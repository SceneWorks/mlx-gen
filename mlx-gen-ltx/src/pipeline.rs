//! S5 — the **2-stage distilled T2V pipeline**: the denoise loop + the stage transition (2× spatial
//! upsample + re-noise) + video output. Port of the `mlx_video` reference `generate_av.py` video path
//! (`denoise_av` with audio disabled + the `generate_video_with_audio` stage orchestration).
//!
//! The shipped `base_q8` is a **unified split-weight** checkpoint (`split_model.json` `format:
//! "split"`), so the reference takes its `legacy_unified_sampler` branch: **fixed distilled sigmas**
//! ([`STAGE1_SIGMAS`] 8 steps + [`STAGE2_SIGMAS`] 3 steps) and the **legacy dtype-preserving Euler**
//! update. The distilled 2.3 model bakes in guidance, so `effective_cfg_scale = 1.0` → **no CFG**
//! (single forward per step, no negative prompt). T2V has no I2V conditioning state.
//!
//! Flow ([`generate_t2v`]): random noise → stage-1 denoise at half-res → [`upsample_latents`] 2× →
//! re-noise (`noise·σ₂₀ + latent·(1−σ₂₀)`) → stage-2 denoise at full-res → VAE decode →
//! `(x+1)/2·255` uint8 frames `[F, H, W, 3]` (the consuming app muxes these to MP4 — matching the
//! Wan sibling, MP4 encoding is out of the crate).
//!
//! **Precision (S5 gate).** Run in the **f32** regime (latents f32, transformer [`Precision::quant_f32`],
//! upsampler/VAE f32) to gate the pipeline *math* — the 2-stage orchestration, the legacy Euler, the
//! re-noise, the flatten/unflatten — bit-tight, isolated from bf16 rounding (consistent with the S3b
//! DiT gate). The bf16-**production** end-to-end px>8 verdict is S6, which wires the real text encoder
//! and the public `generate()`. Honors "divergence is not rounding": the parity test localizes any
//! gap (per-stage latents + decoded frames) rather than writing it off.

use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{add, broadcast_to, divide, maximum, minimum, multiply, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::media::AudioTrack;
use mlx_gen::tiling::{budgeted_plan, TileCandidates, TilingBudgetError, TilingConfig};
use mlx_gen::{CancelFlag, Error, Image, Result};

use crate::audio_vae::AudioDecoder;
use crate::conditioning::{
    append_keyframe_clip, apply_conditioning, apply_denoise_mask, apply_keyframes, token_timesteps,
    unpatchify_grid, I2vConditioning, Keyframe, VideoTokenState,
};
use crate::positions::{DEFAULT_FPS, SPATIAL_SCALE, TEMPORAL_SCALE};
use crate::transformer::{to_denoised, AvDiT, LtxDiT};
use crate::upsampler::{upsample_latents, LatentUpsampler};
use crate::vae::LtxVideoVae;
use crate::vocoder::LtxVocoder;

/// Number of distilled denoise passes (stage-1 + stage-2). Drives the per-pass LoRA strength
/// schedule (sc-2687): `pass_scales` on an [`AdapterSpec`](mlx_gen::AdapterSpec) carries one entry
/// per pass, and [`generate_t2v_latents`] selects the active pass on the DiT before each stage.
pub const NUM_DENOISE_PASSES: usize = 2;

/// Distilled stage-1 sigmas (`DEFAULT_STAGE_1_SIGMAS`, 8 denoise steps).
pub const STAGE1_SIGMAS: [f32; 9] = [
    1.0, 0.993_75, 0.987_5, 0.981_25, 0.975, 0.909_375, 0.725, 0.421_875, 0.0,
];
/// Distilled stage-2 sigmas (`DEFAULT_STAGE_2_SIGMAS`, 3 denoise steps). `STAGE2_SIGMAS[0]` is the
/// stage-transition re-noise scale.
pub const STAGE2_SIGMAS: [f32; 4] = [0.909_375, 0.725, 0.421_875, 0.0];

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Force a logically-contiguous copy (see `vae.rs`): host reads (`as_slice`) return the *physical*
/// buffer, so an array left strided by the `(F,H,W,C)` transpose reads scrambled.
fn contiguous(x: &Array) -> Result<Array> {
    let shape = x.shape().to_vec();
    Ok(x.reshape(&[-1])?.reshape(&shape)?)
}

/// The legacy dtype-preserving Euler update (the `use_legacy_euler` branch): for `σ_next > 0`,
/// `x' = denoised + σ_next·(x − denoised)/σ`; at the final step (`σ_next = 0`), `x' = denoised`.
/// Computed in `x`'s dtype (the σ scalars are cast to it) — algebraically `x + (σ_next − σ)·v` but
/// kept in the reference's exact op order/dtype for bit-parity.
pub fn euler_step(x: &Array, denoised: &Array, sigma: f32, sigma_next: f32) -> Result<Array> {
    if sigma_next <= 0.0 {
        return Ok(denoised.clone());
    }
    let dt = x.dtype();
    let sn = scalar(sigma_next).as_dtype(dt)?;
    let sg = scalar(sigma).as_dtype(dt)?;
    let step = divide(&multiply(&sn, &subtract(x, denoised)?)?, &sg)?;
    Ok(add(denoised, &step)?)
}

/// One stage's denoise loop. Distilled: **no CFG**, **legacy Euler**.
///
/// * `latents` — `(B, 128, F, H, W)` NCFHW, the stage's dtype (f32 here, S5 gate). For I2V this is
///   the conditioned + noised [`I2vConditioning::latent`].
/// * `context` — `(B, ctx, inner)` text embeddings (the connector output / S6's text encoder).
/// * `positions` — `(B, 3, S, 2)` position grid for this stage's latent dims.
/// * `sigmas` — the stage schedule; `sigmas.len() − 1` denoise steps.
/// * `state` — `None` for T2V (uniform per-token σ); `Some` for I2V (per-token `σ·mask`, with the
///   denoised output blended toward the clean conditioning each step — the reference `denoise(...,
///   state=...)` path that pins the conditioned frame).
/// * `on_step` — progress callback, fired once per completed step.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    dit: &LtxDiT,
    latents: &Array,
    context: &Array,
    positions: &Array,
    sigmas: &[f32],
    state: Option<&I2vConditioning>,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let dt = latents.dtype();
    let sh = latents.shape();
    let (b, c, f, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
    let num_tokens = f * h * w;
    let mut lat = latents.clone();

    for i in 0..sigmas.len() - 1 {
        // Honor the engine cancellation contract — check before each (minutes-long) step (sc-5551,
        // the video sibling of chroma's sc-5514). The per-step `eval` below makes this effective.
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        // (B, C, F, H, W) → (B, C, S) → (B, S, C) packed tokens.
        let flat = lat.reshape(&[b, c, -1])?.transpose_axes(&[0, 2, 1])?;
        // Per-token timesteps, shape (B, num_tokens): T2V → uniform σ; I2V → σ·mask (conditioned
        // tokens get 0). Matches the reference `denoise`.
        let ts = match state {
            Some(st) => st.token_timesteps(sigma, h, w)?,
            None => broadcast_to(&scalar(sigma).as_dtype(dt)?, &[b, num_tokens])?,
        };
        let velocity = dit.forward(&flat, &ts, context, None, positions)?;
        // (B, S, C) → (B, C, S) → (B, C, F, H, W).
        let velocity = velocity
            .transpose_axes(&[0, 2, 1])?
            .reshape(&[b, c, f, h, w])?;
        let sig = scalar(sigma).as_dtype(dt)?;
        let mut denoised = to_denoised(&lat, &velocity, &sig)?;
        // I2V: pin the conditioned frame(s) to the clean image latent (reference `apply_denoise_mask`).
        if let Some(st) = state {
            denoised = apply_denoise_mask(&denoised, &st.clean_latent, &st.denoise_mask)?;
        }
        lat = euler_step(&lat, &denoised, sigma, sigma_next)?;
        mlx_rs::transforms::eval([&lat])?;
        on_step(i + 1);
    }
    Ok(lat)
}

/// Stage-transition re-noise: `noise·scale + latent·(1 − scale)`, dtype-preserving. `1 − scale` is
/// computed in `latent`'s dtype (`array(1) − array(scale)`), matching the reference exactly.
pub fn renoise(latents: &Array, noise: &Array, noise_scale: f32) -> Result<Array> {
    let dt = latents.dtype();
    let s = scalar(noise_scale).as_dtype(dt)?;
    let one_minus = subtract(&scalar(1.0).as_dtype(dt)?, &s)?;
    Ok(add(&multiply(noise, &s)?, &multiply(latents, &one_minus)?)?)
}

/// VAE-decode latents `(B, 128, F, H, W)` → `(F, H, W, 3)` uint8 frames. Reference order:
/// squeeze batch → `(F, H, W, 3)` → `clip((x+1)/2, 0, 1)·255` → uint8.
///
/// sc-6894 F-004 — routes the (previously dead-in-production) [`LtxVideoVae::decode_tiled`] behind the
/// shared memory-budgeted selector. The decoded output dims come from the latent geometry (LTX VAE:
/// ×32 spatial, ×8 **causal** temporal ⇒ `out_f = 1 + (T_lat−1)·8`), so an over-budget decode returns
/// a **catchable** error here instead of SIGKILLing the process inside a single-pass full-video decode.
/// Small outputs select `None` → the original single-pass `decode` (byte-identical to before).
pub fn decode_to_frames(vae: &LtxVideoVae, latents: &Array) -> Result<Array> {
    let sh = latents.shape(); // (B, 128, T_lat, H_lat, W_lat)
    let (t_lat, h_lat, w_lat) = (sh[2], sh[3], sh[4]);
    let out_f = 1 + (t_lat - 1) * TEMPORAL_SCALE as i32;
    let out_h = h_lat * SPATIAL_SCALE as i32;
    let out_w = w_lat * SPATIAL_SCALE as i32;
    let decoded = match auto_tiling_budgeted_ltx(out_h, out_w, out_f)? {
        Some(cfg) => vae.decode_tiled(latents, &cfg)?,
        None => vae.decode(latents)?,
    };
    to_uint8_frames(&decoded)
}

/// `(B=1, 3, F, H, W)` video in ~[-1, 1] → `(F, H, W, 3)` uint8. The reference clips `(x+1)/2` to
/// `[0, 1]` *before* scaling by 255, so the result saturates at 255 (truncating cast).
pub fn to_uint8_frames(video: &Array) -> Result<Array> {
    let sh = video.shape(); // (1, 3, F, H, W)
                            // The reshape below drops the batch axis; B>1 would interleave frames across batch items (or
                            // shape-error). Production decode is B==1 — reject anything else with a clear message (F-051).
    if sh[0] != 1 {
        return Err(Error::Msg(format!(
            "ltx to_uint8_frames: batch size must be 1, got {}",
            sh[0]
        )));
    }
    let (c, f, h, w) = (sh[1], sh[2], sh[3], sh[4]);
    let dt = video.dtype();
    let chw = video
        .reshape(&[c, f, h, w])?
        .transpose_axes(&[1, 2, 3, 0])?; // (F, H, W, 3)
    let half = divide(
        &add(&chw, &scalar(1.0).as_dtype(dt)?)?,
        &scalar(2.0).as_dtype(dt)?,
    )?;
    let clipped = minimum(
        &maximum(&half, &scalar(0.0).as_dtype(dt)?)?,
        &scalar(1.0).as_dtype(dt)?,
    )?;
    let scaled = multiply(&clipped, &scalar(255.0).as_dtype(dt)?)?;
    contiguous(&scaled.as_dtype(Dtype::Uint8)?)
}

// --- sc-6894 F-004: LTX VAE decode budgeting ------------------------------------------------------
//
// [`LtxVideoVae::decode_tiled`] shipped parity-validated in sc-2679 S2b but was never wired into the
// pipeline — every production decode ran a single-pass full-video decode with no OOM guard. These
// route it through the shared `budgeted_plan` selector (gen-core) with an LTX cost model fit from the
// real-weight `vae_decode_sweep.rs` anchors. The LTX VAE is causal in time (×8) and upsamples ×32
// spatially, so its per-output-voxel decode working set differs from the Wan VAEs; the constants are
// LTX-specific.

/// **Fixed** decode floor (bytes): the resident decoder weights + base MLX working set, paid
/// regardless of output/tile size. Unlike the Wan VAEs (whose per-voxel cost dwarfs any fixed term),
/// the LTX decoder is light per output voxel (×32 spatial compression) so this ~2.5 GB floor
/// dominates small/mid decodes — omitting it would force a no-fixed model to over-predict the
/// max-size decode by ~140 % and tile pathologically. Fit from `vae_decode_sweep.rs` (5 single-pass
/// points, intercept ~2.5 GB; rounded **up** to 3.3 for headroom — the model must never under-predict).
const LTX_VAE_FIXED_BYTES: f64 = 3.3e9;
/// Per-output-voxel cost of the LTX decode's full-output f32 accumulators (`output` [1,3,F,H,W] +
/// `weights` [1,1,F,H,W]) — paid by every tiled plan. Isolated from the single-pass slope minus the
/// 1024²×25 @512-px tiled anchor (~36 B/voxel); rounded **up** to 40.
const LTX_VAE_ACCUM_BYTES_PER_VOXEL: f64 = 40.0;
/// Per-tile-output-voxel cost of the LTX decoder working set (×32 spatial / ×8 causal-temporal
/// upsample). Fit from the same anchors at ~287 B/voxel; rounded **up** to 300. (Far lighter per
/// voxel than the Wan VAEs — the heavy ×32 upsample runs on a tiny latent.)
const LTX_VAE_TILE_BYTES_PER_OUT_VOXEL: f64 = 300.0;

/// Candidate spatial tile sizes (output px, multiples of the LTX ×32 spatial scale, overlap 64).
const LTX_VAE_SPATIAL_PX: [i32; 8] = [768, 640, 512, 448, 384, 320, 256, 192];
/// Candidate temporal tiles `(tile_frames, overlap_frames)` in output frames (the causal decoder maps
/// `tile_frames/8` latent frames per tile).
const LTX_VAE_TEMPORAL_FR: [(i32, i32); 4] = [(96, 24), (64, 16), (48, 16), (24, 8)];

/// Estimated concurrent GPU peak (GiB) of an LTX decode whose largest tile spans `tile_*` output
/// voxels while assembling an `out_*` video. Pure (no global state) → unit-testable against the
/// `vae_decode_sweep.rs` anchors. Single-pass is `tile_* == out_*`; a zero tile is the
/// accumulator-only floor.
fn estimated_ltx_decode_peak_gib(
    out_f: i64,
    out_h: i64,
    out_w: i64,
    tile_f: i64,
    tile_h: i64,
    tile_w: i64,
) -> f64 {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let out_voxels = (out_f * out_h * out_w) as f64;
    let tile_voxels = (tile_f * tile_h * tile_w) as f64;
    (LTX_VAE_FIXED_BYTES
        + LTX_VAE_ACCUM_BYTES_PER_VOXEL * out_voxels
        + LTX_VAE_TILE_BYTES_PER_OUT_VOXEL * tile_voxels)
        / GIB
}

/// **Memory-budgeted** tiling for the LTX VAE decode (sc-6894 F-004): routes the shared
/// [`budgeted_plan`] selector through the LTX cost model. Caller passes the **output** dims (the
/// decoded video size). `Ok(None)` → a single-pass decode already fits; `Err` → a catchable
/// over-budget signal returned before the decode (not a SIGKILL).
pub fn auto_tiling_budgeted_ltx(
    height: i32,
    width: i32,
    out_frames: i32,
) -> Result<Option<TilingConfig>> {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let budget_gib = get_memory_limit() as f64 / GIB;
    plan_ltx_tiling(height, width, out_frames, budget_gib * 0.85)
}

/// Pure LTX tile selector behind [`auto_tiling_budgeted_ltx`] (the `safe_gib` ceiling is injected so it
/// is unit-testable without touching the global memory limit). Supplies the LTX cost model + candidate
/// grid to the shared [`budgeted_plan`].
fn plan_ltx_tiling(
    height: i32,
    width: i32,
    out_frames: i32,
    safe_gib: f64,
) -> Result<Option<TilingConfig>> {
    let candidates = TileCandidates {
        spatial_px: &LTX_VAE_SPATIAL_PX,
        spatial_overlap_px: 64,
        temporal: &LTX_VAE_TEMPORAL_FR,
    };
    budgeted_plan(
        height,
        width,
        out_frames,
        safe_gib,
        candidates,
        estimated_ltx_decode_peak_gib,
    )
    .map_err(|e| ltx_decode_budget_error(width, height, out_frames, e))
}

/// Map gen-core's neutral [`TilingBudgetError`] to an LTX-facing catchable decode error.
fn ltx_decode_budget_error(
    width: i32,
    height: i32,
    out_frames: i32,
    e: TilingBudgetError,
) -> Error {
    match e {
        TilingBudgetError::AccumulatorsExceedBudget {
            projected_gib,
            safe_gib,
        } => Error::Msg(format!(
            "ltx vae decode: assembling a {width}×{height}×{out_frames} video needs ~{projected_gib:.0} \
             GB just for the output buffers, over this machine's ~{safe_gib:.0} GB safe budget. Reduce \
             the resolution or frame count."
        )),
        TilingBudgetError::SmallestTileExceedsBudget {
            projected_gib,
            safe_gib,
        } => Error::Msg(format!(
            "ltx vae decode: a {width}×{height}×{out_frames} video peaks at ~{projected_gib:.0} GB even \
             with the smallest tile, over this machine's ~{safe_gib:.0} GB safe budget. Reduce the \
             resolution or frame count."
        )),
    }
}

/// Render one preview sample (sc-5637) from the **in-progress training adapter** already installed on
/// `dit`: seeded single-frame noise → the distilled **stage-1** Euler denoise (the real
/// [`STAGE1_SIGMAS`] schedule, no CFG) → VAE decode → the first frame as an [`Image`]. A single-stage
/// stripped [`generate_t2v_latents`] for the trainer's periodic preview (the 2× upsample + stage-2 are
/// a quality refinement skipped here to keep the per-cadence cost low). `context` is the pre-encoded
/// prompt embedding `(1, L, 4096)`; `positions` is the `(1, 1, le, le)` grid; `dtype` is the trainer
/// compute dtype. No progress/cancel plumbing — the caller drives the cadence.
pub(crate) fn render_sample(
    dit: &LtxDiT,
    vae: &LtxVideoVae,
    context: &Array,
    positions: &Array,
    seed: u64,
    latent_edge: usize,
    dtype: Dtype,
) -> Result<Image> {
    let le = latent_edge as i32;
    let init = mlx_rs::random::normal::<f32>(
        &[1, 128, 1, le, le],
        None,
        None,
        Some(&mlx_rs::random::key(seed)?),
    )?
    .as_dtype(dtype)?;
    let latents = denoise(
        dit,
        &init,
        context,
        positions,
        &STAGE1_SIGMAS,
        None,
        &CancelFlag::default(),
        &mut |_| {},
    )?;
    let frames = decode_to_frames(vae, &latents)?; // (F, H, W, 3) uint8
    frame0_to_image(&frames)
}

/// First frame of a decoded `(F, H, W, 3)` uint8 tensor → an RGB8 [`Image`] (a video LoRA's preview
/// is a still thumbnail, sc-5637).
fn frame0_to_image(frames: &Array) -> Result<Image> {
    let sh = frames.shape(); // (F, H, W, 3)
    if sh[0] < 1 {
        return Err(Error::Msg("ltx render_sample: no frames decoded".into()));
    }
    let (h, w, c) = (sh[1] as usize, sh[2] as usize, sh[3] as usize);
    let n = h * w * c;
    let flat = frames.reshape(&[-1])?;
    let pixels = flat.as_slice::<u8>()[..n].to_vec();
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// Prepare an I2V conditioning image for VAE encoding (reference `prepare_image_for_encoding` ∘
/// `load_image`): PIL-LANCZOS scale the RGB8 image to the stage pixel resolution `(target_height,
/// target_width)` (a no-op when already sized), normalize `[0,255] → [-1,1]`, and lay out as **NCFHW**
/// `[1, 3, 1, H, W]` f32 — the single-frame video the [`LtxVideoVae::encode`](crate::vae::LtxVideoVae)
/// expects. The reference resizes the *original* image directly to each stage's pixel resolution, so
/// the caller passes `height/2 × width/2` for stage 1 and `height × width` for stage 2.
pub fn preprocess_conditioning_image(
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (target_width as usize, target_height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(Error::Msg(format!(
            "I2V conditioning image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    // PIL LANCZOS on the uint8 image (no-op when already at target size), matching `load_image`.
    let resized: Vec<f32> = if (ih, iw) == (th, tw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)
    };
    // /255 then [-1,1], as NHWC.
    let norm: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    let nhwc = Array::from_slice(&norm, &[1, th as i32, tw as i32, 3]);
    // NHWC → NCHW → insert the singleton temporal axis → (1, 3, 1, H, W).
    let nchw = nhwc.transpose_axes(&[0, 3, 1, 2])?; // (1, 3, H, W)
    Ok(nchw.reshape(&[1, 3, 1, th as i32, tw as i32])?)
}

/// Prepare a multi-frame conditioning **clip** for VAE encoding (the in-context extend/bridge source):
/// each frame is PIL-LANCZOS resized to `(target_height, target_width)`, normalized `[0,255] → [-1,1]`,
/// and stacked along the temporal axis as **NCFHW** `[1, 3, F, H, W]` f32 — the multi-frame analogue of
/// [`preprocess_conditioning_image`] that [`LtxVideoVae::encode`](crate::vae::LtxVideoVae) compresses to
/// `[1, 128, cf, H/32, W/32]`. `frames` must be non-empty and all the same source size.
pub fn preprocess_conditioning_clip(
    frames: &[Image],
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    if frames.is_empty() {
        return Err(Error::Msg("conditioning clip must have ≥1 frame".into()));
    }
    let (tw, th) = (target_width as usize, target_height as usize);
    // Each frame → NCFHW (1,3,1,th,tw); concat along the temporal axis (axis 2).
    let per_frame: Vec<Array> = frames
        .iter()
        .map(|f| preprocess_conditioning_image(f, target_width, target_height))
        .collect::<Result<_>>()?;
    let refs: Vec<&Array> = per_frame.iter().collect();
    let clip = mlx_rs::ops::concatenate_axis(&refs, 2)?;
    debug_assert_eq!(
        clip.shape(),
        &[1, 3, frames.len() as i32, th as i32, tw as i32]
    );
    Ok(clip)
}

/// The full 2-stage distilled T2V latent pipeline: stage-1 denoise → 2× upsample → re-noise →
/// stage-2 denoise. `stage1_noise`/`stage2_noise` are the (injected) initial + re-noise samples,
/// `context` the shared text embeddings, `*_positions` each stage's grid, `latent_{mean,std}` the VAE
/// `per_channel_statistics`. Returns the final full-res latents `(B, 128, F, H, W)`.
#[allow(clippy::too_many_arguments)]
pub fn generate_t2v_latents(
    dit: &LtxDiT,
    upsampler: &LatentUpsampler,
    stage1_noise: &Array,
    stage1_positions: &Array,
    stage2_noise: &Array,
    stage2_positions: &Array,
    context: &Array,
    latent_mean: &Array,
    latent_std: &Array,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    // sc-2963 (rollout of sc-2957): run the AvDiT's fusable elementwise glue (adaLN affine, gated
    // residuals, tanh-GELU FFN, split-RoPE rotation) through `mx.compile` — bit-exact and the biggest
    // per-step win of the rollout at video sequence (the FFN GELU dominates). Enabled here at the
    // production boundary (not inside the shared `denoise`, which the parity tests reuse eager).
    // sc-4045/F-049: an RAII guard restores the prior process-global on return, so the eager parity
    // gates aren't left running compiled after a generate.
    let _compile_glue = crate::CompileGlueGuard::enable();
    // Select the per-pass LoRA strength for stage 1 (a no-op without adapters; sc-2687).
    dit.set_lora_pass(0);
    let lat = denoise(
        dit,
        stage1_noise,
        context,
        stage1_positions,
        &STAGE1_SIGMAS,
        None,
        cancel,
        on_step,
    )?;
    let lat = upsample_latents(&lat, upsampler, latent_mean, latent_std)?;
    let lat = renoise(&lat, stage2_noise, STAGE2_SIGMAS[0])?;
    dit.set_lora_pass(1);
    denoise(
        dit,
        &lat,
        context,
        stage2_positions,
        &STAGE2_SIGMAS,
        None,
        cancel,
        on_step,
    )
}

/// The full 2-stage distilled **I2V** latent pipeline (reference `generate.py` / `generate_av.py`
/// video path with `state`): stage-1 condition + noise + conditioned denoise → 2× upsample → stage-2
/// condition + re-noise + conditioned denoise. Differs from [`generate_t2v_latents`] only in the
/// conditioning state: each stage injects its VAE-encoded image latent at `frame_idx` (clean latent +
/// per-frame `1 − strength` mask), seeds the loop via the [`I2vConditioning::noised`] noiser (so the
/// conditioned frame is pinned and the rest gets the stage's noise), and runs the conditioned denoise.
///
/// * `stage1_image_latent` `(B, 128, 1, h1, w1)` / `stage2_image_latent` `(B, 128, 1, h2, w2)` — the
///   conditioning image VAE-encoded at each stage's latent resolution.
/// * `stage1_noise` / `stage2_noise` — the stage noise (the reference draws fresh `normal`; the
///   parity seam injects the reference samples). The conditioned frame ignores it (mask).
/// * `frame_idx` / `strength` — single-image I2V uses `frame_idx = 0`; `strength = 1.0` fully pins
///   the conditioned frame.
///
/// Returns the final full-res latents `(B, 128, F, h2, w2)`.
#[allow(clippy::too_many_arguments)]
pub fn generate_i2v_latents(
    dit: &LtxDiT,
    upsampler: &LatentUpsampler,
    stage1_image_latent: &Array,
    stage1_noise: &Array,
    stage1_positions: &Array,
    stage2_image_latent: &Array,
    stage2_noise: &Array,
    stage2_positions: &Array,
    context: &Array,
    latent_mean: &Array,
    latent_std: &Array,
    frame_idx: i32,
    strength: f32,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    // Stage 1: condition over a zero base, noise (σ₀ = 1.0), conditioned denoise. The image latent is
    // cast to the base/noise dtype (the f32 VAE encoder feeds a bf16 path with a sub-ULP cast — the
    // same post-encode quality island as the VAE decode; a no-op when both are already f32/bf16).
    let zeros1 = Array::zeros::<f32>(stage1_noise.shape())?.as_dtype(stage1_noise.dtype())?;
    let cond1 = stage1_image_latent.as_dtype(zeros1.dtype())?;
    let st1 = apply_conditioning(&zeros1, &cond1, frame_idx, strength)?;
    let st1 = st1.noised(stage1_noise, STAGE1_SIGMAS[0])?;
    let lat = denoise(
        dit,
        &st1.latent,
        context,
        stage1_positions,
        &STAGE1_SIGMAS,
        Some(&st1),
        cancel,
        on_step,
    )?;

    // Upsample 2×.
    let lat = upsample_latents(&lat, upsampler, latent_mean, latent_std)?;

    // Stage 2: condition over the upscaled latent, re-noise (σ₀ = STAGE2_SIGMAS[0]), conditioned denoise.
    let cond2 = stage2_image_latent.as_dtype(lat.dtype())?;
    let st2 = apply_conditioning(&lat, &cond2, frame_idx, strength)?;
    let st2 = st2.noised(stage2_noise, STAGE2_SIGMAS[0])?;
    denoise(
        dit,
        &st2.latent,
        context,
        stage2_positions,
        &STAGE2_SIGMAS,
        Some(&st2),
        cancel,
        on_step,
    )
}

/// [`generate_t2v_latents`] + VAE decode → uint8 frames `(F, H, W, 3)`.
#[allow(clippy::too_many_arguments)]
pub fn generate_t2v(
    dit: &LtxDiT,
    upsampler: &LatentUpsampler,
    vae: &LtxVideoVae,
    stage1_noise: &Array,
    stage1_positions: &Array,
    stage2_noise: &Array,
    stage2_positions: &Array,
    context: &Array,
    latent_mean: &Array,
    latent_std: &Array,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let latents = generate_t2v_latents(
        dit,
        upsampler,
        stage1_noise,
        stage1_positions,
        stage2_noise,
        stage2_positions,
        context,
        latent_mean,
        latent_std,
        cancel,
        on_step,
    )?;
    decode_to_frames(vae, &latents)
}

// ===================================================================================================
// AudioVideo pipeline (sc-2684) — the joint `denoise_av` loop + audio decode → waveform.
// ===================================================================================================

/// One stage's **joint** video+audio denoise loop (`denoise_av`). Distilled T2V+A / I2V+A: no CFG,
/// legacy Euler, audio always enabled (the cross-modal attention couples the streams every step).
/// Audio init is pure noise (no audio I2V — the reference's `video_state` conditions only the video).
///
/// * `video` — `(B, 128, F, H, W)` NCFHW; `audio` — `(B, 8, T, 16)` NCTF.
/// * `*_ctx` — the video (4096) / audio (2048) text embeddings.
/// * `*_pos` — the video `(B,3,Sv,2)` / audio `(B,1,Ta,2)` position grids.
/// * `video_state` — `None` for T2V (uniform per-token σ); `Some` for I2V (the video stream gets
///   per-token `σ·mask` + `apply_denoise_mask` each step, pinning the conditioned frame). The audio
///   stream is unaffected, matching `generate_av.py` (`video_state` is video-only).
#[allow(clippy::too_many_arguments)]
pub fn denoise_av(
    dit: &AvDiT,
    video: &Array,
    audio: &Array,
    video_ctx: &Array,
    audio_ctx: &Array,
    video_pos: &Array,
    audio_pos: &Array,
    sigmas: &[f32],
    video_state: Option<&I2vConditioning>,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<(Array, Array)> {
    let dt = video.dtype();
    let v = video.shape();
    let (vb, vc, vf, vh, vw) = (v[0], v[1], v[2], v[3], v[4]);
    let v_tokens = vf * vh * vw;
    let a = audio.shape();
    let (ab, ac, at, af) = (a[0], a[1], a[2], a[3]);

    let mut vlat = video.clone();
    let mut alat = audio.clone();
    for i in 0..sigmas.len() - 1 {
        // Honor the engine cancellation contract — check before each (minutes-long) step (sc-5551).
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        // Flatten: video (B,C,F,H,W)→(B,Sv,C); audio (B,C,T,F)→(B,T,C·F).
        let vflat = vlat.reshape(&[vb, vc, -1])?.transpose_axes(&[0, 2, 1])?;
        let aflat = alat
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[ab, at, ac * af])?;
        // Video per-token σ: I2V → σ·mask (conditioned tokens 0); T2V → uniform σ. Audio uniform σ.
        let vts = match video_state {
            Some(st) => st.token_timesteps(sigma, vh, vw)?,
            None => broadcast_to(&scalar(sigma).as_dtype(dt)?, &[vb, v_tokens])?,
        };
        let ats = broadcast_to(&scalar(sigma).as_dtype(dt)?, &[ab, at])?;
        let (vvel, avel) = dit.forward(
            &vflat, &vts, video_ctx, None, video_pos, &aflat, &ats, audio_ctx, None, audio_pos,
        )?;
        let vvel = vvel
            .transpose_axes(&[0, 2, 1])?
            .reshape(&[vb, vc, vf, vh, vw])?;
        let avel = avel
            .reshape(&[ab, at, ac, af])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let sig = scalar(sigma).as_dtype(dt)?;
        let mut vden = to_denoised(&vlat, &vvel, &sig)?;
        // I2V: pin the conditioned frame(s) to the clean image latent (video only).
        if let Some(st) = video_state {
            vden = apply_denoise_mask(&vden, &st.clean_latent, &st.denoise_mask)?;
        }
        let aden = to_denoised(&alat, &avel, &sig)?;
        vlat = euler_step(&vlat, &vden, sigma, sigma_next)?;
        alat = euler_step(&alat, &aden, sigma, sigma_next)?;
        mlx_rs::transforms::eval([&vlat, &alat])?;
        on_step(i + 1);
    }
    Ok((vlat, alat))
}

/// A replace-latent keyframe at both pipeline stages: the conditioning latent VAE-encoded at stage-1
/// (half-res) and stage-2 (full-res) resolution, the **latent** frame index it pins, and its strength
/// (mask `1 − strength`). Single-image I2V = one keyframe at frame 0; **first_last_frame** = two
/// (frame 0 and the last latent frame). The replace-latent mechanism rewrites grid frames in place, so
/// it drives the existing [`denoise_av`] loop unchanged (no token-native loop).
#[derive(Clone, Copy)]
pub struct StageKeyframe<'a> {
    pub stage1: &'a Array,
    pub stage2: &'a Array,
    pub frame_idx: i32,
    pub strength: f32,
}

/// Build the per-stage [`I2vConditioning`] for a stage's `keyframes` over `base` (zeros for stage 1,
/// the upscaled latent for stage 2), casting each conditioning latent to the base dtype. Empty → T2V.
fn stage_keyframe_state(
    base: &Array,
    keyframes: &[StageKeyframe],
    stage1: bool,
) -> Result<Option<I2vConditioning>> {
    if keyframes.is_empty() {
        return Ok(None);
    }
    let dt = base.dtype();
    let cast: Vec<Array> = keyframes
        .iter()
        .map(|k| Ok((if stage1 { k.stage1 } else { k.stage2 }).as_dtype(dt)?))
        .collect::<Result<_>>()?;
    let kfs: Vec<Keyframe> = keyframes
        .iter()
        .zip(&cast)
        .map(|(k, l)| Keyframe {
            latent: l,
            frame_idx: k.frame_idx,
            strength: k.strength,
        })
        .collect();
    Ok(Some(apply_keyframes(base, &kfs)?))
}

/// Stage-1 **token-native** joint video+audio denoise for the keyframe-append (IC-LoRA) path. The
/// video stream is a [`VideoTokenState`] whose token sequence already includes the appended
/// conditioning clips (so it is **not** a grid); the audio stream is the usual `(B, 8, T, 16)` grid.
/// Mirrors [`denoise_av`] but takes the video as tokens + per-token `positions`/`denoise_mask`/`clean`
/// (so the appended tokens carry their own RoPE positions) and pins the conditioning tokens each step.
/// Returns `(video_token_state_after, audio_grid)`; the caller reads back the generated grid via
/// [`VideoTokenState::target_tokens`] + `unpatchify_grid`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_av_tokens(
    dit: &AvDiT,
    video: &VideoTokenState,
    audio: &Array,
    video_ctx: &Array,
    audio_ctx: &Array,
    audio_pos: &Array,
    sigmas: &[f32],
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<(VideoTokenState, Array)> {
    let dt = video.latent.dtype();
    let a = audio.shape();
    let (ab, ac, at, af) = (a[0], a[1], a[2], a[3]);

    let mut vtok = video.latent.clone();
    let mut alat = audio.clone();
    for i in 0..sigmas.len() - 1 {
        // Honor the engine cancellation contract — check before each (minutes-long) step (sc-5551).
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        // Video already token-native (B, Sv, C). Audio (B,C,T,F) → (B,T,C·F).
        let aflat = alat
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[ab, at, ac * af])?;
        // Video per-token σ = σ·mask (conditioning tokens get σ·(1−strength)); audio uniform σ. The
        // timesteps depend only on the fixed denoise_mask + dtype, so call the free fn directly
        // instead of rebuilding a throwaway VideoTokenState each step (F-060).
        let vts = token_timesteps(&video.denoise_mask, vtok.dtype(), sigma)?;
        let ats = broadcast_to(&scalar(sigma).as_dtype(dt)?, &[ab, at])?;
        let (vvel, avel) = dit.forward(
            &vtok,
            &vts,
            video_ctx,
            None,
            &video.positions,
            &aflat,
            &ats,
            audio_ctx,
            None,
            audio_pos,
        )?;
        let avel = avel
            .reshape(&[ab, at, ac, af])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let sig = scalar(sigma).as_dtype(dt)?;
        // Pin conditioning tokens to their clean latent (token-native apply_denoise_mask).
        let vden = apply_denoise_mask(
            &to_denoised(&vtok, &vvel, &sig)?,
            &video.clean_latent,
            &video.denoise_mask,
        )?;
        let aden = to_denoised(&alat, &avel, &sig)?;
        vtok = euler_step(&vtok, &vden, sigma, sigma_next)?;
        alat = euler_step(&alat, &aden, sigma, sigma_next)?;
        mlx_rs::transforms::eval([&vtok, &alat])?;
        on_step(i + 1);
    }
    Ok((
        VideoTokenState {
            latent: vtok,
            clean_latent: video.clean_latent.clone(),
            denoise_mask: video.denoise_mask.clone(),
            positions: video.positions.clone(),
            target_tokens: video.target_tokens,
        },
        alat,
    ))
}

/// The full 2-stage **AudioVideo** latent pipeline: joint stage-1 denoise → 2× upsample the **video**
/// (audio is not upsampled) → re-noise both → joint stage-2 denoise. Returns `(video_latents (B,128,
/// F,H,W), audio_latents (B,8,T,16))`.
///
/// A non-empty `video_keyframes` switches the **video** stream to replace-latent conditioning (I2V /
/// first_last_frame / multi-keyframe; the audio is always pure-noise, matching `generate_av.py`'s
/// I2V+Audio): each stage injects the VAE-encoded keyframe latents at their frame indices (clean
/// latent plus a `1 − strength` mask), seeds the loop via the [`I2vConditioning::noised`] noiser, and
/// runs the conditioned `denoise_av`.
#[allow(clippy::too_many_arguments)]
pub fn generate_av_latents(
    dit: &AvDiT,
    upsampler: &LatentUpsampler,
    video_s1_noise: &Array,
    video_pos1: &Array,
    video_s2_noise: &Array,
    video_pos2: &Array,
    audio_s1_noise: &Array,
    audio_s2_noise: &Array,
    audio_pos: &Array,
    video_ctx: &Array,
    audio_ctx: &Array,
    latent_mean: &Array,
    latent_std: &Array,
    video_keyframes: &[StageKeyframe],
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<(Array, Array)> {
    // sc-2963 (rollout of sc-2957): compiled elementwise glue across the joint video/audio/cross-modal
    // AvDiT forward — see `generate_t2v_latents`. Bit-exact, dtype-preserving, enabled at the
    // production boundary (the shared `denoise_av` stays eager for the parity tests). sc-4045/F-049:
    // an RAII guard restores the prior process-global on return.
    let _compile_glue = crate::CompileGlueGuard::enable();
    // Stage 1: video init = conditioned+noised (replace-latent) or pure noise (T2V); audio = noise.
    let (vlat1, vstate1): (Array, Option<I2vConditioning>) = {
        let zeros =
            Array::zeros::<f32>(video_s1_noise.shape())?.as_dtype(video_s1_noise.dtype())?;
        match stage_keyframe_state(&zeros, video_keyframes, true)? {
            Some(st) => {
                let st = st.noised(video_s1_noise, STAGE1_SIGMAS[0])?;
                (st.latent.clone(), Some(st))
            }
            None => (video_s1_noise.clone(), None),
        }
    };
    // Select the per-pass LoRA strength for stage 1 (a no-op without adapters; sc-2687).
    dit.set_lora_pass(0);
    let (v, a) = denoise_av(
        dit,
        &vlat1,
        audio_s1_noise,
        video_ctx,
        audio_ctx,
        video_pos1,
        audio_pos,
        &STAGE1_SIGMAS,
        vstate1.as_ref(),
        cancel,
        on_step,
    )?;
    let v = upsample_latents(&v, upsampler, latent_mean, latent_std)?;
    // Stage 2: re-noise / re-condition the upscaled video; re-noise audio (never upsampled).
    let (vlat2, vstate2): (Array, Option<I2vConditioning>) =
        match stage_keyframe_state(&v, video_keyframes, false)? {
            Some(st) => {
                let st = st.noised(video_s2_noise, STAGE2_SIGMAS[0])?;
                (st.latent.clone(), Some(st))
            }
            None => (renoise(&v, video_s2_noise, STAGE2_SIGMAS[0])?, None),
        };
    let a = renoise(&a, audio_s2_noise, STAGE2_SIGMAS[0])?;
    dit.set_lora_pass(1);
    denoise_av(
        dit,
        &vlat2,
        &a,
        video_ctx,
        audio_ctx,
        video_pos2,
        audio_pos,
        &STAGE2_SIGMAS,
        vstate2.as_ref(),
        cancel,
        on_step,
    )
}

/// A stage-1 in-context conditioning clip (extend_clip / video_bridge): the source clip VAE-encoded at
/// **stage-1** (half-res) resolution `(B, 128, cf, h1, w1)`, the **latent** frame index it is appended
/// at (extend = 0; bridge left = 0, right = tail), and its strength (mask `1 − strength`). Per the
/// reference `ICLoraPipeline`, video conditioning is applied in **stage 1 only** (stage 2 re-applies
/// only image/replace-latent conditioning).
#[derive(Clone, Copy)]
pub struct StageClip<'a> {
    pub stage1: &'a Array,
    pub frame_idx: i32,
    pub strength: f32,
}

/// The full 2-stage **IC-LoRA** (keyframe-append) A/V pipeline for extend_clip / video_bridge: stage-1
/// **token-native** denoise with the conditioning clips appended as in-context tokens → read back the
/// generated grid → 2× upsample → stage-2 plain grid denoise (clips are stage-1 only). Audio is
/// pure-noise both stages. `grid_dims = (channels, latent_frames, h1, w1)` describes the stage-1 grid
/// for `unpatchify`. Requires an IC-LoRA adapter installed on `dit` (the appended tokens are inert
/// without it); the mechanism + token layout are weight-independent. Returns `(video_latents (B,128,
/// F,h2,w2), audio_latents)`.
#[allow(clippy::too_many_arguments)]
pub fn generate_av_latents_iclora(
    dit: &AvDiT,
    upsampler: &LatentUpsampler,
    video_s1_noise: &Array,
    video_pos1: &Array,
    video_s2_noise: &Array,
    video_pos2: &Array,
    audio_s1_noise: &Array,
    audio_s2_noise: &Array,
    audio_pos: &Array,
    video_ctx: &Array,
    audio_ctx: &Array,
    latent_mean: &Array,
    latent_std: &Array,
    clips: &[StageClip],
    grid_dims: (i32, i32, i32, i32),
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<(Array, Array)> {
    // sc-2963 compiled elementwise glue at the production boundary; sc-4045/F-049 RAII guard restores
    // the prior process-global on return (the shared joint denoise stays eager for the parity tests).
    let _compile_glue = crate::CompileGlueGuard::enable();
    let (c, f, h1, w1) = grid_dims;

    // Stage 1: build the base token state from the noise grid + main positions, append each clip as
    // in-context conditioning tokens, then run the token-native joint denoise.
    let mut vstate = VideoTokenState::base(video_s1_noise, video_pos1)?;
    for clip in clips {
        vstate = append_keyframe_clip(
            &vstate,
            clip.stage1,
            clip.frame_idx,
            clip.strength,
            TEMPORAL_SCALE,
            SPATIAL_SCALE,
            DEFAULT_FPS,
        )?;
    }
    dit.set_lora_pass(0);
    let (vstate, a) = denoise_av_tokens(
        dit,
        &vstate,
        audio_s1_noise,
        video_ctx,
        audio_ctx,
        audio_pos,
        &STAGE1_SIGMAS,
        cancel,
        on_step,
    )?;
    // Read back the generated grid (the first `target_tokens` tokens) → (B, 128, f, h1, w1).
    let tgt_idx: Vec<i32> = (0..vstate.target_tokens).collect();
    let gen_tokens = vstate
        .latent
        .take_axis(Array::from_slice(&tgt_idx, &[vstate.target_tokens]), 1)?;
    let v = unpatchify_grid(&gen_tokens, c, f, h1, w1)?;

    // Stage 2: plain grid denoise (no clips; reference re-applies only image conditioning here).
    let v = upsample_latents(&v, upsampler, latent_mean, latent_std)?;
    let vlat2 = renoise(&v, video_s2_noise, STAGE2_SIGMAS[0])?;
    let a = renoise(&a, audio_s2_noise, STAGE2_SIGMAS[0])?;
    dit.set_lora_pass(1);
    denoise_av(
        dit,
        &vlat2,
        &a,
        video_ctx,
        audio_ctx,
        video_pos2,
        audio_pos,
        &STAGE2_SIGMAS,
        None,
        cancel,
        on_step,
    )
}

/// Decode audio latents → an interleaved-PCM [`AudioTrack`]: `audio_decoder` → mel `(B,2,T',64)` →
/// `vocoder` → waveform `(B,2,samples)` → interleaved `f32`. `decode_audio = audio_decoder → vocoder`.
pub fn decode_audio_track(
    decoder: &AudioDecoder,
    vocoder: &LtxVocoder,
    audio_latents: &Array,
    sample_rate: u32,
) -> Result<AudioTrack> {
    let mel = decoder.decode(audio_latents)?;
    let wav = vocoder.forward(&mel)?; // (B, channels, samples)
    let sh = wav.shape();
    let (channels, samples) = (sh[1] as usize, sh[2]);
    // (1, C, S) → (S, C) → interleaved.
    let interleaved = contiguous(
        &wav.reshape(&[channels as i32, samples])?
            .transpose_axes(&[1, 0])?,
    )?
    .as_dtype(Dtype::Float32)?;
    Ok(AudioTrack {
        samples: interleaved.as_slice::<f32>().to_vec(),
        sample_rate,
        channels: channels as u16,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(v: &[f32], shape: &[i32]) -> Array {
        Array::from_slice(v, shape)
    }

    #[test]
    fn preprocess_conditioning_image_layout_and_norm() {
        // 1×2 RGB image, white pixel then black pixel (HWC). No-op resize (target == source).
        let image = Image {
            width: 2,
            height: 1,
            pixels: vec![255, 255, 255, 0, 0, 0],
        };
        let got = preprocess_conditioning_image(&image, 2, 1).unwrap();
        // NCFHW (1, 3, 1, 1, 2): 255 → 1.0, 0 → -1.0; each channel holds [w0=1, w1=-1].
        assert_eq!(got.shape(), &[1, 3, 1, 1, 2]);
        let c = mlx_rs::ops::reshape(&got, &[-1]).unwrap();
        assert_eq!(c.as_slice::<f32>(), &[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn preprocess_conditioning_image_resizes_to_target() {
        // 4×4 → 2×2: LANCZOS path (values gated by core image tests); just check the output layout.
        let image = Image {
            width: 4,
            height: 4,
            pixels: vec![128u8; 4 * 4 * 3],
        };
        let got = preprocess_conditioning_image(&image, 2, 2).unwrap();
        assert_eq!(got.shape(), &[1, 3, 1, 2, 2]);
    }

    #[test]
    fn stage_sigmas_are_exact() {
        // F-046: lock the production distilled sigma lists (single source of truth now schedule.rs
        // is gone). These are chaos-sensitive — a silent edit would drift the render.
        assert_eq!(STAGE1_SIGMAS.len(), 9); // 8 steps
        assert_eq!(STAGE2_SIGMAS.len(), 4); // 3 steps
        assert_eq!(STAGE1_SIGMAS[0], 1.0);
        assert_eq!(*STAGE1_SIGMAS.last().unwrap(), 0.0);
        assert_eq!(STAGE2_SIGMAS[0], 0.909_375);
        assert_eq!(*STAGE2_SIGMAS.last().unwrap(), 0.0);
        // The stage boundary: stage 2 starts at stage 1's σ index 5 (the 0.909375 re-noise anchor).
        assert_eq!(STAGE1_SIGMAS[5], STAGE2_SIGMAS[0]);
    }

    #[test]
    fn euler_step_matches_reference_formula() {
        // x' = denoised + σ_next·(x − denoised)/σ.
        let x = arr(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let den = arr(&[0.5, 1.0, 1.5, 2.0], &[4]);
        let (sigma, sigma_next) = (0.5_f32, 0.25_f32);
        let got = euler_step(&x, &den, sigma, sigma_next).unwrap();
        let want: Vec<f32> = (0..4)
            .map(|i| {
                let (xv, dv) = (x.as_slice::<f32>()[i], den.as_slice::<f32>()[i]);
                dv + sigma_next * (xv - dv) / sigma
            })
            .collect();
        for (g, w) in got.as_slice::<f32>().iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "euler {g} vs {w}");
        }
    }

    #[test]
    fn euler_step_final_is_denoised() {
        let x = arr(&[1.0, 2.0], &[2]);
        let den = arr(&[9.0, 8.0], &[2]);
        let got = euler_step(&x, &den, 0.42, 0.0).unwrap();
        assert_eq!(got.as_slice::<f32>(), den.as_slice::<f32>());
    }

    #[test]
    fn renoise_matches_reference_formula() {
        // noise·scale + latent·(1−scale).
        let lat = arr(&[1.0, 2.0, 3.0], &[3]);
        let noise = arr(&[0.0, 1.0, -1.0], &[3]);
        let scale = 0.909_375_f32;
        let got = renoise(&lat, &noise, scale).unwrap();
        let want: Vec<f32> = (0..3)
            .map(|i| {
                let (nv, lv) = (noise.as_slice::<f32>()[i], lat.as_slice::<f32>()[i]);
                nv * scale + lv * (1.0 - scale)
            })
            .collect();
        for (g, w) in got.as_slice::<f32>().iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "renoise {g} vs {w}");
        }
    }

    #[test]
    fn to_uint8_frames_clips_and_scales() {
        // (1,3,1,1,2): values spanning below/within/above [-1,1]. Channel values chosen so
        // (x+1)/2·255 lands on exact integers (no trunc-vs-round ambiguity).
        let video = arr(&[-2.0, -1.0, 0.2, 1.0, 0.6, 2.0], &[1, 3, 1, 1, 2]);
        let frames = to_uint8_frames(&video).unwrap();
        assert_eq!(frames.shape(), &[1, 1, 2, 3]); // (F,H,W,3)
        assert_eq!(frames.dtype(), Dtype::Uint8);
        // Layout (F,H,W,C): channels at w=0 are x=[-2,0.2,0.6]; at w=1 x=[-1,1,2].
        // (x+1)/2 clip[0,1] ·255: -2→0, 0.2→153, 0.6→204, -1→0, 1→255, 2→255.
        let got = frames.as_slice::<u8>();
        assert_eq!(got, &[0, 153, 204, 0, 255, 255]);
    }

    // --- sc-6894 F-004: LTX VAE decode budgeting ------------------------------------------------

    /// Re-derive an LTX plan's peak the way the selector sizes its largest tile.
    fn ltx_chosen_peak(cfg: &TilingConfig, h: i64, w: i64, f: i64) -> f64 {
        let tile_h = cfg.spatial.map(|s| (s.tile_px as i64).min(h)).unwrap_or(h);
        let tile_w = cfg.spatial.map(|s| (s.tile_px as i64).min(w)).unwrap_or(w);
        let tile_f = cfg
            .temporal
            .map(|t| (t.tile_frames as i64).min(f))
            .unwrap_or(f);
        estimated_ltx_decode_peak_gib(f, h, w, tile_f, tile_h, tile_w)
    }

    #[test]
    fn ltx_decode_peak_matches_sweep_anchors() {
        // Real-weight anchors from `vae_decode_sweep.rs` (q8 decoder, 128 GB M-series). The model must
        // be CONSERVATIVE (never below the measured peak — an under-shoot is an OOM) and within ~25 %.
        // (out_f, out_h, out_w, tile_f, tile_h, tile_w, measured_gib)
        let anchors = [
            (25, 512, 512, 25, 512, 512, 4.7561),      // single-pass
            (25, 768, 768, 25, 768, 768, 6.8314),      // single-pass
            (49, 512, 512, 49, 512, 512, 6.1914),      // single-pass (temporal scaling)
            (25, 1024, 1024, 25, 1024, 1024, 10.3689), // single-pass
            (25, 1280, 1280, 25, 1280, 1280, 14.9152), // single-pass (asymptotic slope)
            (25, 1024, 1024, 25, 512, 512, 5.1525),    // tiled @512 px
        ];
        for (of, oh, ow, tf, th, tw, measured) in anchors {
            let est = estimated_ltx_decode_peak_gib(of, oh, ow, tf, th, tw);
            assert!(
                est >= measured,
                "ltx model {est:.2} GiB UNDER-shoots measured {measured} (OOM risk) for tile \
                 [{tf},{th},{tw}] of [{of},{oh},{ow}]"
            );
            assert!(
                est <= measured * 1.25,
                "ltx model {est:.2} GiB over-conservative vs measured {measured} (>25 %)"
            );
        }
    }

    #[test]
    fn ltx_tiling_single_pass_when_small() {
        // A short, low-res LTX clip fits a single-pass decode → no tiling.
        let plan = plan_ltx_tiling(256, 256, 25, 60.0).unwrap();
        assert!(plan.is_none(), "small LTX clip should not tile: {plan:?}");
    }

    #[test]
    fn ltx_tiling_bounds_moderate_res_peak() {
        // 1280×1280×121: single-pass LTX would peak ~66 GB. On a 48 GiB machine the budgeted plan must
        // tile and keep the recomputed peak under the safe budget (the bounded/catchable guarantee).
        let safe = 48.0 * 0.85; // 40.8 GiB
        let cfg = plan_ltx_tiling(1280, 1280, 121, safe)
            .unwrap()
            .expect("moderate-res LTX must tile");
        let peak = ltx_chosen_peak(&cfg, 1280, 1280, 121);
        assert!(
            peak <= safe,
            "ltx chosen peak {peak:.1} GiB over safe {safe:.1}"
        );
    }

    #[test]
    fn ltx_tiling_errors_when_unfittable() {
        // 4K × 257 frames under an 8 GiB budget: the output accumulators (+ fixed floor) alone blow it
        // → a catchable error before the decode, not a SIGKILL.
        let err = plan_ltx_tiling(2160, 3840, 257, 8.0);
        assert!(
            err.is_err(),
            "over-budget LTX decode must error, got {err:?}"
        );
    }
}
