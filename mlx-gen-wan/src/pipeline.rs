//! S4/S5 — the **T2V generation pipeline**: the denoise loop + CFG + VAE decode + frame assembly
//! that turns latents into video. Port of `generate_wan.py`'s `generate_video` — both the
//! single-model dense path ([`denoise`], S4) and the dual-expert MoE path ([`denoise_moe`], S5).
//!
//! This is **reusable machinery**, not a model: the dense loop is exactly what each Wan2.2-A14B MoE
//! expert runs (the MoE adds only the per-step boundary swap) and what the 5B runs (sc-2680, with
//! its z48 VAE). The concrete `Generator::generate` wiring lands in `model.rs`.
//!
//! Shapes are channels-first **`[C, F, H, W]`** (no batch dim) for the latents + scheduler. CFG runs
//! cond + uncond as a **single batched B=2 forward** ([`WanTransformer::forward_cached`]) — the shared
//! latent is patchified once and broadcast across the batch, so each per-step GPU kernel launches once
//! instead of twice (the small-seq win, sc-2853); it stays bit-identical to two B=1 forwards since
//! attention never mixes batch elements. The per-block cross-attention K/V and the RoPE cos/sin are
//! **precomputed once per expert** before the loop (the reference's `prepare_cross_kv` / `prepare_rope`)
//! and reused across all steps, instead of recomputed every forward.

use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{add, concatenate_axis, maximum, minimum, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::tiling::{budgeted_plan, TileCandidates, TilingBudgetError, TilingConfig};
use mlx_gen::{default_seed, CancelFlag, Error, GenerationRequest, Image, Progress, Result};

use crate::scheduler::{compute_sigmas, make_scheduler, SolverKind};
use crate::transformer::WanTransformer;
use crate::vae::WanVae;
use crate::vae22::Wan22Vae;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Align a pixel dimension **down** to a multiple of `patch · vae_stride` (the reference rounds the
/// requested size to the nearest valid grid; sub-tile requests are rejected by `validate`).
pub fn align_dim(value: u32, patch: usize, stride: usize) -> u32 {
    let align = (patch * stride) as u32;
    (value / align) * align
}

/// Resolve the sampler-loop knobs shared **byte-identically** by every Wan generate path (dense 5B,
/// A14B MoE, single- and dual-expert VACE): the step count, scheduler shift, solver kind, and seed
/// (F-010). Each falls back to the config default when the request leaves it unset; an unset sampler
/// maps to UniPC (the `generate_wan.py` default — `validate` has already rejected any unadvertised
/// name), and an unset seed draws a fresh [`default_seed`] so repeated calls vary. The four return
/// types are distinct, so a mis-ordered destructure at a call site is a compile error.
pub fn resolve_sampler_knobs(
    req: &GenerationRequest,
    steps_default: usize,
    shift_default: f32,
) -> (usize, f32, SolverKind, u64) {
    let steps = req.steps.map(|s| s as usize).unwrap_or(steps_default);
    let shift = req.scheduler_shift.unwrap_or(shift_default);
    let kind = SolverKind::from_name(req.sampler.as_deref().unwrap_or("uni_pc"));
    let seed = req.seed.unwrap_or_else(default_seed);
    (steps, shift, kind, seed)
}

/// Latent shape `[z_dim, t_lat, h_lat, w_lat]` for a `frames × H × W` request.
/// `t_lat = (frames − 1) / vae_stride_t + 1`; spatial divide by the vae stride.
pub fn latent_shape(
    frames: usize,
    height: u32,
    width: u32,
    z_dim: usize,
    vae_stride: (usize, usize, usize),
) -> Result<[i32; 4]> {
    // `frames == 0` would underflow `frames - 1` (usize) into a massive `t_lat`; reject it here,
    // co-located with the subtraction, so a config/parse path that bypasses the upstream frame
    // validation gets a clear error rather than a silent wrong latent shape (F-007).
    let frames_minus_1 = frames
        .checked_sub(1)
        .ok_or_else(|| Error::Msg("wan latent_shape: frames must be >= 1".to_string()))?;
    let t_lat = frames_minus_1 / vae_stride.0 + 1;
    let h_lat = height as usize / vae_stride.1;
    let w_lat = width as usize / vae_stride.2;
    Ok([z_dim as i32, t_lat as i32, h_lat as i32, w_lat as i32])
}

/// Transformer sequence length: `ceil(h_lat · w_lat / (patch_h · patch_w) · t_lat)`.
pub fn seq_len(latent: [i32; 4], patch_size: (usize, usize, usize)) -> usize {
    let (_z, t_lat, h_lat, w_lat) = (latent[0], latent[1], latent[2], latent[3]);
    // Exact integer `ceil(h_lat·w_lat·t_lat / (patch_h·patch_w))` — the old f64 ceil was exact only
    // up to 2^24 and could go off-by-one beyond it (F-089).
    let tokens = h_lat as usize * w_lat as usize * t_lat as usize;
    tokens.div_ceil(patch_size.1 * patch_size.2)
}

/// The largest `(width, height)` that fits within `max_area` while preserving the input aspect ratio
/// and staying aligned to the `(dw, dh)` grid (= `patch · vae_stride`). Port of `generate_wan.py`'s
/// `_best_output_size`: it derives the ideal `(ow, oh)` from `√(max_area·ratio)`, then tries
/// width-first and height-first alignment and keeps whichever distorts the aspect ratio less. Applied
/// only when `config.max_area > 0` and the requested area exceeds it (I2V-14B / TI2V-5B cap, 704×1280).
pub fn best_output_size(width: u32, height: u32, dw: u32, dh: u32, max_area: usize) -> (u32, u32) {
    let (w, h, dw_f, dh_f) = (width as f64, height as f64, dw as f64, dh as f64);
    let area = max_area as f64;
    let ratio = w / h;
    let ow = (area * ratio).sqrt();
    let oh = area / ow;

    // Each grid-aligned dimension is clamped to at least one cell (F-030): for a degenerate `max_area`
    // (or an extreme aspect ratio) the `floor(.. / d) * d` could otherwise hit 0, making `area / 0`
    // produce Inf/NaN — a NaN ratio comparison then silently picks a branch or a `(0, …)` size that
    // blows up later in a reshape. For every production input (dims ≫ grid) the clamp is a no-op.
    // Option 1: align width first, derive height from the remaining area. (`int(x // d * d)`.)
    let ow1 = ((ow / dw_f).floor() * dw_f).max(dw_f);
    let oh1 = ((area / ow1 / dh_f).floor() * dh_f).max(dh_f);
    let ratio1 = ow1 / oh1;

    // Option 2: align height first, derive width.
    let oh2 = ((oh / dh_f).floor() * dh_f).max(dh_f);
    let ow2 = ((area / oh2 / dw_f).floor() * dw_f).max(dw_f);
    let ratio2 = ow2 / oh2;

    let dist1 = (ratio / ratio1).max(ratio1 / ratio);
    let dist2 = (ratio / ratio2).max(ratio2 / ratio);
    if dist1 < dist2 {
        (ow1 as u32, oh1 as u32)
    } else {
        (ow2 as u32, oh2 as u32)
    }
}

/// sc-4986 — **pre-flight denoise memory guard.** Estimate the concurrent GPU peak of the
/// DiT-denoise stage (the resident transformer weights + the per-token activation working set of one
/// forward) and return a **catchable** error *before* the expensive text-encode / weight-load when it
/// exceeds this machine's MLX memory budget — instead of letting the OS hard-kill the worker (SIGKILL)
/// or the Metal command buffer abort it (uncaught `kIOGPUCommandBufferCallbackError…` → `terminate`),
/// the two non-recoverable deaths seen in production. Mirrors the z-image sc-4874 `preflight_memory_guard`.
///
/// The staged generate (TE → DiT → VAE each loaded then dropped, see [`crate::model`]) means the DiT
/// stage is the transformer peak; the **14B MoE keeps both experts resident**, so pass the summed
/// expert bytes. `activation_bytes ≈ 72 · batch · tokens · dim` is fit from real Wan2.2 TI2V-5B
/// measurements (peak − weights across L = 1 760 … 32 560, batched B=2; sc-4986). `batch` is 2 with CFG.
///
/// Scope: this guards the **DiT-denoise** stage's memory (OOM / command-buffer abort). It
/// deliberately does *not* encode a wall-time/step-count policy (a long-but-fitting run is the
/// worker's call — sc-4997 / the forward-progress watchdog sc-4984). The z48 VAE-decode peak is a
/// *separate, later* stage (the DiT is freed before the VAE loads), so it has its own budgeted guard
/// in [`auto_tiling_budgeted`] (sc-4998) rather than being summed into this one.
pub fn preflight_denoise_memory_guard(
    model_id: &str,
    dit_resident_bytes: u64,
    tokens: usize,
    dim: usize,
    cfg_enabled: bool,
) -> Result<()> {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let weights_gb = dit_resident_bytes as f64 / GIB;
    let peak_gb = estimated_denoise_peak_gib(dit_resident_bytes, tokens, dim, cfg_enabled);
    let act_gb = peak_gb - weights_gb;
    let budget_gb = get_memory_limit() as f64 / GIB;
    let safe = budget_gb * 0.85;
    if peak_gb > safe {
        return Err(Error::Msg(format!(
            "{model_id}: a denoise step at this resolution/frame-count needs ~{peak_gb:.0} GB \
             (transformer ~{weights_gb:.0} GB resident + ~{act_gb:.0} GB activations for {tokens} \
             attention tokens{}), exceeding this machine's ~{safe:.0} GB safe budget ({budget_gb:.0} \
             GB MLX limit × 0.85). Unmitigated, the OS hard-kills the worker (SIGKILL) or the Metal \
             command buffer aborts (sc-4986). Reduce the resolution or frame count, or load a Q8/Q4 \
             snapshot.",
            if cfg_enabled { ", ×2 for CFG" } else { "" }
        )));
    }
    Ok(())
}

/// Estimated concurrent GPU peak (GiB) of one denoise stage: resident transformer weights + the
/// activation working set of a single forward. `activation ≈ 72 B · batch · tokens · dim`, fit from
/// real Wan2.2 TI2V-5B measurements (sc-4986: peak − weights tracked 0.7→14.4 GiB across
/// L = 1 760…32 560 at batch 2). Pure (no global state) so it is unit-testable against those anchors.
fn estimated_denoise_peak_gib(
    dit_resident_bytes: u64,
    tokens: usize,
    dim: usize,
    cfg_enabled: bool,
) -> f64 {
    const ACT_BYTES_PER_ELEM: f64 = 72.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let batch = if cfg_enabled { 2.0 } else { 1.0 };
    let act = ACT_BYTES_PER_ELEM * batch * tokens as f64 * dim as f64;
    (dit_resident_bytes as f64 + act) / GIB
}

// ===========================================================================================
// sc-4998 — memory-budgeted z48 vae22 decode tiling
// ===========================================================================================
//
// The dense TI2V-5B is welded to the z48 `vae22` decode (it cannot use the lighter 2.1 z16 VAE),
// and once Lightning makes the DiT trivial that decode is ~95 % of wall-clock. The px-threshold
// [`TilingConfig::auto`] picked **512 px** tiles just below its aggressive cutover — peaking at
// **60 GB** on a routine 1024×576×97 video (OOMs a 64 GB Mac) while the *larger* 1280×704×145
// decode peaked at only 12.6 GB with 256 px tiles: it traded memory the wrong way (non-monotonic).
//
// [`auto_tiling_budgeted`] replaces that with a peak-GB target derived from `get_memory_limit()`:
// it picks the *largest* tile whose estimated decode peak stays under the safe budget, so the peak
// is bounded and monotonic in output size, and — being the largest fitting tile — it minimizes the
// overlap-recompute that dominates the aggressive path's wall-clock.

/// Bytes of GPU working-set per **output voxel** (`out_f·out_h·out_w`) for the two terms of a z48
/// `vae22` tiled decode, fit from the real-weight `wedge_sweep.rs` anchors (M5 Max):
///   • 1024×576×97 video, 512 px / 64-frame tiles → **60 GB** peak,
///   • 1280×704×145 video, 256 px / 32-frame tiles → **12.6 GB** peak.
/// The peak splits cleanly into a *fixed* full-output term (the output + blend-weight accumulators
/// plus the per-tile pad/add transients, ≈40 B/voxel) and a *per-tile* term that scales with the
/// largest tile's output volume (≈3800 B/voxel through the decoder's 1024-channel stack). With these
/// two constants both anchors reproduce within ~10 % on the conservative side (the model
/// over-estimates slightly — what a guard wants).
const VAE22_ACCUM_BYTES_PER_VOXEL: f64 = 40.0;
const VAE22_TILE_BYTES_PER_OUT_VOXEL: f64 = 3800.0;
/// Per-tile coefficient for a **bf16** decode (sc-5039). Measured on the same real-weight rig at two
/// tiles of the 1024×576×97 video (cosine 0.99995, no NaN both): 768 px / 64-frame → **79.7 GB**
/// (vs 97.7 GB f32), 640 px / 48-frame → **55.1 GB**. The per-tile term only drops to ~85 % of f32
/// — *not* 50 % — because the `RMS_norm` channel-L2 reduction stays f32 and materializes a full-size
/// f32 temporary of each activation. Calibrated to the **higher** of the two implied coefficients
/// (the 640/48 point) so the estimate never under-shoots a real peak — the 3100 first guess let the
/// selector pick a tile that measured 55.1 GB, just over the 54.4 GB safe line at the 64 GB tier.
/// The fixed accumulator term is unchanged (the blend buffers are f32 either way).
const VAE22_TILE_BYTES_PER_OUT_VOXEL_BF16: f64 = 3400.0;

/// Estimated concurrent GPU peak (GiB) of a z48 `vae22` decode whose **largest tile** spans
/// `tile_f·tile_h·tile_w` output voxels while assembling a `out_f·out_h·out_w` video. `bf16` selects
/// the lighter per-tile coefficient (sc-5039). Pure (no global state) so it is unit-testable against
/// the `wedge_sweep.rs` anchors. A single-pass decode is the special case `tile_* == out_*`; passing
/// a zero tile yields the accumulator-only floor (the unavoidable cost of holding the output).
///
/// No explicit overflow guard: the voxel products are `i64`→`f64`, and the inputs are bounded upstream
/// by the descriptor's `max_size` (1280 px long edge) and the generated frame count, so
/// `out_f·out_h·out_w` stays ~10¹⁰ — many orders below `i64`/`f64` overflow. The model depends on those
/// upstream caps rather than guarding here (sc-6894 Info).
fn estimated_vae22_decode_peak_gib(
    out_f: i64,
    out_h: i64,
    out_w: i64,
    tile_f: i64,
    tile_h: i64,
    tile_w: i64,
    bf16: bool,
) -> f64 {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let tile_coeff = if bf16 {
        VAE22_TILE_BYTES_PER_OUT_VOXEL_BF16
    } else {
        VAE22_TILE_BYTES_PER_OUT_VOXEL
    };
    let out_voxels = (out_f * out_h * out_w) as f64;
    let tile_voxels = (tile_f * tile_h * tile_w) as f64;
    (VAE22_ACCUM_BYTES_PER_VOXEL * out_voxels + tile_coeff * tile_voxels) / GIB
}

/// Candidate spatial tile sizes (output px, multiples of the vae22 ×16 spatial scale, overlap 64).
const VAE22_SPATIAL_PX: [i32; 8] = [768, 640, 512, 448, 384, 320, 256, 192];
/// Candidate temporal tiles `(tile_frames, overlap_frames)` in output frames (matching the preset
/// overlaps: 24 for the longer tiles, 16/8 for the shorter).
const VAE22_TEMPORAL_FR: [(i32, i32); 4] = [(96, 24), (64, 24), (48, 16), (32, 8)];

/// **Memory-budgeted** tiling for the z48 `vae22` decode (sc-4998). Derives a safe peak-GB ceiling
/// from this machine's MLX memory limit (× 0.85, matching [`preflight_denoise_memory_guard`]) and
/// returns the *largest* tile that fits — see [`plan_vae22_tiling`] for the cases and the catchable
/// over-budget error. Caller passes the **output** dimensions (the decoded video size).
pub fn auto_tiling_budgeted(
    height: i32,
    width: i32,
    out_frames: i32,
    bf16: bool,
) -> Result<Option<TilingConfig>> {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let budget_gib = get_memory_limit() as f64 / GIB;
    plan_vae22_tiling(height, width, out_frames, budget_gib * 0.85, bf16)
}

/// Pure tile selector behind [`auto_tiling_budgeted`] (the `safe_gib` ceiling is injected so this is
/// unit-testable without touching the global memory limit). Returns:
///   • `Ok(None)`    — a single-pass decode already fits `safe_gib` (small/short video); the
///                     existing `decode` path runs, so single-pass is reached **only** when safe.
///   • `Ok(Some(c))` — tiling is required; `c` is the largest tile whose estimated peak ≤ `safe_gib`
///                     (largest ⇒ fewest tiles ⇒ least overlap-recompute ⇒ fastest within budget).
///   • `Err(..)`     — even the smallest candidate tile (or the unavoidable full-output
///                     accumulators) exceeds `safe_gib`: a **catchable** error returned before the
///                     decode, so the caller surfaces it instead of the OS hard-killing the worker
///                     (SIGKILL) or the Metal command buffer aborting (`kIOGPUCommandBufferError…`).
fn plan_vae22_tiling(
    height: i32,
    width: i32,
    out_frames: i32,
    safe_gib: f64,
    bf16: bool,
) -> Result<Option<TilingConfig>> {
    // The selector algorithm now lives in gen-core ([`budgeted_plan`], sc-6894) so the LTX and Wan
    // z16 decodes share it. This wrapper supplies only the **vae22-specific** pieces: the candidate
    // tile grid and the [`estimated_vae22_decode_peak_gib`] cost model (its constants were fit to the
    // z48 `wedge_sweep.rs` anchors and are meaningless for any other VAE/backend, so they stay here),
    // then maps the neutral over-budget signal back to the wan-specific message.
    let candidates = TileCandidates {
        spatial_px: &VAE22_SPATIAL_PX,
        spatial_overlap_px: 64,
        temporal: &VAE22_TEMPORAL_FR,
    };
    budgeted_plan(
        height,
        width,
        out_frames,
        safe_gib,
        candidates,
        |of, oh, ow, tf, th, tw| estimated_vae22_decode_peak_gib(of, oh, ow, tf, th, tw, bf16),
    )
    .map_err(|e| wan_budget_error("z48 vae22", width, height, out_frames, e))
}

/// Map gen-core's neutral [`TilingBudgetError`] to a wan-facing message tagged with the VAE `label`
/// (e.g. `"z48 vae22"`, `"z16 vae"`). Shared by the per-VAE budgeted-tiling wrappers so the catchable
/// over-budget wording stays identical across them.
fn wan_budget_error(
    label: &str,
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
            "wan {label} decode: assembling a {width}×{height}×{out_frames} video needs \
             ~{projected_gib:.0} GB just for the output buffers, over this machine's ~{safe_gib:.0} GB \
             safe budget. Reduce the resolution or frame count."
        )),
        TilingBudgetError::SmallestTileExceedsBudget {
            projected_gib,
            safe_gib,
        } => Error::Msg(format!(
            "wan {label} decode: a {width}×{height}×{out_frames} video peaks at ~{projected_gib:.0} GB \
             even with the smallest tile, over this machine's ~{safe_gib:.0} GB safe budget. Reduce \
             the resolution or frame count."
        )),
    }
}

// --- z16 Wan 2.1 VAE decode budgeting (sc-6894 F-009) ---------------------------------------------
//
// The 14B T2V/I2V + VACE decode paths previously used the unbudgeted px-threshold `TilingConfig::auto`
// on the largest-resident models — the same OOM-prone selector the z48 path replaced (sc-4998). These
// route the z16 decode through the shared `budgeted_plan` selector with a z16-specific cost model fit
// from the real `vae16_decode_sweep.rs` anchors (the z16 decoder is non-causal time ×4, spatial ×8).

/// Per-output-voxel cost of the z16 decode's full-output f32 accumulators (`output` [1,3,F,H,W] +
/// `weights` [1,1,F,H,W]) — paid by every tiled plan. Isolated from the `vae16_decode_sweep.rs`
/// anchors (128 GB M-series, f32): the 768²×16 single-pass peak (56.35 GB) minus the same output tiled
/// @384 px (14.46 GB) pins this term at ~57 B/voxel; rounded **up** to 64 for headroom (the model must
/// never under-predict — an under-shoot is an OOM, an over-shoot only tiles slightly more).
const VAE16_ACCUM_BYTES_PER_VOXEL: f64 = 64.0;
/// Per-tile-output-voxel cost of the z16 decoder working set (conv stack + ×8 spatial / ×4 temporal
/// upsample). Fit from the same anchors at ~6355 B/voxel (≈1.7× the z48 `vae22`'s 3800 — the bigger
/// spatial upsample); rounded **up** to 6500. z16 decodes f32 in production, so there is no bf16
/// coefficient (unlike `vae22`, sc-5039).
const VAE16_TILE_BYTES_PER_OUT_VOXEL: f64 = 6500.0;

/// Candidate spatial tile sizes (output px, multiples of the z16 ×8 spatial scale, overlap 64).
const VAE16_SPATIAL_PX: [i32; 8] = [768, 640, 512, 448, 384, 320, 256, 192];
/// Candidate temporal tiles `(tile_frames, overlap_frames)` in output frames.
const VAE16_TEMPORAL_FR: [(i32, i32); 4] = [(96, 24), (64, 24), (48, 16), (32, 8)];

/// Estimated concurrent GPU peak (GiB) of a z16 decode whose largest tile spans `tile_*` output voxels
/// while assembling an `out_*` video. Pure (no global state) → unit-testable against the
/// `vae16_decode_sweep.rs` anchors. Single-pass is the special case `tile_* == out_*`; a zero tile is
/// the accumulator-only floor.
fn estimated_z16_decode_peak_gib(
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
    (VAE16_ACCUM_BYTES_PER_VOXEL * out_voxels + VAE16_TILE_BYTES_PER_OUT_VOXEL * tile_voxels) / GIB
}

/// **Memory-budgeted** tiling for the z16 Wan 2.1 VAE decode (sc-6894 F-009): the z16 analogue of
/// [`auto_tiling_budgeted`], routing the shared [`budgeted_plan`] selector through the z16 cost model.
/// Replaces the unbudgeted [`TilingConfig::auto`] on the 14B T2V/I2V + VACE decode paths. Caller passes
/// the **output** dims (the decoded video size).
pub fn auto_tiling_budgeted_z16(
    height: i32,
    width: i32,
    out_frames: i32,
) -> Result<Option<TilingConfig>> {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let budget_gib = get_memory_limit() as f64 / GIB;
    plan_z16_tiling(height, width, out_frames, budget_gib * 0.85)
}

/// Pure z16 tile selector behind [`auto_tiling_budgeted_z16`] (the `safe_gib` ceiling is injected so it
/// is unit-testable without touching the global memory limit). Supplies the z16 cost model + candidate
/// grid to the shared [`budgeted_plan`]; same `Ok(None)` / `Ok(Some)` / catchable-`Err` contract as
/// [`plan_vae22_tiling`].
fn plan_z16_tiling(
    height: i32,
    width: i32,
    out_frames: i32,
    safe_gib: f64,
) -> Result<Option<TilingConfig>> {
    let candidates = TileCandidates {
        spatial_px: &VAE16_SPATIAL_PX,
        spatial_overlap_px: 64,
        temporal: &VAE16_TEMPORAL_FR,
    };
    budgeted_plan(
        height,
        width,
        out_frames,
        safe_gib,
        candidates,
        estimated_z16_decode_peak_gib,
    )
    .map_err(|e| wan_budget_error("z16 vae", width, height, out_frames, e))
}

/// Classifier-free guidance combine: `uncond + gs·(cond − uncond)`.
fn cfg_combine(cond: &Array, uncond: &Array, gs: f32) -> Result<Array> {
    Ok(add(
        uncond,
        &multiply(&subtract(cond, uncond)?, scalar(gs))?,
    )?)
}

/// Per-generate caches for one transformer/expert, constant across every denoise step: the bf16 RoPE
/// `(cos, sin)` for the (fixed) grid + each block's cross-attention K/V for the (CFG-batched) context.
/// Mirrors the reference's `prepare_rope` / `prepare_cross_kv`, computed once before the loop.
struct StepCache {
    /// Per-block cross-attention `(k, v)`, each `[batch, n, text_len, d]` (bf16).
    cross_kv: Vec<(Array, Array)>,
    cos: Array,
    sin: Array,
    /// Forward batch width: 2 when CFG is on (cond+uncond stacked), else 1.
    batch: usize,
}

/// Build the per-expert [`StepCache`] from the embedded contexts + the (constant) RoPE grid. When CFG
/// is on (`ctx_uncond = Some`) the cond/uncond contexts are stacked on the batch axis so the cross-K/V
/// is `B=2`; otherwise `B=1`. The caches are evaluated once here (the reference's `mx.eval(cross_kv,
/// rope_cos_sin)`) so each per-step graph reuses them instead of recomputing.
fn build_cache(
    transformer: &WanTransformer,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    grid: (usize, usize, usize),
) -> Result<StepCache> {
    let (context_batch, batch) = match ctx_uncond {
        Some(uncond) => (concatenate_axis(&[ctx_cond, uncond], 0)?, 2),
        None => (ctx_cond.clone(), 1),
    };
    let cross_kv = transformer.prepare_cross_kv(&context_batch)?;
    let (cos, sin) = transformer.prepare_rope(grid)?;
    let mut to_eval: Vec<&Array> = vec![&cos, &sin];
    for (k, v) in &cross_kv {
        to_eval.push(k);
        to_eval.push(v);
    }
    mlx_rs::transforms::eval(to_eval)?;
    Ok(StepCache {
        cross_kv,
        cos,
        sin,
        batch,
    })
}

/// One denoise prediction reusing the precomputed [`StepCache`]: a single batched forward yielding
/// `[cond, uncond]`, combined as `uncond + gs·(cond − uncond)` when CFG is on, else the B=1 cond-only
/// forward.
///
/// `y` is the optional I2V channel-concat conditioning `[20, F, H, W]` (mirrors `WanModel.__call__`'s
/// `y`): when `Some`, it is concatenated **onto the channel axis after** the `[16, …]` noise latent —
/// `[noise(16), mask(4), z_video(16)]` → `[36, F, H, W]` — before patchify, exactly the channel order
/// the I2V-14B `patch_embedding` (in_dim 36) was trained on. The DiT prediction stays `out_dim = 16`,
/// so the scheduler step still consumes/produces the 16-channel latent.
fn predict(
    transformer: &WanTransformer,
    latents: &Array,
    t: f32,
    cache: &StepCache,
    guidance: f32,
    y: Option<&Array>,
) -> Result<Array> {
    let x = match y {
        Some(y) => concatenate_axis(&[latents, y], 0)?,
        None => latents.clone(),
    };
    let preds =
        transformer.forward_cached(&x, t, &cache.cross_kv, &cache.cos, &cache.sin, cache.batch)?;
    if cache.batch == 2 {
        // preds[0] = cond (context row 0), preds[1] = uncond (row 1).
        cfg_combine(&preds[0], &preds[1], guidance)
    } else {
        preds
            .into_iter()
            .next()
            .ok_or_else(|| Error::Msg("wan: B=1 forward produced no output".into()))
    }
}

/// The dense denoise loop (single model). `ctx_cond`/`ctx_uncond` are
/// [`WanTransformer::embed_text`] outputs; pass `ctx_uncond = None` for the CFG-disabled B=1 fast
/// path. `init_noise` is `[C, F, H, W]` f32. Returns the denoised latents `[out_dim, F, H, W]`
/// (f32). `on_step(i)` is called after each completed step.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    transformer: &WanTransformer,
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    guidance: f32,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    init_noise: &Array,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let mut sched = make_scheduler(kind, num_train_timesteps);
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();

    // sc-2957: run the DiT's fusable elementwise glue (adaLN affine, gated residual, gated-GELU FFN,
    // RoPE rotation) through `mx.compile` — bit-exact (proven `max|Δ|=0` real + tiny, perf.rs /
    // compile_parity.rs) and ~14% faster/step at production geometry. Scoped + restored on drop by the
    // RAII guard (F-006/F-007) instead of leaking the process-global toggle on.
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();

    // Precompute the RoPE + cross-K/V caches once (grid + context are constant across steps).
    let grid = transformer.patch_grid(init_noise);
    let cache = build_cache(transformer, ctx_cond, ctx_uncond, grid)?;

    let mut latents = init_noise.clone();
    for (i, &t) in timesteps.iter().enumerate() {
        // Honor the engine cancellation contract (sc-5551, the video sibling of chroma's sc-5514):
        // a video render runs minutes, so check before each step. The per-step `eval` below makes
        // this effective — without it MLX's lazy graph defers all compute to VAE decode and this
        // check would pass for every step.
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let pred = predict(transformer, &latents, t, &cache, guidance, None)?;
        latents = sched.step(&pred, &latents)?;
        // Force evaluation each step to bound the lazy graph's peak memory (the reference's
        // per-step `mx.eval(latents)`).
        mlx_rs::transforms::eval([&latents])?;
        on_step(i + 1);
    }
    Ok(latents)
}

/// The dense denoise loop driven by a **curated unified solver** (epic 7114, sc-7121) — the additive
/// fold onto the shared gen-core solver library, alongside the native [`denoise`]. Routes any curated
/// [`mlx_gen::Solver`] (`euler` / `euler_ancestral` / `heun` / `dpmpp_sde` / `ddim` / …) through
/// `mlx_gen::run_flow_sampler` over Wan's own shifted flow-σ schedule ([`compute_sigmas`]).
///
/// Wan's native `unipc`/`dpmpp2m` are a diffusers `FlowDPMSolver`/`FlowUniPC` in flow-SNR space
/// (`λ = log((1−σ)/σ)`), which the gen-core VE-space solvers (`λ = −ln σ`) do NOT reproduce — so the
/// native default stays on [`denoise`] (the N1 default-parity gate), and this path is selected only for
/// the gen-core-only curated solvers. The model is velocity-prediction over the FLOW
/// [`mlx_gen::TimestepConvention::Sigma`] convention, and Wan feeds the DiT the integer-valued timestep
/// `(σ·num_train).trunc()` (the predict closure maps σ → that timestep). `seed` drives the stochastic
/// solvers' per-step noise. Progress / cancel route through `run_flow_sampler`'s per-eval hook.
#[allow(clippy::too_many_arguments)]
pub fn denoise_curated(
    transformer: &WanTransformer,
    sampler_name: &str,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    guidance: f32,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    init_noise: &Array,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();
    let grid = transformer.patch_grid(init_noise);
    let cache = build_cache(transformer, ctx_cond, ctx_uncond, grid)?;
    let sigmas = compute_sigmas(steps, shift, num_train_timesteps);
    let nt = num_train_timesteps as f32;
    mlx_gen::run_flow_sampler(
        Some(sampler_name),
        mlx_gen::TimestepConvention::Sigma,
        &sigmas,
        init_noise.clone(),
        seed,
        cancel,
        on_progress,
        |x, sigma| {
            // Wan feeds the DiT the integer-valued timestep `(σ·num_train).trunc()`, not raw σ.
            let t = (sigma * nt).trunc();
            predict(transformer, x, t, &cache, guidance, None)
        },
    )
}

/// One TI2V prediction with **per-token timesteps**, reusing the precomputed [`StepCache`]: a single
/// batched forward over the per-token timestep vector `t_tokens` `[1, L]` (mask-blend, sc-2680),
/// combined as `uncond + gs·(cond − uncond)` when CFG is on, else the B=1 cond-only forward. Mirrors
/// [`predict`] but routes through [`WanTransformer::forward_tokens_cached`].
fn predict_tokens(
    transformer: &WanTransformer,
    latents: &Array,
    t_tokens: &Array,
    cache: &StepCache,
    guidance: f32,
) -> Result<Array> {
    let preds = transformer.forward_tokens_cached(
        latents,
        t_tokens,
        &cache.cross_kv,
        &cache.cos,
        &cache.sin,
        cache.batch,
    )?;
    if cache.batch == 2 {
        cfg_combine(&preds[0], &preds[1], guidance)
    } else {
        preds
            .into_iter()
            .next()
            .ok_or_else(|| Error::Msg("wan: B=1 forward produced no output".into()))
    }
}

/// The image-conditioned TI2V-5B **mask-blend** denoise loop (port of `generate_wan.py`'s
/// `is_i2v_mask_blend` path, sc-2680). The first latent temporal frame is pinned to the encoded
/// image `z_img` and *frozen*: every step (1) builds the per-token timestep vector `t_tokens =
/// mask_tokens · t` (`0` for the first-frame tokens, so they carry timestep 0), (2) predicts the
/// noise with [`predict_tokens`], (3) scheduler-steps, then (4) re-blends `latents = (1−mask)·z_img +
/// mask·latents` so the first frame stays the conditioning image while the rest denoise.
///
/// `init_latents` is the pre-blended `[C,F,H,W]` start `(1−mask)·z_img + mask·noise`; `z_img` is the
/// VAE-encoded image `[C,1,H,W]` (broadcasts over `F`); `mask` is `[C,F,H,W]` (`0` first frame, `1`
/// rest); `mask_tokens` is `[1,L]` (`0` first-frame tokens, `1` rest), `L` = the patch-token count.
#[allow(clippy::too_many_arguments)]
pub fn denoise_ti2v(
    transformer: &WanTransformer,
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    guidance: f32,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    init_latents: &Array,
    z_img: &Array,
    mask: &Array,
    mask_tokens: &Array,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let mut sched = make_scheduler(kind, num_train_timesteps);
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();

    // sc-2957: compile the DiT's fusable elementwise glue (bit-exact, ~14% faster/step). The per-token
    // modulation shapes differ from T2V's, so `mx.compile` simply re-traces them once. Scoped +
    // restored on drop by the RAII guard (F-006/F-007) instead of leaking the process-global toggle on.
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();

    // Precompute the RoPE + cross-K/V caches once (grid + context constant across steps), exactly like
    // [`denoise`]. The per-token timesteps change each step (the only per-step DiT input besides the
    // latent), so the time embedding is recomputed inside `forward_tokens_cached`.
    let grid = transformer.patch_grid(init_latents);
    let cache = build_cache(transformer, ctx_cond, ctx_uncond, grid)?;

    // `(1−mask)·z_img` — the frozen first-frame content (z_img broadcasts over F); precomputed once.
    let one_minus_mask = subtract(scalar(1.0), mask)?;
    let frozen = multiply(&one_minus_mask, z_img)?;

    let mut latents = init_latents.clone();
    for (i, &t) in timesteps.iter().enumerate() {
        // Honor the engine cancellation contract — check before each (minutes-long) step (sc-5551).
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // Per-token timesteps: 0 for the first-frame tokens (frozen), `t` for the rest. (The 5B's
        // seq_len equals the patch count, so no padding is needed — matches the reference.)
        let t_tokens = multiply(mask_tokens, scalar(t))?;
        let pred = predict_tokens(transformer, &latents, &t_tokens, &cache, guidance)?;
        latents = sched.step(&pred, &latents)?;
        // Re-apply the mask so the first frame stays pinned to the conditioning image.
        latents = add(&frozen, &multiply(mask, &latents)?)?;
        mlx_rs::transforms::eval([&latents])?;
        on_step(i + 1);
    }
    Ok(latents)
}

/// One MoE expert: a full transformer + its own (per-model) embedded contexts + guidance scale.
/// Wan2.2-A14B's "MoE" is two complete checkpoints, not token routing — each carries its own
/// `text_embedding`, so contexts are embedded per expert.
pub struct Expert<'a> {
    pub transformer: &'a WanTransformer,
    /// `embed_text` output for this expert (cond).
    pub ctx_cond: Array,
    /// `embed_text` output for this expert (uncond); `None` ⇒ CFG disabled for this expert.
    pub ctx_uncond: Option<Array>,
    /// This expert's guidance scale (the `low`/`high` of the dual `sample_guide_scale`).
    pub guidance: f32,
}

/// The dual-expert MoE denoise loop (Wan2.2-A14B). Each step picks the **high-noise** expert while
/// the integer timestep is `≥ boundary_timestep` (`config.boundary · num_train_timesteps`, e.g.
/// `0.875 · 1000 = 875`) and the **low-noise** expert below it — switching the transformer, the
/// per-expert contexts, and the per-expert guidance together. Reduces to [`denoise`] when both
/// experts are the same model.
///
/// `y` is the optional I2V-14B channel-concat conditioning `[20, F, H, W]` ([`build_i2v_y`]),
/// concatenated onto each forward's noise latent (see [`predict`]); `None` for T2V. It is constant
/// across steps and shared by both experts (the conditioning doesn't change with the noise level).
#[allow(clippy::too_many_arguments)]
pub fn denoise_moe(
    low: &Expert,
    high: &Expert,
    boundary_timestep: f32,
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    y: Option<&Array>,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let mut sched = make_scheduler(kind, num_train_timesteps);
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();

    // sc-2957: compiled elementwise glue (bit-exact, ~14% faster/step) — see `denoise`. Scoped +
    // restored on drop by the RAII guard (F-006/F-007) instead of leaking the process-global on.
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();

    // Precompute each expert's RoPE + cross-K/V caches once (the grid is shared — the channel-concat
    // `y` doesn't change F/H/W — and each expert's contexts are constant across steps).
    let grid = low.transformer.patch_grid(init_noise);
    let low_cache = build_cache(
        low.transformer,
        &low.ctx_cond,
        low.ctx_uncond.as_ref(),
        grid,
    )?;
    let high_cache = build_cache(
        high.transformer,
        &high.ctx_cond,
        high.ctx_uncond.as_ref(),
        grid,
    )?;

    let mut latents = init_noise.clone();
    for (i, &t) in timesteps.iter().enumerate() {
        // Honor the engine cancellation contract — check before each (minutes-long) step (sc-5551).
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // Boundary swap: high-noise expert at/above the boundary, low-noise below (mirrors
        // `vace::denoise_moe`). `t` is the scheduler's integer-valued timestep and
        // `boundary_timestep = config.boundary · num_train_timesteps`, so this `>=` is an exact
        // integer-vs-constant comparison — the exact boundary deterministically routes to high-noise.
        let (e, cache) = if t >= boundary_timestep {
            (high, &high_cache)
        } else {
            (low, &low_cache)
        };
        let pred = predict(e.transformer, &latents, t, cache, e.guidance, y)?;
        latents = sched.step(&pred, &latents)?;
        mlx_rs::transforms::eval([&latents])?;
        on_step(i + 1);
    }
    Ok(latents)
}

/// The dual-expert MoE denoise loop driven by a **curated unified solver** (epic 7114, sc-7121) — the
/// additive fold onto the shared gen-core solver library, alongside the native [`denoise_moe`]. Same
/// rationale as [`denoise_curated`]: the native `unipc`/`dpmpp2m` (flow-SNR) stay native (N1), and this
/// path serves the gen-core-only curated solvers. The boundary expert swap is applied inside the
/// predict closure (the integer timestep `(σ·num_train).trunc()` is compared to `boundary_timestep`),
/// so a multi-eval solver re-evaluates the correct expert at its intermediate σ.
#[allow(clippy::too_many_arguments)]
pub fn denoise_moe_curated(
    low: &Expert,
    high: &Expert,
    boundary_timestep: f32,
    sampler_name: &str,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    y: Option<&Array>,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();
    let grid = low.transformer.patch_grid(init_noise);
    let low_cache = build_cache(
        low.transformer,
        &low.ctx_cond,
        low.ctx_uncond.as_ref(),
        grid,
    )?;
    let high_cache = build_cache(
        high.transformer,
        &high.ctx_cond,
        high.ctx_uncond.as_ref(),
        grid,
    )?;
    let sigmas = compute_sigmas(steps, shift, num_train_timesteps);
    let nt = num_train_timesteps as f32;
    mlx_gen::run_flow_sampler(
        Some(sampler_name),
        mlx_gen::TimestepConvention::Sigma,
        &sigmas,
        init_noise.clone(),
        seed,
        cancel,
        on_progress,
        |x, sigma| {
            let t = (sigma * nt).trunc();
            let (e, cache) = if t >= boundary_timestep {
                (high, &high_cache)
            } else {
                (low, &low_cache)
            };
            predict(e.transformer, x, t, cache, e.guidance, y)
        },
    )
}

/// Decode denoised latents `[C, F, H, W]` → an RGB video tensor `[F_out, H_out, W_out, 3]` of
/// `uint8` (the reference's `(video + 1)/2 · 255`, clamped). Uses the Wan 2.1 z16 VAE (S2). When
/// `tiling` is `Some`, decodes via [`WanVae::decode_tiled`] (memory-bounded for large/long video;
/// it falls back to a single pass when the config doesn't fire); `None` is always single-pass.
pub fn decode_to_frames(
    vae: &WanVae,
    latents: &Array,
    tiling: Option<&TilingConfig>,
) -> Result<Array> {
    // WanVae::decode[_tiled] expect/return a leading batch dim: [1, 3, F, H, W] in [-1, 1].
    let z = latents.reshape(&prepend1(latents.shape()))?;
    let video = match tiling {
        Some(cfg) => vae.decode_tiled(&z, cfg)?,
        None => vae.decode(&z)?,
    };
    // [1,3,F,H,W] → [F,H,W,3]
    let sh = video.shape(); // [1,3,F,H,W]
    let (f, h, w) = (sh[2], sh[3], sh[4]);
    let chw = video
        .reshape(&[3, f, h, w])?
        .transpose_axes(&[1, 2, 3, 0])?; // [F,H,W,3]
                                         // [-1,1] → [0,255] uint8
    let scaled = multiply(&add(&chw, scalar(1.0))?, scalar(127.5))?;
    let clamped = minimum(&maximum(&scaled, scalar(0.0))?, scalar(255.0))?;
    Ok(clamped.as_dtype(mlx_rs::Dtype::Uint8)?)
}

/// Decode denoised z48 latents `[C, F, H, W]` → an RGB video tensor `[F_out, H_out, W_out, 3]` of
/// `uint8` via the Wan **2.2** z48 [`Wan22Vae`] (sc-2680). The vae22 decoder is **channels-last** and
/// emits `[1, F', 16H, 16W, 3]` in `[-1, 1]` directly (no `[1,3,F,H,W]` transpose, unlike the z16
/// [`decode_to_frames`]); this drops the batch axis and maps `(v+1)/2·255` clamped. `tiling` →
/// [`Wan22Vae::decode_tiled`] (memory-bounded); `None` is single-pass.
pub fn decode_to_frames_22(
    vae: &Wan22Vae,
    latents: &Array,
    tiling: Option<&TilingConfig>,
) -> Result<Array> {
    let video = match tiling {
        Some(cfg) => vae.decode_tiled(latents, cfg)?,
        None => vae.decode(latents)?,
    };
    // [1, F', H', W', 3] → [F', H', W', 3]; [-1,1] → [0,255] uint8.
    let sh = video.shape();
    let (f, h, w) = (sh[1], sh[2], sh[3]);
    let frames = video.reshape(&[f, h, w, 3])?;
    let scaled = multiply(&add(&frames, scalar(1.0))?, scalar(127.5))?;
    let clamped = minimum(&maximum(&scaled, scalar(0.0))?, scalar(255.0))?;
    Ok(clamped.as_dtype(mlx_rs::Dtype::Uint8)?)
}

/// Split a `[F, H, W, 3]` `uint8` video tensor (the [`decode_to_frames`] output) into one
/// [`Image`] per frame. The tensor is transpose-strided, so a raw `as_slice` would read the
/// physical (pre-transpose) buffer — `reshape` first re-materializes it in logical C-order, then we
/// chunk the contiguous bytes `H·W·3` at a time (see [[mlx_rs_as_slice_physical_buffer]]).
pub fn frames_to_images(frames_u8: &Array) -> Result<Vec<Image>> {
    let sh = frames_u8.shape(); // [F, H, W, 3]
    let (f, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let total: i32 = f * h * w * c;
    let flat = frames_u8.reshape(&[total])?; // materialize logical NHWC order
    let bytes = flat.as_slice::<u8>();
    let per = (h * w * c) as usize;
    let mut out = Vec::with_capacity(f as usize);
    for i in 0..f as usize {
        out.push(Image {
            width: w as u32,
            height: h as u32,
            pixels: bytes[i * per..(i + 1) * per].to_vec(),
        });
    }
    Ok(out)
}

/// `[d0, d1, ...]` → `[1, d0, d1, ...]` (prepend a batch axis).
fn prepend1(shape: &[i32]) -> Vec<i32> {
    let mut s = vec![1];
    s.extend_from_slice(shape);
    s
}

// ===========================================================================================
// I2V-14B channel-concat conditioning (port of `generate_wan.py`'s `is_i2v_channel_concat` setup)
// ===========================================================================================

/// Python `round()` — round half to **even** (banker's rounding), matching `round(img.width * scale)`
/// in the reference's image preprocessing. (Rust `f64::round` rounds half away from zero, which would
/// differ on exact `.5` derived sizes.)
fn py_round(x: f64) -> usize {
    let floor = x.floor();
    let frac = x - floor;
    // Round up on frac > 0.5, or on an exact tie (frac == 0.5) when `floor` is odd (→ even).
    let round_up = frac > 0.5 || (frac == 0.5 && (floor as i64) % 2 != 0);
    (if round_up { floor + 1.0 } else { floor }) as usize
}

/// Preprocess an I2V conditioning image to `[3, height, width]` f32 in `[-1, 1]` (CHW), matching the
/// reference's inline pipeline: **cover-fit** LANCZOS resize (`scale = max(W/iw, H/ih)`, new dims
/// `round(·)`), **center-crop** to the target, then `px/255·2 − 1`. The resize is the core PIL-exact
/// fixed-point integer LANCZOS ([`resize_lanczos_u8`]), so it's bit-identical to PIL's `Image.LANCZOS`.
pub fn preprocess_i2v_image(image: &Image, width: u32, height: u32) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (width as usize, height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(Error::Msg(format!(
            "i2v image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    // Cover-fit: scale so the image covers the target, then round to integer dims (PIL `round`).
    let scale = (tw as f64 / iw as f64).max(th as f64 / ih as f64);
    let nw = py_round(iw as f64 * scale).max(tw);
    let nh = py_round(ih as f64 * scale).max(th);
    let resized: Vec<f32> = if (nh, nw) == (ih, iw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, nh, nw)
    };
    // Center-crop the (integer-valued) resized HWC buffer to (th, tw), then normalize → CHW [-1,1].
    let x1 = (nw - tw) / 2;
    let y1 = (nh - th) / 2;
    let mut chw = vec![0f32; 3 * th * tw];
    let plane = th * tw;
    for yy in 0..th {
        for xx in 0..tw {
            let src = ((y1 + yy) * nw + (x1 + xx)) * 3;
            for c in 0..3 {
                chw[c * plane + yy * tw + xx] = 2.0 * (resized[src + c] / 255.0) - 1.0;
            }
        }
    }
    Ok(Array::from_slice(&chw, &[3, th as i32, tw as i32]))
}

/// The I2V-14B 4-channel temporal mask `[4, T_lat, h_lat, w_lat]` (f32): `1.0` for the first latent
/// temporal frame (all 4 channels, all spatial), `0.0` elsewhere. The reference builds this via a
/// `ones`/`zeros` → `repeat(first,4)` → `reshape(·,T_lat,4,·,·)` → `transpose` dance over the
/// `[1, F, h_lat, w_lat]` per-frame mask (first frame 1, rest 0); the result is exactly this pattern
/// (the per-frame mask collapses to "the first 4 of `F+3` temporal slots", which is latent frame 0).
fn build_i2v_mask(t_lat: usize, h_lat: usize, w_lat: usize) -> Array {
    let plane = h_lat * w_lat;
    let mut data = vec![0f32; 4 * t_lat * plane];
    for c in 0..4 {
        let base = c * t_lat * plane; // temporal index 0 of channel c
        for p in 0..plane {
            data[base + p] = 1.0;
        }
    }
    Array::from_slice(&data, &[4, t_lat as i32, h_lat as i32, w_lat as i32])
}

/// Build the I2V-14B channel-concat conditioning `y = [mask(4), z_video(16)]` → `[20, T_lat, h_lat,
/// w_lat]` (f32). Port of `generate_wan.py`'s `is_i2v_channel_concat` branch: a conditioning video
/// (first frame = the preprocessed image, the remaining `frames−1` zero) is encoded by the 2.1 z16
/// `WanVae` → `z_video [16, T_lat, …]`, and concatenated under the temporal mask. `vae` must carry
/// encoder weights. The result is `Some(y)` fed to [`denoise_moe`].
pub fn build_i2v_y(
    vae: &WanVae,
    image: &Image,
    frames: usize,
    height: u32,
    width: u32,
    vae_stride: (usize, usize, usize),
) -> Result<Array> {
    let (h, w) = (height as i32, width as i32);
    // `frames == 0` would make both the `frames − 1` zero-pad count (negative i32) and the `t_lat`
    // subtraction (usize underflow) bogus; reject it up front (F-007).
    let frames_minus_1 = frames
        .checked_sub(1)
        .ok_or_else(|| Error::Msg("wan build_i2v_y: frames must be >= 1".to_string()))?;
    // Conditioning video [3, F, H, W]: first frame = image, rest zeros.
    let first = preprocess_i2v_image(image, width, height)?.reshape(&[3, 1, h, w])?;
    let rest = Array::zeros::<f32>(&[3, frames_minus_1 as i32, h, w])?;
    let video = concatenate_axis(&[&first, &rest], 1)?; // [3, F, H, W]

    // VAE-encode → [1, 16, T_lat, h_lat, w_lat], drop the batch axis → [16, T_lat, h_lat, w_lat].
    let z_video = vae.encode(&video.reshape(&[1, 3, frames as i32, h, w])?)?;
    let z_video = z_video.reshape(&z_video.shape()[1..])?;

    let t_lat = frames_minus_1 / vae_stride.0 + 1;
    let h_lat = height as usize / vae_stride.1;
    let w_lat = width as usize / vae_stride.2;
    let mask = build_i2v_mask(t_lat, h_lat, w_lat);

    Ok(concatenate_axis(&[&mask, &z_video], 0)?)
}

// ===========================================================================================
// TI2V-5B mask-blend conditioning (port of `generate_wan.py`'s `is_i2v_mask_blend` setup + i2v_utils)
// ===========================================================================================

/// Preprocess a TI2V conditioning image to **channels-last** `[1, 1, height, width, 3]` f32 in
/// `[-1, 1]` (batch + temporal dims), the layout the z48 [`Wan22Vae::encode`] consumes. Reuses the
/// PIL-exact cover-fit LANCZOS + center-crop pipeline of [`preprocess_i2v_image`] (which returns CHW),
/// then moves channels last + adds the batch/temporal axes. Mirrors `i2v_utils.preprocess_image`.
pub fn preprocess_ti2v_image(image: &Image, width: u32, height: u32) -> Result<Array> {
    let chw = preprocess_i2v_image(image, width, height)?; // [3, H, W]
    Ok(chw
        .transpose_axes(&[1, 2, 0])?
        .expand_dims(0)?
        .expand_dims(0)?) // [1, 1, H, W, 3]
}

/// Build the TI2V-5B mask-blend tensors (port of `i2v_utils.build_i2v_mask`):
///  - `mask` `[z, T_lat, h_lat, w_lat]` (f32): `0.0` for the first latent temporal frame (all
///    channels/spatial), `1.0` elsewhere — the latent the first frame is frozen, the rest denoise.
///  - `mask_tokens` `[1, L]` (f32): the channel-0 mask subsampled to the patch grid (`0.0` for the
///    first-frame tokens, `1.0` for the rest), `L` = the DiT patch-token count `(T_lat/pt)·(h_lat/ph)·
///    (w_lat/pw)`. Token order is temporal-slowest (matching [`crate::patchify::patchify`]).
pub fn build_ti2v_mask(
    z_dim: usize,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
    patch_size: (usize, usize, usize),
) -> (Array, Array) {
    let plane = h_lat * w_lat;
    // mask: 1.0 everywhere except temporal index 0 (= 0.0).
    let mut mask = vec![1f32; z_dim * t_lat * plane];
    for c in 0..z_dim {
        let base = c * t_lat * plane; // temporal index 0 of channel c
        for p in 0..plane {
            mask[base + p] = 0.0;
        }
    }
    let mask = Array::from_slice(
        &mask,
        &[z_dim as i32, t_lat as i32, h_lat as i32, w_lat as i32],
    );

    // mask_tokens: subsample channel 0 by the patch grid. mask is 0 only at temporal index 0, so a
    // token is 0 iff its source temporal index `t'·pt == 0` (i.e. `t' == 0`) → the first `hg·wg`
    // tokens (temporal-slowest order) are 0, the rest 1.
    let (pt, ph, pw) = patch_size;
    let (tg, hg, wg) = (t_lat / pt, h_lat / ph, w_lat / pw);
    let mut tok = vec![1f32; tg * hg * wg];
    for v in tok.iter_mut().take(hg * wg) {
        *v = 0.0;
    }
    let mask_tokens = Array::from_slice(&tok, &[1, (tg * hg * wg) as i32]);
    (mask, mask_tokens)
}

/// Multi-keyframe generalization of [`build_ti2v_mask`] (epic 3040, Wan-native first_last_frame):
/// pin the latent temporal frames in `indices` (mask `0.0` there, `1.0` elsewhere) instead of only
/// frame 0. first_last_frame = `indices = [0, t_lat-1]`. `mask` `[z, T_lat, h, w]` + `mask_tokens`
/// `[1, L]` (the `hg·wg` tokens of each pinned frame are `0`). Indices must be `< t_lat`; out-of-range
/// indices are ignored (the caller validates). With `indices = [0]` this is exactly `build_ti2v_mask`.
pub fn build_ti2v_multi_mask(
    indices: &[usize],
    z_dim: usize,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
    patch_size: (usize, usize, usize),
) -> (Array, Array) {
    let plane = h_lat * w_lat;
    let mut mask = vec![1f32; z_dim * t_lat * plane];
    for c in 0..z_dim {
        for &t in indices {
            if t >= t_lat {
                continue;
            }
            let base = (c * t_lat + t) * plane;
            for p in 0..plane {
                mask[base + p] = 0.0;
            }
        }
    }
    let mask = Array::from_slice(
        &mask,
        &[z_dim as i32, t_lat as i32, h_lat as i32, w_lat as i32],
    );

    let (pt, ph, pw) = patch_size;
    let (tg, hg, wg) = (t_lat / pt, h_lat / ph, w_lat / pw);
    let mut tok = vec![1f32; tg * hg * wg];
    for &t in indices {
        let tg_idx = t / pt;
        if tg_idx >= tg {
            continue;
        }
        for k in 0..(hg * wg) {
            tok[tg_idx * hg * wg + k] = 0.0;
        }
    }
    let mask_tokens = Array::from_slice(&tok, &[1, (tg * hg * wg) as i32]);
    (mask, mask_tokens)
}

/// Scatter per-keyframe latents into a single `[z, T_lat, h, w]` clean latent for the multi-keyframe
/// mask-blend (epic 3040): each `(z_k, idx)` (with `z_k` shaped `[z, 1, h, w]`) is placed at temporal
/// frame `idx`; every other frame is zeros (those frames have mask `1`, so the zero is never read). The
/// resulting latent feeds [`ti2v_blend_init`] + [`denoise_ti2v`] as the `z_img` per-frame conditioning.
pub fn build_ti2v_keyframe_z(
    frames: &[(Array, usize)],
    z_dim: usize,
    t_lat: usize,
    h_lat: usize,
    w_lat: usize,
) -> Result<Array> {
    let zero = Array::zeros::<f32>(&[z_dim as i32, 1, h_lat as i32, w_lat as i32])?;
    let mut slices: Vec<Array> = (0..t_lat).map(|_| zero.clone()).collect();
    for (z_k, idx) in frames {
        if *idx < t_lat {
            slices[*idx] = z_k.clone();
        }
    }
    let refs: Vec<&Array> = slices.iter().collect();
    Ok(concatenate_axis(&refs, 1)?)
}

/// Blend the encoded image latent with the initial noise for the TI2V start: `latents = (1−mask)·
/// z_img + mask·noise` (port of `generate_wan.py`'s `is_i2v_mask_blend` init). `z_img` is `[z,1,h,w]`
/// (broadcasts over the noise's `T_lat`), `mask`/`noise` are `[z,T_lat,h,w]`.
pub fn ti2v_blend_init(z_img: &Array, mask: &Array, noise: &Array) -> Result<Array> {
    let one_minus_mask = subtract(scalar(1.0), mask)?;
    Ok(add(
        &multiply(&one_minus_mask, z_img)?,
        &multiply(mask, noise)?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denoise_peak_estimate_matches_5b_measurements() {
        // sc-4986 anchors (real 5B, dim 3072, batch 2 / CFG on). bf16 model.safetensors ≈ 11.5 GiB
        // resident; measured total denoise peak at L tokens. The estimate must land within ~2 GiB.
        let weights = (11.5 * 1024.0 * 1024.0 * 1024.0) as u64;
        for (tokens, measured_peak) in [(1760usize, 11.2_f64), (16720, 17.5), (32560, 24.9)] {
            let est = estimated_denoise_peak_gib(weights, tokens, 3072, true);
            assert!(
                (est - measured_peak).abs() < 2.0,
                "L={tokens}: estimate {est:.1} GiB vs measured {measured_peak:.1} GiB"
            );
        }
    }

    #[test]
    fn denoise_peak_scales_with_cfg_and_tokens() {
        let w = 10u64 << 30; // 10 GiB
                             // CFG doubles the activation term.
        let on = estimated_denoise_peak_gib(w, 32560, 3072, true);
        let off = estimated_denoise_peak_gib(w, 32560, 3072, false);
        assert!(
            on - 10.0 > 1.9 * (off - 10.0),
            "CFG should ~2× the activation term"
        );
        // Monotonic in tokens.
        assert!(estimated_denoise_peak_gib(w, 40000, 3072, true) > on);
    }

    #[test]
    fn guard_rejects_over_budget_and_passes_under() {
        use mlx_rs::memory::set_memory_limit;
        // Pin a deterministic budget (32 GiB) so the threshold is exercised on any machine, then
        // restore. set_memory_limit returns the previous value.
        let prev = set_memory_limit(32 << 30);
        // A 14B-class resident (two bf16 experts ≈ 56 GiB) blows the 32 GiB budget on weights alone.
        let res = preflight_denoise_memory_guard("wan_test", 56 << 30, 1024, 5120, true);
        // A tiny model + small request fits comfortably (10 GiB + ~1.5 GiB acts < 27 GiB safe).
        let ok = preflight_denoise_memory_guard("wan_test", 10 << 30, 5280, 3072, true);
        set_memory_limit(prev);
        assert!(
            res.is_err(),
            "56 GiB resident must be rejected under a 32 GiB budget"
        );
        assert!(ok.is_ok(), "11.5 GiB peak must pass under a 32 GiB budget");
    }

    #[test]
    fn resolve_sampler_knobs_falls_back_to_defaults_then_request() {
        // Unset request fields take the config defaults; an unset sampler → UniPC; the seed is some
        // value (drawn fresh). This is the byte-identical inline block the four generate paths used.
        let req = GenerationRequest {
            prompt: "x".into(),
            ..Default::default()
        };
        let (steps, shift, kind, _seed) = resolve_sampler_knobs(&req, 40, 5.0);
        assert_eq!(steps, 40);
        assert_eq!(shift, 5.0);
        assert_eq!(kind, SolverKind::UniPC);

        // Explicit request fields win over the defaults, and the sampler name maps through.
        let req = GenerationRequest {
            prompt: "x".into(),
            steps: Some(12),
            scheduler_shift: Some(3.5),
            sampler: Some("euler".into()),
            seed: Some(99),
            ..Default::default()
        };
        let (steps, shift, kind, seed) = resolve_sampler_knobs(&req, 40, 5.0);
        assert_eq!((steps, shift, kind, seed), (12, 3.5, SolverKind::Euler, 99));
    }

    // --- sc-4998: memory-budgeted z48 vae22 decode tiling ---------------------------------------

    /// The estimated peak of the chosen tiling, recomputed from a returned config + output dims (the
    /// largest tile spans `min(tile_px, dim)` on each spatial axis and `min(tile_frames, f)` frames).
    fn chosen_peak_gib(cfg: &TilingConfig, h: i64, w: i64, f: i64, bf16: bool) -> f64 {
        let tile_h = cfg.spatial.map(|s| (s.tile_px as i64).min(h)).unwrap_or(h);
        let tile_w = cfg.spatial.map(|s| (s.tile_px as i64).min(w)).unwrap_or(w);
        let tile_f = cfg
            .temporal
            .map(|t| (t.tile_frames as i64).min(f))
            .unwrap_or(f);
        estimated_vae22_decode_peak_gib(f, h, w, tile_f, tile_h, tile_w, bf16)
    }

    #[test]
    fn vae22_decode_peak_matches_wedge_anchors() {
        // sc-4998 f32 anchors (real 5B z48 vae22, M5 Max): the model must reproduce both within ~10 %.
        // A: 1024×576×97 video, 512 px / 64-frame tile → 60 GB.
        let a = estimated_vae22_decode_peak_gib(97, 576, 1024, 64, 512, 512, false);
        assert!((a - 60.0).abs() < 6.0, "anchor A estimate {a:.1} GiB vs 60");
        // B: 1280×704×145 video, 256 px / 32-frame tile → 12.6 GB.
        let b = estimated_vae22_decode_peak_gib(145, 704, 1280, 32, 256, 256, false);
        assert!(
            (b - 12.6).abs() < 2.0,
            "anchor B estimate {b:.1} GiB vs 12.6"
        );
        // sc-5039 bf16 anchors (real-weight, 1024×576×97): 768 px/64 f → 79.7 GB, 640 px/48 f →
        // 55.1 GB. bf16 must estimate below f32 and stay **conservative** (never below the measured
        // peak — the guard must not under-shoot). Tile dims use the selector's nominal frame count.
        let bf16_768 = estimated_vae22_decode_peak_gib(97, 576, 1024, 64, 576, 768, true);
        let f32_768 = estimated_vae22_decode_peak_gib(97, 576, 1024, 64, 576, 768, false);
        assert!(bf16_768 < f32_768, "bf16 peak must be below f32");
        assert!(
            bf16_768 >= 79.7,
            "bf16 768/64 estimate {bf16_768:.1} under-shoots 79.7"
        );
        let bf16_640 = estimated_vae22_decode_peak_gib(97, 576, 1024, 48, 576, 640, true);
        assert!(
            bf16_640 >= 55.1,
            "bf16 640/48 estimate {bf16_640:.1} under-shoots 55.1"
        );
    }

    #[test]
    fn vae22_tiling_single_pass_when_small() {
        // A short, low-res clip fits a single-pass decode comfortably → no tiling.
        let plan = plan_vae22_tiling(256, 256, 33, 40.0, false).unwrap();
        assert!(
            plan.is_none(),
            "small clip should not need tiling: {plan:?}"
        );
    }

    #[test]
    fn vae22_tiling_bounds_moderate_res_peak() {
        // The regression: 1024×576×97 on a 64 GiB machine. The px-threshold `auto` chose 512 px tiles
        // → ~60 GB. The budgeted plan must tile, keep the peak under the safe budget, and crucially
        // below the 60 GB blow-up that OOMs a 64 GiB Mac.
        let safe = 64.0 * 0.85; // 54.4 GiB
        let cfg = plan_vae22_tiling(576, 1024, 97, safe, false)
            .unwrap()
            .expect("moderate res must tile");
        let peak = chosen_peak_gib(&cfg, 576, 1024, 97, false);
        assert!(
            peak <= safe,
            "chosen peak {peak:.1} GiB over safe {safe:.1}"
        );
        assert!(
            peak < 60.0,
            "chosen peak {peak:.1} GiB not below the 60 GB blow-up"
        );
    }

    #[test]
    fn vae22_bf16_tiling_stays_under_budget_and_below_f32_peak() {
        // sc-5039: the bf16 plan must keep its chosen tile under the safe budget (the 3100→3400
        // coefficient fix — 3100 let a tile measure 55.1 GB, over the 54.4 GB line), and the bf16
        // peak of a *given* tile is strictly below the f32 peak of the same tile (the headroom that
        // is bf16's only real win — no wall-clock benefit). No claim that it fits a *bigger* tile:
        // at this resolution the candidate grid lands bf16 on the same 384/full-97 tile as f32.
        let safe = 64.0 * 0.85; // 54.4 GiB
        let bf16 = plan_vae22_tiling(576, 1024, 97, safe, true)
            .unwrap()
            .expect("bf16 still needs tiling at 64 GiB");
        let bf16_peak = chosen_peak_gib(&bf16, 576, 1024, 97, true);
        assert!(
            bf16_peak <= safe,
            "bf16 chosen peak {bf16_peak:.1} GiB over safe {safe:.1}"
        );
        // Same tile, bf16 vs f32: bf16 must be the lighter estimate.
        let f32_same = chosen_peak_gib(&bf16, 576, 1024, 97, false);
        assert!(
            bf16_peak < f32_same,
            "bf16 peak {bf16_peak:.1} not below f32 {f32_same:.1} for the same tile"
        );
    }

    #[test]
    fn vae22_tiling_bounds_peak_across_output_sizes() {
        // The px-threshold `auto`'s real defect was *non-monotonic* peak: a moderate 1024×576×97
        // decode spiked to 60 GB while the *larger* 1280×704×145 sat at 12.6 GB — so the dangerous
        // peak hid at a routine resolution. The budgeted plan must hold every size under the safe
        // budget (and below that 60 GB spike), regardless of how output size grows.
        let safe = 64.0 * 0.85; // 54.4 GiB
        for (h, w, f) in [
            (576i64, 1024i64, 49i64),
            (576, 1024, 97),
            (576, 1024, 145),
            (704, 1280, 145),
            (1088, 1920, 97),
        ] {
            let peak = match plan_vae22_tiling(h as i32, w as i32, f as i32, safe, false).unwrap() {
                Some(cfg) => chosen_peak_gib(&cfg, h, w, f, false),
                None => estimated_vae22_decode_peak_gib(f, h, w, f, h, w, false), // single-pass fit
            };
            assert!(
                peak <= safe && peak < 60.0,
                "{w}×{h}×{f}: peak {peak:.1} GiB not bounded under safe {safe:.1} / 60 GB spike"
            );
        }
    }

    #[test]
    fn vae22_tiling_errors_when_unfittable() {
        // A huge video under a tiny budget: even the smallest tile (and the unavoidable output
        // accumulators) cannot fit → a catchable error, not an OOM/abort.
        let err = plan_vae22_tiling(1088, 1920, 241, 8.0, false);
        assert!(err.is_err(), "over-budget decode must error, got {err:?}");
    }

    // --- sc-6894 F-009: z16 Wan 2.1 VAE decode budgeting ----------------------------------------

    /// Re-derive a z16 plan's peak the way the selector sizes its largest tile.
    fn z16_chosen_peak(cfg: &TilingConfig, h: i64, w: i64, f: i64) -> f64 {
        let tile_h = cfg.spatial.map(|s| (s.tile_px as i64).min(h)).unwrap_or(h);
        let tile_w = cfg.spatial.map(|s| (s.tile_px as i64).min(w)).unwrap_or(w);
        let tile_f = cfg
            .temporal
            .map(|t| (t.tile_frames as i64).min(f))
            .unwrap_or(f);
        estimated_z16_decode_peak_gib(f, h, w, tile_f, tile_h, tile_w)
    }

    #[test]
    fn z16_decode_peak_matches_sweep_anchors() {
        // Real-weight anchors from `vae16_decode_sweep.rs` (128 GB M-series, f32). The model must be
        // CONSERVATIVE (never below the measured peak — an under-shoot is an OOM) and within ~10 %.
        // (out_f, out_h, out_w, tile_f, tile_h, tile_w, measured_gib)
        let anchors = [
            (16, 512, 512, 16, 512, 512, 25.39),   // single-pass
            (16, 768, 768, 16, 768, 768, 56.35),   // single-pass
            (32, 512, 512, 32, 512, 512, 50.12),   // single-pass (temporal scaling == spatial)
            (16, 768, 768, 16, 384, 384, 14.46),   // tiled @384 px
            (16, 1024, 1024, 16, 512, 512, 25.66), // tiled @512 px
        ];
        for (of, oh, ow, tf, th, tw, measured) in anchors {
            let est = estimated_z16_decode_peak_gib(of, oh, ow, tf, th, tw);
            assert!(
                est >= measured,
                "z16 model {est:.2} GiB UNDER-shoots measured {measured} (OOM risk) for tile \
                 [{tf},{th},{tw}] of [{of},{oh},{ow}]"
            );
            assert!(
                est <= measured * 1.10,
                "z16 model {est:.2} GiB over-conservative vs measured {measured} (>10 %)"
            );
        }
    }

    #[test]
    fn z16_tiling_single_pass_when_small() {
        // A short, low-res z16 clip fits a single-pass decode → no tiling.
        let plan = plan_z16_tiling(256, 256, 16, 60.0).unwrap();
        assert!(plan.is_none(), "small z16 clip should not tile: {plan:?}");
    }

    #[test]
    fn z16_tiling_bounds_moderate_res_peak() {
        // 1280×720×80 on a 64 GiB machine: single-pass z16 would peak ~450 GB. The budgeted plan must
        // tile and keep the recomputed peak under the safe budget (the bounded/catchable guarantee).
        let safe = 64.0 * 0.85; // 54.4 GiB
        let cfg = plan_z16_tiling(720, 1280, 80, safe)
            .unwrap()
            .expect("moderate-res z16 must tile");
        let peak = z16_chosen_peak(&cfg, 720, 1280, 80);
        assert!(
            peak <= safe,
            "z16 chosen peak {peak:.1} GiB over safe {safe:.1}"
        );
    }

    #[test]
    fn z16_tiling_errors_when_unfittable() {
        // 4K × 240 frames under an 8 GiB budget: the output accumulators alone blow it → a catchable
        // error before the decode, not a SIGKILL.
        let err = plan_z16_tiling(2160, 3840, 240, 8.0);
        assert!(
            err.is_err(),
            "over-budget z16 decode must error, got {err:?}"
        );
    }

    #[test]
    fn vae22_tiling_budgeted_reads_memory_limit() {
        use mlx_rs::memory::set_memory_limit;
        // Exercise the public wrapper end-to-end on a pinned 64 GiB limit (restore after).
        let prev = set_memory_limit(64 << 30);
        let plan = auto_tiling_budgeted(576, 1024, 97, false);
        set_memory_limit(prev);
        let cfg = plan.unwrap().expect("moderate res tiles at 64 GiB");
        assert!(cfg.spatial.is_some() || cfg.temporal.is_some());
    }

    #[test]
    fn align_dim_rounds_down_to_tile() {
        // patch 2 × vae_stride 8 = 16-px grid.
        assert_eq!(align_dim(1280, 2, 8), 1280);
        assert_eq!(align_dim(1281, 2, 8), 1280);
        assert_eq!(align_dim(1295, 2, 8), 1280);
        assert_eq!(align_dim(1296, 2, 8), 1296);
    }

    #[test]
    fn latent_shape_and_seq_len_match_reference_formulas() {
        // 49 frames, 512×512, z16, stride (4,8,8), patch (1,2,2).
        let ls = latent_shape(49, 512, 512, 16, (4, 8, 8)).unwrap();
        assert_eq!(ls, [16, 13, 64, 64]); // (49-1)/4+1=13, 512/8=64
        let sl = seq_len(ls, (1, 2, 2));
        // ceil(64*64/(2*2) * 13) = 1024 * 13 = 13312
        assert_eq!(sl, 13312);
    }

    #[test]
    fn latent_shape_rejects_zero_frames() {
        // frames == 0 must be a clean error, not a usize underflow → huge t_lat (F-007).
        assert!(latent_shape(0, 512, 512, 16, (4, 8, 8)).is_err());
        assert!(latent_shape(1, 512, 512, 16, (4, 8, 8)).is_ok());
    }

    #[test]
    fn cfg_combine_is_uncond_plus_gs_delta() {
        let cond = Array::from_slice(&[2.0f32, 4.0], &[2]);
        let uncond = Array::from_slice(&[1.0f32, 1.0], &[2]);
        let got = cfg_combine(&cond, &uncond, 3.0).unwrap();
        // 1 + 3*(2-1) = 4 ; 1 + 3*(4-1) = 10
        assert_eq!(got.as_slice::<f32>(), &[4.0, 10.0]);
    }

    #[test]
    fn py_round_is_half_to_even() {
        assert_eq!(py_round(19.2), 19);
        assert_eq!(py_round(16.0), 16);
        assert_eq!(py_round(0.5), 0); // half → even (down)
        assert_eq!(py_round(1.5), 2); // half → even (up)
        assert_eq!(py_round(2.5), 2); // half → even (down)
        assert_eq!(py_round(2.500001), 3); // just over half → up
    }

    #[test]
    fn best_output_size_caps_area_and_aligns() {
        // 1280×720 over the I2V/TI2V 704×1280 cap, 16-px grid → width-first wins (less distortion).
        let (w, h) = best_output_size(1280, 720, 16, 16, 704 * 1280);
        assert_eq!((w, h), (1264, 704));
        assert!((w * h) as usize <= 704 * 1280, "must fit within max_area");
        assert_eq!(w % 16, 0);
        assert_eq!(h % 16, 0);
    }

    #[test]
    fn best_output_size_clamps_degenerate_area_to_one_grid_cell() {
        // F-030: a `max_area` smaller than one grid cell would floor a dimension to 0 and divide by
        // it (Inf/NaN, a silent (0, …) result). The guard clamps every dimension to ≥ one cell.
        let (w, h) = best_output_size(8, 8, 16, 16, 100); // ideal ≈ 10×10 < the 16-px cell
        assert!(w >= 16 && h >= 16, "got {w}x{h}");
        assert_eq!((w % 16, h % 16), (0, 0));
        // An extreme aspect ratio (one side floors to 0) is also clamped, not Inf/NaN.
        let (w2, h2) = best_output_size(4096, 1, 16, 16, 16 * 16);
        assert!(w2 >= 16 && h2 >= 16, "got {w2}x{h2}");
    }

    #[test]
    fn build_i2v_mask_is_one_at_first_latent_frame() {
        // [4, T_lat=2, 1, 1]: channel-major, temporal index 0 → 1.0, index 1 → 0.0.
        let m = build_i2v_mask(2, 1, 1);
        assert_eq!(m.shape(), &[4, 2, 1, 1]);
        assert_eq!(m.as_slice::<f32>(), &[1., 0., 1., 0., 1., 0., 1., 0.]);
    }

    #[test]
    fn build_ti2v_mask_freezes_first_frame() {
        // z=2, T_lat=2, h=w=2, patch (1,2,2) → grid (2,1,1) → L=2 tokens.
        let (mask, tokens) = build_ti2v_mask(2, 2, 2, 2, (1, 2, 2));
        assert_eq!(mask.shape(), &[2, 2, 2, 2]);
        // Per channel (8 vals): temporal 0 → 0.0 (4 spatial), temporal 1 → 1.0 (4 spatial).
        assert_eq!(
            mask.as_slice::<f32>(),
            &[0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1.]
        );
        // Token mask: first (t'=0) token frozen (0), second (t'=1) active (1).
        assert_eq!(tokens.shape(), &[1, 2]);
        assert_eq!(tokens.as_slice::<f32>(), &[0., 1.]);
    }

    #[test]
    fn build_ti2v_multi_mask_freezes_first_and_last() {
        // first_last_frame: z=1, T_lat=3, h=w=2, patch (1,2,2) → grid (3,1,1) → 3 tokens.
        // Pin frames [0, 2] (first + last). With indices=[0] it must equal build_ti2v_mask.
        let (mask, tokens) = build_ti2v_multi_mask(&[0, 2], 1, 3, 2, 2, (1, 2, 2));
        assert_eq!(mask.shape(), &[1, 3, 2, 2]);
        // temporal 0 → 0 (4), temporal 1 → 1 (4), temporal 2 → 0 (4).
        assert_eq!(
            mask.as_slice::<f32>(),
            &[0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.]
        );
        // token mask: frame0 + frame2 tokens 0, frame1 token 1.
        assert_eq!(tokens.as_slice::<f32>(), &[0., 1., 0.]);
        // Single-index [0] reproduces build_ti2v_mask exactly.
        let (m1, t1) = build_ti2v_multi_mask(&[0], 2, 2, 2, 2, (1, 2, 2));
        let (m0, t0) = build_ti2v_mask(2, 2, 2, 2, (1, 2, 2));
        assert_eq!(m1.as_slice::<f32>(), m0.as_slice::<f32>());
        assert_eq!(t1.as_slice::<f32>(), t0.as_slice::<f32>());
    }

    #[test]
    fn build_ti2v_keyframe_z_scatters_frames() {
        // z=1, h=w=1, T_lat=3; place A=[7] @0 and B=[9] @2; frame1 = 0.
        let a = Array::from_slice(&[7.0f32], &[1, 1, 1, 1]);
        let b = Array::from_slice(&[9.0f32], &[1, 1, 1, 1]);
        let z = build_ti2v_keyframe_z(&[(a, 0), (b, 2)], 1, 3, 1, 1).unwrap();
        assert_eq!(z.shape(), &[1, 3, 1, 1]);
        assert_eq!(z.as_slice::<f32>(), &[7.0, 0.0, 9.0]);
    }

    #[test]
    fn ti2v_blend_init_freezes_first_frame() {
        // z=1,T=2,h=w=1: mask 0 at t=0, 1 at t=1. z_img=[9] (frame0), noise=[5,7].
        let z_img = Array::from_slice(&[9.0f32], &[1, 1, 1, 1]);
        let mask = Array::from_slice(&[0.0f32, 1.0], &[1, 2, 1, 1]);
        let noise = Array::from_slice(&[5.0f32, 7.0], &[1, 2, 1, 1]);
        let out = ti2v_blend_init(&z_img, &mask, &noise).unwrap();
        // frame0 = z_img (9), frame1 = noise (7).
        assert_eq!(out.as_slice::<f32>(), &[9.0, 7.0]);
    }
}
