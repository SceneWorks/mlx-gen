//! The live SCAIL-2 generation pipeline — the runnable end-to-end denoise loop (sc-5443).
//!
//! Ports `wan/scail.py::SCAIL2Pipeline.generate`: preprocess the reference character (+ optional
//! extra characters) and the driving video into conditioning latents, then run a plain-CFG
//! flow-matching denoise (UniPC / DPM++) over one or more 81-frame **segments** with clean-history
//! continuity, and VAE-decode each segment back to pixels.
//!
//! Reuse map — the heavy components are `mlx-gen-wan`'s (SCAIL-2 *is* Wan2.1-14B I2V): the z16
//! [`WanVae`] (encode/decode), the [`Umt5Encoder`] text encoder, and the flow-matching
//! [`make_scheduler`] (UniPC/DPM++). SCAIL-2's own pieces are the [`Scail2Dit`] forward, the
//! open-CLIP [`ScailClip`] image encode, the 28-channel [`extract_and_compress_mask_to_latent`] mask
//! build, and the [`interpolate`]/[`downsample_half`] resizes — all already parity-gated.
//!
//! Conditioning shapes (latent dims are `vae_stride`-down; `lat_h = H/8`, `lat_w = W/8`):
//!   * reference char → `ref_latent [16,1,lat_h,lat_w]` + `ref_mask_28 [28,1,lat_h,lat_w]` + CLIP
//!     features `[1,257,1280]`.
//!   * driving video (per segment, **half** spatial res) → `pose_latent [16,T,lat_h/2,lat_w/2]` +
//!     `driving_masks [28,T,lat_h/2,lat_w/2]`.
//!   * the noisy target latent `x [16,T,lat_h,lat_w]`; `ref_masks` is `ref_mask_28` padded with a
//!     zero null-noisy-mask over the latent length (`[28,1+T,lat_h,lat_w]`).

use std::path::Path;

use mlx_gen::array::scalar;
use mlx_gen::tiling::TilingConfig;
use mlx_gen::weights::Weights;
use mlx_gen::{AdapterSpec, Error, GenerationOutput, Image, Progress, Quant, Result};
use mlx_gen_wan::{
    frames_to_images, load_tokenizer, make_scheduler, DitMemoryConfig, SolverKind, Umt5Encoder,
    WanVae,
};
use mlx_rs::ops::{add, concatenate_axis, maximum, minimum, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use crate::clip::{ClipVisionConfig, ScailClip};
use crate::config::Scail2Config;
use crate::model::{Scail2Dit, Scail2Inputs};
use crate::preprocess::{extract_and_compress_mask_to_latent, TEMPORAL_STRIDE};
use crate::resize::{clip_preprocess, downsample_half, interpolate, Interp};

/// Wan2.1 flow-matching training horizon (upstream `config.num_train_timesteps`).
const NUM_TRAIN_TIMESTEPS: usize = 1000;
/// VAE-decode temporal-window budget (sc-5681): max output pixel·frames per `decode_tiled` window,
/// so the decode peak stays bounded at every resolution bucket (fewer frames/window as resolution
/// grows). At 832×480/5s a 3.5M budget → 8-frame windows → ~33 GB MLX-active / ~76 GB process
/// footprint (the Metal conv scratch is ~2× the MLX-active), vs. ~139 GB for an un-tiled segment
/// decode. See the per-segment decode for why this is temporal-only rather than `TilingConfig::auto`
/// (a memory bound — `auto`'s combined-plan blend is itself correct, per sc-5690).
const DECODE_TILE_BUDGET_PXFRAMES: i64 = 3_500_000;
/// SCAIL-2 DiT-denoise activation-memory defaults (sc-5681). NOTE: measurement showed the 832×480/5s
/// high-resolution OOM is the **VAE decode** (see the per-segment decode), not the DiT denoise — MLX's
/// `scaled_dot_product_attention` is flash here, so the 40-layer denoise fits even un-chunked. These
/// levers therefore bound the *denoise* peak for headroom + the larger buckets (1280×704) and are the
/// shared-layer "practice" the story carries to Wan/Bernini; they are not what unblocks 832×480.
///
/// - `eval_per_block` — caps the peak at ~one block's activations instead of the whole 40-block lazy
///   graph. Bit-exact, near-zero cost.
/// - `ffn_seq_chunk` — bounds the `[L, ffn_dim]` FFN intermediate (the largest denoise transient).
/// - `attn_query_chunk` — OFF by default (SDPA is flash here, so chunking the query path only adds
///   overhead); available via env for any bucket/model where SDPA falls back to a materialized
///   `[heads, L, L]` score matrix.
///
/// Numerically equivalent to the un-chunked forward (eval is bit-exact; the chunked GEMM differs only
/// by Metal tile-rounding, cosine ≈ 1). Overridable via `MLX_GEN_WAN_*` env
/// (see [`DitMemoryConfig::from_env`]).
const SCAIL2_MEM_DEFAULT: DitMemoryConfig = DitMemoryConfig {
    ffn_seq_chunk: Some(8192),
    attn_query_chunk: None,
    eval_per_block: true,
};
/// Inputs must be divisible by 32: the pose path halves spatially (→ ÷16) before the ÷8 VAE stride,
/// and the 28-channel mask pools 8×, so both the full and half grids must stay integer + even.
const DIM_ALIGN: u32 = 32;

/// One masked character reference (the primary subject or an extra character): an RGB image paired
/// with its color-coded segmentation mask.
pub struct CharacterRef<'a> {
    pub image: &'a Image,
    pub mask: &'a Image,
}

/// A fully-specified SCAIL-2 generation job (the engine-internal form the worker maps a
/// `GenerationRequest` onto). All images are decoded + resized to `(width, height)` here.
pub struct Scail2Job<'a> {
    pub prompt: &'a str,
    pub negative_prompt: &'a str,
    pub width: u32,
    pub height: u32,
    /// The primary character (reference image + color mask).
    pub reference: CharacterRef<'a>,
    /// Extra characters for multi-reference (experimental); each paired with its mask.
    pub additional: Vec<CharacterRef<'a>>,
    /// Driving video frames (the motion source — raw end-to-end or a pose render).
    pub driving_frames: &'a [Image],
    /// Per-frame color-coded driving masks (same count as `driving_frames`).
    pub driving_masks: &'a [Image],
    /// `true` = cross-identity replacement, `false` = animation.
    pub replace_flag: bool,
    pub seed: u64,
    pub steps: usize,
    pub shift: f32,
    pub guidance: f32,
    pub sampler: SolverKind,
    pub fps: u32,
    pub segment_len: usize,
    pub segment_overlap: usize,
}

/// Round a requested dim down to a multiple of [`DIM_ALIGN`] (min one tile).
fn align(value: u32) -> usize {
    (value / DIM_ALIGN).max(1) as usize * DIM_ALIGN as usize
}

/// Decode an `Image` (RGB24 `u8`) → `[3, th, tw]` f32 in `[-1, 1]`, resizing if its native size
/// differs. `mode` is `Bicubic` for photographic images, `Bilinear` for color masks (bounded — avoids
/// the bicubic overshoot that would invent out-of-gamut colors at mask edges).
fn image_to_chw(img: &Image, tw: usize, th: usize, mode: Interp) -> Result<Array> {
    let (iw, ih) = (img.width as usize, img.height as usize);
    if img.pixels.len() != iw * ih * 3 {
        return Err(Error::Msg(format!(
            "scail2: image pixel buffer {} != {iw}x{ih}x3",
            img.pixels.len()
        )));
    }
    let px: Vec<f32> = img.pixels.iter().map(|&p| p as f32 / 127.5 - 1.0).collect();
    let chw = Array::from_slice(&px, &[ih as i32, iw as i32, 3]).transpose_axes(&[2, 0, 1])?; // [3,H,W]
    let nchw = chw.reshape(&[1, 3, ih as i32, iw as i32])?;
    let out = if (ih, iw) != (th, tw) {
        interpolate(&nchw, th, tw, mode)?
    } else {
        nchw
    };
    Ok(out.reshape(&[3, th as i32, tw as i32])?)
}

/// Stack driving frames → `[T, 3, H, W]` (frames-first, the layout `downsample_half` then `c t h w`
/// rearrange expects).
fn stack_frames(frames: &[Image], tw: usize, th: usize) -> Result<Array> {
    let chw: Vec<Array> = frames
        .iter()
        .map(|f| -> Result<Array> {
            Ok(image_to_chw(f, tw, th, Interp::Bicubic)?.reshape(&[1, 3, th as i32, tw as i32])?)
        })
        .collect::<Result<_>>()?;
    let refs: Vec<&Array> = chw.iter().collect();
    Ok(concatenate_axis(&refs, 0)?) // [T,3,H,W]
}

/// Stack per-frame masks → `[3, T, H, W]` (channels-first, the `extract_and_compress` input layout).
fn stack_masks(masks: &[Image], tw: usize, th: usize) -> Result<Array> {
    let chw: Vec<Array> =
        masks
            .iter()
            .map(|m| -> Result<Array> {
                Ok(image_to_chw(m, tw, th, Interp::Bilinear)?
                    .reshape(&[3, 1, th as i32, tw as i32])?)
            })
            .collect::<Result<_>>()?;
    let refs: Vec<&Array> = chw.iter().collect();
    Ok(concatenate_axis(&refs, 1)?) // [3,T,H,W]
}

/// VAE-encode an `[3, T, H, W]` pixel clip (`[-1,1]`) → `[16, T_lat, H/8, W/8]` (drops the batch dim
/// the `WanVae` API carries).
fn vae_encode_cthw(vae: &WanVae, cthw: &Array) -> Result<Array> {
    let s = cthw.shape();
    let z = vae.encode(&cthw.reshape(&[1, s[0], s[1], s[2], s[3]])?)?;
    let zs = z.shape(); // [1,16,T_lat,h,w]
    Ok(z.reshape(&[zs[1], zs[2], zs[3], zs[4]])?)
}

/// `uncond + guidance·(cond − uncond)`.
fn cfg_combine(uncond: &Array, cond: &Array, guidance: f32) -> Result<Array> {
    Ok(add(
        uncond,
        &multiply(&subtract(cond, uncond)?, scalar(guidance))?,
    )?)
}

/// Overwrite the leading `min(history_t, T)` latent frames of `latent [16,T,h,w]` with the clean
/// history (upstream `apply_clean_history`). No-op without history.
fn apply_clean_history(latent: &Array, history: Option<&Array>) -> Result<Array> {
    let Some(h) = history else {
        return Ok(latent.clone());
    };
    let lt = latent.shape()[1];
    let ht = h.shape()[1].min(lt);
    if ht <= 0 {
        return Ok(latent.clone());
    }
    let head = h.take_axis(Array::from_slice(&(0..ht).collect::<Vec<i32>>(), &[ht]), 1)?;
    if ht == lt {
        return Ok(head);
    }
    let tail = latent.take_axis(
        Array::from_slice(&(ht..lt).collect::<Vec<i32>>(), &[lt - ht]),
        1,
    )?;
    Ok(concatenate_axis(&[&head, &tail], 1)?)
}

/// Segment plan over `total` driving frames (upstream `build_segments`): a single VAE-aligned segment
/// when the clip fits, else overlapping `segment_len` windows striding by `len − overlap`.
fn build_segments(total: usize, len: usize, overlap: usize) -> Vec<(usize, usize)> {
    if total <= len {
        let keep = ((total - 1) / TEMPORAL_STRIDE) * TEMPORAL_STRIDE + 1;
        return vec![(0, keep)];
    }
    let mut segs = Vec::new();
    let stride = len - overlap;
    let mut start = 0;
    while start < total {
        let end = start + len;
        if end > total {
            break;
        }
        segs.push((start, end));
        start += stride;
    }
    segs
}

/// `[3, T, H, W]` in `[-1,1]` → `[T, H, W, 3]` `u8` in `[0,255]`.
fn pixels_to_u8(video_cthw: &Array) -> Result<Array> {
    let thwc = video_cthw.transpose_axes(&[1, 2, 3, 0])?; // [T,H,W,3]
    let scaled = multiply(&add(&thwc, scalar(1.0))?, scalar(127.5))?;
    let clamped = minimum(&maximum(&scaled, scalar(0.0))?, scalar(255.0))?;
    Ok(clamped.as_dtype(Dtype::Uint8)?)
}

/// Run the full SCAIL-2 generation for `job`, loading each component from the snapshot `root`.
pub fn generate(
    root: &Path,
    cfg: &Scail2Config,
    job: &Scail2Job,
    quant: Option<Quant>,
    adapters: &[AdapterSpec],
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput> {
    if job.driving_frames.is_empty() {
        return Err(Error::Msg("scail2: a driving video is required".into()));
    }
    if job.driving_masks.len() != job.driving_frames.len() {
        return Err(Error::Msg(format!(
            "scail2: driving_masks ({}) must match driving_frames ({})",
            job.driving_masks.len(),
            job.driving_frames.len()
        )));
    }
    // Partition the inference LoRAs (before the 31 GB DiT load, so the gate below fails fast):
    //   * diff-patch ("lightning") files — full-rank `.diff`/`.diff_b` (+ low-rank factors) that the
    //     residual loader can't consume — are merged *in place* into the dense weights below (sc-5684).
    //   * pure low-rank files (the Bias-Aware DPO LoRA, …) install as forward-time residuals over the
    //     (possibly Q4/Q8) base, the way sc-5451 wired them.
    let mut diff_patch: Vec<&AdapterSpec> = Vec::new();
    let mut residual: Vec<AdapterSpec> = Vec::new();
    for spec in adapters {
        if crate::lora::has_diff_patch_keys(&spec.path)? {
            diff_patch.push(spec);
        } else {
            residual.push(spec.clone());
        }
    }
    // The in-place diff-patch merge folds dense deltas into the weights, so it needs the DENSE (bf16)
    // snapshot — a pre-quantized-on-disk DiT carries packed u32 weights that can't take a dense delta.
    // Fail loudly rather than silently dropping the lightning patch (sc-5684/sc-5445).
    if !diff_patch.is_empty() {
        if let Some(q) = cfg.wan.quantization {
            return Err(Error::Msg(format!(
                "scail2: a lightx2v diff-patch lightning LoRA needs the DENSE (bf16) snapshot, but \
                 this one is pre-quantized on disk (Q{}). Point the loader at the bf16 snapshot — \
                 load-time Q4/Q8 still applies *after* the merge, and the lightning recipe is a \
                 speed lever (8 steps, CFG off) on activation-bound 480p memory where pre-packed Q4 \
                 weights help little (sc-5684/sc-5445).",
                q.bits
            )));
        }
    }
    let (tw, th) = (align(job.width), align(job.height));
    let cfg_disabled = job.guidance <= 1.0;

    // --- decode + resize all pixel inputs to (tw, th) ---
    let ref_chw = image_to_chw(job.reference.image, tw, th, Interp::Bicubic)?; // [3,H,W]
    let ref_mask_chw = image_to_chw(job.reference.mask, tw, th, Interp::Bilinear)?; // [3,H,W]
    let driving = stack_frames(job.driving_frames, tw, th)?; // [T,3,H,W]
    let driving_mask = stack_masks(job.driving_masks, tw, th)?; // [3,T,H,W]

    // --- VAE (kept resident: per-segment pose encode + final decode) ---
    let vae = {
        let w = Weights::from_file(root.join("vae.safetensors"))?;
        WanVae::from_weights(&w)?
    };

    // Reference char latent + its 28-ch mask (1 latent frame).
    let ref_latent = vae_encode_cthw(&vae, &ref_chw.reshape(&[3, 1, th as i32, tw as i32])?)?;
    let ref_mask_28 = extract_and_compress_mask_to_latent(
        &ref_mask_chw.reshape(&[3, 1, th as i32, tw as i32])?,
        TEMPORAL_STRIDE,
    )?;
    let lat_h = ref_latent.shape()[2];
    let lat_w = ref_latent.shape()[3];

    // Extra characters (multi-reference): cat latents + masks on the frame axis.
    let (additional_ref_latent, additional_ref_masks) = if job.additional.is_empty() {
        (None, None)
    } else {
        let mut lats = Vec::new();
        let mut masks = Vec::new();
        for c in &job.additional {
            let img = image_to_chw(c.image, tw, th, Interp::Bicubic)?
                .reshape(&[3, 1, th as i32, tw as i32])?;
            lats.push(vae_encode_cthw(&vae, &img)?);
            let mk = image_to_chw(c.mask, tw, th, Interp::Bilinear)?
                .reshape(&[3, 1, th as i32, tw as i32])?;
            masks.push(extract_and_compress_mask_to_latent(&mk, TEMPORAL_STRIDE)?);
        }
        let lr: Vec<&Array> = lats.iter().collect();
        let mr: Vec<&Array> = masks.iter().collect();
        (
            Some(concatenate_axis(&lr, 1)?),
            Some(concatenate_axis(&mr, 1)?),
        )
    };

    // --- UMT5 text encode (loaded → used → freed) ---
    let (context, context_null) = {
        let tok = load_tokenizer(root.join("tokenizer.json"), cfg.wan.text_len)?;
        let w = Weights::from_file(root.join("t5_encoder.safetensors"))?;
        let enc = Umt5Encoder::from_weights(&w, &cfg.wan)?;
        let c = enc.encode(&tok, job.prompt)?;
        let cn = if cfg_disabled {
            c.clone()
        } else {
            enc.encode(&tok, job.negative_prompt)?
        };
        mlx_rs::transforms::eval([&c, &cn])?;
        (c, cn)
    };

    // --- CLIP reference-image features (loaded → used → freed) ---
    let clip_fea = {
        let w = Weights::from_file(root.join("clip.safetensors"))?;
        let clip = ScailClip::from_weights(&w, &ClipVisionConfig::vit_h_14())?;
        let pixel = clip_preprocess(&ref_chw.reshape(&[1, 3, th as i32, tw as i32])?, 224)?;
        let f = clip.encode(&pixel)?;
        mlx_rs::transforms::eval([&f])?;
        f
    };

    // --- DiT (bf16 production compute; optional Q4/Q8 load-time quant, sc-5445) ---
    let dit = {
        let mut w = Weights::from_file(root.join("dit.safetensors"))?;
        // sc-5684: merge any lightx2v diff-patch ("lightning") LoRA(s) into the dense weights *before*
        // building + quantizing — `.diff`/`.diff_b`/low-rank factors fold uniformly into the raw
        // `{stem}.weight`/`.bias` map (handling the qk-norms, the affine LayerNorms, every bias, and
        // the full-rank `head.head` delta the residual host can't reach), with the in_dim-36 vanilla
        // `patch_embedding` deliberately skipped (shape-incompatible) and surfaced loudly.
        if !diff_patch.is_empty() {
            let report = crate::lora::merge_diff_patch_adapters(&mut w, &diff_patch)?;
            crate::lora::report_outcome(&report, crate::pipeline::MODEL_ID)?;
        }
        let mut d = Scail2Dit::from_weights(&w, cfg)?;
        // f32 matmul compute (sc-5681). The bf16 path overflows to NaN at long sequences: traced to a
        // bf16 quantized-matmul (the self-attention `o` projection is the first to blow up — its
        // SDPA input is clean at max ~15 — but f32-ing only `o` just moves the overflow to the next
        // projection), reproduced at L≈42k (832×480·5s) and clean in f32. bf16 was never parity-gated
        // (the gates run f32); f32 is the validated-correct path and the Q4 weights still bound the
        // weight memory. `SCAIL2_COMPUTE_BF16=1` opts back into the (fast, but high-res-unsafe) bf16
        // path for experiments. A mixed-precision pass to recover bf16 speed at high res is a follow-up.
        let bf16 = std::env::var("SCAIL2_COMPUTE_BF16").is_ok_and(|v| v == "1");
        d.set_compute_dtype(if bf16 {
            Dtype::Bfloat16
        } else {
            Dtype::Float32
        });
        // sc-5681: bound the per-step activation peak so the high-resolution buckets don't OOM.
        d.set_memory_config(DitMemoryConfig::from_env(SCAIL2_MEM_DEFAULT));
        // Quantize the attention + FFN Linears in place (Q4 default in the SceneWorks worker). The
        // packed Q4/Q8 weights are what stays resident; the bf16 source is freed in `quantize`.
        if let Some(q) = quant {
            d.quantize(q.bits(), None)?;
        }
        // Install any pure low-rank inference LoRA(s) — the Bias-Aware DPO refinement LoRA, a
        // diff-stripped lightning subset, … — as forward-time residuals over the (now possibly Q4/Q8)
        // base (sc-5451). Adapters are independent of the base quantization, so they stack cleanly on
        // the packed weights; applying them *after* `quantize` keeps the residual a dense add over the
        // quantized matmul. The family-agnostic strict installer (the Z-Image / Qwen path) resolves the
        // diffusers / PEFT / kohya / LoKr keys against SCAIL-2's raw module names and errors — never
        // silently drops — on a format/prefix mismatch or an unmatched target. (The diff-patch
        // lightning files were already merged into the dense weights above, sc-5684.)
        if !residual.is_empty() {
            mlx_gen::adapters::loader::apply_adapters_strict(
                &mut d,
                &residual,
                crate::pipeline::MODEL_ID,
            )?;
        }
        d
    };

    let segments = build_segments(
        job.driving_frames.len(),
        job.segment_len,
        job.segment_overlap,
    );
    let mut out_pieces: Vec<Array> = Vec::new();
    let mut prev_history_pixel: Option<Array> = None;

    for (seg_idx, &(seg_start, seg_end)) in segments.iter().enumerate() {
        // Pose latent (half spatial res) + driving mask for this segment.
        let pose_seg = driving.take_axis(
            Array::from_slice(
                &(seg_start as i32..seg_end as i32).collect::<Vec<i32>>(),
                &[(seg_end - seg_start) as i32],
            ),
            0,
        )?; // [T,3,H,W]
        let pose_half = downsample_half(&pose_seg)?; // [T,3,H/2,W/2]
        let pose_cthw = pose_half.transpose_axes(&[1, 0, 2, 3])?; // [3,T,H/2,W/2]
        let pose_latent = vae_encode_cthw(&vae, &pose_cthw)?; // [16,T_lat,h/2,w/2]
        let lat_t = pose_latent.shape()[1];

        let dmask_seg = driving_mask.take_axis(
            Array::from_slice(
                &(seg_start as i32..seg_end as i32).collect::<Vec<i32>>(),
                &[(seg_end - seg_start) as i32],
            ),
            1,
        )?; // [3,T,H,W]
        let dmask_half = downsample_half(&dmask_seg)?; // [3,T,H/2,W/2]
        let driving_masks = extract_and_compress_mask_to_latent(&dmask_half, TEMPORAL_STRIDE)?;

        // ref_masks = ref_mask_28 ++ zero null-noisy-mask over the latent length.
        let null_noisy = Array::zeros::<f32>(&[28, lat_t, lat_h, lat_w])?;
        let ref_masks = concatenate_axis(&[&ref_mask_28, &null_noisy], 1)?; // [28,1+T,h,w]

        // Clean-history latent + i2v mask for segments after the first.
        let (history_latent, history_mask) = match &prev_history_pixel {
            Some(hp) if seg_idx > 0 => {
                let hl = vae_encode_cthw(&vae, hp)?; // [16,h_t,h,w]
                let h_t = hl.shape()[1].min(lat_t);
                let ones = Array::ones::<f32>(&[4, h_t, lat_h, lat_w])?;
                let hm = if h_t < lat_t {
                    let z = Array::zeros::<f32>(&[4, lat_t - h_t, lat_h, lat_w])?;
                    concatenate_axis(&[&ones, &z], 1)?
                } else {
                    ones
                };
                (Some(hl), Some(hm))
            }
            _ => (None, None),
        };

        // Seeded init noise (per-segment key; exact RNG values differ mlx-rs vs torch, as expected).
        let key = random::key(job.seed.wrapping_add(seg_idx as u64))?;
        let noise = random::normal::<f32>(&[16, lat_t, lat_h, lat_w], None, None, Some(&key))?;

        // --- denoise (plain CFG, clean-history pinned) ---
        let mut sched = make_scheduler(job.sampler, NUM_TRAIN_TIMESTEPS);
        sched.set_timesteps(job.steps, job.shift);
        let timesteps: Vec<f32> = sched.timesteps().to_vec();
        let total = timesteps.len() as u32;

        let mut latent = apply_clean_history(&noise, history_latent.as_ref())?;
        for (i, &t) in timesteps.iter().enumerate() {
            let x = apply_clean_history(&latent, history_latent.as_ref())?;
            let mut inp = Scail2Inputs {
                x: &x,
                ref_latent: &ref_latent,
                ref_masks: &ref_masks,
                pose_latent: &pose_latent,
                driving_masks: &driving_masks,
                history_mask: history_mask.as_ref(),
                additional_ref_latent: additional_ref_latent.as_ref(),
                additional_ref_masks: additional_ref_masks.as_ref(),
                clip_fea: &clip_fea,
                context: &context,
                t,
                replace_flag: job.replace_flag,
            };
            let pred_cond = dit.forward(&inp)?;
            let pred = if cfg_disabled {
                pred_cond
            } else {
                inp.context = &context_null;
                let pred_uncond = dit.forward(&inp)?;
                cfg_combine(&pred_uncond, &pred_cond, job.guidance)?
            };
            latent = sched.step(&pred, &latent)?;
            latent = apply_clean_history(&latent, history_latent.as_ref())?;
            mlx_rs::transforms::eval([&latent])?;
            on_progress(Progress::Step {
                current: (i + 1) as u32,
                total,
            });
        }
        // sc-5681: release the MLX buffer cache before the VAE decode. The decode is the heaviest
        // phase (Metal conv scratch ~2× its MLX-active), so the cache retained from preprocessing +
        // denoise (~35 GB) would otherwise stack under it — measured 832×480/5s footprint 111 → 76 GB.
        mlx_rs::memory::clear_cache();

        // --- decode this segment → pixels; stitch + carry history ---
        // sc-5681: tile the VAE decode. Decoding the whole `[3, 4·T_lat, 480, 832]` segment in one
        // pass is the high-res activation peak (≈139 GB → OOMs a 128 GB Mac), NOT the DiT denoise
        // (which fits — MLX SDPA is flash). SCAIL-2's hand-rolled loop used the plain `decode` and
        // omitted the tiling the shared wan path already does.
        //
        // We tile **temporally only**, sizing each window so its output pixel-volume stays under a
        // budget — so the peak is bounded at every bucket (fewer frames/window as resolution grows).
        // Deliberately NOT `TilingConfig::auto`: at the buckets that trip both its spatial (>512 px)
        // and temporal (>65 frame) thresholds, `auto` emits a combined plan whose spatial tiles are
        // coarse (512 px / 64 latent), and because they don't shrink with frame count the decode peak
        // stays high (~111 GB at 832×480). The budgeted temporal-only window bounds it much tighter
        // (~33 GB) — this is purely a memory choice. (The combined-plan *blend* itself is correct:
        // sc-5690 verified `tile_decode_accumulate` reconstructs a combined spatial+temporal plan
        // exactly on the real z16 VAE at this exact geometry — an earlier flat-frame symptom was not
        // the decode, most likely the bf16→NaN DiT overflow fixed in the same sc-5681 work.)
        // `decode_tiled` falls back to a single pass when the window covers the whole clip.
        on_progress(Progress::Decoding);
        let zs = latent.shape();
        let z = latent.reshape(&[1, zs[0], zs[1], zs[2], zs[3]])?;
        let out_frames = (seg_end - seg_start) as i32;
        let video = {
            // [`DECODE_TILE_BUDGET_PXFRAMES`] output px·frames/window ≈ a bounded decode peak across
            // buckets; ≥8 output frames (2 latent frames for temporal-conv context), snapped to the
            // z16 ×4 temporal stride.
            let px_per_frame = (th as i64) * (tw as i64);
            let budget_frames = (DECODE_TILE_BUDGET_PXFRAMES / px_per_frame.max(1)) as i32;
            let tile_frames = (budget_frames / 4 * 4).clamp(8, out_frames.max(8));
            let overlap = (tile_frames / 4).max(1);
            let cfg = TilingConfig::temporal_only(tile_frames, overlap);
            vae.decode_tiled(&z, &cfg)? // [1,3,T_out,H,W]; single-pass fallback if it doesn't tile
        };
        let vs = video.shape();
        let seg_video = video.reshape(&[vs[1], vs[2], vs[3], vs[4]])?; // [3,T_out,H,W]
        let t_out = seg_video.shape()[1];

        let keep_from = if seg_idx == 0 {
            0
        } else {
            job.segment_overlap as i32
        };
        let piece = seg_video.take_axis(
            Array::from_slice(
                &(keep_from..t_out).collect::<Vec<i32>>(),
                &[t_out - keep_from],
            ),
            1,
        )?;
        mlx_rs::transforms::eval([&piece])?;
        out_pieces.push(piece);

        if seg_idx + 1 < segments.len() {
            let ov = job.segment_overlap as i32;
            prev_history_pixel = Some(seg_video.take_axis(
                Array::from_slice(&(t_out - ov..t_out).collect::<Vec<i32>>(), &[ov]),
                1,
            )?);
        }
    }

    let piece_refs: Vec<&Array> = out_pieces.iter().collect();
    let full = concatenate_axis(&piece_refs, 1)?; // [3,T_total,H,W]
    let frames_u8 = pixels_to_u8(&full)?;
    let frames = frames_to_images(&frames_u8)?;
    Ok(GenerationOutput::Video {
        frames,
        fps: job.fps,
        audio: None,
    })
}
