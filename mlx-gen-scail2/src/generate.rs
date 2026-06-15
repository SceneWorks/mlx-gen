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
use mlx_gen::weights::Weights;
use mlx_gen::{AdapterSpec, Error, GenerationOutput, Image, Progress, Quant, Result};
use mlx_gen_wan::{
    frames_to_images, load_tokenizer, make_scheduler, SolverKind, Umt5Encoder, WanVae,
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

/// Reject an adapter file the SCAIL-2 LoRA path does not yet faithfully support, rather than
/// silently applying only the part it understands (sc-5451).
///
/// The family-agnostic loader installs the standard `lora_down`/`lora_up` (+ `alpha`) factors as
/// forward-time residuals. The lightx2v cross-architecture step-distill ("lightning") LoRAs are a
/// *hybrid* format: alongside the low-rank factors they carry full-rank **diff-patch** tensors
/// (`.diff` / `.diff_b` — direct weight/bias deltas, including on the norm layers) that the residual
/// loader does not consume, and they target the vanilla Wan2.1-I2V input embeddings whose shapes
/// differ from SCAIL-2's (e.g. `patch_embedding` in_dim 36 vs 20). Applying such a file through the
/// residual path would drop the diff patches and the incompatible-shape targets *silently* — exactly
/// the "never silently drop" failure the strict installer guards against. Until the diff-patch +
/// cross-architecture install lands (sc-5684), reject it loudly so a partially-applied lightning LoRA
/// can't masquerade as a full one. SCAIL-2-native LoRAs (the Bias-Aware DPO refinement
/// LoRA, any trained-on-SCAIL-2 adapter) carry only the standard factors and pass straight through.
fn reject_unsupported_adapter_formats(adapters: &[AdapterSpec]) -> Result<()> {
    for spec in adapters {
        let w = Weights::from_file(&spec.path)?;
        if w.keys()
            .any(|k| k.ends_with(".diff") || k.ends_with(".diff_b"))
        {
            return Err(Error::Msg(format!(
                "scail2 LoRA {}: this is a lightx2v diff-patch ('.diff'/'.diff_b') LoRA. The \
                 residual loader understands only the low-rank 'lora_down'/'lora_up' factors, so \
                 applying it would silently drop the full-rank diff patches (and its Wan2.1-I2V \
                 'patch_embedding' shape differs from SCAIL-2's in_dim). Full diff-patch + \
                 cross-architecture lightning support is tracked as sc-5684.",
                spec.path.display()
            )));
        }
    }
    Ok(())
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
    // Fail fast on an unsupported adapter format (before the 31 GB DiT load) so a lightx2v
    // diff-patch LoRA can't get half-applied (sc-5451).
    reject_unsupported_adapter_formats(adapters)?;
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
        let w = Weights::from_file(root.join("dit.safetensors"))?;
        let mut d = Scail2Dit::from_weights(&w, cfg)?;
        d.set_compute_dtype(Dtype::Bfloat16);
        // Quantize the attention + FFN Linears in place (Q4 default in the SceneWorks worker). The
        // packed Q4/Q8 weights are what stays resident; the bf16 source is freed in `quantize`.
        if let Some(q) = quant {
            d.quantize(q.bits(), None)?;
        }
        // Install any inference LoRA(s) — the Bias-Aware DPO refinement LoRA, a lightx2v step-distill
        // lightning LoRA, … — as forward-time residuals over the (now possibly Q4/Q8) base (sc-5451).
        // Adapters are independent of the base quantization, so they stack cleanly on the packed
        // weights; applying them *after* `quantize` keeps the residual a dense add over the quantized
        // matmul. The family-agnostic strict installer (the Z-Image / Qwen path) resolves the
        // diffusers / PEFT / kohya / LoKr keys against SCAIL-2's raw module names and errors — never
        // silently drops — on a format/prefix mismatch or an unmatched target.
        if !adapters.is_empty() {
            mlx_gen::adapters::loader::apply_adapters_strict(
                &mut d,
                adapters,
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

        // --- decode this segment → pixels; stitch + carry history ---
        on_progress(Progress::Decoding);
        let zs = latent.shape();
        let video = vae.decode(&latent.reshape(&[1, zs[0], zs[1], zs[2], zs[3]])?)?; // [1,3,T_out,H,W]
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
