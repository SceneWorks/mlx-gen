//! Qwen-Image T2I sampling pipeline — ports of the fork's `FluxLatentCreator` (Qwen reuses it),
//! `LinearScheduler` sigma schedule, `QwenImage.compute_guided_noise` (true-CFG with norm
//! correction), the denoise loop (`variants/txt2img/qwen_image.py`), and `ImageUtil.to_image`.
//!
//! Latents live as a **packed** token sequence `[1, (h/16)·(w/16), 64]` throughout the loop
//! (the noise is created already packed, Flux-style), and are unpacked to the VAE's `[1, 16, h/8,
//! w/8]` only at decode. Conditioning runs **two** transformer forwards per step (positive +
//! negative) combined by classifier-free guidance.

use mlx_rs::ops::{add, concatenate_axis, divide, multiply, split_sections, subtract, sum_axes};
use mlx_rs::{random, Array, Dtype};

use mlx_gen::array::scalar;
// The img2img leaves (start-step / init-image preprocess / noise-interp blend) are shared in core;
// re-export so the crate's public surface and internal callers are unchanged.
pub use mlx_gen::img2img::{add_noise_by_interpolation, init_time_step, preprocess_init_image};
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    curated_scheduler_names, default_seed, resolve_flow_schedule, run_flow_sampler, CancelFlag,
    Error, FlowMatchEuler, GenerationRequest, Image, LatentDecoder, Progress, Result,
    TimestepConvention,
};

use crate::control_transformer::QwenFunControlBranch;
use crate::sampler::{lightning_sigmas, LIGHTNING_SHIFT};
use crate::text_encoder::QwenTextEncoder;
use crate::transformer::QwenTransformer;
use crate::vae::QwenVae;

/// Default non-Lightning step count when a request omits `req.steps`. This mirrors the fork's
/// `qwen_image` *function-signature* default of 4 — NOT its recommended production usage: the fork's
/// own README runs 30 steps for every non-Lightning (non-distilled, true-CFG) example. 4 steps on
/// the true-CFG path is heavily under-denoised, so any caller relying on this default gets a quality
/// cliff that can look like an engine bug (F-123). Production callers (SceneWorks) pass an explicit
/// `req.steps` (~30) and never hit this. Kept at 4 to match the fork verbatim rather than silently
/// diverge; raising it to the fork's documented 30 is an owner decision (surfaced on sc-4139).
/// The distilled Lightning path uses [`LIGHTNING_DEFAULT_STEPS`] instead.
pub const DEFAULT_STEPS: u32 = 4;
/// Default true-CFG guidance scale.
pub const DEFAULT_GUIDANCE: f32 = 4.0;
/// Negative-prompt fallback (a single space) for the true-CFG branch when the request omits one.
pub const NEGATIVE_FALLBACK: &str = " ";
/// `req.sampler` value selecting the few-step Lightning recipe (sc-2909).
pub const LIGHTNING_SAMPLER: &str = "lightning";

/// The advertised sampler menu shared by all three Qwen generators (epic 7114 P3): the curated
/// unified-framework integrator menu (Euler / Heun / DPM++ 2M·SDE / UniPC / ancestral / LCM / DDIM)
/// plus the `lightning` few-step acceleration *profile* (sc-2909). The profile picks the schedule and
/// falls back to Euler as the integrator inside [`run_flow_sampler`]; a curated name selects that
/// solver over the production schedule.
pub fn qwen_samplers() -> Vec<&'static str> {
    let mut s = mlx_gen::curated_sampler_names();
    s.push(LIGHTNING_SAMPLER);
    s
}

/// The advertised scheduler menu shared by all three Qwen generators (epic 7114 scheduler axis): the
/// curated sigma-schedule menu (`normal` / `simple` / `karras` / `exponential` / `sgm_uniform` / `beta`
/// / `ddim_uniform`). An unset `req.scheduler` is the native resolution-shifted (production) or
/// static-shift (lightning) schedule — the byte-exact default.
pub fn qwen_schedulers() -> Vec<&'static str> {
    curated_scheduler_names()
}
/// Default step count for the Lightning recipe.
pub const LIGHTNING_DEFAULT_STEPS: u32 = 8;

/// Per-run scalars shared by all three Qwen generators: the Lightning flag, resolved step count and
/// guidance, the per-batch base seed, the selected curated-solver name (epic 7114), and the
/// (seed-independent) flow-match sigma schedule. Extracted so the three `generate` paths can't drift
/// apart (F-117).
pub struct RunParams {
    pub is_lightning: bool,
    pub steps: usize,
    pub guidance: f32,
    pub base_seed: u64,
    /// The requested curated solver name (`req.sampler`), passed to [`run_flow_sampler`] — `lightning`
    /// is an acceleration *profile* (it picks the schedule below, and falls back to Euler as the
    /// integrator), a curated name (e.g. `dpmpp_2m`) selects that solver, and `None` is Euler.
    pub sampler_name: Option<String>,
    /// The flow-match schedule (length `num_steps + 1`, trailing `0.0`).
    pub sigmas: Vec<f32>,
}

/// Resolve the shared run parameters from a request: `sampler == "lightning"` selects the few-step
/// static-shift schedule + its step default; otherwise the production resolution-dependent
/// `qwen_scheduler`. The solver (integration method) is chosen separately by [`run_flow_sampler`]
/// from the same `req.sampler` name. Identical across T2I/Edit/Control (F-117).
pub fn resolve_run_params(req: &GenerationRequest, width: u32, height: u32) -> RunParams {
    let is_lightning = req.sampler.as_deref() == Some(LIGHTNING_SAMPLER);
    let default_steps = if is_lightning {
        LIGHTNING_DEFAULT_STEPS
    } else {
        DEFAULT_STEPS
    };
    let steps = req.steps.unwrap_or(default_steps) as usize;
    // Native schedule (the byte-exact default, epic 7114 N1): lightning's static-shift few-step
    // schedule, or the production resolution-shifted `qwen_scheduler`. A curated `req.scheduler` then
    // re-shapes σ over the SAME mu (lightning: `ln(3)`; production: the qwen area-fit `qwen_mu`).
    let native = if is_lightning {
        lightning_sigmas(steps)
    } else {
        qwen_scheduler(steps, width, height).sigmas
    };
    let mu = if is_lightning {
        LIGHTNING_SHIFT.ln()
    } else {
        qwen_mu(width, height)
    };
    let sigmas = resolve_flow_schedule(req.scheduler.as_deref(), mu, steps, &native);
    RunParams {
        is_lightning,
        steps,
        guidance: req.guidance.unwrap_or(DEFAULT_GUIDANCE),
        base_seed: req.seed.unwrap_or_else(default_seed),
        sampler_name: req.sampler.clone(),
        sigmas,
    }
}

/// The true-CFG negative prompt: the request's negative when non-blank, else [`NEGATIVE_FALLBACK`].
/// Shared by the T2I and Control generators (the Edit path conditions differently).
pub fn negative_or_fallback(req: &GenerationRequest) -> &str {
    match req.negative_prompt.as_deref() {
        Some(s) if !s.trim().is_empty() => s,
        _ => NEGATIVE_FALLBACK,
    }
}

/// Prompt → conditioning embeds (bf16): tokenize, run the text encoder, round to bf16 (the fork is
/// bf16-native on disk). Shared verbatim by the T2I and Control generators; `model_id` only labels
/// the empty-prompt error. The Edit generator uses its own vision-conditioned `encode_edit`.
pub fn encode_prompt(
    tokenizer: &TextTokenizer,
    text_encoder: &QwenTextEncoder,
    prompt: &str,
    model_id: &str,
) -> Result<Array> {
    let t = tokenizer.tokenize(prompt)?;
    if t.ids.is_empty() {
        return Err(Error::Msg(format!("{model_id}: empty prompt")));
    }
    let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&t);
    // PARITY-BF16 (sc-2609): round embeds to bf16 to match the fork (Qwen is bf16-native on disk).
    let embeds = text_encoder.encode(&input_ids, &attention_mask)?;
    Ok(embeds.as_dtype(Dtype::Bfloat16)?)
}

/// The per-count seed → final-latents → decode → collect loop shared by all three generators
/// (F-117). `denoise_one(seed, on_progress)` runs the variant-specific denoise for one sample and
/// returns its packed final latents; this helper handles the seed sequence, the `Decoding` progress
/// tick, unpack + decode, and image collection identically for every variant.
///
/// `decoder` is the latent→pixel decode seam (sc-7844): the native [`QwenVae`] by default, or a PiD
/// decoder for this latent space once wired (sc-7845). PiD output may be larger than VAE-native, so
/// downstream size is taken from the decoded tensor, not assumed.
pub fn decode_and_collect<F>(
    decoder: &dyn LatentDecoder,
    count: u32,
    base_seed: u64,
    width: u32,
    height: u32,
    on_progress: &mut dyn FnMut(Progress),
    mut denoise_one: F,
) -> Result<Vec<Image>>
where
    F: FnMut(u64, &mut dyn FnMut(Progress)) -> Result<Array>,
{
    let mut images = Vec::with_capacity(count as usize);
    for i in 0..count {
        let seed = base_seed.wrapping_add(i as u64);
        let latents = denoise_one(seed, on_progress)?;
        on_progress(Progress::Decoding);
        let unpacked = unpack_latents(&latents, width, height)?;
        let decoded = decoder.decode(&unpacked)?.as_dtype(Dtype::Float32)?;
        images.push(decoded_to_image(&decoded)?);
    }
    Ok(images)
}

/// The PiD backbone (latent-space) tag for the Qwen-Image VAE — shared by all three Qwen generators
/// and by Krea (which reuses [`QwenVae`]). Resolves to the `qwenimage` `2kto4k` student + 4× SR
/// (`mlx_gen_pid::registry`).
pub const PID_BACKBONE: &str = "qwenimage";

/// VAE latent channel count.
pub const LATENT_CHANNELS: i32 = 16;
/// VAE spatial downscale (latent is image/8 per side).
pub const SPATIAL_SCALE: u32 = 8;
/// 2×2 patchify of the latent into the transformer's `in_channels = 16·4 = 64` token features.
pub const PATCH: u32 = 2;

// fork qwen-image scheduler shift params (`ModelConfig.qwen_image`).
const SIGMA_BASE_SHIFT: f32 = 0.5;
const SIGMA_MAX_SHIFT: f32 = 0.9;
const SIGMA_BASE_SEQ_LEN: f32 = 256.0;
const SIGMA_MAX_SEQ_LEN: f32 = 8192.0;
const SIGMA_SHIFT_TERMINAL: f32 = 0.02;

// The decoded-tensor → Image step is identical across families and now lives in core (F-006);
// re-exported so `crate::pipeline::decoded_to_image` and the crate's public surface are unchanged.
pub use mlx_gen::image::decoded_to_image;

/// Seeded txt2img latent noise — shape `[1, (h/16)·(w/16), 64]`, f32. Port of
/// `FluxLatentCreator.create_noise` (`mx.random.normal` with `key(seed)`); the noise is created
/// *already packed*, so packing is a no-op for T2I. The fork casts to the model precision (bf16)
/// when the latents enter the loop; this returns the raw f32 sample for seeded-RNG parity.
pub fn create_noise(seed: u64, width: u32, height: u32) -> Result<Array> {
    let key = random::key(seed)?;
    let seq = ((height / 16) * (width / 16)) as i32;
    let shape = [1, seq, (LATENT_CHANNELS * (PATCH * PATCH) as i32)];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
}

/// Port of `FluxLatentCreator.unpack_latents`: packed tokens `[1, seq, 64]` → VAE latent
/// `[1, 16, h/8, w/8]`. `reshape([1, h/16, w/16, 16, 2, 2]) → transpose(0,3,1,4,2,5) → reshape`.
pub fn unpack_latents(latents: &Array, width: u32, height: u32) -> Result<Array> {
    let (lh, lw) = ((height / 16) as i32, (width / 16) as i32);
    let c = LATENT_CHANNELS;
    let p = PATCH as i32;
    let x = latents.reshape(&[1, lh, lw, c, p, p])?;
    let x = x.transpose_axes(&[0, 3, 1, 4, 2, 5])?;
    Ok(x.reshape(&[1, c, lh * p, lw * p])?)
}

/// Port of `FluxLatentCreator.pack_latents` (inverse of [`unpack_latents`]): VAE latent
/// `[1, 16, h/8, w/8]` → packed tokens `[1, (h/16)·(w/16), 64]`. Used by the Qwen-Image-Edit
/// dual-latent path to fold the encoded reference into the transformer's token sequence.
pub fn pack_latents(latents: &Array, width: u32, height: u32) -> Result<Array> {
    let (lh, lw) = ((height / 16) as i32, (width / 16) as i32);
    let c = LATENT_CHANNELS;
    let p = PATCH as i32;
    let x = latents.reshape(&[1, c, lh, p, lw, p])?;
    let x = x.transpose_axes(&[0, 2, 4, 1, 3, 5])?;
    Ok(x.reshape(&[1, lh * lw, c * p * p])?)
}

/// img2img init image → packed clean latents `[1, (h/16)·(w/16), 64]` (f32). Port of the fork's
/// `LatentCreator.encode_image` ∘ `QwenLatentCreator.pack_latents`: PIL-LANCZOS scale to the target
/// dims, normalize `[0,255] → [-1,1]` NCHW, VAE-encode (causal-Conv3d → scaled 16-ch latent), drop
/// the temporal axis, and pack. Mirrors the Edit `encode_reference_latents` encode, minus the
/// dual-latent `cond_grid` (T2I img2img blends into the noise rather than concatenating).
pub fn encode_init_latents(vae: &QwenVae, image: &Image, width: u32, height: u32) -> Result<Array> {
    let image_nchw = preprocess_init_image(image, width, height)?; // [1, 3, H, W]
    let latent = vae.encode(&image_nchw)?.squeeze_axes(&[2])?; // [1, 16, 1, H/8, W/8] → [1, 16, H/8, W/8]
    pack_latents(&latent, width, height) // [1, (h/16)·(w/16), 64]
}

/// Pack an arbitrary-channel latent `[1, C, h/8, w/8]` → `[1, (h/16)·(w/16), C·4]` (the 2×2 patch
/// pack, generalized over the channel count). Port of the fork's `_pack_latents` used for the
/// 33-channel control context — `pack_latents` hardcodes `C = LATENT_CHANNELS`, this does not.
fn pack_latents_c(latents: &Array, channels: i32, width: u32, height: u32) -> Result<Array> {
    let (lh, lw) = ((height / 16) as i32, (width / 16) as i32);
    let p = PATCH as i32;
    let x = latents.reshape(&[1, channels, lh, p, lw, p])?;
    let x = x.transpose_axes(&[0, 2, 4, 1, 3, 5])?;
    Ok(x.reshape(&[1, lh * lw, channels * p * p])?)
}

/// Build the packed **2512-Fun** control context `[1, seq, 132]` from the pose/union control image —
/// the fork's `pipeline_qwenimage_control`: VAE-encode the control image → 16-ch latent, concatenated
/// (on the channel axis) with a 1-channel mask and a 16-channel inpaint latent, then 2×2-packed
/// (`33 · 4 = 132 = control_in_dim`). v1 is **pose-only** (no inpaint image / mask), where the fork's
/// `1 − ones` mask is `0` and the inpaint latent is `0`, so the layout reduces to
/// `[control_latents | 0(1) | 0(16)]`. The result is constant across denoise steps + the batch.
pub fn encode_fun_control_context(
    vae: &QwenVae,
    image: &Image,
    width: u32,
    height: u32,
) -> Result<Array> {
    let image_nchw = preprocess_init_image(image, width, height)?; // [1, 3, H, W]
    let control_latents = vae.encode(&image_nchw)?.squeeze_axes(&[2])?; // [1, 16, H/8, W/8]
    fun_control_context_from_latents(&control_latents, width, height)
}

/// Build the packed **2512-Fun** 132-ch control context from an already VAE-encoded control latent
/// `[1, 16, H/8, W/8]` — the channel-order/fill + 2×2 pack half of
/// [`encode_fun_control_context`], split out so the sc-8335 numeric golden can byte-confirm it
/// against the fork's `pipeline_qwenimage_control._pack_latents([control_latents | mask | inpaint])`
/// with a synthetic latent (no VAE). Pose-only: zero mask (1ch) + zero inpaint latent (16ch)
/// concatenated on the channel axis → 33 channels, then 2×2-packed → 132.
pub fn fun_control_context_from_latents(
    control_latents: &Array,
    width: u32,
    height: u32,
) -> Result<Array> {
    let (lh8, lw8) = ((height / 8) as i32, (width / 8) as i32);
    let mask = mlx_rs::ops::zeros::<f32>(&[1, 1, lh8, lw8])?;
    let inpaint = mlx_rs::ops::zeros::<f32>(&[1, LATENT_CHANNELS, lh8, lw8])?;
    let ctx = concatenate_axis(&[control_latents, &mask, &inpaint], 1)?; // [1, 33, H/8, W/8]
    pack_latents_c(&ctx, LATENT_CHANNELS * 2 + 1, width, height) // [1, seq, 132]
}

/// Qwen-Image's flow-match sigma schedule: `linspace(1, 1/n, n)` run through the exponential
/// time-shift (`mu` from image area) **and** the terminal-sigma rescale, with a trailing `0`.
/// Port of `LinearScheduler._get_sigmas` (the `requires_sigma_shift` + `sigma_shift_terminal`
/// path). The core [`FlowMatchEuler`] uses FLUX's empirical `mu` and no terminal shift, so we
/// build the Vec here and wrap it.
/// The Qwen-Image production time-shift `mu` — the resolution-dependent linear fit
/// `mu = m·(w·h/256) + b` (the fork's `ModelConfig.qwen_image` shift params). Exposed so the epic 7114
/// scheduler axis builds a curated schedule over Qwen's OWN mu; kept identical to [`qwen_sigmas`] so the
/// native production schedule stays byte-exact.
fn qwen_mu(width: u32, height: u32) -> f32 {
    let m = (SIGMA_MAX_SHIFT - SIGMA_BASE_SHIFT) / (SIGMA_MAX_SEQ_LEN - SIGMA_BASE_SEQ_LEN);
    let b = SIGMA_BASE_SHIFT - m * SIGMA_BASE_SEQ_LEN;
    m * (width as f32 * height as f32 / 256.0) + b
}

pub fn qwen_scheduler(num_steps: usize, width: u32, height: u32) -> FlowMatchEuler {
    FlowMatchEuler {
        sigmas: qwen_sigmas(num_steps, width, height),
    }
}

fn qwen_sigmas(num_steps: usize, width: u32, height: u32) -> Vec<f32> {
    // The terminal-sigma rescale below divides by `1 - shifted.last()`, which is `0` at `n == 1`
    // (`shifted == [1.0]`) → a `[NaN, 0.0]` schedule. The production Generator already rejects
    // `steps < 2` with a clear error (F-113); clamp here so this helper never emits a NaN schedule
    // even when called directly (F-004), yielding a valid 2-step schedule instead.
    let n = num_steps.max(2);
    // linspace(1.0, 1.0/n, n)
    let (start, end) = (1.0_f32, 1.0_f32 / n as f32);
    let linspace: Vec<f32> = (0..n)
        .map(|i| {
            if n == 1 {
                start
            } else {
                start + (end - start) * (i as f32) / ((n - 1) as f32)
            }
        })
        .collect();

    let mu = qwen_mu(width, height);
    let e = mu.exp();
    // exp(mu) / (exp(mu) + (1/sigma - 1))
    let mut shifted: Vec<f32> = linspace
        .iter()
        .map(|&s| e / (e + (1.0 / s - 1.0)))
        .collect();

    // terminal-sigma rescale so the last shifted sigma hits `1 - terminal`.
    let one_minus: Vec<f32> = shifted.iter().map(|&s| 1.0 - s).collect();
    let scale = one_minus[one_minus.len() - 1] / (1.0 - SIGMA_SHIFT_TERMINAL);
    for (s, om) in shifted.iter_mut().zip(&one_minus) {
        *s = 1.0 - om / scale;
    }

    shifted.push(0.0);
    shifted
}

/// True classifier-free guidance with norm correction. Port of `QwenImage.compute_guided_noise`:
/// `combined = neg + g·(pos − neg)`, then rescale `combined` to the L2 norm of the positive
/// prediction (over the channel axis). Keeps the guided velocity's magnitude matched to the
/// conditional one — prevents the over-saturation plain CFG would introduce at `g = 4`.
pub fn compute_guided_noise(pos: &Array, neg: &Array, guidance: f32) -> Result<Array> {
    let combined = add(neg, &multiply(&subtract(pos, neg)?, scalar(guidance))?)?;
    let cond_norm = l2_over_channels(pos)?;
    let comb_norm = l2_over_channels(&combined)?;
    Ok(multiply(&combined, &divide(&cond_norm, &comb_norm)?)?)
}

/// `sqrt(sum(x², axis=-1, keepdims) + 1e-12)` — per-token L2 norm over the channel axis.
fn l2_over_channels(x: &Array) -> Result<Array> {
    let last = (x.shape().len() - 1) as i32;
    let sq = sum_axes(&multiply(x, x)?, &[last], true)?;
    Ok(add(&sq, scalar(1e-12))?.sqrt()?)
}

/// Flow-match denoise loop driven by a [`DiffusionSampler`] (the production [`FlowMatchSampler`]
/// wrapping `qwen_scheduler`, or the few-step Lightning schedule — sc-2909), with progress and
/// cooperative cancellation. The sampler owns the schedule (`num_steps`/`timestep`/`step`); Qwen
/// feeds the **raw sigma** ([`DiffusionSampler::timestep`]) as the transformer timestep. Returns
/// the final packed latents.
///
/// `neg_embeds` selects the guidance mode:
/// - `Some(neg)` → **true CFG**: two forwards/step (positive + negative) combined via
///   [`compute_guided_noise`] at `guidance` (the production path).
/// - `None` → **CFG-off**: a single forward/step (the velocity *is* the positive prediction). This
///   is the Lightning fast path — the distillation LoRAs are CFG-distilled, so the negative forward
///   and norm-correction are skipped (a 2× saving on top of the few steps); `guidance` is ignored.
///
/// `start_step` is the fork's `Config.init_time_step`: `0` for txt2img (loop over every step), or
/// [`init_time_step`] for img2img (loop `range(init_time_step, steps)` so the blended init latents
/// are denoised from the matching sigma). Progress reports `steps - start_step` total steps.
#[allow(clippy::too_many_arguments)]
pub fn denoise_with_progress(
    transformer: &QwenTransformer,
    sampler_name: Option<&str>,
    sigmas: &[f32],
    seed: u64,
    latents: Array,
    pos_embeds: &Array,
    neg_embeds: Option<&Array>,
    guidance: f32,
    width: u32,
    height: u32,
    start_step: usize,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    // sc-2963 (rollout of sc-2957): run the MMDiT's fusable elementwise glue (adaLN affine, gated
    // residual, tanh-GELU FFN, RoPE rotation) through `mx.compile` — bit-exact (`max|Δ|=0`,
    // compile_parity.rs) and a per-step win at production geometry. Scoped + restored on drop by the
    // RAII guard (F-006) instead of leaking the process-global toggle on.
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();
    let (lh, lw) = ((height / 16) as usize, (width / 16) as usize);
    // `None` joint mask: the prompt embeds carry no padding into the transformer, so parity is
    // proven maskless (see `build_joint_mask`). Qwen is flow-match (FLOW prediction) and feeds the
    // raw schedule sigma as the transformer timestep (Sigma convention).
    let predict = |latents: &Array, sigma: f32| -> Result<Array> {
        let pos = transformer.forward(latents, pos_embeds, None, sigma, lh, lw, &[])?;
        match neg_embeds {
            Some(neg) => {
                let neg = transformer.forward(latents, neg, None, sigma, lh, lw, &[])?;
                compute_guided_noise(&pos, &neg, guidance)
            }
            None => Ok(pos),
        }
    };
    // img2img loops `range(start_step, steps)`: slice the schedule from the matching sigma so the
    // blended init latents are denoised from there (the fork's `init_time_step`). Cancellation, the
    // per-step `eval` (F-119), and progress live in `run_flow_sampler` (epic 7114 P3).
    run_flow_sampler(
        sampler_name,
        TimestepConvention::Sigma,
        &sigmas[start_step.min(sigmas.len().saturating_sub(1))..],
        latents,
        seed,
        cancel,
        on_progress,
        predict,
    )
}

/// Qwen-Image **2512-Fun-Controlnet-Union** (pose) denoise loop (sc-8267 — replaces the InstantX
/// loop). Like [`denoise_with_progress`], but each step runs the base transformer with the VACE
/// control branch threaded in ([`QwenTransformer::forward_control`]): the branch computes its 5
/// per-block hints from the post-embedder streams + the (constant) packed 132-ch control context,
/// which the base adds into its image stream at `control_layers` scaled by `control_scale`. Under
/// true CFG the control forward runs once per guidance branch (positive + negative); the Lightning
/// CFG-off path (`neg_embeds = None`) runs it once. `control_scale = 0` reproduces the base T2I
/// forward (the zero-init `after_proj` + `+0` injection).
#[allow(clippy::too_many_arguments)]
pub fn denoise_control_with_progress(
    transformer: &QwenTransformer,
    controlnet: &QwenFunControlBranch,
    sampler_name: Option<&str>,
    sigmas: &[f32],
    seed: u64,
    latents: Array,
    control_cond: &Array,
    pos_embeds: &Array,
    neg_embeds: Option<&Array>,
    guidance: f32,
    control_scale: f32,
    width: u32,
    height: u32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    // Compiled elementwise glue (sc-2963), as in `denoise_with_progress`. Scoped + restored on drop
    // by the RAII guard (F-006) instead of leaking the process-global toggle on.
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();
    let (lh, lw) = ((height / 16) as usize, (width / 16) as usize);
    // Each step runs the base forward with the VACE control branch + the (constant) 132-ch control
    // context injected, scaled by `control_scale` (`= 0` reproduces base T2I). Under true CFG the
    // control forward runs once per guidance branch. Control is pose-only T2I (no img2img-with-control
    // path; F-122).
    let predict = |latents: &Array, sigma: f32| -> Result<Array> {
        let pos = transformer.forward_control(
            latents,
            pos_embeds,
            None,
            sigma,
            lh,
            lw,
            &[],
            Some((controlnet, control_cond)),
            control_scale,
        )?;
        match neg_embeds {
            Some(neg) => {
                let neg = transformer.forward_control(
                    latents,
                    neg,
                    None,
                    sigma,
                    lh,
                    lw,
                    &[],
                    Some((controlnet, control_cond)),
                    control_scale,
                )?;
                compute_guided_noise(&pos, &neg, guidance)
            }
            None => Ok(pos),
        }
    };
    // Cancellation, the per-step `eval` (F-119), and progress live in `run_flow_sampler` (epic 7114).
    run_flow_sampler(
        sampler_name,
        TimestepConvention::Sigma,
        sigmas,
        latents,
        seed,
        cancel,
        on_progress,
        predict,
    )
}

/// Qwen-Image-**Edit** dual-latent denoise loop, driven by a [`DiffusionSampler`] (sc-2909). Each
/// step concatenates the noise latents with the (static) packed reference latents over the sequence
/// axis, runs the transformer with the reference `cond_grids` so the RoPE spans `[noise] +
/// references`, slices the velocity back to the noise prefix, then takes an Euler step. Port of
/// `QwenImageEdit.generate_image`'s loop.
///
/// `neg_embeds` selects the guidance mode (as in [`denoise_with_progress`]): `Some(neg)` = true CFG
/// (two forwards/step), `None` = CFG-off single forward (the Lightning fast path — the velocity is
/// the positive prediction; `guidance` is ignored).
#[allow(clippy::too_many_arguments)]
pub fn denoise_edit_with_progress(
    transformer: &QwenTransformer,
    sampler_name: Option<&str>,
    sigmas: &[f32],
    seed: u64,
    latents: Array,
    static_image_latents: &Array,
    cond_grids: &[(usize, usize)],
    pos_embeds: &Array,
    neg_embeds: Option<&Array>,
    guidance: f32,
    width: u32,
    height: u32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    // sc-2963 (rollout of sc-2957): compiled elementwise glue in the Edit denoise loop too — see
    // `denoise_with_progress`. Bit-exact; scoped + restored on drop by the RAII guard (F-006).
    let _compile_glue = crate::transformer::CompileGlueGuard::enable();
    let (lh, lw) = ((height / 16) as usize, (width / 16) as usize);
    // Each step concatenates the noise latents with the (static) packed reference latents so the RoPE
    // spans `[noise] + references`, then slices the velocity back to the noise prefix. `None` joint
    // mask (as in T2I): the spliced prompt embeds are full-valid.
    let predict = |latents: &Array, sigma: f32| -> Result<Array> {
        let noise_seq = latents.shape()[1];
        let hidden = concatenate_axis(&[latents, static_image_latents], 1)?;
        let pos = slice_seq(
            &transformer.forward(&hidden, pos_embeds, None, sigma, lh, lw, cond_grids)?,
            noise_seq,
        )?;
        match neg_embeds {
            Some(neg) => {
                let neg = slice_seq(
                    &transformer.forward(&hidden, neg, None, sigma, lh, lw, cond_grids)?,
                    noise_seq,
                )?;
                compute_guided_noise(&pos, &neg, guidance)
            }
            None => Ok(pos),
        }
    };
    // Cancellation, the per-step `eval` (the command-buffer-watchdog boundary), and progress live in
    // `run_flow_sampler` (epic 7114 P3).
    run_flow_sampler(
        sampler_name,
        TimestepConvention::Sigma,
        sigmas,
        latents,
        seed,
        cancel,
        on_progress,
        predict,
    )
}

/// Slice the transformer velocity `[1, full_seq, 64]` back to the noise prefix `[1, n, 64]`. A
/// zero-copy strided split at the static boundary, vs the old per-step arange `take_axis` gather
/// (F-114).
fn slice_seq(x: &Array, n: i32) -> Result<Array> {
    Ok(split_sections(x, &[n], 1)?.swap_remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_seq_matches_arange_gather() {
        // F-114: the zero-copy split-at-boundary must return exactly what the old arange `take_axis`
        // gather did — the leading n tokens, same values.
        let x = Array::from_slice(
            &(0..(5 * 2)).map(|i| i as f32).collect::<Vec<_>>(),
            &[1, 5, 2],
        );
        let got = slice_seq(&x, 3).unwrap();
        assert_eq!(got.shape(), &[1, 3, 2]);
        let idx = Array::from_slice(&[0i32, 1, 2], &[3]);
        let want = x.take_axis(&idx, 1).unwrap();
        assert!(mlx_rs::ops::array_eq(&got, &want, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn noise_shape_is_packed() {
        let n = create_noise(0, 1024, 1024).unwrap();
        assert_eq!(n.shape(), &[1, 4096, 64]);
    }

    #[test]
    fn unpack_inverts_to_vae_latent_shape() {
        let packed = create_noise(0, 512, 768).unwrap(); // h=768,w=512
        let lat = unpack_latents(&packed, 512, 768).unwrap();
        // [1, 16, h/8, w/8]
        assert_eq!(lat.shape(), &[1, 16, 96, 64]);
    }

    #[test]
    fn pack_is_inverse_of_unpack() {
        let packed = create_noise(7, 512, 768).unwrap(); // [1, 48*32, 64]
        let vae_latent = unpack_latents(&packed, 512, 768).unwrap(); // [1,16,96,64]
        let repacked = pack_latents(&vae_latent, 512, 768).unwrap();
        assert_eq!(repacked.shape(), packed.shape());
        let (a, b) = (repacked.as_slice::<f32>(), packed.as_slice::<f32>());
        assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6));
    }

    #[test]
    fn qwen_schedule_shape_and_terminal() {
        let s = qwen_sigmas(4, 1024, 1024);
        assert_eq!(s.len(), 5);
        assert_eq!(*s.last().unwrap(), 0.0);
        // strictly decreasing over the shifted part (1.0 → … → terminal).
        assert!(s[..4].windows(2).all(|w| w[0] > w[1]));
        // terminal rescale forces the last shifted sigma to the terminal `0.02`.
        assert!((s[3] - 0.02).abs() < 1e-4, "got {}", s[3]);
        // first sigma stays at 1.0 (linspace start, shift fixes 1.0 -> 1.0).
        assert!((s[0] - 1.0).abs() < 1e-4, "got {}", s[0]);
    }

    #[test]
    fn init_time_step_matches_fork() {
        // txt2img: no/zero strength → 0.
        assert_eq!(init_time_step(4, None), 0);
        assert_eq!(init_time_step(4, Some(0.0)), 0);
        // floor(steps·strength), clamped to >= 1.
        assert_eq!(init_time_step(4, Some(0.6)), 2); // floor(2.4)
        assert_eq!(init_time_step(8, Some(0.5)), 4);
        assert_eq!(init_time_step(4, Some(0.1)), 1); // floor(0.4)=0 → max(1)
        assert_eq!(init_time_step(4, Some(1.0)), 4);
        assert_eq!(init_time_step(4, Some(2.0)), 4); // strength clamps to 1.0
    }

    #[test]
    fn blend_endpoints() {
        let clean = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let noise = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[1, 2, 2]);
        // sigma=0 → all clean; sigma=1 → all noise.
        let c = add_noise_by_interpolation(&clean, &noise, 0.0).unwrap();
        let n = add_noise_by_interpolation(&clean, &noise, 1.0).unwrap();
        assert_eq!(c.as_slice::<f32>(), clean.as_slice::<f32>());
        assert_eq!(n.as_slice::<f32>(), noise.as_slice::<f32>());
    }

    #[test]
    fn preprocess_init_image_shape_and_range() {
        // 2×2 RGB, no resize (target == source): pixels map [0,255] → [-1,1] NCHW.
        let img = Image {
            width: 2,
            height: 2,
            pixels: vec![0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255],
        };
        let pre = preprocess_init_image(&img, 2, 2).unwrap();
        assert_eq!(pre.shape(), &[1, 3, 2, 2]);
        let v = pre.as_slice::<f32>();
        assert!(v.iter().all(|&x| (-1.0..=1.0).contains(&x)));
        // first pixel (0,0,0) → -1 across channels; channel-planar NCHW so index 0,4,8 are R,G,B@(0,0).
        assert!((v[0] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn guided_noise_matches_positive_norm() {
        // when pos == neg, guided == pos (combined == pos, norm ratio == 1).
        let pos = Array::from_slice(&[3.0f32, 4.0, 0.0, 0.0], &[1, 2, 2]);
        let g = compute_guided_noise(&pos, &pos, 4.0).unwrap();
        let got = g.as_slice::<f32>();
        let want = pos.as_slice::<f32>();
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}
