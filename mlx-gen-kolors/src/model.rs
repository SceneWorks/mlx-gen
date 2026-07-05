//! Kolors T2I pipeline (sc-3094) — composes the ChatGLM3 conditioning, the leading-Euler scheduler,
//! the SDXL U-Net (with the ChatGLM context projection), real CFG, and the SDXL VAE decode.
//!
//! Mirrors diffusers `KolorsPipeline`: tokenize → ChatGLM3 `encode_prompt` (context = `hidden[-2]`,
//! pooled = `hidden[-1]` last token, with the left-padded `position_ids`) for the positive AND
//! negative prompt → CFG-batched U-Net denoise over `EulerDiscreteScheduler(leading)` → VAE decode
//! (latents / 0.13025). `time_ids` = `(H, W, 0, 0, H, W)` (the SDXL `_get_add_time_ids`).
//!
//! The whole pipeline is dtype-parametric; the parity gate (`tests/t2i_parity.rs`) runs f32.

use mlx_rs::{random, Array, Dtype};

use mlx_gen::array::scalar;
use mlx_gen::weights::Weights;
use mlx_gen::{
    schedule_sigmas, AdapterSpec, AlphaSchedule, CancelFlag, DiffusionSampler,
    DiscreteModelSampling, Error, Image, Progress, Result, Scheduler,
};

use mlx_gen_sdxl::{
    apply_sdxl_adapters_with, decode_image, denoise, denoise_control, denoise_curated, denoise_ip,
    denoise_ip_control, encode_init_latents, load_unet_kolors_dtype, load_vae,
    preprocess_control_image, Autoencoder, ControlContext, ControlNet, Denoiser, IpImageEncoder,
    LoraCoverage, SdxlLoraReport, UNet2DConditionModel,
};

use crate::chatglm3::{ChatGlmConfig, ChatGlmModel};
use crate::sampler::{KolorsEulerSampler, BETA_END, BETA_START, NUM_TRAIN_TIMESTEPS};
use crate::tokenizer::KolorsTokenizer;

/// VAE spatial downscale (latent is image/8 per side).
pub const SPATIAL_SCALE: i32 = 8;

/// Reject degenerate dimensions at the public struct-API boundary (F-020). The registered
/// `KolorsGenerator::generate_impl` runs `validate_request` (multiple-of-8), but the `pub fn
/// generate*`/`img2img` struct methods beneath it do not — a non-multiple-of-8 or non-positive
/// dimension would otherwise silently produce a wrong latent shape (`width / SPATIAL_SCALE` truncates)
/// or crash deep in an MLX op. Inert on every valid request (registry dims are always multiples of 8).
fn validate_dims(height: i32, width: i32) -> Result<()> {
    if height <= 0 || width <= 0 || height % SPATIAL_SCALE != 0 || width % SPATIAL_SCALE != 0 {
        return Err(Error::Msg(format!(
            "kolors: height and width must be positive multiples of {SPATIAL_SCALE} (got {height}x{width})"
        )));
    }
    Ok(())
}

/// diffusers `KolorsImg2ImgPipeline` default `strength` (how much of the schedule to re-noise/denoise).
pub const DEFAULT_IMG2IMG_STRENGTH: f32 = 0.3;

/// A loaded Kolors model: ChatGLM3 text encoder + tokenizer + SDXL-family U-Net (with the ChatGLM
/// context projection) + SDXL VAE.
pub struct Kolors {
    chatglm: ChatGlmModel,
    tokenizer: KolorsTokenizer,
    unet: UNet2DConditionModel,
    vae: Autoencoder,
    dtype: Dtype,
}

/// The SDXL-style micro-conditioning `time_ids` = `(H, W, 0, 0, H, W)` per row (the diffusers
/// `_get_add_time_ids` for `original_size == target_size`, no crop).
pub(crate) fn kolors_time_ids(batch: i32, height: i32, width: i32) -> Array {
    let (h, w) = (height as f32, width as f32);
    let row = [h, w, 0.0, 0.0, h, w];
    let mut v = Vec::with_capacity(batch as usize * 6);
    for _ in 0..batch {
        v.extend_from_slice(&row);
    }
    Array::from_slice(&v, &[batch, 6])
}

/// Assemble the U-Net conditioning batch (`context`, `pooled`, `time_ids`) so its batch dim matches
/// what `mlx_gen_sdxl::denoise*` feeds the U-Net latents (sc-9091, F-005). The shared engine only
/// CFG-batches the latents to B=2 when `cfg > 1.0` (`denoise_core`, sdxl pipeline.rs); with
/// `cfg <= 1.0` (guidance disabled, valid per capabilities) the latents stay B=1. Every Kolors mode
/// runs through this ONE helper so a single gate keeps all six denoise assemblies' batch dims correct:
///
/// - `cfg > 1.0` → `[positive, negative]` context/pooled (row 0 = cond, row 1 = uncond) + `time_ids(2)`;
///   `neg` **must** be `Some` (the caller encodes it).
/// - `cfg <= 1.0` → the positive row only + `time_ids(1)`; `neg` is ignored (the caller skips the whole
///   ChatGLM3-6B negative encode).
///
/// Before this gate the assemblies unconditionally built B=2 conditioning, so a CFG-off request handed
/// the U-Net B=1 latents with B=2 conditioning and the attention reshape failed mid-denoise.
fn cfg_conditioning(
    pos: &(Array, Array),
    neg: Option<&(Array, Array)>,
    cfg: f32,
    height: i32,
    width: i32,
) -> Result<(Array, Array, Array)> {
    use mlx_rs::ops::concatenate_axis;
    if cfg > 1.0 {
        // CFG batch order is [positive, negative] — `mlx_gen_sdxl::denoise*` reads row 0 as the text
        // (cond) and row 1 as the uncond.
        let neg = neg.ok_or_else(|| {
            Error::Msg(
                "kolors: CFG is on (guidance > 1.0) but no negative conditioning was supplied"
                    .into(),
            )
        })?;
        let conditioning = concatenate_axis(&[&pos.0, &neg.0], 0)?;
        let pooled = concatenate_axis(&[&pos.1, &neg.1], 0)?;
        Ok((conditioning, pooled, kolors_time_ids(2, height, width)))
    } else {
        // CFG off: B=1 conditioning to match the B=1 latents `denoise_core` keeps unbatched.
        Ok((
            pos.0.clone(),
            pos.1.clone(),
            kolors_time_ids(1, height, width),
        ))
    }
}

/// CFG-batch a preprocessed ControlNet control image to match the U-Net latents' batch dim (sc-9343,
/// follow-up to F-005). The Kolors ControlNet sees the SAME CFG-batched input as the U-Net:
///
/// - `cfg > 1.0` → duplicate the control image to `[cond, uncond]` (B=2), matching the CFG-batched
///   latents `denoise_core` builds;
/// - `cfg <= 1.0` → the single control image (B=1), matching the unbatched latents.
///
/// `cimg` is the already-preprocessed control image (`preprocess_control_image` → `[1, H, W, 3]`);
/// this is only the shape-preserving CFG duplication so it is a pure, synthetic-array-testable gate
/// shared by every control mode (`denoise_controlnet_latents`, `denoise_controlnet_ip_latents`, and
/// the curated path) — the per-mode branch that could silently regress the B=1/B=2 contract.
fn cfg_batch_control_image(cimg: &Array, cfg: f32) -> Result<Array> {
    use mlx_rs::ops::concatenate_axis;
    if cfg > 1.0 {
        Ok(concatenate_axis(&[cimg, cimg], 0)?)
    } else {
        Ok(cimg.clone())
    }
}

/// CFG-batch IP-Adapter image tokens to match the U-Net latents' batch dim (sc-9343, follow-up to
/// F-005). Mirrors [`cfg_batch_control_image`] but the uncond row gets **no** image conditioning:
///
/// - `cfg > 1.0` → `[image tokens, zeros]` (B=2) — the uncond half sees a zeroed token stream;
/// - `cfg <= 1.0` → the image tokens alone (B=1).
///
/// `ip_tokens` is `[1, N, 2048]` (from `IpImageEncoder::tokens`). Pure + synthetic-array-testable,
/// shared by every IP mode (`denoise_ip_latents`, `denoise_controlnet_ip_latents`, curated path).
fn cfg_batch_ip_tokens(ip_tokens: &Array, cfg: f32) -> Result<Array> {
    use mlx_rs::ops::{concatenate_axis, zeros};
    if cfg > 1.0 {
        let zero = zeros::<f32>(ip_tokens.shape())?.as_dtype(ip_tokens.dtype())?;
        Ok(concatenate_axis(&[ip_tokens, &zero], 0)?)
    } else {
        Ok(ip_tokens.clone())
    }
}

/// Render one preview sample (sc-5637) from the **in-progress training adapter** already installed
/// on `unet`: seeded prior → leading-Euler CFG denoise → VAE decode → [`Image`]. A stripped
/// [`Kolors::denoise_latents`] + [`Kolors::decode`] for the trainer (which holds the raw components,
/// not a `Kolors`). `context`/`pooled` are the pre-encoded **CFG batch** (`[2, …]` = positive then
/// empty-negative); `dtype` is the trainer compute dtype (the sampler scales the initial noise in it).
/// No progress/cancel plumbing — the caller drives the cadence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_sample(
    unet: &UNet2DConditionModel,
    vae: &Autoencoder,
    context: &Array,
    pooled: &Array,
    guidance: f32,
    seed: u64,
    edge: u32,
    steps: usize,
    dtype: Dtype,
) -> Result<Image> {
    random::seed(seed)?;
    let lh = (edge as i32) / SPATIAL_SCALE;
    let lw = (edge as i32) / SPATIAL_SCALE;
    let init_noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;
    let sampler = KolorsEulerSampler::kolors(steps.max(1), dtype)?;
    // Match the time_ids batch to the conditioning batch the caller supplied (mirrors SDXL's
    // `render_sample`), so this stays correct if the preview ever runs a B=1 (CFG-off) batch.
    let time_ids = kolors_time_ids(pooled.shape()[0], edge as i32, edge as i32);
    let latents = sampler.scale_initial_noise(&init_noise)?;
    let d = Denoiser {
        unet,
        sampler: &sampler,
    };
    let latents = denoise(
        &d,
        latents,
        context,
        pooled,
        &time_ids,
        guidance,
        &CancelFlag::default(),
        &mut |_| {},
    )?;
    // Training preview — native VAE decode only (no PiD overlay in the trainer's render path).
    decode_image(vae, &latents, None)
}

impl Kolors {
    /// Load every Kolors component from the `Kwai-Kolors/Kolors-diffusers` snapshot at `dtype`.
    /// `tokenizer/tokenizer.json` must already be materialized (`tools/build_kolors_tokenizer.py`).
    pub fn load(snapshot: &std::path::Path, dtype: Dtype) -> Result<Self> {
        let te_w = Weights::from_dir(snapshot.join("text_encoder"))?;
        let chatglm = ChatGlmModel::from_weights(&te_w, ChatGlmConfig::chatglm3_6b(), None, dtype)?;
        let tokenizer = KolorsTokenizer::from_dir(snapshot.join("tokenizer"))?;
        let unet = load_unet_kolors_dtype(snapshot, dtype)?;
        let vae = load_vae(snapshot)?; // SDXL VAE (sdxl-vae-fp16-fix), f32
        Ok(Self {
            chatglm,
            tokenizer,
            unet,
            vae,
            dtype,
        })
    }

    /// Load every Kolors component, then **load-time quantize** the memory drivers to `bits` (4 or 8)
    /// — the mlx-gen-sdxl sc-2641 path: the dense fp16 snapshot is loaded and packed in-memory (there
    /// is no pre-quantized Kolors snapshot). Quantizes the 6B ChatGLM3 encoder (the dominant footprint)
    /// **and** the SDXL-family U-Net (reusing its own `quantize`); the VAE stays f32 (it overflows in
    /// low precision — the SDXL-family convention). `bits` ∈ {4, 8}.
    pub fn load_quantized(snapshot: &std::path::Path, dtype: Dtype, bits: i32) -> Result<Self> {
        let mut m = Self::load(snapshot, dtype)?;
        m.quantize(bits)?;
        Ok(m)
    }

    /// Load-time quantize the memory drivers to `bits` (4 or 8) — the 6B ChatGLM3 encoder **and** the
    /// SDXL-family U-Net (the VAE stays f32; the SDXL-family convention). Split out of
    /// [`load_quantized`](Self::load_quantized) so the registry can **merge LoRA/LoKr into the dense
    /// base first, then quantize** (the SDXL ordering — the f32 delta merges into the dense weights,
    /// which are then packed). Idempotent per component.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.chatglm.quantize(bits)?;
        self.unet.quantize(bits)?;
        Ok(())
    }

    /// Merge LoRA / LoKr adapters into the dense U-Net weights at load (sc-4733). The Kolors U-Net is
    /// the SDXL `UNet2DConditionModel`, so this delegates to the SDXL adapter merge
    /// ([`apply_sdxl_adapters_with`]) at **Complete** coverage — the down/mid/up attention surface the
    /// Kolors trainer (sc-4568) targets and the diffusers PEFT suffix-match selects (LoKr specs ignore
    /// coverage and use the vendored down/up surface). Merging (not a forward-time residual) keeps the
    /// denoise loop unchanged. Out-of-surface keys are surfaced in the returned report, not dropped.
    /// Must run **before** [`quantize`](Self::quantize) so the f32 delta lands in the dense base.
    pub fn apply_lora(&mut self, adapters: &[AdapterSpec]) -> Result<SdxlLoraReport> {
        apply_sdxl_adapters_with(&mut self.unet, adapters, LoraCoverage::Complete)
    }

    /// Encode one prompt → `(context [1, 256, 4096], pooled [1, 4096])`, threading the tokenizer's
    /// left-padded `position_ids` into the ChatGLM3 RoPE (as `KolorsPipeline.encode_prompt` does).
    pub fn encode(&self, prompt: &str) -> Result<(Array, Array)> {
        // Kolors tokenizes the raw prompt (no chat template).
        let t = self.tokenizer.encode(prompt)?;
        self.chatglm
            .encode_prompt(&t.input_ids, &t.attention_mask, Some(&t.position_ids))
    }

    /// Decode latents `[1, h, w, 4]` → an RGB [`Image`] (`vae.decode(latents / 0.13025)`). The
    /// lower-level struct API always uses the native VAE; the dispatchable `KolorsGenerator`
    /// (`registry.rs`) is where the optional PiD overlay (sc-7848) is threaded.
    pub fn decode(&self, latents: &Array) -> Result<Image> {
        decode_image(&self.vae, latents, None)
    }

    /// Crate-internal VAE accessor for the registry [`Generator`](crate::registry) wrapper, which
    /// VAE-encodes the img2img init and decodes the final latents around the per-mode denoise
    /// methods it now drives directly (F-146).
    pub(crate) fn vae(&self) -> &Autoencoder {
        &self.vae
    }

    /// Crate-internal compute-dtype accessor for the registry wrapper. Used by the PiD `from_ldm`
    /// early-stop (sc-8049) to build the throwaway [`KolorsEulerSampler`] whose `edm_sigmas()` the
    /// VP-capture plan resolves against — the same dtype the active denoise method passes.
    pub(crate) fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Run the CFG denoise loop from a (raw, unit-normal) initial-noise tensor `init_noise`
    /// `[1, h, w, 4]`. The single denoise assembly for plain T2I: the parity gate feeds diffusers'
    /// exact noise with a no-op `cancel`/`on_progress`, and the registry's production count loop
    /// drives it with the real request `CancelFlag` + progress sink — so the two surfaces can't drift
    /// (F-146). `pos`/`neg` are the `(context, pooled)` from [`encode`](Self::encode). Returns the
    /// final latents `[1, h, w, 4]`.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_latents(
        &self,
        init_noise: &Array,
        pos: &(Array, Array),
        neg: Option<&(Array, Array)>,
        num_steps: usize,
        cfg: f32,
        height: i32,
        width: i32,
        run_steps: Option<usize>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        // PiD from_ldm early-stop (sc-8049): `run_steps = Some(keep-1)` truncates the schedule so the
        // solver stops at the VP-capture σ; `None` runs the full schedule byte-identically.
        let sampler = KolorsEulerSampler::kolors(num_steps, self.dtype)?;
        let sampler = match run_steps {
            Some(rs) => sampler.truncate_to(rs),
            None => sampler,
        };
        let (conditioning, pooled, time_ids) = cfg_conditioning(pos, neg, cfg, height, width)?;
        let latents = sampler.scale_initial_noise(init_noise)?;

        let d = Denoiser {
            unet: &self.unet,
            sampler: &sampler,
        };
        denoise(
            &d,
            latents,
            &conditioning,
            &pooled,
            &time_ids,
            cfg,
            cancel,
            on_progress,
        )
    }

    /// Full T2I: seed the RNG, draw the initial noise, encode the prompt + negative prompt, denoise,
    /// and VAE-decode. `height`/`width` are pixels (multiples of 8). `cfg` ≤ 1 disables guidance.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        negative: &str,
        num_steps: usize,
        cfg: f32,
        seed: u64,
        height: i32,
        width: i32,
    ) -> Result<Image> {
        validate_dims(height, width)?;
        random::seed(seed)?;
        let (lh, lw) = (height / SPATIAL_SCALE, width / SPATIAL_SCALE);
        let init_noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;
        let pos = self.encode(prompt)?;
        // Skip the ChatGLM3-6B negative encode entirely when guidance is off (F-005) — the uncond row
        // is never used, so encoding it would be a large wasted forward.
        let neg = if cfg > 1.0 {
            Some(self.encode(negative)?)
        } else {
            None
        };
        let latents = self.denoise_latents(
            &init_noise,
            &pos,
            neg.as_ref(),
            num_steps,
            cfg,
            height,
            width,
            None,
            &CancelFlag::new(),
            &mut |_p| {},
        )?;
        self.decode(&latents)
    }

    /// Run the img2img CFG denoise loop from pre-encoded init latents + a supplied noise tensor —
    /// split out (like [`denoise_latents`](Self::denoise_latents)) so the parity gate can feed
    /// diffusers' exact VAE-encoded init + noise. `init_latents` is the scaled VAE mean
    /// `[1, h, w, 4]`; the sampler is the strength-sliced schedule, the init is seeded via
    /// [`KolorsEulerSampler::add_noise`] (raw `x₀ + noise·σ_start`, no `scale_initial_noise`), and the
    /// loop runs the remaining `int(num_steps·strength)` steps. Returns the final latents.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_img2img_latents(
        &self,
        init_latents: &Array,
        noise: &Array,
        pos: &(Array, Array),
        neg: Option<&(Array, Array)>,
        num_steps: usize,
        strength: f32,
        cfg: f32,
        height: i32,
        width: i32,
        run_steps: Option<usize>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        // PiD from_ldm early-stop (sc-8049): truncate the (already strength-sliced) schedule to `keep-1`
        // steps when `run_steps = Some`; `None` runs the full sliced schedule byte-identically. The
        // init seed uses `start_sigma` (schedule slice start), unaffected by the trailing truncation.
        let sampler = KolorsEulerSampler::kolors_img2img(num_steps, strength, self.dtype)?;
        let sampler = match run_steps {
            Some(rs) => sampler.truncate_to(rs),
            None => sampler,
        };
        let (conditioning, pooled, time_ids) = cfg_conditioning(pos, neg, cfg, height, width)?;
        // Seed the init: raw `x₀ + noise·σ_start` (diffusers EulerDiscrete add_noise at begin_index).
        let latents = sampler.add_noise(init_latents, noise)?;

        let d = Denoiser {
            unet: &self.unet,
            sampler: &sampler,
        };
        denoise(
            &d,
            latents,
            &conditioning,
            &pooled,
            &time_ids,
            cfg,
            cancel,
            on_progress,
        )
    }

    /// Curated unified-sampler denoise (epic 7114, sc-7121) — the **additive** k-diffusion alternative
    /// to the native leading-Euler default, for txt2img + img2img. Drives any curated solver over a
    /// `DiscreteModelSampling` (the Kolors ε/DDPM schedule: `scaled_linear` betas over
    /// `NUM_TRAIN_TIMESTEPS=1100`) and an [`mlx_gen::Scheduler`]-built σ schedule, through the shared
    /// `mlx_gen_sdxl::denoise_curated`. The native `euler_discrete` default is left untouched (N1).
    ///
    /// `init_latents` is `Some` for img2img (the scaled VAE mean), `None` for txt2img. The latents live
    /// in raw k-diffusion σ-space: txt2img seeds `ε·σ_max`; img2img runs the strength-tail of the
    /// schedule, seeded `x₀ + ε·σ_start`.
    ///
    /// `control` / `ip_tokens` thread the conditioned sub-providers (sc-7297, epic 7114) through the
    /// SAME curated solver — the engine `denoise_curated` already supports ControlNet residuals + the
    /// IP-Adapter decoupled-attn tokens (it is the InstantID dual-conditioning path). `control` is
    /// `(controlnet, control_image, control_scale)`: the pose ControlNet, raw-preprocessed +
    /// CFG-batched here and run with its own `embed_cond`. `ip_tokens` is `([1,N,2048] image tokens,
    /// ip_scale)`, CFG-batched with a zeros uncond row. The Kolors ControlNet cross-attends to the
    /// **text** conditioning (`control_encoder = None` ⇒ `cn_enc = conditioning` in `denoise_curated`),
    /// matching the bespoke `denoise_controlnet*_latents`. Both `None` ⇒ plain txt2img / img2img.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_curated_latents(
        &self,
        sampler_name: Option<&str>,
        scheduler_name: Option<&str>,
        init_latents: Option<&Array>,
        noise: &Array,
        pos: &(Array, Array),
        neg: Option<&(Array, Array)>,
        num_steps: usize,
        strength: f32,
        cfg: f32,
        seed: u64,
        height: i32,
        width: i32,
        control: Option<(&ControlNet, &Image, f32)>,
        ip_tokens: Option<(&Array, f32)>,
        run_steps: Option<usize>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        use mlx_rs::ops::{add, multiply};
        // Kolors DDPM schedule: `scaled_linear` betas (β₀=0.00085, β₁=0.014) over 1100 train timesteps
        // — the same `EulerDiscreteScheduler` config the native sampler interpolates, here as the
        // discrete `ModelSampling` the curated solvers integrate over (ε-prediction, σ_data = 1).
        let sched = AlphaSchedule::scaled_linear(NUM_TRAIN_TIMESTEPS, BETA_START, BETA_END);
        let ms = DiscreteModelSampling::sdxl(&sched);
        let scheduler = scheduler_name
            .and_then(Scheduler::from_name)
            .unwrap_or(Scheduler::Normal);
        let full_sigmas = schedule_sigmas(scheduler, &ms, num_steps);
        let noise = noise.as_dtype(Dtype::Float32)?;
        let (run_sigmas, init) = if let Some(x0) = init_latents {
            let strength = strength.clamp(0.0, 1.0);
            let eff = (num_steps as f32 * strength) as usize;
            let run_start = full_sigmas.len().saturating_sub(1).saturating_sub(eff);
            let rs = full_sigmas[run_start..].to_vec();
            let init = add(
                &x0.as_dtype(Dtype::Float32)?,
                &multiply(&noise, scalar(rs[0]))?,
            )?;
            (rs, init)
        } else {
            (
                full_sigmas.clone(),
                multiply(&noise, scalar(full_sigmas[0]))?,
            )
        };
        // PiD from_ldm early-stop (sc-8049): truncate the curated k-diffusion schedule to the VP-capture
        // `keep = run_steps + 1` nodes so the solver stops at the achieved degrade σ; `None` runs the full
        // schedule byte-identically. (The curated `run_sigmas` — not a `KolorsEulerSampler` — is what this
        // path integrates over, so it is truncated in place, mirroring the SDXL curated anchor's
        // `run_sigmas[..keep]`.) `truncate` clamps, so a `keep` past the end is a no-op.
        let run_sigmas = match run_steps {
            Some(rs) => {
                let mut s = run_sigmas;
                s.truncate(rs + 1);
                s
            }
            None => run_sigmas,
        };
        let (conditioning, pooled, time_ids) = cfg_conditioning(pos, neg, cfg, height, width)?;

        // ControlNet branch: preprocess + CFG-batch the control image, then embed it once (the
        // conditioning embedding is step-invariant, F-069) — exactly as `denoise_controlnet_latents`.
        let controls: Vec<ControlContext> = match control {
            Some((controlnet, control_image, scale)) => {
                let cimg = preprocess_control_image(control_image, width as u32, height as u32)?;
                let cimg = cfg_batch_control_image(&cimg, cfg)?;
                vec![ControlContext {
                    cond_embed: controlnet.embed_cond(&cimg)?,
                    controlnet,
                    scale,
                }]
            }
            None => Vec::new(),
        };

        // IP-Adapter image tokens, batched to match the latents: CFG-batch with a zeros uncond row
        // (the uncond gets no image conditioning) when guidance is on; the image tokens alone when off
        // — exactly as `denoise_ip_latents`.
        let ip_batched = match ip_tokens {
            Some((tokens, scale)) => Some((cfg_batch_ip_tokens(tokens, cfg)?, scale)),
            None => None,
        };

        denoise_curated(
            &self.unet,
            sampler_name,
            &ms,
            &run_sigmas,
            init,
            &conditioning,
            &pooled,
            &time_ids,
            cfg,
            seed,
            cancel,
            on_progress,
            &controls,
            ip_batched.as_ref().map(|(tokens, scale)| (tokens, *scale)),
            // `control_encoder = None` ⇒ the Kolors ControlNet cross-attends to the text
            // `conditioning` (its own `encoder_hid_proj`), matching the bespoke combined-pose path.
            None,
        )
    }

    /// Full img2img: VAE-encode `image` (resized to `height`×`width`) → seed at the strength-derived
    /// start → encode the prompts → denoise the remaining steps → VAE-decode. Mirrors diffusers
    /// `KolorsImg2ImgPipeline` (using the VAE encoder **mean** as the init, consistent with the rest
    /// of mlx-gen-sdxl's img2img — the production fork convention; the diffusers default samples the
    /// latent dist, which is not reproducible cross-backend). `cfg` ≤ 1 disables guidance.
    #[allow(clippy::too_many_arguments)]
    pub fn img2img(
        &self,
        image: &Image,
        prompt: &str,
        negative: &str,
        num_steps: usize,
        strength: f32,
        cfg: f32,
        seed: u64,
        height: i32,
        width: i32,
    ) -> Result<Image> {
        validate_dims(height, width)?;
        // VAE-encode the init (no RNG: mean, not a sample) so the first global-RNG draw is the
        // add_noise noise — matching the reference's `prepare_latents` order.
        let init_latents = encode_init_latents(&self.vae, image, width as u32, height as u32)?;
        random::seed(seed)?;
        let (lh, lw) = (height / SPATIAL_SCALE, width / SPATIAL_SCALE);
        let noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;
        let pos = self.encode(prompt)?;
        let neg = if cfg > 1.0 {
            Some(self.encode(negative)?)
        } else {
            None
        };
        let latents = self.denoise_img2img_latents(
            &init_latents,
            &noise,
            &pos,
            neg.as_ref(),
            num_steps,
            strength,
            cfg,
            height,
            width,
            None,
            &CancelFlag::new(),
            &mut |_p| {},
        )?;
        self.decode(&latents)
    }

    /// Run the CFG denoise loop with a Kolors **ControlNet** branch injecting residuals each step
    /// (sc-3097) — split out (like [`denoise_latents`](Self::denoise_latents)) so the parity gate can
    /// feed diffusers' exact noise. The `controlnet` is loaded via `mlx_gen_sdxl::load_controlnet`
    /// (the Kolors ControlNet is a standard SDXL `ControlNetModel` whose only deltas — its own
    /// `encoder_hid_proj` 4096→2048 + the 5632 add-embedding — are auto-detected/shape-driven). It is
    /// conditioned with the **same ChatGLM3 context** as the U-Net (the branch projects it with its
    /// own `encoder_hid_proj`). `control_scale = 0` ⇒ the residuals vanish ⇒ identical to plain T2I.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_controlnet_latents(
        &self,
        controlnet: &ControlNet,
        init_noise: &Array,
        control_image: &Image,
        pos: &(Array, Array),
        neg: Option<&(Array, Array)>,
        num_steps: usize,
        cfg: f32,
        control_scale: f32,
        height: i32,
        width: i32,
        run_steps: Option<usize>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        // PiD from_ldm early-stop (sc-8049): truncate to `keep-1` steps when `run_steps = Some`; `None`
        // runs the full schedule byte-identically.
        let sampler = KolorsEulerSampler::kolors(num_steps, self.dtype)?;
        let sampler = match run_steps {
            Some(rs) => sampler.truncate_to(rs),
            None => sampler,
        };
        let (conditioning, pooled, time_ids) = cfg_conditioning(pos, neg, cfg, height, width)?;
        let latents = sampler.scale_initial_noise(init_noise)?;

        // The ControlNet sees the same CFG-batched input as the U-Net (cfg>1 ⇒ [cond, uncond]).
        let cimg = preprocess_control_image(control_image, width as u32, height as u32)?;
        let cimg = cfg_batch_control_image(&cimg, cfg)?;
        let cc = ControlContext {
            // The conditioning embedding is step-invariant, computed once per denoise here (F-069).
            // Under the registry's count loop this runs once per image rather than once per run; the
            // cost is a single embed forward ≪ the count × N-step denoise, so it stays negligible
            // while keeping this the single denoise assembly shared with production (F-146).
            cond_embed: controlnet.embed_cond(&cimg)?,
            controlnet,
            scale: control_scale,
        };

        let d = Denoiser {
            unet: &self.unet,
            sampler: &sampler,
        };
        denoise_control(
            &d,
            latents,
            &conditioning,
            &pooled,
            &time_ids,
            cfg,
            cancel,
            on_progress,
            &cc,
        )
    }

    /// Full ControlNet T2I: seed the noise, encode the prompts, denoise with the `controlnet` branch
    /// injecting `control_image`-conditioned residuals (`control_scale`), and VAE-decode. The
    /// `control_image` is preprocessed (LANCZOS resize → `[0,1]` NHWC) by the SDXL primitive.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_controlnet(
        &self,
        controlnet: &ControlNet,
        control_image: &Image,
        prompt: &str,
        negative: &str,
        num_steps: usize,
        cfg: f32,
        control_scale: f32,
        seed: u64,
        height: i32,
        width: i32,
    ) -> Result<Image> {
        validate_dims(height, width)?;
        random::seed(seed)?;
        let (lh, lw) = (height / SPATIAL_SCALE, width / SPATIAL_SCALE);
        let init_noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;
        let pos = self.encode(prompt)?;
        let neg = if cfg > 1.0 {
            Some(self.encode(negative)?)
        } else {
            None
        };
        let latents = self.denoise_controlnet_latents(
            controlnet,
            &init_noise,
            control_image,
            &pos,
            neg.as_ref(),
            num_steps,
            cfg,
            control_scale,
            height,
            width,
            None,
            &CancelFlag::new(),
            &mut |_p| {},
        )?;
        self.decode(&latents)
    }

    /// Install the IP-Adapter decoupled cross-attention K/V pairs (from
    /// [`crate::ip_adapter::load_kolors_ip_adapter`]) into the U-Net's cross-attention layers
    /// (sc-3098). One-time setup; non-destructive to plain T2I (the [`denoise`] path never reads the
    /// IP projections — only [`denoise_ip`] does). 70 pairs for the SDXL-family U-Net.
    pub fn install_ip_adapter(&mut self, pairs: Vec<(Array, Array)>) -> Result<()> {
        self.unet.install_ip_adapter(pairs)
    }

    /// Run the CFG denoise loop with IP-Adapter image tokens injected into every cross-attention at
    /// `ip_scale` (sc-3098) — split out (like [`denoise_latents`](Self::denoise_latents)) for the
    /// parity gate. `ip_tokens` is `[1, N, 2048]` (from [`IpImageEncoder::tokens`]); it is CFG-batched
    /// here with a zeros uncond row. The IP-Adapter pairs must already be installed
    /// ([`install_ip_adapter`](Self::install_ip_adapter)). `ip_scale = 0` ⇒ identical to plain T2I.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_ip_latents(
        &self,
        ip_tokens: &Array,
        init_noise: &Array,
        pos: &(Array, Array),
        neg: Option<&(Array, Array)>,
        num_steps: usize,
        cfg: f32,
        ip_scale: f32,
        height: i32,
        width: i32,
        run_steps: Option<usize>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        // PiD from_ldm early-stop (sc-8049): truncate to `keep-1` steps when `run_steps = Some`; `None`
        // runs the full schedule byte-identically.
        let sampler = KolorsEulerSampler::kolors(num_steps, self.dtype)?;
        let sampler = match run_steps {
            Some(rs) => sampler.truncate_to(rs),
            None => sampler,
        };
        let (conditioning, pooled, time_ids) = cfg_conditioning(pos, neg, cfg, height, width)?;
        let latents = sampler.scale_initial_noise(init_noise)?;

        // IP-Adapter image tokens, batched to match the U-Net latents: [image tokens, zeros] under CFG
        // (the uncond row gets no image conditioning); the image tokens alone when guidance is off.
        let tokens = cfg_batch_ip_tokens(ip_tokens, cfg)?;

        let d = Denoiser {
            unet: &self.unet,
            sampler: &sampler,
        };
        denoise_ip(
            &d,
            latents,
            &conditioning,
            &pooled,
            &time_ids,
            cfg,
            cancel,
            on_progress,
            &tokens,
            ip_scale,
        )
    }

    /// Run the CFG denoise loop combining the Kolors **ControlNet** pose branch AND the
    /// **IP-Adapter** image tokens on an **img2img** init (sc-5012) — the SceneWorks strict-pose tier
    /// (Character Studio pose-locked character variations). One pose ControlNet (the rasterized
    /// skeleton) locks the pose, the IP-Adapter reference drives identity, and the **same** reference
    /// seeds the img2img init. Mirrors the vendored `StableDiffusionXLControlNetImg2ImgPipeline` with
    /// `ip_adapter_image` (the torch `KolorsDiffusersAdapter._run_pose`).
    ///
    /// Reuses the SDXL [`denoise_ip_control`] primitive (built for InstantID, sc-3113/3114) — it runs
    /// the ControlNet branch and injects the IP tokens in the same step. The crucial Kolors-specific
    /// wiring: the ControlNet cross-attends to the **text** `conditioning` (`control_encoder =
    /// conditioning`), NOT the IP tokens — the Kolors ControlNet projects the ChatGLM3 context with
    /// its own `encoder_hid_proj`, unlike InstantID's IdentityNet which cross-attends to face tokens.
    ///
    /// `control_scale` (torch `controlnet_conditioning_scale` ≈ 0.7) and `ip_scale` (torch
    /// `ip_adapter_scale` ≈ 0.6) are independent; `strength` is the img2img init strength (torch
    /// default 1.0 — at full strength the init only seeds latent dimensions, identity comes from the
    /// IP-Adapter). `control_scale = 0` + `ip_scale = 0` ⇒ identical to plain img2img. `init_latents`
    /// is the VAE mean of the reference (`[1, h, w, 4]`); `ip_tokens` is `[1, N, 2048]`. The ControlNet
    /// must be loaded and the IP-Adapter pairs installed.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_controlnet_ip_latents(
        &self,
        controlnet: &ControlNet,
        ip_tokens: &Array,
        init_latents: &Array,
        noise: &Array,
        control_image: &Image,
        pos: &(Array, Array),
        neg: Option<&(Array, Array)>,
        num_steps: usize,
        strength: f32,
        cfg: f32,
        control_scale: f32,
        ip_scale: f32,
        height: i32,
        width: i32,
        run_steps: Option<usize>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        // PiD from_ldm early-stop (sc-8049): truncate the (strength-sliced) schedule to `keep-1` steps
        // when `run_steps = Some`; `None` runs the full sliced schedule byte-identically.
        let sampler = KolorsEulerSampler::kolors_img2img(num_steps, strength, self.dtype)?;
        let sampler = match run_steps {
            Some(rs) => sampler.truncate_to(rs),
            None => sampler,
        };
        let (conditioning, pooled, time_ids) = cfg_conditioning(pos, neg, cfg, height, width)?;
        // Seed the img2img init (raw `x₀ + noise·σ_start`), as in `denoise_img2img_latents`.
        let latents = sampler.add_noise(init_latents, noise)?;

        // The ControlNet sees the same CFG-batched control image as the U-Net (cfg>1 ⇒ [cond, uncond]).
        let cimg = preprocess_control_image(control_image, width as u32, height as u32)?;
        let cimg = cfg_batch_control_image(&cimg, cfg)?;
        let cc = ControlContext {
            cond_embed: controlnet.embed_cond(&cimg)?,
            controlnet,
            scale: control_scale,
        };

        // Batch the IP tokens to match the latents: a zeros uncond row (the uncond gets no image
        // conditioning) under CFG; the image tokens alone when guidance is off — as in
        // `denoise_ip_latents`.
        let tokens = cfg_batch_ip_tokens(ip_tokens, cfg)?;

        let d = Denoiser {
            unet: &self.unet,
            sampler: &sampler,
        };
        // `control_encoder = conditioning`: the Kolors ControlNet cross-attends to the ChatGLM3 text
        // context (its own `encoder_hid_proj`), NOT the IP tokens. `cn_enc = control_encoder
        // .unwrap_or(conditioning)` in `denoise_core`, so passing the text conditioning here is the
        // Kolors-correct override (the InstantID default would feed face tokens).
        denoise_ip_control(
            &d,
            latents,
            &conditioning,
            &pooled,
            &time_ids,
            cfg,
            cancel,
            on_progress,
            &cc,
            &conditioning,
            &tokens,
            ip_scale,
        )
    }

    /// Full combined ControlNet-pose + IP-Adapter img2img (sc-5012): encode the `reference_image` →
    /// IP image tokens + VAE init, seed the noise, encode the prompts, run the combined denoise, and
    /// VAE-decode. The `reference_image` drives **both** the IP-Adapter identity and the img2img init;
    /// `control_image` is the pose skeleton. The ControlNet must be loaded and the IP-Adapter pairs
    /// installed.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_controlnet_ip(
        &self,
        controlnet: &ControlNet,
        ip_encoder: &IpImageEncoder,
        control_image: &Image,
        reference_image: &Image,
        prompt: &str,
        negative: &str,
        num_steps: usize,
        strength: f32,
        cfg: f32,
        control_scale: f32,
        ip_scale: f32,
        seed: u64,
        height: i32,
        width: i32,
    ) -> Result<Image> {
        validate_dims(height, width)?;
        let ip_tokens = ip_encoder.tokens(reference_image)?;
        // VAE-encode the init (no RNG: mean) so the first global-RNG draw is the add_noise noise.
        let init_latents =
            encode_init_latents(&self.vae, reference_image, width as u32, height as u32)?;
        random::seed(seed)?;
        let (lh, lw) = (height / SPATIAL_SCALE, width / SPATIAL_SCALE);
        let noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;
        let pos = self.encode(prompt)?;
        let neg = if cfg > 1.0 {
            Some(self.encode(negative)?)
        } else {
            None
        };
        let latents = self.denoise_controlnet_ip_latents(
            controlnet,
            &ip_tokens,
            &init_latents,
            &noise,
            control_image,
            &pos,
            neg.as_ref(),
            num_steps,
            strength,
            cfg,
            control_scale,
            ip_scale,
            height,
            width,
            None,
            &CancelFlag::new(),
            &mut |_p| {},
        )?;
        self.decode(&latents)
    }

    /// Full IP-Adapter T2I: encode the `reference_image` → image tokens, seed the noise, encode the
    /// prompts, denoise with the IP tokens injected at `ip_scale`, and VAE-decode. The IP-Adapter
    /// pairs must already be installed via [`install_ip_adapter`](Self::install_ip_adapter).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_ip(
        &self,
        ip_encoder: &IpImageEncoder,
        reference_image: &Image,
        prompt: &str,
        negative: &str,
        num_steps: usize,
        cfg: f32,
        ip_scale: f32,
        seed: u64,
        height: i32,
        width: i32,
    ) -> Result<Image> {
        validate_dims(height, width)?;
        let ip_tokens = ip_encoder.tokens(reference_image)?;
        random::seed(seed)?;
        let (lh, lw) = (height / SPATIAL_SCALE, width / SPATIAL_SCALE);
        let init_noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;
        let pos = self.encode(prompt)?;
        let neg = if cfg > 1.0 {
            Some(self.encode(negative)?)
        } else {
            None
        };
        let latents = self.denoise_ip_latents(
            &ip_tokens,
            &init_noise,
            &pos,
            neg.as_ref(),
            num_steps,
            cfg,
            ip_scale,
            height,
            width,
            None,
            &CancelFlag::new(),
            &mut |_p| {},
        )?;
        self.decode(&latents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::IndexOp;

    /// F-020: the struct-API dim guard rejects non-positive / non-multiple-of-8 dimensions (which the
    /// registry validates but the `pub fn generate*` methods previously did not).
    #[test]
    fn validate_dims_rejects_degenerate_dimensions() {
        assert!(validate_dims(1024, 768).is_ok());
        assert!(validate_dims(8, 8).is_ok());
        assert!(
            validate_dims(513, 512).is_err(),
            "513 is not a multiple of 8"
        );
        assert!(
            validate_dims(512, 510).is_err(),
            "510 is not a multiple of 8"
        );
        assert!(validate_dims(0, 512).is_err(), "0 is non-positive");
        assert!(validate_dims(512, -8).is_err(), "negative width");
    }

    /// Tiny synthetic `(context, pooled)` in the Kolors conditioning shapes (`[1, T, D]` / `[1, D]`).
    fn synthetic_cond(tag: f32) -> (Array, Array) {
        // T=4 tokens, D=8 channels — dimension-parametric, not the real 256×4096.
        let ctx = mlx_rs::ops::full::<f32>(&[1, 4, 8], mlx_gen::array::scalar(tag)).unwrap();
        let pooled = mlx_rs::ops::full::<f32>(&[1, 8], mlx_gen::array::scalar(tag)).unwrap();
        (ctx, pooled)
    }

    /// sc-9091 (F-005): with CFG on (`cfg > 1.0`) the conditioning is the B=2 `[positive, negative]`
    /// batch and `time_ids` is B=2 — matching the B=2 latents `denoise_core` builds.
    #[test]
    fn cfg_conditioning_batches_two_when_guidance_on() {
        let pos = synthetic_cond(1.0);
        let neg = synthetic_cond(-1.0);
        let (ctx, pooled, time_ids) =
            cfg_conditioning(&pos, Some(&neg), 5.0, 1024, 768).expect("cfg-on assembly");
        assert_eq!(ctx.shape(), &[2, 4, 8], "context is the [pos, neg] batch");
        assert_eq!(pooled.shape(), &[2, 8], "pooled is the [pos, neg] batch");
        assert_eq!(time_ids.shape(), &[2, 6], "time_ids batches to 2 under CFG");
        // Row 0 must be the positive stream, row 1 the negative (the order denoise reads).
        assert_eq!(ctx.index(0).sum(None).unwrap().item::<f32>(), 32.0); // 4*8*1.0
        assert_eq!(ctx.index(1).sum(None).unwrap().item::<f32>(), -32.0); // 4*8*-1.0
    }

    /// sc-9091 (F-005): with CFG off (`cfg <= 1.0`) the conditioning is B=1 (positive only) and
    /// `time_ids` is B=1 — matching the B=1 latents `denoise_core` keeps unbatched. This is the batch
    /// the pre-fix code got wrong (it always built B=2), which is why the attention reshape failed.
    #[test]
    fn cfg_conditioning_batches_one_when_guidance_off() {
        let pos = synthetic_cond(1.0);
        let neg = synthetic_cond(-1.0);
        for cfg in [1.0f32, 0.0, 0.5] {
            let (ctx, pooled, time_ids) =
                cfg_conditioning(&pos, Some(&neg), cfg, 1024, 768).expect("cfg-off assembly");
            assert_eq!(
                ctx.shape(),
                &[1, 4, 8],
                "context is B=1 (positive only) at cfg={cfg}"
            );
            assert_eq!(pooled.shape(), &[1, 8], "pooled is B=1 at cfg={cfg}");
            assert_eq!(time_ids.shape(), &[1, 6], "time_ids is B=1 at cfg={cfg}");
            assert_eq!(
                ctx.index(0).sum(None).unwrap().item::<f32>(),
                32.0,
                "the positive stream"
            );
        }
    }

    /// sc-9091 (F-005): CFG-off must NOT require a negative conditioning — the caller skips the full
    /// ChatGLM3-6B negative encode and passes `None`, and the assembly still yields a valid B=1 batch.
    #[test]
    fn cfg_conditioning_off_needs_no_negative() {
        let pos = synthetic_cond(1.0);
        let (ctx, pooled, time_ids) =
            cfg_conditioning(&pos, None, 1.0, 512, 512).expect("cfg-off needs no negative");
        assert_eq!(ctx.shape(), &[1, 4, 8]);
        assert_eq!(pooled.shape(), &[1, 8]);
        assert_eq!(time_ids.shape(), &[1, 6]);
    }

    // --- sc-9343: per-mode control-image + IP-token batch-shape contract ------------------------
    //
    // Follow-up to sc-9091 (F-005). `cfg_conditioning` already has default-run tests, but the per-mode
    // control-image / IP-token gates (`cfg_batch_control_image`, `cfg_batch_ip_tokens`) that the six
    // denoise assemblies use to keep the ControlNet / IP token batch dim aligned with the latents were
    // only exercised end-to-end by the `#[ignore]`d real-weight smoke. These default-run tests assert
    // the B=1 (CFG-off, guidance ≤ 1.0) / B=2 (CFG-on, guidance > 1.0) contract for each of the six
    // modes using synthetic arrays (no ChatGLM3-6B / U-Net / VAE), so a future edit can't silently
    // regress it. The extraction is a pure refactor — these assemble exactly what the methods build.

    /// A tiny synthetic preprocessed control image (`preprocess_control_image` shape `[1, H, W, 3]`,
    /// here `[1, 2, 2, 3]`), tagged so the CFG-duplication order is checkable.
    fn synthetic_control_image(tag: f32) -> Array {
        mlx_rs::ops::full::<f32>(&[1, 2, 2, 3], mlx_gen::array::scalar(tag)).unwrap()
    }

    /// Tiny synthetic IP-Adapter image tokens (`IpImageEncoder::tokens` shape `[1, N, 2048]`, here
    /// `[1, 4, 8]`), tagged so the zeros-uncond row is checkable.
    fn synthetic_ip_tokens(tag: f32) -> Array {
        mlx_rs::ops::full::<f32>(&[1, 4, 8], mlx_gen::array::scalar(tag)).unwrap()
    }

    /// The expected batch dim per guidance setting: B=2 under CFG (guidance > 1.0), B=1 with it off.
    fn expected_batch(cfg: f32) -> i32 {
        if cfg > 1.0 {
            2
        } else {
            1
        }
    }

    /// The two CFG regimes to sweep: one CFG-on value and the CFG-off values (guidance ≤ 1.0, all of
    /// which disable the uncond stream and must produce B=1).
    const CFG_CASES: [f32; 4] = [5.0, 1.0, 0.5, 0.0];

    /// Mode: **base** (plain T2I) — conditioning only, no control / IP. `denoise_latents`.
    #[test]
    fn mode_base_batch_shape() {
        let pos = synthetic_cond(1.0);
        let neg = synthetic_cond(-1.0);
        for cfg in CFG_CASES {
            let b = expected_batch(cfg);
            let (ctx, pooled, time_ids) =
                cfg_conditioning(&pos, Some(&neg), cfg, 1024, 768).unwrap();
            assert_eq!(ctx.shape(), &[b, 4, 8], "base context B at cfg={cfg}");
            assert_eq!(pooled.shape(), &[b, 8], "base pooled B at cfg={cfg}");
            assert_eq!(time_ids.shape(), &[b, 6], "base time_ids B at cfg={cfg}");
        }
    }

    /// Mode: **img2img** — same conditioning-only assembly as base (the init latents don't affect the
    /// conditioning batch). `denoise_img2img_latents`.
    #[test]
    fn mode_img2img_batch_shape() {
        let pos = synthetic_cond(1.0);
        let neg = synthetic_cond(-1.0);
        for cfg in CFG_CASES {
            let b = expected_batch(cfg);
            let (ctx, pooled, time_ids) =
                cfg_conditioning(&pos, Some(&neg), cfg, 1024, 768).unwrap();
            assert_eq!(ctx.shape(), &[b, 4, 8], "img2img context B at cfg={cfg}");
            assert_eq!(pooled.shape(), &[b, 8], "img2img pooled B at cfg={cfg}");
            assert_eq!(time_ids.shape(), &[b, 6], "img2img time_ids B at cfg={cfg}");
        }
    }

    /// Mode: **controlnet** — conditioning + a CFG-batched control image. `denoise_controlnet_latents`.
    #[test]
    fn mode_controlnet_batch_shape() {
        let pos = synthetic_cond(1.0);
        let neg = synthetic_cond(-1.0);
        for cfg in CFG_CASES {
            let b = expected_batch(cfg);
            let (ctx, pooled, time_ids) =
                cfg_conditioning(&pos, Some(&neg), cfg, 1024, 768).unwrap();
            assert_eq!(ctx.shape(), &[b, 4, 8], "controlnet context B at cfg={cfg}");
            assert_eq!(pooled.shape(), &[b, 8], "controlnet pooled B at cfg={cfg}");
            assert_eq!(
                time_ids.shape(),
                &[b, 6],
                "controlnet time_ids B at cfg={cfg}"
            );

            let cimg = cfg_batch_control_image(&synthetic_control_image(1.0), cfg).unwrap();
            assert_eq!(
                cimg.shape(),
                &[b, 2, 2, 3],
                "controlnet control image B at cfg={cfg}"
            );
        }
    }

    /// Mode: **ip** — conditioning + CFG-batched IP tokens (a zeros uncond row under CFG).
    /// `denoise_ip_latents`.
    #[test]
    fn mode_ip_batch_shape() {
        let pos = synthetic_cond(1.0);
        let neg = synthetic_cond(-1.0);
        for cfg in CFG_CASES {
            let b = expected_batch(cfg);
            let (ctx, pooled, time_ids) =
                cfg_conditioning(&pos, Some(&neg), cfg, 1024, 768).unwrap();
            assert_eq!(ctx.shape(), &[b, 4, 8], "ip context B at cfg={cfg}");
            assert_eq!(pooled.shape(), &[b, 8], "ip pooled B at cfg={cfg}");
            assert_eq!(time_ids.shape(), &[b, 6], "ip time_ids B at cfg={cfg}");

            let tokens = cfg_batch_ip_tokens(&synthetic_ip_tokens(1.0), cfg).unwrap();
            assert_eq!(tokens.shape(), &[b, 4, 8], "ip tokens B at cfg={cfg}");
            if cfg > 1.0 {
                // Row 0 is the image tokens (tag 1.0 → 4*8), row 1 the zeros uncond (0.0).
                assert_eq!(tokens.index(0).sum(None).unwrap().item::<f32>(), 32.0);
                assert_eq!(
                    tokens.index(1).sum(None).unwrap().item::<f32>(),
                    0.0,
                    "the uncond IP row is zeroed"
                );
            }
        }
    }

    /// Mode: **controlnet_ip** (the combined strict-pose tier, sc-5012) — conditioning + a CFG-batched
    /// control image + CFG-batched IP tokens. `denoise_controlnet_ip_latents`.
    #[test]
    fn mode_controlnet_ip_batch_shape() {
        let pos = synthetic_cond(1.0);
        let neg = synthetic_cond(-1.0);
        for cfg in CFG_CASES {
            let b = expected_batch(cfg);
            let (ctx, pooled, time_ids) =
                cfg_conditioning(&pos, Some(&neg), cfg, 1024, 768).unwrap();
            assert_eq!(
                ctx.shape(),
                &[b, 4, 8],
                "controlnet_ip context B at cfg={cfg}"
            );
            assert_eq!(
                pooled.shape(),
                &[b, 8],
                "controlnet_ip pooled B at cfg={cfg}"
            );
            assert_eq!(
                time_ids.shape(),
                &[b, 6],
                "controlnet_ip time_ids B at cfg={cfg}"
            );

            let cimg = cfg_batch_control_image(&synthetic_control_image(1.0), cfg).unwrap();
            assert_eq!(
                cimg.shape(),
                &[b, 2, 2, 3],
                "controlnet_ip control image B at cfg={cfg}"
            );
            let tokens = cfg_batch_ip_tokens(&synthetic_ip_tokens(1.0), cfg).unwrap();
            assert_eq!(
                tokens.shape(),
                &[b, 4, 8],
                "controlnet_ip tokens B at cfg={cfg}"
            );
        }
    }

    /// Mode: **curated** (the additive unified-sampler path, sc-7121) — the same per-mode gates as the
    /// bespoke dispatch, threaded through `denoise_curated_latents`: conditioning always, plus the
    /// optional CFG-batched control image and IP tokens when the conditioned sub-providers are wired.
    #[test]
    fn mode_curated_batch_shape() {
        let pos = synthetic_cond(1.0);
        let neg = synthetic_cond(-1.0);
        for cfg in CFG_CASES {
            let b = expected_batch(cfg);
            let (ctx, pooled, time_ids) =
                cfg_conditioning(&pos, Some(&neg), cfg, 1024, 768).unwrap();
            assert_eq!(ctx.shape(), &[b, 4, 8], "curated context B at cfg={cfg}");
            assert_eq!(pooled.shape(), &[b, 8], "curated pooled B at cfg={cfg}");
            assert_eq!(time_ids.shape(), &[b, 6], "curated time_ids B at cfg={cfg}");

            // The curated path threads control + IP through the SAME gates (sc-7297).
            let cimg = cfg_batch_control_image(&synthetic_control_image(1.0), cfg).unwrap();
            assert_eq!(
                cimg.shape(),
                &[b, 2, 2, 3],
                "curated control image B at cfg={cfg}"
            );
            let tokens = cfg_batch_ip_tokens(&synthetic_ip_tokens(1.0), cfg).unwrap();
            assert_eq!(tokens.shape(), &[b, 4, 8], "curated tokens B at cfg={cfg}");
        }
    }
}
