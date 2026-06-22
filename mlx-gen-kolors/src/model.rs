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
use crate::sampler::{KolorsEulerSampler, NUM_TRAIN_TIMESTEPS};
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
    let time_ids = kolors_time_ids(2, edge as i32, edge as i32);
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
    decode_image(vae, &latents)
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

    /// Decode latents `[1, h, w, 4]` → an RGB [`Image`] (`vae.decode(latents / 0.13025)`).
    pub fn decode(&self, latents: &Array) -> Result<Image> {
        decode_image(&self.vae, latents)
    }

    /// Crate-internal VAE accessor for the registry [`Generator`](crate::registry) wrapper, which
    /// VAE-encodes the img2img init and decodes the final latents around the per-mode denoise
    /// methods it now drives directly (F-146).
    pub(crate) fn vae(&self) -> &Autoencoder {
        &self.vae
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
        neg: &(Array, Array),
        num_steps: usize,
        cfg: f32,
        height: i32,
        width: i32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        use mlx_rs::ops::concatenate_axis;
        let sampler = KolorsEulerSampler::kolors(num_steps, self.dtype)?;
        // CFG batch order is [positive, negative] — `mlx_gen_sdxl::denoise` reads row 0 as the text
        // (cond) and row 1 as the uncond.
        let conditioning = concatenate_axis(&[&pos.0, &neg.0], 0)?;
        let pooled = concatenate_axis(&[&pos.1, &neg.1], 0)?;
        let time_ids = kolors_time_ids(2, height, width);
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
        let neg = self.encode(negative)?;
        let latents = self.denoise_latents(
            &init_noise,
            &pos,
            &neg,
            num_steps,
            cfg,
            height,
            width,
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
        neg: &(Array, Array),
        num_steps: usize,
        strength: f32,
        cfg: f32,
        height: i32,
        width: i32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        use mlx_rs::ops::concatenate_axis;
        let sampler = KolorsEulerSampler::kolors_img2img(num_steps, strength, self.dtype)?;
        let conditioning = concatenate_axis(&[&pos.0, &neg.0], 0)?;
        let pooled = concatenate_axis(&[&pos.1, &neg.1], 0)?;
        let time_ids = kolors_time_ids(2, height, width);
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
        neg: &(Array, Array),
        num_steps: usize,
        strength: f32,
        cfg: f32,
        seed: u64,
        height: i32,
        width: i32,
        control: Option<(&ControlNet, &Image, f32)>,
        ip_tokens: Option<(&Array, f32)>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        use mlx_rs::ops::{add, concatenate_axis, multiply, zeros};
        // Kolors DDPM schedule: `scaled_linear` betas (β₀=0.00085, β₁=0.014) over 1100 train timesteps
        // — the same `EulerDiscreteScheduler` config the native sampler interpolates, here as the
        // discrete `ModelSampling` the curated solvers integrate over (ε-prediction, σ_data = 1).
        let sched = AlphaSchedule::scaled_linear(NUM_TRAIN_TIMESTEPS, 0.00085, 0.014)?;
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
        let conditioning = concatenate_axis(&[&pos.0, &neg.0], 0)?;
        let pooled = concatenate_axis(&[&pos.1, &neg.1], 0)?;
        let time_ids = kolors_time_ids(2, height, width);

        // ControlNet branch: preprocess + CFG-batch the control image, then embed it once (the
        // conditioning embedding is step-invariant, F-069) — exactly as `denoise_controlnet_latents`.
        let controls: Vec<ControlContext> = match control {
            Some((controlnet, control_image, scale)) => {
                let cimg = preprocess_control_image(control_image, width as u32, height as u32)?;
                let cimg = if cfg > 1.0 {
                    concatenate_axis(&[&cimg, &cimg], 0)?
                } else {
                    cimg
                };
                vec![ControlContext {
                    cond_embed: controlnet.embed_cond(&cimg)?,
                    controlnet,
                    scale,
                }]
            }
            None => Vec::new(),
        };

        // IP-Adapter image tokens: CFG-batch with a zeros uncond row (the uncond gets no image
        // conditioning) — exactly as `denoise_ip_latents`.
        let ip_batched = match ip_tokens {
            Some((tokens, scale)) => {
                let zero = zeros::<f32>(tokens.shape())?.as_dtype(tokens.dtype())?;
                Some((concatenate_axis(&[tokens, &zero], 0)?, scale))
            }
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
        let neg = self.encode(negative)?;
        let latents = self.denoise_img2img_latents(
            &init_latents,
            &noise,
            &pos,
            &neg,
            num_steps,
            strength,
            cfg,
            height,
            width,
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
        neg: &(Array, Array),
        num_steps: usize,
        cfg: f32,
        control_scale: f32,
        height: i32,
        width: i32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        use mlx_rs::ops::concatenate_axis;
        let sampler = KolorsEulerSampler::kolors(num_steps, self.dtype)?;
        let conditioning = concatenate_axis(&[&pos.0, &neg.0], 0)?;
        let pooled = concatenate_axis(&[&pos.1, &neg.1], 0)?;
        let time_ids = kolors_time_ids(2, height, width);
        let latents = sampler.scale_initial_noise(init_noise)?;

        // The ControlNet sees the same CFG-batched input as the U-Net (cfg>1 ⇒ [cond, uncond]).
        let cimg = preprocess_control_image(control_image, width as u32, height as u32)?;
        let cimg = if cfg > 1.0 {
            concatenate_axis(&[&cimg, &cimg], 0)?
        } else {
            cimg
        };
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
        let neg = self.encode(negative)?;
        let latents = self.denoise_controlnet_latents(
            controlnet,
            &init_noise,
            control_image,
            &pos,
            &neg,
            num_steps,
            cfg,
            control_scale,
            height,
            width,
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
        neg: &(Array, Array),
        num_steps: usize,
        cfg: f32,
        ip_scale: f32,
        height: i32,
        width: i32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        use mlx_rs::ops::{concatenate_axis, zeros};
        let sampler = KolorsEulerSampler::kolors(num_steps, self.dtype)?;
        let conditioning = concatenate_axis(&[&pos.0, &neg.0], 0)?;
        let pooled = concatenate_axis(&[&pos.1, &neg.1], 0)?;
        let time_ids = kolors_time_ids(2, height, width);
        let latents = sampler.scale_initial_noise(init_noise)?;

        // CFG batch: [image tokens, zeros] — the uncond row gets no image conditioning.
        let sh = ip_tokens.shape();
        let zero = zeros::<f32>(sh)?.as_dtype(ip_tokens.dtype())?;
        let tokens = concatenate_axis(&[ip_tokens, &zero], 0)?;

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
        neg: &(Array, Array),
        num_steps: usize,
        strength: f32,
        cfg: f32,
        control_scale: f32,
        ip_scale: f32,
        height: i32,
        width: i32,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        use mlx_rs::ops::{concatenate_axis, zeros};
        let sampler = KolorsEulerSampler::kolors_img2img(num_steps, strength, self.dtype)?;
        let conditioning = concatenate_axis(&[&pos.0, &neg.0], 0)?;
        let pooled = concatenate_axis(&[&pos.1, &neg.1], 0)?;
        let time_ids = kolors_time_ids(2, height, width);
        // Seed the img2img init (raw `x₀ + noise·σ_start`), as in `denoise_img2img_latents`.
        let latents = sampler.add_noise(init_latents, noise)?;

        // The ControlNet sees the same CFG-batched control image as the U-Net (cfg>1 ⇒ [cond, uncond]).
        let cimg = preprocess_control_image(control_image, width as u32, height as u32)?;
        let cimg = if cfg > 1.0 {
            concatenate_axis(&[&cimg, &cimg], 0)?
        } else {
            cimg
        };
        let cc = ControlContext {
            cond_embed: controlnet.embed_cond(&cimg)?,
            controlnet,
            scale: control_scale,
        };

        // CFG batch the IP tokens with a zeros uncond row (the uncond gets no image conditioning), as
        // in `denoise_ip_latents`.
        let sh = ip_tokens.shape();
        let zero = zeros::<f32>(sh)?.as_dtype(ip_tokens.dtype())?;
        let tokens = concatenate_axis(&[ip_tokens, &zero], 0)?;

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
        let neg = self.encode(negative)?;
        let latents = self.denoise_controlnet_ip_latents(
            controlnet,
            &ip_tokens,
            &init_latents,
            &noise,
            control_image,
            &pos,
            &neg,
            num_steps,
            strength,
            cfg,
            control_scale,
            ip_scale,
            height,
            width,
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
        let neg = self.encode(negative)?;
        let latents = self.denoise_ip_latents(
            &ip_tokens,
            &init_noise,
            &pos,
            &neg,
            num_steps,
            cfg,
            ip_scale,
            height,
            width,
            &CancelFlag::new(),
            &mut |_p| {},
        )?;
        self.decode(&latents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
