//! Krea 2 **Turbo** text-to-image pipeline (sc-7571) — the vertical that makes Krea 2 runnable:
//! tokenize → Qwen3-VL-4B condition-encode (the 12-layer select stack) → DiT (text_fusion aggregator +
//! single-stream denoise) → Qwen-Image VAE decode. Port of the reference `sampling.py::sample` Turbo
//! path.
//!
//! **CFG-free.** The TDM distillation baked the guided velocity into the weights, so there is no
//! unconditional branch (`guidance == 0` in the reference) — one DiT forward per step. Per-sample
//! `B = 1`: one prompt → no padding → `mask = None` (the DiT runs the full valid context).
//!
//! **Rectified-flow v-param Euler.** The DiT consumes the raw sigma as its timestep
//! ([`TimestepConvention::Sigma`]; it scales ×1000 internally) and predicts the flow velocity
//! directly, so the core [`run_flow_sampler`] Euler step `x + v·(σ_{i+1} − σ_i)` is exactly the
//! reference `img += (tprev − tcurr)·v`. The native exponential-mu schedule ([`turbo_sigmas`]) is the
//! byte-exact default; a per-generation curated sampler/scheduler (epic 7114) reshapes over the same
//! mu. The `clamp(-1,1)` + denormalize the reference applies after decode lives in `decoded_to_image`
//! (`clip(x·0.5 + 0.5, 0, 1)`, the algebraic equal).
//!
//! **Component residency (epic 10834 Phase 1, sc-11101).** The pipeline is split into a [`KreaText`]
//! phase (tokenizer + Qwen3-VL-4B condition encoder + vision tower) and a [`KreaHeavy`] phase
//! (single-stream DiT + Qwen-Image VAE). Each render body takes a *pre-encoded* DiT context Array, so
//! the `Sequential` residency in [`crate::model`] / [`crate::model_control`] can encode → drop the
//! text phase → load the heavy phase → render, bounding peak unified memory to `max(text, DiT+VAE)`
//! instead of the sum. [`KreaPipeline`] stays a thin composition (`text` + `heavy`) whose delegators
//! preserve the byte-exact monolithic behaviour for the trainer-adjacent tests + the `Resident` path.

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use mlx_gen::adapters::loader::apply_adapters_strict;
use mlx_gen::array::scalar;
use mlx_gen::image::{decoded_to_image, validate_multiple_of_16};
use mlx_gen::img2img::{add_noise_by_interpolation, init_time_step, preprocess_init_image};
use mlx_gen::media::Image;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::{
    resolve_flow_schedule, run_flow_sampler, CancelFlag, LatentDecoder, Progress, Result,
    TimestepConvention,
};

use std::path::Path;

use crate::control::Krea2ControlBranch;
use crate::loader::{load_text_encoder, load_transformer, load_vision_tower};
use crate::schedule::{dynamic_mu, krea_sigmas, turbo_sigmas, TURBO_MU};
use crate::text_encoder::{encode_grounded, KreaTextEncoder, KreaTokenizer};
use crate::transformer::Krea2Transformer;
use crate::vae::{load_vae, QwenVae};
use mlx_gen_boogu::VisionTower;

/// Turbo text-to-image knobs, resolved from the [`crate::model`] request. Dimensions are validated at
/// the Generator layer (multiple-of-16, in the resolution range) before the pipeline runs.
pub struct TurboOptions {
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub seed: u64,
    /// Curated sampler override (epic 7114). `None` = the native byte-exact rectified-flow Euler.
    pub sampler: Option<String>,
    /// Curated scheduler override. `None` = the native exponential-mu schedule.
    pub scheduler: Option<String>,
}

/// The **text-encode phase** of a Krea 2 pipeline (epic 10834 Phase 1, sc-11101): tokenizer +
/// Qwen3-VL-4B condition encoder + the Qwen3-VL vision tower (image-grounded edit encoding). This is
/// the component dropped first under `Sequential` residency — it produces the DiT text context
/// Array(s) and holds no DiT/VAE, so once the contexts are materialized (`eval`) it can be freed
/// before the heavy phase loads. The `Resident` path holds it warm for the whole job.
pub struct KreaText {
    tok: KreaTokenizer,
    te: KreaTextEncoder,
    /// Qwen3-VL vision tower for image-grounded (edit) encoding (epic 10871 P2). NB loaded eagerly by
    /// [`Self::from_snapshot`] for the P2.3 validation; production should load it lazily / edit-only so
    /// text-to-image doesn't pay the ~0.6 GB (tracked for P3).
    vision: VisionTower,
}

impl KreaText {
    /// Load the tokenizer + Qwen3-VL-4B condition encoder + vision tower from a Krea 2 snapshot's
    /// `tokenizer/` + `text_encoder/` dirs.
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        Ok(Self {
            tok: KreaTokenizer::from_snapshot(root)?,
            te: load_text_encoder(root)?,
            vision: load_vision_tower(root)?,
        })
    }

    /// Quantize the text-encoder Linears in place (group-wise affine Q4/Q8). The DiT is quantized on the
    /// heavy side ([`KreaHeavy::quantize`]); the VAE + vision tower stay dense (matching the converter's
    /// quant-target set — the monolithic `KreaPipeline::quantize` did `te` + `dit`, not the VAE/vision).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.te.quantize(bits)
    }

    /// Encode a plain text prompt → the DiT text context `[1, n_tok, 12, 2560]` (the 12 selected
    /// Qwen3-VL hidden layers, stacked + prefix-dropped). Used by the Turbo/Raw/control/img2img paths.
    /// Deterministic (no RNG), so encoding once and reusing across a `count` loop is byte-identical to a
    /// per-image re-encode.
    pub fn encode(&self, prompt: &str) -> Result<Array> {
        let (ids, attn) = self.tok.encode_prompt(prompt)?;
        self.te.forward(&ids, &attn)
    }

    /// Image-grounded edit context (epic 10871 P2.3): the `source` image feeds the Qwen3-VL vision
    /// tower alongside the instruction text, replacing the text-only encode for the Kontext edit path.
    pub fn encode_grounded(&self, source: &Image, prompt: &str) -> Result<Array> {
        encode_grounded(&self.vision, &self.tok, &self.te, source, prompt)
    }
}

/// The **heavy render phase** of a Krea 2 pipeline (epic 10834 Phase 1, sc-11101): the single-stream
/// DiT + the Qwen-Image VAE — everything but the text encoder. Each `render_*` body takes a pre-encoded
/// context Array (from [`KreaText`]), so both the `Resident` and `Sequential` residencies drive the
/// exact same render code. Owned by the `Resident` components or loaded per-generate by `Sequential`.
pub struct KreaHeavy {
    dit: Krea2Transformer,
    vae: QwenVae,
}

impl KreaHeavy {
    /// Load the single-stream DiT + Qwen-Image VAE from a Krea 2 snapshot's `transformer/` + `vae/`
    /// dirs.
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        Ok(Self {
            dit: load_transformer(root)?,
            vae: load_vae(root)?,
        })
    }

    /// Quantize the DiT Linears in place (group-wise affine Q4/Q8); the VAE stays dense (the published
    /// `vae/` is f32), matching the converter's quant-target set. A no-op on an already-packed snapshot
    /// (`AdaptableLinear::quantize` skips quantized bases).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.dit.quantize(bits)
    }

    /// Install Raw-trained LoRA/LoKr adapters onto the single-stream DiT (sc-7911). The shared
    /// [`apply_adapters_strict`] seam parses PEFT/diffusers/kohya/LoKr files, folds alpha/rank, and
    /// pushes a residual onto each matched `AdaptableLinear` — erroring (never silently dropping) on an
    /// adapter target that matches no module. The `Krea2Transformer` adapter host routes the trained
    /// `transformer_blocks.{i}.attn.{to_q,to_k,to_v,to_out.0}` paths (+ `text_fusion` + globals); the
    /// residual stacks over the (possibly already-quantized) base, so it composes with the Q8/Q4
    /// turnkey. Multiple + mixed LoRA/LoKr adapters stack by construction.
    pub fn apply_adapters(&mut self, specs: &[AdapterSpec]) -> Result<()> {
        apply_adapters_strict(&mut self.dit, specs, "krea_2")?;
        Ok(())
    }

    /// **Turbo t2i render** — the denoise/decode body of [`KreaPipeline::generate_turbo_with_progress`]
    /// with the text encode hoisted out (`context`, pre-encoded by [`KreaText::encode`]). CFG-free, B=1
    /// → mask = None. `keep` (from_ldm early-stop) truncates the schedule; `usize::MAX` runs it all.
    pub fn render_turbo(
        &self,
        context: &Array,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        keep: usize,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "krea_2_turbo")?;

        // Initial latent noise [1, 16, H/8, W/8] (f32; the DiT casts to its compute dtype).
        let noise = init_noise(opts.height, opts.width, opts.seed)?;

        // Native exponential-mu Turbo sigmas are the byte-exact default; a curated scheduler reshapes
        // over the same mu. Raw sigma → DiT timestep, raw velocity → Euler `x + v·(σ_{i+1} − σ_i)`.
        // `from_ldm` early-stop (sc-7993): truncate to `keep` entries (σ=0 clean path runs them all).
        let full = turbo_schedule(opts.steps, opts.scheduler.as_deref());
        let sigmas = &full[..keep.min(full.len())];
        // Hoist the step-invariant text fusion + joint RoPE out of the per-step closure (F-079): the
        // context (and the latent geometry) are fixed across the denoise, so `prepare` runs ONCE and
        // every step reuses `prep` via `forward_prepared` (bit-identical to the per-step `forward`).
        let prep = self.dit.prepare(context, None, &noise)?;
        let lat = run_flow_sampler(
            opts.sampler.as_deref(),
            TimestepConvention::Sigma,
            sigmas,
            noise,
            opts.seed,
            cancel,
            on_progress,
            |x, timestep| {
                let t = Array::from_slice(&[timestep], &[1]);
                let v = self.dit.forward_prepared(x, &t, &prep)?;
                Ok(v.as_dtype(Dtype::Float32)?)
            },
        )?;

        on_progress(Progress::Decoding);
        self.decode_latents(&lat, decoder)
    }

    /// **Pose-ControlNet Turbo render** (sc-8465, epic 8459 S5) — the denoise/decode body of
    /// [`KreaPipeline::generate_turbo_control_with_progress`] with the text encode hoisted out. The
    /// VAE-encoded pose skeleton is embedded once through the frozen base `img_in` and injected as a
    /// `control_scale`-scaled, RMS-clamped residual on every CFG-free Turbo step via the
    /// [`Krea2ControlBranch`]. `control_scale == 0` is a bit-exact base passthrough (the branch is never
    /// run). The PiD decode / `from_ldm` early-stop and img2img seams are intentionally NOT wired on the
    /// control lane (matching the candle `Krea2Control` provider, sc-8464).
    #[allow(clippy::too_many_arguments)]
    pub fn render_turbo_control(
        &self,
        context: &Array,
        branch: &Krea2ControlBranch,
        control_image: &Image,
        control_scale: f32,
        opts: &TurboOptions,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "krea_2_turbo_control")?;

        // Pose skeleton → control latent [1, 16, H/8, W/8] (`QwenVae::encode` returns the normalized
        // `(e − mean)/std` latent, the same space as `init_noise`; drop the singleton temporal axis).
        // Embed once through the frozen base `img_in` — the pose is fixed across steps (step-invariant).
        let image_nchw = preprocess_init_image(control_image, opts.width, opts.height)?;
        let ctrl_latent = self.vae.encode(&image_nchw)?.squeeze_axes(&[2])?;
        let ctrl_tokens = self.dit.embed_latent(&ctrl_latent)?;

        let noise = init_noise(opts.height, opts.width, opts.seed)?;
        let sigmas = turbo_schedule(opts.steps, opts.scheduler.as_deref());
        // Hoist the step-invariant text fusion + joint RoPE (F-079), as the plain Turbo path.
        let prep = self.dit.prepare(context, None, &noise)?;
        let lat = run_flow_sampler(
            opts.sampler.as_deref(),
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            opts.seed,
            cancel,
            on_progress,
            |x, timestep| {
                let t = Array::from_slice(&[timestep], &[1]);
                let v = branch.forward(&self.dit, x, &t, &prep, &ctrl_tokens, control_scale)?;
                Ok(v.as_dtype(Dtype::Float32)?)
            },
        )?;

        on_progress(Progress::Decoding);
        self.decode_latents(&lat, None)
    }

    /// **img2img latent-init Turbo render** (epic 8588 slice A; sc-8589/sc-8590) — the denoise/decode
    /// body of [`KreaPipeline::generate_turbo_img2img_with_progress`] with the text encode hoisted out.
    /// VAE-encode the reference into the same normalized latent space as [`init_noise`], blend
    /// `(1 − σ_k)·clean + σ_k·noise` at the start sigma `σ_k = sigmas[k]`, and run the rectified-flow
    /// Euler sampler over `sigmas[k..]`. Distilled Turbo is CFG-free → one DiT forward per step.
    #[allow(clippy::too_many_arguments)]
    pub fn render_turbo_img2img(
        &self,
        context: &Array,
        init: &Image,
        strength: f32,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "krea_2_turbo")?;

        // Reference → clean latent [1, 16, H/8, W/8]. `QwenVae::encode` already returns the normalized
        // `(e − mean)/std` latent (the same space as `init_noise`); drop the singleton temporal axis.
        let image_nchw = preprocess_init_image(init, opts.width, opts.height)?;
        let clean = self.vae.encode(&image_nchw)?.squeeze_axes(&[2])?;

        let noise = init_noise(opts.height, opts.width, opts.seed)?;

        // Start step from strength; blend the clean latent with noise at σ_k, then denoise sigmas[k..].
        let full = turbo_schedule(opts.steps, opts.scheduler.as_deref());
        let start = init_time_step(opts.steps, Some(strength)).min(full.len().saturating_sub(1));
        let sigmas = &full[start..];
        let x_start = add_noise_by_interpolation(&clean, &noise, full[start])?;

        let prep = self.dit.prepare(context, None, &x_start)?;
        let lat = run_flow_sampler(
            opts.sampler.as_deref(),
            TimestepConvention::Sigma,
            sigmas,
            x_start,
            opts.seed,
            cancel,
            on_progress,
            |x, timestep| {
                let t = Array::from_slice(&[timestep], &[1]);
                let v = self.dit.forward_prepared(x, &t, &prep)?;
                Ok(v.as_dtype(Dtype::Float32)?)
            },
        )?;

        on_progress(Progress::Decoding);
        self.decode_latents(&lat, decoder)
    }

    /// **Raw true-CFG t2i render** (`krea_2_raw`) — the denoise/decode body of
    /// [`KreaPipeline::generate_base_with_progress`] with the text encodes hoisted out. `ctx_pos` is the
    /// conditional context; `ctx_neg` (`Some` only when `guidance > 0`) the unconditional one. Two DiT
    /// forwards/step combined by the reference `v = cond + guidance·(cond − uncond)`
    /// ([`krea_cfg_combine`]); `ctx_neg == None` collapses to a single conditional forward. Both preps
    /// are hoisted ONCE (the F-079 hoist) against the shared noise.
    #[allow(clippy::too_many_arguments)]
    pub fn render_base(
        &self,
        ctx_pos: &Array,
        ctx_neg: Option<&Array>,
        guidance: f32,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        keep: usize,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "krea_2_raw")?;

        // Initial latent noise [1, 16, H/8, W/8] — shared by both prep branches (same geometry).
        let noise = init_noise(opts.height, opts.width, opts.seed)?;
        let prep_pos = self.dit.prepare(ctx_pos, None, &noise)?;

        // Unconditional branch prepared ONLY when CFG is active (the caller passes `Some` iff
        // `guidance > 0` — reference `cfg = guidance > 0`; the negative prompt defaulted to `""`).
        let prep_neg = match ctx_neg {
            Some(nc) => Some(self.dit.prepare(nc, None, &noise)?),
            None => None,
        };

        // Resolution-dynamic Raw schedule (mu from image-token count), truncated to `keep` for a PiD
        // `from_ldm` early-stop (σ=0 clean path runs them all).
        let full = base_schedule(
            opts.steps,
            opts.width,
            opts.height,
            opts.scheduler.as_deref(),
        );
        let sigmas = &full[..keep.min(full.len())];
        let lat = run_flow_sampler(
            opts.sampler.as_deref(),
            TimestepConvention::Sigma,
            sigmas,
            noise,
            opts.seed,
            cancel,
            on_progress,
            |x, timestep| {
                let t = Array::from_slice(&[timestep], &[1]);
                let cond = self.dit.forward_prepared(x, &t, &prep_pos)?;
                let v = match &prep_neg {
                    Some(neg) => {
                        let uncond = self.dit.forward_prepared(x, &t, neg)?;
                        krea_cfg_combine(&cond, &uncond, guidance)?
                    }
                    None => cond,
                };
                Ok(v.as_dtype(Dtype::Float32)?)
            },
        )?;

        on_progress(Progress::Decoding);
        self.decode_latents(&lat, decoder)
    }

    /// **img2img latent-init Raw true-CFG render** (`krea_2_raw`, epic 8588 slice A / sc-10224) — the
    /// denoise/decode body of [`KreaPipeline::generate_base_img2img_with_progress`] with the text
    /// encodes hoisted out. Seed the denoise from a VAE-encoded `init` reference (strength-derived start
    /// step + noise-blend) instead of pure noise, then run the full-CFG Raw sampler over the tail of the
    /// resolution-dynamic [`base_schedule`]. Both preps are hoisted ONCE against the blended `x_start`.
    #[allow(clippy::too_many_arguments)]
    pub fn render_base_img2img(
        &self,
        ctx_pos: &Array,
        ctx_neg: Option<&Array>,
        guidance: f32,
        init: &Image,
        strength: f32,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "krea_2_raw")?;

        // Reference → clean latent [1, 16, H/8, W/8]. `QwenVae::encode` already returns the normalized
        // `(e − mean)/std` latent (the same space as `init_noise`); drop the singleton temporal axis.
        let image_nchw = preprocess_init_image(init, opts.width, opts.height)?;
        let clean = self.vae.encode(&image_nchw)?.squeeze_axes(&[2])?;

        let noise = init_noise(opts.height, opts.width, opts.seed)?;

        // Start step from strength on the resolution-dynamic Raw schedule; blend the clean latent with
        // noise at σ_k, then denoise sigmas[k..]. (Turbo uses `turbo_schedule`; Raw's mu is dynamic.)
        let full = base_schedule(
            opts.steps,
            opts.width,
            opts.height,
            opts.scheduler.as_deref(),
        );
        let start = init_time_step(opts.steps, Some(strength)).min(full.len().saturating_sub(1));
        let sigmas = &full[start..];
        let x_start = add_noise_by_interpolation(&clean, &noise, full[start])?;

        // Positive + (CFG-active) unconditional preps, hoisted once against the blended start latent.
        let prep_pos = self.dit.prepare(ctx_pos, None, &x_start)?;
        let prep_neg = match ctx_neg {
            Some(nc) => Some(self.dit.prepare(nc, None, &x_start)?),
            None => None,
        };

        let lat = run_flow_sampler(
            opts.sampler.as_deref(),
            TimestepConvention::Sigma,
            sigmas,
            x_start,
            opts.seed,
            cancel,
            on_progress,
            |x, timestep| {
                let t = Array::from_slice(&[timestep], &[1]);
                let cond = self.dit.forward_prepared(x, &t, &prep_pos)?;
                let v = match &prep_neg {
                    Some(neg) => {
                        let uncond = self.dit.forward_prepared(x, &t, neg)?;
                        krea_cfg_combine(&cond, &uncond, guidance)?
                    }
                    None => cond,
                };
                Ok(v.as_dtype(Dtype::Float32)?)
            },
        )?;

        on_progress(Progress::Decoding);
        self.decode_latents(&lat, decoder)
    }

    /// **Kontext-style edit render** on the Krea 2 Raw (true-CFG) path (epic 10871, sc-10876) — the
    /// denoise/decode body of [`KreaPipeline::generate_edit_with_progress`] with the grounded text
    /// encodes hoisted out. `ctx_pos` / `ctx_neg` are the *grounded* contexts (source image + text)
    /// from [`KreaText::encode_grounded`]; `sources` are the reference image(s) VAE-encoded here at the
    /// TARGET resolution as in-context tokens (`prepare_edit`). Denoises from PURE NOISE under full CFG.
    #[allow(clippy::too_many_arguments)]
    pub fn render_edit(
        &self,
        ctx_pos: &Array,
        ctx_neg: Option<&Array>,
        guidance: f32,
        // `true` = the CFG-free distilled Turbo edit (`krea_2_turbo_edit`, sc-11640): denoise on the
        // few-step `turbo_schedule` (fixed mu) the distilled student was trained on. `false` = the Raw
        // edit (`krea_2_edit`, epic 10871): the resolution-dynamic `base_schedule`. Both denoise the
        // whole schedule from pure noise; only the sigma trajectory (and, via the caller, the CFG
        // branching) differ.
        distilled: bool,
        sources: &[&Image],
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "krea_2_raw")?;
        if sources.is_empty() {
            return Err(mlx_gen::Error::Msg(
                "krea_2 edit: at least one source image is required".into(),
            ));
        }

        // Source reference(s) → clean latents `[1, 16, H/8, W/8]` at the TARGET resolution (references
        // share the target grid). `QwenVae::encode` returns the normalized `(e − mean)/std` latent; drop
        // the singleton temporal axis (same front matter as img2img).
        let mut ref_latents: Vec<Array> = Vec::with_capacity(sources.len());
        for &src in sources {
            let image_nchw = preprocess_init_image(src, opts.width, opts.height)?;
            ref_latents.push(self.vae.encode(&image_nchw)?.squeeze_axes(&[2])?);
        }

        // Edit denoises from PURE NOISE — the source is in-context conditioning, not a noised init.
        let noise = init_noise(opts.height, opts.width, opts.seed)?;

        // DUAL CONDITIONING (epic 10871 P2.3): the source image feeds BOTH the in-context VAE tokens
        // (`ref_latents`, above) AND the Qwen3-VL grounded context (`ctx_pos`/`ctx_neg`, hoisted by the
        // caller). Each prep carries the clean reference tokens + the reference-frame RoPE.
        let prep_pos = self.dit.prepare_edit(ctx_pos, None, &noise, &ref_latents)?;
        let prep_neg = match ctx_neg {
            Some(nc) => Some(self.dit.prepare_edit(nc, None, &noise, &ref_latents)?),
            None => None,
        };

        // The edit runs the whole schedule from noise (like t2i). Turbo edit uses the distilled few-step
        // `turbo_schedule` (fixed mu) the CFG-free student expects; Raw edit uses the resolution-dynamic
        // `base_schedule`. This must match `generate_impl`'s capture-σ schedule selector (`is_raw`).
        let sigmas = if distilled {
            turbo_schedule(opts.steps, opts.scheduler.as_deref())
        } else {
            base_schedule(
                opts.steps,
                opts.width,
                opts.height,
                opts.scheduler.as_deref(),
            )
        };
        let lat = run_flow_sampler(
            opts.sampler.as_deref(),
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            opts.seed,
            cancel,
            on_progress,
            |x, timestep| {
                let t = Array::from_slice(&[timestep], &[1]);
                let cond = self.dit.forward_prepared_edit(x, &t, &prep_pos)?;
                let v = match &prep_neg {
                    Some(neg) => {
                        let uncond = self.dit.forward_prepared_edit(x, &t, neg)?;
                        krea_cfg_combine(&cond, &uncond, guidance)?
                    }
                    None => cond,
                };
                Ok(v.as_dtype(Dtype::Float32)?)
            },
        )?;

        on_progress(Progress::Decoding);
        self.decode_latents(&lat, decoder)
    }

    /// Decode a latent to an RGB image through the seam. `decoded_to_image` applies
    /// `clip(x·0.5 + 0.5, 0, 1)` — the algebraic equal of the reference `img.clamp(-1,1)·0.5 + 0.5` —
    /// and drops the singleton temporal axis when present (`QwenVae::decode` is NCTHW with T=1; PiD
    /// returns NCHW at 4× resolution). `decoder` is the native VAE when `None`.
    fn decode_latents(&self, lat: &Array, decoder: Option<&dyn LatentDecoder>) -> Result<Image> {
        let dec: &dyn LatentDecoder = decoder.unwrap_or(&self.vae);
        let decoded = dec.decode(lat)?.as_dtype(Dtype::Float32)?;
        decoded_to_image(&decoded)
    }
}

/// The assembled Krea 2 Turbo pipeline: the [`KreaText`] encode phase + the [`KreaHeavy`] render phase.
/// A thin composition (epic 10834 Phase 1, sc-11101) — the delegators below reproduce the byte-exact
/// monolithic behaviour (encode + render inline) for the `Resident` path and the weight-gated tests,
/// while [`crate::model`] / [`crate::model_control`] drive the two phases directly for `Sequential`.
pub struct KreaPipeline {
    text: KreaText,
    heavy: KreaHeavy,
}

impl KreaPipeline {
    /// Load all Turbo components from a Krea 2 snapshot (`tokenizer/ text_encoder/ transformer/ vae/`).
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        Ok(Self {
            text: KreaText::from_snapshot(root)?,
            heavy: KreaHeavy::from_snapshot(root)?,
        })
    }

    /// Quantize the DiT + text-encoder Linears in place (group-wise affine Q4/Q8); the VAE stays dense
    /// (the published `vae/` is f32), matching the converter's quant-target set. A no-op on an
    /// already-packed snapshot (`AdaptableLinear::quantize` skips quantized bases).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.text.quantize(bits)?;
        self.heavy.quantize(bits)?;
        Ok(())
    }

    /// Install Raw-trained LoRA/LoKr adapters onto the single-stream DiT (sc-7911). See
    /// [`KreaHeavy::apply_adapters`].
    pub fn apply_adapters(&mut self, specs: &[AdapterSpec]) -> Result<()> {
        self.heavy.apply_adapters(specs)
    }

    /// Generate one RGB image from a text prompt. Convenience wrapper over
    /// [`Self::generate_turbo_with_progress`] with no cancellation, a no-op progress sink, and the
    /// native VAE decode (no PiD).
    pub fn generate_turbo(&self, prompt: &str, opts: &TurboOptions) -> Result<Image> {
        // `keep = usize::MAX` → the full schedule (clean σ=0 decode; no from_ldm early-stop).
        self.generate_turbo_with_progress(
            prompt,
            opts,
            None,
            usize::MAX,
            &CancelFlag::new(),
            &mut |_| {},
        )
    }

    /// Generate one RGB image, streaming [`Progress`] and honoring `cancel` at each denoise step. A
    /// pre/mid-flight cancellation returns [`mlx_gen::Error::Canceled`]; the per-step `eval` (inside
    /// [`run_flow_sampler`]) bounds the lazy MLX graph so the cancel check can interrupt mid-render.
    /// `decoder` (epic 7840, sc-7845): the latent→pixel decode seam — `None` uses the native
    /// [`QwenVae`] (the byte-exact default), `Some` routes through a PiD super-resolving decoder
    /// (built per-generation from the prompt by the caller). `keep` (epic 7840, sc-7993) is the PiD
    /// `from_ldm` early-stop truncation (`usize::MAX` runs the full schedule).
    pub fn generate_turbo_with_progress(
        &self,
        prompt: &str,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        keep: usize,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let context = self.text.encode(prompt)?;
        self.heavy
            .render_turbo(&context, opts, decoder, keep, cancel, on_progress)
    }

    /// **Pose-ControlNet on Turbo** (sc-8465, epic 8459 S5). See [`KreaHeavy::render_turbo_control`].
    #[allow(clippy::too_many_arguments)]
    pub fn generate_turbo_control_with_progress(
        &self,
        prompt: &str,
        branch: &Krea2ControlBranch,
        control_image: &Image,
        control_scale: f32,
        opts: &TurboOptions,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let context = self.text.encode(prompt)?;
        self.heavy.render_turbo_control(
            &context,
            branch,
            control_image,
            control_scale,
            opts,
            cancel,
            on_progress,
        )
    }

    /// Generate one RGB image seeded from a reference `init` image at the given `strength` (img2img,
    /// slice A / sc-8590). Convenience wrapper over [`Self::generate_turbo_img2img_with_progress`].
    pub fn generate_turbo_img2img(
        &self,
        prompt: &str,
        init: &Image,
        strength: f32,
        opts: &TurboOptions,
    ) -> Result<Image> {
        self.generate_turbo_img2img_with_progress(
            prompt,
            init,
            strength,
            opts,
            None,
            &CancelFlag::new(),
            &mut |_| {},
        )
    }

    /// **img2img latent-init on Turbo** (epic 8588 slice A). See [`KreaHeavy::render_turbo_img2img`].
    #[allow(clippy::too_many_arguments)]
    pub fn generate_turbo_img2img_with_progress(
        &self,
        prompt: &str,
        init: &Image,
        strength: f32,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let context = self.text.encode(prompt)?;
        self.heavy.render_turbo_img2img(
            &context,
            init,
            strength,
            opts,
            decoder,
            cancel,
            on_progress,
        )
    }

    /// Generate one RGB image through the **Raw** classifier-free-guidance path with no cancellation, a
    /// no-op progress sink, and the native VAE decode (no PiD). Convenience wrapper over
    /// [`Self::generate_base_with_progress`].
    pub fn generate_base(
        &self,
        prompt: &str,
        negative_prompt: &str,
        guidance: f32,
        opts: &TurboOptions,
    ) -> Result<Image> {
        self.generate_base_with_progress(
            prompt,
            negative_prompt,
            guidance,
            opts,
            None,
            usize::MAX,
            &CancelFlag::new(),
            &mut |_| {},
        )
    }

    /// Generate one RGB image through the **Raw** classifier-free-guidance path (`krea_2_raw`). See
    /// [`KreaHeavy::render_base`]. The unconditional context is encoded ONLY when `guidance > 0`
    /// (reference `cfg = guidance > 0`); the negative prompt defaults to `""` at the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_base_with_progress(
        &self,
        prompt: &str,
        negative_prompt: &str,
        guidance: f32,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        keep: usize,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let ctx_pos = self.text.encode(prompt)?;
        let ctx_neg = if guidance > 0.0 {
            Some(self.text.encode(negative_prompt)?)
        } else {
            None
        };
        self.heavy.render_base(
            &ctx_pos,
            ctx_neg.as_ref(),
            guidance,
            opts,
            decoder,
            keep,
            cancel,
            on_progress,
        )
    }

    /// Generate one RGB image seeded from a reference `init` at `strength` through the **Raw**
    /// true-CFG path, with no cancellation, a no-op progress sink, and the native VAE decode (no PiD).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_base_img2img(
        &self,
        prompt: &str,
        negative_prompt: &str,
        guidance: f32,
        init: &Image,
        strength: f32,
        opts: &TurboOptions,
    ) -> Result<Image> {
        self.generate_base_img2img_with_progress(
            prompt,
            negative_prompt,
            guidance,
            init,
            strength,
            opts,
            None,
            &CancelFlag::new(),
            &mut |_| {},
        )
    }

    /// **img2img latent-init on the Raw (true-CFG) path** (`krea_2_raw`, epic 8588 slice A / sc-10224).
    /// See [`KreaHeavy::render_base_img2img`].
    #[allow(clippy::too_many_arguments)]
    pub fn generate_base_img2img_with_progress(
        &self,
        prompt: &str,
        negative_prompt: &str,
        guidance: f32,
        init: &Image,
        strength: f32,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let ctx_pos = self.text.encode(prompt)?;
        let ctx_neg = if guidance > 0.0 {
            Some(self.text.encode(negative_prompt)?)
        } else {
            None
        };
        self.heavy.render_base_img2img(
            &ctx_pos,
            ctx_neg.as_ref(),
            guidance,
            init,
            strength,
            opts,
            decoder,
            cancel,
            on_progress,
        )
    }

    /// Generate one edited RGB image from a source image + an instruction, with no cancellation, a
    /// no-op progress sink, and the native VAE decode. Convenience wrapper over
    /// [`Self::generate_edit_with_progress`] for a single reference.
    pub fn generate_edit(
        &self,
        prompt: &str,
        negative_prompt: &str,
        guidance: f32,
        source: &Image,
        opts: &TurboOptions,
    ) -> Result<Image> {
        self.generate_edit_with_progress(
            prompt,
            negative_prompt,
            guidance,
            &[source],
            opts,
            None,
            &CancelFlag::new(),
            &mut |_| {},
        )
    }

    /// **Kontext-style image edit** on the Krea 2 Raw (true-CFG) path (epic 10871, sc-10876). See
    /// [`KreaHeavy::render_edit`]. The grounded context (`encode_grounded` over `sources[0]`) replaces
    /// the text-only encode; the unconditional grounded context is built ONLY when `guidance > 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_edit_with_progress(
        &self,
        prompt: &str,
        negative_prompt: &str,
        guidance: f32,
        sources: &[&Image],
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        if sources.is_empty() {
            return Err(mlx_gen::Error::Msg(
                "krea_2 edit: at least one source image is required".into(),
            ));
        }
        // Single-reference grounding for now (the VAE-token side already takes N sources); the grounded
        // context sits in both CFG branches (only the instruction text differs).
        let ctx_pos = self.text.encode_grounded(sources[0], prompt)?;
        let ctx_neg = if guidance > 0.0 {
            Some(self.text.encode_grounded(sources[0], negative_prompt)?)
        } else {
            None
        };
        self.heavy.render_edit(
            &ctx_pos,
            ctx_neg.as_ref(),
            guidance,
            // The KreaPipeline delegator is the Raw-edit (`krea_2_edit`) helper — full-CFG base schedule.
            // The distilled Turbo edit is driven through the `Generator` seam (`generate_impl`), not here.
            false,
            sources,
            opts,
            decoder,
            cancel,
            on_progress,
        )
    }
}

/// The Turbo flow-match sigma schedule for `steps` (native exponential-mu by default, or a curated
/// scheduler over the same mu). Length `steps + 1`, strictly descending with a trailing `0.0`. Exposed
/// so the caller can resolve a PiD `from_ldm` early-stop capture (sc-7993, via
/// `mlx_gen_pid::flow_capture_for_request`) before building the decoder — the same schedule
/// [`KreaHeavy::render_turbo`] then runs (the build is pure host math).
pub fn turbo_schedule(steps: usize, scheduler: Option<&str>) -> Vec<f32> {
    let native = turbo_sigmas(steps);
    resolve_flow_schedule(scheduler, TURBO_MU as f32, steps, &native)
}

/// The **Raw** flow-match sigma schedule for `steps` at a given resolution: the exponential-mu shift
/// with a resolution-**dynamic** `mu` interpolated in image-token count ([`dynamic_mu`]), unlike the
/// Turbo fixed `mu = 1.15`. Length `steps + 1`, descending with a trailing `0.0`; a curated scheduler
/// (epic 7114) reshapes over the same dynamic mu. Exposed so the caller can resolve a PiD `from_ldm`
/// early-stop capture from the same schedule [`KreaHeavy::render_base`] runs.
pub fn base_schedule(steps: usize, width: u32, height: u32, scheduler: Option<&str>) -> Vec<f32> {
    // Image token count = (W/16)·(H/16) (latent /8 then patch /2) — the reference `x.shape[1]`.
    let seq_len = (width as f64 / 16.0) * (height as f64 / 16.0);
    let mu = dynamic_mu(seq_len);
    let native = krea_sigmas(steps, mu);
    resolve_flow_schedule(scheduler, mu as f32, steps, &native)
}

/// Krea's classifier-free-guidance velocity combine — the reference `sampling.py:129`
/// `v = v_cond + guidance·(v_cond − v_uncond)`, **NOT** the standard `v_uncond + g·Δ`. Krea's guidance
/// is offset by one: the standard form applies one full step LESS guidance, and at `guidance = 1.0`
/// collapses to exactly `v_cond` (zero effective CFG). Single source of truth so the Raw inference path
/// and the trainer preview (`training::render_sample`) can never drift again (sc-10009). The caller runs
/// this only for `guidance > 0` (a single conditional forward otherwise, matching `cfg = guidance > 0`).
pub(crate) fn krea_cfg_combine(v_cond: &Array, v_uncond: &Array, guidance: f32) -> Result<Array> {
    Ok(add(
        v_cond,
        &multiply(&subtract(v_cond, v_uncond)?, scalar(guidance))?,
    )?)
}

/// Seeded initial Gaussian latent noise `[1, 16, H/8, W/8]` (f32; the VAE's 8× spatial compression).
/// The model layer offsets `seed` per image in a batch, mirroring the reference `seed + i`.
fn init_noise(height: u32, width: u32, seed: u64) -> Result<Array> {
    let (hl, wl) = ((height / 8) as i32, (width / 8) as i32);
    let key = random::key(seed)?;
    Ok(random::normal::<f32>(
        &[1, 16, hl, wl],
        None,
        None,
        Some(&key),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Krea CFG combine is the reference `cond + g·(cond − uncond)`, not the standard
    /// `uncond + g·Δ`. With cond = 2, uncond = 1 (Δ = 1): g = 1 → 3 (the standard form would give 2 —
    /// exactly `cond` — which is why the default `sample_guidance_scale = 1.0` washed previews out); a
    /// larger g pushes further from cond, away from uncond.
    #[test]
    fn cfg_combine_is_reference_offset_by_one() {
        let cond = Array::from_slice(&[2.0f32], &[1]);
        let uncond = Array::from_slice(&[1.0f32], &[1]);
        for (g, want) in [(1.0f32, 3.0f32), (3.5, 5.5), (0.0, 2.0)] {
            let v = krea_cfg_combine(&cond, &uncond, g).unwrap();
            assert!(
                (v.item::<f32>() - want).abs() < 1e-5,
                "g={g}: got {}, want {want}",
                v.item::<f32>()
            );
        }
    }
}
