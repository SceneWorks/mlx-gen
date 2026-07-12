//! `QwenImageEdit` — the Qwen-Image-**Edit** implementation of [`mlx_gen::Generator`] (id
//! `qwen_image_edit`), plus its [`descriptor`]/[`load`] entry points and `inventory` registration.
//!
//! [`load`] assembles the model from a `Qwen/Qwen-Image-Edit` snapshot (the validated reference is
//! `-2511`; `-2509` is superseded — same architecture, sc-2782/sc-2997) — tokenizer + Qwen2-VL
//! image processor, the Qwen2.5-VL vision-language encoder (LM + vision transformer), the 60-layer
//! MMDiT, and the causal-Conv3d VAE. [`QwenImageEdit::generate`] runs the reference-conditioned
//! pipeline: tokenize the edit template with the reference image → VL-encode (vision embeds spliced
//! into the prompt) → **dual-latent** conditioning (VAE-encode the reference, pack, concat with the
//! noise over the sequence axis) → flow-match Euler denoise with the reference `cond_grid` in the
//! RoPE (two forwards/step, CFG) → slice the noise prefix → VAE decode → RGB8. The dual-latent
//! denoise core is parity-proven against the fork (`tests/edit_real_weights.rs`).
//!
//! Component residency (epic 10834 Phase 1, sc-11006 — the fan-out sibling of the T2I sc-11000):
//! under [`OffloadPolicy::Sequential`] the heavy Qwen2.5-VL vision-language encoder (~16 GB) is
//! dropped after the encode phase — which for Edit spans BOTH the **vision tower** pass over the
//! reference image and the **LM** pass over the prompts — so peak unified memory is bounded to
//! `max(VL-encoder, DiT+VAE)` instead of the sum. The dual-latent reference VAE-encode uses the VAE
//! (not the VL encoder) and so runs after the drop, byte-identically. See [`crate::model::QwenImage`]
//! for the T2I template this mirrors.

use mlx_gen::array::host_i32;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    gen_core, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LatentDecoder, LoadSpec, Modality, ModelDescriptor,
    OffloadPolicy, Precision, Progress, Quant, Residency, Result, WeightsSource,
};
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};
use std::path::Path;

use crate::image_processor::{ImageInput, QwenImageProcessor};
use crate::loader;
use crate::model::validate_request;
use crate::pipeline::{
    create_noise, decode_and_collect, denoise_edit_with_progress, qwen_samplers, qwen_schedulers,
    resolve_run_params, PID_BACKBONE,
};
use crate::text_encoder::vision::grid::Grid;
use crate::text_encoder::QwenVisionLanguageEncoder;
use crate::transformer::QwenTransformer;
use crate::vae::QwenVae;
use crate::vl_tokenizer::{
    condition_resize_dims, encode_reference_latents, preprocess_edit_image, tokenize_edit_text,
};

/// Registry id for Qwen-Image-Edit.
pub const MODEL_ID: &str = "qwen_image_edit";

/// Qwen-Image-Edit's identity + capabilities. Accepts one `Reference` or N `MultiReference`
/// conditioning images — the fork's `use_picture_prefix=False` edit path, where every reference is
/// VAE-encoded and folded into the transformer's dual-latent sequence (sc-2529).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "qwen-image",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
            ],
            // LoRA/LoKr wired (sc-2528): shared `QwenTransformer` host; stacked + mixed.
            supports_lora: true,
            supports_lokr: true,
            // `lightning` = the few-step Lightning sampler (sc-2909), e.g.
            // `lightx2v/Qwen-Image-Edit-2511-Lightning`; an unset sampler is the production path.
            // Curated unified-framework integrator menu (epic 7114 P3) + the `lightning` profile.
            samplers: qwen_samplers(),
            schedulers: qwen_schedulers(),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: true,
        },
    }
}

/// A loaded Qwen-Image-Edit generator: the cached descriptor, the (tiny, always-warm) tokenizer +
/// image processor, and the heavy-component residency strategy (sc-11006).
pub struct QwenImageEdit {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    processor: QwenImageProcessor,
    /// Component-residency strategy (epic 10834 Phase 1, sc-11006; hoisted to the shared seam in
    /// sc-11125), selected from [`LoadSpec::offload_policy`]. `Resident` (default) holds the
    /// Qwen2.5-VL vision-language encoder, the DiT, and the VAE warm; `Sequential` holds only the
    /// per-phase loader closures and re-loads per generation in phase order (VL-encode, then **drop
    /// the VL encoder**, then dual-latent/denoise/decode), bounding peak unified memory to
    /// `max(VL-encoder, DiT+VAE)` — the Qwen2.5-VL encoder ≈16 GB is comparable to the DiT. The
    /// [`Residency`] seam owns the eval/drop/clear discipline, the stage-boundary cancel checks, and
    /// the error-safe cache flush.
    residency: Residency<QwenVisionLanguageEncoder, QwenEditHeavyOwned>,
}

/// The heavy render-phase components (the edit MMDiT transformer, the VAE, and the optional PiD
/// decoder) — everything but the VL encoder. Owned by the `Resident` components or by a `Sequential`
/// generate.
struct QwenEditHeavyOwned {
    transformer: QwenTransformer,
    vae: QwenVae,
    /// Optional PiD super-resolving decoder (epic 7840, sc-7845); see [`crate::model::QwenImage`].
    pid: Option<PidEngine>,
}

/// A borrow of the heavy render-phase components, so the denoise/decode body runs identically whether
/// they are held resident or were just loaded by the `Sequential` path (candle's `DitRef`).
struct QwenEditHeavy<'a> {
    transformer: &'a QwenTransformer,
    vae: &'a QwenVae,
    pid: Option<&'a PidEngine>,
}

impl QwenEditHeavyOwned {
    fn as_ref(&self) -> QwenEditHeavy<'_> {
        QwenEditHeavy {
            transformer: &self.transformer,
            vae: &self.vae,
            pid: self.pid.as_ref(),
        }
    }
}

/// Construct a [`QwenImageEdit`] from a [`LoadSpec`] (a `Qwen/Qwen-Image-Edit` snapshot dir; the
/// validated reference is `-2511`, `-2509` superseded — sc-2782/sc-2997).
/// `spec.quantize` (Q4/Q8) quantizes the **transformer only** (group_size 64) after the dense bf16
/// load — same as T2I ([`crate::model::load`]). This is the fork's full `quantize=N` scope, not a
/// descope: the Edit variant uses the same `QwenWeightDefinition`, whose `text_encoder` component
/// (the VL model — **LM + vision tower**, all under `text_encoder/`) is `skip_quantization=True`,
/// and whose VAE is all-conv (no `to_quantized` leaves). So the VL encoder and VAE stay bf16,
/// matching the fork (sc-2565).
///
/// Component residency (epic 10834 Phase 1, sc-11006): `Resident` (default) builds every heavy
/// component now via [`load_components`] and holds it warm; `Sequential` keeps only the spec and
/// re-loads per generate in phase order (VL-encode → drop the VL encoder → dual-latent/denoise/decode)
/// to bound peak memory to `max(VL-encoder, DiT+VAE)`. Both use the same per-phase loaders, so the
/// components are byte-identical.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let (tokenizer, processor, residency) = match spec.offload_policy {
        OffloadPolicy::Resident => {
            let c = load_components(spec)?;
            (
                c.tokenizer,
                c.processor,
                Residency::resident(
                    c.vl_encoder,
                    QwenEditHeavyOwned {
                        transformer: c.transformer,
                        vae: c.vae,
                        pid: c.pid,
                    },
                ),
            )
        }
        OffloadPolicy::Sequential => {
            // Validate precision + snapshot dir up front (fail fast, same as `Resident`); the heavy
            // build is deferred to each generate via the loader closures below.
            let root = resolve_root(spec)?;
            // F-181: Sequential + a load-time quant over a dense snapshot re-quantizes every generate.
            if let Some(q) = spec.quantize {
                if loader::needs_load_time_quant(root, q.bits(), MODEL_ID)? {
                    mlx_gen::residency::warn_sequential_requantize(MODEL_ID, q.bits());
                }
            }
            let tokenizer = loader::load_tokenizer(root)?;
            let spec_text = spec.clone();
            let spec_heavy = spec.clone();
            let residency = Residency::sequential(
                move || load_vl_encoder_only(resolve_root(&spec_text)?),
                move |use_pid| load_heavy(&spec_heavy, resolve_root(&spec_heavy)?, use_pid),
            );
            (tokenizer, QwenImageProcessor::default(), residency)
        }
    };
    Ok(Box::new(QwenImageEdit {
        descriptor: descriptor(),
        tokenizer,
        processor,
        residency,
    }))
}

/// Precision guard (only dense bf16 is wired) + snapshot-dir resolution (rejecting a single-file
/// source), shared by [`load_components`] and the `Sequential` per-phase loaders (sc-11006).
fn resolve_root(spec: &LoadSpec) -> Result<&Path> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "qwen_image_edit: only dense bf16 is wired in the Rust port (drop the precision override)"
                .into(),
        ));
    }
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(Error::Msg(
            "qwen_image_edit expects a snapshot directory (tokenizer/ text_encoder/ transformer/ \
             vae/), not a single .safetensors file"
                .into(),
        )),
    }
}

/// Load the Qwen2.5-VL vision-language encoder — the phase-A component dropped first under
/// `Sequential`. This is the LM (`model.*`) + the vision transformer (`visual.*`), both parsed once
/// from the shared `text_encoder/` shard set (F-080). Never quantized (the fork marks the
/// `text_encoder` component `skip_quantization=True`), so the `Resident` and `Sequential` paths build
/// byte-identical encoders.
fn load_vl_encoder_only(root: &Path) -> Result<QwenVisionLanguageEncoder> {
    loader::load_vision_language_encoder(root)
}

/// Load the heavy render-phase components — the edit MMDiT transformer (+ Q4/Q8 + LoRA/LoKr
/// residuals), VAE, and the optional PiD overlay — everything but the VL encoder. Factored so the
/// `Sequential` path loads these AFTER the encoder is dropped (bounding peak to `max(VL, DiT+VAE)`).
/// Quantize-then-adapters order matches the pre-sc-11006 `load`; the components are independent of the
/// VL encoder (separate weight files, deterministic RNG-free quant), so the `Resident` composition is
/// byte-identical.
fn load_heavy(spec: &LoadSpec, root: &Path, load_pid: bool) -> Result<QwenEditHeavyOwned> {
    // Edit-2511 transformer (zero_cond_t on): clean-timestep modulation for the conditioning tokens.
    let mut transformer = loader::load_transformer_edit(root)?;
    if let Some(q) = spec.quantize {
        // F-076: reject a requested-vs-packed quant-tier mismatch instead of silently serving the
        // snapshot's tier; skip the no-op quantize when the turnkey is already packed at the
        // requested bits (see `loader::needs_load_time_quant`) — same guard as the T2I loader.
        if loader::needs_load_time_quant(root, q.bits(), MODEL_ID)? {
            transformer.quantize(q.bits())?;
        }
    }
    // LoRA/LoKr (sc-2528): same load-time, post-quantize, residual-over-base path as T2I.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_qwen_adapters(&mut transformer, &spec.adapters)?;
    }
    // Optional PiD overlay, loaded only when the spec carries it AND this generate uses it (`load_pid`,
    // F-177) — Resident passes `true`, Sequential passes `req.use_pid` so a non-PiD generate skips the
    // student + its Gemma-2 caption encoder entirely.
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };
    let vae = loader::load_vae(root)?;
    Ok(QwenEditHeavyOwned {
        transformer,
        vae,
        pid,
    })
}

/// The Qwen-Image-Edit components loaded from a snapshot for the `Resident` path — composed
/// (sc-11006) from the same per-phase loaders the `Sequential` residency uses, so both build the
/// identical set.
struct QwenEditComponents {
    tokenizer: TextTokenizer,
    processor: QwenImageProcessor,
    vl_encoder: QwenVisionLanguageEncoder,
    transformer: QwenTransformer,
    vae: QwenVae,
    pid: Option<PidEngine>,
}

fn load_components(spec: &LoadSpec) -> Result<QwenEditComponents> {
    let root = resolve_root(spec)?;
    let vl_encoder = load_vl_encoder_only(root)?;
    // Resident loads the PiD overlay unconditionally (loaded once, reused); the per-request `use_pid`
    // skip (F-177) applies only to the Sequential per-generate loader.
    let heavy = load_heavy(spec, root, true)?;
    let tokenizer = loader::load_tokenizer(root)?;
    Ok(QwenEditComponents {
        tokenizer,
        processor: QwenImageProcessor::default(),
        vl_encoder,
        transformer: heavy.transformer,
        vae: heavy.vae,
        pid: heavy.pid,
    })
}

impl QwenImageEdit {
    /// Edit conditioning embeds (f16, matching the fork) for one prompt: tokenize the edit template
    /// (the `<|image_pad|>` run length is `n_image_tokens`, from the shared image preprocess), then
    /// run the LM over the spliced sequence reusing the already-computed `vision` embeds — so the
    /// vision tower is **not** re-run for the positive vs negative prompt (F-004). Takes the encoder
    /// as an argument so the `Resident` (warm) and `Sequential` (just-loaded) paths share this body.
    fn encode_edit(
        &self,
        vl: &QwenVisionLanguageEncoder,
        prompt: &str,
        n_image_tokens: usize,
        vision: &Array,
    ) -> Result<Array> {
        let tok = tokenize_edit_text(&self.tokenizer, prompt, n_image_tokens)?;
        let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&tok);
        let embeds = vl.encode_with_vision(&input_ids, &attention_mask, vision)?;
        Ok(embeds.as_dtype(Dtype::Float16)?)
    }

    /// Run the full phase-A VL pass (epic 10834 Phase 1, sc-11006): preprocess the **first** reference,
    /// run the **vision tower** over it, then the **LM** over the positive (and, for true CFG, negative)
    /// prompts reusing that vision output. The tower runs once (image-only), so the positive + negative
    /// encodes reuse it (F-004). `neg` is `None` under Lightning (CFG-distilled → one forward/step).
    /// Called by the shared residency seam's encode closure with the phase-A `vl` encoder.
    fn encode_phase_a(
        &self,
        vl: &QwenVisionLanguageEncoder,
        req: &GenerationRequest,
        is_lightning: bool,
    ) -> Result<(Array, Option<Array>)> {
        // The fork's `use_picture_prefix=False` edit template carries a single `<|image_pad|>`, so
        // only the **first** reference enters the prompt embeds; its block-diagonal vision output is
        // identical whether computed alone or alongside the others. `generate_impl` validates
        // non-empty before calling this.
        let references = reference_images(req);
        let first = *references
            .first()
            .ok_or_else(|| Error::Msg("qwen_image_edit: no reference image to encode".into()))?;
        let pre = preprocess_edit_image(&self.processor, image_input(first))?;
        let grids: Vec<Grid> = host_i32(&pre.grid_thw)?
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let vision = vl.encode_vision(&pre.pixel_values, &grids)?;
        let pos = self.encode_edit(vl, &req.prompt, pre.n_image_tokens, &vision)?;
        let neg = if is_lightning {
            None
        } else {
            Some(self.encode_edit(
                vl,
                req.negative_prompt.as_deref().unwrap_or(""),
                pre.n_image_tokens,
                &vision,
            )?)
        };
        Ok((pos, neg))
    }
}

impl Generator for QwenImageEdit {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor.capabilities, req)?;
        validate_reference_images(req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

impl QwenImageEdit {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        // Shared step/sampler/guidance/seed resolution (F-117): `req.sampler == "lightning"` selects
        // the few-step recipe (its matching Edit Lightning LoRA must be supplied via `spec.adapters`),
        // else the production resolution-dependent schedule.
        let (out_w, out_h) = (req.width, req.height);
        let params = resolve_run_params(req, out_w, out_h);

        // Phase A: reference + prompts → conditioning embeds (epic 10834 Phase 1, sc-11006; sc-11125).
        // Under `Sequential` the shared seam loads the Qwen2.5-VL encoder, runs the vision tower over
        // the reference + the LM over pos/neg, materializes, then DROPS it + `clear_cache()` so its
        // ~16 GB frees before the DiT/VAE load below. Under `Resident` it borrows the warm encoder. The
        // encode carries no RNG, so ordering it before the dual-latent VAE-encode is byte-identical.
        self.residency.run(
            &req.cancel,
            req.use_pid,
            |vl: &QwenVisionLanguageEncoder| self.encode_phase_a(vl, req, params.is_lightning),
            // Materialize pos (+neg) while the VL encoder is still alive (Sequential only) — this forces
            // the vision-tower AND LM forwards, else the outputs keep the encoder referenced and the
            // drop would free nothing.
            |(pos, neg)| {
                match neg {
                    Some(neg) => mlx_rs::transforms::eval([pos, neg])?,
                    None => mlx_rs::transforms::eval([pos])?,
                }
                Ok(())
            },
            // ── Establish the heavy render components (edit DiT + VAE + PiD) and run the dual-latent
            // VAE-encode + denoise/decode body once against the `heavy` borrow — identical for both.
            |heavy_owned, enc| {
                let heavy = heavy_owned.as_ref();
                let (pos, neg) = enc;

                let references = reference_images(req);
                let last = *references.last().expect("validated non-empty");

                // VL condition / dual-latent reference resolution (~384² area, /32). The fork's
                // `_compute_dimensions` derives all dims from `image_paths[-1]`, so the dual-latent resolution
                // comes from the **last** reference's aspect (identical to the first when the references share
                // an aspect ratio, the common case).
                let (vl_w, vl_h) = condition_resize_dims(last.width as usize, last.height as usize);

                // Dual-latent references (static across steps + samples): VAE-encode **each** reference at the
                // VL resolution, pack, and concatenate over the sequence axis — one `cond_grid` per reference
                // so the MMDiT RoPE spans `[noise] + references` (fork
                // `QwenEditUtil.create_image_conditioning_latents` + `forward_multi`). This is a deterministic
                // VAE encode, independent of `pos`/`neg`, so under `Sequential` running it here — after the VL
                // drop, with the VAE just loaded — is byte-identical to the Resident order (same hoist
                // argument as the T2I img2img `encode_init_latents`).
                let mut packed = Vec::with_capacity(references.len());
                let mut cond_grids = Vec::with_capacity(references.len());
                for im in &references {
                    let (latents, grid) = encode_reference_latents(
                        heavy.vae,
                        image_input(im),
                        vl_w as u32,
                        vl_h as u32,
                    )?;
                    packed.push(latents);
                    cond_grids.push(grid);
                }
                let static_latents = if packed.len() == 1 {
                    packed.pop().expect("len checked")
                } else {
                    concatenate_axis(&packed.iter().collect::<Vec<_>>(), 1)?
                };

                // Decode seam (sc-7845) + `from_ldm` early-stop (sc-7993): the partially-denoised x_k at the
                // achieved σ (truncated schedule) when use_pid + pid_capture_sigma; else the clean σ=0 path.
                // Edit denoises from full noise (no img2img init), so `start_step = 0`.
                let (capture_sigma, keep) = flow_capture_for_request(req, &params.sigmas, 0);
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    heavy.pid,
                    req,
                    params.base_seed,
                    MODEL_ID,
                    capture_sigma,
                )?;
                let decoder: &dyn LatentDecoder = match &pid_decoder {
                    Some(d) => d,
                    None => heavy.vae,
                };
                let denoise_sigmas = &params.sigmas[..keep];
                let images = decode_and_collect(
                    decoder,
                    req.count,
                    params.base_seed,
                    out_w,
                    out_h,
                    on_progress,
                    |seed, progress| {
                        let noise = create_noise(seed, out_w, out_h)?;
                        denoise_edit_with_progress(
                            heavy.transformer,
                            params.sampler_name.as_deref(),
                            denoise_sigmas,
                            seed,
                            noise,
                            &static_latents,
                            &cond_grids,
                            &pos,
                            neg.as_ref(),
                            params.guidance,
                            out_w,
                            out_h,
                            &req.cancel,
                            progress,
                        )
                    },
                )?;
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

/// Borrow an [`Image`] as an [`ImageInput`] (RGB uint8 HWC) for the preprocess/VAE-encode paths.
fn image_input(im: &Image) -> ImageInput<'_> {
    ImageInput {
        data: &im.pixels,
        height: im.height as usize,
        width: im.width as usize,
    }
}

/// The conditioning reference images, in order — a single `Reference` or every `MultiReference`
/// image. The first drives the text/VL prompt embeds (fork `use_picture_prefix=False`); all of them
/// are VAE-encoded into the dual-latent sequence.
fn reference_images(req: &GenerationRequest) -> Vec<&Image> {
    let mut out = Vec::new();
    for c in &req.conditioning {
        match c {
            Conditioning::Reference { image, .. } => out.push(image),
            Conditioning::MultiReference { images } => out.extend(images.iter()),
            _ => {}
        }
    }
    out
}

/// Require at least one reference image, each with nonzero dims and a `w*h*3` pixel buffer. The edit
/// path feeds reference pixels straight into `resize_bicubic_u8`/`resize_lanczos_u8` (which index
/// `src[(y*in_w + x)*3 + ch]`) and `condition_resize_dims` (which divides by the dims), so an
/// undersized buffer panics OOB, an oversized one reads garbage, and a zero dimension yields NaN dims
/// — exactly what the T2I path already rejects in `preprocess_init_image` (F-112). Validate once here,
/// at the request boundary, so a malformed `qwen_image_edit` request errors cleanly instead of
/// crashing the engine.
fn validate_reference_images(req: &GenerationRequest) -> Result<()> {
    let refs = reference_images(req);
    if refs.is_empty() {
        return Err(Error::Msg(
            "qwen_image_edit requires a Reference or MultiReference conditioning image".into(),
        ));
    }
    for img in refs {
        let (w, h) = (img.width as usize, img.height as usize);
        if w == 0 || h == 0 {
            return Err(Error::Msg(format!(
                "qwen_image_edit: reference image has a zero dimension ({w}x{h})"
            )));
        }
        if img.pixels.len() != w * h * 3 {
            return Err(Error::Msg(format!(
                "qwen_image_edit: reference image pixel buffer {} != {w}x{h}x3",
                img.pixels.len()
            )));
        }
        // An extreme aspect ratio (a thin strip) survives the nonzero-dims check above but rounds
        // a condition-resize side down to 0 (`round_ties_even(side/32) == 0`), which then feeds a
        // zero-dim resize / `sqrt(minp/0)` downstream → empty latents or NaN. Reject it here so the
        // dual-latent (`last`) and VL (`first`) reference dims are both validated (F-005).
        let (cw, ch) = condition_resize_dims(w, h);
        if cw == 0 || ch == 0 {
            return Err(Error::Msg(format!(
                "qwen_image_edit: reference image aspect ratio ({w}x{h}) is too extreme; its \
                 condition-resize collapses to a zero dimension ({cw}x{ch})"
            )));
        }
    }
    Ok(())
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`. The `impl
// Generator` above stays hand-written because `validate` adds a reference-image check beyond the
// shared `validate_request`, so it is not the plain delegation `impl_generator!` expresses.
mlx_gen::register_generators! { descriptor => load }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_qwen_image_edit() {
        let d = descriptor();
        assert_eq!(d.id, "qwen_image_edit");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::MultiReference));
        assert!(!d.capabilities.accepts(ConditioningKind::Depth));
    }

    #[test]
    fn load_accepts_q8_spec() {
        // Q8 is wired (transformer-only, slice 7b): a Q8 spec must get past the quant gate and fail
        // later on the missing snapshot, not on quantization being unsupported.
        let spec =
            LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(mlx_gen::Quant::Q8);
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(!err.contains("not wired"), "got: {err}");
    }

    #[test]
    fn load_rejects_single_file() {
        // A single-file source is rejected up front (the snapshot-dir guard), for both residencies.
        let spec = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    #[test]
    fn generate_requires_a_reference_image() {
        let caps = descriptor().capabilities;
        // A valid-size request with no Reference conditioning fails validation.
        let req = GenerationRequest {
            prompt: "make it autumn".into(),
            ..Default::default()
        };
        // validate_request (size/conditioning) passes, but the edit generator needs a reference.
        assert!(validate_request(&caps, &req).is_ok());
        assert!(reference_images(&req).is_empty());
    }

    #[test]
    fn validate_reference_images_rejects_bad_buffers() {
        use mlx_gen::Conditioning;
        let reference = |image| GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image,
                strength: None,
            }],
            ..Default::default()
        };

        // No reference at all.
        assert!(validate_reference_images(&GenerationRequest::default()).is_err());

        // Short pixel buffer (would index OOB in the resize inner loop).
        let short = reference(Image {
            width: 8,
            height: 8,
            pixels: vec![0u8; 8 * 8 * 3 - 1],
        });
        assert!(validate_reference_images(&short)
            .unwrap_err()
            .to_string()
            .contains("pixel buffer"));

        // Oversized buffer (would silently read garbage).
        let long = reference(Image {
            width: 8,
            height: 8,
            pixels: vec![0u8; 8 * 8 * 3 + 5],
        });
        assert!(validate_reference_images(&long).is_err());

        // Zero dimension (would drive condition_resize_dims to NaN).
        let zero = reference(Image {
            width: 0,
            height: 8,
            pixels: Vec::new(),
        });
        assert!(validate_reference_images(&zero)
            .unwrap_err()
            .to_string()
            .contains("zero dimension"));

        // A well-formed reference passes. One bad image in a MultiReference still fails.
        let good_img = Image {
            width: 8,
            height: 8,
            pixels: vec![0u8; 8 * 8 * 3],
        };
        assert!(validate_reference_images(&reference(good_img.clone())).is_ok());
        let mixed = GenerationRequest {
            conditioning: vec![Conditioning::MultiReference {
                images: vec![
                    good_img,
                    Image {
                        width: 8,
                        height: 8,
                        pixels: vec![0u8; 4],
                    },
                ],
            }],
            ..Default::default()
        };
        assert!(validate_reference_images(&mixed).is_err());
    }

    #[test]
    fn reference_images_collects_single_and_multi() {
        use mlx_gen::Conditioning;
        let img = |w| Image {
            width: w,
            height: 8,
            pixels: vec![0u8; (w * 8 * 3) as usize],
        };
        // A single `Reference` yields one image.
        let single = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: img(8),
                strength: None,
            }],
            ..Default::default()
        };
        assert_eq!(reference_images(&single).len(), 1);
        // `MultiReference` yields every image, in order (first drives the text path, last the dims).
        let multi = GenerationRequest {
            conditioning: vec![Conditioning::MultiReference {
                images: vec![img(8), img(16), img(24)],
            }],
            ..Default::default()
        };
        let got = reference_images(&multi);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].width, 8);
        assert_eq!(got.last().unwrap().width, 24);
    }
}
