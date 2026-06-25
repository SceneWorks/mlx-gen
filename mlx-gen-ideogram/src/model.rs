//! `Ideogram4` — the [`mlx_gen::Generator`] implementation for Ideogram 4.0, plus its
//! [`descriptor`]/[`load`] entry points and the `inventory` registration that wires it into
//! `mlx_gen`'s registry under id `"ideogram_4"` (sc-5988). Linking this crate is all the worker
//! needs to resolve the model by id.
//!
//! [`load`] assembles the pipeline (2 DiTs + Qwen3-VL TE + VAE + tokenizer) from a converted
//! snapshot directory ([`crate::pipeline::Ideogram4Pipeline`]); [`Ideogram4::generate`] runs the
//! full prompt→image flow per requested image — tokenize the (JSON-caption) prompt natively,
//! asymmetric-CFG flow-match denoise, VAE decode → RGB8 — honoring `req.cancel` and streaming
//! `Progress`. `spec.quantize` (Q4/Q8) quantizes the whole model in place after the dense load
//! (sc-5989); a precision override and LoRA adapters are rejected rather than silently ignored.

use mlx_gen::array::host_i32;
use mlx_gen::{
    default_seed, AdapterKind, AdapterSpec, Capabilities, Conditioning, ConditioningKind, Error,
    GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor,
    Precision, Progress, Quant, Result, WeightsSource,
};
use mlx_rs::{Array, Dtype};

use crate::config::{
    DEFAULT_GUIDANCE, DEFAULT_IMG2IMG_STRENGTH, DEFAULT_INPAINT_STRENGTH, DEFAULT_STEPS,
    DEFAULT_TURBO_STEPS, IDEOGRAM_4_ID, IDEOGRAM_4_TURBO_ID, RES_MAX, RES_MIN, RES_MULTIPLE,
    TURBO_LORA_FILE, TURBO_LORA_SCALE,
};
use crate::pipeline::Ideogram4Pipeline;

/// Registry id (matches the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = IDEOGRAM_4_ID;

/// Registry id for the few-step CFG-free turbo variant (issue #488).
pub const MODEL_ID_TURBO: &str = IDEOGRAM_4_TURBO_ID;

/// Max images per request (the image-model standard, shared with the other MLX families).
const MAX_COUNT: u32 = 8;
/// Max aspect ratio (long:short) — the reference supports up to 6:1.
const MAX_ASPECT: u32 = 6;

/// Ideogram 4's identity + capabilities — constructible without loading weights (registry
/// introspection / capability advertisement).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "ideogram",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Asymmetric CFG runs a separate *unconditional* DiT — the "negative" is a fixed
            // trained model, not a user negative prompt — so `guidance` is offered but a negative
            // prompt is not.
            supports_negative_prompt: false,
            supports_guidance: true,
            supports_true_cfg: false,
            // Edit (sc-6303/6330): img2img / Remix via a source `Reference`, and mask inpaint via a
            // `Mask` (white = repaint) alongside the `Reference`. The prompt stays the model's native
            // JSON caption. No control/pose/multi-reference. Edit works in both quality and turbo.
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::Mask],
            supports_lora: false,
            supports_lokr: false,
            // Bespoke-by-architecture (epic 7114, sc-7120, task 7184): Ideogram is NOT routed through
            // the unified curated-sampler framework. Its `LogitNormalSchedule` is an INVERTED, clamped
            // logit-normal time grid (no `σ = 0` terminal), so the FLOW `x0 = x − σ·v` estimate the
            // multistep/2nd-order solvers (heun / dpmpp_2m / uni_pc) require is meaningless; it uses a
            // per-step CFG guidance schedule (POLISH_GUIDANCE on the final steps) and an inpaint
            // mask-blend interleaved BETWEEN Euler steps (no post-step hook in `run_flow_sampler`).
            // Advertising the curated menu would expose solvers that produce broken output — so the
            // native logit-normal Euler is its only valid sampler. See `pipeline::run_denoise`.
            samplers: Vec::new(),
            schedulers: Vec::new(),
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            mac_only: true,
            // Load-time Q4/Q8 over the whole model (both DiTs + TE + VAE), sc-5989. Q8 default is
            // the worker's call; Q4 roughly halves the ~27 GB Q8 weights for smaller Macs.
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// Ideogram 4 **turbo** identity + capabilities (issue #488). Same surface as [`descriptor`] except
/// it is **CFG-free** — the TurboTime LoRA distilled the guided velocity into a single DiT, so
/// `guidance` is not offered (no unconditional branch). Few-step (`DEFAULT_TURBO_STEPS`), single DiT.
pub fn descriptor_turbo() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = MODEL_ID_TURBO;
    // CFG-free: there is no unconditional DiT to mix against, so guidance is inert. (negative-prompt
    // and true_cfg were already off.)
    d.capabilities.supports_guidance = false;
    d
}

/// A loaded Ideogram 4 generator: the assembled pipeline plus the cached descriptor.
pub struct Ideogram4 {
    descriptor: ModelDescriptor,
    pipeline: Ideogram4Pipeline,
}

/// Construct an [`Ideogram4`] from a [`LoadSpec`]. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a converted snapshot (`transformer/ unconditional_transformer/ text_encoder/ vae/
/// tokenizer/`). Dense bf16 by default; `spec.quantize` (Q4/Q8) quantizes the whole model in place
/// after the dense load. A precision override and LoRA/LoKr adapters are not wired and are rejected
/// rather than silently ignored.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "ideogram_4: only dense bf16 is wired (drop the precision override)".into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "ideogram_4: LoRA/LoKr adapters are not supported".into(),
        ));
    }
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p,
            WeightsSource::File(_) => return Err(Error::Msg(
                "ideogram_4 expects a snapshot directory (transformer/ unconditional_transformer/ \
                 text_encoder/ vae/ tokenizer/), not a single .safetensors file"
                    .into(),
            )),
        };
    // Q4/Q8 quantizes the whole model (both DiTs + TE + VAE) in place after the dense bf16 load.
    // NOTE: peak footprint is the *dense* bf16 transient (~53 GB) — pre-quantized-on-disk weights
    // avoid it for smaller Macs (the SCAIL-2 lesson; tracked in this story).
    let mut pipeline = Ideogram4Pipeline::load(root)?;
    if let Some(q) = spec.quantize {
        pipeline.quantize(q.bits())?;
    }
    Ok(Box::new(Ideogram4 {
        descriptor: descriptor(),
        pipeline,
    }))
}

/// Construct an [`Ideogram4`] **turbo** generator (issue #488) from a [`LoadSpec`]. `spec.weights`
/// must be a [`WeightsSource::Dir`] pointing at a turbo snapshot — the conditional `transformer/`,
/// `text_encoder/`, `vae/`, `tokenizer/`, plus the bundled [`TURBO_LORA_FILE`]; the unconditional
/// DiT is not loaded. Loads the single DiT, quantizes (Q4/Q8) if requested, then installs the
/// TurboTime LoRA at scale 1.0 — the CFG-free few-step path. A precision override or **user**
/// LoRA/LoKr adapters are rejected (the TurboTime LoRA is part of the snapshot, not user-supplied).
pub fn load_turbo(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "ideogram_4_turbo: only dense bf16 is wired (drop the precision override)".into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "ideogram_4_turbo: user LoRA/LoKr adapters are not supported (the TurboTime LoRA is \
             bundled in the snapshot)"
                .into(),
        ));
    }
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p,
            WeightsSource::File(_) => return Err(Error::Msg(
                "ideogram_4_turbo expects a snapshot directory (transformer/ text_encoder/ vae/ \
                 tokenizer/ + turbo_lora.safetensors), not a single .safetensors file"
                    .into(),
            )),
        };
    let lora_path = root.join(TURBO_LORA_FILE);
    if !lora_path.exists() {
        return Err(Error::Msg(format!(
            "ideogram_4_turbo: bundled TurboTime LoRA not found at {} (a turbo snapshot must \
             include {TURBO_LORA_FILE})",
            lora_path.display()
        )));
    }
    // Single conditional DiT (no unconditional branch) + TE + VAE + tokenizer.
    let mut pipeline = Ideogram4Pipeline::load_turbo(root)?;
    // Quantize the base first (Q4/Q8), then install the LoRA residual on top — fork-faithful order
    // (matches the flux2 family): the residual is computed and added over the possibly quantized base.
    if let Some(q) = spec.quantize {
        pipeline.quantize(q.bits())?;
    }
    pipeline.apply_adapters(&[AdapterSpec {
        path: lora_path,
        scale: TURBO_LORA_SCALE,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: None,
    }])?;
    Ok(Box::new(Ideogram4 {
        descriptor: descriptor_turbo(),
        pipeline,
    }))
}

mlx_gen::impl_generator!(Ideogram4 {
    validate: |s, req| validate_request(&s.descriptor.capabilities, req),
    generate: generate_impl,
});

impl Ideogram4 {
    /// The rich-`Result` body behind [`Generator::generate`] — kept on the crate's own
    /// [`mlx_gen::Error`] so `?` lifts `mlx_rs` device exceptions transparently; the trait wrapper
    /// bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        validate_request(&self.descriptor.capabilities, req)?;

        // Turbo defaults to the few-step count; quality mode to the 48-step preset. `guidance` is
        // inert in turbo (the pipeline runs CFG-free when the unconditional DiT is absent).
        let default_steps = if self.pipeline.is_turbo() {
            DEFAULT_TURBO_STEPS
        } else {
            DEFAULT_STEPS
        };
        let steps = req.steps.unwrap_or(default_steps) as usize;
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let base_seed = req.seed.unwrap_or_else(default_seed);

        // Edit (img2img / inpaint): resolve a source `Reference` (+ optional `Mask`) and VAE-encode
        // the source once (seed-independent). `None` → the text-to-image path (byte-identical).
        let edit_init = match resolve_edit(req)? {
            Some((source, mask, strength)) => Some(
                self.pipeline
                    .prepare_edit(source, mask, strength, req.height, req.width)?,
            ),
            None => None,
        };

        // Tokenize once — the JSON caption is identical across the count loop; only the seed varies.
        let ids = self.pipeline.tokenize(&req.prompt)?;

        let mut images = Vec::with_capacity(req.count as usize);
        for n in 0..req.count {
            let seed = base_seed.wrapping_add(n as u64);
            let arr = match &edit_init {
                Some(edit) => self.pipeline.generate_edit_with_progress(
                    &ids,
                    req.height,
                    req.width,
                    steps,
                    guidance,
                    seed,
                    edit,
                    &req.cancel,
                    on_progress,
                )?,
                None => self.pipeline.generate_with_progress(
                    &ids,
                    req.height,
                    req.width,
                    steps,
                    guidance,
                    seed,
                    &req.cancel,
                    on_progress,
                )?,
            };
            images.push(array_to_image(&arr)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Resolve the optional edit conditioning: a single img2img/inpaint source [`Conditioning::Reference`]
/// plus an optional [`Conditioning::Mask`]. Returns `(source, mask, strength)`; `None` for pure
/// text-to-image. A per-reference strength wins over `req.strength`, else the img2img/inpaint
/// default. More than one `Reference`/`Mask`, or a `Mask` without a `Reference`, is an error.
fn resolve_edit(req: &GenerationRequest) -> Result<Option<(&Image, Option<&Image>, f32)>> {
    let mut source: Option<(&Image, Option<f32>)> = None;
    let mut mask: Option<&Image> = None;
    for c in &req.conditioning {
        match c {
            Conditioning::Reference { image, strength } => {
                if source.is_some() {
                    return Err(Error::Msg(
                        "ideogram_4: only one reference (source) image is supported for edit"
                            .into(),
                    ));
                }
                source = Some((image, strength.or(req.strength)));
            }
            Conditioning::Mask { image } => {
                if mask.is_some() {
                    return Err(Error::Msg(
                        "ideogram_4: only one inpaint mask is supported".into(),
                    ));
                }
                mask = Some(image);
            }
            // Other conditioning kinds are rejected by the capability floor in `validate_request`.
            _ => {}
        }
    }
    match source {
        Some((image, strength)) => {
            let default = if mask.is_some() {
                DEFAULT_INPAINT_STRENGTH
            } else {
                DEFAULT_IMG2IMG_STRENGTH
            };
            Ok(Some((image, mask, strength.unwrap_or(default))))
        }
        None if mask.is_some() => Err(Error::Msg(
            "ideogram_4: an inpaint mask requires a reference (source) image".into(),
        )),
        None => Ok(None),
    }
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Layers Ideogram's model-specific constraints (non-empty prompt, size multiple-of-16,
/// aspect ≤ 6:1, steps ≥ 1) on top of the shared [`Capabilities::validate_request`] floor
/// (count/size range, negative/guidance/true_cfg flags, conditioning kinds).
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if req.prompt.is_empty() {
        return Err(Error::Msg(
            "ideogram_4: prompt must not be empty (Ideogram 4 expects a JSON caption)".into(),
        ));
    }
    // `?` converts the shared floor's `gen_core::Error` into the crate's `Error` (From impl).
    caps.validate_request(MODEL_ID, req)?;
    if req.steps == Some(0) {
        return Err(Error::Msg("ideogram_4: steps must be >= 1".into()));
    }
    // The pipeline needs dims divisible by patch(2) × ae_scale(8) = 16, or `patchify`'s reshape
    // blows up deep in MLX — reject at the boundary.
    if !req.width.is_multiple_of(RES_MULTIPLE) || !req.height.is_multiple_of(RES_MULTIPLE) {
        return Err(Error::Msg(format!(
            "ideogram_4: {}x{} must be a multiple of {RES_MULTIPLE}",
            req.width, req.height
        )));
    }
    let (long, short) = (req.width.max(req.height), req.width.min(req.height));
    if long > short * MAX_ASPECT {
        return Err(Error::Msg(format!(
            "ideogram_4: aspect ratio of {}x{} exceeds the supported {MAX_ASPECT}:1",
            req.width, req.height
        )));
    }
    // Edit: an inpaint `Mask` is meaningless without a source `Reference` to keep/blend against
    // (the capability floor admits both kinds individually; this enforces the pairing). Multiple
    // references / masks are caught in `resolve_edit` at generate time.
    let has_ref = req
        .conditioning
        .iter()
        .any(|c| matches!(c, Conditioning::Reference { .. }));
    let has_mask = req
        .conditioning
        .iter()
        .any(|c| matches!(c, Conditioning::Mask { .. }));
    if has_mask && !has_ref {
        return Err(Error::Msg(
            "ideogram_4: an inpaint mask requires a reference (source) image".into(),
        ));
    }
    Ok(())
}

/// Host-extract the pipeline's `[H, W, 3]` u8 RGB array into an [`Image`].
fn array_to_image(img: &Array) -> Result<Image> {
    let sh = img.shape();
    let (h, w) = (sh[0] as u32, sh[1] as u32);
    let px = host_i32(&img.as_dtype(Dtype::Int32)?)?;
    Ok(Image {
        width: w,
        height: h,
        pixels: px.into_iter().map(|v| v as u8).collect(),
    })
}

// Link-time registration (epic 3720): the macro emits each `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`. The turbo variant
// (issue #488) registers under `ideogram_4_turbo`.
mlx_gen::register_generators! {
    descriptor => load,
    descriptor_turbo => load_turbo,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core;

    fn caps() -> Capabilities {
        descriptor().capabilities
    }

    /// A valid request with a (stand-in) JSON-caption prompt.
    fn req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: r#"{"high_level_description":"a fox"}"#.into(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_is_ideogram_4() {
        let d = descriptor();
        assert_eq!(d.id, "ideogram_4");
        assert_eq!(d.family, "ideogram");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        // Edit surface (sc-6303/6330): img2img Reference + inpaint Mask.
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::Reference));
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::Mask));
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(
            (d.capabilities.min_size, d.capabilities.max_size),
            (256, 2048)
        );
    }

    #[test]
    fn validate_accepts_in_surface() {
        assert!(validate_request(&caps(), &req(1024, 1024)).is_ok());
        // Exactly 6:1 is allowed (1536 / 256 = 6).
        assert!(validate_request(&caps(), &req(256, 1536)).is_ok());
        // guidance is supported.
        assert!(validate_request(
            &caps(),
            &GenerationRequest {
                guidance: Some(7.0),
                ..req(512, 512)
            }
        )
        .is_ok());
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        let e = validate_request(&caps(), &GenerationRequest::default())
            .unwrap_err()
            .to_string();
        assert!(e.contains("empty"), "got: {e}");
    }

    #[test]
    fn validate_rejects_non_multiple_of_16() {
        for (w, h) in [(1000, 1000), (257, 256), (512, 520)] {
            let e = validate_request(&caps(), &req(w, h))
                .unwrap_err()
                .to_string();
            assert!(e.contains("multiple of 16"), "{w}x{h} got: {e}");
        }
    }

    #[test]
    fn validate_rejects_out_of_range_size() {
        assert!(validate_request(&caps(), &req(128, 128)).is_err()); // below min
        assert!(validate_request(&caps(), &req(2064, 256)).is_err()); // above max
    }

    #[test]
    fn validate_rejects_excessive_aspect() {
        // 1792 / 256 = 7:1 (> 6:1); in range and a multiple of 16, so only the aspect guard fires.
        let e = validate_request(&caps(), &req(256, 1792))
            .unwrap_err()
            .to_string();
        assert!(e.contains("aspect"), "got: {e}");
    }

    #[test]
    fn validate_rejects_zero_steps_and_negative_prompt() {
        assert!(validate_request(
            &caps(),
            &GenerationRequest {
                steps: Some(0),
                ..req(512, 512)
            }
        )
        .is_err());
        assert!(validate_request(
            &caps(),
            &GenerationRequest {
                negative_prompt: Some("x".into()),
                ..req(512, 512)
            }
        )
        .is_err());
    }

    fn img(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
    }

    #[test]
    fn validate_accepts_img2img_reference() {
        // Edit surface (sc-6303): a single img2img source Reference is now accepted.
        let r = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: img(512, 512),
                strength: Some(0.7),
            }],
            ..req(512, 512)
        };
        assert!(validate_request(&caps(), &r).is_ok());
    }

    #[test]
    fn validate_accepts_inpaint_reference_plus_mask() {
        let r = GenerationRequest {
            conditioning: vec![
                Conditioning::Reference {
                    image: img(512, 512),
                    strength: None,
                },
                Conditioning::Mask {
                    image: img(512, 512),
                },
            ],
            ..req(512, 512)
        };
        assert!(validate_request(&caps(), &r).is_ok());
    }

    #[test]
    fn validate_rejects_mask_without_reference() {
        let r = GenerationRequest {
            conditioning: vec![Conditioning::Mask {
                image: img(512, 512),
            }],
            ..req(512, 512)
        };
        let e = validate_request(&caps(), &r).unwrap_err().to_string();
        assert!(e.contains("requires a reference"), "got: {e}");
    }

    #[test]
    fn validate_rejects_unsupported_conditioning() {
        // A control/pose conditioning is out of surface → rejected by the capability floor.
        let r = GenerationRequest {
            conditioning: vec![Conditioning::Control {
                image: img(512, 512),
                kind: mlx_gen::ControlKind::Pose,
                scale: 1.0,
            }],
            ..req(512, 512)
        };
        assert!(validate_request(&caps(), &r).is_err());
    }

    #[test]
    fn resolve_edit_defaults_and_pairing() {
        // No conditioning → no edit.
        assert!(resolve_edit(&req(512, 512)).unwrap().is_none());
        // Reference only → img2img with the img2img default strength.
        let r = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: img(512, 512),
                strength: None,
            }],
            ..req(512, 512)
        };
        let (_, mask, strength) = resolve_edit(&r).unwrap().expect("edit");
        assert!(mask.is_none());
        assert_eq!(strength, DEFAULT_IMG2IMG_STRENGTH);
        // Reference + Mask → inpaint default strength; per-reference strength wins when present.
        let r = GenerationRequest {
            conditioning: vec![
                Conditioning::Reference {
                    image: img(512, 512),
                    strength: None,
                },
                Conditioning::Mask {
                    image: img(512, 512),
                },
            ],
            ..req(512, 512)
        };
        let (_, mask, strength) = resolve_edit(&r).unwrap().expect("edit");
        assert!(mask.is_some());
        assert_eq!(strength, DEFAULT_INPAINT_STRENGTH);
        // A second Reference is an error.
        let r = GenerationRequest {
            conditioning: vec![
                Conditioning::Reference {
                    image: img(512, 512),
                    strength: None,
                },
                Conditioning::Reference {
                    image: img(512, 512),
                    strength: None,
                },
            ],
            ..req(512, 512)
        };
        assert!(resolve_edit(&r).is_err());
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        // `Box<dyn Generator>` isn't Debug → use `.err()`.
        let e = load(&spec).err().expect("expected an error").to_string();
        assert!(e.contains("snapshot directory"), "got: {e}");
    }

    #[test]
    fn load_accepts_quant_spec() {
        // Q4/Q8 is wired (whole model) — a quant spec must get past the entry point and fail later
        // on the missing snapshot, not be rejected as unsupported.
        for q in [Quant::Q4, Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(q);
            let e = load(&spec).err().expect("expected an error").to_string();
            assert!(
                !e.contains("not yet wired"),
                "quant should be accepted: {e}"
            );
        }
    }

    #[test]
    fn reachable_via_registry_by_id() {
        // Linking this crate self-registers ideogram_4; it must be discoverable and resolve to OUR
        // loader (a nonexistent dir fails inside load, NOT with "no generator registered").
        assert!(gen_core::registry::generators().any(|r| (r.descriptor)().id == MODEL_ID));
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-ideogram".into()));
        let e = gen_core::registry::load(MODEL_ID, &spec)
            .err()
            .expect("missing weights → err")
            .to_string();
        assert!(
            !e.contains("no generator registered"),
            "id not resolved: {e}"
        );
    }

    // ── Turbo variant (issue #488) ────────────────────────────────────────────────────────

    #[test]
    fn descriptor_turbo_is_cfg_free_else_matches_base() {
        let (b, t) = (descriptor(), descriptor_turbo());
        assert_eq!(t.id, "ideogram_4_turbo");
        assert_eq!(t.family, b.family);
        assert_eq!(t.backend, b.backend);
        assert_eq!(t.modality, b.modality);
        // The one capability that differs: turbo is CFG-free (no unconditional DiT to mix against).
        assert!(b.capabilities.supports_guidance);
        assert!(!t.capabilities.supports_guidance);
        // Everything else is identical to the base surface.
        assert_eq!(
            t.capabilities.supports_negative_prompt,
            b.capabilities.supports_negative_prompt
        );
        assert_eq!(
            t.capabilities.supported_quants,
            b.capabilities.supported_quants
        );
        assert_eq!(
            (t.capabilities.min_size, t.capabilities.max_size),
            (b.capabilities.min_size, b.capabilities.max_size)
        );
        assert!(t.capabilities.mac_only);
    }

    #[test]
    fn load_turbo_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load_turbo(&spec)
            .err()
            .expect("expected an error")
            .to_string();
        assert!(e.contains("snapshot directory"), "got: {e}");
    }

    #[test]
    fn load_turbo_rejects_user_adapters() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_adapters(vec![
            AdapterSpec {
                path: "/tmp/user.safetensors".into(),
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            },
        ]);
        let e = load_turbo(&spec)
            .err()
            .expect("expected an error")
            .to_string();
        assert!(e.contains("not supported"), "got: {e}");
    }

    #[test]
    fn load_turbo_errors_when_bundled_lora_missing() {
        // A dir with no turbo_lora.safetensors must fail loudly on the missing bundled LoRA (the
        // model-defining component), not silently fall back to a CFG render.
        let dir = std::env::temp_dir().join("ideogram4_turbo_no_lora_test");
        std::fs::create_dir_all(&dir).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
        let e = load_turbo(&spec)
            .err()
            .expect("expected an error")
            .to_string();
        assert!(
            e.contains("turbo_lora.safetensors") || e.contains("TurboTime LoRA not found"),
            "got: {e}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn turbo_reachable_via_registry_by_id() {
        assert!(gen_core::registry::generators().any(|r| (r.descriptor)().id == MODEL_ID_TURBO));
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-ideogram-turbo".into()));
        let e = gen_core::registry::load(MODEL_ID_TURBO, &spec)
            .err()
            .expect("missing weights → err")
            .to_string();
        assert!(
            !e.contains("no generator registered"),
            "turbo id not resolved: {e}"
        );
    }
}
