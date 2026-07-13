//! SCAIL-2 provider: capability surface, registration, snapshot/config resolution, and the
//! [`Generator`] entrypoint.
//!
//! [`Generator::generate`] maps the [`GenerationRequest`] conditioning onto the SCAIL-2 inputs and
//! runs the live [`crate::generate`] denoise pipeline: the primary **reference character** is a
//! [`Conditioning::Reference`] image paired with its color-coded [`Conditioning::Mask`]; the
//! **driving video + per-frame color masks** are a [`Conditioning::ControlClip`]; `video_mode ==
//! "replacement"` toggles the cross-identity `replace_flag` (else animation). Inference LoRA(s) from
//! [`LoadSpec::adapters`] (the Bias-Aware DPO refinement LoRA + a lightx2v step-distill lightning
//! LoRA, sc-5451) install onto the DiT as forward-time residuals. Multi-reference (extra characters,
//! each needing its own paired mask) awaits the sc-5583 request contract; the [`crate::generate`]
//! core already supports extra characters via [`crate::CharacterRef`].

use std::path::PathBuf;

use mlx_gen::{
    default_seed, AdapterSpec, Capabilities, Conditioning, ConditioningKind, Error,
    GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor,
    Progress, Quant, Result, WeightsSource,
};
use mlx_gen_wan::SolverKind;

use crate::config::Scail2Config;
use crate::generate::{CharacterRef, Scail2Job};

/// Default driving-segment window + clean-history overlap (upstream `scail.py` defaults).
const SEGMENT_LEN: usize = 81;
const SEGMENT_OVERLAP: usize = 5;
/// Upstream `generate()` sampler defaults: 40 steps, shift 5.0 (3.0 at 480p), guide 5.0, 16 fps.
const DEFAULT_STEPS: u32 = 40;
const DEFAULT_SHIFT: f32 = 5.0;
const DEFAULT_GUIDANCE: f32 = 5.0;
const DEFAULT_FPS: u32 = 16;

/// SceneWorks/engine model id. A still image is `num_frames == 1`.
pub const MODEL_ID: &str = "scail2_14b";

/// Stable identity + advertised capabilities for SCAIL-2 (Wan2.1-14B I2V end-to-end character
/// animation: reference image + driving video + color-coded masks → animated/identity-replaced video;
/// plain single-scale CFG; packed-token conditioning + per-source RoPE + CLIP image cross-attn).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "scail2",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // Reference character image (Reference) + its color-coded segmentation mask (Mask); extra
            // characters (MultiReference, experimental); the driving video + its per-frame color masks
            // map to ControlClip.
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::Mask,
                ConditioningKind::MultiReference,
                ConditioningKind::ControlClip,
            ],
            // Inference LoRA (the Bias-Aware DPO refinement LoRA + a lightx2v step-distill lightning
            // LoRA) installs as a forward-time residual over the (possibly Q4/Q8) base via the
            // family-agnostic loader — SCAIL-2 is Wan2.1-14B I2V, so a Wan-I2V LoRA resolves directly
            // (sc-5451). LoKr/LoHa ride the same residual path.
            supports_lora: true,
            supports_lokr: true,
            samplers: vec!["unipc", "dpm++"],
            schedulers: Vec::new(),
            supported_guidance_methods: vec![],
            min_size: 32,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: true,
            requires_sigma_shift: false,
            // Not wired onto the shared `Residency` seam (F-176); Sequential is a no-op fallback.
            supports_sequential_offload: false,
        },
    }
}

/// The loaded SCAIL-2 model: resolved config + snapshot dir + optional load-time quant. The heavy
/// components (DiT / VAE / UMT5 / CLIP) are staged per-stage inside [`crate::generate`].
pub struct Scail2 {
    descriptor: ModelDescriptor,
    config: Scail2Config,
    root: PathBuf,
    /// Q4/Q8 load-time quant (sc-5445) — applied to the DiT in [`crate::generate::generate`].
    quant: Option<Quant>,
    /// Inference LoRA(s) from [`LoadSpec::adapters`] (the Bias-Aware DPO / lightx2v lightning LoRA,
    /// sc-5451) — installed onto the DiT as forward-time residuals in [`crate::generate::generate`].
    adapters: Vec<AdapterSpec>,
}

/// Load SCAIL-2 from a converted MLX snapshot directory (`dit.safetensors` + `config.json` +
/// `Wan2.1_VAE.pth` + `umt5-xxl/` + the open-CLIP XLM-RoBERTa ViT-H/14 visual encoder), as published
/// to `SceneWorks/scail2-mlx`.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "scail2: expected a model directory (converted MLX snapshot), not a single file"
                    .into(),
            )),
        };
    if !root.exists() {
        return Err(Error::Msg(format!(
            "scail2: snapshot dir does not exist: {}",
            root.display()
        )));
    }
    let config = Scail2Config::from_model_dir(&root)?;
    Ok(Box::new(Scail2 {
        descriptor: descriptor(),
        config,
        root,
        quant: spec.quantize,
        adapters: spec.adapters.clone(),
    }))
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`.
mlx_gen::register_generators! { descriptor => load }

mlx_gen::impl_generator!(Scail2 {
    validate: |s, req| s
        .descriptor
        .capabilities
        .validate_request(s.descriptor.id, req),
    generate: run,
});

/// The first conditioning input matching `f`.
fn find_conditioning<'a, T>(
    req: &'a GenerationRequest,
    f: impl Fn(&'a Conditioning) -> Option<T>,
) -> Option<T> {
    req.conditioning.iter().find_map(f)
}

impl Scail2 {
    /// Map the request conditioning onto a [`Scail2Job`] and run the denoise pipeline.
    fn run(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        // Self-validate the shared floor first (F-158). `impl_generator!`'s `generate` does NOT call
        // `validate`, so a direct `Generator::generate` on scail2 (the only provider that skipped
        // this) otherwise bypassed count, sampler membership, conditioning allowlist, and the
        // F-053 finiteness guard — `guidance: Some(NAN)` would NaN-poison a multi-minute render into
        // garbage-as-success. Every other provider re-validates at the top of its generate impl.
        //
        // scail2 supports a "match the driving-video size" convention (`width`/`height == 0` → resolved
        // from the driving frames below), which the floor's size-range check would wrongly reject. So
        // the full floor always runs; only the size-range check is gated on explicit dims via
        // `validate_request_skip_size` — count/frame caps, sampler membership, conditioning allowlist,
        // support gating, and the finiteness guard all fire even on the auto-size path.
        if req.width > 0 && req.height > 0 {
            self.descriptor
                .capabilities
                .validate_request(self.descriptor.id, req)?;
        } else {
            self.descriptor
                .capabilities
                .validate_request_skip_size(self.descriptor.id, req)?;
        }
        let reference = find_conditioning(req, |c| match c {
            Conditioning::Reference { image, .. } => Some(image),
            _ => None,
        })
        .ok_or_else(|| Error::Msg("scail2: a Reference character image is required".into()))?;
        let ref_mask = find_conditioning(req, |c| match c {
            Conditioning::Mask { image } => Some(image),
            _ => None,
        })
        .ok_or_else(|| {
            Error::Msg(
                "scail2: a Mask (the reference character's color-coded segmentation mask) is required"
                    .into(),
            )
        })?;
        let driving = req.control_clip().ok_or_else(|| {
            Error::Msg(
                "scail2: a ControlClip (driving video frames + per-frame color masks) is required"
                    .into(),
            )
        })?;

        // Target size: the request's (aligned to 32 in the core), else the driving frame's native size.
        let first: &Image = driving
            .frames
            .first()
            .ok_or_else(|| Error::Msg("scail2: the ControlClip has no driving frames".into()))?;
        let width = if req.width > 0 {
            req.width
        } else {
            first.width
        };
        let height = if req.height > 0 {
            req.height
        } else {
            first.height
        };

        let neg = req.negative_prompt.clone().unwrap_or_default();
        let job = Scail2Job {
            prompt: &req.prompt,
            negative_prompt: &neg,
            width,
            height,
            reference: CharacterRef {
                image: reference,
                mask: ref_mask,
            },
            additional: Vec::new(),
            driving_frames: driving.frames,
            driving_masks: driving.mask,
            replace_flag: req.video_mode.as_deref() == Some("replacement"),
            seed: req.seed.unwrap_or_else(default_seed),
            steps: req.steps.unwrap_or(DEFAULT_STEPS) as usize,
            shift: req.scheduler_shift.unwrap_or(DEFAULT_SHIFT),
            guidance: req.guidance.unwrap_or(DEFAULT_GUIDANCE),
            sampler: SolverKind::from_name(req.sampler.as_deref().unwrap_or("unipc")),
            fps: req.fps.unwrap_or(DEFAULT_FPS),
            segment_len: SEGMENT_LEN,
            segment_overlap: SEGMENT_OVERLAP,
        };
        crate::generate::generate(
            &self.root,
            &self.config,
            &job,
            self.quant,
            &self.adapters,
            &req.cancel,
            on_progress,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A weights-free [`Scail2`] whose [`Scail2::run`] hits the shared floor before any load. Sound only
    /// because validation runs first: a *passing* request would then try to load `root` and fail. Used
    /// to prove the floor rejects an out-of-surface request on the auto-size path without weights.
    fn unloaded() -> Scail2 {
        Scail2 {
            descriptor: descriptor(),
            config: Scail2Config::default(),
            root: PathBuf::from("/nonexistent-scail2-snapshot"),
            quant: None,
            adapters: Vec::new(),
        }
    }

    /// F-158: with `width == height == 0` (scail2's "match the driving-video size" sentinel), `run`
    /// routes through `validate_request_skip_size` — so the whole shared floor except the size-range
    /// check still fires. An oversized count and an unadvertised sampler are both rejected on the
    /// auto-size path, before any weight load.
    #[test]
    fn floor_fires_on_auto_size_path() {
        let m = unloaded();
        let mut noop = |_: Progress| {};

        // Oversized count (max_count == 1) — rejected even though dims are the 0x0 auto sentinel.
        let bad_count = GenerationRequest {
            width: 0,
            height: 0,
            count: 4,
            ..Default::default()
        };
        let err = m
            .run(&bad_count, &mut noop)
            .expect_err("oversized count must be rejected on the auto-size path");
        assert!(
            err.to_string().contains("count"),
            "expected a count-range rejection, got: {err}"
        );

        // Unadvertised sampler (scail2 advertises only `unipc` / `dpm++`); count == 1 so the sampler is
        // the failing check.
        let bad_sampler = GenerationRequest {
            width: 0,
            height: 0,
            count: 1,
            sampler: Some("euler".into()),
            ..Default::default()
        };
        let err = m
            .run(&bad_sampler, &mut noop)
            .expect_err("unadvertised sampler must be rejected on the auto-size path");
        assert!(
            err.to_string().contains("sampler"),
            "expected an unsupported-sampler rejection, got: {err}"
        );
    }
}
