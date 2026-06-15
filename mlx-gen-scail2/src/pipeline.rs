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

use mlx_gen::gen_core;
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
            min_size: 32,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: true,
            requires_sigma_shift: false,
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

/// Registry adapter: bridge the crate's rich [`Result`] into the registry's [`gen_core::Result`].
fn load_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load(spec).map_err(Into::into)
}

inventory::submit! {
    mlx_gen::ModelRegistration { descriptor, load: load_registered }
}

impl Generator for Scail2 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        Ok(self.run(req, on_progress)?)
    }
}

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
            on_progress,
        )
    }
}
