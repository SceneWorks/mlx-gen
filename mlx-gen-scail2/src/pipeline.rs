//! SCAIL-2 provider: capability surface, registration, snapshot/config resolution, and the
//! [`Generator`] entrypoint.
//!
//! WIP (sc-5442): the DiT forward — three patch-embed stems (latent / pose / 28-channel mask) with the
//! mask & pose embeds *added* to the latent embeds, per-source RoPE shifts (the `replace_flag`
//! reference H-shift + pose freq-downsample), Wan-I2V image cross-attention, the open-CLIP XLM-RoBERTa
//! ViT-H/14 image encode, the 28-channel mask preprocessing, and the plain-CFG denoise loop — is being
//! built. This file lands the registration + capability surface + config/snapshot resolution;
//! [`Generator::generate`] returns an explicit not-yet-implemented error until the forward lands.

use std::path::PathBuf;

use mlx_gen::gen_core;
use mlx_gen::{
    Capabilities, ConditioningKind, Error, GenerationOutput, GenerationRequest, Generator,
    LoadSpec, Modality, ModelDescriptor, Progress, Quant, Result, WeightsSource,
};

use crate::config::Scail2Config;

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
            // Reference character (Reference) + extra characters (MultiReference); the driving video +
            // its color-coded mask map to ControlClip; the reference's own mask rides with the
            // Reference in the worker-side mapping (sc-5448).
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
                ConditioningKind::VideoClip,
                ConditioningKind::ControlClip,
            ],
            // LoRA (incl. the Bias-Aware DPO refinement LoRA) is wired in sc-5451.
            supports_lora: false,
            supports_lokr: false,
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
/// components (DiT / VAE / UMT5 / CLIP) are staged inside `generate` once the forward lands.
pub struct Scail2 {
    descriptor: ModelDescriptor,
    #[allow(dead_code)]
    config: Scail2Config,
    #[allow(dead_code)]
    root: PathBuf,
    #[allow(dead_code)]
    quant: Option<Quant>,
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
        _req: &GenerationRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        Err(Error::Msg(
            "scail2: DiT forward not yet implemented (sc-5442 WIP — converter + turnkey snapshot done; \
             3 patch-embeds / per-source RoPE / I2V cross-attn / CLIP encode / denoise in progress)"
                .into(),
        )
        .into())
    }
}
