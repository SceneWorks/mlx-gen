//! `mlx-gen-wan` model entry: the Wan2.2 **TI2V-5B** descriptor, the config-driven `load`, and
//! registry self-registration.
//!
//! **Scope (S0):** the foundation slice — crate scaffold, registry wiring, the config-driven
//! [`WanModelConfig`], the three flow-match solvers, 3-axis 3-D RoPE, and 3-D patchify/unpatchify.
//! The actual denoise pipeline (UMT5-XXL TE → 30-layer DiT → z48 VAE → video output) lands across
//! S1–S5, so `generate` returns an explicit "not yet wired" error until then. `load` already reads
//! and resolves the model's `config.json` so the config seam is exercised end-to-end now.

use mlx_gen::{
    Capabilities, ConditioningKind, Error, GenerationOutput, GenerationRequest, Generator,
    LoadSpec, Modality, ModelDescriptor, Precision, Progress, Result, WeightsSource,
};

use crate::config::WanModelConfig;

/// Public registry id: `mlx_gen::load("wan2_2_ti2v_5b", spec)`.
pub const MODEL_ID: &str = "wan2_2_ti2v_5b";

/// Stable identity + advertised capabilities for the Wan2.2 TI2V-5B (dense text+image→video).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "wan",
        modality: Modality::Video,
        capabilities: Capabilities {
            // 5B uses real CFG (guide 5.0) with the Chinese anti-artifact negative prompt, and
            // accepts a single image as the TI2V mask-blend conditioning reference.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![ConditioningKind::Reference],
            // LoRA/LoKr (sc-2683 / sc-2393) and Q4/Q8 (sc-2682) are sibling slices.
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["unipc", "euler", "dpmpp2m"],
            schedulers: Vec::new(),
            // H/W align to patch×vae_stride = 32; cap the long edge at 1280 (max_area 704×1280).
            min_size: 32,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            // Cross-attention text K/V is cached across denoise steps.
            supports_kv_cache: true,
            // Wan pins a static `sample_shift` from config (not the empirical per-resolution mu).
            requires_sigma_shift: false,
        },
    }
}

/// The loaded Wan model. S0 holds the resolved config; the network components (UMT5 TE, DiT, z48
/// VAE) attach across S1–S5.
pub struct Wan {
    descriptor: ModelDescriptor,
    #[allow(dead_code)] // consumed by the S1–S5 pipeline.
    config: WanModelConfig,
}

impl Wan {
    /// The resolved model config (exposed for the S1–S5 pipeline slices + tests).
    pub fn config(&self) -> &WanModelConfig {
        &self.config
    }
}

/// Load the model from a snapshot directory. Reads + resolves `config.json` (the config seam). The
/// 5B path runs f32 activations (quality + dodging the pmetal bf16 GEMM bug); quantization and
/// adapters are sibling slices, rejected here for now.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p,
            WeightsSource::File(_) => return Err(Error::Msg(
                "wan2_2_ti2v_5b: expected a model directory (split-weight snapshot), not a single \
                 file"
                    .into(),
            )),
        };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan2_2_ti2v_5b: precision override is not wired (the dense path runs f32 activations)"
                .into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(Error::Msg(
            "wan2_2_ti2v_5b: Q4/Q8 quantization is a sibling slice (sc-2682), not yet wired".into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "wan2_2_ti2v_5b: LoRA/LoKr adapters are sibling slices (sc-2683 / sc-2393), not yet \
             wired"
                .into(),
        ));
    }

    let config = WanModelConfig::from_model_dir(root)?;
    Ok(Box::new(Wan {
        descriptor: descriptor(),
        config,
    }))
}

impl Generator for Wan {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        // H/W align to patch×vae_stride (32 for the 5B); the pipeline rounds down, but reject
        // sub-tile sizes outright.
        let align = (self.config.patch_size.1 * self.config.vae_stride.1) as u32;
        if req.width < align || req.height < align {
            return Err(Error::Msg(format!(
                "wan2_2_ti2v_5b: width/height must be ≥ {align} (got {}x{})",
                req.width, req.height
            )));
        }
        if let Some(frames) = req.frames {
            // num_frames must be 1 + 4·k (one VAE temporal chunk + 4× per chunk).
            if frames % 4 != 1 {
                return Err(Error::Msg(format!(
                    "wan2_2_ti2v_5b: num_frames must be 1 + 4·k (got {frames})"
                )));
            }
        }
        Ok(())
    }

    fn generate(
        &self,
        _req: &GenerationRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        Err(Error::Msg(
            "wan2_2_ti2v_5b: the T2V/TI2V denoise pipeline is not yet wired — S0 ships the \
             scaffold, config, 3 flow-match solvers, 3-axis RoPE, and 3-D patchify; the UMT5 TE / \
             z48 VAE / DiT / pipeline land in S1–S5 (sc-2678 / sc-2680)"
                .into(),
        ))
    }
}

inventory::submit! {
    mlx_gen::ModelRegistration { descriptor, load }
}
