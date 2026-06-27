//! FLUX.1 family configuration, lifted from the frozen Python mflux fork's
//! `ModelConfig.{schnell,dev}` and `FluxWeightDefinition.get_tokenizers`.

use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, Capabilities, ConditioningKind, Modality,
    ModelDescriptor, Quant,
};

pub const FLUX1_SCHNELL_ID: &str = "flux1_schnell";
pub const FLUX1_DEV_ID: &str = "flux1_dev";

pub const DEFAULT_WIDTH: u32 = 1024;
pub const DEFAULT_HEIGHT: u32 = 1024;
pub const DEFAULT_GUIDANCE: f32 = 3.5;

/// The base flow-match sampler name in the capability surface (sc-2908). An unset `req.sampler`
/// resolves to this — the standard FLUX flow-match Euler denoise over `build_linear_sigmas`.
pub const DEFAULT_SAMPLER: &str = "flow_match";
/// The Hyper-FLUX few-step acceleration profile (sc-2908): the SAME flow-match schedule at a reduced
/// step count (8) and guidance 3.5, paired with the ByteDance Hyper-FLUX 8-step LoRA loaded at
/// `scale≈0.125` (`spec.adapters`). FLUX.1-dev-only — it is a dev LoRA, so schnell never advertises
/// it. Selecting it without the LoRA loaded just runs 8 base steps (undertrained noise).
pub const HYPER_SAMPLER: &str = "hyper";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FluxVariant {
    Schnell,
    Dev,
}

impl FluxVariant {
    pub fn id(self) -> &'static str {
        match self {
            Self::Schnell => FLUX1_SCHNELL_ID,
            Self::Dev => FLUX1_DEV_ID,
        }
    }

    pub fn hf_model(self) -> &'static str {
        match self {
            Self::Schnell => "black-forest-labs/FLUX.1-schnell",
            Self::Dev => "black-forest-labs/FLUX.1-dev",
        }
    }

    pub fn default_steps(self) -> u32 {
        match self {
            Self::Schnell => 4,
            Self::Dev => 25,
        }
    }

    pub fn max_sequence_length(self) -> usize {
        match self {
            Self::Schnell => 256,
            Self::Dev => 512,
        }
    }

    pub fn supports_guidance(self) -> bool {
        matches!(self, Self::Dev)
    }

    pub fn requires_sigma_shift(self) -> bool {
        matches!(self, Self::Dev)
    }

    pub fn descriptor(self) -> ModelDescriptor {
        ModelDescriptor {
            id: self.id(),
            family: "flux",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities {
                supports_negative_prompt: false,
                supports_guidance: self.supports_guidance(),
                supports_true_cfg: false,
                // FLUX.1 reference-image conditioning is the XLabs IP-Adapter (epic 3621): a single
                // `Reference` rides `Conditioning::Reference { image, strength=ipAdapterScale }`,
                // exactly as SDXL exposes its IP-Adapter. Only wired when a `LoadSpec::ip_adapter`
                // is supplied at load time; a `Reference` request without it errors loudly (no
                // false-capability trap — `validate` rejects MultiReference / multiple references).
                // The Redux/Depth/Fill/Control variants remain later ports.
                conditioning: vec![ConditioningKind::Reference],
                supported_quants: &[Quant::Q4, Quant::Q8],
                supports_lora: true,
                supports_lokr: true,
                // The curated unified-framework integrator menu (epic 7114 P3) + the legacy
                // `flow_match` alias (== Euler) and, for dev, the Hyper-FLUX few-step profile
                // (sc-2908). schnell is already a distilled 4-step checkpoint, so it omits the
                // dev-only Hyper-FLUX LoRA profile. The acceleration profiles (`flow_match`/`hyper`)
                // route to Euler in `run_flow_sampler`; they change the schedule/steps, not the
                // integrator.
                samplers: {
                    let mut s = curated_sampler_names();
                    s.push(DEFAULT_SAMPLER);
                    if matches!(self, Self::Dev) {
                        s.push(HYPER_SAMPLER);
                    }
                    s
                },
                // Scheduler axis (epic 7114): the native `linear` schedule (resolution-shifted flow) is
                // the byte-exact default; a curated name re-shapes σ over FLUX.1's own mu.
                schedulers: {
                    let mut s = curated_scheduler_names();
                    s.push("linear");
                    s
                },
                supported_guidance_methods: vec![],
                min_size: 256,
                max_size: 2048,
                max_count: 8,
                mac_only: true,
                supports_kv_cache: false,
                requires_sigma_shift: self.requires_sigma_shift(),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FluxTokenizerKind {
    Clip,
    T5,
}

impl FluxTokenizerKind {
    pub fn subdir(self) -> &'static str {
        match self {
            Self::Clip => "tokenizer",
            Self::T5 => "tokenizer_2",
        }
    }

    pub fn max_length(self, variant: FluxVariant) -> usize {
        match self {
            Self::Clip => 77,
            Self::T5 => variant.max_sequence_length(),
        }
    }

    pub fn pad_token_id(self) -> i32 {
        match self {
            // CLIP's `<|endoftext|>` id in the FLUX.1 tokenizer.
            Self::Clip => 49407,
            // T5's `<pad>` id.
            Self::T5 => 0,
        }
    }
}
