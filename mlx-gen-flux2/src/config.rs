//! FLUX.2-klein configuration, lifted from the frozen Python mflux fork's
//! `ModelConfig.flux2_klein_*` (`models/common/config/model_config.py`) and the FLUX.2
//! transformer / Qwen3 text-encoder / VAE constructors.
//!
//! The config is **dimension-parametric**: the same Rust code runs the real 9b model, the 4b
//! variant, and tiny parity fixtures. The fork distinguishes the variants only by a handful of
//! `transformer_overrides` / `text_encoder_overrides` (block/head counts, the Qwen3 hidden /
//! intermediate sizes); everything else is shared.

use mlx_gen::{
    curated_sampler_names, Capabilities, ConditioningKind, Modality, ModelDescriptor, Quant,
};

pub const FLUX2_KLEIN_9B_ID: &str = "flux2_klein_9b";
pub const FLUX2_KLEIN_9B_EDIT_ID: &str = "flux2_klein_9b_edit";
/// The KV-cache edit variant (sc-2347). Loads the `-kv`-distilled weights and caches reference
/// K/V across denoise steps for the ~2.4× single-ref edit speedup. Edit-only — the cache is
/// meaningless without reference tokens.
pub const FLUX2_KLEIN_9B_KV_EDIT_ID: &str = "flux2_klein_9b_kv_edit";
/// FLUX.2-dev txt2img (sc-5916). Undistilled 32B flagship: embedded guidance + more steps.
pub const FLUX2_DEV_ID: &str = "flux2_dev";
/// FLUX.2-dev image-conditioned edit (sc-5919): single + multi reference. Same dev snapshot as
/// [`FLUX2_DEV_ID`] — references condition via the **DiT token concat** (the klein edit mechanism,
/// per the diffusers `Flux2Pipeline`), NOT the Pixtral vision tower (that feeds caption upsampling,
/// a separate feature).
pub const FLUX2_DEV_EDIT_ID: &str = "flux2_dev_edit";
/// FLUX.2-dev strict-pose ControlNet (sc-2292): the `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union`
/// VACE-style control branch overlaid on the dev base. Loads the dev snapshot (`weights`) + the
/// control checkpoint (`control`); a required `Control` conditioning (pose/union skeleton) drives it.
pub const FLUX2_DEV_CONTROL_ID: &str = "flux2_dev_control";

pub const DEFAULT_WIDTH: u32 = 1024;
pub const DEFAULT_HEIGHT: u32 = 1024;
/// Distilled klein default; the fork generates in 4 steps at guidance 1.0.
pub const DEFAULT_STEPS: u32 = 4;
/// Distilled klein runs at guidance 1.0 (no CFG). Base (non-distilled) variants allow >1.0.
pub const DEFAULT_GUIDANCE: f32 = 1.0;

/// FLUX.2-dev is guidance-distilled (embedded scalar, the FLUX.1-dev pattern): ~28 steps (24–50)
/// at guidance ~4.0 — NOT true CFG with a negative prompt. (BFL reference call: `guidance_scale=4`,
/// `num_inference_steps=50` with 28 a good trade-off.)
pub const DEFAULT_STEPS_DEV: u32 = 28;
pub const DEFAULT_GUIDANCE_DEV: f32 = 4.0;

/// A pre-quantized-snapshot manifest (sc-5917): the `quantization` block written into a
/// component's `config.json` by [`crate::convert`]. Its presence on disk flips the matching
/// loader from the dense path to building each predicate Linear (and the TE token embedding)
/// directly from packed Q4/Q8 parts — so no dense bf16 weight is ever materialized, which is
/// what keeps the dev load-time memory floor under the 128 GB ceiling (60 GB DiT + 45 GB TE bf16
/// would peak ~105 GB dense *before* any in-place quantization). The consume-side mirror of
/// [`mlx_gen_wan::config::WanQuant`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flux2Quant {
    pub bits: i32,
    pub group_size: i32,
}

/// The FLUX.2-klein variants this crate targets. 9b is the story target; the enum keeps the
/// door open for 4b (a near-free addition — only the dims in [`Flux2Config`] change).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flux2Variant {
    /// FLUX.2-klein-9b, distilled, txt2img.
    Klein9b,
    /// FLUX.2-klein-9b, distilled, image-conditioned edit (single + multi reference).
    Klein9bEdit,
    /// FLUX.2-klein-9b-kv, distilled, edit with the reference-K/V cache (sc-2347).
    Klein9bKvEdit,
    /// FLUX.2-dev, guidance-distilled txt2img (sc-5916). Larger MMDiT (48 single blocks, 48 heads,
    /// joint 15360) + the Mistral text encoder; embedded guidance ~4 over ~28 steps.
    Dev,
    /// FLUX.2-dev, image-conditioned edit (sc-5919): single + multi reference, loading the same dev
    /// snapshot as [`Dev`](Self::Dev). Per the diffusers `Flux2Pipeline`, dev references condition
    /// via the **DiT token concat** (VAE-encode → pack → concat to the image stream at t=10+10·i,
    /// the klein edit mechanism), with a **text-only** Mistral prompt — the Pixtral vision tower is
    /// not on this path (it feeds caption upsampling). Embedded guidance, same as `Dev`.
    DevEdit,
}

impl Flux2Variant {
    pub fn id(self) -> &'static str {
        match self {
            Self::Klein9b => FLUX2_KLEIN_9B_ID,
            Self::Klein9bEdit => FLUX2_KLEIN_9B_EDIT_ID,
            Self::Klein9bKvEdit => FLUX2_KLEIN_9B_KV_EDIT_ID,
            Self::Dev => FLUX2_DEV_ID,
            Self::DevEdit => FLUX2_DEV_EDIT_ID,
        }
    }

    pub fn hf_model(self) -> &'static str {
        match self {
            // Both txt2img and plain edit load the same base 9b snapshot; the edit path differs
            // only in how reference images are tokenized into extra sequence tokens.
            Self::Klein9b | Self::Klein9bEdit => "black-forest-labs/FLUX.2-klein-9B",
            // The KV-cache variant is a separately distilled checkpoint (same architecture).
            Self::Klein9bKvEdit => "black-forest-labs/FLUX.2-klein-9b-kv",
            // dev txt2img and dev edit share the one dev snapshot (no separate -edit checkpoint).
            Self::Dev | Self::DevEdit => "black-forest-labs/FLUX.2-dev",
        }
    }

    pub fn is_edit(self) -> bool {
        matches!(
            self,
            Self::Klein9bEdit | Self::Klein9bKvEdit | Self::DevEdit
        )
    }

    /// dev variants (txt2img + edit) — the ones loading the Mistral3 snapshot through the `*_dev`
    /// loaders and using the embedded-guidance forward.
    pub fn is_dev(self) -> bool {
        matches!(self, Self::Dev | Self::DevEdit)
    }

    /// The 9b-kv variant, which caches reference K/V across denoise steps.
    pub fn is_kv(self) -> bool {
        matches!(self, Self::Klein9bKvEdit)
    }

    /// The dimension-parametric model config for this variant.
    pub fn config(self) -> Flux2Config {
        if self.is_dev() {
            Flux2Config::dev()
        } else {
            Flux2Config::klein_9b()
        }
    }

    /// Default denoise steps. Distilled klein = 4; guidance-distilled dev ≈ 28 (range 24–50).
    pub fn default_steps(self) -> u32 {
        if self.is_dev() {
            DEFAULT_STEPS_DEV
        } else {
            DEFAULT_STEPS
        }
    }

    /// Default guidance. klein runs CFG-free (1.0); dev uses embedded guidance ~4.0.
    pub fn default_guidance(self) -> f32 {
        if self.is_dev() {
            DEFAULT_GUIDANCE_DEV
        } else {
            DEFAULT_GUIDANCE
        }
    }

    /// Whether the guidance scale is consumed as an **embedded scalar** fed into the transformer's
    /// guidance embedder (the guidance-distilled dev, FLUX.1-dev pattern) rather than as a true-CFG
    /// dual-forward over a negative prompt. dev (txt2img + edit) = `true` (single forward, no
    /// negative pass); klein = `false` (distilled, CFG-free at guidance 1.0; a base variant would do
    /// true-CFG when >1).
    pub fn uses_embedded_guidance(self) -> bool {
        self.is_dev()
    }

    pub fn descriptor(self) -> ModelDescriptor {
        // Conditioning surface by variant: the edit variant consumes one `Reference` (single image,
        // token concat, sc-2346) or one `MultiReference` (N images, sc-2645); txt2img consumes a
        // single `Reference` as an **img2img** init image seeding the latents via the noise blend
        // (sc-2644). Advertise what this port delivers, no more, no less.
        let conditioning = if self.is_edit() {
            vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
            ]
        } else {
            vec![ConditioningKind::Reference]
        };
        ModelDescriptor {
            id: self.id(),
            family: "flux2",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities {
                supports_negative_prompt: false,
                // klein is distilled (guidance 1.0); base variants would flip this on. The
                // fork's `supports_guidance` is True, but the distilled klein the story targets
                // runs CFG-free, so we expose guidance but default it to 1.0.
                supports_guidance: true,
                supports_true_cfg: false,
                conditioning,
                // Transformer-only LoRA/LoKr (sc-2646): both variants share the `Flux2Transformer`,
                // which hosts the adapters; the VAE + Qwen3 TE are not adapter targets.
                supports_lora: true,
                supports_lokr: true,
                supported_quants: &[Quant::Q4, Quant::Q8],
                // Curated unified-framework integrator menu (epic 7114 P3). An unset `req.sampler` is
                // the curated Euler over the resolution-shifted flow schedule.
                samplers: curated_sampler_names(),
                schedulers: vec!["flow_match_euler"],
                min_size: 256,
                max_size: 2048,
                max_count: 8,
                mac_only: true,
                // Only the 9b-kv edit variant runs the reference-K/V cache (sc-2347).
                supports_kv_cache: self.is_kv(),
                // FLUX.2 uses the empirical-mu shifted flow-match schedule.
                requires_sigma_shift: true,
            },
        }
    }
}

/// Dimension-parametric FLUX.2 model dimensions. Field values come from the fork's
/// `ModelConfig` + the FLUX.2 module constructors; the 9b values are the story target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flux2Config {
    // --- MMDiT transformer ---
    /// Double (joint img+txt) blocks. 9b: 8, 4b: 5.
    pub num_double_layers: usize,
    /// Single (fused parallel attention+SwiGLU) blocks. 9b: 24, 4b: 20.
    pub num_single_layers: usize,
    /// Attention heads. 9b: 32, 4b: 24.
    pub num_heads: usize,
    /// Per-head dim (constant across variants). `inner_dim = num_heads * head_dim`.
    pub head_dim: usize,
    /// Latent channels entering/leaving the transformer = `num_latent_channels * 4` (2×2 patch).
    pub in_channels: usize,
    /// Reference-config mirror (= `in_channels`); the output projection width is read from the
    /// loaded weights, so the forward never consumes this. Carried for variant identity / parity.
    pub out_channels: usize,
    /// Text-embedding width entering the joint blocks = `3 * te_hidden_size` (the concat of the
    /// three Qwen3 hidden-state layers). 9b: 12288, 4b: 7680.
    pub joint_attention_dim: usize,
    /// Single-block SwiGLU expansion ratio (mlp_hidden = mlp_ratio * inner_dim).
    pub mlp_ratio: f32,
    /// Sinusoidal timestep-embedding width feeding `time_guidance_embed.linear_1` (klein: 256).
    pub timestep_channels: usize,

    // --- 4-axis RoPE over ids (t, h, w, layer) ---
    pub axes_dim: [usize; 4],
    pub rope_theta: f32,

    // --- Qwen3 text encoder (consumed in S1; carried here for variant identity) ---
    pub te_hidden_size: usize,
    /// Qwen3 FFN width; read from the TE weights in S1, so this mirror is variant-identity only.
    pub te_intermediate_size: usize,
    /// Concatenated hidden-state layers forming `prompt_embeds` (`joint_attention_dim` wide).
    pub te_out_layers: [usize; 3],
    /// Reference prompt-length cap; the encoder bounds sequence length from its inputs, so this
    /// mirror is variant-identity only (not read by the forward).
    pub max_sequence_length: usize,

    // --- VAE / latent geometry ---
    pub num_latent_channels: usize,
    /// VAE spatial downsample (= 8); the pipeline applies the 8×/16× latent geometry inline, so
    /// this mirror documents the variant and is not read by the forward.
    pub vae_scale_factor: usize,
}

impl Flux2Config {
    /// FLUX.2-klein-9b (the story target).
    pub fn klein_9b() -> Self {
        Self {
            num_double_layers: 8,
            num_single_layers: 24,
            num_heads: 32,
            head_dim: 128,
            in_channels: 128,
            out_channels: 128,
            joint_attention_dim: 12288,
            mlp_ratio: 3.0,
            timestep_channels: 256,
            axes_dim: [32, 32, 32, 32],
            rope_theta: 2000.0,
            te_hidden_size: 4096,
            te_intermediate_size: 12288,
            te_out_layers: [9, 18, 27],
            max_sequence_length: 512,
            num_latent_channels: 32,
            vae_scale_factor: 8,
        }
    }

    /// FLUX.2-dev — the same MMDiT arch as klein, scaled up: 48 single blocks, 48 heads (inner
    /// 6144), `joint_attention_dim` 15360 (= 3 × the Mistral TE hidden 5120). The VAE + RoPE +
    /// patch geometry are identical to klein; only the block/head counts, the joint width, and the
    /// text-encoder dims change. Values from the dev `transformer/config.json` + `text_encoder/config.json`.
    pub fn dev() -> Self {
        Self {
            num_double_layers: 8,
            num_single_layers: 48,
            num_heads: 48,
            head_dim: 128,
            in_channels: 128,
            out_channels: 128,
            joint_attention_dim: 15360,
            mlp_ratio: 3.0,
            timestep_channels: 256,
            axes_dim: [32, 32, 32, 32],
            rope_theta: 2000.0,
            te_hidden_size: 5120,
            te_intermediate_size: 32768,
            te_out_layers: [10, 20, 30],
            max_sequence_length: 512,
            num_latent_channels: 32,
            vae_scale_factor: 8,
        }
    }

    /// `num_heads * head_dim` — the transformer inner width (9b: 4096).
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn klein_9b_dims_match_fork() {
        let c = Flux2Config::klein_9b();
        assert_eq!(c.num_double_layers, 8);
        assert_eq!(c.num_single_layers, 24);
        assert_eq!(c.num_heads, 32);
        assert_eq!(c.inner_dim(), 4096);
        assert_eq!(c.joint_attention_dim, 3 * c.te_hidden_size);
        assert_eq!(c.in_channels, c.num_latent_channels * 4);
        // RoPE axes sum to the head dim; each axis emits dim/2 freqs → cos/sin width head_dim/2.
        assert_eq!(c.axes_dim.iter().sum::<usize>(), c.head_dim);
    }

    #[test]
    fn descriptors_have_expected_ids() {
        assert_eq!(Flux2Variant::Klein9b.id(), FLUX2_KLEIN_9B_ID);
        assert_eq!(Flux2Variant::Klein9bEdit.id(), FLUX2_KLEIN_9B_EDIT_ID);
        assert_eq!(Flux2Variant::Klein9bKvEdit.id(), FLUX2_KLEIN_9B_KV_EDIT_ID);
        assert!(Flux2Variant::Klein9bEdit.is_edit());
        assert!(!Flux2Variant::Klein9b.is_edit());
    }

    #[test]
    fn dev_dims_match_reference() {
        let c = Flux2Config::dev();
        assert_eq!(c.num_double_layers, 8);
        assert_eq!(c.num_single_layers, 48);
        assert_eq!(c.num_heads, 48);
        assert_eq!(c.inner_dim(), 6144);
        // joint_attention_dim = 3 × the Mistral TE hidden (5120) = 15360.
        assert_eq!(c.joint_attention_dim, 3 * c.te_hidden_size);
        assert_eq!(c.joint_attention_dim, 15360);
        assert_eq!(c.in_channels, c.num_latent_channels * 4);
        assert_eq!(c.axes_dim.iter().sum::<usize>(), c.head_dim);
        assert_eq!(c.te_out_layers, [10, 20, 30]);
    }

    #[test]
    fn dev_variant_is_txt2img_with_embedded_guidance() {
        let v = Flux2Variant::Dev;
        assert_eq!(v.id(), FLUX2_DEV_ID);
        assert_eq!(v.hf_model(), "black-forest-labs/FLUX.2-dev");
        assert!(!v.is_edit());
        assert!(!v.is_kv());
        // Guidance-distilled (embedded scalar): ~28 steps at guidance ~4, not 4-step CFG-free.
        assert_eq!(v.default_steps(), 28);
        assert_eq!(v.default_guidance(), 4.0);
        let caps = v.descriptor().capabilities;
        assert!(caps.supports_guidance);
        assert!(!caps.supports_negative_prompt);
        assert!(!caps.supports_true_cfg);
        assert!(!caps.supports_kv_cache);
        assert!(caps.accepts(ConditioningKind::Reference));
        // config() now returns the dev dims, not klein's (the previous hardcode was a latent bug).
        assert_eq!(v.config().num_single_layers, 48);
    }

    #[test]
    fn dev_edit_variant_is_edit_with_dev_dims_and_embedded_guidance() {
        let e = Flux2Variant::DevEdit;
        assert_eq!(e.id(), FLUX2_DEV_EDIT_ID);
        // Same dev snapshot as txt2img dev (no separate -edit checkpoint).
        assert_eq!(e.hf_model(), "black-forest-labs/FLUX.2-dev");
        assert!(e.is_dev() && Flux2Variant::Dev.is_dev());
        assert!(e.is_edit() && !e.is_kv());
        // Embedded guidance + dev dims, like Dev.
        assert!(e.uses_embedded_guidance());
        assert_eq!(e.default_steps(), 28);
        assert_eq!(e.default_guidance(), 4.0);
        assert_eq!(e.config().num_single_layers, 48);
        // Edit conditioning surface = single + multi reference, like the klein edit variant; no
        // negative/true-CFG (embedded guidance), no KV cache (dev has no -kv checkpoint).
        let caps = e.descriptor().capabilities;
        assert!(caps.accepts(ConditioningKind::Reference));
        assert!(caps.accepts(ConditioningKind::MultiReference));
        assert!(
            caps.supports_guidance && !caps.supports_negative_prompt && !caps.supports_true_cfg
        );
        assert!(!caps.supports_kv_cache);
        assert!(caps.mac_only);
    }

    #[test]
    fn kv_variant_is_edit_only_and_caches() {
        let kv = Flux2Variant::Klein9bKvEdit;
        assert!(kv.is_edit());
        assert!(kv.is_kv());
        assert!(!Flux2Variant::Klein9bEdit.is_kv());
        assert!(!Flux2Variant::Klein9b.is_kv());
        // Only the kv variant advertises the cache; it loads the distinct -kv checkpoint.
        assert!(kv.descriptor().capabilities.supports_kv_cache);
        assert!(
            !Flux2Variant::Klein9bEdit
                .descriptor()
                .capabilities
                .supports_kv_cache
        );
        assert_eq!(kv.hf_model(), "black-forest-labs/FLUX.2-klein-9b-kv");
        // Edit conditioning surface (single + multi reference), same as the plain edit variant.
        let caps = kv.descriptor().capabilities;
        assert!(caps.accepts(ConditioningKind::Reference));
        assert!(caps.accepts(ConditioningKind::MultiReference));
    }
}
