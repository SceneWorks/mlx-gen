//! `mlx-gen-wan` model entries: the Wan2.2 **TI2V-5B** (`wan2_2_ti2v_5b`, dense, z48 VAE — S0
//! scaffold, denoise pending in sc-2680), the Wan2.2 **T2V-A14B** (`wan2_2_t2v_14b`, dual-expert MoE,
//! z16 VAE — fully wired here on the S1–S5 core), and the Wan2.2 **I2V-A14B** (`wan2_2_i2v_14b`,
//! dual-expert MoE, channel-concat image conditioning, in_dim 36 — sc-2681), plus their registry
//! self-registration.
//!
//! The 5B `load` resolves `config.json` and stubs `generate` (its z48 VAE + dense denoise are
//! sc-2680). The shared [`Wan14b`] struct serves both A14B variants — [`Wan14b::generate`] runs the
//! complete pipeline: UMT5-XXL encode → (I2V only) build the channel-concat conditioning `y` →
//! per-step dual-expert MoE denoise (boundary-switched high/low experts, [`denoise_moe`]) → z16 VAE
//! decode → RGB8 frames, **staging** each heavy component (T5, the two 27 GB experts, the VAE) in and
//! out to bound peak memory (mirrors `generate_wan.py`). The I2V variant differs only by the `y`
//! conditioning (the image's first-frame VAE latent + temporal mask, channel-concatenated to in_dim
//! 36) and the max-area resolution cap.

use std::path::PathBuf;

use mlx_gen::tiling::TilingConfig;
use mlx_gen::weights::Weights;
use mlx_gen::{
    default_seed, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, Precision, Progress,
    Result, WeightsSource,
};
use mlx_rs::random;

use crate::config::{GuideScale, WanModelConfig};
use crate::pipeline::{
    align_dim, best_output_size, build_i2v_y, decode_to_frames, denoise_moe, frames_to_images,
    latent_shape, Expert,
};
use crate::scheduler::SolverKind;
use crate::text_encoder::{load_tokenizer, Umt5Encoder};
use crate::transformer::WanTransformer;
use crate::vae::WanVae;

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

// ===========================================================================================
// Wan2.2 T2V-A14B — dual-expert MoE text→video (the S1–S5 core, fully wired)
// ===========================================================================================

/// Public registry id for the dual-expert MoE T2V model: `mlx_gen::load("wan2_2_t2v_14b", spec)`.
pub const MODEL_ID_T2V_14B: &str = "wan2_2_t2v_14b";

/// Stable identity + advertised capabilities for the Wan2.2 T2V-A14B (dual-expert MoE text→video).
pub fn descriptor_t2v_14b() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID_T2V_14B,
        family: "wan",
        modality: Modality::Video,
        capabilities: Capabilities {
            // CFG with the per-expert (low, high) guidance pair + the Chinese anti-artifact negative
            // prompt. Pure text→video: no image conditioning.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: Vec::new(),
            // LoRA/LoKr (sc-2683 / sc-2393) and Q4/Q8 (sc-2682) are sibling slices.
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["unipc", "euler", "dpmpp2m"],
            schedulers: Vec::new(),
            // H/W align to patch×vae_stride = 16 (z16 VAE, spatial stride 8); long edge cap 1280.
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            // Cross-attention text K/V is cached across denoise steps (per expert).
            supports_kv_cache: true,
            requires_sigma_shift: false,
        },
    }
}

/// The loaded Wan2.2 T2V-A14B. Holds the resolved config + the snapshot directory; the heavy
/// components (UMT5 TE, the two 14B experts, the z16 VAE) are **staged** inside
/// [`Wan14b::generate`] — loaded, used, then dropped in turn — to bound peak memory (mirrors
/// `generate_wan.py`, which never holds the T5 encoder and both 27 GB experts resident at once).
pub struct Wan14b {
    descriptor: ModelDescriptor,
    config: WanModelConfig,
    root: PathBuf,
}

impl Wan14b {
    /// The resolved model config.
    pub fn config(&self) -> &WanModelConfig {
        &self.config
    }
}

/// Map a request `sampler` string to a [`SolverKind`] (default UniPC, the reference's default).
fn solver_kind(sampler: Option<&str>) -> SolverKind {
    match sampler {
        Some("euler") => SolverKind::Euler,
        Some("dpmpp2m") | Some("dpm++") => SolverKind::Dpmpp2m,
        _ => SolverKind::UniPC,
    }
}

/// Load the Wan2.2 T2V-A14B from a converted MLX snapshot directory (`convert_wan.py` output:
/// `low_noise_model.safetensors` + `high_noise_model.safetensors` + `t5_encoder.safetensors` +
/// `vae.safetensors` + `tokenizer.json` + `config.json`). Quantization + adapters are sibling slices.
pub fn load_t2v_14b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(Error::Msg(
            "wan2_2_t2v_14b: expected a model directory (converted MLX snapshot), not a single \
                 file"
                .into(),
        )),
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan2_2_t2v_14b: precision override is not wired (the experts run bf16 GEMMs over an \
             f32 residual stream — the parity regime)"
                .into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(Error::Msg(
            "wan2_2_t2v_14b: Q4/Q8 quantization is a sibling slice (sc-2682), not yet wired".into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "wan2_2_t2v_14b: LoRA/LoKr adapters are sibling slices (sc-2683 / sc-2393), not yet \
             wired"
                .into(),
        ));
    }

    let config = WanModelConfig::from_model_dir(&root)?;
    if !config.dual_model {
        return Err(Error::Msg(format!(
            "wan2_2_t2v_14b: config.json is not a dual-expert model (dual_model=false, \
             model_type={}); expected the converted Wan2.2 A14B MoE checkpoint",
            config.model_type
        )));
    }
    Ok(Box::new(Wan14b {
        descriptor: descriptor_t2v_14b(),
        config,
        root,
    }))
}

impl Generator for Wan14b {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        let id = self.descriptor.id;
        // H/W align to patch×vae_stride (16 for the z16 VAE); the pipeline rounds down, but reject
        // sub-tile sizes outright.
        let align = (self.config.patch_size.1 * self.config.vae_stride.1) as u32;
        if req.width < align || req.height < align {
            return Err(Error::Msg(format!(
                "{id}: width/height must be ≥ {align} (got {}x{})",
                req.width, req.height
            )));
        }
        if let Some(frames) = req.frames {
            // num_frames must be 1 + 4·k (one VAE temporal chunk + 4× per chunk).
            if frames % 4 != 1 {
                return Err(Error::Msg(format!(
                    "{id}: num_frames must be 1 + 4·k (got {frames})"
                )));
            }
        }
        // I2V channel-concat requires a single reference image (the first conditioning frame), and
        // does not support `trim_first_frames` (the reference builds `y` from `num_frames`, so an
        // extended noise length would mismatch the conditioning's temporal dim).
        if self.config.is_i2v_concat() {
            if i2v_reference(req).is_none() {
                return Err(Error::Msg(format!(
                    "{id}: image-to-video requires a Reference conditioning image"
                )));
            }
            if req.trim_first_frames.unwrap_or(0) > 0 {
                return Err(Error::Msg(format!(
                    "{id}: trim_first_frames is not supported for I2V (the conditioning `y` is built \
                     from num_frames)"
                )));
            }
        }
        Ok(())
    }

    /// The full dual-expert MoE pipeline (port of `generate_wan.py`'s dual-model path) — serves both
    /// **T2V-A14B** and **I2V-A14B** (the struct's config selects). Resolves request knobs against the
    /// config defaults, then **stages** the phases to bound memory: (1) load UMT5, encode the prompt +
    /// negative prompt, drop the encoder; (1b, I2V only) load the z16 VAE encoder, build the
    /// channel-concat conditioning `y` from the reference image, drop it; (2) load both 14B experts,
    /// embed the contexts per expert, run the boundary-switched [`denoise_moe`] loop (with `y` for
    /// I2V), drop the experts; (3) load the z16 VAE, decode to RGB8 frames. CFG runs with the
    /// per-expert (low, high) guidance.
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let cfg = &self.config;

        // --- Resolve request knobs against config defaults ---
        let frames = req.frames.map(|f| f as usize).unwrap_or(cfg.frame_num);
        // trim_first_frames: generate `trim` extra leading temporal chunks (each = vae_stride_t = 4
        // latent frames → 4 output frames after the non-causal T→4T decode) and discard them after
        // decode, so the first kept frame sees a full temporal receptive field (port of
        // generate_wan.py). gen_frames stays 1+4k since frames is and we add a multiple of 4.
        let trim = req.trim_first_frames.unwrap_or(0) as usize;
        let trim_out = trim * cfg.vae_stride.0; // discarded output frames = trim · 4
        let gen_frames = frames + trim * cfg.vae_stride.0;
        // validate() already rejected sub-tile + bad frame counts; round H/W down to the grid.
        let mut width = align_dim(req.width, cfg.patch_size.2, cfg.vae_stride.2);
        let mut height = align_dim(req.height, cfg.patch_size.1, cfg.vae_stride.1);
        // Enforce the model's max-area cap (I2V-14B / TI2V-5B: 704×1280) with an aspect-preserving,
        // grid-aligned fit (no-op for T2V, whose `max_area` is 0). Mirrors `generate_wan.py`.
        if cfg.max_area > 0 && (width as usize) * (height as usize) > cfg.max_area {
            let dw = (cfg.patch_size.2 * cfg.vae_stride.2) as u32;
            let dh = (cfg.patch_size.1 * cfg.vae_stride.1) as u32;
            (width, height) = best_output_size(width, height, dw, dh, cfg.max_area);
        }
        let steps = req.steps.map(|s| s as usize).unwrap_or(cfg.sample_steps);
        let shift = req.scheduler_shift.unwrap_or(cfg.sample_shift);
        let kind = solver_kind(req.sampler.as_deref());
        let seed = req.seed.unwrap_or_else(default_seed);
        // A scalar request `guidance` overrides both experts; otherwise use the config (low, high).
        let (low_gs, high_gs) = match (cfg.sample_guide_scale, req.guidance) {
            (_, Some(g)) => (g, g),
            (GuideScale::Dual { low, high }, None) => (low, high),
            (GuideScale::Single(s), None) => (s, s),
        };
        let neg_prompt = req
            .negative_prompt
            .clone()
            .unwrap_or_else(|| cfg.sample_neg_prompt.clone());

        // Init-noise latent geometry: [z_dim, t_lat, h_lat, w_lat] for the (possibly trim-extended)
        // generation length.
        let lat = latent_shape(gen_frames, height, width, cfg.vae_z_dim, cfg.vae_stride);

        // --- Stage 1: UMT5 text encode (loaded → used → freed) ---
        let tokenizer = load_tokenizer(self.root.join("tokenizer.json"), cfg.text_len)?;
        let (context, context_null) = {
            let w = Weights::from_file(self.root.join("t5_encoder.safetensors"))?;
            let enc = Umt5Encoder::from_weights(&w, cfg)?;
            let context = enc.encode(&tokenizer, &req.prompt)?;
            let context_null = enc.encode(&tokenizer, &neg_prompt)?;
            mlx_rs::transforms::eval([&context, &context_null])?;
            (context, context_null)
        };

        // Seeded init noise (f32, no batch dim) — matches the reference's `mx.random.normal(shape)`
        // shape; exact seeded-RNG values differ across the mlx-python/mlx-rs split (expected). I2V
        // (like the reference) starts from pure noise — the image enters via the `y` channel-concat.
        let key = random::key(seed)?;
        let init_noise = random::normal::<f32>(&lat[..], None, None, Some(&key))?;

        // --- Stage 1b (I2V only): build the channel-concat conditioning `y` (→ VAE encoder freed) ---
        // First frame = the reference image, the rest zero, VAE-encoded under a temporal mask →
        // `[20, T_lat, h_lat, w_lat]` (f32), concatenated onto each forward's noise latent in
        // `denoise_moe`. `frames` (not `gen_frames`) — validate() rejected `trim` for I2V.
        let y = if cfg.is_i2v_concat() {
            let image = i2v_reference(req).ok_or_else(|| {
                Error::Msg(format!(
                    "{}: image-to-video requires a Reference conditioning image",
                    self.descriptor.id
                ))
            })?;
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = WanVae::from_weights(&w)?;
            let y = build_i2v_y(&vae, image, frames, height, width, cfg.vae_stride)?;
            mlx_rs::transforms::eval([&y])?;
            Some(y)
        } else {
            None
        };

        // --- Stage 2: load both experts, embed per-expert, dual-expert MoE denoise (→ freed) ---
        let latents = {
            let low_w = Weights::from_file(self.root.join("low_noise_model.safetensors"))?;
            let high_w = Weights::from_file(self.root.join("high_noise_model.safetensors"))?;
            let low_dit = WanTransformer::from_weights(&low_w, cfg)?;
            let high_dit = WanTransformer::from_weights(&high_w, cfg)?;

            // Each expert has its own text_embedding weights, so contexts are embedded per expert.
            let low = Expert {
                transformer: &low_dit,
                ctx_cond: low_dit.embed_text(&context)?,
                ctx_uncond: Some(low_dit.embed_text(&context_null)?),
                guidance: low_gs,
            };
            let high = Expert {
                transformer: &high_dit,
                ctx_cond: high_dit.embed_text(&context)?,
                ctx_uncond: Some(high_dit.embed_text(&context_null)?),
                guidance: high_gs,
            };
            let boundary_timestep = cfg.boundary * cfg.num_train_timesteps as f32;
            let total = steps as u32;
            let mut on_step = |i: usize| {
                on_progress(Progress::Step {
                    current: i as u32,
                    total,
                })
            };
            denoise_moe(
                &low,
                &high,
                boundary_timestep,
                kind,
                cfg.num_train_timesteps,
                steps,
                shift,
                &init_noise,
                y.as_ref(),
                &mut on_step,
            )?
        };

        // --- Stage 3: z16 VAE decode → RGB8 frames ---
        on_progress(Progress::Decoding);
        // Auto-select VAE decode tiling from the actual decoded output dims (t_lat·4 frames after the
        // non-causal decode); `None` for small outputs → single-pass. decode_to_frames re-checks
        // `needs_tiling`.
        let out_frames = lat[1] * cfg.vae_stride.0 as i32;
        let tiling = TilingConfig::auto(height as i32, width as i32, out_frames);
        let frames_u8 = {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = WanVae::from_weights(&w)?;
            decode_to_frames(&vae, &latents, tiling.as_ref())?
        };
        let mut images = frames_to_images(&frames_u8)?;
        // Discard the extra leading frames generated for `trim_first_frames`.
        if trim_out > 0 {
            images.drain(0..trim_out.min(images.len()));
        }

        let fps = req.fps.unwrap_or(cfg.sample_fps);
        Ok(GenerationOutput::Video {
            frames: images,
            fps,
            audio: None,
        })
    }
}

inventory::submit! {
    mlx_gen::ModelRegistration { descriptor: descriptor_t2v_14b, load: load_t2v_14b }
}

// ===========================================================================================
// Wan2.2 I2V-A14B — dual-expert MoE image→video (channel-concat conditioning, in_dim 36)
// ===========================================================================================

/// Public registry id for the channel-concat I2V model: `mlx_gen::load("wan2_2_i2v_14b", spec)`.
pub const MODEL_ID_I2V_14B: &str = "wan2_2_i2v_14b";

/// The single conditioning reference image for I2V (the first video frame), if present.
fn i2v_reference(req: &GenerationRequest) -> Option<&Image> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, .. } => Some(image),
        _ => None,
    })
}

/// Stable identity + advertised capabilities for the Wan2.2 I2V-A14B (dual-expert MoE image→video).
/// Identical to the T2V-A14B but advertises a single `Reference` conditioning image (the channel-
/// concat first frame) and the (3.5, 3.5) per-expert guidance.
pub fn descriptor_i2v_14b() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID_I2V_14B,
        family: "wan",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // A single image is channel-concatenated as the first-frame conditioning (in_dim 36).
            conditioning: vec![ConditioningKind::Reference],
            // LoRA/LoKr (sc-2683 / sc-2393) and Q4/Q8 (sc-2682) are sibling slices.
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["unipc", "euler", "dpmpp2m"],
            schedulers: Vec::new(),
            // H/W align to patch×vae_stride = 16 (z16 VAE, spatial stride 8); long edge cap 1280.
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supports_kv_cache: true,
            requires_sigma_shift: false,
        },
    }
}

/// Load the Wan2.2 I2V-A14B from a converted MLX snapshot directory (same layout as the T2V-A14B:
/// `low_noise_model` + `high_noise_model` + `t5_encoder` + `vae` (with encoder) + `tokenizer.json` +
/// `config.json`). Requires `model_type == "i2v"` (in_dim 36) and a dual-expert checkpoint.
/// Quantization + adapters are sibling slices.
pub fn load_i2v_14b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(Error::Msg(
            "wan2_2_i2v_14b: expected a model directory (converted MLX snapshot), not a single \
                 file"
                .into(),
        )),
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan2_2_i2v_14b: precision override is not wired (the experts run bf16 GEMMs over an \
             f32 residual stream — the parity regime)"
                .into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(Error::Msg(
            "wan2_2_i2v_14b: Q4/Q8 quantization is a sibling slice (sc-2682), not yet wired".into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "wan2_2_i2v_14b: LoRA/LoKr adapters are sibling slices (sc-2683 / sc-2393), not yet \
             wired"
                .into(),
        ));
    }

    let config = WanModelConfig::from_model_dir(&root)?;
    if !config.is_i2v_concat() {
        return Err(Error::Msg(format!(
            "wan2_2_i2v_14b: config.json is not a channel-concat I2V model (model_type={}, \
             in_dim={}); expected the converted Wan2.2 I2V-A14B checkpoint (model_type=i2v, \
             in_dim=36)",
            config.model_type, config.in_dim
        )));
    }
    if !config.dual_model {
        return Err(Error::Msg(
            "wan2_2_i2v_14b: config.json is not a dual-expert model (dual_model=false); expected \
             the converted Wan2.2 I2V-A14B MoE checkpoint"
                .into(),
        ));
    }
    Ok(Box::new(Wan14b {
        descriptor: descriptor_i2v_14b(),
        config,
        root,
    }))
}

inventory::submit! {
    mlx_gen::ModelRegistration { descriptor: descriptor_i2v_14b, load: load_i2v_14b }
}
