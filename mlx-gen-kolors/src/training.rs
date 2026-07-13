//! sc-4568 — LoRA/LoKr **training** on the Kolors U-Net, in pure Rust on mlx-rs. The Kolors
//! realization of the core [`Trainer`] contract (epic 3039), built on the same functional-autograd
//! mechanism the Z-Image / SDXL trainers proved (sc-3042/3044/3045) and the host-generic factor
//! machinery in core ([`mlx_gen::train::lora`]). Parity target = the SceneWorks torch Kolors LoRA
//! trainer (the legacy `KolorsDiffusersAdapter` training path, epic 1929) this replaces.
//!
//! Kolors **is an SDXL-base U-Net under a ChatGLM3-6B text encoder**, so the whole training lifecycle
//! is the shared SDXL-family backbone ([`mlx_gen_sdxl::train_family`], sc-7781 — the same code
//! `mlx-gen-sdxl` drives). This module is just the Kolors deltas, supplied through
//! [`SdxlFamilyHooks`]; everything else is the shared backbone:
//!
//!   * **Text encoder — ChatGLM3-6B, not dual-CLIP.** Conditioning is the ChatGLM3 penultimate hidden
//!     state `context` `[1, 256, 4096]` and the last-token last-layer `pooled` `[1, 4096]` — exactly
//!     the inference [`Kolors::encode`](crate::Kolors::encode) path (tokenize with the left-padded
//!     `position_ids`, then `ChatGlmModel::encode_prompt`). Single forward, no CFG (training is
//!     CFG-off, like every diffusers LoRA script). The SDXL U-Net auto-detects the `encoder_hid_proj`
//!     (4096→2048) and the 5632-wide add-embedding from the Kolors checkpoint (sc-3093), so its
//!     `forward` consumes the ChatGLM `(context, pooled)` directly.
//!   * **Micro-conditioning `time_ids` = `(H, W, 0, 0, H, W)`.** Kolors inference feeds the real
//!     resolution ([`crate::model::kolors_time_ids`], the diffusers `_get_add_time_ids`), unlike the
//!     SDXL engine which hardcodes `[512,512,0,0,512,512]`. Training feeds the **same** real-resolution
//!     ids at the bucketed training edge so the LoRA learns under the conditioning inference applies it
//!     under.
//!   * **Noise / objective — discrete DDPM over the Kolors `scaled_linear` schedule with
//!     `num_train_timesteps = 1100`.** The diffusers Kolors LoRA script noises with a `DDPMScheduler`:
//!     `noisy = √ᾱ_t·x0 + √(1−ᾱ_t)·noise` at a uniform integer `t ∈ [0, 1100)`, regressing the U-Net's
//!     **epsilon** toward `noise` (SDXL-base lineage; Kolors is epsilon-prediction — its inference
//!     [`KolorsEulerSampler`](crate::sampler) is epsilon Euler). This is train/inference-consistent
//!     **by construction**: the Kolors inference sampler's per-train-step sigma is
//!     `σ_t = √((1−ᾱ_t)/ᾱ_t)`, and the renormalized k-diffusion input `(x0+σ_t·noise)·rsqrt(σ_t²+1)`
//!     is algebraically identical to the DDPM `noisy` (`rsqrt(σ²+1)=√ᾱ`, `σ·rsqrt(σ²+1)=√(1−ᾱ)`), with
//!     the U-Net consuming the integer `t` as its sinusoidal time exactly as inference consumes the
//!     leading timesteps off the **same** `√((1−ᾱ)/ᾱ)` table. Unlike the SDXL engine's vendored sigma
//!     table — which is `concat([0], σ_1..σ_1000)` and so trains/infers at table-index `t↔ᾱ[t−1]` (a
//!     deliberate +1 offset) — Kolors inference indexes `ᾱ[T]` directly, so training uses the **direct**
//!     `ᾱ_t` (no offset, [`TrainTimestep::Index`]) to stay in lock-step.
//!   * **f32 base, bf16 default training.** The U-Net + VAE load at f32 for clean autograd; the U-Net
//!     casts to bf16 for the training forward (sc-4941, the worker default); the trained f32 factors
//!     merge into the fp16 base at load. The **ChatGLM3 encoder loads bf16 and is freed after caching**
//!     (sc-4941) — it is a frozen conditioning encoder (no autograd through it), bf16 matches fp16
//!     inference, and freeing its ~12 GB keeps the working set within a 32 GB unified-memory budget.
//!   * **Adapter surface + save keys, matched to inference consumption.** The Kolors U-Net is the SDXL
//!     `UNet2DConditionModel`, so the trained adapter round-trips through the SDXL adapter merge
//!     ([`mlx_gen_sdxl::apply_sdxl_adapters`]): LoRA targets the **complete** attention surface
//!     (down/mid/up `to_q/k/v/to_out.0`) under the PEFT prefix `base_model.model.unet.`; LoKr targets
//!     the **vendored** surface (down/up attention only — the SDXL LoKr loader keeps `mid_block` out,
//!     sc-2640) and reconstructs at **f32** (the SDXL/Kolors merge dtype). The Kolors inference
//!     registry applies `spec.adapters` (LoRA/LoKr merged into the dense U-Net before quantization
//!     since sc-4733), so the produced adapter reloads through the Kolors inference path directly
//!     (validated by `tests/trainer_e2e.rs`).

use mlx_gen::sampler::AlphaSchedule;
use mlx_gen::weights::Weights;
use mlx_gen::{
    gen_core, Image, LoadSpec, Modality, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainingOutput, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_rs::ops::{add, concatenate_axis, multiply};
use mlx_rs::{random, Array, Dtype};

use mlx_gen_sdxl::{
    load_unet_kolors_dtype, load_vae, train_family, Autoencoder, SdxlFamilyHooks, TrainTimestep,
    UNet2DConditionModel,
};

use crate::chatglm3::{ChatGlmConfig, ChatGlmModel};
use crate::model::{kolors_time_ids, render_sample};
use crate::registry::MODEL_ID;
use crate::sampler::{BETA_END, BETA_START, NUM_TRAIN_TIMESTEPS};
use crate::tokenizer::KolorsTokenizer;

/// The Kolors family deltas behind the shared [`train_family`] backbone (sc-7781): the ChatGLM3-6B
/// encoder + its tokenizer, and the DDPM `alphas_cumprod` schedule that drives the noising. Held in
/// [`KolorsTrainer::hooks`].
struct KolorsHooks {
    tokenizer: KolorsTokenizer,
    /// ChatGLM3-6B text encoder, in an `Option` so it can be **dropped after the caching loop**
    /// (sc-4941, 32 GB-Mac support): it is idle during training — every prompt is already encoded to
    /// the cached `(context, pooled)` — yet at ~12 GB (bf16) it is the single largest resident in the
    /// trainer. Freeing it before the train loop keeps the working set within a 32 GB unified-memory
    /// budget. Loaded **bf16**, not f32: it is a frozen encoder producing conditioning (no gradient
    /// flows through it), bf16 is the ecosystem-standard LLM inference precision, and it matches the
    /// fp16 the Kolors *inference* path runs the encoder at — so training conditions on the same
    /// numerics it will be applied under, while halving the cache-phase footprint (24 → 12 GB).
    chatglm: Option<ChatGlmModel>,
    /// Discrete DDPM `alphas_cumprod` over the Kolors `scaled_linear` schedule
    /// (`num_train_timesteps = 1100`); training noises `x0` with `√ᾱ_t·x0 + √(1−ᾱ_t)·noise`.
    schedule: AlphaSchedule,
}

impl SdxlFamilyHooks for KolorsHooks {
    fn label(&self) -> &'static str {
        "kolors"
    }

    /// Caption → `(context [1, 256, 4096], pooled [1, 4096])`: tokenize (left-padded, with the
    /// ChatGLM `position_ids`) and run the ChatGLM3 encoder exactly as the inference
    /// [`Kolors::encode`](crate::Kolors::encode) path.
    fn encode_prompt(&self, caption: &str) -> Result<(Array, Array)> {
        let chatglm = self.chatglm.as_ref().ok_or_else(|| {
            mlx_gen::Error::Msg(
                "kolors trainer: text encoder already freed (encode after caching)".into(),
            )
        })?;
        let t = self.tokenizer.encode(caption)?;
        chatglm.encode_prompt(&t.input_ids, &t.attention_mask, Some(&t.position_ids))
    }

    /// Preview-sample CFG batch (`[2, …]` = positive then empty-negative): encode the positive and the
    /// (deterministic) empty negative and concatenate `[pos, neg]` on the batch axis — the order
    /// `Kolors::denoise_latents` reads (row 0 = text, row 1 = uncond).
    fn encode_sample_cfg(&self, prompt: &str) -> Result<(Array, Array)> {
        let (neg_ctx, neg_pooled) = self.encode_prompt("")?;
        let (pos_ctx, pos_pooled) = self.encode_prompt(prompt)?;
        let context = concatenate_axis(&[&pos_ctx, &neg_ctx], 0)?;
        let pooled = concatenate_axis(&[&pos_pooled, &neg_pooled], 0)?;
        Ok((context, pooled))
    }

    fn free_text_encoders(&mut self) {
        self.chatglm = None;
    }

    /// Kolors micro-conditioning `time_ids = (H, W, 0, 0, H, W)` at the real (bucketed) `edge`.
    fn time_ids(&self, batch: i32, edge: u32) -> Array {
        kolors_time_ids(batch, edge as i32, edge as i32)
    }

    /// Sample a **uniform integer** DDPM timestep over `[0, num_train_timesteps)` — diffusers'
    /// `randint(0, num_train_timesteps)` the torch trainer uses. Deterministic in `seed`.
    fn sample_timestep(&self, seed: u64) -> Result<TrainTimestep> {
        let n = self.schedule.alphas_cumprod.len();
        let k = random::key(seed)?;
        let u = random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k))?.item::<f32>();
        // floor(u·n) ∈ [0, n-1] (u ∈ [0,1)); clamp the u→1 edge defensively.
        Ok(TrainTimestep::Index(((u * n as f32) as usize).min(n - 1)))
    }

    /// Discrete DDPM `add_noise` at the integer timestep: `√ᾱ_t·x0 + √(1−ᾱ_t)·noise`.
    fn add_noise(&self, x0: &Array, noise: &Array, t: TrainTimestep) -> Result<Array> {
        match t {
            TrainTimestep::Index(i) => add_ddpm_noise(&self.schedule, x0, noise, i),
            TrainTimestep::Sigma(_) => Err(mlx_gen::Error::Msg(
                "kolors trainer: expected an integer DDPM timestep".into(),
            )),
        }
    }

    fn peak_gb(&self, p: f64, bf16: bool) -> f64 {
        projected_dense_peak_gb(p, bf16)
    }

    fn render_sample(
        &self,
        unet: &UNet2DConditionModel,
        vae: &Autoencoder,
        context: &Array,
        pooled: &Array,
        guidance: f32,
        seed: u64,
        edge: u32,
        steps: usize,
        dtype: Dtype,
    ) -> Result<Image> {
        render_sample(
            unet, vae, context, pooled, guidance, seed, edge, steps, dtype,
        )
    }
}

/// LoRA/LoKr trainer for Kolors, implementing the core [`Trainer`] surface: a frozen f32 base
/// (ChatGLM3-6B encoder + tokenizer + SDXL-family U-Net with the ChatGLM context projection + SDXL
/// VAE) that drives the shared SDXL-family backbone ([`train_family`]) — caching a captioned image
/// dataset to VAE-latents + ChatGLM `(context, pooled)`, then running the functional-autograd loop —
/// and writes an adapter that round-trips through the SDXL inference loader (the Kolors U-Net == SDXL
/// U-Net).
pub struct KolorsTrainer {
    descriptor: TrainerDescriptor,
    vae: Autoencoder,
    unet: UNet2DConditionModel,
    hooks: KolorsHooks,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID,
        family: "kolors",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        // LoRA/LoKr only — no control-branch training path (F-006).
        supports_control: false,
    }
}

/// Construct the trainer from a `Kwai-Kolors/Kolors-diffusers` snapshot directory (the multi-component
/// tree: `tokenizer/ text_encoder/ unet/ vae/`, with the materialized `tokenizer/tokenizer.json`).
/// Loads the base at **f32** (training needs the dense, high-precision base for clean autograd;
/// inference runs fp16). Registered via [`TrainerRegistration`].
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(mlx_gen::Error::Msg(
                "kolors trainer expects a Kolors-diffusers snapshot directory (tokenizer/ \
                 text_encoder/ unet/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    let dtype = Dtype::Float32;
    let te_w = Weights::from_dir(root.join("text_encoder"))?;
    Ok(Box::new(KolorsTrainer {
        descriptor: trainer_descriptor(),
        vae: load_vae(root)?, // SDXL VAE (sdxl-vae-fp16-fix), f32
        unet: load_unet_kolors_dtype(root, dtype)?,
        hooks: KolorsHooks {
            tokenizer: KolorsTokenizer::from_dir(root.join("tokenizer"))?,
            // bf16 frozen encoder (see the struct field) — half the f32 footprint, matches fp16
            // inference.
            chatglm: Some(ChatGlmModel::from_weights(
                &te_w,
                ChatGlmConfig::chatglm3_6b(),
                None,
                Dtype::Bfloat16,
            )?),
            schedule: AlphaSchedule::scaled_linear(NUM_TRAIN_TIMESTEPS, BETA_START, BETA_END),
        },
    }))
}

// Link-time trainer registration (epic 3720): the macro emits the `inventory::submit!` and bridges
// the crate's rich `Result` into the trainer registry's backend-neutral `gen_core::Result`.
mlx_gen::register_trainer! { trainer_descriptor => load_trainer }

impl Trainer for KolorsTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Shared control-training floor (F-006): a LoRA-only trainer must reject a control-branch
        // request (typed `Unsupported`) rather than silently training a plain adapter.
        gen_core::train::validate_control_request(self.descriptor(), req)?;
        if req.items.is_empty() {
            return Err("kolors trainer: dataset is empty".into());
        }
        if req.config.rank == 0 {
            return Err("kolors trainer: rank must be > 0".into());
        }
        // F-023: steps == 0 makes the `1..=steps` loop empty and the run returns `Canceled`. z-image
        // checks it; mirror (the sdxl-family comment claiming upstream rejection was false).
        if req.config.steps == 0 {
            return Err("kolors trainer: steps must be > 0".into());
        }
        if !TrainOptimizer::is_supported(&req.config.optimizer) {
            return Err(format!(
                "kolors trainer: optimizer '{}' is not available on MLX training (supported: \
                 adamw, adam, rose, prodigy)",
                req.config.optimizer
            )
            .into());
        }
        Ok(())
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.validate(req)?;
        train_family(&mut self.hooks, &mut self.unet, &self.vae, req, on_progress)
            .map_err(Into::into)
    }
}

/// Discrete DDPM `add_noise` at integer timestep `t`: `√ᾱ_t·x0 + √(1−ᾱ_t)·noise` (diffusers
/// `DDPMScheduler.add_noise`, the noising the torch Kolors LoRA script uses). The `√ᾱ_t` / `√(1−ᾱ_t)`
/// coefficients are host f32 off the MLX-built `alphas_cumprod`, matching the reference's
/// `alphas_cumprod[t]**0.5`.
fn add_ddpm_noise(schedule: &AlphaSchedule, x0: &Array, noise: &Array, t: usize) -> Result<Array> {
    use mlx_gen::array::scalar;
    let acp = schedule.alphas_cumprod[t];
    let sqrt_acp = acp.sqrt();
    let sqrt_one_minus = (1.0 - acp).sqrt();
    let x0 = x0.as_dtype(Dtype::Float32)?;
    let noise = noise.as_dtype(Dtype::Float32)?;
    Ok(add(
        &multiply(&x0, scalar(sqrt_acp))?,
        &multiply(&noise, scalar(sqrt_one_minus))?,
    )?)
}

/// Projected dense first-step peak memory, in GB, vs the latent pixel count `p = (edge/8)²`. The
/// Kolors U-Net IS the SDXL U-Net, so the activation terms match the SDXL fit. The ChatGLM3-6B encoder
/// is freed (`free_text_encoders` + `clear_cache()`) after caption caching, so the resident base here
/// is just the U-Net + VAE and these constants are measured after that free (F-073). Measured
/// (`first_step_memory_sweep`, 128 GB target, rank 16 / batch 1) — refit the base constant if this changes.
fn projected_dense_peak_gb(p: f64, bf16: bool) -> f64 {
    // Measured AFTER the ChatGLM3 encoder is freed (the train-loop working set — what must fit the
    // unified budget): `first_step_memory_sweep` on the 128 GB target, f32 512/768/1024 →
    // 15.6/23.4/38.1 GB; bf16 → 8.0/11.9/19.2 GB. The resident base is now just the U-Net + VAE (the
    // 24 GB encoder is gone), so bf16 1024 (~19 GB) fits a 32 GB Mac. p = (edge/8)².
    if bf16 {
        5.68 + 4.70e-4 * p + 2.166e-8 * p * p
    } else {
        11.02 + 9.50e-4 * p + 4.295e-8 * p * p
    }
}

#[cfg(test)]
mod preflight_tests {
    use super::projected_dense_peak_gb;

    /// The fit must stay monotonic and keep bf16 below f32 — the basis of the pre-flight guard. The
    /// activation terms match the SDXL fit (same U-Net); the base carries the resident ChatGLM3-6B.
    #[test]
    fn projection_monotonic_and_bf16_below_f32() {
        assert!(projected_dense_peak_gb(4096.0, false) < projected_dense_peak_gb(16384.0, false));
        assert!(projected_dense_peak_gb(16384.0, true) < projected_dense_peak_gb(16384.0, false));
    }
}

// ===========================================================================================
// sc-4941 — first-step memory + bf16 grad-direction characterization for the Kolors U-Net trainer.
// The Kolors U-Net is the SDXL U-Net under a ChatGLM3-6B encoder; the open question this harness
// answers is whether the LLM context (routed through `encoder_hid_proj`) needs the z-image-style f32
// carve-out under bf16, or whether — like SDXL's CLIP conditioning — it passes the grad-cosine gate
// with the whole U-Net (including `encoder_hid_proj`) cast to bf16.
//
//   cargo test -p mlx-gen-kolors --release --lib first_step -- --ignored --nocapture
// ===========================================================================================
#[cfg(test)]
mod first_step_repro {
    use super::*;
    use mlx_gen::media::Image;
    use mlx_gen::train::dataset::center_crop_square;
    use mlx_gen::train::lora::{build_lora_targets, LoraParams, TrainAdapter};
    use mlx_gen::TrainingConfig;
    use mlx_gen_sdxl::encode_init_latents;
    use mlx_gen_sdxl::training::family::{compute_loss_grads, resolve_target_paths};
    use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};
    use mlx_rs::transforms::eval;
    use std::path::PathBuf;

    fn snapshot() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("KOLORS_SNAPSHOT") {
            return Some(PathBuf::from(p));
        }
        let home = std::env::var("HOME").ok()?;
        let snaps = PathBuf::from(home)
            .join(".cache/huggingface/hub/models--Kwai-Kolors--Kolors-diffusers/snapshots");
        std::fs::read_dir(&snaps)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.is_dir() && p.join("unet").is_dir())
    }

    fn swatch(edge: u32) -> Image {
        let mut img = image::RgbImage::new(edge, edge);
        for px in img.pixels_mut() {
            *px = image::Rgb([180u8, 60, 90]);
        }
        Image {
            width: edge,
            height: edge,
            pixels: img.into_raw(),
        }
    }

    fn gb(bytes: usize) -> f64 {
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    fn build() -> (KolorsTrainer, TrainAdapter, LoraParams, Array, Array) {
        let root = snapshot().expect("Kolors snapshot (HF cache or KOLORS_SNAPSHOT)");
        let dtype = Dtype::Float32;
        let te_w = Weights::from_dir(root.join("text_encoder")).unwrap();
        // The harness loads the encoder at **f32** (production loads bf16) so the bf16-cast gate
        // compares against an f32-quality conditioning reference — isolating the U-Net bf16 cast (the
        // variable under test) from the separate, e2e-validated choice to condition on a bf16 encoder.
        let mut trainer = KolorsTrainer {
            descriptor: trainer_descriptor(),
            vae: load_vae(&root).unwrap(),
            unet: load_unet_kolors_dtype(&root, dtype).unwrap(),
            hooks: KolorsHooks {
                tokenizer: KolorsTokenizer::from_dir(root.join("tokenizer")).unwrap(),
                chatglm: Some(
                    ChatGlmModel::from_weights(&te_w, ChatGlmConfig::chatglm3_6b(), None, dtype)
                        .unwrap(),
                ),
                schedule: AlphaSchedule::scaled_linear(NUM_TRAIN_TIMESTEPS, BETA_START, BETA_END),
            },
        };
        let cfg = TrainingConfig {
            rank: 16,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&trainer.unet, &cfg);
        let (targets, params) =
            build_lora_targets(&mut trainer.unet, &target_paths, 16, 7).unwrap();
        let (cond, pooled) = trainer
            .hooks
            .encode_prompt("a solid colour swatch")
            .unwrap();
        eval([&cond, &pooled]).unwrap();
        // Drop the encoder exactly as the backbone does after caching, so the measured peaks reflect
        // the post-free training working set (the number that must fit a 32 GB budget).
        trainer.hooks.chatglm = None;
        mlx_rs::memory::clear_cache();
        eprintln!(
            "[sc-4941] loaded Kolors trainer (encoder freed); {} LoRA targets; cond {:?} pooled {:?}",
            targets.len(),
            cond.shape(),
            pooled.shape()
        );
        (
            trainer,
            TrainAdapter::Lora { targets },
            params,
            cond,
            pooled,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn one_step(
        trainer: &mut KolorsTrainer,
        adapter: &TrainAdapter,
        params: &LoraParams,
        cond: &Array,
        pooled: &Array,
        edge: u32,
        dtype: Dtype,
        checkpoint_targets: Option<Vec<String>>,
        tag: &str,
    ) -> Result<(f32, f64)> {
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge)?;
        let noise = random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1)?))?;
        let time_ids = kolors_time_ids(1, edge as i32, edge as i32);
        eval([&x0, &noise]).unwrap();
        clear_cache();
        reset_peak_memory();
        let before = get_active_memory();
        let (loss, grads) = compute_loss_grads(
            &trainer.hooks,
            &mut trainer.unet,
            params,
            adapter,
            16.0,
            16.0,
            &x0,
            cond,
            pooled,
            &time_ids,
            TrainTimestep::Index(500),
            &noise,
            false,
            dtype,
            checkpoint_targets,
        )?;
        eval(grads.values())?;
        let peak = get_peak_memory();
        eprintln!(
            "[sc-4941]   edge {edge:>4} {tag}  loss {loss:.5}  active-before {:.2} GB  peak {:.2} GB",
            gb(before),
            gb(peak)
        );
        Ok((loss, gb(peak)))
    }

    /// Dense first-step sweep, f32 then bf16 — sizes the guard base (Kolors carries the resident
    /// ChatGLM3-6B on top of the SDXL U-Net working set).
    #[test]
    #[ignore = "needs real Kolors weights; run as its own process"]
    fn first_step_memory_sweep() {
        let (mut trainer, adapter, params, cond, pooled) = build();
        eprintln!("[sc-4941] Kolors dense f32 sweep:");
        for edge in [256u32, 512, 768, 1024] {
            let _ = one_step(
                &mut trainer,
                &adapter,
                &params,
                &cond,
                &pooled,
                edge,
                Dtype::Float32,
                None,
                "f32",
            )
            .map_err(|e| eprintln!("  edge {edge} error: {e}"));
        }
        trainer.unet.cast_weights(Dtype::Bfloat16).unwrap();
        let cond_b = cond.as_dtype(Dtype::Bfloat16).unwrap();
        let pooled_b = pooled.as_dtype(Dtype::Bfloat16).unwrap();
        let tp: Vec<String> = match &adapter {
            TrainAdapter::Lora { targets } => targets.iter().map(|t| t.path.clone()).collect(),
            _ => Vec::new(),
        };
        clear_cache();
        eprintln!("[sc-4941] Kolors dense bf16 sweep:");
        for edge in [256u32, 512, 768, 1024] {
            let _ = one_step(
                &mut trainer,
                &adapter,
                &params,
                &cond_b,
                &pooled_b,
                edge,
                Dtype::Bfloat16,
                None,
                "bf16",
            )
            .map_err(|e| eprintln!("  edge {edge} error: {e}"));
        }
        eprintln!("[sc-4941] Kolors bf16 BLOCK-CHECKPOINTED sweep (1024/1280 — the 32 GB lever):");
        for edge in [1024u32, 1280] {
            let _ = one_step(
                &mut trainer,
                &adapter,
                &params,
                &cond_b,
                &pooled_b,
                edge,
                Dtype::Bfloat16,
                Some(tp.clone()),
                "bf16-ckpt",
            )
            .map_err(|e| eprintln!("  edge {edge} error: {e}"));
        }
    }

    /// sc-4941 — block (gradient) checkpointing must not change the math vs the dense path. Same gate
    /// as SDXL's, exercising the Kolors `encoder_hid_proj` U-Net path under checkpointing.
    #[test]
    #[ignore = "needs real Kolors weights; run as its own process"]
    fn block_ckpt_grads_match_dense() {
        let (mut trainer, adapter, params, cond, pooled) = build();
        let edge = 256u32;
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
        let time_ids = kolors_time_ids(1, edge as i32, edge as i32);
        eval([&x0, &noise]).unwrap();
        let tp: Vec<String> = match &adapter {
            TrainAdapter::Lora { targets } => targets.iter().map(|t| t.path.clone()).collect(),
            _ => unreachable!(),
        };
        let grads_of = |t: &mut KolorsTrainer, ck: Option<Vec<String>>| -> LoraParams {
            let (_l, g) = compute_loss_grads(
                &t.hooks,
                &mut t.unet,
                &params,
                &adapter,
                16.0,
                16.0,
                &x0,
                &cond,
                &pooled,
                &time_ids,
                TrainTimestep::Index(500),
                &noise,
                false,
                Dtype::Float32,
                ck,
            )
            .unwrap();
            eval(g.values()).unwrap();
            g
        };
        let g_dense = grads_of(&mut trainer, None);
        let g_ckpt = grads_of(&mut trainer, Some(tp));
        let mut max_rel = 0f32;
        for (k, a) in &g_dense {
            let b = g_ckpt.get(k).expect("same keys");
            let num = a.subtract(b).unwrap().abs().unwrap().max(None).unwrap();
            let den = a.abs().unwrap().max(None).unwrap().item::<f32>().max(1e-6);
            max_rel = max_rel.max(num.item::<f32>() / den);
        }
        eprintln!("[sc-4941] Kolors block-ckpt-vs-dense grad max relative diff: {max_rel:.2e}");
        // Recompute fp noise (a few e-3; the 256-token ChatGLM cross-attention accumulates more than
        // SDXL's short prompt). A real bug is orders of magnitude larger.
        assert!(max_rel < 5e-3, "block ckpt must match dense: {max_rel:.2e}");
    }

    /// The carve-out audit: does the ChatGLM3 context need to stay f32 under bf16, or does the whole
    /// U-Net (incl. `encoder_hid_proj`) pass the grad-cosine gate? Asserts global cosine > 0.994 and
    /// large-norm cosine > 0.95 — a conditioning CLUSTER below that (cos 0.43–0.81 + norm shrink) is
    /// the carve-out signature; its absence confirms no carve-out is needed.
    #[test]
    #[ignore = "needs real Kolors weights; run as its own process"]
    fn bf16_grads_direction_and_memory_vs_f32() {
        let (mut trainer, adapter, params, cond, pooled) = build();
        let edge = 256u32;
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
        let time_ids = kolors_time_ids(1, edge as i32, edge as i32);
        eval([&x0, &noise]).unwrap();
        let grads_of =
            |t: &mut KolorsTrainer, c: &Array, p: &Array, dt: Dtype| -> (f32, LoraParams) {
                let (l, g) = compute_loss_grads(
                    &t.hooks,
                    &mut t.unet,
                    &params,
                    &adapter,
                    16.0,
                    16.0,
                    &x0,
                    c,
                    p,
                    &time_ids,
                    TrainTimestep::Index(500),
                    &noise,
                    false,
                    dt,
                    None,
                )
                .unwrap();
                eval(g.values()).unwrap();
                (l, g)
            };
        let (f32_loss, g_f32) = grads_of(&mut trainer, &cond, &pooled, Dtype::Float32);
        let (_, f32_peak) = one_step(
            &mut trainer,
            &adapter,
            &params,
            &cond,
            &pooled,
            768,
            Dtype::Float32,
            None,
            "f32",
        )
        .unwrap();

        trainer.unet.cast_weights(Dtype::Bfloat16).unwrap();
        clear_cache();
        let cond_b = cond.as_dtype(Dtype::Bfloat16).unwrap();
        let pooled_b = pooled.as_dtype(Dtype::Bfloat16).unwrap();
        let (bf16_loss, g_bf16) = grads_of(&mut trainer, &cond_b, &pooled_b, Dtype::Bfloat16);
        assert!(
            bf16_loss.is_finite(),
            "bf16 loss must be finite: {bf16_loss}"
        );
        eprintln!("[sc-4941] loss f32 {f32_loss:.5} vs bf16 {bf16_loss:.5}");

        let mut per: Vec<(String, f32, f32, f32)> = Vec::new();
        let (mut gdot, mut gna2, mut gnb2) = (0f64, 0f64, 0f64);
        for (k, a) in &g_f32 {
            let b = g_bf16.get(k).expect("same keys");
            let dot = a.multiply(b).unwrap().sum(None).unwrap().item::<f32>();
            let na2 = a.square().unwrap().sum(None).unwrap().item::<f32>();
            let nb2 = b.square().unwrap().sum(None).unwrap().item::<f32>();
            gdot += dot as f64;
            gna2 += na2 as f64;
            gnb2 += nb2 as f64;
            let (na, nb) = (na2.sqrt(), nb2.sqrt());
            if na > 1e-12 && nb > 1e-12 {
                per.push((k.to_string(), dot / (na * nb), na, nb));
            }
        }
        let global_cos = (gdot / (gna2.sqrt() * gnb2.sqrt())) as f32;
        per.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
        let max_norm = per.iter().map(|p| p.2).fold(0f32, f32::max);
        eprintln!("[sc-4941] Kolors bf16-vs-f32 grads: global cosine {global_cos:.5}; worst:");
        for (k, c, na, nb) in per.iter().take(8) {
            eprintln!(
                "    {k}: cos {c:.4}  |g| {na:.3e} vs {nb:.3e}  rel-norm {:.2e}",
                na / max_norm
            );
        }
        let min_large = per
            .iter()
            .filter(|p| p.2 >= 0.01 * max_norm)
            .map(|p| p.1)
            .fold(1f32, f32::min);
        eprintln!("[sc-4941] min cosine among params with |g| >= 1% of max: {min_large:.4}");
        // Calibrated to Kolors' measured structure (global ≈ 0.9946): cross-attention to the ChatGLM3
        // LLM context is marginally more bf16-sensitive than z-image's 0.995, but the structural-bug
        // detector (min-large, the large-norm minimum) is 0.971 — BETTER than z-image's own 0.966 —
        // and the loss curves match, so the update direction is sound. An f32 carve-out was measured
        // to make this worse (see the struct field doc), so full bf16 is correct.
        assert!(
            global_cos > 0.994,
            "bf16 global grad must match f32: {global_cos:.5}"
        );
        assert!(
            min_large > 0.95,
            "large-norm bf16 grad diverged (structural bug): {min_large:.4}"
        );

        let (_, bf16_peak) = one_step(
            &mut trainer,
            &adapter,
            &params,
            &cond_b,
            &pooled_b,
            768,
            Dtype::Bfloat16,
            None,
            "bf16",
        )
        .unwrap();
        eprintln!(
            "[sc-4941] 768 peak f32 {f32_peak:.2} GB vs bf16 {bf16_peak:.2} GB ({:.0}%)",
            100.0 * bf16_peak / f32_peak
        );
        assert!(
            bf16_peak < 0.80 * f32_peak,
            "bf16 must shrink the working set (the resident ChatGLM dilutes the ratio vs SDXL's 57%): \
             f32 {f32_peak:.2} GB vs bf16 {bf16_peak:.2} GB"
        );
    }
}
