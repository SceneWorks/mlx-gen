//! sc-3045 — LoRA/LoKr **training** on the SDXL U-Net, in pure Rust on mlx-rs. The SDXL realization
//! of the core [`Trainer`] contract (epic 3039), built on the same functional-autograd mechanism the
//! Z-Image trainer proved (sc-3042/3044) and the host-generic factor machinery hoisted to core
//! ([`mlx_gen::train::lora`], sc-3045). Parity target = the SceneWorks torch `SdxlLoraTrainer` /
//! `_SdxlLoraBackend`.
//!
//! **What is SDXL-specific here** (everything else is the shared core machinery):
//!   * **Noise / objective — discrete DDPM in the vendored sigma-space.** SDXL inference runs the
//!     vendored k-diffusion Euler-Ancestral sampler ([`EulerSampler`]): latents are stored
//!     *renormalized*, `scale_model_input` is the identity, and the per-step time `t` is the float
//!     sigma-table index in `[0, 1000]` that the U-Net's sinusoidal embedding consumes. Crucially the
//!     renormalized model input `(x0 + σ·noise)·rsqrt(σ²+1)` is **algebraically identical** to the
//!     diffusers DDPM `noisy = √(ᾱ)·x0 + √(1−ᾱ)·noise` (since `rsqrt(σ²+1) = √(ᾱ)`,
//!     `σ·rsqrt(σ²+1) = √(1−ᾱ)`), and the **epsilon** target is the unit `noise`. So training reuses
//!     the crate's own [`EulerSampler::add_noise_with`] at a sampled integer table-index `t` — making
//!     train/inference consistent **by construction** — and regresses the U-Net's `eps` toward
//!     `noise`. (SDXL-base is epsilon-prediction; the v-prediction the torch reference's
//!     `prediction_type` branch supports is never taken for SDXL-base, and the crate's eps-only
//!     sampler could not consume a v-pred adapter — so eps is the correct and only objective here.)
//!     `t` is sampled **uniform over the integer table indices `[1, 1000]`**, which maps 1:1 onto the
//!     diffusers `randint(0, 1000)` the torch trainer uses (the table is `concat([0], σ_1..σ_1000)`).
//!   * **`added_cond_kwargs`.** The U-Net forward takes the pooled `text_embeds` (CLIP-bigG pooled)
//!     and the 6-element `time_ids`. The crate's inference path hardcodes
//!     `time_ids = [512,512,0,0,512,512]` (the vendored `generate_latents` quirk — it ignores the
//!     real size); training feeds the **same** [`text_time_ids`] so the conditioning the LoRA learns
//!     under matches what inference applies it under. (This deliberately diverges from the torch
//!     trainer's real-resolution time_ids — that would mismatch this engine's inference.)
//!   * **Dual-CLIP conditioning.** `encoder_hidden_states = concat(CLIP-L.hidden[-2], bigG.hidden[-2])`
//!     and pooled `text_embeds = bigG.pooled`, via [`encode_conditioning`]. Single forward, no CFG
//!     (the torch ref encodes with `do_classifier_free_guidance=False`).
//!   * **f32 base.** The U-Net + both text encoders + VAE load at f32 for clean autograd (the
//!     inference path runs fp16; the trained f32 factors merge into the fp16 base at load, casts
//!     handled by the loader). The VAE encodes the f32 init image to the scaled latent `x0`.
//!   * **Adapter surface, matched to inference consumption.** LoRA targets the **complete** UNet
//!     attention surface (down/mid/up `to_q/k/v/to_out.0`) — what `LoraCoverage::Complete`
//!     (`model::load`'s default) merges, and what the torch PEFT suffix-match selects. LoKr targets
//!     the **vendored** surface (down/up attention only): the SDXL LoKr loader keeps `mid_block` out
//!     (sc-2640), so training mid_block LoKr would produce factors no inference path reads. LoRA
//!     saves PEFT keys under `base_model.model.unet.` (what `_SdxlLoraBackend` emits); LoKr saves the
//!     bare `<path>.lokr_*` keys; both reconstruct at **f32** (the SDXL merge dtype).

use std::path::Path;

use mlx_gen::train::checkpoint::checkpoint_filename;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, average_grads, build_lokr_targets, build_lora_targets, LoraParams,
    TrainAdapter,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::{
    gen_core, LoadSpec, Modality, NetworkType, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::subtract;
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::config::DiffusionConfig;
use crate::model::MODEL_ID;
use crate::pipeline::{encode_conditioning, encode_init_latents, text_time_ids};
use crate::sampler::EulerSampler;
use crate::text_encoder::ClipTextEncoder;
use crate::tokenizer::ClipBpeTokenizer;
use crate::unet::UNet2DConditionModel;
use crate::vae::Autoencoder;

/// SDXL reconstructs its LoKr delta at **f32** (the f32-everywhere merge path); training must match
/// so the adapter round-trips through the inference loader.
const LOKR_DTYPE: Dtype = Dtype::Float32;

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence (sc-5637).
const SAMPLE_PROMPT_CAP: usize = 4;

/// PEFT save-key prefix for the LoRA adapter — what `peft.save_pretrained()` / the SceneWorks
/// `_SdxlLoraBackend` emit, and what the SDXL loader's PEFT key classifier
/// (`adapters::classify_key`) expects.
const PEFT_PREFIX: &str = "base_model.model.unet.";

/// The default SDXL attention LoRA targets — the suffixes `to_q`/`to_k`/`to_v`/`to_out.0` the torch
/// trainer uses (`DEFAULT_LORA_TARGET_MODULES`, `training_adapters.py:72`), suffix-matched across the
/// UNet attention modules exactly as PEFT's `LoraConfig(target_modules=…)` does.
const DEFAULT_TARGET_SUFFIXES: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// LoRA/LoKr trainer for Stable Diffusion XL, implementing the core [`Trainer`] surface: a frozen
/// f32 base (U-Net + dual CLIP + VAE + tokenizer) that caches a captioned image dataset to
/// VAE-latents + dual-CLIP conditioning/pooled embeds, then runs the functional-autograd loop with
/// the sc-3043 runtime glue (LR schedule, gradient accumulation, checkpoint cadence, cancel,
/// progress bands), and writes an adapter that round-trips through the SDXL inference loader.
pub struct SdxlTrainer {
    descriptor: TrainerDescriptor,
    tokenizer: ClipBpeTokenizer,
    /// The dual CLIP encoders, in `Option`s so they can be **dropped after the caching loop** (sc-4941,
    /// 32 GB-Mac headroom): they are idle during training (every prompt is already encoded to the
    /// cached conditioning), so freeing them (~3.3 GB at f32) before the train loop leaves more of the
    /// unified-memory budget for the U-Net working set.
    te1: Option<ClipTextEncoder>,
    te2: Option<ClipTextEncoder>,
    vae: Autoencoder,
    unet: UNet2DConditionModel,
    /// The SDXL noise schedule (the same sigma table the inference Euler-Ancestral sampler uses);
    /// training reuses its [`EulerSampler::add_noise_with`] for the renormalized DDPM noising.
    sampler: EulerSampler,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID,
        family: "sdxl",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// Construct the trainer from an SDXL snapshot directory (the diffusers multi-component tree:
/// `tokenizer/ text_encoder/ text_encoder_2/ unet/ vae/`). Loads the base at **f32** (training needs
/// the dense, high-precision base for clean autograd; inference runs fp16). Registered via
/// [`TrainerRegistration`].
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(mlx_gen::Error::Msg(
                "sdxl trainer expects a snapshot directory (tokenizer/ text_encoder/ \
                 text_encoder_2/ unet/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    Ok(Box::new(SdxlTrainer {
        descriptor: trainer_descriptor(),
        tokenizer: crate::loader::load_tokenizer(root)?,
        te1: Some(crate::loader::load_text_encoder_1(root)?),
        te2: Some(crate::loader::load_text_encoder_2(root)?),
        vae: crate::loader::load_vae(root)?,
        unet: crate::loader::load_unet(root)?,
        sampler: EulerSampler::new(&DiffusionConfig::sdxl_base(), true)?,
    }))
}

// Link-time trainer registration (epic 3720): the macro emits the `inventory::submit!` and bridges
// the crate's rich `Result` into the trainer registry's backend-neutral `gen_core::Result`.
mlx_gen::register_trainer! { trainer_descriptor => load_trainer }

impl SdxlTrainer {
    /// Caption → `(conditioning [1, N, 2048], pooled [1, 1280])`: tokenize (no negative — training is
    /// CFG-off), run both CLIP encoders, and assemble the SDXL dual-CLIP conditioning + pooled embed
    /// exactly as the inference [`encode_conditioning`] path.
    fn encode_prompt(&self, caption: &str) -> Result<(Array, Array)> {
        let (te1, te2) = match (&self.te1, &self.te2) {
            (Some(a), Some(b)) => (a, b),
            _ => {
                return Err(mlx_gen::Error::Msg(
                    "sdxl trainer: text encoders already freed (encode_prompt after caching)"
                        .into(),
                ))
            }
        };
        let tokens = self.tokenizer.tokenize_batch(caption, None)?;
        encode_conditioning(te1, te2, &tokens)
    }
}

impl Trainer for SdxlTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        if req.items.is_empty() {
            return Err("sdxl trainer: dataset is empty".into());
        }
        if req.config.rank == 0 {
            return Err("sdxl trainer: rank must be > 0".into());
        }
        if !TrainOptimizer::is_supported(&req.config.optimizer) {
            return Err(format!(
                "sdxl trainer: optimizer '{}' is not available on MLX training (supported: \
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
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

impl SdxlTrainer {
    /// The rich-`Result` body behind [`Trainer::train`]; the trait wrapper bridges its tail into
    /// [`gen_core::Error`] (epic 3720), keeping `?` on `mlx_rs`/family helpers transparent here.
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        self.validate(req)?;
        let cfg = &req.config;
        on_progress(TrainingProgress::Preparing);
        let edge = bucket_resolution(cfg.resolution);

        // sc-4941 — training compute dtype. bf16 (the worker default, passed through since sc-4881)
        // halves the activation working set and is the ecosystem-standard mixed precision; the
        // trainable factors / loss / grads / optimizer stay f32 (master-weights). The U-Net f32→bf16
        // cast is destructive, so a trainer already cast to bf16 cannot honor a later f32 request —
        // reload instead of silently training at the wrong precision. Unlike z-image (which OOM-killed
        // at 1024), the SDXL working set is modest; combined with freeing the CLIP encoders after
        // caching, bf16 1024 LoRA training fits a 32 GB Mac (~18 GB peak vs ~36 GB f32).
        let use_bf16 = cfg.train_dtype.trim().eq_ignore_ascii_case("bf16")
            || cfg.train_dtype.trim().eq_ignore_ascii_case("bfloat16");
        let compute_dtype = if use_bf16 {
            Dtype::Bfloat16
        } else {
            Dtype::Float32
        };
        if !use_bf16 && self.unet.compute_dtype() == Some(Dtype::Bfloat16) {
            return Err(
                "sdxl trainer: this trainer instance was already cast to bf16 by a previous run; \
                 reload the trainer for f32 training"
                    .into(),
            );
        }

        // sc-4941 — opt-in gradient checkpointing (the SceneWorks "Gradient Checkpointing" toggle,
        // passed through since sc-4881). When on, each down/up macro-block recomputes its activations
        // in the backward (`forward_block_checkpointed`) instead of retaining them — the lever that
        // makes 1280+ training fit a 32 GB Mac (1024 already fits dense bf16). LoRA-only: LoKr falls
        // back to the dense path (a distinct Kronecker reconstruction), where the pre-flight guard
        // refuses a run that would exceed the memory budget. The block recompute already covers
        // attention, so the standalone SDPA-segment checkpoint stays off (nesting = double recompute).
        let use_checkpoint =
            matches!(cfg.network_type, NetworkType::Lora) && cfg.gradient_checkpointing;
        if !use_checkpoint {
            preflight_memory_guard(edge, use_bf16)?;
        }
        self.unet.set_sdpa_checkpoint(false);
        if use_bf16 {
            self.unet.cast_weights(Dtype::Bfloat16)?;
        }

        // --- prepare → load → cache: VAE-latents + dual-CLIP (conditioning, pooled) into memory ---
        on_progress(TrainingProgress::LoadingModel); // base already resident from load_trainer
        let total = req.items.len() as u32;
        let mut cache: Vec<(Array, Array, Array)> = Vec::with_capacity(req.items.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = center_crop_square(&decode_image(&item.image_path)?);
            let x0 = encode_init_latents(&self.vae, &img, edge, edge)?; // scaled latent [1,h,w,4]
            let (cond, pooled) = self.encode_prompt(&item.caption)?;
            eval([&x0, &cond, &pooled])?;
            cache.push((x0, cond, pooled));
        }
        if cache.is_empty() {
            // sc-4895 — a cancel tripped during caching is a genuine cancellation → typed
            // `Error::Canceled` (bridged 1:1 to `gen_core::Error::Canceled`); an empty cache with no
            // cancel is a real "no usable dataset items" error.
            if req.cancel.is_cancelled() {
                return Err(mlx_gen::Error::Canceled);
            }
            return Err("sdxl trainer: no usable dataset items".into());
        }

        // sc-5637 — pre-encode the preview-sample prompts as a **CFG batch** (`[2, …]` = positive
        // then empty-negative) while the dual CLIP encoders are still resident (freed just below).
        // SDXL renders previews with real classifier-free guidance, so the denoise needs both streams.
        let sample_caps: Vec<(String, Array, Array)> =
            if cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() && !req.cancel.is_cancelled()
            {
                let (te1, te2) = match (&self.te1, &self.te2) {
                    (Some(a), Some(b)) => (a, b),
                    _ => {
                        return Err(mlx_gen::Error::Msg(
                            "sdxl trainer: text encoders already freed (sample pre-encode)".into(),
                        ))
                    }
                };
                let mut caps = Vec::with_capacity(cfg.sample_prompts.len().min(SAMPLE_PROMPT_CAP));
                for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                    let tokens = self.tokenizer.tokenize_batch(prompt, Some(""))?;
                    let (cond, pooled) = encode_conditioning(te1, te2, &tokens)?;
                    let cond = if compute_dtype == Dtype::Float32 {
                        cond
                    } else {
                        cond.as_dtype(compute_dtype)?
                    };
                    let pooled = if compute_dtype == Dtype::Float32 {
                        pooled
                    } else {
                        pooled.as_dtype(compute_dtype)?
                    };
                    eval([&cond, &pooled])?;
                    caps.push((prompt.clone(), cond, pooled));
                }
                caps
            } else {
                Vec::new()
            };
        let sampling_enabled = !sample_caps.is_empty();

        // sc-4941 (32 GB-Mac headroom) — the prompts are all encoded into `cache`, so the dual CLIP
        // encoders are dead weight for the rest of the run. Drop them and evict their buffers before
        // the train loop, reclaiming ~3.3 GB for the U-Net working set.
        self.te1 = None;
        self.te2 = None;
        mlx_rs::memory::clear_cache();

        // SDXL micro-conditioning `time_ids`, built once and shared (B=1). Matches the inference
        // path's hardcoded `[512,512,0,0,512,512]` so the LoRA trains under the conditioning it is
        // applied under.
        let time_ids = text_time_ids(1);

        // --- adapter targets + params (LoRA or LoKr) + optimizer ---
        let target_paths = resolve_target_paths(&self.unet, cfg);
        // When block checkpointing is on, the per-step forward threads these target paths' LoRA
        // factors through the block checkpoints; `None` selects the dense forward.
        let checkpoint_targets: Option<Vec<String>> = use_checkpoint.then(|| target_paths.clone());
        let rank = cfg.rank as f32;
        let (adapter, mut params) = match cfg.network_type {
            NetworkType::Lora => {
                let (targets, params) =
                    build_lora_targets(&mut self.unet, &target_paths, cfg.rank as i32, cfg.seed)?;
                (TrainAdapter::Lora { targets }, params)
            }
            NetworkType::Lokr => {
                let (targets, params) = build_lokr_targets(
                    &mut self.unet,
                    &target_paths,
                    cfg.rank as i32,
                    cfg.decompose_factor,
                    cfg.seed,
                )?;
                (TrainAdapter::Lokr { targets }, params)
            }
        };
        let alpha = cfg.alpha;
        let mae = {
            let lt = cfg.loss_type.to_ascii_lowercase();
            lt == "mae" || lt == "l1"
        };
        // AdamW with wd=0 is identical to Adam, so the one optimizer covers both choices.
        let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
            0.0
        } else {
            cfg.weight_decay
        };
        let mut opt = TrainOptimizer::from_config(&cfg.optimizer, cfg.learning_rate, weight_decay)?;

        let accum = cfg.gradient_accumulation.max(1);
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
        let stem = Path::new(&req.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lora")
            .to_string();

        // --- train loop ---
        let mut accumulated: Option<LoraParams> = None;
        let mut update_idx: u32 = 0;
        let mut last_loss = 0.0f32;
        let mut steps_run = 0u32;
        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (x0, cond, pooled) = &cache[((step - 1) as usize) % cache.len()];
            // Uniform integer DDPM timestep over the sigma-table indices [1, max_time].
            let t = sample_timestep(
                &self.sampler,
                cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64),
            )?;
            let noise = random::normal::<f32>(
                x0.shape(),
                None,
                None,
                Some(&random::key(
                    cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
                )?),
            )?;
            let (loss, grads) = compute_loss_grads(
                &mut self.unet,
                &self.sampler,
                &params,
                &adapter,
                alpha,
                rank,
                x0,
                cond,
                pooled,
                &time_ids,
                t,
                &noise,
                mae,
                compute_dtype,
                checkpoint_targets.clone(),
            )?;
            last_loss = loss;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;

            if step % accum == 0 || step == cfg.steps {
                let mult =
                    lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                opt.set_lr_scaled(mult);
                let avg = average_grads(
                    accumulated
                        .take()
                        .expect("an update fires only after accumulation"),
                    accum,
                )?;
                let (clipped, _norm) = clip_grad_norm(&avg, 1.0)?;
                let clipped: LoraParams = clipped
                    .into_iter()
                    .map(|(k, v)| (k, v.into_owned()))
                    .collect();
                opt.step(&mut params, &clipped)?;
                eval(params.values())?;
                update_idx += 1;
            }

            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                std::fs::create_dir_all(&req.output_dir)?;
                let ckpt = req.output_dir.join(checkpoint_filename(&stem, step));
                adapter.save(
                    &params,
                    alpha,
                    rank,
                    cfg.decompose_factor,
                    PEFT_PREFIX,
                    &ckpt,
                )?;
                on_progress(TrainingProgress::Checkpoint { step });
            }

            // sc-5637 — periodic best-effort previews from the in-progress adapter (mirrors z-image).
            // Install the current factors as concrete adapters for the forward-only render; the next
            // step's traced `loss_fn` re-installs them, so no teardown is needed. A render failure must
            // NOT abort the long training run — log and continue.
            if sampling_enabled && step % cfg.sample_every == 0 {
                let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);
                adapter.install_as(&mut self.unet, &params, alpha, rank, lora_dtype, LOKR_DTYPE)?;
                let total = sample_caps.len() as u32;
                for (i, (prompt, cond, pooled)) in sample_caps.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    let sample_seed = cfg
                        .seed
                        .wrapping_add(step as u64)
                        .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                        .wrapping_add(i as u64);
                    match crate::pipeline::render_sample(
                        &self.unet,
                        &self.vae,
                        &self.sampler,
                        cond,
                        pooled,
                        cfg.sample_guidance_scale,
                        sample_seed,
                        edge,
                        cfg.sample_steps.max(1) as usize,
                    ) {
                        Ok(image) => on_progress(TrainingProgress::Sample {
                            step,
                            index: i as u32 + 1,
                            total,
                            prompt: prompt.clone(),
                            image,
                        }),
                        Err(e) => eprintln!(
                            "[sc-5637] {MODEL_ID} preview sample failed at step {step} \
                             (prompt {}): {e} — skipping this preview, training continues",
                            i + 1
                        ),
                    }
                }
            }
        }

        // Cancelled before completing a single step (`steps == 0` is rejected upstream by
        // `validate`): the LoRA factors are still freshly initialized with `B = 0`, a no-op adapter.
        // Surface the typed `Error::Canceled` (sc-4895, bridged 1:1 to `gen_core::Error::Canceled`)
        // rather than writing a valid-looking `.safetensors` and returning `Ok` — downstream tooling
        // would otherwise ship an identity LoRA as a trained artifact (F-040).
        if steps_run == 0 {
            return Err(mlx_gen::Error::Canceled);
        }

        // --- save final adapter ---
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        adapter.save(
            &params,
            alpha,
            rank,
            cfg.decompose_factor,
            PEFT_PREFIX,
            &adapter_path,
        )?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Resolve the config's target-module *suffixes* (default `to_q`/`to_k`/`to_v`/`to_out.0`) to full
/// dotted UNet paths by suffix-matching them against the routable Linear surface — the same match
/// PEFT's `LoraConfig(target_modules=…)` does over the UNet attention modules.
///
/// The surface is chosen to match each adapter kind's **inference consumption** (so nothing trains
/// that no inference path reads, and the adapter round-trips cleanly):
///   * **LoRA** → the **complete** surface ([`UNet2DConditionModel::lora_target_paths_complete`]),
///     which `LoraCoverage::Complete` (`model::load`'s default) merges — down / **mid** / up
///     attention. Matches the torch PEFT suffix-match (which hits mid_block too).
///   * **LoKr** → the **vendored** surface ([`UNet2DConditionModel::lora_target_paths`]), down / up
///     attention only: the SDXL LoKr loader keeps `mid_block` out (sc-2640), so a mid_block LoKr
///     factor would be skipped at load. Training to the vendored surface keeps train/inference in
///     lock-step. (Extending the LoKr inference surface to mid_block is a separate engine change.)
fn resolve_target_paths(unet: &UNet2DConditionModel, cfg: &TrainingConfig) -> Vec<String> {
    let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
        DEFAULT_TARGET_SUFFIXES
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        cfg.lora_target_modules.clone()
    };
    let surface = match cfg.network_type {
        NetworkType::Lora => unet.lora_target_paths_complete(),
        NetworkType::Lokr => unet.lora_target_paths(),
    };
    surface
        .into_iter()
        .filter(|path| {
            suffixes
                .iter()
                .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
}

/// One forward+backward over the trainable adapter factors: build the renormalized DDPM input at
/// integer table-index `t`, inject `params` (LoRA or LoKr), run the U-Net, regress the predicted
/// `eps` toward the unit `noise`, return `(loss, grads)`.
///
/// `dtype` is the training compute dtype (sc-4941): for bf16 the noisy latent / conditioning / pooled
/// inputs are cast to bf16 at entry (the U-Net weights were cast once in `train_impl`) and the LoRA
/// factors / LoKr delta are reconstructed at bf16 inside the traced install — so the whole U-Net
/// graph runs bf16 with no silent f32 re-promotion (a mixed-dtype matmul would promote the chain back
/// to f32, defeating the cast). The noise target, loss, and grads stay f32 (master-weights pattern).
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    unet: &mut UNet2DConditionModel,
    sampler: &EulerSampler,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    cond: &Array,
    pooled: &Array,
    time_ids: &Array,
    t: f32,
    noise: &Array,
    mae: bool,
    dtype: Dtype,
    checkpoint_targets: Option<Vec<String>>,
) -> Result<(f32, LoraParams)> {
    // Renormalized model input = `(x0 + σ(t)·noise)·rsqrt(σ(t)²+1)` — algebraically the diffusers
    // DDPM `noisy`; the epsilon target is the unit `noise`. Reusing the sampler's own `add_noise_with`
    // makes the training input bit-consistent with the inference convention.
    let noisy = sampler.add_noise_with(x0, noise, t)?.as_dtype(dtype)?;
    let target = noise.clone(); // f32 — the loss is computed in f32 (eps promotes on subtract)
    let (cond, pooled, time_ids) = (
        cond.as_dtype(dtype)?,
        pooled.as_dtype(dtype)?,
        time_ids.clone(),
    );
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    // Reconstruct the LoKr delta at the compute dtype so its residual matches the bf16 activation
    // stream; the SAVED factors stay f32 (the inference round-trip dtype) — `save` writes the raw
    // factor arrays, not this delta.
    let lokr_dtype = if dtype == Dtype::Float32 {
        LOKR_DTYPE
    } else {
        dtype
    };
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // Install ALL adapters: under block checkpointing the mid block + embedders train through
        // these on the (non-checkpointed) dense path, while each down/up block's adapters are replaced
        // inside its checkpoint segment by the explicit-input factors — so installing them here costs
        // nothing on the checkpointed path.
        adapter.install_as(unet, &p, alpha, rank, lora_dtype, lokr_dtype)?;
        let eps = match &checkpoint_targets {
            Some(tp) => unet
                .forward_block_checkpointed(&noisy, t, &cond, &pooled, &time_ids, tp, &p, alpha)
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => unet
                .forward(&noisy, t, &cond, &pooled, &time_ids)
                .map_err(|e| Exception::custom(e.to_string()))?,
        };
        let diff = subtract(&eps, &target)?;
        // MSE / MAE — `mean(None)` reduces to a 0-d scalar (grad requires a scalar cotangent).
        let loss = if mae {
            diff.abs()?.mean(None)?
        } else {
            diff.square()?.mean(None)?
        };
        Ok(vec![loss])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (val, grads) = vg(params.clone(), 0)?;
    Ok((val[0].item::<f32>(), grads))
}

/// Projected dense first-step peak memory, in GB, as a function of the latent pixel count
/// `p = (edge/8)²` (the SDXL VAE downscales /8; the U-Net working set is dominated by the conv-resnet
/// activations that scale with this spatial extent — unlike z-image, whose peak is the attention
/// seq² term). An empirical fit to peaks measured on the 128 GB target (`first_step_memory_sweep`,
/// rank 16 / 560 LoRA targets / batch 1), AFTER the CLIP encoders are freed (the train-loop working
/// set — what the guard models, since the guard projects the loop peak that follows caching).
/// The structure is `resident + linear·p + quad·p²`: the constant is the resident base (U-Net + VAE,
/// encoders freed), the linear term the per-pixel conv activations across the down/mid/up stack, the
/// small quadratic the attention seq² at the 64²/32² grids. bf16 roughly halves all three. Assumes
/// micro-batch 1 (the loop's actual shape); refit if the LoRA-target count or batch changes.
fn projected_dense_peak_gb(p: f64, bf16: bool) -> f64 {
    // Measured AFTER the CLIP encoders are freed (the train-loop working set): `first_step_memory_sweep`
    // on the 128 GB target, f32 512/768/1024/1280 → 15.0/22.3/36.4/60.4 GB; bf16 → 7.7/11.3/18.3/30.4
    // GB. The bf16 1024 peak (~18 GB) fits a 32 GB Mac; 1280 (~30 GB) is the headroom edge. p=(edge/8)².
    if bf16 {
        5.56 + 4.258e-4 * p + 2.157e-8 * p * p
    } else {
        10.78 + 8.582e-4 * p + 4.293e-8 * p * p
    }
}

/// Refuse a run whose dense first step would exceed this machine's memory budget, returning a
/// catchable, actionable error instead of risking an uncatchable SIGKILL (sc-4874/sc-4941). For SDXL
/// the dense first step fits unified memory at every production resolution (≤1280 → ≤63 GB f32 on a
/// 128 GB machine), so this guard only fires for very large edges / smaller machines — cheap insurance
/// that converts a would-be kill into an error recommending Gradient Checkpointing or a lower
/// resolution. Only consulted when gradient checkpointing is OFF. `edge` is the bucketed training edge.
fn preflight_memory_guard(edge: u32, bf16: bool) -> Result<()> {
    let latent_side = (edge as f64 / 8.0).ceil();
    let p = latent_side * latent_side;
    let projected = projected_dense_peak_gb(p, bf16);
    let budget_gb = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);
    let safe = budget_gb * 0.85;
    if projected > safe {
        return Err(format!(
            "sdxl trainer: a dense first training step at resolution {edge} needs ~{projected:.0} GB \
             (the forward working set materializes in one allocation), exceeding this machine's ~{safe:.0} GB \
             safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the OS could hard-kill the \
             worker (SIGKILL) at the first step with no recoverable error. Enable Gradient Checkpointing or \
             reduce the training resolution."
        )
        .into());
    }
    Ok(())
}

/// Sample a **uniform integer** DDPM timestep over the sigma-table indices `[1, max_time]` (the
/// vendored table is `concat([0], σ_1..σ_1000)`, so index `t` maps to diffusers `ᾱ[t-1]` — i.e. a
/// uniform draw here equals the torch trainer's `randint(0, num_train_timesteps)`). Deterministic in
/// `seed`. At an integer `t` the sampler's sigma interpolation is exact (`σ = σ_t`).
fn sample_timestep(sampler: &EulerSampler, seed: u64) -> Result<f32> {
    let k = random::key(seed)?;
    let max_t = sampler.max_time(); // 1000.0
    let u = random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k))?.item::<f32>();
    // floor(1 + u·max_t) ∈ [1, max_t] (u ∈ [0,1)); clamp the u→1 edge defensively.
    let t = (1.0 + u * max_t).floor().clamp(1.0, max_t);
    Ok(t)
}

#[cfg(test)]
mod preflight_tests {
    use super::projected_dense_peak_gb;

    /// The empirical fit must reproduce the measured (post-encoder-free) first-step peaks within a few
    /// GB and stay monotonic — it is the basis of the pre-flight OOM guard. p = (edge/8)²: edge
    /// 512→4096, 768→9216, 1024→16384, 1280→25600. Measured (`first_step_memory_sweep`, 128 GB):
    /// f32 15.0/22.3/36.4/60.4 GB; bf16 7.7/11.3/18.3/30.4 GB.
    #[test]
    fn projection_matches_measured_curve() {
        for (p, measured) in [
            (4096.0, 15.0),
            (9216.0, 22.3),
            (16384.0, 36.4),
            (25600.0, 60.4),
        ] {
            let proj = projected_dense_peak_gb(p, false);
            assert!(
                (proj - measured).abs() < 3.0,
                "f32 projection at p={p} = {proj:.1} GB, expected ≈{measured} GB"
            );
        }
        for (p, measured) in [
            (4096.0, 7.7),
            (9216.0, 11.3),
            (16384.0, 18.3),
            (25600.0, 30.4),
        ] {
            let proj = projected_dense_peak_gb(p, true);
            assert!(
                (proj - measured).abs() < 3.0,
                "bf16 projection at p={p} = {proj:.1} GB, expected ≈{measured} GB"
            );
        }
        // Monotonic increasing; bf16 strictly below f32.
        assert!(projected_dense_peak_gb(4096.0, false) < projected_dense_peak_gb(16384.0, false));
        assert!(projected_dense_peak_gb(16384.0, true) < projected_dense_peak_gb(16384.0, false));
        // sc-4941's 32 GB-Mac goal: bf16 1024 LoRA training fits a 32 GB box. Such a box reports an
        // MLX working-set limit of ~22 GB; the guard's budget is limit × 0.85 ≈ 18.7 GB. bf16 1024
        // (~18.3 GB) clears it; f32 1024 (~36 GB) does not — so an f32 run on a 32 GB box is correctly
        // steered to bf16 / lower resolution rather than SIGKILLed.
        assert!(projected_dense_peak_gb(16384.0, true) < 18.7); // bf16 1024 fits a 32 GB box
        assert!(projected_dense_peak_gb(16384.0, false) > 18.7); // f32 1024 does not
    }
}

/// Decode an image file (PNG/JPEG) into the core RGB8 [`Image`](mlx_gen::media::Image).
fn decode_image(path: &Path) -> Result<mlx_gen::media::Image> {
    let dynimg = image::open(path)
        .map_err(|e| mlx_gen::Error::Msg(format!("decode image {}: {e}", path.display())))?;
    let rgb = dynimg.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(mlx_gen::media::Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

// ===========================================================================================
// sc-4941 (sibling of z-image sc-4874) — first-step peak-memory characterization for the SDXL
// U-Net LoRA trainer. The story's explicit mandate is "measure before assuming z-image magnitude":
// the SDXL U-Net's attention runs at SMALLER latent grids (64²/32², not z-image's unified 64²×30
// blocks) and its conv resnets have no seq² term, so the first-step working set must be measured,
// not extrapolated from z-image's 135 GB-at-1024 curve. This harness drives the exact inner step
// (`compute_loss_grads` + the backward grad `eval` the real loop forces) at swept resolution with
// MLX peak probes around it.
//
//   cargo test -p mlx-gen-sdxl --release --lib first_step -- --ignored --nocapture
// ===========================================================================================
#[cfg(test)]
mod first_step_repro {
    use super::*;
    use mlx_gen::media::Image;
    use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};
    use std::path::PathBuf;

    /// Resolve an SDXL diffusers snapshot (env `SDXL_SNAPSHOT`, else the HF cache).
    fn snapshot() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
            return Some(PathBuf::from(p));
        }
        let home = std::env::var("HOME").ok()?;
        let snaps = PathBuf::from(home).join(
            ".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots",
        );
        std::fs::read_dir(&snaps)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.is_dir() && p.join("unet").is_dir())
    }

    /// A solid-colour `edge`×`edge` RGB source image (latent magnitude is irrelevant; the graph
    /// size — driven by resolution — is the variable under test).
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

    /// Run one first training step at `edge` in `dtype` (caller casts the U-Net to match) and report
    /// the peak GPU memory across forward+backward (forces the grad eval — the real step-1 kill point).
    #[allow(clippy::too_many_arguments)]
    fn one_step(
        trainer: &mut SdxlTrainer,
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
        let time_ids = text_time_ids(1);
        eval([&x0, &noise]).unwrap();

        clear_cache();
        reset_peak_memory();
        let before = get_active_memory();
        let t0 = std::time::Instant::now();
        let (loss, grads) = compute_loss_grads(
            &mut trainer.unet,
            &trainer.sampler,
            params,
            adapter,
            16.0,
            16.0,
            &x0,
            cond,
            pooled,
            &time_ids,
            500.0,
            &noise,
            false,
            dtype,
            checkpoint_targets,
        )?;
        eval(grads.values())?; // force the backward (true working set)
        let secs = t0.elapsed().as_secs_f64();
        let peak = get_peak_memory();
        eprintln!(
            "[sc-4941]   edge {edge:>4} {tag}  loss {loss:.5}  active-before {:.2} GB  peak {:.2} GB  step {secs:.2}s",
            gb(before),
            gb(peak)
        );
        Ok((loss, gb(peak)))
    }

    /// Max relative grad diff between two param maps.
    fn max_rel_diff(ga: &LoraParams, gb_: &LoraParams) -> f32 {
        let mut max_rel = 0f32;
        for (k, a) in ga {
            let b = gb_.get(k).expect("same keys");
            let num = a.subtract(b).unwrap().abs().unwrap().max(None).unwrap();
            let den = a.abs().unwrap().max(None).unwrap().item::<f32>().max(1e-6);
            max_rel = max_rel.max(num.item::<f32>() / den);
        }
        max_rel
    }

    fn build_trainer_and_adapter() -> (SdxlTrainer, TrainAdapter, LoraParams, Array, Array) {
        let root = snapshot().expect("SDXL snapshot (HF cache or SDXL_SNAPSHOT)");
        let mut trainer = SdxlTrainer {
            descriptor: trainer_descriptor(),
            tokenizer: crate::loader::load_tokenizer(&root).unwrap(),
            te1: Some(crate::loader::load_text_encoder_1(&root).unwrap()),
            te2: Some(crate::loader::load_text_encoder_2(&root).unwrap()),
            vae: crate::loader::load_vae(&root).unwrap(),
            unet: crate::loader::load_unet(&root).unwrap(),
            sampler: EulerSampler::new(&DiffusionConfig::sdxl_base(), true).unwrap(),
        };
        let cfg = TrainingConfig {
            rank: 16,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&trainer.unet, &cfg);
        let (targets, params) =
            build_lora_targets(&mut trainer.unet, &target_paths, 16, 7).unwrap();
        let (cond, pooled) = trainer.encode_prompt("a solid colour swatch").unwrap();
        eval([&cond, &pooled]).unwrap();
        // Drop the CLIP encoders exactly as `train_impl` does after caching, so the measured peaks
        // reflect the post-free training working set.
        trainer.te1 = None;
        trainer.te2 = None;
        mlx_rs::memory::clear_cache();
        eprintln!(
            "[sc-4941] loaded SDXL trainer (encoders freed); {} LoRA targets; cond {:?} pooled {:?}",
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

    /// Sweep resolution tiny → production, printing the dense first-step peak curve in f32 then bf16
    /// (cond/pooled cast to match). These measured points are the basis of the `projected_dense_peak_gb`
    /// guard fit — refit the constants if this prints materially different numbers.
    #[test]
    #[ignore = "needs real SDXL weights; run as its own process"]
    fn first_step_memory_sweep() {
        let (mut trainer, adapter, params, cond, pooled) = build_trainer_and_adapter();
        eprintln!("[sc-4941] SDXL dense f32 first-step sweep:");
        for edge in [256u32, 512, 768, 1024, 1280] {
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
            .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
        }
        eprintln!("[sc-4941] casting U-Net to bf16…");
        trainer.unet.cast_weights(Dtype::Bfloat16).unwrap();
        let cond_b = cond.as_dtype(Dtype::Bfloat16).unwrap();
        let pooled_b = pooled.as_dtype(Dtype::Bfloat16).unwrap();
        let tp: Vec<String> = match &adapter {
            TrainAdapter::Lora { targets } => targets.iter().map(|t| t.path.clone()).collect(),
            _ => Vec::new(),
        };
        clear_cache();
        eprintln!("[sc-4941] SDXL dense bf16 first-step sweep:");
        for edge in [256u32, 512, 768, 1024, 1280] {
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
            .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
        }
        eprintln!("[sc-4941] SDXL bf16 BLOCK-CHECKPOINTED first-step sweep (1024/1280 — the 32 GB lever):");
        for edge in [1024u32, 1280, 1536] {
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
            .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
        }
        eprintln!("[sc-4941] sweep complete");
    }

    /// sc-4941 — always-bit-identical: the SDPA-segment checkpoint (opt-in) must not change grads vs
    /// the retained backward. Same decomposed attention, recomputed instead of retained.
    #[test]
    #[ignore = "needs real SDXL weights; run as its own process"]
    fn attn_ckpt_grads_match_retained() {
        let (mut trainer, adapter, params, cond, pooled) = build_trainer_and_adapter();
        let edge = 256u32;
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
        let time_ids = text_time_ids(1);
        eval([&x0, &noise]).unwrap();
        let grads_of = |t: &mut SdxlTrainer, on: bool| -> LoraParams {
            t.unet.set_sdpa_checkpoint(on);
            let (_l, g) = compute_loss_grads(
                &mut t.unet,
                &t.sampler,
                &params,
                &adapter,
                16.0,
                16.0,
                &x0,
                &cond,
                &pooled,
                &time_ids,
                500.0,
                &noise,
                false,
                Dtype::Float32,
                None,
            )
            .unwrap();
            eval(g.values()).unwrap();
            g
        };
        let g_retained = grads_of(&mut trainer, false);
        let g_ckpt = grads_of(&mut trainer, true);
        let max_rel = max_rel_diff(&g_retained, &g_ckpt);
        eprintln!("[sc-4941] attn-ckpt-vs-retained grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-5,
            "attention-segment checkpointing must not change grads: max rel {max_rel:.2e}"
        );
    }

    /// sc-4941 — block (gradient) checkpointing must not change the math: the per-block checkpointed
    /// forward+grads must match the dense path within fp tolerance (it reuses the same install + block
    /// forward, recompute-only). This is the correctness gate for the `gradient_checkpointing` lever.
    #[test]
    #[ignore = "needs real SDXL weights; run as its own process"]
    fn block_ckpt_grads_match_dense() {
        let (mut trainer, adapter, params, cond, pooled) = build_trainer_and_adapter();
        let edge = 256u32; // math is resolution-agnostic; small enough that the dense path is cheap
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
        let time_ids = text_time_ids(1);
        eval([&x0, &noise]).unwrap();
        let tp: Vec<String> = match &adapter {
            TrainAdapter::Lora { targets } => targets.iter().map(|t| t.path.clone()).collect(),
            _ => unreachable!("LoRA adapter in this harness"),
        };
        let grads_of = |t: &mut SdxlTrainer, ck: Option<Vec<String>>| -> LoraParams {
            let (_l, g) = compute_loss_grads(
                &mut t.unet,
                &t.sampler,
                &params,
                &adapter,
                16.0,
                16.0,
                &x0,
                &cond,
                &pooled,
                &time_ids,
                500.0,
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
        let max_rel = max_rel_diff(&g_dense, &g_ckpt);
        eprintln!("[sc-4941] block-ckpt-vs-dense grad max relative diff: {max_rel:.2e}");
        // Recompute-vs-retained fp noise (not a structural diff — a real checkpointing bug, like the
        // duplicate-output VJP corruption this gate caught, shows up at ~1e0). The bound is a few e-3
        // because the conv-heavy recompute reorders fp accumulation; a genuine mismatch is orders of
        // magnitude larger.
        assert!(
            max_rel < 5e-3,
            "block checkpointing must match the dense grads: max rel {max_rel:.2e}"
        );
    }

    /// sc-4941 — bf16 is mixed precision, NOT bit parity: assert the bf16 grads point the same way as
    /// f32 (global cosine + large-norm cosine) and the bf16 working set is genuinely smaller (a silent
    /// f32 re-promotion in the forward would pass the cosine check while saving nothing — the memory
    /// ratio IS the dtype assertion). Runs f32 first (the cast is destructive), then casts to bf16.
    #[test]
    #[ignore = "needs real SDXL weights; run as its own process"]
    fn bf16_grads_direction_and_memory_vs_f32() {
        let (mut trainer, adapter, params, cond, pooled) = build_trainer_and_adapter();

        // Grad reference at 256 in f32.
        let edge = 256u32;
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
        let time_ids = text_time_ids(1);
        eval([&x0, &noise]).unwrap();
        let grads_of =
            |t: &mut SdxlTrainer, c: &Array, p: &Array, dt: Dtype| -> (f32, LoraParams) {
                let (l, g) = compute_loss_grads(
                    &mut t.unet,
                    &t.sampler,
                    &params,
                    &adapter,
                    16.0,
                    16.0,
                    &x0,
                    c,
                    p,
                    &time_ids,
                    500.0,
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

        // Memory A/B at 768 in f32 (big enough that activations dominate).
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

        let mut per: Vec<(String, f32, f32, f32)> = Vec::new(); // (key, cos, na, nb)
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
        eprintln!("[sc-4941] bf16-vs-f32 grads: global cosine {global_cos:.5}; worst per-param:");
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
        assert!(
            global_cos > 0.995,
            "bf16 global grad must point the same way as f32: {global_cos:.5}"
        );
        assert!(
            min_large > 0.95,
            "a large-norm param's bf16 grad diverged from f32 (systematic bug): {min_large:.4}"
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
            bf16_peak < 0.70 * f32_peak,
            "bf16 must materially shrink the working set: f32 {f32_peak:.2} GB vs bf16 {bf16_peak:.2} GB"
        );
    }
}
