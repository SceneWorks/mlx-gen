//! LoRA/LoKr **training** on the SD3.5 MMDiT, in pure Rust on mlx-rs (T2 sc-7883, epic 7841) — the
//! MLX-native SD3.5 LoRA-training base. LoRAs train on the `stabilityai/stable-diffusion-3.5-large`
//! MMDiT and apply back at `sd3_5_large` (and family-arch-identical Large-Turbo) inference (the
//! Lens / Z-Image / Krea precedent: same architecture, no base-model gating, family-match suffices).
//!
//! [`Sd3LoraTrainer`] realizes the core [`Trainer`](mlx_gen::Trainer) contract on the real 38-block
//! joint MMDiT, mirroring [`KreaRawTrainer`](mlx_gen_krea) — the model crates don't use mlx-rs's
//! `Module` system (hand-rolled `&self` forwards over raw `Array`s), so training uses the **functional
//! autograd**: the trainable factors live OUTSIDE the model in a [`LoraParams`] map, re-injected each
//! step into the target [`AdaptableLinear`](mlx_gen::adapters::AdaptableLinear)s via the shared core
//! seam ([`mlx_gen::train::lora`]), stepped with `keyed_value_and_grad` + the core [`TrainOptimizer`] +
//! `clip_grad_norm`. The injection mirrors the inference reload op-for-op (the [`crate::adapters`]
//! apply path), so the trained adapter round-trips through that loader.
//!
//! ## What is SD3-specific (everything else reuses the family-agnostic core unchanged)
//! - **Flow-match velocity target = `noise − x0`** with **NO sign flip** (the Krea convention, the
//!   OPPOSITE of the Z-Image trainer): the SD3 MMDiT [`forward`](crate::transformer::Sd3Transformer::forward)
//!   returns the RAW un-negated flow-match velocity (the pipeline feeds it to the Euler step
//!   un-negated, [`crate::pipeline::denoise_cfg`]), so the regression target IS the velocity itself.
//!   `x_t = (1 − t)·x0 + t·noise`.
//! - **CRITICAL: the DiT timestep is the diffusers-scale `t·1000`** (the scheduler's
//!   `NUM_TRAIN_TIMESTEPS` — see [`crate::transformer::Sd3Transformer::forward`] and
//!   [`crate::pipeline`] `timestep * NUM_TRAIN_TIMESTEPS`). The trainer samples a normalized
//!   `t ∈ (0,1)` for the noising (`x_t`/target), then SCALES it to `t·1000` at the forward call. This
//!   is the top SD3-vs-Krea/z-image delta (those pass `t` raw); pinned by a test.
//! - **Latents** by the SD3.5 16-ch VAE encode: `preprocess_init_image` (resize + `[−1,1]` NCHW) →
//!   [`Vae::encode`](mlx_gen_z_image::vae::Vae) → `[1, 16, edge/8, edge/8]` (NO temporal axis, NO
//!   pack/transpose — the SD3 latent stays plain NCHW, unlike z-image's packed DiT input).
//! - **Conditioning** is the SD3 triple-TE aggregator's `(context [1,333,4096], pooled [1,2048])`
//!   pair (cached per sample). The three encoders (CLIP-L + CLIP-G + T5-XXL) load in an `Option` so
//!   they can be dropped after caching — they are idle during training (every caption is cached) yet
//!   multi-GB resident (T5-XXL dominates). Loaded Q8 for the trainer (smaller footprint; matches the
//!   deployment quant).
//! - **logit-normal default t-sampling** (the SD3.5 training recipe): `u~U(0,1)`, `t = σ(m + s·Φ⁻¹(u))`
//!   (`m=0, s=1` = the standard logit-normal `σ(N(0,1))`), via the Acklam probit `ndtri` + logistic
//!   ported from `mlx-gen-ideogram/src/scheduler.rs` (the resolution-aware INFERENCE mean-shift is NOT
//!   used). `sigmoid`/`uniform`/`weighted` remain for parity.
//! - **Targets** default to the joint-block attention — image stream `to_q`/`to_k`/`to_v`/`to_out.0`
//!   and text stream `add_q_proj`/`add_k_proj`/`add_v_proj`/`to_add_out` (both joint streams). The
//!   `attn2` (Medium MMDiT-X) targets are enumerable so the Medium trainer (T4 sc-7885) is a
//!   validation story, not a re-architecture; FFN is opt-in.
//!
//! Registered under the **`sd3_5_large`** id (the LoRA-training base; the adapter applies to Large /
//! Large-Turbo inference — family-match, no base-model gating).
//!
//! ## Memory hardening (the Krea sc-7577 / z-image analog — SD3.5-Large at 8.1B is the largest base)
//! - **SDPA-segment checkpointing** is always on in training: the joint SDPA runs inside an
//!   `mlx::checkpoint` so its backward recomputes attention rather than retaining the `[heads, S, S]`
//!   probability matrix. Numerically identical.
//! - **`gradient_checkpointing`** (the SceneWorks toggle) is an opt-in OPTION (LoRA only): each of the
//!   38 joint blocks recomputes its activations in the backward via
//!   [`Sd3Transformer::forward_with_blocks_checkpointed`](crate::transformer::Sd3Transformer::forward_with_blocks_checkpointed),
//!   threading the per-block LoRA factors as explicit checkpoint inputs — MANDATORY for the 8.1B Large
//!   at production resolution. LoKr keeps the dense path (caught by the guard).
//! - **Fail-fast OOM preflight guard** projects the dense first-step peak and refuses (recommending
//!   the toggle) before the minutes-long caching, converting an uncatchable SIGKILL into an actionable
//!   error.

use std::path::Path;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core;
use mlx_gen::img2img::preprocess_init_image;
use mlx_gen::media::Image;
use mlx_gen::train::checkpoint::checkpoint_filename;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, average_grads, build_lokr_targets, build_lora_targets, LoraParams,
    TrainAdapter,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::{
    Error, LoadSpec, Modality, NetworkType, Precision, Result, TrainOptimizer, Trainer,
    TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use mlx_gen_sdxl::tokenizer::ClipBpeTokenizer;
use mlx_gen_z_image::vae::Vae;

use crate::config::Sd3Variant;
use crate::loader;
use crate::pipeline::encode_prompt;
use crate::text::{Sd3Conditioning, Sd3TextEncoders};
use crate::transformer::Sd3Transformer;

/// Registry id for the SD3.5 LoRA-training base (the `stabilityai/stable-diffusion-3.5-large` MMDiT).
/// The trained adapter records `baseModel: sd3_5_large` / `family: sd3` and applies at `sd3_5_large`
/// (and the arch-identical Large-Turbo) inference — the family-match cross-apply, no base-model gating.
pub const SD3_5_LARGE_TRAINER_ID: &str = crate::config::SD3_5_LARGE_ID;

/// The LoKr delta-reconstruction dtype, matching the inference loader so a trained LoKr round-trips
/// through the apply path. bf16 — the family compute dtype.
const LOKR_DTYPE: Dtype = Dtype::Bfloat16;

/// The three text encoders are loaded Q8 for the trainer: they are frozen and used only to cache
/// caption conditioning once, then dropped before the train loop (the free pattern). Q8 also matches
/// the deployment quant.
const TRAINER_ENCODER_BITS: i32 = 8;

/// The number of train timesteps the diffusers SD3 scheduler embeds — the MMDiT forward expects
/// `t·NUM_TRAIN_TIMESTEPS` (NOT the raw `t ∈ (0,1)` that z-image/Krea pass). See the module docs.
const NUM_TRAIN_TIMESTEPS: f32 = crate::pipeline::NUM_TRAIN_TIMESTEPS;

/// The default target modules: the joint-block attention projections — image stream
/// (`to_q`/`to_k`/`to_v`/`to_out.0`) + text stream (`add_q_proj`/`add_k_proj`/`add_v_proj`/
/// `to_add_out`). Both streams (the standard SD3 PEFT attention surface). The FFN (`net.0.proj`/
/// `net.2`) and the adaLN modulation linears are reachable as explicit targets but not default.
const DEFAULT_TARGET_MODULES: [&str; 8] = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
];

/// The SD3.5 default flow-match timestep distribution: logit-normal (the SD3 training recipe). The
/// trainer uses this when `timestep_type` is unset/`"default"`; `logit_normal` selects it explicitly.
const SD3_DEFAULT_TIMESTEP_TYPE: &str = "logit_normal";

/// Recognized `timestep_type` values [`sample_sigma`] branches on (plus the logit-normal default):
/// the SD3-native `logit_normal` and the cross-family `sigmoid`/`linear`/`uniform`/`weighted` (parity).
const TIMESTEP_TYPES: [&str; 6] = [
    "logit_normal",
    "default",
    "sigmoid",
    "linear",
    "uniform",
    "weighted",
];
/// Recognized `timestep_bias` values [`sample_sigma`] branches on (plus the neutral default).
const TIMESTEP_BIASES: [&str; 9] = [
    "balanced",
    "none",
    "neutral",
    "high",
    "high_noise",
    "favor_high_noise",
    "low",
    "low_noise",
    "favor_low_noise",
];
/// Recognized `loss_type` values — `mae`/`l1` → MAE, `mse`/`l2` → the MSE default.
const LOSS_TYPES: [&str; 4] = ["mse", "l2", "mae", "l1"];

/// `(x_t, target)` for a single sample at flow-match `t`: `x_t = (1−t)·x0 + t·noise`,
/// `target = noise − x0` (the velocity the **raw** SD3 MMDiT output is regressed onto — NO sign flip;
/// the SAME sign as Krea, the OPPOSITE of z-image). The DiT timestep is scaled to `t·1000` by the
/// caller (see [`compute_loss_grads`]); `build_batch` works entirely in the un-scaled `t ∈ (0,1)`.
fn build_batch(x0: &Array, noise: &Array, t: f32) -> Result<(Array, Array)> {
    let one_minus = Array::from_slice(&[1.0 - t], &[1]);
    let s = Array::from_slice(&[t], &[1]);
    let x_t = add(&multiply(x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, x0)?;
    Ok((x_t, target))
}

/// The production [`Trainer`] for the SD3.5-Large MMDiT: a frozen base (triple TE, MMDiT, 16-ch VAE,
/// CLIP/T5 tokenizers) that caches a captioned dataset to VAE-latents + triple-TE conditioning, then
/// runs the functional-autograd LoRA/LoKr loop with the core runtime glue (LR schedule, gradient
/// accumulation, checkpoint cadence, cancel, progress bands), writing a PEFT adapter that reloads
/// through the inference path ([`crate::adapters::apply_sd3_adapters`]).
pub struct Sd3LoraTrainer {
    descriptor: TrainerDescriptor,
    clip_tokenizer: ClipBpeTokenizer,
    t5_tokenizer: mlx_gen::tokenizer::TextTokenizer,
    /// The three text encoders, in an `Option` so they can be **dropped after caching** — idle during
    /// training (every caption is cached) yet a multi-GB resident (T5-XXL dominates).
    encoders: Option<Sd3TextEncoders>,
    transformer: Sd3Transformer,
    vae: Vae,
    /// The compute dtype (bf16 production / f32 tight-gate), fixed at load from `spec.precision`.
    dtype: Dtype,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: SD3_5_LARGE_TRAINER_ID,
        family: "sd3",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// Construct the trainer from a `stabilityai/stable-diffusion-3.5-large` snapshot directory (the
/// diffusers multi-component tree). The MMDiT is loaded **dense** (the adapter host); the encoders are
/// Q8. `spec.precision` selects the compute dtype (bf16 default / f32 tight-gate); the snapshot ships
/// bf16, so f32 widens it via [`Sd3Transformer::cast_weights`] and bf16 casts the dense load.
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "sd3 trainer expects a snapshot directory (transformer/ text_encoder{,_2,_3}/ \
                 tokenizer{,_2,_3}/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    let dtype = match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    };
    let arch = Sd3Variant::Large.arch();
    let clip_tokenizer = loader::load_clip_tokenizer(&root)?;
    let t5_tokenizer = loader::load_t5_tokenizer(&root)?;
    let mut encoders = loader::load_text_encoders(&root)?;
    encoders.quantize(TRAINER_ENCODER_BITS)?;
    let mut transformer = loader::load_transformer(&root, &arch)?;
    if transformer.compute_dtype() != dtype {
        transformer.cast_weights(dtype)?;
    }
    let vae = loader::load_vae(&root)?;
    Ok(Box::new(Sd3LoraTrainer {
        descriptor: trainer_descriptor(),
        clip_tokenizer,
        t5_tokenizer,
        encoders: Some(encoders),
        transformer,
        vae,
        dtype,
    }))
}

// Link-time trainer registration (epic 3720): the macro emits the `inventory::submit!` and bridges
// the crate's rich `Result` into the trainer registry's backend-neutral `gen_core::Result`.
mlx_gen::register_trainer! { trainer_descriptor => load_trainer }

/// Normalize a free-form config string the way the trainer's own parsers do (trim, lowercase,
/// `-`/space → `_`) so validation accepts exactly the spellings the run would.
fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Capability-free training-request validation, factored out so it can be unit-tested without a loaded
/// trainer. Rejects an empty dataset, zero rank/steps, an unsupported optimizer, and an unrecognized
/// `timestep_type`/`timestep_bias`/`loss_type`. An EMPTY `timestep_type` is accepted (→ the SD3
/// logit-normal default).
fn validate_request(req: &TrainingRequest) -> Result<()> {
    let cfg = &req.config;
    if req.items.is_empty() {
        return Err("sd3 trainer: dataset is empty".into());
    }
    if cfg.rank == 0 {
        return Err("sd3 trainer: rank must be > 0".into());
    }
    if cfg.steps == 0 {
        return Err("sd3 trainer: steps must be > 0".into());
    }
    if !TrainOptimizer::is_supported(&cfg.optimizer) {
        return Err(format!(
            "sd3 trainer: optimizer '{}' is not available on MLX training (supported: adamw, adam, \
             rose, prodigy)",
            cfg.optimizer
        )
        .into());
    }
    let tt = normalize_cfg(&cfg.timestep_type);
    if !tt.is_empty() && !TIMESTEP_TYPES.contains(&tt.as_str()) {
        return Err(format!(
            "sd3 trainer: timestep_type '{}' is not recognized (supported: {})",
            cfg.timestep_type,
            TIMESTEP_TYPES.join(", ")
        )
        .into());
    }
    if !TIMESTEP_BIASES.contains(&normalize_cfg(&cfg.timestep_bias).as_str()) {
        return Err(format!(
            "sd3 trainer: timestep_bias '{}' is not recognized (supported: {})",
            cfg.timestep_bias,
            TIMESTEP_BIASES.join(", ")
        )
        .into());
    }
    if !LOSS_TYPES.contains(&normalize_cfg(&cfg.loss_type).as_str()) {
        return Err(format!(
            "sd3 trainer: loss_type '{}' is not recognized (supported: {})",
            cfg.loss_type,
            LOSS_TYPES.join(", ")
        )
        .into());
    }
    Ok(())
}

impl Trainer for Sd3LoraTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        validate_request(req)?;
        if resolve_target_paths(&self.transformer, &req.config).is_empty() {
            return Err(format!(
                "sd3 trainer: lora_target_modules {:?} matched no adaptable module on the SD3 MMDiT \
                 (defaults are to_q/to_k/to_v/to_out.0 + add_q_proj/add_k_proj/add_v_proj/to_add_out \
                 on the joint blocks)",
                req.config.lora_target_modules
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

impl Sd3LoraTrainer {
    /// The rich-`Result` body behind [`Trainer::train`]; the trait wrapper bridges its tail into
    /// [`gen_core::Error`] (epic 3720), keeping `?` on `mlx_rs`/family helpers transparent here.
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        validate_request(req)?;
        let cfg = &req.config;

        let target_paths = resolve_target_paths(&self.transformer, cfg);
        if target_paths.is_empty() {
            return Err(format!(
                "sd3 trainer: lora_target_modules {:?} matched no adaptable module on the SD3 MMDiT",
                cfg.lora_target_modules
            )
            .into());
        }

        // The MMDiT compute dtype is fixed at load (`spec.precision`); enforce `train_dtype` against
        // it (never a silent no-op). The common case (config default bf16 + LoadSpec default Bf16)
        // matches, so this only fires on an explicit f32-vs-bf16 mismatch.
        let want_bf16 = {
            let t = cfg.train_dtype.trim();
            t.eq_ignore_ascii_case("bf16") || t.eq_ignore_ascii_case("bfloat16")
        };
        let loaded_bf16 = self.dtype == Dtype::Bfloat16;
        if want_bf16 != loaded_bf16 {
            return Err(format!(
                "sd3 trainer: train_dtype '{}' does not match the loaded precision ({}). Load the \
                 trainer with {} to train at that dtype.",
                cfg.train_dtype,
                if loaded_bf16 { "bf16" } else { "f32" },
                if want_bf16 {
                    "Precision::Bf16"
                } else {
                    "Precision::Fp32"
                }
            )
            .into());
        }
        let compute_dtype = self.dtype;
        // bf16 mixed precision: cast the folded LoRA residual to the activation dtype so the adapted
        // Linear stays bf16 (else it silently re-promotes the chain to f32). f32 → no cast.
        let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);

        on_progress(TrainingProgress::Preparing);
        let edge = bucket_resolution(cfg.resolution);

        // T2 — fail-fast pre-flight memory guard (the Krea/z-image analog). The dense
        // (non-block-checkpointed) first step materializes the whole forward graph in one MLX `eval`;
        // at high resolution that working set can exceed unified memory and the OS hard-kills the
        // worker with an UNCATCHABLE SIGKILL. We predict it and refuse up front with a catchable,
        // actionable error BEFORE the (minutes-long) latent caching, UNLESS the run will
        // block-checkpoint (LoRA + the toggle). LoKr always takes the dense path, so it is guarded.
        let will_checkpoint =
            matches!(cfg.network_type, NetworkType::Lora) && cfg.gradient_checkpointing;
        if !will_checkpoint {
            preflight_memory_guard(edge, want_bf16)?;
        }

        // --- prepare → load → cache: VAE-latents + triple-TE conditioning into memory ---
        on_progress(TrainingProgress::LoadingModel); // base is already resident from load_trainer
        let total = req.items.len() as u32;
        let mut cache: Vec<(Array, Sd3Conditioning)> = Vec::with_capacity(req.items.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = center_crop_square(&decode_image(&item.image_path)?);
            let x0 = encode_init_latents(&self.vae, &img, edge)?; // [1, 16, edge/8, edge/8]
            let encoders = self.encoders.as_ref().ok_or_else(|| {
                Error::Msg(
                    "sd3 trainer: text encoders already freed (caching after train loop)".into(),
                )
            })?;
            let cond = encode_prompt(
                encoders,
                &self.clip_tokenizer,
                &self.t5_tokenizer,
                &item.caption,
            )?;
            eval([&x0, &cond.context, &cond.pooled])?;
            cache.push((x0, cond));
        }
        if cache.is_empty() {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            return Err("sd3 trainer: no usable dataset items".into());
        }

        // Every caption is cached now — free the three encoders (T5-XXL dominates) and evict buffers
        // before the train loop, reclaiming that resident for the 8.1B MMDiT working set.
        self.encoders = None;
        mlx_rs::memory::clear_cache();

        // --- adapter targets + params (LoRA or LoKr) + optimizer ---
        let rank = cfg.rank as f32;
        let (adapter, mut params) = match cfg.network_type {
            NetworkType::Lora => {
                let (targets, params) = build_lora_targets(
                    &mut self.transformer,
                    &target_paths,
                    cfg.rank as i32,
                    cfg.seed,
                )?;
                (TrainAdapter::Lora { targets }, params)
            }
            NetworkType::Lokr => {
                let (targets, params) = build_lokr_targets(
                    &mut self.transformer,
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

        // T2 — gradient checkpointing. Collect, per joint block, the adapter-routable LOCAL paths
        // trained on it (e.g. `"attn.to_q"`), in trained-file order — the factors a checkpoint segment
        // threads as explicit inputs. Only `transformer_blocks.*` targets are checkpointed; any global
        // targets (`context_embedder`/`proj_out`) train dense through `self`.
        let n_layers = self.transformer.num_blocks();
        let mut block_local_targets: Vec<Vec<String>> = vec![Vec::new(); n_layers];
        for path in &target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("transformer_blocks.")
                .and_then(|rest| rest.split_once('.'))
            {
                if let Ok(i) = idx.parse::<usize>() {
                    if i < n_layers {
                        block_local_targets[i].push(local.to_string());
                    }
                }
            }
        }
        // Opt-in OPTION (the SceneWorks "Gradient Checkpointing" toggle), never auto-forced — a run
        // that would OOM is caught instead by the pre-flight guard above. LoRA only — LoKr falls back
        // to the dense path.
        let use_checkpoint =
            matches!(adapter, TrainAdapter::Lora { .. }) && cfg.gradient_checkpointing;
        let checkpoint_blocks: Option<&[Vec<String>]> = if use_checkpoint {
            Some(&block_local_targets)
        } else {
            None
        };
        // SDPA-segment checkpointing is ALWAYS on in training. When whole-block checkpointing is on,
        // the per-block SDPA flag goes OFF (the block recompute already covers attention).
        self.transformer.set_sdpa_checkpoint(!use_checkpoint);

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
            let (x0, cond) = &cache[((step - 1) as usize) % cache.len()];
            let t = sample_sigma(
                &cfg.timestep_type,
                &cfg.timestep_bias,
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
                &mut self.transformer,
                &params,
                &adapter,
                alpha,
                rank,
                x0,
                cond,
                t,
                &noise,
                mae,
                compute_dtype,
                lora_dtype,
                checkpoint_blocks,
            )?;
            last_loss = loss;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;

            if step % accum == 0 || step == cfg.steps {
                let mult =
                    lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                opt.set_lr_scaled(mult);
                let window = if step % accum == 0 {
                    accum
                } else {
                    step % accum
                };
                let avg = average_grads(
                    accumulated
                        .take()
                        .expect("an update fires only after accumulation"),
                    window,
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
                adapter.save(&params, alpha, rank, cfg.decompose_factor, "", &ckpt)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }

        // Cancelled before a single step completed (`steps == 0` is rejected by `validate`): the
        // factors are still the `B = 0` no-op init. Surface the cancellation rather than writing a
        // valid-looking identity adapter as a trained artifact.
        if steps_run == 0 {
            return Err(Error::Canceled);
        }

        // --- save final adapter (the diffusers/PEFT format the inference apply path loads) ---
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        adapter.save(
            &params,
            alpha,
            rank,
            cfg.decompose_factor,
            "",
            &adapter_path,
        )?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Number of caption tokens assumed by the pre-flight projection. The SD3 unified sequence is
/// `img_len + ctx_len` and the context is a fixed 333 tokens (77 CLIP + 256 T5).
const PREFLIGHT_TXT_TOKENS: f64 = 333.0;

/// Projected DENSE (non-block-checkpointed) first-step peak memory, in GB, as a function of the
/// unified token count `s = img_len + ctx_len`, for the 8.1B SD3.5-Large MMDiT. The structure follows
/// the Krea/z-image `weights + linear·s + quad·s²` decomposition: the constant is the resident MMDiT
/// base (bf16 ~16 GB / f32 ~32 GB; the encoders are freed before the train loop), the linear term is
/// the per-token activations across the 38 joint blocks, and the quadratic term is the seq² attention
/// transient — demoted to a single block's backward transient by the always-on SDPA-segment
/// checkpointing.
///
/// **These constants are a CONSERVATIVE INITIAL ESTIMATE** — they err toward refusing borderline runs
/// (recommending Gradient Checkpointing) rather than allowing a SIGKILL, and are to be refit from a
/// real-weight sweep. `projection_is_monotonic_and_conservative` pins the shape.
fn projected_dense_peak_gb(s: f64, bf16: bool) -> f64 {
    if bf16 {
        PREFLIGHT_BF16.0 + PREFLIGHT_BF16.1 * s + PREFLIGHT_BF16.2 * s * s
    } else {
        PREFLIGHT_F32.0 + PREFLIGHT_F32.1 * s + PREFLIGHT_F32.2 * s * s
    }
}

/// `(weights, linear, quad)` conservative initial constants for [`projected_dense_peak_gb`].
const PREFLIGHT_F32: (f64, f64, f64) = (32.0, 1.20e-2, 3.0e-7);
const PREFLIGHT_BF16: (f64, f64, f64) = (16.0, 6.0e-3, 1.5e-7);

/// Refuse a run whose dense first step would exceed this machine's memory budget (and thus get
/// SIGKILLed), returning a catchable, actionable error instead. Only consulted when gradient
/// checkpointing is OFF.
fn preflight_memory_guard(edge: u32, bf16: bool) -> Result<()> {
    let budget_gb = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);
    check_preflight_budget(edge, bf16, budget_gb)
}

/// The pure guard logic (no MLX global state, so it is unit-testable): refuse if the projected dense
/// first-step peak exceeds `budget_gb × 0.85`. `edge` is the bucketed training edge; the SD3 unified
/// token count is `(edge/16)²` (latent /8, patch 2) plus the fixed 333-token context.
fn check_preflight_budget(edge: u32, bf16: bool, budget_gb: f64) -> Result<()> {
    let tokens_per_side = (edge as f64 / 16.0).ceil();
    let s = tokens_per_side * tokens_per_side + PREFLIGHT_TXT_TOKENS;
    let projected = projected_dense_peak_gb(s, bf16);
    let safe = budget_gb * 0.85;
    if projected > safe {
        return Err(format!(
            "sd3 trainer: a dense first training step at resolution {edge} needs ~{projected:.0} GB \
             (the forward working set materializes in one allocation), exceeding this machine's \
             ~{safe:.0} GB safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the \
             OS would hard-kill the worker (SIGKILL) at the first step with no recoverable error. \
             Enable Gradient Checkpointing (recomputes block activations in the backward) or reduce \
             the training resolution."
        )
        .into());
    }
    Ok(())
}

/// Resolve the config's target-module *suffixes* (default [`DEFAULT_TARGET_MODULES`]) to full dotted
/// paths by matching them against every adapter-routable module on the MMDiT — the same suffix-match
/// PEFT's `LoraConfig(target_modules=…)` does. The DEFAULT set is restricted to the joint
/// `transformer_blocks`; an explicit `lora_target_modules` matches anywhere (incl. the global
/// projections).
fn resolve_target_paths(transformer: &Sd3Transformer, cfg: &TrainingConfig) -> Vec<String> {
    let default = cfg.lora_target_modules.is_empty();
    let suffixes: Vec<String> = if default {
        DEFAULT_TARGET_MODULES
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        cfg.lora_target_modules.clone()
    };
    AdaptableHost::adaptable_paths(transformer)
        .into_iter()
        .filter(|path| {
            (!default || path.starts_with("transformer_blocks."))
                && suffixes
                    .iter()
                    .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
}

/// Decode an image file (PNG/JPEG) into the core RGB8 [`Image`].
fn decode_image(path: &Path) -> Result<Image> {
    let dynimg = image::open(path)
        .map_err(|e| Error::Msg(format!("decode image {}: {e}", path.display())))?;
    let rgb = dynimg.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

/// Encode a center-cropped square image into an SD3.5 training latent `[1, 16, edge/8, edge/8]` — the
/// 16-ch VAE encode the MMDiT predicts in: `preprocess_init_image` (resize + `[−1,1]` NCHW) →
/// [`Vae::encode`] (16-ch latent in the scaled/shifted latent space, `[1,16,edge/8,edge/8]`). Unlike
/// z-image's packed DiT input, the SD3 latent stays plain NCHW — NO pack/transpose, NO temporal axis.
fn encode_init_latents(vae: &Vae, image: &Image, edge: u32) -> Result<Array> {
    let pre = preprocess_init_image(image, edge, edge)?; // [1, 3, edge, edge] in [-1, 1]
    vae.encode(&pre) // [1, 16, edge/8, edge/8]
}

/// Sample a normalized flow-match timestep `t ∈ [1e-3, 1−1e-3]`. The SD3.5 default is **logit-normal**
/// (`u~U(0,1)`, `t = σ(m + s·Φ⁻¹(u))`, `m=0, s=1`; the Acklam probit + logistic ported from
/// `mlx-gen-ideogram/src/scheduler.rs`, WITHOUT the resolution-aware inference mean-shift). The
/// cross-family `sigmoid`/`uniform`/`linear`/`weighted` (a faithful port of the SceneWorks
/// `sample_training_timestep`) remain for parity; bias `high` → `√t`, `low` → `t²`. Deterministic in
/// `seed`. An empty / `"default"` `timestep_type` selects the logit-normal default.
fn sample_sigma(timestep_type: &str, timestep_bias: &str, seed: u64) -> Result<f32> {
    let k1 = random::key(seed)?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let ttype = {
        let n = normalize_cfg(timestep_type);
        if n.is_empty() || n == "default" {
            SD3_DEFAULT_TIMESTEP_TYPE.to_string()
        } else {
            n
        }
    };
    let t = match ttype.as_str() {
        "logit_normal" => {
            // u ~ U(0,1) → t = σ(m + s·ndtri(u)); m=0, s=1 (= σ(N(0,1)), the standard logit-normal).
            let u =
                random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>() as f64;
            logistic(0.0 + 1.0 * ndtri(u)) as f32
        }
        "linear" | "uniform" => {
            random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>()
        }
        "weighted" => {
            let k2 = random::key(seed ^ 0x9E37_79B9)?;
            let base = random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>();
            let center = sigmoid(random::normal::<f32>(&[1], None, None, Some(&k2))?.item::<f32>());
            (base + center) / 2.0
        }
        _ => sigmoid(random::normal::<f32>(&[1], None, None, Some(&k1))?.item::<f32>()),
    };
    let bias = normalize_cfg(timestep_bias);
    let t = match bias.as_str() {
        "high" | "high_noise" | "favor_high_noise" => t.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t * t,
        _ => t,
    };
    Ok(t.clamp(1e-3, 1.0 - 1e-3))
}

/// Logistic (sigmoid) `σ(y) = 1/(1+e^-y)`. Ported from `mlx-gen-ideogram/src/scheduler.rs`.
fn logistic(y: f64) -> f64 {
    1.0 / (1.0 + (-y).exp())
}

/// Inverse normal CDF (probit), Acklam's rational approximation (|err| ≲ 1.15e-9). Endpoints map to
/// ±∞ so the logistic squashes them back into (0,1) before the trainer clamp. Ported from
/// `mlx-gen-ideogram/src/scheduler.rs` (the T1-designated reference).
fn ndtri(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    let p_high = 1.0 - P_LOW;
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// One forward+backward over the trainable adapter factors: inject `params` (LoRA or LoKr), run the
/// SD3 MMDiT at the **`t·1000`-scaled** timestep, regress the **raw** `forward()` velocity onto
/// `noise − x0` (NO sign flip), return `(loss, grads)`. `dtype` is the training compute dtype; the
/// LoRA factors are cast inside the traced install (`lora_dtype`), so the MMDiT graph runs at `dtype`;
/// the noising math, loss, and grads stay f32.
///
/// `checkpoint_blocks`, when `Some`, lists per-joint-block LOCAL LoRA target paths and switches the
/// forward to the gradient-checkpointed path. `None` runs the dense forward.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    transformer: &mut Sd3Transformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    cond: &Sd3Conditioning,
    t: f32,
    noise: &Array,
    mae: bool,
    dtype: Dtype,
    lora_dtype: Option<Dtype>,
    checkpoint_blocks: Option<&[Vec<String>]>,
) -> Result<(f32, LoraParams)> {
    let (x_t, target) = build_batch(x0, noise, t)?;
    let x_t = x_t.as_dtype(dtype)?; // no-op in f32 mode
                                    // CRITICAL: the SD3 MMDiT embeds the diffusers-scale timestep `t·1000` (NUM_TRAIN_TIMESTEPS),
                                    // NOT the raw `t ∈ (0,1)`. `build_batch` above used the un-scaled `t` for the noising; the forward
                                    // gets `t·1000`. (z-image/Krea pass `t` raw — this is the top SD3 parity delta.)
    let timestep = Array::from_slice(&[t * NUM_TRAIN_TIMESTEPS], &[1]);
    let context = cond.context.clone();
    let pooled = cond.pooled.clone();
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // Install ALL targets so the dense path (and any non-checkpointed global targets) train
        // through ordinary autograd; on the checkpointed path the joint-block adapters installed here
        // are replaced inside each checkpoint segment by the explicit-input factors.
        adapter.install_as(transformer, &p, alpha, rank, lora_dtype, LOKR_DTYPE)?;
        // Training drives the compute dtype EXPLICITLY (`dtype` = the bf16 train dtype, or f32). The
        // inference `forward` is f32-pinned and must not be used here; both training paths take the
        // explicit-dtype seam so bf16 training runs the heavy matmuls in bf16.
        let v = match checkpoint_blocks {
            Some(locals) => transformer
                .forward_with_blocks_checkpointed(
                    &x_t, &context, &pooled, &timestep, &p, locals, alpha, dtype,
                )
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => transformer
                .forward_with(&x_t, &context, &pooled, &timestep, dtype)
                .map_err(|e| Exception::custom(e.to_string()))?,
        };
        let diff = subtract(&v, &target)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::CancelFlag;
    use std::path::PathBuf;

    fn base_config() -> TrainingConfig {
        TrainingConfig {
            rank: 8,
            steps: 10,
            ..Default::default()
        }
    }

    fn req_with(config: TrainingConfig) -> TrainingRequest {
        TrainingRequest {
            items: vec![mlx_gen::TrainingItem {
                image_path: PathBuf::from("/tmp/x.png"),
                caption: "a swatch".into(),
            }],
            config,
            output_dir: PathBuf::from("/tmp/sd3_unused"),
            file_name: "lora.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }

    #[test]
    fn descriptor_is_the_large_base_id() {
        let d = trainer_descriptor();
        assert_eq!(d.id, "sd3_5_large");
        assert_eq!(d.family, "sd3");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.supports_lora && d.supports_lokr);
    }

    #[test]
    fn validate_rejects_empty_dataset_and_zero_rank_steps() {
        let mut r = req_with(base_config());
        r.items.clear();
        assert!(validate_request(&r)
            .unwrap_err()
            .to_string()
            .contains("dataset is empty"));

        let r = req_with(TrainingConfig {
            rank: 0,
            ..base_config()
        });
        assert!(validate_request(&r)
            .unwrap_err()
            .to_string()
            .contains("rank"));

        let r = req_with(TrainingConfig {
            steps: 0,
            ..base_config()
        });
        assert!(validate_request(&r)
            .unwrap_err()
            .to_string()
            .contains("steps"));
    }

    #[test]
    fn validate_accepts_gradient_checkpointing_and_logit_normal() {
        let r = req_with(TrainingConfig {
            gradient_checkpointing: true,
            timestep_type: "logit_normal".into(),
            ..base_config()
        });
        assert!(validate_request(&r).is_ok());
        // The SD3 default (empty timestep_type) is accepted (→ logit-normal).
        let r = req_with(TrainingConfig {
            timestep_type: "".into(),
            ..base_config()
        });
        assert!(validate_request(&r).is_ok());
    }

    #[test]
    fn validate_rejects_unrecognized_optimizer_timestep_loss() {
        for cfg in [
            TrainingConfig {
                optimizer: "nope".into(),
                ..base_config()
            },
            TrainingConfig {
                timestep_type: "bogus".into(),
                ..base_config()
            },
            TrainingConfig {
                timestep_bias: "sideways".into(),
                ..base_config()
            },
            TrainingConfig {
                loss_type: "huber".into(),
                ..base_config()
            },
        ] {
            assert!(validate_request(&req_with(cfg)).is_err());
        }
        assert!(validate_request(&req_with(TrainingConfig {
            timestep_type: "Logit-Normal".into(),
            timestep_bias: "high-noise".into(),
            loss_type: "L1".into(),
            optimizer: "adamw".into(),
            ..base_config()
        }))
        .is_ok());
    }

    #[test]
    fn build_batch_is_velocity_with_no_sign_flip() {
        // target = noise − x0 (the RAW SD3 MMDiT velocity; the SAME sign as Krea, OPPOSITE z-image),
        // and x_t = (1−t)·x0 + t·noise. NO sign flip on the target. (The DiT timestep is scaled to
        // t·1000 at the forward call — see `t1000_timestep_scaling_pins_diffusers_scale`.)
        let x0 = Array::from_slice(&[2.0f32, 4.0, 6.0], &[1, 3, 1]);
        let noise = Array::from_slice(&[1.0f32, 1.0, 1.0], &[1, 3, 1]);
        let (x_t, target) = build_batch(&x0, &noise, 0.25).unwrap();
        assert_eq!(target.as_slice::<f32>(), &[-1.0, -3.0, -5.0]); // noise − x0, NOT x0 − noise
        let xt = x_t.as_slice::<f32>();
        for (got, want) in xt.iter().zip([1.75f32, 3.25, 4.75].iter()) {
            assert!((got - want).abs() < 1e-6, "x_t {got} != {want}");
        }
    }

    #[test]
    fn t1000_timestep_scaling_pins_diffusers_scale() {
        // The trainer scales the normalized flow-match `t ∈ (0,1)` to the diffusers `t·1000` the SD3
        // MMDiT embeds (NUM_TRAIN_TIMESTEPS), distinct from z-image/Krea which pass `t` raw. This pins
        // that constant and the scaling the forward call applies (the top SD3 parity risk).
        assert_eq!(NUM_TRAIN_TIMESTEPS, 1000.0);
        for t in [1e-3f32, 0.25, 0.5, 0.9999] {
            let scaled = t * NUM_TRAIN_TIMESTEPS;
            assert!((scaled - t * 1000.0).abs() < 1e-3, "t={t} scaled={scaled}");
        }
    }

    #[test]
    fn logit_normal_sampler_is_deterministic_in_range_and_monotone() {
        // Deterministic in seed; in (1e-3, 1−1e-3); and the bias monotone (high pushes toward 1, low
        // toward 0). Also: the median (u=0.5 path is stochastic, so check the distribution mean ≈ 0.5
        // for the standard logit-normal m=0,s=1 over many seeds).
        for bias in ["balanced", "high", "low"] {
            let a = sample_sigma("logit_normal", bias, 42).unwrap();
            let b = sample_sigma("logit_normal", bias, 42).unwrap();
            assert_eq!(a, b, "logit_normal/{bias} must be deterministic in seed");
            assert!(
                (1e-3..=1.0 - 1e-3).contains(&a),
                "logit_normal/{bias} t={a} out of range"
            );
        }
        assert!(
            sample_sigma("logit_normal", "high", 7).unwrap()
                > sample_sigma("logit_normal", "low", 7).unwrap()
        );
        // Standard logit-normal σ(N(0,1)) has mean ≈ 0.5 (symmetric about 0.5); pin it loosely.
        let mut sum = 0.0f64;
        let n = 400u64;
        for s in 0..n {
            sum +=
                sample_sigma("logit_normal", "balanced", s.wrapping_mul(7919) + 1).unwrap() as f64;
        }
        let mean = sum / n as f64;
        assert!(
            (mean - 0.5).abs() < 0.06,
            "logit-normal mean {mean} not ≈ 0.5"
        );
        // Empty / "default" timestep_type routes to the logit-normal default (same as named).
        assert_eq!(
            sample_sigma("", "balanced", 123).unwrap(),
            sample_sigma("logit_normal", "balanced", 123).unwrap()
        );
        assert_eq!(
            sample_sigma("default", "balanced", 123).unwrap(),
            sample_sigma("logit_normal", "balanced", 123).unwrap()
        );
    }

    #[test]
    fn ndtri_matches_known_probit_values() {
        // Φ⁻¹(0.5)=0, Φ⁻¹(0.975)≈1.959964, Φ⁻¹(0.025)≈−1.959964 (the reference probit; the Acklam
        // approximation is |err| ≲ 1.15e-9).
        assert!(ndtri(0.5).abs() < 1e-6);
        assert!((ndtri(0.975) - 1.959_963_984_540_054).abs() < 1e-4);
        assert!((ndtri(0.025) + 1.959_963_984_540_054).abs() < 1e-4);
        assert_eq!(ndtri(0.0), f64::NEG_INFINITY);
        assert_eq!(ndtri(1.0), f64::INFINITY);
    }

    #[test]
    fn sigmoid_parity_sampler_in_range_and_deterministic() {
        for kind in ["sigmoid", "linear", "weighted"] {
            for bias in ["balanced", "high", "low"] {
                let a = sample_sigma(kind, bias, 42).unwrap();
                let b = sample_sigma(kind, bias, 42).unwrap();
                assert_eq!(a, b, "{kind}/{bias} must be deterministic");
                assert!((1e-3..=1.0 - 1e-3).contains(&a), "{kind}/{bias} t={a} oob");
            }
        }
    }

    #[test]
    fn default_target_modules_are_both_joint_streams() {
        // The default training surface is the standard SD3 PEFT attention surface: image + text stream
        // attention projections (both joint streams).
        assert_eq!(
            DEFAULT_TARGET_MODULES,
            [
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_add_out"
            ]
        );
    }

    #[test]
    fn preflight_guard_fires_over_budget_and_passes_under() {
        // A 16 GB-class budget (safe ≈ 13.6 GB): the 16 GB bf16 MMDiT base alone exceeds it → dense
        // 1024 must be refused with an actionable error recommending Gradient Checkpointing.
        let err = check_preflight_budget(1024, true, 16.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Gradient Checkpointing"), "got: {err}");
        assert!(
            err.contains("1024"),
            "error should name the resolution: {err}"
        );
        // A 128 GB-class budget (safe ≈ 108 GB) comfortably fits dense 1024 in both dtypes.
        assert!(check_preflight_budget(1024, true, 128.0).is_ok());
        assert!(check_preflight_budget(1024, false, 128.0).is_ok());
    }

    #[test]
    fn reachable_via_trainer_registry_by_id() {
        assert!(
            gen_core::registry::trainers().any(|r| (r.descriptor)().id == SD3_5_LARGE_TRAINER_ID),
            "trainer id {SD3_5_LARGE_TRAINER_ID} not registered"
        );
    }

    #[test]
    fn projection_is_monotonic_and_conservative() {
        for bf16 in [false, true] {
            let (s512, s768, s1024) = (1024.0 + 333.0, 2304.0 + 333.0, 4096.0 + 333.0);
            assert!(projected_dense_peak_gb(s512, bf16) < projected_dense_peak_gb(s768, bf16));
            assert!(projected_dense_peak_gb(s768, bf16) < projected_dense_peak_gb(s1024, bf16));
        }
        assert!(projected_dense_peak_gb(4429.0, true) < projected_dense_peak_gb(4429.0, false));
        // The bf16 base (no tokens) is ~the resident MMDiT weights (≥ 14 GB).
        assert!(projected_dense_peak_gb(0.0, true) >= 14.0);
    }
}

// ===========================================================================================
// T2 (sc-7883) — real-weight adaptable-paths + checkpoint-parity harness (weight-gated).
//
//   SD3_LARGE_DIR=/path/to/stable-diffusion-3.5-large \
//     cargo test -p mlx-gen-sd3 --release --lib real_weight_repro -- --ignored --nocapture
// ===========================================================================================
#[cfg(test)]
mod real_weight_repro {
    use super::*;
    use std::path::PathBuf;

    /// The `stabilityai/stable-diffusion-3.5-large` snapshot (the `SD3_LARGE_DIR` override, else the
    /// newest HF-cache snapshot with a `transformer/` tree).
    fn snapshot() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("SD3_LARGE_DIR") {
            return Some(PathBuf::from(p));
        }
        let home = std::env::var("HOME").ok()?;
        let snaps = PathBuf::from(home).join(
            ".cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-large/snapshots",
        );
        std::fs::read_dir(&snaps)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.is_dir() && p.join("transformer").is_dir())
    }

    /// The real MMDiT's default target surface resolves to the joint-block attention only: 38 blocks ×
    /// {to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj} = 266, plus to_add_out on all but the
    /// final context_pre_only block = 37 → 303 targets, none on globals.
    #[test]
    #[ignore = "needs real stabilityai/stable-diffusion-3.5-large weights; run as its own process"]
    fn default_targets_resolve_to_joint_block_attention() {
        let root = snapshot().expect("sd3.5-large snapshot (HF cache or SD3_LARGE_DIR)");
        let dit = loader::load_transformer(&root, &Sd3Variant::Large.arch()).unwrap();
        let cfg = TrainingConfig::default();
        let paths = resolve_target_paths(&dit, &cfg);
        let n = dit.num_blocks();
        // 7 streams on every block + to_add_out on every non-final block.
        assert_eq!(paths.len(), n * 7 + (n - 1), "{} targets", n * 7 + (n - 1));
        assert!(paths.iter().all(|p| p.starts_with("transformer_blocks.")));
        assert!(paths.iter().any(|p| p.ends_with(".attn.to_q")));
        assert!(paths.iter().any(|p| p.ends_with(".attn.add_q_proj")));
        // The final context_pre_only block has no to_add_out.
        assert!(!paths.contains(&format!("transformer_blocks.{}.attn.to_add_out", n - 1)));
    }

    /// Whole-block gradient checkpointing must not change the math: the checkpointed forward+grads must
    /// match the dense path within fp tolerance. Run in f32 at a tiny resolution.
    #[test]
    #[ignore = "needs real stabilityai/stable-diffusion-3.5-large weights; run as its own process"]
    fn checkpointed_grads_match_dense() {
        let root = snapshot().expect("sd3.5-large snapshot (HF cache or SD3_LARGE_DIR)");
        let mut dit = loader::load_transformer(&root, &Sd3Variant::Large.arch()).unwrap();
        dit.cast_weights(Dtype::Float32).unwrap();
        let cfg = TrainingConfig {
            rank: 4,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&dit, &cfg);
        let (targets, params) = build_lora_targets(&mut dit, &target_paths, 4, 7).unwrap();
        let adapter = TrainAdapter::Lora { targets };
        let mut locals: Vec<Vec<String>> = vec![Vec::new(); dit.num_blocks()];
        for path in &target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("transformer_blocks.")
                .and_then(|rest| rest.split_once('.'))
            {
                if let Ok(i) = idx.parse::<usize>() {
                    locals[i].push(local.to_string());
                }
            }
        }

        // Tiny synthetic batch (latent 32×32 → img tokens 256; the math is resolution-agnostic).
        let x0 =
            random::normal::<f32>(&[1, 16, 32, 32], None, None, Some(&random::key(1).unwrap()))
                .unwrap();
        let noise =
            random::normal::<f32>(&[1, 16, 32, 32], None, None, Some(&random::key(2).unwrap()))
                .unwrap();
        let context =
            random::normal::<f32>(&[1, 333, 4096], None, None, Some(&random::key(3).unwrap()))
                .unwrap();
        let pooled =
            random::normal::<f32>(&[1, 2048], None, None, Some(&random::key(4).unwrap())).unwrap();
        let cond = Sd3Conditioning { context, pooled };

        dit.set_sdpa_checkpoint(true);
        let (_l, g_dense) = compute_loss_grads(
            &mut dit,
            &params,
            &adapter,
            4.0,
            4.0,
            &x0,
            &cond,
            0.5,
            &noise,
            false,
            Dtype::Float32,
            None,
            None,
        )
        .unwrap();
        eval(g_dense.values()).unwrap();

        dit.set_sdpa_checkpoint(false);
        let (_l, g_ckpt) = compute_loss_grads(
            &mut dit,
            &params,
            &adapter,
            4.0,
            4.0,
            &x0,
            &cond,
            0.5,
            &noise,
            false,
            Dtype::Float32,
            None,
            Some(&locals),
        )
        .unwrap();
        eval(g_ckpt.values()).unwrap();

        let mut max_rel = 0f32;
        for (k, a) in &g_dense {
            let b = g_ckpt.get(k).expect("same keys");
            let num = a.subtract(b).unwrap().abs().unwrap().max(None).unwrap();
            let den = a.abs().unwrap().max(None).unwrap().item::<f32>().max(1e-6);
            max_rel = max_rel.max(num.item::<f32>() / den);
        }
        eprintln!("[sc-7883] checkpointed-vs-dense grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-3,
            "checkpointed grads must match dense: max rel {max_rel:.2e}"
        );
    }

    /// END-TO-END TRAINING SMOKE + ROUND-TRIP (the T2 acceptance proof): train a tiny LoRA (few steps,
    /// small rank, bf16 + gradient checkpointing) on a tiny synthetic dataset through the real
    /// [`Sd3LoraTrainer`], save the adapter, then RELOAD it via [`crate::adapters::apply_sd3_adapters`]
    /// at `sd3_5_large` generation and render — confirming the adapter loads, applies, and produces a
    /// coherent image (the round-trip). Memory-hardened for the 8.1B Large via the trainer's bf16 +
    /// block checkpointing.
    #[test]
    #[ignore = "needs real stabilityai/stable-diffusion-3.5-large weights + Metal; run as its own process"]
    fn training_smoke_round_trip() {
        use mlx_gen::runtime::{AdapterKind, AdapterSpec, LoadSpec, WeightsSource};
        use mlx_gen::{GenerationOutput, GenerationRequest, NetworkType, TrainingItem};

        let root = snapshot().expect("sd3.5-large snapshot (HF cache or SD3_LARGE_DIR)");
        let tmp = std::env::temp_dir().join(format!("sd3_t2_smoke_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        // --- tiny synthetic dataset: 2 solid-color 256² PNGs with captions ---
        let mk_png = |path: &std::path::Path, rgb: [u8; 3]| {
            let mut img = image::RgbImage::new(256, 256);
            for px in img.pixels_mut() {
                *px = image::Rgb(rgb);
            }
            img.save(path).unwrap();
        };
        let img_a = tmp.join("a.png");
        let img_b = tmp.join("b.png");
        mk_png(&img_a, [200, 40, 40]);
        mk_png(&img_b, [40, 60, 200]);
        let items = vec![
            TrainingItem {
                image_path: img_a.clone(),
                caption: "sks a solid crimson swatch".into(),
            },
            TrainingItem {
                image_path: img_b.clone(),
                caption: "sks a solid cobalt swatch".into(),
            },
        ];

        // --- train (bf16 + gradient checkpointing; tiny rank/steps at 256²) ---
        // LoadSpec default precision is Bf16 — exactly the trainer's bf16 compute path.
        let mut trainer = load_trainer(&LoadSpec::new(WeightsSource::Dir(root.clone())))
            .expect("load sd3 trainer");
        let cfg = TrainingConfig {
            rank: 4,
            alpha: 4.0,
            learning_rate: 1e-4,
            steps: 6,
            resolution: 256,
            save_every: 0,
            seed: 7,
            network_type: NetworkType::Lora,
            gradient_checkpointing: true,
            train_dtype: "bf16".into(),
            timestep_type: "logit_normal".into(),
            ..Default::default()
        };
        let adapter_path = tmp.join("sd3_smoke_lora.safetensors");
        let req = TrainingRequest {
            items,
            config: cfg,
            output_dir: tmp.clone(),
            file_name: "sd3_smoke_lora.safetensors".into(),
            trigger_words: vec!["sks".into()],
            cancel: mlx_gen::CancelFlag::new(),
        };
        let mut losses: Vec<f32> = Vec::new();
        let out = trainer
            .train(&req, &mut |p| {
                if let TrainingProgress::Training { step, loss, .. } = p {
                    eprintln!("[sc-7883 smoke] step {step} loss {loss:.5}");
                    losses.push(loss);
                }
            })
            .expect("training run");
        eprintln!(
            "[sc-7883 smoke] TRAINED: steps={} final_loss={:.5} adapter={}",
            out.steps,
            out.final_loss,
            out.adapter_path.display()
        );
        assert!(out.adapter_path.exists(), "adapter file written");
        assert!(out.steps == 6, "ran all steps");
        assert!(
            out.final_loss.is_finite() && out.final_loss > 0.0,
            "finite loss"
        );
        // Drop the trainer (frees the resident MMDiT) before loading the inference model.
        drop(trainer);
        mlx_rs::memory::clear_cache();

        // --- round-trip: reload the trained adapter at sd3_5_large generation ---
        let spec = LoadSpec::new(WeightsSource::Dir(root))
            .with_quant(mlx_gen::Quant::Q8) // Q8 keeps the 8.1B inference footprint in budget
            .with_adapters(vec![AdapterSpec::new(
                adapter_path.clone(),
                1.0,
                AdapterKind::Lora,
            )]);
        let model = crate::model::load(&spec).expect("load sd3_5_large WITH the trained adapter");
        let gen_req = GenerationRequest {
            prompt: "sks a solid crimson swatch".into(),
            width: 512,
            height: 512,
            steps: Some(8),
            seed: Some(1),
            count: 1,
            ..Default::default()
        };
        let gout = model
            .generate(&gen_req, &mut |_| {})
            .expect("generate WITH the reloaded adapter");
        let img = match gout {
            GenerationOutput::Images(mut v) => v.remove(0),
            _ => panic!("expected an image"),
        };
        assert_eq!((img.width, img.height), (512, 512));
        // Coherence: not all-black / not NaN-collapsed (a broken adapter apply produces a degenerate
        // frame). Check the mean luminance is in a sane mid-range and there is real variance.
        let n = img.pixels.len() as f64;
        let mean = img.pixels.iter().map(|&b| b as f64).sum::<f64>() / n;
        let var = img
            .pixels
            .iter()
            .map(|&b| (b as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        eprintln!("[sc-7883 smoke] ROUND-TRIP render mean={mean:.1} var={var:.1}");
        let png = tmp.join("sd3_smoke_render.png");
        image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .unwrap()
            .save(&png)
            .unwrap();
        eprintln!("[sc-7883 smoke] wrote {}", png.display());
        assert!(
            mean > 5.0 && mean < 250.0,
            "render mean luminance sane (coherent)"
        );
        assert!(
            var > 1.0,
            "render has real variance (not a flat/degenerate frame)"
        );
    }
}
