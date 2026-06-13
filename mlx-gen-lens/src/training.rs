//! LoRA/LoKr **training** on the Lens DiT, in pure Rust on mlx-rs (sc-5148, epic 3164) — the
//! native-MLX replacement for the Python `lens_train_runner.py` (torch in `/opt/lens-venv`), the last
//! Python holdout for Lens (zero-Python north star, epic 3482).
//!
//! [`LensTrainer`] realizes the core [`Trainer`](mlx_gen::Trainer) contract on the real 48-block Lens
//! MMDiT, mirroring [`ZImageTurboTrainer`](mlx_gen_z_image) — the model crates don't use mlx-rs's
//! `Module` system (hand-rolled `&self` forwards over raw `Array`s), so training uses the **functional
//! autograd**: the trainable factors live OUTSIDE the model in a [`LoraParams`] map, re-injected each
//! step into the target [`AdaptableLinear`](mlx_gen::adapters::AdaptableLinear)s via the shared core
//! seam ([`mlx_gen::train::lora`]), stepped with `keyed_value_and_grad` + the core [`TrainOptimizer`] +
//! `clip_grad_norm`. The injection mirrors the inference reload op-for-op, so the trained adapter
//! round-trips through [`apply_lens_adapters`](crate::adapters::apply_lens_adapters) (sc-3174)
//! bit-for-bit.
//!
//! ## What is Lens-specific (everything else reuses the family-agnostic core unchanged)
//!
//! Ported from `lens_train_runner.py`:
//!   * **Flow-match velocity target = `noise − x0`** with the transformer **timestep = `t`** (the noise
//!     fraction) fed directly. The Lens DiT [`forward`](crate::dit::LensTransformer::forward) returns
//!     the **raw** patch-space velocity (no negation — the pipeline feeds it to `FlowMatchEuler::step`
//!     un-negated), so the regression target is the velocity itself. This is the **opposite sign** of
//!     the Z-Image trainer, whose Rust `forward()` negates → target `noise − x0` *with* `timestep =
//!     1 − σ`. `x_t = (1 − t)·x0 + t·noise`.
//!   * **Latents by inverting the Lens `_decode`.** The Lens latent space *is* the Flux.2 one, so a
//!     pixel → `[1, seq, 128]` training latent is exactly the Flux.2 `encode_init_latents` chain
//!     (`preprocess_ref_image → Flux2Vae::encode_mean → patchify → bn-normalize → pack`). Uses the
//!     deterministic latent **mean** (the only public encode path + the established mlx-gen img2img
//!     convention); the Python's `latent_dist.sample()` reparam-noise is a minor regularizer dropped
//!     deliberately.
//!   * **Caption features.** The pipeline's positive-only `encode_one`: tokenize → the gpt-oss
//!     [`encode`](crate::text_encoder::encoder::LensTextEncoder::encode) (4 captured layers) → slice
//!     at [`TXT_OFFSET`] → a ones mask. Single-conditional (no CFG), matching the Python.
//!   * **Targets** default to `img_qkv`/`txt_qkv`/`to_out.0`/`to_add_out` (the `AdaptableHost for
//!     LensTransformer` paths, sc-3174); LoKr reconstructs at [`LOKR_DTYPE`] (what the lens adapter
//!     loader uses, so the trained LoKr round-trips). The gpt-oss encoder loads **Q8** (~12 GB vs
//!     ~40 GB dense bf16) — frozen, used only to cache caption features, then dropped before the train
//!     loop (the 32 GB-Mac free pattern); Q8 also matches the Q8 inference default (sc-3172/sc-5105).
//!
//! Registered under the **`lens`** id (the base, non-distilled `microsoft/Lens` — sc-1583; arch-
//! identical to `lens_turbo`, so the adapter applies to both, sc-3174).
//!
//! ## Scope (sc-5148): functional trainer. Production-resolution **memory hardening is a follow-up**.
//! The Python trainer exposes `gradient_checkpointing`; the Lens DiT has no checkpointed forward, and
//! the z-image analog (a checkpointed forward + SDPA-segment checkpointing + a fitted OOM preflight
//! guard, with grad-parity gates) was three separate stories (sc-4874/4886/4887) *after* its base
//! trainer. So this trainer **rejects** `gradient_checkpointing = true` with a pointer (NOT a silent
//! no-op) rather than mislead the caller into thinking they are protected; the default is off.

use std::path::Path;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core;
use mlx_gen::media::Image;
use mlx_gen::train::checkpoint::checkpoint_filename;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, average_grads, build_lokr_targets, build_lora_targets, LoraParams,
    TrainAdapter,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::weights::Weights;
use mlx_gen::{
    Error, LoadSpec, Modality, NetworkType, Precision, Quant, Result, TrainOptimizer, Trainer,
    TrainerDescriptor, TrainerRegistration, TrainingConfig, TrainingOutput, TrainingProgress,
    TrainingRequest, WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::{add, multiply, ones, split_sections, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use mlx_gen_flux2::{load_vae, pack_latents, patchify_latents, preprocess_ref_image, Flux2Vae};

use crate::config::GptOssConfig;
use crate::dit::{LensDitConfig, LensTransformer};
use crate::pipeline::{DEFAULT_DATE, VAE_SCALE_FACTOR};
use crate::registry::MODEL_ID_BASE;
use crate::text::{LensTokenizer, TXT_OFFSET};
use crate::text_encoder::encoder::LensTextEncoder;

/// The lens adapter loader reconstructs LoKr deltas at bf16 (`src/adapters/loader.rs`); training must
/// reconstruct at the same dtype so the trained LoKr round-trips through `apply_lens_adapters`.
const LOKR_DTYPE: Dtype = Dtype::Bfloat16;

/// The gpt-oss encoder is loaded Q8 for the trainer (~12 GB vs ~40 GB dense bf16): it is frozen and
/// used only to cache caption features once, then dropped. Q8 is the Lens inference default (sc-3172),
/// so the cached features match the deployed encode path.
const TRAINER_ENCODER_QUANT: Option<Quant> = Some(Quant::Q8);

/// The Lens trainer default target modules (`lens_train_runner.DEFAULT_LORA_TARGET_MODULES`): the
/// dual-stream joint-attention projections. `to_out` is an `nn.ModuleList([Linear, Identity])`, so the
/// trainable Linear is `to_out.0` (sc-2218); `img_qkv`/`txt_qkv` are the fused per-stream QKV.
const DEFAULT_TARGET_MODULES: [&str; 4] = ["img_qkv", "txt_qkv", "to_out.0", "to_add_out"];

/// Recognized `timestep_type` values [`sample_sigma`] branches on (plus the `sigmoid` default).
const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "linear", "uniform", "weighted"];
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
/// `target = noise − x0` (the velocity the **raw** Lens DiT output is regressed onto; the transformer
/// timestep is `t` itself, fed by the caller — see the module docs for the sign vs Z-Image).
fn build_batch(x0: &Array, noise: &Array, t: f32) -> Result<(Array, Array)> {
    let one_minus = Array::from_slice(&[1.0 - t], &[1]);
    let s = Array::from_slice(&[t], &[1]);
    let x_t = add(&multiply(x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, x0)?;
    Ok((x_t, target))
}

/// The production [`Trainer`] for the base `microsoft/Lens` DiT: a frozen base (gpt-oss encoder + Lens
/// MMDiT + Flux.2 VAE + tokenizer) that caches a captioned dataset to VAE-latents/caption-features,
/// then runs the functional-autograd LoRA/LoKr loop with the core runtime glue (LR schedule, gradient
/// accumulation, checkpoint cadence, cancel, progress bands), writing a PEFT adapter that reloads
/// through the inference path.
pub struct LensTrainer {
    descriptor: TrainerDescriptor,
    tokenizer: LensTokenizer,
    /// The 20 B-param gpt-oss encoder, in an `Option` so it can be **dropped after caching** — it is
    /// idle during training (every caption is already cached), yet a multi-GB resident.
    encoder: Option<LensTextEncoder>,
    transformer: LensTransformer,
    vae: Flux2Vae,
    /// The compute dtype (bf16 production / f32 tight-gate), fixed at load from `spec.precision`.
    dtype: Dtype,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID_BASE,
        family: "lens",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// Construct the trainer from a `microsoft/Lens` snapshot directory (the diffusers multi-component
/// tree: `tokenizer/ text_encoder/ transformer/ vae/`). The DiT is loaded **dense** (the adapter host);
/// the encoder is Q8. `spec.precision` selects the compute dtype (bf16 default / f32 tight-gate).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "lens trainer expects a snapshot directory (tokenizer/ text_encoder/ transformer/ \
                 vae/), not a single .safetensors file"
                    .into(),
            )),
        };
    let dtype = match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    };
    let tokenizer = LensTokenizer::from_file(root.join("tokenizer").join("tokenizer.json"))?;
    let enc_cfg = GptOssConfig::lens();
    let enc_w = Weights::from_dir(root.join("text_encoder"))?;
    let encoder =
        LensTextEncoder::from_weights_quant(&enc_w, &enc_cfg, dtype, TRAINER_ENCODER_QUANT)?;
    let dit_cfg = LensDitConfig::lens();
    let dit_w = Weights::from_dir(root.join("transformer"))?;
    let transformer = LensTransformer::from_weights(&dit_w, &dit_cfg, dtype)?;
    let vae = load_vae(&root)?;
    Ok(Box::new(LensTrainer {
        descriptor: trainer_descriptor(),
        tokenizer,
        encoder: Some(encoder),
        transformer,
        vae,
        dtype,
    }))
}

/// Registry adapter: the trainer registry's `load` slot is typed on [`gen_core::Result`] (epic 3720);
/// bridge the crate's rich-`Result` [`load_trainer`] into it.
fn load_trainer_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Trainer>> {
    load_trainer(spec).map_err(Into::into)
}

inventory::submit! {
    TrainerRegistration { descriptor: trainer_descriptor, load: load_trainer_registered }
}

/// Normalize a free-form config string the way the trainer's own parsers do (trim, lowercase,
/// `-`/space → `_`) so validation accepts exactly the spellings the run would.
fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Capability-free training-request validation, factored out so it can be unit-tested without a loaded
/// trainer. Rejects an empty dataset, zero rank, **zero steps** (a 0-step run would write a no-op
/// `B = 0` identity adapter), `gradient_checkpointing` (not yet implemented for the Lens DiT — a
/// pointer, not a silent no-op), an unsupported optimizer, and an unrecognized
/// `timestep_type`/`timestep_bias`/`loss_type` (rather than silently falling back to a default).
fn validate_request(req: &TrainingRequest) -> Result<()> {
    let cfg = &req.config;
    if req.items.is_empty() {
        return Err("lens trainer: dataset is empty".into());
    }
    if cfg.rank == 0 {
        return Err("lens trainer: rank must be > 0".into());
    }
    if cfg.steps == 0 {
        return Err("lens trainer: steps must be > 0".into());
    }
    if cfg.gradient_checkpointing {
        return Err("lens trainer: gradient_checkpointing is not yet implemented for the Lens DiT \
                    (tracked follow-up: a checkpointed forward + OOM preflight guard, the z-image \
                    sc-4874/4886/4887 analogs). Disable the Gradient Checkpointing toggle and train \
                    at a resolution that fits unified memory."
            .into());
    }
    if !TrainOptimizer::is_supported(&cfg.optimizer) {
        return Err(format!(
            "lens trainer: optimizer '{}' is not available on MLX training (supported: adamw, adam, \
             rose, prodigy)",
            cfg.optimizer
        )
        .into());
    }
    if !TIMESTEP_TYPES.contains(&normalize_cfg(&cfg.timestep_type).as_str()) {
        return Err(format!(
            "lens trainer: timestep_type '{}' is not recognized (supported: {})",
            cfg.timestep_type,
            TIMESTEP_TYPES.join(", ")
        )
        .into());
    }
    if !TIMESTEP_BIASES.contains(&normalize_cfg(&cfg.timestep_bias).as_str()) {
        return Err(format!(
            "lens trainer: timestep_bias '{}' is not recognized (supported: {})",
            cfg.timestep_bias,
            TIMESTEP_BIASES.join(", ")
        )
        .into());
    }
    if !LOSS_TYPES.contains(&normalize_cfg(&cfg.loss_type).as_str()) {
        return Err(format!(
            "lens trainer: loss_type '{}' is not recognized (supported: {})",
            cfg.loss_type,
            LOSS_TYPES.join(", ")
        )
        .into());
    }
    Ok(())
}

impl Trainer for LensTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        validate_request(req)?;
        // Non-default `lora_target_modules` that match no adaptable module on the DiT would train zero
        // parameters yet "succeed". Catch it here, where the loaded DiT is available to match against.
        if resolve_target_paths(&self.transformer, &req.config).is_empty() {
            return Err(format!(
                "lens trainer: lora_target_modules {:?} matched no adaptable module on the Lens DiT \
                 (targets are img_qkv/txt_qkv/to_out.0/to_add_out)",
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

impl LensTrainer {
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
                "lens trainer: lora_target_modules {:?} matched no adaptable module on the Lens DiT",
                cfg.lora_target_modules
            )
            .into());
        }

        // The DiT compute dtype is fixed at load (`spec.precision`); the Lens DiT has no
        // cast-after-load, so `train_dtype` is *enforced* against it (never a silent no-op). The
        // common case (TrainingConfig default bf16 + LoadSpec default Bf16) matches, so this only fires
        // on an explicit f32-vs-bf16 mismatch — telling the caller to load at the matching precision.
        let want_bf16 = {
            let t = cfg.train_dtype.trim();
            t.eq_ignore_ascii_case("bf16") || t.eq_ignore_ascii_case("bfloat16")
        };
        let loaded_bf16 = self.dtype == Dtype::Bfloat16;
        if want_bf16 != loaded_bf16 {
            return Err(format!(
                "lens trainer: train_dtype '{}' does not match the loaded precision ({}). Load the \
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
        // Lens latent grid: a cell maps to a 16×16 pixel tile (Flux.2 8× VAE ∘ 2× DiT patchify). The
        // ÷32 bucket guarantees the VAE-encoded `edge/8` is even, so the 2×2 patchify divides cleanly.
        let latent = (edge / VAE_SCALE_FACTOR) as usize; // latent_h == latent_w (square)

        // --- prepare → load → cache: VAE-latents + 4-layer caption features into memory ---
        on_progress(TrainingProgress::LoadingModel); // base model is already resident from load_trainer
        let total = req.items.len() as u32;
        let mut cache: Vec<(Array, Vec<Array>, Array)> = Vec::with_capacity(req.items.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = center_crop_square(&decode_image(&item.image_path)?);
            let x0 = encode_latents(&self.vae, &img, edge)?; // [1, seq, 128]
            let encoder = self.encoder.as_ref().ok_or_else(|| {
                Error::Msg(
                    "lens trainer: text encoder already freed (caching after train loop)".into(),
                )
            })?;
            let (features, mask) =
                encode_caption(&self.tokenizer, encoder, &item.caption, compute_dtype)?;
            let mut to_eval: Vec<&Array> = Vec::with_capacity(features.len() + 2);
            to_eval.push(&x0);
            to_eval.push(&mask);
            to_eval.extend(features.iter());
            eval(to_eval)?;
            cache.push((x0, features, mask));
        }
        if cache.is_empty() {
            // A cancel mid-cache is a genuine cancellation → typed `Error::Canceled`; an empty cache
            // with no cancel is a real "no usable dataset items" error.
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            return Err("lens trainer: no usable dataset items".into());
        }

        // Every caption is cached now — free the 20 B-param encoder and evict its buffers before the
        // train loop, reclaiming that resident for the DiT working set.
        self.encoder = None;
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
            let (x0, features, mask) = &cache[((step - 1) as usize) % cache.len()];
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
                features,
                mask,
                t,
                &noise,
                mae,
                compute_dtype,
                lora_dtype,
                latent,
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

        // --- save final adapter (the diffusers/PEFT format `apply_lens_adapters` loads, sc-3174) ---
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

/// Resolve the config's target-module *suffixes* (default [`DEFAULT_TARGET_MODULES`]) to full dotted
/// paths by matching them against every adapter-routable module on the DiT — the same suffix-match
/// PEFT's `LoraConfig(target_modules=…)` does (`transformer_blocks.{i}.attn.{suffix}`).
fn resolve_target_paths(transformer: &LensTransformer, cfg: &TrainingConfig) -> Vec<String> {
    let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
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
            suffixes
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

/// Encode a center-cropped square image into a Lens training latent `[1, latent·latent, 128]` — the
/// inverse of the Lens `_decode`. The Lens latent space *is* the Flux.2 one, so this is the Flux.2
/// `encode_init_latents` chain built from public helpers: `preprocess_ref_image` (resize + `[−1,1]`
/// NHWC) → `Flux2Vae::encode_mean` (latent mean) → NCHW → 2×2 `patchify_latents` →
/// `Flux2Vae::bn_normalize_nchw` → `pack_latents`. `pack_latents`/`patchify_latents` are plain
/// row-major reshapes consistent with the lens `vae::decode` plain-reshape path, so the latent lives in
/// exactly the space the DiT predicts in. `crop_to_even`/`match_latent_spatial_size` (in the fork's
/// `encode_init_latents`) are no-ops at the ÷32-bucketed square edge, so they are elided.
fn encode_latents(vae: &Flux2Vae, image: &Image, edge: u32) -> Result<Array> {
    let pre = preprocess_ref_image(image, edge, edge)?; // NHWC [1, edge, edge, 3]
    let enc = vae.encode_mean(&pre)?; // NHWC [1, edge/8, edge/8, 32]
    let enc = enc.transpose_axes(&[0, 3, 1, 2])?; // → NCHW for the packing helpers
    let patchified = patchify_latents(&enc)?; // [1, 128, edge/16, edge/16]
    let normed = vae.bn_normalize_nchw(&patchified)?; // (x − mean)/std on the packed 128-ch
    pack_latents(&normed) // [1, latent·latent, 128]
}

/// Encode a caption into its per-layer DiT text features (sliced at [`TXT_OFFSET`]) + the valid mask —
/// the pipeline's positive-only `encode_one` (single-conditional training; the Python keeps the
/// positives of `encode_prompt(neg="")`). Returns `(features, mask)`: `features` is 4 × `[1, S, 2880]`,
/// `mask` is `[1, S]` (all-1; a single prompt is unpadded).
fn encode_caption(
    tokenizer: &LensTokenizer,
    encoder: &LensTextEncoder,
    caption: &str,
    dtype: Dtype,
) -> Result<(Vec<Array>, Array)> {
    let out = tokenizer.encode(caption, DEFAULT_DATE)?;
    let l = out.ids.len() as i32;
    let offset = TXT_OFFSET as i32;
    if l <= offset {
        return Err(format!(
            "lens trainer: caption tokenized to {l} tokens (≤ the {offset}-token harmony preamble), \
             leaving no conditioning tokens"
        )
        .into());
    }
    let input_ids = Array::from_slice(&out.ids, &[1, l]);
    let layers = encoder.encode(&input_ids)?; // num_text_layers × [1, L, 2880]
                                              // `[:, offset:, :]` — split at the offset along the sequence axis, keep the tail.
    let features = layers
        .iter()
        .map(|f| Ok(split_sections(f, &[offset], 1)?[1].as_dtype(dtype)?))
        .collect::<Result<Vec<_>>>()?;
    let mask = ones::<f32>(&[1, l - offset])?;
    Ok((features, mask))
}

/// Sample a normalized flow-match timestep (interpolation coefficient) `t ∈ [1e-3, 1−1e-3]` — a
/// faithful port of the SceneWorks `sample_training_timestep` (identical to the Z-Image trainer):
/// `sigmoid(randn)` by default, `uniform` for linear, `(uniform + sigmoid(randn))/2` for weighted;
/// bias `high` → `√t`, `low` → `t²`. Deterministic in `seed`.
fn sample_sigma(timestep_type: &str, timestep_bias: &str, seed: u64) -> Result<f32> {
    let k1 = random::key(seed)?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let ttype = timestep_type.trim().to_ascii_lowercase().replace('-', "_");
    let t = match ttype.as_str() {
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
    let bias = timestep_bias
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-'], "_");
    let t = match bias.as_str() {
        "high" | "high_noise" | "favor_high_noise" => t.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t * t,
        _ => t,
    };
    Ok(t.clamp(1e-3, 1.0 - 1e-3))
}

/// One forward+backward over the trainable adapter factors: inject `params` (LoRA or LoKr), run the
/// Lens DiT, regress the **raw** `forward()` velocity onto `noise − x0`, return `(loss, grads)`. The
/// transformer timestep is `t` (the noise fraction) directly. `dtype` is the training compute dtype:
/// `x_t`/features are cast at entry (the weights were loaded at this dtype), the LoRA factors are cast
/// inside the traced install (`lora_dtype`), so the DiT graph runs at `dtype`; the noising math, loss,
/// and grads stay f32.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    transformer: &mut LensTransformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    features: &[Array],
    mask: &Array,
    t: f32,
    noise: &Array,
    mae: bool,
    dtype: Dtype,
    lora_dtype: Option<Dtype>,
    latent: usize,
) -> Result<(f32, LoraParams)> {
    let (x_t, target) = build_batch(x0, noise, t)?;
    let x_t = x_t.as_dtype(dtype)?; // no-op in f32 mode
    let timestep = Array::from_slice(&[t], &[1]);
    let feats: Vec<Array> = features
        .iter()
        .map(|f| f.as_dtype(dtype))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mask = mask.clone();
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        adapter.install_as(transformer, &p, alpha, rank, lora_dtype, LOKR_DTYPE)?;
        let v = transformer
            .forward(&x_t, &feats, Some(&mask), &timestep, 1, latent, latent)
            .map_err(|e| Exception::custom(e.to_string()))?;
        let diff = subtract(&v, &target)?;
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
            output_dir: PathBuf::from("/tmp/lens_unused"),
            file_name: "lora.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }

    #[test]
    fn descriptor_is_the_base_lens_id() {
        let d = trainer_descriptor();
        assert_eq!(d.id, "lens");
        assert_eq!(d.family, "lens");
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
    fn validate_rejects_gradient_checkpointing_with_pointer_not_noop() {
        let r = req_with(TrainingConfig {
            gradient_checkpointing: true,
            ..base_config()
        });
        let err = validate_request(&r).unwrap_err().to_string();
        assert!(err.contains("gradient_checkpointing"), "got: {err}");
        assert!(
            err.contains("follow-up"),
            "should point to the follow-up, not be a silent no-op"
        );
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
        // The recognized spellings (incl. alias normalization) pass.
        assert!(validate_request(&req_with(TrainingConfig {
            timestep_type: "Weighted".into(),
            timestep_bias: "high-noise".into(),
            loss_type: "L1".into(),
            optimizer: "adamw".into(),
            ..base_config()
        }))
        .is_ok());
    }

    #[test]
    fn build_batch_is_lens_velocity_with_no_sign_flip() {
        // target = noise − x0 (the RAW Lens DiT velocity; the OPPOSITE sign of z-image's negated
        // forward), and x_t = (1−t)·x0 + t·noise. Timestep `t` is passed straight to the DiT (the
        // caller), unlike z-image's `1 − σ` — covered by checking the interpolation here.
        let x0 = Array::from_slice(&[2.0f32, 4.0, 6.0], &[1, 3, 1]);
        let noise = Array::from_slice(&[1.0f32, 1.0, 1.0], &[1, 3, 1]);
        let t = 0.25f32;
        let (x_t, target) = build_batch(&x0, &noise, t).unwrap();
        // target = noise − x0 = [-1, -3, -5]
        assert_eq!(target.as_slice::<f32>(), &[-1.0, -3.0, -5.0]);
        // x_t = 0.75·x0 + 0.25·noise = [1.75, 3.25, 4.75]
        let xt = x_t.as_slice::<f32>();
        for (got, want) in xt.iter().zip([1.75f32, 3.25, 4.75].iter()) {
            assert!((got - want).abs() < 1e-6, "x_t {got} != {want}");
        }
    }

    #[test]
    fn sample_sigma_is_deterministic_and_in_range() {
        for kind in ["sigmoid", "linear", "weighted"] {
            for bias in ["balanced", "high", "low"] {
                let a = sample_sigma(kind, bias, 42).unwrap();
                let b = sample_sigma(kind, bias, 42).unwrap();
                assert_eq!(a, b, "{kind}/{bias} must be deterministic in seed");
                assert!(
                    (1e-3..=1.0 - 1e-3).contains(&a),
                    "{kind}/{bias} t={a} out of range"
                );
            }
        }
        // high-noise bias (√t) lifts the value vs low-noise bias (t²) for the same draw.
        assert!(
            sample_sigma("sigmoid", "high", 7).unwrap()
                > sample_sigma("sigmoid", "low", 7).unwrap()
        );
    }
}
