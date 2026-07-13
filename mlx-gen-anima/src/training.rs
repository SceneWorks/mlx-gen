//! LoRA/LoKr **training** on the Anima flow-match loop (sc-10522, epic 10512) — pure Rust on mlx-rs.
//!
//! Anima is trainable, so per standing guidance it ships with training. This realizes the core
//! [`Trainer`] contract (gen-core `train.rs`) on the real Anima model — the 28-block Cosmos-Predict2
//! DiT **and** the bundled `AnimaTextConditioner` (the `llm_adapter`). It is modeled on
//! `mlx-gen-z-image::training` (the closest analogue: a Qwen3-encoded flow-match DiT that already
//! trains LoRA + LoKr), and reuses the family-agnostic factor machinery in [`mlx_gen::train::lora`].
//!
//! ## Two hosts, one adapter surface
//! Unlike z-image (one DiT host), Anima adapters route into **two** sub-models via the sc-10521
//! [`AnimaAdapterHost`] seam: `llm_adapter.*` → the conditioner, everything else → the DiT. The
//! trainer enumerates its trainable targets from exactly that host, so the default trainable set is
//! the **508** targets the official `anima-turbo-lora-v0.2` carries — **448** DiT (`blocks.N.*`:
//! self/cross-attn q/k/v/o, mlp ×2, and the three `adaln_modulation_*.{1,2}` down/up pairs) + **60**
//! conditioner (6 blocks × {self/cross-attn q/k/v/o, mlp.0, mlp.2}). Training only the DiT would
//! produce a 448-target file that cannot reproduce the official injection surface.
//!
//! ## The conditioner genuinely trains
//! The conditioner (`llm_adapter`) is a first-class trainable target, so its adapter factors must
//! receive real gradients. We therefore **cache the conditioner's INPUTS** — the (masked) Qwen3
//! `last_hidden_state` + the T5 query-token ids, which are deterministic per caption and produced by
//! the multi-GB Qwen3 encoder (freed post-caching) — and run the (cheap, 6-block) conditioner forward
//! **inside the traced grad graph** each step, with the conditioner adapters injected. Caching the
//! conditioner's *output* instead would make its adapters inert (no gradient path), so the input-cache
//! is the faithful analogue of OneTrainer/mgds `EncodeAnimaText`: the expensive deterministic encoder
//! output is cached, the trained component stays live.
//!
//! ## Flow-match objective (`shift=3.0`)
//! Anima's DiT is a standard flow denoiser: it predicts the velocity `v ≈ ε − x0` and embeds the raw
//! (shifted) σ as its timestep (`pipeline.rs`), so the regression target is `noise − x0` with **no**
//! output negation (the opposite of z-image, whose `forward()` is pre-negated and whose timestep is
//! `1 − σ`). A base timestep is sampled, then run through the same static `shift = 3.0`
//! (`3σ/(1+2σ)`, [`SIGMA_SHIFT`]) the inference schedule uses, and that shifted σ drives both the
//! `x_t = (1−σ)·x0 + σ·noise` interpolation and the DiT timestep.
//!
//! ## Mid-run resume (sc-10642)
//! At each `save_every` the trainer additionally writes a resume bundle (the shared sc-9560 engine in
//! [`mlx_gen::train::checkpoint`]): the optimizer state + the raw trainable factors + `{step,
//! update_idx, optimizer}`. With `cfg.resume`, a fresh run of the same adapter restores that state and
//! continues from `step + 1`. On restore it **asserts the full 448 DiT + 60 `llm_adapter` = 508 target
//! surface** against the live model (never inferred from the file) — a checkpoint that had dropped the
//! 60 conditioner targets would resume an inert conditioner while every structural check still passed
//! (the sc-10522 trap), so [`assert_resume_surface_matches`] fails it loudly.
//!
//! ## Round-trip
//! The trained adapter round-trips through the sc-10521 inference loader: LoRA is saved with the
//! ComfyUI `diffusion_model.` prefix + PEFT `lora_A`/`lora_B` keys and **no alpha** (the shipped Anima
//! convention — the α/rank fold is baked into `lora_B` so scale-1.0 loading is exact for any alpha);
//! LoKr is saved by the shared [`save_lokr`] in the bare-path `lokr_*` convention the sc-10521 LoKr
//! path consumes. Both reconstruct the residual at **bf16** to match the inference loader.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use mlx_gen::adapters::{prefixed_paths, AdaptableHost};
use mlx_gen::gen_core;
use mlx_gen::media::Image;
use mlx_gen::train::checkpoint;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, average_grads, build_lokr_targets, build_lora_targets, save_lokr, LoraParams,
    TrainAdapter,
};
pub use mlx_gen::train::lora::{LokrTarget, LoraTarget};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::{
    LoadSpec, Modality, NetworkType, Precision, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{multiply, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::adapters::AnimaAdapterHost;
use crate::conditioner::AnimaTextConditioner;
use crate::config::{Variant, SIGMA_SHIFT};
use crate::loader::AnimaComponents;
use crate::pipeline::render_preview;
use crate::text_encoder::AnimaQwen3;
use crate::tokenizer::AnimaTokenizers;
use crate::transformer::CosmosDiT;
use crate::vae::QwenVae;

/// The inference LoKr loader (`apply_lokr`) reconstructs the Kronecker delta at bf16 (`loader.rs`);
/// training must match so the adapter round-trips.
const LOKR_DTYPE: Dtype = Dtype::Bfloat16;

/// Saved-adapter storage dtype. The trainable factors are f32 master-weights, but the shipped Anima
/// adapters (`anima-turbo-lora-v0.2`, `anima-greg-rutkowski-style`) are **bf16**, and the inference
/// loader reconstructs the residual at bf16 regardless — so the saved factors are cast to bf16 to
/// match the shipped convention and halve file size (~138 MB → ~69 MB), with no round-trip loss
/// beyond the bf16 rounding the loader would apply anyway.
const SAVE_DTYPE: Dtype = Dtype::Bfloat16;

/// ComfyUI adapter key prefix the official Anima LoRAs (and the sc-10521 inference loader) use —
/// `diffusion_model.blocks.*` for the DiT, `diffusion_model.llm_adapter.blocks.*` for the conditioner.
const KEY_PREFIX: &str = "diffusion_model.";

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence (sc-10641), matching
/// z-image's cap; extra prompts beyond this are ignored (a preview is a quick convergence check, not a
/// full sweep, and each one is an inference denoise on the live 2B DiT).
const SAMPLE_PROMPT_CAP: usize = 4;

// ==================================================================================================
// Flow-match batch construction
// ==================================================================================================

/// The static flow-match time-shift the Anima schedule applies (`shift·σ / (1 + (shift−1)·σ)`),
/// concentrating sampling toward higher noise. Mirrors `pipeline::anima_sigmas` so training and
/// inference share the same σ warp. `shift = 1` is the identity.
fn apply_static_shift(sigma: f32, shift: f32) -> f32 {
    shift * sigma / (1.0 + (shift - 1.0) * sigma)
}

/// `(x_t, target, timestep)` for a single sample at flow-match `sigma` (already shift-warped):
/// `x_t = (1−σ)·x0 + σ·noise`, `target = noise − x0` (the velocity `ε − x0`), `timestep = σ`.
/// Anima's `forward()` is **not** pre-negated (unlike z-image), so the target is the raw velocity and
/// the timestep is the raw σ (matching the inference `TimestepConvention::Sigma`).
fn build_batch(x0: &Array, noise: &Array, sigma: f32) -> Result<(Array, Array, f32)> {
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    let x_t = mlx_rs::ops::add(&multiply(x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, x0)?;
    Ok((x_t, target, sigma))
}

// ==================================================================================================
// Request validation (capability-free, unit-testable)
// ==================================================================================================

/// Recognized `timestep_type` values (`sigmoid` is the fall-through default).
const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "linear", "uniform", "weighted"];
/// Recognized `timestep_bias` values (`balanced`/`none`/`neutral` are neutral).
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
/// Recognized `loss_type` values (`mse`/`l2` = MSE, `mae`/`l1` = MAE).
const LOSS_TYPES: [&str; 4] = ["mse", "l2", "mae", "l1"];

/// Normalize a free-form config string the way the parsers do (trim, lowercase, `-`/space → `_`).
fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Capability-free training-request validation (mirrors z-image's `validate_request`): rejects an
/// empty dataset, zero rank, zero steps, an unsupported optimizer, and an unrecognized
/// `timestep_type` / `timestep_bias` / `loss_type` (rather than silently falling back to a default).
/// The non-empty target-module resolution is checked in [`Trainer::validate`], which has the loaded
/// model to enumerate adaptable paths against.
fn validate_request(req: &TrainingRequest) -> Result<()> {
    if req.items.is_empty() {
        return Err("anima trainer: dataset is empty".into());
    }
    if req.config.rank == 0 {
        return Err("anima trainer: rank must be > 0".into());
    }
    if req.config.steps == 0 {
        return Err("anima trainer: steps must be > 0".into());
    }
    if !TrainOptimizer::is_supported(&req.config.optimizer) {
        return Err(format!(
            "anima trainer: optimizer '{}' is not available on MLX training (supported: adamw, \
             adam, rose, prodigy)",
            req.config.optimizer
        )
        .into());
    }
    if !TIMESTEP_TYPES.contains(&normalize_cfg(&req.config.timestep_type).as_str()) {
        return Err(format!(
            "anima trainer: timestep_type '{}' is not recognized (supported: {})",
            req.config.timestep_type,
            TIMESTEP_TYPES.join(", ")
        )
        .into());
    }
    if !TIMESTEP_BIASES.contains(&normalize_cfg(&req.config.timestep_bias).as_str()) {
        return Err(format!(
            "anima trainer: timestep_bias '{}' is not recognized (supported: {})",
            req.config.timestep_bias,
            TIMESTEP_BIASES.join(", ")
        )
        .into());
    }
    if !LOSS_TYPES.contains(&normalize_cfg(&req.config.loss_type).as_str()) {
        return Err(format!(
            "anima trainer: loss_type '{}' is not recognized (supported: {})",
            req.config.loss_type,
            LOSS_TYPES.join(", ")
        )
        .into());
    }
    Ok(())
}

/// Resolve the config's target modules to the full dotted-path set on the Anima adapter surface (DiT
/// `blocks.*` + `llm_adapter.blocks.*`). An empty `lora_target_modules` (the default) trains the whole
/// **508**-target surface the official LoRAs carry; a non-empty set filters those paths by suffix (the
/// same suffix match PEFT's `LoraConfig(target_modules=…)` does). Computed from `&self` refs (no
/// `&mut` host needed) so [`Trainer::validate`] can call it. Mirrors [`AnimaAdapterHost::adaptable_paths`].
fn resolve_target_paths(
    dit: &CosmosDiT,
    conditioner: &AnimaTextConditioner,
    cfg: &TrainingConfig,
) -> Vec<String> {
    let mut all = dit.adaptable_paths();
    all.extend(prefixed_paths("llm_adapter", conditioner));
    if cfg.lora_target_modules.is_empty() {
        return all;
    }
    let suffixes = &cfg.lora_target_modules;
    all.into_iter()
        .filter(|path| {
            suffixes
                .iter()
                .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
}

/// The distinct adapter *target paths* present in a trainable-factor map, split into the DiT
/// (`blocks.*`) surface and the conditioner (`llm_adapter.*`) surface. Every factor is keyed
/// `{path}.<factor>` (`.lora_a`/`.lora_b`/`.lokr_w1`/`.lokr_w2`/`.lokr_w2_a`/`.lokr_w2_b`), so the
/// target path is the key with its trailing factor segment stripped. Used by the sc-10642 resume guard
/// to count the 448 DiT + 60 `llm_adapter` = 508 targets a full-surface run trains.
fn split_target_surface(params: &LoraParams) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut dit = BTreeSet::new();
    let mut cond = BTreeSet::new();
    for key in params.keys() {
        let path = key.rsplit_once('.').map_or(key.as_ref(), |(p, _)| p);
        if path.starts_with("llm_adapter.") {
            cond.insert(path.to_string());
        } else {
            dit.insert(path.to_string());
        }
    }
    (dit, cond)
}

/// The sc-10522 / sc-10642 restore guard. A resumed run rebuilds its trainable surface from the LIVE
/// model (`resolve_target_paths` → `build_lora/lokr_targets`, the full **508** targets = 448 DiT +
/// 60 `llm_adapter` for the default config) and then swaps the checkpoint's factors in. If the
/// checkpoint had dropped the 60 conditioner (`llm_adapter.*`) targets — or otherwise disagreed with
/// this run's surface — resume would silently continue with an **inert conditioner** while every
/// structural check (converging DiT loss, valid adapter file) still passed: the exact sc-10522 trap.
/// So before swapping the factors we ASSERT the checkpoint's factor surface is EXACTLY the one this run
/// rebuilt — same DiT paths, same `llm_adapter` paths, same factor keys — and fail loudly with a typed
/// [`Error`](mlx_gen::Error) otherwise, naming the DiT/conditioner target split so the mismatch is
/// diagnosable. The count is asserted against the live model's surface, never inferred from the file.
fn assert_resume_surface_matches(restored: &LoraParams, expected: &LoraParams) -> Result<()> {
    // Exact factor-key equality is the strongest guarantee — it also catches a LoRA↔LoKr network-type
    // mismatch or a single dropped factor, not just a wholesale missing target class.
    let restored_keys: BTreeSet<&str> = restored.keys().map(|k| k.as_ref()).collect();
    let expected_keys: BTreeSet<&str> = expected.keys().map(|k| k.as_ref()).collect();
    if restored_keys == expected_keys {
        return Ok(());
    }
    let (rd, rc) = split_target_surface(restored);
    let (ed, ec) = split_target_surface(expected);
    Err(mlx_gen::Error::Msg(format!(
        "anima resume: checkpoint trainable surface does not match this run — refusing to resume into a \
         silently-different model (sc-10522). checkpoint carries {rt} targets ({rdn} DiT `blocks.*` + \
         {rcn} `llm_adapter` conditioner); this run rebuilt {et} targets ({edn} DiT + {ecn} \
         conditioner). a checkpoint missing the {ecn} conditioner targets would resume an inert \
         conditioner — start a fresh run or resume the matching adapter.",
        rt = rd.len() + rc.len(),
        rdn = rd.len(),
        rcn = rc.len(),
        et = ed.len() + ec.len(),
        edn = ed.len(),
        ecn = ec.len(),
    )))
}

/// Whether in-training preview sampling (sc-10641) is active for this run: a positive cadence AND at
/// least one prompt AND not already cancelled. `false` ⇒ no conditioner-input pre-encode and no render,
/// so a run that does not opt in (the default) behaves exactly as before.
fn previews_enabled(cfg: &TrainingConfig, cancelled: bool) -> bool {
    cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() && !cancelled
}

/// Whether micro-step `step` (1-based) lands on the preview cadence. `sample_every == 0` never fires
/// (also guarded upstream by [`previews_enabled`], but kept total here so the predicate is self-contained).
fn preview_due(step: u32, sample_every: u32) -> bool {
    sample_every > 0 && step.is_multiple_of(sample_every)
}

// ==================================================================================================
// The trainer
// ==================================================================================================

/// A LoRA/LoKr trainer for one Anima variant: the frozen base (Cosmos DiT + `AnimaTextConditioner` +
/// Qwen3 TE + Qwen-Image VAE + dual tokenizers), which caches a captioned dataset to VAE latents +
/// Qwen3 conditioner-inputs, then runs the functional-autograd flow-match loop and writes an adapter
/// that round-trips through the sc-10521 inference loader.
pub struct AnimaTrainer {
    descriptor: TrainerDescriptor,
    /// Which Anima variant this trainer wraps — decides whether in-training previews (sc-10641) run CFG
    /// (base/aesthetic) or a single guidance-free forward (turbo).
    variant: Variant,
    tokenizers: AnimaTokenizers,
    /// The Qwen3 encoder — in an `Option` so it can be **dropped after caching** (it is idle during
    /// training; every caption is already encoded to its cached `source_hidden`, and it is a multi-GB
    /// resident). The conditioner it feeds is NOT freed — it is a trained target.
    text_encoder: Option<AnimaQwen3>,
    vae: QwenVae,
    dit: CosmosDiT,
    conditioner: AnimaTextConditioner,
}

fn trainer_descriptor_for(variant: Variant) -> TrainerDescriptor {
    TrainerDescriptor {
        id: variant.id(),
        family: "anima",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        // LoRA/LoKr only — no control-branch training path (F-006).
        supports_control: false,
    }
}

pub fn trainer_descriptor_base() -> TrainerDescriptor {
    trainer_descriptor_for(Variant::Base)
}
pub fn trainer_descriptor_aesthetic() -> TrainerDescriptor {
    trainer_descriptor_for(Variant::Aesthetic)
}
pub fn trainer_descriptor_turbo() -> TrainerDescriptor {
    trainer_descriptor_for(Variant::Turbo)
}

pub fn load_trainer_base(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load_variant_trainer(spec, Variant::Base)
}
pub fn load_trainer_aesthetic(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load_variant_trainer(spec, Variant::Aesthetic)
}
pub fn load_trainer_turbo(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    load_variant_trainer(spec, Variant::Turbo)
}

/// Construct the trainer from a `split_files/` snapshot (the multi-file Anima layout). No
/// quantization — training needs the dense bf16 base (the single wired precision).
fn load_variant_trainer(spec: &LoadSpec, variant: Variant) -> Result<Box<dyn Trainer>> {
    let id = variant.id();
    if spec.precision != Precision::Bf16 {
        return Err(mlx_gen::Error::Msg(format!(
            "{id} trainer: only the dense bf16 base is wired for training (drop the precision override)"
        )));
    }
    if spec.quantize.is_some() {
        return Err(mlx_gen::Error::Msg(format!(
            "{id} trainer: training needs the dense base; quantized tiers are not trainable"
        )));
    }
    let components = AnimaComponents::load(&spec.weights, variant)?;
    Ok(Box::new(AnimaTrainer {
        descriptor: trainer_descriptor_for(variant),
        variant,
        tokenizers: components.tokenizers,
        text_encoder: Some(components.text_encoder),
        vae: components.vae,
        dit: components.dit,
        conditioner: components.conditioner,
    }))
}

// Link-time trainer registration (epic 3720) for all three variants, mirroring the generator side.
mlx_gen::register_trainer! {
    trainer_descriptor_base => load_trainer_base,
    trainer_descriptor_aesthetic => load_trainer_aesthetic,
    trainer_descriptor_turbo => load_trainer_turbo,
}

impl Trainer for AnimaTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Shared control-training floor (F-006): a LoRA-only trainer must reject a control-branch
        // request (typed `Unsupported`) rather than silently training a plain adapter.
        gen_core::train::validate_control_request(self.descriptor(), req)?;
        validate_request(req)?;
        if resolve_target_paths(&self.dit, &self.conditioner, &req.config).is_empty() {
            return Err(format!(
                "anima trainer: lora_target_modules {:?} matched no adaptable module on the DiT or \
                 conditioner",
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

impl AnimaTrainer {
    /// The rich-`Result` body behind [`Trainer::train`]; the trait wrapper bridges its tail into
    /// [`gen_core::Error`] (epic 3720).
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        self.validate(req)?;
        let cfg = &req.config;
        on_progress(TrainingProgress::Preparing);
        let edge = bucket_resolution(cfg.resolution);

        // Anima's base is bf16 on disk (there is no dense-f32 cast path), so training runs bf16
        // mixed-precision: the frozen base + activation stream are bf16, and the trainable factors /
        // loss / grads / optimizer stay f32 (master-weights). This matches the inference dtype.
        let compute_dtype = Dtype::Bfloat16;

        // sc-10576 — fail-fast pre-flight memory guard. The dense (non-block-checkpointed) first step
        // materializes the whole DiT forward graph — at 1536² the retained per-block seq² self-attention
        // (≈9216 image tokens) makes that working set exceed unified memory, and the OS hard-kills the
        // worker with an UNCATCHABLE SIGKILL (the run just appears to hang at the last cached latent).
        // We predict it and refuse up front with an actionable, catchable error — BEFORE the (minutes-
        // long) latent caching — whenever gradient checkpointing is NOT enabled (LoRA checkpointing OR
        // the LoKr/dense fallback). With whole-block checkpointing on, the first step fits, so skip it.
        let will_checkpoint =
            matches!(cfg.network_type, NetworkType::Lora) && cfg.gradient_checkpointing;
        if !will_checkpoint {
            preflight_memory_guard(edge, compute_dtype == Dtype::Bfloat16)?;
        }

        // --- prepare → cache: VAE latents + (masked Qwen3 states, T5 ids) into memory ---
        on_progress(TrainingProgress::LoadingModel);
        let total = req.items.len() as u32;
        // (x0 latent, masked Qwen3 source_hidden, T5 query-token ids).
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
            let nchw = mlx_gen_qwen_image::preprocess_init_image(&img, edge, edge)?; // [1,3,edge,edge]
            let x0 = self.vae.encode(&nchw)?; // [1,16,1,edge/8,edge/8], normalized
            let (source, t5_ids) = self.encode_conditioner_inputs(&item.caption)?;
            eval([&x0, &source, &t5_ids])?;
            cache.push((x0, source, t5_ids));
        }
        if cache.is_empty() {
            if req.cancel.is_cancelled() {
                return Err(mlx_gen::Error::Canceled);
            }
            return Err("anima trainer: no usable dataset items".into());
        }

        // sc-10641 — pre-encode the preview-sample prompts' conditioner INPUTS while the Qwen3 encoder
        // is still resident (it is freed just below). We cache the conditioner INPUTS (masked Qwen3
        // states + T5 ids), NOT its output: the conditioner (`llm_adapter`) is a trained target, so each
        // preview must re-run it through the live graph to reflect its adapters (the sc-10522 trap — a
        // cached conditioner output would render silently-inert conditioner adapters). For CFG variants
        // (base/aesthetic) the empty-prompt uncond inputs are cached once too. Skipped when sampling is
        // off (the default) or the run is already cancelled.
        let previews_on = previews_enabled(cfg, req.cancel.is_cancelled());
        let sample_inputs: Vec<(String, Array, Array)> = if previews_on {
            let mut v = Vec::with_capacity(cfg.sample_prompts.len().min(SAMPLE_PROMPT_CAP));
            for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                let (source, t5_ids) = self.encode_conditioner_inputs(prompt)?;
                eval([&source, &t5_ids])?;
                v.push((prompt.clone(), source, t5_ids));
            }
            v
        } else {
            Vec::new()
        };
        // Uncond (empty-prompt) conditioner inputs — cached once, only for CFG variants at guidance ≠ 1.
        let uncond_inputs: Option<(Array, Array)> =
            if !sample_inputs.is_empty() && self.variant.uses_cfg() {
                let (s, ids) = self.encode_conditioner_inputs("")?;
                eval([&s, &ids])?;
                Some((s, ids))
            } else {
                None
            };

        // Every caption is encoded into `cache`; the multi-GB Qwen3 encoder is now dead weight for
        // the rest of the run. Drop it and evict its buffers before the train loop.
        self.text_encoder = None;
        mlx_rs::memory::clear_cache();

        // --- adapter targets + trainable factors (LoRA or LoKr) + optimizer ---
        let target_paths = resolve_target_paths(&self.dit, &self.conditioner, cfg);
        let rank = cfg.rank as f32;
        let (adapter, mut params) = {
            let mut host = AnimaAdapterHost {
                dit: &mut self.dit,
                conditioner: &mut self.conditioner,
            };
            match cfg.network_type {
                NetworkType::Lora => {
                    let (targets, params) =
                        build_lora_targets(&mut host, &target_paths, cfg.rank as i32, cfg.seed)?;
                    (TrainAdapter::Lora { targets }, params)
                }
                NetworkType::Lokr => {
                    let (targets, params) = build_lokr_targets(
                        &mut host,
                        &target_paths,
                        cfg.rank as i32,
                        cfg.decompose_factor,
                        cfg.seed,
                    )?;
                    (TrainAdapter::Lokr { targets }, params)
                }
            }
        };
        let alpha = cfg.alpha;
        let mae = {
            let lt = normalize_cfg(&cfg.loss_type);
            lt == "mae" || lt == "l1"
        };

        // sc-10576 — gradient checkpointing. Collect, per DiT block, the adapter-routable LOCAL paths
        // trained on it (`self_attn.q_proj`, `adaln_modulation_mlp.2`, …); the 28-block DiT stack is
        // where the first-step activation memory concentrates (its per-block seq² self-attention), so
        // that is what we whole-block checkpoint. The 60 conditioner (`llm_adapter.*`) targets are NOT
        // collected here — the conditioner runs UN-checkpointed inside the traced grad graph (only 512
        // text tokens), so its factors train through ordinary autograd, and — critically — its gradient
        // path stays live because the DiT checkpoints thread `encoder` as an explicit input (sc-10522).
        let n_layers = self.dit.config().num_layers;
        let block_local_targets = collect_dit_block_local_targets(&target_paths, n_layers);

        // Gradient checkpointing is an OPT-IN option, never auto-forced — a run that would OOM is caught
        // by the fail-fast pre-flight guard above (which recommends this flag) rather than silently
        // changing the user's training dynamics. Only the LoRA path is whole-block checkpointed today;
        // LoKr (a distinct Kronecker reconstruction) falls back to the dense path, exactly like z-image.
        let is_lora = matches!(adapter, TrainAdapter::Lora { .. });
        let use_checkpoint = is_lora && cfg.gradient_checkpointing;
        let checkpoint_blocks: Option<&[Vec<String>]> = if use_checkpoint {
            Some(&block_local_targets)
        } else {
            None
        };
        // SDPA-segment checkpointing: the conditioner is never whole-block checkpointed, so it keeps
        // segment ckpt ON (bounds its retained attention). The DiT keeps segment ckpt ON only when
        // whole-block checkpointing is OFF (the dense / LoKr path) — when whole-block is on, the block
        // recompute already covers attention and nesting would recompute it twice for no memory win.
        self.dit.set_sdpa_checkpoint(!use_checkpoint);
        self.conditioner.set_sdpa_checkpoint(true);

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
            .unwrap_or("adapter")
            .to_string();

        // sc-10642 — mid-run resume (the shared sc-9560 / F-125 engine). When `cfg.resume` is set and a
        // prior interrupted run of THIS adapter left a snapshot in `output_dir`, restore its optimizer
        // state + trainable factors + step/update index and continue from `start_step + 1` rather than
        // step 0. The restored factors REPLACE the fresh `B = 0`/zeroed init built above; the fresh build
        // still ran first so `params` carries the live model's full expected surface for the sc-10522
        // guard. Snapshots land on optimizer-update boundaries at each `save_every`, so resume is
        // bit-exact for `gradient_accumulation = 1` (the default) and when `save_every` is a multiple of
        // the accumulation otherwise.
        let mut update_idx: u32 = 0;
        let mut start_step: u32 = 0;
        if cfg.resume {
            if let Some((snapshot, _)) = checkpoint::find_latest_resume(&req.output_dir, &stem) {
                let (loaded, meta) = checkpoint::load_resume(&snapshot, &mut opt)?;
                // sc-10522: the checkpoint MUST carry the full 448 DiT + 60 `llm_adapter` surface this
                // run rebuilt, or the conditioner resumes inert. Assert against the live model's surface
                // BEFORE swapping the factors in — a typed error, not a silent inert-conditioner resume.
                assert_resume_surface_matches(&loaded, &params)?;
                params = loaded;
                start_step = meta.step;
                update_idx = meta.update_idx;
                eprintln!(
                    "[sc-10642] anima resuming '{stem}' from step {start_step} (optimizer update \
                     {update_idx})"
                );
            }
        }

        // --- train loop ---
        let mut accumulated: Option<LoraParams> = None;
        let mut last_loss = 0.0f32;
        let mut steps_run = start_step;
        for step in start_step + 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (x0, source, t5_ids) = &cache[((step - 1) as usize) % cache.len()];
            let sigma = sample_sigma(
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
                &mut self.dit,
                &mut self.conditioner,
                &params,
                &adapter,
                alpha,
                rank,
                x0,
                source,
                t5_ids,
                sigma,
                &noise,
                mae,
                checkpoint_blocks,
                compute_dtype,
            )?;
            last_loss = loss;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;

            if step % accum == 0 || step == cfg.steps {
                let mult =
                    lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                opt.set_lr_scaled(mult);
                // The final update can fire with fewer than `accum` grads; divide by the actual
                // in-window count so a short tail step isn't down-scaled.
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
                let ckpt = req
                    .output_dir
                    .join(intermediate_filename(&req.file_name, step));
                save_adapter(&adapter, &params, &target_paths, alpha, rank, cfg, &ckpt)?;
                // sc-10642 — alongside the user-facing adapter checkpoint, write the resume bundle
                // (optimizer state + raw trainable factors + `{step, update_idx, optimizer}`), so an
                // interrupted run can continue from here with `cfg.resume` rather than restarting at 0.
                checkpoint::save_resume(&req.output_dir, &stem, step, update_idx, &opt, &params)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }

            // sc-10641 — periodic preview samples from the in-progress adapter so the user can watch the
            // LoRA converge (the sc-5637 `TrainingProgress::Sample` contract). Install the CURRENT factors
            // as concrete adapters for the forward-only render (the traced `loss_fn` re-installs them at
            // the next step, so no teardown is needed), then render each cached sample prompt. The render
            // re-runs the conditioner through the LIVE graph (`render_preview` → `conditioner.forward`),
            // so the preview reflects the 60 `llm_adapter` conditioner adapters' training — never a cached
            // output (sc-10522). Best-effort: a render failure logs and continues the (long) run.
            if previews_on && preview_due(step, cfg.sample_every) {
                let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);
                {
                    let mut host = AnimaAdapterHost {
                        dit: &mut self.dit,
                        conditioner: &mut self.conditioner,
                    };
                    adapter.install_as(&mut host, &params, alpha, rank, lora_dtype, LOKR_DTYPE)?;
                }
                let guidance = if self.variant.uses_cfg() {
                    cfg.sample_guidance_scale
                } else {
                    1.0 // turbo is the merged CFG-free student — a single forward, guidance inert.
                };
                let total = sample_inputs.len() as u32;
                for (i, (prompt, source, t5_ids)) in sample_inputs.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    let sample_seed = cfg
                        .seed
                        .wrapping_add(step as u64)
                        .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                        .wrapping_add(i as u64);
                    match render_preview(
                        &self.dit,
                        &self.conditioner,
                        &self.vae,
                        source,
                        t5_ids,
                        uncond_inputs.as_ref(),
                        cfg.sample_steps.max(1) as usize,
                        guidance,
                        edge,
                        sample_seed,
                        compute_dtype,
                        &req.cancel,
                    ) {
                        Ok(image) => on_progress(TrainingProgress::Sample {
                            step,
                            index: i as u32 + 1,
                            total,
                            prompt: prompt.clone(),
                            image,
                        }),
                        // F-117: a cancelled preview denoise exits the preview loop (the outer loop's
                        // cancel check then unwinds the run); other failures skip one preview.
                        Err(mlx_gen::Error::Canceled) => break,
                        Err(e) => eprintln!(
                            "[sc-10641] anima preview sample failed at step {step} (prompt {}): {e} \
                             — skipping this preview, training continues",
                            i + 1
                        ),
                    }
                }
                // sc-5567 — release the preview's transient forward+VAE-decode residency before
                // training resumes. MLX pools freed buffers, so a full `sample_steps` denoise plus
                // VAE decode on the 2B DiT can push peak residency above the train steady state and
                // SIGKILL a tightly-budgeted run (sc-10576 memory focus). Preview path only — the
                // training-step working set is untouched.
                mlx_rs::memory::clear_cache();
            }
        }

        // Cancelled before completing a single step: the factors are still the no-op init (LoRA
        // `B = 0` / LoKr zeroed `w1`). Surface the cancellation as the typed `Error::Canceled` rather
        // than writing a valid-looking identity adapter and returning `Ok`.
        if steps_run == 0 {
            return Err(mlx_gen::Error::Canceled);
        }

        // --- save final adapter ---
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_adapter(
            &adapter,
            &params,
            &target_paths,
            alpha,
            rank,
            cfg,
            &adapter_path,
        )?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }

    /// Encode a caption to the conditioner's **inputs**: the mask-multiplied Qwen3 `last_hidden_state`
    /// `[1, S, 1024]` (bf16) and the T5 query-token ids `[1, St]` (int32) — the deterministic,
    /// cacheable half of the text path (mirrors `pipeline::encode_prompt` up to the conditioner). The
    /// conditioner itself is run per-step in the grad graph (it is a trained target), so its output is
    /// deliberately NOT cached.
    fn encode_conditioner_inputs(&self, caption: &str) -> Result<(Array, Array)> {
        let te = self.text_encoder.as_ref().ok_or_else(|| {
            mlx_gen::Error::Msg(
                "anima trainer: text encoder already freed (caching after loop)".into(),
            )
        })?;
        let (qwen_ids, qwen_mask) = self.tokenizers.encode_qwen(caption)?;
        let source = te.forward(&qwen_ids, &qwen_mask)?; // [1, S, 1024] bf16
        let mask = qwen_mask.as_dtype(source.dtype())?.expand_dims(2)?; // [1, S, 1]
        let source = multiply(&source, &mask)?;
        let t5_ids = self.tokenizers.encode_t5(caption)?; // [1, St]
        Ok((source, t5_ids))
    }
}

/// Dispatch the save: LoRA is written with the ComfyUI `diffusion_model.` prefix, PEFT `lora_A`/
/// `lora_B` keys, and NO alpha (the α/rank fold baked into `lora_B` — the shipped Anima convention);
/// LoKr uses the shared [`save_lokr`] (bare `lokr_*` keys the sc-10521 LoKr path consumes).
fn save_adapter(
    adapter: &TrainAdapter,
    params: &LoraParams,
    target_paths: &[String],
    alpha: f32,
    rank: f32,
    cfg: &TrainingConfig,
    path: &Path,
) -> Result<()> {
    match adapter {
        TrainAdapter::Lora { .. } => save_anima_lora(params, target_paths, alpha, rank, path),
        TrainAdapter::Lokr { targets } => {
            // Store the Kronecker factors bf16 ([`SAVE_DTYPE`]) too — the inference LoKr loader
            // reconstructs the delta at bf16 ([`LOKR_DTYPE`]) regardless, so casting the f32 master
            // factors here is round-trip-lossless and halves the file, matching the LoRA convention.
            let bf16: LoraParams = params
                .iter()
                .map(|(k, v)| Ok((k.clone(), v.as_dtype(SAVE_DTYPE)?)))
                .collect::<Result<_>>()?;
            save_lokr(&bf16, targets, alpha, rank, cfg.decompose_factor, path)
        }
    }
}

/// Write the trainable LoRA factors in the shipped Anima convention: keys
/// `diffusion_model.{path}.lora_A.weight` `[r,in]` / `diffusion_model.{path}.lora_B.weight` `[out,r]`,
/// with the `alpha/rank` scale **baked into `lora_B`** so the file carries no alpha (α = r ⇒ the
/// sc-10521 inference loader applies it at scale 1.0, exactly reproducing the trained residual). The
/// factors are stored **bf16** ([`SAVE_DTYPE`]) and the metadata is `{"format":"pt"}` only, matching
/// the official `anima-turbo-lora-v0.2` `__metadata__` and dtype.
fn save_anima_lora(
    params: &LoraParams,
    target_paths: &[String],
    alpha: f32,
    rank: f32,
    path: &Path,
) -> Result<()> {
    let scale = Array::from_slice(&[alpha / rank], &[1]);
    // Own the baked `lora_B` arrays so their borrows outlive the entry list.
    let mut owned: Vec<(String, Array)> = Vec::with_capacity(target_paths.len() * 2);
    for p in target_paths {
        let a = params
            .get(format!("{p}.lora_a").as_str())
            .ok_or_else(|| mlx_gen::Error::Msg(format!("LoRA param missing: {p}.lora_a")))?;
        let b = params
            .get(format!("{p}.lora_b").as_str())
            .ok_or_else(|| mlx_gen::Error::Msg(format!("LoRA param missing: {p}.lora_b")))?;
        owned.push((
            format!("{KEY_PREFIX}{p}.lora_A.weight"),
            a.as_dtype(SAVE_DTYPE)?,
        ));
        owned.push((
            format!("{KEY_PREFIX}{p}.lora_B.weight"),
            multiply(b, &scale)?.as_dtype(SAVE_DTYPE)?,
        ));
    }
    let entries: Vec<(String, &Array)> = owned.iter().map(|(k, v)| (k.clone(), v)).collect();
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".to_string(), "pt".to_string());
    Array::save_safetensors(entries, Some(&meta), path)?;
    Ok(())
}

/// `{stem}-step{step}{ext}` — the intermediate-checkpoint name for `save_every`.
fn intermediate_filename(file_name: &str, step: u32) -> String {
    let p = Path::new(file_name);
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("safetensors");
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("adapter");
    format!("{stem}-step{step}.{ext}")
}

/// Decode an image file (PNG/JPEG) into the core RGB8 [`Image`].
fn decode_image(path: &Path) -> Result<Image> {
    let dynimg = image::open(path)
        .map_err(|e| mlx_gen::Error::Msg(format!("decode image {}: {e}", path.display())))?;
    let rgb = dynimg.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

/// Sample a normalised flow-match base timestep `σ ∈ [1e-3, 1−1e-3]` (a faithful port of the
/// SceneWorks `sample_training_timestep`: `sigmoid(randn)` default, `uniform` for linear,
/// `(uniform + sigmoid(randn))/2` weighted; bias `high` → `√σ`, `low` → `σ²`), then run it through
/// the static `shift = 3.0` warp the inference schedule uses. Deterministic in `seed`.
fn sample_sigma(timestep_type: &str, timestep_bias: &str, seed: u64) -> Result<f32> {
    let k1 = random::key(seed)?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let ttype = normalize_cfg(timestep_type);
    let t = match ttype.as_str() {
        "linear" | "uniform" => {
            random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>()
        }
        "weighted" => {
            let k2 = random::key(seed.wrapping_add(0x9E37_79B9))?;
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
    let shifted = apply_static_shift(t, SIGMA_SHIFT);
    Ok(shifted.clamp(1e-3, 1.0 - 1e-3))
}

/// Image tokens the DiT self-attends at a square training `edge`: VAE /8 then patch /2 ⇒ `edge/16`
/// per side, squared. The `+512` is the conditioner's fixed padded text length (cross-attended, not
/// self-attended) — folded in so `s` is the "total token" proxy the projection is fit against.
fn unified_tokens(edge: u32) -> f64 {
    let per_side = (edge as f64 / 16.0).ceil();
    per_side * per_side + 512.0
}

/// Projected DENSE first-step peak GPU memory, in GB, as a function of the token proxy `s`
/// ([`unified_tokens`]) — an empirical fit to peaks measured on the 128 GB target with the Anima base.
///
/// Structure follows the z-image / sc-4874 decomposition `weights + linear·s + quad·s²`: the constant
/// is the resident base (the ~2B-param bf16 Cosmos DiT + the small conditioner + Qwen-Image VAE, after
/// the Qwen3 text encoder is freed post-caching), the linear term is the per-token hidden-state
/// activations retained across the 28 blocks (+ the s·512 cross-attention), and the quadratic term is
/// the residual seq² self-attention. Since the DENSE training path runs with SDPA-segment
/// checkpointing ON ([`AnimaTrainer::train_impl`] sets `dit.set_sdpa_checkpoint(!use_checkpoint)`), the
/// quadratic term is demoted from "one retained `[16-heads, s, s]` matrix per block" to a single
/// layer's backward transient. bf16 roughly halves the weights + activation terms vs an f32 base
/// (Anima has no f32 base, but the parameter is kept for symmetry with the z-image guard).
///
/// **Calibrated** against `first_step_dense_peak_sweep` (128 GB Mac, rank 16, batch 1) — see that
/// `#[ignore]`d test and the `preflight_tests` fit check; refit both if it prints materially different
/// numbers.
fn projected_dense_peak_gb(s: f64, bf16: bool) -> f64 {
    if bf16 {
        ANIMA_PEAK_CONST_BF16 + ANIMA_PEAK_LINEAR_BF16 * s + ANIMA_PEAK_QUAD_BF16 * s * s
    } else {
        // No f32 Anima base exists; a ~1.8× scale of the bf16 fit is a conservative upper bound.
        1.8 * (ANIMA_PEAK_CONST_BF16 + ANIMA_PEAK_LINEAR_BF16 * s + ANIMA_PEAK_QUAD_BF16 * s * s)
    }
}

// sc-10576 memory-projection coefficients — an exact `a + b·s + c·s²` fit to THREE measured Anima
// dense first-step peaks (edge 512/768/1024 ⇒ s 1536/2816/4608 ⇒ 21.2/32.0/49.4 GB, bf16, rank16) on
// the 128 GB target (`first_step_dense_peak_sweep`). The linear term dominates (per-token activations
// retained across the 28 blocks); the quadratic is small because the dense path runs SDPA-segment
// checkpointing. Extrapolates to ~113 GB at edge 1536 (s 9728) — over this machine's working set, so
// the guard refuses a dense 1536² run. CALIBRATED — re-run the sweep + `preflight_tests` fit check if
// the model/activation shape changes; do not hand-edit.
const ANIMA_PEAK_CONST_BF16: f64 = 9.84;
const ANIMA_PEAK_LINEAR_BF16: f64 = 6.775e-3;
const ANIMA_PEAK_QUAD_BF16: f64 = 3.946e-7;

/// Refuse a run whose dense first step would exceed this machine's memory budget (and thus get
/// SIGKILLed), returning a catchable, actionable error instead. The budget is MLX's own reported
/// memory limit (≈ the device's recommended working set), scaled by 0.85 for worker/host headroom —
/// exceeding it is the regime where the dense run dies. Only consulted when whole-block gradient
/// checkpointing is OFF.
fn preflight_memory_guard(edge: u32, bf16: bool) -> Result<()> {
    let s = unified_tokens(edge);
    let projected = projected_dense_peak_gb(s, bf16);
    let budget_gb = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);
    let safe = budget_gb * 0.85;
    if projected > safe {
        return Err(format!(
            "anima trainer: a dense first training step at resolution {edge} needs ~{projected:.0} GB \
             (the DiT forward working set materializes in one allocation), exceeding this machine's \
             ~{safe:.0} GB safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the OS \
             would hard-kill the worker (SIGKILL) at the first step with no recoverable error \
             (sc-10576). Enable Gradient Checkpointing (recomputes block activations in the backward) \
             or reduce the training resolution."
        )
        .into());
    }
    Ok(())
}

/// Per-DiT-block LOCAL LoRA target paths (`block_local_targets[i]` for block `i`), extracted from the
/// combined `target_paths`: keep only the DiT `blocks.{i}.{local}` entries (the conditioner's
/// `llm_adapter.blocks.…` entries are deliberately excluded — the conditioner is never whole-block
/// checkpointed). Mirrors z-image's `main_block_local_targets` collection; the order per block matches
/// the params keys `blocks.{i}.{local}.lora_a` that [`build_lora_targets`] produced.
fn collect_dit_block_local_targets(target_paths: &[String], n_layers: usize) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = vec![Vec::new(); n_layers];
    for path in target_paths {
        if path.starts_with("llm_adapter.") {
            continue; // conditioner target — trains through ordinary autograd, not checkpointed
        }
        if let Some((idx, local)) = path.strip_prefix("blocks.").and_then(|r| r.split_once('.')) {
            if let Ok(i) = idx.parse::<usize>() {
                if i < n_layers {
                    out[i].push(local.to_string());
                }
            }
        }
    }
    out
}

/// One forward+backward over the trainable adapter factors: inject `params` (LoRA or LoKr) onto BOTH
/// the DiT and the conditioner, run the conditioner (→ `encoder_hidden_states`) then the DiT, regress
/// the velocity `forward()` output toward `noise − x0`, return `(loss, grads)`. The conditioner runs
/// **inside** the traced graph, so its adapter factors receive gradients. `dtype` is the bf16 compute
/// dtype: `x_t` is cast at entry, the LoRA factors are cast inside the traced install, and the DiT/
/// conditioner run bf16; the noising math, loss, and grads stay f32.
///
/// `checkpoint_blocks`, when `Some`, lists per-DiT-block LOCAL LoRA target paths and switches the DiT
/// forward to the gradient-checkpointed path (sc-10576) — each block recomputes its activations in the
/// backward instead of retaining them, and threads `encoder` as an explicit checkpoint input so the
/// conditioner keeps its gradient. `None` runs the dense (activation-retaining) DiT forward.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &mut CosmosDiT,
    conditioner: &mut AnimaTextConditioner,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    source: &Array,
    t5_ids: &Array,
    sigma: f32,
    noise: &Array,
    mae: bool,
    checkpoint_blocks: Option<&[Vec<String>]>,
    dtype: Dtype,
) -> Result<(f32, LoraParams)> {
    let (x_t, target, timestep) = build_batch(x0, noise, sigma)?;
    let x_t = x_t.as_dtype(dtype)?;
    let src = source.clone();
    let ids = t5_ids.clone();
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // Install ALL targets (DiT + conditioner) via the combined host, then drop the host so the
        // `&self` forwards can borrow the two sub-models. On the checkpointed path the DiT block
        // adapters installed here are simply REPLACED inside each checkpoint segment by the explicit-
        // input factors (so they cost nothing there); the conditioner adapters always train through
        // this install (ordinary autograd). F-149: NEVER check the cancel flag inside this traced
        // closure (it would be stringified through `Exception::custom` and lose the typed
        // `Error::Canceled`); cancellation is the caller's job at the step boundary.
        {
            let mut host = AnimaAdapterHost {
                dit: &mut *dit,
                conditioner: &mut *conditioner,
            };
            adapter.install_as(&mut host, &p, alpha, rank, lora_dtype, LOKR_DTYPE)?;
        }
        let enc = conditioner
            .forward(&src, &ids, dtype)
            .map_err(|e| Exception::custom(e.to_string()))?;
        let s = Array::from_slice(&[timestep], &[1]);
        let v = match checkpoint_blocks {
            Some(blocks) => dit
                .forward_with_main_checkpointed(&x_t, &s, &enc, dtype, &p, blocks, alpha)
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => dit
                .forward(&x_t, &s, &enc, dtype)
                .map_err(|e| Exception::custom(e.to_string()))?,
        };
        let v = v.as_dtype(Dtype::Float32)?;
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
    use mlx_gen::{TrainingItem, TrainingRequest};
    use std::path::PathBuf;

    fn request(items: usize, steps: u32, rank: u32) -> TrainingRequest {
        TrainingRequest {
            items: (0..items)
                .map(|i| TrainingItem {
                    image_path: PathBuf::from(format!("img{i}.png")),
                    caption: "1girl, silver hair".into(),
                    control_image_path: None,
                })
                .collect(),
            config: TrainingConfig {
                steps,
                rank,
                ..Default::default()
            },
            output_dir: PathBuf::from("/tmp/anima-trainer-test"),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: Default::default(),
        }
    }

    #[test]
    fn three_trainer_variants_registered() {
        for id in ["anima_base", "anima_aesthetic", "anima_turbo"] {
            assert!(
                gen_core::registry::trainers().any(|r| (r.descriptor)().id == id),
                "trainer id {id} not registered"
            );
        }
    }

    #[test]
    fn descriptor_advertises_lora_and_lokr() {
        let d = trainer_descriptor_base();
        assert_eq!(d.id, "anima_base");
        assert_eq!(d.family, "anima");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.supports_lora && d.supports_lokr);
    }

    #[test]
    fn validate_request_guards() {
        assert!(validate_request(&request(1, 100, 16)).is_ok());
        assert!(validate_request(&request(0, 100, 16)).is_err()); // empty dataset
        assert!(validate_request(&request(1, 0, 16)).is_err()); // zero steps
        assert!(validate_request(&request(1, 100, 0)).is_err()); // zero rank
    }

    #[test]
    fn validate_rejects_unrecognized_schedule_and_loss() {
        let with = |f: fn(&mut TrainingConfig)| {
            let mut r = request(1, 100, 16);
            f(&mut r.config);
            validate_request(&r)
        };
        assert!(with(|c| c.timestep_type = "sgmoid".into()).is_err());
        assert!(with(|c| c.timestep_bias = "hihg_noise".into()).is_err());
        assert!(with(|c| c.loss_type = "huber".into()).is_err());
        assert!(with(|c| c.optimizer = "sophia".into()).is_err());
        // Documented spellings still pass, case/separator-insensitively.
        assert!(with(|c| c.timestep_type = "Linear".into()).is_ok());
        assert!(with(|c| c.timestep_bias = "High-Noise".into()).is_ok());
        assert!(with(|c| c.loss_type = "L1".into()).is_ok());
    }

    #[test]
    fn static_shift_matches_inference_schedule() {
        // shift(σ)=3σ/(1+2σ): shift(1)=1, shift(0)=0, shift(0.5)=1.5/2=0.75.
        assert!((apply_static_shift(1.0, SIGMA_SHIFT) - 1.0).abs() < 1e-6);
        assert!((apply_static_shift(0.0, SIGMA_SHIFT)).abs() < 1e-6);
        assert!((apply_static_shift(0.5, SIGMA_SHIFT) - 0.75).abs() < 1e-6);
        // Monotone increasing on [0,1].
        assert!(apply_static_shift(0.2, SIGMA_SHIFT) < apply_static_shift(0.8, SIGMA_SHIFT));
    }

    #[test]
    fn build_batch_is_flow_match_velocity() {
        // x_t = (1-σ)x0 + σ·noise; target = noise - x0; timestep = σ (raw, not 1-σ).
        let x0 = Array::from_slice(&[2.0f32, 4.0], &[1, 2]);
        let noise = Array::from_slice(&[1.0f32, 1.0], &[1, 2]);
        let (x_t, target, ts) = build_batch(&x0, &noise, 0.25).unwrap();
        assert!((ts - 0.25).abs() < 1e-6);
        // x_t = 0.75*[2,4] + 0.25*[1,1] = [1.75, 3.25]
        assert!((x_t.as_slice::<f32>()[0] - 1.75).abs() < 1e-5);
        assert!((x_t.as_slice::<f32>()[1] - 3.25).abs() < 1e-5);
        // target = [1,1] - [2,4] = [-1,-3]
        assert_eq!(target.as_slice::<f32>(), &[-1.0, -3.0]);
    }

    /// CI-runnable, **weights-free** guard on the trainable-target surface (sc-10522). Builds the DiT +
    /// conditioner *structurally* (placeholder 1×1 weights, no licensed checkpoint, no Metal compute),
    /// then asserts the trainer enumerates exactly **508** targets = **448** DiT (`blocks.*`) + **60**
    /// conditioner (`llm_adapter.blocks.*`). Deliberately **structural** (path strings, no numerics),
    /// so — unlike a Metal golden — it cannot go device-dependently red: a regression that drops the
    /// conditioner leg (the sc-10274 partial-injection class) collapses this to 448 and fails here.
    /// This is the always-on analogue of the real-weights `trainable_surface_is_508_*` (which is
    /// `#[ignore]`d and needs the snapshot); it runs under a plain `cargo test -p mlx-gen-anima`.
    #[test]
    fn trainable_surface_is_508_dit_plus_60_conditioner_weightsfree() {
        use crate::config::{ConditionerConfig, DitConfig};
        let dit = CosmosDiT::structural(DitConfig::anima());
        let cond = AnimaTextConditioner::structural(ConditionerConfig::anima());

        // Empty `lora_target_modules` ⇒ the full 508-target surface the official Anima LoRAs carry.
        let cfg = TrainingConfig::default();
        let paths = resolve_target_paths(&dit, &cond, &cfg);
        assert_eq!(
            paths.len(),
            508,
            "full trainable surface must be 508 (448 DiT + 60 conditioner), got {}",
            paths.len()
        );

        let cond_paths: Vec<&String> = paths
            .iter()
            .filter(|p| p.starts_with("llm_adapter."))
            .collect();
        let dit_paths: Vec<&String> = paths
            .iter()
            .filter(|p| !p.starts_with("llm_adapter."))
            .collect();
        assert_eq!(
            cond_paths.len(),
            60,
            "conditioner (llm_adapter) must contribute exactly 60 targets — dropping it is the \
             sc-10274 partial-injection regression"
        );
        assert_eq!(
            dit_paths.len(),
            448,
            "DiT must contribute exactly 448 targets (28 blocks × 16)"
        );
        assert!(
            cond_paths
                .iter()
                .all(|p| p.starts_with("llm_adapter.blocks.")),
            "every conditioner target must be a per-block llm_adapter path"
        );
        assert!(
            dit_paths.iter().all(|p| p.starts_with("blocks.")),
            "every DiT target must be a per-block path"
        );

        // A non-empty target filter narrows the surface by PEFT-style suffix match, and must stay
        // non-empty and strictly narrower than the full surface.
        let filtered = TrainingConfig {
            lora_target_modules: vec![
                "q_proj".into(),
                "k_proj".into(),
                "v_proj".into(),
                "output_proj".into(),
            ],
            ..Default::default()
        };
        let attn_only = resolve_target_paths(&dit, &cond, &filtered);
        assert!(
            !attn_only.is_empty() && attn_only.len() < paths.len(),
            "suffix filter must narrow the surface, got {}",
            attn_only.len()
        );
        assert!(
            attn_only
                .iter()
                .all(|p| ["q_proj", "k_proj", "v_proj", "output_proj"]
                    .iter()
                    .any(|s| p.ends_with(&format!(".{s}")))),
            "every filtered path must match a requested suffix"
        );
    }

    // ======================================================================================
    // sc-10576 — gradient checkpointing + pre-flight OOM guard
    // ======================================================================================

    use crate::config::{ConditionerConfig, DitConfig};

    /// A tiny but structurally-complete DiT config (2 heads × 8 = hidden 16, 2 blocks) for a
    /// Metal-cheap grad-parity model. `text_embed_dim` MUST equal the conditioner `target_dim` (the
    /// DiT cross-attends the conditioner output).
    fn tiny_dit_cfg() -> DitConfig {
        DitConfig {
            in_channels: 4,
            out_channels: 4,
            num_attention_heads: 2,
            attention_head_dim: 8,
            num_layers: 2,
            mlp_ratio: 2.0,
            text_embed_dim: 16,
            adaln_lora_dim: 8,
            max_size: (4, 16, 16),
            patch_size: (1, 2, 2),
            rope_scale: (1.0, 4.0, 4.0),
            concat_padding_mask: true,
        }
    }

    fn tiny_cond_cfg() -> ConditionerConfig {
        ConditionerConfig {
            source_dim: 16,
            target_dim: 16,
            model_dim: 16,
            num_layers: 2,
            num_attention_heads: 2,
            mlp_ratio: 2.0,
            target_vocab_size: 32,
            min_sequence_length: 8,
            rope_theta: 10000.0,
            norm_eps: 1e-6,
        }
    }

    /// `(x0 latent, source Qwen states, T5 ids, noise)` for the tiny model, all f32 (the parity test
    /// runs f32 so the fp tolerance isn't loosened by bf16 rounding).
    fn tiny_inputs(
        dcfg: &DitConfig,
        ccfg: &ConditionerConfig,
        edge: i32,
    ) -> (Array, Array, Array, Array) {
        let hl = edge / 8;
        let x0 = random::normal::<f32>(
            &[1, dcfg.in_channels as i32, 1, hl, hl],
            None,
            None,
            Some(&random::key(11).unwrap()),
        )
        .unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(12).unwrap())).unwrap();
        let source = random::normal::<f32>(
            &[1, 6, ccfg.source_dim as i32],
            None,
            None,
            Some(&random::key(13).unwrap()),
        )
        .unwrap();
        let ids_f = random::uniform::<_, f32>(
            0.0,
            ccfg.target_vocab_size as f32,
            &[1, 4],
            Some(&random::key(14).unwrap()),
        )
        .unwrap();
        let t5_ids = ids_f.as_dtype(Dtype::Int32).unwrap();
        eval([&x0, &noise, &source, &t5_ids]).unwrap();
        (x0, source, t5_ids, noise)
    }

    /// Σ|grad| over the conditioner (`llm_adapter.*`) `lora_b` factors — non-zero iff the encoder
    /// gradient path is live (lora_a starts at B=0, so its grad is 0; lora_b carries the signal).
    fn cond_lora_b_grad(g: &LoraParams) -> f32 {
        g.iter()
            .filter(|(k, _)| k.starts_with("llm_adapter.") && k.ends_with(".lora_b"))
            .map(|(_, v)| v.abs().unwrap().sum(None).unwrap().item::<f32>())
            .sum()
    }

    /// Σ|grad| over the DiT (`blocks.*`) `lora_b` factors.
    fn dit_lora_b_grad(g: &LoraParams) -> f32 {
        g.iter()
            .filter(|(k, _)| !k.starts_with("llm_adapter.") && k.ends_with(".lora_b"))
            .map(|(_, v)| v.abs().unwrap().sum(None).unwrap().item::<f32>())
            .sum()
    }

    /// Max relative grad diff between two param maps (per key: `‖a−b‖∞ / max(‖a‖∞, 1e-6)`).
    fn max_rel_diff(ga: &LoraParams, gb: &LoraParams) -> f32 {
        let mut m = 0f32;
        for (k, a) in ga {
            let b = gb.get(k).expect("same keys");
            let num = a
                .subtract(b)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            let den = a.abs().unwrap().max(None).unwrap().item::<f32>().max(1e-6);
            m = m.max(num / den);
        }
        m
    }

    /// Build a tiny synthetic (DiT + conditioner) and the combined LoRA factor surface on it.
    #[allow(clippy::type_complexity)]
    fn tiny_model_and_adapter() -> (
        CosmosDiT,
        AnimaTextConditioner,
        LoraParams,
        TrainAdapter,
        Vec<String>,
        Vec<Vec<String>>,
    ) {
        let dcfg = tiny_dit_cfg();
        let ccfg = tiny_cond_cfg();
        let mut dit = CosmosDiT::synthetic(dcfg, 42);
        let mut cond = AnimaTextConditioner::synthetic(ccfg, 43);
        let tcfg = TrainingConfig {
            rank: 4,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&dit, &cond, &tcfg);
        let (targets, params) = {
            let mut host = AnimaAdapterHost {
                dit: &mut dit,
                conditioner: &mut cond,
            };
            build_lora_targets(&mut host, &target_paths, 4, 7).unwrap()
        };
        let blocks = collect_dit_block_local_targets(&target_paths, dcfg.num_layers);
        (
            dit,
            cond,
            params,
            TrainAdapter::Lora { targets },
            target_paths,
            blocks,
        )
    }

    /// Grad-parity: whole-block checkpointed grads == dense grads for BOTH the DiT and the conditioner
    /// (`llm_adapter`) factors, to fp tolerance. Synthetic small model, Metal, no real weights — the
    /// always-on analogue of z-image's `#[ignore]`d `checkpointed_grads_match_dense`. Also asserts the
    /// conditioner actually receives gradient (the encoder path is live) in both legs.
    #[test]
    fn checkpointed_grads_match_dense_dit_and_conditioner() {
        let (mut dit, mut cond, params, adapter, _tp, blocks) = tiny_model_and_adapter();
        // Hold every SDPA-segment flag OFF so the ONLY difference between the two legs is whole-block
        // checkpointing of the DiT (isolates the sc-10522 encoder-threading correctness).
        dit.set_sdpa_checkpoint(false);
        cond.set_sdpa_checkpoint(false);
        let (x0, source, t5_ids, noise) = tiny_inputs(&tiny_dit_cfg(), &tiny_cond_cfg(), 32);

        let grads_of =
            |dit: &mut CosmosDiT, cond: &mut AnimaTextConditioner, ck: Option<&[Vec<String>]>| {
                let (_l, g) = compute_loss_grads(
                    dit,
                    cond,
                    &params,
                    &adapter,
                    4.0,
                    4.0,
                    &x0,
                    &source,
                    &t5_ids,
                    0.5,
                    &noise,
                    false,
                    ck,
                    Dtype::Float32,
                )
                .unwrap();
                eval(g.values()).unwrap();
                g
            };
        let g_dense = grads_of(&mut dit, &mut cond, None);
        let g_ckpt = grads_of(&mut dit, &mut cond, Some(&blocks));

        let max_rel = max_rel_diff(&g_dense, &g_ckpt);
        eprintln!("[sc-10576] checkpointed-vs-dense grad max rel diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-3,
            "checkpointed grads must match dense within fp tolerance: max rel {max_rel:.2e}"
        );
        // Both legs must actually train the conditioner (proves encoder grad flows) and the DiT.
        assert!(
            cond_lora_b_grad(&g_dense) > 1e-6,
            "conditioner must receive gradient (dense)"
        );
        assert!(
            cond_lora_b_grad(&g_ckpt) > 1e-6,
            "conditioner must receive gradient (checkpointed) — the sc-10522 inert-adapter trap"
        );
        assert!(
            dit_lora_b_grad(&g_ckpt) > 1e-6,
            "DiT must receive gradient (checkpointed)"
        );
    }

    /// Grad-parity in the EXACT production combined config (sc-10576). `train_impl` runs a checkpointed
    /// LoRA step with THREE flags at once: DiT whole-block checkpoint ON, DiT sdpa-segment OFF
    /// (`dit.set_sdpa_checkpoint(!use_checkpoint)` → OFF), and conditioner sdpa-segment ON
    /// (`conditioner.set_sdpa_checkpoint(true)`). The other grad-parity tests exercise those mechanisms
    /// only in isolation (whole-block with cond-segment OFF; sdpa-segment on the dense path), so none of
    /// them pins the interaction of all three. This one reproduces the production combination verbatim
    /// and asserts its grads equal the fully-dense reference (no checkpointing anywhere) to fp tolerance
    /// for BOTH the DiT and the conditioner (`llm_adapter`) factors, and that both actually train.
    /// Synthetic small model, Metal, no real weights. Failure-capable: regressing the production forward
    /// to capture `encoder` collapses the conditioner grad to zero and reddens this test (verified by
    /// temporarily routing `compute_loss_grads` through `_encoder_captured`).
    #[test]
    fn production_combined_config_grads_match_dense() {
        let (mut dit, mut cond, params, adapter, _tp, blocks) = tiny_model_and_adapter();
        let (x0, source, t5_ids, noise) = tiny_inputs(&tiny_dit_cfg(), &tiny_cond_cfg(), 32);

        let grads_of = |dit: &mut CosmosDiT,
                        cond: &mut AnimaTextConditioner,
                        dit_seg: bool,
                        cond_seg: bool,
                        ck: Option<&[Vec<String>]>| {
            dit.set_sdpa_checkpoint(dit_seg);
            cond.set_sdpa_checkpoint(cond_seg);
            let (_l, g) = compute_loss_grads(
                dit,
                cond,
                &params,
                &adapter,
                4.0,
                4.0,
                &x0,
                &source,
                &t5_ids,
                0.5,
                &noise,
                false,
                ck,
                Dtype::Float32,
            )
            .unwrap();
            eval(g.values()).unwrap();
            g
        };

        // Fully-dense reference: no whole-block checkpointing and every SDPA-segment flag OFF, so every
        // activation is retained — the autograd ground truth.
        let g_dense = grads_of(&mut dit, &mut cond, false, false, None);
        // Production combined config, verbatim: whole-block ON, DiT segment OFF, conditioner segment ON.
        let g_prod = grads_of(&mut dit, &mut cond, false, true, Some(&blocks));

        let max_rel = max_rel_diff(&g_dense, &g_prod);
        eprintln!("[sc-10576] production-combined-vs-dense grad max rel diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-3,
            "production combined-config grads must match fully-dense within fp tolerance: max rel {max_rel:.2e}"
        );
        // Both factor groups must actually train in the production config (proves the conditioner grad
        // path stays live once whole-block + segment checkpointing are combined — the sc-10522 trap).
        assert!(
            cond_lora_b_grad(&g_prod) > 1e-6,
            "conditioner must receive gradient in the production combined config"
        );
        assert!(
            dit_lora_b_grad(&g_prod) > 1e-6,
            "DiT must receive gradient in the production combined config"
        );
    }

    /// SDPA-segment checkpointing (the dense/LoKr path) must not change grads either: dense grads with
    /// segment ckpt ON == OFF, to fp tolerance.
    #[test]
    fn sdpa_segment_checkpoint_grads_match_retained() {
        let (mut dit, mut cond, params, adapter, _tp, _blocks) = tiny_model_and_adapter();
        let (x0, source, t5_ids, noise) = tiny_inputs(&tiny_dit_cfg(), &tiny_cond_cfg(), 32);
        let grads_of = |dit: &mut CosmosDiT, cond: &mut AnimaTextConditioner, on: bool| {
            dit.set_sdpa_checkpoint(on);
            cond.set_sdpa_checkpoint(on);
            let (_l, g) = compute_loss_grads(
                dit,
                cond,
                &params,
                &adapter,
                4.0,
                4.0,
                &x0,
                &source,
                &t5_ids,
                0.5,
                &noise,
                false,
                None,
                Dtype::Float32,
            )
            .unwrap();
            eval(g.values()).unwrap();
            g
        };
        let g_off = grads_of(&mut dit, &mut cond, false);
        let g_on = grads_of(&mut dit, &mut cond, true);
        let max_rel = max_rel_diff(&g_off, &g_on);
        eprintln!("[sc-10576] sdpa-seg-ckpt-vs-retained grad max rel diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-3,
            "SDPA-segment checkpointing must not change grads: max rel {max_rel:.2e}"
        );
    }

    /// The sc-10522 trap made executable: threading `encoder` keeps the conditioner live; CAPTURING it
    /// (the deliberately-wrong impl) drops the conditioner gradient to ZERO while the DiT still trains
    /// and the loss still falls — exactly the silent inert-adapter failure. A mutation guard: if the
    /// production forward regressed to capturing `encoder`, `checkpointed_grads_match_dense_…` would go
    /// red here (conditioner grad ≈ 0 ≠ the dense non-zero).
    #[test]
    fn captured_encoder_zeros_conditioner_grad_mutation() {
        let (mut dit, mut cond, params, adapter, _tp, blocks) = tiny_model_and_adapter();
        dit.set_sdpa_checkpoint(false);
        cond.set_sdpa_checkpoint(false);
        let (x0, source, t5_ids, noise) = tiny_inputs(&tiny_dit_cfg(), &tiny_cond_cfg(), 32);

        // Correct (threaded) path.
        let (_l, g_ok) = compute_loss_grads(
            &mut dit,
            &mut cond,
            &params,
            &adapter,
            4.0,
            4.0,
            &x0,
            &source,
            &t5_ids,
            0.5,
            &noise,
            false,
            Some(&blocks),
            Dtype::Float32,
        )
        .unwrap();
        eval(g_ok.values()).unwrap();

        // Wrong (captured) path — same math except `encoder` is a captured constant in each block's
        // checkpoint segment, so the backward produces no cotangent for it.
        let g_bad = grads_encoder_captured(
            &mut dit, &mut cond, &params, &adapter, &x0, &source, &t5_ids, 0.5, &noise, &blocks,
        );

        let ok = cond_lora_b_grad(&g_ok);
        let bad = cond_lora_b_grad(&g_bad);
        let dit_bad = dit_lora_b_grad(&g_bad);
        eprintln!(
            "[sc-10576] conditioner lora_b Σ|grad|: threaded {ok:.3e} vs captured {bad:.3e}; DiT (captured) {dit_bad:.3e}"
        );
        assert!(ok > 1e-6, "threaded encoder: conditioner must train");
        assert!(
            bad < 1e-9,
            "captured encoder: conditioner grad must collapse to ZERO (the trap), got {bad:.3e}"
        );
        assert!(
            dit_bad > 1e-6,
            "captured encoder still trains the DiT — that is why the bug is silent"
        );
    }

    /// Compute grads via the deliberately-wrong `encoder`-captured checkpoint forward (mutation test).
    #[allow(clippy::too_many_arguments)]
    fn grads_encoder_captured(
        dit: &mut CosmosDiT,
        cond: &mut AnimaTextConditioner,
        params: &LoraParams,
        adapter: &TrainAdapter,
        x0: &Array,
        source: &Array,
        t5_ids: &Array,
        sigma: f32,
        noise: &Array,
        blocks: &[Vec<String>],
    ) -> LoraParams {
        let (x_t, target, timestep) = build_batch(x0, noise, sigma).unwrap();
        let src = source.clone();
        let ids = t5_ids.clone();
        let blk = blocks.to_vec();
        let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
            {
                let mut host = AnimaAdapterHost {
                    dit: &mut *dit,
                    conditioner: &mut *cond,
                };
                adapter.install_as(&mut host, &p, 4.0, 4.0, None, LOKR_DTYPE)?;
            }
            let enc = cond
                .forward(&src, &ids, Dtype::Float32)
                .map_err(|e| Exception::custom(e.to_string()))?;
            let s = Array::from_slice(&[timestep], &[1]);
            let v = dit
                .forward_with_main_checkpointed_encoder_captured(
                    &x_t,
                    &s,
                    &enc,
                    Dtype::Float32,
                    &p,
                    &blk,
                    4.0,
                )
                .map_err(|e| Exception::custom(e.to_string()))?;
            let diff = subtract(&v, &target)?;
            Ok(vec![diff.square()?.mean(None)?])
        };
        let mut vg = keyed_value_and_grad(loss_fn);
        vg(params.clone(), 0).unwrap().1
    }

    #[test]
    fn collect_dit_block_local_targets_excludes_conditioner() {
        let tp = vec![
            "blocks.0.self_attn.q_proj".to_string(),
            "blocks.1.mlp.layer2".to_string(),
            "blocks.1.adaln_modulation_mlp.2".to_string(),
            "llm_adapter.blocks.0.self_attn.q_proj".to_string(),
        ];
        let out = collect_dit_block_local_targets(&tp, 2);
        assert_eq!(out[0], vec!["self_attn.q_proj".to_string()]);
        assert_eq!(
            out[1],
            vec![
                "mlp.layer2".to_string(),
                "adaln_modulation_mlp.2".to_string()
            ]
        );
        let total: usize = out.iter().map(|b| b.len()).sum();
        assert_eq!(
            total, 3,
            "the conditioner llm_adapter target must be excluded"
        );
    }

    #[test]
    fn unified_tokens_grows_with_edge() {
        assert_eq!(unified_tokens(512), 32.0 * 32.0 + 512.0);
        assert_eq!(unified_tokens(1024), 64.0 * 64.0 + 512.0);
        assert!(unified_tokens(1536) > unified_tokens(1024));
    }

    /// The pre-flight guard mechanism, deterministically (no real weights): an over-budget projection
    /// returns a catchable, flag-recommending error; a within-budget one is `Ok`. Drives the MLX
    /// memory limit directly so the assertion is machine-independent.
    #[test]
    fn preflight_guard_refuses_over_budget() {
        use mlx_rs::memory::set_memory_limit;
        let prev = set_memory_limit(8 * 1024 * 1024 * 1024); // 8 GB budget → safe ~6.8 GB
        let over = preflight_memory_guard(1536, true);
        set_memory_limit(256 * 1024 * 1024 * 1024); // 256 GB budget → safe ~217 GB
        let under = preflight_memory_guard(512, true);
        set_memory_limit(prev); // restore
        let err = over.unwrap_err().to_string();
        assert!(
            err.contains("Gradient Checkpointing") && err.contains("1536"),
            "over-budget error must be actionable + name the resolution: {err}"
        );
        assert!(
            under.is_ok(),
            "a 512² run under a 256 GB budget must pass the guard"
        );
    }

    // ======================================================================================
    // sc-10641 — in-training preview sampling
    // ======================================================================================

    /// Max abs element-wise diff between two same-shape latents.
    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        a.subtract(b)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>()
    }

    /// The interval contract, deterministically (no weights): previews are enabled only with a positive
    /// cadence + at least one prompt + not cancelled, and fire on EXACTLY the cadence multiples — nowhere
    /// else. A regression to `>=`, an off-by-one, or firing when disabled reddens this.
    #[test]
    fn preview_cadence_contract() {
        let base = TrainingConfig {
            sample_every: 5,
            sample_prompts: vec!["1girl".into()],
            ..Default::default()
        };
        assert!(previews_enabled(&base, false));
        assert!(!previews_enabled(&base, true), "cancelled ⇒ disabled");
        assert!(
            !previews_enabled(
                &TrainingConfig {
                    sample_every: 0,
                    ..base.clone()
                },
                false
            ),
            "cadence 0 ⇒ disabled"
        );
        assert!(
            !previews_enabled(
                &TrainingConfig {
                    sample_prompts: vec![],
                    ..base.clone()
                },
                false
            ),
            "no prompts ⇒ disabled"
        );
        // The default config never opts in.
        assert!(!previews_enabled(&TrainingConfig::default(), false));

        // Fires on multiples of the cadence, nowhere else.
        let due: Vec<u32> = (1..=12).filter(|&s| preview_due(s, 5)).collect();
        assert_eq!(due, vec![5, 10]);
        assert!(!preview_due(1, 5) && !preview_due(7, 5) && !preview_due(11, 5));
        assert!(
            (1..=10).all(|s| !preview_due(s, 0)),
            "cadence 0 never fires"
        );
    }

    /// sc-10641 live-graph guard (the critical correctness point). The preview MUST re-run the
    /// conditioner (`llm_adapter`) through the LIVE graph so it reflects the in-training conditioner
    /// adapters — never a cached output (the sc-10522 trap). Synthetic small model, Metal, no real
    /// weights. A trained (non-zero `lora_b`) adapter is installed on the CONDITIONER ONLY (the DiT
    /// factors stay at the `B=0` no-op), so the ONLY thing that can move the preview latent is the live
    /// conditioner. The production `render_preview_latent` (conditioner run live) is compared against
    /// `render_latent_with_enc` fed a STALE conditioner output captured BEFORE install:
    ///   - live ≠ stale  ⇒ the preview genuinely re-ran the conditioner adapters — FAILS if it ever
    ///     regressed to caching the conditioner output.
    ///   - live == render_latent_with_enc(live enc) ⇒ the conditioner is the sole live-dependent input,
    ///     so the first assertion can't pass for a spurious reason.
    #[test]
    fn preview_samples_conditioner_through_live_graph() {
        let (mut dit, mut cond, params0, adapter, _tp, _blocks) = tiny_model_and_adapter();
        // `tiny_inputs` returns (x0, source, t5_ids, noise); the noise is the preview's starting latent.
        let (_x0, source, t5_ids, init) = tiny_inputs(&tiny_dit_cfg(), &tiny_cond_cfg(), 32);

        // Stale conditioner output — captured from the BASE conditioner (no adapters installed). This is
        // exactly what caching the conditioner OUTPUT would freeze into the preview (the trap).
        let enc_stale = cond.forward(&source, &t5_ids, Dtype::Float32).unwrap();
        eval([&enc_stale]).unwrap();

        // "Train" the conditioner: give its `lora_b` non-zero values (build_lora_targets inits them to
        // 0). DiT `lora_b` left at 0 (inert) ⇒ the DiT forward is constant across both legs.
        let mut trained = params0.clone();
        for (k, v) in trained.iter_mut() {
            if k.starts_with("llm_adapter.") && k.ends_with(".lora_b") {
                *v = random::normal::<f32>(v.shape(), None, None, Some(&random::key(99).unwrap()))
                    .unwrap();
            }
        }
        {
            let mut host = AnimaAdapterHost {
                dit: &mut dit,
                conditioner: &mut cond,
            };
            adapter
                .install_as(&mut host, &trained, 4.0, 4.0, None, LOKR_DTYPE)
                .unwrap();
        }

        let steps = 3usize;
        // Production preview path: conditioner run LIVE (reflects the trained conditioner adapters).
        let latent_live = crate::pipeline::render_preview_latent(
            &dit,
            &cond,
            &source,
            &t5_ids,
            None,
            &init,
            steps,
            1.0,
            7,
            Dtype::Float32,
            &Default::default(),
        )
        .unwrap();
        // Same DiT, but a STALE (pre-training) conditioner output — the trap.
        let latent_stale = crate::pipeline::render_latent_with_enc(
            &dit,
            &enc_stale,
            None,
            &init,
            steps,
            1.0,
            7,
            Dtype::Float32,
            &Default::default(),
        )
        .unwrap();
        // Positive control: feed render_latent_with_enc the LIVE conditioner output → must equal the
        // live preview (proves the conditioner is the only live-dependent input; no spurious diff).
        let enc_live = cond.forward(&source, &t5_ids, Dtype::Float32).unwrap();
        let latent_ctrl = crate::pipeline::render_latent_with_enc(
            &dit,
            &enc_live,
            None,
            &init,
            steps,
            1.0,
            7,
            Dtype::Float32,
            &Default::default(),
        )
        .unwrap();
        eval([&latent_live, &latent_stale, &latent_ctrl]).unwrap();

        let live_vs_stale = max_abs_diff(&latent_live, &latent_stale);
        let live_vs_ctrl = max_abs_diff(&latent_live, &latent_ctrl);
        eprintln!(
            "[sc-10641] preview latent live-vs-stale {live_vs_stale:.3e}, live-vs-ctrl {live_vs_ctrl:.3e}"
        );
        assert!(
            live_vs_stale > 1e-4,
            "preview must re-run the conditioner LIVE: a cached (stale) conditioner output yields the \
             same latent (max abs diff {live_vs_stale:.3e}) — the sc-10522 inert-adapter trap"
        );
        assert!(
            live_vs_ctrl < 1e-5,
            "render_preview_latent's only live input is the conditioner: feeding the live enc must \
             reproduce the preview (max abs diff {live_vs_ctrl:.3e})"
        );
    }

    // -------- real-weights measurement + validation (sc-10576), #[ignore]d + snapshot-gated --------

    fn anima_split() -> Option<std::path::PathBuf> {
        let home = std::env::var("HOME").ok()?;
        let base = std::path::PathBuf::from(home)
            .join(".cache/huggingface/hub/models--circlestone-labs--Anima/snapshots");
        std::fs::read_dir(&base)
            .ok()?
            .filter_map(|e| e.ok())
            .find_map(|e| {
                let p = e.path().join("split_files");
                p.join("diffusion_models").is_dir().then_some(p)
            })
    }

    /// Load the real DiT + conditioner (+ keep the VAE resident like the train loop); drop the Qwen3
    /// text encoder + tokenizers, exactly as `train_impl` does post-caching.
    fn load_real_dit_cond(
        split: &std::path::Path,
    ) -> (CosmosDiT, AnimaTextConditioner, crate::vae::QwenVae) {
        use mlx_gen::WeightsSource;
        let comps =
            AnimaComponents::load(&WeightsSource::Dir(split.to_path_buf()), Variant::Base).unwrap();
        let AnimaComponents {
            dit,
            conditioner,
            vae,
            text_encoder,
            tokenizers,
        } = comps;
        drop(text_encoder);
        drop(tokenizers);
        mlx_rs::memory::clear_cache();
        (dit, conditioner, vae)
    }

    /// Run one real bf16 first training step at `edge` and return `(peak GB, loss)`. Synthesizes the
    /// cached inputs (latent + Qwen states + T5 ids) directly — the memory profile is value-independent,
    /// so this needs only the real DiT/conditioner weights, not the VAE/text encoders.
    fn measure_first_step(
        dit: &mut CosmosDiT,
        cond: &mut AnimaTextConditioner,
        edge: i32,
        use_checkpoint: bool,
    ) -> (f32, f32) {
        use mlx_rs::memory::{get_peak_memory, reset_peak_memory};
        let tcfg = TrainingConfig {
            rank: 16,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(dit, cond, &tcfg);
        let (targets, params) = {
            let mut host = AnimaAdapterHost {
                dit,
                conditioner: cond,
            };
            build_lora_targets(&mut host, &target_paths, 16, 7).unwrap()
        };
        let adapter = TrainAdapter::Lora { targets };
        let blocks = collect_dit_block_local_targets(&target_paths, dit.config().num_layers);
        let ck: Option<&[Vec<String>]> = if use_checkpoint { Some(&blocks) } else { None };
        dit.set_sdpa_checkpoint(!use_checkpoint);
        cond.set_sdpa_checkpoint(true);

        let hl = edge / 8;
        let x0 = random::normal::<f32>(
            &[1, 16, 1, hl, hl],
            None,
            None,
            Some(&random::key(1).unwrap()),
        )
        .unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(2).unwrap())).unwrap();
        let source =
            random::normal::<f32>(&[1, 64, 1024], None, None, Some(&random::key(3).unwrap()))
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
        let ids = random::uniform::<_, f32>(0.0, 32000.0, &[1, 32], Some(&random::key(4).unwrap()))
            .unwrap()
            .as_dtype(Dtype::Int32)
            .unwrap();
        eval([&x0, &noise, &source, &ids]).unwrap();

        reset_peak_memory();
        let (loss, grads) = compute_loss_grads(
            dit,
            cond,
            &params,
            &adapter,
            16.0,
            16.0,
            &x0,
            &source,
            &ids,
            0.5,
            &noise,
            false,
            ck,
            Dtype::Bfloat16,
        )
        .unwrap();
        eval(grads.values()).unwrap();
        let peak = get_peak_memory() as f32 / (1024.0 * 1024.0 * 1024.0);
        (peak, loss)
    }

    /// sc-10576 CALIBRATION: sweep the dense (whole-block OFF, SDPA-segment ON) first-step peak at
    /// 512/768/1024 to fit `projected_dense_peak_gb`. Prints measured vs. projected — refit the
    /// `ANIMA_PEAK_*` constants if these move materially.
    #[test]
    #[ignore = "needs the circlestone-labs/Anima snapshot; measures GPU peak (sc-10576 calibration)"]
    fn first_step_dense_peak_sweep() {
        let Some(split) = anima_split() else {
            eprintln!("skip: no Anima snapshot");
            return;
        };
        let (mut dit, mut cond, _vae) = load_real_dit_cond(&split);
        eprintln!("[sc-10576] dense first-step peak sweep (bf16, rank16, SDPA-seg ckpt ON):");
        for edge in [512, 768, 1024] {
            let (peak, loss) = measure_first_step(&mut dit, &mut cond, edge, false);
            let s = unified_tokens(edge as u32);
            eprintln!(
                "[sc-10576]   edge {edge:>4}  s {s:>7.0}  peak {peak:6.2} GB  loss {loss:.4}  (projected {:6.2} GB)",
                projected_dense_peak_gb(s, true)
            );
        }
    }

    /// sc-10576 VALIDATION: (1) a direct measured A/B at 1024² (both fit) proving whole-block
    /// checkpointing reduces the first-step peak, then (2) the 1536² criterion — checkpointing makes
    /// the step fit, while the dense projection is refused by the pre-flight guard.
    #[test]
    #[ignore = "needs the circlestone-labs/Anima snapshot; SLOW (2B DiT steps at 1024²/1536²)"]
    fn first_step_1536_checkpointed_vs_dense() {
        let Some(split) = anima_split() else {
            eprintln!("skip: no Anima snapshot");
            return;
        };
        let (mut dit, mut cond, _vae) = load_real_dit_cond(&split);
        let budget = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);

        // (1) Measured A/B at 1024 — both fit, so the reduction is observed, not projected.
        let (dense_1024, _) = measure_first_step(&mut dit, &mut cond, 1024, false);
        let (ckpt_1024, _) = measure_first_step(&mut dit, &mut cond, 1024, true);
        eprintln!(
            "[sc-10576] edge 1024  dense {dense_1024:.2} GB  ckpt {ckpt_1024:.2} GB  ({:.0}% reduction)",
            100.0 * (1.0 - ckpt_1024 / dense_1024)
        );
        assert!(
            ckpt_1024 < dense_1024,
            "whole-block checkpointing must reduce the first-step peak: dense {dense_1024:.2} GB vs ckpt {ckpt_1024:.2} GB"
        );

        // (2) The 1536² criterion: checkpointed fits, dense is over the safe budget → guard refuses.
        let (ck_peak, ck_loss) = measure_first_step(&mut dit, &mut cond, 1536, true);
        let dense_proj = projected_dense_peak_gb(unified_tokens(1536), true);
        let refused = preflight_memory_guard(1536, true).is_err();
        eprintln!(
            "[sc-10576] edge 1536 CHECKPOINTED peak {ck_peak:.2} GB loss {ck_loss:.4} | budget {budget:.0} GB | dense projected {dense_proj:.1} GB | preflight-refuses {refused}"
        );
        assert!(
            ck_peak as f64 <= budget,
            "checkpointed 1536 must fit this machine's budget: {ck_peak:.2} GB vs {budget:.0} GB"
        );
        assert!(
            refused,
            "dense 1536 (projected {dense_proj:.0} GB) must be refused by the pre-flight guard"
        );
    }

    // ==============================================================================================
    // sc-10642 — mid-run resume (CI, Metal-synthetic; no real weights)
    // ==============================================================================================

    /// Round-trip: a few optimizer steps over the Anima trainable surface → `save_resume` → discover →
    /// `load_resume` restores the optimizer state + trainable factors + step/update index. The restored
    /// factors match bit-for-bit, the step count comes back as N (not 0), and the restored optimizer's
    /// next step matches the uninterrupted optimizer's — the sc-9560 exactness the resume wiring relies
    /// on, exercised on Anima's own two-host (DiT + `llm_adapter`) factor map.
    #[test]
    fn resume_round_trips_optimizer_factors_and_step() {
        let (_dit, _cond, params, _adapter, _tp, _blocks) = tiny_model_and_adapter();
        // Synthetic non-zero grads keyed exactly as the factors (so both optimizer instances step
        // identically). `lora_b` starts at 0, so a real grad is needed to move the state.
        let grads: LoraParams = params
            .iter()
            .map(|(k, v)| {
                let g =
                    random::normal::<f32>(v.shape(), None, None, Some(&random::key(99).unwrap()))
                        .unwrap();
                (k.clone(), g)
            })
            .collect();
        let mut opt = TrainOptimizer::from_config("adamw", 1e-3, 0.0).unwrap();
        opt.set_lr_scaled(1.0);
        let mut p = params.clone();
        opt.step(&mut p, &grads).unwrap();
        opt.step(&mut p, &grads).unwrap();
        eval(p.values()).unwrap();

        let dir = std::env::temp_dir().join("mlxgen_anima_resume_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let stem = "anima_style";
        checkpoint::save_resume(&dir, stem, 4, 2, &opt, &p).unwrap();

        // Discovery returns the snapshot's step; load restores it.
        let (found, step) = checkpoint::find_latest_resume(&dir, stem).expect("resume snapshot");
        assert_eq!(step, 4, "find_latest_resume returns the snapshot step");
        let mut opt2 = TrainOptimizer::from_config("adamw", 1e-3, 0.0).unwrap();
        let (loaded, meta) = checkpoint::load_resume(&found, &mut opt2).unwrap();

        // The Anima surface guard passes on a faithful round-trip (fresh-build keys == restored keys).
        assert_resume_surface_matches(&loaded, &params).unwrap();

        // Step count + update index restored — resume continues at N, not 0.
        assert_eq!(meta.step, 4, "resumes at the recorded step, not 0");
        assert_eq!(meta.update_idx, 2, "optimizer-update index restored");
        assert_eq!(meta.optimizer, "adamw");

        // Trainable factors (adapter weights) restored bit-for-bit.
        assert_eq!(loaded.len(), p.len(), "same factor count");
        for (k, v) in &p {
            let l = loaded.get(k).expect("restored factor");
            let d = v
                .subtract(l)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            assert!(d == 0.0, "factor {k} not restored bit-exact: |Δ| = {d:e}");
        }

        // Optimizer state restored: the restored optimizer's next step == the uninterrupted one's.
        let mut a = p.clone();
        let mut b = loaded;
        opt.set_lr_scaled(1.0);
        opt2.set_lr_scaled(1.0);
        opt.step(&mut a, &grads).unwrap();
        opt2.step(&mut b, &grads).unwrap();
        eval(a.values()).unwrap();
        eval(b.values()).unwrap();
        let m = max_rel_diff(&a, &b);
        assert!(
            m <= 1e-6,
            "restored optimizer's next step diverged: max_rel {m:e}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The sc-10522 restore assertion. The guard passes when the checkpoint's factor surface matches the
    /// live model's, and ERRORS (failure-capable) when the checkpoint dropped the conditioner
    /// (`llm_adapter.*`) targets — the trap where the DiT keeps training and every structural check
    /// passes while the conditioner is silently inert — or dropped any single factor.
    #[test]
    fn resume_restore_asserts_full_target_surface() {
        let (_dit, _cond, params, _adapter, _tp, _blocks) = tiny_model_and_adapter();

        // The tiny synthetic surface is a scaled-down analogue of the real 448 DiT + 60 `llm_adapter`
        // split; the guard is size-agnostic (it asserts restored == live surface). Both classes present.
        let (dit_paths, cond_paths) = split_target_surface(&params);
        assert!(!dit_paths.is_empty(), "tiny surface has DiT targets");
        assert!(
            !cond_paths.is_empty(),
            "tiny surface has conditioner targets"
        );

        // Faithful restore (identical surface) passes.
        assert_resume_surface_matches(&params, &params).unwrap();

        // Drop the conditioner (`llm_adapter.*`) factors — the sc-10522 inert-conditioner trap. ERROR.
        let mut dropped = params.clone();
        let cond_keys: Vec<_> = dropped
            .keys()
            .filter(|k| k.starts_with("llm_adapter."))
            .cloned()
            .collect();
        assert!(
            !cond_keys.is_empty(),
            "there are conditioner factors to drop"
        );
        for k in cond_keys {
            dropped.remove(&k);
        }
        let err = assert_resume_surface_matches(&dropped, &params)
            .expect_err("a checkpoint missing the conditioner targets must be refused");
        let msg = err.to_string();
        assert!(msg.contains("sc-10522"), "error names the trap: {msg}");
        assert!(
            msg.contains("0 `llm_adapter` conditioner"),
            "error reports 0 conditioner targets in the checkpoint: {msg}"
        );

        // Dropping a single DiT factor is also caught (wrong count within a class, not a whole class).
        let mut one_missing = params.clone();
        let a_dit = one_missing
            .keys()
            .find(|k| !k.starts_with("llm_adapter.") && k.ends_with(".lora_a"))
            .cloned()
            .expect("a DiT lora_a factor");
        one_missing.remove(&a_dit);
        assert_resume_surface_matches(&one_missing, &params)
            .expect_err("a checkpoint missing a single factor must be refused");
    }
}

/// The empirical fit must reproduce the measured Anima dense first-step peaks (sc-10576) within a few
/// GB, stay monotone in `s`, and put the checkpointing regime on the right side of a typical budget.
#[cfg(test)]
mod preflight_tests {
    use super::{projected_dense_peak_gb, unified_tokens};

    #[test]
    fn projected_peak_reproduces_measured_points() {
        // SCOPE: this guards coefficient-TRANSCRIPTION only, NOT fit accuracy. `projected_dense_peak_gb`
        // is an EXACT 3-point quadratic fit (`a + b·s + c·s²`) through these same three calibration
        // points, so an intact fit ALWAYS reproduces its own interpolation points — this test can catch a
        // fat-fingered/drifted coefficient (or a broken `unified_tokens`), but by construction it cannot
        // vouch for how well the curve predicts a HELD-OUT edge. Real fit accuracy is validated against
        // fresh GPU measurements by the `#[ignore]`d `first_step_dense_peak_sweep` (refit both if it
        // prints materially different peaks). We deliberately do NOT add a synthetic 4th point here: a
        // meaningful held-out check needs a real measurement, and a made-up one would prove nothing.
        //
        // Measured on the 128 GB target (first_step_dense_peak_sweep, bf16, rank16, SDPA-seg ON):
        //   edge 512  (s 1536) → 21.18 GB
        //   edge 768  (s 2816) → 32.05 GB
        //   edge 1024 (s 4608) → 49.44 GB
        // The exact 3-point fit must reproduce each within ~1 GB.
        let approx = |edge: u32, want: f64| {
            let got = projected_dense_peak_gb(unified_tokens(edge), true);
            assert!(
                (got - want).abs() < 1.0,
                "edge {edge}: projected {got:.2} GB vs measured {want:.2} GB"
            );
        };
        approx(512, 21.18);
        approx(768, 32.05);
        approx(1024, 49.44);
    }

    #[test]
    fn projected_peak_puts_1536_over_a_128gb_budget() {
        // Dense 1536² (s 9728) extrapolates well above a 128 GB machine's working set — so the guard
        // refuses it (or it SIGKILLs), which is exactly the regime whole-block checkpointing unblocks.
        let dense_1536 = projected_dense_peak_gb(unified_tokens(1536), true);
        assert!(
            dense_1536 > 100.0,
            "dense 1536² must project over budget (got {dense_1536:.0} GB)"
        );
    }

    #[test]
    fn projected_peak_is_monotone_and_ordered() {
        let p = |edge: u32| projected_dense_peak_gb(unified_tokens(edge), true);
        assert!(p(512) < p(1024));
        assert!(p(1024) < p(1536));
        assert!(p(1536) < p(2048));
        // f32 upper bound is strictly above the bf16 fit.
        let s = unified_tokens(1024);
        assert!(projected_dense_peak_gb(s, false) > projected_dense_peak_gb(s, true));
    }
}
