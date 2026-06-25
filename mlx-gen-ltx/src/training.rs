//! sc-3047 — LTX-2.3 LoRA **training** on the video DiT, in pure Rust on mlx-rs. The Rust port of
//! SceneWorks' pure-MLX `_LtxMlxLoraBackend` / `LtxMlxLoraTrainer` (`training_adapters.py:3249-3628`),
//! realizing the core [`Trainer`] contract (epic 3039). Retiring the Python version removes the last
//! Python-MLX trainer (blocks sc-3049 cutover → sc-3242 `mlx-video` drop).
//!
//! Built on the same functional-autograd mechanism the Z-Image spike proved (sc-3042) and the
//! sc-3043 runtime glue, but LTX has its **own** adapter seam: its [`crate::transformer::Linear`]
//! carries a per-pass [`LoraStack`](crate::transformer) (not the core `AdaptableLinear`), so this
//! module uses the LTX-local [`Linear::set_train_lora`] training seam and its own target
//! enumeration / save, while reusing the core [`LoraParams`] + grad-accumulation helpers and the
//! runtime (schedule / dataset / checkpoint).
//!
//! **What is LTX-specific:**
//!   * **Video-only forward over `LtxDiT`.** The reference loads the AV model and trains with
//!     `audio=None`; [`LtxDiT`] is exactly that video-only reduction (the AV checkpoint embeds the
//!     same `transformer_blocks.{i}` video blocks), and the trained video-attention adapter reloads
//!     onto the AvDiT inference path unchanged.
//!   * **Rectified-flow target = `noise - clean`.** LTX denoises with `x_t - σ·v` over
//!     `x_t = (1-σ)·x0 + σ·noise` and feeds the **raw** transformer output straight to `to_denoised`
//!     (no negation, unlike Z-Image), so the velocity that recovers `x0` is `v = noise - x0`. The
//!     **timestep fed to the DiT is the raw σ** (broadcast over tokens), σ ~ U(1e-3, 1-1e-3). MSE.
//!   * **Latent layout.** A still image VAE-encodes (single frame T=1) to a normalized latent
//!     `(1,128,1,h,w)`, flattened to the patchified `(1, S, 128)` the DiT consumes; the position
//!     grid is built once for the fixed latent resolution. The 24 GB Gemma text encoder is freed
//!     after the one-time prompt-embed cache (mirroring the reference), before the train loop.
//!   * **Adapter surface.** `attn1`/`attn2` (self + text cross-attention) `to_q/k/v/to_out.0`, the
//!     reference `inject_video_attention_lora` default. Residual LoRA over the (Q4) base — the base
//!     is frozen, gradients flow only through the trainable factors (functional autograd handles the
//!     `quantized_matmul` base as a constant). Saved as `{module}.lora_A/B.weight` + `.alpha` (the
//!     `to_out.0` diffusers spelling the inference loader normalizes), so it round-trips through
//!     [`crate::apply_ltx_adapters`].
//!   * **LoRA-only.** The reference LTX MLX trainer has no LoKr (LTX *inference* supports LoKr via
//!     sc-2393, but no LoKr trainer exists); LoKr requests are rejected with that explanation.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use mlx_gen::media::Image;
use mlx_gen::train::checkpoint::checkpoint_filename;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{accumulate_grads, average_grads, LoraParams};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{
    gen_core, LoadSpec, Modality, NetworkType, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainingOutput, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::config::{LtxConfig, LtxVaeConfig, SplitModel};
use crate::gemma::GemmaConfig;
use crate::model::MODEL_ID;
use crate::pipeline::preprocess_conditioning_image;
use crate::positions::{create_position_grid, SPATIAL_SCALE};
use crate::text_encoder::LtxTextEncoder;
use crate::tokenizer::LtxTokenizer;
use crate::transformer::{BlockLoraRef, LtxDiT, Precision};
use crate::vae::LtxVideoVae;

/// Gemma prompt token budget for caption encoding (the captions are short; padding tokens are
/// attended with `mask=None`, matching the reference `Modality(context_mask=None)`).
const MAX_PROMPT_TOKENS: usize = 128;

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence (sc-5637).
const SAMPLE_PROMPT_CAP: usize = 4;

/// The reference `inject_video_attention_lora` default targets (`DEFAULT_LORA_TARGET_MODULES`,
/// `training_adapters.py:72`), restricted to `attn1`/`attn2`. `to_out.0` is the diffusers spelling
/// the inference loader normalizes to the checkpoint's `to_out`.
const DEFAULT_TARGET_SUFFIXES: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// One LoRA-trained attention `Linear`: its diffusers save spelling (e.g. `…attn1.to_out.0`), the
/// resolution segments after the `to_out.0`→`to_out` normalization, and the factor-map keys.
struct LtxLoraTarget {
    save_path: String,
    segs: Vec<String>,
    a_key: Rc<str>,
    b_key: Rc<str>,
}

/// LoRA trainer for LTX-2.3, implementing the core [`Trainer`] surface: a frozen LtxDiT (f32
/// activations × Q4/Q8 weights) + VAE + Gemma text encoder + tokenizer that caches a captioned
/// image dataset to (normalized latent, prompt-embed) pairs, then runs the functional-autograd
/// rectified-flow loop with the sc-3043 runtime glue, and writes a LoRA that round-trips through
/// [`crate::apply_ltx_adapters`].
///
/// **Single-use** (F-055): `train` frees the Gemma text encoder + tokenizer (~24 GB) after the
/// embed cache, so the instance cannot run a second job — `validate` (hence `train`) rejects a reuse
/// up front. Construct a fresh trainer (via [`load_trainer`]) per job.
pub struct LtxTrainer {
    descriptor: TrainerDescriptor,
    /// Freed after the one-time prompt-embed cache (the 24 GB Gemma backbone), before the loop.
    tokenizer: Option<LtxTokenizer>,
    text_encoder: Option<LtxTextEncoder>,
    vae: LtxVideoVae,
    transformer: LtxDiT,
    cfg: LtxConfig,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID,
        family: "ltx",
        backend: "mlx",
        modality: Modality::Video,
        supports_lora: true,
        // The reference LTX MLX trainer is LoRA-only; LoKr training is unsupported (see `validate`).
        supports_lokr: false,
    }
}

/// Construct the trainer from an LTX-2.3 split-weight snapshot directory (transformer / VAE /
/// connector + the Gemma-3-12B text-encoder snapshot resolved like inference). The transformer loads
/// at **f32 activations × quantized weights** (`quant_f32`) for clean autograd — the base is frozen,
/// gradients flow only through the trainable LoRA factors. Registered via [`TrainerRegistration`].
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => return Err(mlx_gen::Error::Msg(
            "ltx_2_3 trainer expects a split-weight snapshot directory (transformer.safetensors \
                 / vae_*.safetensors / connector.safetensors), not a single file"
                .into(),
        )),
    };
    Ok(Box::new(load_trainer_from_dir(root)?))
}

/// The concrete-typed loader behind [`load_trainer`] (sc-4942 — the first-step memory harness needs
/// the concrete [`LtxTrainer`] to reach `.transformer` / `.vae`, which a `Box<dyn Trainer>` hides).
fn load_trainer_from_dir(root: &Path) -> Result<LtxTrainer> {
    let split = SplitModel::from_model_dir(root)?;
    let cfg = LtxConfig::from_model_dir(root)?;
    let vae_config = LtxVaeConfig::from_model_dir(root)?;

    let gemma_dir = crate::model::resolve_gemma_dir()?;
    let gemma_w = Weights::from_dir(&gemma_dir)?;
    let gemma_quant = crate::model::resolve_gemma_quant(&gemma_dir)?;
    let connector_w = Weights::from_file(root.join("connector.safetensors"))?;
    let transformer_w = Weights::from_file(root.join("transformer.safetensors"))?;
    let vae_dec_w = Weights::from_file(root.join("vae_decoder.safetensors"))?;
    let vae_enc_w = Weights::from_file(root.join("vae_encoder.safetensors"))?;

    // Video-only text encoder (bf16, the reference TE dtype); we cast its embeds to f32 per-item for
    // the f32 training forward.
    let text_encoder = LtxTextEncoder::from_weights(
        &gemma_w,
        &connector_w,
        GemmaConfig::gemma_3_12b(),
        gemma_quant,
        &cfg,
        Dtype::Bfloat16,
    )?;
    let transformer = LtxDiT::from_weights(
        &transformer_w,
        &cfg,
        Precision::quant_f32(split.bits, split.group),
    )?;
    let vae = LtxVideoVae::from_weights(&vae_dec_w, Some(&vae_enc_w), &vae_config)?;
    let tokenizer = LtxTokenizer::from_dir(&gemma_dir)?;

    Ok(LtxTrainer {
        descriptor: trainer_descriptor(),
        tokenizer: Some(tokenizer),
        text_encoder: Some(text_encoder),
        vae,
        transformer,
        cfg,
    })
}

// Link-time trainer registration (epic 3720): the macro emits the `inventory::submit!` and bridges
// the crate's rich `Result` into the trainer registry's backend-neutral `gen_core::Result`.
mlx_gen::register_trainer! { trainer_descriptor => load_trainer }

/// Capability-free request validation, factored out of [`Trainer::validate`] so it can be
/// unit-tested without a loaded trainer. Rejects an empty dataset, zero rank, LoKr (LoRA-only
/// trainer), and unsupported optimizers. The single-use / text-encoder-present check stays in
/// [`Trainer::validate`], which has the trainer state to inspect (F-055).
fn validate_request(req: &TrainingRequest) -> Result<()> {
    if req.items.is_empty() {
        return Err("ltx_2_3 trainer: dataset is empty".into());
    }
    if req.config.rank == 0 {
        return Err("ltx_2_3 trainer: rank must be > 0".into());
    }
    if req.config.network_type == NetworkType::Lokr {
        return Err(
            "ltx_2_3 trainer: LoKr training is not supported — the reference LTX MLX \
                    trainer is LoRA-only. (LTX *inference* supports LoKr via sc-2393, but no \
                    LoKr trainer exists yet; that would be a separate extension.)"
                .into(),
        );
    }
    if !TrainOptimizer::is_supported(&req.config.optimizer) {
        return Err(format!(
            "ltx_2_3 trainer: optimizer '{}' is not available on MLX training (supported: \
             adamw, adam, rose, prodigy)",
            req.config.optimizer
        )
        .into());
    }
    Ok(())
}

impl Trainer for LtxTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Single-use enforcement (F-055): `train` frees the Gemma text encoder + tokenizer (~24 GB)
        // after the embed cache, so a second `train` on the same instance can't re-encode. Fail here,
        // up front (validate runs before any progress is emitted), instead of with a late, confusing
        // "text encoder missing" mid-run. Construct a fresh trainer (via `load_trainer`) per job.
        if self.text_encoder.is_none() || self.tokenizer.is_none() {
            return Err(
                "ltx_2_3 trainer: single-use — the Gemma text encoder was freed after the \
                        first train() to reclaim ~24 GB; construct a fresh trainer for each job"
                    .into(),
            );
        }
        validate_request(req).map_err(Into::into)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

impl LtxTrainer {
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
        let edge = bucket_resolution(cfg.resolution); // pixel edge, multiple of 32
        let latent_edge = (edge / SPATIAL_SCALE as u32).max(1) as usize; // latent tokens per side

        // sc-4942 — LTX trains in **f32 activations** (× the Q-packed base), NOT bf16, even though the
        // SceneWorks worker passes `train_dtype=bf16` (sc-4881). MEASURED on real weights (the
        // `bf16_grads_direction_and_memory_vs_f32` harness): a bf16 activation cast DECORRELATES the
        // gradient from the f32 (quality) path — global cosine 0.31–0.45, with the early/deep K
        // projections of BOTH attentions pointing ~opposite — because the 48-block distilled DiT is
        // chaos-sensitive (the same reason inference uses `quant_f32`, not `quant_bf16`, for quality;
        // see `transformer::Precision`/`Mode::QuantF32`). bf16 would save ~30% memory (1024: 28 vs 43 GB)
        // but the f32 working set already fits the video tier with the checkpointing levers below
        // (1024 ≈ 43 GB attn-ckpt / 27 GB block-ckpt, measured 128 GB box; the guard auto-scales), so
        // honoring bf16 would trade training quality for memory we do not need. (`LtxDiT::cast_weights`
        // stays available — the bf16 harness that produced this finding exercises it — but the
        // production trainer never invokes it; this trainer is f32-only.)
        let compute_dtype = Dtype::Float32;

        // sc-4942 — fail-fast pre-flight memory guard (the sc-4874 mechanism, ported to LTX). The dense
        // (non-block-checkpointed) first step materializes the whole forward graph in one MLX `eval`; at
        // high resolution that working set can exceed unified memory and the OS hard-kills the worker
        // with an UNCATCHABLE SIGKILL. We predict it and refuse up front with an actionable, catchable
        // error — BEFORE the (~minutes-long) latent caching — when gradient checkpointing is not
        // enabled. (LTX is LoRA-only, so the LoRA-path condition is always met.)
        let will_checkpoint = cfg.gradient_checkpointing;
        if !will_checkpoint {
            preflight_memory_guard(latent_edge)?;
        }

        // --- prepare → load → cache: normalized latents + prompt embeds (then free the TE) ---
        on_progress(TrainingProgress::LoadingModel);
        let total = req.items.len() as u32;
        let mut cache: Vec<(Array, Array)> = Vec::with_capacity(req.items.len());
        // sc-5637 — preview-sample prompts, pre-encoded inside the `te`/`tok` scope below (the Gemma
        // encoder is freed before the train loop). LTX is distilled (no CFG) → one ctx per prompt.
        let mut sample_ctxs: Vec<(String, Array)> = Vec::new();
        {
            let te = self.text_encoder.as_ref().ok_or_else(|| {
                mlx_gen::Error::Msg("ltx_2_3 trainer: text encoder missing".into())
            })?;
            let tok = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| mlx_gen::Error::Msg("ltx_2_3 trainer: tokenizer missing".into()))?;
            for (i, item) in req.items.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    break;
                }
                on_progress(TrainingProgress::Caching {
                    current: i as u32 + 1,
                    total,
                });
                let img = center_crop_square(&decode_image(&item.image_path)?);
                let prep = preprocess_conditioning_image(&img, edge, edge)?; // (1,3,1,edge,edge)
                let latent = self.vae.encode(&prep)?; // (1,128,1,le,le), normalized, f32
                let clean = flatten_latent(&latent)?; // (1, S, 128)
                let (ids, mask) = tok.encode(&item.caption, MAX_PROMPT_TOKENS)?;
                let ctx = to_dtype(&te.encode(&ids, &mask)?, Dtype::Float32)?; // (1, L, 4096)
                eval([&clean, &ctx])?;
                cache.push((clean, ctx));
            }
            // sc-5637 — pre-encode the preview-sample prompts while the encoder is still resident.
            if cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() && !req.cancel.is_cancelled()
            {
                for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                    let (ids, mask) = tok.encode(prompt, MAX_PROMPT_TOKENS)?;
                    let ctx = to_dtype(&te.encode(&ids, &mask)?, Dtype::Float32)?;
                    eval([&ctx])?;
                    sample_ctxs.push((prompt.clone(), ctx));
                }
            }
        }
        if cache.is_empty() {
            // sc-4895 — a cancel tripped during caching is a genuine cancellation → typed
            // `Error::Canceled` (bridged 1:1 to `gen_core::Error::Canceled`); an empty cache with no
            // cancel is a real "no usable dataset items" error.
            if req.cancel.is_cancelled() {
                return Err(mlx_gen::Error::Canceled);
            }
            return Err("ltx_2_3 trainer: no usable dataset items".into());
        }
        // Free the Gemma text encoder + tokenizer (~24 GB) before training — they are only needed for
        // the one-time embed cache (mirrors the reference `prepare_dataset` release).
        self.text_encoder = None;
        self.tokenizer = None;

        let sampling_enabled = !sample_ctxs.is_empty();

        // The RoPE position grid is identical across items at a fixed latent resolution (single
        // frame) — build it once. Reused for preview-sample rendering (sc-5637).
        let positions = create_position_grid(1, 1, latent_edge, latent_edge);

        // --- adapter targets + trainable factors ---
        let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
            DEFAULT_TARGET_SUFFIXES
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            cfg.lora_target_modules.clone()
        };
        let (targets, mut params) = build_targets(
            &mut self.transformer,
            self.cfg.num_layers,
            &suffixes,
            cfg.rank as i32,
            cfg.seed,
        )?;
        if targets.is_empty() {
            return Err(
                "ltx_2_3 trainer: no LoRA targets resolved (check lora_target_modules)".into(),
            );
        }
        let alpha = cfg.alpha;
        let rank = cfg.rank as f32;
        let mae = {
            let lt = cfg.loss_type.to_ascii_lowercase();
            lt == "mae" || lt == "l1"
        };

        // sc-4942 — gradient checkpointing. Group the resolved targets by their owning block so the
        // checkpointed forward can thread each block's LoRA factors as explicit recompute inputs. Every
        // target of this trainer lives in a `transformer_blocks.{i}.attn{1,2}` leaf, so the grouping
        // covers the whole adapter surface.
        let n_layers = self.cfg.num_layers as usize;
        let block_targets = group_block_targets(&targets, n_layers);
        // Gradient checkpointing is an OPT-IN OPTION (the SceneWorks "Gradient Checkpointing" toggle),
        // never auto-forced — a run that would OOM is caught instead by the fail-fast pre-flight guard
        // above, which recommends this flag rather than silently changing the user's training dynamics.
        let use_checkpoint = cfg.gradient_checkpointing;
        let checkpoint_block: Option<&[Vec<BlockLoraRef>]> = if use_checkpoint {
            Some(&block_targets)
        } else {
            None
        };
        // sc-4942 — attention-segment checkpointing is on for the dense (non-block-checkpointed) path:
        // it is numerically identical to the retained backward (same decomposed attention, recomputed)
        // and removes the dominant seq² per-block retention — the flash-backward surrogate every torch
        // trainer gets from its fused SDPA kernel. When whole-block checkpointing is on it goes OFF (the
        // block recompute already covers attention; nesting would recompute it twice for no win).
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
            let (clean, ctx) = &cache[((step - 1) as usize) % cache.len()];
            // σ ~ U(1e-3, 1-1e-3), deterministic in seed (the reference's uniform timestep).
            let sigma = {
                let k = random::key(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64))?;
                random::uniform::<_, f32>(1e-3f32, 1.0 - 1e-3, &[1], Some(&k))?.item::<f32>()
            };
            let noise = random::normal::<f32>(
                clean.shape(),
                None,
                None,
                Some(&random::key(
                    cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
                )?),
            )?;
            let (loss, grads) = compute_loss_grads(
                &mut self.transformer,
                &params,
                &targets,
                alpha,
                rank,
                clean,
                ctx,
                &positions,
                sigma,
                &noise,
                mae,
                checkpoint_block,
                compute_dtype,
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
                save_lora(&params, &targets, alpha, cfg.rank, &ckpt)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }

            // sc-5637 — periodic best-effort preview frames from the in-progress adapter. Install the
            // current factors concretely for the forward-only render (the next step's traced `loss_fn`
            // re-installs them); a render failure logs and is skipped, never failing the training run.
            if sampling_enabled && step % cfg.sample_every == 0 {
                let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);
                install_train_lora(
                    &mut self.transformer,
                    &params,
                    &targets,
                    alpha,
                    rank,
                    lora_dtype,
                )?;
                let total = sample_ctxs.len() as u32;
                for (i, (prompt, ctx)) in sample_ctxs.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    let sample_seed = cfg
                        .seed
                        .wrapping_add(step as u64)
                        .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                        .wrapping_add(i as u64);
                    match crate::pipeline::render_sample(
                        &self.transformer,
                        &self.vae,
                        ctx,
                        &positions,
                        sample_seed,
                        latent_edge,
                        compute_dtype,
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
        save_lora(&params, &targets, alpha, cfg.rank, &adapter_path)?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Flatten a single-frame VAE latent `(1, 128, 1, le, le)` to the patchified `(1, S, 128)` the DiT
/// consumes (`S = le·le`) — the reference's `transpose(reshape(latent, (B, C, -1)), (0, 2, 1))`.
fn flatten_latent(latent: &Array) -> Result<Array> {
    let sh = latent.shape(); // [1, 128, 1, le, le]
    let (b, c) = (sh[0], sh[1]);
    let s = sh[2] * sh[3] * sh[4];
    let flat = latent.reshape(&[b, c, s])?; // (1, 128, S)
    Ok(flat.transpose_axes(&[0, 2, 1])?) // (1, S, 128)
}

/// `to_out.0` → `to_out`, the only diffusers→checkpoint rename in the attention LoRA surface (the
/// inference loader does the same in `adapters::normalize`); other suffixes pass through.
fn resolve_segments(save_path: &str) -> Vec<String> {
    save_path
        .replace(".to_out.0", ".to_out")
        .split('.')
        .map(String::from)
        .collect()
}

/// Enumerate the `attn1`/`attn2` × `suffixes` targets across the DiT's `num_layers` blocks, resolve
/// each on the (mutable) DiT, read its `[out,in]` base shape, and initialise the trainable factors
/// the reference `_MlxLoRALinear` way — `A ~ N(0, 0.02)` `[rank,in]`, `B = 0` `[out,rank]` — keyed
/// `{save_path}.lora_a` / `.lora_b`. Targets that do not resolve (a missing gated branch, a typo'd
/// suffix) are skipped.
fn build_targets(
    dit: &mut LtxDiT,
    num_layers: i32,
    suffixes: &[String],
    rank: i32,
    seed: u64,
) -> Result<(Vec<LtxLoraTarget>, LoraParams)> {
    let mut targets = Vec::new();
    let mut params = LoraParams::new();
    let small = Array::from_slice(&[0.02f32], &[1]);
    let mut idx: u64 = 0;
    for i in 0..num_layers {
        for attn in ["attn1", "attn2"] {
            for suf in suffixes {
                let save_path = format!("transformer_blocks.{i}.{attn}.{suf}");
                let segs = resolve_segments(&save_path);
                let seg_refs: Vec<&str> = segs.iter().map(String::as_str).collect();
                let Some(lin) = dit.adaptable_mut(&seg_refs) else {
                    continue;
                };
                let shape = lin.base_shape(); // [out, in]
                let (out_f, in_f) = (shape[0], shape[1]);
                let a_key: Rc<str> = Rc::from(format!("{save_path}.lora_a"));
                let b_key: Rc<str> = Rc::from(format!("{save_path}.lora_b"));
                let ka = random::key(seed.wrapping_add(2 * idx + 1))?;
                let a = multiply(
                    &random::normal::<f32>(&[rank, in_f], None, None, Some(&ka))?,
                    &small,
                )?;
                let b = Array::zeros::<f32>(&[out_f, rank])?;
                eval([&a, &b])?;
                params.insert(a_key.clone(), a);
                params.insert(b_key.clone(), b);
                targets.push(LtxLoraTarget {
                    save_path,
                    segs,
                    a_key,
                    b_key,
                });
                idx += 1;
            }
        }
    }
    Ok((targets, params))
}

/// Inject the current trainable factors as one LoRA residual per target via the LTX training seam —
/// transpose `[r,in]`→`[in,r]` and `[out,r]`→`[r,out]`, fold `alpha/rank` into `b` — so the residual
/// is `(x·Aᵀ·Bᵀ)·(alpha/rank)`, matching the reference `_MlxLoRALinear`. Differentiable.
fn install_train_lora(
    dit: &mut LtxDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    lora_dtype: Option<Dtype>,
) -> MlxResult<()> {
    for t in targets {
        let a = params[&t.a_key].t(); // [r,in] -> [in,r]
        let b = params[&t.b_key]
            .t()
            .multiply(Array::from_slice(&[alpha / rank], &[1]))?; // [out,r] -> [r,out] · (α/r)
                                                                  // sc-4942 — under the bf16 training cast the f32 factors must join the bf16 stream, or every
                                                                  // adapted Linear re-promotes its block to f32 (defeating the activation saving). No-op in f32.
        let (a, b) = match lora_dtype {
            Some(dt) => (a.as_dtype(dt)?, b.as_dtype(dt)?),
            None => (a, b),
        };
        let seg_refs: Vec<&str> = t.segs.iter().map(String::as_str).collect();
        let lin = dit
            .adaptable_mut(&seg_refs)
            .ok_or_else(|| Exception::custom(format!("LoRA target not found: {}", t.save_path)))?;
        lin.set_train_lora(a, b);
    }
    Ok(())
}

/// Group resolved targets by their owning block (sc-4942) — `block_targets[i]` lists block `i`'s
/// trainable LoRA targets as the block-local path (`segs` minus the `transformer_blocks.{i}` prefix)
/// plus the factor-map keys, for the gradient-checkpoint closure. Every target lives in a
/// `transformer_blocks.{i}.attn{1,2}` leaf, so the grouping is exhaustive.
fn group_block_targets(targets: &[LtxLoraTarget], n_layers: usize) -> Vec<Vec<BlockLoraRef>> {
    let mut out: Vec<Vec<BlockLoraRef>> = (0..n_layers).map(|_| Vec::new()).collect();
    for t in targets {
        // segs = ["transformer_blocks", "{i}", attn, suffix...]; the block-local path is segs[2..].
        if t.segs.len() < 3 || t.segs[0] != "transformer_blocks" {
            continue;
        }
        let Ok(i) = t.segs[1].parse::<usize>() else {
            continue;
        };
        if i >= n_layers {
            continue;
        }
        out[i].push(BlockLoraRef {
            local: t.segs[2..].to_vec(),
            a_key: t.a_key.to_string(),
            b_key: t.b_key.to_string(),
        });
    }
    out
}

/// One forward+backward over the trainable factors: build the rectified-flow input `x_t`, inject the
/// factors, run the video DiT, regress the raw velocity toward `noise - clean`, return `(loss, grads)`.
///
/// `checkpoint_block`, when `Some`, lists each block's trainable targets and switches the forward to
/// the gradient-checkpointed path (sc-4942) — each block recomputes its activations in the backward
/// instead of retaining them. `None` runs the dense (attention-segment-checkpointed) forward.
/// `dtype` is the training compute dtype (sc-4942): for bf16 the DiT weights were cast once in
/// `train_impl` and `preprocess` casts the activation stream, so the LoRA factors are cast at install
/// (here and inside the checkpoint segment) to keep the whole graph bf16; the noising / loss / grads
/// stay f32.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &mut LtxDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    clean: &Array,
    context: &Array,
    positions: &Array,
    sigma: f32,
    noise: &Array,
    mae: bool,
    checkpoint_block: Option<&[Vec<BlockLoraRef>]>,
    dtype: Dtype,
) -> Result<(f32, LoraParams)> {
    // x_t = (1-σ)·clean + σ·noise; target = noise - clean (the raw-output velocity); timestep = σ.
    // x_t / context stay f32 here; `preprocess` casts the activation stream to the compute dtype.
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    let x_t = add(&multiply(clean, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, clean)?;
    let timestep = Array::from_slice(&[sigma], &[1, 1]); // (B, 1), broadcast over tokens
    let ctx = context.clone();
    let pos = positions.clone();
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        let v = match checkpoint_block {
            Some(bt) => dit
                .forward_with_main_checkpointed(&x_t, &timestep, &ctx, None, &pos, &p, bt, alpha)
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => {
                install_train_lora(dit, &p, targets, alpha, rank, lora_dtype)?;
                // `None`: content-keyed RoPE memo (the per-stage epoch fast path is inference-only;
                // training positions are constant within a step, so the content compare hits — sc-7141).
                dit.forward(&x_t, &timestep, &ctx, None, &pos, None)
                    .map_err(|e| Exception::custom(e.to_string()))?
            }
        };
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

/// Projected DENSE (non-block-checkpointed) first-step peak memory, in GB, as a function of the LTX
/// latent token count `s` (the trainer trains single-frame still latents, so `s = (edge/32)²`). With
/// attention-segment checkpointing always on (sc-4942) the seq² attention term is demoted to a single
/// layer's backward transient, so the measured f32 curve is essentially **linear** in `s`: the
/// constant is the resident base (the Q-packed 22B DiT after the Gemma TE is freed) plus the
/// f32-activation working set, the slope the per-token retained hidden activations across the 48
/// blocks.
///
/// CALIBRATED from `first_step_attn_ckpt_sweep` (128 GB Mac, rank 8 / 384 targets / batch 1, f32):
/// s=256 → 23.3 GB, s=576 → 31.5 GB, s=1024 → 42.6 GB (fit error < 0.2 GB). Refit if that harness
/// prints materially different numbers. (LTX trains f32 only — see the `compute_dtype` note in
/// `train_impl`; there is no bf16 production path to size.)
fn projected_dense_peak_gb(s: f64) -> f64 {
    16.9 + 0.0251 * s
}

/// Refuse a run whose dense first step would exceed this machine's memory budget (and thus get
/// SIGKILLed), returning a catchable, actionable error instead (sc-4942 — the sc-4874 mechanism).
/// `latent_edge` is the latent tokens per side (`edge/32`); the token count is `latent_edge²` (the
/// trainer trains single-frame still latents). The budget is MLX's reported memory limit (≈ the
/// device's recommended working set) × 0.85 for worker/host headroom. Only consulted when gradient
/// checkpointing is OFF.
fn preflight_memory_guard(latent_edge: usize) -> Result<()> {
    let s = (latent_edge * latent_edge) as f64;
    let projected = projected_dense_peak_gb(s);
    let budget_gb = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);
    let safe = budget_gb * 0.85;
    if projected > safe {
        let px = latent_edge * SPATIAL_SCALE as usize;
        return Err(format!(
            "ltx_2_3 trainer: a dense first training step at resolution {px} needs ~{projected:.0} GB \
             (the forward working set materializes in one allocation), exceeding this machine's ~{safe:.0} GB \
             safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the OS would hard-kill the \
             worker (SIGKILL) at the first step with no recoverable error (sc-4874/sc-4942). Enable Gradient \
             Checkpointing (recomputes block activations in the backward) or reduce the training resolution."
        )
        .into());
    }
    Ok(())
}

/// Write the trained LoRA as safetensors keyed by the LTX module paths — `{module}.lora_A.weight`
/// `[rank,in]`, `{module}.lora_B.weight` `[out,rank]`, scalar `{module}.alpha` (= `alpha`) — the
/// reference `_save_lora` format, reloadable by [`crate::apply_ltx_adapters`] (which folds
/// `scale = alpha/rank`). `networkType`/`rank`/`alpha` metadata mirrors the other family trainers.
fn save_lora(
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: u32,
    path: &Path,
) -> Result<()> {
    let alphas: Vec<(String, Array)> = targets
        .iter()
        .map(|t| {
            (
                format!("{}.alpha", t.save_path),
                Array::from_slice(&[alpha], &[1]),
            )
        })
        .collect();
    let mut entries: Vec<(String, &Array)> = Vec::with_capacity(targets.len() * 3);
    for t in targets {
        entries.push((format!("{}.lora_A.weight", t.save_path), &params[&t.a_key]));
        entries.push((format!("{}.lora_B.weight", t.save_path), &params[&t.b_key]));
    }
    for (k, v) in &alphas {
        entries.push((k.clone(), v));
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("networkType".to_string(), "lora".to_string());
    meta.insert("rank".to_string(), rank.to_string());
    meta.insert("alpha".to_string(), alpha.to_string());
    Array::save_safetensors(entries, Some(&meta), path)?;
    Ok(())
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

// ===========================================================================================
// sc-4942 — first-step memory + grad-parity harness (weight-gated, run as its own process).
//
// Ports the z-image sc-4874/4886/4887 `first_step_repro` harness to LTX: drives the exact inner
// training step (`compute_loss_grads` + the step-1 grad `eval`) directly, sweeping resolution with
// MLX peak-memory probes, and asserts the three levers' invariants on REAL weights:
//   * attention-segment checkpointing is bit-identical to the retained backward,
//   * block (gradient) checkpointing matches the dense path within fp tolerance,
//   * bf16 grads point the same way as f32 and materially shrink the working set.
//
//   cargo test -p mlx-gen-ltx --release --lib first_step -- --ignored --nocapture
// ===========================================================================================
#[cfg(test)]
mod first_step_repro {
    use super::*;
    use mlx_gen::media::Image;
    use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};
    use std::path::PathBuf;

    const RANK: i32 = 8;
    const ALPHA: f32 = 8.0;

    fn snapshot() -> PathBuf {
        if let Ok(p) = std::env::var("LTX_BASE_DIR") {
            return PathBuf::from(p);
        }
        let home = std::env::var("HOME").unwrap();
        PathBuf::from(home)
            .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
    }

    /// A solid-colour `edge`×`edge` RGB source (the latent magnitude is irrelevant; the graph size —
    /// driven by resolution — is the variable under test).
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

    /// Build a real LTX trainer + LoRA targets, encode one caption into a cached f32 context, then free
    /// the Gemma TE + tokenizer (so the measured peaks reflect the post-free training working set, like
    /// `train_impl`). Returns the trainer, the targets/params, the cached context, and the per-block
    /// target grouping for the checkpointed path.
    #[allow(clippy::type_complexity)]
    fn build() -> (
        LtxTrainer,
        Vec<LtxLoraTarget>,
        LoraParams,
        Array,
        Vec<Vec<BlockLoraRef>>,
    ) {
        let mut trainer = load_trainer_from_dir(&snapshot())
            .expect("LTX-2.3 base snapshot (SceneWorks cache or $LTX_BASE_DIR) + Gemma TE");
        let suffixes: Vec<String> = DEFAULT_TARGET_SUFFIXES
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (targets, params) = build_targets(
            &mut trainer.transformer,
            trainer.cfg.num_layers,
            &suffixes,
            RANK,
            7,
        )
        .unwrap();
        let n_layers = trainer.cfg.num_layers as usize;
        let block_targets = group_block_targets(&targets, n_layers);
        let ctx = {
            let te = trainer.text_encoder.as_ref().unwrap();
            let tok = trainer.tokenizer.as_ref().unwrap();
            let (ids, mask) = tok
                .encode("a solid colour swatch", MAX_PROMPT_TOKENS)
                .unwrap();
            to_dtype(&te.encode(&ids, &mask).unwrap(), Dtype::Float32).unwrap()
        };
        eval([&ctx]).unwrap();
        trainer.text_encoder = None;
        trainer.tokenizer = None;
        clear_cache();
        eprintln!(
            "[sc-4942] loaded LTX trainer (TE freed); {} LoRA targets; ctx {:?}",
            targets.len(),
            ctx.shape()
        );
        (trainer, targets, params, ctx, block_targets)
    }

    /// Run a single first training step at `edge` and report peak GPU memory across forward+backward.
    /// Forces the backward (grad eval) — the real step-1 kill point. `checkpoint` selects the
    /// block-checkpointed forward; the caller sets the SDPA-checkpoint flag on the transformer.
    #[allow(clippy::too_many_arguments)]
    fn one_step(
        trainer: &mut LtxTrainer,
        targets: &[LtxLoraTarget],
        params: &LoraParams,
        ctx: &Array,
        block_targets: &[Vec<BlockLoraRef>],
        edge: u32,
        checkpoint: bool,
        dtype: Dtype,
        tag: &str,
    ) -> Result<(f32, f64, Vec<i32>)> {
        let le = (edge / SPATIAL_SCALE as u32).max(1) as usize;
        let img = center_crop_square(&swatch(edge));
        let prep = preprocess_conditioning_image(&img, edge, edge)?;
        let latent = trainer.vae.encode(&prep)?;
        let clean = flatten_latent(&latent)?;
        eval([&clean])?;
        let positions = create_position_grid(1, 1, le, le);
        let noise = random::normal::<f32>(clean.shape(), None, None, Some(&random::key(1)?))?;
        eval([&noise])?;

        let ck = if checkpoint {
            Some(block_targets)
        } else {
            None
        };
        clear_cache();
        reset_peak_memory();
        let before = get_active_memory();
        let t0 = std::time::Instant::now();
        let (loss, grads) = compute_loss_grads(
            &mut trainer.transformer,
            params,
            targets,
            ALPHA,
            RANK as f32,
            &clean,
            ctx,
            &positions,
            0.5,
            &noise,
            false,
            ck,
            dtype,
        )?;
        // `compute_loss_grads` only forces the loss (forward). The real trainer forces the backward at
        // the step-1 optimizer `eval`; do the same here so the peak reflects the true working set.
        eval(grads.values())?;
        let secs = t0.elapsed().as_secs_f64();
        let peak = get_peak_memory();
        let shape = clean.shape().to_vec();
        eprintln!(
            "  [edge {edge:>4} {tag}] latent {shape:?}  loss {loss:.5}  active-before {:.2} GB  peak {:.2} GB  step {secs:.2}s",
            gb(before),
            gb(peak)
        );
        Ok((loss, gb(peak), shape))
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

    /// Grads at `edge` for a given (checkpoint, dtype, sdpa) configuration, backward forced.
    #[allow(clippy::too_many_arguments)]
    fn grads_of(
        trainer: &mut LtxTrainer,
        targets: &[LtxLoraTarget],
        params: &LoraParams,
        ctx: &Array,
        block_targets: &[Vec<BlockLoraRef>],
        edge: u32,
        checkpoint: bool,
        dtype: Dtype,
    ) -> LoraParams {
        let le = (edge / SPATIAL_SCALE as u32).max(1) as usize;
        let img = center_crop_square(&swatch(edge));
        let prep = preprocess_conditioning_image(&img, edge, edge).unwrap();
        let latent = trainer.vae.encode(&prep).unwrap();
        let clean = flatten_latent(&latent).unwrap();
        let positions = create_position_grid(1, 1, le, le);
        let noise =
            random::normal::<f32>(clean.shape(), None, None, Some(&random::key(1).unwrap()))
                .unwrap();
        eval([&clean, &noise]).unwrap();
        let ck = if checkpoint {
            Some(block_targets)
        } else {
            None
        };
        let (_l, g) = compute_loss_grads(
            &mut trainer.transformer,
            params,
            targets,
            ALPHA,
            RANK as f32,
            &clean,
            ctx,
            &positions,
            0.5,
            &noise,
            false,
            ck,
            dtype,
        )
        .unwrap();
        eval(g.values()).unwrap();
        g
    }

    /// sc-4942 — the always-on attention-segment checkpointing must not change the math: grads with the
    /// SDPA checkpoint on must match the retained backward (flag off). Same decomposed attention,
    /// recomputed instead of retained → (near-)bit-identical.
    #[test]
    #[ignore = "needs real LTX-2.3 + Gemma weights; run as its own process"]
    fn attn_ckpt_grads_match_retained() {
        let (mut trainer, targets, params, ctx, bt) = build();
        let edge = 256u32; // small; the math is resolution-agnostic
        trainer.transformer.set_sdpa_checkpoint(false);
        let g_retained = grads_of(
            &mut trainer,
            &targets,
            &params,
            &ctx,
            &bt,
            edge,
            false,
            Dtype::Float32,
        );
        trainer.transformer.set_sdpa_checkpoint(true);
        let g_ckpt = grads_of(
            &mut trainer,
            &targets,
            &params,
            &ctx,
            &bt,
            edge,
            false,
            Dtype::Float32,
        );
        let max_rel = max_rel_diff(&g_retained, &g_ckpt);
        eprintln!("[sc-4942] attn-ckpt-vs-retained grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-5,
            "attention-segment checkpointing must not change grads: max rel {max_rel:.2e}"
        );
    }

    /// sc-4942 — block (gradient) checkpointing must not change the math: the checkpointed forward+grads
    /// must match the dense path within fp tolerance (same install + block forward, recompute-only).
    /// This gate also catches the multi-output-VJP duplicate-cotangent bug (each checkpoint returns one
    /// distinct array, so it should pass).
    #[test]
    #[ignore = "needs real LTX-2.3 + Gemma weights; run as its own process"]
    fn block_ckpt_grads_match_dense() {
        let (mut trainer, targets, params, ctx, bt) = build();
        let edge = 256u32;
        trainer.transformer.set_sdpa_checkpoint(true);
        let g_dense = grads_of(
            &mut trainer,
            &targets,
            &params,
            &ctx,
            &bt,
            edge,
            false,
            Dtype::Float32,
        );
        trainer.transformer.set_sdpa_checkpoint(false);
        let g_ckpt = grads_of(
            &mut trainer,
            &targets,
            &params,
            &ctx,
            &bt,
            edge,
            true,
            Dtype::Float32,
        );
        let max_rel = max_rel_diff(&g_dense, &g_ckpt);
        eprintln!("[sc-4942] block-ckpt-vs-dense grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 5e-3,
            "block checkpointing must match dense within tolerance: max rel {max_rel:.2e}"
        );
    }

    /// sc-4942 — the MEASURED finding that LTX trains f32, not bf16 (the rest of the family casts to
    /// bf16; LTX deliberately does not — see the `compute_dtype` note in `train_impl`). bf16 *does*
    /// shrink the working set (~30 %), but its gradient DECORRELATES from the f32 (quality) path:
    /// global cosine 0.31–0.45, with the early/deep K projections of both attentions pointing
    /// ~opposite — the 48-block distilled DiT's chaos-sensitivity (the same reason inference uses
    /// `quant_f32`). This test pins that finding (asserts the decorrelation, NOT agreement) so a future
    /// change that accidentally re-enables bf16 training is caught, and documents the memory delta that
    /// makes the trade unattractive (f32 already fits the video tier). Runs f32 first (the cast is
    /// destructive), then casts the same trainer to bf16.
    #[test]
    #[ignore = "needs real LTX-2.3 + Gemma weights; run as its own process"]
    fn bf16_grads_decorrelate_justifying_f32() {
        let (mut trainer, targets, params, ctx, bt) = build();
        trainer.transformer.set_sdpa_checkpoint(true);

        // Memory A/B at 768 (activations dominate the peak).
        let (_, f32_peak, _) = one_step(
            &mut trainer,
            &targets,
            &params,
            &ctx,
            &bt,
            768,
            false,
            Dtype::Float32,
            "attn-ckpt f32",
        )
        .expect("f32 step");
        // Grad reference at 256 in f32.
        let g_f32 = grads_of(
            &mut trainer,
            &targets,
            &params,
            &ctx,
            &bt,
            256,
            false,
            Dtype::Float32,
        );

        trainer
            .transformer
            .cast_weights(Dtype::Bfloat16)
            .expect("cast");
        clear_cache();
        let g_bf16 = grads_of(
            &mut trainer,
            &targets,
            &params,
            &ctx,
            &bt,
            256,
            false,
            Dtype::Bfloat16,
        );

        // Cosine between bf16 and f32 grads (both arrive f32 through the astype VJP). Gate on the GLOBAL
        // cosine (the concatenated gradient the optimizer follows) and the large-norm minimum; tiny-norm
        // params whose direction bf16 rounding scrambles contribute nothing to the update.
        let mut per: Vec<(String, f32, f32)> = Vec::new(); // (key, cos, na)
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
                per.push((k.to_string(), dot / (na * nb), na));
            }
        }
        let global_cos = (gdot / (gna2.sqrt() * gnb2.sqrt())) as f32;
        per.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
        let max_norm = per.iter().map(|p| p.2).fold(0f32, f32::max);
        eprintln!("[sc-4942] bf16-vs-f32 grads: global cosine {global_cos:.5}; worst per-param:");
        for (k, c, na) in per.iter().take(5) {
            eprintln!(
                "    {k}: cos {c:.4}  |g| {na:.3e}  rel-norm {:.2e}",
                na / max_norm
            );
        }
        let min_large = per
            .iter()
            .filter(|p| p.2 >= 0.01 * max_norm)
            .map(|p| p.1)
            .fold(1f32, f32::min);
        eprintln!("[sc-4942] min cosine among params with |g| >= 1% of max: {min_large:.4}");
        // The FINDING (not a regression): bf16 grads do NOT track f32 on this distilled stack. Pin it
        // loosely (global cosine well under the family's >0.99 bar) so an accidental re-enable is caught.
        assert!(
            global_cos < 0.9,
            "expected bf16 to decorrelate from f32 on the LTX distilled stack (the reason LTX trains \
             f32); if this is now high, bf16 training may be viable — re-evaluate: {global_cos:.5}"
        );

        let (_, bf16_peak, _) = one_step(
            &mut trainer,
            &targets,
            &params,
            &ctx,
            &bt,
            768,
            false,
            Dtype::Bfloat16,
            "attn-ckpt bf16",
        )
        .expect("bf16 step");
        // Informational: bf16 does shrink the working set — but f32 already fits the tier, so the
        // quality cost above is not worth taking.
        eprintln!(
            "[sc-4942] 768 peak f32 {f32_peak:.2} GB vs bf16 {bf16_peak:.2} GB ({:.0}%) — f32 fits the \
             video tier, so bf16's saving is not needed",
            100.0 * bf16_peak / f32_peak
        );
    }

    /// sc-4942 — first-step peak sweep on the dense path (attention-segment checkpointing always on),
    /// f32 then bf16, plus a block-ckpt point. These measured points are the basis of the
    /// `projected_dense_peak_gb` guard fit — refit the constants if this prints materially different
    /// numbers.
    #[test]
    #[ignore = "needs real LTX-2.3 + Gemma weights; run as its own process (may SIGKILL at large edge)"]
    fn first_step_attn_ckpt_sweep() {
        let (mut trainer, targets, params, ctx, bt) = build();
        trainer.transformer.set_sdpa_checkpoint(true);
        eprintln!("[sc-4942] attn-ckpt dense sweep, f32:");
        for edge in [512u32, 768, 1024] {
            let _ = one_step(
                &mut trainer,
                &targets,
                &params,
                &ctx,
                &bt,
                edge,
                false,
                Dtype::Float32,
                "attn-ckpt f32",
            )
            .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
        }
        eprintln!("[sc-4942] block-ckpt at 1024, f32:");
        trainer.transformer.set_sdpa_checkpoint(false);
        let _ = one_step(
            &mut trainer,
            &targets,
            &params,
            &ctx,
            &bt,
            1024,
            true,
            Dtype::Float32,
            "blk-ckpt f32",
        )
        .map_err(|e| eprintln!("  blk-ckpt CATCHABLE error: {e}"));

        eprintln!("[sc-4942] casting weights to bf16…");
        trainer
            .transformer
            .cast_weights(Dtype::Bfloat16)
            .expect("cast");
        trainer.transformer.set_sdpa_checkpoint(true);
        clear_cache();
        eprintln!("[sc-4942] attn-ckpt dense sweep, bf16:");
        for edge in [512u32, 768, 1024] {
            let _ = one_step(
                &mut trainer,
                &targets,
                &params,
                &ctx,
                &bt,
                edge,
                false,
                Dtype::Bfloat16,
                "attn-ckpt bf16",
            )
            .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
        }
        eprintln!("[sc-4942] block-ckpt + bf16 at 1024:");
        trainer.transformer.set_sdpa_checkpoint(false);
        let _ = one_step(
            &mut trainer,
            &targets,
            &params,
            &ctx,
            &bt,
            1024,
            true,
            Dtype::Bfloat16,
            "blk-ckpt bf16",
        )
        .map_err(|e| eprintln!("  blk-ckpt bf16 CATCHABLE error: {e}"));
    }

    /// sc-4942 — block checkpointing must drop the first-step peak below the dense path at production
    /// resolution. Runs the dense step first (baseline), then the checkpointed step.
    #[test]
    #[ignore = "needs real LTX-2.3 + Gemma weights; run as its own process"]
    fn block_ckpt_reduces_peak_vs_dense() {
        let (mut trainer, targets, params, ctx, bt) = build();
        trainer.transformer.set_sdpa_checkpoint(true);
        let (_, dense_peak, _) = one_step(
            &mut trainer,
            &targets,
            &params,
            &ctx,
            &bt,
            1024,
            false,
            Dtype::Float32,
            "dense",
        )
        .expect("dense step");
        trainer.transformer.set_sdpa_checkpoint(false);
        let (_, ckpt_peak, _) = one_step(
            &mut trainer,
            &targets,
            &params,
            &ctx,
            &bt,
            1024,
            true,
            Dtype::Float32,
            "blk-ckpt",
        )
        .expect("checkpointed step");
        eprintln!(
            "[sc-4942] edge 1024  dense {dense_peak:.2} GB  ckpt {ckpt_peak:.2} GB  ({:.0}% reduction)",
            100.0 * (1.0 - ckpt_peak / dense_peak)
        );
        assert!(
            ckpt_peak < dense_peak,
            "block checkpointing must reduce the first-step peak: dense {dense_peak:.2} vs ckpt {ckpt_peak:.2}"
        );
    }
}

#[cfg(test)]
mod preflight_tests {
    use super::projected_dense_peak_gb;

    /// The empirical fit must reproduce the measured first-step peaks (the basis of the pre-flight OOM
    /// guard) and stay monotonic. Measured by `first_step_repro::first_step_attn_ckpt_sweep` (128 GB
    /// Mac, f32, attention-segment checkpointing on): s = (edge/32)² → 512/768/1024 = 256/576/1024
    /// tokens → 23.3/31.5/42.6 GB.
    #[test]
    fn projection_matches_measured_curve() {
        for (s, measured) in [(256.0, 23.3), (576.0, 31.5), (1024.0, 42.6)] {
            let p = projected_dense_peak_gb(s);
            assert!(
                (p - measured).abs() < 1.5,
                "f32 projection at s={s} = {p:.1} GB, expected ≈{measured} GB"
            );
        }
        // Monotonic increasing in token count.
        assert!(projected_dense_peak_gb(256.0) < projected_dense_peak_gb(576.0));
        assert!(projected_dense_peak_gb(576.0) < projected_dense_peak_gb(1024.0));
        // 1024 still fits a 64 GB video tier (budget ≈ 54 GB) without block-checkpointing.
        assert!(projected_dense_peak_gb(1024.0) < 54.0);
    }
}

#[cfg(test)]
mod validate_request_tests {
    use super::validate_request;
    use mlx_gen::{NetworkType, TrainingConfig, TrainingItem, TrainingRequest};
    use std::path::PathBuf;

    fn request(items: usize) -> TrainingRequest {
        TrainingRequest {
            items: (0..items)
                .map(|i| TrainingItem {
                    image_path: PathBuf::from(format!("img{i}.png")),
                    caption: "a cat".into(),
                })
                .collect(),
            config: TrainingConfig::default(),
            output_dir: PathBuf::from("/tmp/ltx-trainer-test"),
            file_name: "adapter.safetensors".into(),
            trigger_words: vec![],
            cancel: Default::default(),
        }
    }

    #[test]
    fn accepts_valid_and_rejects_bad_requests() {
        assert!(validate_request(&request(1)).is_ok());
        assert!(validate_request(&request(0)).is_err()); // empty dataset

        let mut r = request(1);
        r.config.rank = 0;
        assert!(validate_request(&r).is_err()); // zero rank

        let mut r = request(1);
        r.config.network_type = NetworkType::Lokr;
        assert!(validate_request(&r).is_err()); // LoKr is LoRA-only here

        let mut r = request(1);
        r.config.optimizer = "sgd".into();
        assert!(validate_request(&r).is_err()); // unsupported optimizer
    }
}
