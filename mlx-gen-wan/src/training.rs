//! sc-3046 — Wan2.2 LoRA/LoKr **training** on the A14B dual-expert MoE DiT, in pure Rust on mlx-rs.
//! The Rust port of SceneWorks' `WanMoeLoraTrainer` (the dual-expert path) realizing the core
//! [`Trainer`] contract (epic 3039), on the functional-autograd mechanism the spike proved.
//!
//! **Dual-expert MoE.** Wan2.2-A14B is two full transformers — a **high-noise** expert (denoise
//! timesteps `[boundary, 1]`) and a **low-noise** expert (`[0, boundary]`), `boundary = 0.875`. Each
//! gets its **own** LoRA (separate factor map + optimizer + LR schedule). Training **alternates** per
//! micro-step (the reference's `step % 2`): odd steps train the high expert on a timestep sampled in
//! its band, even steps the low expert in its band. Two adapters are saved —
//! `{stem}.high_noise.safetensors` + `{stem}.low_noise.safetensors` — which the inference loader
//! consumes per expert (`AdapterSpec.moe_expert{High,Low}`, sc-2683).
//!
//! **Wan-specific pieces** (the rest is the shared core `train::lora` machinery + sc-3043 runtime):
//!   * **Engine seam.** Wan inference *merges* LoRA into the weight map at load (no forward-time
//!     residual), so `WanTransformer` gained an [`AdaptableHost`] impl (sc-3046) for training: the
//!     trainer injects a fresh `Adapter::Lora` per target each step via
//!     [`AdaptableLinear::set_adapters`](mlx_gen::adapters::AdaptableLinear::set_adapters) and the
//!     block forward applies it.
//!   * **Flow-match target = `noise - clean`** (the raw transformer output is the velocity, no
//!     negation — opposite to Z-Image), **timestep = `t·1000`** (`t∈[0,1]` scaled, NOT `1-σ`), `t`
//!     sampled per-expert *within its noise band*.
//!   * **3D-VAE latents** — a still image VAE-encodes (single frame T=1) to the z16 **normalized**
//!     latent `[16, 1, h, w]` (the VAE applies the per-channel mean/std), the space the DiT operates
//!     in. The UMT5 context is embedded **per expert** (each expert has its own `text_embedding`); the
//!     ~11 GB UMT5 encoder is freed after the one-time cache.
//!   * **Native target naming** — `blocks.{i}.self_attn.{q,k,v,o}` / `cross_attn.{q,k,v,o}` (the
//!     reference `to_q/k/v/to_out.0` suffix surface in Wan's native checkpoint naming). Saved bare,
//!     so `apply_wan_adapters`' `normalize_wan_key` resolves them on reload. LoRA + LoKr (LoKr
//!     reconstructs f32, matching the Wan merge path).
//!
//! **sc-3279 — the two Wan siblings, folded into this one trainer.** The mechanism is identical
//! (still-image T=1 flow-match velocity regression, per the reference `_WanLoraBackend`/
//! `_WanMoeLoraBackend`); only the latent space + input channels differ, so the same struct serves
//! all three Wan trainers via three registrations:
//!   * **Dense TI2V-5B** (`wan2_2_ti2v_5b`, single expert) — the z48 [`Wan22Vae`] (channels-last
//!     `[1,1,H,W,3]` encode, transposed to channels-first `[48,1,h,w]`), in_dim 48.
//!   * **I2V-A14B** (`wan2_2_i2v_14b`, dual expert, boundary 0.900) — the same z16 VAE, but the
//!     forward needs in_dim 36, so the 16-channel noisy latent is concatenated with a **zero** 20-
//!     channel `y` (the reference trains the no-conditioning T2V velocity objective; the attention
//!     LoRA is the trained surface and is blind to the padded conditioning channels).

use std::path::Path;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::media::Image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, average_grads, build_lokr_targets, build_lora_targets, LoraParams,
    TrainAdapter,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::weights::Weights;
use mlx_gen::{
    gen_core, CancelFlag, LoadSpec, Modality, NetworkType, Result, TrainOptimizer, Trainer,
    TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{add, concatenate_axis, multiply, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::config::WanModelConfig;
use crate::model::{MODEL_ID as MODEL_ID_TI2V_5B, MODEL_ID_I2V_14B, MODEL_ID_T2V_14B};
use crate::pipeline::{
    decode_to_frames, decode_to_frames_22, denoise, denoise_moe, frames_to_images,
    preprocess_i2v_image, preprocess_ti2v_image, Expert,
};
use crate::scheduler::SolverKind;
use crate::text_encoder::{load_tokenizer, Umt5Encoder};
use crate::transformer::{BlockLoraRef, WanTransformer};
use crate::vae::WanVae;
use crate::vae22::Wan22Vae;

/// Wan reconstructs its LoKr delta at **f32** (the f32 merge path, `merge_one_lokr`); training matches
/// so the adapter round-trips.
const LOKR_DTYPE: Dtype = Dtype::Float32;

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence (sc-5637).
const SAMPLE_PROMPT_CAP: usize = 4;

/// The reference attention LoRA targets `to_q/to_k/to_v/to_out.0` in Wan's **native** naming: the
/// self/cross-attention `q/k/v/o`. Suffix-matched against the per-block adaptable surface.
const DEFAULT_TARGET_SUFFIXES: [&str; 4] = ["q", "k", "v", "o"];

/// Render one preview frame (sc-5637) from the **in-progress training adapters** already installed on
/// `experts`: a single-frame (`F=1`) seeded denoise reusing the family's own inference loop — the
/// boundary-switched dual-expert [`denoise_moe`] for the A14B MoE, or the dense single-expert
/// [`denoise`] for TI2V-5B — then VAE decode and the first frame as an [`Image`] (a video LoRA's
/// preview is a still thumbnail). `ctxs` are this prompt's per-expert [`WanTransformer::embed_text`]
/// outputs; `latent_shape` is a cached clean latent's `[z, 1, h, w]` (so the init noise matches the
/// VAE's exact latent geometry without re-deriving the per-family stride). CFG is off (guidance 1.0).
/// No progress/cancel plumbing — the caller drives the cadence.
fn render_wan_sample(
    experts: &[WanTransformer],
    vae: &WanTrainVae,
    cfg: &WanModelConfig,
    ctxs: &[Array],
    latent_shape: &[i32],
    seed: u64,
    steps: usize,
) -> Result<Image> {
    let init = random::normal::<f32>(latent_shape, None, None, Some(&random::key(seed)?))?;
    let kind = SolverKind::from_name("uni_pc");
    let ntt = cfg.num_train_timesteps;
    let shift = cfg.sample_shift;
    let latents = if experts.len() == 2 {
        let boundary_ts = cfg.boundary * ntt as f32;
        let low = Expert {
            transformer: &experts[0],
            ctx_cond: ctxs[0].clone(),
            ctx_uncond: None,
            guidance: 1.0,
        };
        let high = Expert {
            transformer: &experts[1],
            ctx_cond: ctxs[1].clone(),
            ctx_uncond: None,
            guidance: 1.0,
        };
        denoise_moe(
            &low,
            &high,
            boundary_ts,
            kind,
            ntt,
            steps.max(1),
            shift,
            &init,
            None,
            &CancelFlag::default(),
            &mut |_| {},
        )?
    } else {
        denoise(
            &experts[0],
            kind,
            ntt,
            steps.max(1),
            shift,
            1.0,
            &ctxs[0],
            None,
            &init,
            &CancelFlag::default(),
            &mut |_| {},
        )?
    };
    let frames = match vae {
        WanTrainVae::Z16(v) => decode_to_frames(v, &latents, None)?,
        WanTrainVae::Z48(v) => decode_to_frames_22(v, &latents, None)?,
    };
    frames_to_images(&frames)?
        .into_iter()
        .next()
        .ok_or_else(|| mlx_gen::Error::Msg("wan trainer: preview produced no frames".into()))
}

/// The VAE the trainer encodes its dataset through — the family-specific latent space the DiT
/// regresses velocity in. **z16** 2.1 [`WanVae`] for the 14B T2V/I2V experts; **z48** [`Wan22Vae`]
/// for the dense TI2V-5B (sc-3279). Both produce a normalized **channels-first** clean latent
/// `[z, 1, h, w]` (a still image is one latent frame, `T = 1`).
enum WanTrainVae {
    Z16(WanVae),
    Z48(Wan22Vae),
}

impl WanTrainVae {
    /// Encode a center-cropped square still image → the normalized channels-first clean latent
    /// `[z, 1, h, w]`, exactly as each model's inference encode: the z16 video encode squeezed to
    /// one frame; the z48 channels-last image encode transposed channels-first (`model.rs`'s TI2V
    /// conditioning, `transpose([3,0,1,2])`).
    fn encode_clean(&self, img: &Image, edge: u32) -> Result<Array> {
        match self {
            WanTrainVae::Z16(vae) => {
                let chw = preprocess_i2v_image(img, edge, edge)?; // [3, H, W] in [-1,1]
                let nct_hw = chw.reshape(&[1, 3, 1, edge as i32, edge as i32])?; // [1,3,1,H,W]
                let latent = vae.encode(&nct_hw)?; // [1,16,1,h,w] normalized
                let s = latent.shape();
                Ok(latent.reshape(&[s[1], s[2], s[3], s[4]])?) // [16,1,h,w]
            }
            WanTrainVae::Z48(vae) => {
                let thwc = preprocess_ti2v_image(img, edge, edge)?; // [1,1,H,W,3] channels-last
                let latent = vae.encode(&thwc)?; // [1,1,h,w,48] channels-last normalized
                                                 // [1,1,h,w,48] → [1,h,w,48] → [48,1,h,w] (channels-first, the latent convention).
                Ok(latent
                    .reshape(&latent.shape()[1..])?
                    .transpose_axes(&[3, 0, 1, 2])?)
            }
        }
    }
}

/// Per-expert training state: its save suffix + noise band, plus its own adapter / factor map /
/// optimizer / grad accumulator / LR-schedule bookkeeping (each expert trains independently on the
/// micro-steps routed to it).
struct ExpertState {
    suffix: &'static str, // "" for dense, "high_noise"/"low_noise" for the MoE files
    band: (f32, f32),     // the timestep band this expert is sampled in
    adapter: TrainAdapter,
    params: LoraParams,
    /// sc-4942 — this expert's per-block trainable targets, for the gradient-checkpointed forward
    /// (`block_targets[i]` = block `i`'s targets). Empty when block checkpointing is off.
    block_targets: Vec<Vec<BlockLoraRef>>,
    opt: TrainOptimizer,
    accumulated: Option<LoraParams>,
    micro: u32,      // micro-steps routed to this expert so far (drives accumulation)
    update_idx: u32, // optimizer updates applied (drives the LR schedule)
    total_updates: u32,
    warmup_updates: u32,
}

/// Dual-expert (A14B MoE) Wan2.2 LoRA/LoKr trainer. Loads both experts + the z16 VAE + UMT5 + the
/// tokenizer, caches a captioned image dataset to (normalized latent, per-expert context) pairs, then
/// runs the alternating per-expert functional-autograd flow-match loop with the sc-3043 runtime glue,
/// and writes one adapter per expert that round-trips through `apply_wan_adapters`.
pub struct WanMoeTrainer {
    descriptor: TrainerDescriptor,
    tokenizer: Option<TextTokenizer>,
    text_encoder: Option<Umt5Encoder>,
    vae: WanTrainVae,
    /// `[low, high]` for the dual-expert MoE; `[single]` for a dense checkpoint.
    experts: Vec<WanTransformer>,
    cfg: WanModelConfig,
}

fn trainer_descriptor(id: &'static str) -> TrainerDescriptor {
    TrainerDescriptor {
        id,
        family: "wan",
        backend: "mlx",
        modality: Modality::Video,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// Dual-expert MoE T2V-A14B trainer (z16 VAE, in_dim 16) — sc-3046.
fn descriptor_t2v_14b() -> TrainerDescriptor {
    trainer_descriptor(MODEL_ID_T2V_14B)
}
/// Dual-expert MoE I2V-A14B trainer (z16 VAE, channel-concat in_dim 36, boundary 0.900) — sc-3279.
fn descriptor_i2v_14b() -> TrainerDescriptor {
    trainer_descriptor(MODEL_ID_I2V_14B)
}
/// Dense TI2V-5B trainer (z48 vae22, in_dim 48, single expert) — sc-3279.
fn descriptor_ti2v_5b() -> TrainerDescriptor {
    trainer_descriptor(MODEL_ID_TI2V_5B)
}

/// Construct the Wan trainer from a converted MLX snapshot directory, picking the VAE (z16 vs z48)
/// and expert layout (dual `low_noise_model`/`high_noise_model` vs dense `model.safetensors`) from
/// the checkpoint's `config.json`. The transformers load bf16 (Wan's native dtype; the trainable f32
/// factors promote against the bf16 base — clean autograd, the base frozen). `descriptor.id` selects
/// which Wan variant this registration serves, and is checked against the config.
fn build_trainer(spec: &LoadSpec, descriptor: TrainerDescriptor) -> Result<Box<dyn Trainer>> {
    Ok(Box::new(build_trainer_concrete(spec, descriptor)?))
}

/// The concrete-typed loader behind [`build_trainer`] (sc-4942 — the first-step memory harness needs
/// the concrete [`WanMoeTrainer`] to reach `.experts` / `.vae`, which a `Box<dyn Trainer>` hides).
fn build_trainer_concrete(spec: &LoadSpec, descriptor: TrainerDescriptor) -> Result<WanMoeTrainer> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(mlx_gen::Error::Msg(format!(
                "{} trainer expects a converted snapshot directory (model/expert safetensors + \
                 t5_encoder + vae + tokenizer.json), not a single file",
                descriptor.id
            )))
        }
    };
    let cfg = WanModelConfig::from_model_dir(root)?;
    check_config_matches(descriptor.id, &cfg)?;
    let tokenizer = load_tokenizer(root.join("tokenizer.json"), cfg.text_len)?;
    let t5_w = Weights::from_file(root.join("t5_encoder.safetensors"))?;
    let text_encoder = Umt5Encoder::from_weights(&t5_w, &cfg)?;
    let vae_w = Weights::from_file(root.join("vae.safetensors"))?;
    // The 5B operates in the z48 vae22 latent space; the 14B T2V/I2V experts in the z16 2.1 VAE.
    let vae = if cfg.vae_z_dim == 48 {
        WanTrainVae::Z48(Wan22Vae::from_weights(&vae_w)?)
    } else {
        WanTrainVae::Z16(WanVae::from_weights(&vae_w)?)
    };

    let experts = if cfg.dual_model {
        let low_w = Weights::from_file(root.join("low_noise_model.safetensors"))?;
        let high_w = Weights::from_file(root.join("high_noise_model.safetensors"))?;
        vec![
            WanTransformer::from_weights(&low_w, &cfg)?,
            WanTransformer::from_weights(&high_w, &cfg)?,
        ]
    } else {
        let w = Weights::from_file(root.join("model.safetensors"))?;
        vec![WanTransformer::from_weights(&w, &cfg)?]
    };

    Ok(WanMoeTrainer {
        descriptor,
        tokenizer: Some(tokenizer),
        text_encoder: Some(text_encoder),
        vae,
        experts,
        cfg,
    })
}

/// Reject a snapshot whose `config.json` doesn't match the registration id (a dense 5B routed to the
/// MoE id, or vice-versa) before the expensive weight load — the same shape checks the `model.rs`
/// generators apply.
fn check_config_matches(id: &str, cfg: &WanModelConfig) -> Result<()> {
    let ok = match id {
        _ if id == MODEL_ID_TI2V_5B => !cfg.dual_model && cfg.vae_z_dim == 48,
        _ if id == MODEL_ID_I2V_14B => cfg.dual_model && cfg.is_i2v_concat(),
        _ if id == MODEL_ID_T2V_14B => cfg.dual_model && !cfg.is_i2v_concat(),
        _ => true,
    };
    if !ok {
        return Err(mlx_gen::Error::Msg(format!(
            "{id} trainer: config.json does not match (model_type={}, in_dim={}, dual_model={}, \
             vae_z_dim={})",
            cfg.model_type, cfg.in_dim, cfg.dual_model, cfg.vae_z_dim
        )));
    }
    Ok(())
}

/// Construct the dual-expert MoE T2V-A14B trainer (sc-3046).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    build_trainer(spec, descriptor_t2v_14b())
}
/// Construct the dual-expert MoE I2V-A14B trainer (sc-3279) — same MoE machinery, channel-concat
/// in_dim 36 (the 16-channel latent is zero-`y`-padded to 36 each step).
pub fn load_trainer_i2v_14b(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    build_trainer(spec, descriptor_i2v_14b())
}
/// Construct the dense TI2V-5B trainer (sc-3279) — z48 vae22, single expert.
pub fn load_trainer_ti2v_5b(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    build_trainer(spec, descriptor_ti2v_5b())
}

// Link-time trainer registration (epic 3720): the macro emits each `inventory::submit!` and bridges
// the crate's rich `Result` into the trainer registry's backend-neutral `gen_core::Result`.
mlx_gen::register_trainer! {
    descriptor_t2v_14b => load_trainer,
    descriptor_i2v_14b => load_trainer_i2v_14b,
    descriptor_ti2v_5b => load_trainer_ti2v_5b,
}

impl Trainer for WanMoeTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        let id = self.descriptor.id;
        if req.items.is_empty() {
            return Err(format!("{id} trainer: dataset is empty").into());
        }
        if req.config.rank == 0 {
            return Err(format!("{id} trainer: rank must be > 0").into());
        }
        if !TrainOptimizer::is_supported(&req.config.optimizer) {
            return Err(format!(
                "{id} trainer: optimizer '{}' is not available on MLX training (supported: \
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

impl WanMoeTrainer {
    /// Train body — kept on the crate's own [`mlx_gen::Error`] so `?` on mlx-rs ops and crate
    /// helpers lifts transparently; the trait wrapper bridges the tail into [`gen_core::Error`]
    /// (epic 3720).
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        self.validate(req)?;
        let id = self.descriptor.id;
        let cfg = &req.config;
        let n_experts = self.experts.len();
        let dual = n_experts == 2;
        let boundary = self.cfg.boundary;
        // Channel-concat conditioning width (I2V-A14B: in_dim 36 = latent 16 + y 20; 0 for T2V/5B).
        // The forward gets a zero `y` of this width appended to the noisy latent (sc-3279).
        let y_channels = self.cfg.in_dim as i32 - self.cfg.vae_z_dim as i32;
        on_progress(TrainingProgress::Preparing);
        let edge = bucket_resolution(cfg.resolution);

        // sc-4942 — fail-fast pre-flight memory guard (the sc-4874 mechanism). The dense (non-block-
        // checkpointed) first step materializes the whole forward graph in one MLX `eval`; at high
        // resolution that working set, on top of the resident expert(s), can exceed unified memory and
        // the OS hard-kills the worker with an UNCATCHABLE SIGKILL. Predict it and refuse up front —
        // BEFORE the (~minutes-long) latent/UMT5 caching — when gradient checkpointing is off. (Block-
        // checkpointed runs recompute per block, so they are not subject to the dense peak.)
        let will_checkpoint = cfg.gradient_checkpointing && cfg.network_type == NetworkType::Lora;
        if !will_checkpoint {
            preflight_memory_guard(&self.cfg, edge, n_experts, id)?;
        }

        // --- prepare → load → cache: normalized latents + per-expert UMT5 context (then free the TE) ---
        on_progress(TrainingProgress::LoadingModel);
        let total = req.items.len() as u32;
        let mut cache: Vec<(Array, Vec<Array>)> = Vec::with_capacity(req.items.len());
        // sc-5637 — preview-sample prompts, embedded per expert inside the `te`/`tok` scope below
        // (the UMT5 encoder is freed before the train loop).
        let mut sample_ctxs: Vec<(String, Vec<Array>)> = Vec::new();
        {
            let te = self.text_encoder.as_ref().ok_or_else(|| {
                mlx_gen::Error::Msg(format!("{id} trainer: text encoder missing"))
            })?;
            let tok = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| mlx_gen::Error::Msg(format!("{id} trainer: tokenizer missing")))?;
            for (i, item) in req.items.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    break;
                }
                on_progress(TrainingProgress::Caching {
                    current: i as u32 + 1,
                    total,
                });
                let img = center_crop_square(&decode_image(&item.image_path)?);
                // [z,1,h,w] normalized channels-first (z16 14B / z48 5B — dispatched by the VAE kind).
                let clean = self.vae.encode_clean(&img, edge)?;
                let t5_embed = te.encode(tok, &item.caption)?; // [L, text_dim]
                                                               // Each expert has its own text_embedding, so embed the context per expert.
                let mut ctxs = Vec::with_capacity(n_experts);
                for e in &self.experts {
                    ctxs.push(e.embed_text(&t5_embed)?); // [1, text_len, dim]
                }
                let mut to_eval: Vec<&Array> = vec![&clean];
                to_eval.extend(ctxs.iter());
                eval(to_eval)?;
                cache.push((clean, ctxs));
            }
            // sc-5637 — pre-encode the preview-sample prompts (per expert) while the UMT5 encoder is
            // still resident. Mirrors the per-item embed above: one ctx per expert per prompt.
            if cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() && !req.cancel.is_cancelled()
            {
                for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                    let t5_embed = te.encode(tok, prompt)?;
                    let mut ctxs = Vec::with_capacity(n_experts);
                    for e in &self.experts {
                        ctxs.push(e.embed_text(&t5_embed)?);
                    }
                    let to_eval: Vec<&Array> = ctxs.iter().collect();
                    eval(to_eval)?;
                    sample_ctxs.push((prompt.clone(), ctxs));
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
            return Err(format!("{id} trainer: no usable dataset items").into());
        }
        // Free the UMT5 encoder + tokenizer (~11 GB) before training (the reference frees it post-cache).
        self.text_encoder = None;
        self.tokenizer = None;

        // sc-5637 — preview-sample geometry: a cached clean latent's exact `[z, 1, h, w]` shape (so the
        // preview's init noise matches the VAE's latent geometry without re-deriving the per-family
        // spatial stride). Cache is non-empty here (checked above).
        let sampling_enabled = !sample_ctxs.is_empty();
        let sample_latent_shape: Vec<i32> = cache[0].0.shape().to_vec();

        // --- per-expert adapter targets + factors + optimizer + schedule ---
        let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
            DEFAULT_TARGET_SUFFIXES
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            cfg.lora_target_modules.clone()
        };
        let alpha = cfg.alpha;
        let rank = cfg.rank as f32;
        let mae = {
            let lt = cfg.loss_type.to_ascii_lowercase();
            lt == "mae" || lt == "l1"
        };
        let accum = cfg.gradient_accumulation.max(1);
        let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
            0.0
        } else {
            cfg.weight_decay
        };

        // sc-4942 — gradient checkpointing (opt-in via the SceneWorks toggle). Block-checkpoint only the
        // LoRA path (LoKr reconstructs a Kronecker delta — the dense path handles it, matching the image
        // families); attention-segment checkpointing is ON for the dense path and OFF under block-ckpt
        // (the block recompute already covers attention, so nesting would recompute it twice). Wan is
        // bf16-native (the base loads bf16, every matmul already runs bf16 with an f32 residual), so the
        // `train_dtype=bf16` contract is honored by the load — there is no f32→bf16 cast lever here; the
        // memory fix is the two checkpointing levers + the pre-flight guard.
        let use_checkpoint = cfg.gradient_checkpointing && cfg.network_type == NetworkType::Lora;

        let mut states: Vec<ExpertState> = Vec::with_capacity(n_experts);
        for (idx, expert) in self.experts.iter_mut().enumerate() {
            let target_paths = resolve_target_paths(expert, &suffixes);
            if target_paths.is_empty() {
                return Err(format!(
                    "{id} trainer: no LoRA targets resolved (check lora_target_modules)"
                )
                .into());
            }
            // Distinct seed per expert so the two experts' gaussian init differs.
            let seed = cfg.seed.wrapping_add(idx as u64 * 0x9E37_79B9);
            let (adapter, params) = build_adapter(expert, &target_paths, cfg, seed)?;
            // sc-4942 — group this expert's targets by block for the checkpointed forward, and arm
            // attention-segment checkpointing on the dense path (off under block-ckpt).
            let block_targets = if use_checkpoint {
                group_block_targets(&target_paths, expert.num_blocks())
            } else {
                Vec::new()
            };
            expert.set_sdpa_checkpoint(!use_checkpoint);
            let (band, suffix) = if dual {
                if idx == 0 {
                    ((0.0, boundary), "low_noise")
                } else {
                    ((boundary, 1.0), "high_noise")
                }
            } else {
                ((0.0, 1.0), "")
            };
            // This expert sees ~steps/n_experts micro-steps; size its LR schedule accordingly.
            let expert_micro = (cfg.steps / n_experts as u32).max(1);
            let (total_updates, warmup_updates) =
                schedule_updates(expert_micro, accum, cfg.lr_warmup_steps);
            let opt = TrainOptimizer::from_config(&cfg.optimizer, cfg.learning_rate, weight_decay)?;
            states.push(ExpertState {
                suffix,
                band,
                adapter,
                params,
                block_targets,
                opt,
                accumulated: None,
                micro: 0,
                update_idx: 0,
                total_updates,
                warmup_updates,
            });
        }

        let stem = Path::new(&req.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lora")
            .to_string();
        let ext = Path::new(&req.file_name)
            .extension()
            .and_then(|s| s.to_str())
            .map(|e| format!(".{e}"))
            .unwrap_or_else(|| ".safetensors".to_string());

        // --- train loop: alternate experts (high on odd steps, low on even — the reference's step%2) ---
        let mut last_loss = 0.0f32;
        let mut steps_run = 0u32;
        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            // Dual: high (index 1) on odd steps, low (index 0) on even. Dense: the single expert.
            let ei = if dual && step % 2 == 1 { 1 } else { 0 };
            let (clean, ctxs) = &cache[((step - 1) as usize) % cache.len()];
            let ctx = &ctxs[ei];
            let band = states[ei].band;
            let t = sample_band_timestep(
                &cfg.timestep_type,
                &cfg.timestep_bias,
                band,
                cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64),
            )?;
            let noise = random::normal::<f32>(
                clean.shape(),
                None,
                None,
                Some(&random::key(
                    cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
                )?),
            )?;
            let checkpoint_block = if use_checkpoint {
                Some(states[ei].block_targets.as_slice())
            } else {
                None
            };
            let (loss, grads) = compute_loss_grads(
                &mut self.experts[ei],
                &states[ei].adapter,
                &states[ei].params,
                alpha,
                rank,
                clean,
                ctx,
                t,
                &noise,
                mae,
                y_channels,
                checkpoint_block,
            )?;
            last_loss = loss;
            steps_run = step;

            let st = &mut states[ei];
            st.micro += 1;
            accumulate_grads(&mut st.accumulated, grads)?;
            // Fire an optimizer update every `accum` micro-steps for THIS expert (or on the final step).
            if st.micro.is_multiple_of(accum) || step == cfg.steps {
                let mult = lr_multiplier(
                    cfg.lr_scheduler,
                    st.update_idx,
                    st.total_updates,
                    st.warmup_updates,
                );
                st.opt.set_lr_scaled(mult);
                let avg = average_grads(
                    st.accumulated
                        .take()
                        .expect("an update fires only after accumulation"),
                    accum,
                )?;
                let (clipped, _norm) = clip_grad_norm(&avg, 1.0)?;
                let clipped: LoraParams = clipped
                    .into_iter()
                    .map(|(k, v)| (k, v.into_owned()))
                    .collect();
                st.opt.step(&mut st.params, &clipped)?;
                eval(st.params.values())?;
                st.update_idx += 1;
            }

            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                save_experts(
                    &states,
                    &req.output_dir,
                    &stem,
                    &ext,
                    Some(step),
                    alpha,
                    rank,
                    cfg,
                )?;
                on_progress(TrainingProgress::Checkpoint { step });
            }

            // sc-5637 — periodic best-effort preview frames from the in-progress adapters. Install
            // EVERY expert's current adapter concretely (a render switches experts at the boundary), run
            // the family inference denoise → decode → first frame. The next train step's traced
            // `loss_fn` re-installs the active expert's factors, so no teardown is needed. A render
            // failure logs and is skipped — it never aborts the (long) training run.
            if sampling_enabled && step % cfg.sample_every == 0 {
                for (ei2, st) in states.iter().enumerate() {
                    st.adapter.install(
                        &mut self.experts[ei2],
                        &st.params,
                        alpha,
                        rank,
                        LOKR_DTYPE,
                    )?;
                }
                let total = sample_ctxs.len() as u32;
                for (i, (prompt, ctxs)) in sample_ctxs.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    let sample_seed = cfg
                        .seed
                        .wrapping_add(step as u64)
                        .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                        .wrapping_add(i as u64);
                    match render_wan_sample(
                        &self.experts,
                        &self.vae,
                        &self.cfg,
                        ctxs,
                        &sample_latent_shape,
                        sample_seed,
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
                            "[sc-5637] {id} preview sample failed at step {step} (prompt {}): {e} \
                             — skipping this preview, training continues",
                            i + 1
                        ),
                    }
                }
            }
        }

        // Cancelled before completing a single step (`steps == 0` is rejected upstream by
        // `validate`): the adapter factors are still freshly initialized with `B = 0`, a no-op
        // adapter. Surface the typed `Error::Canceled` (sc-4895, bridged 1:1 to
        // `gen_core::Error::Canceled`) rather than writing valid-looking `.safetensors` and returning
        // `Ok` — downstream tooling would otherwise ship an identity adapter as a trained artifact
        // (F-040).
        if steps_run == 0 {
            return Err(mlx_gen::Error::Canceled);
        }

        // --- save one adapter per expert (the MoE high/low pair, or a single dense file) ---
        on_progress(TrainingProgress::Saving);
        let paths = save_experts(
            &states,
            &req.output_dir,
            &stem,
            &ext,
            None,
            alpha,
            rank,
            cfg,
        )?;
        // The high-noise file is the primary returned path (the worker discovers the low pair).
        let adapter_path = paths
            .iter()
            .zip(&states)
            .find(|(_, st)| st.suffix == "high_noise" || st.suffix.is_empty())
            .map(|(p, _)| p.clone())
            .unwrap_or_else(|| req.output_dir.join(&req.file_name));
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Per-expert adapter filename: `{stem}.{suffix}{ext}` (final) or `{stem}-step{step:06}.{suffix}{ext}`
/// (checkpoint, matching the [`checkpoint_filename`](mlx_gen::train::checkpoint::checkpoint_filename)
/// convention), with the `.{suffix}` dropped for the single dense file.
fn expert_filename(stem: &str, ext: &str, suffix: &str, step: Option<u32>) -> String {
    let base = match step {
        Some(s) => format!("{stem}-step{s:06}"),
        None => stem.to_string(),
    };
    if suffix.is_empty() {
        format!("{base}{ext}")
    } else {
        format!("{base}.{suffix}{ext}")
    }
}

/// Write each expert's adapter (the MoE high/low pair, or a single dense file). `step = Some` writes
/// intermediate checkpoints; `None` the final adapter. Returns the written paths (expert order).
#[allow(clippy::too_many_arguments)]
fn save_experts(
    states: &[ExpertState],
    dir: &Path,
    stem: &str,
    ext: &str,
    step: Option<u32>,
    alpha: f32,
    rank: f32,
    cfg: &TrainingConfig,
) -> Result<Vec<std::path::PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let mut paths = Vec::with_capacity(states.len());
    for st in states {
        let path = dir.join(expert_filename(stem, ext, st.suffix, step));
        st.adapter
            .save(&st.params, alpha, rank, cfg.decompose_factor, "", &path)?;
        paths.push(path);
    }
    Ok(paths)
}

/// Resolve the config's target-module suffixes (default native `q/k/v/o` — the reference
/// `to_q/k/v/to_out.0` attention surface) against the expert's per-block adaptable Linears.
fn resolve_target_paths(expert: &WanTransformer, suffixes: &[String]) -> Vec<String> {
    AdaptableHost::adaptable_paths(expert)
        .into_iter()
        .filter(|path| {
            suffixes
                .iter()
                .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
}

/// Build one expert's trainable adapter (LoRA or LoKr per `cfg.network_type`) over `target_paths`.
fn build_adapter(
    expert: &mut WanTransformer,
    target_paths: &[String],
    cfg: &TrainingConfig,
    seed: u64,
) -> Result<(TrainAdapter, LoraParams)> {
    match cfg.network_type {
        NetworkType::Lora => {
            let (targets, params) =
                build_lora_targets(expert, target_paths, cfg.rank as i32, seed)?;
            Ok((TrainAdapter::Lora { targets }, params))
        }
        NetworkType::Lokr => {
            let (targets, params) = build_lokr_targets(
                expert,
                target_paths,
                cfg.rank as i32,
                cfg.decompose_factor,
                seed,
            )?;
            Ok((TrainAdapter::Lokr { targets }, params))
        }
    }
}

/// One forward+backward over an expert's trainable factors: build the flow-match input `x_t`, inject
/// the factors, run the DiT, regress the raw velocity toward `noise - clean`, return `(loss, grads)`.
/// `y_channels > 0` (I2V-A14B, = 20) appends a **zero** channel-concat conditioning block to the
/// noisy latent so the in_dim-36 forward runs — the reference's no-conditioning T2V objective; the
/// padded channels are a constant input (not differentiated), and the trained surface is attention.
///
/// `checkpoint_block`, when `Some`, lists each block's trainable targets and switches the forward to
/// the per-block gradient-checkpointed path (sc-4942) — each block recomputes its activations (and its
/// cross-attention K/V) in the backward instead of retaining them. `None` runs the dense
/// (attention-segment-checkpointed) forward via the installed adapter.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    expert: &mut WanTransformer,
    adapter: &TrainAdapter,
    params: &LoraParams,
    alpha: f32,
    rank: f32,
    clean: &Array,
    context: &Array,
    t: f32,
    noise: &Array,
    mae: bool,
    y_channels: i32,
    checkpoint_block: Option<&[Vec<BlockLoraRef>]>,
) -> Result<(f32, LoraParams)> {
    // x_t = (1-t)·clean + t·noise; target = noise - clean (raw velocity); transformer timestep = t·1000.
    let one_minus = Array::from_slice(&[1.0 - t], &[1]);
    let tt = Array::from_slice(&[t], &[1]);
    let x_t = add(&multiply(clean, &one_minus)?, &multiply(noise, &tt)?)?;
    let target = subtract(noise, clean)?;
    // I2V channel-concat: append a zero `y` `[y_channels, F, h, w]` → in_dim 36 (mirrors `predict`'s
    // `[latents, y]` order). A constant input, built once outside the autograd closure.
    let x_in = if y_channels > 0 {
        let s = x_t.shape(); // [z, F, h, w]
        let zero_y = Array::zeros::<f32>(&[y_channels, s[1], s[2], s[3]])?;
        concatenate_axis(&[&x_t, &zero_y], 0)?
    } else {
        x_t
    };
    let timestep = t * 1000.0;
    let ctx = context.clone();
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        let v = match checkpoint_block {
            Some(bt) => expert
                .forward_train_checkpointed(&x_in, timestep, &ctx, &p, bt, alpha)
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => {
                adapter.install(expert, &p, alpha, rank, LOKR_DTYPE)?;
                expert
                    .forward(&x_in, timestep, &ctx)
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

/// Group resolved target paths by their owning block (sc-4942) — `block_targets[i]` lists block `i`'s
/// trainable LoRA targets as the block-local path (`blocks.{i}.` stripped) plus the factor-map keys
/// (`{path}.lora_a`/`.lora_b`, the core `build_lora_targets` convention), for the gradient-checkpoint
/// closure. Every target lives in a `blocks.{i}.…` leaf, so the grouping is exhaustive.
fn group_block_targets(target_paths: &[String], n_blocks: usize) -> Vec<Vec<BlockLoraRef>> {
    let mut out: Vec<Vec<BlockLoraRef>> = (0..n_blocks).map(|_| Vec::new()).collect();
    for path in target_paths {
        let segs: Vec<&str> = path.split('.').collect();
        if segs.len() < 3 || segs[0] != "blocks" {
            continue;
        }
        let Ok(i) = segs[1].parse::<usize>() else {
            continue;
        };
        if i >= n_blocks {
            continue;
        }
        out[i].push(BlockLoraRef {
            local: segs[2..].iter().map(|s| s.to_string()).collect(),
            a_key: format!("{path}.lora_a"),
            b_key: format!("{path}.lora_b"),
        });
    }
    out
}

/// Sample a flow-match timestep within `band = (lo, hi)`: a faithful port of the reference
/// `sample_training_timestep` (`sigmoid(randn)` default, `uniform` for linear, `(uniform+sigmoid)/2`
/// for weighted; bias `high` → `√t`, `low` → `t²`), then scaled into the expert's noise band
/// `lo + t_unit·(hi-lo)`. Deterministic in `seed`.
fn sample_band_timestep(
    timestep_type: &str,
    timestep_bias: &str,
    band: (f32, f32),
    seed: u64,
) -> Result<f32> {
    let k1 = random::key(seed)?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let ttype = timestep_type.trim().to_ascii_lowercase().replace('-', "_");
    let t_unit = match ttype.as_str() {
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
    let t_unit = match bias.as_str() {
        "high" | "high_noise" | "favor_high_noise" => t_unit.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t_unit * t_unit,
        _ => t_unit,
    };
    let t_unit = t_unit.clamp(1e-3, 1.0 - 1e-3);
    Ok(band.0 + t_unit * (band.1 - band.0))
}

/// Projected DENSE (non-block-checkpointed) first-step peak memory, in GB, for a Wan variant — an
/// empirical fit used by the pre-flight OOM guard (sc-4942). With attention-segment checkpointing
/// always on, the working set is essentially linear in the token count `tokens` (the seq² term is
/// demoted to a per-layer transient). The peak is the **resident weights** (all `n_experts` stay
/// resident across the MoE alternation) plus the active expert's working set:
///   `peak = n_experts · resident_per_expert + working_set(tokens)`.
///
/// CALIBRATED from `first_step_repro::first_step_peak_sweep_t2v` on the converted **T2V-A14B bf16**
/// (dim 5120, 40 layers): one expert resident ≈ 27 GB; dense working set ≈ 10.3 + 0.0177·L GB
/// (L=256/576/1024 → 14.8/20.6/28.4 GB). The resident scales ~ params (∝ dim²·layers) and the working
/// set ∝ dim·layers, so the fit extrapolates to the I2V-A14B (same size) and the smaller dense
/// TI2V-5B (the A14B is the exact anchor; the 5B/quantized scalings are estimates → the guard is
/// conservative there, which is the safe direction). Refit the anchors if the sweep changes.
///
/// sc-4972 — CONFIRMED on the other two variants' real weights (`projection_matches_measured_curve` +
/// `first_step_peak_sweep_{i2v,ti2v_5b}`): the I2V-A14B single-expert curve is bit-for-bit the T2V
/// anchor (same dim/layers), and the TI2V-5B extrapolation lands within ~2 GB of the measured
/// 13.66/14.56/15.88 GB (edge 256/384/512); the guard stays conservative for the 5B (see below).
fn projected_dense_peak_gb(tokens: f64, dim: usize, n_layers: usize, n_experts: usize) -> f64 {
    let param_scale = (dim * dim * n_layers) as f64 / (5120.0 * 5120.0 * 40.0);
    let resident = n_experts as f64 * 27.0 * param_scale;
    let act_scale = (dim * n_layers) as f64 / (5120.0 * 40.0);
    let working = (10.3 + 0.0177 * tokens) * act_scale;
    resident + working
}

/// The DiT token count for a still-image (single-frame) training latent at pixel `edge`: the VAE
/// downscales ×8, then the patchifier divides by `(ph, pw)` (frames `F = 1` / `pt`). `L = (edge/8/ph)·
/// (edge/8/pw)`.
///
/// NOTE (sc-4972): the ×8 is the z16 14B VAE (T2V/I2V). The z48 TI2V-5B VAE downscales ×16, so this
/// OVERESTIMATES the 5B's true L by 4× — which only feeds the OOM `preflight_memory_guard`, where an
/// overcount keeps the projection conservatively above the true peak (the safe direction; the measured
/// 5B peaks confirm this — see `projected_dense_peak_gb`). Make it `cfg`-stride-aware before reusing
/// it anywhere correctness (not just a safety bound) depends on the exact token count.
fn training_tokens(cfg: &WanModelConfig, edge: u32) -> f64 {
    let (_pt, ph, pw) = cfg.patch_size;
    let le = (edge / 8).max(1) as usize;
    ((le / ph.max(1)) * (le / pw.max(1))) as f64
}

/// Refuse a run whose dense first step would exceed this machine's memory budget (and thus get
/// SIGKILLed), returning a catchable, actionable error instead (sc-4942 — the sc-4874 mechanism, ported
/// to Wan). The budget is MLX's reported memory limit × 0.85 for worker/host headroom. Only consulted
/// when gradient checkpointing is OFF. `n_experts` resident is the MoE floor (both stay loaded across
/// the alternation), which is itself most of the cost — so on a tier that can't hold the experts, the
/// guard correctly recommends the dense TI2V-5B or a lower resolution.
fn preflight_memory_guard(
    cfg: &WanModelConfig,
    edge: u32,
    n_experts: usize,
    id: &str,
) -> Result<()> {
    let tokens = training_tokens(cfg, edge);
    let projected = projected_dense_peak_gb(tokens, cfg.dim, cfg.num_layers, n_experts);
    let budget_gb = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);
    let safe = budget_gb * 0.85;
    if projected > safe {
        return Err(format!(
            "{id} trainer: a dense first training step at resolution {edge} needs ~{projected:.0} GB \
             ({n_experts} resident expert(s) + the forward working set, materialized in one allocation), \
             exceeding this machine's ~{safe:.0} GB safe budget ({budget_gb:.0} GB MLX limit × 0.85). \
             Without mitigation the OS would hard-kill the worker (SIGKILL) at the first step with no \
             recoverable error (sc-4874/sc-4942). Enable Gradient Checkpointing (recomputes block \
             activations in the backward) or reduce the training resolution."
        )
        .into());
    }
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
// Ports the z-image/LTX first-step harness to Wan: drives the exact inner training step
// (`compute_loss_grads` + the step-1 grad `eval`) on ONE expert, sweeping resolution with MLX
// peak-memory probes, and asserts the checkpointing levers' invariants on REAL weights:
//   * attention-segment checkpointing is bit-identical to the retained backward,
//   * block (gradient) checkpointing matches the dense path within fp tolerance,
//   * block checkpointing materially shrinks the first-step working set.
// (Wan is bf16-native, so there is no bf16-vs-f32 grad gate — there is no f32 training path.)
//
//   cargo test -p mlx-gen-wan --release --lib first_step -- --ignored --nocapture
// ===========================================================================================
#[cfg(test)]
mod first_step_repro {
    use super::*;
    use mlx_gen::train::lora::build_lora_targets;
    use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};
    use std::path::PathBuf;

    const RANK: i32 = 8;
    const ALPHA: f32 = 8.0;

    /// Which converted bf16 Wan variant the harness drives. T2V-A14B is the reference (widest DiT,
    /// dual-expert); sc-4972 adds I2V-A14B (channel-concat in_dim 36) and the dense TI2V-5B (z48
    /// vae22, 30 layers) so grad parity + the first-step peak are confirmed on each variant's own
    /// real weights, not just extrapolated from T2V.
    #[derive(Clone, Copy)]
    enum Variant {
        T2v,
        I2v,
        Ti2v5b,
    }

    impl Variant {
        fn descriptor(self) -> TrainerDescriptor {
            match self {
                Variant::T2v => descriptor_t2v_14b(),
                Variant::I2v => descriptor_i2v_14b(),
                Variant::Ti2v5b => descriptor_ti2v_5b(),
            }
        }

        /// The converted snapshot dir; `$<ENV>` overrides, else the mlx-gen model cache default.
        fn snapshot(self) -> PathBuf {
            let (env, default) = match self {
                Variant::T2v => (
                    "WAN_A14B_MODEL_DIR",
                    ".cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16",
                ),
                Variant::I2v => (
                    "WAN_I2V_A14B_MODEL_DIR",
                    ".cache/mlx-gen-models/wan2_2_i2v_a14b_mlx_bf16",
                ),
                Variant::Ti2v5b => (
                    "WAN_TI2V_5B_MODEL_DIR",
                    ".cache/mlx-gen-models/wan_2_2_ti2v_5b_mlx_bf16",
                ),
            };
            if let Ok(p) = std::env::var(env) {
                return PathBuf::from(p);
            }
            PathBuf::from(std::env::var("HOME").unwrap()).join(default)
        }
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

    /// Build a real Wan trainer, keep ONLY the low-noise expert (frees ~28 GB of the unused high
    /// expert), build its LoRA targets, encode one caption to a cached bf16 context, then free the UMT5
    /// TE + tokenizer — so the measured peaks reflect the post-free single-expert training working set.
    #[allow(clippy::type_complexity)]
    fn build(
        variant: Variant,
    ) -> (
        WanMoeTrainer,
        TrainAdapter,
        LoraParams,
        Array,
        Vec<Vec<BlockLoraRef>>,
    ) {
        let spec = LoadSpec::new(WeightsSource::Dir(variant.snapshot()));
        let mut trainer = build_trainer_concrete(&spec, variant.descriptor())
            .expect("converted Wan2.2 bf16 snapshot ($WAN_*_MODEL_DIR or the cache)");
        // Encode one context with expert 0, then keep only expert 0 + free the TE.
        let ctx = {
            let te = trainer.text_encoder.as_ref().unwrap();
            let tok = trainer.tokenizer.as_ref().unwrap();
            let t5 = te.encode(tok, "a solid colour swatch").unwrap();
            trainer.experts[0].embed_text(&t5).unwrap()
        };
        eval([&ctx]).unwrap();
        trainer.experts.truncate(1);
        trainer.text_encoder = None;
        trainer.tokenizer = None;
        clear_cache();

        let suffixes: Vec<String> = DEFAULT_TARGET_SUFFIXES
            .iter()
            .map(|s| s.to_string())
            .collect();
        let target_paths = resolve_target_paths(&trainer.experts[0], &suffixes);
        let (targets, params) =
            build_lora_targets(&mut trainer.experts[0], &target_paths, RANK, 7).unwrap();
        let block_targets = group_block_targets(&target_paths, trainer.experts[0].num_blocks());
        eprintln!(
            "[sc-4942] loaded Wan trainer ({}, 1 expert, TE freed); in_dim {} vae_z {} → y_channels {}; {} LoRA targets; ctx {:?}",
            trainer.descriptor.id,
            trainer.cfg.in_dim,
            trainer.cfg.vae_z_dim,
            (trainer.cfg.in_dim as i32 - trainer.cfg.vae_z_dim as i32).max(0),
            target_paths.len(),
            ctx.shape()
        );
        (
            trainer,
            TrainAdapter::Lora { targets },
            params,
            ctx,
            block_targets,
        )
    }

    /// Run a single first training step at `edge` and report peak GPU memory across forward+backward,
    /// forcing the backward. `checkpoint` selects the block-checkpointed forward; the caller sets the
    /// SDPA-checkpoint flag on the expert.
    #[allow(clippy::too_many_arguments)]
    fn one_step(
        trainer: &mut WanMoeTrainer,
        adapter: &TrainAdapter,
        params: &LoraParams,
        ctx: &Array,
        block_targets: &[Vec<BlockLoraRef>],
        edge: u32,
        checkpoint: bool,
        tag: &str,
    ) -> Result<(f32, f64)> {
        let clean = trainer
            .vae
            .encode_clean(&center_crop_square(&swatch(edge)), edge)?;
        eval([&clean])?;
        let noise = random::normal::<f32>(clean.shape(), None, None, Some(&random::key(1)?))?;
        eval([&noise])?;
        let ck = if checkpoint {
            Some(block_targets)
        } else {
            None
        };

        // I2V-A14B concatenates a zero `y` (in_dim 36 = latent 16 + y 20); T2V/5B have in_dim == z.
        let y_channels = (trainer.cfg.in_dim as i32 - trainer.cfg.vae_z_dim as i32).max(0);
        clear_cache();
        reset_peak_memory();
        let before = get_active_memory();
        let t0 = std::time::Instant::now();
        let (loss, grads) = compute_loss_grads(
            &mut trainer.experts[0],
            adapter,
            params,
            ALPHA,
            RANK as f32,
            &clean,
            ctx,
            0.5,
            &noise,
            false,
            y_channels,
            ck,
        )?;
        eval(grads.values())?;
        let secs = t0.elapsed().as_secs_f64();
        let peak = get_peak_memory();
        eprintln!(
            "  [edge {edge:>4} {tag}] latent {:?}  loss {loss:.5}  active-before {:.2} GB  peak {:.2} GB  step {secs:.2}s",
            clean.shape(),
            gb(before),
            gb(peak)
        );
        Ok((loss, gb(peak)))
    }

    /// Grads for a given (checkpoint, sdpa) configuration at `edge`, backward forced.
    fn grads_of(
        trainer: &mut WanMoeTrainer,
        adapter: &TrainAdapter,
        params: &LoraParams,
        ctx: &Array,
        block_targets: &[Vec<BlockLoraRef>],
        edge: u32,
        checkpoint: bool,
    ) -> LoraParams {
        let clean = trainer
            .vae
            .encode_clean(&center_crop_square(&swatch(edge)), edge)
            .unwrap();
        let noise =
            random::normal::<f32>(clean.shape(), None, None, Some(&random::key(1).unwrap()))
                .unwrap();
        eval([&clean, &noise]).unwrap();
        let ck = if checkpoint {
            Some(block_targets)
        } else {
            None
        };
        let y_channels = (trainer.cfg.in_dim as i32 - trainer.cfg.vae_z_dim as i32).max(0);
        let (_l, g) = compute_loss_grads(
            &mut trainer.experts[0],
            adapter,
            params,
            ALPHA,
            RANK as f32,
            &clean,
            ctx,
            0.5,
            &noise,
            false,
            y_channels,
            ck,
        )
        .unwrap();
        eval(g.values()).unwrap();
        g
    }

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

    /// sc-4942/sc-4972 — attention-segment checkpointing must not change the math on `variant`'s real
    /// weights: grads with the SDPA checkpoint on must match the retained backward (flag off).
    /// Bit-identical recompute.
    fn check_attn_ckpt(variant: Variant) {
        let (mut t, adapter, params, ctx, bt) = build(variant);
        let edge = 256u32;
        t.experts[0].set_sdpa_checkpoint(false);
        let g_retained = grads_of(&mut t, &adapter, &params, &ctx, &bt, edge, false);
        t.experts[0].set_sdpa_checkpoint(true);
        let g_ckpt = grads_of(&mut t, &adapter, &params, &ctx, &bt, edge, false);
        let max_rel = max_rel_diff(&g_retained, &g_ckpt);
        eprintln!("[sc-4972] attn-ckpt-vs-retained grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-5,
            "attention-segment checkpointing must not change grads: max rel {max_rel:.2e}"
        );
    }

    #[test]
    #[ignore = "needs the converted Wan2.2-T2V-A14B bf16 checkpoint; run as its own process"]
    fn attn_ckpt_grads_match_retained_t2v() {
        check_attn_ckpt(Variant::T2v);
    }
    #[test]
    #[ignore = "needs the converted Wan2.2-I2V-A14B bf16 checkpoint; run as its own process"]
    fn attn_ckpt_grads_match_retained_i2v() {
        check_attn_ckpt(Variant::I2v);
    }
    #[test]
    #[ignore = "needs the converted Wan2.2-TI2V-5B bf16 checkpoint; run as its own process"]
    fn attn_ckpt_grads_match_retained_ti2v_5b() {
        check_attn_ckpt(Variant::Ti2v5b);
    }

    /// sc-4942/sc-4972 — block (gradient) checkpointing must match the dense path within fp tolerance
    /// on `variant`'s real weights (same install + block forward, recompute-only). Also guards the
    /// multi-output-VJP duplicate-cotangent trap (each checkpoint returns one distinct array).
    fn check_block_ckpt(variant: Variant) {
        let (mut t, adapter, params, ctx, bt) = build(variant);
        let edge = 256u32;
        t.experts[0].set_sdpa_checkpoint(true);
        let g_dense = grads_of(&mut t, &adapter, &params, &ctx, &bt, edge, false);
        t.experts[0].set_sdpa_checkpoint(false);
        let g_ckpt = grads_of(&mut t, &adapter, &params, &ctx, &bt, edge, true);
        let max_rel = max_rel_diff(&g_dense, &g_ckpt);
        eprintln!("[sc-4972] block-ckpt-vs-dense grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 5e-3,
            "block checkpointing must match dense within tolerance: max rel {max_rel:.2e}"
        );
    }

    #[test]
    #[ignore = "needs the converted Wan2.2-T2V-A14B bf16 checkpoint; run as its own process"]
    fn block_ckpt_grads_match_dense_t2v() {
        check_block_ckpt(Variant::T2v);
    }
    #[test]
    #[ignore = "needs the converted Wan2.2-I2V-A14B bf16 checkpoint; run as its own process"]
    fn block_ckpt_grads_match_dense_i2v() {
        check_block_ckpt(Variant::I2v);
    }
    #[test]
    #[ignore = "needs the converted Wan2.2-TI2V-5B bf16 checkpoint; run as its own process"]
    fn block_ckpt_grads_match_dense_ti2v_5b() {
        check_block_ckpt(Variant::Ti2v5b);
    }

    /// sc-4942/sc-4972 — first-step peak sweep (attention-segment checkpointing on), plus block-ckpt
    /// points. Calibrates the `projected_dense_peak_gb` guard fit per variant — refit the anchors if
    /// these print materially different. The harness keeps ONE expert resident (build() truncates), so
    /// the printed peak is the single-expert working set; the projection scales `n_experts` resident.
    fn run_peak_sweep(variant: Variant) {
        let (mut t, adapter, params, ctx, bt) = build(variant);
        t.experts[0].set_sdpa_checkpoint(true);
        eprintln!("[sc-4972] attn-ckpt dense sweep (bf16-native):");
        for edge in [256u32, 384, 512] {
            let _ = one_step(
                &mut t,
                &adapter,
                &params,
                &ctx,
                &bt,
                edge,
                false,
                "attn-ckpt",
            )
            .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
        }
        eprintln!("[sc-4972] block-ckpt:");
        t.experts[0].set_sdpa_checkpoint(false);
        for edge in [384u32, 512] {
            let _ = one_step(&mut t, &adapter, &params, &ctx, &bt, edge, true, "blk-ckpt")
                .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
        }
    }

    #[test]
    #[ignore = "needs the converted Wan2.2-T2V-A14B bf16 checkpoint; run as its own process (may SIGKILL)"]
    fn first_step_peak_sweep_t2v() {
        run_peak_sweep(Variant::T2v);
    }
    #[test]
    #[ignore = "needs the converted Wan2.2-I2V-A14B bf16 checkpoint; run as its own process (may SIGKILL)"]
    fn first_step_peak_sweep_i2v() {
        run_peak_sweep(Variant::I2v);
    }
    #[test]
    #[ignore = "needs the converted Wan2.2-TI2V-5B bf16 checkpoint; run as its own process"]
    fn first_step_peak_sweep_ti2v_5b() {
        run_peak_sweep(Variant::Ti2v5b);
    }

    /// sc-4942 — the empirical fit must reproduce the measured A14B peak (the basis of the pre-flight
    /// OOM guard) and stay monotonic / ordered across variants. Measured anchor (T2V-A14B, one expert,
    /// edge 512 → L=1024): 55.4 GB.
    ///
    /// sc-4972 — confirmed on the OTHER two variants' own real weights (first_step_peak_sweep_{i2v,
    /// ti2v_5b}, one resident expert):
    ///   * **I2V-A14B** (dim 5120, 40 layers) edge 256/384/512 → L 256/576/1024 → 41.80/47.58/55.41 GB,
    ///     i.e. *bit-for-bit the T2V curve* (same DiT; the in_dim-36 zero-`y` concat is negligible) —
    ///     the A14B anchor IS the I2V anchor.
    ///   * **TI2V-5B** (dim 3072, 30 layers, z48 VAE ×16) edge 256/384/512 → L 64/144/256 →
    ///     13.66/14.56/15.88 GB. The dim²·layers / dim·layers extrapolation lands within ~2 GB of each
    ///     (slightly under at the true L). The guard reaches the fit through `training_tokens`, which
    ///     treats every VAE as ×8 (true for the z16 14B, ×16-coarse for the z48 5B) → it feeds L 4×
    ///     too large, so the guard projection sits *above* the true peak (the safe direction).
    #[test]
    fn projection_matches_measured_curve() {
        // A14B one-expert at L=1024 = the measured 55.4 GB anchor.
        let p = projected_dense_peak_gb(1024.0, 5120, 40, 1);
        assert!(
            (p - 55.4).abs() < 1.0,
            "A14B L=1024 projection = {p:.1} GB, expected ≈55.4"
        );
        // A14B sweep points (working set 14.8/20.6 GB at L=256/576 + 27 resident).
        assert!((projected_dense_peak_gb(256.0, 5120, 40, 1) - 41.8).abs() < 1.5);
        assert!((projected_dense_peak_gb(576.0, 5120, 40, 1) - 47.6).abs() < 1.5);
        // sc-4972 — I2V-A14B: same dim/layers ⇒ the fit reproduces the measured 41.80/47.58/55.41 GB.
        assert!((projected_dense_peak_gb(256.0, 5120, 40, 1) - 41.80).abs() < 1.0);
        assert!((projected_dense_peak_gb(576.0, 5120, 40, 1) - 47.58).abs() < 1.0);
        assert!((projected_dense_peak_gb(1024.0, 5120, 40, 1) - 55.41).abs() < 1.0);
        // sc-4972 — TI2V-5B: the fit at the TRUE 5B token counts (L=64/144/256) is within ~2 GB of the
        // measured 13.66/14.56/15.88 GB single-expert peaks.
        assert!((projected_dense_peak_gb(64.0, 3072, 30, 1) - 13.66).abs() < 2.0);
        assert!((projected_dense_peak_gb(144.0, 3072, 30, 1) - 14.56).abs() < 2.0);
        assert!((projected_dense_peak_gb(256.0, 3072, 30, 1) - 15.88).abs() < 2.0);
        // sc-4972 — and the guard's `training_tokens` path (5B fed L=1024 at edge 512, the ×8 overcount)
        // stays conservatively ABOVE the true 15.88 GB peak — the safe direction for an OOM guard.
        assert!(
            projected_dense_peak_gb(1024.0, 3072, 30, 1) > 15.88,
            "5B guard projection must bound the measured peak conservatively"
        );
        // Monotonic in tokens; dual-expert adds a second resident; the dense 5B is much smaller.
        assert!(
            projected_dense_peak_gb(256.0, 5120, 40, 1)
                < projected_dense_peak_gb(1024.0, 5120, 40, 1)
        );
        assert!(
            projected_dense_peak_gb(1024.0, 5120, 40, 2)
                > projected_dense_peak_gb(1024.0, 5120, 40, 1)
        );
        assert!(
            projected_dense_peak_gb(1024.0, 3072, 30, 1)
                < projected_dense_peak_gb(1024.0, 5120, 40, 1)
        );
    }

    /// sc-4942 — block checkpointing must drop the first-step peak below the dense path.
    #[test]
    #[ignore = "needs the converted Wan2.2-T2V-A14B bf16 checkpoint; run as its own process"]
    fn block_ckpt_reduces_peak_vs_dense() {
        let (mut t, adapter, params, ctx, bt) = build(Variant::T2v);
        t.experts[0].set_sdpa_checkpoint(true);
        let (_, dense_peak) =
            one_step(&mut t, &adapter, &params, &ctx, &bt, 512, false, "dense").expect("dense");
        t.experts[0].set_sdpa_checkpoint(false);
        let (_, ckpt_peak) =
            one_step(&mut t, &adapter, &params, &ctx, &bt, 512, true, "blk-ckpt").expect("ckpt");
        eprintln!(
            "[sc-4942] edge 512  dense {dense_peak:.2} GB  ckpt {ckpt_peak:.2} GB  ({:.0}% reduction)",
            100.0 * (1.0 - ckpt_peak / dense_peak)
        );
        assert!(
            ckpt_peak < dense_peak,
            "block checkpointing must reduce the first-step peak: dense {dense_peak:.2} vs ckpt {ckpt_peak:.2}"
        );
    }
}
