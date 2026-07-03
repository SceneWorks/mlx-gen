//! sc-7781 — the parameterized **SDXL-family** LoRA/LoKr training backbone, shared by
//! [`SdxlTrainer`](super::SdxlTrainer) (dual-CLIP) and the Kolors trainer (ChatGLM3). Kolors *is* an
//! SDXL-base U-Net under a different text encoder, so its trainer was an ~85–90% line-for-line clone
//! of the SDXL one — same private-fn roster, same `train_impl` control flow, same tests — differing
//! only at a handful of injection points. This module collapses that duplication: the whole
//! prepare→cache→train→save lifecycle lives once in [`train_family`], and each crate supplies its
//! deltas through the [`SdxlFamilyHooks`] trait.
//!
//! Everything generic here was **moved verbatim** from the two trainers (no behavioral change — the
//! committed grad-parity gates in each crate are the safety case), with the per-family pieces routed
//! through hooks:
//!   1. **Prompt encoding** — SDXL dual-CLIP vs Kolors ChatGLM3 ([`SdxlFamilyHooks::encode_prompt`]);
//!      the preview-sample CFG batch via [`SdxlFamilyHooks::encode_sample_cfg`].
//!   2. **Micro-conditioning `time_ids`** — SDXL's hardcoded `[512,512,0,0,512,512]` vs Kolors'
//!      real-resolution `(H,W,0,0,H,W)` ([`SdxlFamilyHooks::time_ids`]).
//!   3. **Noise / objective** — SDXL's renormalized k-diffusion sigma noising (float table-index `t`)
//!      vs Kolors' direct DDPM `√ᾱ_t·x0 + √(1−ᾱ_t)·noise` (integer `t`). The float-vs-integer split is
//!      surfaced explicitly as [`TrainTimestep`]; the noise call hides behind
//!      [`SdxlFamilyHooks::add_noise`] and the draw behind [`SdxlFamilyHooks::sample_timestep`].
//!   4. **Memory-fit coefficients** — each family's fitted peak-GB curve
//!      ([`SdxlFamilyHooks::peak_gb`]) + the error-string family label ([`SdxlFamilyHooks::label`]).
//!   5. **Preview render** — SDXL's Euler-Ancestral vs Kolors' leading-Euler render
//!      ([`SdxlFamilyHooks::render_sample`]); freeing the text encoder(s) after caching via
//!      [`SdxlFamilyHooks::free_text_encoders`].
//!
//! The U-Net and VAE are the **same** concrete SDXL types in both crates (Kolors reuses
//! `mlx_gen_sdxl::{UNet2DConditionModel, Autoencoder}`), so they are threaded as plain parameters
//! rather than hidden behind the trait.

use std::path::Path;

use mlx_gen::train::checkpoint::{self, checkpoint_filename};
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, average_grads, build_lokr_targets, build_lora_targets, LoraParams,
    TrainAdapter,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::{
    Image, NetworkType, Result, TrainOptimizer, TrainingConfig, TrainingOutput, TrainingProgress,
    TrainingRequest,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::subtract;
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::pipeline::encode_init_latents;
use crate::unet::UNet2DConditionModel;
use crate::vae::Autoencoder;

/// The SDXL-family reconstructs its LoKr delta at **f32** (the f32-everywhere merge path); training
/// must match so the adapter round-trips through the inference loader. Shared by both families.
const LOKR_DTYPE: Dtype = Dtype::Float32;

/// Max preview-sample prompts rendered per [`TrainingConfig::sample_every`] cadence (sc-5637).
const SAMPLE_PROMPT_CAP: usize = 4;

/// PEFT save-key prefix for the LoRA adapter — what `peft.save_pretrained()` / the SceneWorks
/// backends emit, and what the SDXL loader's PEFT key classifier (`adapters::classify_key`) expects.
/// The Kolors U-Net IS the SDXL `UNet2DConditionModel`, so it shares this prefix.
const PEFT_PREFIX: &str = "base_model.model.unet.";

/// The default SDXL-family attention LoRA targets — the suffixes `to_q`/`to_k`/`to_v`/`to_out.0` the
/// torch trainers use, suffix-matched across the UNet attention modules exactly as PEFT's
/// `LoraConfig(target_modules=…)` does.
const DEFAULT_TARGET_SUFFIXES: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// The sampled training timestep, surfacing the one fiddly cross-family split: SDXL noises in the
/// vendored **sigma** space at a *float* table-index, while Kolors noises with direct DDPM
/// `alphas_cumprod` at an *integer* index. [`SdxlFamilyHooks::add_noise`] consumes this verbatim; the
/// U-Net's sinusoidal time embedding consumes [`Self::unet_time`] (Kolors' integer cast to f32).
#[derive(Clone, Copy, Debug)]
pub enum TrainTimestep {
    /// SDXL: a float sigma-table index in `[1, max_time]`; fed verbatim to both the noiser and the
    /// U-Net's time embedding.
    Sigma(f32),
    /// Kolors: an integer `alphas_cumprod` index in `[0, num_train_timesteps)`; the U-Net consumes it
    /// as `t as f32`.
    Index(usize),
}

impl TrainTimestep {
    /// The f32 timestep the U-Net's sinusoidal embedding consumes.
    pub fn unet_time(self) -> f32 {
        match self {
            TrainTimestep::Sigma(s) => s,
            TrainTimestep::Index(i) => i as f32,
        }
    }
}

/// The per-family injection points the generic [`train_family`] backbone routes through. Everything
/// *not* on this trait is identical across the SDXL family and lives in [`train_family`].
pub trait SdxlFamilyHooks {
    /// Family id prefix for error strings + log lines, e.g. `"sdxl"` / `"kolors"`.
    fn label(&self) -> &'static str;

    /// Encode one caption → `(conditioning, pooled)` for a single (B=1) cached training item.
    fn encode_prompt(&self, caption: &str) -> Result<(Array, Array)>;

    /// Encode one preview-sample prompt into the **CFG batch** (`[2, …]` = positive then
    /// empty-negative) the preview render's classifier-free guidance needs. Returns the f32 (or
    /// encoder-native) conditioning/pooled; [`train_family`] casts to the compute dtype.
    fn encode_sample_cfg(&self, prompt: &str) -> Result<(Array, Array)>;

    /// Free the text encoder(s) after the caching phase (SDXL frees both CLIPs; Kolors frees the
    /// ChatGLM3) — the 32 GB-Mac headroom lever. After this, [`Self::encode_prompt`] /
    /// [`Self::encode_sample_cfg`] must not be called again.
    fn free_text_encoders(&mut self);

    /// Micro-conditioning `time_ids` for the given latent `edge`, `batch` rows.
    fn time_ids(&self, batch: i32, edge: u32) -> Array;

    /// Sample a training timestep for this step, deterministic in `seed`.
    fn sample_timestep(&self, seed: u64) -> Result<TrainTimestep>;

    /// Add noise at the sampled timestep — hides the VE-sigma vs DDPM-`alpha_bar` convention (and the
    /// f32/usize split carried by [`TrainTimestep`]).
    fn add_noise(&self, x0: &Array, noise: &Array, t: TrainTimestep) -> Result<Array>;

    /// Fitted dense first-step peak-GB curve vs the latent pixel count `p = (edge/8)²`.
    fn peak_gb(&self, p: f64, bf16: bool) -> f64;

    /// Render one preview sample from the **in-progress adapter** already installed on `unet`:
    /// seeded prior → CFG denoise → VAE decode. `conditioning`/`pooled` are the pre-encoded CFG batch;
    /// `dtype` is the trainer compute dtype (used by Kolors' sampler; SDXL ignores it).
    #[allow(clippy::too_many_arguments)]
    fn render_sample(
        &self,
        unet: &UNet2DConditionModel,
        vae: &Autoencoder,
        conditioning: &Array,
        pooled: &Array,
        guidance: f32,
        seed: u64,
        edge: u32,
        steps: usize,
        dtype: Dtype,
    ) -> Result<Image>;
}

/// Resolve the config's target-module *suffixes* (default `to_q`/`to_k`/`to_v`/`to_out.0`) to full
/// dotted UNet paths by suffix-matching them against the routable Linear surface — the same match
/// PEFT's `LoraConfig(target_modules=…)` does over the UNet attention modules.
///
/// The surface is chosen to match each adapter kind's **inference consumption** (so nothing trains
/// that no inference path reads, and the adapter round-trips cleanly):
///   * **LoRA** → the **complete** surface ([`UNet2DConditionModel::lora_target_paths_complete`]),
///     which `LoraCoverage::Complete` (the SDXL-family `model::load` default) merges — down / **mid** /
///     up attention. Matches the torch PEFT suffix-match (which hits `mid_block` too).
///   * **LoKr** → the **vendored** surface ([`UNet2DConditionModel::lora_target_paths`]), down / up
///     attention only: the SDXL LoKr loader keeps `mid_block` out (sc-2640), so a `mid_block` LoKr
///     factor would be skipped at load. Training to the vendored surface keeps train/inference in
///     lock-step.
pub fn resolve_target_paths(unet: &UNet2DConditionModel, cfg: &TrainingConfig) -> Vec<String> {
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

/// One forward+backward over the trainable adapter factors: build the noisy input at the sampled
/// timestep (via [`SdxlFamilyHooks::add_noise`]), inject `params` (LoRA or LoKr), run the U-Net,
/// regress the predicted `eps` toward the unit `noise`, return `(loss, grads)`.
///
/// `dtype` is the training compute dtype (sc-4941): for bf16 the noisy latent / conditioning / pooled
/// are cast to bf16 at entry (the U-Net weights were cast once in [`train_family`]) and the LoRA
/// factors / LoKr delta are reconstructed at bf16 inside the traced install — so the whole U-Net graph
/// runs bf16 with no silent f32 re-promotion. The noise target, loss, and grads stay f32
/// (master-weights pattern).
#[allow(clippy::too_many_arguments)]
pub fn compute_loss_grads<H: SdxlFamilyHooks>(
    hooks: &H,
    unet: &mut UNet2DConditionModel,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    cond: &Array,
    pooled: &Array,
    time_ids: &Array,
    t: TrainTimestep,
    noise: &Array,
    mae: bool,
    dtype: Dtype,
    checkpoint_targets: Option<Vec<String>>,
) -> Result<(f32, LoraParams)> {
    // The renormalized DDPM noisy input at the sampled timestep — the family hook hides the SDXL
    // sigma-space vs Kolors `alphas_cumprod` convention. The epsilon target is the unit `noise`.
    let noisy = hooks.add_noise(x0, noise, t)?.as_dtype(dtype)?;
    let t_f = t.unet_time();
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
                .forward_block_checkpointed(&noisy, t_f, &cond, &pooled, &time_ids, tp, &p, alpha)
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => unet
                .forward(&noisy, t_f, &cond, &pooled, &time_ids)
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

/// Refuse a run whose dense first step would exceed this machine's memory budget, returning a
/// catchable, actionable error instead of risking an uncatchable SIGKILL (sc-4874/sc-4941). Only
/// consulted when gradient checkpointing is OFF. `edge` is the bucketed training edge; the projection
/// is the family's fitted [`SdxlFamilyHooks::peak_gb`] curve.
fn preflight_memory_guard<H: SdxlFamilyHooks>(hooks: &H, edge: u32, bf16: bool) -> Result<()> {
    let latent_side = (edge as f64 / 8.0).ceil();
    let p = latent_side * latent_side;
    let projected = hooks.peak_gb(p, bf16);
    let budget_gb = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);
    let safe = budget_gb * 0.85;
    if projected > safe {
        return Err(format!(
            "{} trainer: a dense first training step at resolution {edge} needs ~{projected:.0} GB \
             (the forward working set materializes in one allocation), exceeding this machine's \
             ~{safe:.0} GB safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the OS \
             could hard-kill the worker (SIGKILL) at the first step with no recoverable error. Enable \
             Gradient Checkpointing or reduce the training resolution.",
            hooks.label()
        )
        .into());
    }
    Ok(())
}

/// Decode a dataset image file (PNG/JPEG) into the core RGB8 [`Image`](mlx_gen::media::Image).
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

/// The shared SDXL-family LoRA/LoKr training lifecycle: prepare → load → cache (VAE-latents +
/// conditioning/pooled) → functional-autograd loop (LR schedule, gradient accumulation, checkpoint
/// cadence, cancel, preview samples) → save an adapter that round-trips through the family's inference
/// loader. The per-family deltas are supplied by `hooks`; `unet`/`vae` are the (shared) SDXL types the
/// trainer owns. `validate` runs in the caller's `Trainer::train` before this is entered.
pub fn train_family<H: SdxlFamilyHooks>(
    hooks: &mut H,
    unet: &mut UNet2DConditionModel,
    vae: &Autoencoder,
    req: &TrainingRequest,
    on_progress: &mut dyn FnMut(TrainingProgress),
) -> Result<TrainingOutput> {
    let cfg = &req.config;
    let label = hooks.label();
    on_progress(TrainingProgress::Preparing);
    let edge = bucket_resolution(cfg.resolution);

    // sc-4941 — training compute dtype. bf16 (the worker default, passed through since sc-4881) halves
    // the activation working set and is the ecosystem-standard mixed precision; the trainable factors /
    // loss / grads / optimizer stay f32 (master-weights). The U-Net f32→bf16 cast is destructive, so a
    // trainer already cast to bf16 cannot honor a later f32 request — reload instead of silently
    // training at the wrong precision.
    let use_bf16 = cfg.train_dtype.trim().eq_ignore_ascii_case("bf16")
        || cfg.train_dtype.trim().eq_ignore_ascii_case("bfloat16");
    let compute_dtype = if use_bf16 {
        Dtype::Bfloat16
    } else {
        Dtype::Float32
    };
    if !use_bf16 && unet.compute_dtype() == Some(Dtype::Bfloat16) {
        return Err(format!(
            "{label} trainer: this trainer instance was already cast to bf16 by a previous run; \
             reload the trainer for f32 training"
        )
        .into());
    }

    // sc-4941 — opt-in gradient checkpointing (the SceneWorks "Gradient Checkpointing" toggle). When
    // on, each down/up macro-block recomputes its activations in the backward
    // (`forward_block_checkpointed`) instead of retaining them — the lever that makes 1280+ training fit
    // a 32 GB Mac (1024 already fits dense bf16). LoRA-only: LoKr falls back to the dense path (a
    // distinct Kronecker reconstruction), where the pre-flight guard refuses a run that would exceed the
    // memory budget. The block recompute already covers attention, so the standalone SDPA-segment
    // checkpoint stays off (nesting = double recompute).
    let use_checkpoint =
        matches!(cfg.network_type, NetworkType::Lora) && cfg.gradient_checkpointing;
    if !use_checkpoint {
        preflight_memory_guard(hooks, edge, use_bf16)?;
    }
    unet.set_sdpa_checkpoint(false);
    if use_bf16 {
        unet.cast_weights(Dtype::Bfloat16)?;
    }

    // --- prepare → load → cache: VAE-latents + (conditioning, pooled) into memory ---
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
        let x0 = encode_init_latents(vae, &img, edge, edge)?; // scaled latent [1,h,w,4]
        let (cond, pooled) = hooks.encode_prompt(&item.caption)?;
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
        return Err(format!("{label} trainer: no usable dataset items").into());
    }

    // sc-5637 — pre-encode the preview-sample prompts as a **CFG batch** (`[2, …]` = positive then
    // empty-negative) while the text encoder(s) are still resident (freed just below). The family
    // renders previews with real classifier-free guidance, so the denoise needs both streams.
    let sample_caps: Vec<(String, Array, Array)> =
        if cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() && !req.cancel.is_cancelled() {
            let mut caps = Vec::with_capacity(cfg.sample_prompts.len().min(SAMPLE_PROMPT_CAP));
            for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                let (cond, pooled) = hooks.encode_sample_cfg(prompt)?;
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

    // sc-4941 (32 GB-Mac headroom) — the prompts are all encoded into `cache`, so the text encoder(s)
    // are dead weight for the rest of the run. Drop them and evict their buffers before the train loop,
    // reclaiming their footprint for the U-Net working set.
    hooks.free_text_encoders();
    mlx_rs::memory::clear_cache();

    // Family micro-conditioning `time_ids`, built once and shared (B=1) — matches the inference path so
    // the LoRA trains under the conditioning it is applied under.
    let time_ids = hooks.time_ids(1, edge);

    // --- adapter targets + params (LoRA or LoKr) + optimizer ---
    let target_paths = resolve_target_paths(unet, cfg);
    // When block checkpointing is on, the per-step forward threads these target paths' LoRA factors
    // through the block checkpoints; `None` selects the dense forward.
    let checkpoint_targets: Option<Vec<String>> = use_checkpoint.then(|| target_paths.clone());
    let rank = cfg.rank as f32;
    let (adapter, mut params) = match cfg.network_type {
        NetworkType::Lora => {
            let (targets, params) =
                build_lora_targets(unet, &target_paths, cfg.rank as i32, cfg.seed)?;
            (TrainAdapter::Lora { targets }, params)
        }
        NetworkType::Lokr => {
            let (targets, params) = build_lokr_targets(
                unet,
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
    let (total_updates, warmup_updates) = schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
    let stem = Path::new(&req.file_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("lora")
        .to_string();

    // --- resume (F-125): continue from the latest snapshot of THIS adapter in output_dir, if any ---
    let mut update_idx: u32 = 0;
    let mut start_step: u32 = 0;
    if cfg.resume {
        if let Some((snapshot, _)) = checkpoint::find_latest_resume(&req.output_dir, &stem) {
            let (loaded, meta) = checkpoint::load_resume(&snapshot, &mut opt)?;
            params = loaded;
            start_step = meta.step;
            update_idx = meta.update_idx;
            eprintln!(
                "[F-125] {label} resuming from step {start_step} (optimizer update {update_idx})"
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
        let (x0, cond, pooled) = &cache[((step - 1) as usize) % cache.len()];
        let t =
            hooks.sample_timestep(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64))?;
        let noise = random::normal::<f32>(
            x0.shape(),
            None,
            None,
            Some(&random::key(
                cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
            )?),
        )?;
        let (loss, grads) = compute_loss_grads(
            hooks,
            unet,
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
            let mult = lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
            opt.set_lr_scaled(mult);
            // F-017: average by the ACTUAL in-window count, not the full `accum`. The final-step
            // flush is usually a partial window (cfg.steps % accum != 0); dividing by `accum`
            // down-scaled that update (halved effective LR on the tail) for BOTH the SDXL and
            // Kolors trainers. Mirrors z-image/lens F-069. (When step%accum==0 the window is the
            // full `accum`.)
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
            adapter.save(
                &params,
                alpha,
                rank,
                cfg.decompose_factor,
                PEFT_PREFIX,
                &ckpt,
            )?;
            // F-125: the resume bundle (raw factors + optimizer state + step/update index) siblings the
            // PEFT checkpoint, so a later `config.resume` run continues this schedule from here.
            checkpoint::save_resume(&req.output_dir, &stem, step, update_idx, &opt, &params)?;
            on_progress(TrainingProgress::Checkpoint { step });
        }

        // sc-5637 — periodic best-effort previews from the in-progress adapter (mirrors z-image).
        // Install the current factors as concrete adapters for the forward-only render; the next step's
        // traced `loss_fn` re-installs them, so no teardown is needed. A render failure must NOT abort
        // the long training run — log and continue.
        if sampling_enabled && step % cfg.sample_every == 0 {
            let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);
            adapter.install_as(unet, &params, alpha, rank, lora_dtype, LOKR_DTYPE)?;
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
                match hooks.render_sample(
                    unet,
                    vae,
                    cond,
                    pooled,
                    cfg.sample_guidance_scale,
                    sample_seed,
                    edge,
                    cfg.sample_steps.max(1) as usize,
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
                        "[sc-5637] {label} preview sample failed at step {step} \
                         (prompt {}): {e} — skipping this preview, training continues",
                        i + 1
                    ),
                }
            }
        }
    }

    // Cancelled before completing a single step (`steps == 0` is rejected upstream by `validate`): the
    // LoRA factors are still freshly initialized with `B = 0`, a no-op adapter. Surface the typed
    // `Error::Canceled` (sc-4895, bridged 1:1 to `gen_core::Error::Canceled`) rather than writing a
    // valid-looking `.safetensors` and returning `Ok` — downstream tooling would otherwise ship an
    // identity LoRA as a trained artifact (F-040).
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
