//! The end-to-end Lens-Turbo / Lens T2I pipeline (sc-3173) — wires the four ported components into a
//! single `generate`: [`LensTokenizer`](crate::text::LensTokenizer) → the gpt-oss
//! [`LensTextEncoder`](crate::text_encoder::encoder::LensTextEncoder) (multi-layer capture + the
//! `txt_offset = 97` slice) → the [`LensTransformer`](crate::dit::LensTransformer) denoising DiT (with
//! the [`schedule`](crate::schedule) flow-match sigmas + norm-rescaled CFG) → the Flux.2
//! [`vae::decode`](crate::vae) shim.
//!
//! A faithful port of `_vendor/lens/pipeline.py::LensPipeline.__call__`. The two model variants
//! (`lens`, `lens_turbo`) share this code and arch — they differ only in their sampling defaults
//! (registered in [`crate::registry`]).
//!
//! ## Parity-critical details (from the reference `__call__`)
//! - **Encode → offset slice → align.** Positives and negatives are each encoded to the four captured
//!   gpt-oss layers, sliced at `input_ids[97:]`, then zero-padded to a shared `S_txt` and stacked
//!   `[pos; neg]` along the batch axis for the joint CFG forward. An **empty** negative is the
//!   unconditional branch: zero text features + an all-`false` mask (no text tokens), *not* a second
//!   encode (`encode_prompt`).
//! - **Joint CFG batch.** Each step runs the DiT once over `B·2` (here `B = 1`): `hidden = [x; x]`,
//!   `encoder_features = [pos; neg]`. The output splits `cond, uncond`; the per-step guidance is the
//!   **norm-rescaled** CFG ([`schedule::cfg_rescale`]).
//! - **Timestep.** The transformer is fed the *shifted sigma* directly (the reference `timestep /
//!   1000`, where `scheduler.timesteps = sigma · 1000`) — i.e. [`schedule::timesteps`].
//! - **Latents.** `[B, latent_h · latent_w, 128]`, `latent_{h,w} = {height,width} / 16`; the denoise
//!   is the core flow-match Euler step.

use mlx_rs::ops::{concatenate_axis, split, split_sections};
use mlx_rs::{Array, Dtype};

use mlx_gen::scheduler::compute_mu;
use mlx_gen::weights::Weights;
use mlx_gen::{
    resolve_flow_schedule, run_flow_sampler, CancelFlag, Error, Image, Progress, Quant, Result,
    TimestepConvention,
};
use mlx_gen_flux2::{load_vae, Flux2Vae};

use crate::config::GptOssConfig;
use crate::dit::{LensDitConfig, LensTransformer};
use crate::schedule::{self, cfg_rescale, lens_schedule};
use crate::text::{LensTokenizer, TXT_OFFSET};
use crate::text_encoder::encoder::LensTextEncoder;
use crate::vae;

/// The VAE downsample factor (`vae_scale_factor`): a Lens latent cell maps to a 16×16 pixel tile
/// (Flux.2's 8× conv VAE composed with the 2× DiT patchify).
pub const VAE_SCALE_FACTOR: u32 = 16;

/// Default harmony-preamble date (`Current date:`). The preamble is the first [`TXT_OFFSET`] tokens,
/// which are **sliced off** before the DiT conditioning, so its `date` line never reaches the image
/// path — a fixed constant keeps generation deterministic regardless of wall-clock. (The Python
/// worker passes the live date; the value is image-irrelevant.)
pub const DEFAULT_DATE: &str = "2025-01-01";

/// Options for a single [`LensPipeline::generate`] call.
pub struct GenerateOptions<'a> {
    pub prompt: &'a str,
    /// Empty ⇒ the unconditional branch (zero text features), matching the reference default `""`.
    pub negative_prompt: &'a str,
    /// Output pixels — both must be divisible by [`VAE_SCALE_FACTOR`] (use [`crate::resolution`]).
    pub height: u32,
    pub width: u32,
    pub num_steps: usize,
    pub guidance_scale: f32,
    /// Curated sampler name (epic 7114 sc-7305); `None` ⇒ the engine default `euler`, byte-equivalent
    /// to the legacy flow-match loop (N1). The legacy `flow_match_euler` alias / any unknown name falls
    /// back to `euler` (N3).
    pub sampler: Option<&'a str>,
    /// Curated scheduler name (epic 7114 sc-7305); `None` ⇒ the native empirical-μ flow-match schedule
    /// (the byte-exact N1 default). The legacy `flow_match` alias / any unknown name falls back to it (N3).
    pub scheduler: Option<&'a str>,
    pub seed: u64,
    /// Harmony-preamble `Current date:` (image-irrelevant for the encode path; the **reasoner** uses
    /// it for its own preamble — see [`DEFAULT_DATE`]).
    pub date: &'a str,
    /// Refine the prompt through the attached local [`LensReasoner`](crate::reasoner::LensReasoner)
    /// before encoding (sc-3176, the vendor `enable_reasoner`). Requires
    /// [`attach_reasoner`](LensPipeline::attach_reasoner); off by default.
    pub enable_reasoner: bool,
}

/// A loaded Lens pipeline: the four components, shared by both variants, plus the **optional** local
/// prompt reasoner (sc-3176 — `None` unless [`attach_reasoner`](LensPipeline::attach_reasoner)d, so the
/// default pipeline carries no extra gpt-oss footprint for an off-by-default feature).
pub struct LensPipeline {
    tokenizer: LensTokenizer,
    encoder: LensTextEncoder,
    transformer: LensTransformer,
    vae: Flux2Vae,
    reasoner: Option<crate::reasoner::LensReasoner>,
    num_text_layers: usize,
    dtype: Dtype,
}

impl LensPipeline {
    /// Load all four components from a `microsoft/Lens-Turbo` (or `microsoft/Lens`) snapshot directory
    /// at `dtype` (bf16 production / f32 tight-gate). The snapshot is the diffusers multi-component
    /// tree: `tokenizer/tokenizer.json`, `text_encoder/`, `transformer/`, `vae/`. The VAE always runs
    /// f32 internally (the shared Flux.2 decoder).
    pub fn load(snapshot_dir: impl AsRef<std::path::Path>, dtype: Dtype) -> Result<Self> {
        Self::load_quant(snapshot_dir, dtype, None)
    }

    /// As [`load`](Self::load) but quantizes the gpt-oss encoder's MoE experts to Q4/Q8 (sc-3172) so
    /// the encoder loads at `~12 GB` instead of `~40 GB` bf16 — the per-layer dequant is the only
    /// transient. The DiT and VAE stay dense (DiT quant is sc-3175); the dominant footprint is the
    /// 20 B-param encoder, so quantizing it is the memory win.
    pub fn load_quant(
        snapshot_dir: impl AsRef<std::path::Path>,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Result<Self> {
        let root = snapshot_dir.as_ref();
        let tokenizer = LensTokenizer::from_file(root.join("tokenizer").join("tokenizer.json"))?;

        let enc_cfg = GptOssConfig::lens();
        let enc_w = Weights::from_dir(root.join("text_encoder"))?;
        let encoder = LensTextEncoder::from_weights_quant(&enc_w, &enc_cfg, dtype, quant)?;

        let dit_cfg = LensDitConfig::lens();
        let dit_w = Weights::from_dir(root.join("transformer"))?;
        let transformer = LensTransformer::from_weights(&dit_w, &dit_cfg, dtype)?;

        let vae = load_vae(root)?;

        Ok(Self {
            tokenizer,
            encoder,
            transformer,
            vae,
            reasoner: None,
            num_text_layers: dit_cfg.num_text_layers,
            dtype,
        })
    }

    /// Attach a local prompt [`LensReasoner`](crate::reasoner::LensReasoner) (sc-3176), enabling the
    /// `enable_reasoner` path in [`generate`](Self::generate). Loaded separately (its own gpt-oss copy)
    /// so the base pipeline pays nothing for this off-by-default feature.
    pub fn attach_reasoner(&mut self, reasoner: crate::reasoner::LensReasoner) {
        self.reasoner = Some(reasoner);
    }

    /// Encode one prompt to its per-layer DiT text features (sliced at [`TXT_OFFSET`]) + the valid
    /// mask. Returns `(features, mask)` where `features` is `num_text_layers × [1, S, 2880]` and
    /// `mask` is `[1, S]` (all-`1`; a single prompt is unpadded). When the rendered prompt is ≤ the
    /// offset (never, for real prompts) the features collapse to length 0 (`_get_text_embeddings`).
    fn encode_one(&self, prompt: &str, date: &str) -> Result<(Vec<Array>, Array)> {
        let out = self.tokenizer.encode(prompt, date)?;
        let l = out.ids.len();
        let input_ids = Array::from_slice(&out.ids, &[1, l as i32]);
        let layers = self.encoder.encode(&input_ids)?; // num_text_layers × [1, L, 2880]

        let offset = TXT_OFFSET as i32;
        if l as i32 > offset {
            let s = l as i32 - offset;
            // `[:, offset:, :]` — split at `offset` along the sequence axis, keep the tail.
            let features = layers
                .iter()
                .map(|f| Ok(split_sections(f, &[offset], 1)?[1].clone()))
                .collect::<Result<Vec<_>>>()?;
            // Single unpadded prompt ⇒ every retained token is valid.
            let mask = mlx_rs::ops::ones::<f32>(&[1, s])?;
            Ok((features, mask))
        } else {
            // `input_ids` shorter than the offset (never for a real prompt): length-0 features.
            let dim = layers[0].shape()[2];
            let features = (0..self.num_text_layers)
                .map(|_| Ok(mlx_rs::ops::zeros::<f32>(&[1, 0, dim])?.as_dtype(self.dtype)?))
                .collect::<Result<Vec<_>>>()?;
            let mask = mlx_rs::ops::zeros::<f32>(&[1, 0])?;
            Ok((features, mask))
        }
    }

    /// Encode positives + negatives and assemble the joint CFG batch (`encode_prompt` +
    /// `_align_text_features` + the `[pos; neg]` stack). Returns `(encoder_features, encoder_mask)`
    /// where each feature layer is `[2, S_txt, 2880]` and the mask is `[2, S_txt]` (`1` = valid).
    pub fn encode_prompt(
        &self,
        prompt: &str,
        negative_prompt: &str,
        date: &str,
    ) -> Result<(Vec<Array>, Array)> {
        let (pos_feats, pos_mask) = self.encode_one(prompt, date)?;
        let s_pos = pos_feats[0].shape()[1];

        // Empty negative ⇒ the unconditional branch: zero text features matching the positive shape +
        // an all-`false` (all-zero) mask. A non-empty negative is encoded normally.
        let (neg_feats, neg_mask) = if negative_prompt.trim().is_empty() {
            let zeros = pos_feats
                .iter()
                .map(mlx_rs::ops::zeros_like)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            (zeros, mlx_rs::ops::zeros_like(&pos_mask)?)
        } else {
            self.encode_one(negative_prompt, date)?
        };
        let s_neg = neg_feats[0].shape()[1];

        // Pad both to a shared S_txt = max(s_pos, s_neg).
        let target = s_pos.max(s_neg);
        let pos_feats = pad_features(&pos_feats, s_pos, target)?;
        let neg_feats = pad_features(&neg_feats, s_neg, target)?;
        let pos_mask = pad_mask(&pos_mask, s_pos, target)?;
        let neg_mask = pad_mask(&neg_mask, s_neg, target)?;

        // Stack [pos; neg] along the batch axis → the joint CFG forward.
        let mut encoder_features = Vec::with_capacity(self.num_text_layers);
        for (pf, nf) in pos_feats.iter().zip(neg_feats.iter()) {
            encoder_features.push(concatenate_axis(&[pf, nf], 0)?.as_dtype(self.dtype)?);
        }
        let encoder_mask = concatenate_axis(&[&pos_mask, &neg_mask], 0)?; // [2, S_txt]
        Ok((encoder_features, encoder_mask))
    }

    /// The denoising loop over pre-encoded conditioning + an initial latent, on the engine **default**
    /// sampler/scheduler (`euler` over the native empirical-μ flow-match schedule). Exposed for the e2e
    /// parity gate (which injects the reference's initial latents to factor out cross-RNG noise). A thin
    /// wrapper over [`denoise_with_sampler`](Self::denoise_with_sampler) forcing the default — `euler`
    /// reproduces the legacy `FlowMatchEuler::step` loop within the N1 parity tolerance (the shared
    /// `flow_match_euler_step` + gen-core keystone equivalence).
    ///
    /// - `encoder_features`: `num_text_layers × [2, S_txt, 2880]` (`[pos; neg]`).
    /// - `encoder_mask`: `[2, S_txt]` (`1` = valid).
    /// - `init_latents`: `[1, latent_h · latent_w, 128]`.
    ///
    /// Returns the final latents `[1, latent_h · latent_w, 128]` (patch-space; feed to [`vae::decode`]).
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        encoder_features: &[Array],
        encoder_mask: &Array,
        init_latents: &Array,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Array> {
        self.denoise_with_sampler(
            encoder_features,
            encoder_mask,
            init_latents,
            latent_h,
            latent_w,
            num_steps,
            guidance_scale,
            None, // default sampler — `euler` == the legacy flow-match loop
            None, // native empirical-μ schedule
            0,    // seed unused by the deterministic default; stochastic solvers key off it
            cancel,
            on_step,
        )
    }

    /// The denoising loop routed through the unified curated-sampler framework (epic 7114 sc-7305): the
    /// per-generation `sampler` (integration method) + `scheduler` (σ schedule) knobs. Lens is
    /// rectified-flow (FLOW prediction) fed the **raw shifted sigma** as its timestep
    /// ([`TimestepConvention::Sigma`]); each step's velocity is the norm-rescaled joint-CFG combination
    /// (one DiT forward over `[pos; neg]`, [`schedule::cfg_rescale`]) — the body of the legacy loop, now
    /// driven by the curated solver.
    ///
    /// - `sampler_name`: a curated solver name (`euler` / `heun` / `dpmpp_2m` / `uni_pc` / …). `None`,
    ///   the legacy `flow_match_euler` alias, or any unknown name falls back to `euler` (N3) — the
    ///   byte-equivalent default.
    /// - `scheduler_name`: a curated scheduler name (`karras` / `exponential` / …). `None`, the legacy
    ///   `flow_match` alias, or any unknown name returns the native schedule verbatim (the N1 byte-exact
    ///   default).
    /// - `seed`: drives the per-step noise of the stochastic solvers (`euler_ancestral` / `dpmpp_sde` /
    ///   `lcm`); the deterministic solvers ignore it.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_with_sampler(
        &self,
        encoder_features: &[Array],
        encoder_mask: &Array,
        init_latents: &Array,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        sampler_name: Option<&str>,
        scheduler_name: Option<&str>,
        seed: u64,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Array> {
        // The native Lens schedule is the byte-exact N1 default: empirical-μ flow-match sigmas (length
        // num_steps + 1, trailing 0). A curated `scheduler_name` re-shapes σ over the SAME empirical μ
        // (`compute_mu(seq_len, steps)`), so `karras` / `exponential` / … stay consistent with Lens's
        // resolution-/step-dependent shift instead of degrading to a linear ramp; an unset / legacy /
        // unknown name returns this native schedule verbatim.
        let native = lens_schedule(num_steps, latent_h, latent_w).sigmas;
        let mu = compute_mu(latent_h * latent_w, num_steps);
        let sigmas = resolve_flow_schedule(scheduler_name, mu, num_steps, &native);

        let dtype = self.dtype;
        let transformer = &self.transformer;
        // FLOW velocity field: the norm-rescaled joint-CFG combination over one DiT forward. Lens feeds
        // the raw shifted sigma as the timestep directly (Sigma convention), matching the legacy loop.
        let predict = |latents: &Array, sigma: f32| -> Result<Array> {
            // Joint CFG batch: duplicate the latent (cond/uncond share x_t), one DiT call.
            let hidden = concatenate_axis(&[latents, latents], 0)?; // [2, seq, 128]
            let timestep = Array::from_slice(&[sigma, sigma], &[2]).as_dtype(dtype)?;
            let noise = transformer.forward(
                &hidden,
                encoder_features,
                Some(encoder_mask),
                &timestep,
                1,
                latent_h,
                latent_w,
            )?;
            // chunk(2) → cond (positive, batch 0), uncond (negative, batch 1).
            let parts = split(&noise, 2, 0)?;
            cfg_rescale(&parts[0], &parts[1], guidance_scale)
        };

        // Adapt the framework's `Progress` callback to the pipeline's (completed, total) step callback.
        let mut on_progress = |p: Progress| {
            if let Progress::Step { current, total } = p {
                on_step(current as usize, total as usize);
            }
        };

        // Route through the unified curated-sampler framework (epic 7114 P3 seam): cancellation, the
        // per-step `eval` (sc-5399 — bounds the lazy graph so a mid-render cancel lands within ~1 model
        // eval), and progress are handled inside `run_flow_sampler`. `euler` reproduces the legacy
        // `FlowMatchEuler::step` loop within the N1 parity tolerance.
        run_flow_sampler(
            sampler_name,
            TimestepConvention::Sigma,
            &sigmas,
            init_latents.as_dtype(dtype)?,
            seed,
            cancel,
            &mut on_progress,
            predict,
        )
    }

    /// Generate a single image (no cancellation / progress). Draws the initial latents from the
    /// global RNG seeded with `opts.seed`. Native-VAE decode (no PiD overlay).
    pub fn generate(&self, opts: &GenerateOptions) -> Result<Image> {
        self.generate_with_progress(opts, None, &CancelFlag::default(), &mut |_| {})
    }

    /// Generate a single image, threading a cancel flag and a per-step progress callback
    /// (`on_step(completed_step)`). The registry loops `count` with per-image seeds over this.
    ///
    /// `pid_decoder`: an optional PiD super-resolving decoder (epic 7840, sc-7847) that replaces the
    /// native Flux.2 VAE decode (4× SR). `None` → the byte-exact VAE path.
    pub fn generate_with_progress(
        &self,
        opts: &GenerateOptions,
        pid_decoder: Option<&dyn mlx_gen::LatentDecoder>,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize),
    ) -> Result<Image> {
        if !opts.width.is_multiple_of(VAE_SCALE_FACTOR)
            || !opts.height.is_multiple_of(VAE_SCALE_FACTOR)
        {
            return Err(Error::Msg(format!(
                "lens: height/width must be divisible by {VAE_SCALE_FACTOR} (got {}x{})",
                opts.height, opts.width
            )));
        }
        if opts.num_steps == 0 {
            return Err(Error::Msg("lens: num_steps must be >= 1".into()));
        }
        let latent_h = (opts.height / VAE_SCALE_FACTOR) as usize;
        let latent_w = (opts.width / VAE_SCALE_FACTOR) as usize;
        let seq_len = (latent_h * latent_w) as i32;

        // Optional prompt refinement via the local reasoner (sc-3176) — before encoding.
        let refined;
        let prompt = if opts.enable_reasoner {
            let reasoner = self.reasoner.as_ref().ok_or_else(|| {
                Error::Msg(
                    "lens: enable_reasoner set but no reasoner attached (call attach_reasoner)"
                        .into(),
                )
            })?;
            refined = reasoner.refine(
                opts.prompt,
                crate::reasoner::DEFAULT_MAX_NEW_TOKENS,
                opts.date,
                Some(cancel),
            )?;
            &refined
        } else {
            opts.prompt
        };

        let (encoder_features, encoder_mask) =
            self.encode_prompt(prompt, opts.negative_prompt, opts.date)?;

        mlx_rs::random::seed(opts.seed)?;
        let init = mlx_rs::random::normal::<f32>(&[1, seq_len, 128], None, None, None)?;

        let latents = self.denoise_with_sampler(
            &encoder_features,
            &encoder_mask,
            &init,
            latent_h,
            latent_w,
            opts.num_steps,
            opts.guidance_scale,
            opts.sampler,
            opts.scheduler,
            opts.seed,
            cancel,
            &mut |cur, _total| on_step(cur),
        )?;

        // `vae::decode` returns NHWC [1, H, W, 3] (native) or [1, 4H, 4W, 3] (PiD, 4× SR) — both in
        // [-1,1] — so the NHWC `decoded_to_image` below handles either.
        let decoded = vae::decode(&self.vae, &latents, latent_h, latent_w, pid_decoder)?;
        decoded_to_image(&decoded)
    }

    /// The loaded VAE (for the e2e parity test's decode step).
    pub fn vae(&self) -> &Flux2Vae {
        &self.vae
    }

    /// Apply LoRA/LoKr adapters to the DiT (sc-3174) — stacked, mixed, strict (errors on an unmatched
    /// target). A LoRA trained on base `microsoft/Lens` applies to `Lens-Turbo` (same architecture).
    pub fn apply_adapters(&mut self, specs: &[mlx_gen::AdapterSpec]) -> Result<()> {
        crate::adapters::apply_lens_adapters(&mut self.transformer, specs)?;
        Ok(())
    }

    /// Quantize the DiT's linears to Q4/Q8 (sc-3175 — the complement to the encoder quant in
    /// [`load_quant`](Self::load_quant)). Call **after** [`apply_adapters`](Self::apply_adapters): the
    /// adapters are forward-time residuals over the now-quantized base, so quantize-after-merge is the
    /// correct order (the registry orchestrates it).
    pub fn quantize_dit(&mut self, quant: Quant) -> Result<()> {
        self.transformer.quantize(quant.bits())
    }
}

/// Render one preview sample (sc-5637) from the **in-progress training adapter** already installed on
/// `transformer`: seeded init latents → flow-match Euler norm-rescaled-CFG denoise → VAE decode →
/// [`Image`]. A stripped [`LensPipeline::denoise`] + decode for the trainer (which holds the raw DiT +
/// VAE, not a `LensPipeline` — the gpt-oss encoder is freed after caching). `encoder_features`/
/// `encoder_mask` are the pre-encoded joint CFG batch (`[2, …]` = positive then empty-negative);
/// `dtype` is the trainer compute dtype. No progress/cancel plumbing — the caller drives the cadence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_sample(
    transformer: &LensTransformer,
    vae: &Flux2Vae,
    encoder_features: &[Array],
    encoder_mask: &Array,
    seed: u64,
    edge: u32,
    num_steps: usize,
    guidance_scale: f32,
    dtype: Dtype,
) -> Result<Image> {
    let latent = (edge / VAE_SCALE_FACTOR) as usize;
    let seq_len = (latent * latent) as i32;
    let schedule = lens_schedule(num_steps.max(1), latent, latent);
    let timesteps = schedule::timesteps(&schedule);
    mlx_rs::random::seed(seed)?;
    let init = mlx_rs::random::normal::<f32>(&[1, seq_len, 128], None, None, None)?;
    let mut latents = init.as_dtype(dtype)?;
    for (i, &sigma) in timesteps.iter().enumerate() {
        // Joint CFG batch: duplicate the latent (cond/uncond share x_t), one DiT call (mirrors
        // `LensPipeline::denoise`).
        let hidden = concatenate_axis(&[&latents, &latents], 0)?;
        let timestep = Array::from_slice(&[sigma, sigma], &[2]).as_dtype(dtype)?;
        let noise = transformer.forward(
            &hidden,
            encoder_features,
            Some(encoder_mask),
            &timestep,
            1,
            latent,
            latent,
        )?;
        let parts = split(&noise, 2, 0)?;
        let noise_pred = cfg_rescale(&parts[0], &parts[1], guidance_scale)?;
        latents = schedule.step(&latents, &noise_pred, i)?;
        latents.eval()?;
    }
    let decoded = vae::decode(vae, &latents, latent, latent, None)?; // training preview: native VAE
    decoded_to_image(&decoded)
}

/// Zero-pad each `[B, cur, C]` feature layer along the sequence axis to length `target`.
fn pad_features(features: &[Array], cur: i32, target: i32) -> Result<Vec<Array>> {
    if cur == target {
        return Ok(features.to_vec());
    }
    let pad = target - cur;
    features
        .iter()
        .map(|f| {
            let (b, c) = (f.shape()[0], f.shape()[2]);
            let z = mlx_rs::ops::zeros::<f32>(&[b, pad, c])?.as_dtype(f.dtype())?;
            Ok(concatenate_axis(&[f, &z], 1)?)
        })
        .collect()
}

/// Zero-pad a `[B, cur]` mask along the sequence axis to length `target`.
fn pad_mask(mask: &Array, cur: i32, target: i32) -> Result<Array> {
    if cur == target {
        return Ok(mask.clone());
    }
    let pad = target - cur;
    let b = mask.shape()[0];
    let z = mlx_rs::ops::zeros::<f32>(&[b, pad])?;
    Ok(concatenate_axis(&[mask, &z], 1)?)
}

/// Convert a decoded image `[1, H, W, 3]` (NHWC) in `[-1, 1]` to an RGB8 [`Image`]
/// (`((x·0.5+0.5).clamp(0,1)·255).round()`), matching the reference `_to_pil` quantization.
fn decoded_to_image(decoded: &Array) -> Result<Image> {
    let x = decoded.as_dtype(Dtype::Float32)?;
    let half = Array::from_f32(0.5);
    let x = mlx_rs::ops::add(&mlx_rs::ops::multiply(&x, &half)?, &half)?;
    let x = mlx_rs::ops::clip(&x, (0.0, 1.0))?;
    let x = mlx_rs::ops::round(&mlx_rs::ops::multiply(&x, Array::from_f32(255.0))?, 0)?;
    let sh = x.shape();
    // One image per call: reject B>1 instead of silently keeping only batch 0, and size in usize /
    // flatten via -1 to avoid the u32/i32 product overflow at large resolutions (F-068).
    if sh[0] != 1 {
        return Err(Error::Msg(format!(
            "lens decoded_to_image: expected batch size 1, got {}",
            sh[0]
        )));
    }
    let (h, w, c) = (sh[1] as usize, sh[2] as usize, sh[3] as usize);
    let n = h * w * c;
    let flat = x.reshape(&[-1])?;
    let pixels: Vec<u8> = flat.as_slice::<f32>()[..n]
        .iter()
        .map(|&v| v as u8)
        .collect();
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}
