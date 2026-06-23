//! Boogu Base text-to-image pipeline (E5): tokenize → condition-encode → flow-match denoise with
//! true-CFG → VAE decode. Port of the core `BooguImagePipeline.__call__` path (T2I, no reference
//! images, no rewriter / boosted-orthogonal / image-guidance extras).
//!
//! Scheduler is the snapshot's `FlowMatchEulerDiscreteScheduler` in its **static v1** configuration
//! (`do_shift=true`, `dynamic_time_shift=false`, `time_shift_version="v1"`, `seq_len=4096`): the
//! `linspace(0,1,n+1)[:-1]` grid is logistic-shifted by a constant `mu = lin(seq_len) = 1.15`, then a
//! trailing `1.0` is appended; each Euler step is `x += (t_next − t)·v` (t ascending 0→1, latent
//! initialized as pure noise). True-CFG: `pred = cond + (scale − 1)·(cond − uncond)` with the uncond
//! pass run on the empty (drop) instruction. Per-sample `B=1` (the DiT runs once per condition).

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use mlx_gen::image::{decoded_to_image, validate_multiple_of_16};
use mlx_gen::media::Image;
use mlx_gen::{
    resolve_flow_schedule, run_flow_sampler, CancelFlag, Error, Progress, Result,
    TimestepConvention,
};

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use crate::loader::{load_text_encoder, load_transformer, load_vae, load_vision_tower};
use crate::text_encoder::BooguTextEncoder;
use crate::tokenizer::BooguTokenizer;
use crate::transformer::BooguTransformer;
use crate::vision::preprocess::preprocess_image;
use crate::vision::VisionTower;
use mlx_gen_z_image::vae::Vae;

/// Qwen3-VL image placeholder token (`mllm/config.json::image_token_id`) — the position the vision
/// tower's merged embeds are spliced into for image-conditioned editing.
const IMAGE_TOKEN_ID: i32 = 151655;

/// Static-v1 time-shift parameters from the snapshot `scheduler/scheduler_config.json`
/// (`base_shift 0.5`, `max_shift 1.15`, `seq_len 4096`). The linear map saturates at `seq_len=4096`,
/// so `mu` is the constant `max_shift`.
const SEQ_LEN: f64 = 4096.0;
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

/// Text-to-image generation knobs. Defaults mirror the reference `__call__`.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub height: u32,
    pub width: u32,
    pub steps: usize,
    pub text_guidance_scale: f32,
    pub seed: u64,
    /// Curated unified-framework integrator (epic 7114). `None` (or an unknown name) is the curated
    /// Euler — the legacy flow-match step within the N1 tolerance.
    pub sampler: Option<String>,
    /// Curated unified-framework scheduler (epic 7114). `None` keeps the native static-shift schedule
    /// (`mu = 1.15`) byte-exact; a curated name re-shapes σ over the same shift.
    pub scheduler: Option<String>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            height: 1024,
            width: 1024,
            steps: 50,
            text_guidance_scale: 4.0,
            seed: 0,
            sampler: None,
            scheduler: None,
        }
    }
}

/// Turbo (DMD few-step) generation knobs. Defaults mirror the standalone turbo pipeline.
#[derive(Debug, Clone)]
pub struct TurboOptions {
    pub height: u32,
    pub width: u32,
    pub steps: usize,
    pub seed: u64,
    /// DMD conditioning sigma — the first (lowest) sigma in the schedule.
    pub conditioning_sigma: f32,
    /// Curated unified-framework integrator (epic 7114). `None` is the native DMD student loop
    /// (predict → flow-renoise with fresh noise), the byte-exact default. A curated name routes the
    /// few-step denoise through [`run_flow_sampler`] over the DMD σ grid instead — the experimental
    /// Turbo sampler axis (sc-7491): the DMD x0 estimate is identical, only the renoise
    /// differs (curated `lcm`/ancestral re-noise VE-additively, the deterministic solvers integrate).
    /// See sc-7491 for the real-weight survey behind the advertised subset.
    pub sampler: Option<String>,
    /// Curated unified-framework scheduler (epic 7114). `None` keeps the native DMD σ grid byte-exact.
    pub scheduler: Option<String>,
}

impl Default for TurboOptions {
    fn default() -> Self {
        Self {
            height: 1024,
            width: 1024,
            steps: 4,
            seed: 0,
            conditioning_sigma: 0.001,
            sampler: None,
            scheduler: None,
        }
    }
}

/// Edit (text+image-to-image, one or more references) generation knobs. The output resolution is
/// `height`/`width`; each reference image's own dimensions drive its reference latent (all must be
/// multiples of 16). Defaults mirror the Base `__call__` (true-CFG, 50 steps).
#[derive(Debug, Clone)]
pub struct EditOptions {
    pub height: u32,
    pub width: u32,
    pub steps: usize,
    pub text_guidance_scale: f32,
    pub seed: u64,
    /// Faithful Boogu edit (default): route the reference image through the Qwen3-VL vision tower so
    /// the MLLM "sees" it (image-conditioned instruction features). When `false`, the instruction is
    /// encoded text-only (the E7 fallback) — the DiT still gets the spatial reference latent either way.
    pub condition_on_image: bool,
    /// Reference `use_input_images_4_neg_instruct`: also condition the CFG-negative (empty-instruction)
    /// pass on the reference image. Default `false` (the reference default + inference script) — the
    /// negative is the text-only empty/drop instruction. Only meaningful when `condition_on_image`.
    pub use_input_images_4_neg_instruct: bool,
    /// Curated unified-framework integrator (epic 7114); `None` is the curated Euler. As the Base path.
    pub sampler: Option<String>,
    /// Curated unified-framework scheduler (epic 7114); `None` keeps the native static-shift schedule.
    pub scheduler: Option<String>,
}

impl Default for EditOptions {
    fn default() -> Self {
        Self {
            height: 1024,
            width: 1024,
            steps: 50,
            text_guidance_scale: 4.0,
            seed: 0,
            condition_on_image: true,
            use_input_images_4_neg_instruct: false,
            sampler: None,
            scheduler: None,
        }
    }
}

/// The assembled Boogu Base pipeline: tokenizer + Qwen3-VL condition encoder + DiT + FLUX.1 VAE.
///
/// The Qwen3-VL **vision tower** (image-conditioned editing) is loaded lazily on the first
/// [`Self::generate_edit`] call and cached, so the text-to-image paths keep their original footprint.
pub struct BooguPipeline {
    tok: BooguTokenizer,
    te: BooguTextEncoder,
    dit: BooguTransformer,
    vae: Vae,
    /// Snapshot root — kept so the vision tower can be lazily loaded from `mllm/` on first edit.
    root: PathBuf,
    /// Lazily-loaded (f32) Qwen3-VL vision tower; `None` until the first image-conditioned edit.
    vision: RefCell<Option<VisionTower>>,
}

impl BooguPipeline {
    /// Load the text-to-image components from a standard Boogu snapshot (`mllm/`, `transformer/`,
    /// `vae/`). The vision tower (edit-only) loads lazily on the first [`Self::generate_edit`].
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        Ok(Self {
            tok: BooguTokenizer::from_snapshot(root)?,
            te: load_text_encoder(root)?,
            dit: load_transformer(root)?,
            vae: load_vae(root)?,
            root: root.to_path_buf(),
            vision: RefCell::new(None),
        })
    }

    /// Generate one RGB image from a text prompt. Convenience wrapper over
    /// [`Self::generate_with_progress`] with no cancellation and a no-op progress sink.
    pub fn generate(&self, prompt: &str, opts: &GenerateOptions) -> Result<Image> {
        self.generate_with_progress(prompt, opts, &CancelFlag::new(), &mut |_| {})
    }

    /// Generate one RGB image from a text prompt, streaming [`Progress`] and honoring `cancel` at
    /// each denoise step. A pre/mid-flight cancellation returns [`Error::Canceled`]; the per-step
    /// `eval` bounds the lazy MLX graph and lets the cancel check interrupt mid-render.
    ///
    /// The `RES_MIN`/`RES_MAX` (256–2048) resolution range and the multiple-of-`RES_MULTIPLE`
    /// constraint are a **Generator-layer guarantee** — enforced by `model::validate_request` (via
    /// the shared `Capabilities` min/max-size + the explicit multiple check) before this pipeline
    /// runs. This entry trusts the already-validated [`GenerateOptions`] dims (sc-6983).
    pub fn generate_with_progress(
        &self,
        prompt: &str,
        opts: &GenerateOptions,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "boogu")?;

        // Condition encoding: positive instruction + CFG-negative (empty/drop) instruction.
        let (cond_ids, cond_mask) = self.tok.encode_t2i(prompt)?;
        let cond = self.te.last_hidden(&cond_ids, &cond_mask)?;
        let do_cfg = opts.text_guidance_scale > 1.0;
        let uncond = if do_cfg {
            let (u_ids, u_mask) = self.tok.encode_negative()?;
            Some((self.te.last_hidden(&u_ids, &u_mask)?, u_mask))
        } else {
            None
        };

        // Initial latent noise [1, 16, H/8, W/8] (f32; the DiT casts to its compute dtype).
        let lat = init_noise(opts.height, opts.width, opts.seed, 0)?;

        // Route the flow-match denoise through the unified curated-sampler framework (epic 7114): the
        // native static-shift (`mu = 1.15`, v1 logistic) schedule is the byte-exact default and the
        // curated sampler/scheduler menus become selectable. Boogu feeds the shifted clean-fraction
        // timestep `t = 1 − σ` to the DiT (OneMinusSigma) and its DiT predicts the velocity in
        // clean-fraction time (`dz/dt`), so `predict` negates it into `run_flow_sampler`'s noise-fraction
        // FLOW convention (`dz/dσ = −dz/dt`) and casts to f32 (the native step's dtype). Cancellation,
        // the per-step `eval`, and progress all live in `run_flow_sampler`.
        let native = base_native_sigmas(opts.steps);
        let sigmas = resolve_flow_schedule(
            opts.scheduler.as_deref(),
            base_shift_mu(),
            opts.steps,
            &native,
        );
        let scale = opts.text_guidance_scale;
        let lat = run_flow_sampler(
            opts.sampler.as_deref(),
            TimestepConvention::OneMinusSigma,
            &sigmas,
            lat,
            opts.seed,
            cancel,
            on_progress,
            |x, timestep| {
                let t = Array::from_slice(&[timestep], &[1]);
                let cond_v = self.dit.forward(x, &t, &cond, &cond_mask)?;
                let pred = match &uncond {
                    Some((u_hidden, u_mask)) => {
                        let uncond_v = self.dit.forward(x, &t, u_hidden, u_mask)?;
                        // pred = cond + (scale − 1)·(cond − uncond)
                        add(
                            &cond_v,
                            &multiply(
                                &subtract(&cond_v, &uncond_v)?,
                                Array::from_f32(scale - 1.0),
                            )?,
                        )?
                    }
                    None => cond_v,
                };
                Ok(multiply(
                    &pred.as_dtype(Dtype::Float32)?,
                    Array::from_f32(-1.0),
                )?)
            },
        )?;

        on_progress(Progress::Decoding);
        self.decode_latents(&lat)
    }

    /// Generate one RGB image via the **Turbo** DMD student few-step sampler (Boogu-Image-0.1-Turbo).
    ///
    /// Pure T2I, **no CFG** (the distilled student needs `text_guidance_scale == 1`). The sigma grid
    /// is `linspace(conditioning_sigma, 1.0, steps+1)[:-1]` (ascending; `sigma` is the clean-fraction,
    /// so the latent starts as noise). Each step: predict → `x += (1 − sigma)·v`, then (except the
    /// last) renoise to the next level `x = (1 − sigma_next)·noise + sigma_next·x` with fresh noise.
    /// Same DiT/TE/VAE as Base — only the sampler differs — so load this from a Turbo snapshot.
    pub fn generate_turbo(&self, prompt: &str, opts: &TurboOptions) -> Result<Image> {
        self.generate_turbo_with_progress(prompt, opts, &CancelFlag::new(), &mut |_| {})
    }

    /// [`Self::generate_turbo`] with [`Progress`] streaming and per-step cooperative cancellation
    /// ([`Error::Canceled`]).
    pub fn generate_turbo_with_progress(
        &self,
        prompt: &str,
        opts: &TurboOptions,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "boogu")?;

        let (ids, mask) = self.tok.encode_t2i(prompt)?;
        let cond = self.te.last_hidden(&ids, &mask)?;

        // Curated sampler axis (epic 7114 sc-7297): a selected sampler/scheduler routes the few-step
        // denoise through the unified framework over the DMD σ grid. The DMD x0 estimate is identical
        // to the native loop (`x0 = x + (1−c)·v`, the OneMinusSigma flow denoise with the velocity
        // negated); only the renoise convention differs (the curated solver re-noises, the native loop
        // flow-blends). Unset (the default) is the native DMD student loop, byte-exact below.
        if opts.sampler.is_some() || opts.scheduler.is_some() {
            let lat = init_noise(opts.height, opts.width, opts.seed, 0)?;
            let native = turbo_native_sigmas(opts.conditioning_sigma, opts.steps);
            // The DMD grid is linear in clean-fraction (no logistic shift), so mu = 0 for a curated
            // scheduler re-shape over the same σ span.
            let sigmas = resolve_flow_schedule(opts.scheduler.as_deref(), 0.0, opts.steps, &native);
            let lat = run_flow_sampler(
                opts.sampler.as_deref(),
                TimestepConvention::OneMinusSigma,
                &sigmas,
                lat,
                opts.seed,
                cancel,
                on_progress,
                |x, timestep| {
                    let t = Array::from_slice(&[timestep], &[1]);
                    let v = self.dit.forward(x, &t, &cond, &mask)?;
                    Ok(multiply(
                        &v.as_dtype(Dtype::Float32)?,
                        Array::from_f32(-1.0),
                    )?)
                },
            )?;
            on_progress(Progress::Decoding);
            return self.decode_latents(&lat);
        }

        let mut lat = init_noise(opts.height, opts.width, opts.seed, 0)?;
        let sigmas = dmd_sigmas(opts.conditioning_sigma, opts.steps);

        for i in 0..opts.steps {
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let sigma = sigmas[i];
            let t = Array::from_slice(&[sigma], &[1]);
            let pred = self.dit.forward(&lat, &t, &cond, &mask)?;
            // Predict (clean estimate): x += (1 − sigma)·v, in f32.
            lat = add(
                &lat.as_dtype(Dtype::Float32)?,
                &multiply(
                    &pred.as_dtype(Dtype::Float32)?,
                    Array::from_f32(1.0 - sigma),
                )?,
            )?;
            // Renoise to the next sigma level with fresh noise (all but the final step).
            if i + 1 < opts.steps {
                let sigma_next = sigmas[i + 1];
                let noise = init_noise(opts.height, opts.width, opts.seed, (i + 1) as u64)?;
                lat = add(
                    &multiply(&noise, Array::from_f32(1.0 - sigma_next))?,
                    &multiply(&lat, Array::from_f32(sigma_next))?,
                )?;
            }
            eval([&lat])?;
            on_progress(Progress::Step {
                current: (i + 1) as u32,
                total: opts.steps as u32,
            });
        }

        on_progress(Progress::Decoding);
        self.decode_latents(&lat)
    }

    /// Generate one RGB image via the **Edit** path: VAE-encode each reference image into a clean
    /// reference latent, then flow-match denoise (true-CFG) with those references packed into the DiT's
    /// image sequence (`forward_edit`). The references shape the output spatially; the instruction
    /// drives the edit. Same static-v1 scheduler / true-CFG as [`Self::generate`]. Generalized to
    /// multiple references (the OmniGen2-lineage multi-image path, max 5).
    ///
    /// Faithful Boogu edit (the default, `opts.condition_on_image`) feeds every reference image through
    /// the Qwen3-VL **vision tower** so the MLLM "sees" them — the instruction features are
    /// image-conditioned (semantic path) **and** the DiT gets the spatial reference latents. With
    /// `condition_on_image = false` the instruction is encoded text-only (the E7 fallback); the DiT
    /// spatial reference path is identical either way. `references` must be non-empty.
    pub fn generate_edit(
        &self,
        references: &[&Image],
        instruction: &str,
        opts: &EditOptions,
    ) -> Result<Image> {
        self.generate_edit_with_progress(
            references,
            instruction,
            opts,
            &CancelFlag::new(),
            &mut |_| {},
        )
    }

    /// [`Self::generate_edit`] with [`Progress`] streaming and per-step cooperative cancellation
    /// ([`Error::Canceled`]).
    pub fn generate_edit_with_progress(
        &self,
        references: &[&Image],
        instruction: &str,
        opts: &EditOptions,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "boogu")?;
        for r in references {
            validate_multiple_of_16(r.width, r.height, "boogu")?;
        }

        // Each reference → clean VAE latent [1, 16, rH/8, rW/8].
        let ref_latents: Vec<Array> = references
            .iter()
            .map(|r| self.vae.encode(&image_to_pixels(r)?))
            .collect::<Result<_>>()?;

        // Condition encoding: edit instruction + CFG-negative (empty/drop) instruction. Both DiT
        // passes carry the same reference latents — only the instruction differs (TI2I text guidance).
        let (cond, cond_mask) = if opts.condition_on_image {
            self.encode_image_instruction(references, instruction)?
        } else {
            let (ids, mask) = self.tok.encode_edit(instruction)?;
            (self.te.last_hidden(&ids, &mask)?, mask)
        };
        let do_cfg = opts.text_guidance_scale > 1.0;
        let uncond = if do_cfg {
            // Reference `use_input_images_4_neg_instruct`: condition the negative on the images too
            // (empty instruction); otherwise the text-only empty/drop instruction.
            if opts.condition_on_image && opts.use_input_images_4_neg_instruct {
                Some(self.encode_image_instruction(references, "")?)
            } else {
                let (u_ids, u_mask) = self.tok.encode_negative()?;
                Some((self.te.last_hidden(&u_ids, &u_mask)?, u_mask))
            }
        } else {
            None
        };

        let lat = init_noise(opts.height, opts.width, opts.seed, 0)?;

        // Same unified-framework routing as the Base path (epic 7114), with the spatial reference latents
        // threaded through `forward_edit`. Native static-shift schedule is the byte-exact default;
        // velocity is negated into the noise-fraction FLOW convention. The output resolution starts from
        // pure noise (the references shape the DiT sequence, not the init latent), so there is no
        // start-step slicing.
        let native = base_native_sigmas(opts.steps);
        let sigmas = resolve_flow_schedule(
            opts.scheduler.as_deref(),
            base_shift_mu(),
            opts.steps,
            &native,
        );
        let scale = opts.text_guidance_scale;
        let lat = run_flow_sampler(
            opts.sampler.as_deref(),
            TimestepConvention::OneMinusSigma,
            &sigmas,
            lat,
            opts.seed,
            cancel,
            on_progress,
            |x, timestep| {
                let t = Array::from_slice(&[timestep], &[1]);
                let cond_v = self
                    .dit
                    .forward_edit(x, &ref_latents, &t, &cond, &cond_mask)?;
                let pred = match &uncond {
                    Some((u_hidden, u_mask)) => {
                        let uncond_v =
                            self.dit
                                .forward_edit(x, &ref_latents, &t, u_hidden, u_mask)?;
                        add(
                            &cond_v,
                            &multiply(
                                &subtract(&cond_v, &uncond_v)?,
                                Array::from_f32(scale - 1.0),
                            )?,
                        )?
                    }
                    None => cond_v,
                };
                Ok(multiply(
                    &pred.as_dtype(Dtype::Float32)?,
                    Array::from_f32(-1.0),
                )?)
            },
        )?;

        on_progress(Progress::Decoding);
        self.decode_latents(&lat)
    }

    /// Image-conditioned instruction features for the edit path: preprocess each reference, run the
    /// (lazily-loaded, f32) Qwen3-VL vision tower **per reference** (separately — no cross-image
    /// attention in the ViT, matching Qwen3-VL's per-image encoding), build the chat template with one
    /// `<|image_pad|>` block per reference, and run the multi-image MLLM forward. Returns
    /// `(features [1, L, 4096], mask [1, L])` — each reference's `<|image_pad|>` block carries that
    /// reference's merged vision embeds + deepstack injections.
    fn encode_image_instruction(
        &self,
        references: &[&Image],
        instruction: &str,
    ) -> Result<(Array, Array)> {
        self.ensure_vision()?;
        let vision = self.vision.borrow();
        let tower = vision
            .as_ref()
            .expect("vision tower loaded by ensure_vision");

        let mut image_embeds = Vec::with_capacity(references.len());
        let mut deepstacks = Vec::with_capacity(references.len());
        let mut grids = Vec::with_capacity(references.len());
        let mut counts = Vec::with_capacity(references.len());
        for r in references {
            // Reference → Qwen3-VL preprocessing (its own smart-resize / grid) → vision tower.
            let rgb = image::RgbImage::from_raw(r.width, r.height, r.pixels.clone()).ok_or_else(
                || Error::Msg("boogu edit: reference pixels != width·height·3".into()),
            )?;
            let (pixel_values, grid) = preprocess_image(&rgb)?;
            let (embeds, deepstack) = tower.forward(&pixel_values, &[grid])?;
            counts.push(embeds.shape()[0] as usize);
            image_embeds.push(embeds);
            deepstacks.push(deepstack);
            grids.push(grid);
        }

        // Chat template with one block of merged vision tokens (`<|image_pad|>`) per reference, then the
        // multi-image MLLM forward (per-block vision splice + 3-D MRoPE + deepstack injection).
        let (ids, mask) = self.tok.encode_edit_with_images(instruction, &counts)?;
        let feats = self.te.last_hidden_with_images(
            &ids,
            &mask,
            &image_embeds,
            &deepstacks,
            &grids,
            IMAGE_TOKEN_ID,
        )?;
        Ok((feats, mask))
    }

    /// Ensure the Qwen3-VL vision tower is loaded (lazy, cached). Loaded from the snapshot's `mllm/`
    /// (f32 — see [`load_vision_tower`]) only on the first image-conditioned edit, so the T2I paths
    /// never pay for it.
    fn ensure_vision(&self) -> Result<()> {
        let needs_load = self.vision.borrow().is_none();
        if needs_load {
            let tower = load_vision_tower(&self.root)?;
            *self.vision.borrow_mut() = Some(tower);
        }
        Ok(())
    }

    /// VAE-decode a final latent `[1, 16, H/8, W/8]` → RGB8 image. z-image `Vae::decode`
    /// de-normalizes (`z/scaling + shift`) internally, so the raw post-denoise latent is passed.
    fn decode_latents(&self, lat: &Array) -> Result<Image> {
        let decoded = self.vae.decode(lat)?.as_dtype(Dtype::Float32)?; // [1,3,1,H,W]
        decoded_to_image(&decoded)
    }

    /// Quantize the two large weight stacks — the DiT (~20.6 GB bf16) and the Qwen3-VL TE (~17.5 GB
    /// bf16) — to Q4/Q8 in place (E8 / memory). The FLUX.1 VAE (0.34 GB, decode-precision-sensitive)
    /// and the lazily-loaded f32 vision tower (E7b-1 parity finding) stay dense. The same group-wise
    /// affine packing the on-disk converter ([`crate::convert::quantize_transformer`]) writes.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.dit.quantize(bits)?;
        self.te.quantize(bits)?;
        Ok(())
    }
}

/// Seeded initial/renoise latent noise `[1, 16, H/8, W/8]` (f32). `step` derives a distinct RNG key
/// per renoise so successive renoise draws differ (mirroring the reference's advancing generator).
fn init_noise(height: u32, width: u32, seed: u64, step: u64) -> Result<Array> {
    let (hl, wl) = ((height / 8) as i32, (width / 8) as i32);
    let key = random::key(seed.wrapping_add(step))?;
    Ok(random::normal::<f32>(
        &[1, 16, hl, wl],
        None,
        None,
        Some(&key),
    )?)
}

/// Convert an RGB8 [`Image`] (NHWC, `[0, 255]`) into the VAE encoder's expected `[1, 3, H, W]` f32
/// tensor in `[-1, 1]` — the inverse of [`decoded_to_image`]'s `x·0.5 + 0.5` denormalize.
fn image_to_pixels(img: &Image) -> Result<Array> {
    let (h, w) = (img.height as i32, img.width as i32);
    // Reject a buffer that violates the `w·h·3` Image invariant before `Array::from_slice` panics on
    // the shape mismatch — mirrors ideogram's `image_to_pixels` guard (pipeline.rs ~109) (F-020/L-A).
    let expected = (img.height as usize) * (img.width as usize) * 3;
    if img.pixels.len() != expected {
        return Err(Error::Msg(format!(
            "boogu: reference pixel buffer {} bytes != {}x{}x3 ({expected})",
            img.pixels.len(),
            img.width,
            img.height
        )));
    }
    let f: Vec<f32> = img
        .pixels
        .iter()
        .map(|&p| (p as f32 / 255.0) * 2.0 - 1.0)
        .collect();
    let nhwc = Array::from_slice(&f, &[1, h, w, 3]);
    Ok(nhwc.transpose_axes(&[0, 3, 1, 2])?)
}

/// DMD sigma schedule: `linspace(conditioning_sigma, 1.0, steps+1)[:-1]` — `steps` ascending values
/// from `conditioning_sigma` toward (but excluding) `1.0`. These are **clean-fraction** sigmas.
fn dmd_sigmas(conditioning_sigma: f32, steps: usize) -> Vec<f32> {
    let span = 1.0 - conditioning_sigma;
    (0..steps)
        .map(|k| conditioning_sigma + span * (k as f32) / (steps as f32))
        .collect()
}

/// The Turbo DMD grid as the curated framework's **noise-fraction** schedule: `σ_i = 1 − c_i` for each
/// clean-fraction [`dmd_sigmas`] entry (descending), plus the trailing `0.0` the curated solvers
/// integrate toward. `run_flow_sampler` feeds `1 − σ = c_i` (the clean-fraction) back to the DiT
/// (OneMinusSigma), so each curated step's x0 estimate matches the native DMD loop's; the curated
/// solver then supplies the renoise. The final node `σ = 0` is the last native x0 estimate (the DMD
/// loop's last step never renoises), so a consistency solver lands on the same terminal prediction.
fn turbo_native_sigmas(conditioning_sigma: f32, steps: usize) -> Vec<f32> {
    let mut s: Vec<f32> = dmd_sigmas(conditioning_sigma, steps)
        .iter()
        .map(|&c| 1.0 - c)
        .collect();
    s.push(0.0);
    s
}

/// The Base/Edit static-shift `mu` — `lin_mu(SEQ_LEN) = 1.15` for the saturated `seq_len = 4096` (the
/// snapshot's `time_shift_version="v1"` static config). Fed to the epic 7114 scheduler axis so a curated
/// `normal` / `sgm_uniform` / … schedule re-shapes σ over the SAME shift the native schedule uses.
fn base_shift_mu() -> f32 {
    lin_mu(SEQ_LEN) as f32
}

/// The Base/Edit native sigma schedule (noise-fraction, descending to a trailing `0.0`) — the
/// `OneMinusSigma` view of [`build_timesteps_v1`]'s shifted clean-fraction timesteps: `σ_i = 1 − ts_i`.
/// `run_flow_sampler` feeds `1 − σ = ts_i` back to the DiT, so the unified Euler default is byte-exact
/// with the legacy `x += (ts_{i+1} − ts_i)·v` loop (the trailing `ts = 1.0` becomes the terminal `σ = 0`).
fn base_native_sigmas(steps: usize) -> Vec<f32> {
    build_timesteps_v1(steps)
        .iter()
        .map(|&t| 1.0 - t as f32)
        .collect()
}

/// Build the static-v1 shifted timestep schedule plus the trailing `1.0`.
///
/// Returns a `Vec<f64>` of length `steps + 1`: the `steps` shifted samples of
/// `linspace(0,1,steps+1)[:-1]` followed by `1.0` (so `ts[i+1]` is always valid in the Euler step).
fn build_timesteps_v1(steps: usize) -> Vec<f64> {
    let mu = lin_mu(SEQ_LEN);
    let mut ts: Vec<f64> = (0..steps)
        .map(|i| time_shift_v1(i as f64 / steps as f64, mu))
        .collect();
    ts.push(1.0);
    ts
}

/// Reference `_get_lin_function(x1=256,y1=base_shift,x2=4096,y2=max_shift)(seq_len)` → `mu`.
fn lin_mu(seq_len: f64) -> f64 {
    let (x1, y1, x2, y2) = (256.0, BASE_SHIFT, 4096.0, MAX_SHIFT);
    let m = (y2 - y1) / (x2 - x1);
    let b = y1 - m * x1;
    m * seq_len + b
}

/// Reference `_time_shift_v1(t, mu, sigma=1.0)`: `t1=1−t` (clipped); `y = e^mu / (e^mu + (1/t1 − 1))`;
/// return `1 − y`.
fn time_shift_v1(t: f64, mu: f64) -> f64 {
    let eps = 1e-8;
    let t1 = (1.0 - t).clamp(eps, 1.0 - eps);
    let num = mu.exp();
    let denom = num + (1.0 / t1 - 1.0);
    1.0 - num / denom
}
