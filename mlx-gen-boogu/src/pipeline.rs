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
//!
//! ## Component residency (epic 10834, sc-10840)
//!
//! The assembled model splits into two component bundles so a `Sequential`-offload generate can drop
//! the ~17.5 GB Qwen3-VL `mllm/` encoder before the ~20.6 GB DiT loads:
//!
//! * [`BooguEncoders`] — the phase-A **mllm** (Qwen3-VL condition encoder + its lazily-loaded vision
//!   tower). Turns a prompt/instruction (+ optional reference images) into DiT conditioning; dropped
//!   first under `Sequential`.
//! * [`BooguHeavy`] — the phase-B **render** bundle (mixed-stream DiT + FLUX.1 16-ch VAE). Loaded after
//!   the mllm is dropped; also VAE-encodes reference/init images (edit / img2img) and decodes the final
//!   latent — so the reference *latents* are produced in the render phase (like qwen-image's
//!   `encode_init_latents`, F-118), and only the mllm-derived conditioning has to persist across the
//!   drop.
//!
//! [`BooguPipeline`] holds both bundles resident and is the byte-exact warm path the real-weight tests
//! drive directly; the [`crate::model`] generator wires the two bundles onto the shared
//! [`mlx_gen::Residency`] seam.

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use mlx_gen::image::{decoded_to_image, validate_multiple_of_16};
use mlx_gen::img2img::{add_noise_by_interpolation, preprocess_init_image};
use mlx_gen::media::Image;
use mlx_gen::{
    resolve_flow_schedule, run_flow_sampler, CancelFlag, Conditioning, Error, GenerationRequest,
    LatentDecoder, Progress, Result, TimestepConvention,
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

/// Edit (single-reference text+image-to-image) generation knobs. The output resolution is
/// `height`/`width`; the reference image's own dimensions drive the reference latent (both must be
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

/// The phase-A **mllm** encoder bundle (epic 10834, sc-10840) — the Qwen3-VL condition encoder (`mllm/`
/// text tower, ~17.5 GB bf16) plus its lazily-loaded (f32) vision tower for image-conditioned editing.
/// Everything the mllm needs to turn a prompt/instruction (+ optional reference images) into DiT
/// conditioning lives here; nothing it produces references the DiT/VAE, so the whole bundle can be
/// dropped before the render bundle loads (`Sequential` offload).
pub(crate) struct BooguEncoders {
    te: BooguTextEncoder,
    /// Snapshot root — kept so the vision tower can be lazily loaded from `mllm/` on first edit.
    root: PathBuf,
    /// Lazily-loaded (f32) Qwen3-VL vision tower; `None` until the first image-conditioned edit.
    vision: RefCell<Option<VisionTower>>,
}

/// The phase-B **render** bundle: the mixed-stream DiT (~20.6 GB bf16) + the FLUX.1 16-ch VAE. Loaded
/// AFTER the mllm is dropped under `Sequential`, bounding peak unified memory to `max(mllm, DiT+VAE)`
/// instead of their sum. The VAE both encodes reference/init images (edit / img2img — [`Self::
/// encode_ref_latents`] / [`Self::encode_init_clean`]) and decodes the final latent, so it lives with
/// the DiT (never the mllm): the reference latents are produced in the render phase, and only the
/// mllm-derived conditioning has to persist across the drop.
pub(crate) struct BooguHeavy {
    dit: BooguTransformer,
    vae: Vae,
}

/// The materialized mllm conditioning for the true-CFG Base and instruction-Edit paths: the positive
/// instruction hidden states + mask, and (when CFG is active) the empty/drop-instruction uncond twin.
/// Produced in phase A; under `Sequential` [`Self::materialize`] `eval`s it before the mllm is dropped
/// so no un-evaluated output keeps the ~17.5 GB encoder referenced through MLX's lazy graph.
pub(crate) struct BooguBaseCond {
    cond: Array,
    cond_mask: Array,
    uncond: Option<(Array, Array)>,
}

impl BooguBaseCond {
    /// Force-evaluate the conditioning tensors (cond + mask + optional uncond twin) while the encoder
    /// is still alive — MLX is lazy, so an un-evaluated output would keep the encoder referenced and
    /// the `Sequential` drop would free nothing. No-op-equivalent under `Resident` (never called).
    pub(crate) fn materialize(&self) -> Result<()> {
        let mut arrays = vec![&self.cond, &self.cond_mask];
        if let Some((hidden, mask)) = &self.uncond {
            arrays.push(hidden);
            arrays.push(mask);
        }
        eval(arrays)?;
        Ok(())
    }
}

impl BooguEncoders {
    /// Load the Qwen3-VL condition encoder from a snapshot's `mllm/`. The vision tower (edit-only)
    /// loads lazily on the first image-conditioned edit, so the text-to-image paths keep their footprint.
    pub(crate) fn load(root: &Path) -> Result<Self> {
        Ok(Self {
            te: load_text_encoder(root)?,
            root: root.to_path_buf(),
            vision: RefCell::new(None),
        })
    }

    /// Quantize the Qwen3-VL text tower (Q4/Q8) in place — the mllm half of the E8 quant scope. The
    /// lazily-loaded f32 vision tower stays dense (E7b-1 parity finding). No-op on an already-packed base.
    pub(crate) fn quantize(&mut self, bits: i32) -> Result<()> {
        self.te.quantize(bits)
    }

    /// Base / img2img true-CFG conditioning: the positive instruction, plus (when `guidance > 1`) the
    /// empty/drop CFG-negative instruction. Byte-identical to the pre-split inline encode.
    pub(crate) fn encode_base(
        &self,
        tok: &BooguTokenizer,
        prompt: &str,
        guidance: f32,
    ) -> Result<BooguBaseCond> {
        let (cond_ids, cond_mask) = tok.encode_t2i(prompt)?;
        let cond = self.te.last_hidden(&cond_ids, &cond_mask)?;
        let uncond = if guidance > 1.0 {
            let (u_ids, u_mask) = tok.encode_negative()?;
            Some((self.te.last_hidden(&u_ids, &u_mask)?, u_mask))
        } else {
            None
        };
        Ok(BooguBaseCond {
            cond,
            cond_mask,
            uncond,
        })
    }

    /// Turbo (CFG-free DMD student) conditioning: the positive instruction only (no unconditional
    /// branch — the guided velocity is distilled into the weights).
    pub(crate) fn encode_turbo(
        &self,
        tok: &BooguTokenizer,
        prompt: &str,
    ) -> Result<(Array, Array)> {
        let (ids, mask) = tok.encode_t2i(prompt)?;
        let cond = self.te.last_hidden(&ids, &mask)?;
        Ok((cond, mask))
    }

    /// Instruction-Edit conditioning (true-CFG). Faithful edit (`condition_on_image`) runs the
    /// references through the Qwen3-VL vision tower so the MLLM sees them (image-conditioned instruction
    /// features); otherwise the instruction is encoded text-only. The CFG-negative is the empty/drop
    /// instruction, optionally image-conditioned (`use_input_images_4_neg_instruct`).
    pub(crate) fn encode_edit(
        &self,
        tok: &BooguTokenizer,
        references: &[Image],
        instruction: &str,
        opts: &EditOptions,
    ) -> Result<BooguBaseCond> {
        let (cond, cond_mask) = if opts.condition_on_image {
            self.encode_image_instruction(tok, references, instruction)?
        } else {
            let (ids, mask) = tok.encode_edit(instruction)?;
            (self.te.last_hidden(&ids, &mask)?, mask)
        };
        let uncond = if opts.text_guidance_scale > 1.0 {
            if opts.condition_on_image && opts.use_input_images_4_neg_instruct {
                Some(self.encode_image_instruction(tok, references, "")?)
            } else {
                let (u_ids, u_mask) = tok.encode_negative()?;
                Some((self.te.last_hidden(&u_ids, &u_mask)?, u_mask))
            }
        } else {
            None
        };
        Ok(BooguBaseCond {
            cond,
            cond_mask,
            uncond,
        })
    }

    /// Image-conditioned instruction features for the edit path: preprocess each reference, run the
    /// (lazily-loaded, f32) Qwen3-VL vision tower over each, build the chat template with one image
    /// block per reference, and run the multi-image-conditioned MLLM forward. Returns
    /// `(features [1, L, 4096], mask [1, L])` — the same `(hidden, mask)` shape the DiT
    /// `forward_edit_multi` consumes, but with each `<|image_pad|>` run carrying its reference's merged
    /// embeds + deepstack injections. A single reference is the `references.len() == 1` case.
    fn encode_image_instruction(
        &self,
        tok: &BooguTokenizer,
        references: &[Image],
        instruction: &str,
    ) -> Result<(Array, Array)> {
        self.ensure_vision()?;
        let vision = self.vision.borrow();
        let tower = vision
            .as_ref()
            .expect("vision tower loaded by ensure_vision");

        // Each reference → Qwen3-VL preprocessing (its own smart-resize / grid) → vision tower; collect
        // per-image embeds / deepstack / grid + the merged-token count that drives the chat template.
        let mut image_embeds: Vec<Array> = Vec::with_capacity(references.len());
        let mut deepstacks: Vec<Vec<Array>> = Vec::with_capacity(references.len());
        let mut grids: Vec<[i32; 3]> = Vec::with_capacity(references.len());
        let mut counts: Vec<usize> = Vec::with_capacity(references.len());
        for r in references {
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

        // Chat template with one `<|image_pad|>` block per reference, then the multi-image-conditioned
        // MLLM forward (per-image vision splice + 3-D MRoPE advancing per image + deepstack injection).
        let (ids, mask) = tok.encode_edit_with_images(instruction, &counts)?;
        let feats = self.te.last_hidden_with_image_multi(
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
}

impl BooguHeavy {
    /// Load the DiT + FLUX.1 VAE from a standard Boogu snapshot (`transformer/`, `vae/`).
    pub(crate) fn load(root: &Path) -> Result<Self> {
        Ok(Self {
            dit: load_transformer(root)?,
            vae: load_vae(root)?,
        })
    }

    /// Quantize the DiT (~20.6 GB bf16) to Q4/Q8 in place — the render half of the E8 quant scope. The
    /// FLUX.1 VAE (decode-precision-sensitive) stays dense. No-op on an already-packed base.
    pub(crate) fn quantize(&mut self, bits: i32) -> Result<()> {
        self.dit.quantize(bits)
    }

    /// VAE-encode a single init/reference image into a clean latent `[1, 16, H/8, W/8]` at the output
    /// resolution (img2img latent-init). `preprocess_init_image` LANCZOS-resizes to W×H and normalizes
    /// to `[-1,1]` NCHW. Seed-independent — the caller hoists it once across the `count` loop.
    pub(crate) fn encode_init_clean(&self, init: &Image, width: u32, height: u32) -> Result<Array> {
        self.vae
            .encode(&preprocess_init_image(init, width, height)?)
    }

    /// VAE-encode each edit reference into its own clean latent `[1, 16, rH/8, rW/8]` (packed into the
    /// DiT image sequence by `forward_edit_multi`). Validates the `1..` count + per-reference
    /// multiple-of-16 dims. Seed-independent — hoisted once across the `count` loop.
    pub(crate) fn encode_ref_latents(&self, references: &[Image]) -> Result<Vec<Array>> {
        if references.is_empty() {
            return Err(Error::Msg(
                "boogu edit: at least one reference image is required".into(),
            ));
        }
        references
            .iter()
            .map(|r| {
                validate_multiple_of_16(r.width, r.height, "boogu")?;
                self.vae.encode(&image_to_pixels(r)?)
            })
            .collect()
    }

    /// True-CFG flow velocity for the Base / img2img paths: `pred = cond + (scale − 1)·(cond − uncond)`
    /// (or `cond` alone when CFG is off), negated into `run_flow_sampler`'s noise-fraction FLOW
    /// convention and cast to f32. Shared by [`Self::render_base_t2i`] and [`Self::render_base_img2img`].
    fn base_velocity(
        &self,
        x: &Array,
        timestep: f32,
        c: &BooguBaseCond,
        scale: f32,
    ) -> Result<Array> {
        let t = Array::from_slice(&[timestep], &[1]);
        let cond_v = self.dit.forward(x, &t, &c.cond, &c.cond_mask)?;
        let pred = match &c.uncond {
            Some((u_hidden, u_mask)) => {
                let uncond_v = self.dit.forward(x, &t, u_hidden, u_mask)?;
                add(
                    &cond_v,
                    &multiply(&subtract(&cond_v, &uncond_v)?, Array::from_f32(scale - 1.0))?,
                )?
            }
            None => cond_v,
        };
        Ok(multiply(
            &pred.as_dtype(Dtype::Float32)?,
            Array::from_f32(-1.0),
        )?)
    }

    /// True-CFG flow velocity for the Edit path — identical to [`Self::base_velocity`] but with the
    /// reference latents threaded through the DiT's `forward_edit_multi` (`[ref₀; …; ref_{N-1}; noise]`).
    fn edit_velocity(
        &self,
        x: &Array,
        ref_latents: &[Array],
        timestep: f32,
        c: &BooguBaseCond,
        scale: f32,
    ) -> Result<Array> {
        let t = Array::from_slice(&[timestep], &[1]);
        let cond_v = self
            .dit
            .forward_edit_multi(x, ref_latents, &t, &c.cond, &c.cond_mask)?;
        let pred = match &c.uncond {
            Some((u_hidden, u_mask)) => {
                let uncond_v = self
                    .dit
                    .forward_edit_multi(x, ref_latents, &t, u_hidden, u_mask)?;
                add(
                    &cond_v,
                    &multiply(&subtract(&cond_v, &uncond_v)?, Array::from_f32(scale - 1.0))?,
                )?
            }
            None => cond_v,
        };
        Ok(multiply(
            &pred.as_dtype(Dtype::Float32)?,
            Array::from_f32(-1.0),
        )?)
    }

    /// CFG-free flow velocity for the Turbo DMD student: one DiT forward, negated into the FLOW
    /// convention and cast to f32.
    fn turbo_velocity(
        &self,
        x: &Array,
        timestep: f32,
        cond: &Array,
        mask: &Array,
    ) -> Result<Array> {
        let t = Array::from_slice(&[timestep], &[1]);
        let v = self.dit.forward(x, &t, cond, mask)?;
        Ok(multiply(
            &v.as_dtype(Dtype::Float32)?,
            Array::from_f32(-1.0),
        )?)
    }

    /// Render one Base (true-CFG) image from pure noise. `sigmas` is the (possibly `from_ldm`-truncated)
    /// schedule; `c` is the phase-A conditioning; the per-image seed lives on `opts.seed`.
    pub(crate) fn render_base_t2i(
        &self,
        c: &BooguBaseCond,
        opts: &GenerateOptions,
        sigmas: &[f32],
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "boogu")?;
        let lat = init_noise(opts.height, opts.width, opts.seed, 0)?;
        let scale = opts.text_guidance_scale;
        let lat = run_flow_sampler(
            opts.sampler.as_deref(),
            TimestepConvention::OneMinusSigma,
            sigmas,
            lat,
            opts.seed,
            cancel,
            on_progress,
            |x, timestep| self.base_velocity(x, timestep, c, scale),
        )?;
        on_progress(Progress::Decoding);
        self.decode_latents(&lat, decoder)
    }

    /// Render one Base (true-CFG) **img2img** image: seed the denoise from the (pre-encoded) `clean`
    /// reference latent blended with noise at `σ_k = sigmas[start_step]` and run the true-CFG Euler over
    /// the tail `sigmas[start..]`. `clean` is hoisted by the caller (seed-independent). Byte-identical to
    /// the pre-split path — only the VAE-encode of the reference moved out to [`Self::encode_init_clean`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render_base_img2img(
        &self,
        c: &BooguBaseCond,
        clean: &Array,
        start_step: usize,
        opts: &GenerateOptions,
        sigmas: &[f32],
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "boogu")?;
        let noise = init_noise(opts.height, opts.width, opts.seed, 0)?;
        let start = start_step.min(sigmas.len().saturating_sub(1));
        let lat = add_noise_by_interpolation(clean, &noise, sigmas[start])?;
        let denoise_sigmas = &sigmas[start..];
        let scale = opts.text_guidance_scale;
        let lat = run_flow_sampler(
            opts.sampler.as_deref(),
            TimestepConvention::OneMinusSigma,
            denoise_sigmas,
            lat,
            opts.seed,
            cancel,
            on_progress,
            |x, timestep| self.base_velocity(x, timestep, c, scale),
        )?;
        on_progress(Progress::Decoding);
        self.decode_latents(&lat, decoder)
    }

    /// Render one Turbo (DMD few-step, CFG-free) image from pure noise. A selected sampler/scheduler
    /// routes through the curated unified framework over the DMD σ grid; unset is the native DMD loop.
    pub(crate) fn render_turbo_t2i(
        &self,
        cond: &Array,
        mask: &Array,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "boogu")?;

        if turbo_uses_curated(opts.sampler.as_deref(), opts.scheduler.as_deref()) {
            // F-093 / sc-11122: a scheduler-only request defaults the sampler to `lcm` so the curated
            // branch never resolves to the excluded Euler on the DMD student.
            let sampler = turbo_curated_sampler(opts.sampler.as_deref());
            let lat = init_noise(opts.height, opts.width, opts.seed, 0)?;
            let native = turbo_native_sigmas(opts.conditioning_sigma, opts.steps);
            let sigmas = resolve_flow_schedule(opts.scheduler.as_deref(), 0.0, opts.steps, &native);
            let lat = run_flow_sampler(
                Some(sampler),
                TimestepConvention::OneMinusSigma,
                &sigmas,
                lat,
                opts.seed,
                cancel,
                on_progress,
                |x, timestep| self.turbo_velocity(x, timestep, cond, mask),
            )?;
            on_progress(Progress::Decoding);
            return self.decode_latents(&lat, decoder);
        }

        let lat = init_noise(opts.height, opts.width, opts.seed, 0)?;
        let sigmas = dmd_sigmas(opts.conditioning_sigma, opts.steps);
        let lat = self.denoise_turbo_native(
            cond,
            mask,
            lat,
            &sigmas,
            0,
            opts.height,
            opts.width,
            opts.seed,
            cancel,
            on_progress,
        )?;
        on_progress(Progress::Decoding);
        self.decode_latents(&lat, decoder)
    }

    /// Render one Turbo **img2img** image: seed the few-step DMD denoise from the (pre-encoded) `clean`
    /// reference latent noise-blended at the start σ. Curated + native branches mirror the t2i path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render_turbo_img2img(
        &self,
        cond: &Array,
        mask: &Array,
        clean: &Array,
        start_step: usize,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "boogu")?;

        let noise = init_noise(opts.height, opts.width, opts.seed, 0)?;

        if turbo_uses_curated(opts.sampler.as_deref(), opts.scheduler.as_deref()) {
            let sampler = turbo_curated_sampler(opts.sampler.as_deref());
            let native = turbo_native_sigmas(opts.conditioning_sigma, opts.steps);
            let full = resolve_flow_schedule(opts.scheduler.as_deref(), 0.0, opts.steps, &native);
            let start = start_step.min(full.len().saturating_sub(1));
            let lat = add_noise_by_interpolation(clean, &noise, full[start])?;
            let sigmas = &full[start..];
            let lat = run_flow_sampler(
                Some(sampler),
                TimestepConvention::OneMinusSigma,
                sigmas,
                lat,
                opts.seed,
                cancel,
                on_progress,
                |x, timestep| self.turbo_velocity(x, timestep, cond, mask),
            )?;
            on_progress(Progress::Decoding);
            return self.decode_latents(&lat, decoder);
        }

        // Native DMD student loop seeded from the noise-blended reference (see the pre-split note: the
        // native clean-fraction `c_k` → noise-fraction `1 − c_k` matches the curated blend byte-for-byte).
        let sigmas = dmd_sigmas(opts.conditioning_sigma, opts.steps);
        let start = start_step.min(sigmas.len().saturating_sub(1));
        let lat = add_noise_by_interpolation(clean, &noise, 1.0 - sigmas[start])?;
        let lat = self.denoise_turbo_native(
            cond,
            mask,
            lat,
            &sigmas,
            start,
            opts.height,
            opts.width,
            opts.seed,
            cancel,
            on_progress,
        )?;
        on_progress(Progress::Decoding);
        self.decode_latents(&lat, decoder)
    }

    /// Render one Edit (true-CFG) image: flow-match denoise from pure noise with the (pre-encoded)
    /// reference latents packed into the DiT image sequence. `c` is the phase-A (optionally
    /// image-conditioned) instruction conditioning; `ref_latents` is hoisted by the caller.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render_edit(
        &self,
        c: &BooguBaseCond,
        ref_latents: &[Array],
        opts: &EditOptions,
        sigmas: &[f32],
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "boogu")?;
        let lat = init_noise(opts.height, opts.width, opts.seed, 0)?;
        let scale = opts.text_guidance_scale;
        let lat = run_flow_sampler(
            opts.sampler.as_deref(),
            TimestepConvention::OneMinusSigma,
            sigmas,
            lat,
            opts.seed,
            cancel,
            on_progress,
            |x, timestep| self.edit_velocity(x, ref_latents, timestep, c, scale),
        )?;
        on_progress(Progress::Decoding);
        self.decode_latents(&lat, decoder)
    }

    /// The native DMD student few-step loop, shared by the Turbo t2i and img2img paths so the two never
    /// diverge (F-093 / sc-11122). Runs `sigmas[start..]` (clean-fraction ascending): each step predicts
    /// the clean estimate `x += (1 − σ)·v` (f32), then — except on the final entry — renoises to the next
    /// level `x = (1 − σ_next)·noise + σ_next·x` with fresh per-step noise. `lat` is the already-seeded
    /// start latent; progress streams over the `sigmas.len() − start` steps, cancellation per step.
    #[allow(clippy::too_many_arguments)]
    fn denoise_turbo_native(
        &self,
        cond: &Array,
        mask: &Array,
        mut lat: Array,
        sigmas: &[f32],
        start: usize,
        height: u32,
        width: u32,
        seed: u64,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let total = (sigmas.len() - start) as u32;
        for i in start..sigmas.len() {
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let sigma = sigmas[i];
            let t = Array::from_slice(&[sigma], &[1]);
            let pred = self.dit.forward(&lat, &t, cond, mask)?;
            // Predict (clean estimate): x += (1 − sigma)·v, in f32.
            lat = add(
                &lat.as_dtype(Dtype::Float32)?,
                &multiply(
                    &pred.as_dtype(Dtype::Float32)?,
                    Array::from_f32(1.0 - sigma),
                )?,
            )?;
            // Renoise to the next sigma level with fresh noise (all but the final step).
            if i + 1 < sigmas.len() {
                let sigma_next = sigmas[i + 1];
                let noise = init_noise(height, width, seed, (i + 1) as u64)?;
                lat = add(
                    &multiply(&noise, Array::from_f32(1.0 - sigma_next))?,
                    &multiply(&lat, Array::from_f32(sigma_next))?,
                )?;
            }
            eval([&lat])?;
            on_progress(Progress::Step {
                current: (i - start + 1) as u32,
                total,
            });
        }
        Ok(lat)
    }

    /// VAE-decode (or PiD-decode) a final latent `[1, 16, H/8, W/8]` → RGB8 image. z-image `Vae::decode`
    /// de-normalizes (`z/scaling + shift`) internally, so the raw post-denoise latent is passed; PiD
    /// consumes that same normalized latent and additionally 4× super-resolves. `decoder` is `Some` only
    /// when the request set `use_pid` and a PiD overlay was loaded (epic 7840, sc-7846); else native VAE.
    fn decode_latents(&self, lat: &Array, decoder: Option<&dyn LatentDecoder>) -> Result<Image> {
        let decoder: &dyn LatentDecoder = decoder.unwrap_or(&self.vae);
        let decoded = decoder.decode(lat)?.as_dtype(Dtype::Float32)?; // VAE [1,3,1,H,W]; PiD [1,3,4H,4W]
        decoded_to_image(&decoded)
    }
}

/// The assembled Boogu pipeline holding both component bundles resident: tokenizer + [`BooguEncoders`]
/// (Qwen3-VL condition encoder + lazy vision tower) + [`BooguHeavy`] (DiT + FLUX.1 VAE). This is the
/// byte-exact warm path the real-weight tests drive directly; the [`crate::model`] generator stages the
/// same two bundles onto the shared [`mlx_gen::Residency`] seam for `Sequential` offload.
pub struct BooguPipeline {
    tok: BooguTokenizer,
    enc: BooguEncoders,
    heavy: BooguHeavy,
}

impl BooguPipeline {
    /// Load the text-to-image components from a standard Boogu snapshot (`mllm/`, `transformer/`,
    /// `vae/`). The vision tower (edit-only) loads lazily on the first edit.
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        Ok(Self {
            tok: BooguTokenizer::from_snapshot(root)?,
            enc: BooguEncoders::load(root)?,
            heavy: BooguHeavy::load(root)?,
        })
    }

    /// Generate one RGB image from a text prompt. Convenience wrapper over
    /// [`Self::generate_with_progress`] with no cancellation and a no-op progress sink.
    pub fn generate(&self, prompt: &str, opts: &GenerateOptions) -> Result<Image> {
        let sigmas = base_flow_schedule(opts.steps, opts.scheduler.as_deref());
        self.generate_with_progress(prompt, opts, &sigmas, None, &CancelFlag::new(), &mut |_| {})
    }

    /// Generate one RGB image from a text prompt, streaming [`Progress`] and honoring `cancel` at
    /// each denoise step. A pre/mid-flight cancellation returns [`Error::Canceled`]; the per-step
    /// `eval` bounds the lazy MLX graph and lets the cancel check interrupt mid-render.
    ///
    /// The `RES_MIN`/`RES_MAX` (256–2048) resolution range and the multiple-of-`RES_MULTIPLE`
    /// constraint are a **Generator-layer guarantee** — enforced by `model::validate_request` before
    /// this pipeline runs. This entry trusts the already-validated [`GenerateOptions`] dims (sc-6983).
    pub fn generate_with_progress(
        &self,
        prompt: &str,
        opts: &GenerateOptions,
        sigmas: &[f32],
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let cond = self
            .enc
            .encode_base(&self.tok, prompt, opts.text_guidance_scale)?;
        self.heavy
            .render_base_t2i(&cond, opts, sigmas, decoder, cancel, on_progress)
    }

    /// **img2img latent-init on the Base (true-CFG) path** (epic 8588 A4.3, sc-10191). Seed the
    /// flow-match denoise from a VAE-encoded `init` reference instead of pure noise. See
    /// [`BooguHeavy::render_base_img2img`] for the noise-blend detail; `sigmas` is the model layer's
    /// schedule already truncated for the PiD `from_ldm` early-stop (`[..keep]`).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_base_img2img_with_progress(
        &self,
        prompt: &str,
        init: &Image,
        start_step: usize,
        opts: &GenerateOptions,
        sigmas: &[f32],
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let cond = self
            .enc
            .encode_base(&self.tok, prompt, opts.text_guidance_scale)?;
        let clean = self
            .heavy
            .encode_init_clean(init, opts.width, opts.height)?;
        self.heavy.render_base_img2img(
            &cond,
            &clean,
            start_step,
            opts,
            sigmas,
            decoder,
            cancel,
            on_progress,
        )
    }

    /// Generate one RGB image via the **Turbo** DMD student few-step sampler (Boogu-Image-0.1-Turbo).
    /// Pure T2I, **no CFG**. Convenience wrapper over [`Self::generate_turbo_with_progress`].
    pub fn generate_turbo(&self, prompt: &str, opts: &TurboOptions) -> Result<Image> {
        self.generate_turbo_with_progress(prompt, opts, None, &CancelFlag::new(), &mut |_| {})
    }

    /// [`Self::generate_turbo`] with [`Progress`] streaming and per-step cooperative cancellation
    /// ([`Error::Canceled`]).
    pub fn generate_turbo_with_progress(
        &self,
        prompt: &str,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let (cond, mask) = self.enc.encode_turbo(&self.tok, prompt)?;
        self.heavy
            .render_turbo_t2i(&cond, &mask, opts, decoder, cancel, on_progress)
    }

    /// **img2img latent-init on the Turbo (DMD few-step, CFG-free) path** (epic 8588 A4.3, sc-10191).
    /// The distilled analog of [`Self::generate_base_img2img_with_progress`].
    #[allow(clippy::too_many_arguments)]
    pub fn generate_turbo_img2img_with_progress(
        &self,
        prompt: &str,
        init: &Image,
        start_step: usize,
        opts: &TurboOptions,
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let (cond, mask) = self.enc.encode_turbo(&self.tok, prompt)?;
        let clean = self
            .heavy
            .encode_init_clean(init, opts.width, opts.height)?;
        self.heavy.render_turbo_img2img(
            &cond,
            &mask,
            &clean,
            start_step,
            opts,
            decoder,
            cancel,
            on_progress,
        )
    }

    /// Generate one RGB image via the **Edit** path (single-reference convenience over
    /// [`Self::generate_edit_multi`]).
    pub fn generate_edit(
        &self,
        reference: &Image,
        instruction: &str,
        opts: &EditOptions,
    ) -> Result<Image> {
        self.generate_edit_multi(std::slice::from_ref(reference), instruction, opts)
    }

    /// Multi-reference Edit (`N ∈ [1, 5]`): VAE-encode each reference and pack them all into the DiT
    /// image sequence. `N = 1` is [`Self::generate_edit`].
    pub fn generate_edit_multi(
        &self,
        references: &[Image],
        instruction: &str,
        opts: &EditOptions,
    ) -> Result<Image> {
        let sigmas = base_flow_schedule(opts.steps, opts.scheduler.as_deref());
        self.generate_edit_multi_with_progress(
            references,
            instruction,
            opts,
            &sigmas,
            None,
            &CancelFlag::new(),
            &mut |_| {},
        )
    }

    /// Multi-reference Edit with [`Progress`] + cancellation. VAE-encodes each of the `N ∈ [1, 5]`
    /// references into its own clean latent, optionally runs each through the Qwen3-VL vision tower
    /// (faithful semantic path), and flow-match denoises (true-CFG) with all references packed into the
    /// DiT image sequence. `N = 1` matches [`Self::generate_edit`]. The `sigmas` schedule is the model
    /// layer's, already truncated for the PiD `from_ldm` early-stop.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_edit_multi_with_progress(
        &self,
        references: &[Image],
        instruction: &str,
        opts: &EditOptions,
        sigmas: &[f32],
        decoder: Option<&dyn LatentDecoder>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let cond = self
            .enc
            .encode_edit(&self.tok, references, instruction, opts)?;
        let ref_latents = self.heavy.encode_ref_latents(references)?;
        self.heavy.render_edit(
            &cond,
            &ref_latents,
            opts,
            sigmas,
            decoder,
            cancel,
            on_progress,
        )
    }

    /// Quantize the two large weight stacks — the DiT (~20.6 GB bf16) and the Qwen3-VL TE (~17.5 GB
    /// bf16) — to Q4/Q8 in place (E8 / memory). The FLUX.1 VAE (0.34 GB, decode-precision-sensitive)
    /// and the lazily-loaded f32 vision tower (E7b-1 parity finding) stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.heavy.quantize(bits)?;
        self.enc.quantize(bits)?;
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

/// The single img2img reference for the Base/Turbo t2i path (epic 8588 A4.3, sc-10191): at most one
/// [`Conditioning::Reference`] — multiple is an error (Boogu's multi-image path is the Edit
/// checkpoint's `resolve_edit_references`, not img2img) — with its per-reference `strength` falling
/// back to `req.strength`. `None` ⇒ pure txt2img. Mirrors Z-Image's `resolve_reference`.
pub(crate) fn resolve_reference<'a>(
    req: &'a GenerationRequest,
    id: &str,
) -> Result<Option<(&'a Image, Option<f32>)>> {
    let mut reference = None;
    for c in &req.conditioning {
        if let Conditioning::Reference { image, strength } = c {
            if reference.is_some() {
                return Err(Error::Msg(format!(
                    "{id}: multiple reference images are not supported on the t2i path (single img2img \
                     init only; the Edit checkpoint handles multi-image edits)"
                )));
            }
            reference = Some((image, strength.or(req.strength)));
        }
    }
    Ok(reference)
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

/// Whether a Turbo request selects the curated unified-sampler framework (epic 7114): true when a
/// sampler **or** a scheduler name is set. `false` — the default — is the native DMD student loop
/// (predict → flow-renoise), the byte-exact baseline. Shared by the Turbo t2i **and** img2img paths so
/// neither silently falls through to `run_flow_sampler`, whose `None → Euler` mapping is the
/// deterministic solver `descriptor_turbo` deliberately excludes as out-of-regime for the few-step DMD
/// student (sc-7491). This shared predicate is the F-093 / sc-11122 guard: the img2img path used to
/// route the default request straight through Euler.
pub(crate) fn turbo_uses_curated(sampler: Option<&str>, scheduler: Option<&str>) -> bool {
    sampler.is_some() || scheduler.is_some()
}

/// The curated sampler to feed [`run_flow_sampler`] once a request has already routed into the curated
/// branch ([`turbo_uses_curated`]). A scheduler-only request (sampler unset) would otherwise pass `None`
/// to `run_flow_sampler`, whose `None → Euler` mapping is the deterministic solver `descriptor_turbo`
/// deliberately excludes as out-of-regime for the few-step DMD student (sc-7491). Default it instead to
/// `lcm` — the surveyed closest match (the curated few-step default per [`TURBO_SAMPLERS`]) — so the
/// curated branch never falls through to Euler on Turbo. This is the F-093 / sc-11122 fix.
pub(crate) fn turbo_curated_sampler(sampler: Option<&str>) -> &str {
    sampler.unwrap_or(TURBO_CURATED_DEFAULT_SAMPLER)
}

/// The curated few-step default sampler for the Turbo DMD student: `lcm` (sc-7491's surveyed closest
/// match to the re-noised distillation regime). Used when a Turbo request selects a scheduler without a
/// sampler so the curated branch resolves to `lcm` rather than the excluded Euler.
pub(crate) const TURBO_CURATED_DEFAULT_SAMPLER: &str = "lcm";

/// The Base/Edit static-shift `mu` — `lin_mu(SEQ_LEN) = 1.15` for the saturated `seq_len = 4096` (the
/// snapshot's `time_shift_version="v1"` static config). Fed to the epic 7114 scheduler axis so a curated
/// `normal` / `sgm_uniform` / … schedule re-shapes σ over the SAME shift the native schedule uses.
fn base_shift_mu() -> f32 {
    lin_mu(SEQ_LEN) as f32
}

/// The resolved Base/Edit flow-match σ schedule (native static-shift + optional curated re-shape),
/// exposed so the model-layer PiD `from_ldm` early-stop (sc-8048) can mint the decoder at the achieved
/// capture σ and truncate the denoise — the schedule must be known where the decoder is resolved. This
/// is the exact `sigmas` the Base/Edit `run_flow_sampler` runs over (noise-fraction, `vp_frame=false`,
/// so the schedule σ *is* the degrade σ).
pub fn base_flow_schedule(steps: usize, scheduler: Option<&str>) -> Vec<f32> {
    let native = base_native_sigmas(steps);
    resolve_flow_schedule(scheduler, base_shift_mu(), steps, &native)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// F-093 / sc-11122: the default Turbo request (no sampler, no scheduler) must select the native
    /// DMD student loop on BOTH the t2i and img2img paths — never fall through to `run_flow_sampler`,
    /// whose `None → Euler` mapping is the deterministic solver `descriptor_turbo` deliberately
    /// excludes as out-of-regime for the few-step DMD student (sc-7491). A curated sampler OR scheduler
    /// still routes to the unified framework.
    #[test]
    fn turbo_default_request_selects_native_dmd_not_curated_euler() {
        let default = TurboOptions::default();
        assert!(
            default.sampler.is_none() && default.scheduler.is_none(),
            "the Turbo default must be sampler/scheduler-unset"
        );
        // The default (img2img OR t2i) takes the native DMD branch, not the curated Euler runner.
        assert!(!turbo_uses_curated(
            default.sampler.as_deref(),
            default.scheduler.as_deref()
        ));
        // A curated sampler routes to the unified framework.
        assert!(turbo_uses_curated(Some("lcm"), None));
        // A curated scheduler alone (the ComfyUI lcm/sgm_uniform combo half) also routes to it.
        assert!(turbo_uses_curated(None, Some("sgm_uniform")));
        assert!(turbo_uses_curated(
            Some("euler_ancestral"),
            Some("sgm_uniform")
        ));
    }

    /// F-093 / sc-11122 (the adversarial-review follow-up): once a request has routed into the curated
    /// branch, the sampler fed to `run_flow_sampler` must never be `None` — `None → Euler` is the
    /// deterministic solver `descriptor_turbo` deliberately excludes (sc-7491). A **scheduler-only**
    /// Turbo request (sampler unset, scheduler set) is exactly the case that used to fall through to
    /// Euler on both t2i and img2img; it must now default to `lcm`, the surveyed curated few-step
    /// default. This test is the guard against re-introducing the Euler fall-through.
    #[test]
    fn turbo_scheduler_only_request_defaults_to_lcm_not_euler() {
        // The curated default is `lcm` and is an advertised Turbo sampler (not the excluded Euler).
        assert_eq!(TURBO_CURATED_DEFAULT_SAMPLER, "lcm");
        assert!(crate::model::descriptor_turbo()
            .capabilities
            .samplers
            .contains(&TURBO_CURATED_DEFAULT_SAMPLER));

        // Scheduler-only (sampler=None): both the t2i and img2img curated branches resolve the sampler
        // via `turbo_curated_sampler` — it must default to `lcm`, never `None` (→ Euler) and never
        // `euler`/`ddim`/... .
        for scheduler in ["sgm_uniform", "normal", "karras"] {
            assert!(
                turbo_uses_curated(None, Some(scheduler)),
                "a scheduler-only request still routes into the curated framework"
            );
            let resolved = turbo_curated_sampler(None);
            assert_eq!(
                resolved, "lcm",
                "scheduler-only ({scheduler}) must default the sampler to lcm, not fall through to Euler"
            );
            assert_ne!(
                resolved, "euler",
                "must never resolve to the excluded Euler solver"
            );
        }

        // An explicit sampler is preserved (the default only fills an unset sampler).
        assert_eq!(
            turbo_curated_sampler(Some("euler_ancestral")),
            "euler_ancestral"
        );
        assert_eq!(turbo_curated_sampler(Some("dpmpp_sde")), "dpmpp_sde");
    }

    /// The native DMD img2img loop seeds the blended reference at the SAME noise level the curated
    /// path uses, so moving the default request off the excluded Euler solver does not shift the
    /// img2img start point: native clean-fraction `c_k` → noise-fraction `1 − c_k`, which equals the
    /// curated `turbo_native_sigmas[k]` fed to `add_noise_by_interpolation`. F-093 / sc-11122.
    #[test]
    fn turbo_img2img_native_seed_matches_curated_blend() {
        let (cs, steps) = (DEFAULT_TURBO_SIGMA_TEST, 4usize);
        let clean = dmd_sigmas(cs, steps); // native clean-fraction, ascending, len == steps
        let curated = turbo_native_sigmas(cs, steps); // noise-fraction, len == steps + 1 (trailing 0)
        assert_eq!(clean.len(), steps);
        assert_eq!(curated.len(), steps + 1);
        for k in 0..steps {
            assert!(
                (1.0 - clean[k] - curated[k]).abs() < 1e-6,
                "native seed noise-fraction (1 − c_{k}) must equal curated turbo_native_sigmas[{k}]"
            );
        }
    }

    /// The DMD clean-fraction grid is `linspace(conditioning_sigma, 1.0, steps+1)[:-1]` — `steps`
    /// ascending values starting at `conditioning_sigma`, strictly increasing toward (but excluding)
    /// `1.0`. Guards the native img2img loop's `start`-clamp assumption (`sigmas.len() == steps`).
    #[test]
    fn dmd_sigmas_are_ascending_clean_fractions() {
        let steps = 6usize;
        let s = dmd_sigmas(DEFAULT_TURBO_SIGMA_TEST, steps);
        assert_eq!(s.len(), steps);
        assert!((s[0] - DEFAULT_TURBO_SIGMA_TEST).abs() < 1e-6);
        for w in s.windows(2) {
            assert!(w[1] > w[0], "clean-fraction sigmas must ascend");
        }
        assert!(*s.last().unwrap() < 1.0, "the trailing 1.0 is excluded");
    }

    const DEFAULT_TURBO_SIGMA_TEST: f32 = 0.001;
}
