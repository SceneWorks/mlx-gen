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
use mlx_rs::{random, Array, Dtype};

use mlx_gen::image::{decoded_to_image, validate_multiple_of_16};
use mlx_gen::media::Image;
use mlx_gen::Result;

use std::path::Path;

use crate::loader::{load_text_encoder, load_transformer, load_vae};
use crate::text_encoder::BooguTextEncoder;
use crate::tokenizer::BooguTokenizer;
use crate::transformer::BooguTransformer;
use mlx_gen_z_image::vae::Vae;

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
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            height: 1024,
            width: 1024,
            steps: 50,
            text_guidance_scale: 4.0,
            seed: 0,
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
}

impl Default for TurboOptions {
    fn default() -> Self {
        Self {
            height: 1024,
            width: 1024,
            steps: 4,
            seed: 0,
            conditioning_sigma: 0.001,
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
}

impl Default for EditOptions {
    fn default() -> Self {
        Self {
            height: 1024,
            width: 1024,
            steps: 50,
            text_guidance_scale: 4.0,
            seed: 0,
        }
    }
}

/// The assembled Boogu Base pipeline: tokenizer + Qwen3-VL condition encoder + DiT + FLUX.1 VAE.
pub struct BooguPipeline {
    tok: BooguTokenizer,
    te: BooguTextEncoder,
    dit: BooguTransformer,
    vae: Vae,
}

impl BooguPipeline {
    /// Load all four components from a standard Boogu snapshot (`mllm/`, `transformer/`, `vae/`).
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        Ok(Self {
            tok: BooguTokenizer::from_snapshot(root)?,
            te: load_text_encoder(root)?,
            dit: load_transformer(root)?,
            vae: load_vae(root)?,
        })
    }

    /// Generate one RGB image from a text prompt.
    pub fn generate(&self, prompt: &str, opts: &GenerateOptions) -> Result<Image> {
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
        let mut lat = init_noise(opts.height, opts.width, opts.seed, 0)?;

        // Static-v1 timesteps + the trailing 1.0 the Euler step reads as `t_next` at the last step.
        let ts = build_timesteps_v1(opts.steps);
        let scale = opts.text_guidance_scale;

        for i in 0..opts.steps {
            let t = Array::from_slice(&[ts[i] as f32], &[1]);
            let cond_v = self.dit.forward(&lat, &t, &cond, &cond_mask)?;
            let pred = match &uncond {
                Some((u_hidden, u_mask)) => {
                    let uncond_v = self.dit.forward(&lat, &t, u_hidden, u_mask)?;
                    // pred = cond + (scale − 1)·(cond − uncond)
                    add(
                        &cond_v,
                        &multiply(&subtract(&cond_v, &uncond_v)?, Array::from_f32(scale - 1.0))?,
                    )?
                }
                None => cond_v,
            };

            // Euler step in f32: x += (t_next − t)·v.
            let dt = (ts[i + 1] - ts[i]) as f32;
            lat = add(
                &lat.as_dtype(Dtype::Float32)?,
                &multiply(&pred.as_dtype(Dtype::Float32)?, Array::from_f32(dt))?,
            )?;
        }

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
        validate_multiple_of_16(opts.width, opts.height, "boogu")?;

        let (ids, mask) = self.tok.encode_t2i(prompt)?;
        let cond = self.te.last_hidden(&ids, &mask)?;

        let mut lat = init_noise(opts.height, opts.width, opts.seed, 0)?;
        let sigmas = dmd_sigmas(opts.conditioning_sigma, opts.steps);

        for i in 0..opts.steps {
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
        }

        self.decode_latents(&lat)
    }

    /// Generate one RGB image via the **Edit** path: VAE-encode a reference image into a clean
    /// reference latent, then flow-match denoise (true-CFG) with that reference packed into the DiT's
    /// image sequence (`forward_edit`). The reference shapes the output spatially; the instruction
    /// drives the edit. Same static-v1 scheduler / true-CFG as [`Self::generate`].
    ///
    /// Scope: this wires the DiT's **spatial** reference path (the story's named scope). Faithful
    /// Boogu edit additionally feeds the reference image through the Qwen3-VL vision tower so the MLLM
    /// "sees" it (image-conditioned instruction features); that semantic path is tracked as E7b.
    pub fn generate_edit(
        &self,
        reference: &Image,
        instruction: &str,
        opts: &EditOptions,
    ) -> Result<Image> {
        validate_multiple_of_16(opts.width, opts.height, "boogu")?;
        validate_multiple_of_16(reference.width, reference.height, "boogu")?;

        // Reference → clean VAE latent [1, 16, rH/8, rW/8].
        let ref_pixels = image_to_pixels(reference);
        let ref_latent = self.vae.encode(&ref_pixels)?;

        // Condition encoding: edit instruction + CFG-negative (empty/drop) instruction. Both DiT
        // passes carry the same reference latent — only the instruction differs (TI2I text guidance).
        let (cond_ids, cond_mask) = self.tok.encode_edit(instruction)?;
        let cond = self.te.last_hidden(&cond_ids, &cond_mask)?;
        let do_cfg = opts.text_guidance_scale > 1.0;
        let uncond = if do_cfg {
            let (u_ids, u_mask) = self.tok.encode_negative()?;
            Some((self.te.last_hidden(&u_ids, &u_mask)?, u_mask))
        } else {
            None
        };

        let mut lat = init_noise(opts.height, opts.width, opts.seed, 0)?;
        let ts = build_timesteps_v1(opts.steps);
        let scale = opts.text_guidance_scale;

        for i in 0..opts.steps {
            let t = Array::from_slice(&[ts[i] as f32], &[1]);
            let cond_v = self
                .dit
                .forward_edit(&lat, &ref_latent, &t, &cond, &cond_mask)?;
            let pred = match &uncond {
                Some((u_hidden, u_mask)) => {
                    let uncond_v =
                        self.dit
                            .forward_edit(&lat, &ref_latent, &t, u_hidden, u_mask)?;
                    add(
                        &cond_v,
                        &multiply(&subtract(&cond_v, &uncond_v)?, Array::from_f32(scale - 1.0))?,
                    )?
                }
                None => cond_v,
            };

            let dt = (ts[i + 1] - ts[i]) as f32;
            lat = add(
                &lat.as_dtype(Dtype::Float32)?,
                &multiply(&pred.as_dtype(Dtype::Float32)?, Array::from_f32(dt))?,
            )?;
        }

        self.decode_latents(&lat)
    }

    /// VAE-decode a final latent `[1, 16, H/8, W/8]` → RGB8 image. z-image `Vae::decode`
    /// de-normalizes (`z/scaling + shift`) internally, so the raw post-denoise latent is passed.
    fn decode_latents(&self, lat: &Array) -> Result<Image> {
        let decoded = self.vae.decode(lat)?.as_dtype(Dtype::Float32)?; // [1,3,1,H,W]
        decoded_to_image(&decoded)
    }

    /// Quantize the DiT + VAE to Q4/Q8 (E8 / memory).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.dit.quantize(bits)?;
        self.vae.quantize(bits)?;
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
fn image_to_pixels(img: &Image) -> Array {
    let (h, w) = (img.height as i32, img.width as i32);
    let f: Vec<f32> = img
        .pixels
        .iter()
        .map(|&p| (p as f32 / 255.0) * 2.0 - 1.0)
        .collect();
    let nhwc = Array::from_slice(&f, &[1, h, w, 3]);
    nhwc.transpose_axes(&[0, 3, 1, 2]).expect("NHWC→NCHW")
}

/// DMD sigma schedule: `linspace(conditioning_sigma, 1.0, steps+1)[:-1]` — `steps` ascending values
/// from `conditioning_sigma` toward (but excluding) `1.0`.
fn dmd_sigmas(conditioning_sigma: f32, steps: usize) -> Vec<f32> {
    let span = 1.0 - conditioning_sigma;
    (0..steps)
        .map(|k| conditioning_sigma + span * (k as f32) / (steps as f32))
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
