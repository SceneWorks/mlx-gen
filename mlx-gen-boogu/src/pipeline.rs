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
        let (hl, wl) = ((opts.height / 8) as i32, (opts.width / 8) as i32);
        let key = random::key(opts.seed)?;
        let mut lat = random::normal::<f32>(&[1, 16, hl, wl], None, None, Some(&key))?;

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

        // VAE decode (z-image `Vae::decode` de-normalizes z/scaling+shift internally) → RGB8.
        let decoded = self.vae.decode(&lat)?.as_dtype(Dtype::Float32)?; // [1,3,1,H,W]
        decoded_to_image(&decoded)
    }

    /// Quantize the DiT + VAE to Q4/Q8 (E8 / memory).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.dit.quantize(bits)?;
        self.vae.quantize(bits)?;
        Ok(())
    }
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
