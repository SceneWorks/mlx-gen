//! sc-5139: the Bernini planner's flow-matching ViT diffusion head (`DiffLoss_FM` / `SimpleMLPAdaLN`,
//! `bernini/models/diffloss_fm.py`) + its `FlowMatchScheduler` (`bernini/models/scheduler.py`).
//!
//! The planner's MAR loop (sc-5140) samples a target ViT embedding by running this small AdaLN MLP as
//! a flow-matching denoiser conditioned on `c` = the connector's `for_gen` projection of the planner
//! hidden states. Inference-only: `diffusion_batch_mul` (a train-time noise-replication trick) is
//! dropped, and the `eps`/`rest` output split is vestigial here (the net's out channels == in channels
//! == 3584, so `rest` is empty).
//!
//! `SimpleMLPAdaLN`: `input_proj`(in→width) + `time_embed`(GLIDE sinusoidal→MLP) + `cond_embed`(z→width)
//! → `y = t+c` → N adaLN-zero `ResBlock`s → `FinalLayer`. f32 islands: LayerNorm reductions (via
//! `mlx_rs::fast::layer_norm`).

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{add, concatenate_axis, multiply, split, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

const LN_EPS: f32 = 1e-6;

fn linear(w: &Weights, prefix: &str) -> Result<AdaptableLinear> {
    Ok(AdaptableLinear::dense(
        w.require(&format!("{prefix}.weight"))?.clone(),
        Some(w.require(&format!("{prefix}.bias"))?.clone()),
    ))
}

fn require(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?.clone())
}

// ---------------------------------------------------------------------------
// FlowMatchScheduler (inference) — analog of mlx-gen-wan/src/scheduler.rs.
// ---------------------------------------------------------------------------

/// The clip-diff flow-matching scheduler: a shifted linear σ schedule + an Euler velocity step.
/// Host-side `f32` σ/timestep vectors (tiny — `num_inference_steps` entries).
pub struct FlowMatchScheduler {
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    num_train: f32,
    shift: f32,
    sigma_min: f32,
    sigma_max: f32,
    extra_one_step: bool,
}

impl FlowMatchScheduler {
    /// Bernini clip-diff defaults: `shift 2.0`, `extra_one_step true`, `sigma_min 0.003/1.002`,
    /// `sigma_max 1.0`, `num_train_timesteps 1000`.
    pub fn new(shift: f32, extra_one_step: bool) -> Self {
        let mut s = Self {
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            num_train: 1000.0,
            shift,
            sigma_min: 0.003 / 1.002,
            sigma_max: 1.0,
            extra_one_step,
        };
        s.set_timesteps(100);
        s
    }

    /// Build the σ schedule for `steps` inference steps (`denoising_strength = 1.0`): a linspace
    /// `[sigma_start … sigma_min]` (one extra step then dropped when `extra_one_step`), shifted by
    /// `shift·σ / (1 + (shift-1)·σ)`; `timesteps = σ · num_train`.
    pub fn set_timesteps(&mut self, steps: usize) {
        let sigma_start = self.sigma_max; // denoising_strength 1.0 → sigma_min + (max-min)·1 = max
        let n = if self.extra_one_step {
            steps + 1
        } else {
            steps
        };
        let mut sigmas: Vec<f32> = (0..n)
            .map(|i| {
                let frac = if n <= 1 {
                    0.0
                } else {
                    i as f32 / (n as f32 - 1.0)
                };
                sigma_start + (self.sigma_min - sigma_start) * frac
            })
            .collect();
        if self.extra_one_step {
            sigmas.pop(); // [:-1]
        }
        let shift = self.shift;
        for s in &mut sigmas {
            *s = shift * *s / (1.0 + (shift - 1.0) * *s);
        }
        self.timesteps = sigmas.iter().map(|&s| s * self.num_train).collect();
        self.sigmas = sigmas;
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// Euler velocity step at schedule index `i`: `sample + v · (σ_{i+1} − σ_i)` (σ_{last+1} = 0).
    pub fn step(&self, model_output: &Array, i: usize, sample: &Array) -> Result<Array> {
        let sigma = self.sigmas[i];
        let sigma_next = if i + 1 >= self.sigmas.len() {
            0.0
        } else {
            self.sigmas[i + 1]
        };
        add(
            sample,
            &multiply(model_output, Array::from_f32(sigma_next - sigma))?,
        )
        .map_err(Error::from)
    }
}

// ---------------------------------------------------------------------------
// SimpleMLPAdaLN building blocks.
// ---------------------------------------------------------------------------

/// `x*(1+scale) + shift` (DiT adaLN modulation).
fn modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array> {
    let scaled = multiply(x, &add(scale, Array::from_f32(1.0))?)?;
    add(&scaled, shift).map_err(Error::from)
}

/// GLIDE sinusoidal timestep embedding `[N, dim]` (f32): `half = dim/2`,
/// `freqs[k] = exp(-ln(max_period)·k/half)`, `emb = cat(cos(t·freqs), sin(t·freqs))`.
fn timestep_embedding(t: &Array, dim: usize, max_period: f32) -> Result<Array> {
    let half = dim / 2;
    let ln = max_period.ln();
    let freqs: Vec<f32> = (0..half)
        .map(|k| (-ln * k as f32 / half as f32).exp())
        .collect();
    let freqs = Array::from_slice(&freqs, &[1, half as i32]);
    let n = t.shape()[0];
    let t_col = t.as_dtype(Dtype::Float32)?.reshape(&[n, 1])?;
    let args = multiply(&t_col, &freqs)?; // [N, half]
    concatenate_axis(&[&args.cos()?, &args.sin()?], 1).map_err(Error::from)
}

struct TimestepEmbedder {
    mlp0: AdaptableLinear,
    mlp2: AdaptableLinear,
    freq_size: usize,
}

impl TimestepEmbedder {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            mlp0: linear(w, &format!("{prefix}.mlp.0"))?,
            mlp2: linear(w, &format!("{prefix}.mlp.2"))?,
            freq_size: 256,
        })
    }

    /// `mlp(timestep_embedding(t))`, with the sinusoidal embedding cast to `dtype` before the MLP
    /// (the reference `t_freq.to(t.dtype)`).
    fn forward(&self, t: &Array, dtype: Dtype) -> Result<Array> {
        let freq = timestep_embedding(t, self.freq_size, 10000.0)?.as_dtype(dtype)?;
        self.mlp2.forward(&silu(&self.mlp0.forward(&freq)?)?)
    }
}

struct ResBlock {
    in_ln_w: Array,
    in_ln_b: Array,
    mlp0: AdaptableLinear,
    mlp2: AdaptableLinear,
    adaln: AdaptableLinear, // adaLN_modulation.1 (Linear width→3·width); SiLU is applied in forward
}

impl ResBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            in_ln_w: require(w, &format!("{prefix}.in_ln.weight"))?,
            in_ln_b: require(w, &format!("{prefix}.in_ln.bias"))?,
            mlp0: linear(w, &format!("{prefix}.mlp.0"))?,
            mlp2: linear(w, &format!("{prefix}.mlp.2"))?,
            adaln: linear(w, &format!("{prefix}.adaLN_modulation.1"))?,
        })
    }

    /// `shift,scale,gate = adaLN(silu(y))`; `h = mlp(modulate(LN(x), shift, scale))`; `x + gate·h`.
    fn forward(&self, x: &Array, y: &Array) -> Result<Array> {
        let mods = self.adaln.forward(&silu(y)?)?;
        let parts = split(&mods, 3, -1)?; // shift, scale, gate
        let h = layer_norm(x, Some(&self.in_ln_w), Some(&self.in_ln_b), LN_EPS)?;
        let h = modulate(&h, &parts[0], &parts[1])?;
        let h = self.mlp2.forward(&silu(&self.mlp0.forward(&h)?)?)?;
        add(x, &multiply(&parts[2], &h)?).map_err(Error::from)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.mlp0.quantize(bits, None)?;
        self.mlp2.quantize(bits, None)?;
        self.adaln.quantize(bits, None)
    }
}

struct FinalLayer {
    linear: AdaptableLinear,
    adaln: AdaptableLinear, // adaLN_modulation.1 (Linear width→2·width)
}

impl FinalLayer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear: linear(w, &format!("{prefix}.linear"))?,
            adaln: linear(w, &format!("{prefix}.adaLN_modulation.1"))?,
        })
    }

    /// `shift,scale = adaLN(silu(c))`; `linear(modulate(LN_noaffine(x), shift, scale))`.
    fn forward(&self, x: &Array, c: &Array) -> Result<Array> {
        let mods = self.adaln.forward(&silu(c)?)?;
        let parts = split(&mods, 2, -1)?; // shift, scale
        let h = layer_norm(x, None, None, LN_EPS)?; // norm_final: elementwise_affine=False
        let h = modulate(&h, &parts[0], &parts[1])?;
        self.linear.forward(&h)
    }
}

/// `SimpleMLPAdaLN`: the clip-diff denoiser network.
struct SimpleMlpAdaLn {
    time_embed: TimestepEmbedder,
    cond_embed: AdaptableLinear,
    input_proj: AdaptableLinear,
    res_blocks: Vec<ResBlock>,
    final_layer: FinalLayer,
}

impl SimpleMlpAdaLn {
    fn from_weights(w: &Weights, prefix: &str, depth: usize) -> Result<Self> {
        let res_blocks = (0..depth)
            .map(|i| ResBlock::from_weights(w, &format!("{prefix}.res_blocks.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            time_embed: TimestepEmbedder::from_weights(w, &format!("{prefix}.time_embed"))?,
            cond_embed: linear(w, &format!("{prefix}.cond_embed"))?,
            input_proj: linear(w, &format!("{prefix}.input_proj"))?,
            res_blocks,
            final_layer: FinalLayer::from_weights(w, &format!("{prefix}.final_layer"))?,
        })
    }

    /// `x` `[N, in]`, `t` `[N]` (or `[1]`), `c` `[N, z]` → `[N, out]`.
    fn forward(&self, x: &Array, t: &Array, c: &Array) -> Result<Array> {
        let mut h = self.input_proj.forward(x)?;
        let te = self.time_embed.forward(t, h.dtype())?;
        let ce = self.cond_embed.forward(c)?;
        let y = add(&te, &ce)?;
        for block in &self.res_blocks {
            h = block.forward(&h, &y)?;
        }
        self.final_layer.forward(&h, &y)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.cond_embed.quantize(bits, None)?;
        self.input_proj.quantize(bits, None)?;
        self.time_embed.mlp0.quantize(bits, None)?;
        self.time_embed.mlp2.quantize(bits, None)?;
        for b in &mut self.res_blocks {
            b.quantize(bits)?;
        }
        self.final_layer.linear.quantize(bits, None)?;
        self.final_layer.adaln.quantize(bits, None)
    }
}

/// The flow-matching ViT diffusion head: `SimpleMLPAdaLN` + its `FlowMatchScheduler`.
pub struct DiffLossFm {
    net: SimpleMlpAdaLn,
    scheduler: FlowMatchScheduler,
    in_channels: i32,
}

impl DiffLossFm {
    /// Build from a converted `vit_decoder.safetensors` (`net.*`). `prefix` = the net namespace
    /// (`"net"` for the sc-5144 layout). `depth` = number of res blocks (16), `in_channels` = 3584,
    /// `shift` = 2.0.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        depth: usize,
        in_channels: i32,
        shift: f32,
    ) -> Result<Self> {
        Ok(Self {
            net: SimpleMlpAdaLn::from_weights(w, prefix, depth)?,
            scheduler: FlowMatchScheduler::new(shift, true),
            in_channels,
        })
    }

    /// Raw denoiser forward (no CFG).
    pub fn forward(&self, x: &Array, t: &Array, c: &Array) -> Result<Array> {
        self.net.forward(x, t, c)
    }

    /// Standard CFG over a 2-tiled batch: `uncond + cfg·(cond − uncond)`.
    fn forward_with_cfg(&self, x: &Array, t: &Array, c: &Array, cfg: f32) -> Result<Array> {
        let half = &split(x, 2, 0)?[0];
        let combined = concatenate_axis(&[half, half], 0)?;
        let out = self.net.forward(&combined, t, c)?;
        let p = split(&out, 2, 0)?; // cond, uncond
        let half_eps = add(
            &p[1],
            &multiply(&subtract(&p[0], &p[1])?, Array::from_f32(cfg))?,
        )?;
        concatenate_axis(&[&half_eps, &half_eps], 0).map_err(Error::from)
    }

    /// Triple CFG over a 3-tiled batch (txt/img guidance):
    /// `uncond + img·(imgcond − uncond) + txt·(cond − imgcond)`.
    fn forward_with_txt_img_cfg(
        &self,
        x: &Array,
        t: &Array,
        c: &Array,
        txt_cfg: f32,
        img_cfg: f32,
    ) -> Result<Array> {
        let part = &split(x, 3, 0)?[0];
        let combined = concatenate_axis(&[part, part, part], 0)?;
        let out = self.net.forward(&combined, t, c)?;
        let p = split(&out, 3, 0)?; // cond, uncond, imgcond
        let img_term = multiply(&subtract(&p[2], &p[1])?, Array::from_f32(img_cfg))?;
        let txt_term = multiply(&subtract(&p[0], &p[2])?, Array::from_f32(txt_cfg))?;
        let part_eps = add(&p[1], &add(&img_term, &txt_term)?)?;
        concatenate_axis(&[&part_eps, &part_eps, &part_eps], 0).map_err(Error::from)
    }

    /// Denoise a target ViT embedding from `noise_base` `[N, in]`, conditioned on `z`. Mirrors
    /// `DiffLoss_FM.sample`: `img_cfg.is_some() && cfg>1` → triple CFG (z tiled ×3, noise ×3);
    /// `cfg>1` → standard CFG (×2); else plain. `num_steps` denoise steps (`vit_denoising_step`).
    /// `z` must already be tiled to match the chosen mode (×3 / ×2 / ×1), as the reference's caller
    /// does. Returns the tiled samples `[mode·N, in]`.
    pub fn sample(
        &mut self,
        z: &Array,
        cfg: f32,
        num_steps: usize,
        img_cfg: Option<f32>,
        noise_base: &Array,
    ) -> Result<Array> {
        self.scheduler.set_timesteps(num_steps);
        let dtype = z.dtype();
        let tiles = if img_cfg.is_some() && cfg > 1.0 {
            3
        } else if cfg > 1.0 {
            2
        } else {
            1
        };
        let refs: Vec<&Array> = (0..tiles).map(|_| noise_base).collect();
        let mut samples = concatenate_axis(&refs, 0)?.as_dtype(dtype)?;

        let timesteps: Vec<f32> = self.scheduler.timesteps().to_vec();
        for (i, &ts) in timesteps.iter().enumerate() {
            let t = Array::from_slice(&[ts], &[1]).as_dtype(dtype)?;
            let pred = match (img_cfg, cfg > 1.0) {
                (Some(img), true) => self.forward_with_txt_img_cfg(&samples, &t, z, cfg, img)?,
                (None, true) => self.forward_with_cfg(&samples, &t, z, cfg)?,
                _ => self.net.forward(&samples, &t, z)?,
            };
            samples = self.scheduler.step(&pred, i, &samples)?;
        }
        Ok(samples)
    }

    pub fn in_channels(&self) -> i32 {
        self.in_channels
    }

    /// Quantize the net's linears (Q4/Q8). (sc-5146)
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.net.quantize(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FlowMatchScheduler σ schedule: shift 2.0 / extra_one_step → first σ is the shifted σ_max,
    /// monotonically decreasing, `timesteps = σ·1000`, and `step` is the Euler velocity update.
    #[test]
    fn scheduler_schedule_and_step() {
        let sched = FlowMatchScheduler::new(2.0, true);
        // shift·1/(1+(shift-1)·1) = 2/2 = 1.0 for σ_max=1.
        assert!(
            (sched.sigmas[0] - 1.0).abs() < 1e-6,
            "first σ = shifted σ_max"
        );
        for w in sched.sigmas.windows(2) {
            assert!(w[1] < w[0], "σ strictly decreasing");
        }
        assert_eq!(sched.timesteps.len(), 100);
        assert!((sched.timesteps[0] - sched.sigmas[0] * 1000.0).abs() < 1e-3);

        // Euler step: sample + v·(σ_next − σ).
        let sample = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let v = Array::from_slice(&[0.5f32, -0.5], &[1, 2]);
        let out = sched.step(&v, 0, &sample).unwrap();
        let d = sched.sigmas[1] - sched.sigmas[0];
        let got: Vec<f32> = out.flatten(None, None).unwrap().as_slice::<f32>().to_vec();
        assert!((got[0] - (1.0 + 0.5 * d)).abs() < 1e-5);
        assert!((got[1] - (2.0 - 0.5 * d)).abs() < 1e-5);
    }

    /// `set_timesteps(3)` yields 3 steps (extra_one_step drops the tail of a 4-point linspace).
    #[test]
    fn scheduler_step_count() {
        let mut sched = FlowMatchScheduler::new(2.0, true);
        sched.set_timesteps(3);
        assert_eq!(sched.timesteps().len(), 3);
    }
}
