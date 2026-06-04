//! S4 — the LTX-2.3 **spatial latent upsampler** (2× spatial). Port of the `mlx_video` reference
//! `models/ltx/upsampler.py` (`LatentUpsampler` + `upsample_latents`), gated against it
//! (`tests/upsampler_parity.rs`, real `upsampler.safetensors`).
//!
//! Sits between the two-stage distilled denoise (S5): stage-1 runs at half resolution, its latents
//! are upsampled 2× spatially here, then stage-2 refines at full resolution. The reference loads the
//! `ltx-2-spatial-upscaler-x2` checkpoint **bf16** and runs the whole path bf16 (weights, latents,
//! and the un-/re-normalize `latents_mean`/`latents_std` are all bf16) — so this matches that exactly
//! rather than the VAE's f32 (which is its own gated choice). The one f32 island is `GroupNorm3d`,
//! which the reference upcasts to f32 internally and casts back — replicated here verbatim.
//!
//! Architecture (`num_blocks_per_stage = 4`, `mid_channels = 1024`, structure-from-weights):
//!   `initial_conv 128→1024` → `initial_norm` → SiLU → 4× pre-`ResBlock3D` → `SpatialRationalResampler`
//!   (frame-by-frame `Conv2d 1024→4096` + `PixelShuffle2D(2)`) → 4× post-`ResBlock3D` → `final_conv
//!   1024→128`. I/O is channels-first `NCFHW`, transposed to `NFHWC` only for the conv ops.
//!
//! Reference quirks carried over verbatim:
//!  - Conv weights are on-disk **PyTorch** layout (Conv3d `[O,I,D,H,W]`, Conv2d `[O,I,H,W]`) — unlike
//!    the VAE's pre-transposed MLX layout — so they're transposed to MLX `[O,…,I]` at load. The
//!    Conv2d lives under `upsampler.0.*` on disk (the reference renames it `upsampler.conv.*`).
//!  - `GroupNorm3d` (32 groups, eps 1e-5) reshapes `NFHWC → (N, F·H·W, groups, C/groups)`, takes
//!    mean/var over the spatial+within-group axes `(1, 3)`, normalizes, then scale/shifts — all in
//!    **f32**, output cast back to the input dtype.
//!  - `ResBlock3D` applies its SiLU **after** the residual add (`silu(conv2(norm)→ + residual)`).
//!  - `PixelShuffle2D` is the channels-last `(N,H,W,C·r²) → (N,H·r,W·r,C)` rearrange.
//!
//! The parity gate honors "divergence is not rounding": every op here is the same mlx op the
//! reference uses at the same dtype, so a >1% gap would be a real bug, not bf16 noise.

use mlx_rs::ops::{add, divide, mean_axes, multiply, subtract, var_axes};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{conv2d, conv3d, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// `GroupNorm3d` group count (`GroupNorm3d(32, …)` throughout the reference).
const GROUPS: i32 = 32;
/// `GroupNorm3d` epsilon (`GroupNorm3d.__init__(eps=1e-5)`).
const NORM_EPS: f32 = 1e-5;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// A bias-carrying conv whose on-disk weight is **PyTorch** layout; transposed to MLX `[O,…,I]` at
/// load. Stride 1, padding 1 (kernel 3, "same"), matching every conv in the upsampler.
struct Conv {
    w: Array,
    b: Array,
    /// `true` → 3-D (`NDHWC`), `false` → 2-D (`NHWC`).
    is_3d: bool,
}

impl Conv {
    /// `is_3d` picks the PyTorch→MLX transpose: 3-D `[O,I,D,H,W]→[O,D,H,W,I]` (`0,2,3,4,1`); 2-D
    /// `[O,I,H,W]→[O,H,W,I]` (`0,2,3,1`). Weights stay at their on-disk dtype (bf16).
    fn load(w: &Weights, prefix: &str, is_3d: bool) -> Result<Self> {
        let raw = w.require(&format!("{prefix}.weight"))?;
        let weight = if is_3d {
            raw.transpose_axes(&[0, 2, 3, 4, 1])?
        } else {
            raw.transpose_axes(&[0, 2, 3, 1])?
        };
        Ok(Self {
            w: weight,
            b: w.require(&format!("{prefix}.bias"))?.clone(),
            is_3d,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        if self.is_3d {
            conv3d(x, &self.w, Some(&self.b), (1, 1, 1), (1, 1, 1))
        } else {
            conv2d(x, &self.w, Some(&self.b), 1, 1)
        }
    }
}

/// `GroupNorm3d` — group norm over `NFHWC` computed in **f32** then cast back. Mirrors the reference
/// reshape `(N, F·H·W, groups, C/groups)` + mean/var over `(1, 3)` exactly (mlx `mean`/`var`, ddof 0).
struct GroupNorm {
    weight: Array,
    bias: Array,
}

impl GroupNorm {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.clone(),
            bias: w.require(&format!("{prefix}.bias"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let input_dtype = x.dtype();
        let x = x.as_dtype(Dtype::Float32)?;
        let sh = x.shape(); // (n, f, h, w, c)
        let (n, f, h, wd, c) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let gs = c / GROUPS;
        let xr = x.reshape(&[n, f * h * wd, GROUPS, gs])?;
        let mean = mean_axes(&xr, &[1, 3], true)?;
        let var = var_axes(&xr, &[1, 3], true, None)?; // ddof 0 — matches mx.var
        let denom = add(&var, scalar(NORM_EPS))?.sqrt()?;
        let normed = divide(&subtract(&xr, &mean)?, &denom)?.reshape(&[n, f, h, wd, c])?;
        let wf = self.weight.as_dtype(Dtype::Float32)?;
        let bf = self.bias.as_dtype(Dtype::Float32)?;
        let out = add(&multiply(&normed, &wf)?, &bf)?;
        Ok(out.as_dtype(input_dtype)?)
    }
}

/// `PixelShuffle2D(r)` — channels-last `(N, H, W, C·r²) → (N, H·r, W·r, C)`.
fn pixel_shuffle_2d(x: &Array, r: i32) -> Result<Array> {
    let sh = x.shape(); // (n, h, w, c)
    let (n, h, wd, c) = (sh[0], sh[1], sh[2], sh[3]);
    let out_c = c / (r * r);
    let x = x.reshape(&[n, h, wd, out_c, r, r])?;
    let x = x.transpose_axes(&[0, 1, 4, 2, 5, 3])?;
    Ok(x.reshape(&[n, h * r, wd * r, out_c])?)
}

/// `SpatialRationalResampler` — frame-by-frame 2× spatial upsample: fold frames into the batch, one
/// `Conv2d 1024→4096`, `PixelShuffle2D(2)`, unfold.
struct SpatialResampler {
    conv: Conv,
}

impl SpatialResampler {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        // On disk the Conv2d is `{prefix}.0.*` (the reference renames it `{prefix}.conv.*`).
        Ok(Self {
            conv: Conv::load(w, &format!("{prefix}.0"), false)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape(); // (n, f, h, w, c)
        let (n, f, h, wd, c) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let x = x.reshape(&[n * f, h, wd, c])?;
        let x = self.conv.forward(&x)?;
        let x = pixel_shuffle_2d(&x, 2)?;
        Ok(x.reshape(&[n, f, h * 2, wd * 2, c])?)
    }
}

/// `ResBlock3D` — `conv1 → norm1 → SiLU → conv2 → norm2`, then `SiLU(· + residual)`.
struct ResBlock {
    conv1: Conv,
    norm1: GroupNorm,
    conv2: Conv,
    norm2: GroupNorm,
}

impl ResBlock {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv1: Conv::load(w, &format!("{prefix}.conv1"), true)?,
            norm1: GroupNorm::load(w, &format!("{prefix}.norm1"))?,
            conv2: Conv::load(w, &format!("{prefix}.conv2"), true)?,
            norm2: GroupNorm::load(w, &format!("{prefix}.norm2"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let residual = x.clone();
        let h = self.conv1.forward(x)?;
        let h = self.norm1.forward(&h)?;
        let h = silu(&h)?;
        let h = self.conv2.forward(&h)?;
        let h = self.norm2.forward(&h)?;
        silu(&add(&h, &residual)?)
    }
}

/// The LTX-2.3 spatial latent upsampler. `num_blocks_per_stage` is read from the checkpoint (count of
/// `res_blocks.{i}`), `mid_channels` follows from the conv weights.
pub struct LatentUpsampler {
    initial_conv: Conv,
    initial_norm: GroupNorm,
    res_blocks: Vec<ResBlock>,
    upsampler: SpatialResampler,
    post_upsample_res_blocks: Vec<ResBlock>,
    final_conv: Conv,
}

impl LatentUpsampler {
    /// Build from a loaded `upsampler.safetensors` (or `spatial_upscaler_x2_v1_1.safetensors`).
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let load_stage = |stem: &str| -> Result<Vec<ResBlock>> {
            let mut blocks = Vec::new();
            let mut i = 0;
            while w.get(&format!("{stem}.{i}.conv1.weight")).is_some() {
                blocks.push(ResBlock::load(w, &format!("{stem}.{i}"))?);
                i += 1;
            }
            Ok(blocks)
        };
        Ok(Self {
            initial_conv: Conv::load(w, "initial_conv", true)?,
            initial_norm: GroupNorm::load(w, "initial_norm")?,
            res_blocks: load_stage("res_blocks")?,
            upsampler: SpatialResampler::load(w, "upsampler")?,
            post_upsample_res_blocks: load_stage("post_upsample_res_blocks")?,
            final_conv: Conv::load(w, "final_conv", true)?,
        })
    }

    /// Upsample a channels-first `NCFHW` latent → `NCF(H·2)(W·2)`.
    pub fn forward(&self, latent_ncfhw: &Array) -> Result<Array> {
        // NCFHW → NFHWC for the channels-last conv ops.
        let mut x = latent_ncfhw.transpose_axes(&[0, 2, 3, 4, 1])?;
        x = self.initial_conv.forward(&x)?;
        x = self.initial_norm.forward(&x)?;
        x = silu(&x)?;
        for b in &self.res_blocks {
            x = b.forward(&x)?;
        }
        x = self.upsampler.forward(&x)?;
        for b in &self.post_upsample_res_blocks {
            x = b.forward(&x)?;
        }
        x = self.final_conv.forward(&x)?;
        // NFHWC → NCFHW.
        Ok(x.transpose_axes(&[0, 4, 1, 2, 3])?)
    }
}

/// `upsample_latents` — un-normalize (`latent·std + mean`), upsample 2×, re-normalize
/// (`(latent − mean)/std`). `latent_mean`/`latent_std` are the VAE `per_channel_statistics` (bf16,
/// shape `[C]`), reshaped to broadcast over `NCFHW`.
pub fn upsample_latents(
    latent: &Array,
    upsampler: &LatentUpsampler,
    latent_mean: &Array,
    latent_std: &Array,
) -> Result<Array> {
    let mean = latent_mean.reshape(&[1, -1, 1, 1, 1])?;
    let std = latent_std.reshape(&[1, -1, 1, 1, 1])?;
    let unnorm = add(&multiply(latent, &std)?, &mean)?;
    let up = upsampler.forward(&unnorm)?;
    Ok(divide(&subtract(&up, &mean)?, &std)?)
}
