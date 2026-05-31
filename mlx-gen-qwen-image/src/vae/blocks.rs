//! Leaf + stage building blocks of the Qwen-Image causal-Conv3d VAE, ported 1:1 from the frozen
//! fork (`~/repos/mflux/src/mflux/models/qwen/model/qwen_vae/`). Tensors are kept **NCTHW**
//! (channels-first) throughout — mirroring the fork — and transposed to channels-last only inside
//! the conv ops, since mlx convs are channels-last.
//!
//! Two notes carried from the fork:
//!  - The VAE "RMSNorm" is actually a **channel-L2 normalization** (`x / max(‖x‖₂ over C, eps) · √C
//!    · weight`), not feature-RMS. See [`rms_norm_channels`].
//!  - `Resample3d` constructs a temporal `time_conv` that the fork **never calls** in `forward`;
//!    for T2I (T=1) up/down-sampling is purely spatial. We don't port the unused `time_conv`.

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, divide, maximum, multiply, pad, split, sum_axes};
use mlx_rs::Array;

use mlx_gen::nn::{conv2d, conv3d, silu, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// VAE channel-L2 norm eps (fork `QwenImageRMSNorm` default).
const NORM_EPS: f32 = 1e-12;

/// Channel-L2 normalization over axis 1: `x / max(‖x‖₂ over C, eps) · √C · weight`.
/// `weight` is 1-D `(C,)`; `x` is any rank with the channel axis at index 1 (NCTHW or NCHW).
pub fn rms_norm_channels(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    let shape = x.shape();
    let nd = shape.len();
    let c = shape[1];
    let sum_sq = sum_axes(&multiply(x, x)?, &[1], true)?;
    let l2 = sum_sq.sqrt()?;
    let denom = maximum(&l2, Array::from_slice(&[eps], &[1]))?;
    let normed = divide(x, &denom)?;
    let scale = (c as f32).sqrt();
    let mut wshape = vec![1i32; nd];
    wshape[1] = c;
    let wt = weight.reshape(&wshape)?;
    let scaled = multiply(&normed, Array::from_slice(&[scale], &[1]))?;
    Ok(multiply(&scaled, &wt)?)
}

/// Causal 3-D conv: pad time on the **left only** (`2·pad_t, 0`) + symmetric H/W, then a valid
/// (padding-0) conv3d. NCTHW I/O. Weight is the fork's already-transposed MLX `[out,kD,kH,kW,in]`.
pub struct CausalConv3d {
    w: Array,
    b: Array,
    padding: i32,
}

impl CausalConv3d {
    pub fn from_weights(w: &Weights, prefix: &str, padding: i32) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{prefix}.conv3d.weight"))?.clone(),
            b: w.require(&format!("{prefix}.conv3d.bias"))?.clone(),
            padding,
        })
    }

    pub fn forward(&self, x_ncthw: &Array) -> Result<Array> {
        let p = self.padding;
        let x = if p > 0 {
            pad(
                x_ncthw,
                &[(0, 0), (0, 0), (2 * p, 0), (p, p), (p, p)][..],
                None,
                None,
            )?
        } else {
            x_ncthw.clone()
        };
        let x = x.transpose_axes(&[0, 2, 3, 4, 1])?; // NDHWC
        let y = conv3d(&x, &self.w, Some(&self.b), (1, 1, 1), (0, 0, 0))?;
        Ok(y.transpose_axes(&[0, 4, 1, 2, 3])?) // NCTHW
    }
}

/// `norm1 → SiLU → conv1(3×3×3) → norm2 → SiLU → conv2`, residual (1×1×1 skip when channels differ).
pub struct ResBlock3D {
    norm1: Array,
    conv1: CausalConv3d,
    norm2: Array,
    conv2: CausalConv3d,
    skip: Option<CausalConv3d>,
}

impl ResBlock3D {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let skip = if w
            .get(&format!("{prefix}.skip_conv.conv3d.weight"))
            .is_some()
        {
            Some(CausalConv3d::from_weights(
                w,
                &format!("{prefix}.skip_conv"),
                0,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: w.require(&format!("{prefix}.norm1.weight"))?.clone(),
            conv1: CausalConv3d::from_weights(w, &format!("{prefix}.conv1"), 1)?,
            norm2: w.require(&format!("{prefix}.norm2.weight"))?.clone(),
            conv2: CausalConv3d::from_weights(w, &format!("{prefix}.conv2"), 1)?,
            skip,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let h = rms_norm_channels(x, &self.norm1, NORM_EPS)?;
        let h = self.conv1.forward(&silu(&h)?)?;
        let h = rms_norm_channels(&h, &self.norm2, NORM_EPS)?;
        let h = self.conv2.forward(&silu(&h)?)?;
        let residual = match &self.skip {
            Some(s) => s.forward(x)?,
            None => x.clone(),
        };
        Ok(add(&h, &residual)?)
    }
}

/// Per-frame spatial self-attention (single head, head_dim = C). NCTHW I/O. The fork builds Q/K/V
/// via a 1×1 Conv2d (`to_qkv`) over channels, attends over the H·W tokens, then projects.
pub struct AttentionBlock3D {
    norm: Array,
    qkv_w: Array,
    qkv_b: Array,
    proj_w: Array,
    proj_b: Array,
}

impl AttentionBlock3D {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: w.require(&format!("{prefix}.norm.weight"))?.clone(),
            qkv_w: w.require(&format!("{prefix}.to_qkv.weight"))?.clone(),
            qkv_b: w.require(&format!("{prefix}.to_qkv.bias"))?.clone(),
            proj_w: w.require(&format!("{prefix}.proj.weight"))?.clone(),
            proj_b: w.require(&format!("{prefix}.proj.bias"))?.clone(),
        })
    }

    pub fn forward(&self, x_ncthw: &Array) -> Result<Array> {
        let sh = x_ncthw.shape();
        let (b, c, t, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let bt = b * t;
        // NCTHW -> (B·T, C, H, W)
        let x = x_ncthw
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[bt, c, h, w])?;
        // channel-L2 norm over C (axis 1), then to NHWC for the 1×1 convs.
        let normed = rms_norm_channels(&x, &self.norm, NORM_EPS)?.transpose_axes(&[0, 2, 3, 1])?;
        let qkv = conv2d(&normed, &self.qkv_w, Some(&self.qkv_b), 1, 0)?; // (BT,H,W,3C)
        let qkv = qkv.reshape(&[bt, h * w, 3 * c])?;
        let parts = split(&qkv, 3, 2)?; // q,k,v each (BT, H·W, C)
                                        // single head: (BT, 1, H·W, C)
        let q = parts[0].expand_dims(1)?;
        let k = parts[1].expand_dims(1)?;
        let v = parts[2].expand_dims(1)?;
        let scale = (c as f32).powf(-0.5);
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let o = o.reshape(&[bt, h, w, c])?;
        let o = conv2d(&o, &self.proj_w, Some(&self.proj_b), 1, 0)?; // (BT,H,W,C)
                                                                     // back to NCTHW
        let o = o
            .transpose_axes(&[0, 3, 1, 2])?
            .reshape(&[b, t, c, h, w])?
            .transpose_axes(&[0, 2, 1, 3, 4])?;
        Ok(add(&o, x_ncthw)?)
    }
}

/// Spatial 2× resample (up = nearest-2× + 3×3 conv to C/2; down = pad-(0,1) + stride-2 3×3 conv).
/// The fork's `time_conv` is never invoked, so only `resample_conv` is ported. NCTHW I/O.
pub struct Resample3d {
    conv_w: Array,
    conv_b: Array,
    upsample: bool,
}

impl Resample3d {
    pub fn from_weights(w: &Weights, prefix: &str, upsample: bool) -> Result<Self> {
        Ok(Self {
            conv_w: w
                .require(&format!("{prefix}.resample_conv.weight"))?
                .clone(),
            conv_b: w.require(&format!("{prefix}.resample_conv.bias"))?.clone(),
            upsample,
        })
    }

    pub fn forward(&self, x_ncthw: &Array) -> Result<Array> {
        let sh = x_ncthw.shape();
        let (b, c, t, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let bt = b * t;
        let x = x_ncthw
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[bt, c, h, w])?
            .transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let x = if self.upsample {
            let up = upsample_nearest(&x, 2)?;
            conv2d(&up, &self.conv_w, Some(&self.conv_b), 1, 1)?
        } else {
            // pad bottom/right by 1, then valid stride-2 conv (fork's asymmetric downsample).
            let padded = pad(&x, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
            conv2d(&padded, &self.conv_w, Some(&self.conv_b), 2, 0)?
        };
        let nsh = x.shape();
        let (nc, nh, nw) = (nsh[3], nsh[1], nsh[2]);
        Ok(x.transpose_axes(&[0, 3, 1, 2])? // (BT, nc, nh, nw)
            .reshape(&[b, t, nc, nh, nw])?
            .transpose_axes(&[0, 2, 1, 3, 4])?) // NCTHW
    }
}

/// `resnet → attention → resnet` bottleneck.
pub struct MidBlock3D {
    resnet0: ResBlock3D,
    attn: AttentionBlock3D,
    resnet1: ResBlock3D,
}

impl MidBlock3D {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            resnet0: ResBlock3D::from_weights(w, &format!("{prefix}.resnets.0"))?,
            attn: AttentionBlock3D::from_weights(w, &format!("{prefix}.attentions.0"))?,
            resnet1: ResBlock3D::from_weights(w, &format!("{prefix}.resnets.1"))?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.resnet0.forward(x)?;
        let x = self.attn.forward(&x)?;
        self.resnet1.forward(&x)
    }
}

/// N resnets followed by an optional spatial downsample.
pub struct DownBlock3D {
    resnets: Vec<ResBlock3D>,
    downsampler: Option<Resample3d>,
}

impl DownBlock3D {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_res_blocks: usize,
        has_downsampler: bool,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_res_blocks);
        for i in 0..num_res_blocks {
            resnets.push(ResBlock3D::from_weights(
                w,
                &format!("{prefix}.resnets.{i}"),
            )?);
        }
        let downsampler = if has_downsampler {
            Some(Resample3d::from_weights(
                w,
                &format!("{prefix}.downsamplers.0"),
                false,
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsampler,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x)?;
        }
        if let Some(d) = &self.downsampler {
            x = d.forward(&x)?;
        }
        Ok(x)
    }
}

/// `num_res_blocks + 1` resnets followed by an optional spatial upsample.
pub struct UpBlock3D {
    resnets: Vec<ResBlock3D>,
    upsampler: Option<Resample3d>,
}

impl UpBlock3D {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_res_blocks: usize,
        has_upsampler: bool,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_res_blocks + 1);
        for i in 0..(num_res_blocks + 1) {
            resnets.push(ResBlock3D::from_weights(
                w,
                &format!("{prefix}.resnets.{i}"),
            )?);
        }
        let upsampler = if has_upsampler {
            Some(Resample3d::from_weights(
                w,
                &format!("{prefix}.upsamplers.0"),
                true,
            )?)
        } else {
            None
        };
        Ok(Self { resnets, upsampler })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x)?;
        }
        if let Some(u) = &self.upsampler {
            x = u.forward(&x)?;
        }
        Ok(x)
    }
}
