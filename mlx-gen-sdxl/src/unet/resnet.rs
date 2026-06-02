//! UNet `ResnetBlock2D` (NHWC): GroupNorm→SiLU→Conv3×3, add the projected time embedding, then
//! GroupNorm→SiLU→Conv3×3, plus a residual (a 1×1-conv-as-Linear shortcut when channels change).
//! Port of the vendored `unet.ResnetBlock2D`. The whole UNet runs NHWC (mlx conv layout), so unlike
//! the Z-Image VAE port there is no per-conv transpose — only the stored weights are transposed at
//! load (NCHW→NHWC).

use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::nn::{conv2d, group_norm, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::nchw_to_nhwc;

/// UNet GroupNorm: 32 groups, eps 1e-5 (mlx `nn.GroupNorm` default, which the vendored uses).
const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-5;

pub struct ResnetBlock2D {
    norm1_w: Array,
    norm1_b: Array,
    conv1_w: Array,
    conv1_b: Array,
    norm2_w: Array,
    norm2_b: Array,
    conv2_w: Array,
    conv2_b: Array,
    /// Time-embedding projection — `Some` for UNet resnets, `None` for VAE resnets (no temb).
    time_emb_proj: Option<AdaptableLinear>,
    /// 1×1-conv-as-Linear residual projection when in≠out channels.
    shortcut: Option<AdaptableLinear>,
}

impl ResnetBlock2D {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let g = |n: &str| w.require(&format!("{prefix}.{n}")).cloned();
        let shortcut = match w.get(&format!("{prefix}.conv_shortcut.weight")) {
            Some(sw) => {
                // [out, in, 1, 1] → [out, in] Linear over NHWC channels.
                let sh = sw.shape();
                let w2 = sw.reshape(&[sh[0], sh[1]])?;
                let b = w.require(&format!("{prefix}.conv_shortcut.bias"))?.clone();
                Some(AdaptableLinear::dense(w2, Some(b)))
            }
            None => None,
        };
        // VAE resnets carry no time embedding (`temb_channels=None`).
        let time_emb_proj = match w.get(&format!("{prefix}.time_emb_proj.weight")) {
            Some(tw) => Some(AdaptableLinear::dense(
                tw.clone(),
                Some(w.require(&format!("{prefix}.time_emb_proj.bias"))?.clone()),
            )),
            None => None,
        };
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            conv1_w: nchw_to_nhwc(&g("conv1.weight")?)?,
            conv1_b: g("conv1.bias")?,
            time_emb_proj,
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            conv2_w: nchw_to_nhwc(&g("conv2.weight")?)?,
            conv2_b: g("conv2.bias")?,
            shortcut,
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        if let Some(t) = &mut self.time_emb_proj {
            t.quantize(bits, None)?;
        }
        if let Some(s) = &mut self.shortcut {
            s.quantize(bits, None)?;
        }
        Ok(())
    }

    /// `x`: NHWC `[B, H, W, in]`; `temb`: `Some([B, temb_dim])` for the UNet, `None` for the VAE.
    pub fn forward(&self, x: &Array, temb: Option<&Array>) -> Result<Array> {
        let y = group_norm(x, &self.norm1_w, &self.norm1_b, GN_GROUPS, GN_EPS)?;
        let mut y = conv2d(&silu(&y)?, &self.conv1_w, Some(&self.conv1_b), 1, 1)?;
        // Add the projected time embedding (UNet only), broadcast over H,W.
        if let (Some(proj), Some(t)) = (&self.time_emb_proj, temb) {
            let tp = proj.forward(&silu(t)?)?;
            let tb = tp.shape();
            y = add(&y, &tp.reshape(&[tb[0], 1, 1, tb[1]])?)?;
        }
        let y = group_norm(&y, &self.norm2_w, &self.norm2_b, GN_GROUPS, GN_EPS)?;
        let y = conv2d(&silu(&y)?, &self.conv2_w, Some(&self.conv2_b), 1, 1)?;

        let residual = match &self.shortcut {
            Some(sc) => sc.forward(x)?,
            None => x.clone(),
        };
        Ok(add(&residual, &y)?)
    }

    /// The one LoRA-targetable Linear on a U-Net resnet — `time_emb_proj`. `conv_shortcut` is a
    /// 1×1 conv in the fork (4-D conv LoRAs target it, which the vendored Linear-only merge skips),
    /// so it is intentionally not a target.
    pub fn lora_target_paths(&self, prefix: &str, out: &mut Vec<String>) {
        if self.time_emb_proj.is_some() {
            out.push(format!("{prefix}.time_emb_proj"));
        }
    }
}

impl AdaptableHost for ResnetBlock2D {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["time_emb_proj"] => self.time_emb_proj.as_mut(),
            _ => None,
        }
    }
}
