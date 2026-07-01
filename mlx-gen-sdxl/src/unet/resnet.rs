//! UNet `ResnetBlock2D` (NHWC): GroupNorm→SiLU→Conv3×3, add the projected time embedding, then
//! GroupNorm→SiLU→Conv3×3, plus a residual (a 1×1-conv-as-Linear shortcut when channels change).
//! Port of the vendored `unet.ResnetBlock2D`. The whole UNet runs NHWC (mlx conv layout), so unlike
//! the Z-Image VAE port there is no per-conv transpose — only the stored weights are transposed at
//! load (NCHW→NHWC).

use mlx_rs::ops::add;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableConv2d, AdaptableHost, AdaptableLinear};
use mlx_gen::nn::{conv2d, group_norm};

use crate::silu_glue;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::nchw_to_nhwc;

/// UNet GroupNorm: 32 groups, eps 1e-5 (mlx `nn.GroupNorm` default, which the vendored uses).
const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-5;

#[derive(Clone)]
pub struct ResnetBlock2D {
    norm1_w: Array,
    norm1_b: Array,
    /// 3×3 conv (NHWC) — a conv-layer LoRA target (sc-2919).
    conv1: AdaptableConv2d,
    norm2_w: Array,
    norm2_b: Array,
    /// 3×3 conv (NHWC) — a conv-layer LoRA target (sc-2919).
    conv2: AdaptableConv2d,
    /// Time-embedding projection — `Some` for UNet resnets, `None` for VAE resnets (no temb).
    time_emb_proj: Option<AdaptableLinear>,
    /// 1×1-conv-as-Linear residual projection when in≠out channels. A conv-shortcut LoRA (sc-2919)
    /// merges here via a 1×1 `[out,in,1,1]→[out,in]` reshape (kept a Linear so its forward — a
    /// matmul over NHWC channels — is unchanged; switching to `conv2d` would risk a 1-ULP shift on
    /// the chaos-sensitive ancestral sampler).
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
        // VAE resnets carry no time embedding (`temb_channels=None`). The U-Net's `time_emb_proj` is
        // quantized (packed-detect, sc-8746); the VAE resnets never reach this crate's quantize path.
        // Detect presence by the dense `.weight` OR a packed `.scales` (a pre-quantized snapshot has
        // no dense `.weight` key beyond the u32 codes, but always carries `.scales`).
        let has_time_proj = w.get(&format!("{prefix}.time_emb_proj.weight")).is_some()
            || w.get(&format!("{prefix}.time_emb_proj.scales")).is_some();
        let time_emb_proj = if has_time_proj {
            Some(crate::quant::lin(
                w,
                &format!("{prefix}.time_emb_proj"),
                true,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            conv1: AdaptableConv2d::new(nchw_to_nhwc(&g("conv1.weight")?)?, Some(g("conv1.bias")?)),
            time_emb_proj,
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            conv2: AdaptableConv2d::new(nchw_to_nhwc(&g("conv2.weight")?)?, Some(g("conv2.bias")?)),
            shortcut,
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        if let Some(t) = &mut self.time_emb_proj {
            t.quantize(bits, None)?;
        }
        // `shortcut` is the 1×1 `conv_shortcut` stored as a Linear (for the NHWC channel-matmul
        // forward), not a true Linear — so it stays DENSE like every other conv in the U-Net
        // (`conv_in`/`conv_out`, resnet `conv1`/`conv2`, the up/down samplers; see
        // `UNet2DConditionModel::quantize`). It sits directly on the resnet residual/skip path, so
        // int8 rounding error there injects straight into the residual stream and, at 1024², compounds
        // across the denoise loop into a runaway latent outlier that blows out the VAE → a flat image
        // (sc-3329). The defect is the int8 rounding itself, not a dtype/overflow issue (f32 scales +
        // f32 activations reproduce it identically), and it stayed sub-threshold at ≤512² so sc-2641
        // missed it. The `mlx_sd` reference also stores `conv_shortcut` as `nn.Linear` and quantizes
        // it, so leaving it dense is a deliberate, correct divergence from a reference that shares the
        // bug. (LoRA still merges into `shortcut` as a reshaped conv delta — that path is unaffected.)
        Ok(())
    }

    /// Cast every dtype-bearing leaf to `dtype` (sc-4941 bf16 training): the GroupNorm weights/biases,
    /// both convs, the time-embedding projection, and the 1×1 shortcut.
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        super::cast_array(&mut self.norm1_w, dtype)?;
        super::cast_array(&mut self.norm1_b, dtype)?;
        super::cast_array(&mut self.norm2_w, dtype)?;
        super::cast_array(&mut self.norm2_b, dtype)?;
        self.conv1.cast_weights(dtype)?;
        self.conv2.cast_weights(dtype)?;
        if let Some(t) = &mut self.time_emb_proj {
            t.cast_weights(dtype)?;
        }
        if let Some(sc) = &mut self.shortcut {
            sc.cast_weights(dtype)?;
        }
        Ok(())
    }

    /// `x`: NHWC `[B, H, W, in]`; `temb`: `Some([B, temb_dim])` for the UNet, `None` for the VAE.
    pub fn forward(&self, x: &Array, temb: Option<&Array>) -> Result<Array> {
        let y = group_norm(x, &self.norm1_w, &self.norm1_b, GN_GROUPS, GN_EPS)?;
        let mut y = conv2d(
            &silu_glue(&y)?,
            self.conv1.weight(),
            self.conv1.bias(),
            1,
            1,
        )?;
        // Add the projected time embedding (UNet only), broadcast over H,W.
        if let (Some(proj), Some(t)) = (&self.time_emb_proj, temb) {
            let tp = proj.forward(&silu_glue(t)?)?;
            let tb = tp.shape();
            y = add(&y, &tp.reshape(&[tb[0], 1, 1, tb[1]])?)?;
        }
        let y = group_norm(&y, &self.norm2_w, &self.norm2_b, GN_GROUPS, GN_EPS)?;
        let y = conv2d(
            &silu_glue(&y)?,
            self.conv2.weight(),
            self.conv2.bias(),
            1,
            1,
        )?;

        let residual = match &self.shortcut {
            Some(sc) => sc.forward(x)?,
            None => x.clone(),
        };
        Ok(add(&residual, &y)?)
    }

    /// The one LoRA-targetable **Linear** on a U-Net resnet — `time_emb_proj`. The convs
    /// (`conv1`/`conv2`/`conv_shortcut`) are conv-layer targets, enumerated by
    /// [`conv_target_paths`](Self::conv_target_paths) instead (sc-2919).
    pub fn lora_target_paths(&self, prefix: &str, out: &mut Vec<String>) {
        if self.time_emb_proj.is_some() {
            out.push(format!("{prefix}.time_emb_proj"));
        }
    }

    /// This resnet's conv-layer LoRA targets (sc-2919): `conv1`, `conv2`, and — when present —
    /// `conv_shortcut`. Merged only under [`crate::adapters::LoraCoverage::Complete`]; the
    /// Linear-only vendored coverage drops them (preserving byte-parity with the retired Python path).
    pub fn conv_target_paths(&self, prefix: &str, out: &mut Vec<String>) {
        out.push(format!("{prefix}.conv1"));
        out.push(format!("{prefix}.conv2"));
        if self.shortcut.is_some() {
            out.push(format!("{prefix}.conv_shortcut"));
        }
    }
}

impl AdaptableHost for ResnetBlock2D {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["time_emb_proj"] => self.time_emb_proj.as_mut(),
            // conv_shortcut is a 1×1 conv stored as a Linear; a conv LoRA merges into it as a
            // reshaped 2-D delta (sc-2919, routed by the SDXL adapter's conv path).
            ["conv_shortcut"] => self.shortcut.as_mut(),
            _ => None,
        }
    }

    fn adaptable_conv_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableConv2d> {
        match path {
            ["conv1"] => Some(&mut self.conv1),
            ["conv2"] => Some(&mut self.conv2),
            _ => None,
        }
    }
}
