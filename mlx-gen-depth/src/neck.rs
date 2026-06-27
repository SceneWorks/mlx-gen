//! DPT neck — reassemble stage + RefineNet feature-fusion stage. Faithful port of the HF
//! `transformers` `DepthAnythingNeck` / `DepthAnythingReassembleStage` /
//! `DepthAnythingFeatureFusionStage` for the `neck.*` weight tree.
//!
//! Stage 1 — **reassemble**: each of the four captured hidden states `[B, grid²+1, hidden]` has its
//! CLS token dropped, is reshaped to a 2-D map `[B, grid, grid, hidden]`, projected by a 1×1 conv to
//! `neck_hidden_sizes[i]`, then resized by `reassemble_factors[i]`:
//!   - factor > 1 → `ConvTranspose2d(kernel=factor, stride=factor)` (upsample),
//!   - factor == 1 → identity,
//!   - factor < 1 → `Conv2d(kernel=3, stride=1/factor, pad=1)` (downsample).
//!
//! Stage 2 — `convs`: a 3×3 (pad 1, **no bias**) conv projects each reassembled map to
//! `fusion_hidden_size` (64).
//!
//! Stage 3 — **feature fusion** (`fusion_stage`), processed deepest→shallowest: a pre-activation
//! residual unit refines each level; from the second level on the running fused map is bilinearly
//! resized to the incoming residual, summed, refined again, ×2 bilinearly upsampled, and 1×1
//! projected. The shallowest fused map (×2 upsampled to 2·grid·factor[0]) is the head input.

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{add, conv_transpose2d};
use mlx_rs::Array;

use mlx_gen::nn::conv2d;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::DepthAnythingConfig;
use crate::util::{bilinear_resize, conv_transpose_w, conv_w_ohwi, join};

/// How a reassemble layer resizes its projected map.
enum Resize {
    /// `ConvTranspose2d(kernel=stride=factor)`: OHWI weight + bias.
    Up { w: Array, b: Array, stride: i32 },
    /// Identity (factor == 1).
    Same,
    /// `Conv2d(kernel=3, stride, pad=1)`: OHWI weight + bias.
    Down { w: Array, b: Array, stride: i32 },
}

/// One reassemble layer: 1×1 projection + factor resize.
struct ReassembleLayer {
    proj_w: Array, // 1×1 conv OHWI
    proj_b: Array,
    resize: Resize,
}

impl ReassembleLayer {
    fn from_weights(w: &Weights, prefix: &str, factor: f32) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let resize = if factor > 1.0 {
            Resize::Up {
                w: conv_transpose_w(w.require(&p("resize.weight"))?)?,
                b: w.require(&p("resize.bias"))?.clone(),
                stride: factor as i32,
            }
        } else if (factor - 1.0).abs() < f32::EPSILON {
            Resize::Same
        } else {
            Resize::Down {
                w: conv_w_ohwi(w.require(&p("resize.weight"))?)?,
                b: w.require(&p("resize.bias"))?.clone(),
                stride: (1.0 / factor).round() as i32,
            }
        };
        Ok(Self {
            proj_w: conv_w_ohwi(w.require(&p("projection.weight"))?)?,
            proj_b: w.require(&p("projection.bias"))?.clone(),
            resize,
        })
    }

    /// `hidden`: a captured backbone state `[B, grid²+1, hidden]` → an NHWC feature map.
    fn forward(&self, hidden: &Array, grid: i32, hidden_dim: i32) -> Result<Array> {
        // Drop CLS (index 0), reshape patch tokens to [B, grid, grid, hidden] (NHWC).
        let b = hidden.shape()[0];
        let patches = hidden.index((.., 1..));
        let map = patches.reshape(&[b, grid, grid, hidden_dim])?;
        // 1×1 projection.
        let map = conv2d(&map, &self.proj_w, Some(&self.proj_b), 1, 0)?;
        match &self.resize {
            Resize::Up { w, b: bias, stride } => {
                Ok(
                    conv_transpose2d(&map, w, (*stride, *stride), None, None, None, None)?
                        .add(bias)?,
                )
            }
            Resize::Same => Ok(map),
            Resize::Down { w, b: bias, stride } => Ok(conv2d(&map, w, Some(bias), *stride, 1)?),
        }
    }
}

/// Pre-activation residual unit (`PreActResidualLayer`): `ReLU → conv3×3 → ReLU → conv3×3`, added to
/// the input. Both convs are pad-1 stride-1 with bias.
struct PreActResidual {
    c1_w: Array,
    c1_b: Array,
    c2_w: Array,
    c2_b: Array,
}

impl PreActResidual {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            c1_w: conv_w_ohwi(w.require(&p("convolution1.weight"))?)?,
            c1_b: w.require(&p("convolution1.bias"))?.clone(),
            c2_w: conv_w_ohwi(w.require(&p("convolution2.weight"))?)?,
            c2_b: w.require(&p("convolution2.bias"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = relu(x)?;
        let h = conv2d(&h, &self.c1_w, Some(&self.c1_b), 1, 1)?;
        let h = relu(&h)?;
        let h = conv2d(&h, &self.c2_w, Some(&self.c2_b), 1, 1)?;
        Ok(add(x, &h)?)
    }
}

/// One feature-fusion layer (`DepthAnythingFeatureFusionLayer`).
struct FusionLayer {
    res1: PreActResidual,
    res2: PreActResidual,
    proj_w: Array, // 1×1 conv OHWI
    proj_b: Array,
}

impl FusionLayer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            res1: PreActResidual::from_weights(w, &p("residual_layer1"))?,
            res2: PreActResidual::from_weights(w, &p("residual_layer2"))?,
            proj_w: conv_w_ohwi(w.require(&p("projection.weight"))?)?,
            proj_b: w.require(&p("projection.bias"))?.clone(),
        })
    }

    /// Fuse: when `residual` is present, refine it (res1) + add to the running map (bilinearly
    /// resized to the residual's HW when they differ, align_corners=False), then refine (res2),
    /// ×2 bilinear upsample (align_corners=True), 1×1 project.
    fn forward(&self, hidden: &Array, residual: Option<&Array>) -> Result<Array> {
        let mut x = hidden.clone();
        if let Some(res) = residual {
            let (rh, rw) = (res.shape()[1], res.shape()[2]);
            if x.shape()[1] != rh || x.shape()[2] != rw {
                x = bilinear_resize(&x, rh, rw, false)?;
            }
            x = add(&x, &self.res1.forward(res)?)?;
        }
        x = self.res2.forward(&x)?;
        // ×2 upsample (align_corners=True).
        let (h, w) = (x.shape()[1], x.shape()[2]);
        x = bilinear_resize(&x, h * 2, w * 2, true)?;
        // 1×1 projection.
        conv2d(&x, &self.proj_w, Some(&self.proj_b), 1, 0)
    }
}

/// The full DPT neck: reassemble + project (`convs`) + fusion stage.
pub struct DptNeck {
    reassemble: Vec<ReassembleLayer>,
    // 3×3 pad-1 NO-bias projection per level (`neck.convs.{i}`).
    convs: Vec<Array>,
    fusion: Vec<FusionLayer>,
}

impl DptNeck {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &DepthAnythingConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let reassemble = (0..4)
            .map(|i| {
                ReassembleLayer::from_weights(
                    w,
                    &p(&format!("reassemble_stage.layers.{i}")),
                    cfg.reassemble_factors[i],
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let convs = (0..4)
            .map(|i| conv_w_ohwi(w.require(&p(&format!("convs.{i}.weight")))?))
            .collect::<Result<Vec<_>>>()?;
        let fusion = (0..4)
            .map(|i| FusionLayer::from_weights(w, &p(&format!("fusion_stage.layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            reassemble,
            convs,
            fusion,
        })
    }

    /// `hidden_states`: the four captured backbone states (shallow→deep), each `[B, grid²+1, hidden]`.
    /// Returns the fused NHWC feature map the head consumes.
    pub fn forward(&self, hidden_states: &[Array], grid: i32, hidden_dim: i32) -> Result<Array> {
        // Reassemble + project each level (shallow→deep order).
        let mut feats = Vec::with_capacity(4);
        for (i, hs) in hidden_states.iter().enumerate() {
            let re = self.reassemble[i].forward(hs, grid, hidden_dim)?;
            // 3×3 pad-1 no-bias projection to fusion_hidden_size.
            let f = conv2d(&re, &self.convs[i], None, 1, 1)?;
            feats.push(f);
        }
        // Fusion runs deepest→shallowest. `transformers` reverses the feature list so `fusion[0]`
        // pairs with the deepest level (`feats[3]`) and fuses with no residual; each subsequent
        // `fusion[k]` folds in the next-shallower feature (`feats[3-k]`) as its residual.
        let mut fused = self.fusion[0].forward(&feats[3], None)?;
        for k in 1..4 {
            fused = self.fusion[k].forward(&fused, Some(&feats[3 - k]))?;
        }
        Ok(fused)
    }
}

/// ReLU over an [`Array`] (no in-place; matches the `maximum(x, 0)` MLX idiom).
fn relu(x: &Array) -> Result<Array> {
    Ok(mlx_rs::ops::maximum(x, Array::from_f32(0.0))?)
}
