//! ArcFace recognition embedding — a native MLX port of antelopev2 `glintr100` (iresnet100).
//!
//! The fidelity-critical core of the face stack (sc-3081): PuLID-FLUX and InstantID were trained on
//! *this exact checkpoint's* 512-d embeddings, so the port must reproduce the onnx output numerically
//! (embedding cosine ≈ 1.0). Weights come from [`tools/convert_glintr100.py`], which walks the onnx
//! graph and emits canonical keys.
//!
//! ## Architecture (verified from the onnx graph)
//! - **stem**: `Conv(3→64, 3×3, s1, p1)[+bias] → PReLU(64)`
//! - **layers `[3,13,30,3]`** of IBasicBlock — `BN(bn1) → Conv(conv1, 3×3, s1, p1)[+bias] → PReLU
//!   → Conv(conv2, 3×3, s{block}, p1)[+bias] → [Conv(downsample, 1×1, s{block}) on block 0] → +identity`.
//!   Each layer's block 0 has stride 2 + a 1×1 downsample on the identity path; later blocks stride 1.
//! - **head**: `BN(bn2) → flatten(NCHW order) → Linear(25088→512)[fc] → BN(features)` → 512-d.
//!
//! The onnx export already folded the post-conv BatchNorms (`bn2`/`bn3`/downsample-bn) into their
//! convs (every Conv carries a bias). The remaining *pre-activation* BNs (per-block `bn1`, the final
//! `bn2`, and `features`) are folded to per-channel affine (`scale`/`shift`) at conversion time, so
//! the forward here is pure conv + PReLU + affine + add + linear — no runtime BatchNorm op.
//!
//! Runs in **f32** (the reference is f32; cosine parity is comfortable at f32). Conv weights are
//! stored MLX-native OHWI `[out, kH, kW, in]`; input is NHWC `[N, 112, 112, 3]`.

use mlx_gen::array::scalar;
use mlx_gen::nn;
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::ops::{add, maximum, minimum, multiply};
use mlx_rs::Array;

use crate::common::Conv;

/// iresnet100 block counts per layer (`layer1..layer4`).
const LAYERS: [usize; 4] = [3, 13, 30, 3];
/// Flattened head input = 512 channels × 7 × 7 feature map.
const FLAT: i32 = 512 * 7 * 7;

/// PReLU with a per-channel `slope` (`[C]`, broadcast over the NHWC channel axis):
/// `max(x, 0) + slope · min(x, 0)`.
fn prelu(x: &Array, slope: &Array) -> Result<Array> {
    let zero = scalar(0.0);
    let pos = maximum(x, &zero)?;
    let neg = minimum(x, &zero)?;
    Ok(add(&pos, &multiply(&neg, slope)?)?)
}

/// Per-channel affine `x · scale + shift` (a folded BatchNorm; `scale`/`shift` are `[C]`,
/// broadcasting over the last axis for both NHWC maps and `[N, C]` feature vectors).
fn affine(x: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    Ok(add(&multiply(x, scale)?, shift)?)
}

/// A folded BatchNorm (per-channel `scale`/`shift`).
struct Affine {
    scale: Array,
    shift: Array,
}

impl Affine {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            scale: w.require(&format!("{prefix}.scale"))?.clone(),
            shift: w.require(&format!("{prefix}.shift"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        affine(x, &self.scale, &self.shift)
    }
}

/// One IBasicBlock: `bn1 → conv1 → prelu → conv2(stride) → + downsample(identity)`.
struct Block {
    bn1: Affine,
    conv1: Conv,
    prelu: Array,
    conv2: Conv,
    stride: i32,
    downsample: Option<Conv>,
}

impl Block {
    fn forward(&self, x: &Array) -> Result<Array> {
        let t = self.bn1.forward(x)?;
        let t = self.conv1.forward(&t, 1, 1)?;
        let t = prelu(&t, &self.prelu)?;
        let t = self.conv2.forward(&t, self.stride, 1)?;
        let identity = match &self.downsample {
            Some(ds) => ds.forward(x, self.stride, 0)?,
            None => x.clone(),
        };
        Ok(add(&t, &identity)?)
    }
}

/// ArcFace iresnet100 recognition network → 512-d embedding.
pub struct ArcFace {
    stem_conv: Conv,
    stem_prelu: Array,
    layers: Vec<Vec<Block>>,
    bn2: Affine,
    fc: Conv, // `fc.weight` `[512, 25088]`, `fc.bias` `[512]` — applied as a Linear.
    features: Affine,
}

impl ArcFace {
    /// Load from the converted `arcface_iresnet100.safetensors` (see `tools/convert_glintr100.py`).
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let mut layers = Vec::with_capacity(LAYERS.len());
        for (li, &nb) in LAYERS.iter().enumerate() {
            let l = li + 1;
            let mut blocks = Vec::with_capacity(nb);
            for b in 0..nb {
                let p = format!("layer{l}.{b}");
                let stride = if b == 0 { 2 } else { 1 };
                let downsample = if b == 0 {
                    Some(Conv::load(w, &format!("{p}.downsample"))?)
                } else {
                    None
                };
                blocks.push(Block {
                    bn1: Affine::load(w, &format!("{p}.bn1"))?,
                    conv1: Conv::load(w, &format!("{p}.conv1"))?,
                    prelu: w.require(&format!("{p}.prelu.weight"))?.clone(),
                    conv2: Conv::load(w, &format!("{p}.conv2"))?,
                    stride,
                    downsample,
                });
            }
            layers.push(blocks);
        }
        Ok(Self {
            stem_conv: Conv::load(w, "stem.conv")?,
            stem_prelu: w.require("stem.prelu.weight")?.clone(),
            layers,
            bn2: Affine::load(w, "bn2")?,
            fc: Conv::load(w, "fc")?,
            features: Affine::load(w, "features")?,
        })
    }

    /// Compute the 512-d recognition embedding for a batch of aligned face crops.
    ///
    /// `x`: NHWC `[N, 112, 112, 3]` f32, normalized as `(rgb - 127.5) / 127.5` (the antelopev2
    /// ArcFace blob). Returns the raw `[N, 512]` embedding (un-normalized — match insightface's
    /// `face['embedding']`; L2-normalize at the call site for cosine).
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = self.stem_conv.forward(x, 1, 1)?;
        h = prelu(&h, &self.stem_prelu)?;
        for blocks in &self.layers {
            for blk in blocks {
                h = blk.forward(&h)?;
            }
        }
        // Head: bn2 over NHWC, then flatten in NCHW (channel-major) order to match the onnx
        // `Flatten`, then fc Linear + features BN.
        h = self.bn2.forward(&h)?;
        let n = h.shape()[0];
        h = h.transpose_axes(&[0, 3, 1, 2])?.reshape(&[n, FLAT])?;
        h = nn::linear(&h, &self.fc.w, &self.fc.b)?;
        h = self.features.forward(&h)?;
        Ok(h)
    }
}
