//! Shared leaf modules for the face sub-models (SCRFD / ArcFace / BiSeNet): the BN-folded biased
//! [`Conv`], the bias-less [`ConvW`], and [`relu`]. These were byte-identical copies in each
//! sub-model file; hoisting them here means a loader / error-message / forward fix lands once (F-084).

use mlx_gen::array::scalar;
use mlx_gen::nn::conv2d;
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::ops::maximum;
use mlx_rs::Array;

/// `relu(x) = max(x, 0)`.
pub(crate) fn relu(x: &Array) -> Result<Array> {
    Ok(maximum(x, scalar(0.0))?)
}

/// A biased convolution (the BN-folded convs all carry a bias, folded in at conversion). Fields are
/// crate-visible: ArcFace reuses the loaded `fc` weights as a `linear` (`iresnet.rs`).
pub(crate) struct Conv {
    pub w: Array,
    pub b: Array,
}

impl Conv {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{prefix}.weight"))?.clone(),
            b: w.require(&format!("{prefix}.bias"))?.clone(),
        })
    }

    pub fn forward(&self, x: &Array, stride: i32, padding: i32) -> Result<Array> {
        conv2d(x, &self.w, Some(&self.b), stride, padding)
    }

    /// ConvBNReLU = biased conv → ReLU (used by BiSeNet's ConvBNReLU blocks).
    pub fn forward_relu(&self, x: &Array, stride: i32, padding: i32) -> Result<Array> {
        relu(&self.forward(x, stride, padding)?)
    }
}

/// A bias-less convolution (the BiSeNet FFM SE 1×1s and the final 1×1 head — no BN). Always stride 1,
/// pad 0 (1×1 convs).
pub(crate) struct ConvW {
    pub w: Array,
}

impl ConvW {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{prefix}.weight"))?.clone(),
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        conv2d(x, &self.w, None, 1, 0)
    }
}
