//! Pixtral patch embedding: a **bias-less** `Conv2d(num_channels → hidden, kernel = stride =
//! patch_size)` over an NHWC image. Port of the fork's `PixtralVisionModel.patch_conv`.

use mlx_rs::Array;

use mlx_gen::nn::conv2d;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::text_encoder::join;

pub struct PatchConv {
    /// MLX `Conv2d` weight `[out, kH, kW, in]` (transposed from the stored torch `[out, in, kH, kW]`).
    w: Array,
    patch: i32,
}

impl PatchConv {
    pub fn from_weights(w: &Weights, prefix: &str, patch: i32) -> Result<Self> {
        // torch `[out, in, patch, patch]` → MLX channels-last `[out, patch, patch, in]`.
        Ok(Self {
            w: w.require(&join(prefix, "weight"))?
                .transpose_axes(&[0, 2, 3, 1])?,
            patch,
        })
    }

    /// `image`: NHWC `[1, H, W, num_channels]` → patches `[gh·gw, embed]` in row-major (h-major)
    /// order — matching the reference's `conv(…).flatten(2).permute(0, 2, 1)`.
    pub fn forward(&self, image: &Array) -> Result<Array> {
        // kernel == stride == patch, no padding → one output cell per patch → [1, gh, gw, embed].
        let y = conv2d(image, &self.w, None, self.patch, 0)?;
        let sh = y.shape();
        Ok(y.reshape(&[sh[1] * sh[2], sh[3]])?)
    }
}
