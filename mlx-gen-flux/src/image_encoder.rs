//! CLIP-ViT-L/14 image encoder for the XLabs FLUX IP-Adapter (sc-3622).
//!
//! XLabs' `flux-ip-adapter` conditions on `openai/clip-vit-large-patch14` loaded as a
//! `CLIPVisionModelWithProjection`. The torch path takes `clip_image_encoder(image).image_embeds`
//! — i.e. the **projected pooled CLS token**, `[B, 768]` — and feeds it to the `ImageProjModel`
//! (`image_proj`, sc-3623). This module reproduces exactly that `.image_embeds` output in MLX.
//!
//! The transformer body is `mlx-gen-sdxl`'s [`ClipVisionEncoder`] parameterised to ViT-L/14
//! ([`VisionConfig::vit_l_14`]) — the same crate-reuse pattern `mlx-gen-svd` uses for its ViT-H
//! tower. Only the projection head differs from [`mlx_gen_svd`]: `pooled = post_layernorm(
//! last_hidden_state[:, 0])`, `image_embeds = visual_projection(pooled)` (Linear 1024→768, no
//! bias) — diffusers `CLIPVisionTransformer` pooling + `CLIPVisionModelWithProjection`.

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::matmul;
use mlx_rs::{Array, Dtype};

use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_gen_sdxl::{preprocess_clip_image, ClipVisionEncoder, VisionConfig};

/// CLIP LN epsilon (matches the body + diffusers `layer_norm_eps`).
const LN_EPS: f32 = 1e-5;

/// The XLabs IP-Adapter image tower: ViT-L/14 body + the `CLIPVisionModelWithProjection` head.
pub struct FluxIpImageEncoder {
    body: ClipVisionEncoder,
    post_ln_w: Array,
    post_ln_b: Array,
    /// `visual_projection.weight` `[projection_dim, hidden]` (no bias). 768×1024 for ViT-L/14.
    visual_projection: Array,
    /// Compute dtype (the loaded weight dtype).
    dtype: Dtype,
}

impl FluxIpImageEncoder {
    /// Load from an `openai/clip-vit-large-patch14` checkpoint: the `vision_model.*` body +
    /// `vision_model.post_layernorm.*` + top-level `visual_projection.weight`.
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let body = ClipVisionEncoder::from_weights(w, &VisionConfig::vit_l_14())?;
        let visual_projection = w.require("visual_projection.weight")?.clone();
        Ok(Self {
            body,
            post_ln_w: w.require("vision_model.post_layernorm.weight")?.clone(),
            post_ln_b: w.require("vision_model.post_layernorm.bias")?.clone(),
            dtype: visual_projection.dtype(),
            visual_projection,
        })
    }

    /// `pixel_values` NHWC `[B, 224, 224, 3]` (CLIP-normalised) → `image_embeds` `[B, 768]`.
    /// Mirrors diffusers `self.image_encoder(image).image_embeds`: run the tower → CLS token of the
    /// last hidden state → `post_layernorm` → `visual_projection`. Returns f32 (pipeline-facing).
    pub fn image_embeds(&self, pixel_values: &Array) -> Result<Array> {
        let pixel_values = pixel_values.as_dtype(self.dtype)?;
        let states = self.body.hidden_states(&pixel_values)?;
        let last = states.last().expect("hidden_states non-empty"); // [B, 257, 1024]
        let cls = last.take_axis(Array::from_int(0), 1)?; // [B, 1024] (CLS token, axis dropped)
        let pooled = layer_norm(&cls, Some(&self.post_ln_w), Some(&self.post_ln_b), LN_EPS)?;
        // visual_projection is a bias-free Linear with weight [proj, hidden] → embeds = pooled · Wᵀ.
        let embeds = matmul(&pooled, &self.visual_projection.transpose_axes(&[1, 0])?)?;
        Ok(embeds.as_dtype(Dtype::Float32)?)
    }

    /// Reference image → `image_embeds` `[1, 768]`. CLIP preprocess (resize/center-crop 224² +
    /// mean/std) → tower → projection head.
    pub fn encode(&self, image: &Image) -> Result<Array> {
        let pixel_values = preprocess_clip_image(image)?;
        self.image_embeds(&pixel_values)
    }
}
