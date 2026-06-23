//! CLIP ViT-L/14 image embedder — the `gen_core::ImageEmbedder` provider for the Dataset Doctor
//! analysis job (epic 6529 P2, sc-6535).
//!
//! Produces the **canonical OpenAI CLIP image embedding** (`openai/clip-vit-large-patch14` loaded as
//! `CLIPVisionModelWithProjection`): the ViT-L/14 tower → CLS token of the last hidden state →
//! `post_layernorm` → `visual_projection` (Linear 1024→768, no bias) → `[768]`. This is the same
//! `.image_embeds` head `mlx-gen-flux`'s IP-adapter uses, surfaced here as a backend-neutral embedder
//! and registered so the worker can `load_image_embedder("clip_vit_l14", …)`. The vector is returned
//! **raw** (un-normalized) — callers L2-normalize for cosine, exactly like `FaceEmbedder`.
//!
//! The transformer body, ViT-L/14 config, and CLIP preprocessing are reused from `mlx-gen-sdxl`
//! ([`ClipVisionEncoder`], [`VisionConfig::vit_l_14`], [`preprocess_clip_image`]); only the small
//! pooling + projection head lives here.

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::matmul;
use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::registry::{ImageEmbedderRegistration, TextEmbedderRegistration};
use mlx_gen::gen_core::runtime::{LoadSpec, WeightsSource};
use mlx_gen::gen_core::{
    ImageEmbedder, ImageEmbedderDescriptor, Result as GenResult, TextEmbedder,
    TextEmbedderDescriptor,
};
use mlx_gen::media::Image;
use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_gen_sdxl::{
    preprocess_clip_image, ClipTextConfig, ClipTextEncoder, ClipVisionEncoder, VisionConfig,
};

/// CLIP LN epsilon (matches the body + diffusers `layer_norm_eps`).
const LN_EPS: f32 = 1e-5;

/// The provider id used to load this embedder (`load_image_embedder("clip_vit_l14", …)`).
pub const MODEL_ID: &str = "clip_vit_l14";
/// The provider id used to load the matching CLIP text embedder.
pub const TEXT_MODEL_ID: &str = "clip_vit_l14_text";
const CLIP_MAX_LENGTH: usize = 77;
const CLIP_EOS_ID: i32 = 49407;

static DESCRIPTOR: ImageEmbedderDescriptor = ImageEmbedderDescriptor {
    id: MODEL_ID,
    family: "image-embed",
    backend: "mlx",
    embedding_dim: 768,
    space: "clip-vit-l14",
    mac_only: true,
};

static TEXT_DESCRIPTOR: TextEmbedderDescriptor = TextEmbedderDescriptor {
    id: TEXT_MODEL_ID,
    family: "text-embed",
    backend: "mlx",
    embedding_dim: 768,
    space: "clip-vit-l14",
    mac_only: true,
};

/// The descriptor for the registry (constructible without loading weights).
pub fn descriptor() -> ImageEmbedderDescriptor {
    DESCRIPTOR.clone()
}

/// The text-embedder descriptor for the registry (constructible without loading weights).
pub fn text_descriptor() -> TextEmbedderDescriptor {
    TEXT_DESCRIPTOR.clone()
}

/// CLIP ViT-L/14 image embedder: the `mlx-gen-sdxl` ViT body + the `CLIPVisionModelWithProjection`
/// pooling + projection head.
pub struct ClipImageEmbedder {
    body: ClipVisionEncoder,
    post_ln_w: Array,
    post_ln_b: Array,
    /// `visual_projection.weight` `[projection_dim, hidden]` (no bias). 768×1024 for ViT-L/14.
    visual_projection: Array,
    /// Compute dtype (the loaded weight dtype).
    dtype: Dtype,
}

impl ClipImageEmbedder {
    /// Load from an `openai/clip-vit-large-patch14` checkpoint dir: the `vision_model.*` body +
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

    /// `pixel_values` NHWC `[B, 224, 224, 3]` (CLIP-normalised) → `image_embeds` `[B, 768]`. Mirrors
    /// diffusers `self.image_encoder(image).image_embeds`: tower → CLS of the last hidden state →
    /// `post_layernorm` → `visual_projection`. Returns f32.
    pub fn image_embeds(&self, pixel_values: &Array) -> Result<Array> {
        let pixel_values = pixel_values.as_dtype(self.dtype)?;
        let states = self.body.hidden_states(&pixel_values)?;
        let last = states
            .last()
            .ok_or_else(|| Error::Msg("clip image embedder produced no hidden states".into()))?; // [B, 257, 1024]
        let cls = last.take_axis(Array::from_int(0), 1)?; // [B, 1024] (CLS token, axis dropped)
        let pooled = layer_norm(&cls, Some(&self.post_ln_w), Some(&self.post_ln_b), LN_EPS)?;
        // visual_projection is a bias-free Linear with weight [proj, hidden] → embeds = pooled · Wᵀ.
        let embeds = matmul(&pooled, &self.visual_projection.transpose_axes(&[1, 0])?)?;
        Ok(embeds.as_dtype(Dtype::Float32)?)
    }

    /// One image → its raw 768-d CLIP embedding as host floats. CLIP preprocess (resize/center-crop
    /// 224² + mean/std) → tower → projection head → `Vec<f32>`.
    fn embed_internal(&self, image: &Image) -> Result<Vec<f32>> {
        let pixel_values = preprocess_clip_image(image)?;
        let embeds = self.image_embeds(&pixel_values)?; // [1, 768]
        let flat = embeds.reshape(&[-1])?; // [768]
        flat.eval()?;
        Ok(flat.as_slice::<f32>().to_vec())
    }
}

impl ImageEmbedder for ClipImageEmbedder {
    fn descriptor(&self) -> &ImageEmbedderDescriptor {
        &DESCRIPTOR
    }

    fn embed(&self, image: &Image) -> GenResult<Vec<f32>> {
        self.embed_internal(image).map_err(Into::into)
    }
}

/// CLIP ViT-L/14 text embedder: the `mlx-gen-sdxl` CLIP-L text body + the
/// `CLIPTextModelWithProjection` pooled `text_projection` head.
pub struct ClipTextEmbedder {
    encoder: ClipTextEncoder,
    tokenizer: TextTokenizer,
}

impl ClipTextEmbedder {
    /// Load from an `openai/clip-vit-large-patch14` checkpoint dir: `text_model.*`,
    /// top-level `text_projection.weight`, and `tokenizer.json`.
    pub fn from_weights_dir(root: &std::path::Path) -> Result<Self> {
        let weights = Weights::from_dir(root)?;
        let tokenizer = TextTokenizer::from_file(
            root.join("tokenizer.json"),
            TokenizerConfig {
                max_length: CLIP_MAX_LENGTH,
                pad_token_id: CLIP_EOS_ID,
                chat_template: ChatTemplate::None,
                pad_to_max_length: true,
            },
        )?;
        Ok(Self {
            encoder: ClipTextEncoder::from_weights(&weights, "text_model", &clip_text_config())?,
            tokenizer,
        })
    }

    /// `text` → projected CLIP `text_embeds` `[1, 768]` as f32. The SDXL encoder's projected path
    /// applies `text_projection.weight` to the pooled EOS hidden state.
    pub fn text_embeds(&self, text: &str) -> Result<Array> {
        let tokens = self.tokenizer.tokenize(text)?;
        let (input_ids, _) = mlx_gen::tokenizer::to_arrays(&tokens);
        let output = self.encoder.forward(&input_ids)?;
        Ok(output.pooled.as_dtype(Dtype::Float32)?)
    }

    fn embed_text_internal(&self, text: &str) -> Result<Vec<f32>> {
        let embeds = self.text_embeds(text)?; // [1, 768]
        let flat = embeds.reshape(&[-1])?;
        flat.eval()?;
        Ok(flat.as_slice::<f32>().to_vec())
    }
}

impl TextEmbedder for ClipTextEmbedder {
    fn descriptor(&self) -> &TextEmbedderDescriptor {
        &TEXT_DESCRIPTOR
    }

    fn embed_text(&self, text: &str) -> GenResult<Vec<f32>> {
        self.embed_text_internal(text).map_err(Into::into)
    }
}

fn clip_text_config() -> ClipTextConfig {
    let mut cfg = ClipTextConfig::sdxl_te1();
    cfg.projection_dim = Some(768);
    cfg
}

/// Load the embedder from a weights directory (the `openai/clip-vit-large-patch14` snapshot).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn ImageEmbedder>> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        _ => {
            return Err(Error::Msg(
                "clip_vit_l14 requires a weights directory (WeightsSource::Dir)".into(),
            ))
        }
    };
    let weights = Weights::from_dir(root)?;
    Ok(Box::new(ClipImageEmbedder::from_weights(&weights)?))
}

/// Load the text embedder from a weights directory (the `openai/clip-vit-large-patch14` snapshot).
pub fn load_text(spec: &LoadSpec) -> Result<Box<dyn TextEmbedder>> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        _ => {
            return Err(Error::Msg(
                "clip_vit_l14_text requires a weights directory (WeightsSource::Dir)".into(),
            ))
        }
    };
    Ok(Box::new(ClipTextEmbedder::from_weights_dir(root)?))
}

/// Registry adapter: bridge the crate's rich `Result` into the backend-neutral `gen_core::Result`.
fn load_registered(spec: &LoadSpec) -> GenResult<Box<dyn ImageEmbedder>> {
    load(spec).map_err(Into::into)
}

fn load_text_registered(spec: &LoadSpec) -> GenResult<Box<dyn TextEmbedder>> {
    load_text(spec).map_err(Into::into)
}

inventory::submit! {
    ImageEmbedderRegistration { descriptor, load: load_registered }
}

inventory::submit! {
    TextEmbedderRegistration { descriptor: text_descriptor, load: load_text_registered }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_advertises_clip_vit_l14() {
        let d = descriptor();
        assert_eq!(d.id, "clip_vit_l14");
        assert_eq!(d.embedding_dim, 768);
        assert_eq!(d.space, "clip-vit-l14");
        assert_eq!(d.backend, "mlx");
        assert!(d.mac_only);
    }

    #[test]
    fn non_dir_weights_source_is_rejected() {
        // A single-file source is rejected up front (a CLIP snapshot is a directory).
        let spec = LoadSpec::new(WeightsSource::File(std::path::PathBuf::from(
            "model.safetensors",
        )));
        assert!(load(&spec).is_err());
    }

    #[test]
    fn registered_and_discoverable_by_id() {
        // The `inventory::submit!` registration is linked in this crate's test binary, so the registry
        // must find `clip_vit_l14` by id and route to our loader — the error is the weights complaint,
        // NOT "no image embedder registered" (which would mean the registration didn't link).
        let spec = LoadSpec::new(WeightsSource::File(std::path::PathBuf::from("x")));
        let err = mlx_gen::gen_core::load_image_embedder(MODEL_ID, &spec)
            .err()
            .expect("bogus weights should fail to load");
        assert!(
            !format!("{err}").contains("no image embedder registered"),
            "embedder should be discovered by id, got: {err}"
        );
    }

    #[test]
    fn text_descriptor_advertises_clip_vit_l14_joint_space() {
        let d = text_descriptor();
        assert_eq!(d.id, "clip_vit_l14_text");
        assert_eq!(d.family, "text-embed");
        assert_eq!(d.embedding_dim, 768);
        assert_eq!(d.space, "clip-vit-l14");
        assert_eq!(d.backend, "mlx");
        assert!(d.mac_only);
    }

    #[test]
    fn text_registered_and_discoverable_by_id() {
        let spec = LoadSpec::new(WeightsSource::File(std::path::PathBuf::from("x")));
        let err = mlx_gen::gen_core::load_text_embedder(TEXT_MODEL_ID, &spec)
            .err()
            .expect("bogus weights should fail to load");
        assert!(
            !format!("{err}").contains("no text embedder registered"),
            "text embedder should be discovered by id, got: {err}"
        );
    }
}
