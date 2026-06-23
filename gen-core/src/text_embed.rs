//! The `TextEmbedder` contract: a single global text embedding (CLIP-style) per prompt, for
//! Dataset Doctor caption/image alignment (epic 6529 P2, sc-6537).
//!
//! Backend-neutral like every other gen-core contract: host types only (`&str`, `Vec<f32>`), no
//! backend tensors. MLX implements it in `mlx-gen-clip`, candle in `candle-gen-clip`. The returned
//! vector is **raw** (un-normalized); callers L2-normalize for cosine similarity.

use crate::Result;

/// A whole-text embedding provider (a CLIP-style text encoder).
///
/// No `Send`/`Sync` bound — matches [`ImageEmbedder`](crate::image_embed::ImageEmbedder) and the
/// MLX/candle provider pattern. The worker loads and runs the provider inside one blocking task.
pub trait TextEmbedder {
    /// Stable identity + advertised shape, constructible without loading weights.
    fn descriptor(&self) -> &TextEmbedderDescriptor;

    /// Embed one text string into its raw vector of length
    /// [`TextEmbedderDescriptor::embedding_dim`]. Callers L2-normalize for cosine similarity.
    fn embed_text(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed a batch of texts. The default maps [`embed_text`](Self::embed_text) over the slice; a
    /// provider can override with a batched forward. Order matches the input.
    fn embed_text_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.embed_text(text)).collect()
    }
}

/// A text embedder's stable identity + advertised shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextEmbedderDescriptor {
    /// Stable id (e.g. `"clip_vit_l14_text"`).
    pub id: &'static str,
    /// Provider family (`"text-embed"`).
    pub family: &'static str,
    /// Tensor backend that registered this embedder (`"mlx"` | `"candle"`); used by the worker's
    /// per-backend capability advertisement.
    pub backend: &'static str,
    /// Dimensionality of the returned embedding (768 for CLIP ViT-L/14).
    pub embedding_dim: usize,
    /// The embedding-space identifier (e.g. `"clip-vit-l14"`). Text and image vectors are only
    /// comparable when their `space` matches.
    pub space: &'static str,
    /// Whether this embedder only runs on macOS (the MLX implementation); candle sets this `false`.
    pub mac_only: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstTextEmbedder {
        descriptor: TextEmbedderDescriptor,
        value: Vec<f32>,
    }

    impl TextEmbedder for ConstTextEmbedder {
        fn descriptor(&self) -> &TextEmbedderDescriptor {
            &self.descriptor
        }

        fn embed_text(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(self.value.clone())
        }
    }

    #[test]
    fn embed_text_returns_the_raw_vector() {
        let embedder = ConstTextEmbedder {
            descriptor: TextEmbedderDescriptor {
                id: "test",
                family: "text-embed",
                backend: "mlx",
                embedding_dim: 3,
                space: "test-space",
                mac_only: true,
            },
            value: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(embedder.descriptor().embedding_dim, 3);
        assert_eq!(embedder.embed_text("caption").unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn default_embed_text_batch_maps_over_embed_text_preserving_order() {
        let embedder = ConstTextEmbedder {
            descriptor: TextEmbedderDescriptor {
                id: "test",
                family: "text-embed",
                backend: "mlx",
                embedding_dim: 2,
                space: "test-space",
                mac_only: true,
            },
            value: vec![0.5, 0.5],
        };
        let batch = embedder.embed_text_batch(&["one", "two"]).unwrap();
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|v| v == &vec![0.5, 0.5]));
    }
}
