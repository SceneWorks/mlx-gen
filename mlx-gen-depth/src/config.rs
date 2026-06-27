//! Depth Anything V2 configuration — mirrors the HF `transformers`
//! `DepthAnythingConfig` (+ its DINOv2 `backbone_config`) for the
//! `depth-anything/Depth-Anything-V2-Small-hf` checkpoint (epic 8236, sc-8242).
//!
//! Only the **Small** (ViT-S/14) variant is wired as the default — the preprocessing
//! tier favors speed/size (the standard DA-V2 ControlNet-preprocessor choice). The
//! Base (ViT-B) and Large (ViT-L) checkpoints share the identical module graph and
//! differ only in these scalars, so they plug in by swapping the config (see
//! [`DepthAnythingConfig::small`]).

/// DINOv2 ViT backbone + DPT neck/head hyperparameters. Defaults are the shipped
/// `depth-anything/Depth-Anything-V2-Small-hf` values.
#[derive(Clone, Debug)]
pub struct DepthAnythingConfig {
    // --- backbone (DINOv2 ViT) ---
    /// Backbone embedding dim (384 for ViT-S).
    pub hidden_size: i32,
    /// Number of transformer layers (12).
    pub num_hidden_layers: i32,
    /// Attention heads (6); `head_dim = hidden_size / num_attention_heads` (64).
    pub num_attention_heads: i32,
    /// FFN expansion ratio (4 ⇒ intermediate = 1536).
    pub mlp_ratio: i32,
    /// Input channels (3).
    pub num_channels: i32,
    /// Default inference image size (518) → `image_size / patch_size` token grid (37).
    pub image_size: i32,
    /// Patch / conv-stem stride (14).
    pub patch_size: i32,
    /// LayerNorm epsilon (1e-6, the DINOv2 default).
    pub layer_norm_eps: f32,
    /// 1-based backbone layer indices whose **output** hidden states feed the neck
    /// (`out_indices` = [3, 6, 9, 12]). The reassemble stage consumes these four.
    pub out_indices: [i32; 4],

    // --- neck (DPT reassemble + fusion) ---
    /// Per-stage reassemble output channels (`neck_hidden_sizes` = [48, 96, 192, 384]).
    pub neck_hidden_sizes: [i32; 4],
    /// Per-stage spatial resize factors over the backbone token grid
    /// (`reassemble_factors` = [4.0, 2.0, 1.0, 0.5]): >1 → transposed-conv upsample,
    /// ==1 → identity, <1 → strided-conv downsample.
    pub reassemble_factors: [f32; 4],
    /// Channel dim every neck `conv` projects into and the fusion stage runs at
    /// (`fusion_hidden_size` = 64).
    pub fusion_hidden_size: i32,

    // --- head ---
    /// Penultimate head conv channel dim (`head_hidden_size` = 32).
    pub head_hidden_size: i32,
}

impl Default for DepthAnythingConfig {
    fn default() -> Self {
        Self::small()
    }
}

impl DepthAnythingConfig {
    /// The shipped `depth-anything/Depth-Anything-V2-Small-hf` (ViT-S/14) configuration.
    pub fn small() -> Self {
        Self {
            hidden_size: 384,
            num_hidden_layers: 12,
            num_attention_heads: 6,
            mlp_ratio: 4,
            num_channels: 3,
            image_size: 518,
            patch_size: 14,
            layer_norm_eps: 1e-6,
            out_indices: [3, 6, 9, 12],
            neck_hidden_sizes: [48, 96, 192, 384],
            reassemble_factors: [4.0, 2.0, 1.0, 0.5],
            fusion_hidden_size: 64,
            head_hidden_size: 32,
        }
    }

    /// `head_dim = hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }

    /// FFN intermediate dim (`hidden_size * mlp_ratio`).
    pub fn intermediate_size(&self) -> i32 {
        self.hidden_size * self.mlp_ratio
    }

    /// Token grid side for the **default** image size (`image_size / patch_size` = 37).
    pub fn grid(&self) -> i32 {
        self.image_size / self.patch_size
    }

    /// Zero-based backbone layer indices whose output the neck consumes
    /// (`out_indices` are 1-based; the captured hidden is the *output* of that layer).
    pub fn capture_layers(&self) -> [i32; 4] {
        [
            self.out_indices[0] - 1,
            self.out_indices[1] - 1,
            self.out_indices[2] - 1,
            self.out_indices[3] - 1,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_geometry_is_the_shipped_vits() {
        let c = DepthAnythingConfig::small();
        assert_eq!(c.head_dim(), 64, "384 / 6 = 64");
        assert_eq!(c.intermediate_size(), 1536, "384 * 4 = 1536");
        assert_eq!(c.grid(), 37, "518 / 14 = 37");
        assert_eq!(
            c.grid() * c.grid() + 1,
            1370,
            "37² + 1 = 1370 (the pos-embed length)"
        );
        assert_eq!(
            c.capture_layers(),
            [2, 5, 8, 11],
            "1-based out_indices [3,6,9,12] → zero-based [2,5,8,11]"
        );
    }
}
