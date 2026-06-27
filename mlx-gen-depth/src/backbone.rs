//! DINOv2 ViT-S/14 backbone — the Depth Anything V2 encoder. Faithful port of the HF
//! `transformers` `Dinov2Backbone` (`modeling_dinov2.py`) for the `backbone.*` weight
//! tree of `depth-anything/Depth-Anything-V2-Small-hf`.
//!
//! Pipeline: `Conv2d` patch-embed (kernel=stride=14, NHWC) → prepend a learned CLS token + add the
//! learned absolute `position_embeddings` → 12 standard pre-norm transformer layers → a final
//! `layernorm`. Each layer is two residual sub-blocks: (a) LN → MHSA (separate Q/K/V linears, full
//! SDPA) → LayerScale; (b) LN → MLP (fc1 → GELU → fc2) → LayerScale.
//!
//! For the DPT neck this backbone returns the **per-layer output hidden states** of the
//! four `out_indices` layers ([3,6,9,12], 1-based → captured at layer-output indices
//! [2,5,8,11]). Each is `[B, grid²+1, hidden]` *including* the CLS token; the neck drops
//! the CLS token itself (matching `transformers`).
//!
//! Fixed-size note: the host preprocessor always feeds the default 518² square, so the
//! token grid is exactly 37×37 (1369 patches + 1 CLS = 1370) and the shipped
//! `position_embeddings` (length 1370) is added **directly** — no DINOv2 pos-embed
//! interpolation is needed (that path is only exercised at off-default resolutions, which
//! the preprocessor never produces).

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, multiply};
use mlx_rs::Array;

use mlx_gen::nn::{conv2d, gelu_exact, linear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::DepthAnythingConfig;
use crate::util::{conv_w_ohwi, join};

/// One DINOv2 transformer layer (`backbone.encoder.layer.{i}`).
struct Dinov2Layer {
    norm1_w: Array,
    norm1_b: Array,
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    out_w: Array,
    out_b: Array,
    ls1: Array, // layer_scale1.lambda1  [hidden]
    norm2_w: Array,
    norm2_b: Array,
    fc1_w: Array,
    fc1_b: Array,
    fc2_w: Array,
    fc2_b: Array,
    ls2: Array, // layer_scale2.lambda1  [hidden]
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
}

impl Dinov2Layer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &DepthAnythingConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            norm1_w: w.require(&p("norm1.weight"))?.clone(),
            norm1_b: w.require(&p("norm1.bias"))?.clone(),
            q_w: w.require(&p("attention.attention.query.weight"))?.clone(),
            q_b: w.require(&p("attention.attention.query.bias"))?.clone(),
            k_w: w.require(&p("attention.attention.key.weight"))?.clone(),
            k_b: w.require(&p("attention.attention.key.bias"))?.clone(),
            v_w: w.require(&p("attention.attention.value.weight"))?.clone(),
            v_b: w.require(&p("attention.attention.value.bias"))?.clone(),
            out_w: w.require(&p("attention.output.dense.weight"))?.clone(),
            out_b: w.require(&p("attention.output.dense.bias"))?.clone(),
            ls1: w.require(&p("layer_scale1.lambda1"))?.clone(),
            norm2_w: w.require(&p("norm2.weight"))?.clone(),
            norm2_b: w.require(&p("norm2.bias"))?.clone(),
            fc1_w: w.require(&p("mlp.fc1.weight"))?.clone(),
            fc1_b: w.require(&p("mlp.fc1.bias"))?.clone(),
            fc2_w: w.require(&p("mlp.fc2.weight"))?.clone(),
            fc2_b: w.require(&p("mlp.fc2.bias"))?.clone(),
            ls2: w.require(&p("layer_scale2.lambda1"))?.clone(),
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
            scale: (cfg.head_dim() as f32).powf(-0.5),
            eps: cfg.layer_norm_eps,
        })
    }

    /// `x`: `[B, N, C]` (N = 1 CLS + grid² patches) → `[B, N, C]`.
    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, n) = (sh[0], sh[1]);
        let (h, hd) = (self.num_heads, self.head_dim);

        // --- self-attention sub-block (pre-norm + LayerScale + residual) ---
        let hn = layer_norm(x, Some(&self.norm1_w), Some(&self.norm1_b), self.eps)?;
        let q = linear(&hn, &self.q_w, &self.q_b)?;
        let k = linear(&hn, &self.k_w, &self.k_b)?;
        let v = linear(&hn, &self.v_w, &self.v_b)?;
        let to_heads = |t: &Array| -> Result<Array> {
            Ok(t.reshape(&[b, n, h, hd])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let (q, k, v) = (to_heads(&q)?, to_heads(&k)?, to_heads(&v)?);
        let attn = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        // [B, heads, N, hd] -> [B, N, C]
        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, n, h * hd])?;
        let attn = linear(&attn, &self.out_w, &self.out_b)?;
        let attn = multiply(&attn, &self.ls1)?;
        let x = add(x, &attn)?;

        // --- MLP sub-block (pre-norm + LayerScale + residual) ---
        let hn = layer_norm(&x, Some(&self.norm2_w), Some(&self.norm2_b), self.eps)?;
        let y = linear(&hn, &self.fc1_w, &self.fc1_b)?;
        let y = gelu_exact(&y)?;
        let y = linear(&y, &self.fc2_w, &self.fc2_b)?;
        let y = multiply(&y, &self.ls2)?;
        Ok(add(&x, &y)?)
    }
}

/// The DINOv2 ViT backbone (patch-embed + CLS/pos embed + 12 layers + final LN).
pub struct Dinov2Backbone {
    proj_w: Array, // patch_embeddings.projection.weight, OHWI [embed, 14, 14, 3]
    proj_b: Array, // [embed]
    cls_token: Array,
    pos_embed: Array, // [1, grid²+1, embed]
    layers: Vec<Dinov2Layer>,
    final_ln_w: Array,
    final_ln_b: Array,
    cfg: DepthAnythingConfig,
}

impl Dinov2Backbone {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: DepthAnythingConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Dinov2Layer::from_weights(w, &p(&format!("encoder.layer.{i}")), &cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            proj_w: conv_w_ohwi(w.require(&p("embeddings.patch_embeddings.projection.weight"))?)?,
            proj_b: w
                .require(&p("embeddings.patch_embeddings.projection.bias"))?
                .clone(),
            cls_token: w.require(&p("embeddings.cls_token"))?.clone(),
            pos_embed: w.require(&p("embeddings.position_embeddings"))?.clone(),
            layers,
            final_ln_w: w.require(&p("layernorm.weight"))?.clone(),
            final_ln_b: w.require(&p("layernorm.bias"))?.clone(),
            cfg,
        })
    }

    pub fn config(&self) -> &DepthAnythingConfig {
        &self.cfg
    }

    /// `pixel_values`: NHWC `[B, H, W, 3]` (H=W=image_size, ImageNet-normalized) → the four
    /// captured hidden states (outputs of the `out_indices` layers), each `[B, grid²+1, hidden]`
    /// **including** the CLS token. The final `layernorm` is applied to the captured states (the
    /// DPT reassemble stage in `transformers` consumes the normalized hidden states for DA-V2).
    pub fn forward(&self, pixel_values: &Array) -> Result<Vec<Array>> {
        let sh = pixel_values.shape();
        let b = sh[0];
        let embed = self.cfg.hidden_size;

        // Patch embed: conv (stride=patch, no pad) → [B, g, g, embed] → [B, g², embed].
        let y = conv2d(
            pixel_values,
            &self.proj_w,
            Some(&self.proj_b),
            self.cfg.patch_size,
            0,
        )?;
        let g = y.shape()[1];
        let mut x = y.reshape(&[b, g * g, embed])?;

        // Prepend CLS, add absolute position embedding.
        let cls = broadcast_to(&self.cls_token, &[b, 1, embed])?;
        x = concatenate_axis(&[&cls, &x], 1)?;
        x = add(&x, &self.pos_embed)?;

        let capture = self.cfg.capture_layers();
        let mut out = Vec::with_capacity(4);
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if capture.contains(&(idx as i32)) {
                // DA-V2 applies the backbone's final LayerNorm to each captured state.
                let normed = layer_norm(
                    &x,
                    Some(&self.final_ln_w),
                    Some(&self.final_ln_b),
                    self.cfg.layer_norm_eps,
                )?;
                out.push(normed);
            }
        }
        Ok(out)
    }
}
