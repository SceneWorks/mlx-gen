//! SAM3 mask head + instance post-processing — porting `Sam3MaskDecoder` / `Sam3PixelDecoder` /
//! `Sam3MaskEmbedder` and `post_process_instance_segmentation` (epic 4910, sc-4922).
//!
//! MaskFormer-style: the 200 decoder queries are embedded and dot-producted (`einsum bqc,bchw→bqhw`)
//! against a pixel-embedding FPN built from the backbone features (with the DETR-encoded 72² level
//! swapped in for the finest) to yield per-query masks; a 1×1 conv gives the semantic map. Prompt
//! (text) cross-attention conditions the pixel features. Layout NHWC. Post-process scores each query
//! `σ(logits)·σ(presence)`, keeps `> threshold`, and binarizes `σ(mask) > 0.5`.

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::nn::relu;
use mlx_rs::ops::{add, matmul, sigmoid};
use mlx_rs::Array;

use mlx_gen::nn::{conv2d, group_norm, linear, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::Sam3DetrConfig;

const LN_EPS: f32 = 1e-5; // nn.LayerNorm / GroupNorm default eps in the mask decoder
const NUM_GROUPS: i32 = 8;

fn join(prefix: &str, leaf: &str) -> String {
    format!("{prefix}.{leaf}")
}

/// Torch conv weight `[out, in, kH, kW]` (OIHW) → MLX `[out, kH, kW, in]` (OHWI).
fn conv_w(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 1])?)
}

/// Prompt cross-attention (text-conditioned), `[B, Nq, D]` query over `[B, Nk, D]` key/value.
struct PromptAttn {
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    o_w: Array,
    o_b: Array,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl PromptAttn {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3DetrConfig) -> Result<Self> {
        let g = |n: &str| -> Result<Array> { Ok(w.require(&join(prefix, n))?.clone()) };
        let head_dim = cfg.head_dim();
        Ok(Self {
            q_w: g("q_proj.weight")?,
            q_b: g("q_proj.bias")?,
            k_w: g("k_proj.weight")?,
            k_b: g("k_proj.bias")?,
            v_w: g("v_proj.weight")?,
            v_b: g("v_proj.bias")?,
            o_w: g("o_proj.weight")?,
            o_b: g("o_proj.bias")?,
            num_heads: cfg.num_attention_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, query: &Array, kv: &Array, mask: &Array) -> Result<Array> {
        let b = query.shape()[0];
        let (nq, nk) = (query.shape()[1], kv.shape()[1]);
        let (nh, hd) = (self.num_heads, self.head_dim);
        let heads = |t: Array, n: i32| -> Result<Array> {
            Ok(t.reshape(&[b, n, nh, hd])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = heads(linear(query, &self.q_w, &self.q_b)?, nq)?;
        let k = heads(linear(kv, &self.k_w, &self.k_b)?, nk)?;
        let v = heads(linear(kv, &self.v_w, &self.v_b)?, nk)?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, mask, None)?;
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, nq, nh * hd])?;
        linear(&o, &self.o_w, &self.o_b)
    }
}

/// 3-layer ReLU MLP embedding the queries for mask prediction (`Sam3MaskEmbedder`).
struct MaskEmbedder {
    layers: Vec<(Array, Array)>,
}

impl MaskEmbedder {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let layers = (0..3)
            .map(|i| -> Result<(Array, Array)> {
                Ok((
                    w.require(&join(prefix, &format!("layers.{i}.weight")))?
                        .clone(),
                    w.require(&join(prefix, &format!("layers.{i}.bias")))?
                        .clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = x.clone();
        for (i, (w, b)) in self.layers.iter().enumerate() {
            h = linear(&h, w, b)?;
            if i < 2 {
                h = relu(&h)?;
            }
        }
        Ok(h)
    }
}

/// FPN pixel decoder (`Sam3PixelDecoder`): coarse→fine, nearest-upsample + skip-add + conv/GN/ReLU.
struct PixelDecoder {
    convs: Vec<(Array, Array)>, // OHWI conv3×3 + bias
    norms: Vec<(Array, Array)>, // GroupNorm weight/bias
}

impl PixelDecoder {
    fn from_weights(w: &Weights, prefix: &str, stages: usize) -> Result<Self> {
        let convs = (0..stages)
            .map(|i| -> Result<(Array, Array)> {
                Ok((
                    conv_w(w.require(&join(prefix, &format!("conv_layers.{i}.weight")))?)?,
                    w.require(&join(prefix, &format!("conv_layers.{i}.bias")))?
                        .clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let norms = (0..stages)
            .map(|i| -> Result<(Array, Array)> {
                Ok((
                    w.require(&join(prefix, &format!("norms.{i}.weight")))?
                        .clone(),
                    w.require(&join(prefix, &format!("norms.{i}.bias")))?
                        .clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { convs, norms })
    }

    /// `features`: NHWC, fine→coarse `[288², 144², 72²]`. Returns the finest pixel embedding NHWC.
    fn forward(&self, features: &[Array]) -> Result<Array> {
        let mut prev = features[features.len() - 1].clone(); // coarsest (72²)
                                                             // iterate the remaining levels coarse→fine: 144², then 288²
        for (layer_idx, feat) in features[..features.len() - 1].iter().rev().enumerate() {
            prev = upsample_nearest(&prev, 2)?; // exact 2× (72→144→288)
            prev = add(&prev, feat)?;
            let (cw, cb) = &self.convs[layer_idx];
            prev = conv2d(&prev, cw, Some(cb), 1, 1)?; // 3×3 pad 1
            let (nw, nb) = &self.norms[layer_idx];
            prev = group_norm(&prev, nw, nb, NUM_GROUPS, LN_EPS)?;
            prev = relu(&prev)?;
        }
        Ok(prev)
    }
}

/// SAM3 mask head: prompt-conditioned pixel features + query embeddings → per-instance masks.
pub struct Sam3MaskHead {
    pixel_decoder: PixelDecoder,
    mask_embedder: MaskEmbedder,
    instance_proj_w: Array, // OHWI 1×1
    instance_proj_b: Array,
    semantic_proj_w: Array, // OHWI 1×1 (→ 1 channel)
    semantic_proj_b: Array,
    prompt_attn: PromptAttn,
    prompt_norm_w: Array,
    prompt_norm_b: Array,
}

/// Mask-head outputs.
pub struct MaskOutput {
    /// `[1, Q, 288, 288]` per-query mask logits.
    pub pred_masks: Array,
    /// `[1, 288, 288, 1]` semantic-segmentation logits (NHWC).
    pub semantic_seg: Array,
}

impl Sam3MaskHead {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3DetrConfig) -> Result<Self> {
        let p = join(prefix, "mask_decoder");
        Ok(Self {
            pixel_decoder: PixelDecoder::from_weights(w, &join(&p, "pixel_decoder"), 3)?,
            mask_embedder: MaskEmbedder::from_weights(w, &join(&p, "mask_embedder"))?,
            instance_proj_w: conv_w(w.require(&join(&p, "instance_projection.weight"))?)?,
            instance_proj_b: w.require(&join(&p, "instance_projection.bias"))?.clone(),
            semantic_proj_w: conv_w(w.require(&join(&p, "semantic_projection.weight"))?)?,
            semantic_proj_b: w.require(&join(&p, "semantic_projection.bias"))?.clone(),
            prompt_attn: PromptAttn::from_weights(w, &join(&p, "prompt_cross_attn"), cfg)?,
            prompt_norm_w: w
                .require(&join(&p, "prompt_cross_attn_norm.weight"))?
                .clone(),
            prompt_norm_b: w.require(&join(&p, "prompt_cross_attn_norm.bias"))?.clone(),
        })
    }

    /// `query_hidden`: `[1, Q, D]`; `backbone_features`: NHWC fine→coarse `[288²,144²,72²]`;
    /// `encoder_hidden`: `[1, 72², D]` (DETR-encoded 72² level); `prompt`: text `[1, L, D]`.
    pub fn forward(
        &self,
        query_hidden: &Array,
        backbone_features: &[Array],
        encoder_hidden: &Array,
        prompt: &Array,
        prompt_key_mask: &Array,
    ) -> Result<MaskOutput> {
        // prompt cross-attention: encoder features attend to text prompt
        let normed = layer_norm(
            encoder_hidden,
            Some(&self.prompt_norm_w),
            Some(&self.prompt_norm_b),
            LN_EPS,
        )?;
        let attn = self.prompt_attn.forward(&normed, prompt, prompt_key_mask)?;
        let enc = add(encoder_hidden, &attn)?; // [1, 72², D]

        // swap the encoded 72² level in for the finest backbone level, then run the FPN
        let coarse = backbone_features.last().unwrap();
        let (h, w) = (coarse.shape()[1], coarse.shape()[2]);
        let d = coarse.shape()[3];
        let enc_spatial = enc.reshape(&[1, h, w, d])?;
        let mut feats: Vec<Array> = backbone_features.to_vec();
        *feats.last_mut().unwrap() = enc_spatial;
        let pixel_embed = self.pixel_decoder.forward(&feats)?; // NHWC [1, 288, 288, D]

        // instance masks: dot product of query mask-embeddings with the projected pixel embedding
        let instance = conv2d(
            &pixel_embed,
            &self.instance_proj_w,
            Some(&self.instance_proj_b),
            1,
            0,
        )?; // [1,288,288,D]
        let mask_emb = self.mask_embedder.forward(query_hidden)?; // [1, Q, D]
        let (ph, pw) = (instance.shape()[1], instance.shape()[2]);
        let inst_flat = instance
            .reshape(&[1, ph * pw, d])?
            .transpose_axes(&[0, 2, 1])?; // [1, D, HW]
        let pred_masks =
            matmul(&mask_emb, &inst_flat)?.reshape(&[1, mask_emb.shape()[1], ph, pw])?;

        let semantic_seg = conv2d(
            &pixel_embed,
            &self.semantic_proj_w,
            Some(&self.semantic_proj_b),
            1,
            0,
        )?;
        Ok(MaskOutput {
            pred_masks,
            semantic_seg,
        })
    }
}

/// One detected instance from the post-process.
pub struct Instance {
    /// `σ(logit)·σ(presence)` confidence.
    pub score: f32,
    /// Query index into the 200 queries.
    pub query: usize,
    /// Box xyxy in pixels at `target_size`.
    pub box_xyxy: [f32; 4],
    /// Binary mask `[h, w]` (0/1) at the mask-head resolution (288²) — caller resizes to the image.
    pub mask: Array,
}

/// `post_process_instance_segmentation`: keep queries whose `σ(logits)·σ(presence) > threshold`,
/// binarize `σ(mask) > mask_threshold`. Boxes (xyxy∈[0,1]) are scaled to `target_wh`. Masks are
/// returned at the native 288² resolution (resize-to-image is the caller's concern).
pub fn post_process_instances(
    pred_logits: &Array,     // [1, Q]
    pred_boxes: &Array,      // [1, Q, 4] xyxy ∈ [0,1]
    presence_logits: &Array, // [1, 1]
    pred_masks: &Array,      // [1, Q, h, w]
    target_wh: (f32, f32),
    threshold: f32,
    mask_threshold: f32,
) -> Result<Vec<Instance>> {
    let presence = sigmoid(presence_logits)?.item::<f32>();
    let scores: Vec<f32> = sigmoid(pred_logits)?
        .as_slice::<f32>()
        .iter()
        .map(|&s| s * presence)
        .collect();
    let boxes: Vec<f32> = pred_boxes.as_slice::<f32>().to_vec();
    let (tw, th) = target_wh;

    let mut out = Vec::new();
    for (qi, &score) in scores.iter().enumerate() {
        if score <= threshold {
            continue;
        }
        let b = &boxes[qi * 4..qi * 4 + 4];
        let mask_logits = pred_masks.take_axis(Array::from_slice(&[qi as i32], &[1]), 1)?; // [1,1,h,w]
        let mask = sigmoid(&mask_logits)?
            .gt(Array::from_f32(mask_threshold))?
            .as_dtype(mlx_rs::Dtype::Uint8)?
            .reshape(&[pred_masks.shape()[2], pred_masks.shape()[3]])?;
        out.push(Instance {
            score,
            query: qi,
            box_xyxy: [b[0] * tw, b[1] * th, b[2] * tw, b[3] * th],
            mask,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_process_filters_by_score_and_binarizes() {
        // presence ~1; query 0 logit high (kept), query 1 low (dropped).
        let presence = Array::from_slice(&[10.0f32], &[1, 1]);
        let logits = Array::from_slice(&[8.0f32, -8.0], &[1, 2]);
        let boxes = Array::from_slice(&[0.0f32, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0], &[1, 2, 4]);
        // [1,2,2,2] masks: query 0 = all positive → all-1; query 1 = all negative.
        let masks = Array::from_slice(
            &[5.0f32, 5.0, 5.0, 5.0, -5.0, -5.0, -5.0, -5.0],
            &[1, 2, 2, 2],
        );
        let inst =
            post_process_instances(&logits, &boxes, &presence, &masks, (100.0, 200.0), 0.5, 0.5)
                .unwrap();
        assert_eq!(inst.len(), 1, "only query 0 passes threshold");
        assert_eq!(inst[0].query, 0);
        assert!(inst[0].score > 0.99);
        // box scaled by target_wh (100,200): [0,0,0.5,0.5] → [0,0,50,100]
        assert_eq!(inst[0].box_xyxy, [0.0, 0.0, 50.0, 100.0]);
        // mask binarized to all-1 (4 px)
        let s = inst[0].mask.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        assert_eq!(mlx_rs::ops::sum(&s, None).unwrap().item::<f32>(), 4.0);
    }
}
