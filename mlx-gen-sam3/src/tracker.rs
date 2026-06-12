//! SAM3 tracker — single-frame path (epic 4910, sc-4924, Phase F).
//!
//! The SAM3 tracker is **SAM2.1** architecture (mask decoder 256/2L/8H + dynamic-multimask via
//! stability; prompt encoder; memory bank for video). It is fed by the **shared PE vision encoder**
//! ([`crate::vision`]) + its own `tracker_neck` FPN (NOT a separate Hiera trunk). The checkpoint
//! stores it under `tracker_model.*` / `tracker_neck.*` in **`transformers`-5 module naming**, which
//! diverges from the original `facebookresearch/sam2` `.pt` naming `mlx-gen-sam2` was ported against
//! (`o_proj` vs `out_proj`, `upscale_conv1/2` vs `output_upscaling_*`, `mlp.proj_in/out` vs
//! `mlp.layers.*`, `feature_projection` is net-new, …). So this is a direct port mirroring the public
//! Apache-2.0 `transformers` reference (`modeling_sam3_tracker_video.py`), reusing this crate's own
//! NHWC primitives, rather than a key-translation over `mlx-gen-sam2`.
//!
//! This module is the **single-frame box-prompt** path (`tracker_neck` → prompt encoder → two-way
//! mask decoder), the SAM2-`Sam2Segmenter`-equivalent that the memory/video layer (F2) builds on.
//! Layout is NHWC end-to-end (matching [`crate::vision`] / [`crate::mask`]).

use std::f32::consts::PI;

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::nn::relu;
use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, conv_transpose2d, multiply, sigmoid, stack_axis,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{conv2d, gelu_exact, linear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::Sam3VisionConfig;
use crate::vision::{Backbone, FpnLayer};

/// Take a single index `i` along `axis`, dropping that axis.
fn take1(x: &Array, i: i32, axis: i32) -> Result<Array> {
    Ok(x.take_axis(Array::from_int(i), axis)?)
}

/// Slice `[start, end)` along `axis` (keeps the axis).
fn slice_axis(x: &Array, axis: i32, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[idx.len() as i32]), axis)?)
}

// --- fixed facebook/sam3 tracker hyperparameters (Sam3TrackerVideoMaskDecoderConfig) -------------
const HIDDEN: i32 = 256;
const NUM_HEADS: i32 = 8;
const NUM_MASK_TOKENS: i32 = 4; // num_multimask_outputs (3) + 1
const ATTN_DOWNSAMPLE: i32 = 2;
const LN_EPS: f32 = 1e-6;
const INPUT_SIZE: f32 = 1008.0; // image_size
                                // NB: `dynamic_multimask_via_stability` (stability_delta 0.05 / thresh 0.98) selects the best single
                                // mask on *no-prompt* video frames (multimask_output=false). That branch is exercised only by the
                                // memory/video layer (F2); the box-prompt PVS path here always requests multimask. Lands with F2.

fn join(prefix: &str, leaf: &str) -> String {
    format!("{prefix}.{leaf}")
}

/// Torch conv weight `[out, in, kH, kW]` (OIHW) → MLX `[out, kH, kW, in]` (OHWI).
fn conv_w_ohwi(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 1])?)
}

/// Torch transposed-conv weight `[in, out, kH, kW]` (IOHW) → MLX `[out, kH, kW, in]` (OHWI).
fn conv_transpose_w(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[1, 2, 3, 0])?)
}

fn weight_bias(w: &Weights, prefix: &str) -> Result<(Array, Array)> {
    Ok((
        w.require(&join(prefix, "weight"))?.clone(),
        w.require(&join(prefix, "bias"))?.clone(),
    ))
}

fn ln(x: &Array, p: &(Array, Array)) -> Result<Array> {
    Ok(layer_norm(x, Some(&p.0), Some(&p.1), LN_EPS)?)
}

// --- FeedForward (Sam3TrackerVideoFeedForward) ---------------------------------------------------

/// `proj_in → act → (layers.i → act)* → proj_out → [sigmoid]`. `act` is ReLU throughout the tracker.
struct FeedForward {
    proj_in: (Array, Array),
    layers: Vec<(Array, Array)>,
    proj_out: (Array, Array),
    sigmoid_output: bool,
}

impl FeedForward {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        num_layers: i32,
        sigmoid_output: bool,
    ) -> Result<Self> {
        let layers = (0..num_layers - 2)
            .map(|i| weight_bias(w, &join(prefix, &format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            proj_in: weight_bias(w, &join(prefix, "proj_in"))?,
            layers,
            proj_out: weight_bias(w, &join(prefix, "proj_out"))?,
            sigmoid_output,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = relu(&linear(x, &self.proj_in.0, &self.proj_in.1)?)?;
        for l in &self.layers {
            h = relu(&linear(&h, &l.0, &l.1)?)?;
        }
        h = linear(&h, &self.proj_out.0, &self.proj_out.1)?;
        if self.sigmoid_output {
            h = sigmoid(&h)?;
        }
        Ok(h)
    }
}

// --- Attention (Sam3TrackerVideoAttention, with q/k/v down-projection) ---------------------------

/// MHA on `[b, n, hidden]` tokens; q/k/v project to `internal = hidden / downsample`, split into
/// `NUM_HEADS`, SDPA, then `o_proj` back to `hidden`.
struct Attention {
    q: (Array, Array),
    k: (Array, Array),
    v: (Array, Array),
    o: (Array, Array),
    num_heads: i32,
}

impl Attention {
    fn from_weights(w: &Weights, prefix: &str, downsample: i32) -> Result<Self> {
        let _ = downsample; // head split is derived from the loaded projection width at forward time
        Ok(Self {
            q: weight_bias(w, &join(prefix, "q_proj"))?,
            k: weight_bias(w, &join(prefix, "k_proj"))?,
            v: weight_bias(w, &join(prefix, "v_proj"))?,
            o: weight_bias(w, &join(prefix, "o_proj"))?,
            num_heads: NUM_HEADS,
        })
    }

    fn forward(&self, q: &Array, k: &Array, v: &Array) -> Result<Array> {
        let sep = |x: &Array| -> Result<Array> {
            let sh = x.shape();
            let (b, n, c) = (sh[0], sh[1], sh[2]);
            Ok(x.reshape(&[b, n, self.num_heads, c / self.num_heads])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = sep(&linear(q, &self.q.0, &self.q.1)?)?;
        let k = sep(&linear(k, &self.k.0, &self.k.1)?)?;
        let v = sep(&linear(v, &self.v.0, &self.v.1)?)?;
        let scale = 1.0 / (q.shape()[3] as f32).sqrt();
        let out = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let sh = out.shape();
        let (b, h, n, c) = (sh[0], sh[1], sh[2], sh[3]);
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, n, h * c])?;
        linear(&out, &self.o.0, &self.o.1)
    }
}

// --- Two-way attention block + transformer (Sam3TrackerVideoTwoWayTransformer) -------------------

struct TwoWayBlock {
    self_attn: Attention,
    norm1: (Array, Array),
    cross_t2i: Attention,
    norm2: (Array, Array),
    mlp: FeedForward,
    norm3: (Array, Array),
    cross_i2t: Attention,
    norm4: (Array, Array),
    skip_first_layer_pe: bool,
}

impl TwoWayBlock {
    fn from_weights(w: &Weights, prefix: &str, skip_first_layer_pe: bool) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::from_weights(w, &join(prefix, "self_attn"), 1)?,
            norm1: weight_bias(w, &join(prefix, "layer_norm1"))?,
            cross_t2i: Attention::from_weights(
                w,
                &join(prefix, "cross_attn_token_to_image"),
                ATTN_DOWNSAMPLE,
            )?,
            norm2: weight_bias(w, &join(prefix, "layer_norm2"))?,
            mlp: FeedForward::from_weights(w, &join(prefix, "mlp"), 2, false)?,
            norm3: weight_bias(w, &join(prefix, "layer_norm3"))?,
            cross_i2t: Attention::from_weights(
                w,
                &join(prefix, "cross_attn_image_to_token"),
                ATTN_DOWNSAMPLE,
            )?,
            norm4: weight_bias(w, &join(prefix, "layer_norm4"))?,
            skip_first_layer_pe,
        })
    }

    /// `queries`/`keys`: `[b, nq, D]` / `[b, nk, D]`; `query_pe`/`key_pe`: same shapes.
    fn forward(
        &self,
        queries: &Array,
        keys: &Array,
        query_pe: &Array,
        key_pe: &Array,
    ) -> Result<(Array, Array)> {
        let mut queries = if self.skip_first_layer_pe {
            self.self_attn.forward(queries, queries, queries)?
        } else {
            let q = add(queries, query_pe)?;
            add(queries, &self.self_attn.forward(&q, &q, queries)?)?
        };
        queries = ln(&queries, &self.norm1)?;

        let q = add(&queries, query_pe)?;
        let k = add(keys, key_pe)?;
        queries = ln(
            &add(&queries, &self.cross_t2i.forward(&q, &k, keys)?)?,
            &self.norm2,
        )?;
        queries = ln(&add(&queries, &self.mlp.forward(&queries)?)?, &self.norm3)?;

        let q = add(&queries, query_pe)?;
        let k = add(keys, key_pe)?;
        let keys = ln(
            &add(keys, &self.cross_i2t.forward(&k, &q, &queries)?)?,
            &self.norm4,
        )?;
        Ok((queries, keys))
    }
}

struct TwoWayTransformer {
    layers: Vec<TwoWayBlock>,
    final_attn: Attention,
    norm_final: (Array, Array),
}

impl TwoWayTransformer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            layers: vec![
                TwoWayBlock::from_weights(w, &join(prefix, "layers.0"), true)?,
                TwoWayBlock::from_weights(w, &join(prefix, "layers.1"), false)?,
            ],
            final_attn: Attention::from_weights(
                w,
                &join(prefix, "final_attn_token_to_image"),
                ATTN_DOWNSAMPLE,
            )?,
            norm_final: weight_bias(w, &join(prefix, "layer_norm_final_attn"))?,
        })
    }

    /// `image_embedding`/`image_pe`: token-flattened `[b, hw, D]`; `point_embedding`: `[b, n, D]`.
    fn forward(
        &self,
        image_embedding: &Array,
        image_pe: &Array,
        point_embedding: &Array,
    ) -> Result<(Array, Array)> {
        let mut queries = point_embedding.clone();
        let mut keys = image_embedding.clone();
        for layer in &self.layers {
            let (q, k) = layer.forward(&queries, &keys, point_embedding, image_pe)?;
            queries = q;
            keys = k;
        }
        let q = add(&queries, point_embedding)?;
        let k = add(&keys, image_pe)?;
        queries = ln(
            &add(&queries, &self.final_attn.forward(&q, &k, &keys)?)?,
            &self.norm_final,
        )?;
        Ok((queries, keys))
    }
}

// --- Prompt encoder (Sam3TrackerVideoPromptEncoder, box path) ------------------------------------

/// `PositionEmbeddingRandom` (`shared_embedding`): `cat[sin, cos](2π · (2·coord−1) @ gaussian)`.
/// `gaussian` is `[2, HIDDEN/2]`. Coords are normalized to `[0,1]`.
struct PositionEmbeddingRandom {
    gaussian: Array, // [2, 128]
}

impl PositionEmbeddingRandom {
    /// `coords_norm`: `[..., 2]` in `[0,1]`. Returns `[..., HIDDEN]`.
    fn forward(&self, coords_norm: &Array) -> Result<Array> {
        let c = add(
            &coords_norm.multiply(Array::from_f32(2.0))?,
            Array::from_f32(-1.0),
        )?;
        // [..., 2] @ [2, 128] → [..., 128]
        let proj = c.matmul(&self.gaussian)?;
        let proj = proj.multiply(Array::from_f32(2.0 * PI))?;
        Ok(concatenate_axis(&[&proj.sin()?, &proj.cos()?], -1)?)
    }

    /// Dense positional grid for a `g×g` feature map → NHWC `[1, g, g, HIDDEN]` (each cell at its
    /// pixel center `(i+0.5)/g`).
    fn dense_pe(&self, g: i32) -> Result<Array> {
        let n = (g * g) as usize;
        let mut coords = vec![0f32; n * 2];
        for y in 0..g {
            for x in 0..g {
                let idx = (y * g + x) as usize * 2;
                coords[idx] = (x as f32 + 0.5) / g as f32;
                coords[idx + 1] = (y as f32 + 0.5) / g as f32;
            }
        }
        let coords = Array::from_slice(&coords, &[1, g, g, 2]);
        self.forward(&coords)
    }
}

/// SAM3 tracker prompt encoder — box path only (single-frame PVS). `point_embed[2]`/`[3]` are the
/// box-corner embeddings; the padding point uses `not_a_point_embed`. Dense embedding for a box-only
/// prompt is the broadcast `no_mask_embed`.
struct PromptEncoder {
    pe: PositionEmbeddingRandom,
    point_embed: Array,   // [4, 256]
    not_a_point: Array,   // [1, 256]
    no_mask_embed: Array, // [1, 256]
}

impl PromptEncoder {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            pe: PositionEmbeddingRandom {
                gaussian: w
                    .require(&join(prefix, "shared_embedding.positional_embedding"))?
                    .clone(),
            },
            point_embed: w.require(&join(prefix, "point_embed.weight"))?.clone(),
            not_a_point: w
                .require(&join(prefix, "not_a_point_embed.weight"))?
                .clone(),
            no_mask_embed: w.require(&join(prefix, "no_mask_embed.weight"))?.clone(),
        })
    }

    /// `box_xyxy` in **1008-input** pixel space → `(sparse [1, 1, 3, 256], dense [1, g, g, 256])`.
    /// Mirrors `_embed_boxes`: corners +0.5, normalize by the input size, pos-encode, add the
    /// corner-2/3 point embeddings, set the pad point to `not_a_point`.
    fn encode_box(&self, box_xyxy: [f32; 4], g: i32) -> Result<(Array, Array)> {
        // 3 corner coords (x0,y0),(x1,y1),(pad 0,0), normalized; pad stays 0.
        let norm = |v: f32| (v + 0.5) / INPUT_SIZE;
        let coords = [
            norm(box_xyxy[0]),
            norm(box_xyxy[1]),
            norm(box_xyxy[2]),
            norm(box_xyxy[3]),
            0.0,
            0.0,
        ];
        let coords = Array::from_slice(&coords, &[1, 1, 3, 2]);
        let emb = self.pe.forward(&coords)?; // [1,1,3,256]
                                             // corner 0 += point_embed[2], corner 1 += point_embed[3], corner 2 = not_a_point.
        let pe2 = take1(&self.point_embed, 2, 0)?.reshape(&[1, 1, 1, HIDDEN])?;
        let pe3 = take1(&self.point_embed, 3, 0)?.reshape(&[1, 1, 1, HIDDEN])?;
        let rows: Vec<Array> = vec![
            add(&take1(&emb, 0, 2)?.reshape(&[1, 1, 1, HIDDEN])?, &pe2)?,
            add(&take1(&emb, 1, 2)?.reshape(&[1, 1, 1, HIDDEN])?, &pe3)?,
            self.not_a_point.reshape(&[1, 1, 1, HIDDEN])?,
        ];
        let sparse = concatenate_axis(&rows.iter().collect::<Vec<_>>(), 2)?; // [1,1,3,256]

        let dense = broadcast_to(
            &self.no_mask_embed.reshape(&[1, 1, 1, HIDDEN])?,
            &[1, g, g, HIDDEN],
        )?;
        Ok((sparse, dense))
    }
}

// --- Mask decoder (Sam3TrackerVideoMaskDecoder) --------------------------------------------------

struct MaskDecoder {
    transformer: TwoWayTransformer,
    iou_token: Array,       // [1, 256]
    mask_tokens: Array,     // [4, 256]
    obj_score_token: Array, // [1, 256]
    upscale_conv1: (Array, Array),
    upscale_layer_norm: (Array, Array),
    upscale_conv2: (Array, Array),
    hypernet: Vec<FeedForward>,
    iou_head: FeedForward,
    obj_score_head: FeedForward,
}

impl MaskDecoder {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let hypernet = (0..NUM_MASK_TOKENS)
            .map(|i| {
                FeedForward::from_weights(
                    w,
                    &join(prefix, &format!("output_hypernetworks_mlps.{i}")),
                    3,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let (uc1w, uc1b) = weight_bias(w, &join(prefix, "upscale_conv1"))?;
        let (uc2w, uc2b) = weight_bias(w, &join(prefix, "upscale_conv2"))?;
        Ok(Self {
            transformer: TwoWayTransformer::from_weights(w, &join(prefix, "transformer"))?,
            iou_token: w.require(&join(prefix, "iou_token.weight"))?.clone(),
            mask_tokens: w.require(&join(prefix, "mask_tokens.weight"))?.clone(),
            obj_score_token: w.require(&join(prefix, "obj_score_token.weight"))?.clone(),
            upscale_conv1: (conv_transpose_w(&uc1w)?, uc1b),
            upscale_layer_norm: weight_bias(w, &join(prefix, "upscale_layer_norm"))?,
            upscale_conv2: (conv_transpose_w(&uc2w)?, uc2b),
            hypernet,
            iou_head: FeedForward::from_weights(w, &join(prefix, "iou_prediction_head"), 3, true)?,
            obj_score_head: FeedForward::from_weights(
                w,
                &join(prefix, "pred_obj_score_head"),
                3,
                false,
            )?,
        })
    }

    /// `image_embedding`: NHWC `[1, g, g, 256]`; `image_pe`: NHWC `[1, g, g, 256]`; `sparse`:
    /// `[1, 1, n, 256]`; `dense`: NHWC `[1, g, g, 256]`; `high_res`: `[feat_s0 (NHWC, 4g², 32),
    /// feat_s1 (NHWC, 2g², 64)]`. Returns `(masks [1, k, mg, mg], ious [1, k], obj_score [1, 1])`
    /// with `multimask_output` selecting the 3 multimask candidates (slice 1..).
    fn forward(
        &self,
        image_embedding: &Array,
        image_pe: &Array,
        sparse: &Array,
        dense: &Array,
        high_res: &[Array; 2],
        multimask_output: bool,
    ) -> Result<(Array, Array, Array)> {
        let g = image_embedding.shape()[1];
        // output tokens: [obj_score, iou, mask_tokens(4)] → [1, 6, 256]
        let out_tokens = concatenate_axis(
            &[&self.obj_score_token, &self.iou_token, &self.mask_tokens],
            0,
        )?
        .reshape(&[1, 1, 2 + NUM_MASK_TOKENS, HIDDEN])?;
        let sparse2 = sparse.reshape(&[1, 1, sparse.shape()[2], HIDDEN])?;
        let tokens = concatenate_axis(&[&out_tokens, &sparse2], 2)?.reshape(&[1, -1, HIDDEN])?; // [1, n_tok, 256]

        // image + dense, flattened to tokens [1, g*g, 256].
        let img = add(image_embedding, dense)?.reshape(&[1, g * g, HIDDEN])?;
        let img_pe = image_pe.reshape(&[1, g * g, HIDDEN])?;

        let (hs, src) = self.transformer.forward(&img, &img_pe, &tokens)?;
        let iou_token_out = take1(&hs, 1, 1)?.reshape(&[1, HIDDEN])?;
        let mask_tokens_out = slice_axis(&hs, 1, 2, 2 + NUM_MASK_TOKENS)?; // [1, 4, 256]
        let obj_score = self
            .obj_score_head
            .forward(&take1(&hs, 0, 1)?.reshape(&[1, HIDDEN])?)?; // [1, 1]

        // upscale: NHWC src [1, g, g, 256] → +feat_s1 (2g²) → +feat_s0 (4g²).
        let src = src.reshape(&[1, g, g, HIDDEN])?;
        let mut up = conv_transpose2d(&src, &self.upscale_conv1.0, (2, 2), None, None, None, None)?;
        up = add(&up, &self.upscale_conv1.1)?;
        up = add(&up, &high_res[1])?;
        up = gelu_exact(&ln(&up, &self.upscale_layer_norm)?)?;
        let mut up2 = conv_transpose2d(&up, &self.upscale_conv2.0, (2, 2), None, None, None, None)?;
        up2 = add(&up2, &self.upscale_conv2.1)?;
        up2 = gelu_exact(&add(&up2, &high_res[0])?)?; // [1, 4g, 4g, 32]
        let (mg, ch) = (up2.shape()[1], up2.shape()[3]);

        // hypernetwork: per mask token MLP → [1, 4, 32]; mask = hyper @ upscaled.
        let hyper: Vec<Array> = (0..NUM_MASK_TOKENS as usize)
            .map(|i| {
                self.hypernet[i]
                    .forward(&take1(&mask_tokens_out, i as i32, 1)?.reshape(&[1, HIDDEN])?)
            })
            .collect::<Result<Vec<_>>>()?;
        let hyper =
            stack_axis(&hyper.iter().collect::<Vec<_>>(), 1)?.reshape(&[1, NUM_MASK_TOKENS, ch])?;
        let up_flat = up2
            .reshape(&[1, (mg * mg), ch])?
            .transpose_axes(&[0, 2, 1])?; // [1, ch, mg²]
        let masks = hyper
            .matmul(&up_flat)?
            .reshape(&[1, NUM_MASK_TOKENS, mg, mg])?;
        let ious = self.iou_head.forward(&iou_token_out)?; // [1, 4]

        let (masks, ious) = if multimask_output {
            (
                slice_axis(&masks, 1, 1, NUM_MASK_TOKENS)?,
                slice_axis(&ious, 1, 1, NUM_MASK_TOKENS)?,
            )
        } else {
            // single-mask: dynamic multimask via stability is applied by the caller's policy; for the
            // box-prompt PVS path we always request multimask, so the else-arm returns mask 0.
            (slice_axis(&masks, 1, 0, 1)?, slice_axis(&ious, 1, 0, 1)?)
        };
        Ok((masks, ious, obj_score))
    }
}

// --- tracker neck (Sam3VisionNeck over tracker_neck.* + conv_s0/s1) ------------------------------

/// The tracker's FPN neck (same `FpnLayer` pyramid as the detector neck, separate `tracker_neck.*`
/// weights) plus the `conv_s0`/`conv_s1` high-res projections (which live under the mask decoder).
struct TrackerNeck {
    fpn_layers: Vec<FpnLayer>,
    conv_s0: (Array, Array), // 1×1 256→32
    conv_s1: (Array, Array), // 1×1 256→64
}

impl TrackerNeck {
    fn from_weights(
        w: &Weights,
        neck_prefix: &str,
        decoder_prefix: &str,
        cfg: &Sam3VisionConfig,
    ) -> Result<Self> {
        let fpn_layers = cfg
            .scale_factors
            .iter()
            .enumerate()
            .map(|(i, &scale)| {
                FpnLayer::from_weights(w, &join(neck_prefix, &format!("fpn_layers.{i}")), scale)
            })
            .collect::<Result<Vec<_>>>()?;
        let (s0w, s0b) = weight_bias(w, &join(decoder_prefix, "conv_s0"))?;
        let (s1w, s1b) = weight_bias(w, &join(decoder_prefix, "conv_s1"))?;
        Ok(Self {
            fpn_layers,
            conv_s0: (conv_w_ohwi(&s0w)?, s0b),
            conv_s1: (conv_w_ohwi(&s1w)?, s1b),
        })
    }

    /// `backbone`: NHWC `[1, g0, g0, C]` PE features. Returns `(image_embedding [1, g, g, 256],
    /// high_res [feat_s0 (4g²,32), feat_s1 (2g²,64)])`. The coarsest FPN level (36²) is dropped
    /// (`fpn_hidden_states[:-1]`).
    fn forward(&self, backbone: &Array) -> Result<(Array, [Array; 2])> {
        let fpn: Vec<Array> = self
            .fpn_layers
            .iter()
            .map(|l| l.forward(backbone))
            .collect::<Result<Vec<_>>>()?; // [288²,144²,72²,36²]
        let feat_s0 = conv2d(&fpn[0], &self.conv_s0.0, Some(&self.conv_s0.1), 1, 0)?; // 288², 32
        let feat_s1 = conv2d(&fpn[1], &self.conv_s1.0, Some(&self.conv_s1.1), 1, 0)?; // 144², 64
        let image_embedding = fpn[2].clone(); // 72², 256
        Ok((image_embedding, [feat_s0, feat_s1]))
    }
}

// --- memory encoder (Sam3TrackerVideoMemoryEncoder, F2) ------------------------------------------
//
// Encodes a frame's image features + its predicted mask into a 64-channel spatial memory + sine
// position encoding, stored in the memory bank for later frames. Mirrors `_encode_new_memory`
// (modeling_sam3_tracker_video.py ~2658) and the encoder forward (~1136). NHWC throughout; the
// reference is NCHW (channels-first `LayerNorm` ⇒ a plain LayerNorm over the channel axis here).

const SIGMOID_SCALE_FOR_MEM: f32 = 20.0;
const SIGMOID_BIAS_FOR_MEM: f32 = -10.0;
const MEM_OUT_CHANNELS: i32 = 64; // memory_encoder_output_channels
const MEM_POS_FEATS: i32 = MEM_OUT_CHANNELS / 2; // PositionEmbeddingSine num_position_features (32)
const MEM_SINE_TEMPERATURE: f32 = 10000.0;

/// Grouped 2-D conv on an **NHWC** input with an OHWI weight (+ channel bias). `groups == channels`
/// is the depthwise case (`memory_fuser` 7×7); everything else is `groups = 1`.
fn conv2d_g(
    x: &Array,
    w_ohwi: &Array,
    b: &Array,
    stride: i32,
    pad: i32,
    groups: i32,
) -> Result<Array> {
    let y = mlx_rs::ops::conv2d(x, w_ohwi, (stride, stride), (pad, pad), (1, 1), groups)?;
    Ok(add(&y, b)?)
}

/// Separable bilinear-resize weight matrix `W` `[out, in]` for `align_corners=False`
/// (`out = W @ in @ Wᵀ`). Matches `torch.nn.functional.interpolate(mode="bilinear")`; the SAM3 mask
/// prep is 1008²→1152² (**upsampling**, so the reference `antialias=True` is a documented no-op).
fn bilinear_resize_matrix(in_size: i32, out_size: i32) -> Array {
    let mut data = vec![0f32; (out_size * in_size) as usize];
    let scale = in_size as f32 / out_size as f32;
    for o in 0..out_size {
        // area_pixel source index (align_corners=False), clamped to [0, in-1].
        let src = ((o as f32 + 0.5) * scale - 0.5).clamp(0.0, (in_size - 1) as f32);
        let x0 = src.floor() as i32;
        let x1 = (x0 + 1).min(in_size - 1);
        let frac = src - x0 as f32;
        data[(o * in_size + x0) as usize] += 1.0 - frac;
        data[(o * in_size + x1) as usize] += frac;
    }
    Array::from_slice(&data, &[out_size, in_size])
}

/// `Sam3TrackerVideoPositionEmbeddingSine(num_position_features=32, normalize=True)` over a `g×g`
/// grid → NHWC `[1, g, g, 64]`. Channel layout is `cat(pos_y[32], pos_x[32])`; within each half the
/// `2k`/`2k+1` pair is `(sin, cos)` at frequency `10000^(k/16)`. Host-computed (weight-free).
fn position_embedding_sine(g: i32) -> Array {
    let num_pos = MEM_POS_FEATS; // 32
    let half = (num_pos / 2) as usize; // 16 frequencies
    let scale = 2.0 * PI;
    let eps = 1e-6f32;
    let denom = g as f32 + eps;
    let freqs: Vec<f32> = (0..half)
        .map(|k| MEM_SINE_TEMPERATURE.powf((2.0 * k as f32) / num_pos as f32))
        .collect();
    let ch = (2 * num_pos) as usize; // 64
    let mut data = vec![0f32; (g * g) as usize * ch];
    for y in 0..g {
        let ye = (y as f32 + 1.0) / denom * scale;
        for x in 0..g {
            let xe = (x as f32 + 1.0) / denom * scale;
            let base = ((y * g + x) as usize) * ch;
            for k in 0..half {
                let (py, px) = (ye / freqs[k], xe / freqs[k]);
                data[base + 2 * k] = py.sin();
                data[base + 2 * k + 1] = py.cos();
                data[base + num_pos as usize + 2 * k] = px.sin();
                data[base + num_pos as usize + 2 * k + 1] = px.cos();
            }
        }
    }
    Array::from_slice(&data, &[1, g, g, ch as i32])
}

/// `MaskDownSampler`: 4× (k3/s2/p1 conv → channels-first LayerNorm → GELU), channels 1→4→16→64→256,
/// then a 1×1 `final_conv` (256→256). Shrinks `[1,1152,1152,1]` → `[1,72,72,256]`. NHWC.
struct MaskDownSampler {
    layers: Vec<((Array, Array), (Array, Array))>, // (conv (OHWI,bias), layer_norm (w,b))
    final_conv: (Array, Array),
}

impl MaskDownSampler {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let layers = (0..4)
            .map(|i| -> Result<((Array, Array), (Array, Array))> {
                let lp = join(prefix, &format!("layers.{i}"));
                let (cw, cb) = weight_bias(w, &join(&lp, "conv"))?;
                Ok((
                    (conv_w_ohwi(&cw)?, cb),
                    weight_bias(w, &join(&lp, "layer_norm"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let (fw, fb) = weight_bias(w, &join(prefix, "final_conv"))?;
        Ok(Self {
            layers,
            final_conv: (conv_w_ohwi(&fw)?, fb),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for ((cw, cb), norm) in &self.layers {
            x = conv2d_g(&x, cw, cb, 2, 1, 1)?;
            x = gelu_exact(&ln(&x, norm)?)?;
        }
        conv2d_g(&x, &self.final_conv.0, &self.final_conv.1, 1, 0, 1)
    }
}

/// `MemoryFuserCXBlock`: ConvNeXt-style residual — 7×7 depthwise conv → channels-first LayerNorm →
/// 1×1 expand (256→1024) → GELU → 1×1 project (1024→256) → per-channel `scale` → +input. NHWC, so
/// the pointwise convs are last-axis linears with no permute.
struct CxBlock {
    depthwise: (Array, Array), // OHWI [256,7,7,1], depthwise (groups=256)
    norm: (Array, Array),
    pw1: (Array, Array),
    pw2: (Array, Array),
    scale: Array, // [256]
}

impl CxBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let (dw, db) = weight_bias(w, &join(prefix, "depthwise_conv"))?;
        Ok(Self {
            depthwise: (conv_w_ohwi(&dw)?, db),
            norm: weight_bias(w, &join(prefix, "layer_norm"))?,
            pw1: weight_bias(w, &join(prefix, "pointwise_conv1"))?,
            pw2: weight_bias(w, &join(prefix, "pointwise_conv2"))?,
            scale: w.require(&join(prefix, "scale"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = conv2d_g(x, &self.depthwise.0, &self.depthwise.1, 1, 3, HIDDEN)?; // groups == channels
        let h = ln(&h, &self.norm)?;
        let h = linear(
            &gelu_exact(&linear(&h, &self.pw1.0, &self.pw1.1)?)?,
            &self.pw2.0,
            &self.pw2.1,
        )?;
        let h = multiply(&self.scale, &h)?;
        Ok(add(x, &h)?)
    }
}

/// `Sam3TrackerVideoMemoryEncoder`: `mask_downsampler` + `feature_projection` (1×1 256→256) +
/// `memory_fuser` (2 CXBlocks) + `projection` (1×1 256→64), returning the 64-channel memory map +
/// its sine position encoding.
struct MemoryEncoder {
    mask_downsampler: MaskDownSampler,
    feature_projection: (Array, Array), // 1×1 256→256
    fuser: Vec<CxBlock>,
    projection: (Array, Array), // 1×1 256→64
}

impl MemoryEncoder {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let (fpw, fpb) = weight_bias(w, &join(prefix, "feature_projection"))?;
        let (pw, pb) = weight_bias(w, &join(prefix, "projection"))?;
        Ok(Self {
            mask_downsampler: MaskDownSampler::from_weights(w, &join(prefix, "mask_downsampler"))?,
            feature_projection: (conv_w_ohwi(&fpw)?, fpb),
            fuser: vec![
                CxBlock::from_weights(w, &join(prefix, "memory_fuser.layers.0"))?,
                CxBlock::from_weights(w, &join(prefix, "memory_fuser.layers.1"))?,
            ],
            projection: (conv_w_ohwi(&pw)?, pb),
        })
    }

    /// `pix_feat`: NHWC `[1,72,72,256]` raw image embedding; `mask_for_mem`: NHWC `[1,1152,1152,1]`
    /// scaled mask. Returns `(features [1,72,72,64], pos_enc [1,72,72,64])`.
    fn forward(&self, pix_feat: &Array, mask_for_mem: &Array) -> Result<(Array, Array)> {
        let masks = self.mask_downsampler.forward(mask_for_mem)?; // [1,72,72,256]
        let mut x = conv2d_g(
            pix_feat,
            &self.feature_projection.0,
            &self.feature_projection.1,
            1,
            0,
            1,
        )?;
        x = add(&x, &masks)?;
        for layer in &self.fuser {
            x = layer.forward(&x)?;
        }
        let features = conv2d_g(&x, &self.projection.0, &self.projection.1, 1, 0, 1)?; // [1,72,72,64]
        let pos = position_embedding_sine(features.shape()[1]);
        Ok((features, pos))
    }
}

// --- single-frame tracker -----------------------------------------------------------------------

/// SAM3 single-frame box-prompt tracker (PVS path). Reuses the shared PE [`Backbone`]; loads the
/// tracker neck + prompt encoder + mask decoder from `tracker_neck.*` / `tracker_model.*`.
pub struct Sam3Tracker {
    backbone: Backbone,
    neck: TrackerNeck,
    prompt: PromptEncoder,
    decoder: MaskDecoder,
    /// `tracker_model.shared_image_embedding` — a **separate** random-Gaussian table from the prompt
    /// encoder's `shared_embedding`; supplies the dense image positional encoding (the two are not
    /// the same matrix in the checkpoint — `get_image_wide_positional_embeddings`).
    image_pe_embed: PositionEmbeddingRandom,
    /// `tracker_model.no_memory_embedding` `[1, 1, 256]` — the learned "no memory yet" bias added to
    /// the image embedding on a frame with no memory conditioning (the single-frame / init path).
    /// On a memory-conditioned video frame this is replaced by memory attention (F2).
    no_memory_embedding: Array,
    /// `tracker_model.memory_encoder` — encodes a frame's image features + predicted mask into the
    /// 64-channel spatial memory stored in the memory bank (F2).
    memory_encoder: MemoryEncoder,
    /// `tracker_model.occlusion_spatial_embedding_parameter` `[1, 64]` — added to the spatial memory
    /// when the object is predicted absent (`object_score_logits ≤ 0`).
    occlusion: Array,
}

/// A frame's encoded memory: the 64-channel spatial feature map + its sine position encoding, both
/// NHWC `[1, 72, 72, 64]` (f32). The reference casts the features to bf16 for storage; that cast is
/// applied by the memory-bank layer, not here.
pub struct MemoryFeatures {
    pub features: Array,
    pub pos: Array,
}

const MASK_MEM_SIZE: i32 = 1152; // mask_input_size (4·72) · 4

/// A single-frame tracker prediction: the best (argmax-IoU) low-res mask logits + its IoU + the
/// object-score logit.
pub struct TrackerMask {
    /// Low-res mask logits `[mg, mg]` (f32) for the best candidate.
    pub low_res: Array,
    pub iou: f32,
    pub object_score: f32,
}

impl Sam3Tracker {
    /// Load from a `facebook/sam3` weight map (shared PE backbone + `tracker_neck` + `tracker_model`).
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let cfg = Sam3VisionConfig::sam3();
        Ok(Self {
            backbone: Backbone::from_weights(w, "detector_model.vision_encoder.backbone", &cfg)?,
            neck: TrackerNeck::from_weights(w, "tracker_neck", "tracker_model.mask_decoder", &cfg)?,
            prompt: PromptEncoder::from_weights(w, "tracker_model.prompt_encoder")?,
            decoder: MaskDecoder::from_weights(w, "tracker_model.mask_decoder")?,
            image_pe_embed: PositionEmbeddingRandom {
                gaussian: w
                    .require("tracker_model.shared_image_embedding.positional_embedding")?
                    .clone(),
            },
            no_memory_embedding: w.require("tracker_model.no_memory_embedding")?.clone(),
            memory_encoder: MemoryEncoder::from_weights(w, "tracker_model.memory_encoder")?,
            occlusion: w
                .require("tracker_model.occlusion_spatial_embedding_parameter")?
                .clone(),
        })
    }

    /// Mask prep for `_encode_new_memory`: resize the image-resolution mask logits to the 1152² mask
    /// memory size (separable bilinear, `align_corners=False`), then `sigmoid` (or `>0` binarize for
    /// point/box-prompted frames), then `·20 −10`. Returns NHWC `[1, 1152, 1152, 1]`.
    pub fn prepare_mask_for_mem(
        &self,
        pred_high_res: &Array,
        is_mask_from_pts: bool,
    ) -> Result<Array> {
        let sh = pred_high_res.shape();
        let (in_h, in_w) = (sh[sh.len() - 2], sh[sh.len() - 1]);
        let m = pred_high_res.reshape(&[in_h, in_w])?;
        let resized = if in_h == MASK_MEM_SIZE && in_w == MASK_MEM_SIZE {
            m
        } else {
            let wh = bilinear_resize_matrix(in_h, MASK_MEM_SIZE);
            let ww = bilinear_resize_matrix(in_w, MASK_MEM_SIZE);
            wh.matmul(&m)?.matmul(&ww.transpose_axes(&[1, 0])?)?
        };
        let prob = if is_mask_from_pts {
            resized.gt(Array::from_f32(0.0))?.as_dtype(Dtype::Float32)?
        } else {
            sigmoid(&resized)?
        };
        let scaled = add(
            &prob.multiply(Array::from_f32(SIGMOID_SCALE_FOR_MEM))?,
            Array::from_f32(SIGMOID_BIAS_FOR_MEM),
        )?;
        Ok(scaled.reshape(&[1, MASK_MEM_SIZE, MASK_MEM_SIZE, 1])?)
    }

    /// `_encode_new_memory`: encode a frame's raw image embedding + its predicted mask into spatial
    /// memory. `pix_feat`: NHWC `[1, 72, 72, 256]` raw image embedding (the [`Self::encode_frame`]
    /// first output — **no** `no_memory_embedding` bias). `pred_high_res`: `[1, 1, 1008, 1008]`
    /// image-resolution mask logits. `object_score`: the decoder object-score logit (drives the
    /// occlusion add). `is_mask_from_pts`: binarize vs sigmoid the mask (true for point/box frames).
    pub fn encode_new_memory(
        &self,
        pix_feat: &Array,
        pred_high_res: &Array,
        object_score: f32,
        is_mask_from_pts: bool,
    ) -> Result<MemoryFeatures> {
        let mask_for_mem = self.prepare_mask_for_mem(pred_high_res, is_mask_from_pts)?;
        let (mut features, pos) = self.memory_encoder.forward(pix_feat, &mask_for_mem)?;
        if object_score <= 0.0 {
            // object predicted absent → add the occlusion spatial embedding over the grid.
            features = add(
                &features,
                &self.occlusion.reshape(&[1, 1, 1, MEM_OUT_CHANNELS])?,
            )?;
        }
        Ok(MemoryFeatures { features, pos })
    }

    /// Encode a frame's pixels `[1, 3, 1008, 1008]` → `(image_embedding, high_res)`. Runs the shared
    /// PE backbone once; the detector path can run its own neck over the same backbone separately.
    pub fn encode_frame(&self, pixel_values: &Array) -> Result<(Array, [Array; 2])> {
        let backbone = self.backbone.forward(pixel_values)?;
        self.neck.forward(&backbone)
    }

    /// Box-prompt a pre-encoded frame: `box_xyxy` in **1008-input** space → best low-res mask.
    pub fn segment_encoded(
        &self,
        image_embedding: &Array,
        high_res: &[Array; 2],
        box_xyxy_1008: [f32; 4],
    ) -> Result<TrackerMask> {
        let g = image_embedding.shape()[1];
        // No-memory single-frame path: add the learned no-memory bias to the image embedding
        // (broadcast over the spatial grid). A memory-conditioned video frame skips this (F2).
        let image_embedding = add(
            image_embedding,
            &self.no_memory_embedding.reshape(&[1, 1, 1, HIDDEN])?,
        )?;
        let (sparse, dense) = self.prompt.encode_box(box_xyxy_1008, g)?;
        let image_pe = self.image_pe_embed.dense_pe(g)?;
        let (masks, ious, obj_score) =
            self.decoder
                .forward(&image_embedding, &image_pe, &sparse, &dense, high_res, true)?;
        // argmax IoU over the 3 candidates (host).
        let iv = ious
            .reshape(&[-1])?
            .as_dtype(Dtype::Float32)?
            .as_slice::<f32>()
            .to_vec();
        let best = argmax(&iv);
        let mg = masks.shape()[2];
        let low_res = take1(&take1(&masks, 0, 0)?, best as i32, 0)?
            .reshape(&[mg, mg])?
            .as_dtype(Dtype::Float32)?;
        Ok(TrackerMask {
            low_res,
            iou: iv[best],
            object_score: obj_score
                .reshape(&[-1])?
                .as_dtype(Dtype::Float32)?
                .as_slice::<f32>()[0],
        })
    }

    /// End-to-end single-frame: pixels + box (1008-input space) → best low-res mask.
    pub fn segment(&self, pixel_values: &Array, box_xyxy_1008: [f32; 4]) -> Result<TrackerMask> {
        let (emb, high_res) = self.encode_frame(pixel_values)?;
        self.segment_encoded(&emb, &high_res, box_xyxy_1008)
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}
