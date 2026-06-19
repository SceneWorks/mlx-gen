//! E7b-1: Boogu's **Qwen3-VL vision tower** — the ViT that turns a reference image into the merged
//! vision tokens (and 3 deepstack features) the MLLM consumes for image-conditioned editing.
//!
//! Port of `Qwen3VLVisionModel` (transformers `models/qwen3_vl/modeling_qwen3_vl.py`). Structure:
//!   - **Patch embed** — a `Conv3d` with kernel == stride == `[temporal 2, 16, 16]`; the kernel spans
//!     the whole patch so it is a per-patch matmul (fold `[embed, in, t, ph, pw]` → `[embed, in·t·ph·pw]`).
//!   - **Learned `pos_embed`** — an `nn.Embedding(num_position_embeddings, hidden)` (a
//!     `√n × √n` grid) **bilinearly interpolated** to the image grid (merge-grouped order) and added.
//!   - **`depth` blocks** — pre-`LayerNorm` (eps 1e-6) → full attention (fused-QKV + bias, 2-D NeoX
//!     rotary, per-image `cu_seqlens` blocks — single-image ⇒ full unmasked) → `proj`; pre-LayerNorm →
//!     **GELU-tanh** 2-linear MLP (`linear_fc1`/`linear_fc2`, bias). No windowing (unlike Qwen2.5-VL).
//!   - **Patch merger** — `LayerNorm` → concat `merge²` (=4) group → `Linear → GELU → Linear` → `out`.
//!   - **Deepstack** — at vision layers `deepstack_visual_indexes` ([8,16,24]), a post-shuffle-norm
//!     merger produces a feature the LM later injects into its early layers.
//!
//! The grid-derived host-side math (rope table, bilinear pos-embed indices/weights) mirrors the
//! reference `get_vision_position_ids` / `get_vision_bilinear_indices_and_weights` and is computed in
//! [`VisionTower::build_plan`]. Linears are [`AdaptableLinear`]s so they quantize at load.

pub mod preprocess;

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{add, concatenate_axis, matmul, multiply, softmax_axis, split};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::{gelu_exact, gelu_tanh};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::quant::lin;

const LN_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 10000.0;

/// Qwen3-VL vision-tower config (the `vision_config` block of `mllm/config.json`).
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub hidden_size: i32,
    pub num_heads: i32,
    pub intermediate_size: i32,
    pub depth: i32,
    pub out_hidden_size: i32,
    pub patch_size: i32,
    pub temporal_patch_size: i32,
    pub spatial_merge_size: i32,
    pub in_channels: i32,
    pub num_position_embeddings: i32,
    pub deepstack_visual_indexes: Vec<i32>,
}

impl VisionConfig {
    /// Boogu's Qwen3-VL-8B vision tower (`mllm/config.json::vision_config`).
    pub fn qwen3_vl() -> Self {
        Self {
            hidden_size: 1152,
            num_heads: 16,
            intermediate_size: 4304,
            depth: 27,
            out_hidden_size: 4096,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            num_position_embeddings: 2304,
            deepstack_visual_indexes: vec![8, 16, 24],
        }
    }

    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_heads
    }

    /// `spatial_merge_size²` — patches per merged token.
    fn merge_unit(&self) -> i32 {
        self.spatial_merge_size * self.spatial_merge_size
    }

    /// `√num_position_embeddings` — the learned pos-embed grid side.
    fn num_grid_per_side(&self) -> i32 {
        (self.num_position_embeddings as f64).sqrt() as i32
    }
}

/// LayerNorm in the model dtype (matches the reference's bf16 `nn.LayerNorm`).
fn ln(x: &Array, w: &Array, b: &Array) -> Result<Array> {
    Ok(layer_norm(x, Some(w), Some(b), LN_EPS)?)
}

/// HF half-split rotary `rotate_half`: `cat(-x[d/2:], x[:d/2])` on the last axis.
fn rotate_half(x: &Array) -> Result<Array> {
    let ax = x.ndim() as i32 - 1;
    let parts = split(x, 2, ax)?;
    Ok(concatenate_axis(&[&parts[1].negative()?, &parts[0]], ax)?)
}

/// One vision block: pre-LayerNorm full attention + pre-LayerNorm GELU-tanh MLP, both residual.
struct Block {
    norm1_w: Array,
    norm1_b: Array,
    norm2_w: Array,
    norm2_b: Array,
    qkv: AdaptableLinear,
    proj: AdaptableLinear,
    fc1: AdaptableLinear,
    fc2: AdaptableLinear,
}

impl Block {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let req = |s: &str| -> Result<Array> { Ok(w.require(&format!("{prefix}.{s}"))?.clone()) };
        Ok(Self {
            norm1_w: req("norm1.weight")?,
            norm1_b: req("norm1.bias")?,
            norm2_w: req("norm2.weight")?,
            norm2_b: req("norm2.bias")?,
            qkv: lin(w, &format!("{prefix}.attn.qkv"), true)?,
            proj: lin(w, &format!("{prefix}.attn.proj"), true)?,
            fc1: lin(w, &format!("{prefix}.mlp.linear_fc1"), true)?,
            fc2: lin(w, &format!("{prefix}.mlp.linear_fc2"), true)?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.qkv.quantize(bits, None)?;
        self.proj.quantize(bits, None)?;
        self.fc1.quantize(bits, None)?;
        self.fc2.quantize(bits, None)
    }

    /// Full attention over `x` `[seq, dim]` with precomputed `cos`/`sin` `[seq, head_dim]` (f32).
    /// Single-image (one `cu_seqlens` block) ⇒ unmasked. Manual `matmul → f32 softmax → matmul`,
    /// mirroring the reference `eager_attention_forward` (softmax accumulated in f32).
    fn attention(&self, x: &Array, cos: &Array, sin: &Array, nh: i32) -> Result<Array> {
        let seq = x.shape()[0];
        let dim = x.shape()[1];
        let hd = dim / nh;
        let dtype = x.dtype();

        let qkv = self.qkv.forward(x)?.reshape(&[seq, 3, nh, hd])?;
        let parts = split(&qkv, 3, 1)?; // 3 × [seq, 1, nh, hd]
        let q = parts[0].reshape(&[seq, nh, hd])?;
        let k = parts[1].reshape(&[seq, nh, hd])?;
        let v = parts[2].reshape(&[seq, nh, hd])?;

        // 2-D NeoX rope in f32; cos/sin broadcast over heads ([seq, 1, head_dim]).
        let cos = cos.reshape(&[seq, 1, hd])?;
        let sin = sin.reshape(&[seq, 1, hd])?;
        let rope = |t: &Array| -> Result<Array> {
            let f = t.as_dtype(Dtype::Float32)?;
            let r = add(&multiply(&f, &cos)?, &multiply(&rotate_half(&f)?, &sin)?)?;
            Ok(r.as_dtype(dtype)?)
        };
        // [seq, nh, hd] → [nh, seq, hd].
        let q = rope(&q)?.transpose_axes(&[1, 0, 2])?;
        let k = rope(&k)?.transpose_axes(&[1, 0, 2])?;
        let v = v.transpose_axes(&[1, 0, 2])?;

        let scale = (hd as f32).powf(-0.5);
        let scores = multiply(
            &matmul(&q, &k.transpose_axes(&[0, 2, 1])?)?,
            Array::from_f32(scale),
        )?; // [nh, seq, seq]
        let weights = softmax_axis(&scores, -1, true)?; // f32-accumulated softmax
        let o = matmul(&weights, &v)? // [nh, seq, hd]
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq, dim])?;
        self.proj.forward(&o)
    }

    fn mlp(&self, x: &Array) -> Result<Array> {
        self.fc2.forward(&gelu_tanh(&self.fc1.forward(x)?)?)
    }

    fn forward(&self, x: &Array, cos: &Array, sin: &Array, nh: i32) -> Result<Array> {
        let a = self.attention(&ln(x, &self.norm1_w, &self.norm1_b)?, cos, sin, nh)?;
        let x = add(x, &a)?;
        let m = self.mlp(&ln(&x, &self.norm2_w, &self.norm2_b)?)?;
        Ok(add(&x, &m)?)
    }
}

/// Patch merger: `LayerNorm` → concat `merge²` group → `linear_fc1 → GELU → linear_fc2`.
/// The main merger norms pre-shuffle (over `hidden`); deepstack mergers norm post-shuffle
/// (over `hidden·merge²`).
struct Merger {
    norm_w: Array,
    norm_b: Array,
    fc1: AdaptableLinear,
    fc2: AdaptableLinear,
    postshuffle: bool,
    merged_dim: i32, // hidden · merge²
}

impl Merger {
    fn from_weights(w: &Weights, prefix: &str, postshuffle: bool, merged_dim: i32) -> Result<Self> {
        Ok(Self {
            norm_w: w.require(&format!("{prefix}.norm.weight"))?.clone(),
            norm_b: w.require(&format!("{prefix}.norm.bias"))?.clone(),
            fc1: lin(w, &format!("{prefix}.linear_fc1"), true)?,
            fc2: lin(w, &format!("{prefix}.linear_fc2"), true)?,
            postshuffle,
            merged_dim,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.fc1.quantize(bits, None)?;
        self.fc2.quantize(bits, None)
    }

    /// `x` `[seq, hidden]` → `[merged, out_hidden]` (`merged = seq / merge²`).
    fn forward(&self, x: &Array, merged: i32) -> Result<Array> {
        let x = if self.postshuffle {
            // group merge-units first, then norm over hidden·merge².
            let g = x.reshape(&[merged, self.merged_dim])?;
            ln(&g, &self.norm_w, &self.norm_b)?
        } else {
            // norm over hidden per-patch, then group merge-units.
            let n = ln(x, &self.norm_w, &self.norm_b)?;
            n.reshape(&[merged, self.merged_dim])?
        };
        self.fc2.forward(&gelu_exact(&self.fc1.forward(&x)?)?)
    }
}

/// Host-side `grid_thw`-derived plan: the rope `cos`/`sin` (f32, merge-grouped order) and the
/// 4 bilinear corner indices + weights for the learned pos-embed interpolation.
struct Plan {
    merged: i32,
    cos: Array,               // f32 [seq, head_dim]
    sin: Array,               // f32 [seq, head_dim]
    bilinear_idx: [Array; 4], // i32 [seq]
    bilinear_w: [Array; 4],   // f32 [seq]
}

/// The native Qwen3-VL vision tower.
pub struct VisionTower {
    patch_embed: AdaptableLinear,
    pos_embed: Array, // [num_position_embeddings, hidden]
    blocks: Vec<Block>,
    merger: Merger,
    deepstack_mergers: Vec<Merger>,
    cfg: VisionConfig,
}

impl VisionTower {
    /// Build from the mllm weight set (`{prefix}.*`, e.g. `"model.visual"`).
    pub fn from_weights(w: &Weights, cfg: VisionConfig, prefix: &str) -> Result<Self> {
        // Fold the Conv3d patch-embed weight `[embed, in, t, ph, pw]` → `[embed, in·t·ph·pw]` so the
        // full-kernel conv runs as a per-patch matmul; keep its bias.
        let conv = w
            .require(&format!("{prefix}.patch_embed.proj.weight"))?
            .clone();
        let embed = conv.shape()[0];
        let in_dim = conv.shape().iter().skip(1).product::<i32>();
        let bias = w
            .require(&format!("{prefix}.patch_embed.proj.bias"))?
            .clone();
        let patch_embed = AdaptableLinear::dense(conv.reshape(&[embed, in_dim])?, Some(bias));

        let blocks = (0..cfg.depth)
            .map(|i| Block::from_weights(w, &format!("{prefix}.blocks.{i}")))
            .collect::<Result<Vec<_>>>()?;

        let merged_dim = cfg.hidden_size * cfg.merge_unit();
        let merger = Merger::from_weights(w, &format!("{prefix}.merger"), false, merged_dim)?;
        let deepstack_mergers = (0..cfg.deepstack_visual_indexes.len())
            .map(|i| {
                Merger::from_weights(
                    w,
                    &format!("{prefix}.deepstack_merger_list.{i}"),
                    true,
                    merged_dim,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            patch_embed,
            pos_embed: w.require(&format!("{prefix}.pos_embed.weight"))?.clone(),
            blocks,
            merger,
            deepstack_mergers,
            cfg,
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    /// Quantize every block + the (deepstack) mergers. LayerNorm / pos-embed weights stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for b in &mut self.blocks {
            b.quantize(bits)?;
        }
        self.merger.quantize(bits)?;
        for m in &mut self.deepstack_mergers {
            m.quantize(bits)?;
        }
        Ok(())
    }

    /// Host-side plan from `grid_thw` (rows `[t, h, w]` in patches), merge-grouped order — mirrors
    /// `get_vision_position_ids` (rope) + `get_vision_bilinear_indices_and_weights` (pos-embed).
    fn build_plan(&self, grid: &[[i32; 3]]) -> Plan {
        let c = &self.cfg;
        let m = c.spatial_merge_size;
        let hd = c.head_dim();
        let rd = (hd / 2) as usize; // rope width per token (= head_dim/2)
        let nfreq = rd / 2; // inv_freq length (= head_dim/4)
        let side = c.num_grid_per_side();
        let inv: Vec<f32> = (0..nfreq)
            .map(|j| ROPE_THETA.powf(-((2 * j) as f32) / rd as f32))
            .collect();

        let mut seq = 0i32;
        let mut merged = 0i32;
        let mut rope_rows: Vec<f32> = Vec::new(); // [seq, rd]
                                                  // bilinear corner indices + weights, merge-grouped order
        let mut idx: [Vec<i32>; 4] = [vec![], vec![], vec![], vec![]];
        let mut wts: [Vec<f32>; 4] = [vec![], vec![], vec![], vec![]];

        for g in grid {
            let (t, h, w) = (g[0], g[1], g[2]);
            seq += t * h * w;
            merged += t * (h / m) * (w / m);

            // linspace(0, side-1, n): value at index i.
            let lin = |i: i32, n: i32| -> f64 {
                if n <= 1 {
                    0.0
                } else {
                    (side - 1) as f64 * i as f64 / (n - 1) as f64
                }
            };

            for _f in 0..t {
                for bh in 0..(h / m) {
                    for bw in 0..(w / m) {
                        for ih in 0..m {
                            for iw in 0..m {
                                let hpos = bh * m + ih;
                                let wpos = bw * m + iw;
                                // rope: [hpos·inv(nfreq), wpos·inv(nfreq)] → rd
                                for &fq in &inv {
                                    rope_rows.push(hpos as f32 * fq);
                                }
                                for &fq in &inv {
                                    rope_rows.push(wpos as f32 * fq);
                                }
                                // bilinear pos-embed interpolation corners (into the side×side grid).
                                let hc = lin(hpos, h);
                                let wc = lin(wpos, w);
                                let hf = hc.floor();
                                let wf = wc.floor();
                                let h0 = hf as i32;
                                let w0 = wf as i32;
                                let h1 = (h0 + 1).min(side - 1);
                                let w1 = (w0 + 1).min(side - 1);
                                let hfr = (hc - hf) as f32;
                                let wfr = (wc - wf) as f32;
                                idx[0].push(h0 * side + w0);
                                idx[1].push(h0 * side + w1);
                                idx[2].push(h1 * side + w0);
                                idx[3].push(h1 * side + w1);
                                wts[0].push((1.0 - hfr) * (1.0 - wfr));
                                wts[1].push((1.0 - hfr) * wfr);
                                wts[2].push(hfr * (1.0 - wfr));
                                wts[3].push(hfr * wfr);
                            }
                        }
                    }
                }
            }
        }

        // rope table → cos/sin over the full head_dim (emb = cat(rope, rope)).
        let rope = Array::from_slice(&rope_rows, &[seq, rd as i32]);
        let emb = concatenate_axis(&[&rope, &rope], 1).unwrap(); // [seq, head_dim]
        let cos = emb.cos().unwrap();
        let sin = emb.sin().unwrap();

        let mk_i = |v: &[i32]| Array::from_slice(v, &[seq]);
        let mk_w = |v: &[f32]| Array::from_slice(v, &[seq, 1]);
        Plan {
            merged,
            cos,
            sin,
            bilinear_idx: [mk_i(&idx[0]), mk_i(&idx[1]), mk_i(&idx[2]), mk_i(&idx[3])],
            bilinear_w: [mk_w(&wts[0]), mk_w(&wts[1]), mk_w(&wts[2]), mk_w(&wts[3])],
        }
    }

    /// Bilinearly-interpolated learned pos-embed `[seq, hidden]` (f32) for the plan.
    fn pos_embeds(&self, plan: &Plan) -> Result<Array> {
        let pe = self.pos_embed.as_dtype(Dtype::Float32)?;
        let mut acc: Option<Array> = None;
        for k in 0..4 {
            let gathered = pe.take_axis(&plan.bilinear_idx[k], 0)?; // [seq, hidden]
            let term = multiply(&gathered, &plan.bilinear_w[k])?;
            acc = Some(match acc {
                Some(a) => add(&a, &term)?,
                None => term,
            });
        }
        Ok(acc.unwrap())
    }

    /// Encode packed patches → (merged image embeds `[merged, out_hidden]`, deepstack features —
    /// one `[merged, out_hidden]` per `deepstack_visual_indexes` entry).
    ///
    /// `pixel_values` is `[seq, in·t·ph·pw]`; `grid_thw` rows are `[t, h, w]` (patches).
    pub fn forward(
        &self,
        pixel_values: &Array,
        grid_thw: &[[i32; 3]],
    ) -> Result<(Array, Vec<Array>)> {
        let (embeds, deepstack, _prenorm) = self.encode(pixel_values, grid_thw)?;
        Ok((embeds, deepstack))
    }

    /// Like [`Self::forward`] but also returns the pre-merger ViT hidden `[seq, hidden]` (the
    /// reference `last_hidden_state`) for parity isolation.
    pub fn forward_debug(
        &self,
        pixel_values: &Array,
        grid_thw: &[[i32; 3]],
    ) -> Result<(Array, Vec<Array>, Array)> {
        self.encode(pixel_values, grid_thw)
    }

    fn encode(
        &self,
        pixel_values: &Array,
        grid_thw: &[[i32; 3]],
    ) -> Result<(Array, Vec<Array>, Array)> {
        let c = &self.cfg;
        let nh = c.num_heads;
        let plan = self.build_plan(grid_thw);
        let merged = plan.merged;

        // Patch embed + learned (interpolated) position embedding.
        let h = self.patch_embed.forward(pixel_values)?;
        let pos = self.pos_embeds(&plan)?.as_dtype(h.dtype())?;
        let mut h = add(&h, &pos)?;

        let mut deepstack = Vec::with_capacity(c.deepstack_visual_indexes.len());
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward(&h, &plan.cos, &plan.sin, nh)?;
            if let Some(di) = c
                .deepstack_visual_indexes
                .iter()
                .position(|&x| x == i as i32)
            {
                deepstack.push(self.deepstack_mergers[di].forward(&h, merged)?);
            }
        }

        let embeds = self.merger.forward(&h, merged)?;
        Ok((embeds, deepstack, h))
    }
}
