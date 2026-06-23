//! Boogu Qwen3-VL condition encoder: token embedding → all `num_layers` causal decoder layers →
//! final RMSNorm → **last_hidden_state** `[B, L, 4096]` (the per-token instruction features the DiT
//! caption embedder consumes). Differs from the ideogram TE only in the head: Boogu applies the
//! final norm and returns a single layer, vs ideogram's 13-layer pre-final-norm interleave.
//!
//! Two forwards: [`BooguTextEncoder::last_hidden`] (text-only, plain 1-D RoPE) and
//! [`BooguTextEncoder::last_hidden_with_image`] (Edit / E7b-2 — splices the vision tower's image
//! embeds at the `<|image_pad|>` positions, switches to the 3-D **interleaved MRoPE**, and injects
//! the deepstack features into LM layers 0/1/2 at the image positions).

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{add, concatenate_axis};
use mlx_rs::{Array, Dtype};

use mlx_gen::array::host_i32;
use mlx_gen::nn::{build_mask, TextRope, TokenEmbedding};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, BooguTextEncoderConfig, Qwen3DecoderLayer};

/// Qwen3-VL MRoPE section split (`text_config.rope_parameters.mrope_section`) — T/H/W frequency
/// counts over `head_dim/2 = 64`.
const MROPE_SECTION: [i32; 3] = [24, 20, 20];
/// Vision spatial merge (the LM sees one token per `merge²` patches).
const SPATIAL_MERGE: i32 = 2;

pub struct BooguTextEncoder {
    embed_tokens: TokenEmbedding,
    layers: Vec<Qwen3DecoderLayer>,
    rope: TextRope,
    final_norm: Array,
    eps: f32,
    head_dim: i32,
    rope_theta: f32,
}

impl BooguTextEncoder {
    /// Load from the `mllm` weights under `prefix` (`"model.language_model"`):
    /// `{prefix}.embed_tokens.weight`, `{prefix}.layers.{i}.…`, `{prefix}.norm.weight`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &BooguTextEncoderConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers as usize);
        for i in 0..cfg.num_layers {
            layers.push(Qwen3DecoderLayer::from_weights(
                w,
                &join(prefix, &format!("layers.{i}")),
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
        }
        Ok(Self {
            embed_tokens: crate::quant::embedding(w, &join(prefix, "embed_tokens"))?,
            layers,
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            final_norm: w.require(&join(prefix, "norm.weight"))?.clone(),
            eps: cfg.rms_norm_eps,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    /// Quantize every decoder-layer projection in place (group-wise Q4/Q8 at [`crate::quant::GROUP_SIZE`]
    /// = 32). The **token embedding stays dense**: its only quantizer (`TokenEmbedding::quantize`)
    /// hardcodes group 64 in shared gen-core, which would clash with the group-32 Linears under the
    /// single-group-size packed loader — and the embedding is a precision-sensitive lookup table
    /// (~1.2 GB bf16), a standard dense-keep. The per-layer norms + final norm also stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// `input_ids` / `attention_mask`: `[b, s]` int32. Returns `last_hidden_state` `[b, s, 4096]`
    /// (f32) — all layers run, final norm applied.
    pub fn last_hidden(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
        }
        Ok(rms_norm(&hidden, &self.final_norm, self.eps)?)
    }

    /// Single-reference image-conditioned forward (Edit) — a thin wrapper over
    /// [`Self::last_hidden_with_images`] for one reference image (`grid_thw`, `image_embeds`
    /// `[n, 4096]`, and its 3 `deepstack` features). Kept for single-reference call sites and the
    /// component-parity harness.
    pub fn last_hidden_with_image(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        image_embeds: &Array,
        deepstack: &[Array],
        grid_thw: [i32; 3],
        image_token_id: i32,
    ) -> Result<Array> {
        self.last_hidden_with_images(
            input_ids,
            attention_mask,
            std::slice::from_ref(image_embeds),
            std::slice::from_ref(&deepstack.to_vec()),
            std::slice::from_ref(&grid_thw),
            image_token_id,
        )
    }

    /// Multi-reference image-conditioned forward (Edit). Splices each reference's `image_embeds[k]`
    /// (`[n_k, 4096]`, the vision tower's merged output) into the token embeddings at the k-th
    /// contiguous `image_token_id` block (in input-id order), runs the 36 decoder layers under the
    /// 3-D **interleaved MRoPE** (each image's grid advancing the shared position counter), and injects
    /// each reference's `deepstack[k]` features at its image block after layers 0/1/2 — mirroring
    /// `Qwen3VLTextModel` with multiple `<|image_pad|>` blocks. `grids[k]` is image k's patch grid
    /// `[t, h, w]`. `b = 1`. The block order must match the reference order (the chat template emits
    /// the references' vision blocks before the instruction, in order).
    pub fn last_hidden_with_images(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        image_embeds: &[Array],
        deepstack: &[Vec<Array>],
        grids: &[[i32; 3]],
        image_token_id: i32,
    ) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let ids = host_i32(input_ids)?;

        // Contiguous `<|image_pad|>` blocks, in order; block k carries reference k.
        let blocks = image_blocks(&ids, image_token_id);
        if blocks.len() != image_embeds.len() {
            return Err(mlx_gen::Error::Msg(format!(
                "boogu image-conditioned encode: {} image-token blocks in input_ids but {} reference embeds",
                blocks.len(),
                image_embeds.len()
            )));
        }

        // Token embeddings, then splice each reference's vision embeds at its block. Each replacement
        // is the same length as the block it replaces, so earlier splices don't shift later indices.
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let dt = hidden.dtype();
        for (k, &(start, len)) in blocks.iter().enumerate() {
            let img = image_embeds[k].expand_dims(0)?.as_dtype(dt)?; // [1, n_k, 4096]
            hidden = replace_seq(&hidden, &img, start, start + len, s)?;
        }

        // 3-D interleaved MRoPE (per-image grids) + causal mask.
        let (pt, ph, pw) = mrope_positions(&ids, image_token_id, grids);
        let (cos, sin) = mrope_cos_sin(&pt, &ph, &pw, self.head_dim, self.rope_theta, dt)?;
        let mask = build_mask(attention_mask, b, s)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            // Deepstack: after LM layers 0/1/2, add each reference's layer-i feature at its block.
            for (k, &(start, len)) in blocks.iter().enumerate() {
                if i < deepstack[k].len() {
                    let ds = deepstack[k][i].expand_dims(0)?.as_dtype(dt)?; // [1, n_k, 4096]
                    let mid = add(&slice_seq(&hidden, start, start + len)?, &ds)?;
                    hidden = replace_seq(&hidden, &mid, start, start + len, s)?;
                }
            }
        }
        Ok(rms_norm(&hidden, &self.final_norm, self.eps)?)
    }
}

/// Contiguous runs of `image_token_id` in `ids`, returned as `(start, len)` in input-id order. Each
/// run is one reference image's `<|image_pad|>` block.
fn image_blocks(ids: &[i32], image_token_id: i32) -> Vec<(i32, i32)> {
    let mut blocks = Vec::new();
    let mut i = 0i32;
    let n = ids.len() as i32;
    while i < n {
        if ids[i as usize] == image_token_id {
            let start = i;
            while i < n && ids[i as usize] == image_token_id {
                i += 1;
            }
            blocks.push((start, i - start));
        } else {
            i += 1;
        }
    }
    blocks
}

/// Slice `[b, s, d]` along the sequence axis (axis 1) to `[start, end)`.
fn slice_seq(x: &Array, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), 1)?)
}

/// Replace `x[:, start:end, :]` with `repl` (`[b, end-start, d]`) via concat of the surrounding slices.
fn replace_seq(x: &Array, repl: &Array, start: i32, end: i32, s: i32) -> Result<Array> {
    let before = slice_seq(x, 0, start)?;
    let after = slice_seq(x, end, s)?;
    Ok(concatenate_axis(&[&before, repl, &after], 1)?)
}

/// 3-D MRoPE positions per token (mirrors `get_rope_index` + `get_vision_position_ids`): text tokens
/// advance `(i, i, i)`; the k-th image block (at running offset `cur`) takes `grids[k] = [t, h, w]`,
/// gets `t = cur`, `h = cur + row`, `w = cur + col` over its `(h/merge)×(w/merge)` merged grid, then
/// advances `cur += max(h, w) / merge`. Multiple image blocks consume `grids` in order.
fn mrope_positions(
    ids: &[i32],
    image_token_id: i32,
    grids: &[[i32; 3]],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let (mut pt, mut ph, mut pw) = (Vec::new(), Vec::new(), Vec::new());
    let mut cur = 0i32;
    let mut i = 0usize;
    let mut img_k = 0usize;
    while i < ids.len() {
        if ids[i] == image_token_id {
            let g = grids[img_k];
            let (llm_h, llm_w) = (g[1] / SPATIAL_MERGE, g[2] / SPATIAL_MERGE);
            let step = g[1].max(g[2]) / SPATIAL_MERGE;
            for idx in 0..(llm_h * llm_w) {
                pt.push(cur);
                ph.push(cur + idx / llm_w);
                pw.push(cur + idx % llm_w);
            }
            cur += step;
            i += (llm_h * llm_w) as usize;
            img_k += 1;
        } else {
            pt.push(cur);
            ph.push(cur);
            pw.push(cur);
            cur += 1;
            i += 1;
        }
    }
    (pt, ph, pw)
}

/// Build the interleaved-MRoPE `cos`/`sin` `[1, s, head_dim]` (cast to `dt`). Each of the `head_dim/2`
/// frequencies takes its position from the T/H/W axis per the interleave: within the first
/// `mrope_section[1]·3` indices, `j%3==1 → H`, `j%3==2 → W`, else `T` (the tail stays `T`).
fn mrope_cos_sin(
    pt: &[i32],
    ph: &[i32],
    pw: &[i32],
    head_dim: i32,
    theta: f32,
    dt: Dtype,
) -> Result<(Array, Array)> {
    let s = pt.len();
    let half = (head_dim / 2) as usize;
    let sec_h = (MROPE_SECTION[1] * 3) as usize;
    let sec_w = (MROPE_SECTION[2] * 3) as usize;
    let inv: Vec<f32> = (0..half)
        .map(|j| (theta as f64).powf(-(2.0 * j as f64) / head_dim as f64) as f32)
        .collect();

    let hd = head_dim as usize;
    let mut emb = vec![0f32; s * hd];
    for i in 0..s {
        for j in 0..half {
            let pos = if j < sec_h && j % 3 == 1 {
                ph[i]
            } else if j < sec_w && j % 3 == 2 {
                pw[i]
            } else {
                pt[i]
            };
            let angle = pos as f32 * inv[j];
            emb[i * hd + j] = angle;
            emb[i * hd + half + j] = angle; // emb = cat(freqs, freqs)
        }
    }
    let arr = Array::from_slice(&emb, &[1, s as i32, head_dim]);
    Ok((arr.cos()?.as_dtype(dt)?, arr.sin()?.as_dtype(dt)?))
}
