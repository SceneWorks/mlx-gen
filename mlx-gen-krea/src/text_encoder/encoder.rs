//! The Krea Qwen3-VL-4B text encoder forward: token embedding → causal Qwen3 decoder layers,
//! capturing the hidden states at `select_hidden` and **stacking** them on a new axis →
//! `[B, L, num_select, hidden]`, then dropping the leading template-prefix tokens. This is the exact
//! `context` the DiT's `TextFusionTransformer` consumes (sc-7568) — the aggregation happens there, not
//! here.
//!
//! HF `hidden_states` indexing: `hidden_states[i]` is the state after running `i` decoder layers
//! (`hidden_states[0]` = the raw embedding). So the reference's `select_hidden = [2,5,…,35]` capture
//! the OUTPUT of 0-indexed layers `[1,4,…,34]`. The final `language_model.norm` is never applied (all
//! selected layers are pre-final-norm), and only `max+1` layers are run (later layers can't matter).

use mlx_rs::ops::{add, concatenate_axis};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{build_mask, TextRope, TokenEmbedding};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use super::{embedding, join, KreaTeConfig, Qwen3DecoderLayer};

/// Qwen3-VL spatial merge factor (`spatial_merge_size`); a `merge×merge` block of ViT patches collapses
/// to one LM image token. Fixed at 2 across the family.
const SPATIAL_MERGE: i32 = 2;

pub struct KreaTextEncoder {
    embed_tokens: TokenEmbedding,
    layers: Vec<Qwen3DecoderLayer>,
    rope: TextRope,
    /// 0-indexed decoder-layer OUTPUT indices to capture (= `select_hidden[i] - 1`), in stack order.
    out_layers: Vec<usize>,
    prefix_tokens: i32,
    /// Image-grounded (edit) encoding params (epic 10871 P2): the `<|image_pad|>` id whose positions the
    /// vision features replace, the interleaved-MRoPE section widths, and the head-dim/θ the MRoPE
    /// `cos`/`sin` are built from. Unused by the text-only [`forward`](Self::forward).
    image_token_id: i32,
    mrope_section: [i32; 3],
    head_dim: i32,
    rope_theta: f32,
}

impl KreaTextEncoder {
    /// Load from the `text_encoder` weights under `prefix` (`"language_model"`):
    /// `{prefix}.embed_tokens.weight`, `{prefix}.layers.{i}.…`. The final `{prefix}.norm.weight` is
    /// intentionally not loaded.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &KreaTeConfig) -> Result<Self> {
        let out_layers: Vec<usize> = cfg
            .select_hidden
            .iter()
            .map(|&s| {
                s.checked_sub(1).ok_or_else(|| {
                    Error::Msg("krea te: select_hidden index 0 has no layer output".into())
                })
            })
            .collect::<Result<_>>()?;
        let max_layer = *out_layers.iter().max().unwrap_or(&0);
        if max_layer as i32 >= cfg.num_layers {
            return Err(Error::Msg(format!(
                "krea te: select_hidden needs layer {max_layer} but the encoder has {} layers",
                cfg.num_layers
            )));
        }

        let mut layers = Vec::with_capacity(max_layer + 1);
        for i in 0..=max_layer {
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
            embed_tokens: embedding(w, &join(prefix, "embed_tokens"))?,
            layers,
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            out_layers,
            prefix_tokens: cfg.prefix_tokens as i32,
            image_token_id: cfg.image_token_id,
            mrope_section: cfg.mrope_section,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    /// Quantize the token table + every decoder-layer projection in place (group-wise affine Q4/Q8).
    /// `cast_to_bf16=true` for the embedding matches the Qwen3 TE convention; the norms stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.embed_tokens.quantize(bits, true)?;
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// `input_ids` / `attention_mask`: `[b, s]` int32. Returns the stacked conditioning
    /// `[b, s - prefix_tokens, num_select, hidden]` (the DiT's `context`). The final norm is never
    /// applied; only layers up to `max(out_layers)` are run.
    pub fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let mut saved: Vec<(usize, Array)> = Vec::with_capacity(self.out_layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            if self.out_layers.contains(&i) {
                saved.push((i, hidden.clone()));
            }
        }

        self.stack_and_trim(saved)
    }

    /// The shared tail of the text-only [`forward`](Self::forward) and the grounded
    /// [`forward_with_image`](Self::forward_with_image): stack the captured `select_hidden` layers on a
    /// new axis 2 → `[b, s, n, hidden]` (the reference `torch.stack([hidden_states[i] …], dim=2)`), then
    /// drop the leading template-prefix tokens. Dropping needs strictly more tokens than the prefix; a
    /// shorter sequence would build an empty index and hit an opaque `take_axis` panic (F-081).
    fn stack_and_trim(&self, saved: Vec<(usize, Array)>) -> Result<Array> {
        let pick = |idx: usize| -> Result<&Array> {
            saved
                .iter()
                .find(|(k, _)| *k == idx)
                .map(|(_, v)| v)
                .ok_or_else(|| Error::Msg(format!("krea te: hidden state {idx} not captured")))
        };
        let expanded: Vec<Array> = self
            .out_layers
            .iter()
            .map(|&idx| Ok(pick(idx)?.expand_dims(2)?))
            .collect::<Result<_>>()?;
        let refs: Vec<&Array> = expanded.iter().collect();
        let stacked = concatenate_axis(&refs, 2)?; // [b, s, n, hidden]

        let n = stacked.shape()[1];
        if n <= self.prefix_tokens {
            return Err(Error::Msg(format!(
                "krea text encoder: prompt has {n} token(s), must exceed the {} dropped template-prefix tokens",
                self.prefix_tokens
            )));
        }
        // Drop the leading `prefix_tokens` via a contiguous split (F-078), not an arange-gather: keep
        // the tail `[prefix_tokens, n)`.
        Ok(stacked.split_axis(&[self.prefix_tokens], 1)?.swap_remove(1))
    }

    /// **Image-grounded** condition encoding (epic 10871 P2.1) — the Qwen3-VL "dual conditioning" text
    /// half: run the encoder with the source image's vision features spliced over the `<|image_pad|>`
    /// block and 3-D MRoPE positions, so the LM "sees" the image while reading the edit instruction.
    /// Mirrors [`forward`](Self::forward) but (a) replaces the `<|image_pad|>` token embeddings with the
    /// vision tower's merged `image_embeds` `[n, hidden]`, (b) additively injects the `deepstack`
    /// features at those positions for the first `deepstack.len()` layers, and (c) uses interleaved
    /// MRoPE (the image block carries its 2-D merged grid position; text stays sequential). Returns the
    /// same stacked `[b, s - prefix_tokens, num_select, hidden]` the DiT `TextFusionTransformer` consumes.
    /// `image_embeds` / `deepstack` come from [`mlx_gen_boogu::VisionTower::forward`]; `grid_thw` is that
    /// image's `[t, h, w]` patch grid.
    ///
    /// NB `prefix_tokens` is the text-to-image template's system-prefix length; the edit template's
    /// prefix must match (or be re-derived) so the drop stays aligned — verified on the real edit
    /// template + weights in P2.3.
    pub fn forward_with_image(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        image_embeds: &Array,
        deepstack: &[Array],
        grid_thw: [i32; 3],
    ) -> Result<Array> {
        // The single-image forward is the N = 1 case of the multi-image path (epic 10871 P2.3, F-071):
        // one embeds array, one deepstack stack, one grid. Kept as a convenience for single-source edits
        // and the existing tests; byte-identical to the multi splice for one image.
        self.forward_with_images(
            input_ids,
            attention_mask,
            std::slice::from_ref(image_embeds),
            std::slice::from_ref(&deepstack.to_vec()),
            &[grid_thw],
        )
    }

    /// **Multi-image-grounded** condition encoding (epic 10871 P1.3/P2.3, F-071) — the scene+person (or
    /// N-source) generalization of [`forward_with_image`]. Splices each reference's `image_embeds[j]`
    /// (`[nⱼ, hidden]`) at its own contiguous `<|image_pad|>` run, runs the decoder under the 3-D
    /// interleaved MRoPE (positions advance `max(hⱼ, wⱼ)/merge` per image, [`mrope_positions_multi`]),
    /// and additively injects each image's `deepstack[j][i]` feature at that image's run for the first
    /// `deepstack.len()` layers. `grids[j]` is image `j`'s `[t, h, w]` patch grid, in the same order the
    /// runs appear in `input_ids`. `b = 1`. Mirrors [`mlx_gen_boogu::…::last_hidden_with_image_multi`],
    /// so BOTH edit sources reach the grounded encode — not just the first (the prior single-source gap).
    pub fn forward_with_images(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        image_embeds: &[Array],
        deepstack: &[Vec<Array>],
        grids: &[[i32; 3]],
    ) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let ids_arr = input_ids.as_dtype(Dtype::Int32)?;
        let ids: Vec<i32> = ids_arr.as_slice::<i32>().to_vec();

        // One contiguous <|image_pad|> run per reference image, in sequence order.
        let runs = image_token_runs(&ids, self.image_token_id, s);
        if runs.is_empty() {
            return Err(Error::Msg(
                "krea te (grounded): prompt has no <|image_pad|> tokens".into(),
            ));
        }
        if runs.len() != image_embeds.len() || runs.len() != grids.len() {
            return Err(Error::Msg(format!(
                "krea te (grounded): {} <|image_pad|> run(s) but {} embeds / {} grids",
                runs.len(),
                image_embeds.len(),
                grids.len()
            )));
        }

        // Embed tokens, then splice each image's vision features over its <|image_pad|> block.
        let mut hidden = self.embed_tokens.forward(input_ids)?; // [1, s, hidden]
        let dt = hidden.dtype();
        for (&(start, end), emb) in runs.iter().zip(image_embeds) {
            let img = emb.expand_dims(0)?.as_dtype(dt)?; // [1, nⱼ, hidden]
            hidden = replace_seq(&hidden, &img, start, end, s)?;
        }

        // 3-D MRoPE: text tokens sequential; each image block carries its merged (h/m × w/m) grid, with
        // the per-image position advance.
        let (pt, ph, pw) = mrope_positions_multi(&ids, self.image_token_id, grids);
        let (cos, sin) = mrope_cos_sin(
            &pt,
            &ph,
            &pw,
            self.head_dim,
            self.rope_theta,
            self.mrope_section,
            dt,
        )?;
        let mask = build_mask(attention_mask, b, s)?;

        let mut saved: Vec<(usize, Array)> = Vec::with_capacity(self.out_layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            // Deepstack: add each image's i-th merged vision feature at its run for LM layers
            // 0..deepstack.len() (the Qwen3-VL deepstack contract).
            for (&(start, end), ds_img) in runs.iter().zip(deepstack) {
                if i < ds_img.len() {
                    let mid = slice_seq(&hidden, start, end)?;
                    let inj = add(&mid, &ds_img[i].expand_dims(0)?.as_dtype(dt)?)?;
                    hidden = replace_seq(&hidden, &inj, start, end, s)?;
                }
            }
            if self.out_layers.contains(&i) {
                saved.push((i, hidden.clone()));
            }
        }
        self.stack_and_trim(saved)
    }
}

// ── Image-grounded helpers (ported from `mlx-gen-boogu`'s Qwen3-VL text encoder) ─────────────────

/// Slice `[b, s, d]` along the sequence axis to `[start, end)`. A contiguous `split_axis` (F-078), not
/// an arange + `take_axis` gather: MLX's `as_slice` ignores strides, so a gather round-trip through
/// host indices is both slower and stride-fragile — the split returns the same elements directly.
fn slice_seq(x: &Array, start: i32, end: i32) -> Result<Array> {
    Ok(x.split_axis(&[start, end], 1)?.swap_remove(1))
}

/// Replace `x[:, start:end, :]` with `repl` (`[b, end-start, d]`) via concat of the surrounding
/// (contiguous-split) slices — the masked-replace splice (no in-place scatter). F-078: the surrounding
/// slices come from `split_axis`, not an arange-gather. `s` is unused now (the split names the
/// trailing boundary implicitly) but kept in the signature for call-site symmetry with the runs.
fn replace_seq(x: &Array, repl: &Array, start: i32, end: i32, _s: i32) -> Result<Array> {
    let mut parts = x.split_axis(&[start, end], 1)?;
    let after = parts.swap_remove(2);
    let before = parts.swap_remove(0);
    Ok(concatenate_axis(&[&before, repl, &after], 1)?)
}

/// Contiguous runs of `image_token_id` in `ids` (`[start, end)` per run), in sequence order — one run
/// per reference image (the tokenizer separates images with `<|vision_end|><|vision_start|>`).
fn image_token_runs(ids: &[i32], image_token_id: i32, s: i32) -> Vec<(i32, i32)> {
    let mut runs = Vec::new();
    let mut i = 0i32;
    while i < s {
        if ids[i as usize] == image_token_id {
            let start = i;
            while i < s && ids[i as usize] == image_token_id {
                i += 1;
            }
            runs.push((start, i));
        } else {
            i += 1;
        }
    }
    runs
}

/// 3-D MRoPE positions per token for a SINGLE image: text tokens advance `(i, i, i)`; an image block
/// (at offset `cur`) gets `t = cur`, `h = cur + row`, `w = cur + col` over its `(h/merge)×(w/merge)`
/// merged grid, then `cur += max(h, w) / merge`. Mirrors Qwen3-VL `get_rope_index`. F-071: production
/// now always goes through [`mrope_positions_multi`] (the single case is `grids.len() == 1`); this is
/// retained as the test oracle that pins the multi path equal on one image.
#[cfg(test)]
fn mrope_positions(
    ids: &[i32],
    image_token_id: i32,
    grid_h: i32,
    grid_w: i32,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let (llm_h, llm_w) = (grid_h / SPATIAL_MERGE, grid_w / SPATIAL_MERGE);
    let step = grid_h.max(grid_w) / SPATIAL_MERGE;
    let (mut pt, mut ph, mut pw) = (Vec::new(), Vec::new(), Vec::new());
    let mut cur = 0i32;
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == image_token_id {
            for idx in 0..(llm_h * llm_w) {
                pt.push(cur);
                ph.push(cur + idx / llm_w);
                pw.push(cur + idx % llm_w);
            }
            cur += step;
            i += (llm_h * llm_w) as usize;
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

/// Multi-image 3-D MRoPE positions (F-071; mirrors Qwen3-VL `get_rope_index` over `image_grid_thw`):
/// text tokens advance `(i, i, i)`; the `k`-th image block (using `grids[k]`) at offset `cur` gets
/// `t = cur`, `h = cur + row`, `w = cur + col` over its `(h/merge)×(w/merge)` merged grid, then
/// `cur += max(h, w) / merge`. The single-image [`mrope_positions`] is exactly the `grids.len() == 1`
/// case (verified by [`tests::mrope_multi_matches_single_for_one_image`]).
fn mrope_positions_multi(
    ids: &[i32],
    image_token_id: i32,
    grids: &[[i32; 3]],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let (mut pt, mut ph, mut pw) = (Vec::new(), Vec::new(), Vec::new());
    let mut cur = 0i32;
    let mut img_i = 0usize;
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == image_token_id {
            let g = grids[img_i];
            let (llm_h, llm_w) = (g[1] / SPATIAL_MERGE, g[2] / SPATIAL_MERGE);
            let step = g[1].max(g[2]) / SPATIAL_MERGE;
            for idx in 0..(llm_h * llm_w) {
                pt.push(cur);
                ph.push(cur + idx / llm_w);
                pw.push(cur + idx % llm_w);
            }
            cur += step;
            i += (llm_h * llm_w) as usize;
            img_i += 1;
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

/// Build the interleaved-MRoPE `cos`/`sin` `[1, s, head_dim]` (cast to `dt`). For each of the
/// `head_dim/2` freqs `j`: within the first `section[1]·3` indices `j%3==1 → H`, within `section[2]·3`
/// `j%3==2 → W`, else `T`; `angle = pos·θ^(−2j/head_dim)`, written to both halves (`emb = cat(f, f)`).
fn mrope_cos_sin(
    pt: &[i32],
    ph: &[i32],
    pw: &[i32],
    head_dim: i32,
    theta: f32,
    section: [i32; 3],
    dt: Dtype,
) -> Result<(Array, Array)> {
    let s = pt.len();
    let half = (head_dim / 2) as usize;
    let sec_h = (section[1] * 3) as usize;
    let sec_w = (section[2] * 3) as usize;
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
            emb[i * hd + half + j] = angle;
        }
    }
    let arr = Array::from_slice(&emb, &[1, s as i32, head_dim]);
    Ok((arr.cos()?.as_dtype(dt)?, arr.sin()?.as_dtype(dt)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MRoPE positions: text tokens advance sequentially `(i,i,i)`; the image block sits at the running
    /// offset with its 2-D merged grid on the H/W axes, and the cursor jumps by `max(h,w)/merge` after.
    #[test]
    fn mrope_positions_lays_out_text_then_image_grid() {
        // ids: [txt, txt, img×4, txt] with a 4×4 patch grid → merged 2×2 = 4 image tokens.
        let img = 99;
        let ids = vec![1, 2, img, img, img, img, 3];
        let (pt, ph, pw) = mrope_positions(&ids, img, 4, 4);
        // Text @0,1 → (0,0,0),(1,1,1). Image @cur=2, llm 2×2: idx→(2, 2+row, 2+col). Then cur += 4/2=2
        // → next text @ (4,4,4).
        assert_eq!(pt, vec![0, 1, 2, 2, 2, 2, 4]);
        assert_eq!(ph, vec![0, 1, 2, 2, 3, 3, 4]);
        assert_eq!(pw, vec![0, 1, 2, 3, 2, 3, 4]);
    }

    /// Text-only MRoPE (no image tokens) is a plain sequential ramp on all three axes — the reduction
    /// that lets the text path keep using 1-D `TextRope`.
    #[test]
    fn mrope_positions_text_only_is_sequential() {
        let (pt, ph, pw) = mrope_positions(&[10, 11, 12, 13], 99, 4, 4);
        assert_eq!(pt, vec![0, 1, 2, 3]);
        assert_eq!(pt, ph);
        assert_eq!(pt, pw);
    }

    /// F-071: the multi-image position layout is the exact generalization of the single-image one — for
    /// a single image + grid it produces byte-identical `(pt, ph, pw)`, so the multi path is a safe
    /// superset (the single `forward_with_image` delegates to it).
    #[test]
    fn mrope_multi_matches_single_for_one_image() {
        let img = 99;
        let ids = vec![1, 2, img, img, img, img, 3];
        let single = mrope_positions(&ids, img, 4, 4);
        let multi = mrope_positions_multi(&ids, img, &[[1, 4, 4]]);
        assert_eq!(single, multi);
    }

    /// F-071: TWO reference images (scene + person) each get their own merged grid block, and the second
    /// block's cursor starts AFTER the first image's `max(h,w)/merge` advance — so the person image is
    /// grounded at a distinct frame, not overlaid on the scene. This is the layout the two-source edit
    /// LoRA was trained against.
    #[test]
    fn mrope_multi_lays_out_two_image_blocks() {
        let img = 99;
        // [txt, img×4 (2×2 grid), img×4 (2×2 grid), txt]
        let ids = vec![1, img, img, img, img, img, img, img, img, 2];
        let (pt, ph, pw) = mrope_positions_multi(&ids, img, &[[1, 4, 4], [1, 4, 4]]);
        // txt@0 → 0. img1@cur=1: (1, 1+row, 1+col) over 2×2 → advance +2. img2@cur=3: (3, 3+row, 3+col)
        // → advance +2. txt@cur=5 → 5.
        assert_eq!(pt, vec![0, 1, 1, 1, 1, 3, 3, 3, 3, 5]);
        assert_eq!(ph, vec![0, 1, 1, 2, 2, 3, 3, 4, 4, 5]);
        assert_eq!(pw, vec![0, 1, 2, 1, 2, 3, 4, 3, 4, 5]);
    }

    /// F-078: the contiguous-`split_axis` `slice_seq`/`replace_seq` return element-identical results to
    /// the arange + `take_axis` gather they replaced. `as_slice` ignores strides, so a split view and a
    /// gather could in principle diverge — this pins them equal on a strided sequence axis.
    #[test]
    fn split_slice_and_replace_match_arange_gather() {
        // [1, 5, 2] so the sequence axis (axis 1) is strided over the last dim.
        let x = Array::from_slice(&[0.0f32, 1., 2., 3., 4., 5., 6., 7., 8., 9.], &[1, 5, 2]);
        let (start, end, s) = (1i32, 4i32, 5i32);

        // Old gather reference for slice_seq.
        let idx: Vec<i32> = (start..end).collect();
        let gathered = x
            .take_axis(Array::from_slice(&idx, &[idx.len() as i32]), 1)
            .unwrap();
        let split = slice_seq(&x, start, end).unwrap();
        assert_eq!(
            split.as_slice::<f32>(),
            gathered.as_slice::<f32>(),
            "slice_seq split must equal the arange-gather"
        );

        // replace_seq: swap [start,end) for a marker block, compare against a gather-built reference.
        let repl = Array::from_slice(&[-1.0f32, -2., -3., -4., -5., -6.], &[1, 3, 2]);
        let before_idx: Vec<i32> = (0..start).collect();
        let after_idx: Vec<i32> = (end..s).collect();
        let before = x
            .take_axis(
                Array::from_slice(&before_idx, &[before_idx.len() as i32]),
                1,
            )
            .unwrap();
        let after = x
            .take_axis(Array::from_slice(&after_idx, &[after_idx.len() as i32]), 1)
            .unwrap();
        let gather_ref = concatenate_axis(&[&before, &repl, &after], 1).unwrap();
        let via_split = replace_seq(&x, &repl, start, end, s).unwrap();
        assert_eq!(via_split.shape(), gather_ref.shape());
        assert_eq!(
            via_split.as_slice::<f32>(),
            gather_ref.as_slice::<f32>(),
            "replace_seq split must equal the arange-gather splice"
        );
    }
}
