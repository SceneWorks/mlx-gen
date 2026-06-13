//! sc-5140: the Bernini planner's **MAR semantic-planning loop** + the planner→renderer **handoff** —
//! filling the target ViT tokens, then turning the planner's penultimate hidden states into the
//! renderer's 4 conditioning streams. Port of `sample_vit_embed` (`pipeline.py` 724-884) +
//! `post_process_input_embeds`/`feat_from_planner_to_renderer` (`bernini.py`).
//!
//!   - [`post_process_input_embeds`] (inference) — set every gen-ViT slot (`visual_output_mask`) to
//!     the `mask_token`; the MAR loop starts fully masked.
//!   - [`mar_schedule`] — the MaskGIT/MAR cosine reveal schedule (`mask_ratio = cos(π/2·(s+1)/N)`,
//!     `mask_len = floor(n·ratio)` clamped to `[1, masked−1]`, the newly-revealed chunk
//!     `order[mask_len : prev]`, the last step revealing everything still masked). Driven by a
//!     **seeded** reveal permutation `order` so it can be matched bit-for-bit against torch.
//!   - [`sample_vit_embed`] — the `planning_step`-iteration loop: a single shared target buffer
//!     spliced into all 3 streams (cond/uncond/imgcond) → Qwen2.5-VL backbone → penultimate at the
//!     gen slots → `connector.for_vit` → gather the revealed tokens → `clip_diff.sample` (triple CFG)
//!     → scatter the predictions back. Both RNG consumers (the reveal `order` and the per-step FM
//!     noise) are **injectable** for deterministic parity.
//!   - [`feat_to_renderer`] (inference) — `connector.for_gen` over *all* tokens (`cond_embed_mask =
//!     ¬gen | gen = all`), plus the txt (`¬gen`) / vit (`gen`) position sub-masks.
//!   - [`four_streams`] — `feature_type = masked_tgt_embed_with_qwen_txt_vit_tokens`: `wtxt_wvit` =
//!     cond contexts; `wtxt_wovit` = cond[txt]; `wotxt_wvit` = cond[vit]; `wotxt_wovit` = uncond[txt].
//!     These feed the renderer's ViT-conditioned APG guidance (sc-5142).
//!
//! Sequence-axis gather is `take_axis`; scatter (write a few rows back into the buffer) is the
//! one-hot-selection matmul idiom (no in-place index assignment), matching the sensenova port.

use mlx_rs::ops::{add, concatenate_axis, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::Result;

use crate::clip_diff::DiffLossFm;
use crate::connector::MlpConnector;
use crate::qwen2_5_vl::Qwen25VlText;

/// Positions where `mask == want`.
fn mask_to_idx(mask: &[bool], want: bool) -> Vec<i32> {
    mask.iter()
        .enumerate()
        .filter(|(_, &m)| m == want)
        .map(|(i, _)| i as i32)
        .collect()
}

/// `post_process_input_embeds` (inference): set the gen-ViT slots of `input_embeds` `[1, L, H]` to
/// `mask_token` `[1, 1, H]` (broadcast). `gen_mask[i]` marks a gen-ViT slot.
pub fn post_process_input_embeds(
    input_embeds: &Array,
    gen_mask: &[bool],
    mask_token: &Array,
) -> Result<Array> {
    let l = gen_mask.len() as i32;
    let g: Vec<f32> = gen_mask
        .iter()
        .map(|&m| if m { 1.0 } else { 0.0 })
        .collect();
    let gv = Array::from_slice(&g, &[1, l, 1]);
    let keep = subtract(Array::from_f32(1.0), &gv)?; // 1 - g
    Ok(add(
        &multiply(input_embeds, &keep)?,
        &multiply(mask_token, &gv)?, // [1,1,H] * [1,L,1] → [1,L,H]
    )?)
}

/// `feat_from_planner_to_renderer` (inference): `for_gen` over all tokens + the txt/vit sub-masks.
pub struct RendererFeat {
    /// `connector.for_gen(hidden)` — `[1, L, gen_dim]`.
    pub contexts: Array,
    /// Token positions that are **not** gen-ViT (text + input-vit).
    pub txt_idx: Vec<i32>,
    /// Token positions that **are** gen-ViT.
    pub vit_idx: Vec<i32>,
}

/// Run the renderer-feature projection over `hidden` `[1, L, H]`; `gen_mask` marks the gen-ViT slots.
pub fn feat_to_renderer(
    hidden: &Array,
    gen_mask: &[bool],
    connector: &MlpConnector,
) -> Result<RendererFeat> {
    Ok(RendererFeat {
        contexts: connector.for_gen(hidden)?,
        txt_idx: mask_to_idx(gen_mask, false),
        vit_idx: mask_to_idx(gen_mask, true),
    })
}

/// The 4 renderer conditioning streams.
pub struct FourStreams {
    pub wtxt_wvit: Array,
    pub wtxt_wovit: Array,
    pub wotxt_wvit: Array,
    pub wotxt_wovit: Array,
}

fn take_seq(a: &Array, idx: &[i32]) -> Result<Array> {
    Ok(a.take_axis(Array::from_slice(idx, &[idx.len() as i32]), 1)?)
}

/// Build the 4 streams from the cond + uncond planner hidden states (the `sample_vit_embed` tail,
/// `else`/`masked_tgt_embed_with_qwen_txt_vit_tokens` branch).
pub fn four_streams(
    cond_hidden: &Array,
    cond_gen_mask: &[bool],
    uncond_hidden: &Array,
    uncond_gen_mask: &[bool],
    connector: &MlpConnector,
) -> Result<FourStreams> {
    let c = feat_to_renderer(cond_hidden, cond_gen_mask, connector)?;
    let u = feat_to_renderer(uncond_hidden, uncond_gen_mask, connector)?;
    Ok(FourStreams {
        wtxt_wovit: take_seq(&c.contexts, &c.txt_idx)?,
        wotxt_wvit: take_seq(&c.contexts, &c.vit_idx)?,
        wotxt_wovit: take_seq(&u.contexts, &u.txt_idx)?,
        wtxt_wvit: c.contexts,
    })
}

/// Bool mask of length `len` with `true` at every position in `idx` (the inverse of [`mask_to_idx`]).
fn idx_to_bool(idx: &[i32], len: usize) -> Vec<bool> {
    let mut m = vec![false; len];
    for &i in idx {
        m[i as usize] = true;
    }
    m
}

// ---------------------------------------------------------------------------
// MAR cosine reveal schedule (`sample_vit_embed` 744-813).
// ---------------------------------------------------------------------------

/// The MaskGIT/MAR reveal schedule: for each of `planning_step` steps, the **sorted** target-token
/// positions revealed (un-masked) on that step. Port of the mask bookkeeping in `sample_vit_embed`:
///
///   - `mask_ratio = cos(π/2·(s+1)/N)` (float64, like numpy); `mask_len = floor(n_query·ratio)`.
///   - clamp `mask_len = max(1, min(masked_now − 1, mask_len))` (always leave ≥1 masked, never grow).
///   - the still-masked set is always a prefix `order[:mask_len]`, so the newly-revealed chunk is
///     `order[mask_len : prev_mask_len]` (`xor(mask, mask_next)`); the **last** step reveals the whole
///     remaining masked set `order[:prev_mask_len]` (`mask_to_pred = mask`).
///   - positions are returned **ascending** (the reference gathers/scatters by `nonzero`, i.e. sorted
///     token position, not reveal order).
///
/// `order` is the seeded reveal permutation of `[0, n_query)`. The chunks across steps are disjoint
/// and cover every token exactly once, so after the loop every target slot is filled.
pub fn mar_schedule(n_query: i32, planning_step: usize, order: &[i32]) -> Vec<Vec<i32>> {
    let mut out = Vec::with_capacity(planning_step);
    let mut prev = n_query; // currently-masked count (mask starts all-ones)
    for step in 0..planning_step {
        let ratio = (std::f64::consts::PI / 2.0 * (step + 1) as f64 / planning_step as f64).cos();
        let raw = (n_query as f64 * ratio).floor() as i32;
        let mask_len = raw.min(prev - 1).max(1);
        // newly-revealed token positions this step
        let revealed: &[i32] = if step >= planning_step - 1 {
            &order[..prev as usize] // last step: everything still masked
        } else {
            &order[mask_len as usize..prev as usize]
        };
        let mut sorted: Vec<i32> = revealed.to_vec();
        sorted.sort_unstable();
        out.push(sorted);
        prev = mask_len;
    }
    out
}

// ---------------------------------------------------------------------------
// Sequence-axis scatter (row gather; bit-exact, no matmul reduction).
// ---------------------------------------------------------------------------

/// Overwrite the rows of `base` `[1, L, H]` at positions `idx` with `src` `[1, n, H]` (row `j` ←
/// `src[0, j]`), leaving the other rows untouched. `idx.len() == n`. Implemented as a pure row gather
/// over `concat([base; src])` — `out[l] = base[l]` for un-touched rows, `src[j]` where `idx[j] == l` —
/// so it is **bit-exact** (a one-hot matmul would pick up the Metal f32 matmul floor instead). Also the
/// `masked_scatter` primitive for [`crate::assembly::format_mllm_inputs_embeds`].
pub(crate) fn scatter_rows(base: &Array, idx: &[i32], src: &Array) -> Result<Array> {
    let sh = base.shape();
    let (l, h) = (sh[1], sh[2]);
    let n = idx.len() as i32;
    let stacked = concatenate_axis(&[&base.reshape(&[l, h])?, &src.reshape(&[n, h])?], 0)?; // [L+n, H]

    let mut gi: Vec<i32> = (0..l).collect(); // un-touched rows keep their own index
    for (j, &pos) in idx.iter().enumerate() {
        gi[pos as usize] = l + j as i32; // touched rows point into the src block
    }
    Ok(stacked
        .take_axis(Array::from_slice(&gi, &[l]), 0)?
        .reshape(&[1, l, h])?)
}

// ---------------------------------------------------------------------------
// sample_vit_embed orchestration.
// ---------------------------------------------------------------------------

/// One conditioning stream's planner inputs (cond / uncond / imgcond). `input_embeds` is the
/// post-processed embed `[1, L, H]` (gen-ViT slots already set to the `mask_token`); `position_ids`
/// is `[3, L]` int32; `mask` is the additive 4D attention mask (`[1, L, L]` or `[1, 1, L, L]`);
/// `gen_idx` is the **sorted** target-ViT slot positions (`visual_output_token_mask`).
pub struct StreamState {
    pub input_embeds: Array,
    pub position_ids: Array,
    pub mask: Array,
    pub gen_idx: Vec<i32>,
}

/// MAR planning knobs (`sample_vit_embed` / `__call__` defaults: 25 / 3 / 1.4 / 1.2).
pub struct VitCfg {
    pub planning_step: usize,
    pub vit_denoising_step: usize,
    pub vit_txt_cfg: f32,
    pub vit_img_cfg: f32,
}

impl Default for VitCfg {
    fn default() -> Self {
        Self {
            planning_step: 25,
            vit_denoising_step: 3,
            vit_txt_cfg: 1.4,
            vit_img_cfg: 1.2,
        }
    }
}

/// The output of [`sample_vit_embed`]: the renderer's 4 conditioning streams (pre-T5) + the filled
/// target ViT embeds.
pub struct SampledStreams {
    pub wtxt_wvit: Array,
    pub wtxt_wovit: Array,
    pub wotxt_wvit: Array,
    pub wotxt_wovit: Array,
    /// The fully-revealed target ViT embeds `[1, n_query, H]` (`pred_vit_embed`).
    pub pred_vit_embed: Array,
}

/// Splice the shared target buffer `target` `[1, n_query, H]` into stream `s`'s gen-ViT slots,
/// yielding the current `input_embeds` for this step's backbone forward.
fn splice_target(s: &StreamState, target: &Array) -> Result<Array> {
    scatter_rows(&s.input_embeds, &s.gen_idx, target)
}

/// `[1, L, H]` penultimate hidden state at stream `s`'s gen slots → `connector.for_vit` → the ViT
/// prediction `[1, n_query, H_vit]`.
fn stream_for_vit(
    backbone: &Qwen25VlText,
    connector: &MlpConnector,
    s: &StreamState,
    target: &Array,
) -> Result<Array> {
    let embeds = splice_target(s, target)?;
    let hidden = backbone.penultimate(&embeds, &s.position_ids, &s.mask)?;
    connector.for_vit(&take_seq(&hidden, &s.gen_idx)?)
}

/// The MAR planning loop (`sample_vit_embed`): fill the target ViT tokens over `cfg.planning_step`
/// MaskGIT steps, then run the planner→renderer handoff ([`four_streams`]).
///
/// `order` is the seeded reveal permutation of `[0, n_query)`; `step_noise[step]` is the base FM noise
/// `[n_revealed_step, H]` for that step's `clip_diff.sample` (`torch.randn(z.shape[0]//3, in)` in the
/// reference — tiled ×3 internally). Both are injected so the trajectory matches torch bit-for-bit;
/// a step whose revealed set is empty (or `{token 0}` only — the reference's `nonzero().sum()==0`
/// skip) consumes no noise. `mask_token` is `[1, 1, H]`.
#[allow(clippy::too_many_arguments)]
pub fn sample_vit_embed(
    backbone: &Qwen25VlText,
    connector: &MlpConnector,
    clip_diff: &mut DiffLossFm,
    cond: &StreamState,
    uncond: &StreamState,
    imgcond: &StreamState,
    cfg: &VitCfg,
    order: &[i32],
    step_noise: &[Array],
    mask_token: &Array,
) -> Result<SampledStreams> {
    let n_query = order.len() as i32;
    let h = mask_token.shape()[2];
    let schedule = mar_schedule(n_query, cfg.planning_step, order);

    // Single shared target buffer, init = mask_token broadcast over the n_query target slots.
    let mut target = mlx_rs::ops::broadcast_to(mask_token, &[1, n_query, h])?;

    for (step, revealed) in schedule.iter().enumerate() {
        // Every step runs all 3 backbones over the current (partially-filled) embeds.
        let cond_vit = stream_for_vit(backbone, connector, cond, &target)?;
        let uncond_vit = stream_for_vit(backbone, connector, uncond, &target)?;
        let imgcond_vit = stream_for_vit(backbone, connector, imgcond, &target)?;

        // `nonzero().sum() == 0` → nothing to predict this step (empty, or {token 0} only).
        if revealed.iter().sum::<i32>() == 0 {
            continue;
        }
        let np = revealed.len() as i32;
        let hv = cond_vit.shape()[2];

        // z = cat([cond, uncond, imgcond], dim=1)[0] → [3·np, H_vit] (pre-tiled triple-CFG cond).
        let c = take_seq(&cond_vit, revealed)?;
        let u = take_seq(&uncond_vit, revealed)?;
        let ic = take_seq(&imgcond_vit, revealed)?;
        let z = concatenate_axis(&[&c, &u, &ic], 1)?.reshape(&[3 * np, hv])?;

        // Triple-CFG denoise; take the first third (the cond tile) → [1, np, H].
        let sampled = clip_diff.sample(
            &z,
            cfg.vit_txt_cfg,
            cfg.vit_denoising_step,
            Some(cfg.vit_img_cfg),
            &step_noise[step],
        )?;
        let cur = take_first_rows(&sampled, np)?.reshape(&[1, np, h])?;

        target = scatter_rows(&target, revealed, &cur)?;
    }

    // ---- handoff: final cond + uncond forwards → feat_from_planner_to_renderer → 4 streams ----
    let cond_embeds = splice_target(cond, &target)?;
    let uncond_embeds = splice_target(uncond, &target)?;
    let cond_hidden = backbone.penultimate(&cond_embeds, &cond.position_ids, &cond.mask)?;
    let uncond_hidden = backbone.penultimate(&uncond_embeds, &uncond.position_ids, &uncond.mask)?;

    let cond_gen = idx_to_bool(&cond.gen_idx, cond.input_embeds.shape()[1] as usize);
    let uncond_gen = idx_to_bool(&uncond.gen_idx, uncond.input_embeds.shape()[1] as usize);
    let s = four_streams(
        &cond_hidden,
        &cond_gen,
        &uncond_hidden,
        &uncond_gen,
        connector,
    )?;

    Ok(SampledStreams {
        wtxt_wvit: s.wtxt_wvit,
        wtxt_wovit: s.wtxt_wovit,
        wotxt_wvit: s.wotxt_wvit,
        wotxt_wovit: s.wotxt_wovit,
        pred_vit_embed: target,
    })
}

/// First `n` rows of a `[R, C]` array → `[n, C]` (`x[:n]`).
fn take_first_rows(x: &Array, n: i32) -> Result<Array> {
    let idx: Vec<i32> = (0..n).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[n]), 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_idx_split() {
        let m = [false, true, true, false];
        assert_eq!(mask_to_idx(&m, true), vec![1, 2]);
        assert_eq!(mask_to_idx(&m, false), vec![0, 3]);
    }

    #[test]
    fn idx_bool_roundtrip() {
        let idx = [1, 3];
        let b = idx_to_bool(&idx, 4);
        assert_eq!(b, vec![false, true, false, true]);
        assert_eq!(mask_to_idx(&b, true), vec![1, 3]);
    }

    /// The schedule covers every token exactly once (disjoint chunks), reveals ascending positions,
    /// and the last step empties the remaining masked set.
    #[test]
    fn schedule_covers_all_once() {
        let order = [4, 0, 2, 5, 1, 3]; // a permutation of 0..6
        let n = order.len() as i32;
        for planning_step in [1usize, 3, 4, 8] {
            let sched = mar_schedule(n, planning_step, &order);
            assert_eq!(sched.len(), planning_step);
            let mut all: Vec<i32> = sched.iter().flatten().copied().collect();
            all.sort_unstable();
            assert_eq!(all, (0..n).collect::<Vec<_>>(), "every token revealed once");
            for step in &sched {
                let mut s = step.clone();
                s.sort_unstable();
                assert_eq!(&s, step, "revealed positions ascending");
            }
        }
    }

    /// Step 0 reveals exactly the last element of `order` (the one token un-masked when
    /// `mask_len = n−1`); the clamp keeps ≥1 masked until the final step.
    #[test]
    fn schedule_step0_reveals_order_tail() {
        let order = [4, 0, 2, 5, 1, 3];
        let sched = mar_schedule(6, 4, &order);
        assert_eq!(sched[0], vec![3]); // order[5] = 3, sorted singleton
    }
}
