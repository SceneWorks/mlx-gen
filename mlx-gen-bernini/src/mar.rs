//! sc-5140: the Bernini planner→renderer **handoff** — turning the planner's penultimate hidden
//! states into the renderer's 4 conditioning streams. Port of `post_process_input_embeds` +
//! `feat_from_planner_to_renderer` (`_vendor/bernini/bernini/models/bernini.py`) and the 4-stream
//! extraction of `sample_vit_embed` (`pipeline.py`).
//!
//!   - [`post_process_input_embeds`] (inference) — set every gen-ViT slot (`visual_output_mask`) to
//!     the `mask_token`; the MAR loop starts fully masked.
//!   - [`feat_to_renderer`] (inference) — `connector.for_gen` over *all* tokens (`cond_embed_mask =
//!     ¬gen | gen = all`), plus the txt (`¬gen`) / vit (`gen`) position sub-masks.
//!   - [`four_streams`] — `feature_type = masked_tgt_embed_with_qwen_txt_vit_tokens`: `wtxt_wvit` =
//!     cond contexts; `wtxt_wovit` = cond[txt]; `wotxt_wvit` = cond[vit]; `wotxt_wovit` = uncond[txt].
//!     These feed the renderer's ViT-conditioned APG guidance (sc-5142).
//!
//! The full MAR loop ([`sample_vit_embed`]-equivalent) that fills the gen-ViT slots before this
//! handoff is a following sc-5140 sub-piece.

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::Result;

use crate::connector::MlpConnector;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_idx_split() {
        let m = [false, true, true, false];
        assert_eq!(mask_to_idx(&m, true), vec![1, 2]);
        assert_eq!(mask_to_idx(&m, false), vec![0, 3]);
    }
}
