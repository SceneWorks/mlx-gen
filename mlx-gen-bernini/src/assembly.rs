//! sc-5140: the planner-input assembly glue that brackets the MAR loop — `format_mllm_inputs_embeds`
//! (`bernini.py`) before it, and the T5 `concat_with_zero_init` (`pipeline.__call__`) after it.
//!
//!   - [`format_mllm_inputs_embeds`] — embed the token ids, then `masked_scatter` the ViT visual
//!     features into the visual slots (`visual_input_mask | visual_output_mask`). The gen-output slots
//!     get placeholder features here; [`crate::mar::post_process_input_embeds`] then overwrites them
//!     with the `mask_token` before the loop starts.
//!   - [`concat_with_zero_init`] — the renderer streams' text combine: prepend the UMT5 prompt embeds
//!     (positive for the txt streams, negative for the wotxt streams) to each planner stream, then
//!     zero-pad / truncate to `max_sequence_length` (512). The UMT5 encoder itself is the wan
//!     foundation's ([`mlx_gen_wan`]); this is only the prepend + pad/truncate mechanics.

use mlx_rs::ops::{concatenate_axis, pad};
use mlx_rs::Array;

use mlx_gen::Result;

use crate::mar::scatter_rows;
use crate::qwen2_5_vl::Qwen25VlText;

/// `format_mllm_inputs_embeds` (`bernini.py`): `embed_tokens(input_ids)` `[1, L, H]`, then scatter the
/// `visual_embeds` `[n, H]` (input-ViT + target-ViT features, in sequence order) into the visual slots
/// (`visual_input_mask | visual_output_mask`). With no visual features it's just the token embedding.
///
/// `input_ids` is the flat id list (length `L`); the two masks are per-token booleans of length `L`.
/// The scatter is row-order: visual feature `j` fills the `j`-th visual slot in ascending position,
/// matching torch `masked_scatter` (which fills `True` positions in flattened order).
pub fn format_mllm_inputs_embeds(
    backbone: &Qwen25VlText,
    input_ids: &[i32],
    visual_embeds: Option<&Array>,
    visual_input_mask: &[bool],
    visual_output_mask: &[bool],
) -> Result<Array> {
    let l = input_ids.len();
    let ids = Array::from_slice(input_ids, &[1, l as i32]);
    let embeds = backbone.embed(&ids)?; // [1, L, H]

    let ve = match visual_embeds {
        Some(v) if v.shape()[0] > 0 => v,
        _ => return Ok(embeds),
    };

    let visual_idx: Vec<i32> = (0..l)
        .filter(|&i| visual_input_mask[i] || visual_output_mask[i])
        .map(|i| i as i32)
        .collect();
    let n = visual_idx.len() as i32;
    if n != ve.shape()[0] {
        return Err(mlx_gen::Error::Msg(format!(
            "format_mllm_inputs_embeds: {n} visual slots but {} visual features",
            ve.shape()[0]
        )));
    }
    let h = embeds.shape()[2];
    scatter_rows(&embeds, &visual_idx, &ve.reshape(&[1, n, h])?)
}

/// `concat_with_zero_init` (`pipeline.__call__`): prepend the T5 prompt embeds `[1, T, W]` to a planner
/// stream `[1, S, W]`, then zero-pad (or truncate) the result to `max_sequence_length` tokens →
/// `[1, max_seq, W]`. Padding appends zero rows on the sequence axis (the reference `feat.new_zeros`).
pub fn concat_with_zero_init(t5_embeds: &Array, stream: &Array, max_seq: i32) -> Result<Array> {
    let combined = concatenate_axis(&[t5_embeds, stream], 1)?; // [1, T+S, W]
    pad_and_truncate(&combined, max_seq)
}

/// Zero-pad (append) or truncate `feat` `[1, S, W]` to exactly `max_seq` tokens on the sequence axis.
pub fn pad_and_truncate(feat: &Array, max_seq: i32) -> Result<Array> {
    let s = feat.shape()[1];
    if s < max_seq {
        // append (max_seq - s) zero rows on axis 1; no pad on axes 0 and 2 (value default 0).
        let widths = [(0, 0), (0, max_seq - s), (0, 0)];
        Ok(pad(feat, &widths[..], None, None)?)
    } else if s > max_seq {
        let idx: Vec<i32> = (0..max_seq).collect();
        Ok(feat.take_axis(Array::from_slice(&idx, &[max_seq]), 1)?)
    } else {
        Ok(feat.clone())
    }
}
