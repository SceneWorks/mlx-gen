//! `QwenVisionLanguageEncoder` (sc-2465 slice 6b) — the Qwen-Image-Edit conditioning encoder. Port
//! of the fork's `QwenEncoder.__call__` (image path) + `QwenVisionLanguageEncoder`:
//!
//! 1. Embed `input_ids`, then **splice** the vision-transformer embeds into the positions of the
//!    `<|image_pad|>` (151655) tokens (consumed in order; per-image split is implicit — the vision
//!    embeds are already concatenated across images).
//! 2. Run the 28 LM layers + final RMSNorm (the verified [`QwenTextEncoder`] stack). The fork uses
//!    **sequential** RoPE here (`position_ids = arange(seq)` broadcast to all mrope sections — NOT
//!    spatial mrope), so the standard `TextRope` applies unchanged.
//! 3. **Drop the first 64** template tokens (vs the T2I text path's 34).
//!
//! Single un-padded sequence per call (the per-prompt pipeline case); the fork's batch valid-len
//! trim/repad reduces to a plain `[drop_idx:]` slice there.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::array::host_i32;
use mlx_gen::{Error, Result};

use super::vision::grid::Grid;
use super::vision::VisionTransformer;
use super::QwenTextEncoder;

pub struct QwenVisionLanguageEncoder {
    lm: QwenTextEncoder,
    visual: VisionTransformer,
}

impl QwenVisionLanguageEncoder {
    /// `<|image_pad|>` token id (the placeholder replaced by vision embeds).
    pub const IMAGE_TOKEN_ID: i32 = 151655;
    /// Tokens dropped from the front of the Edit chat template.
    pub const EDIT_DROP_IDX: i32 = 64;

    pub fn new(lm: QwenTextEncoder, visual: VisionTransformer) -> Self {
        Self { lm, visual }
    }

    /// Run the vision transformer over the reference patches → vision embeds `[n_vis, hidden]`.
    /// Depends only on the image (`pixel_values` + `grids`), so the caller computes it **once** and
    /// reuses it for the positive + negative prompts instead of re-running the 32-block tower (F-004).
    pub fn encode_vision(&self, pixel_values: &Array, grids: &[Grid]) -> Result<Array> {
        self.visual.forward(pixel_values, grids)
    }

    /// Splice precomputed `vision` embeds into the prompt token stream, run the 28 LM layers, and
    /// drop the leading template tokens. Pair with [`encode_vision`](Self::encode_vision) so the
    /// vision tower runs once across the positive/negative encodes.
    pub fn encode_with_vision(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        vision: &Array,
    ) -> Result<Array> {
        let embeds = self.lm.embed(input_ids)?; // [b, s, hidden] f32
        let spliced = self.splice(&embeds, input_ids, vision)?;
        let hidden = self.lm.forward_from_embeds(&spliced, attention_mask)?; // [b, s, hidden]

        // Drop the leading template tokens (single un-padded sequence per row). Needs more than
        // EDIT_DROP_IDX tokens or the index is empty and `take_axis` panics (F-020/L-A).
        let s = hidden.shape()[1];
        if s <= Self::EDIT_DROP_IDX {
            return Err(Error::Msg(format!(
                "qwen VL encoder: prompt has {s} token(s), must exceed the {} dropped template tokens",
                Self::EDIT_DROP_IDX
            )));
        }
        let idx: Vec<i32> = (Self::EDIT_DROP_IDX..s).collect();
        let idx = Array::from_slice(&idx, &[idx.len() as i32]);
        Ok(hidden.take_axis(&idx, 1)?)
    }

    /// `input_ids` / `attention_mask`: `[b, s]` int32; `pixel_values`: `[n_patches, 1176]`; `grids`:
    /// one `(t, grid_h, grid_w)` per reference image. Returns the prompt embeds `[b, s-64, hidden]`.
    /// Convenience wrapper = [`encode_vision`](Self::encode_vision) then
    /// [`encode_with_vision`](Self::encode_with_vision).
    pub fn encode(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        pixel_values: &Array,
        grids: &[Grid],
    ) -> Result<Array> {
        let vision = self.encode_vision(pixel_values, grids)?;
        self.encode_with_vision(input_ids, attention_mask, &vision)
    }

    /// Replace `<|image_pad|>` embeddings with the vision embeds (in order) via a single gather:
    /// build `[text_embeds ‖ vision_embeds]` and index each output row at either its text position
    /// or the next vision row.
    fn splice(&self, embeds: &Array, input_ids: &Array, vision: &Array) -> Result<Array> {
        let sh = embeds.shape();
        let (b, s, h) = (sh[0], sh[1], sh[2]);
        let n_text = b * s;
        let n_vis = vision.shape()[0];
        let ids = host_i32(input_ids)?;

        let gather = image_gather_index(&ids, Self::IMAGE_TOKEN_ID, n_vis, n_text);
        let embeds_flat = embeds.reshape(&[n_text, h])?;
        let src = concatenate_axis(&[&embeds_flat, vision], 0)?; // [n_text + n_vis, h]
        let idx = Array::from_slice(&gather, &[n_text]);
        Ok(src.take_axis(&idx, 0)?.reshape(&[b, s, h])?)
    }
}

/// Gather indices into `[text_embeds(n_text) ‖ vision_embeds(n_vis)]`: image-token positions map to
/// the next vision row (`n_text + vi`), all others to their own text position. Mirrors the fork's
/// in-order replacement loop. Pure — unit-tested directly.
pub fn image_gather_index(ids: &[i32], image_token_id: i32, n_vis: i32, n_text: i32) -> Vec<i32> {
    let mut out = Vec::with_capacity(n_text as usize);
    let mut vi = 0i32;
    for (p, &id) in ids.iter().enumerate() {
        if id == image_token_id && vi < n_vis {
            out.push(n_text + vi);
            vi += 1;
        } else {
            out.push(p as i32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::image_gather_index;

    #[test]
    fn gather_replaces_image_tokens_in_order() {
        // ids: [a, PAD, PAD, b] with 4 text rows + 2 vision rows → vision at indices 4,5.
        let ids = [10, 151655, 151655, 11];
        let got = image_gather_index(&ids, 151655, 2, 4);
        assert_eq!(got, vec![0, 4, 5, 3]);
    }

    #[test]
    fn gather_stops_when_vision_exhausted() {
        // Only 1 vision row for 2 PADs: the second PAD keeps its text position.
        let ids = [151655, 151655, 7];
        let got = image_gather_index(&ids, 151655, 1, 3);
        assert_eq!(got, vec![3, 1, 2]);
    }
}
