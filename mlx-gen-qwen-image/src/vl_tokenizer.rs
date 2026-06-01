//! Qwen-Image-Edit VL tokenization (sc-2465 slice 6b-2). Port of the fork's
//! `QwenVisionLanguageTokenizer` (`use_picture_prefix=False`) + `QwenVisionLanguageProcessor`:
//! turn a reference image + edit prompt into the four inputs the [`QwenVisionLanguageEncoder`]
//! consumes — `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`.
//!
//! Pipeline: resize the reference to the **condition size** (~384² area, sides rounded to /32,
//! BICUBIC) → the [`QwenImageProcessor`] (smart_resize to /28 + patchify, already parity-tested) →
//! expand the single `<|image_pad|>` in the fixed edit template to `prod(grid)//merge²` copies →
//! tokenize the formatted string with special tokens.
//!
//! [`QwenVisionLanguageEncoder`]: crate::text_encoder::QwenVisionLanguageEncoder

use mlx_rs::Array;

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::Result;

use crate::image_processor::{resize_bicubic_u8, ImageInput, QwenImageProcessor};

/// Inputs for [`crate::text_encoder::QwenVisionLanguageEncoder::encode`].
pub struct EditInputs {
    pub input_ids: Array,
    pub attention_mask: Array,
    pub pixel_values: Array,
    pub grid_thw: Array,
}

/// Target condition area (fork `CONDITION_IMAGE_SIZE`): the reference is scaled to ~384²,
/// preserving aspect, with each side rounded to a multiple of 32.
const CONDITION_AREA: f64 = 384.0 * 384.0;

/// Condition-resize dims `(width, height)`: `w = round(√(area·ratio)/32)·32`, `h = round((w/ratio)
/// /32)·32`, `ratio = width/height`. Round-half-to-even (Python `round`).
pub fn condition_resize_dims(width: usize, height: usize) -> (usize, usize) {
    let ratio = width as f64 / height as f64;
    let cw = (CONDITION_AREA * ratio).sqrt();
    let ch = cw / ratio;
    let cw = (cw / 32.0).round_ties_even() * 32.0;
    let ch = (ch / 32.0).round_ties_even() * 32.0;
    (cw as usize, ch as usize)
}

/// The fork's edit chat template (`use_picture_prefix=False`) with the single `<|image_pad|>`
/// expanded to `n_image_tokens` copies and the user prompt inserted.
pub fn build_edit_text(prompt: &str, n_image_tokens: usize) -> String {
    let pads = "<|image_pad|>".repeat(n_image_tokens);
    format!(
        "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, \
         texture, objects, background), then explain how the user's text instruction should alter \
         or modify the image. Generate a new image that meets the user's requirements while \
         maintaining consistency with the original input where appropriate.<|im_end|>\n\
         <|im_start|>user\n<|vision_start|>{pads}<|vision_end|>{prompt}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Tokenize a reference image + edit prompt for the VL encoder. `image` is RGB uint8 HWC.
pub fn tokenize_edit(
    tokenizer: &TextTokenizer,
    processor: &QwenImageProcessor,
    prompt: &str,
    image: ImageInput,
) -> Result<EditInputs> {
    // 1. Condition resize (BICUBIC, /32) — clip8-rounded f32 back to u8 for the processor.
    let (cw, ch) = condition_resize_dims(image.width, image.height);
    let resized: Vec<u8> = if (image.height, image.width) == (ch, cw) {
        image.data.to_vec()
    } else {
        resize_bicubic_u8(image.data, image.height, image.width, ch, cw)
            .iter()
            .map(|&v| v as u8)
            .collect()
    };

    // 2. Patchify (smart_resize to /28 + flatten) — parity-tested in image_processor.
    let processed = processor.preprocess(ImageInput {
        data: &resized,
        height: ch,
        width: cw,
    })?;

    // 3. Expand <|image_pad|> to prod(grid)//merge² and tokenize the formatted template.
    let grid = processed.grid_thw.as_slice::<i32>(); // [grid_t, grid_h, grid_w]
    let merge2 = (processor.merge_size * processor.merge_size) as i32;
    let n_image_tokens = (grid[0] * grid[1] * grid[2] / merge2) as usize;
    let text = build_edit_text(prompt, n_image_tokens);
    let tok = tokenizer.tokenize_preformatted(&text)?;

    Ok(EditInputs {
        input_ids: tok.input_ids,
        attention_mask: tok.attention_mask,
        pixel_values: processed.pixel_values,
        grid_thw: processed.grid_thw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_text_expands_image_pad_and_keeps_structure() {
        let t = build_edit_text("make it night", 3);
        assert_eq!(t.matches("<|image_pad|>").count(), 3);
        assert!(t.contains("<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>make it night<|im_end|>"));
        assert!(t.starts_with("<|im_start|>system\nDescribe the key features"));
        assert!(t.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn condition_resize_dims_match_reference() {
        assert_eq!(condition_resize_dims(512, 512), (384, 384)); // square
        assert_eq!(condition_resize_dims(768, 512), (480, 320)); // 3:2 landscape
        assert_eq!(condition_resize_dims(512, 768), (320, 480)); // 2:3 portrait
    }
}
