//! FLUX.2-dev **caption upsampling** (sc-6030): the Pixtral vision tower's only consumer.
//!
//! An optional prompt-preprocess that rewrites the user prompt with the dev `Mistral3` multimodal
//! LLM before the diffusion encode — the diffusers `Flux2Pipeline.upsample_prompt` (gated there by
//! `caption_upsample_temperature`). Text-only for T2I, image-conditioned for edit (the reference
//! images flow through the vision tower → projector and are spliced into the Mistral input embeds at
//! the `[IMG]` positions; the language tower then autoregressively generates the rewritten prompt).
//!
//! Faithful to the reference flow: `format_input` (system message + `[INST]` turns) →
//! `PixtralProcessor` `[IMG]`/`[IMG_BREAK]`/`[IMG_END]` expansion → masked-scatter merge →
//! `generate(do_sample=True, temperature=0.15, max_new_tokens=512)` → decode the new tokens. The
//! Mistral language tower is the **same packed-Q4 [`Qwen3TextEncoder`]** the T2I path uses (no second
//! copy); only the final norm + LM head are extra (loaded by `load_generation_head`). Multimodal
//! splice reuses the shared `mlx_gen::mllm::splice_image_features` helper.
//!
//! Pixel preprocessing matches the reference up to the resample kernel and a small rounding choice:
//! references are horizontally concatenated (white bg, center), area-capped to 768², and resized to
//! a multiple of `patch·spatial_merge = 28` (the reference rounds to the patch (14) and floor-drops
//! the odd row/col in the projector's 2×2 unfold; rounding to 28 keeps the [`crate::vision`] projector's
//! strict even-grid contract, a ≤14 px difference that does not affect the sampled text). Pixel
//! parity is not the gate (the whole FLUX.2 path runs f32 and the upsampled prompt is sampled).

use mlx_rs::Array;

use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::media::Image;
use mlx_gen::mllm::splice_image_features;
use mlx_gen::runtime::CancelFlag;
use mlx_gen::{Error, Result};

use crate::text_encoder::{Qwen3TextEncoder, UpsampleSampling};
use crate::vision::{Mistral3Projector, PixtralVisionTower};

/// Mistral/Pixtral special-token ids (dev `tokenizer.json` `added_tokens`).
pub const IMAGE_TOKEN_ID: i32 = 10;
const IMAGE_BREAK_TOKEN_ID: i32 = 12;
const IMAGE_END_TOKEN_ID: i32 = 13;
/// `</s>` — the Mistral EOS that stops generation.
pub const EOS_TOKEN_ID: i32 = 2;

/// Pixtral vision config: patch 14, 2×2 spatial merge (dev `text_encoder/config.json`).
const PATCH_SIZE: i32 = 14;
const SPATIAL_MERGE: i32 = 2;
/// `patch · spatial_merge` — the effective downsample the projector's 2×2 merge applies, and the
/// multiple every upsampling image dimension is rounded to (so the patch grid is even).
const MERGE_PATCH: i32 = PATCH_SIZE * SPATIAL_MERGE;

/// Upsampling pixel-area cap (diffusers `UPSAMPLING_MAX_IMAGE_SIZE = 768²`).
const UPSAMPLING_MAX_AREA: f64 = 768.0 * 768.0;
/// CLIP image normalization (dev `preprocessor_config.json`) — the digits are the verbatim
/// reference config values (kept for provenance; a couple carry more precision than f32 holds).
#[allow(clippy::excessive_precision)]
const IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
#[allow(clippy::excessive_precision)]
const IMAGE_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// The reference `caption_upsample_temperature` default (`upsample_prompt(temperature=0.15)`).
pub const DEFAULT_TEMPERATURE: f32 = 0.15;
/// The reference `generate(max_new_tokens=512)`.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 512;
/// Hard ceiling on upsample decode length (F-012). Each decode step is a full ~32B forward over a
/// growing KV cache, so a request-supplied `enhance_max_tokens` must be capped or a single upsample
/// becomes an effectively unbounded job. 4× the 512 reference default leaves room for legitimately
/// long rewrites while bounding the worst case to ~2048 forwards instead of billions.
pub const MAX_NEW_TOKENS_CAP: usize = 2048;

/// Resolve the decode length from the request's `enhance_max_tokens`: the reference default
/// ([`DEFAULT_MAX_NEW_TOKENS`]) when unset, otherwise the requested value clamped to
/// [`MAX_NEW_TOKENS_CAP`] (F-012). A request is never *rejected* for asking too much — the advisory
/// knob is silently capped — so callers stay infallible.
pub fn clamp_max_new_tokens(requested: Option<u32>) -> usize {
    requested
        .map(|m| (m as usize).min(MAX_NEW_TOKENS_CAP))
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS)
}

/// `SYSTEM_MESSAGE_UPSAMPLING_T2I` (diffusers `flux2/system_messages.py`), used when no reference
/// images are present (pure T2I).
pub const SYSTEM_MESSAGE_UPSAMPLING_T2I: &str = "You are an expert prompt engineer for FLUX.2 by Black Forest Labs. Rewrite user prompts to be more descriptive while strictly preserving their core subject and intent.\n\nGuidelines:\n1. Structure: Keep structured inputs structured (enhance within fields). Convert natural language to detailed paragraphs.\n2. Details: Add concrete visual specifics - form, scale, textures, materials, lighting (quality, direction, color), shadows, spatial relationships, and environmental context.\n3. Text in Images: Put ALL text in quotation marks, matching the prompt's language. Always provide explicit quoted text for objects that would contain text in reality (signs, labels, screens, etc.) - without it, the model generates gibberish.\n\nOutput only the revised prompt and nothing else.";

/// `SYSTEM_MESSAGE_UPSAMPLING_I2I` (diffusers `flux2/system_messages.py`), used when reference
/// images are present (edit / image-conditioned).
pub const SYSTEM_MESSAGE_UPSAMPLING_I2I: &str = "You are FLUX.2 by Black Forest Labs, an image-editing expert. You convert editing requests into one concise instruction (50-80 words, ~30 for brief requests).\n\nRules:\n- Single instruction only, no commentary\n- Use clear, analytical language (avoid \"whimsical,\" \"cascading,\" etc.)\n- Specify what changes AND what stays the same (face, lighting, composition)\n- Reference actual image elements\n- Turn negatives into positives (\"don't change X\" → \"keep X\")\n- Make abstractions concrete (\"futuristic\" → \"glowing cyan neon, metallic panels\")\n- Keep content PG-13\n\nOutput only the final instruction in plain text and nothing else.";

/// Run the FLUX.2-dev caption-upsampling rewrite. `references` are the (already collected) edit
/// reference images: empty ⇒ the T2I path (text-only system message, no vision tower); non-empty ⇒
/// the I2I path (images concatenated + run through the tower, projected features spliced into the
/// prompt embeds). Returns the rewritten prompt text (the generated tokens, special tokens skipped).
///
/// `tokenizer` is the dev Mistral/Pixtral tokenizer; `encoder` must have its generation head loaded
/// ([`Qwen3TextEncoder::load_generation_head`]). Mirrors `Flux2Pipeline.upsample_prompt`.
#[allow(clippy::too_many_arguments)]
pub fn upsample_prompt(
    tokenizer: &mlx_gen::tokenizer::TextTokenizer,
    encoder: &Qwen3TextEncoder,
    vision_tower: &PixtralVisionTower,
    projector: &Mistral3Projector,
    prompt: &str,
    references: &[&Image],
    temperature: f32,
    max_new_tokens: usize,
    seed: u64,
    cancel: &CancelFlag,
) -> Result<String> {
    // System message + image grid by whether references are present (reference's T2I/I2I branch).
    let (system_message, image) = if references.is_empty() {
        (SYSTEM_MESSAGE_UPSAMPLING_T2I, None)
    } else {
        let (pixels, grid) = preprocess_upsample_image(references)?;
        (SYSTEM_MESSAGE_UPSAMPLING_I2I, Some((pixels, grid)))
    };

    // Build the prompt ids (chat template + image-token expansion), then the input embeds (splicing
    // the projected image features into the `[IMG]` positions for the I2I path).
    let merged_grid = image
        .as_ref()
        .map(|(_, (gh, gw))| (gh / SPATIAL_MERGE, gw / SPATIAL_MERGE));
    let ids = build_upsample_input_ids(tokenizer, system_message, prompt, merged_grid)?;
    let input_ids = Array::from_slice(&ids, &[1, ids.len() as i32]);
    let mut embeds = encoder.embed(&input_ids)?;

    if let Some((pixels, grid)) = &image {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let features = vision_tower.forward(&[pixels], &[*grid])?;
        let projected = projector.forward(&features, &[*grid])?;
        embeds = splice_image_features(&embeds, &input_ids, &projected, IMAGE_TOKEN_ID)?;
    }

    let sampling = UpsampleSampling {
        temperature,
        top_p: 1.0,
        max_new_tokens,
        seed,
    };
    let generated = encoder.generate_from_embeds(&embeds, EOS_TOKEN_ID, sampling, cancel)?;
    let tokens: Vec<u32> = generated.iter().map(|&id| id.max(0) as u32).collect();
    let text = tokenizer.decode(&tokens, true)?;
    Ok(text.trim().to_string())
}

/// Build the caption-upsampling input ids: render the chat template
/// (`[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]…[/INST]`, BOS auto-prepended) and, when an image
/// is present, expand its single `[IMG]` placeholder into the Pixtral grid layout. T2I omits the
/// image turn entirely (the reference's text-only `format_input`). `merged_grid` is the I2I image's
/// merged (post-2×2) patch grid `(num_h, num_w)`; `None` for T2I. Public so the token-layout parity
/// test exercises the exact production path against the reference `PixtralProcessor`.
pub fn build_upsample_input_ids(
    tokenizer: &mlx_gen::tokenizer::TextTokenizer,
    system_message: &str,
    prompt: &str,
    merged_grid: Option<(i32, i32)>,
) -> Result<Vec<i32>> {
    // Strip any literal `[IMG]` the user typed (the reference's `cleaned_txt`), so it never collides
    // with the image-token expansion.
    let cleaned = prompt.replace("[IMG]", "");
    let text = match merged_grid {
        // I2I: an image `[INST]` turn (a single `[IMG]` placeholder) followed by the text turn.
        Some(_) => format!(
            "[SYSTEM_PROMPT]{system_message}[/SYSTEM_PROMPT][INST][IMG][/INST][INST]{cleaned}[/INST]"
        ),
        // T2I: text turn only.
        None => format!("[SYSTEM_PROMPT]{system_message}[/SYSTEM_PROMPT][INST]{cleaned}[/INST]"),
    };
    // `add_special_tokens = true` → the tokenizer's post-processor prepends `<s>` (BOS), matching the
    // chat template's `{{ bos_token }}` (and the existing dev prompt-embeds path).
    let ids = tokenizer.encode_ids(&text, true)?;
    match merged_grid {
        Some(grid) => Ok(expand_pixtral_image_tokens(&ids, grid)),
        None => Ok(ids),
    }
}

/// Expand the single `[IMG]` (id 10) placeholder into the Pixtral 2-D token layout for a merged
/// patch grid `(num_h, num_w)`: each of `num_h` rows is `num_w` × `[IMG]` then one `[IMG_BREAK]`
/// (id 12), and the final `[IMG_BREAK]` becomes `[IMG_END]` (id 13). The `[IMG]` count is therefore
/// `num_h·num_w` — exactly the projector's merged-token (and projected-feature) count, so the splice
/// lines up. Port of `PixtralProcessor`'s token replacement. Only the leading `[IMG]` is expanded
/// (the upsampling path concatenates all references into one image).
pub fn expand_pixtral_image_tokens(ids: &[i32], merged_grid: (i32, i32)) -> Vec<i32> {
    let (num_h, num_w) = merged_grid;
    let mut out = Vec::with_capacity(ids.len() + (num_h * (num_w + 1)) as usize);
    let mut expanded = false;
    for &id in ids {
        if id == IMAGE_TOKEN_ID && !expanded {
            for row in 0..num_h {
                out.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, num_w.max(0) as usize));
                out.push(if row == num_h - 1 {
                    IMAGE_END_TOKEN_ID
                } else {
                    IMAGE_BREAK_TOKEN_ID
                });
            }
            expanded = true;
        } else {
            out.push(id);
        }
    }
    out
}

/// Preprocess reference images for the Pixtral tower (upsampling I2I path): horizontally concatenate
/// (white background, vertical center) → cap area to 768² → round to a multiple of `MERGE_PATCH`
/// (28) → LANCZOS resize → rescale `1/255` + CLIP per-channel normalize → NHWC `[1, H, W, 3]` f32.
/// Returns the pixel tensor and the **patch** grid `(gh, gw) = (H/14, W/14)` (always even).
fn preprocess_upsample_image(references: &[&Image]) -> Result<(Array, (i32, i32))> {
    if references.is_empty() {
        return Err(Error::Msg(
            "flux2 caption-upsample: no reference image to preprocess".to_owned(),
        ));
    }
    let concat = concatenate_horizontal(references)?;
    let (w0, h0) = (concat.width as f64, concat.height as f64);

    // Area cap (diffusers `_resize_if_exceeds_area`), then round each side up to a MERGE_PATCH
    // multiple (≥ 1 tile) so the patch grid is even.
    let area = w0 * h0;
    let scale = if area > UPSAMPLING_MAX_AREA {
        (UPSAMPLING_MAX_AREA / area).sqrt()
    } else {
        1.0
    };
    let round_up = |x: f64| -> i32 {
        let v = (x * scale).round() as i32;
        let tiles = ((v - 1).max(0) / MERGE_PATCH) + 1;
        tiles * MERGE_PATCH
    };
    let (tw, th) = (round_up(w0), round_up(h0));

    let resized: Vec<f32> = resize_lanczos_u8(
        &concat.pixels,
        concat.height as usize,
        concat.width as usize,
        th as usize,
        tw as usize,
    );
    // rescale + per-channel normalize, NHWC row-major (channel index is the innermost dim).
    let norm: Vec<f32> = resized
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let c = i % 3;
            (v / 255.0 - IMAGE_MEAN[c]) / IMAGE_STD[c]
        })
        .collect();
    let pixels = Array::from_slice(&norm, &[1, th, tw, 3]);
    Ok((pixels, (th / PATCH_SIZE, tw / PATCH_SIZE)))
}

/// Horizontally concatenate RGB images on a white background with vertical center alignment (the
/// reference `concatenate_images`). A single image is returned as-is.
fn concatenate_horizontal(images: &[&Image]) -> Result<Image> {
    for im in images {
        if im.pixels.len() != im.width as usize * im.height as usize * 3 {
            return Err(Error::Msg(format!(
                "flux2 caption-upsample: image pixel buffer {} != {}x{}x3",
                im.pixels.len(),
                im.width,
                im.height
            )));
        }
    }
    if images.len() == 1 {
        return Ok((*images[0]).clone());
    }
    // Sum widths in u64 (the sum itself can overflow u32 on adversarial inputs) and require the total
    // to fit the output `Image`'s u32 width. Then size the canvas with checked usize arithmetic so a
    // wrapped (too-small) allocation can't later trigger an OOB `copy_from_slice` (L-E): `total_w *
    // max_h * 3` can overflow u32 — and even usize on 64-bit for huge dims.
    let total_w_u64: u64 = images.iter().map(|im| im.width as u64).sum();
    let max_h: u32 = images.iter().map(|im| im.height).max().unwrap_or(0);
    let total_w = u32::try_from(total_w_u64).map_err(|_| {
        Error::Msg(format!(
            "flux2 caption-upsample: concatenated width {total_w_u64} exceeds u32"
        ))
    })?;
    let canvas_len = (total_w as usize)
        .checked_mul(max_h as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| {
            Error::Msg(format!(
                "flux2 caption-upsample: canvas {total_w}×{max_h}×3 overflows usize"
            ))
        })?;
    let mut canvas = vec![255u8; canvas_len];
    let mut x_off: u32 = 0;
    for im in images {
        let y_off = (max_h - im.height) / 2;
        for row in 0..im.height {
            // Index in usize, not u32: `(dst_row * total_w + x_off) * 3` can wrap u32 even when the
            // (checked) canvas length fits, which would corrupt the copy bounds.
            let dst_row = (y_off + row) as usize;
            let dst_start = (dst_row * total_w as usize + x_off as usize) * 3;
            let src_start = (row as usize * im.width as usize) * 3;
            let len = im.width as usize * 3;
            canvas[dst_start..dst_start + len]
                .copy_from_slice(&im.pixels[src_start..src_start + len]);
        }
        x_off += im.width;
    }
    Ok(Image {
        width: total_w,
        height: max_h,
        pixels: canvas,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F-012: `clamp_max_new_tokens` defaults when unset and caps the request to `MAX_NEW_TOKENS_CAP`.
    #[test]
    fn clamp_max_new_tokens_defaults_and_caps() {
        assert_eq!(clamp_max_new_tokens(None), DEFAULT_MAX_NEW_TOKENS);
        assert_eq!(clamp_max_new_tokens(Some(100)), 100);
        assert_eq!(
            clamp_max_new_tokens(Some(MAX_NEW_TOKENS_CAP as u32)),
            MAX_NEW_TOKENS_CAP
        );
        assert_eq!(
            clamp_max_new_tokens(Some(MAX_NEW_TOKENS_CAP as u32 + 1)),
            MAX_NEW_TOKENS_CAP
        );
        assert_eq!(clamp_max_new_tokens(Some(u32::MAX)), MAX_NEW_TOKENS_CAP);
    }

    #[test]
    fn expands_single_img_into_grid_with_breaks_and_end() {
        // ids: [<s>, [SYSTEM_PROMPT]..., [INST], [IMG], [/INST], ...]; only the [IMG] expands.
        let ids = vec![1, 17, 99, 18, 3, IMAGE_TOKEN_ID, 4, 3, 100, 4];
        let out = expand_pixtral_image_tokens(&ids, (2, 3)); // 2 rows × 3 cols
                                                             // 2 rows: [10,10,10,12] then [10,10,10,13] → 8 tokens replacing the one [IMG].
        let img_region = vec![10, 10, 10, 12, 10, 10, 10, 13];
        let mut expected = vec![1, 17, 99, 18, 3];
        expected.extend(img_region);
        expected.extend([4, 3, 100, 4]);
        assert_eq!(out, expected);
        // [IMG](10) count == num_h·num_w == projector merged rows.
        assert_eq!(out.iter().filter(|&&t| t == IMAGE_TOKEN_ID).count(), 6);
    }

    #[test]
    fn expand_is_noop_without_image_token() {
        let ids = vec![1, 17, 50, 18, 3, 100, 4];
        assert_eq!(expand_pixtral_image_tokens(&ids, (4, 4)), ids);
    }

    #[test]
    fn single_image_preprocess_rounds_to_even_patch_grid() {
        // 100×60 image (area < 768²) → rounded up to multiples of 28 → 112×84 (gh=8, gw... w=112).
        let img = Image {
            width: 100,
            height: 60,
            pixels: vec![128u8; 100 * 60 * 3],
        };
        let (pixels, (gh, gw)) = preprocess_upsample_image(&[&img]).unwrap();
        let sh = pixels.shape();
        assert_eq!(sh[0], 1);
        assert_eq!(sh[3], 3);
        // dims are multiples of 28 → patch grid (H/14, W/14) is even.
        assert_eq!(sh[1] % MERGE_PATCH, 0);
        assert_eq!(sh[2] % MERGE_PATCH, 0);
        assert_eq!(gh, sh[1] / PATCH_SIZE);
        assert_eq!(gw, sh[2] / PATCH_SIZE);
        assert_eq!(gh % 2, 0);
        assert_eq!(gw % 2, 0);
    }

    #[test]
    fn area_cap_shrinks_large_image_below_768_squared() {
        let img = Image {
            width: 2000,
            height: 2000,
            pixels: vec![10u8; 2000 * 2000 * 3],
        };
        let (pixels, _) = preprocess_upsample_image(&[&img]).unwrap();
        let sh = pixels.shape();
        // area must be at/under the 768² cap (after rounding to 28, allow one tile of slack).
        assert!((sh[1] as f64) * (sh[2] as f64) <= 768.0 * 768.0 + 2.0 * 28.0 * 768.0);
    }

    #[test]
    fn horizontal_concat_widths_sum_heights_max() {
        let a = Image {
            width: 4,
            height: 2,
            pixels: vec![1u8; 4 * 2 * 3],
        };
        let b = Image {
            width: 6,
            height: 3,
            pixels: vec![2u8; 6 * 3 * 3],
        };
        let c = concatenate_horizontal(&[&a, &b]).unwrap();
        assert_eq!(c.width, 10);
        assert_eq!(c.height, 3);
        assert_eq!(c.pixels.len(), 10 * 3 * 3);
    }

    #[test]
    fn t2i_and_i2i_system_messages_differ() {
        assert!(SYSTEM_MESSAGE_UPSAMPLING_T2I.contains("expert prompt engineer"));
        assert!(SYSTEM_MESSAGE_UPSAMPLING_I2I.contains("image-editing expert"));
        assert_ne!(SYSTEM_MESSAGE_UPSAMPLING_T2I, SYSTEM_MESSAGE_UPSAMPLING_I2I);
    }
}
