//! sc-5136: the Bernini planner's host-side **data / templating pipeline** — turning a task + media
//! into the planner's tokenized inputs (token slots, 3-D MRoPE position ids, 4-D flex attention mask).
//!
//! This module holds the pieces that are pure host computation (no MLX graph, no tokenizer):
//!   - [`generate_unified_inputs`] — task + media → the `inputs_structure` conversation
//!     (`bernini/data_utils.py`): `[CLS]`, input videos, input images, the text prompt, the gen-target
//!     slot (`[SOG]`+`image_gen`+`[EOG]` for images / `[SOV]`+`video_gen`+`[EOV]` for video), `[EOS]`.
//!   - [`mrope_position_ids`] — `Qwen2_5_VLModel.get_rope_index`
//!     (`bernini/models/modeling_qwen2_5_vl.py`): the 3-D MRoPE `(3, L)` ids. Text runs are equal 1-D
//!     ramps carrying a running `max+1`; each vision block lays out t/h/w with temporal step
//!     `second_per_grid_t · tokens_per_second` (images use `0`).
//!   - [`build_attention_mask_4d`] — `build_custom_attention_mask`
//!     (`bernini/data/utils/attention_utils.py`): the `(1, L, L)` flex mask. Text / input-vit queries
//!     are causal over text+input-vit keys; gen-output queries additionally attend bidirectionally
//!     within their own segment id; nothing attends *into* the gen latents from the text side.
//!
//! The token layout these consume (vision_start + pad·N + vision_end, indexed pads remapped to the
//! plain image/video pad ids, the `token_type` 0/2/3 + `token_segment_ids` tagging) is produced by the
//! templating sub-piece ([`BerniniTemplate::encode_messages`], a later sc-5136 commit).

use mlx_rs::Array;
use serde_json::{json, Value};

use mlx_gen::{Error, Result};

/// Qwen2.5-VL / Bernini token-id + MRoPE constants (from the snapshot config).
#[derive(Clone, Debug)]
pub struct MRopeConfig {
    pub spatial_merge_size: i64,
    pub tokens_per_second: f64,
    pub image_token_id: i64,
    pub video_token_id: i64,
    pub vision_start_token_id: i64,
}

impl Default for MRopeConfig {
    fn default() -> Self {
        Self {
            spatial_merge_size: 2,
            tokens_per_second: 2.0,
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_token_id: 151652,
        }
    }
}

/// Build the Bernini `inputs_structure` conversation (`generate_unified_inputs`).
///
/// `input_image_hw` is `(height, width)` per input image (the reference reads these from the files);
/// `input_video_count` is the number of input videos (each is a bare `video` slot — the pixels are
/// resolved later). `output_t == 1` selects image generation (`[SOG]` / `image_gen` / `[EOG]`),
/// otherwise video generation (`[SOV]` / `video_gen` / `[EOV]`). Mirrors the reference's index
/// numbering: input image `image_index = i + input_video_count`; the gen-image `image_index =
/// num_input_images`; the gen-video `video_index = input_video_count`.
pub fn generate_unified_inputs(
    prompt: &str,
    input_image_hw: &[(i64, i64)],
    input_video_count: usize,
    output_t: i64,
    output_h: i64,
    output_w: i64,
) -> Vec<Value> {
    let mut s: Vec<Value> = vec![json!({"type": "special_token", "text": "[CLS]", "has_loss": 0})];

    let mut video_index = 0i64;
    for _ in 0..input_video_count {
        s.push(json!({"type": "video", "video_index": video_index, "decode_mode": "video"}));
        video_index += 1;
    }

    for (i, &(h, w)) in input_image_hw.iter().enumerate() {
        let idx = i as i64 + input_video_count as i64;
        s.push(json!({"type": "image", "image_index": idx, "height": h, "width": w}));
    }

    s.push(json!({"type": "text", "text": prompt, "has_loss": 0}));

    if output_t == 1 {
        s.push(json!({"type": "special_token", "text": "[SOG]", "has_loss": 1}));
        let target_idx = input_image_hw.len() as i64;
        s.push(json!({
            "type": "image_gen", "image_index": target_idx,
            "height": output_h, "width": output_w, "has_loss": 1,
        }));
        s.push(json!({"type": "special_token", "text": "[EOG]", "has_loss": 1}));
    } else {
        s.push(json!({"type": "special_token", "text": "[SOV]", "has_loss": 1}));
        s.push(json!({"type": "video_gen", "video_index": video_index, "decode_mode": "video"}));
        s.push(json!({"type": "special_token", "text": "[EOV]", "has_loss": 1}));
    }
    s.push(json!({"type": "special_token", "text": "[EOS]", "has_loss": 1}));
    s
}

/// 3-D MRoPE position ids `(3, L)` (`get_rope_index`), at inference (single sequence, full attention
/// mask). `image_grid_thw` / `video_grid_thw` are the per-item `[t, h, w]` grids in the order their
/// pad runs appear in `input_ids`. With no vision items this is the pure-text ramp `[0..L)` on all
/// three rows.
pub fn mrope_position_ids(
    input_ids: &[i64],
    image_grid_thw: &[[i64; 3]],
    video_grid_thw: &[[i64; 3]],
    cfg: &MRopeConfig,
) -> Result<Array> {
    let l = input_ids.len();
    let mut rows: [Vec<i64>; 3] = [
        Vec::with_capacity(l),
        Vec::with_capacity(l),
        Vec::with_capacity(l),
    ];

    if image_grid_thw.is_empty() && video_grid_thw.is_empty() {
        for r in rows.iter_mut() {
            *r = (0..l as i64).collect();
        }
    } else {
        let sms = cfg.spatial_merge_size;
        // image_nums / video_nums via the token right after each vision_start.
        let (mut image_nums, mut video_nums) = (0i64, 0i64);
        for (i, &t) in input_ids.iter().enumerate() {
            if t == cfg.vision_start_token_id && i + 1 < l {
                match input_ids[i + 1] {
                    x if x == cfg.image_token_id => image_nums += 1,
                    x if x == cfg.video_token_id => video_nums += 1,
                    _ => {}
                }
            }
        }

        let find_from = |tok: i64, from: usize| -> usize {
            input_ids[from..]
                .iter()
                .position(|&x| x == tok)
                .map(|p| p + from)
                .unwrap_or(l + 1)
        };

        let (mut image_index, mut video_index) = (0usize, 0usize);
        let (mut remain_images, mut remain_videos) = (image_nums, video_nums);
        let mut st = 0usize;
        let mut last_max: i64 = 0;
        let mut appended = false;

        for _ in 0..(image_nums + video_nums) {
            let ed_image = if remain_images > 0 {
                find_from(cfg.image_token_id, st)
            } else {
                l + 1
            };
            let ed_video = if remain_videos > 0 {
                find_from(cfg.video_token_id, st)
            } else {
                l + 1
            };

            let (g, second_per_grid_t, ed) = if ed_image < ed_video {
                let g = image_grid_thw[image_index];
                image_index += 1;
                remain_images -= 1;
                (g, 0.0f64, ed_image)
            } else {
                let g = video_grid_thw[video_index];
                video_index += 1;
                remain_videos -= 1;
                (g, 1.0f64, ed_video)
            };
            // `find_from` returns the `l+1` sentinel when a counted vision token isn't actually
            // present; using it as `ed` would make `text_len` overshoot and push more than `l` rows,
            // silently corrupting the MRoPE positions (or erroring at the final reshape). Reject the
            // malformed token sequence instead (F-023).
            if ed > l {
                return Err(Error::Msg(
                    "bernini mrope: a counted vision token was not found in input_ids \
                     (malformed token sequence)"
                        .into(),
                ));
            }
            let (lt, lh, lw) = (g[0], g[1] / sms, g[2] / sms);
            let text_len = (ed - st) as i64;
            let st_idx = if appended { last_max + 1 } else { 0 };

            // text ramp (all three rows equal): arange(text_len) + st_idx.
            for k in 0..text_len {
                for r in rows.iter_mut() {
                    r.push(st_idx + k);
                }
            }

            // vision block: t/h/w grid + (text_len + st_idx).
            let base = text_len + st_idx;
            let mut blk_max = base;
            for ti in 0..lt {
                let tval = ((ti as f64) * second_per_grid_t * cfg.tokens_per_second) as i64 + base;
                for hh in 0..lh {
                    for ww in 0..lw {
                        rows[0].push(tval);
                        rows[1].push(hh + base);
                        rows[2].push(ww + base);
                        blk_max = blk_max.max(tval).max(hh + base).max(ww + base);
                    }
                }
            }
            last_max = blk_max; // = llm_pos_ids_list[-1].max() (the vision block, appended last)
            appended = true;
            st = ed + (lt * lh * lw) as usize;
        }

        if st < l {
            let st_idx = if appended { last_max + 1 } else { 0 };
            let text_len = (l - st) as i64;
            for k in 0..text_len {
                for r in rows.iter_mut() {
                    r.push(st_idx + k);
                }
            }
        }
    }

    let mut data = Vec::with_capacity(3 * l);
    for r in &rows {
        data.extend(r.iter().map(|&v| v as i32));
    }
    Ok(Array::from_slice(&data, &[3, l as i32]))
}

/// 4-D flex attention mask `(1, L, L)` additive f32 (`0` visible / `-inf` masked)
/// (`build_custom_attention_mask`). `token_type`: `0` text, `1` p, `2` input-vit, `3` gen-output.
pub fn build_attention_mask_4d(token_type: &[i32], token_segment_ids: &[i32]) -> Result<Array> {
    let l = token_type.len();
    let mut data = vec![0f32; l * l];
    for qi in 0..l {
        let qt = token_type[qi];
        let qid = token_segment_ids[qi];
        for ki in 0..l {
            let kt = token_type[ki];
            let causal = ki <= qi;
            let k_is_ti = kt == 0 || kt == 2;
            let visible_base_ti = causal && k_is_ti;
            let ids_match = qid == token_segment_ids[ki];
            let visible = match qt {
                0 | 2 => visible_base_ti,
                1 => visible_base_ti || (kt == 1 && ids_match),
                3 => visible_base_ti || (kt == 3 && ids_match),
                _ => false,
            };
            if !visible {
                data[qi * l + ki] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Array::from_slice(&data, &[1, l as i32, l as i32]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// generate_unified_inputs lays out the conversation for image vs video generation.
    #[test]
    fn unified_inputs_image_vs_video() {
        // t2i: [CLS], text, [SOG], image_gen, [EOG], [EOS].
        let s = generate_unified_inputs("a cat", &[], 0, 1, 64, 64);
        let types: Vec<&str> = s.iter().map(|m| m["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            [
                "special_token",
                "text",
                "special_token",
                "image_gen",
                "special_token",
                "special_token"
            ]
        );
        assert_eq!(s[3]["image_index"], 0);
        assert_eq!(s[3]["height"], 64);

        // i2i: one input image, image gen — input image_index then gen image_index = 1.
        let s = generate_unified_inputs("edit", &[(48, 64)], 0, 1, 64, 64);
        assert_eq!(s[1]["type"], "image");
        assert_eq!(s[1]["image_index"], 0);
        assert_eq!(s[1]["height"], 48);
        let gen = s.iter().find(|m| m["type"] == "image_gen").unwrap();
        assert_eq!(gen["image_index"], 1);

        // rv2v: one input video + video gen — gen video_index follows the input count.
        let s = generate_unified_inputs("v", &[], 1, 5, 64, 64);
        assert_eq!(s[1]["type"], "video");
        assert_eq!(s[1]["video_index"], 0);
        let gen = s.iter().find(|m| m["type"] == "video_gen").unwrap();
        assert_eq!(gen["video_index"], 1);
    }

    /// Pure-text MRoPE is the plain ramp on all three rows.
    #[test]
    fn mrope_text_only_ramp() {
        let ids = vec![10i64, 11, 12, 13];
        let pos = mrope_position_ids(&ids, &[], &[], &MRopeConfig::default()).unwrap();
        assert_eq!(pos.shape(), &[3, 4]);
        let v: Vec<i32> = pos.flatten(None, None).unwrap().as_slice::<i32>().to_vec();
        assert_eq!(v, vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
    }

    /// The mask is causal for text, and gen-output tokens attend bidirectionally within their segment.
    #[test]
    fn mask_gen_bidirectional_within_segment() {
        // [text, gen, gen]  types [0,3,3], segs [0,1,1].
        let tt = [0, 3, 3];
        let seg = [0, 1, 1];
        let m = build_attention_mask_4d(&tt, &seg).unwrap();
        let v: Vec<f32> = m.flatten(None, None).unwrap().as_slice::<f32>().to_vec();
        let vis = |q: usize, k: usize| v[q * 3 + k].is_finite();
        // text query (causal over text): sees self only.
        assert!(vis(0, 0) && !vis(0, 1) && !vis(0, 2));
        // gen query 1 sees text (causal) + both gen of same segment (bidirectional, incl. future).
        assert!(vis(1, 0) && vis(1, 1) && vis(1, 2));
        // gen query 2 likewise.
        assert!(vis(2, 0) && vis(2, 1) && vis(2, 2));
    }
}
