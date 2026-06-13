//! sc-5136: the Bernini planner's ChatML templating (`BerniniTemplate.encode_messages`).
//!
//! Turns the [`crate::process::generate_unified_inputs`] conversation + the per-visual token counts
//! (`grid.prod / merge²`) into the planner's `input_ids` plus the `token_type` / `token_segment_ids`
//! / `flex_token_types` tagging that [`crate::process::build_attention_mask_4d`] and the MAR loop
//! consume. Port of `BerniniTemplate.encode_messages`
//! (`_vendor/bernini/bernini/data/bernini_template.py`).
//!
//! Layout per message: `"<|im_start|>{role}\n"` (role tokens) + the content tokens, with no
//! `<|im_end|>` separator (the reference omits it). A system message (per-task system prompt) leads;
//! the conversation is grouped into a `user` message (`has_loss 0`: input visuals + text) and an
//! `assistant` message (`has_loss 1`: the gen-target visual). The special-token markers
//! (`[CLS]`/`[SOG]`/…) carry no tokens.
//!
//! Each visual is `<|vision_start|>` + `pad·token_num` + `<|vision_end|>`. The reference builds
//! *indexed* visual pads, tokenizes the whole content string, then remaps them to the plain
//! `image_pad`/`video_pad` ids; this port emits the **plain pad ids directly** and tracks
//! `token_type` (`2` input-vit / `3` gen-output), `token_segment_ids` (`visual_id + 1` on a visual's
//! pads, else the position index), and `flex_token_types` (the gen visual's running indicator id)
//! **during assembly** — equivalent because special tokens always split BPE, so piece-wise
//! tokenization equals the reference's whole-content-string tokenization (proven by the golden).
//!
//! This is the inference path: no train-time dropout, `vit_mask_ratio = 1` (mask all gen-vit tokens).

use serde_json::Value;
use tokenizers::Tokenizer;

use mlx_gen::{Error, Result};

const VISION_START_ID: i64 = 151652;
const VISION_END_ID: i64 = 151653;
const IMAGE_PAD_ID: i64 = 151655;
const VIDEO_PAD_ID: i64 = 151656;

/// Per-task system prompt (`bernini_template.SYSTEM_PROMPT`).
fn system_prompt(task: &str) -> &'static str {
    match task {
        "t2i" => "You are a helpful assistant specialized in text-to-image generation.",
        "t2v" => "You are a helpful assistant specialized in text-to-video generation.",
        "i2i" => "You are a helpful assistant specialized in image editing.",
        "v2v" => "You are a helpful assistant specialized in video editing.",
        "r2v" => "You are a helpful assistant specialized in subject-to-video generation.",
        "rv2v" => "You are a helpful assistant specialized in video editing with reference.",
        _ => "You are a helpful assistant.",
    }
}

/// The structural outputs of [`BerniniTemplate::encode_messages`] (all host-side `Vec`s; the MLX
/// arrays — `position_ids`, the 4-D mask — are built by [`crate::process`] from these).
#[derive(Clone, Debug, Default)]
pub struct TemplateOutput {
    /// MLLM token ids (visual pads are the plain `image_pad`/`video_pad` ids).
    pub input_ids: Vec<i64>,
    /// `0` text, `2` input-vit, `3` gen-output.
    pub token_type: Vec<i32>,
    /// `visual_id + 1` on a visual's pads, else the token's position index.
    pub token_segment_ids: Vec<i32>,
    /// `-1` everywhere except gen-output pads (their visual's indicator id).
    pub flex_token_types: Vec<i32>,
    /// `0` image / `1` video, one per visual in conversation order.
    pub vit_type_list: Vec<i32>,
    /// The per-type running id (image→`img_id`, video→`vid_id`) of each visual.
    pub vit_img_and_vid_id_list: Vec<i32>,
    /// `has_loss` per image visual (input `0` / gen `1`).
    pub image_target_mask: Vec<i32>,
    /// `has_loss` per video visual.
    pub video_target_mask: Vec<i32>,
    /// `0` image / `1` video, one per visual (drives VAE packing order).
    pub vae_type_list: Vec<i32>,
}

/// One content element of a grouped message.
enum Op {
    Text(String),
    Visual {
        pad_id: i64,
        ttype: i32,
        seg: i32,
        flex: i32,
        n: i64,
    },
}

/// The Bernini ChatML template over the Qwen2.5-VL tokenizer.
pub struct BerniniTemplate {
    tok: Tokenizer,
}

impl BerniniTemplate {
    /// Load from a `tokenizer.json` (the snapshot `mllm/tokenizer.json`).
    pub fn from_tokenizer_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let tok = Tokenizer::from_file(path.as_ref())
            .map_err(|e| Error::Msg(format!("load tokenizer: {e}")))?;
        Ok(Self { tok })
    }

    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| Error::Msg(format!("tokenize: {e}")))?;
        Ok(enc.get_ids().iter().map(|&x| x as i64).collect())
    }

    /// Encode a Bernini conversation into the planner's tokenized inputs. `image_token_nums` /
    /// `video_token_nums` are `grid.prod / merge²` per image / video, in conversation order.
    pub fn encode_messages(
        &self,
        conversation: &[Value],
        image_token_nums: &[i64],
        video_token_nums: &[i64],
        task: &str,
    ) -> Result<TemplateOutput> {
        let mut out = TemplateOutput::default();

        // ---- Phase 1: group by has_loss into ordered content ops + collect visual metadata. ----
        let mut groups: Vec<(i32, Vec<Op>)> = Vec::new();
        let mut cur_has_loss = 0i32;
        let mut cur_ops: Vec<Op> = Vec::new();
        let mut started = false;

        let (mut img_cur, mut vid_cur) = (0usize, 0usize);
        let (mut visual_id, mut img_id, mut vid_id) = (0i32, 0i32, 0i32);
        let mut indicator = 2i32;

        for m in conversation {
            let ty = m.get("type").and_then(Value::as_str).unwrap_or("");
            if ty == "special_token" {
                continue;
            }
            let has_loss = m
                .get("has_loss")
                .and_then(Value::as_i64)
                .map(|x| x as i32)
                .unwrap_or(if ty == "video_gen" { 1 } else { 0 });

            if !started {
                cur_has_loss = has_loss;
                started = true;
            } else if has_loss != cur_has_loss {
                groups.push((cur_has_loss, std::mem::take(&mut cur_ops)));
                cur_has_loss = has_loss;
            }

            match ty {
                "text" | "cot_text" => {
                    cur_ops.push(Op::Text(
                        m.get("text")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                    ));
                }
                "image" | "image_gen" => {
                    let n = image_token_nums[img_cur];
                    img_cur += 1;
                    let (ttype, flex) = if has_loss == 1 {
                        indicator += 1;
                        (3, indicator)
                    } else {
                        (2, -1)
                    };
                    indicator += 1;
                    cur_ops.push(Op::Visual {
                        pad_id: IMAGE_PAD_ID,
                        ttype,
                        seg: visual_id + 1,
                        flex,
                        n,
                    });
                    out.vit_type_list.push(0);
                    out.vit_img_and_vid_id_list.push(img_id);
                    out.image_target_mask.push(has_loss);
                    out.vae_type_list.push(0);
                    img_id += 1;
                    visual_id += 1;
                }
                "video" | "frame_gen" | "video_gen" => {
                    let n = video_token_nums[vid_cur];
                    vid_cur += 1;
                    let (ttype, flex) = if has_loss == 1 {
                        indicator += 1;
                        (3, indicator)
                    } else {
                        (2, -1)
                    };
                    indicator += 1;
                    cur_ops.push(Op::Visual {
                        pad_id: VIDEO_PAD_ID,
                        ttype,
                        seg: visual_id + 1,
                        flex,
                        n,
                    });
                    out.vit_type_list.push(1);
                    out.vit_img_and_vid_id_list.push(vid_id);
                    out.video_target_mask.push(has_loss);
                    out.vae_type_list.push(1);
                    vid_id += 1;
                    visual_id += 1;
                }
                other => return Err(Error::Msg(format!("unknown message type: {other}"))),
            }
        }
        if started {
            groups.push((cur_has_loss, cur_ops));
        }

        // ---- Phase 2: tokenize system + each group, tracking type/segment/flex. ----
        self.emit_message(
            "system",
            &[Op::Text(system_prompt(task).to_string())],
            &mut out,
        )?;
        for (has_loss, ops) in &groups {
            let role = if *has_loss == 0 { "user" } else { "assistant" };
            self.emit_message(role, ops, &mut out)?;
        }
        Ok(out)
    }

    /// Emit one ChatML message: `"<|im_start|>{role}\n"` role tokens + content. Content is the ops in
    /// order, with `str.strip()` applied to the whole concatenation (so the leading text op is
    /// l-trimmed and the trailing text op is r-trimmed). A fully-empty message is skipped.
    fn emit_message(&self, role: &str, ops: &[Op], out: &mut TemplateOutput) -> Result<()> {
        // Apply the outer strip: trim the first op if Text, the last op if Text.
        let mut texts: Vec<Option<String>> = ops
            .iter()
            .map(|op| match op {
                Op::Text(t) => Some(t.clone()),
                Op::Visual { .. } => None,
            })
            .collect();
        if let Some(Some(t)) = texts.first_mut() {
            *t = t.trim_start().to_string();
        }
        if let Some(Some(t)) = texts.last_mut() {
            *t = t.trim_end().to_string();
        }

        // Is the content empty? (no visuals and all text trimmed away)
        let has_visual = ops.iter().any(|op| matches!(op, Op::Visual { .. }));
        let any_text = texts
            .iter()
            .any(|t| t.as_ref().is_some_and(|s| !s.is_empty()));
        if !has_visual && !any_text {
            return Ok(());
        }

        for t in self.encode(&format!("<|im_start|>{role}\n"))? {
            out.push_token(t, 0, -1);
        }
        // Coalesce consecutive text, emit visuals in place.
        let mut buf = String::new();
        let flush = |buf: &mut String, out: &mut TemplateOutput| -> Result<()> {
            if !buf.is_empty() {
                for t in self.encode(buf)? {
                    out.push_token(t, 0, -1);
                }
                buf.clear();
            }
            Ok(())
        };
        for (op, text) in ops.iter().zip(texts) {
            match op {
                Op::Text(_) => {
                    if let Some(t) = text {
                        buf.push_str(&t);
                    }
                }
                Op::Visual {
                    pad_id,
                    ttype,
                    seg,
                    flex,
                    n,
                } => {
                    flush(&mut buf, out)?;
                    out.push_token(VISION_START_ID, 0, -1);
                    for _ in 0..*n {
                        out.push_pad(*pad_id, *ttype, *seg, *flex);
                    }
                    out.push_token(VISION_END_ID, 0, -1);
                }
            }
        }
        flush(&mut buf, out)?;
        Ok(())
    }
}

impl TemplateOutput {
    /// Push a non-pad token: `token_segment_ids` defaults to the position index.
    fn push_token(&mut self, id: i64, ttype: i32, flex: i32) {
        let pos = self.input_ids.len() as i32;
        self.input_ids.push(id);
        self.token_type.push(ttype);
        self.token_segment_ids.push(pos);
        self.flex_token_types.push(flex);
    }

    /// Push a visual pad token with its visual's segment id.
    fn push_pad(&mut self, id: i64, ttype: i32, seg: i32, flex: i32) {
        self.input_ids.push(id);
        self.token_type.push(ttype);
        self.token_segment_ids.push(seg);
        self.flex_token_types.push(flex);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompts_by_task() {
        assert!(system_prompt("i2i").contains("image editing"));
        assert!(system_prompt("rv2v").contains("video editing with reference"));
        assert_eq!(system_prompt("???"), "You are a helpful assistant.");
    }
}
