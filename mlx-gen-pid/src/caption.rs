//! PiD caption encoding — the host-side glue around the Gemma-2 decoder. Faithful port of
//! `pixeldit_model._encode_text_raw`: prepend the fixed **Chi-prompt**, tokenize (`add_special_tokens`
//! → leading `<bos>`) and right-pad/truncate to `num_chi_tokens + model_max_length − 2`, run the Gemma
//! decoder (with the padding mask), then gather `select_index = [0] + range(-(model_max_length−1), 0)`
//! → `caption_embs [1, model_max_length, 2304]`.
//!
//! Note: the `y_norm`/`y_norm_scale_factor` config knob is **never applied** in the reference code
//! (dead config), so we do not scale; and the inference net runs **without** a caption mask (the
//! `emb_masks` are discarded by `generate_samples_from_batch`), so only Gemma sees the padding mask.

use std::path::Path;

use mlx_rs::Array;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::Result;

use crate::gemma2::Gemma2;

/// The fixed "Chi-prompt" instruction prefix (`experiment/shared_config.py::_CHI_PROMPT`, joined by
/// `\n`). The user caption is appended directly after the trailing `"User Prompt: "`.
pub const CHI_PROMPT: &str = "Given a user prompt, generate an \"Enhanced prompt\" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:\n- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.\n- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\nHere are examples of how to transform or refine prompts:\n- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\nPlease generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:\nUser Prompt: ";

const MODEL_MAX_LENGTH: i32 = 300;
const PAD_ID: i32 = 0;

/// Gemma-2 caption encoder: tokenizer + Chi-prompt + the released token-selection policy.
pub struct CaptionEncoder {
    gemma: Gemma2,
    tok: TextTokenizer,
    num_chi_tokens: i32,
}

impl CaptionEncoder {
    /// Build from a constructed [`Gemma2`] and the gemma `tokenizer.json` path.
    pub fn new(gemma: Gemma2, tokenizer_json: impl AsRef<Path>) -> Result<Self> {
        let tok = TextTokenizer::from_file(
            tokenizer_json,
            TokenizerConfig {
                max_length: 4096,
                pad_token_id: PAD_ID,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )?;
        // num_chi_tokens counts the Chi-prompt WITH its special tokens (the reference's
        // `tokenizer.encode(chi_prompt_str)` adds the bos).
        let num_chi_tokens = tok.encode_ids(CHI_PROMPT, true)?.len() as i32;
        Ok(Self {
            gemma,
            tok,
            num_chi_tokens,
        })
    }

    /// Chi-prompt token count (the reference's `_num_chi_tokens`).
    pub fn num_chi_tokens(&self) -> i32 {
        self.num_chi_tokens
    }

    /// The padded `[input_ids, attention_mask]` for a caption — exposed so the tokenizer + Chi-prompt
    /// + length policy can be parity-checked against the reference without the Gemma weights.
    pub fn token_ids(&self, caption: &str) -> Result<(Vec<i32>, Vec<i32>)> {
        let max_len = self.num_chi_tokens + MODEL_MAX_LENGTH - 2;
        let mut ids = self
            .tok
            .encode_ids(&format!("{CHI_PROMPT}{caption}"), true)?;
        ids.truncate(max_len as usize);
        let real = ids.len();
        ids.resize(max_len as usize, PAD_ID);
        let mask = (0..max_len as usize).map(|i| (i < real) as i32).collect();
        Ok((ids, mask))
    }

    /// Encode one caption to `[1, 300, 2304]` caption embeddings.
    pub fn encode(&self, caption: &str) -> Result<Array> {
        let (ids, mask) = self.token_ids(caption)?;
        let max_len = ids.len() as i32;
        let ids_arr = Array::from_slice(&ids, &[1, max_len]);
        let mask_arr = Array::from_slice(&mask, &[1, max_len]);
        let hidden = self.gemma.forward(&ids_arr, Some(&mask_arr))?; // [1, max_len, 2304]

        // select_index = [0] + range(max_len-(300-1), max_len)
        let mut sel = Vec::with_capacity(MODEL_MAX_LENGTH as usize);
        sel.push(0);
        sel.extend((max_len - (MODEL_MAX_LENGTH - 1))..max_len);
        let sel_arr = Array::from_slice(&sel, &[MODEL_MAX_LENGTH]);
        Ok(hidden.take_axis(&sel_arr, 1)?) // [1, 300, 2304]
    }
}
