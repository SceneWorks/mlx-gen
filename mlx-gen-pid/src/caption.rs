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

/// PiD's fixed "Chi-prompt" instruction prefix (`experiment/shared_config.py::_CHI_PROMPT`, joined by
/// `\n`). The user caption is appended directly after the trailing `"User Prompt: "`.
///
/// This is the same Complex-Human-Instruction (CHI) template SANA uses (the two architectures share
/// the gemma-2-2b-it CHI caption-encoder lineage); they differ **only** in the quoting around
/// `Enhanced prompt` — PiD's released text uses escaped double-quotes here, SANA's
/// `complex_human_instruction` list uses single-quotes (see [`crate::caption::CaptionEncoder::with_chi_prompt`]
/// and `mlx-gen-sana`'s `SANA_CHI_PROMPT`). Because that difference changes the tokenization, the
/// CHI prompt is parameterized rather than hardcoded — do NOT assume PiD's text for SANA.
pub const CHI_PROMPT: &str = "Given a user prompt, generate an \"Enhanced prompt\" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:\n- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.\n- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\nHere are examples of how to transform or refine prompts:\n- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\nPlease generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:\nUser Prompt: ";

const MODEL_MAX_LENGTH: i32 = 300;
const PAD_ID: i32 = 0;

/// Gemma-2 caption encoder: tokenizer + CHI-prompt + the released token-selection policy.
///
/// This is the shared SANA-lineage text-conditioning path. PiD and SANA both: prepend a fixed CHI
/// prompt, tokenize (`add_special_tokens` → leading `<bos>`) and right-pad/truncate to
/// `num_chi_tokens + 300 − 2`, run the Gemma-2 decoder (encoder/last-hidden mode, with the padding
/// mask), then gather `select_index = [0] + range(-(300−1), 0)` → `[1, 300, 2304]`. The only knob
/// that differs is the CHI prompt text, so it is a constructor parameter (see [`Self::with_chi_prompt`]).
pub struct CaptionEncoder {
    gemma: Gemma2,
    tok: TextTokenizer,
    chi_prompt: String,
    num_chi_tokens: i32,
}

impl CaptionEncoder {
    /// Build the PiD caption encoder (uses PiD's [`CHI_PROMPT`]) from a constructed [`Gemma2`] and the
    /// gemma `tokenizer.json` path.
    pub fn new(gemma: Gemma2, tokenizer_json: impl AsRef<Path>) -> Result<Self> {
        Self::with_chi_prompt(gemma, tokenizer_json, CHI_PROMPT)
    }

    /// Build the caption encoder with an explicit CHI-prompt prefix — the reuse seam for SANA, which
    /// shares PiD's entire encoder body but ships a CHI template that differs in quoting (and hence
    /// tokenization). The `chi_prompt` is the already-joined (`"\n".join(complex_human_instruction)`)
    /// instruction string; the user caption is appended directly after it.
    pub fn with_chi_prompt(
        gemma: Gemma2,
        tokenizer_json: impl AsRef<Path>,
        chi_prompt: impl Into<String>,
    ) -> Result<Self> {
        let chi_prompt = chi_prompt.into();
        let tok = TextTokenizer::from_file(
            tokenizer_json,
            TokenizerConfig {
                max_length: 4096,
                pad_token_id: PAD_ID,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )?;
        // num_chi_tokens counts the CHI-prompt WITH its special tokens (the reference's
        // `tokenizer.encode(chi_prompt_str)` adds the bos).
        let num_chi_tokens = tok.encode_ids(&chi_prompt, true)?.len() as i32;
        Ok(Self {
            gemma,
            tok,
            chi_prompt,
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
        // The caption is fed RAW (no lowercasing) because PiD was TRAINED on raw-case captions, so raw
        // is what the checkpoint expects (sc-9935). Evidence, not just parity: PiD's training-time
        // caption conditioner (`modules/conditioner.py`) passes captions through unmodified; its
        // reference inference (`_encode_text_raw`) appends `chi_prompt_str + cap` unmodified; the
        // authors' own demo manifest (`assets/clean_image_manifest.jsonl`) uses mixed-case prompts; and
        // there is no `.lower()` on captions anywhere in the PiD repo. This deliberately differs from
        // SANA, whose reference is the diffusers `SanaPipeline` (`_text_preprocessing` lowercases the
        // user prompt) — so `mlx-gen-sana` applies its own `preprocess()` before this shared encoder
        // (sc-9927). Lowercasing here would feed PiD OOD-cased captions. (The effect is also small — an
        // A/B moved a PiD SR decode 0.034% since the decode caption is weak, LQ-dominated — but the
        // reason to keep raw is correctness, not impact.)
        let mut ids = self
            .tok
            .encode_ids(&format!("{}{caption}", self.chi_prompt), true)?;
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

    /// Encode one caption to `([1, 300, 2304]` embeddings, `[1, 300]` padding mask`)`. The mask marks
    /// real (`1.0`) vs padding (`0.0`) tokens among the `select_index`-gathered positions. Consumers
    /// whose trunk applies a cross-attention padding mask (diffusers SANA passes `prompt_attention_mask`
    /// into the transformer) use it; PiD's inference net ignores the mask ([`Self::encode`]). Because a
    /// short caption leaves the 300 slots dominated by PAD embeddings, dropping this mask lets the
    /// padding swamp the real conditioning — so a masked consumer MUST use this variant.
    pub fn encode_with_mask(&self, caption: &str) -> Result<(Array, Array)> {
        let (ids, mask) = self.token_ids(caption)?;
        let max_len = ids.len() as i32;
        let ids_arr = Array::from_slice(&ids, &[1, max_len]);
        let mask_arr = Array::from_slice(&mask, &[1, max_len]);
        let hidden = self.gemma.forward(&ids_arr, Some(&mask_arr))?; // [1, max_len, 2304]

        let mut sel = Vec::with_capacity(MODEL_MAX_LENGTH as usize);
        sel.push(0);
        sel.extend((max_len - (MODEL_MAX_LENGTH - 1))..max_len);
        // Gather the padding mask at the SAME select_index so it stays aligned with the embeddings.
        let sel_mask: Vec<f32> = sel.iter().map(|&i| mask[i as usize] as f32).collect();
        let sel_arr = Array::from_slice(&sel, &[MODEL_MAX_LENGTH]);
        let sel_mask_arr = Array::from_slice(&sel_mask, &[1, MODEL_MAX_LENGTH]);
        Ok((hidden.take_axis(&sel_arr, 1)?, sel_mask_arr)) // ([1,300,2304], [1,300])
    }
}
