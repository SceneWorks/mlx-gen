//! Boogu instruction tokenization (sc-6390) — the Qwen3-VL chat template + tokenizer that turns a
//! text prompt into the `input_ids` the condition encoder consumes.
//!
//! The reference builds messages `[system, user]` and calls `processor.apply_chat_template(...,
//! tokenize=True)` with **`add_generation_prompt=False`** (no trailing `assistant` turn — verified
//! by decoding the captured golden `tok_input_ids`). For text-to-image the system prompt is
//! [`SYSTEM_PROMPT_T2I`]; the classifier-free-guidance negative is the **empty** instruction, which
//! the reference routes to [`SYSTEM_PROMPT_DROP`] with empty user text. We render the exact ChatML
//! string ourselves and encode with `add_special_tokens=false` (the `<|im_start|>` / `<|im_end|>`
//! markers are literal special tokens already in the string), mirroring the reference
//! `tokenizer(text, add_special_tokens=False)` path.

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::Result;
use mlx_rs::Array;
use std::path::Path;

/// Text-to-image system prompt (reference `SYSTEM_PROMPT_4_T2I`).
pub const SYSTEM_PROMPT_T2I: &str = "You are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows.";

/// Empty-instruction (CFG negative) system prompt (reference `SYSTEM_PROMPT_DROP` =
/// `SYSTEM_PROMPT_4_TI2I_UNIFIED`).
pub const SYSTEM_PROMPT_DROP: &str = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.";

/// Render the ChatML string for a `(system, user)` turn pair with no generation prompt:
/// `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n`.
fn render_chat(system: &str, user: &str) -> String {
    format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n")
}

/// The Boogu condition tokenizer: the snapshot's `mllm/tokenizer.json` wrapped so we can render the
/// Boogu chat templates and encode them. Chat templating is done here (not via the core
/// [`ChatTemplate`]) because Boogu needs a per-call choice of system prompt.
pub struct BooguTokenizer {
    inner: TextTokenizer,
}

impl BooguTokenizer {
    /// Load from a snapshot's `mllm/tokenizer.json`.
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let inner = TextTokenizer::from_file(
            root.as_ref().join("mllm").join("tokenizer.json"),
            TokenizerConfig {
                // We render the chat string ourselves and call `encode_ids` directly, so the config
                // template/padding are unused; keep them inert.
                max_length: 1280,
                pad_token_id: 151643, // Qwen <|endoftext|>; unused (no padding on this path)
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )?;
        Ok(Self { inner })
    }

    /// Encode a rendered chat string to ids (`add_special_tokens=false`, matching the reference).
    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        Ok(self.inner.encode_ids(text, false)?)
    }

    /// Encode the **positive** text-to-image instruction → `(input_ids, attention_mask)` `[1, L]`.
    pub fn encode_t2i(&self, prompt: &str) -> Result<(Array, Array)> {
        ids_to_arrays(self.encode(&render_chat(SYSTEM_PROMPT_T2I, prompt))?)
    }

    /// Encode the CFG **negative** (empty instruction with the drop system prompt) → `[1, L]`.
    pub fn encode_negative(&self) -> Result<(Array, Array)> {
        ids_to_arrays(self.encode(&render_chat(SYSTEM_PROMPT_DROP, ""))?)
    }

    /// Raw id vector for the positive instruction (parity testing against the golden).
    pub fn t2i_ids(&self, prompt: &str) -> Result<Vec<i32>> {
        self.encode(&render_chat(SYSTEM_PROMPT_T2I, prompt))
    }
}

/// `Vec<i32>` ids → `(input_ids, attention_mask)` `[1, L]` int32 arrays (mask all-ones: no padding).
fn ids_to_arrays(ids: Vec<i32>) -> Result<(Array, Array)> {
    let len = ids.len() as i32;
    let mask = vec![1i32; ids.len()];
    Ok((
        Array::from_slice(&ids, &[1, len]),
        Array::from_slice(&mask, &[1, len]),
    ))
}
