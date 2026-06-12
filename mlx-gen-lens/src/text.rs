//! Lens text input — the gpt-oss **o200k_harmony** tokenizer + the Lens chat-template (sc-3167).
//!
//! Reproduces `LensPipeline._build_chat_inputs`: wrap the prompt in the harmony chat format (a fixed
//! `system` preamble + the `developer` instruction + the `user` prompt + an `assistant`/`analysis`
//! thinking turn), then tokenize the rendered text via the model's `tokenizer.json` (loaded through
//! the shared [`mlx_gen::tokenizer`] seam — the same HF `tokenizers` core `transformers` wraps, so
//! the ids are byte-identical).
//!
//! **txt_offset = 97.** The encoder runs the *whole* sequence (the 97-token preamble is real causal
//! context), but the DiT conditioning is only `input_ids[97:]` — the user caption + the trailing
//! assistant scaffold. The preamble's `Current date:` line is dynamic, so [`LensTokenizer::encode`]
//! takes the date as a parameter (the worker passes today's date; tests pass the golden's). The
//! preamble is always exactly [`TXT_OFFSET`] tokens, which is why the offset is a fixed constant.

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig, TokenizerOutput};
use mlx_gen::Result;

/// Number of fixed harmony-preamble tokens the DiT conditioning skips (`DEFAULT_TXT_OFFSET`).
pub const TXT_OFFSET: usize = 97;

/// The Lens `developer` instruction (`_CHAT_SYSTEM`).
const SYSTEM_INSTRUCTION: &str =
    "Describe the image by detailing the color, shape, size, texture, \
     quantity, text, spatial relationships of the objects and background.";
/// The Lens `assistant`/`analysis` thinking turn (`_CHAT_ASSISTANT_THINKING`).
const ASSISTANT_THINKING: &str = "Need to generate one image according to the description.";

/// Render the harmony-formatted Lens prompt (== `_build_chat_inputs` after the `<|return|>` split,
/// pre-tokenization). `date` fills the preamble's `Current date:` line (ISO `YYYY-MM-DD`).
fn render(prompt: &str, date: &str) -> String {
    format!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\
         Knowledge cutoff: 2024-06\n\
         Current date: {date}\n\n\
         Reasoning: medium\n\n\
         # Valid channels: analysis, commentary, final. Channel must be included for every message.\
         <|end|><|start|>developer<|message|># Instructions\n\n\
         {SYSTEM_INSTRUCTION}\n\n\
         <|end|><|start|>user<|message|>{prompt}\
         <|end|><|start|>assistant<|channel|>analysis<|message|>{ASSISTANT_THINKING}\
         <|end|><|start|>assistant<|channel|>final<|message|>"
    )
}

/// The Lens text tokenizer: the model's `tokenizer.json` + the harmony chat-template wrapping.
pub struct LensTokenizer {
    inner: TextTokenizer,
}

impl LensTokenizer {
    /// Load from the snapshot's `tokenizer/tokenizer.json`.
    pub fn from_file(tokenizer_json: impl AsRef<Path>) -> Result<Self> {
        // The harmony wrapping is done here in [`render`]; the core tokenizer encodes verbatim.
        let cfg = TokenizerConfig {
            max_length: 512,
            pad_token_id: 199_999, // gpt-oss `pad_token_id`
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        };
        Ok(Self {
            inner: TextTokenizer::from_file(tokenizer_json, cfg)?,
        })
    }

    /// Tokenize `prompt` into `(1, L)` `input_ids` + attention mask (mask all-`1`; a single prompt is
    /// unpadded). `date` is the ISO `YYYY-MM-DD` for the harmony preamble. The DiT consumes
    /// `ids[TXT_OFFSET..]`.
    pub fn encode(&self, prompt: &str, date: &str) -> Result<TokenizerOutput> {
        let ids = self.inner.encode_ids(&render(prompt, date), true)?;
        let mask = vec![1i32; ids.len()];
        Ok(TokenizerOutput { ids, mask })
    }
}
