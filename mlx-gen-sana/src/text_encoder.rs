//! SANA **text conditioning** — a thin wrapper that REUSES PiD's native gemma-2-2b-it CHI caption
//! encoder (epic 8485, story sc-8488).
//!
//! SANA and PiD share the **exact same** text-conditioning lineage: a fixed Complex-Human-Instruction
//! (CHI) prompt is prepended to the user caption, the pair is tokenized (`add_special_tokens=True` →
//! leading `<bos>`) and right-padded/truncated to `num_chi_tokens + max_sequence_length − 2`, the
//! gemma-2-2b-it **decoder** is run in encoder/feature-extraction mode (its **last-hidden** states,
//! NOT generation logits — diffusers `_get_gemma_prompt_embeds` does `prompt_embeds[0]`, i.e.
//! `last_hidden_state`), and finally `select_index = [0] + range(-(max_seq_len − 1), 0)` gathers
//! exactly `max_sequence_length` tokens → `[1, 300, 2304]`.
//!
//! That is **byte-for-byte** [`mlx_gen_pid::CaptionEncoder`]'s algorithm (verified against the frozen
//! `mflux` reference and against diffusers `SanaPipeline.encode_prompt` /
//! `_get_gemma_prompt_embeds`). The two diverge in exactly **one** place: the CHI prompt text. PiD's
//! released `_CHI_PROMPT` wraps `Enhanced prompt` in escaped **double**-quotes; SANA's
//! `complex_human_instruction` list (diffusers `pipeline_sana.py`, NVlabs/Sana) wraps it in
//! **single**-quotes. Everything else in the joined string — wording, the two examples, the trailing
//! `"User Prompt: "`, the `\n` joins — is identical. Because the quote difference changes the
//! tokenization, we do NOT reuse PiD's CHI text: we pass SANA's exact text via
//! [`mlx_gen_pid::CaptionEncoder::with_chi_prompt`].
//!
//! Reuse seam: `mlx-gen-sana` depends on `mlx-gen-pid` and constructs a [`CaptionEncoder`] with
//! [`SANA_CHI_PROMPT`]. The Gemma-2 forward, padding-mask handling, and token-selection policy live
//! once, in `mlx-gen-pid`; nothing is copy-pasted here.
//!
//! ## Mask handling
//! `_get_gemma_prompt_embeds` returns `(prompt_embeds, prompt_attention_mask)` and `encode_prompt`
//! gathers the **same** `select_index` from both. But the SANA *transformer*
//! ([`crate::transformer::SanaTransformer::forward`]) consumes only the `[1, 300, 2304]` embedding —
//! its `attn2` cross-attention is plain full softmax over all 300 caption tokens with **no** mask
//! (same as PiD's inference net, which discards `emb_masks`). So the 300-token attention mask is
//! exposed here for completeness/parity ([`SanaTextEncoder::token_ids`]) but is not fed to the trunk.

use std::path::Path;

use mlx_rs::Array;

use mlx_gen::Result;
use mlx_gen_pid::CaptionEncoder;
pub use mlx_gen_pid::{Gemma2, Gemma2Config};

/// SANA's max caption sequence length (`max_sequence_length=300` in diffusers `SanaPipeline`).
pub const MAX_SEQUENCE_LENGTH: i32 = 300;

/// SANA's Complex-Human-Instruction (CHI) prompt: `"\n".join(complex_human_instruction)` from
/// diffusers `pipeline_sana.py` (matching NVlabs/Sana). Identical to PiD's CHI template **except**
/// `Enhanced prompt` is wrapped in single-quotes here (PiD uses double-quotes); the difference is
/// load-bearing because it changes the tokenization. The user caption is appended directly after the
/// trailing `"User Prompt: "`.
pub const SANA_CHI_PROMPT: &str = "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:\n- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.\n- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\nHere are examples of how to transform or refine prompts:\n- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\nPlease generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:\nUser Prompt: ";

/// SANA Gemma-2 CHI text encoder: prompt → `[1, 300, 2304]` caption embedding for the SANA trunk's
/// `attn2` cross-attention. A thin wrapper over the reused [`mlx_gen_pid::CaptionEncoder`], wired with
/// [`SANA_CHI_PROMPT`].
pub struct SanaTextEncoder {
    inner: CaptionEncoder,
}

impl SanaTextEncoder {
    /// Build from a constructed [`Gemma2`] decoder (loaded from the un-gated
    /// `SceneWorks/gemma-2-2b-it` mirror, epic 7840) and the gemma `tokenizer.json` path. Uses SANA's
    /// [`SANA_CHI_PROMPT`].
    pub fn new(gemma: Gemma2, tokenizer_json: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            inner: CaptionEncoder::with_chi_prompt(gemma, tokenizer_json, SANA_CHI_PROMPT)?,
        })
    }

    /// Load gemma-2-2b-it from a snapshot directory (`<dir>/gemma-2-2b-it.safetensors` +
    /// `<dir>/tokenizer.json`) and build the SANA text encoder.
    pub fn from_snapshot(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let w = mlx_gen::weights::Weights::from_file(dir.join("gemma-2-2b-it.safetensors"))?;
        let gemma = Gemma2::from_weights(&w, "model.", &Gemma2Config::gemma_2_2b())?;
        Self::new(gemma, dir.join("tokenizer.json"))
    }

    /// CHI-prompt token count (`num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))` in the
    /// reference — includes the leading `<bos>`).
    pub fn num_chi_tokens(&self) -> i32 {
        self.inner.num_chi_tokens()
    }

    /// The padded `(input_ids, attention_mask)` for a caption (length `num_chi_tokens + 300 − 2`,
    /// pre token-selection). Exposed so the tokenizer + CHI-prompt + length policy can be
    /// parity-checked against the reference without the gemma weights, and so the attention mask is
    /// available even though the trunk does not consume it.
    pub fn token_ids(&self, caption: &str) -> Result<(Vec<i32>, Vec<i32>)> {
        self.inner.token_ids(caption)
    }

    /// Encode one caption to the SANA caption embedding `[1, 300, 2304]` (gemma last-hidden,
    /// `select_index`-gathered). Byte/shape-compatible with
    /// [`crate::transformer::SanaTransformer::forward`]'s `caption` argument.
    pub fn encode(&self, caption: &str) -> Result<Array> {
        self.inner.encode(caption)
    }
}
