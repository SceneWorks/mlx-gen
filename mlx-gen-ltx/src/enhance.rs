//! S7 — LTX-2.3 prompt enhancement (sc-2845): rewrite the user prompt with Gemma-3 as an
//! autoregressive LLM before encoding. Optional, **default off**, and **not** numeric-parity (text
//! generation is stochastic and mlx-rs RNG isn't portable to mlx-python — a behavioral/smoke gate).
//!
//! Port of `mlx_video/models/ltx/text_encoder.py::LTX2TextEncoder.enhance_t2v / enhance_i2v` and
//! `models/ltx/enhance_prompt.py::enhance_with_model`, with the wiring from `generate_av.py`:
//! - Build the Gemma chat template (system turn + `"user prompt: {prompt}"` user turn + model turn).
//! - Tokenize with `add_special_tokens=false` (the template supplies the `<start_of_turn>` markers).
//! - Autoregressively sample (temperature 0.7; the censored path adds repetition-penalty 1.3 over a
//!   20-token window; top-k / top-p are disabled at the reference defaults but supported here) up to
//!   `max_tokens`, stopping on an end-of-turn / eos token.
//! - Detokenize the generated tokens and run [`clean_response`].
//!
//! The censored variant reuses the **already-loaded** text-encoder Gemma backbone
//! ([`GemmaModel::decode_logits`]); the uncensored variant loads a separate 4-bit Gemma — both go
//! through the same loop here ([`enhance`]), differing only in model + [`SampleParams`].
//!
//! **Stop tokens.** The reference hardcodes `token == 1 or token == 107`, but in the Gemma-3
//! tokenizer **107 is `\n`** (a newline) and `<end_of_turn>` is **106**; `generation_config.json`
//! gives the authoritative `eos_token_id = [1, 106]`. We stop on **{1, 106}** ([`STOP_TOKENS`]) —
//! the reference's `107` would truncate at the first newline (a latent bug in the reference).

use mlx_rs::{Array, Dtype};

use mlx_gen::{CancelFlag, Error, Result};
// The token sampler (temperature / top-k / top-p / repetition penalty) + seeded PRNG live in the core
// crate's shared `text_sample` module (sc-9561 / F-105) so the lens PromptReasoner reuses them rather
// than cloning. `SampleParams` stays part of this crate's public API via the re-export.
pub use mlx_gen::text_sample::SampleParams;
use mlx_gen::text_sample::{sample_token, SplitMix64};

use crate::gemma::GemmaModel;
use crate::tokenizer::LtxTokenizer;

/// Vendored default system prompts (the mlx_video wheel ships `enhance_prompt.py` / `text_encoder.py`
/// but **omits** the `prompts/` dir — so its enhancer silently FileNotFound→falls back; we vendor the
/// canonical `ltx_core` copies, identical across the SceneWorks venv and the upstream git checkout).
pub const T2V_SYSTEM_PROMPT: &str = include_str!("prompts/gemma_t2v_system_prompt.txt");
pub const I2V_SYSTEM_PROMPT: &str = include_str!("prompts/gemma_i2v_system_prompt.txt");

/// Reference enhancement defaults (`generate_av.py` CLI).
pub const DEFAULT_MAX_TOKENS: usize = 512;
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
/// Reference enhancement default seed (`enhance_t2v(..., seed=42)`).
pub const DEFAULT_SEED: u64 = 42;

/// Hard ceiling on enhance decode length (F-012 twin of the flux2 cap). Each decode step is a full
/// Gemma forward over a growing KV cache, so a request-supplied `enhance_max_tokens` must be capped
/// or a single `enhance_prompt=true` request becomes an effectively unbounded job. 4× the 512
/// reference default leaves room for legitimately long rewrites while bounding the worst case to
/// ~2048 forwards instead of billions. Cooperative cancellation ([`enhance`]'s `cancel`) also
/// interrupts the loop per decoded token (F-018).
pub const MAX_TOKENS_CAP: usize = 2048;

/// Resolve the decode budget from the request's `enhance_max_tokens`: the reference default
/// ([`DEFAULT_MAX_TOKENS`]) when unset, otherwise the requested value clamped to [`MAX_TOKENS_CAP`]
/// (F-012). A request is never *rejected* for asking too much — the advisory knob is silently capped
/// — so callers stay infallible. Inert on the happy path (the reference default is well under the cap).
pub fn clamp_max_tokens(requested: Option<u32>) -> usize {
    requested
        .map(|m| (m as usize).min(MAX_TOKENS_CAP))
        .unwrap_or(DEFAULT_MAX_TOKENS)
}

/// Stop tokens: `<eos>` (1) and `<end_of_turn>` (106) — see the module note on the reference's `107`.
pub const STOP_TOKENS: [i32; 2] = [1, 106];

/// Per-call generation budget.
#[derive(Clone, Copy, Debug)]
pub struct EnhanceConfig {
    pub max_tokens: usize,
    pub seed: u64,
}

impl Default for EnhanceConfig {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_TOKENS,
            seed: DEFAULT_SEED,
        }
    }
}

/// Build the Gemma-3 chat-templated string: a system turn, a `"user prompt: {prompt}"` user turn, and
/// the model generation prompt. Mirrors `_apply_chat_template([system, user])` and
/// `enhance_prompt._apply_chat_template(system, "user prompt: " + prompt)` (both produce this exact
/// string — system and user are both emitted as `user` turns in the reference).
fn chat_template(system_prompt: &str, user_prompt: &str) -> String {
    format!(
        "<start_of_turn>user\n{system_prompt}<end_of_turn>\n\
         <start_of_turn>user\nuser prompt: {user_prompt}<end_of_turn>\n\
         <start_of_turn>model\n"
    )
}

/// Reference `_clean_response`: strip surrounding whitespace, then drop a leading run of characters
/// that are neither word (`\w`: alphanumeric or `_`) nor whitespace (`\s`) — i.e. leading punctuation
/// / symbols (`re.sub(r"^[^\w\s]+", "", response)`).
pub fn clean_response(response: &str) -> String {
    let trimmed = response.trim();
    let cleaned = trimmed
        .trim_start_matches(|c: char| !(c.is_alphanumeric() || c == '_' || c.is_whitespace()));
    cleaned.to_string()
}

/// Run the autoregressive enhancement loop over `gemma` + `tokenizer`, returning the cleaned rewrite.
/// May return an empty string (e.g. the model immediately emits a stop token) — the caller decides
/// whether to fall back to the original prompt (the reference treats empty output as a failure).
/// `cancel` is the request's cooperative cancellation handle (F-018): checked before each of the up
/// to [`MAX_TOKENS_CAP`] Gemma decode steps and after the prefill, returning [`Error::Canceled`] so a
/// cancel during a multi-minute enhancement is honored (matching the denoise loops' per-step
/// contract). Each `decode_logits` step already forces a host sync, so the check observes the trip.
#[allow(clippy::too_many_arguments)]
pub fn enhance(
    gemma: &GemmaModel,
    tokenizer: &LtxTokenizer,
    system_prompt: &str,
    user_prompt: &str,
    cfg: &EnhanceConfig,
    sampler: &SampleParams,
    cancel: Option<&CancelFlag>,
) -> Result<String> {
    // Honor a cancel tripped before enhancement even begins (before the ~12B prefill forward, F-018).
    if cancel.is_some_and(CancelFlag::is_cancelled) {
        return Err(Error::Canceled);
    }
    let formatted = chat_template(system_prompt, user_prompt);
    let prompt_ids = tokenizer.encode_chat(&formatted)?;
    if prompt_ids.is_empty() {
        return Ok(String::new());
    }

    // `history` carries the prompt + generated tokens; the repetition penalty looks at its tail (the
    // reference applies the penalty over `tokens[-context_size:]` of the running sequence).
    let mut history = prompt_ids.clone();
    let mut cache = gemma.new_cache();
    let mut rng = SplitMix64::new(cfg.seed);

    // Prefill on the full prompt → logits for the first generated token.
    let prompt_len = prompt_ids.len() as i32;
    let ids = Array::from_slice(&prompt_ids, &[1, prompt_len]);
    let mut logits = gemma.decode_logits(&ids, &mut cache, 0)?;

    let mut generated: Vec<i32> = Vec::new();
    for step in 0..cfg.max_tokens {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        // Pull the `[vocab]` logits to the host once, then draw from the shared host-side sampler.
        let logits_host = logits.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
        let next = sample_token(&logits_host, &history, sampler, &mut rng);
        generated.push(next);
        history.push(next);
        if STOP_TOKENS.contains(&next) {
            break;
        }
        // Feed the token back at its absolute position (the generated token at index `step` sits at
        // `prompt_len + step`, just past the prefilled prompt).
        let nxt = Array::from_slice(&[next], &[1, 1]);
        logits = gemma.decode_logits(&nxt, &mut cache, prompt_len + step as i32)?;
    }

    let text = tokenizer.decode(&generated)?;
    Ok(clean_response(&text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_response_strips_leading_punctuation_and_whitespace() {
        assert_eq!(clean_response("  \n**Style: a fox"), "Style: a fox");
        assert_eq!(clean_response("\"quoted start"), "quoted start");
        // Faithful to the reference: `strip()` then `re.sub(r"^[^\w\s]+", "", …)` with NO final strip,
        // so the regex stops at the first whitespace and a space after the punctuation run survives.
        assert_eq!(clean_response("...:: hello"), " hello");
        // Already clean → unchanged (modulo surrounding whitespace).
        assert_eq!(clean_response("  a red fox  "), "a red fox");
        // Leading digits / underscores are word chars → preserved.
        assert_eq!(clean_response("3 cats"), "3 cats");
        // Empty / all-punctuation collapses to empty.
        assert_eq!(clean_response("   "), "");
        assert_eq!(clean_response("!!!"), "");
    }

    #[test]
    fn clamp_max_tokens_caps_pathological_request_only() {
        // Unset → reference default, untouched.
        assert_eq!(clamp_max_tokens(None), DEFAULT_MAX_TOKENS);
        // Below the cap → honored verbatim (happy path stays inert).
        assert_eq!(clamp_max_tokens(Some(1)), 1);
        assert_eq!(clamp_max_tokens(Some(256)), 256);
        // Exactly at the cap → honored.
        assert_eq!(
            clamp_max_tokens(Some(MAX_TOKENS_CAP as u32)),
            MAX_TOKENS_CAP
        );
        // Above the cap (incl. u32::MAX, the unbounded-job case) → clamped to the cap, not rejected.
        assert_eq!(
            clamp_max_tokens(Some(MAX_TOKENS_CAP as u32 + 1)),
            MAX_TOKENS_CAP
        );
        assert_eq!(clamp_max_tokens(Some(u32::MAX)), MAX_TOKENS_CAP);
    }

    #[test]
    fn chat_template_matches_reference_format() {
        let t = chat_template("SYS", "a fox");
        assert_eq!(
            t,
            "<start_of_turn>user\nSYS<end_of_turn>\n\
             <start_of_turn>user\nuser prompt: a fox<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn vendored_prompts_are_present_and_nonempty() {
        assert!(T2V_SYSTEM_PROMPT.contains("Creative Assistant"));
        assert!(I2V_SYSTEM_PROMPT.contains("image-to-video"));
    }

    // `SampleParams` presets + `SplitMix64` determinism are covered in the shared
    // `mlx_gen::text_sample` tests (the sampler now lives there — sc-9561 / F-105).
}
