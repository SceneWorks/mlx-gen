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

use mlx_gen::Result;

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
/// or a single `enhance_prompt=true` request becomes an effectively unbounded job (only cooperative
/// `cancel` breaks it). 4× the 512 reference default leaves room for legitimately long rewrites while
/// bounding the worst case to ~2048 forwards instead of billions.
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

/// Sampling parameters. `top_k <= 0` and `top_p >= 1.0` disable those filters (the reference default
/// for both variants). `repetition_penalty` / `repetition_context` are the censored path's
/// `make_logits_processors(None, 1.3, 20)`; the uncensored path leaves `repetition_penalty` `None`.
#[derive(Clone, Copy, Debug)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repetition_penalty: Option<f32>,
    pub repetition_context: usize,
}

impl SampleParams {
    /// The censored `enhance_t2v` sampler: `make_sampler(temp, 1.0, top_k=-1)` +
    /// `make_logits_processors(None, repetition_penalty=1.3, repetition_context_size=20)`.
    pub fn censored(temperature: f32) -> Self {
        Self {
            temperature,
            top_k: -1,
            top_p: 1.0,
            repetition_penalty: Some(1.3),
            repetition_context: 20,
        }
    }

    /// The uncensored `enhance_with_model` sampler: `make_sampler(temp, 1.0, 0.0, 1, top_k=0)` — pure
    /// temperature sampling, no repetition penalty.
    pub fn uncensored(temperature: f32) -> Self {
        Self {
            temperature,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: None,
            repetition_context: 0,
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
pub fn enhance(
    gemma: &GemmaModel,
    tokenizer: &LtxTokenizer,
    system_prompt: &str,
    user_prompt: &str,
    cfg: &EnhanceConfig,
    sampler: &SampleParams,
) -> Result<String> {
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
        let next = sample_token(&logits, &history, sampler, &mut rng)?;
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

/// Sample the next token id from `(1, vocab)` logits, applying the repetition penalty over the tail of
/// `history`, then temperature + optional top-k / top-p. Host-side (CPU) for a faithful repetition
/// penalty + nucleus filter; deterministic given `rng` (no numeric-parity requirement).
fn sample_token(
    logits: &Array,
    history: &[i32],
    p: &SampleParams,
    rng: &mut SplitMix64,
) -> Result<i32> {
    let lf = logits.as_dtype(Dtype::Float32)?;
    let mut v: Vec<f32> = lf.as_slice::<f32>().to_vec();
    let vocab = v.len();

    // Repetition penalty over the last `repetition_context` tokens (incl. the prompt tail).
    if let Some(pen) = p.repetition_penalty {
        if pen > 0.0 && p.repetition_context > 0 {
            let start = history.len().saturating_sub(p.repetition_context);
            for &t in &history[start..] {
                let idx = t as usize;
                if idx < vocab {
                    v[idx] = if v[idx] < 0.0 {
                        v[idx] * pen
                    } else {
                        v[idx] / pen
                    };
                }
            }
        }
    }

    // Greedy when temperature collapses to 0 (reference `make_sampler` argmaxes at temp == 0).
    if p.temperature <= 0.0 {
        return Ok(argmax_f32(&v));
    }

    // Candidate set: all tokens, optionally narrowed by top-k then top-p. Disabled at the reference
    // defaults (`top_k <= 0`, `top_p >= 1.0`), in which case every token is a candidate.
    let mut idx: Vec<usize> = (0..vocab).collect();
    if p.top_k > 0 && (p.top_k as usize) < vocab {
        let k = p.top_k as usize;
        idx.select_nth_unstable_by(k - 1, |&a, &b| v[b].total_cmp(&v[a]));
        idx.truncate(k);
    }
    // Temperature-scaled softmax over the candidates (numerically stable).
    let max = idx.iter().map(|&i| v[i]).fold(f32::NEG_INFINITY, f32::max);
    let inv_t = 1.0 / p.temperature;
    let mut probs: Vec<(usize, f32)> = idx
        .iter()
        .map(|&i| (i, ((v[i] - max) * inv_t).exp()))
        .collect();
    // Nucleus (top-p): sort by prob desc, keep the smallest prefix whose cumulative mass ≥ top_p.
    if p.top_p < 1.0 {
        probs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let total: f32 = probs.iter().map(|x| x.1).sum();
        let threshold = p.top_p * total;
        let mut cum = 0.0;
        let mut keep = probs.len();
        for (n, x) in probs.iter().enumerate() {
            cum += x.1;
            if cum >= threshold {
                keep = n + 1;
                break;
            }
        }
        probs.truncate(keep.max(1));
    }

    // Sample from the (unnormalized) categorical via inverse-CDF. Fall back to greedy if the mass is
    // not a positive finite number (all-filtered / NaN / inf).
    let total: f32 = probs.iter().map(|x| x.1).sum();
    if total <= 0.0 || !total.is_finite() {
        return Ok(argmax_f32(&v));
    }
    let mut target = rng.next_f32() * total;
    for (i, prob) in &probs {
        target -= prob;
        if target <= 0.0 {
            return Ok(*i as i32);
        }
    }
    Ok(probs.last().map(|x| x.0).unwrap_or(0) as i32)
}

/// Argmax over a logit vector (greedy fallback).
fn argmax_f32(v: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as i32
}

/// SplitMix64 — a tiny deterministic PRNG for host-side categorical sampling. (Generation is
/// stochastic and not parity-gated; this just makes the rewrite reproducible given a seed.)
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in `[0, 1)` (24-bit mantissa).
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
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

    #[test]
    fn sampler_param_presets_match_reference() {
        let c = SampleParams::censored(0.7);
        assert_eq!(c.repetition_penalty, Some(1.3));
        assert_eq!(c.repetition_context, 20);
        let u = SampleParams::uncensored(0.7);
        assert_eq!(u.repetition_penalty, None);
        assert_eq!(u.top_k, 0);
    }

    #[test]
    fn splitmix64_is_deterministic_and_in_range() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            let x = a.next_f32();
            assert_eq!(x, b.next_f32());
            assert!((0.0..1.0).contains(&x));
        }
    }
}
