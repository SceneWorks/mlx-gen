//! Sampling knobs for the Mistral language tower's caption-upsampling decode (FLUX.2-dev, sc-6030).
//!
//! The same packed-Q4 [`Qwen3TextEncoder`](super::Qwen3TextEncoder) that extracts the T2I
//! `prompt_embeds` (a single bidirectional forward) also drives the caption-upsampling
//! `generate()` loop (a KV-cached causal decode). The decode loop itself lives on
//! `Qwen3TextEncoder` in `encoder.rs`; the on-device pieces it needs — the per-layer growable K/V
//! cache and the temperature/top-p token sampler — are the shared `mlx-llm` decode primitives
//! ([`mlx_llm::primitives::ContiguousKvCache`] + [`mlx_llm::primitives::sample`]), not a hand-rolled
//! copy (sc-7160). This module is left with only the caller-facing sampling knobs.

/// Sampling knobs for the caption-upsampling decode (the reference `generate(do_sample=True,
/// temperature, max_new_tokens)`). `temperature <= 0` is greedy argmax; `top_p < 1` nucleus-filters.
/// Mapped onto [`mlx_llm::primitives::SamplingParams`] (with `top_k = 0`, `repetition_penalty = 1.0`
/// — the FLUX.2 reference uses plain temperature sampling) inside the decode loop; `max_new_tokens`
/// bounds the loop and `seed` seeds the shared [`mlx_llm::primitives::SplitMix64`] draw.
#[derive(Clone, Copy, Debug)]
pub struct UpsampleSampling {
    pub temperature: f32,
    pub top_p: f32,
    pub max_new_tokens: usize,
    pub seed: u64,
}
