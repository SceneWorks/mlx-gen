//! Shared host-side text/LLM token sampler — temperature / top-k / top-p / repetition-penalty over a
//! `[vocab]` logit slice, with a seeded PRNG.
//!
//! Hoisted from `mlx-gen-ltx`'s Gemma prompt-enhancer (sc-2845) so the lens PromptReasoner
//! (sc-9561 / F-105) and any future LLM-decode path share ONE implementation rather than cloning it
//! (the 2026-07-01 review's T6 duplication theme). It is **pure host math** over a `&[f32]` logit
//! slice — no MLX / tensor dependency, and no numeric-parity requirement: generation is stochastic;
//! the [`SplitMix64`] PRNG only makes a run reproducible given a seed.
//!
//! A caller pulls its `[vocab]` logits to the host once per step
//! (`logits.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec()`) and calls [`sample_token`]. Greedy
//! decode stays each crate's concern (e.g. the lens reasoner keeps its on-device `argmax` as the
//! KV-cache parity oracle); this module supplies the *sampled* path.

/// Sampling parameters. `top_k <= 0` and `top_p >= 1.0` disable those filters. `repetition_penalty`
/// (`None` ⇒ off) divides the logit of a token seen in the last `repetition_context` positions
/// (multiplies when the logit is negative), matching the reference `make_logits_processors`.
#[derive(Clone, Copy, Debug)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repetition_penalty: Option<f32>,
    pub repetition_context: usize,
}

impl SampleParams {
    /// Pure temperature sampling: no top-k / top-p narrowing, no repetition penalty. The neutral
    /// preset for an LLM decode that only wants `temperature` (e.g. the lens PromptReasoner's vendor
    /// default `temperature = 0.7`). `temperature <= 0` ⇒ greedy (argmax) in [`sample_token`].
    pub fn temperature(temperature: f32) -> Self {
        Self {
            temperature,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: None,
            repetition_context: 0,
        }
    }

    /// The LTX censored `enhance_t2v` sampler: `make_sampler(temp, 1.0, top_k=-1)` +
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

    /// The LTX uncensored `enhance_with_model` sampler: `make_sampler(temp, 1.0, 0.0, 1, top_k=0)` —
    /// pure temperature sampling, no repetition penalty (identical to [`temperature`](Self::temperature)).
    pub fn uncensored(temperature: f32) -> Self {
        Self::temperature(temperature)
    }
}

/// Sample a token id from host `logits` `[vocab]`, applying the repetition penalty over the tail of
/// `history`, then temperature + optional top-k / top-p. Host-side (CPU) for a faithful repetition
/// penalty + nucleus filter; deterministic given `rng`. Greedy (argmax) when `temperature <= 0` or the
/// filtered candidate mass is not a positive finite number (all-filtered / NaN / inf).
pub fn sample_token(
    logits: &[f32],
    history: &[i32],
    p: &SampleParams,
    rng: &mut SplitMix64,
) -> i32 {
    let mut v: Vec<f32> = logits.to_vec();
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
        return argmax_f32(&v);
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
        return argmax_f32(&v);
    }
    let mut target = rng.next_f32() * total;
    for (i, prob) in &probs {
        target -= prob;
        if target <= 0.0 {
            return *i as i32;
        }
    }
    probs.last().map(|x| x.0).unwrap_or(0) as i32
}

/// Argmax over a logit vector (greedy fallback). Ties break to the lowest index.
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
/// stochastic and not parity-gated; this just makes a run reproducible given a seed.)
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in `[0, 1)` (24-bit mantissa).
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn presets_match_reference() {
        let t = SampleParams::temperature(0.7);
        assert_eq!(t.top_k, 0);
        assert_eq!(t.top_p, 1.0);
        assert_eq!(t.repetition_penalty, None);

        let c = SampleParams::censored(0.7);
        assert_eq!(c.repetition_penalty, Some(1.3));
        assert_eq!(c.repetition_context, 20);

        let u = SampleParams::uncensored(0.7);
        assert_eq!(u.repetition_penalty, None);
        assert_eq!(u.top_k, 0);
    }

    #[test]
    fn temperature_zero_is_greedy_argmax() {
        // Deterministic argmax regardless of the rng draw.
        let logits = [0.1, 2.5, -1.0, 2.4, 0.0];
        let mut rng = SplitMix64::new(1);
        let params = SampleParams::temperature(0.0);
        for _ in 0..8 {
            assert_eq!(sample_token(&logits, &[], &params, &mut rng), 1);
        }
    }

    #[test]
    fn sampling_is_seed_reproducible() {
        let logits = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let params = SampleParams::temperature(1.0);
        let seq = |seed: u64| {
            let mut rng = SplitMix64::new(seed);
            (0..64)
                .map(|_| sample_token(&logits, &[], &params, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(seq(7), seq(7), "same seed must reproduce the same draws");
        // A uniform categorical over 6 tokens with two different seeds should (overwhelmingly) differ.
        assert_ne!(seq(7), seq(8));
    }

    #[test]
    fn top_k_one_forces_the_argmax_token() {
        // top_k = 1 collapses the candidate set to the single largest logit → deterministic.
        let logits = [0.0, 0.5, 3.0, 0.5, 1.0];
        let mut params = SampleParams::temperature(1.0);
        params.top_k = 1;
        let mut rng = SplitMix64::new(3);
        for _ in 0..8 {
            assert_eq!(sample_token(&logits, &[], &params, &mut rng), 2);
        }
    }

    #[test]
    fn repetition_penalty_suppresses_recent_tokens() {
        // Token 0 has the top logit; penalizing it (in history) below token 1 flips the argmax at temp 0.
        let logits = [2.0, 1.5, 0.0];
        let mut params = SampleParams::temperature(0.0); // greedy so the penalty effect is exact
        params.repetition_penalty = Some(2.0);
        params.repetition_context = 4;
        let mut rng = SplitMix64::new(0);
        // 2.0 / 2.0 = 1.0 < 1.5 → token 1 wins once token 0 is in the recent history.
        assert_eq!(sample_token(&logits, &[0], &params, &mut rng), 1);
        // Without the history the penalty does not apply → token 0 wins.
        assert_eq!(sample_token(&logits, &[], &params, &mut rng), 0);
    }
}
