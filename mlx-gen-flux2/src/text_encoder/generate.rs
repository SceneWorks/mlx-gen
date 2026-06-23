//! Autoregressive-decode support for the Mistral language tower (FLUX.2-dev caption upsampling,
//! sc-6030): a per-layer growable K/V cache and a temperature/top-p token sampler.
//!
//! The same packed-Q4 [`Qwen3TextEncoder`](super::Qwen3TextEncoder) that extracts the T2I
//! `prompt_embeds` (a single bidirectional forward) also drives the caption-upsampling
//! `generate()` loop (a KV-cached causal decode). This module holds the two pieces that are
//! independent of the encoder's private state; the decode/generate methods themselves live on
//! `Qwen3TextEncoder` in `encoder.rs` (they need its layers + lm_head). The sampler shape mirrors
//! the proven decoders that have since moved to the unified `mlx-llm` LLM engine.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use mlx_gen::Result;

/// Per-layer growable K/V cache for the autoregressive decode (immutable-array MLX: each step
/// `concat`s the new K/V onto the prior, along the sequence axis of `[b, n_kv_heads, s, head_dim]`).
/// One slot per decoder layer; built fresh per generation.
pub struct Qwen3KvCache {
    layers: Vec<Option<(Array, Array)>>,
}

impl Qwen3KvCache {
    pub(crate) fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
        }
    }

    /// Append layer `i`'s freshly projected `(k, v)` (`[b, n_kv_heads, s, head_dim]`) and return the
    /// full cached `(k, v)` to attend over.
    pub(crate) fn append(&mut self, i: usize, k: Array, v: Array) -> Result<(Array, Array)> {
        let merged = match self.layers[i].take() {
            Some((pk, pv)) => (
                concatenate_axis(&[&pk, &k], 2)?,
                concatenate_axis(&[&pv, &v], 2)?,
            ),
            None => (k, v),
        };
        self.layers[i] = Some((merged.0.clone(), merged.1.clone()));
        Ok(merged)
    }
}

/// Sampling knobs for the caption-upsampling decode (the reference `generate(do_sample=True,
/// temperature, max_new_tokens)`). `temperature <= 0` is greedy argmax; `top_p < 1` nucleus-filters.
#[derive(Clone, Copy, Debug)]
pub struct UpsampleSampling {
    pub temperature: f32,
    pub top_p: f32,
    pub max_new_tokens: usize,
    pub seed: u64,
}

/// Sample the next token from logits `[1, vocab]` (or `[vocab]`). Matches the
/// `mlx_gen::caption::joycaption` sampler shape: greedy at `temperature <= 0`, else
/// softmax-with-temperature → optional top-p nucleus → categorical draw. No repetition penalty
/// (the FLUX.2 reference uses plain temperature sampling).
pub fn sample_token(
    logits: &Array,
    sampling: &UpsampleSampling,
    rng: &mut SplitMix64,
) -> Result<i32> {
    let lf = logits.as_dtype(Dtype::Float32)?;
    let v: Vec<f32> = lf.as_slice::<f32>().to_vec();
    let vocab = v.len();

    if sampling.temperature <= 0.0 {
        return Ok(argmax_f32(&v));
    }

    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let inv_t = 1.0 / sampling.temperature;
    let mut probs: Vec<(usize, f32)> = (0..vocab)
        .map(|i| (i, ((v[i] - max) * inv_t).exp()))
        .collect();

    if sampling.top_p < 1.0 {
        probs = nucleus_select(&probs, sampling.top_p);
    }

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

/// Top-p (nucleus) selection via a partial max-heap (popped only until the cumulative-weight
/// threshold), identical kept set + descending order to a full sort for distinct weights. At least
/// one token is always kept. Mirrors the joycaption helper (F-039).
fn nucleus_select(probs: &[(usize, f32)], top_p: f32) -> Vec<(usize, f32)> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    struct ByWeight(usize, f32);
    impl PartialEq for ByWeight {
        fn eq(&self, o: &Self) -> bool {
            self.1.total_cmp(&o.1) == Ordering::Equal
        }
    }
    impl Eq for ByWeight {}
    impl PartialOrd for ByWeight {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for ByWeight {
        fn cmp(&self, o: &Self) -> Ordering {
            self.1.total_cmp(&o.1)
        }
    }

    let total: f32 = probs.iter().map(|x| x.1).sum();
    let threshold = top_p.max(0.0) * total;
    let mut heap: BinaryHeap<ByWeight> = probs.iter().map(|&(i, p)| ByWeight(i, p)).collect();
    let mut kept: Vec<(usize, f32)> = Vec::new();
    let mut cum = 0.0f32;
    while let Some(ByWeight(i, p)) = heap.pop() {
        kept.push((i, p));
        cum += p;
        if cum >= threshold {
            break;
        }
    }
    kept
}

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

/// Deterministic SplitMix64 PRNG for the categorical draw (seedable + reproducible) — the same
/// generator the joycaption / prompt-refine samplers use.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax_at_temperature_zero() {
        let logits = Array::from_slice(&[0.1f32, 4.0, 2.0], &[1, 3]);
        let mut rng = SplitMix64::new(0);
        let s = UpsampleSampling {
            temperature: 0.0,
            top_p: 1.0,
            max_new_tokens: 1,
            seed: 0,
        };
        assert_eq!(sample_token(&logits, &s, &mut rng).unwrap(), 1);
    }

    #[test]
    fn top_p_keeps_at_least_one_token() {
        let logits = Array::from_slice(&[5.0f32, 4.0, 1.0], &[1, 3]);
        let mut rng = SplitMix64::new(0);
        let s = UpsampleSampling {
            temperature: 0.7,
            top_p: 0.0,
            max_new_tokens: 1,
            seed: 0,
        };
        assert_eq!(sample_token(&logits, &s, &mut rng).unwrap(), 0);
    }

    #[test]
    fn sampling_is_seed_reproducible_and_varies() {
        let logits = Array::from_slice(&[0.0f32; 64], &[1, 64]);
        let draw = |seed: u64| -> Vec<i32> {
            let mut rng = SplitMix64::new(seed);
            let s = UpsampleSampling {
                temperature: 1.0,
                top_p: 1.0,
                max_new_tokens: 32,
                seed,
            };
            (0..32)
                .map(|_| sample_token(&logits, &s, &mut rng).unwrap())
                .collect()
        };
        assert_eq!(draw(7), draw(7), "same seed reproduces the same samples");
        assert_ne!(draw(7), draw(99), "different seeds differ");
    }

    #[test]
    fn cache_append_grows_along_sequence_axis() {
        let mut cache = Qwen3KvCache::new(1);
        let k0 = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 2]);
        let v0 = Array::from_slice(&[3.0f32, 4.0], &[1, 1, 1, 2]);
        let (k, _) = cache.append(0, k0, v0).unwrap();
        assert_eq!(k.shape(), &[1, 1, 1, 2]);
        let k1 = Array::from_slice(&[5.0f32, 6.0], &[1, 1, 1, 2]);
        let v1 = Array::from_slice(&[7.0f32, 8.0], &[1, 1, 1, 2]);
        let (k, v) = cache.append(0, k1, v1).unwrap();
        assert_eq!(k.shape(), &[1, 1, 2, 2]);
        assert_eq!(v.shape(), &[1, 1, 2, 2]);
        assert_eq!(k.as_slice::<f32>(), &[1.0, 2.0, 5.0, 6.0]);
    }
}
