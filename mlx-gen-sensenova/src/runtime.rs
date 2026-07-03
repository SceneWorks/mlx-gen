//! The autoregressive text-generation runtime (sc-3187) — KV-cache incremental decode, token
//! sampling, and the `_generate_think` think/no-think rollout.
//!
//! SenseNova-U1 is a *generating* LLM, so beyond a forward pass it needs an AR runtime: a prefix is
//! prefilled into a [`KvCache`](crate::qwen3::KvCache), then tokens are decoded one at a time —
//! each new token forwarded through the cached backbone at the next temporal position to produce
//! the logits for the token after it. This module ports the reference's three pieces
//! (`modeling_neo_chat.py`):
//!
//! * [`Qwen3Backbone::decode_logits`] — one cached single-token forward → next-token logits (the
//!   inner step of the reference's `self.language_model(input_ids=next_token.unsqueeze(0), …)`).
//! * [`Qwen3Backbone::append_tokens`] — splice a run of known tokens into the cache without
//!   sampling (the reference `_append_text_tokens_to_cache`; e.g. the `\n\n<img>` that follows a
//!   think block).
//! * [`Qwen3Backbone::generate`] — greedy/sampled rollout to an EOS or token budget (the runtime
//!   under `chat`/`answer_question`, sc-3191).
//! * [`Qwen3Backbone::generate_think`] — the `_generate_think` think rollout: greedy-decode a
//!   `<think>…</think>` block, then append `\n\n<img>` to the cache, leaving it primed for image
//!   generation (sc-3187's deliverable for T2I think-mode + interleave).
//!
//! Positions: text tokens advance the temporal axis by one per token (`h = w = 0`), matching the
//! reference, which sets `model.current_index = t_idx` before each step and lets the forward
//! increment it. The understanding path ([`Path::Und`]) drives text decode.

use mlx_rs::{Array, Dtype};

use mlx_gen::{CancelFlag, Error, Result};

// Shared decode sampler (sc-7159): on-device greedy argmax + the unified temperature/top-k/top-p
// sampler + the deterministic SplitMix64. The bespoke think/no-think + dual-path rollout stays here.
use mlx_llm::primitives::sampler::{argmax_device, argmax_host, sample as mll_sample};
use mlx_llm::primitives::{SamplingParams, SplitMix64};

use crate::qwen3::{KvCache, Path, Qwen3Backbone};

/// Map an mlx-llm primitive error onto the gen-core contract error.
fn mll<E: std::fmt::Display>(e: E) -> Error {
    Error::Msg(e.to_string())
}

/// How the next token is chosen from a logits row.
#[derive(Clone, Copy, Debug)]
pub enum Sampler {
    /// Argmax — the reference `_generate_think` rollout and the deterministic chat path.
    Greedy,
    /// Temperature + nucleus (top-p) + top-k sampling. `top_p`/`top_k` of `1.0`/`0` disable that
    /// stage; `temperature` must be `> 0`.
    Sample {
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: u64,
    },
}

impl Sampler {
    /// Pick a token id from a `[vocab]` logits row, advancing `rng` for the stochastic variants.
    ///
    /// Delegates to the shared sampler (sc-7159): greedy is the shared lowest-index [`argmax_host`];
    /// the stochastic variant is the unified [`mll_sample`] (temperature + top-k + nucleus top-p, no
    /// repetition penalty) over the row lifted to an `Array`. The greedy path is bit-identical to the
    /// prior local argmax; the stochastic path is a valid resample from the same shaped distribution
    /// (the shared sampler draws over candidates in index/top-k order rather than the prior
    /// sorted-descending order — no golden pins a stochastic sequence, and every gated rollout, plus
    /// all image-token decoding, is greedy).
    fn pick(&self, logits: &[f32], rng: &mut SplitMix64) -> Result<i32> {
        match *self {
            Sampler::Greedy => Ok(argmax_host(logits)),
            Sampler::Sample {
                temperature,
                top_p,
                top_k,
                ..
            } => {
                let row = Array::from_slice(logits, &[1, logits.len() as i32]);
                let params = SamplingParams {
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty: 1.0,
                    repetition_context: 0,
                };
                mll_sample(&row, &[], &params, rng, None).map_err(mll)
            }
        }
    }

    fn seed(&self) -> u64 {
        match *self {
            Sampler::Greedy => 0,
            Sampler::Sample { seed, .. } => seed,
        }
    }
}

/// The result of a [`Qwen3Backbone::generate_think`] rollout.
pub struct ThinkRollout {
    /// The think-block token ids (everything the model emitted up to and including `</think>`, or
    /// up to EOS). Decode with the tokenizer for the human-readable reasoning text.
    pub think_token_ids: Vec<i32>,
    /// The temporal index after the rollout and the appended `\n\n<img>` — the `text_len` the first
    /// image block is placed after.
    pub t_idx: i32,
}

impl Qwen3Backbone {
    /// One cached single-token forward on the understanding path: embed `token`, run it at temporal
    /// position `pos_t` (`h = w = 0`), persist its K/V, and return the `[vocab]` next-token logits.
    pub fn decode_logits(&self, token: i32, pos_t: i32, cache: &mut KvCache) -> Result<Vec<f32>> {
        let ids = Array::from_slice(&[token], &[1, 1]);
        let embeds = self.embed(&ids)?;
        let hidden = self.forward_cached(&embeds, &[pos_t], &[0], &[0], Path::Und, cache, true)?;
        let logits = self.lm_head(&hidden)?; // [1, 1, vocab]
        let vocab = logits.shape()[2];
        // F-144: `as_slice::<f32>()` reinterprets the raw buffer — it is only correct if `logits` is
        // f32. Today that holds by accident (the RoPE path promotes to f32), but make it explicit so a
        // future bf16/f16 lm_head can't silently mis-read the bytes. A no-op when already f32.
        let logits = logits.reshape(&[vocab])?.as_dtype(Dtype::Float32)?;
        Ok(logits.as_slice::<f32>().to_vec())
    }

    /// Like [`decode_logits`](Self::decode_logits) but reduces to the greedy next token **on device**
    /// — only the single argmax index is copied to host, not the whole `[vocab]` f32 row (~600 KB).
    /// MLX `argmax` breaks ties to the lowest index, matching the host [`argmax`] (`torch.argmax`), so
    /// the greedy stream is bit-identical (F-140).
    pub fn decode_argmax(&self, token: i32, pos_t: i32, cache: &mut KvCache) -> Result<i32> {
        let ids = Array::from_slice(&[token], &[1, 1]);
        let embeds = self.embed(&ids)?;
        let hidden = self.forward_cached(&embeds, &[pos_t], &[0], &[0], Path::Und, cache, true)?;
        // The shared on-device argmax flattens the `[1, 1, vocab]` logits internally and breaks ties
        // to the lowest index — the same single-element host transfer + tie rule as the prior local
        // `argmax_device` (F-140).
        let logits = self.lm_head(&hidden)?;
        argmax_device(&logits).map_err(mll)
    }

    /// Splice a run of known tokens into the cache (no sampling), advancing the temporal axis from
    /// `t_idx`. Returns the new `t_idx`. Mirrors the reference `_append_text_tokens_to_cache`: the
    /// tokens take positions `t_idx+1 .. t_idx+len` (`h = w = 0`), so the within-run mask is causal
    /// and they attend to all cached context.
    pub fn append_tokens(&self, ids: &[i32], t_idx: i32, cache: &mut KvCache) -> Result<i32> {
        if ids.is_empty() {
            return Ok(t_idx);
        }
        let n = ids.len() as i32;
        let ids_arr = Array::from_slice(ids, &[1, n]);
        let embeds = self.embed(&ids_arr)?;
        let temporal: Vec<i32> = (t_idx + 1..=t_idx + n).collect();
        let zeros = vec![0i32; ids.len()];
        self.forward_cached(&embeds, &temporal, &zeros, &zeros, Path::Und, cache, true)?;
        Ok(t_idx + n)
    }

    /// Greedy/sampled AR text rollout. `first_logits` are the prefix's last-position logits (the
    /// distribution over the first generated token); `t_idx` is the prefix's max temporal index.
    /// Decoding stops at any id in `eos` (not emitted) or after `max_new_tokens`. Returns the
    /// generated token ids. This is the runtime under `chat`/`answer_question` (sc-3191).
    ///
    /// `cancel` is the cooperative cancellation handle (F-037): checked before each decoded token so a
    /// worker-consumed VQA / Document Studio rollout is cancellable (each token forces a host sync, so
    /// the check observes the trip). Returns [`Error::Canceled`] on trip.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        eos: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
        cancel: Option<&CancelFlag>,
    ) -> Result<Vec<i32>> {
        // Greedy decodes argmax on device (single-index host transfer per token); sampling needs the
        // full logits row on host. Split so the common greedy path avoids the ~600 KB copy (F-140).
        if let Sampler::Greedy = sampler {
            let mut next = argmax(first_logits);
            let mut out = Vec::new();
            let mut t = t_idx;
            for _ in 0..max_new_tokens {
                if cancel.is_some_and(CancelFlag::is_cancelled) {
                    return Err(Error::Canceled);
                }
                if eos.contains(&next) {
                    break;
                }
                out.push(next);
                t += 1;
                next = self.decode_argmax(next, t, cache)?;
            }
            return Ok(out);
        }

        let mut rng = SplitMix64::new(sampler.seed());
        let mut logits = first_logits.to_vec();
        let mut out = Vec::new();
        let mut t = t_idx;
        for _ in 0..max_new_tokens {
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                return Err(Error::Canceled);
            }
            let next = sampler.pick(&logits, &mut rng)?;
            if eos.contains(&next) {
                break;
            }
            out.push(next);
            t += 1;
            logits = self.decode_logits(next, t, cache)?;
        }
        Ok(out)
    }

    /// The `_generate_think` think/no-think rollout. Greedily decodes a think block from
    /// `first_logits` (the prefix's last-position logits) until `</think>` (`think_end_id`) or any
    /// `eos`, forwarding each emitted token into `cache`; on `</think>` it forwards that token too
    /// (keeping the cache aligned). It then appends `append_ids` (the tokenizer's `\n\n<img>`,
    /// `add_special_tokens=False`) so the cache is primed at the image boundary. Returns the think
    /// token ids and the post-append temporal index. Greedy-only, matching the reference.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_think(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        think_end_id: i32,
        eos: i32,
        append_ids: &[i32],
        max_think_tokens: usize,
        cancel: Option<&CancelFlag>,
    ) -> Result<ThinkRollout> {
        let mut t = t_idx;
        let mut next = argmax(first_logits);
        let mut think_token_ids = Vec::new();
        let mut closed = false;
        for _ in 0..max_think_tokens {
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                return Err(Error::Canceled);
            }
            if next == eos {
                break;
            }
            if next == think_end_id {
                // Forward `</think>` so the cache includes it, then stop. No logits needed here, so
                // splice it in without an lm_head projection (F-140).
                t = self.append_tokens(&[next], t, cache)?;
                think_token_ids.push(next);
                closed = true;
                break;
            }
            think_token_ids.push(next);
            // Greedy: argmax the next-token logits on device (single-index transfer; F-140).
            next = self.decode_argmax(next, t + 1, cache)?;
            t += 1;
        }
        // Budget exhausted before `</think>` (and the model didn't emit `eos`): synthesize the close
        // so the cache is not primed on an unclosed `<think>` token sequence the model was never
        // trained on, which would degrade the subsequent image generation (F-013).
        if !closed && next != eos {
            t = self.append_tokens(&[think_end_id], t, cache)?;
            think_token_ids.push(think_end_id);
        }
        t = self.append_tokens(append_ids, t, cache)?;
        Ok(ThinkRollout {
            think_token_ids,
            t_idx: t,
        })
    }
}

/// Index of the maximum logit (ties → lowest index, matching `torch.argmax`). Delegates to the
/// shared host argmax (sc-7159); kept as a crate-internal name so the gen-path image-token decode in
/// `t2i.rs` (greedy over `[vocab]` rows already on host) keeps one call site.
pub(crate) fn argmax(logits: &[f32]) -> i32 {
    argmax_host(logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_breaks_ties_to_lowest_index() {
        assert_eq!(argmax(&[0.1, 0.5, 0.5, 0.2]), 1);
        assert_eq!(argmax(&[3.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn top_k_one_is_argmax() {
        // top_k = 1 collapses the shaped distribution to the single max → deterministic argmax,
        // whatever the seed. Exercises the shared sampler through `Sampler::pick`.
        let logits = [0.1, 2.0, 0.5, 1.0];
        let s = Sampler::Sample {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            seed: 123,
        };
        let mut rng = SplitMix64::new(s.seed());
        for _ in 0..16 {
            assert_eq!(s.pick(&logits, &mut rng).unwrap(), 1);
        }
    }

    #[test]
    fn sampling_is_seed_deterministic() {
        let logits = [0.2, 1.5, 0.3, 0.9, 0.1];
        let s = Sampler::Sample {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 42,
        };
        let run = || {
            let mut rng = SplitMix64::new(s.seed());
            (0..8)
                .map(|_| s.pick(&logits, &mut rng).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "same seed → identical token sequence");
    }
}
