//! KV-cache for the FLUX.2-klein-9b-kv edit variant (sc-2347).
//!
//! The `-kv` checkpoint is distilled from FLUX.2-klein-9b at 4 inference steps with an
//! attention-side optimisation: the reference-image K/V tensors are computed once on step 0 and
//! re-used on steps 1..N, skipping the redundant reference-token processing. This is the faithful
//! Rust analog of the fork's `Flux2KVCache`
//! (`models/flux2/model/flux2_transformer/flux2_kv_cache.py`).
//!
//! **Why plain mutable state, not `compile_with_state`.** The model forward is **not** whole-graph
//! compiled — only stateless elementwise chains (the adaLN modulate / gated residuals, sc-2963) run
//! through `mx.compile`, and the cache never participates in one — so there is no compiled graph that
//! needs a mutable-state threading mechanism. The 2.4× speedup comes entirely from the cache
//! *reducing work* on steps 1..N (only `[txt, target]` queries, and no ref K/V recompute) — exactly
//! as in the fork, which also disables compile for this path. The cache is therefore an ordinary
//! interior-mutability container the `&self` transformer forward writes on step 0 and reads thereafter.
//!
//! **Token layout** (mflux order, which the Rust transformer already matches): the joint attention
//! sequence is `[txt, target, ref]`. The reference tokens are the trailing `num_ref` slice. In
//! `Extract` mode each attention layer stores that post-RoPE trailing slice of (K, V); in `Cached`
//! mode the input carries no ref tokens (`[txt, target]`) and each layer splices the cached ref K/V
//! onto the **end** of the fresh K/V, so the `[txt, target]` queries attend over the full
//! `[txt, target, ref]` K/V.

use std::cell::{Cell, RefCell};

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::{Error, Result};

/// Which denoise-step role the cache is playing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheMode {
    /// Step 0: run the full `[txt, target, ref]` forward and store the trailing ref K/V per layer.
    Extract,
    /// Steps 1..N: run the `[txt, target]` forward; splice the stored ref K/V back in per layer.
    Cached,
}

/// Which transformer stack a cache slot belongs to (the two stacks are indexed independently).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Double,
    Single,
}

/// Per-layer reference K/V store for one edit generation.
///
/// Interior mutability (the transformer forward borrows it as `&self`): the per-layer slots are
/// written once in [`CacheMode::Extract`] and read in [`CacheMode::Cached`]. Build a fresh cache
/// per seed — the reference K/V depend on the step-0 target latents (which are seed-dependent).
pub struct Flux2KvCache {
    double: RefCell<Vec<Option<(Array, Array)>>>,
    single: RefCell<Vec<Option<(Array, Array)>>>,
    mode: Cell<Option<CacheMode>>,
    /// Count of trailing reference tokens to cache (the static slice). `0` disables caching.
    num_ref_tokens: Cell<i32>,
}

impl Flux2KvCache {
    /// One slot per double-stream and per single-stream block.
    pub fn new(num_double_layers: usize, num_single_layers: usize) -> Self {
        Self {
            double: RefCell::new(vec![None; num_double_layers]),
            single: RefCell::new(vec![None; num_single_layers]),
            mode: Cell::new(None),
            num_ref_tokens: Cell::new(0),
        }
    }

    /// Set the mode + reference-token count for the upcoming forward (the fork's
    /// `Flux2KVCache.configure`).
    pub fn configure(&self, mode: CacheMode, num_ref_tokens: usize) {
        self.mode.set(Some(mode));
        self.num_ref_tokens.set(num_ref_tokens as i32);
    }

    pub fn mode(&self) -> Option<CacheMode> {
        self.mode.get()
    }

    /// The cache hook a single attention layer calls, **after RoPE, before SDPA**, with its freshly
    /// projected `(key, value)` in `[B, H, S, D]`. Returns the `(key, value)` to attend over:
    /// - [`CacheMode::Extract`]: stores the trailing `num_ref` slice for `(stream, layer_idx)` and
    ///   returns `(key, value)` unchanged (the full `[txt, target, ref]` attention is identical to
    ///   the no-cache forward — only the side-effect of storing differs).
    /// - [`CacheMode::Cached`]: loads the stored ref K/V and concatenates them onto the **end** of
    ///   the fresh `(key, value)` along the sequence axis, reconstructing the `[txt, target, ref]`
    ///   K/V layout for the `[txt, target]` queries.
    ///
    /// A no-op (returns the inputs unchanged) when no mode is set or `num_ref == 0`.
    pub fn apply(
        &self,
        stream: Stream,
        layer_idx: usize,
        key: Array,
        value: Array,
    ) -> Result<(Array, Array)> {
        let num_ref = self.num_ref_tokens.get();
        match self.mode.get() {
            Some(CacheMode::Extract) if num_ref > 0 => {
                let ref_k = trailing(&key, num_ref)?;
                let ref_v = trailing(&value, num_ref)?;
                self.slots(stream).borrow_mut()[layer_idx] = Some((ref_k, ref_v));
                Ok((key, value))
            }
            Some(CacheMode::Cached) if num_ref > 0 => {
                let slots = self.slots(stream).borrow();
                let (cached_k, cached_v) = slots[layer_idx].as_ref().ok_or_else(|| {
                    Error::Msg(format!(
                        "flux2 kv-cache: {stream:?} layer {layer_idx} slot is empty — \
                         the step-0 extract pass did not run"
                    ))
                })?;
                let key = concatenate_axis(&[&key, cached_k], 2)?;
                let value = concatenate_axis(&[&value, cached_v], 2)?;
                Ok((key, value))
            }
            _ => Ok((key, value)),
        }
    }

    fn slots(&self, stream: Stream) -> &RefCell<Vec<Option<(Array, Array)>>> {
        match stream {
            Stream::Double => &self.double,
            Stream::Single => &self.single,
        }
    }

    /// All per-layer slots are populated (both stacks) — i.e. the extract pass has run.
    pub fn is_populated(&self) -> bool {
        self.double.borrow().iter().all(Option::is_some)
            && self.single.borrow().iter().all(Option::is_some)
    }
}

/// The trailing `n` tokens of `a` along the sequence axis (axis 2 in `[B, H, S, D]`).
fn trailing(a: &Array, n: i32) -> Result<Array> {
    let s = a.shape()[2];
    let idx = Array::from_slice(&((s - n)..s).collect::<Vec<i32>>(), &[n]);
    Ok(a.take_axis(&idx, 2)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(seq: i32, fill: f32) -> Array {
        // [B=1, H=1, S=seq, D=2], each token = (fill + token_index).
        let data: Vec<f32> = (0..seq)
            .flat_map(|i| [fill + i as f32, fill + i as f32])
            .collect();
        Array::from_slice(&data, &[1, 1, seq, 2])
    }

    #[test]
    fn extract_stores_trailing_ref_and_returns_input_unchanged() {
        let cache = Flux2KvCache::new(1, 1);
        cache.configure(CacheMode::Extract, 2); // 2 ref tokens
        let k = arr(5, 100.0); // [txt+target+ref] = 5 tokens
        let v = arr(5, 200.0);
        let (ok, ov) = cache
            .apply(Stream::Double, 0, k.clone(), v.clone())
            .unwrap();
        // Extract returns the inputs unchanged.
        assert_eq!(ok.shape(), &[1, 1, 5, 2]);
        assert!(mlx_rs::ops::array_eq(&ok, &k, false)
            .unwrap()
            .item::<bool>());
        assert!(mlx_rs::ops::array_eq(&ov, &v, false)
            .unwrap()
            .item::<bool>());
        assert!(!cache.is_populated()); // single slot still empty
    }

    #[test]
    fn cached_splices_stored_ref_onto_the_end() {
        let cache = Flux2KvCache::new(1, 1);
        // Populate via extract on a [txt,target,ref] = 5-token forward, ref=2.
        cache.configure(CacheMode::Extract, 2);
        let k_full = arr(5, 100.0);
        let v_full = arr(5, 200.0);
        cache
            .apply(Stream::Double, 0, k_full.clone(), v_full.clone())
            .unwrap();
        cache.apply(Stream::Single, 0, k_full, v_full).unwrap();
        assert!(cache.is_populated());

        // Cached: fresh input is [txt,target] = 3 tokens; expect spliced length 3 + 2 = 5.
        cache.configure(CacheMode::Cached, 2);
        let k_short = arr(3, 100.0);
        let v_short = arr(3, 200.0);
        let (ok, ov) = cache.apply(Stream::Double, 0, k_short, v_short).unwrap();
        assert_eq!(ok.shape(), &[1, 1, 5, 2]);
        assert_eq!(ov.shape(), &[1, 1, 5, 2]);
        // The spliced tail must equal the cached trailing ref slice (tokens 3,4 of the full key).
        let tail = trailing(&ok, 2).unwrap();
        let want = trailing(&arr(5, 100.0), 2).unwrap();
        assert!(mlx_rs::ops::array_eq(&tail, &want, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn cached_without_extract_errors() {
        let cache = Flux2KvCache::new(1, 1);
        cache.configure(CacheMode::Cached, 2);
        let err = cache
            .apply(Stream::Double, 0, arr(3, 1.0), arr(3, 2.0))
            .unwrap_err()
            .to_string();
        assert!(err.contains("slot is empty"), "got: {err}");
    }

    #[test]
    fn num_ref_zero_is_a_noop() {
        let cache = Flux2KvCache::new(1, 1);
        cache.configure(CacheMode::Extract, 0);
        let k = arr(4, 1.0);
        let (ok, _) = cache
            .apply(Stream::Double, 0, k.clone(), arr(4, 2.0))
            .unwrap();
        assert!(mlx_rs::ops::array_eq(&ok, &k, false)
            .unwrap()
            .item::<bool>());
        // Nothing stored.
        assert!(cache.slots(Stream::Double).borrow()[0].is_none());
    }
}
