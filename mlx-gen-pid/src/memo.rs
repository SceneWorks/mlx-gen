//! Per-decode memoization of the PiD backbone's **step-invariant host tables** (F-153).
//!
//! `PidNet::forward` rebuilds several pure functions of grid geometry with scalar host loops + an H2D
//! upload on *every* forward: the 2-D sin/cos pixel positional table (`~268 MB` host alloc + 134 M
//! sin/cos per 2048² tile), both RoPE `(cos, sin)` tables, and (under tiling) the per-tile feather
//! weights. All are identical across the 4 sampler steps and identical for same-sized tiles, so a
//! 6144² decode recomputes the identical pixel-pos table 36 times (~9.6 GB of host churn) serialized
//! against the GPU. Memoizing them per decode — keyed on *every* varying dimension — removes that
//! churn while returning byte-identical tables (a cache hit is a cheap clone of the refcounted
//! `Array` handle).
//!
//! Scope is one decode: the caches live on the per-generation [`crate::lq::PidNet`] /
//! [`crate::sampler::Sampler`], both minted fresh per [`crate::PidDecoder`], so nothing persists
//! across generations and there is no cross-decode staleness risk.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;

/// A per-decode table cache keyed by geometry `K` → cached table `V`.
pub(crate) type TableCache<K, V> = RefCell<HashMap<K, V>>;

/// Look the table up by `key`; build + insert it on a miss. `key` MUST capture every input the table
/// varies on (all other inputs are per-decode constants — head_dim / θ / ref-grid from the config),
/// or a hit would return a stale table. `build` is only called on a miss.
pub(crate) fn memo<K, V, F>(cache: &TableCache<K, V>, key: K, build: F) -> V
where
    K: Eq + Hash,
    V: Clone,
    F: FnOnce() -> V,
{
    if let Some(v) = cache.borrow().get(&key) {
        return v.clone();
    }
    let v = build();
    cache.borrow_mut().insert(key, v.clone());
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn memo_builds_once_per_key_and_returns_equal() {
        let cache: TableCache<(i32, i32), Vec<i32>> = RefCell::new(HashMap::new());
        let builds = Cell::new(0);
        let build = |k: (i32, i32)| {
            builds.set(builds.get() + 1);
            vec![k.0, k.1, k.0 + k.1]
        };
        let a = memo(&cache, (2, 3), || build((2, 3)));
        let b = memo(&cache, (2, 3), || build((2, 3)));
        assert_eq!(a, b, "same key → identical table");
        assert_eq!(
            builds.get(),
            1,
            "second lookup is a cache hit, not a rebuild"
        );
        // A distinct key builds again — the cache does not collide across geometries.
        let c = memo(&cache, (4, 5), || build((4, 5)));
        assert_eq!(c, vec![4, 5, 9]);
        assert_eq!(builds.get(), 2);
    }
}
