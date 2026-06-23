//! Krea 2 DiT 3-axis (t, h, w) unified RoPE — the reference `mmdit.py` `PositionalEncoding` + `rope`
//! + `ropeapply`.
//!
//! Two facts from the reference, both parity-critical:
//!  1. **Complex *interleaved* rotation** (GPT-J / "lumina"): adjacent dims `(2k, 2k+1)` form a complex
//!     pair `x[2k] + i·x[2k+1]` rotated by `e^{iθ_k}`, *not* the half-split `[x1,x2]→[-x2,x1]`. The
//!     reference's `ropeapply` builds a per-(pos,freq) 2×2 rotation `[[cos,-sin],[sin,cos]]` and applies
//!     `out[2k] = cos·x[2k] − sin·x[2k+1]`, `out[2k+1] = sin·x[2k] + cos·x[2k+1]`. (Byte-identical to
//!     `mlx-gen-boogu`'s `apply_interleaved_rope`, reused verbatim here.)
//!  2. **Three position axes with UNEQUAL sub-dims** `axes_dims_rope = [32,48,48]` (boogu's are equal,
//!     so its table builder doesn't generalize). The head-dim freq index `k ∈ [0, head_dim/2)` is split
//!     into three contiguous blocks of `axes[i]/2`, each block `i` using its own inverse frequencies
//!     `θ^(−2j/axes[i])` over its own position axis.
//!
//! **Position scheme** (reference `sampling.py::prepare`): text tokens are all `(0,0,0)`; image patch
//! tokens are `(0, row, col)` — the t-axis is **always 0**, so only the h/w axes carry position and the
//! text tokens get identity RoPE. The joint `[text; image]` table is applied to the whole single-stream
//! sequence (the text-fusion blocks use no RoPE).
//!
//! Inverse frequencies + angles are built on the host in **f64** (the reference's `rope` uses
//! `torch.float64`), then the `cos`/`sin` tables are materialized as f32 (`rope(...).float()`).

use mlx_rs::ops::{add, concatenate_axis, multiply, split, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::Result;

/// Precomputed `cos`/`sin` rotary tables for one forward pass, laid out `[1, cap_len + img_len,
/// head_dim/2]` (f32) in joint `[text; image]` order.
pub struct RopeTables {
    cos: Array,
    sin: Array,
}

impl RopeTables {
    /// Build the joint table for a text-to-image forward: `cap_len` text positions `(0,0,0)` followed
    /// by an `h_tokens × w_tokens` row-major image grid `(0, row, col)`. `axes` are the per-axis RoPE
    /// sub-dims (`[t,h,w]`, summing to `head_dim`); `theta` is `rope_theta`.
    pub fn build_t2i(
        cap_len: usize,
        h_tokens: usize,
        w_tokens: usize,
        axes: [usize; 3],
        theta: f64,
    ) -> Self {
        let mut positions = Vec::with_capacity(cap_len + h_tokens * w_tokens);
        for _ in 0..cap_len {
            positions.push((0.0, 0.0, 0.0));
        }
        for r in 0..h_tokens {
            for c in 0..w_tokens {
                positions.push((0.0, r as f64, c as f64));
            }
        }
        from_positions(&positions, axes, theta)
    }

    /// `(cos, sin)` for the full joint `[text; image]` sequence (the single-stream blocks).
    pub fn joint(&self) -> (Array, Array) {
        (self.cos.clone(), self.sin.clone())
    }
}

/// Build the `cos`/`sin` tables from 3-axis positions. For freq block `i` (sub-dim `axes[i]`, so
/// `axes[i]/2` complex freqs) the inverse frequencies are `θ^(−2j/axes[i])` (`j ∈ [0, axes[i]/2)`),
/// each multiplied by that token's position on axis `i`. Computed in f64, stored f32.
fn from_positions(positions: &[(f64, f64, f64)], axes: [usize; 3], theta: f64) -> RopeTables {
    // Per-axis inverse frequencies in f64 (reference `rope`: `1 / (theta ** (arange(0,d,2)/d))`).
    let inv: Vec<Vec<f64>> = axes
        .iter()
        .map(|&d| {
            (0..d / 2)
                .map(|j| 1.0 / theta.powf((2 * j) as f64 / d as f64))
                .collect()
        })
        .collect();
    let half: usize = axes.iter().map(|d| d / 2).sum(); // head_dim/2

    let total = positions.len();
    let mut cos = vec![0f32; total * half];
    let mut sin = vec![0f32; total * half];
    for (t, &(p0, p1, p2)) in positions.iter().enumerate() {
        let pos = [p0, p1, p2];
        let mut k = 0usize; // running freq index across the three concatenated blocks
        for (axis, freqs) in inv.iter().enumerate() {
            for &f in freqs {
                let angle = pos[axis] * f;
                cos[t * half + k] = angle.cos() as f32;
                sin[t * half + k] = angle.sin() as f32;
                k += 1;
            }
        }
    }

    let shape = [1, total as i32, half as i32];
    RopeTables {
        cos: Array::from_slice(&cos, &shape),
        sin: Array::from_slice(&sin, &shape),
    }
}

/// Apply the complex-interleaved rotary embedding to `x` in `[b, s, heads, head_dim]` layout.
///
/// `cos`/`sin` are `[1, s, head_dim/2]` (broadcast over heads). For each adjacent pair
/// `(x[2k], x[2k+1])`:
///   `out[2k]   = x[2k]·cos_k − x[2k+1]·sin_k`
///   `out[2k+1] = x[2k]·sin_k + x[2k+1]·cos_k`
/// Computed in f32 (the reference upcasts), then cast back to `x`'s dtype.
pub fn apply_interleaved_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let dt = x.dtype();
    let sh = x.shape();
    let (b, s, h, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let half = hd / 2;

    let cos = cos.as_dtype(Dtype::Float32)?.expand_dims(2)?; // [1, s, 1, half]
    let sin = sin.as_dtype(Dtype::Float32)?.expand_dims(2)?;

    let xr = x.as_dtype(Dtype::Float32)?.reshape(&[b, s, h, half, 2])?;
    let parts = split(&xr, 2, 4)?; // 2 × [b, s, h, half, 1]
    let xe = parts[0].reshape(&[b, s, h, half])?;
    let xo = parts[1].reshape(&[b, s, h, half])?;

    let out_e = subtract(&multiply(&xe, &cos)?, &multiply(&xo, &sin)?)?;
    let out_o = add(&multiply(&xe, &sin)?, &multiply(&xo, &cos)?)?;

    let out = concatenate_axis(&[&out_e.expand_dims(4)?, &out_o.expand_dims(4)?], 4)?;
    Ok(out.reshape(&[b, s, h, hd])?.as_dtype(dt)?)
}
