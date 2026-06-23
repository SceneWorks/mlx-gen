//! Boogu DiT 3-axis (t, h, w) unified RoPE — the OmniGen2 `BooguImageDoubleStreamRotaryPosEmbed`.
//!
//! Two things differ from the Qwen3-VL text encoder's RoPE and matter for parity:
//!  1. **Complex *interleaved* rotation** (`apply_rotary_emb(use_real=False)`, the "lumina" branch):
//!     adjacent dims `(2k, 2k+1)` form a complex pair `x[2k] + i·x[2k+1]` rotated by `e^{iθ_k}`
//!     (GPT-J / interleaved), *not* the text encoder's half-split `[x1, x2] → [-x2, x1]`. MLX has no
//!     `view_as_complex`, so we do the real arithmetic directly.
//!  2. **Three position axes**: per token the rotary frequency index `k ∈ [0, 60)` is grouped into
//!     three contiguous blocks of 20 (`axes_dim_rope = [40,40,40]` ⇒ 20 complex freqs each), one per
//!     axis. Text tokens use position `(i, i, i)`; image patch tokens use `(cap_len, row, col)`.
//!
//! Each axis shares the same 20-vector of inverse frequencies `θ^(−2j/40)` (`θ = 10000`). We build the
//! `cos`/`sin` tables on the CPU in f32 (the reference builds the freqs in f32 on MPS) and slice the
//! joint table into its text-only / image-only sub-ranges.

use mlx_rs::ops::{add, concatenate_axis, multiply, split, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::Result;

/// Precomputed `cos`/`sin` rotary tables for one forward pass.
///
/// Layout is `[1, cap_len + ref_len + img_len, head_dim/2]` (f32) in the joint
/// `[instruct; ref-image; noise-image]` order. For text-to-image there is no reference image
/// (`ref_len == 0`) and the layout collapses to `[instruct; image]`.
pub struct RopeTables {
    cos: Array,
    sin: Array,
    cap_len: i32,
    ref_len: i32,
}

impl RopeTables {
    /// Build the joint table for a text-to-image forward (no reference images): `cap_len` text
    /// positions followed by an `h_tokens × w_tokens` image grid (row-major, `h` outer).
    pub fn build_t2i(
        cap_len: usize,
        h_tokens: usize,
        w_tokens: usize,
        axes_dim: usize,
        theta: f32,
    ) -> Self {
        let mut positions = Vec::with_capacity(cap_len + h_tokens * w_tokens);
        text_positions(&mut positions, cap_len);
        grid_positions(&mut positions, cap_len as f32, h_tokens, w_tokens);
        from_positions(&positions, axes_dim, theta, cap_len as i32, 0)
    }

    /// Build the joint table for an **edit** forward with one or more reference images (the OmniGen2
    /// unified-RoPE multi-image scheme): `cap_len` text positions, then each reference's `rh × rw`
    /// grid placed at its own t-axis position `pe_shift` (starting at `cap_len` and advancing by
    /// `max(rh, rw)` after each reference), then the `h × w` target grid at the final
    /// `pe_shift = cap_len + Σ max(rh_j, rw_j)` — matching the `[instruct; ref₀; …; ref_{N-1}; noise]`
    /// packing the DiT runs the single-stream over. `ref_grids` are `(rh_tokens, rw_tokens)` per
    /// reference, in order; a single-element slice reproduces the single-reference table exactly.
    pub fn build_edit(
        cap_len: usize,
        ref_grids: &[(usize, usize)],
        h_tokens: usize,
        w_tokens: usize,
        axes_dim: usize,
        theta: f32,
    ) -> Self {
        let ref_len: usize = ref_grids.iter().map(|(h, w)| h * w).sum();
        let mut positions = Vec::with_capacity(cap_len + ref_len + h_tokens * w_tokens);
        text_positions(&mut positions, cap_len);
        // Each reference grid at its own t-axis position; `pe_shift` advances by the reference's longer
        // side after each (the OmniGen2 `pe_shift += max(ref_H_tokens, ref_W_tokens)`), so references
        // occupy disjoint t-ranges and the noise grid follows the last one.
        let mut pe_shift = cap_len;
        for &(rh, rw) in ref_grids {
            grid_positions(&mut positions, pe_shift as f32, rh, rw);
            pe_shift += rh.max(rw);
        }
        grid_positions(&mut positions, pe_shift as f32, h_tokens, w_tokens);
        from_positions(&positions, axes_dim, theta, cap_len as i32, ref_len as i32)
    }

    /// `(cos, sin)` for the text tokens only (`context_refiner`).
    pub fn text(&self) -> Result<(Array, Array)> {
        Ok((
            axis1(&self.cos, 0, self.cap_len)?,
            axis1(&self.sin, 0, self.cap_len)?,
        ))
    }

    /// `(cos, sin)` for **all** reference-image patch tokens (the full `[ref₀; …; ref_{N-1}]` block).
    /// Empty-safe via `ref_len == 0` callers (T2I never calls this).
    pub fn ref_image(&self) -> Result<(Array, Array)> {
        let start = self.cap_len;
        let end = self.cap_len + self.ref_len;
        Ok((axis1(&self.cos, start, end)?, axis1(&self.sin, start, end)?))
    }

    /// `(cos, sin)` for one reference image's tokens — the sub-range at local offset `local_start`
    /// (relative to the start of the reference block) of length `len`. Used to refine each reference
    /// independently (the OmniGen2 per-image batched `ref_image_refiner`: no cross-image attention).
    pub fn ref_image_slice(&self, local_start: usize, len: usize) -> Result<(Array, Array)> {
        let start = self.cap_len + local_start as i32;
        Ok((
            axis1(&self.cos, start, start + len as i32)?,
            axis1(&self.sin, start, start + len as i32)?,
        ))
    }

    /// `(cos, sin)` for the target (noise) patch tokens only (`noise_refiner`). These sit after the
    /// reference block, so the range is `[cap_len + ref_len, end)`.
    pub fn image(&self) -> Result<(Array, Array)> {
        let end = self.cos.shape()[1];
        let start = self.cap_len + self.ref_len;
        Ok((axis1(&self.cos, start, end)?, axis1(&self.sin, start, end)?))
    }

    /// `(cos, sin)` for the combined image sequence `[ref; noise]` (the double-stream image
    /// self-attention). For T2I (`ref_len == 0`) this equals [`Self::image`].
    pub fn combined_image(&self) -> Result<(Array, Array)> {
        let end = self.cos.shape()[1];
        Ok((
            axis1(&self.cos, self.cap_len, end)?,
            axis1(&self.sin, self.cap_len, end)?,
        ))
    }

    /// `(cos, sin)` for the full joint `[text; ref; noise]` sequence (double / single stream).
    pub fn joint(&self) -> (Array, Array) {
        (self.cos.clone(), self.sin.clone())
    }
}

/// Push `cap_len` text positions `(i, i, i)`.
fn text_positions(out: &mut Vec<(f32, f32, f32)>, cap_len: usize) {
    for i in 0..cap_len {
        out.push((i as f32, i as f32, i as f32));
    }
}

/// Push an `h × w` row-major image grid at a fixed t-axis position: `(t, row, col)`.
fn grid_positions(out: &mut Vec<(f32, f32, f32)>, t: f32, h: usize, w: usize) {
    for r in 0..h {
        for c in 0..w {
            out.push((t, r as f32, c as f32));
        }
    }
}

/// Build the `cos`/`sin` tables from 3-axis positions: each rotary freq index `k ∈ [0, 3·axes_dim/2)`
/// is grouped into three contiguous blocks of `axes_dim/2`, one per axis, all sharing the inverse
/// frequencies `θ^(−2j/axes_dim)`.
fn from_positions(
    positions: &[(f32, f32, f32)],
    axes_dim: usize,
    theta: f32,
    cap_len: i32,
    ref_len: i32,
) -> RopeTables {
    let half_axis = axes_dim / 2; // 20 complex freqs per axis
    let half = half_axis * 3; // 60 for head_dim 120
    let inv: Vec<f32> = (0..half_axis)
        .map(|j| theta.powf(-(2.0 * j as f32) / axes_dim as f32))
        .collect();

    let total = positions.len();
    let mut cos = vec![0f32; total * half];
    let mut sin = vec![0f32; total * half];
    for (t, &(p0, p1, p2)) in positions.iter().enumerate() {
        for k in 0..half {
            let p = match k / half_axis {
                0 => p0,
                1 => p1,
                _ => p2,
            };
            let angle = p * inv[k % half_axis];
            cos[t * half + k] = angle.cos();
            sin[t * half + k] = angle.sin();
        }
    }

    let shape = [1, total as i32, half as i32];
    RopeTables {
        cos: Array::from_slice(&cos, &shape),
        sin: Array::from_slice(&sin, &shape),
        cap_len,
        ref_len,
    }
}

/// Slice `[1, L, D]` along the sequence axis (axis 1) to `[start, end)`.
fn axis1(x: &Array, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), 1)?)
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
