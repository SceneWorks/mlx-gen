//! Pixtral 2-D vision RoPE + the per-image `cu_seqlens` for block-diagonal attention.
//!
//! Pure functions of the image patch grids — no model weights — so this is the error-prone,
//! easily-verifiable core (the SAM3 discipline): unit-tested independently of the weight-bearing
//! modules. Port of `PixtralRotaryEmbedding` + `position_ids_in_meshgrid` (`modeling_pixtral.py`).
//!
//! For a patch at grid position `(h, w)` the reference builds, with
//! `base[i] = 1/θ^(2i/head_dim)` (`i ∈ 0..head_dim/2`):
//!   `freqs = [ h·base[0], h·base[2], … (the EVEN-indexed base) ‖ w·base[1], w·base[3], … (ODD) ]`
//! (each half `head_dim/4` wide), then `emb = cat(freqs, freqs)` (→ head_dim), `cos/sin = emb.cos()/
//! .sin()`. Combined with the **`rotate_half`** rotation in [`super::attention`]. `position_ids =
//! h·max_patches_per_side + w` only *indexes* a precomputed table in the reference; the freqs depend
//! solely on `(h, w)`, so we build them directly.

use mlx_rs::Array;

/// Cumulative patch counts at image boundaries, `[0, n₀, n₀+n₁, …]` (patch units). Drives the
/// block-diagonal SDPA: each reference image attends only within its own patches.
pub fn cu_seqlens(grids: &[(i32, i32)]) -> Vec<i32> {
    let mut out = vec![0];
    let mut offset = 0;
    for &(gh, gw) in grids {
        offset += gh * gw;
        out.push(offset);
    }
    out
}

/// 2-D RoPE `(cos, sin)`, each `[seq, head_dim]`, over the concatenated image patch grids in
/// row-major (h-major) order. `seq = Σ gh·gw`. Built in f32 (exact integer-driven math); the
/// rotation itself runs in f32 too (see [`super::attention::apply_rope`]).
pub fn rope_2d(grids: &[(i32, i32)], head_dim: i32, theta: f32) -> (Array, Array) {
    let dim = head_dim as usize;
    let half = dim / 2; // head_dim/2 base freqs
    let base: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
        .collect();
    // freqs_h ← base[::2], freqs_w ← base[1::2] (each head_dim/4 wide).
    let fh: Vec<f32> = base.iter().step_by(2).copied().collect();
    let fw: Vec<f32> = base.iter().skip(1).step_by(2).copied().collect();

    let seq: i32 = grids.iter().map(|&(gh, gw)| gh * gw).sum();
    let mut cos_data = Vec::with_capacity(seq as usize * dim);
    let mut sin_data = Vec::with_capacity(seq as usize * dim);
    for &(gh, gw) in grids {
        for hh in 0..gh {
            for ww in 0..gw {
                // freqs = [h·fh ‖ w·fw] (head_dim/2), then emb = cat(freqs, freqs) (head_dim).
                let mut freqs = Vec::with_capacity(half);
                for &f in &fh {
                    freqs.push(hh as f32 * f);
                }
                for &f in &fw {
                    freqs.push(ww as f32 * f);
                }
                for _ in 0..2 {
                    for &v in &freqs {
                        cos_data.push(v.cos());
                        sin_data.push(v.sin());
                    }
                }
            }
        }
    }
    (
        Array::from_slice(&cos_data, &[seq, head_dim]),
        Array::from_slice(&sin_data, &[seq, head_dim]),
    )
}
