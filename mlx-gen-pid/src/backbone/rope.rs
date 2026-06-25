//! 2-D NTK-aware image RoPE + 1-D text RoPE — host-computed `(cos, sin)` tables, then the rotation
//! applied to q/k. Faithful port of `pixeldit_official.py`'s `precompute_freqs_cis_2d_ntk`,
//! `fetch_pos_text`, and `apply_rotary_emb`.
//!
//! The reference packs each per-axis angle into an interleaved real `[N, head_dim/2, 2]` (cos, sin)
//! tensor where consecutive dim-pairs alternate the x-axis and y-axis rotation (element `2j` = x,
//! `2j+1` = y). We compute the `cos`/`sin` halves on the host (deterministic f32 functions of the
//! grid) and rotate the interleaved `(real, imag)` pairs exactly as `apply_rotary_emb` does.

use mlx_rs::ops::{concatenate_axis, split};
use mlx_rs::Array;

use mlx_gen::nn::rope_rotate;
use mlx_gen::Result;

/// Host `(cos, sin)` tables `[L, head_dim/2]` for the 2-D NTK-aware image RoPE.
///
/// Mirrors `precompute_freqs_cis_2d_ntk(dim=head_dim, height=hs, width=ws, ref_grid_h, ref_grid_w,
/// theta=10000, scale=16)`. Token order is row-major over `(hs, ws)` with `ws` fastest (matching the
/// reference `meshgrid(..., indexing="ij").reshape(-1)`); dim-pair `m=2j` rotates by the x-axis
/// (width) angle, `m=2j+1` by the y-axis (height) angle.
pub fn rope_2d_ntk(
    head_dim: i32,
    hs: i32,
    ws: i32,
    ref_grid_h: i32,
    ref_grid_w: i32,
    theta: f32,
    scale: f32,
) -> (Array, Array) {
    let dim = head_dim as f64;
    let dim_axis = dim / 2.0;
    let ntk_exp = if dim_axis > 2.0 {
        dim_axis / (dim_axis - 2.0)
    } else {
        1.0
    };
    let h_scale = hs as f64 / ref_grid_h as f64;
    let w_scale = ws as f64 / ref_grid_w as f64;
    let h_theta = theta as f64 * h_scale.powf(ntk_exp);
    let w_theta = theta as f64 * w_scale.powf(ntk_exp);

    let lin = |n: i32, idx: i32| -> f64 {
        if n <= 1 {
            0.0
        } else {
            idx as f64 * scale as f64 / (n - 1) as f64
        }
    };
    let n_pairs = (head_dim / 4) as usize; // dim//4 complex pairs per axis -> dim//2 real angles
    let freqs_w: Vec<f64> = (0..n_pairs)
        .map(|j| 1.0 / w_theta.powf((4 * j) as f64 / dim))
        .collect();
    let freqs_h: Vec<f64> = (0..n_pairs)
        .map(|j| 1.0 / h_theta.powf((4 * j) as f64 / dim))
        .collect();

    let half = (head_dim / 2) as usize;
    let l = (hs * ws) as usize;
    let mut cos = vec![0f32; l * half];
    let mut sin = vec![0f32; l * half];
    for r in 0..hs {
        let yp = lin(hs, r);
        for c in 0..ws {
            let xp = lin(ws, c);
            let p = (r * ws + c) as usize;
            for m in 0..half {
                let j = m / 2;
                let angle = if m % 2 == 0 {
                    xp * freqs_w[j]
                } else {
                    yp * freqs_h[j]
                };
                cos[p * half + m] = angle.cos() as f32;
                sin[p * half + m] = angle.sin() as f32;
            }
        }
    }
    (
        Array::from_slice(&cos, &[l as i32, half as i32]),
        Array::from_slice(&sin, &[l as i32, half as i32]),
    )
}

/// Host `(cos, sin)` tables `[length, head_dim/2]` for the 1-D text RoPE.
///
/// Mirrors `fetch_pos_text`: `freqs[m] = theta^(-2m/head_dim)`, `angle[l,m] = l·freqs[m]`.
pub fn rope_1d_text(head_dim: i32, length: i32, theta: f32) -> (Array, Array) {
    let half = (head_dim / 2) as usize;
    let freqs: Vec<f64> = (0..half)
        .map(|m| 1.0 / (theta as f64).powf((2 * m) as f64 / head_dim as f64))
        .collect();
    let len = length as usize;
    let mut cos = vec![0f32; len * half];
    let mut sin = vec![0f32; len * half];
    for l in 0..len {
        for m in 0..half {
            let angle = l as f64 * freqs[m];
            cos[l * half + m] = angle.cos() as f32;
            sin[l * half + m] = angle.sin() as f32;
        }
    }
    (
        Array::from_slice(&cos, &[length, half as i32]),
        Array::from_slice(&sin, &[length, half as i32]),
    )
}

/// Apply interleaved RoPE to `q`/`k` in `[B, H, S, D]` with `cos`/`sin` `[S, D/2]`. Pairs
/// `(x[2i], x[2i+1])` as `(real, imag)` and rotates by `cos/sin[i]` — bit-equivalent to
/// `apply_rotary_emb`'s `_rotate` (the head axis is a pure broadcast, so applying after the
/// `[B,H,S,D]` transpose is identical to the reference's pre-transpose `[B,S,H,D]` apply).
pub fn apply_rope(q: &Array, k: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
    let s = cos.shape()[0];
    let half = cos.shape()[1];
    let cos = cos.reshape(&[1, 1, s, half])?;
    let sin = sin.reshape(&[1, 1, s, half])?;
    let one = |x: &Array| -> Result<Array> {
        let sh = x.shape();
        let (b, h, seq, hd) = (sh[0], sh[1], sh[2], sh[3]);
        let x5 = x.reshape(&[b, h, seq, hd / 2, 2])?;
        let p = split(&x5, 2, 4)?;
        let real = p[0].reshape(&[b, h, seq, hd / 2])?;
        let imag = p[1].reshape(&[b, h, seq, hd / 2])?;
        let (out0, out1) = rope_rotate(&real, &imag, &cos, &sin)?;
        Ok(
            concatenate_axis(&[&out0.expand_dims(4)?, &out1.expand_dims(4)?], 4)?
                .reshape(&[b, h, seq, hd])?,
        )
    };
    Ok((one(q)?, one(k)?))
}
