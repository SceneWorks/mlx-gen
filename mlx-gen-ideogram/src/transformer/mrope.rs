//! Interleaved 3D MRoPE (`Ideogram4MRoPE`). For each rotary frequency index `d` (`0..head_dim/2`)
//! the position used is the **t** axis by default, overridden to **h** at `d ≡ 1 (mod 3)` for
//! `d < section_h·3` and to **w** at `d ≡ 2 (mod 3)` for `d < section_w·3` — exactly the upstream
//! `freqs_t[..., idx] = freqs[axis][..., idx]` interleave. The text-only TE path (1-D) is a special
//! case where t = h = w, but the DiT mixes image-grid (t,h,w) positions so the 3 axes differ.

use mlx_rs::ops::{concatenate_axis, cos, multiply, sin, split};
use mlx_rs::{Array, Dtype};

pub struct Ideogram4MRoPE {
    /// `[1, 1, head_dim/2]` inverse frequencies.
    inv_freq: Array,
    /// `[1, 1, head_dim/2]` 0/1 axis selectors (t/h/w) for each frequency index.
    mask_t: Array,
    mask_h: Array,
    mask_w: Array,
}

impl Ideogram4MRoPE {
    pub fn new(head_dim: i32, theta: f32, mrope_section: [i32; 3]) -> Self {
        let half = (head_dim / 2) as usize;
        let mut inv = vec![0f32; half];
        let mut mt = vec![0f32; half];
        let mut mh = vec![0f32; half];
        let mut mw = vec![0f32; half];
        let (len_h, len_w) = (mrope_section[1] * 3, mrope_section[2] * 3);
        for d in 0..half {
            // arange(0, head_dim, 2)[d] / head_dim = 2d / head_dim.
            inv[d] = theta.powf(-(2.0 * d as f32) / head_dim as f32);
            let di = d as i32;
            let axis = if di % 3 == 1 && di < len_h {
                1
            } else if di % 3 == 2 && di < len_w {
                2
            } else {
                0
            };
            match axis {
                1 => mh[d] = 1.0,
                2 => mw[d] = 1.0,
                _ => mt[d] = 1.0,
            }
        }
        let shape = [1, 1, half as i32];
        Self {
            inv_freq: Array::from_slice(&inv, &shape),
            mask_t: Array::from_slice(&mt, &shape),
            mask_h: Array::from_slice(&mh, &shape),
            mask_w: Array::from_slice(&mw, &shape),
        }
    }

    /// `position_ids`: `[B, L, 3]` int (t, h, w). Returns `(cos, sin)` `[B, L, head_dim]`.
    pub fn forward(&self, position_ids: &Array) -> mlx_rs::error::Result<(Array, Array)> {
        let pos = position_ids.as_dtype(Dtype::Float32)?;
        let parts = split(&pos, 3, 2)?; // 3 × [B, L, 1]
                                        // sel[b,l,d] = pos_axis_of(d)[b,l]  (broadcast [B,L,1] · [1,1,half] → [B,L,half])
        let sel = multiply(&parts[0], &self.mask_t)?
            + multiply(&parts[1], &self.mask_h)?
            + multiply(&parts[2], &self.mask_w)?;
        let freqs = multiply(&sel, &self.inv_freq)?; // [B, L, half]
        let emb = concatenate_axis(&[&freqs, &freqs], 2)?; // [B, L, head_dim]
        Ok((cos(&emb)?, sin(&emb)?))
    }
}
