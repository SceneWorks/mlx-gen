//! EVA `VisionRotaryEmbeddingFast` — the 2-D vision RoPE applied to the patch tokens of each
//! attention block. Port of `eva_clip/rope.py`.
//!
//! TWO things make this distinct from the half-split text/Qwen-vision RoPE (`mlx_gen::nn::TextRope`,
//! `mlx-gen-qwen-image` vision):
//!   1. **Interleaved** `rotate_half`: pairs are adjacent — `out[2i] = -x[2i+1]; out[2i+1] = x[2i]`
//!      (einops `rearrange(x,'(d r)->d r',r=2)` then `stack(-x2,x1)`), with the per-freq table
//!      duplicated adjacently (`repeat('n -> (n r)', r=2)`). NOT the `[-x2, x1]` half-split form.
//!   2. **2-D** table: the first half of each head dim is driven by the patch *row*, the second half
//!      by the patch *column* — `broadcat((freqs[:,None,:], freqs[None,:,:]))`.
//!
//! The freqs are deterministic (no weights); we rebuild them and gate the result against the
//! checkpoint's `rope.freqs_cos/sin` buffers. Built in f64 on the host (theta-pow seed), cast to f32.

use mlx_rs::ops::{add, concatenate_axis, multiply, split};
use mlx_rs::{Array, Dtype};

use mlx_gen::Result;

/// The fixed RoPE cos/sin tables `[grid²=576, head_dim=64]` for EVA02-CLIP-L-14-336.
pub struct VisionRope {
    cos: Array,
    sin: Array,
}

impl VisionRope {
    /// Rebuild the `VisionRotaryEmbeddingFast` cos/sin tables for a square patch grid.
    ///
    /// `head_dim` = 64, `grid` = image/patch = 24, `pt_seq_len` = 16 (the pretraining grid; EVA
    /// interpolates frequencies via `t = arange(grid)/grid * pt_seq_len`).
    pub fn build(head_dim: i32, grid: i32, pt_seq_len: i32, theta: f64) -> Result<Self> {
        let half = (head_dim / 2) as usize; // rope dim = head_dim/2 = 32
        let nfreq = half / 2; // 16 base frequencies (arange(0,half,2))
                              // freqs[j] = 1 / theta^(2j / half)
        let freqs: Vec<f64> = (0..nfreq)
            .map(|j| 1.0 / theta.powf((2 * j) as f64 / half as f64))
            .collect();
        let grid_u = grid as usize;
        // t[i] = i/grid * pt_seq_len
        let t: Vec<f64> = (0..grid_u)
            .map(|i| i as f64 / grid as f64 * pt_seq_len as f64)
            .collect();
        // per-axis table rg[pos, d] for d in 0..half (32): rg[pos, 2j] = rg[pos, 2j+1] = t[pos]*freqs[j]
        let axis_table = |pos: usize| -> Vec<f64> {
            let mut row = vec![0.0f64; half];
            for j in 0..nfreq {
                let val = t[pos] * freqs[j];
                row[2 * j] = val;
                row[2 * j + 1] = val;
            }
            row
        };
        // full[h*grid+w, :] = concat(axis_table(h), axis_table(w))  -> [576, 64]
        let n = grid_u * grid_u;
        let mut cos = vec![0.0f32; n * head_dim as usize];
        let mut sin = vec![0.0f32; n * head_dim as usize];
        for h in 0..grid_u {
            let th = axis_table(h);
            for w in 0..grid_u {
                let tw = axis_table(w);
                let base = (h * grid_u + w) * head_dim as usize;
                for d in 0..half {
                    let (vh, vw) = (th[d], tw[d]);
                    cos[base + d] = vh.cos() as f32;
                    sin[base + d] = vh.sin() as f32;
                    cos[base + half + d] = vw.cos() as f32;
                    sin[base + half + d] = vw.sin() as f32;
                }
            }
        }
        Ok(Self {
            cos: Array::from_slice(&cos, &[n as i32, head_dim]),
            sin: Array::from_slice(&sin, &[n as i32, head_dim]),
        })
    }

    pub fn cos(&self) -> &Array {
        &self.cos
    }
    pub fn sin(&self) -> &Array {
        &self.sin
    }

    /// Apply RoPE to patch-token q/k: `x·cos + rotate_half_interleaved(x)·sin`, computed in f32.
    /// `x`: `[B, heads, grid², head_dim]` (CLS token already sliced off by the caller).
    pub fn apply(&self, x: &Array) -> Result<Array> {
        let orig = x.dtype();
        let xf = x.as_dtype(Dtype::Float32)?;
        // cos/sin [576,64] -> [1,1,576,64] for broadcast over (B, heads)
        let cos = self
            .cos
            .reshape(&[1, 1, self.cos.shape()[0], self.cos.shape()[1]])?;
        let sin = self
            .sin
            .reshape(&[1, 1, self.sin.shape()[0], self.sin.shape()[1]])?;
        let rot = rotate_half_interleaved(&xf)?;
        let out = add(&multiply(&xf, &cos)?, &multiply(&rot, &sin)?)?;
        Ok(out.as_dtype(orig)?)
    }
}

/// Interleaved `rotate_half`: reshape the last dim `(.. , d, 2)`, then rebuild `(-x_odd, x_even)`.
/// Result: `out[2i] = -x[2i+1]`, `out[2i+1] = x[2i]`.
fn rotate_half_interleaved(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let last = sh[sh.len() - 1];
    let mut pair_shape = sh.to_vec();
    *pair_shape.last_mut().unwrap() = last / 2;
    pair_shape.push(2);
    let xr = x.reshape(&pair_shape)?; // [.., d, 2]
    let axis = (pair_shape.len() - 1) as i32;
    let halves = split(&xr, 2, axis)?; // even=[..,d,1], odd=[..,d,1]
    let rotated = concatenate_axis(&[&halves[1].negative()?, &halves[0]], axis)?; // (-odd, even)
    Ok(rotated.reshape(sh)?)
}
