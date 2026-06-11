//! FLUX.2 4-axis rotary position embedding — a faithful mirror of the fork's
//! `Flux2PosEmbed` (`models/flux2/model/flux2_transformer/pos_embed.py`).
//!
//! The transformer's joint sequence is positioned by 4 integer coordinate axes
//! `(t, h, w, layer)`:
//! - target latents: `[0, h, w, 0]` over the latent grid,
//! - text tokens: `[0, 0, 0, token_index]`,
//! - edit reference tokens: `[10 + 10·i, h, w, 0]` (a per-reference time offset).
//!
//! Each axis `i` contributes `axes_dim[i] / 2` rotary frequencies, so the returned `cos`/`sin`
//! tables have width `sum(axes_dim) / 2` (= `head_dim / 2`, i.e. 64 for klein-9b). Pure trig —
//! no weights — so it is parity-checked tight.

use mlx_rs::ops::{concatenate_axis, multiply, split};
use mlx_rs::{Array, Dtype};

use mlx_gen::Result;

/// 4-axis RoPE table generator. `theta` and `axes_dim` are fixed by the model config
/// (klein: θ = 2000, axes = (32, 32, 32, 32)).
#[derive(Clone, Copy, Debug)]
pub struct Flux2PosEmbed {
    theta: f32,
    axes_dim: [usize; 4],
}

impl Flux2PosEmbed {
    pub fn new(theta: f32, axes_dim: [usize; 4]) -> Self {
        Self { theta, axes_dim }
    }

    /// `ids`: integer coordinates with the 4 coordinate axes last — `[…, 4]` (e.g. `[seq, 4]`
    /// after the batch is dropped, or `[batch, seq, 4]`). Returns `(cos, sin)`, each `[…,
    /// sum(axes_dim)/2]` in f32. The coordinate axis is taken relative to the end, matching the
    /// fork's `pos[..., i]`.
    pub fn forward(&self, ids: &Array) -> Result<(Array, Array)> {
        let pos = ids.as_dtype(Dtype::Float32)?;
        let last = (pos.shape().len() - 1) as i32;
        // Split the 4 coordinate axes → each `[…, 1]`.
        let axes = split(&pos, 4, last)?;
        let mut cos_parts: Vec<Array> = Vec::with_capacity(4);
        let mut sin_parts: Vec<Array> = Vec::with_capacity(4);
        for (i, &dim) in self.axes_dim.iter().enumerate() {
            // omega[j] = 1 / theta^((2j)/dim), j in 0..dim/2 — the fork's
            // `1.0 / (theta ** (arange(0, dim, 2) / dim))`, computed in f32.
            // Host `f32::powf` is fine here (unlike FLUX.1, which builds omega with MLX ops —
            // `power`/`divide` — to bit-match a frozen mflux fork whose ~4e-7 host-vs-MLX `powf`
            // drift the chaotic 57-block stack amplifies, sc-2787). FLUX.2 has no frozen-fork
            // bit-parity contract, so the static omega table is computed host-side. If FLUX.2 ever
            // needs MLX-reference parity, switch this to the FLUX.1 MLX-op pattern.
            let half = dim / 2;
            let omega: Vec<f32> = (0..half)
                .map(|j| {
                    let scale = (2 * j) as f32 / dim as f32;
                    1.0 / self.theta.powf(scale)
                })
                .collect();
            // `[1, half]` broadcasts against `[…, 1]` → `[…, half]` for any leading rank.
            let omega = Array::from_slice(&omega, &[1, half as i32]);
            let out = multiply(&axes[i], &omega)?;
            cos_parts.push(out.cos()?);
            sin_parts.push(out.sin()?);
        }
        let cos_refs: Vec<&Array> = cos_parts.iter().collect();
        let sin_refs: Vec<&Array> = sin_parts.iter().collect();
        let cos = concatenate_axis(&cos_refs, last)?;
        let sin = concatenate_axis(&sin_refs, last)?;
        Ok((cos, sin))
    }
}
