//! The NEO-Unify **vision embedder** (sc-3183) — port of `modeling_neo_vit.py`.
//!
//! For the 8B-MoT checkpoint the "vision tower" has no transformer blocks: it is a full-kernel
//! `patch_embedding` (Conv2d 3→`hidden_size`, kernel=stride=`patch_size`) + GELU, an **interleaved**
//! 2D RoPE over the patch grid, then a 2×2-strided `dense_embedding` (Conv2d
//! `hidden_size`→`llm_hidden_size`) that merges each 2×2 block of patches into one LLM token. The
//! same module backs the understanding-path `vision_model` and the generation-path
//! `fm_modules.vision_model_mot_gen` — construct two instances with different prefixes.
//!
//! The full-kernel `patch_embedding` is computed as a Linear over the flattened patch (an exact
//! equivalent), while `dense_embedding` is a genuine strided conv. The 2D RoPE here is the
//! **interleaved** (`x[0::2]`/`x[1::2]`) variant — distinct from the backbone's half-split RoPE.

use mlx_rs::ops::{add, concatenate_axis, matmul, multiply, split, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{conv2d, gelu_exact};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::NeoChatConfig;

fn require(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?.clone())
}

/// A NEO vision embedder (Conv patch-embed → 2D RoPE → Conv patch-merge).
pub struct NeoVisionEmbedder {
    /// `patch_embedding` weight flattened to `[embed_dim, num_channels*patch*patch]`.
    patch_w: Array,
    patch_b: Array,
    /// `dense_embedding` weight in MLX conv layout `[llm_dim, factor, factor, embed_dim]`.
    dense_w: Array,
    dense_b: Array,
    embed_dim: i32,
    patch_size: i32,
    num_channels: i32,
    downsample_factor: i32,
    rope_theta: f32,
}

impl NeoVisionEmbedder {
    /// Build from a checkpoint. `prefix` = the embeddings namespace, e.g.
    /// `"vision_model.embeddings"` (understanding) or
    /// `"fm_modules.vision_model_mot_gen.embeddings"` (generation).
    pub fn from_weights(w: &Weights, cfg: &NeoChatConfig, prefix: &str) -> Result<Self> {
        let v = &cfg.vision;
        let embed_dim = v.hidden_size as i32;
        let patch = v.patch_size as i32;
        let ch = v.num_channels as i32;
        let factor = (1.0 / v.downsample_ratio).round() as i32;

        // patch_embedding: torch Conv2d weight [embed, ch, patch, patch] -> flat [embed, ch*patch*patch].
        let patch_w = require(w, &format!("{prefix}.patch_embedding.weight"))?
            .reshape(&[embed_dim, ch * patch * patch])?;
        // dense_embedding: torch [llm, embed, factor, factor] -> MLX [llm, factor, factor, embed].
        let dense_w = require(w, &format!("{prefix}.dense_embedding.weight"))?
            .transpose_axes(&[0, 2, 3, 1])?;

        Ok(Self {
            patch_w,
            patch_b: require(w, &format!("{prefix}.patch_embedding.bias"))?,
            dense_w,
            dense_b: require(w, &format!("{prefix}.dense_embedding.bias"))?,
            embed_dim,
            patch_size: patch,
            num_channels: ch,
            downsample_factor: factor,
            rope_theta: v.rope_theta_vision,
        })
    }

    /// Embed `pixel_values` `[N, num_channels*patch*patch]` (row-major patch list) for one or more
    /// images described by `grid` (each `(h, w)` patch-grid). Returns `[N/factor², llm_hidden]`
    /// tokens in row-major order, concatenated across images.
    pub fn forward(&self, pixel_values: &Array, grid: &[(usize, usize)]) -> Result<Array> {
        let in_dtype = pixel_values.dtype();
        // patch_embedding (full-kernel conv == linear over the flat patch) + GELU.
        let pe = add(&matmul(pixel_values, self.patch_w.t())?, &self.patch_b)?;
        let pe = gelu_exact(&pe)?; // [N, embed]

        // Interleaved 2D RoPE in f32 (RoPE is computed in float32 in the reference).
        let (abs_x, abs_y) = abs_positions(grid);
        let pe_f32 = pe.as_dtype(Dtype::Float32)?;
        let halves = split(&pe_f32, 2, 1)?; // [N, embed/2] each
        let p1 = rope_1d_interleaved(&halves[0], &abs_x, self.rope_theta)?;
        let p2 = rope_1d_interleaved(&halves[1], &abs_y, self.rope_theta)?;
        let roped = concatenate_axis(&[&p1, &p2], 1)?.as_dtype(in_dtype)?; // [N, embed]

        // dense_embedding (2×2 patch merge) per image.
        let f = self.downsample_factor;
        let mut outs: Vec<Array> = Vec::with_capacity(grid.len());
        let mut cur = 0usize;
        for &(h, w) in grid {
            let n = (h * w) as i32;
            let idx = Array::from_slice(&(cur as i32..cur as i32 + n).collect::<Vec<_>>(), &[n]);
            let block =
                roped
                    .take_axis(&idx, 0)?
                    .reshape(&[1, h as i32, w as i32, self.embed_dim])?; // NHWC
            let merged = conv2d(&block, &self.dense_w, Some(&self.dense_b), f, 0)?; // [1, h/f, w/f, llm]
            let llm = merged.shape()[3];
            outs.push(merged.reshape(&[(h as i32 / f) * (w as i32 / f), llm])?);
            cur += h * w;
        }
        let refs: Vec<&Array> = outs.iter().collect();
        concatenate_axis(&refs, 0).map_err(Error::from)
    }

    /// `num_channels * patch_size²` — the flattened patch length the forward expects.
    pub fn patch_len(&self) -> i32 {
        self.num_channels * self.patch_size * self.patch_size
    }
}

/// Row-major patch coordinates `(abs_x, abs_y)` for the concatenated images: `abs_x = i % w`,
/// `abs_y = i / w` within each image.
fn abs_positions(grid: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for &(h, w) in grid {
        for i in 0..(h * w) {
            xs.push((i % w) as i32);
            ys.push((i / w) as i32);
        }
    }
    (xs, ys)
}

/// Interleaved 1D RoPE on `x` `[N, part]` at integer `positions` (length N), base `theta`. Pairs
/// `(x[2j], x[2j+1])` rotate by `positions * theta^(-2j/part)`. Mirrors `apply_rotary_emb_1d`.
fn rope_1d_interleaved(x: &Array, positions: &[i32], theta: f32) -> Result<Array> {
    let sh = x.shape();
    let (n, part) = (sh[0], sh[1]);
    let half = part / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| 1.0f32 / theta.powf((2 * j) as f32 / part as f32))
        .collect();
    let pos: Vec<f32> = positions.iter().map(|&p| p as f32).collect();
    let pos = Array::from_slice(&pos, &[n, 1]);
    let inv = Array::from_slice(&inv_freq, &[1, half]);
    let freqs = matmul(&pos, &inv)?; // [N, half]
    let cos = freqs.cos()?.reshape(&[n, half, 1])?;
    let sin = freqs.sin()?.reshape(&[n, half, 1])?;

    let xr = x.reshape(&[n, half, 2])?;
    let parts = split(&xr, 2, 2)?; // x1 = [...,0], x2 = [...,1], each [N, half, 1]
    let (x1, x2) = (&parts[0], &parts[1]);
    let rot1 = subtract(&multiply(x1, &cos)?, &multiply(x2, &sin)?)?;
    let rot2 = add(&multiply(x1, &sin)?, &multiply(x2, &cos)?)?;
    let out = concatenate_axis(&[&rot1, &rot2], 2)?; // [N, half, 2] interleaved
    Ok(out.reshape(&[n, part])?)
}
