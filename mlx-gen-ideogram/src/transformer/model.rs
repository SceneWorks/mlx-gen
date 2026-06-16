//! The full Ideogram 4 DiT: token composition (`[text ; image]`), scalar-`t` AdaLN conditioning,
//! 34 blocks, and the affine-less final layer. Port of `Ideogram4Transformer.forward`.
//!
//! Token roles (`indicator`): `LLM_TOKEN_INDICATOR = 3` (text), `OUTPUT_IMAGE_INDICATOR = 2`
//! (image). Text positions carry the projected Qwen3-VL features (`llm_cond_proj`); image
//! positions carry the patchified noise latents (`input_proj`). Both streams live in one sequence,
//! mixed every block by full (segment-masked) attention + interleaved 3D MRoPE.

use mlx_rs::fast::{layer_norm, rms_norm};
use mlx_rs::ops::{add, concatenate_axis, cos as mcos, multiply, sin as msin};
use mlx_rs::Array;

use mlx_gen::array::host_i32;
use mlx_gen::nn::{silu, TokenEmbedding};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::block::Ideogram4Block;
use super::mrope::Ideogram4MRoPE;
use super::{join, lin};
use crate::config::Ideogram4DitConfig;

/// Token role constants (upstream `ideogram4.constants`).
const OUTPUT_IMAGE_INDICATOR: i32 = 2;
const LLM_TOKEN_INDICATOR: i32 = 3;

pub struct Ideogram4Transformer {
    input_proj: mlx_gen::adapters::AdaptableLinear,
    llm_cond_norm: Array,
    llm_cond_proj: mlx_gen::adapters::AdaptableLinear,
    t_mlp_in: mlx_gen::adapters::AdaptableLinear,
    t_mlp_out: mlx_gen::adapters::AdaptableLinear,
    adaln_proj: mlx_gen::adapters::AdaptableLinear,
    embed_image_indicator: TokenEmbedding,
    rotary_emb: Ideogram4MRoPE,
    layers: Vec<Ideogram4Block>,
    final_norm_eps: f32,
    final_adaln: mlx_gen::adapters::AdaptableLinear,
    final_linear: mlx_gen::adapters::AdaptableLinear,
    /// Sinusoidal frequencies for the `t` embedding (`[emb_dim/2]`).
    t_freqs: Array,
}

impl Ideogram4Transformer {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Ideogram4DitConfig) -> Result<Self> {
        let p = |n: &str| join(prefix, n);
        let head_dim = cfg.emb_dim / cfg.num_heads;
        let mut layers = Vec::with_capacity(cfg.num_layers as usize);
        for i in 0..cfg.num_layers {
            layers.push(Ideogram4Block::from_weights(
                w,
                &p(&format!("layers.{i}")),
                cfg.num_heads,
                head_dim,
                cfg.norm_eps,
            )?);
        }
        // Sinusoidal freqs: half = emb_dim/2, freq = log(1e4)/(half-1), f[d] = exp(-freq·d).
        let half = (cfg.emb_dim / 2) as usize;
        let lf = (1e4f32).ln() / (half as f32 - 1.0);
        let t_freqs: Vec<f32> = (0..half).map(|d| (-lf * d as f32).exp()).collect();
        Ok(Self {
            input_proj: lin(w, &p("input_proj"), true)?,
            llm_cond_norm: w.require(&p("llm_cond_norm.weight"))?.clone(),
            llm_cond_proj: lin(w, &p("llm_cond_proj"), true)?,
            t_mlp_in: lin(w, &p("t_embedding.mlp_in"), true)?,
            t_mlp_out: lin(w, &p("t_embedding.mlp_out"), true)?,
            adaln_proj: lin(w, &p("adaln_proj"), true)?,
            embed_image_indicator: TokenEmbedding::Dense(
                w.require(&p("embed_image_indicator.weight"))?.clone(),
            ),
            rotary_emb: Ideogram4MRoPE::new(head_dim, cfg.rope_theta, cfg.mrope_section),
            layers,
            final_norm_eps: 1e-6,
            final_adaln: lin(w, &p("final_layer.adaln_modulation"), true)?,
            final_linear: lin(w, &p("final_layer.linear"), true)?,
            t_freqs: Array::from_slice(&t_freqs, &[1, half as i32]),
        })
    }

    /// Sinusoidal scalar-`t` embedding → MLP. `t`: `[B]` in `[0,1]` → `[B, emb_dim]`.
    fn t_embedding(&self, t: &Array) -> Result<Array> {
        let scaled = multiply(&t.as_dtype(mlx_rs::Dtype::Float32)?, Array::from_f32(1e4))?;
        let emb = multiply(&scaled.expand_dims(1)?, &self.t_freqs)?; // [B, half]
        let emb = concatenate_axis(&[&msin(&emb)?, &mcos(&emb)?], 1)?; // [B, emb_dim]
        let h = silu(&self.t_mlp_in.forward(&emb)?)?;
        self.t_mlp_out.forward(&h)
    }

    /// Velocity prediction `[B, L, in_channels]` (f32). Inputs follow the upstream packing:
    /// `llm_features [B,L,llm_dim]`, `x [B,L,in_ch]`, `t [B]`, `position_ids [B,L,3]`,
    /// `segment_ids [B,L]`, `indicator [B,L]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        llm_features: &Array,
        x: &Array,
        t: &Array,
        position_ids: &Array,
        segment_ids: &Array,
        indicator: &Array,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, l) = (sh[0], sh[1]);
        let (llm_mask, img_mask, img_idx) = role_tensors(indicator, b, l)?;

        let llm_features = multiply(llm_features, &llm_mask)?;
        let x = multiply(x, &img_mask)?;
        let x = multiply(&self.input_proj.forward(&x)?, &img_mask)?;

        let t_cond = self.t_embedding(t)?.expand_dims(1)?; // [B,1,emb]
        let adaln_input = silu(&self.adaln_proj.forward(&t_cond)?)?; // [B,1,adaln]

        let llm = rms_norm(&llm_features, &self.llm_cond_norm, 1e-6)?;
        let llm = multiply(&self.llm_cond_proj.forward(&llm)?, &llm_mask)?;

        let mut h = add(&x, &llm)?;
        h = add(&h, &self.embed_image_indicator.forward(&img_idx)?)?;

        let (cos, sin) = self.rotary_emb.forward(position_ids)?;
        let mask = segment_mask(segment_ids, b, l)?;

        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, &mask, &adaln_input)?;
        }

        // Final layer: scale = 1 + adaln(silu(c)); linear(layernorm_no_affine(h) · scale).
        let scale = add(
            &self.final_adaln.forward(&silu(&adaln_input)?)?,
            Array::from_f32(1.0),
        )?;
        let normed = layer_norm(&h, None, None, self.final_norm_eps)?;
        let out = self.final_linear.forward(&multiply(&normed, &scale)?)?;
        Ok(out.as_dtype(mlx_rs::Dtype::Float32)?)
    }
}

/// From `indicator [B,L]`: `(llm_mask [B,L,1] f32, img_mask [B,L,1] f32, img_idx [B,L] i32)`.
/// `img_idx` = 1 at image tokens, 0 elsewhere (the `embed_image_indicator` lookup index).
fn role_tensors(indicator: &Array, b: i32, l: i32) -> Result<(Array, Array, Array)> {
    let ind = host_i32(indicator)?;
    let n = (b * l) as usize;
    let mut llm = vec![0f32; n];
    let mut img = vec![0f32; n];
    let mut idx = vec![0i32; n];
    for (p, &v) in ind.iter().enumerate().take(n) {
        if v == LLM_TOKEN_INDICATOR {
            llm[p] = 1.0;
        }
        if v == OUTPUT_IMAGE_INDICATOR {
            img[p] = 1.0;
            idx[p] = 1;
        }
    }
    Ok((
        Array::from_slice(&llm, &[b, l, 1]),
        Array::from_slice(&img, &[b, l, 1]),
        Array::from_slice(&idx, &[b, l]),
    ))
}

/// Additive attention mask `[B, 1, L, L]`: `0` where two tokens share a `segment_id`, `-inf`
/// otherwise (full bidirectional attention within a packed sample — not causal).
fn segment_mask(segment_ids: &Array, b: i32, l: i32) -> Result<Array> {
    let seg = host_i32(segment_ids)?;
    let (bu, lu) = (b as usize, l as usize);
    let mut data = vec![0f32; bu * lu * lu];
    for bi in 0..bu {
        for i in 0..lu {
            for j in 0..lu {
                if seg[bi * lu + i] != seg[bi * lu + j] {
                    data[(bi * lu + i) * lu + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Ok(Array::from_slice(&data, &[b, 1, l, l]))
}
