//! Dual-stream MMDiT block. Port of the fork's `QwenTransformerBlock`. Each stream (image, text)
//! gets two AdaLN modulations from the timestep embedding — `mod1` around attention, `mod2` around
//! the feed-forward — with gated residuals; attention itself is joint over `[txt, img]`.

use mlx_rs::error::Exception;
use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{add, multiply, split};
use mlx_rs::transforms::compile::compile;
use mlx_rs::Array;

use mlx_gen::adapters::{prefixed_paths, AdaptableHost, AdaptableLinear};
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{compile_glue, join, linear_from, FeedForward, QwenJointAttention};

const LN_EPS: f32 = 1e-6;

pub struct QwenTransformerBlock {
    img_mod: AdaptableLinear,
    txt_mod: AdaptableLinear,
    attn: QwenJointAttention,
    img_ff: FeedForward,
    txt_ff: FeedForward,
}

impl AdaptableHost for QwenTransformerBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // The fork's `QwenLoRAMapping` targets the joint attention + the two stream MLPs (no
        // `*_mod` targets). Routes the trained-file naming (`attn.*`, `{img,txt}_mlp.*`).
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_mut(rest),
            ["img_mlp", rest @ ..] => self.img_ff.adaptable_mut(rest),
            ["txt_mlp", rest @ ..] => self.txt_ff.adaptable_mut(rest),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = prefixed_paths("attn", &self.attn);
        out.extend(prefixed_paths("img_mlp", &self.img_ff));
        out.extend(prefixed_paths("txt_mlp", &self.txt_ff));
        out
    }
}

impl QwenTransformerBlock {
    pub fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            img_mod: linear_from(w, &join(prefix, "img_mod_linear"), true)?,
            txt_mod: linear_from(w, &join(prefix, "txt_mod_linear"), true)?,
            attn: QwenJointAttention::from_weights(w, &join(prefix, "attn"), num_heads, head_dim)?,
            img_ff: FeedForward::from_weights(w, &join(prefix, "img_ff"))?,
            txt_ff: FeedForward::from_weights(w, &join(prefix, "txt_ff"))?,
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.img_mod.quantize(bits, None)?;
        self.txt_mod.quantize(bits, None)?;
        self.attn.quantize(bits)?;
        self.img_ff.quantize(bits)?;
        self.txt_ff.quantize(bits)?;
        Ok(())
    }

    /// Returns `(encoder_hidden_states, hidden_states)` (text, image) — matching the fork's order.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Array,         // image [B, img_seq, dim]
        encoder_hidden_states: &Array, // text  [B, txt_seq, dim]
        text_embeddings: &Array,       // [B, dim]
        img_cos: &Array,
        img_sin: &Array,
        txt_cos: &Array,
        txt_sin: &Array,
        mask: Option<&Array>,
    ) -> Result<(Array, Array)> {
        // Both modulation projections take the same SiLU'd timestep embedding — compute it once.
        let act = silu(text_embeddings)?;
        let img_mod = self.img_mod.forward(&act)?;
        let txt_mod = self.txt_mod.forward(&act)?;
        let img_mod = split(&img_mod, 2, 1)?; // [mod1, mod2], each [B, 3*dim]
        let txt_mod = split(&txt_mod, 2, 1)?;

        let (img_modulated, img_gate1) =
            modulate(&layer_norm(hidden_states, None, None, LN_EPS)?, &img_mod[0])?;
        let (txt_modulated, txt_gate1) = modulate(
            &layer_norm(encoder_hidden_states, None, None, LN_EPS)?,
            &txt_mod[0],
        )?;

        let (img_attn, txt_attn) = self.attn.forward(
            &img_modulated,
            &txt_modulated,
            img_cos,
            img_sin,
            txt_cos,
            txt_sin,
            mask,
        )?;

        let hidden_states = gated(hidden_states, &img_gate1, &img_attn)?;
        let encoder_hidden_states = gated(encoder_hidden_states, &txt_gate1, &txt_attn)?;

        let (img_mod2, img_gate2) = modulate(
            &layer_norm(&hidden_states, None, None, LN_EPS)?,
            &img_mod[1],
        )?;
        let img_ff = self.img_ff.forward(&img_mod2)?;
        let hidden_states = gated(&hidden_states, &img_gate2, &img_ff)?;

        let (txt_mod2, txt_gate2) = modulate(
            &layer_norm(&encoder_hidden_states, None, None, LN_EPS)?,
            &txt_mod[1],
        )?;
        let txt_ff = self.txt_ff.forward(&txt_mod2)?;
        let encoder_hidden_states = gated(&encoder_hidden_states, &txt_gate2, &txt_ff)?;

        Ok((encoder_hidden_states, hidden_states))
    }
}

/// `(x·(1+scale) + shift, gate)` from a `[B, 3*dim]` modulation (shift, scale, gate). Scale/shift/
/// gate broadcast over the sequence axis. The split + expand run eagerly (a shapeless `mx.compile`
/// can't infer a split's output shapes); when the sc-2963 glue toggle is on, the big affine
/// `x·(1+scale)+shift` is fused into one kernel. Bit-identical to the eager form — the `1.0` stays an
/// f32 scalar (the existing strong-`1` behaviour: `bf16 scale + f32 1` → f32, then `·` the f32 x).
fn modulate(x: &Array, mod_params: &Array) -> Result<(Array, Array)> {
    let p = split(mod_params, 3, 1)?; // shift, scale, gate — each [B, dim]
    let shift = p[0].expand_dims(1)?; // [B, 1, dim]
    let scale = p[1].expand_dims(1)?;
    let gate = p[2].expand_dims(1)?;
    let f = |(x, sc, sh): (&Array, &Array, &Array)| -> std::result::Result<Array, Exception> {
        add(&multiply(x, &add(sc, &Array::from_slice(&[1.0f32], &[1]))?)?, sh)
    };
    let out = if compile_glue() {
        compile(f, true)((x, &scale, &shift))?
    } else {
        f((x, &scale, &shift))?
    };
    Ok((out, gate))
}

/// Gated residual `x + gate·y` — one fused kernel when the sc-2963 glue toggle is on; bit-identical
/// to the eager `add(x, gate·y)`.
fn gated(x: &Array, gate: &Array, y: &Array) -> Result<Array> {
    let f = |(x, g, y): (&Array, &Array, &Array)| -> std::result::Result<Array, Exception> {
        add(x, &multiply(g, y)?)
    };
    if compile_glue() {
        Ok(compile(f, true)((x, gate, y))?)
    } else {
        Ok(f((x, gate, y))?)
    }
}

#[cfg(test)]
mod sc2963 {
    use super::*;
    use crate::transformer::compile_test_util::{max_abs, rnd};
    use crate::transformer::set_compile_glue;
    use mlx_rs::Dtype::{Bfloat16, Float32};

    // sc-2963: the compiled adaLN affine + gated residual are bit-identical to eager (`max|Δ|=0`) at
    // the production dtypes — f32 latents, bf16 modulation (the `mod_params` come from a bf16 Linear).
    #[test]
    fn compiled_modulate_and_gated_bit_identical_to_eager() {
        let (b, s, dim) = (2i32, 16i32, 128i32);
        // modulate: f32 hidden, bf16 mod_params [B, 3·dim].
        let x = rnd(&[b, s, dim], Float32);
        let mod_params = rnd(&[b, 3 * dim], Bfloat16);
        set_compile_glue(false);
        let (eo, eg) = modulate(&x, &mod_params).unwrap();
        set_compile_glue(true);
        let (co, cg) = modulate(&x, &mod_params).unwrap();
        set_compile_glue(false);
        assert_eq!(max_abs(&co, &eo), 0.0, "modulate out");
        assert_eq!(max_abs(&cg, &eg), 0.0, "modulate gate");

        // gated: f32 x + (bf16 gate · f32 y).
        let gate = rnd(&[b, 1, dim], Bfloat16);
        let y = rnd(&[b, s, dim], Float32);
        set_compile_glue(false);
        let e = gated(&x, &gate, &y).unwrap();
        set_compile_glue(true);
        let c = gated(&x, &gate, &y).unwrap();
        set_compile_glue(false);
        assert_eq!(max_abs(&c, &e), 0.0, "gated");
    }
}
