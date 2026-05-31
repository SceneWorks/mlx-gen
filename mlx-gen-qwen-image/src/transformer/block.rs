//! Dual-stream MMDiT block. Port of the fork's `QwenTransformerBlock`. Each stream (image, text)
//! gets two AdaLN modulations from the timestep embedding — `mod1` around attention, `mod2` around
//! the feed-forward — with gated residuals; attention itself is joint over `[txt, img]`.

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{add, multiply, split};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, linear_from, FeedForward, QwenJointAttention};

const LN_EPS: f32 = 1e-6;

pub struct QwenTransformerBlock {
    img_mod: AdaptableLinear,
    txt_mod: AdaptableLinear,
    attn: QwenJointAttention,
    img_ff: FeedForward,
    txt_ff: FeedForward,
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
        let img_mod = self.img_mod.forward(&silu(text_embeddings)?)?;
        let txt_mod = self.txt_mod.forward(&silu(text_embeddings)?)?;
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

        let hidden_states = add(hidden_states, &multiply(&img_gate1, &img_attn)?)?;
        let encoder_hidden_states = add(encoder_hidden_states, &multiply(&txt_gate1, &txt_attn)?)?;

        let (img_mod2, img_gate2) = modulate(
            &layer_norm(&hidden_states, None, None, LN_EPS)?,
            &img_mod[1],
        )?;
        let hidden_states = add(
            &hidden_states,
            &multiply(&img_gate2, &self.img_ff.forward(&img_mod2)?)?,
        )?;

        let (txt_mod2, txt_gate2) = modulate(
            &layer_norm(&encoder_hidden_states, None, None, LN_EPS)?,
            &txt_mod[1],
        )?;
        let encoder_hidden_states = add(
            &encoder_hidden_states,
            &multiply(&txt_gate2, &self.txt_ff.forward(&txt_mod2)?)?,
        )?;

        Ok((encoder_hidden_states, hidden_states))
    }
}

/// `(x·(1+scale) + shift, gate)` from a `[B, 3*dim]` modulation (shift, scale, gate). Scale/shift/
/// gate broadcast over the sequence axis.
fn modulate(x: &Array, mod_params: &Array) -> Result<(Array, Array)> {
    let p = split(mod_params, 3, 1)?; // shift, scale, gate — each [B, dim]
    let shift = p[0].expand_dims(1)?; // [B, 1, dim]
    let scale = add(&p[1], Array::from_slice(&[1.0f32], &[1]))?.expand_dims(1)?;
    let gate = p[2].expand_dims(1)?;
    let out = add(&multiply(x, &scale)?, &shift)?;
    Ok((out, gate))
}
