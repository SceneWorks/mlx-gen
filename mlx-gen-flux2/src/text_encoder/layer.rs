//! Qwen3 decoder block (pre-norm residual): `h += attn(input_ln(h))`, then
//! `h += mlp(post_ln(h))`. Port of `Qwen3VLDecoderLayer`.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, Qwen3Attention, Qwen3Mlp};
use crate::config::Flux2Quant;

pub struct Qwen3DecoderLayer {
    input_ln: Array,
    post_ln: Array,
    attn: Qwen3Attention,
    mlp: Qwen3Mlp,
    eps: f32,
}

impl Qwen3DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        eps: f32,
        qk_norm: bool,
        quant: Option<Flux2Quant>,
    ) -> Result<Self> {
        Ok(Self {
            input_ln: w.require(&join(prefix, "input_layernorm.weight"))?.clone(),
            post_ln: w
                .require(&join(prefix, "post_attention_layernorm.weight"))?
                .clone(),
            attn: Qwen3Attention::from_weights(
                w,
                &join(prefix, "self_attn"),
                num_heads,
                num_kv_heads,
                head_dim,
                eps,
                qk_norm,
                quant,
            )?,
            mlp: Qwen3Mlp::from_weights(w, &join(prefix, "mlp"), quant)?,
            eps,
        })
    }

    /// Quantize the attention + MLP linears (group_size 64); the two RMSNorms stay full precision.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.mlp.quantize(bits)?;
        Ok(())
    }

    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(x, &self.attn.forward(&normed, cos, sin, mask)?)?;
        let normed2 = rms_norm(&h, &self.post_ln, self.eps)?;
        Ok(add(&h, &self.mlp.forward(&normed2)?)?)
    }

    /// KV-cached causal decode step (caption-upsampling generate, sc-6030) — the [`forward`](Self::forward)
    /// companion that threads the per-layer `cache` through the attention's `forward_step`.
    pub(crate) fn forward_step(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        cache: &mut super::generate::Qwen3KvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(
            x,
            &self
                .attn
                .forward_step(&normed, cos, sin, cache, layer_idx)?,
        )?;
        let normed2 = rms_norm(&h, &self.post_ln, self.eps)?;
        Ok(add(&h, &self.mlp.forward(&normed2)?)?)
    }
}
