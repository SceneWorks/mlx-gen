//! Qwen2.5 self-attention: GQA (28 query / 4 kv heads), **biased** q/k/v projections + bias-less
//! o_proj, HF half-split RoPE, masked SDPA. No per-head q_norm/k_norm (that's Qwen3 / Z-Image).

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::Array;

use mlx_gen::nn::{apply_text_rope as apply_rope, linear, repeat_kv};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, matmul_t};

pub struct QwenTextAttention {
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    o_w: Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl QwenTextAttention {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> Result<Self> {
        Ok(Self {
            q_w: w.require(&join(prefix, "q_proj.weight"))?.clone(),
            q_b: w.require(&join(prefix, "q_proj.bias"))?.clone(),
            k_w: w.require(&join(prefix, "k_proj.weight"))?.clone(),
            k_b: w.require(&join(prefix, "k_proj.bias"))?.clone(),
            v_w: w.require(&join(prefix, "v_proj.weight"))?.clone(),
            v_b: w.require(&join(prefix, "v_proj.bias"))?.clone(),
            o_w: w.require(&join(prefix, "o_proj.weight"))?.clone(),
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[1, s, head_dim]`; `mask`: additive `[b,1,s,s]`.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        let q = linear(x, &self.q_w, &self.q_b)?.reshape(&[b, s, self.num_heads, self.head_dim])?;
        let k =
            linear(x, &self.k_w, &self.k_b)?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;
        let v =
            linear(x, &self.v_w, &self.v_b)?.reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // GQA: repeat each kv head `groups` times to match the query heads.
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        // [b,s,h,hd] → [b,h,s,hd]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Match the mask dtype to the query (the fork's `mask.astype(query.dtype)`), else a dtype
        // mismatch can drop the additive mask in SDPA. trailing `None` = MLX ≥0.30 `sinks` arg.
        let mask = mask.as_dtype(q.dtype())?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, &mask, None)?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, s, self.num_heads * self.head_dim])?;
        matmul_t(&o, &self.o_w)
    }
}
// F-078: the HF half-split RoPE apply + GQA `repeat_kv` were open-coded identically here and in the
// Krea TE; both now come from `mlx_gen::nn::{apply_text_rope, repeat_kv}`.
