//! Z-Image attention: QKV projections → optional QK-RMSNorm → 3D RoPE → SDPA → out
//! projection. Port of the Python fork's `models/z_image/.../attention.py`, made
//! dimension-parametric. `to_q/to_k/to_v/to_out` are adapter hosts (LoRA/LoKr targets).
//!
//! Numeric parity proven stage-by-stage in the sc-2338 spike (tolerance 1e-2 — MLX runs
//! fp32 matmul in reduced precision on Metal). The DiT is **intentionally maskless** (SDPA
//! `mask=None`, final by design): the fork builds all-ones masks, and padded positions are
//! handled by the learned pad-token embeddings rather than attention masking — so there is no
//! mask to wire (see `transformer.rs`).

use mlx_rs::{
    error::Exception,
    fast::{rms_norm, scaled_dot_product_attention},
    ops::{add, multiply, split, stack_axis, subtract},
    transforms::checkpoint,
    transforms::compile::compile,
    Array, Dtype,
};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

#[derive(Clone)]
pub struct ZImageAttention {
    pub to_q: AdaptableLinear,
    pub to_k: AdaptableLinear,
    pub to_v: AdaptableLinear,
    pub to_out: AdaptableLinear,
    norm_q: Option<Array>,
    norm_k: Option<Array>,
    n_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
    /// sc-4886 — run the SDPA segment inside an `mlx::checkpoint` so its backward recomputes the
    /// attention instead of retaining the `[heads, s, s]` probability matrix (the grad through
    /// `fast::scaled_dot_product_attention` decomposes to naive attention — MLX has no fused SDPA
    /// backward — and that one retained seq² array per block is ~half the dense training working
    /// set). Numerically identical (same math, recomputed); inference never sets it (default off,
    /// zero cost), the trainer enables it unconditionally.
    ckpt_sdpa: bool,
}

impl ZImageAttention {
    /// Load from `{prefix}.{to_q,to_k,to_v,to_out.0,norm_q,norm_k}.weight`. QK-norm weights
    /// are optional (present iff `qk_norm=True`, which Z-Image-turbo uses).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        dim: i32,
        n_heads: i32,
        eps: f32,
    ) -> Result<Self> {
        let head_dim = dim / n_heads;
        Ok(Self {
            to_q: AdaptableLinear::dense(
                w.require(&format!("{prefix}.to_q.weight"))?.clone(),
                None,
            ),
            to_k: AdaptableLinear::dense(
                w.require(&format!("{prefix}.to_k.weight"))?.clone(),
                None,
            ),
            to_v: AdaptableLinear::dense(
                w.require(&format!("{prefix}.to_v.weight"))?.clone(),
                None,
            ),
            to_out: AdaptableLinear::dense(
                w.require(&format!("{prefix}.to_out.0.weight"))?.clone(),
                None,
            ),
            norm_q: w.get(&format!("{prefix}.norm_q.weight")).cloned(),
            norm_k: w.get(&format!("{prefix}.norm_k.weight")).cloned(),
            n_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps,
            ckpt_sdpa: false,
        })
    }

    /// Toggle SDPA-segment gradient checkpointing (sc-4886). Training-only knob — see `ckpt_sdpa`.
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.ckpt_sdpa = on;
    }

    /// Cast the projection weights + QK-norm scales to `dtype` (sc-4887 bf16 training). Quantized
    /// projections are left untouched (see [`AdaptableLinear::cast_weights`]).
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        for lin in [
            &mut self.to_q,
            &mut self.to_k,
            &mut self.to_v,
            &mut self.to_out,
        ] {
            lin.cast_weights(dtype)?;
        }
        for norm in [&mut self.norm_q, &mut self.norm_k].into_iter().flatten() {
            if norm.dtype() != dtype {
                *norm = norm.as_dtype(dtype)?;
            }
        }
        Ok(())
    }

    pub fn forward(&self, x: &Array, freqs_cis: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let dim = self.n_heads * self.head_dim;

        let mut q = self
            .to_q
            .forward(x)?
            .reshape(&[b, s, self.n_heads, self.head_dim])?;
        let mut k = self
            .to_k
            .forward(x)?
            .reshape(&[b, s, self.n_heads, self.head_dim])?;
        let v = self
            .to_v
            .forward(x)?
            .reshape(&[b, s, self.n_heads, self.head_dim])?;

        if let Some(nq) = &self.norm_q {
            q = rms_norm(&q, nq, self.eps)?;
        }
        if let Some(nk) = &self.norm_k {
            k = rms_norm(&k, nk, self.eps)?;
        }

        q = self.apply_rope(&q, freqs_cis)?;
        k = self.apply_rope(&k, freqs_cis)?;

        // (b, s, h, hd) -> (b, h, s, hd)
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // 6th arg is `sinks` (MLX ≥0.30 attention-sinks); `None` = standard attention.
        let o = if self.ckpt_sdpa {
            // sc-4886: checkpoint just the SDPA. q/k/v are the threaded inputs (grads to the
            // QKV projections — and their LoRA — flow through them); only the f32 scale is
            // captured. The backward recomputes the decomposed attention for THIS layer alone,
            // so the seq² probability matrix is a per-layer transient, never 30× retained.
            let scale = self.scale;
            let mut seg = checkpoint(move |inp: &[Array]| -> mlx_rs::error::Result<Vec<Array>> {
                Ok(vec![scaled_dot_product_attention(
                    &inp[0], &inp[1], &inp[2], scale, None, None,
                )?])
            });
            seg(&[q, k, v])?
                .into_iter()
                .next()
                .ok_or_else(|| Error::Msg("z-image: checkpoint SDPA produced no output".into()))?
        } else {
            scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?
        };
        let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, dim])?;
        self.to_out.forward(&o)
    }

    /// Quantize the QKV + output projections to Q4/Q8 (group_size 64). QK-norm weights are
    /// RMSNorm scales (not Linears), so they stay dense — matching the fork's `nn.quantize`.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for lin in [
            &mut self.to_q,
            &mut self.to_k,
            &mut self.to_v,
            &mut self.to_out,
        ] {
            lin.quantize(bits, None)?;
        }
        Ok(())
    }

    /// Port of `ZImageAttention._apply_rotary_emb`. `x`:(b,s,h,hd), `fc`:(s,hd/2,2).
    fn apply_rope(&self, x: &Array, fc: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s, h, hd) = (sh[0], sh[1], sh[2], sh[3]);
        let half = hd / 2;
        let x5 = x.reshape(&[b, s, h, half, 2])?;
        let xp = split(&x5, 2, 4)?;
        let xr = xp[0].reshape(&[b, s, h, half])?;
        let xi = xp[1].reshape(&[b, s, h, half])?;
        let fc5 = fc.reshape(&[1, s, 1, half, 2])?;
        let fp = split(&fc5, 2, 4)?;
        let cos = fp[0].reshape(&[1, s, 1, half])?;
        let sin = fp[1].reshape(&[1, s, 1, half])?;
        let (out_r, out_i) = rope_rotate(&xr, &xi, &cos, &sin)?;
        Ok(stack_axis(&[out_r, out_i], 4)?.reshape(&[b, s, h, hd])?)
    }
}

/// The complex RoPE rotation `(xr + xi·i)·(cos + sin·i)` → `(out_r, out_i)`. Fused into one kernel
/// when the sc-2963 glue toggle is on (vs 6 eager ops, applied to q and k every block);
/// dtype-preserving, bit-identical to the eager form.
fn rope_rotate(xr: &Array, xi: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
    let f = |inp: &[Array]| -> std::result::Result<Vec<Array>, Exception> {
        let (xr, xi, cos, sin) = (&inp[0], &inp[1], &inp[2], &inp[3]);
        let out_r = subtract(&multiply(xr, cos)?, &multiply(xi, sin)?)?;
        let out_i = add(&multiply(xr, sin)?, &multiply(xi, cos)?)?;
        Ok(vec![out_r, out_i])
    };
    let args = [xr.clone(), xi.clone(), cos.clone(), sin.clone()];
    let out = if crate::compile_glue() {
        compile(f, true)(&args)?
    } else {
        f(&args)?
    };
    let [out_r, out_i]: [Array; 2] = out.try_into().map_err(|v: Vec<Array>| {
        Error::Msg(format!(
            "z-image rope_rotate: expected 2 rotated outputs, got {}",
            v.len()
        ))
    })?;
    Ok((out_r, out_i))
}

impl AdaptableHost for ZImageAttention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["to_q"] => Some(&mut self.to_q),
            ["to_k"] => Some(&mut self.to_k),
            ["to_v"] => Some(&mut self.to_v),
            ["to_out", "0"] => Some(&mut self.to_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["to_q", "to_k", "to_v", "to_out.0"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}
