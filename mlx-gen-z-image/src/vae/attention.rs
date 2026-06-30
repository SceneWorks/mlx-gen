//! VAE mid-block spatial self-attention: GroupNorm → QKV → single-head SDPA over the H·W
//! tokens → out projection, with a residual. NCHW I/O.

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::group_norm;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-6;

pub struct VaeAttention {
    gn_w: Array,
    gn_b: Array,
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    out: AdaptableLinear,
}

impl VaeAttention {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let g = |s: &str| w.require(&format!("{prefix}.{s}")).cloned();
        // Packed-detect (sc-8670): the mid-block attention QKV/out projections load packed from a
        // pre-quantized snapshot or dense otherwise; all carry a bias. They are the only quantizable
        // leaves in the otherwise-conv VAE. The pre-quantized `{base}.{weight,scales,biases,bias}`
        // pass through the loader's diffusers→internal VAE remap untouched (not conv weights).
        let lin = |name: &str| crate::quant::lin(w, &format!("{prefix}.{name}"), true);
        Ok(Self {
            gn_w: g("group_norm.weight")?,
            gn_b: g("group_norm.bias")?,
            q: lin("to_q")?,
            k: lin("to_k")?,
            v: lin("to_v")?,
            out: lin("to_out.0")?,
        })
    }

    /// Quantize the QKV + output projections to Q4/Q8 (group_size 64) — the fork's `nn.quantize`
    /// over the VAE hits these Linears (the only quantizable leaves in the otherwise-conv VAE).
    /// GroupNorm scales stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for lin in [&mut self.q, &mut self.k, &mut self.v, &mut self.out] {
            lin.quantize(bits, None)?;
        }
        Ok(())
    }

    pub fn forward(&self, x_nchw: &Array) -> Result<Array> {
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let sh = x.shape();
        let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);

        let normed = group_norm(&x, &self.gn_w, &self.gn_b, GN_GROUPS, GN_EPS)?;
        // (B,H,W,C) -> (B, H*W, 1, C) -> (B, 1, H*W, C) [single head, head_dim = C].
        let proj = |lin: &AdaptableLinear| -> Result<Array> {
            Ok(lin
                .forward(&normed)?
                .reshape(&[b, h * w, 1, c])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = proj(&self.q)?;
        let k = proj(&self.k)?;
        let v = proj(&self.v)?;

        let scale = (c as f32).powf(-0.5);
        // trailing `None` is the MLX ≥0.30 `sinks` arg (no attention sinks).
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, h, w, c])?;
        let o = self.out.forward(&o)?;

        Ok(add(&x, &o)?.transpose_axes(&[0, 3, 1, 2])?) // residual, back to NCHW
    }
}
