//! Z-Image final layer. Port of `FinalLayer`: affine-free LayerNorm scaled by
//! `1 + adaLN(silu(t_emb))`, then a Linear projection back to patch space.

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{add, multiply};
use mlx_rs::Array;

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

const LN_EPS: f32 = 1e-6;

pub struct FinalLayer {
    linear: AdaptableLinear,
    ada: AdaptableLinear,
}

impl FinalLayer {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        // Packed-detect (sc-8670): both Linears load packed from a pre-quantized snapshot or dense
        // otherwise; both carry a bias.
        let lin = |name: &str| crate::quant::lin(w, &format!("{prefix}.{name}"), true);
        Ok(Self {
            linear: lin("linear")?,
            ada: lin("adaLN_modulation.0")?,
        })
    }

    /// Quantize both Linears to Q4/Q8 (group_size 64) — the patch-space projection and the
    /// adaLN modulation; both are `nn.Linear` in the fork.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for lin in [&mut self.linear, &mut self.ada] {
            lin.quantize(bits, None)?;
        }
        Ok(())
    }

    /// Cast both Linears to `dtype` (sc-4887 bf16 training).
    pub fn cast_weights(&mut self, dtype: mlx_rs::Dtype) -> Result<()> {
        for lin in [&mut self.linear, &mut self.ada] {
            lin.cast_weights(dtype)?;
        }
        Ok(())
    }

    /// `x`: `(B, S, H)`, `c` (timestep emb): `(B, min(H,256))` → `(B, S, out_channels)`.
    pub fn forward(&self, x: &Array, c: &Array) -> Result<Array> {
        let ada_out = self.ada.forward(&silu(c)?)?;
        // Dtype-following `1` (sc-4887): a hard f32 scalar would promote the modulated final
        // projection back to f32 under the bf16 training cast. No-op when the path is f32.
        let one = Array::from_slice(&[1.0f32], &[1]).as_dtype(ada_out.dtype())?;
        let scale = add(&ada_out, &one)?;
        let scale = scale.expand_dims(1)?; // (B, 1, H)
        let normed = layer_norm(x, None, None, LN_EPS)?; // affine=False
        self.linear.forward(&multiply(&normed, &scale)?)
    }
}

impl AdaptableHost for FinalLayer {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // Trained-file naming (fork mapping): `all_final_layer.{p}-{pf}.linear` and, for the adaLN
        // modulation, Sequential index **1** (SiLU at 0) — unlike the transformer blocks whose
        // adaLN file key is index 0. The base checkpoint stores this Linear at `adaLN_modulation.0`,
        // hence the Rust field name, but the adapter file addresses it as `.1`.
        match path {
            ["linear"] => Some(&mut self.linear),
            ["adaLN_modulation", "1"] => Some(&mut self.ada),
            _ => None,
        }
    }
}
