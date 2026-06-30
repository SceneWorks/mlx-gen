//! Z-Image SwiGLU feed-forward: `w2(silu(w1(x)) * w3(x))`. Port of the Python fork's
//! `models/z_image/.../feed_forward.py`. `w1`/`w2`/`w3` are adapter hosts (LoRA/LoKr targets).

use mlx_rs::{
    error::Exception,
    ops::{multiply, sigmoid},
    transforms::compile::compile,
    Array,
};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

#[derive(Clone)]
pub struct FeedForward {
    pub w1: AdaptableLinear,
    pub w2: AdaptableLinear,
    pub w3: AdaptableLinear,
}

impl FeedForward {
    /// Load the three projections (no bias) from `{prefix}.w{1,2,3}.weight`.
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        // Packed-detect (sc-8670): the three SwiGLU projections load packed from a pre-quantized
        // snapshot or dense otherwise. No biases.
        Ok(Self {
            w1: crate::quant::lin(w, &format!("{prefix}.w1"), false)?,
            w2: crate::quant::lin(w, &format!("{prefix}.w2"), false)?,
            w3: crate::quant::lin(w, &format!("{prefix}.w3"), false)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let h1 = self.w1.forward(x)?;
        let h3 = self.w3.forward(x)?;
        self.w2.forward(&swiglu(&h1, &h3)?)
    }

    /// Quantize the three projections to Q4/Q8 (group_size 64) — the fork's `nn.quantize` hits
    /// every Linear in the SwiGLU FFN.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for lin in [&mut self.w1, &mut self.w2, &mut self.w3] {
            lin.quantize(bits, None)?;
        }
        Ok(())
    }

    /// Cast the three projections to `dtype` (sc-4887 bf16 training).
    pub fn cast_weights(&mut self, dtype: mlx_rs::Dtype) -> Result<()> {
        for lin in [&mut self.w1, &mut self.w2, &mut self.w3] {
            lin.cast_weights(dtype)?;
        }
        Ok(())
    }
}

/// SwiGLU gate `silu(h1)·h3` = `(h1·sigmoid(h1))·h3`. The `w1`/`w3` GEMMs run eagerly; this fusable
/// arithmetic is compiled into one kernel when the sc-2963 glue toggle is on. Dtype-preserving and
/// bit-identical to the eager `multiply(silu, h3)` — the mixed-precision flow (sc-2720) is untouched.
fn swiglu(h1: &Array, h3: &Array) -> Result<Array> {
    let f = |(h1, h3): (&Array, &Array)| -> std::result::Result<Array, Exception> {
        multiply(&multiply(h1, &sigmoid(h1)?)?, h3)
    };
    if crate::compile_glue() {
        Ok(compile(f, true)((h1, h3))?)
    } else {
        Ok(f((h1, h3))?)
    }
}

impl AdaptableHost for FeedForward {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["w1"] => Some(&mut self.w1),
            ["w2"] => Some(&mut self.w2),
            ["w3"] => Some(&mut self.w3),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["w1", "w2", "w3"].into_iter().map(String::from).collect()
    }
}

#[cfg(test)]
mod sc2963 {
    use super::*;
    use mlx_rs::{random, Dtype};

    // sc-2720 guard: the compiled SwiGLU must preserve the input dtype (bf16 stays bf16 — a silent
    // f32 promotion would break the mixed-precision flow) AND be bit-identical to the eager form.
    #[test]
    fn compiled_swiglu_preserves_bf16_and_matches_eager() {
        let k = random::key(0).unwrap();
        let mk = |dt: Dtype| {
            random::normal::<f32>(&[2, 16, 64], None, None, Some(&k))
                .unwrap()
                .as_dtype(dt)
                .unwrap()
        };
        for dt in [Dtype::Float32, Dtype::Bfloat16] {
            let (h1, h3) = (mk(dt), mk(dt));
            crate::set_compile_glue(false);
            let e = swiglu(&h1, &h3).unwrap();
            crate::set_compile_glue(true);
            let c = swiglu(&h1, &h3).unwrap();
            crate::set_compile_glue(false);
            assert_eq!(c.dtype(), dt, "swiglu dtype {dt:?}");
            assert_eq!(e.dtype(), dt, "eager swiglu dtype {dt:?}");
            let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(&c, &e).unwrap()).unwrap();
            let m = mlx_rs::ops::max(&d, None)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap()
                .item::<f32>();
            assert_eq!(m, 0.0, "swiglu compiled vs eager {dt:?}");
        }
    }
}
