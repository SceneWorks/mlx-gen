//! sc-5139: the Bernini planner's `MLPConnector` (`bernini/models/bernini.py`).
//!
//! Two projection heads off the planner's penultimate hidden state (3584):
//!   - **`for_gen`** (the generation branch → renderer prompt-embed width 4096):
//!     `Linear(3584→4096) → GELU → RMSNorm(4096) → Linear(4096→4096)`.
//!   - **`for_vit`** (the ViT branch → clip-diff condition width 3584):
//!     `Linear(3584→3584) → GELU → Linear(3584→3584) → RMSNorm(3584) → Linear(3584→3584)`.
//!
//! The norm is the reference's local `RMSNorm` (f32 reduction, weight-only, eps 1e-6 — distinct from
//! the Qwen2 RMSNorm but numerically the same op); GELU is exact (`nn.GELU()`). Both branch linears
//! carry bias. `gen_head_type: zerolinear` only affected training init and is a no-op at inference.

use mlx_rs::fast::rms_norm;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::gelu_exact;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

const RMS_EPS: f32 = 1e-6;

fn linear(w: &Weights, prefix: &str) -> Result<AdaptableLinear> {
    Ok(AdaptableLinear::dense(
        w.require(&format!("{prefix}.weight"))?.clone(),
        Some(w.require(&format!("{prefix}.bias"))?.clone()),
    ))
}

/// The Bernini `MLPConnector` (gen + vit branches).
pub struct MlpConnector {
    gen0: AdaptableLinear,
    gen_rms: Array,
    gen3: AdaptableLinear,
    vit0: AdaptableLinear,
    vit2: AdaptableLinear,
    vit_rms: Array,
    vit4: AdaptableLinear,
}

impl MlpConnector {
    /// Build from a converted `connector.safetensors` (`proj_gen.*` / `pred_vit.*`). `prefix` is the
    /// connector namespace (`""` for the sc-5144 layout).
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |leaf: &str| {
            if prefix.is_empty() {
                leaf.to_string()
            } else {
                format!("{prefix}.{leaf}")
            }
        };
        Ok(Self {
            gen0: linear(w, &p("proj_gen.0"))?,
            gen_rms: w.require(&p("proj_gen.2.weight"))?.clone(),
            gen3: linear(w, &p("proj_gen.3"))?,
            vit0: linear(w, &p("pred_vit.0"))?,
            vit2: linear(w, &p("pred_vit.2"))?,
            vit_rms: w.require(&p("pred_vit.3.weight"))?.clone(),
            vit4: linear(w, &p("pred_vit.4"))?,
        })
    }

    /// Generation branch → `[*, 4096]` renderer-prompt embeds.
    pub fn for_gen(&self, x: &Array) -> Result<Array> {
        let x = gelu_exact(&self.gen0.forward(x)?)?;
        let x = rms_norm(&x, &self.gen_rms, RMS_EPS)?;
        self.gen3.forward(&x)
    }

    /// ViT branch → `[*, 3584]` clip-diff condition.
    pub fn for_vit(&self, x: &Array) -> Result<Array> {
        let x = gelu_exact(&self.vit0.forward(x)?)?;
        let x = self.vit2.forward(&x)?;
        let x = rms_norm(&x, &self.vit_rms, RMS_EPS)?;
        self.vit4.forward(&x)
    }

    /// Quantize the branch linears (Q4/Q8, group 64). The two RMSNorm weights stay dense. (sc-5146)
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for lin in [
            &mut self.gen0,
            &mut self.gen3,
            &mut self.vit0,
            &mut self.vit2,
            &mut self.vit4,
        ] {
            lin.quantize(bits, None)?;
        }
        Ok(())
    }
}
