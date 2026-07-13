//! Krea 2 pose-ControlNet inference branch (sc-8465, epic 8459 S5) — the MLX twin of the candle
//! `candle-gen-krea::control` branch (sc-8460 spike / sc-8464 provider). Same recipe, same math:
//!
//! - Copy the first `N` (S0-proven `7`) of the 28 gated single-stream DiT blocks into a trainable
//!   branch. Each branch block is a full [`SingleStreamBlock`] plus a **zero-initialised** output
//!   projection (`proj_out`, `[hidden, hidden]`) — the ControlNet identity seam.
//! - Add the VAE-encoded pose latent (embedded through the frozen base `img_in`) onto the **image-token
//!   slice** of the joint sequence, run the branch, and collect each block's `proj_out(image tokens)`
//!   as a residual.
//! - Inject residual `k` into the frozen main stream **before main block `k + inject_offset`** (offset
//!   `1` → skip block 0, the degenerate overwrite site), added to the **image tokens only**, scaled by
//!   `control_scale` and clamped so `‖residual‖ ≤ τ·RMS(main image tokens)` (τ = 0.15). `control_scale`
//!   = 0 short-circuits to a bit-exact base forward.
//!
//! Parity is judged by "no quality regression vs. the candle lane / comparable PCK" (the epic's
//! parity-reframe), not bit-exact candle numerics — the branch blocks reuse the crate's own faithful
//! [`SingleStreamBlock`] forward, so the residuals match the candle branch within the same MLX-vs-candle
//! tolerance the base DiT already carries.
//!
//! The branch loads the candle overlay checkpoint (`control_step5000.safetensors`) DIRECTLY — no offline
//! convert step. The only candle↔MLX on-disk difference is the four RMSNorm scales, which candle stores
//! pre-folded as `*.weight_p1` (`= scale + 1`); [`crate::transformer::block::RmsScale`] accepts that
//! convention verbatim (alongside the base snapshot's raw `*.weight`), so every branch block loads
//! through the unmodified [`SingleStreamBlock::from_weights`] against the same file the candle lane uses.

use mlx_rs::ops::{add, concatenate_axis, divide, maximum, minimum, multiply, sqrt};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::runtime::WeightsSource;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::Krea2Config;
use crate::quant::lin;
use crate::transformer::block::SingleStreamBlock;
use crate::transformer::{split_axis1, Krea2Transformer};

pub use crate::transformer::JointPrep;

/// Residual RMS clamp τ (candle `DEFAULT_RESIDUAL_CLAMP`): cap `‖residual‖` at `τ·RMS(main image
/// tokens)` so the zero-init branch can only fade a bounded correction into the frozen base.
pub const DEFAULT_RESIDUAL_CLAMP: f32 = 0.15;

/// Injection offset (candle `DEFAULT_INJECT_OFFSET`): branch residual `i` feeds the INPUT of main block
/// `i + offset`. Offset `1` skips main block 0 (its degeneracy-preferred overwrite site; the standard
/// "feed the next block" layout). Persisted in the overlay as the [`META_INJECT_OFFSET`] tensor.
pub const DEFAULT_INJECT_OFFSET: usize = 1;

/// Default `control_scale` when a request leaves it unset (candle `DEFAULT_CONTROL_SCALE`): a comfortable
/// mid pose-lock. The worker hard-caps the exposed range at ≤ 0.85 (S0 GO verdict); the cap is NOT
/// enforced here (parity with the candle crate, which documents "the worker applies the cap").
pub const DEFAULT_CONTROL_SCALE: f32 = 0.6;

/// The S0-proven number of copied branch blocks (candle `N_CONTROL_BLOCKS`). The real count is inferred
/// from the overlay's `blocks.{i}.*` keys; this is the expected value.
pub const DEFAULT_N_CONTROL_BLOCKS: usize = 7;

/// Overlay meta tensor (`[1]` f32) carrying the trained `inject_offset` (candle `META_INJECT_OFFSET`).
const META_INJECT_OFFSET: &str = "meta.inject_offset";

/// One control-branch block: a copy of a base single-stream block plus its zero-init output projection.
struct ControlBlock {
    /// A full copy of a base gated single-stream block (loaded from the overlay's `blocks.{i}.*`).
    block: SingleStreamBlock,
    /// Zero-init `[hidden, hidden]` output projection (no bias) — the ControlNet identity seam; at
    /// step 0 (untrained) it is exactly zero, so the branch is a no-op over the frozen base.
    proj_out: AdaptableLinear,
}

/// The Krea 2 pose control branch: `N` copied single-stream blocks with zero-init output projections,
/// injected into the frozen [`Krea2Transformer`] main stream. Loaded from a converted MLX overlay.
pub struct Krea2ControlBranch {
    blocks: Vec<ControlBlock>,
    /// RMS clamp τ (`Some(0.15)` at load); `None` disables the clamp (unused in production).
    clamp_tau: Option<f32>,
    /// Branch residual `i` → main block `i + inject_offset` (read from the overlay; default 1).
    inject_offset: usize,
}

impl Krea2ControlBranch {
    /// Load the branch from a control checkpoint [`WeightsSource`] (a single `.safetensors` `File`, or a
    /// `Dir` of shards), against the base DiT `cfg` (block dims must match the frozen base).
    pub fn from_source(control: &WeightsSource, cfg: &Krea2Config) -> Result<Self> {
        let w = match control {
            WeightsSource::File(p) => Weights::from_file(p)?,
            WeightsSource::Dir(p) => Weights::from_dir(p)?,
        };
        Self::from_weights(&w, cfg)
    }

    /// Assemble the branch from an already-loaded overlay. Infers `N` from the `blocks.{i}.*` keys and
    /// builds each block through the unmodified [`SingleStreamBlock::from_weights`] plus its `proj_out`.
    /// F-075: there is NO offline convert step — the candle-gen pose overlay ships the RMSNorm scales
    /// pre-folded as `*.weight_p1` (`= scale + 1`), and `crate::transformer::block::RmsScale::from_weights`
    /// loads that verbatim (native `weight_p1` load, sc-8465), so the overlay drops straight in.
    pub fn from_weights(w: &Weights, cfg: &Krea2Config) -> Result<Self> {
        let (heads, kv, hd, hidden, eps) = (
            cfg.num_attention_heads as i32,
            cfg.num_kv_heads as i32,
            cfg.attention_head_dim as i32,
            cfg.hidden_size as i32,
            cfg.norm_eps,
        );
        let n = infer_num_blocks(w);
        if n == 0 {
            return Err(Error::Msg(
                "krea_2_turbo_control: overlay has no `blocks.{i}.*` control-branch tensors — is this \
                 the Krea 2 pose-control overlay (the candle-gen `.safetensors` with `blocks.{i}.*` + \
                 `blocks.{i}.proj_out` and pre-folded `*.weight_p1` RMSNorm scales)?"
                    .into(),
            ));
        }
        let blocks = (0..n)
            .map(|i| {
                let prefix = format!("blocks.{i}");
                Ok(ControlBlock {
                    block: SingleStreamBlock::from_weights(w, &prefix, heads, kv, hd, hidden, eps)?,
                    proj_out: lin(w, &format!("{prefix}.proj_out"), false)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            blocks,
            clamp_tau: Some(DEFAULT_RESIDUAL_CLAMP),
            inject_offset: read_inject_offset(w),
        })
    }

    /// Number of copied branch blocks (`N`).
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// The injection offset read from the overlay.
    pub fn inject_offset(&self) -> usize {
        self.inject_offset
    }

    /// Velocity prediction with the pose-control residual injected — the MLX twin of the candle
    /// `forward_with_control`. `ctrl_tokens` is the base-`img_in`-embedded pose latent
    /// ([`Krea2Transformer::embed_latent`]), precomputed once per generation (step-invariant). Called
    /// per denoise step by the pipeline.
    ///
    /// `control_scale == 0.0` short-circuits to the straight-through base forward (bit-exact base
    /// passthrough — the zero branch is never run), matching the candle guarantee.
    pub(crate) fn forward(
        &self,
        dit: &Krea2Transformer,
        latent: &Array,
        timestep: &Array,
        prep: &JointPrep,
        ctrl_tokens: &Array,
        control_scale: f32,
    ) -> Result<Array> {
        if control_scale == 0.0 {
            return dit.forward_prepared(latent, timestep, prep);
        }

        let j = dit.joint_inputs(latent, timestep, prep)?;
        let residuals = self.residuals(
            &j.combined,
            ctrl_tokens,
            j.cap_len,
            &j.tvec,
            &j.rcos,
            &j.rsin,
        )?;

        // Run the frozen 28-block stack ourselves, adding residual `k` to the image tokens BEFORE main
        // block `k + inject_offset` runs (candle's injection order; text tokens pass through untouched).
        let mut x = j.combined.clone();
        for (idx, blk) in dit.blocks().iter().enumerate() {
            if let Some(k) = self.residual_index_for_main_block(idx) {
                let parts = split_axis1(&x, j.cap_len)?;
                let txt = &parts[0];
                let img = &parts[1];
                // Scale, then RMS-clamp against the current main image slice, then cast back to the
                // stream dtype (candle scales in f64, clamps, then `to_dtype(x.dtype())`).
                let scaled = multiply(&residuals[k], scalar(control_scale))?;
                let scaled = self.apply_clamp(&scaled, img)?.as_dtype(x.dtype())?;
                let img = add(img, &scaled)?;
                x = concatenate_axis(&[txt, &img], 1)?;
            }
            x = blk.forward(&x, &j.tvec, &j.rcos, &j.rsin)?;
        }
        dit.finalize(&x, &j.t, &j)
    }

    /// Run the branch over the joint hidden state to produce one image-token residual per branch block.
    /// The pose `ctrl_tokens` are added onto the image-token slice of the branch input (candle
    /// `residuals_mode`), then each block's output image tokens are passed through its `proj_out`.
    fn residuals(
        &self,
        combined: &Array,
        ctrl_tokens: &Array,
        cap_len: i32,
        tvec: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<Vec<Array>> {
        let parts = split_axis1(combined, cap_len)?;
        let txt = &parts[0];
        let img = add(&parts[1], ctrl_tokens)?; // pose conditioning onto the image tokens
        let mut h = concatenate_axis(&[txt, &img], 1)?;

        let mut out = Vec::with_capacity(self.blocks.len());
        for cb in &self.blocks {
            h = cb.block.forward(&h, tvec, cos, sin)?;
            let h_img = split_axis1(&h, cap_len)?.swap_remove(1);
            out.push(cb.proj_out.forward(&h_img)?);
        }
        Ok(out)
    }

    /// Branch residual index injected before main block `main_block`, or `None` if that block gets no
    /// residual. Branch block `i` → main block `i + inject_offset`.
    fn residual_index_for_main_block(&self, main_block: usize) -> Option<usize> {
        residual_index(main_block, self.inject_offset, self.blocks.len())
    }

    /// Cap `‖residual‖` at `τ·RMS(main image tokens)` (candle `apply_clamp`): `res·min(1, τ·rms(main)/
    /// rms(res))`. RMS is the per-element root-mean-square over all elements, computed in f32; the
    /// clamp factor is a stop-grad scalar (inference has no grad). A zero residual (step 0) or an
    /// in-budget residual passes through unchanged (factor 1).
    fn apply_clamp(&self, res: &Array, main_img: &Array) -> Result<Array> {
        let Some(tau) = self.clamp_tau else {
            return Ok(res.clone());
        };
        let rn = rms(res)?;
        let cap = multiply(&rms(main_img)?, scalar(tau))?;
        // min(1, cap / max(rn, ε)) — ε avoids a 0/0 at step 0 (res == 0 → factor 1, res stays 0).
        let factor = minimum(scalar(1.0), &divide(&cap, &maximum(&rn, scalar(1e-20))?)?)?;
        Ok(multiply(res, &factor)?)
    }
}

/// Per-element root-mean-square `sqrt(mean(x²))` as a 0-d scalar array, reduced in f32 (candle upcasts).
fn rms(t: &Array) -> Result<Array> {
    Ok(sqrt(
        &t.as_dtype(mlx_rs::Dtype::Float32)?.square()?.mean(None)?,
    )?)
}

/// Branch residual index injected before main block `main_block` (branch `i` → main `i + inject_offset`),
/// or `None`. Extracted from [`Krea2ControlBranch::residual_index_for_main_block`] so the offset↔block
/// mapping is unit-testable without loaded weights.
fn residual_index(main_block: usize, inject_offset: usize, n: usize) -> Option<usize> {
    main_block.checked_sub(inject_offset).filter(|&i| i < n)
}

/// Highest `blocks.{i}.*` index + 1 in the overlay (the branch block count `N`), or 0 if none.
fn infer_num_blocks(w: &Weights) -> usize {
    w.keys()
        .filter_map(|k| k.strip_prefix("blocks."))
        .filter_map(|rest| rest.split('.').next())
        .filter_map(|n| n.parse::<usize>().ok())
        .max()
        .map(|max| max + 1)
        .unwrap_or(0)
}

/// Read the trained `inject_offset` from the overlay meta tensor; absent → [`DEFAULT_INJECT_OFFSET`]
/// (every trained overlay carries it — the candle trainer always writes `1`).
fn read_inject_offset(w: &Weights) -> usize {
    w.get(META_INJECT_OFFSET)
        .map(|t| t.item::<f32>().round() as usize)
        .unwrap_or(DEFAULT_INJECT_OFFSET)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_index_maps_branch_to_offset_block() {
        // inject_offset = 1, N = 7 (the S0 recipe): main block 0 gets nothing; branch 0..7 → main 1..8.
        assert_eq!(residual_index(0, 1, 7), None);
        assert_eq!(residual_index(1, 1, 7), Some(0));
        assert_eq!(residual_index(7, 1, 7), Some(6));
        assert_eq!(residual_index(8, 1, 7), None); // past the branch
        assert_eq!(residual_index(27, 1, 7), None);
        // offset 0 would inject into main block 0 (the degenerate site the recipe avoids).
        assert_eq!(residual_index(0, 0, 7), Some(0));
    }

    #[test]
    fn infer_num_blocks_reads_max_index() {
        let mut w = Weights::empty();
        w.insert("blocks.0.proj_out.weight", scalar(0.0));
        w.insert("blocks.1.attn.to_q.weight", scalar(0.0));
        w.insert("blocks.6.norm1.weight", scalar(0.0));
        w.insert(META_INJECT_OFFSET, scalar(1.0));
        assert_eq!(infer_num_blocks(&w), 7);
        assert_eq!(infer_num_blocks(&Weights::empty()), 0);
    }

    #[test]
    fn read_inject_offset_defaults_when_absent() {
        let mut w = Weights::empty();
        assert_eq!(read_inject_offset(&w), DEFAULT_INJECT_OFFSET);
        w.insert(META_INJECT_OFFSET, Array::from_slice(&[1.0f32], &[1]));
        assert_eq!(read_inject_offset(&w), 1);
        w.insert(META_INJECT_OFFSET, Array::from_slice(&[2.0f32], &[1]));
        assert_eq!(read_inject_offset(&w), 2);
    }

    /// The RMS clamp caps an over-budget residual at `τ·RMS(main)` and leaves an in-budget one alone.
    #[test]
    fn apply_clamp_caps_over_budget_residual() {
        let branch = Krea2ControlBranch {
            blocks: vec![],
            clamp_tau: Some(0.15),
            inject_offset: 1,
        };
        let main = Array::ones::<f32>(&[1, 16, 6]).unwrap(); // RMS(main) = 1 → cap = 0.15

        // Over-budget: RMS(res) = 1 ≫ 0.15 → scaled down to RMS 0.15.
        let big = Array::ones::<f32>(&[1, 16, 6]).unwrap();
        let clamped = branch.apply_clamp(&big, &main).unwrap();
        let got = rms(&clamped).unwrap().item::<f32>();
        assert!(
            (got - 0.15).abs() < 1e-4,
            "over-budget RMS should clamp to 0.15, got {got}"
        );

        // In-budget: RMS(res) = 0.1 ≤ 0.15 → unchanged.
        let small = multiply(Array::ones::<f32>(&[1, 16, 6]).unwrap(), scalar(0.1)).unwrap();
        let passed = branch.apply_clamp(&small, &main).unwrap();
        let got = rms(&passed).unwrap().item::<f32>();
        assert!(
            (got - 0.1).abs() < 1e-4,
            "in-budget RMS should pass through, got {got}"
        );
    }

    /// A zero residual (step 0, zero-init `proj_out`) survives the clamp as zero — the identity that
    /// keeps `control_scale`-scaled step 0 an exact base forward.
    #[test]
    fn apply_clamp_zero_residual_is_identity() {
        let branch = Krea2ControlBranch {
            blocks: vec![],
            clamp_tau: Some(0.15),
            inject_offset: 1,
        };
        let main = Array::ones::<f32>(&[1, 16, 6]).unwrap();
        let zero = Array::zeros::<f32>(&[1, 16, 6]).unwrap();
        let out = branch.apply_clamp(&zero, &main).unwrap();
        assert_eq!(rms(&out).unwrap().item::<f32>(), 0.0);
    }
}
