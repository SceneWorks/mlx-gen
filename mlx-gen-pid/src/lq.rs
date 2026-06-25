//! Sigma-aware LQ adapter (`LQProjection2D` + `SigmaAwareGatePerTokenPerDim`) and the `PidNet`
//! wrapper that injects its controlnet-style gate between the backbone's patch blocks. Faithful port
//! of `pid/_src/networks/lq_projection_2d.py` + `pid_net.py` (the inference subset).
//!
//! Scope: the **latent-only** path every in-scope catalog student uses (`lq_in_channels=0`,
//! `z_to_patch_ratio = (sr_scale·lsdf)/patch_size = 2` → nearest-upsample, `lq_interval=2`). The
//! image branch (`lq_in_channels>0`, PixelUnshuffle + bilinear align) and the merge path are never
//! exercised by any released catalog checkpoint and are intentionally not ported (additive if one
//! ever ships an image-conditioned student); the latent `z_to_patch_ratio<1` fold branch likewise
//! never occurs for the 16-/4-channel catalog spaces.

use mlx_rs::ops::{add, concatenate_axis, exp, multiply, sigmoid, subtract};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::{conv2d, group_norm, silu, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::backbone::{PatchInjector, PixDiT};
use crate::config::PidConfig;

const GN_EPS: f32 = 1e-5; // torch nn.GroupNorm default
const GN_GROUPS: i32 = 4; // ResBlock default num_groups

/// Load a dense Linear (`[out, in]` weight + optional bias).
fn lin(w: &Weights, prefix: &str) -> Result<AdaptableLinear> {
    let weight = w.require(&format!("{prefix}.weight"))?.clone();
    let bias = w.get(&format!("{prefix}.bias")).cloned();
    Ok(AdaptableLinear::dense(weight, bias))
}

/// A Conv2d that stores its weight in mlx NHWC `[out, kH, kW, in]` (transposed from the torch
/// `[out, in, kH, kW]` at load) and runs over NHWC activations.
struct Conv2d {
    weight: Array,
    bias: Option<Array>,
    padding: i32,
}

impl Conv2d {
    fn from_weights(w: &Weights, prefix: &str, padding: i32) -> Result<Self> {
        let weight = w
            .require(&format!("{prefix}.weight"))?
            .transpose_axes(&[0, 2, 3, 1])?; // [out,in,kH,kW] -> [out,kH,kW,in]
        Ok(Self {
            weight,
            bias: w.get(&format!("{prefix}.bias")).cloned(),
            padding,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        conv2d(x, &self.weight, self.bias.as_ref(), 1, self.padding)
    }
}

/// Per-activation GroupNorm (NHWC).
struct GroupNorm {
    weight: Array,
    bias: Array,
}

impl GroupNorm {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.clone(),
            bias: w.require(&format!("{prefix}.bias"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        group_norm(x, &self.weight, &self.bias, GN_GROUPS, GN_EPS)
    }
}

/// Pre-activation residual block: `x + Conv(SiLU(GN(Conv(SiLU(GN(x))))))`. Indices match the torch
/// `nn.Sequential` (0 GN, 2 Conv, 3 GN, 5 Conv).
struct ResBlock {
    gn0: GroupNorm,
    conv2: Conv2d,
    gn3: GroupNorm,
    conv5: Conv2d,
}

impl ResBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gn0: GroupNorm::from_weights(w, &format!("{prefix}.block.0"))?,
            conv2: Conv2d::from_weights(w, &format!("{prefix}.block.2"), 1)?,
            gn3: GroupNorm::from_weights(w, &format!("{prefix}.block.3"))?,
            conv5: Conv2d::from_weights(w, &format!("{prefix}.block.5"), 1)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.conv2.forward(&silu(&self.gn0.forward(x)?)?)?;
        let h = self.conv5.forward(&silu(&self.gn3.forward(&h)?)?)?;
        Ok(add(x, &h)?)
    }
}

/// `Conv(in→hidden) → SiLU → Conv(hidden→hidden) → ResBlock×N` over NHWC.
struct ConvStack {
    conv0: Conv2d,
    conv2: Conv2d,
    res: Vec<ResBlock>,
}

impl ConvStack {
    fn from_weights(w: &Weights, prefix: &str, num_res_blocks: i32) -> Result<Self> {
        Ok(Self {
            conv0: Conv2d::from_weights(w, &format!("{prefix}.0"), 1)?,
            conv2: Conv2d::from_weights(w, &format!("{prefix}.2"), 1)?,
            res: (0..num_res_blocks)
                .map(|i| ResBlock::from_weights(w, &format!("{prefix}.{}", i + 3)))
                .collect::<Result<_>>()?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = self.conv2.forward(&silu(&self.conv0.forward(x)?)?)?;
        for rb in &self.res {
            x = rb.forward(&x)?;
        }
        Ok(x)
    }
}

/// `SigmaAwareGatePerTokenPerDim`: `out = x + sigmoid(content_proj([x;lq]) − exp(log_alpha)·σ)·lq`.
struct SigmaGate {
    content_proj: AdaptableLinear,
    log_alpha: Array,
}

impl SigmaGate {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            content_proj: lin(w, &format!("{prefix}.content_proj"))?,
            log_alpha: w.require(&format!("{prefix}.log_alpha"))?.clone(),
        })
    }

    /// `x`, `lq`: `[B, N, D]`; `sigma`: `[B]`.
    fn forward(&self, x: &Array, lq: &Array, sigma: &Array) -> Result<Array> {
        let logit = self
            .content_proj
            .forward(&concatenate_axis(&[x, lq], -1)?)?; // [B,N,D]
        let b = sigma.shape()[0];
        let sigma_off = multiply(&exp(&self.log_alpha)?, &sigma.reshape(&[b, 1, 1])?)?; // exp(log_alpha)·σ
        let gate = sigmoid(&subtract(&logit, &sigma_off)?)?;
        Ok(add(x, &multiply(&gate, lq)?)?)
    }
}

/// `LQProjection2D` (latent-only): nearest-upsample the latent to the patch grid, run the conv stack,
/// then project to `num_outputs` per-block token feature sets; plus the per-block sigma gates.
pub struct LqAdapter {
    latent_proj: ConvStack,
    output_heads: Vec<AdaptableLinear>,
    gates: Vec<SigmaGate>,
    interval: i32,
    upsample_ratio: i32,
}

impl LqAdapter {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &PidConfig) -> Result<Self> {
        let num_outputs = cfg.num_lq_outputs();
        let z_to_patch = (cfg.sr_scale * cfg.latent_spatial_down_factor) / cfg.patch_size;
        Ok(Self {
            latent_proj: ConvStack::from_weights(
                w,
                &format!("{prefix}.latent_proj"),
                cfg.lq_num_res_blocks,
            )?,
            output_heads: (0..num_outputs)
                .map(|i| lin(w, &format!("{prefix}.output_heads.{i}")))
                .collect::<Result<_>>()?,
            gates: (0..num_outputs)
                .map(|i| SigmaGate::from_weights(w, &format!("{prefix}.gate_modules.{i}")))
                .collect::<Result<_>>()?,
            interval: cfg.lq_interval,
            upsample_ratio: z_to_patch.max(1),
        })
    }

    /// Project an LQ latent `[B, z_dim, zH, zW]` to `num_outputs` token feature sets `[B, N, out_dim]`
    /// (`N = pH·pW`).
    pub fn forward(&self, lq_latent: &Array, p_h: i32, p_w: i32) -> Result<Vec<Array>> {
        let b = lq_latent.shape()[0];
        let mut x = lq_latent.transpose_axes(&[0, 2, 3, 1])?; // NCHW -> NHWC
        if self.upsample_ratio > 1 {
            x = upsample_nearest(&x, self.upsample_ratio)?;
        }
        let x = self.latent_proj.forward(&x)?; // [B, pH, pW, hidden]
        let hidden = x.shape()[3];
        let tokens = x.reshape(&[b, p_h * p_w, hidden])?;
        self.output_heads
            .iter()
            .map(|h| h.forward(&tokens))
            .collect()
    }

    /// Whether the gate fires at this patch-block index (`interval>1` → every `interval`-th block).
    pub fn is_gate_active(&self, block_idx: i32) -> bool {
        self.interval <= 1 || block_idx % self.interval == 0
    }

    /// Map a patch-block index to its output-head / gate index.
    pub fn output_index(&self, block_idx: i32) -> i32 {
        if self.interval > 1 {
            block_idx / self.interval
        } else {
            block_idx
        }
    }

    /// Apply the `out_idx`-th sigma-aware gate: `x + sigmoid(content_proj([x;lq]) − α·σ)·lq`.
    pub fn gate(&self, out_idx: usize, x: &Array, lq: &Array, sigma: &Array) -> Result<Array> {
        self.gates[out_idx].forward(x, lq, sigma)
    }
}

/// `PidNet` — the backbone plus the LQ adapter wired as a between-blocks gate injector.
pub struct PidNet {
    backbone: PixDiT,
    lq: LqAdapter,
    patch_size: i32,
}

impl PidNet {
    /// `prefix` is `""` for a bare-key fixture or `"net."` for the released checkpoint's nesting.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &PidConfig) -> Result<Self> {
        Ok(Self {
            backbone: PixDiT::from_weights(w, prefix, cfg)?,
            lq: LqAdapter::from_weights(w, &format!("{prefix}lq_proj"), cfg)?,
            patch_size: cfg.patch_size,
        })
    }

    /// `x`: `[B, 3, H, W]`; `t`: `[B]`; `y`: caption embeddings `[B, Ltxt, txt_embed_dim]`;
    /// `lq_latent`: `[B, z_dim, zH, zW]`; `sigma`: per-sample LQ noise level `[B]`.
    pub fn forward(
        &self,
        x: &Array,
        t: &Array,
        y: &Array,
        lq_latent: &Array,
        sigma: &Array,
    ) -> Result<Array> {
        let sh = x.shape();
        let (p_h, p_w) = (sh[2] / self.patch_size, sh[3] / self.patch_size);
        let feats = self.lq.forward(lq_latent, p_h, p_w)?;
        let inj = LqInjection {
            lq: &self.lq,
            feats,
            sigma: sigma.clone(),
        };
        self.backbone.forward_with(x, t, y, &inj)
    }

    /// Access the LQ adapter (e.g. to parity-test its projection in isolation).
    pub fn lq(&self) -> &LqAdapter {
        &self.lq
    }
}

/// Binds the LQ adapter + this generation's precomputed features + sigma into the patch-block hook.
struct LqInjection<'a> {
    lq: &'a LqAdapter,
    feats: Vec<Array>,
    sigma: Array,
}

impl PatchInjector for LqInjection<'_> {
    fn inject(&self, block_idx: i32, s_main: &Array) -> Result<Array> {
        if self.lq.is_gate_active(block_idx) {
            let out_idx = self.lq.output_index(block_idx) as usize;
            if out_idx < self.feats.len() {
                return self
                    .lq
                    .gate(out_idx, s_main, &self.feats[out_idx], &self.sigma);
            }
        }
        Ok(s_main.clone())
    }
}
