//! Wan DiT — port of `models/wan/{model.py,transformer.py,attention.py}` (`WanModel`,
//! `WanAttentionBlock`, `WanSelfAttention`, `WanCrossAttention`, `Head`).
//!
//! The backbone is dimension-parametric; this slice targets the dense 5B (dim 3072, 30 layers,
//! 24 heads, head_dim 128, in/out 48). A latent `[C, F, H, W]` is 3-D patchified + linearly embedded,
//! refined by `num_layers` blocks (self-attn with qk-RMSNorm + 3-axis RoPE, cross-attn over the text
//! context, adaLN-6vec modulation, gated-GELU FFN), then projected by the modulated `Head` and
//! unpatchified back to `[out_dim, F, H, W]`.
//!
//! ## Dtype regime — mirrors the reference exactly (the parity contract)
//! Weights load **as stored** (bf16; modulation tables upcast to f32, equivalently). Every
//! projection / attention matmul runs **bf16** (the reference's `x.astype(_linear_dtype)` before
//! each), but the **residual stream and modulation are f32**: the modulation tables + the time
//! embedding `e` are f32, so `norm(x)·(1+e_scale)+e_shift` and `x + gate·y` promote the stream to f32
//! from the first block on (the reference's `autocast(float32)` residual). qk-RMSNorm runs on the
//! bf16 q/k; RoPE applies in f32 on **bf16-precision** cos/sin (the reference's `prepare_rope` builds
//! them in the weight dtype) then casts back to bf16. The patch embedding and the head run
//! f32-promoted matmuls (bf16 weight × f32 activation), as the reference does (no `.astype` there).
//!
//! The NAX 16-bit GEMM + SDPA are correct on the pinned build (sc-2772 fixed the metal compile
//! target to macOS 26.2, where the `mpp::tensor_ops::matmul2d` matrix-unit kernels are valid — see
//! [[pmetal-mlx-bf16-matmul-bug]]). An earlier f32-activation version was a workaround for the
//! then-broken bf16 SDPA; it's obsolete. The pinned build is MLX **0.31.1**; the production reference
//! is **0.31.2** (which reworked the NAX bf16 kernels), so bf16 parity is exact only up to that
//! cross-version kernel difference (f32 is bit-exact across the two) until the pin moves to 0.31.2.

use std::sync::atomic::{AtomicBool, Ordering};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear, Adapter};
use mlx_gen::array::scalar;
use mlx_gen::train::lora::LoraParams;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, cos, multiply, power, sigmoid, sin, split, tanh,
};
use mlx_rs::transforms::checkpoint;
use mlx_rs::transforms::compile::compile;
use mlx_rs::{Array, Dtype};

use crate::config::{WanModelConfig, WanQuant};
use crate::patchify::{patchify, unpatchify};
use crate::rope::RopeTable;
use crate::text_encoder::gelu_tanh;

/// sc-2957: when on, the Wan DiT's fusable elementwise *glue* (adaLN affine, gated residual,
/// gated-GELU FFN activation, RoPE rotation) runs through `mx.compile` so MLX fuses each chain into a
/// single kernel (vs one Metal kernel per primitive op when eager) — **bit-exact** and **+14.1% /
/// step** at production geometry (480p×25f A14B: 23.07→19.81 s/step, matching Python's whole-model
/// `mx.compile` ceiling; `tests/perf.rs`). **Enabled by the production denoise loops** ([`denoise`](
/// crate::pipeline::denoise) / [`denoise_moe`](crate::pipeline::denoise_moe)); left **off by default**
/// so the tiny reference-parity gates run the eager form and `compile_parity.rs` can A/B both.
static COMPILE_GLUE: AtomicBool = AtomicBool::new(false);

/// Enable/disable compiled elementwise glue (sc-2957). Process-global; prefer the scoped
/// [`CompileGlueGuard`] in production (the raw setter is for the A/B `compile_parity`/`perf` gates).
pub fn set_compile_glue(on: bool) {
    COMPILE_GLUE.store(on, Ordering::Relaxed);
}

pub(crate) fn compile_glue() -> bool {
    COMPILE_GLUE.load(Ordering::Relaxed)
}

/// RAII guard (F-006/F-007, mirroring core `mlx_gen::nn::CompileGlueGuard` and z-image's) that enables
/// compiled glue for its lifetime and **restores the prior [`COMPILE_GLUE`] value on drop** — even on
/// an early `?`. The production denoise loops ([`denoise`](crate::pipeline::denoise) /
/// [`denoise_moe`](crate::pipeline::denoise_moe)) bind one across the render so the toggle is scoped,
/// not left stuck `true` process-wide, and same-process eager code (the `compile_parity`/`perf` gates)
/// sees the restored value.
#[must_use = "dropping the guard restores the prior compile-glue setting; bind it for the render's lifetime"]
pub(crate) struct CompileGlueGuard {
    prev: bool,
}

impl CompileGlueGuard {
    /// Turn compiled glue on, remembering the prior value to restore on drop.
    pub(crate) fn enable() -> Self {
        Self {
            prev: COMPILE_GLUE.swap(true, Ordering::Relaxed),
        }
    }
}

impl Drop for CompileGlueGuard {
    fn drop(&mut self) {
        COMPILE_GLUE.store(self.prev, Ordering::Relaxed);
    }
}

/// adaLN affine `m·(1+e_scale)+e_shift` — one fused kernel when compiled, else 2 eager ops. The
/// `mx.compile` graph is bit-exact to the eager form (proven `max|Δ|=0`, `tests/compile_micro.rs`).
fn modulate(m: &Array, e_scale: &Array, e_shift: &Array) -> Result<Array> {
    let f = |(m, s, sh): (&Array, &Array, &Array)| -> std::result::Result<Array, Exception> {
        add(&multiply(m, &add(s, scalar(1.0))?)?, sh)
    };
    if compile_glue() {
        Ok(compile(f, true)((m, e_scale, e_shift))?)
    } else {
        Ok(f((m, e_scale, e_shift))?)
    }
}

/// Gated residual `x + y·gate` — one fused kernel when compiled.
fn gated(x: &Array, y: &Array, gate: &Array) -> Result<Array> {
    let f = |(x, y, g): (&Array, &Array, &Array)| -> std::result::Result<Array, Exception> {
        add(x, &multiply(y, g)?)
    };
    if compile_glue() {
        Ok(compile(f, true)((x, y, gate))?)
    } else {
        Ok(f((x, y, gate))?)
    }
}

/// Gated-GELU FFN activation. Body mirrors [`mlx_gen::nn::gelu_tanh`] exactly (bit-exact, dtype-
/// preserving); when compiled, MLX fuses its ~8 elementwise ops into one kernel (the single biggest
/// per-step glue cost — ~600 MB bf16 tensor × 40 layers, sc-2957). Off ⇒ defers to core `gelu_tanh`.
fn gelu_ffn(x: &Array) -> Result<Array> {
    if !compile_glue() {
        return gelu_tanh(x);
    }
    let f = |x_: &Array| -> std::result::Result<Array, Exception> {
        let dt = x_.dtype();
        let s = |v: f32| -> std::result::Result<Array, Exception> { scalar(v).as_dtype(dt) };
        let c = (2.0_f64 / std::f64::consts::PI).sqrt() as f32;
        let x3 = power(x_, Array::from_int(3))?;
        let inner = multiply(&add(x_, &multiply(&x3, &s(0.044_715)?)?)?, &s(c)?)?;
        let gate = add(&tanh(&inner)?, &s(1.0)?)?;
        multiply(&multiply(x_, &s(0.5)?)?, &gate)
    };
    Ok(compile(f, true)(x)?)
}

/// Load a biased `[out, in]` Linear as a core [`AdaptableLinear`] (every Wan DiT `nn.Linear` is
/// biased). The dense base mirrors MLX's `nn.Linear` exactly — a **fused** `addmm(bias, x, Wᵀ)`
/// (accumulate `x·Wᵀ`, add bias, round once) — so it is bit-for-bit identical to the previous
/// hand-rolled `Linear::forward` on every existing path (a separate `matmul`+`add` double-rounds in
/// bf16, ~1.4e-3/layer, the gap that once localized to `q_proj`; the fusion is a no-op under f32
/// activations). Using `AdaptableLinear` makes the base **quantizable in place** ([`quantize`](
/// AdaptableLinear::quantize), sc-2682) and adapter-ready (sc-2683 / sc-2393) without changing the
/// dense numerics. `forward` is dtype-agnostic: the result dtype follows `x`.
///
/// When `quant` is `Some` (a **pre-quantized snapshot** — the `config.json` `quantization` block) and
/// this Linear carries packed weights on disk (`.scales` present), build the base **quantized
/// directly** from the on-disk parts (the `loading.py` consume path) instead of loading dense bf16.
/// `.scales` presence is the per-Linear signal (only the `_quantize_predicate` Linears are packed;
/// embeddings/norms/head stay dense), exactly mirroring the reference's predicate.
fn load_linear(w: &Weights, prefix: &str, quant: Option<WanQuant>) -> Result<AdaptableLinear> {
    if let (Some(q), Some(scales)) = (quant, w.get(&format!("{prefix}.scales"))) {
        return Ok(AdaptableLinear::from_quantized_parts(
            w.require(&format!("{prefix}.weight"))?.clone(),
            scales.clone(),
            w.require(&format!("{prefix}.biases"))?.clone(),
            w.get(&format!("{prefix}.bias")).cloned(),
            q.group_size,
            q.bits,
        ));
    }
    Ok(AdaptableLinear::dense(
        w.require(&format!("{prefix}.weight"))?.clone(),
        Some(w.require(&format!("{prefix}.bias"))?.clone()),
    ))
}

/// SiLU `x·σ(x)` (the reference `nn.SiLU`), bit-exact and dtype-preserving.
fn silu(x: &Array) -> Result<Array> {
    Ok(multiply(x, &sigmoid(x)?)?)
}

fn f32(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Float32)?)
}

fn bf16(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Bfloat16)?)
}

/// `WanLayerNorm` with `elementwise_affine=False` — `mx.fast.layer_norm(x, None, None, eps)`.
fn ln(x: &Array, eps: f32) -> Result<Array> {
    Ok(layer_norm(x, None, None, eps)?)
}

/// Push a linear's quantized packs (`wq`/`scales`/`biases` + optional bias) into `out` for the
/// eval-to-free pass (sc-5360); a dense linear contributes nothing.
fn push_quant_arrays<'a>(lin: &'a AdaptableLinear, out: &mut Vec<&'a Array>) {
    if let Some((wq, scales, biases, bias, _, _)) = lin.quantized_params() {
        out.push(wq);
        out.push(scales);
        out.push(biases);
        if let Some(bias) = bias {
            out.push(bias);
        }
    }
}

#[derive(Clone)]
struct SelfAttention {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    norm_q: Array, // qk-RMSNorm over the full dim
    norm_k: Array,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    eps: f32,
    /// sc-4942 — run the SDPA segment inside an `mlx::checkpoint` so its backward recomputes the
    /// decomposed attention (MLX has no fused SDPA backward) instead of retaining the per-layer seq²
    /// probability matrix. Training-only; `false` on the inference path (byte-identical).
    ckpt_sdpa: bool,
}

impl SelfAttention {
    fn load(w: &Weights, prefix: &str, cfg: &WanModelConfig) -> Result<Self> {
        let head_dim = cfg.dim / cfg.num_heads;
        let q = cfg.quantization;
        Ok(Self {
            q: load_linear(w, &format!("{prefix}.q"), q)?,
            k: load_linear(w, &format!("{prefix}.k"), q)?,
            v: load_linear(w, &format!("{prefix}.v"), q)?,
            o: load_linear(w, &format!("{prefix}.o"), q)?,
            norm_q: w.require(&format!("{prefix}.norm_q.weight"))?.clone(),
            norm_k: w.require(&format!("{prefix}.norm_k.weight"))?.clone(),
            num_heads: cfg.num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps: cfg.eps as f32,
            ckpt_sdpa: false,
        })
    }

    /// Quantize the four projections to Q4/Q8 in place (`_quantize_predicate`'s `.self_attn.{q,k,v,o}`).
    fn quantize(&mut self, bits: i32, group: Option<i32>) -> Result<()> {
        self.q.quantize(bits, group)?;
        self.k.quantize(bits, group)?;
        self.v.quantize(bits, group)?;
        self.o.quantize(bits, group)?;
        Ok(())
    }

    /// Collect this attention's quantized packs (sc-5360 eval-to-free).
    fn push_quant_arrays<'a>(&'a self, out: &mut Vec<&'a Array>) {
        for lin in [&self.q, &self.k, &self.v, &self.o] {
            push_quant_arrays(lin, out);
        }
    }

    /// Route a LoRA-training target (`q`/`k`/`v`/`o`) to its [`AdaptableLinear`] (sc-3046).
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["q"] => Some(&mut self.q),
            ["k"] => Some(&mut self.k),
            ["v"] => Some(&mut self.v),
            ["o"] => Some(&mut self.o),
            _ => None,
        }
    }

    /// `x_mod`: `[B, L, dim]` (f32). `cos`/`sin`: `[L, 1, half_d]` (bf16). Returns `[B, L, dim]` bf16.
    /// Batched over `B` (the CFG cond/uncond branches) — attention never mixes batch elements, so the
    /// `B=2` result is bit-identical to two `B=1` calls (the cos/sin broadcast across batch + heads).
    fn forward(&self, x_mod: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        // Matmuls run bf16 (the reference's `x.astype(w_dtype)`); the f32 residual is restored by the
        // block's modulation. q/k get full-dim bf16 RMSNorm before the head split; RoPE applies in
        // f32 on bf16 cos/sin then casts back to bf16 for the bf16 SDPA.
        let xw = bf16(x_mod)?;
        let (n, d) = (self.num_heads as i32, self.head_dim as i32);
        let b = x_mod.shape()[0];
        let s = x_mod.shape()[1];

        let q = rms_norm(&self.q.forward(&xw)?, &self.norm_q, self.eps)?;
        let k = rms_norm(&self.k.forward(&xw)?, &self.norm_k, self.eps)?;
        let q = bf16(&crate::rope::rope_apply(
            &f32(&q.reshape(&[b, s, n, d])?)?,
            cos,
            sin,
        )?)?
        .transpose_axes(&[0, 2, 1, 3])?;
        let k = bf16(&crate::rope::rope_apply(
            &f32(&k.reshape(&[b, s, n, d])?)?,
            cos,
            sin,
        )?)?
        .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v
            .forward(&xw)?
            .reshape(&[b, s, n, d])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let out = sdpa_maybe_checkpoint(&q, &k, &v, self.scale, self.ckpt_sdpa)?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, n * d])?;
        self.o.forward(&out)
    }

    /// Toggle SDPA-segment checkpointing (sc-4942). Training-only — see `ckpt_sdpa`.
    fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.ckpt_sdpa = on;
    }
}

/// Run `scaled_dot_product_attention`, optionally inside an `mlx::checkpoint` segment (sc-4942). When
/// `ckpt` is on, q/k/v are the threaded inputs (grads to the QKV projections — and their LoRA — flow
/// through them) and only the f32 `scale` is captured; the backward recomputes the decomposed
/// attention for this layer alone, so the seq² probability matrix is a per-layer transient rather than
/// retained across all blocks. `ckpt = false` is byte-identical to the bare call (inference).
fn sdpa_maybe_checkpoint(q: &Array, k: &Array, v: &Array, scale: f32, ckpt: bool) -> Result<Array> {
    if !ckpt {
        return Ok(scaled_dot_product_attention(q, k, v, scale, None, None)?);
    }
    let mut seg = checkpoint(move |inp: &[Array]| -> MlxResult<Vec<Array>> {
        Ok(vec![scaled_dot_product_attention(
            &inp[0], &inp[1], &inp[2], scale, None, None,
        )?])
    });
    seg(&[q.clone(), k.clone(), v.clone()])?
        .into_iter()
        .next()
        .ok_or_else(|| Error::Msg("wan: checkpoint SDPA produced no output".into()))
}

#[derive(Clone)]
struct CrossAttention {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    eps: f32,
    /// sc-4942 — SDPA-segment checkpointing (training-only). See [`SelfAttention::ckpt_sdpa`].
    ckpt_sdpa: bool,
}

impl CrossAttention {
    fn load(w: &Weights, prefix: &str, cfg: &WanModelConfig) -> Result<Self> {
        let head_dim = cfg.dim / cfg.num_heads;
        let q = cfg.quantization;
        Ok(Self {
            q: load_linear(w, &format!("{prefix}.q"), q)?,
            k: load_linear(w, &format!("{prefix}.k"), q)?,
            v: load_linear(w, &format!("{prefix}.v"), q)?,
            o: load_linear(w, &format!("{prefix}.o"), q)?,
            norm_q: w.require(&format!("{prefix}.norm_q.weight"))?.clone(),
            norm_k: w.require(&format!("{prefix}.norm_k.weight"))?.clone(),
            num_heads: cfg.num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps: cfg.eps as f32,
            ckpt_sdpa: false,
        })
    }

    /// Toggle SDPA-segment checkpointing (sc-4942). Training-only — see [`SelfAttention::ckpt_sdpa`].
    fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.ckpt_sdpa = on;
    }

    /// Quantize the four projections to Q4/Q8 in place (`_quantize_predicate`'s `.cross_attn.{q,k,v,o}`).
    fn quantize(&mut self, bits: i32, group: Option<i32>) -> Result<()> {
        self.q.quantize(bits, group)?;
        self.k.quantize(bits, group)?;
        self.v.quantize(bits, group)?;
        self.o.quantize(bits, group)?;
        Ok(())
    }

    /// Collect this attention's quantized packs (sc-5360 eval-to-free).
    fn push_quant_arrays<'a>(&'a self, out: &mut Vec<&'a Array>) {
        for lin in [&self.q, &self.k, &self.v, &self.o] {
            push_quant_arrays(lin, out);
        }
    }

    /// Route a LoRA-training target (`q`/`k`/`v`/`o`) to its [`AdaptableLinear`] (sc-3046).
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["q"] => Some(&mut self.q),
            ["k"] => Some(&mut self.k),
            ["v"] => Some(&mut self.v),
            ["o"] => Some(&mut self.o),
            _ => None,
        }
    }

    /// Cached K/V from the (bf16) text context `[B, L_ctx, dim]` — computed once, reused per step.
    /// `B` is the forward batch (2 for CFG cond+uncond, 1 otherwise); returns `(k, v)` each
    /// `[B, n, L_ctx, d]`.
    fn prepare_kv(&self, context: &Array) -> Result<(Array, Array)> {
        let (n, d) = (self.num_heads as i32, self.head_dim as i32);
        let b = context.shape()[0];
        let ctx = bf16(context)?;
        let k = rms_norm(&self.k.forward(&ctx)?, &self.norm_k, self.eps)?
            .reshape(&[b, -1, n, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v
            .forward(&ctx)?
            .reshape(&[b, -1, n, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        Ok((k, v))
    }

    /// `x`: `[B, L, dim]` (f32). `(k, v)`: cached `[B, n, L_ctx, d]` (bf16). Returns `[B, L, dim]` bf16.
    fn forward(&self, x: &Array, kv: &(Array, Array)) -> Result<Array> {
        let (n, d) = (self.num_heads as i32, self.head_dim as i32);
        let b = x.shape()[0];
        let s = x.shape()[1];
        let q = rms_norm(&self.q.forward(&bf16(x)?)?, &self.norm_q, self.eps)?
            .reshape(&[b, s, n, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let out = sdpa_maybe_checkpoint(&q, &kv.0, &kv.1, self.scale, self.ckpt_sdpa)?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, n * d])?;
        self.o.forward(&out)
    }
}

#[derive(Clone)]
struct Block {
    modulation: Array, // [1, 6, dim]
    self_attn: SelfAttention,
    cross_attn: CrossAttention,
    norm3_w: Array, // cross-attn norm (affine LayerNorm)
    norm3_b: Array,
    ffn_fc1: AdaptableLinear,
    ffn_fc2: AdaptableLinear,
    eps: f32,
}

impl Block {
    fn load(w: &Weights, i: usize, cfg: &WanModelConfig) -> Result<Self> {
        let p = format!("blocks.{i}");
        Ok(Self {
            modulation: f32(w.require(&format!("{p}.modulation"))?)?,
            self_attn: SelfAttention::load(w, &format!("{p}.self_attn"), cfg)?,
            cross_attn: CrossAttention::load(w, &format!("{p}.cross_attn"), cfg)?,
            norm3_w: f32(w.require(&format!("{p}.norm3.weight"))?)?,
            norm3_b: f32(w.require(&format!("{p}.norm3.bias"))?)?,
            ffn_fc1: load_linear(w, &format!("{p}.ffn.fc1"), cfg.quantization)?,
            ffn_fc2: load_linear(w, &format!("{p}.ffn.fc2"), cfg.quantization)?,
            eps: cfg.eps as f32,
        })
    }

    /// Quantize this block's `_quantize_predicate` surface (self/cross attn `q/k/v/o` + `ffn.fc1/fc2`)
    /// to Q4/Q8 in place. The modulation table, `norm3`, and the qk-RMSNorm weights stay dense.
    fn quantize(&mut self, bits: i32, group: Option<i32>) -> Result<()> {
        self.self_attn.quantize(bits, group)?;
        self.cross_attn.quantize(bits, group)?;
        self.ffn_fc1.quantize(bits, group)?;
        self.ffn_fc2.quantize(bits, group)?;
        Ok(())
    }

    /// Collect this block's quantized packs (sc-5360 eval-to-free).
    fn push_quant_arrays<'a>(&'a self, out: &mut Vec<&'a Array>) {
        self.self_attn.push_quant_arrays(out);
        self.cross_attn.push_quant_arrays(out);
        push_quant_arrays(&self.ffn_fc1, out);
        push_quant_arrays(&self.ffn_fc2, out);
    }

    /// Route a LoRA-training target into this block's `self_attn`/`cross_attn` projections or its
    /// `ffn.fc1`/`ffn.fc2` (sc-3046) — the per-block adaptable surface.
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["self_attn", rest @ ..] => self.self_attn.adaptable_mut(rest),
            ["cross_attn", rest @ ..] => self.cross_attn.adaptable_mut(rest),
            ["ffn", "fc1"] => Some(&mut self.ffn_fc1),
            ["ffn", "fc2"] => Some(&mut self.ffn_fc2),
            _ => None,
        }
    }

    fn prepare_kv(&self, context: &Array) -> Result<(Array, Array)> {
        self.cross_attn.prepare_kv(context)
    }

    /// Toggle SDPA-segment checkpointing on both attentions (sc-4942).
    fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.self_attn.set_sdpa_checkpoint(on);
        self.cross_attn.set_sdpa_checkpoint(on);
    }

    /// `x`: `[B, L, dim]` (f32, `B` = CFG batch). `e`: `[1, L_e, 6, dim]` f32 time modulation —
    /// `L_e = 1` for the **scalar** timestep (T2V; broadcasts over both tokens and the CFG batch),
    /// `L_e = L` for the **per-token** timestep (TI2V mask-blend `t_tokens`, sc-2680; broadcasts over
    /// the CFG batch only). Every modulation/residual op below is broadcast over `B`; only the
    /// self/cross attention reshapes to `B`.
    fn forward(
        &self,
        x: &Array,
        e: &Array,
        kv: &(Array, Array),
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        // adaLN-6vec modulation, f32 (promotes the residual stream). [1,6,dim] + [1,L_e,6,dim] →
        // [1,L_e,6,dim], split into 6 × [1,L_e,dim] (shift/scale/gate for self-attn, then FFN).
        let dim = self.self_attn.num_heads as i32 * self.self_attn.head_dim as i32;
        let m = add(&self.modulation, e)?; // [1, L_e, 6, dim] f32
        let l_e = m.shape()[1];
        let p = split(&m, 6, 2)?;
        let v = |i: usize| -> Result<Array> { Ok(p[i].reshape(&[1, l_e, dim])?) };
        let (e0, e1, e2) = (v(0)?, v(1)?, v(2)?);
        let (e3, e4, e5) = (v(3)?, v(4)?, v(5)?);

        // Self-attention.
        let x_mod = modulate(&ln(x, self.eps)?, &e1, &e0)?;
        let y = self.self_attn.forward(&x_mod, cos, sin)?;
        let x = gated(x, &y, &e2)?;

        // Cross-attention (affine LayerNorm on context-side query, no modulation).
        let x_cross = layer_norm(&x, Some(&self.norm3_w), Some(&self.norm3_b), self.eps)?;
        let x = add(&x, &self.cross_attn.forward(&x_cross, kv)?)?;

        // Gated-GELU FFN (bf16 matmuls; the reference's `x.astype(w_dtype)`).
        let x_mod = modulate(&ln(&x, self.eps)?, &e4, &e3)?;
        let y = gelu_ffn(&self.ffn_fc1.forward(&bf16(&x_mod)?)?)?;
        let y = self.ffn_fc2.forward(&y)?;
        gated(&x, &y, &e5)
    }
}

/// A per-block trainable LoRA target threaded into the gradient-checkpoint closure (sc-4942): the
/// block-local resolution path (e.g. `["self_attn", "q"]` or `["ffn", "fc1"]`) and the factor-map keys
/// for its A/B, so [`WanTransformer::forward_train_checkpointed`] can pull each block's live factors
/// out of the traced `LoraParams` and re-install them inside the recompute segment. Built by the
/// trainer from its resolved target paths.
pub struct BlockLoraRef {
    pub local: Vec<String>,
    pub a_key: String,
    pub b_key: String,
}

/// The Wan DiT (5B dense T2V). Holds the loaded weights + the precomputed RoPE table.
pub struct WanTransformer {
    patch_embedding: AdaptableLinear,
    text_embedding_0: AdaptableLinear,
    text_embedding_1: AdaptableLinear,
    time_embedding_0: AdaptableLinear,
    time_embedding_1: AdaptableLinear,
    time_projection: AdaptableLinear,
    blocks: Vec<Block>,
    head_modulation: Array, // [1, 2, dim]
    head: AdaptableLinear,
    rope: RopeTable,
    inv_freq: Array, // [freq_dim/2] f32, for the sinusoidal time embedding
    cfg: WanModelConfig,
}

/// LoRA **training** adapter routing (sc-3046). The Wan DiT's Linears are already core
/// [`AdaptableLinear`]s; inference *merges* deltas into the weight map at load (`apply_wan_adapters`)
/// rather than installing forward-time residuals, so the static generate path never needs this. The
/// trainer does: it injects a fresh trainable `Adapter::Lora` per target each optimizer step (via
/// [`AdaptableLinear::set_adapters`]) and the block forward applies it. Routing uses the **native**
/// Wan checkpoint naming (`blocks.{i}.self_attn.{q,k,v,o}` / `cross_attn.{…}` / `ffn.fc1`/`fc2`) — the
/// same names `apply_wan_adapters`' `normalize_wan_key` resolves to, so a saved adapter round-trips.
impl AdaptableHost for WanTransformer {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["blocks", i, rest @ ..] => self
                .blocks
                .get_mut(i.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            _ => None,
        }
    }

    /// Every per-block adaptable Linear, in native naming: the self/cross-attention `q/k/v/o` and the
    /// FFN `fc1`/`fc2`. The trainer filters these by the configured target suffixes (default
    /// `q`/`k`/`v`/`o` — the native analog of the reference's `to_q/k/v/to_out.0` attention surface).
    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(self.blocks.len() * 10);
        for i in 0..self.blocks.len() {
            for attn in ["self_attn", "cross_attn"] {
                for proj in ["q", "k", "v", "o"] {
                    out.push(format!("blocks.{i}.{attn}.{proj}"));
                }
            }
            out.push(format!("blocks.{i}.ffn.fc1"));
            out.push(format!("blocks.{i}.ffn.fc2"));
        }
        out
    }
}

impl WanTransformer {
    pub fn from_weights(w: &Weights, cfg: &WanModelConfig) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::load(w, i, cfg)?);
        }
        let half = cfg.freq_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|j| (10000.0_f64.powf(-(j as f64) / half as f64)) as f32)
            .collect();
        // The patch/text/time embeddings, time_projection, and head are NOT in the reference's
        // `_quantize_predicate` (precision-sensitive) → always dense, even in a pre-quantized snapshot
        // (they carry no `.scales`), so `None`. Only the per-block attn/FFN Linears consume packed
        // weights, gated by `cfg.quantization` inside `Block::load`/`SelfAttention::load`.
        Ok(Self {
            patch_embedding: load_linear(w, "patch_embedding_proj", None)?,
            text_embedding_0: load_linear(w, "text_embedding_0", None)?,
            text_embedding_1: load_linear(w, "text_embedding_1", None)?,
            time_embedding_0: load_linear(w, "time_embedding_0", None)?,
            time_embedding_1: load_linear(w, "time_embedding_1", None)?,
            time_projection: load_linear(w, "time_projection", None)?,
            blocks,
            head_modulation: f32(w.require("head.modulation")?)?,
            head: load_linear(w, "head.head", None)?,
            rope: RopeTable::new(cfg.dim / cfg.num_heads),
            inv_freq: Array::from_slice(&inv, &[half as i32]),
            cfg: cfg.clone(),
        })
    }

    /// Quantize the transformer-only attention + FFN Linears to Q4/Q8 **in place** — the reference's
    /// `_quantize_predicate` surface: every block's self/cross-attention `q/k/v/o` and `ffn.fc1/fc2`.
    /// The patch/text/time embeddings, `time_projection`, modulation tables, qk/`norm3` norms, and the
    /// output head stay dense (small + precision-sensitive — the reference skips them). Mirrors
    /// `convert_wan.py::_quantize_predicate` + `loading.py` (sc-2682).
    ///
    /// `group` is the quantization group size; `None` ⇒ the mflux/reference default of 64. The core
    /// [`AdaptableLinear::quantize`] casts weight+bias to bf16 before packing so the group scales
    /// byte-match the reference's — a **no-op** for Wan, whose converted DiT is bf16-native (so the
    /// [[zimage-q8-f32-checkpoint-scales-sc2604]] chokepoint applies but never bites). Per-step compute
    /// then runs `quantized_matmul` (fp32-accumulate) on the bf16 activations the blocks already feed.
    pub fn quantize(&mut self, bits: i32, group: Option<i32>) -> Result<()> {
        for block in &mut self.blocks {
            block.quantize(bits, group)?;
        }
        // sc-5360: force-materialize the quantized packs now so the per-weight bf16 dequant transient
        // frees here, instead of lazily at the first forward where (for the dual-expert renderer) both
        // experts' transients would co-reside. The packed Q4/Q8 arrays are what stays resident; eval'ing
        // them releases the bf16 source. Transparent to the result (it forces work the first forward
        // would do anyway); the win is the load-time peak.
        let mut arrays: Vec<&Array> = Vec::new();
        for block in &self.blocks {
            block.push_quant_arrays(&mut arrays);
        }
        if !arrays.is_empty() {
            mlx_rs::transforms::eval(arrays)?;
        }
        Ok(())
    }

    /// Number of transformer blocks (for the trainer's target enumeration).
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Toggle SDPA-segment gradient checkpointing across the whole block stack (sc-4942) — both the
    /// self- and cross-attention SDPA of every block. The trainer arms it for the dense training
    /// forward (numerically identical to the retained backward; bounds the per-block seq² retention to
    /// one layer's transient) and turns it OFF when whole-block checkpointing is on (the block recompute
    /// already covers attention). Inference never calls this (the flag stays `false`).
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        for b in &mut self.blocks {
            b.set_sdpa_checkpoint(on);
        }
    }

    /// The DiT's current compute dtype (sc-4942), probed from the patch-embedding weight. Wan is
    /// **bf16-native** — the converted checkpoint loads bf16 and every matmul already runs bf16 (with an
    /// f32 residual stream), so this reports `Bfloat16` and the worker's `train_dtype=bf16` is honored
    /// by the load itself (no cast lever, unlike the f32-base families). Lets the trainer confirm the
    /// base precision (`None` if the patch embedding were ever quantized, which it is not — it is one of
    /// the precision-sensitive dense Linears the reference never packs).
    pub fn compute_dtype(&self) -> Option<Dtype> {
        self.patch_embedding.weight_dtype()
    }

    /// Training forward with **per-block gradient checkpointing** (sc-4942). Identical compute to
    /// [`forward`](Self::forward) at `batch = 1`, but each block runs inside an `mlx::checkpoint`
    /// segment whose explicit inputs are the block hidden state plus that block's trainable LoRA factors
    /// — so the reverse pass recomputes the block (including its cross-attention K/V from the threaded
    /// context constant) instead of retaining its activations, while gradients still flow to the LoRA
    /// params. The patchify/embed, time embedding, RoPE, and output head run normally.
    ///
    /// `params` is the live (traced) factor map; `block_targets[i]` lists block `i`'s trainable targets
    /// (block-local path + factor keys), in the order their factors are threaded as checkpoint inputs.
    /// Factors install at their f32 master dtype (matching the dense `install_training_lora_as` path),
    /// so the recompute is numerically identical. Returns the denoised `[out_dim, F, H, W]` (f32).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_train_checkpointed(
        &self,
        latent: &Array,
        t: f32,
        context_embed: &Array,
        params: &LoraParams,
        block_targets: &[Vec<BlockLoraRef>],
        alpha: f32,
    ) -> Result<Array> {
        let (tokens, grid) = patchify(latent, self.cfg.patch_size)?;
        let l = (grid.0 * grid.1 * grid.2) as i32;
        let dim = self.cfg.dim as i32;
        let mut x = bf16(&self.patch_embedding.forward(&tokens)?)?.reshape(&[1, l, dim])?;
        let (e, e0) = self.time_embed(t)?;
        let (cos_t, sin_t) = self.prepare_rope(grid)?;

        for (i, block) in self.blocks.iter().enumerate() {
            // Cheap clone (Arrays are refcounted): the closure must OWN its state because the backward
            // recompute runs after this frame is gone; a borrow of `self` would dangle.
            let mut blk = block.clone();
            let empty: Vec<BlockLoraRef> = Vec::new();
            let refs = block_targets.get(i).unwrap_or(&empty);
            let locals: Vec<Vec<String>> = refs.iter().map(|r| r.local.clone()).collect();
            let context_c = context_embed.clone();
            let e0_c = e0.clone();
            let cos_c = cos_t.clone();
            let sin_c = sin_t.clone();
            let alpha_c = alpha;

            // Threaded inputs: [hidden, a_0, b_0, a_1, b_1, ...] (raw `[r,in]`/`[out,r]` f32 factors).
            let mut inputs: Vec<Array> = Vec::with_capacity(1 + 2 * refs.len());
            inputs.push(x.clone());
            for r in refs {
                inputs.push(
                    params
                        .get(r.a_key.as_str())
                        .ok_or_else(|| {
                            mlx_gen::Error::Msg(format!("LoRA param missing: {}", r.a_key))
                        })?
                        .clone(),
                );
                inputs.push(
                    params
                        .get(r.b_key.as_str())
                        .ok_or_else(|| {
                            mlx_gen::Error::Msg(format!("LoRA param missing: {}", r.b_key))
                        })?
                        .clone(),
                );
            }

            let mut seg = checkpoint(move |inp: &[Array]| -> MlxResult<Vec<Array>> {
                // Reinstall the explicit-input factors with the SAME `(transpose, alpha/rank fold)`
                // `install_training_lora_as` applies (dtype=None → factors stay f32), so the
                // checkpointed block forward is numerically identical to the installed-adapter path and
                // grads route to `inp`.
                for (j, local) in locals.iter().enumerate() {
                    let a = inp[1 + 2 * j].t(); // [r,in] -> [in,r]
                    let rank = a.shape()[1] as f32;
                    let b = inp[2 + 2 * j].t().multiply(scalar(alpha_c / rank))?; // [out,r]->[r,out]·(α/r)
                    let seg_refs: Vec<&str> = local.iter().map(String::as_str).collect();
                    blk.adaptable_mut(&seg_refs)
                        .ok_or_else(|| {
                            Exception::custom(format!(
                                "checkpoint LoRA target not found: {}",
                                local.join(".")
                            ))
                        })?
                        .set_adapters(vec![Adapter::Lora { a, b, scale: 1.0 }]);
                }
                let kv = blk
                    .prepare_kv(&context_c)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let out = blk
                    .forward(&inp[0], &e0_c, &kv, &cos_c, &sin_c)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                Ok(vec![out])
            });
            x = seg(&inputs)?
                .into_iter()
                .next()
                .ok_or_else(|| Error::Msg("wan: checkpoint block produced no output".into()))?;
        }

        let x = self.apply_head(&x, &e)?; // [1, L, out_dim·∏patch] f32
        let op = x.shape()[2];
        let xb = x.reshape(&[l, op])?;
        unpatchify(&xb, grid, self.cfg.out_dim, self.cfg.patch_size)
    }

    /// Embed a single T5 prompt embedding `[L_text, text_dim]` → `[1, text_len, dim]` (bf16),
    /// zero-padded to `text_len`. Mirrors `WanModel.embed_text` for one prompt.
    pub fn embed_text(&self, t5_embed: &Array) -> Result<Array> {
        let text_len = self.cfg.text_len as i32;
        let l = t5_embed.shape()[0];
        let dim_text = t5_embed.shape()[1];
        let ctx = if l < text_len {
            let pad = Array::zeros::<f32>(&[text_len - l, dim_text])?.as_dtype(t5_embed.dtype())?;
            concatenate_axis(&[t5_embed, &pad], 0)?
        } else {
            t5_embed.clone()
        };
        let ctx = ctx.reshape(&[1, text_len, dim_text])?;
        let h = self.text_embedding_0.forward(&ctx)?;
        let h = gelu_tanh(&h)?;
        // Cast to bf16 (the reference's `.astype(model_dtype)`); the cross-attn K/V run bf16.
        bf16(&self.text_embedding_1.forward(&h)?)
    }

    /// Sinusoidal time embedding `e` `[1, dim]` (f32) + the 6-vector modulation `e0` `[1,1,6,dim]`.
    fn time_embed(&self, t: f32) -> Result<(Array, Array)> {
        let pos = Array::from_slice(&[t], &[1, 1]); // [1, 1]
        let sinusoid = multiply(&pos, &self.inv_freq)?; // [1, half] f32
        let sin_emb = concatenate_axis(&[&cos(&sinusoid)?, &sin(&sinusoid)?], 1)?; // [1, freq_dim]
        let e = self
            .time_embedding_1
            .forward(&silu(&self.time_embedding_0.forward(&sin_emb)?)?)?; // [1, dim] f32
        let e0 =
            self.time_projection
                .forward(&silu(&e)?)?
                .reshape(&[1, 1, 6, self.cfg.dim as i32])?; // [1,1,6,dim] f32
        Ok((e, e0))
    }

    /// Per-token time embedding `e` `[1, L, dim]` (f32) + the 6-vector modulation `e0`
    /// `[1, L, 6, dim]` from a per-token timestep vector `t_tokens` `[1, L]` (the TI2V mask-blend
    /// path, sc-2680 — first-frame tokens frozen at `t=0`). Mirrors `WanModel.__call__`'s `t.ndim==2`
    /// branch (`sinusoidal_embedding_1d` over the `[1,L]` positions, then the same time MLP/projection).
    fn time_embed_tokens(&self, t_tokens: &Array) -> Result<(Array, Array)> {
        let l = t_tokens.shape()[1];
        let pos = t_tokens.reshape(&[1, l, 1])?; // [1, L, 1]
        let sinusoid = multiply(&pos, &self.inv_freq)?; // [1, L, half] f32
        let sin_emb = concatenate_axis(&[&cos(&sinusoid)?, &sin(&sinusoid)?], 2)?; // [1, L, freq_dim]
        let e = self
            .time_embedding_1
            .forward(&silu(&self.time_embedding_0.forward(&sin_emb)?)?)?; // [1, L, dim] f32
        let e0 =
            self.time_projection
                .forward(&silu(&e)?)?
                .reshape(&[1, l, 6, self.cfg.dim as i32])?; // [1, L, 6, dim] f32
        Ok((e, e0))
    }

    /// Output head: modulated LayerNorm + projection → `[1, L, out_dim·∏patch]` (f32). `e` is the
    /// time embedding `[1, dim]` (T2V) or per-token `[1, L, dim]` (TI2V, sc-2680); `head.modulation`
    /// `[1,2,dim]` broadcasts over `L_e` either way (the reference `Head.__call__`'s `e.ndim` branch).
    fn apply_head(&self, x: &Array, e: &Array) -> Result<Array> {
        let dim = self.cfg.dim as i32;
        let l_e = if e.shape().len() == 2 {
            1
        } else {
            e.shape()[1]
        };
        // head.modulation [1,2,dim] + e [1,L_e,1,dim] → [1,L_e,2,dim], split into shift e0 / scale e1.
        let m = add(&self.head_modulation, &e.reshape(&[1, l_e, 1, dim])?)?;
        let p = split(&m, 2, 2)?;
        let e0 = p[0].reshape(&[1, l_e, dim])?;
        let e1 = p[1].reshape(&[1, l_e, dim])?;
        let x_mod = add(
            &multiply(&ln(x, self.cfg.eps as f32)?, &add(&e1, scalar(1.0))?)?,
            &e0,
        )?;
        self.head.forward(&x_mod)
    }

    /// Per-stage capture for parity bisection: `(x_embed, e_time, e0_mod, x_block0, x_blocks,
    /// x_head)`. Mirrors [`forward`](Self::forward) but returns the intermediate hiddens.
    pub fn forward_capture(
        &self,
        latent: &Array,
        t: f32,
        context_embed: &Array,
    ) -> Result<Vec<Array>> {
        let (tokens, grid) = patchify(latent, self.cfg.patch_size)?;
        let l = (grid.0 * grid.1 * grid.2) as i32;
        let x_embed =
            bf16(&self.patch_embedding.forward(&tokens)?)?.reshape(&[1, l, self.cfg.dim as i32])?;
        let (e, e0) = self.time_embed(t)?;
        // bf16 cos/sin (the reference's `prepare_rope` builds them in the weight dtype = bf16).
        let (cos_t, sin_t) = self.rope.precompute_cos_sin(grid)?;
        let (cos_t, sin_t) = (bf16(&cos_t)?, bf16(&sin_t)?);
        let mut x = x_embed.clone();
        let mut x_block0 = x_embed.clone();
        for (i, block) in self.blocks.iter().enumerate() {
            let kv = block.prepare_kv(context_embed)?;
            x = block.forward(&x, &e0, &kv, &cos_t, &sin_t)?;
            if i == 0 {
                x_block0 = x.clone();
            }
        }
        let x_head = self.apply_head(&x, &e)?;
        Ok(vec![x_embed, e, e0, x_block0, x, x_head])
    }

    /// The patchify grid `(f, h, w)` for a latent `[C, F, H, W]` — constant across denoise steps (the
    /// channel count is irrelevant, so the I2V channel-concat `y` doesn't change it). Used to size the
    /// per-generate RoPE cache ([`prepare_rope`](Self::prepare_rope)).
    pub fn patch_grid(&self, latent: &Array) -> (usize, usize, usize) {
        let sh = latent.shape(); // [C, F, H, W]
        let (pt, ph, pw) = self.cfg.patch_size;
        (
            sh[1] as usize / pt,
            sh[2] as usize / ph,
            sh[3] as usize / pw,
        )
    }

    /// Precompute the **bf16** RoPE `(cos, sin)` for a constant grid — call once per generate, reuse
    /// across every denoise step (mirrors the reference's `prepare_rope`). The cos/sin depend only on
    /// the grid (not the weights), so they are identical for both MoE experts.
    pub fn prepare_rope(&self, grid: (usize, usize, usize)) -> Result<(Array, Array)> {
        let (cos_t, sin_t) = self.rope.precompute_cos_sin(grid)?;
        Ok((bf16(&cos_t)?, bf16(&sin_t)?))
    }

    /// Precompute every block's cross-attention K/V from the (CFG-batched) embedded context — call
    /// once per generate, reuse across all steps (mirrors the reference's `prepare_cross_kv`).
    /// `context_batch`: `[B, text_len, dim]` (bf16) from [`embed_text`](Self::embed_text), with the
    /// cond/uncond contexts stacked on the batch axis when CFG is on. Returns one `(k, v)` per block,
    /// each `[B, n, text_len, d]`.
    pub fn prepare_cross_kv(&self, context_batch: &Array) -> Result<Vec<(Array, Array)>> {
        self.blocks
            .iter()
            .map(|block| block.prepare_kv(context_batch))
            .collect()
    }

    /// Full DiT forward over a **single** latent shared across the CFG batch, reusing the per-generate
    /// RoPE + cross-K/V caches. `latent`: `[C, F, H, W]` (f32, already channel-concatenated with the
    /// I2V `y` by the caller). `t`: integer-valued timestep. `cross_kv`: per-block `(k, v)` from
    /// [`prepare_cross_kv`](Self::prepare_cross_kv) (`[batch, n, text_len, d]`). `cos`/`sin`: from
    /// [`prepare_rope`](Self::prepare_rope). `batch` is the cross-K/V batch width (2 for CFG, 1 for the
    /// cfg-disabled path). Returns one denoised `[out_dim, F, H, W]` (f32) **per batch element**
    /// (`[cond, uncond]` for CFG).
    ///
    /// The single latent is patchified once and broadcast to `batch` (the reference's `all_same`
    /// path) — the cond/uncond branches diverge only at the first cross-attention (different context
    /// K/V), so the `B=2` forward is bit-identical to two `B=1` forwards but launches each GPU kernel
    /// once instead of twice (the small-seq CFG win, sc-2853).
    /// The shared cached DiT forward (F-022): patchify → embed → broadcast over the CFG batch → block
    /// loop with the per-block modulation `e0` → head with `e` → per-batch unpatchify. The scalar
    /// [`forward_cached`](Self::forward_cached) and per-token
    /// [`forward_tokens_cached`](Self::forward_tokens_cached) paths differ **only** in how they build
    /// `(e, e0)` (scalar `time_embed` vs per-token `time_embed_tokens`), so they precompute it and pass
    /// it in here — keeping the ~45-line body in one place so the TI2V path can't silently diverge.
    #[allow(clippy::too_many_arguments)]
    fn forward_with_modulation(
        &self,
        latent: &Array,
        e: &Array,
        e0: &Array,
        cross_kv: &[(Array, Array)],
        cos: &Array,
        sin: &Array,
        batch: usize,
    ) -> Result<Vec<Array>> {
        // Patchify + embed once; cast to bf16 to start the block stream (reference casts to w_dtype).
        let (tokens, grid) = patchify(latent, self.cfg.patch_size)?;
        let l = (grid.0 * grid.1 * grid.2) as i32;
        let dim = self.cfg.dim as i32;
        let x1 = bf16(&self.patch_embedding.forward(&tokens)?)?.reshape(&[1, l, dim])?;
        // Broadcast the shared patch embedding across the CFG batch (the reference's `broadcast_to`).
        let mut x = if batch > 1 {
            broadcast_to(&x1, &[batch as i32, l, dim])?
        } else {
            x1
        };

        for (block, kv) in self.blocks.iter().zip(cross_kv.iter()) {
            x = block.forward(&x, e0, kv, cos, sin)?;
        }

        let x = self.apply_head(&x, e)?; // [batch, L, out_dim·∏patch] f32
        let op = x.shape()[2];

        // Unpatchify each batch element back to [out_dim, F, H, W].
        if batch == 1 {
            let xb = x.reshape(&[l, op])?;
            return Ok(vec![unpatchify(
                &xb,
                grid,
                self.cfg.out_dim,
                self.cfg.patch_size,
            )?]);
        }
        let mut out = Vec::with_capacity(batch);
        for part in split(&x, batch as i32, 0)? {
            let xb = part.reshape(&[l, op])?;
            out.push(unpatchify(
                &xb,
                grid,
                self.cfg.out_dim,
                self.cfg.patch_size,
            )?);
        }
        Ok(out)
    }

    pub fn forward_cached(
        &self,
        latent: &Array,
        t: f32,
        cross_kv: &[(Array, Array)],
        cos: &Array,
        sin: &Array,
        batch: usize,
    ) -> Result<Vec<Array>> {
        let (e, e0) = self.time_embed(t)?;
        self.forward_with_modulation(latent, &e, &e0, cross_kv, cos, sin, batch)
    }

    /// Full DiT forward for a single latent (B=1). `latent`: `[C, F, H, W]` (f32). `t`: integer-valued
    /// timestep. `context_embed`: `[1, text_len, dim]` (bf16) from [`embed_text`](Self::embed_text).
    /// Returns the denoised `[out_dim, F, H, W]` (f32).
    ///
    /// A convenience wrapper that builds the RoPE + cross-K/V caches on the fly and runs the B=1
    /// [`forward_cached`](Self::forward_cached) path — bit-identical to the cached denoise loop for a
    /// single branch. The denoise loops ([`denoise`](crate::pipeline::denoise) /
    /// [`denoise_moe`](crate::pipeline::denoise_moe)) build the caches once per generate instead.
    pub fn forward(&self, latent: &Array, t: f32, context_embed: &Array) -> Result<Array> {
        let grid = self.patch_grid(latent);
        let (cos_t, sin_t) = self.prepare_rope(grid)?;
        let cross_kv = self.prepare_cross_kv(context_embed)?;
        let mut preds = self.forward_cached(latent, t, &cross_kv, &cos_t, &sin_t, 1)?;
        preds
            .pop()
            .ok_or_else(|| Error::Msg("wan: forward_cached produced no output for batch=1".into()))
    }

    /// Patch-embed one latent `[C, F, H, W]` (f32) into the DiT token stream `[1, L, dim]` (bf16) +
    /// its patch grid `(f, h, w)` — the embedding half of [`forward_cached`] exposed as a seam for the
    /// Bernini renderer (sc-4706), which patch-embeds the noisy target **and** each conditioning source
    /// separately (each with its own source-id RoPE) and concatenates them on the token axis before a
    /// single packed forward. `L = f·h·w`.
    pub fn patch_embed_tokens(&self, latent: &Array) -> Result<(Array, (usize, usize, usize))> {
        let (tokens, grid) = patchify(latent, self.cfg.patch_size)?;
        let l = (grid.0 * grid.1 * grid.2) as i32;
        let x =
            bf16(&self.patch_embedding.forward(&tokens)?)?.reshape(&[1, l, self.cfg.dim as i32])?;
        Ok((x, grid))
    }

    /// Run the block stack + output head over a **pre-embedded, pre-packed** token sequence
    /// `tokens` `[1, L, dim]` (bf16) with caller-supplied RoPE `cos`/`sin` `[L, 1, half_d]` and the
    /// (scalar-timestep) cross-attention K/V cache — returning the per-token velocity
    /// `[1, L, out_dim·∏patch]` (f32) **without** unpatchifying. This is [`forward_cached`]'s body
    /// minus the (single-latent) patchify in / unpatchify out, the seam the Bernini renderer (sc-4706)
    /// uses: at batch 1 the packed `[sources…, target]` sequence is plain full self-attention (the
    /// reference's varlen attention with a single `cu_seqlens` segment), so the caller assembles the
    /// token + RoPE concat, calls this once, then slices the target tokens and unpatchifies them.
    pub fn forward_packed(
        &self,
        tokens: &Array,
        t: f32,
        cross_kv: &[(Array, Array)],
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        let (e, e0) = self.time_embed(t)?;
        let mut x = tokens.clone();
        for (block, kv) in self.blocks.iter().zip(cross_kv.iter()) {
            x = block.forward(&x, &e0, kv, cos, sin)?;
        }
        self.apply_head(&x, &e)
    }

    /// TI2V **per-token-timestep** batched DiT forward (sc-2680). Identical to [`forward_cached`](
    /// Self::forward_cached) but the timestep is a **per-token** vector `t_tokens` `[1, L]` (`L` = the
    /// patch-token count = the reference's `i2v_mask_tokens · timestep`), so each token carries its own
    /// modulation — the mask-blend path freezes the first-frame tokens at `t = 0`. The per-token time
    /// embedding `e0` `[1, L, 6, dim]` is shared across the CFG batch (cond/uncond use the same
    /// timesteps), exactly like the scalar path's `e0`, so it broadcasts over `B` and the cond/uncond
    /// branches diverge only at the first cross-attention. Mirrors `WanModel.__call__`'s `t.ndim == 2`
    /// branch, run through the same cached/compiled machinery as T2V.
    pub fn forward_tokens_cached(
        &self,
        latent: &Array,
        t_tokens: &Array,
        cross_kv: &[(Array, Array)],
        cos: &Array,
        sin: &Array,
        batch: usize,
    ) -> Result<Vec<Array>> {
        let (e, e0) = self.time_embed_tokens(t_tokens)?;
        self.forward_with_modulation(latent, &e, &e0, cross_kv, cos, sin, batch)
    }

    /// B=1 per-token convenience wrapper (builds the caches on the fly + runs [`forward_tokens_cached`]
    /// for a single branch) — bit-identical to the cached TI2V denoise loop for one branch. Used by the
    /// parity probes; the denoise loop ([`denoise_ti2v`](crate::pipeline::denoise_ti2v)) builds the
    /// caches once per generate.
    pub fn forward_tokens(
        &self,
        latent: &Array,
        t_tokens: &Array,
        context_embed: &Array,
    ) -> Result<Array> {
        let grid = self.patch_grid(latent);
        let (cos_t, sin_t) = self.prepare_rope(grid)?;
        let cross_kv = self.prepare_cross_kv(context_embed)?;
        let mut preds =
            self.forward_tokens_cached(latent, t_tokens, &cross_kv, &cos_t, &sin_t, 1)?;
        preds.pop().ok_or_else(|| {
            Error::Msg("wan: forward_tokens_cached produced no output for batch=1".into())
        })
    }
}
