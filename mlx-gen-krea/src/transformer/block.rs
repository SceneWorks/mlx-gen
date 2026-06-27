//! Krea 2 DiT building blocks — the reference `mmdit.py` modules: the sigmoid-**gated** GQA attention
//! (`Attention` + `QKNorm`), the `SwiGLU` FFN, the `+1` `RMSNorm`, the un-modulated `TextFusionBlock`,
//! the `DoubleSharedModulation` single-stream block, and the `TextFusionTransformer` layer aggregator.
//!
//! Every `RMSNorm` here computes `weight = scale + 1` in f32 (the reference stores the raw `scale`,
//! centered at 0 — verified against the real Turbo weights), distinct from boogu's apply-weight-directly
//! norms. Attention adds a `to_gate` projection: the post-attention output is multiplied by
//! `sigmoid(to_gate(x))` before `to_out`. Block gates (`pregate`/`postgate`) are raw (no activation).

use mlx_rs::error::Result as MlxResult;
use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, multiply, sigmoid, split};
use mlx_rs::transforms::checkpoint;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{prefixed_paths, AdaptableHost, AdaptableLinear};
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::rope::apply_interleaved_rope;
use super::{join, repeat_kv};
use crate::quant::lin;

/// `1.0 + a`, broadcasting the scalar (the `(1 + scale)` modulation factor).
fn plus1(a: &Array) -> Result<Array> {
    Ok(add(a, Array::from_f32(1.0))?)
}

// ── `+1` RMSNorm ────────────────────────────────────────────────────────────────────────────
/// Reference `RMSNorm`: `F.rms_norm(x.float(), weight = scale.float() + 1.0)` then cast back. The
/// stored param is the raw `scale` (centered at 0); we pre-fold the `+1` into an f32 weight at load and
/// always reduce in f32 (the reference upcasts), preserving the input dtype on the way out.
#[derive(Clone)]
pub struct RmsScale {
    weight: Array, // f32, = scale + 1
    eps: f32,
}

impl RmsScale {
    pub fn from_weights(w: &Weights, key: &str, eps: f32) -> Result<Self> {
        let scale = w.require(key)?.as_dtype(Dtype::Float32)?;
        Ok(Self {
            weight: plus1(&scale)?,
            eps,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let dt = x.dtype();
        let y = rms_norm(&x.as_dtype(Dtype::Float32)?, &self.weight, self.eps)?;
        Ok(y.as_dtype(dt)?)
    }
}

// ── Sigmoid-gated GQA attention (reference `Attention`) ─────────────────────────────────────
#[derive(Clone)]
pub struct GatedAttention {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    gate: AdaptableLinear,
    o: AdaptableLinear,
    norm_q: RmsScale,
    norm_k: RmsScale,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    scale: f32,
    /// SDPA-segment gradient checkpointing (sc-7577, training only; the z-image / Lens pattern). When
    /// `true`, the fused `scaled_dot_product_attention` runs inside an `mlx::checkpoint` so its backward
    /// recomputes the attention rather than retaining the `[heads, s, s]` probability matrix (the
    /// dominant seq² term; MLX decomposes the fused SDPA to naive attention for the backward).
    /// Numerically identical; off in every inference path (default `false`), set by the trainer via
    /// [`Self::set_sdpa_checkpoint`] — which turns it OFF when whole-block checkpointing already covers
    /// the recompute.
    sdpa_checkpoint: bool,
}

impl GatedAttention {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            q: lin(w, &join(prefix, "to_q"), false)?,
            k: lin(w, &join(prefix, "to_k"), false)?,
            v: lin(w, &join(prefix, "to_v"), false)?,
            gate: lin(w, &join(prefix, "to_gate"), false)?,
            o: lin(w, &join(prefix, "to_out.0"), false)?,
            norm_q: RmsScale::from_weights(w, &join(prefix, "norm_q.weight"), eps)?,
            norm_k: RmsScale::from_weights(w, &join(prefix, "norm_k.weight"), eps)?,
            heads,
            kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            sdpa_checkpoint: false,
        })
    }

    /// Toggle SDPA-segment gradient checkpointing (sc-7577, training only). See the field docs.
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.sdpa_checkpoint = on;
    }

    /// Cast the projection weights to the training compute `dtype` in place (sc-7577). The `RmsScale`
    /// q/k norms stay f32 (they always reduce in f32). Inference never calls this.
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        for p in [
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.gate,
            &mut self.o,
        ] {
            p.cast_weights(dtype)?;
        }
        Ok(())
    }

    /// `x`: `[b, s, hidden]`. `rope`: `Some((cos, sin))` (`[1, s, head_dim/2]`) for the single-stream
    /// blocks; `None` for the text-fusion blocks (no positional encoding). Unmasked (B=1 full sequence).
    pub fn forward(&self, x: &Array, rope: Option<(&Array, &Array)>) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let q = self
            .q
            .forward(x)?
            .reshape(&[b, s, self.heads, self.head_dim])?;
        let k = self
            .k
            .forward(x)?
            .reshape(&[b, s, self.kv_heads, self.head_dim])?;
        let v = self
            .v
            .forward(x)?
            .reshape(&[b, s, self.kv_heads, self.head_dim])?;
        let gate = self.gate.forward(x)?; // [b, s, hidden]

        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;
        let (q, k) = match rope {
            Some((cos, sin)) => (
                apply_interleaved_rope(&q, cos, sin)?,
                apply_interleaved_rope(&k, cos, sin)?,
            ),
            None => (q, k),
        };

        let groups = self.heads / self.kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let q = q.transpose_axes(&[0, 2, 1, 3])?; // [b, heads, s, hd]
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        // SDPA — optionally inside an `mlx::checkpoint` segment (training memory hardening): the
        // backward recomputes the attention rather than retaining the seq² probability matrix.
        // Numerically identical to the retained path; off in inference.
        let o = if self.sdpa_checkpoint {
            let scale = self.scale;
            let mut seg = checkpoint(move |inp: &[Array]| -> MlxResult<Vec<Array>> {
                Ok(vec![scaled_dot_product_attention(
                    &inp[0], &inp[1], &inp[2], scale, None, None,
                )?])
            });
            seg(&[q, k, v])?.into_iter().next().ok_or_else(|| {
                mlx_gen::Error::Msg("krea: SDPA checkpoint produced no output".into())
            })?
        } else {
            scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?
        };
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, self.heads * self.head_dim])?;

        // Sigmoid gate the attention output, then the shared output projection.
        let gated = multiply(&o, &sigmoid(&gate)?)?;
        self.o.forward(&gated)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for p in [
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.gate,
            &mut self.o,
        ] {
            p.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        }
        Ok(())
    }
}

/// LoRA/LoKr target routing for the gated attention (sc-7577 / sc-7578): the diffusers leaf names
/// `to_q`/`to_k`/`to_v`/`to_gate`/`to_out.0` (the trained-file naming an applied Krea LoRA uses).
impl AdaptableHost for GatedAttention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            // Diffusers leaf names (our converter/trainer) and the native Krea-2 names ai-toolkit
            // keys its LoRAs to (`wq`/`wk`/`wv`/`wo`/`gate`) are interchangeable aliases (sc-8185).
            ["to_q" | "wq"] => Some(&mut self.q),
            ["to_k" | "wk"] => Some(&mut self.k),
            ["to_v" | "wv"] => Some(&mut self.v),
            ["to_gate" | "gate"] => Some(&mut self.gate),
            ["to_out", "0"] | ["wo"] => Some(&mut self.o),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["to_q", "to_k", "to_v", "to_gate", "to_out.0"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
}

// ── SwiGLU feed-forward (reference `SwiGLU`: `down(silu(gate(x)) * up(x))`) ──────────────────
#[derive(Clone)]
pub struct SwiGlu {
    gate: AdaptableLinear,
    up: AdaptableLinear,
    down: AdaptableLinear,
}

impl SwiGlu {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: lin(w, &join(prefix, "gate"), false)?,
            up: lin(w, &join(prefix, "up"), false)?,
            down: lin(w, &join(prefix, "down"), false)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gated = multiply(&silu(&self.gate.forward(x)?)?, &self.up.forward(x)?)?;
        self.down.forward(&gated)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.gate.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.up.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.down.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        Ok(())
    }

    /// Cast the projection weights to the training compute `dtype` in place (sc-7577).
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.gate.cast_weights(dtype)?;
        self.up.cast_weights(dtype)?;
        self.down.cast_weights(dtype)?;
        Ok(())
    }
}

/// LoRA/LoKr target routing for the SwiGLU FFN (sc-7577 / sc-7578): leaves `gate`/`up`/`down`.
impl AdaptableHost for SwiGlu {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["gate"] => Some(&mut self.gate),
            ["up"] => Some(&mut self.up),
            ["down"] => Some(&mut self.down),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["gate", "up", "down"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
}

// ── Un-modulated text-fusion block (reference `TextFusionBlock`) ─────────────────────────────
/// `x = x + attn(prenorm(x)); x = x + mlp(postnorm(x))`. No modulation, no RoPE.
pub struct TextFusionBlock {
    prenorm: RmsScale,
    postnorm: RmsScale,
    attn: GatedAttention,
    mlp: SwiGlu,
}

impl TextFusionBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            prenorm: RmsScale::from_weights(w, &join(prefix, "norm1.weight"), eps)?,
            postnorm: RmsScale::from_weights(w, &join(prefix, "norm2.weight"), eps)?,
            attn: GatedAttention::from_weights(
                w,
                &join(prefix, "attn"),
                heads,
                kv_heads,
                head_dim,
                eps,
            )?,
            mlp: SwiGlu::from_weights(w, &join(prefix, "ff"))?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = add(x, &self.attn.forward(&self.prenorm.forward(x)?, None)?)?;
        Ok(add(&x, &self.mlp.forward(&self.postnorm.forward(&x)?)?)?)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.mlp.quantize(bits)
    }

    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.attn.set_sdpa_checkpoint(on);
    }

    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.attn.cast_weights(dtype)?;
        self.mlp.cast_weights(dtype)
    }
}

/// LoRA target routing for a text-fusion block: `attn.{…}` / `ff.{…}` (sc-7577 / sc-7578).
impl AdaptableHost for TextFusionBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_mut(rest),
            // `ff` (diffusers) ≡ `mlp` (native ai-toolkit) (sc-8185).
            ["ff" | "mlp", rest @ ..] => self.mlp.adaptable_mut(rest),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = prefixed_paths("attn", &self.attn);
        out.extend(prefixed_paths("ff", &self.mlp));
        out
    }
}

// ── DoubleSharedModulation single-stream block (reference `SingleStreamBlock`) ──────────────
/// `mod(tvec) = tvec + scale_shift_table` → 6 chunks `(prescale, preshift, pregate, postscale,
/// postshift, postgate)`; then
/// `x += pregate · attn((1+prescale)·prenorm(x) + preshift)` and
/// `x += postgate · mlp((1+postscale)·postnorm(x) + postshift)`. Gates are raw (no activation).
#[derive(Clone)]
pub struct SingleStreamBlock {
    scale_shift_table: Array, // [1, 1, 6·hidden]
    prenorm: RmsScale,
    postnorm: RmsScale,
    attn: GatedAttention,
    mlp: SwiGlu,
}

impl SingleStreamBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        hidden: i32,
        eps: f32,
    ) -> Result<Self> {
        // Stored `[6, hidden]`; flatten row-major to `[1, 1, 6·hidden]` so a single broadcast-add onto
        // `tvec` (`[b, 1, 6·hidden]`) and a 6-way split reproduce the reference's `chunk(6, -1)` order.
        let sst = w
            .require(&join(prefix, "scale_shift_table"))?
            .reshape(&[1, 1, 6 * hidden])?;
        Ok(Self {
            scale_shift_table: sst,
            prenorm: RmsScale::from_weights(w, &join(prefix, "norm1.weight"), eps)?,
            postnorm: RmsScale::from_weights(w, &join(prefix, "norm2.weight"), eps)?,
            attn: GatedAttention::from_weights(
                w,
                &join(prefix, "attn"),
                heads,
                kv_heads,
                head_dim,
                eps,
            )?,
            mlp: SwiGlu::from_weights(w, &join(prefix, "ff"))?,
        })
    }

    /// `x`: `[b, s, hidden]`, `tvec`: `[b, 1, 6·hidden]` (shared `time_mod_proj` output), `cos`/`sin`:
    /// `[1, s, head_dim/2]`.
    pub fn forward(&self, x: &Array, tvec: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let m = add(tvec, &self.scale_shift_table)?; // [b, 1, 6·hidden]
        let m = split(&m, 6, 2)?; // 6 × [b, 1, hidden]
        let (prescale, preshift, pregate) = (&m[0], &m[1], &m[2]);
        let (postscale, postshift, postgate) = (&m[3], &m[4], &m[5]);

        let pre = add(
            &multiply(&self.prenorm.forward(x)?, &plus1(prescale)?)?,
            preshift,
        )?;
        let attn = self.attn.forward(&pre, Some((cos, sin)))?;
        let x = add(x, &multiply(pregate, &attn)?)?;

        let post = add(
            &multiply(&self.postnorm.forward(&x)?, &plus1(postscale)?)?,
            postshift,
        )?;
        let mlp = self.mlp.forward(&post)?;
        Ok(add(&x, &multiply(postgate, &mlp)?)?)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.mlp.quantize(bits)
    }

    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.attn.set_sdpa_checkpoint(on);
    }

    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        if self.scale_shift_table.dtype() != dtype {
            self.scale_shift_table = self.scale_shift_table.as_dtype(dtype)?;
        }
        self.attn.cast_weights(dtype)?;
        self.mlp.cast_weights(dtype)
    }
}

/// LoRA target routing for a single-stream block: `attn.{…}` / `ff.{…}` (sc-7577 / sc-7578) — the
/// trainable attention + FFN projections. The `scale_shift_table` / norms are not adapter targets.
impl AdaptableHost for SingleStreamBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_mut(rest),
            // `ff` (diffusers) ≡ `mlp` (native ai-toolkit) (sc-8185).
            ["ff" | "mlp", rest @ ..] => self.mlp.adaptable_mut(rest),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = prefixed_paths("attn", &self.attn);
        out.extend(prefixed_paths("ff", &self.mlp));
        out
    }
}

// ── TextFusionTransformer (reference `TextFusionTransformer`) ────────────────────────────────
/// Aggregates the `num_layers` stacked Qwen3-VL hidden states into one conditioning stream:
/// `layerwise_blocks` attend across the layer axis (per token) → `projector` collapses `num_layers→1`
/// → `refiner_blocks` attend across the token axis.
pub struct TextFusionTransformer {
    layerwise: Vec<TextFusionBlock>,
    projector: AdaptableLinear, // Linear(num_layers → 1), no bias
    refiner: Vec<TextFusionBlock>,
}

impl TextFusionTransformer {
    pub fn from_weights(
        w: &Weights,
        num_layerwise: usize,
        num_refiner: usize,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        let block = |i: usize, kind: &str| {
            TextFusionBlock::from_weights(
                w,
                &format!("text_fusion.{kind}.{i}"),
                heads,
                kv_heads,
                head_dim,
                eps,
            )
        };
        Ok(Self {
            layerwise: (0..num_layerwise)
                .map(|i| block(i, "layerwise_blocks"))
                .collect::<Result<_>>()?,
            projector: lin(w, "text_fusion.projector", false)?,
            refiner: (0..num_refiner)
                .map(|i| block(i, "refiner_blocks"))
                .collect::<Result<_>>()?,
        })
    }

    /// `x`: `[b, n_tokens, num_layers, txt_dim]` (the stacked select-layer hidden states). Returns the
    /// fused conditioning `[b, n_tokens, txt_dim]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, n_tok, n_layers, d) = (sh[0], sh[1], sh[2], sh[3]);

        // Layerwise attention: each token's `num_layers` stack is a sequence (batch = b·n_tokens).
        let mut h = x.reshape(&[b * n_tok, n_layers, d])?;
        for blk in &self.layerwise {
            h = blk.forward(&h)?;
        }

        // `(b n_tok) n_layers d -> b n_tok d n_layers`, project `num_layers → 1`, drop the axis.
        let h = h
            .reshape(&[b, n_tok, n_layers, d])?
            .transpose_axes(&[0, 1, 3, 2])?; // [b, n_tok, d, n_layers]
        let h = self
            .projector
            .forward(&h.reshape(&[b * n_tok * d, n_layers])?)?; // [b·n_tok·d, 1]
        let mut h = h.reshape(&[b, n_tok, d])?;

        // Token-axis refinement.
        for blk in &self.refiner {
            h = blk.forward(&h)?;
        }
        Ok(h)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for b in &mut self.layerwise {
            b.quantize(bits)?;
        }
        for b in &mut self.refiner {
            b.quantize(bits)?;
        }
        Ok(())
    }

    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        for b in &mut self.layerwise {
            b.set_sdpa_checkpoint(on);
        }
        for b in &mut self.refiner {
            b.set_sdpa_checkpoint(on);
        }
    }

    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        for b in &mut self.layerwise {
            b.cast_weights(dtype)?;
        }
        self.projector.cast_weights(dtype)?;
        for b in &mut self.refiner {
            b.cast_weights(dtype)?;
        }
        Ok(())
    }
}

/// LoRA target routing for the text-fusion aggregator (sc-7577 / sc-7578): the per-block attention +
/// FFN of the `layerwise_blocks` / `refiner_blocks`, plus the `projector` collapse linear.
impl AdaptableHost for TextFusionTransformer {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["layerwise_blocks", n, rest @ ..] => self
                .layerwise
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["refiner_blocks", n, rest @ ..] => self
                .refiner
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["projector"] => Some(&mut self.projector),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, b) in self.layerwise.iter().enumerate() {
            out.extend(prefixed_paths(&format!("layerwise_blocks.{i}"), b));
        }
        for (i, b) in self.refiner.iter().enumerate() {
            out.extend(prefixed_paths(&format!("refiner_blocks.{i}"), b));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense `[out, in]` weight at `key` (values irrelevant — these tests route by shape, never
    /// forward), so each projection gets a distinct `out` dim we can match on.
    fn put(w: &mut Weights, key: &str, out: i32, in_: i32) {
        let n = (out * in_) as usize;
        w.insert(key, Array::from_slice(&vec![0f32; n], &[out, in_]));
    }

    /// A `GatedAttention` whose five projections have distinct output dims (q=11, k=12, v=13,
    /// gate=14, o=15), so [`AdaptableLinear::base_shape`]`()[0]` identifies which projection a path
    /// resolved to.
    fn gated() -> GatedAttention {
        let mut w = Weights::empty();
        put(&mut w, "to_q.weight", 11, 8);
        put(&mut w, "to_k.weight", 12, 8);
        put(&mut w, "to_v.weight", 13, 8);
        put(&mut w, "to_gate.weight", 14, 8);
        put(&mut w, "to_out.0.weight", 15, 8);
        w.insert("norm_q.weight", Array::from_slice(&[0f32; 4], &[4]));
        w.insert("norm_k.weight", Array::from_slice(&[0f32; 4], &[4]));
        GatedAttention::from_weights(&w, "", 2, 1, 4, 1e-6).unwrap()
    }

    /// sc-8185: the native ai-toolkit (ostris) attn leaf names (`wq`/`wk`/`wv`/`wo`/`gate`) must
    /// route to the *same* projections as the diffusers names our converter/trainer emit — in
    /// particular `wo` → `to_out.0`. Distinct output dims prove correct routing, not just a match.
    #[test]
    fn gated_attention_accepts_native_aitoolkit_aliases() {
        for (canon, native, out) in [
            (["to_q"].as_slice(), ["wq"].as_slice(), 11),
            (["to_k"].as_slice(), ["wk"].as_slice(), 12),
            (["to_v"].as_slice(), ["wv"].as_slice(), 13),
            (["to_gate"].as_slice(), ["gate"].as_slice(), 14),
            (["to_out", "0"].as_slice(), ["wo"].as_slice(), 15),
        ] {
            let mut a = gated();
            assert_eq!(
                a.adaptable_mut(canon).unwrap().base_shape()[0],
                out,
                "canonical {canon:?} routed to the wrong projection"
            );
            assert_eq!(
                a.adaptable_mut(native).unwrap().base_shape()[0],
                out,
                "native {native:?} must route to the same projection as {canon:?}"
            );
        }
    }

    #[test]
    fn gated_attention_rejects_unknown_leaf() {
        assert!(gated().adaptable_mut(&["nope"]).is_none());
    }
}
