//! UNet spatial transformer: `Transformer2D` (GroupNorm → linear `proj_in` → N `TransformerBlock`s
//! → linear `proj_out`, residual) and its `TransformerBlock` (self-attn → cross-attn → GEGLU FFN).
//! Port of the vendored `unet.Transformer2D` / `TransformerBlock`. SDXL uses linear `proj_in/out`
//! (`use_linear_projection`), no attention masks, and exact `gelu` in the GEGLU. NHWC I/O.

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, multiply};
use mlx_rs::transforms::checkpoint;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::array::scalar;
use mlx_gen::nn::{gelu_exact, group_norm};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-5;
const LN_EPS: f32 = 1e-5;

/// Multi-head attention as the vendored `nn.MultiHeadAttention`: q/k/v projections without bias,
/// output projection with bias, no mask. Used for both self-attention (context = `x`) and
/// cross-attention (context = the text `memory`).
#[derive(Clone)]
struct AttentionMHA {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    out: AdaptableLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    /// IP-Adapter decoupled cross-attention (sc-3059): extra bias-free K/V projections for the image
    /// tokens, installed only on the cross-attention (`attn2`) modules. When present and IP tokens
    /// are supplied, `o += scale · sdpa(q, to_k_ip(ip), to_v_ip(ip))` before the output projection
    /// (ref diffusers `IPAdapterAttnProcessor2_0`). The token source is the caller's (ViT-H→Resampler
    /// here; an ArcFace Resampler for InstantID sc-3113) — this is just the injection primitive.
    to_k_ip: Option<AdaptableLinear>,
    to_v_ip: Option<AdaptableLinear>,
    /// sc-4941 — run the SDPA segment inside an `mlx::checkpoint` so its backward recomputes the
    /// decomposed attention (MLX has no fused SDPA backward) instead of retaining the `[heads,s,s]`
    /// probability matrix. Training-only knob (opt-in via the U-Net's `set_sdpa_checkpoint`); for
    /// SDXL the seq² term is small (64²/32² grids), so this is gated behind `gradient_checkpointing`
    /// rather than always-on. Grads are bit-identical. Default off (inference unaffected).
    ckpt_sdpa: bool,
}

impl AttentionMHA {
    fn from_weights(w: &Weights, prefix: &str, model_dims: i32, num_heads: i32) -> Result<Self> {
        // Packed-detect (sc-8746): q/k/v (bias-free) + to_out.0 (biased) are all quantized in
        // [`Self::quantize`], so `crate::quant::lin` loads their packed triple or the dense weight.
        let no_bias = |n: &str| crate::quant::lin(w, &format!("{prefix}.{n}"), false);
        let head_dim = model_dims / num_heads;
        Ok(Self {
            q: no_bias("to_q")?,
            k: no_bias("to_k")?,
            v: no_bias("to_v")?,
            out: crate::quant::lin(w, &format!("{prefix}.to_out.0"), true)?,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            to_k_ip: None,
            to_v_ip: None,
            ckpt_sdpa: false,
        })
    }

    /// Toggle SDPA-segment gradient checkpointing (sc-4941). Training-only — see `ckpt_sdpa`.
    fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.ckpt_sdpa = on;
    }

    /// Cast the attention projections (and any IP K/V) to `dtype` (sc-4941 bf16 training).
    fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        for lin in [&mut self.q, &mut self.k, &mut self.v, &mut self.out] {
            lin.cast_weights(dtype)?;
        }
        for lin in [&mut self.to_k_ip, &mut self.to_v_ip].into_iter().flatten() {
            lin.cast_weights(dtype)?;
        }
        Ok(())
    }

    /// Install the IP-Adapter decoupled K/V projections (sc-3059). `k_ip`/`v_ip` are the
    /// `ip_adapter.{n}.to_k_ip/to_v_ip` weights (`[hidden, cross_attention_dim]`, bias-free).
    fn install_ip(&mut self, k_ip: Array, v_ip: Array) {
        self.to_k_ip = Some(AdaptableLinear::dense(k_ip, None));
        self.to_v_ip = Some(AdaptableLinear::dense(v_ip, None));
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        for lin in [&mut self.q, &mut self.k, &mut self.v, &mut self.out] {
            lin.quantize(bits, None)?;
        }
        for lin in [&mut self.to_k_ip, &mut self.to_v_ip].into_iter().flatten() {
            lin.quantize(bits, None)?;
        }
        Ok(())
    }

    /// `x`: `[B, L, D]` (queries); `context`: `[B, S, Dctx]` (keys/values; == `x` for self-attn).
    /// Fused `scaled_dot_product_attention` (mathematically the reference's `nn.MultiHeadAttention`;
    /// an explicit softmax matmul was tried and gave no measurable parity gain at large e2e cost).
    /// The four LoRA-targetable attention projections, by diffusers leaf name (the `.to_out.0`
    /// dot is the GEGLU-style indexed leaf the kohya flattener turns into `to_out_0`).
    fn lora_target_paths(&self, prefix: &str, out: &mut Vec<String>) {
        for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
            out.push(format!("{prefix}.{leaf}"));
        }
    }

    fn forward(&self, x: &Array, context: &Array) -> Result<Array> {
        self.forward_ip(x, context, None)
    }

    /// As [`forward`](Self::forward), plus the IP-Adapter branch when `ip = Some((tokens, scale))`
    /// and this module has IP projections installed: `o += scale · sdpa(q, to_k_ip(tokens),
    /// to_v_ip(tokens))`, sharing the query `q`, before the output projection.
    fn forward_ip(&self, x: &Array, context: &Array, ip: Option<(&Array, f32)>) -> Result<Array> {
        let (b, l) = (x.shape()[0], x.shape()[1]);
        let s = context.shape()[1];
        let to_heads = |a: Array, n: i32| -> Result<Array> {
            Ok(a.reshape(&[b, n, self.num_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_heads(self.q.forward(x)?, l)?;
        let k = to_heads(self.k.forward(context)?, s)?;
        let v = to_heads(self.v.forward(context)?, s)?;
        // 6th arg is `sinks`; `None` = standard attention. sc-4941: optionally checkpoint just the
        // SDPA so its backward recomputes the decomposed attention (q/k/v threaded as inputs, scale
        // captured) rather than retaining the seq² probability matrix. Grads stay bit-identical.
        let mut o = if self.ckpt_sdpa {
            let scale = self.scale;
            let mut seg = checkpoint(move |inp: &[Array]| -> mlx_rs::error::Result<Vec<Array>> {
                Ok(vec![scaled_dot_product_attention(
                    &inp[0], &inp[1], &inp[2], scale, None, None,
                )?])
            });
            seg(&[q.clone(), k.clone(), v.clone()])?
                .into_iter()
                .next()
                .ok_or_else(|| Error::Msg("sdxl: checkpoint SDPA produced no output".into()))?
        } else {
            scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?
        };

        // IP-Adapter decoupled cross-attention (shares the query `q`).
        if let (Some(k_ip), Some(v_ip), Some((tokens, scale))) = (&self.to_k_ip, &self.to_v_ip, ip)
        {
            let n_ip = tokens.shape()[1];
            let k_i = to_heads(k_ip.forward(tokens)?, n_ip)?;
            let v_i = to_heads(v_ip.forward(tokens)?, n_ip)?;
            let o_ip = scaled_dot_product_attention(&q, &k_i, &v_i, self.scale, None, None)?;
            o = add(
                &o,
                &multiply(&o_ip, &scalar(scale).as_dtype(o_ip.dtype())?)?,
            )?;
        }

        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, l, self.num_heads * self.head_dim])?;
        self.out.forward(&o)
    }
}

/// One spatial-transformer block: pre-norm self-attn, pre-norm cross-attn to the text memory, and a
/// pre-norm GEGLU FFN (`linear1(y) * gelu(linear2(y)) → linear3`). All residual.
#[derive(Clone)]
struct TransformerBlock {
    norm1_w: Array,
    norm1_b: Array,
    norm2_w: Array,
    norm2_b: Array,
    norm3_w: Array,
    norm3_b: Array,
    attn1: AttentionMHA,
    attn2: AttentionMHA,
    /// GEGLU value half (`ff.net.0.proj` rows `[0:hidden]`).
    linear1: AdaptableLinear,
    /// GEGLU gate half (`ff.net.0.proj` rows `[hidden:2*hidden]`).
    linear2: AdaptableLinear,
    /// FFN output (`ff.net.2`).
    linear3: AdaptableLinear,
}

impl TransformerBlock {
    fn from_weights(w: &Weights, prefix: &str, model_dims: i32, num_heads: i32) -> Result<Self> {
        // GEGLU: `ff.net.0.proj` is one `[2*hidden, D]` Linear on disk, row-split into value/gate
        // halves. Determine `2*hidden` from the packed `.scales` grid (rows) when present, else from
        // the dense weight — the split rows are identical either way (sc-8746). The packed row-slice
        // is byte-identical to the dense split-then-quantize (quantization is per-row).
        let ff_proj = format!("{prefix}.ff.net.0.proj");
        let two_h = match w.get(&format!("{ff_proj}.scales")) {
            Some(scales) => scales.shape()[0],
            None => w.require(&format!("{ff_proj}.weight"))?.shape()[0],
        };
        if two_h % 2 != 0 {
            return Err(Error::Msg(format!(
                "sdxl GEGLU: ff.net.0.proj has odd output dim {two_h}; cannot split value/gate halves"
            )));
        }
        let hidden = two_h / 2;
        // Value/gate halves (rows `[0:hidden]` / `[hidden:2*hidden]`) — packed or dense.
        let linear1 = crate::quant::lin_geglu_half(w, &ff_proj, 0, hidden)?;
        let linear2 = crate::quant::lin_geglu_half(w, &ff_proj, hidden, two_h)?;
        // FFN output (`ff.net.2`) is a plain quantized Linear.
        let linear3 = crate::quant::lin(w, &format!("{prefix}.ff.net.2"), true)?;
        let g = |n: &str| w.require(&format!("{prefix}.{n}")).cloned();
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            norm3_w: g("norm3.weight")?,
            norm3_b: g("norm3.bias")?,
            attn1: AttentionMHA::from_weights(
                w,
                &format!("{prefix}.attn1"),
                model_dims,
                num_heads,
            )?,
            attn2: AttentionMHA::from_weights(
                w,
                &format!("{prefix}.attn2"),
                model_dims,
                num_heads,
            )?,
            linear1,
            linear2,
            linear3,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn1.quantize(bits)?;
        self.attn2.quantize(bits)?;
        self.linear1.quantize(bits, None)?;
        self.linear2.quantize(bits, None)?;
        self.linear3.quantize(bits, None)?;
        Ok(())
    }

    /// Install this block's IP-Adapter K/V projections into its cross-attention (sc-3059).
    fn install_ip(&mut self, k_ip: Array, v_ip: Array) {
        self.attn2.install_ip(k_ip, v_ip);
    }

    /// Toggle SDPA-segment checkpointing on both attentions (sc-4941).
    fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.attn1.set_sdpa_checkpoint(on);
        self.attn2.set_sdpa_checkpoint(on);
    }

    /// Cast all dtype-bearing leaves (norms, attentions, GEGLU FFN) to `dtype` (sc-4941 bf16).
    fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        for a in [
            &mut self.norm1_w,
            &mut self.norm1_b,
            &mut self.norm2_w,
            &mut self.norm2_b,
            &mut self.norm3_w,
            &mut self.norm3_b,
        ] {
            super::cast_array(a, dtype)?;
        }
        self.attn1.cast_weights(dtype)?;
        self.attn2.cast_weights(dtype)?;
        self.linear1.cast_weights(dtype)?;
        self.linear2.cast_weights(dtype)?;
        self.linear3.cast_weights(dtype)?;
        Ok(())
    }

    /// Run the block. The cross-attention (`attn2`) also injects the IP-Adapter branch when `ip` is
    /// supplied (sc-3059); self-attention never gets IP. (No no-IP wrapper — callers always thread
    /// the `Option`, passing `None` when there's no IP-Adapter.)
    fn forward_ip(&self, x: &Array, memory: &Array, ip: Option<(&Array, f32)>) -> Result<Array> {
        // Self-attention.
        let y = layer_norm(x, Some(&self.norm1_w), Some(&self.norm1_b), LN_EPS)?;
        let x = add(x, &self.attn1.forward(&y, &y)?)?;
        // Cross-attention to the text memory (+ optional IP-Adapter branch).
        let y = layer_norm(&x, Some(&self.norm2_w), Some(&self.norm2_b), LN_EPS)?;
        let x = add(&x, &self.attn2.forward_ip(&y, memory, ip)?)?;
        // GEGLU FFN.
        let y = layer_norm(&x, Some(&self.norm3_w), Some(&self.norm3_b), LN_EPS)?;
        let y = multiply(
            &self.linear1.forward(&y)?,
            &gelu_exact(&self.linear2.forward(&y)?)?,
        )?;
        let y = self.linear3.forward(&y)?;
        Ok(add(&x, &y)?)
    }
}

/// A 2-D spatial transformer over NHWC features, cross-attending to the text `encoder_x`.
#[derive(Clone)]
pub struct Transformer2D {
    norm_w: Array,
    norm_b: Array,
    proj_in: AdaptableLinear,
    blocks: Vec<TransformerBlock>,
    proj_out: AdaptableLinear,
}

impl Transformer2D {
    /// `prefix` addresses the `attentions.{i}` module. `num_layers` = transformer blocks.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        model_dims: i32,
        num_heads: i32,
        num_layers: i32,
    ) -> Result<Self> {
        let blocks = (0..num_layers)
            .map(|i| {
                TransformerBlock::from_weights(
                    w,
                    &format!("{prefix}.transformer_blocks.{i}"),
                    model_dims,
                    num_heads,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            norm_w: w.require(&format!("{prefix}.norm.weight"))?.clone(),
            norm_b: w.require(&format!("{prefix}.norm.bias"))?.clone(),
            // Packed-detect (sc-8746): SDXL uses linear `proj_in`/`proj_out`, both quantized.
            proj_in: crate::quant::lin(w, &format!("{prefix}.proj_in"), true)?,
            blocks,
            proj_out: crate::quant::lin(w, &format!("{prefix}.proj_out"), true)?,
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.proj_in.quantize(bits, None)?;
        self.proj_out.quantize(bits, None)?;
        for b in &mut self.blocks {
            b.quantize(bits)?;
        }
        Ok(())
    }

    /// Toggle SDPA-segment checkpointing across every transformer block (sc-4941).
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        for b in &mut self.blocks {
            b.set_sdpa_checkpoint(on);
        }
    }

    /// Cast the GroupNorm, `proj_in`/`proj_out`, and every block to `dtype` (sc-4941 bf16).
    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        super::cast_array(&mut self.norm_w, dtype)?;
        super::cast_array(&mut self.norm_b, dtype)?;
        self.proj_in.cast_weights(dtype)?;
        self.proj_out.cast_weights(dtype)?;
        for b in &mut self.blocks {
            b.cast_weights(dtype)?;
        }
        Ok(())
    }

    /// `x`: NHWC `[B, H, W, C]`; `encoder_x`: text memory `[B, S, Dctx]`.
    pub fn forward(&self, x: &Array, encoder_x: &Array) -> Result<Array> {
        self.forward_ip(x, encoder_x, None)
    }

    /// As [`forward`](Self::forward) but threads the IP-Adapter tokens + scale into each block's
    /// cross-attention (sc-3059).
    pub fn forward_ip(
        &self,
        x: &Array,
        encoder_x: &Array,
        ip: Option<(&Array, f32)>,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, h, w_, c) = (sh[0], sh[1], sh[2], sh[3]);
        let y = group_norm(x, &self.norm_w, &self.norm_b, GN_GROUPS, GN_EPS)?;
        let mut y = self.proj_in.forward(&y.reshape(&[b, h * w_, c])?)?;
        for block in &self.blocks {
            y = block.forward_ip(&y, encoder_x, ip)?;
        }
        let y = self.proj_out.forward(&y)?.reshape(&[b, h, w_, c])?;
        Ok(add(&y, x)?)
    }

    /// Install IP-Adapter K/V projections into each block's cross-attention, consuming one
    /// `(to_k_ip, to_v_ip)` pair per block from `pairs` (sc-3059). Pairs are consumed in block order.
    pub fn install_ip(&mut self, pairs: &mut impl Iterator<Item = (Array, Array)>) -> Result<()> {
        for block in &mut self.blocks {
            let (k_ip, v_ip) = pairs
                .next()
                .ok_or_else(|| mlx_gen::Error::Msg("ip_adapter: not enough K/V pairs".into()))?;
            block.install_ip(k_ip, v_ip);
        }
        Ok(())
    }

    /// LoRA-targetable Linears under this `attentions.{i}` module, by diffusers path: `proj_in`,
    /// `proj_out`, and each transformer block's attention projections. The GEGLU FF (`linear1/2/3`)
    /// is intentionally excluded — the vendored `lora.py` can't reach it (mlx-examples renames it),
    /// so faithfully porting that path omits it too (sc-2671 adds it).
    pub fn lora_target_paths(&self, prefix: &str, out: &mut Vec<String>) {
        out.push(format!("{prefix}.proj_in"));
        out.push(format!("{prefix}.proj_out"));
        for (k, b) in self.blocks.iter().enumerate() {
            b.attn1
                .lora_target_paths(&format!("{prefix}.transformer_blocks.{k}.attn1"), out);
            b.attn2
                .lora_target_paths(&format!("{prefix}.transformer_blocks.{k}.attn2"), out);
        }
    }

    /// The GEGLU feed-forward LoRA targets (diffusers naming) under this `attentions.{i}` module:
    /// each block's `ff.net.0.proj` (the fused value+gate proj, row-split across `linear1`/`linear2`
    /// at merge) and `ff.net.2`. Kept separate from [`Transformer2D::lora_target_paths`] because the
    /// vendored `lora.py` can't reach the FF (mlx-examples renames it) — complete coverage (sc-2671)
    /// adds it on top of the faithful surface.
    pub fn lora_target_paths_ff(&self, prefix: &str, out: &mut Vec<String>) {
        for k in 0..self.blocks.len() {
            out.push(format!("{prefix}.transformer_blocks.{k}.ff.net.0.proj"));
            out.push(format!("{prefix}.transformer_blocks.{k}.ff.net.2"));
        }
    }
}

// LoRA key→module routing (sc-2639). Diffusers leaf naming; the GEGLU FF and (for the U-Net)
// mid_block are intentionally unreachable here to mirror the vendored `lora.py` surface.
impl AdaptableHost for AttentionMHA {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["to_q"] => Some(&mut self.q),
            ["to_k"] => Some(&mut self.k),
            ["to_v"] => Some(&mut self.v),
            ["to_out", "0"] => Some(&mut self.out),
            _ => None,
        }
    }
}

impl AdaptableHost for TransformerBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["attn1", rest @ ..] => self.attn1.adaptable_mut(rest),
            ["attn2", rest @ ..] => self.attn2.adaptable_mut(rest),
            // GEGLU FF (sc-2671 complete coverage). The diffusers `ff.net.0.proj` is row-split into
            // `linear1` (value) + `linear2` (gate) and `ff.net.2` is `linear3`; the SDXL adapter
            // merge translates those diffusers FF keys into these internal `ff.linearN` names (and
            // row-splits a `ff.net.0.proj` delta across linear1/linear2). Unreachable under the
            // vendored coverage (the merge gates FF keys out there).
            ["ff", "linear1"] => Some(&mut self.linear1),
            ["ff", "linear2"] => Some(&mut self.linear2),
            ["ff", "linear3"] => Some(&mut self.linear3),
            _ => None,
        }
    }
}

impl AdaptableHost for Transformer2D {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["proj_in"] => Some(&mut self.proj_in),
            ["proj_out"] => Some(&mut self.proj_out),
            ["transformer_blocks", k, rest @ ..] => self
                .blocks
                .get_mut(k.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            _ => None,
        }
    }
}
