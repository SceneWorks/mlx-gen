//! sc-5132: native Qwen2.5-VL-7B planner **LLM backbone** — a stateless feature extractor.
//!
//! Port of the text decoder of `_vendor/bernini/bernini/models/modeling_qwen2_5_vl.py`
//! (`Qwen2_5_VLModel`, architecturally stock HF Qwen2.5-VL). The Bernini planner runs this as a
//! single forward pass over `inputs_embeds` and taps `hidden_states[-2]` (the penultimate residual
//! stream — the input to the final decoder layer, pre-final-norm). There is **no** token generation,
//! KV-cache, or `lm_head` here — those are dropped (the `lm_head` tensor is dropped at conversion,
//! sc-5144).
//!
//! Deltas vs the sensenova Qwen3 backbone this adapts: **single path** (no dual MoT stacks), attention
//! `q/k/v_proj` carry a **bias** while `o_proj` does not, there is **no q/k-norm**, and the rotary is
//! the net-new **3D multimodal RoPE** (`apply_multimodal_rotary_pos_emb`) driven by externally
//! supplied `(3, L)` position ids and an externally supplied additive 4D attention mask (the planner's
//! flex mask — text/in-vit causal, gen-target bidirectional — not a hardcoded causal mask).
//!
//! f32 islands match the reference: RMSNorm reduction (via `mlx_rs::fast::rms_norm`), the rotary
//! table, and the attention softmax (`precise = true`). Linears are [`AdaptableLinear`]s so sc-5146
//! can quantize them Q4/Q8 at load.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, matmul, multiply, softmax_axis, split, split_sections,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// Text-decoder config for the Qwen2.5-VL-7B planner backbone (the LLM half of `mllm/config.json`).
#[derive(Clone, Debug)]
pub struct QwenVlTextConfig {
    pub hidden_size: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// The 3 MRoPE channel sections (temporal, height, width); sum·2 = head_dim.
    pub mrope_section: [usize; 3],
}

impl Default for QwenVlTextConfig {
    /// Qwen2.5-VL-7B-Instruct (the Bernini planner base).
    fn default() -> Self {
        Self {
            hidden_size: 3584,
            num_layers: 28,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            intermediate_size: 18944,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            mrope_section: [16, 24, 24],
        }
    }
}

impl QwenVlTextConfig {
    /// Read from a `qwen2_5_vl_config.json` (the sc-5144 snapshot copy of `mllm/config.json`). The
    /// text fields live at the top level; `head_dim = hidden_size / num_attention_heads`.
    pub fn from_config_json(path: &std::path::Path) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| Error::Msg(format!("parse {}: {e}", path.display())))?;
        let i = |k: &str, d: i64| v.get(k).and_then(serde_json::Value::as_i64).unwrap_or(d) as i32;
        let f = |k: &str, d: f64| v.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d) as f32;
        let hidden_size = i("hidden_size", 3584);
        let num_heads = i("num_attention_heads", 28);
        let mrope = v
            .get("rope_scaling")
            .and_then(|r| r.get("mrope_section"))
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                let g = |idx: usize, d: usize| {
                    a.get(idx)
                        .and_then(serde_json::Value::as_u64)
                        .map(|x| x as usize)
                        .unwrap_or(d)
                };
                [g(0, 16), g(1, 24), g(2, 24)]
            })
            .unwrap_or([16, 24, 24]);
        Ok(Self {
            hidden_size,
            num_layers: i("num_hidden_layers", 28),
            num_heads,
            num_kv_heads: i("num_key_value_heads", 4),
            head_dim: hidden_size / num_heads,
            intermediate_size: i("intermediate_size", 18944),
            rms_norm_eps: f("rms_norm_eps", 1e-6),
            rope_theta: f("rope_theta", 1_000_000.0),
            mrope_section: mrope,
        })
    }
}

fn require(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?.clone())
}

/// A Linear with optional bias from `{prefix}.weight` (+ `{prefix}.bias`), quantizable.
fn linear(w: &Weights, prefix: &str, bias: bool) -> Result<AdaptableLinear> {
    let weight = require(w, &format!("{prefix}.weight"))?;
    let b = if bias {
        Some(require(w, &format!("{prefix}.bias"))?)
    } else {
        None
    };
    Ok(AdaptableLinear::dense(weight, b))
}

/// Per-layer attention: `q/k/v_proj` carry bias, `o_proj` does not (Qwen2.5-VL; no q/k-norm).
struct Attn {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
}

impl Attn {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            q: linear(w, &format!("{prefix}.q_proj"), true)?,
            k: linear(w, &format!("{prefix}.k_proj"), true)?,
            v: linear(w, &format!("{prefix}.v_proj"), true)?,
            o: linear(w, &format!("{prefix}.o_proj"), false)?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.q.quantize(bits, None)?;
        self.k.quantize(bits, None)?;
        self.v.quantize(bits, None)?;
        self.o.quantize(bits, None)
    }
}

/// SwiGLU MLP (bias-free), the stock Qwen2 MLP.
struct Mlp {
    gate: AdaptableLinear,
    up: AdaptableLinear,
    down: AdaptableLinear,
}

impl Mlp {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: linear(w, &format!("{prefix}.gate_proj"), false)?,
            up: linear(w, &format!("{prefix}.up_proj"), false)?,
            down: linear(w, &format!("{prefix}.down_proj"), false)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gated = multiply(&silu(&self.gate.forward(x)?)?, &self.up.forward(x)?)?;
        self.down.forward(&gated)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.gate.quantize(bits, None)?;
        self.up.quantize(bits, None)?;
        self.down.quantize(bits, None)
    }
}

struct Layer {
    input_ln: Array,
    post_ln: Array,
    attn: Attn,
    mlp: Mlp,
}

impl Layer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            input_ln: require(w, &format!("{prefix}.input_layernorm.weight"))?,
            post_ln: require(w, &format!("{prefix}.post_attention_layernorm.weight"))?,
            attn: Attn::from_weights(w, &format!("{prefix}.self_attn"))?,
            mlp: Mlp::from_weights(w, &format!("{prefix}.mlp"))?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.mlp.quantize(bits)
    }
}

/// HF half-split rotary `rotate_half`: `cat(-x[d/2:], x[:d/2])` on the last axis.
fn rotate_half(x: &Array) -> Result<Array> {
    let ax = x.ndim() as i32 - 1;
    let parts = split(x, 2, ax)?;
    Ok(concatenate_axis(&[&parts[1].negative()?, &parts[0]], ax)?)
}

/// The native Qwen2.5-VL-7B text decoder, run as a stateless penultimate-hidden-state extractor.
pub struct Qwen25VlText {
    embed_tokens: Array,
    layers: Vec<Layer>,
    norm: Array,
    cfg: QwenVlTextConfig,
}

impl Qwen25VlText {
    /// Build from a converted planner snapshot's `qwen2_5_vl.safetensors` (keys `model.*`). `prefix`
    /// is the model namespace — `"model"` for the sc-5144 layout.
    pub fn from_weights(w: &Weights, cfg: QwenVlTextConfig, prefix: &str) -> Result<Self> {
        let layers = (0..cfg.num_layers)
            .map(|i| Layer::from_weights(w, &format!("{prefix}.layers.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: require(w, &format!("{prefix}.embed_tokens.weight"))?,
            layers,
            norm: require(w, &format!("{prefix}.norm.weight"))?,
            cfg,
        })
    }

    pub fn config(&self) -> &QwenVlTextConfig {
        &self.cfg
    }

    /// Quantize every decoder layer's attention projections + SwiGLU linears (Q4/Q8, group 64). The
    /// token embedding and RMSNorms stay dense. (sc-5146 load-time quantization.)
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// Token embedding: `input_ids` `[B,L]` int32 → `[B,L,hidden]` (the reference's
    /// `get_input_embeddings()`), preserving the embedding dtype (bf16).
    pub fn embed(&self, input_ids: &Array) -> Result<Array> {
        let sh = input_ids.shape().to_vec();
        let flat = input_ids.reshape(&[-1])?;
        let g = self.embed_tokens.take_axis(&flat, 0)?;
        let h = self.embed_tokens.shape()[1];
        Ok(g.reshape(&[sh[0], sh[1], h])?)
    }

    /// Assemble the multimodal rotary `(cos, sin)` for the layer apply, each `[1, L, head_dim]` in
    /// `dtype`. `position_ids` is `[3, L]` int32 (temporal / height / width rows).
    ///
    /// Mirrors `Qwen2_5_VLRotaryEmbedding` + the channel interleave of
    /// `apply_multimodal_rotary_pos_emb`: one rotary table per axis (`inv_freq[j] =
    /// theta^(-2j/head_dim)`, `emb = cat(freqs, freqs)`), then the 128 channels are stitched from the
    /// three axes by the doubled `mrope_section` `[16,24,24,16,24,24]`, chunk `i` taking axis `i%3`.
    pub fn mrope_cos_sin(&self, position_ids: &Array, dtype: Dtype) -> Result<(Array, Array)> {
        let hd = self.cfg.head_dim as usize;
        let half = hd / 2;
        let theta = self.cfg.rope_theta;
        let inv_freq: Vec<f32> = (0..half)
            .map(|j| 1.0f32 / theta.powf((2 * j) as f32 / hd as f32))
            .collect();
        let inv = Array::from_slice(&inv_freq, &[1, half as i32]);

        let l = position_ids.shape()[1];
        let pos_f32 = position_ids.as_dtype(Dtype::Float32)?; // [3, L]
        let rows = split(&pos_f32, 3, 0)?; // 3 × [1, L]

        // Doubled-section channel split points: [16,24,24,16,24,24] → cut points [16,40,64,80,104].
        let s = self.cfg.mrope_section;
        let doubled = [s[0], s[1], s[2], s[0], s[1], s[2]];
        let mut pts = Vec::with_capacity(5);
        let mut acc = 0i32;
        for &d in doubled.iter().take(5) {
            acc += d as i32;
            pts.push(acc);
        }

        // One rotary table per axis (its own position row), each split into the 6 channel pieces.
        let mut cos_pieces: Vec<Vec<Array>> = Vec::with_capacity(3);
        let mut sin_pieces: Vec<Vec<Array>> = Vec::with_capacity(3);
        for row in rows.iter() {
            let p = row.reshape(&[l, 1])?; // [L, 1]
            let freqs = matmul(&p, &inv)?; // [L, half]
            let emb = concatenate_axis(&[&freqs, &freqs], 1)?; // [L, head_dim]
            let cos = emb.cos()?.expand_dims(0)?; // [1, L, head_dim]
            let sin = emb.sin()?.expand_dims(0)?;
            cos_pieces.push(split_sections(&cos, &pts, 2)?);
            sin_pieces.push(split_sections(&sin, &pts, 2)?);
        }
        // Stitch: channel chunk `i` is taken from axis `i % 3` (`apply_multimodal_rotary_pos_emb`).
        let cos_sel: Vec<&Array> = (0..6).map(|i| &cos_pieces[i % 3][i]).collect();
        let sin_sel: Vec<&Array> = (0..6).map(|i| &sin_pieces[i % 3][i]).collect();
        let cos = concatenate_axis(&cos_sel, 2)?.as_dtype(dtype)?; // [1, L, head_dim]
        let sin = concatenate_axis(&sin_sel, 2)?.as_dtype(dtype)?;
        Ok((cos, sin))
    }

    /// Apply MRoPE to a `[B, L, H, head_dim]` projection given assembled `cos`/`sin` `[1, L, head_dim]`
    /// (broadcast over the head axis): `x*cos + rotate_half(x)*sin`.
    fn apply_mrope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let cos = cos.expand_dims(2)?; // [1, L, 1, head_dim]
        let sin = sin.expand_dims(2)?;
        Ok(add(
            &multiply(x, &cos)?,
            &multiply(&rotate_half(x)?, &sin)?,
        )?)
    }

    /// Expand `[B,L,Hkv,D]` → `[B,L,Hkv*groups,D]` (GQA repeat).
    fn repeat_kv(x: &Array, groups: i32) -> Result<Array> {
        if groups == 1 {
            return Ok(x.clone());
        }
        let s = x.shape();
        let (b, l, hkv, d) = (s[0], s[1], s[2], s[3]);
        let x = x.expand_dims(3)?;
        let x = broadcast_to(&x, &[b, l, hkv, groups, d])?;
        Ok(x.reshape(&[b, l, hkv * groups, d])?)
    }

    /// Eager attention with an external additive 4D `mask` (`[*,*,L,L]`, broadcast): q/k/v project →
    /// reshape to heads → MRoPE q,k → GQA → `softmax(q·kᵀ/√d + mask)·v` (f32 softmax) → o_proj.
    fn attention(
        &self,
        x: &Array,
        a: &Attn,
        cos: &Array,
        sin: &Array,
        mask: &Array,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let hd = self.cfg.head_dim;
        let (nh, nkv) = (self.cfg.num_heads, self.cfg.num_kv_heads);

        let q = a.q.forward(x)?.reshape(&[b, s, nh, hd])?;
        let k = a.k.forward(x)?.reshape(&[b, s, nkv, hd])?;
        let v = a.v.forward(x)?.reshape(&[b, s, nkv, hd])?;
        let q = Self::apply_mrope(&q, cos, sin)?;
        let k = Self::apply_mrope(&k, cos, sin)?;

        let groups = nh / nkv;
        let q = q.transpose_axes(&[0, 2, 1, 3])?; // [B,H,L,D]
        let k = Self::repeat_kv(&k, groups)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = Self::repeat_kv(&v, groups)?.transpose_axes(&[0, 2, 1, 3])?;

        let scale = (hd as f32).powf(-0.5);
        let scores = multiply(
            &matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?,
            Array::from_f32(scale),
        )?;
        let scores = add(&scores, mask)?;
        let weights = softmax_axis(&scores, -1, true)?; // f32 accumulation
        let out = matmul(&weights, &v)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, nh * hd])?;
        a.o.forward(&out)
    }

    /// Full stateless forward. `embeds` `[B,L,hidden]`; `position_ids` `[3,L]` int32; `mask` an
    /// additive 4D attention mask (`[1,1,L,L]` or `[B,1,L,L]`, `0`/`-inf`).
    ///
    /// Returns **all** hidden states exactly as HF `output_hidden_states=True`: `[embeds,
    /// layer0_out, …, layer_{N-2}_out, final_norm(layer_{N-1}_out)]` — `N+1` entries. The planner's
    /// penultimate tap is [`Self::penultimate`] (`[-2]` = the input to the final decoder layer).
    pub fn forward(
        &self,
        embeds: &Array,
        position_ids: &Array,
        mask: &Array,
    ) -> Result<Vec<Array>> {
        let (cos, sin) = self.mrope_cos_sin(position_ids, embeds.dtype())?;
        let eps = self.cfg.rms_norm_eps;
        let mut hidden = embeds.clone();
        let mut all = Vec::with_capacity(self.layers.len() + 1);
        for layer in &self.layers {
            all.push(hidden.clone()); // HF appends the pre-layer hidden state
            let normed = rms_norm(&hidden, &layer.input_ln, eps)?;
            hidden = add(
                &hidden,
                &self.attention(&normed, &layer.attn, &cos, &sin, mask)?,
            )?;
            let normed = rms_norm(&hidden, &layer.post_ln, eps)?;
            hidden = add(&hidden, &layer.mlp.forward(&normed)?)?;
        }
        all.push(rms_norm(&hidden, &self.norm, eps)?);
        Ok(all)
    }

    /// The planner feature: the penultimate hidden state `hidden_states[-2]` (the residual stream
    /// feeding the final decoder layer, pre-final-norm) — `[B,L,hidden]`.
    pub fn penultimate(&self, embeds: &Array, position_ids: &Array, mask: &Array) -> Result<Array> {
        let all = self.forward(embeds, position_ids, mask)?;
        Ok(all[all.len() - 2].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MRoPE channel stitch: for a text segment all three position rows are equal, so the assembled
    /// cos/sin must equal a plain 1D rotary table (the reference note: "the three rotary position
    /// index of text embedding is always the same → no difference with modern LLMs"). We check that
    /// property holds, and that the table has the right shape.
    #[test]
    fn mrope_text_equals_1d() {
        let cfg = QwenVlTextConfig::default();
        let backbone = Qwen25VlText {
            embed_tokens: Array::zeros::<f32>(&[8, cfg.hidden_size]).unwrap(),
            layers: Vec::new(),
            norm: Array::ones::<f32>(&[cfg.hidden_size]).unwrap(),
            cfg: cfg.clone(),
        };
        let l = 5;
        // All three axes share the same positions [0..5) → a pure-text segment.
        let row: Vec<i32> = (0..l).collect();
        let mut data = Vec::new();
        for _ in 0..3 {
            data.extend_from_slice(&row);
        }
        let pos = Array::from_slice(&data, &[3, l]);
        let (cos, sin) = backbone.mrope_cos_sin(&pos, Dtype::Float32).unwrap();
        assert_eq!(cos.shape(), &[1, l, cfg.head_dim]);

        // A plain 1D rotary table over the same positions.
        let half = (cfg.head_dim / 2) as usize;
        let inv: Vec<f32> = (0..half)
            .map(|j| 1.0f32 / cfg.rope_theta.powf((2 * j) as f32 / cfg.head_dim as f32))
            .collect();
        let inv = Array::from_slice(&inv, &[1, half as i32]);
        let p = Array::from_slice(&row.iter().map(|&x| x as f32).collect::<Vec<_>>(), &[l, 1]);
        let freqs = matmul(&p, &inv).unwrap();
        let emb = concatenate_axis(&[&freqs, &freqs], -1).unwrap();
        let cos1d = emb.cos().unwrap().expand_dims(0).unwrap();

        let diff = mlx_rs::ops::subtract(&cos, &cos1d)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(
            diff < 1e-6,
            "text MRoPE must equal 1D rotary, max|Δ|={diff}"
        );
        let _ = sin;
    }

    /// rotate_half is the NeoX half-split: `[a,b,c,d] → [-c,-d,a,b]`.
    #[test]
    fn rotate_half_neox() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);
        let r = rotate_half(&x).unwrap();
        let got: Vec<f32> = r.flatten(None, None).unwrap().as_slice::<f32>().to_vec();
        assert_eq!(got, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    /// The default config matches Qwen2.5-VL-7B (head_dim derived = 128, GQA groups = 7).
    #[test]
    fn config_shapes() {
        let c = QwenVlTextConfig::default();
        assert_eq!(c.head_dim, c.hidden_size / c.num_heads);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.num_heads / c.num_kv_heads, 7);
        assert_eq!(
            c.mrope_section.iter().sum::<usize>() * 2,
            c.head_dim as usize
        );
    }
}
