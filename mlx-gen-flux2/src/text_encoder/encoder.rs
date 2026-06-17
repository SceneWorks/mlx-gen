//! Full Qwen3 text encoder: token embedding → 36 pre-norm decoder layers, collecting the
//! intermediate hidden states. `prompt_embeds` concatenates the outputs of layers 9/18/27 into
//! the transformer's conditioning. Port of the fork's `Qwen3TextEncoder.get_prompt_embeds`.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::host_i32;
use mlx_gen::nn::TokenEmbedding;
use mlx_gen::runtime::CancelFlag;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use super::generate::{sample_token, Qwen3KvCache, SplitMix64, UpsampleSampling};
use super::{join, lin, Qwen3DecoderLayer, TextRope};
use crate::config::Flux2Quant;

/// Decoder-LM text-encoder dimensions. Covers both FLUX.2 text encoders: klein's **Qwen3**
/// (`klein_9b`) and dev's **Mistral** (`mistral_dev`) — they share the GQA + SwiGLU + HF-RoPE
/// graph and differ only in these fields (chiefly `qk_norm`, the dims, θ, and layer count).
pub struct Qwen3TextEncoderConfig {
    pub hidden_size: i32,
    pub n_layers: usize,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Per-head q/k RMSNorm before RoPE — Qwen3 has it, Mistral does not.
    pub qk_norm: bool,
    /// Hidden-state indices (into a list whose entry 0 is the token embedding) concatenated into
    /// `prompt_embeds`. klein Qwen3: (9, 18, 27) → 3·4096 = 12288; dev Mistral: (10, 20, 30) →
    /// 3·5120 = 15360.
    pub out_layers: [usize; 3],
}

impl Qwen3TextEncoderConfig {
    pub fn klein_9b() -> Self {
        Self {
            hidden_size: 4096,
            n_layers: 36,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            qk_norm: true,
            out_layers: [9, 18, 27],
        }
    }

    /// FLUX.2-dev `text_encoder` — the **Mistral** language tower of `Mistral3ForConditionalGeneration`
    /// (text_config: 40 layers, hidden 5120, GQA 32/8, head_dim 128, θ=1e9, eps 1e-5, no qk-norm).
    /// The vision tower + projector are not part of the T2I path (sc-5918). Hidden states 10/20/30
    /// are concatenated into the 15360-wide `prompt_embeds` (dev pipeline `_get_mistral_3_small_prompt_embeds`).
    pub fn mistral_dev() -> Self {
        Self {
            hidden_size: 5120,
            n_layers: 40,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            rope_theta: 1_000_000_000.0,
            rms_norm_eps: 1e-5,
            qk_norm: false,
            out_layers: [10, 20, 30],
        }
    }
}

pub struct Qwen3TextEncoder {
    embed_tokens: TokenEmbedding,
    layers: Vec<Qwen3DecoderLayer>,
    rope: TextRope,
    out_layers: [usize; 3],
    rms_norm_eps: f32,
    /// Final RMSNorm + LM head — loaded only for the **autoregressive-generate** path (FLUX.2-dev
    /// caption upsampling, sc-6030) via [`load_generation_head`](Self::load_generation_head). The
    /// `prompt_embeds` T2I path never touches them (it discards the final norm and has no LM head),
    /// so klein and the dev T2I/edit encoders leave these `None` until upsampling needs them.
    norm: Option<Array>,
    lm_head: Option<AdaptableLinear>,
}

impl Qwen3TextEncoder {
    /// Loads from the on-disk `model.*` tree under `prefix` (`"model"`):
    /// `{prefix}.embed_tokens.weight`, `{prefix}.layers.{i}.…`. The final `{prefix}.norm.weight`
    /// is intentionally **not** loaded — the fork computes the final norm but `get_prompt_embeds`
    /// discards it, using only the raw (pre-final-norm) intermediate layer outputs.
    /// Load from a **dense** weight map (the parity-test + dense-snapshot path).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Qwen3TextEncoderConfig) -> Result<Self> {
        Self::from_weights_quant(w, prefix, cfg, None)
    }

    /// Load the encoder, building each Linear and the token embedding from packed Q4/Q8 parts when
    /// `quant` is `Some` AND the on-disk weights carry the packed `.scales`/`.biases` (a
    /// pre-quantized snapshot, sc-5917). `quant == None` ⇒ the dense path. The packed path never
    /// materializes a dense bf16 weight, so the dev Mistral TE loads at its Q4 footprint (~13 GB)
    /// rather than the ~45 GB bf16 load transient.
    pub fn from_weights_quant(
        w: &Weights,
        prefix: &str,
        cfg: &Qwen3TextEncoderConfig,
        quant: Option<Flux2Quant>,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(Qwen3DecoderLayer::from_weights(
                w,
                &join(prefix, &format!("layers.{i}")),
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
                cfg.qk_norm,
                quant,
            )?);
        }
        Ok(Self {
            embed_tokens: load_embed(w, &join(prefix, "embed_tokens"), quant)?,
            layers,
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            out_layers: cfg.out_layers,
            rms_norm_eps: cfg.rms_norm_eps,
            norm: None,
            lm_head: None,
        })
    }

    /// Load the final RMSNorm + LM head so this encoder can run the caption-upsampling
    /// `generate()` loop (sc-6030), in addition to the `prompt_embeds` extraction it already does.
    /// `parent_prefix` is the `Mistral3ForConditionalGeneration` language-model root (`language_model`):
    /// the final norm is `{parent}.model.norm.weight` and the LM head is `{parent}.lm_head.weight`.
    /// The LM head is dense bf16 in the dev snapshot (no packed `.scales`), so [`lin`] returns the
    /// dense path even with `quant = Some`; the final norm stays full precision. Idempotent-ish: a
    /// second call reloads.
    pub fn load_generation_head(
        &mut self,
        w: &Weights,
        parent_prefix: &str,
        quant: Option<Flux2Quant>,
    ) -> Result<()> {
        self.norm = Some(
            w.require(&join(parent_prefix, "model.norm.weight"))?
                .clone(),
        );
        self.lm_head = Some(lin(w, &join(parent_prefix, "lm_head.weight"), quant)?);
        Ok(())
    }

    /// `true` once [`load_generation_head`](Self::load_generation_head) has run — i.e. this encoder
    /// can drive the caption-upsampling decode.
    pub fn can_generate(&self) -> bool {
        self.norm.is_some() && self.lm_head.is_some()
    }

    /// Quantize the text encoder to Q4/Q8 (group_size 64): the token embedding + every layer's
    /// q/k/v/o + gate/up/down linears — the full set the fork's `nn.quantize(text_encoder, …)` hits
    /// (`nn.Embedding` + `nn.Linear`). RMSNorms stay full precision; the final `norm` is never loaded.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        // `cast_to_bf16 = true`: byte-match the fork's bf16 `nn.quantize` (no-op for the bf16-native
        // checkpoint) — the FLUX.2 behaviour preserved by the shared `TokenEmbedding` (F-083).
        self.embed_tokens.quantize(bits, true)?;
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// Test-only (sc-2643 byte-parity gate): the quantized `(wq, scales, biases, group_size, bits)`
    /// of the token `embed_tokens` — the unique `nn.Embedding` → `QuantizedEmbedding` case. `None` if
    /// the encoder is still dense.
    #[doc(hidden)]
    pub fn probe_quant_embed(&self) -> Option<(&Array, &Array, &Array, i32, i32)> {
        match &self.embed_tokens {
            TokenEmbedding::Quantized {
                wq,
                scales,
                biases,
                group_size,
                bits,
            } => Some((wq, scales, biases, *group_size, *bits)),
            TokenEmbedding::Dense(_) => None,
        }
    }

    /// Token embedding (f32): `input_ids` `[b, s]` int32 → `[b, s, hidden]`. `pub(crate)` so the
    /// caption-upsampling driver can embed the prompt ids before splicing the projected image
    /// features into them (sc-6030).
    pub(crate) fn embed(&self, input_ids: &Array) -> Result<Array> {
        self.embed_tokens.forward(input_ids)
    }

    /// `input_ids` / `attention_mask`: `[b, s]` int32. Returns `prompt_embeds`
    /// `[b, s, 3·hidden]` (f32): the layer-9/18/27 hidden states concatenated along the feature
    /// axis. Equivalent to the fork's `stack(axis=1) → transpose(0,2,1,3) → reshape` (which, per
    /// position, lays the three layers out contiguously = a feature-axis concat).
    pub fn prompt_embeds(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;

        // hidden_states index 0 = embeddings; index k = output of layer k-1. Only states a/b_/c are
        // used (9/18/27), and the final norm is never applied — so run layers only up to the highest
        // needed index and keep just those states, not all 37 (F-098): layers past max(out_layers)
        // cannot influence the result. Bit-identical to running all 36 + indexing.
        let [a, b_, c] = self.out_layers;
        let max_idx = a.max(b_).max(c);
        let needed = [a, b_, c];

        let mut hidden = self.embed(input_ids)?;
        let mut saved: Vec<(usize, Array)> = Vec::with_capacity(3);
        if needed.contains(&0) {
            saved.push((0, hidden.clone()));
        }
        for (i, layer) in self.layers.iter().take(max_idx).enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            let idx = i + 1;
            if needed.contains(&idx) {
                saved.push((idx, hidden.clone()));
            }
        }

        let pick = |idx: usize| -> Result<&Array> {
            saved
                .iter()
                .find(|(k, _)| *k == idx)
                .map(|(_, v)| v)
                .ok_or_else(|| Error::Msg(format!("flux2 te: hidden state {idx} not captured")))
        };
        Ok(concatenate_axis(&[pick(a)?, pick(b_)?, pick(c)?], 2)?)
    }

    // ---- Autoregressive caption-upsampling generate (sc-6030) ----------------------------------

    /// One generation forward: run pre-embedded tokens `[1, q_len, hidden]` at absolute `offset`,
    /// append each layer's K/V to `cache`, and return logits for the **last** position `[1, vocab]`.
    /// Requires [`load_generation_head`](Self::load_generation_head). All-layers causal decode (no
    /// early-stop, unlike `prompt_embeds`): the LM head consumes the final layer's normed output.
    pub(crate) fn decode_logits_from_embeds(
        &self,
        input_embeds: &Array,
        cache: &mut Qwen3KvCache,
        offset: i32,
    ) -> Result<Array> {
        let norm = self.norm.as_ref().ok_or_else(gen_head_missing)?;
        let lm_head = self.lm_head.as_ref().ok_or_else(gen_head_missing)?;
        let sh = input_embeds.shape();
        let (b, q_len, hidden) = (sh[0], sh[1], sh[2]);
        let (cos, sin) = self.rope.forward_offset(q_len, offset)?;

        let mut hidden_states = input_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward_step(&hidden_states, &cos, &sin, cache, i)?;
        }

        let last_idx = Array::from_slice(&[q_len - 1], &[1]);
        let last = hidden_states
            .take_axis(&last_idx, 1)?
            .reshape(&[b, hidden])?;
        let normed = rms_norm(&last, norm, self.rms_norm_eps)?;
        lm_head.forward(&normed)
    }

    /// [`decode_logits_from_embeds`](Self::decode_logits_from_embeds) from token ids `[1, q_len]`.
    pub(crate) fn decode_logits(
        &self,
        input_ids: &Array,
        cache: &mut Qwen3KvCache,
        offset: i32,
    ) -> Result<Array> {
        let embeds = self.embed(input_ids)?;
        self.decode_logits_from_embeds(&embeds, cache, offset)
    }

    /// Autoregressive generation from already-built prompt embeds `[1, prompt_len, hidden]` (the
    /// caption-upsampling prompt, whose embeds carry the spliced image features). Returns the
    /// generated token ids with the `eos_token` excluded. Stops at `eos_token` or
    /// `max_new_tokens`; honors `cancel`. Evals per step so the lazy graph (and the growing K/V
    /// cache) stays bounded over the up-to-512-token loop — the same per-step-eval discipline the
    /// denoise loop uses (sc-5522).
    pub fn generate_from_embeds(
        &self,
        prompt_embeds: &Array,
        eos_token: i32,
        sampling: UpsampleSampling,
        cancel: &CancelFlag,
    ) -> Result<Vec<i32>> {
        let sh = prompt_embeds.shape();
        if sh.len() != 3 || sh[0] != 1 {
            return Err(Error::Msg(format!(
                "flux2 caption-upsample: prompt embeds must be [1, seq, hidden], got {sh:?}"
            )));
        }
        if !self.can_generate() {
            return Err(gen_head_missing());
        }
        let prompt_len = sh[1];
        let mut cache = Qwen3KvCache::new(self.layers.len());
        let mut rng = SplitMix64::new(sampling.seed);
        let mut generated: Vec<i32> = Vec::new();

        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let mut logits = self.decode_logits_from_embeds(prompt_embeds, &mut cache, 0)?;
        logits.eval()?;

        for step in 0..sampling.max_new_tokens {
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let next = sample_token(&logits, &sampling, &mut rng)?;
            if next == eos_token {
                break;
            }
            generated.push(next);
            if step + 1 == sampling.max_new_tokens {
                break;
            }
            let token = Array::from_slice(&[next], &[1, 1]);
            logits = self.decode_logits(&token, &mut cache, prompt_len + step as i32)?;
            logits.eval()?;
        }
        Ok(generated)
    }
}

/// The error the generate path returns when [`load_generation_head`](Qwen3TextEncoder::load_generation_head)
/// has not run — a programming error (the dev caption loader always loads the head), surfaced loudly
/// rather than panicking.
fn gen_head_missing() -> Error {
    Error::Msg(
        "flux2 caption-upsample: the generation head (final norm + lm_head) is not loaded; \
         call load_generation_head first"
            .to_owned(),
    )
}

/// Load the `embed_tokens` table dense, or — with `quant == Some` and packed `.scales` on disk
/// (pre-quantized snapshot, sc-5917) — directly from the packed `{base}.weight` (u32 codes) /
/// `.scales` / `.biases`, mirroring [`lin`](super::lin) for the embedding case. `base` is the
/// embedding's prefix (`….embed_tokens`).
fn load_embed(w: &Weights, base: &str, quant: Option<Flux2Quant>) -> Result<TokenEmbedding> {
    if let Some(q) = quant {
        if let Some(scales) = w.get(&format!("{base}.scales")) {
            return Ok(TokenEmbedding::from_quantized_parts(
                w.require(&format!("{base}.weight"))?.clone(),
                scales.clone(),
                w.require(&format!("{base}.biases"))?.clone(),
                q.group_size,
                q.bits,
            ));
        }
    }
    Ok(TokenEmbedding::Dense(
        w.require(&format!("{base}.weight"))?.clone(),
    ))
}

/// Additive attention mask `[b, 1, s, s]`: `0` where a query may attend (key is causal **and**
/// not padding), `-inf` otherwise. Built host-side (one-time `O(b·s²)` fill per encode).
fn build_mask(attention_mask: &Array, b: i32, s: i32) -> Result<Array> {
    let am = host_i32(attention_mask)?;
    let (b, s) = (b as usize, s as usize);
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        for i in 0..s {
            for j in 0..s {
                let allowed = j <= i && am[bi * s + j] == 1;
                if !allowed {
                    data[(bi * s + i) * s + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Ok(Array::from_slice(&data, &[b as i32, 1, s as i32, s as i32]))
}
