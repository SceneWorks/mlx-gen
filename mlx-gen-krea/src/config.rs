//! `Krea2Transformer2DModel` configuration — parsed from a diffusers `transformer/config.json`, or
//! constructed directly via [`Krea2Config::turbo`].
//!
//! Mirrors the reference `Krea2Transformer2DModel.__init__` config surface (verified against the
//! published `krea/Krea-2-Turbo` `transformer/config.json` + safetensors index). The DiT is a **dense
//! single-stream** transformer: text + image tokens are concatenated and run through one stack of
//! `num_layers` gated single-stream blocks, with a `text_fusion` (`TextFusionTransformer`) front-end
//! that aggregates the 12 selected Qwen3-VL hidden layers down to one conditioning stream.
//!
//! Krea-2-Raw shares this architecture (only the DiT weights differ — distilled vs base), so one
//! config covers both surfaces (confirmed by the spike; re-checked when the Raw snapshot lands in
//! sc-7576).

use std::path::Path;

use mlx_gen::{Error, Result};

/// Architecture config for the Krea 2 dense single-stream DiT.
#[derive(Debug, Clone, PartialEq)]
pub struct Krea2Config {
    /// Patch-token input width to `img_in` (= `z_dim`·patch²; the latent is patchified 2×2 *outside*
    /// the DiT in the pipeline). The published `in_channels` is 64 = 16 latent ch × 2×2.
    pub in_channels: usize,
    /// Spatial patch size the pipeline applies before `img_in` (lives in `model_index.json`, not
    /// `transformer/config.json`; reference `Krea2Pipeline.patch_size`).
    pub patch_size: usize,
    /// Transformer width = `num_attention_heads`·`attention_head_dim` (not stored directly).
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub attention_head_dim: usize,
    /// Number of single-stream `transformer_blocks`.
    pub num_layers: usize,
    /// SwiGLU FFN inner width of a single-stream block.
    pub intermediate_size: usize,
    pub norm_eps: f32,
    /// Per-axis (t, h, w) RoPE sub-dimensions; must sum to `attention_head_dim`.
    pub axes_dims_rope: [usize; 3],
    pub rope_theta: f32,
    /// Sinusoidal timestep-embedding width fed to `time_embed.linear_1`.
    pub timestep_embed_dim: usize,

    // ── text_fusion (TextFusionTransformer) ────────────────────────────────────────────────────
    /// Number of Qwen3-VL hidden layers stacked + aggregated (the `text_encoder_select_layers` count).
    pub num_text_layers: usize,
    /// `layerwise_blocks`: attention ACROSS the `num_text_layers` axis (the learned aggregator).
    pub num_layerwise_text_blocks: usize,
    /// `refiner_blocks`: attention over the token axis, after the 12→1 projector collapse.
    pub num_refiner_text_blocks: usize,
    /// Qwen3-VL text hidden width (the text-fusion stream width).
    pub text_hidden_dim: usize,
    /// SwiGLU FFN inner width of a text-fusion block.
    pub text_intermediate_size: usize,
    pub text_num_attention_heads: usize,
    pub text_num_kv_heads: usize,
}

impl Krea2Config {
    /// Krea-2-Turbo / -Raw DiT architecture (verified from the published `transformer/config.json` +
    /// safetensors index: 430 tensors, hidden 6144, GQA 48Q/12KV, 28 single-stream blocks).
    pub fn turbo() -> Self {
        Self {
            in_channels: 64,
            patch_size: 2,
            hidden_size: 6144,
            num_attention_heads: 48,
            num_kv_heads: 12,
            attention_head_dim: 128,
            num_layers: 28,
            intermediate_size: 16384,
            norm_eps: 1e-5,
            axes_dims_rope: [32, 48, 48],
            rope_theta: 1000.0,
            timestep_embed_dim: 256,
            num_text_layers: 12,
            num_layerwise_text_blocks: 2,
            num_refiner_text_blocks: 2,
            text_hidden_dim: 2560,
            text_intermediate_size: 6912,
            text_num_attention_heads: 20,
            text_num_kv_heads: 20,
        }
    }

    /// Parse `<root>/transformer/config.json` (+ `<root>/model_index.json` for `patch_size`). Missing
    /// scalar fields fall back to [`Krea2Config::turbo`]; the validated invariants (RoPE-sum,
    /// head-divisibility, text width) are checked here.
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        let path = root.join("transformer").join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Msg(format!("krea: read {}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("krea: parse {}: {e}", path.display())))?;
        let mut cfg = Self::from_json(&v)?;
        // `patch_size` lives in the pipeline manifest, not the transformer config; read it if present.
        let idx = root.join("model_index.json");
        if let Ok(t) = std::fs::read_to_string(&idx) {
            if let Ok(mv) = serde_json::from_str::<serde_json::Value>(&t) {
                if let Some(p) = mv.get("patch_size").and_then(serde_json::Value::as_u64) {
                    cfg.patch_size = p as usize;
                }
            }
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Build from an already-parsed `transformer/config.json` value. `hidden_size` is derived from
    /// `num_attention_heads`·`attention_head_dim`; `patch_size` defaults to the reference 2.
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let d = Krea2Config::turbo();
        let u = |k: &str, dflt: usize| {
            v.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(dflt)
        };
        let f = |k: &str, dflt: f32| {
            v.get(k)
                .and_then(serde_json::Value::as_f64)
                .map(|n| n as f32)
                .unwrap_or(dflt)
        };

        let num_attention_heads = u("num_attention_heads", d.num_attention_heads);
        let attention_head_dim = u("attention_head_dim", d.attention_head_dim);

        let cfg = Self {
            in_channels: u("in_channels", d.in_channels),
            patch_size: d.patch_size,
            hidden_size: num_attention_heads * attention_head_dim,
            num_attention_heads,
            num_kv_heads: u("num_key_value_heads", d.num_kv_heads),
            attention_head_dim,
            num_layers: u("num_layers", d.num_layers),
            intermediate_size: u("intermediate_size", d.intermediate_size),
            norm_eps: f("norm_eps", d.norm_eps),
            axes_dims_rope: read_triple(v.get("axes_dims_rope"), d.axes_dims_rope),
            rope_theta: f("rope_theta", d.rope_theta),
            timestep_embed_dim: u("timestep_embed_dim", d.timestep_embed_dim),
            num_text_layers: u("num_text_layers", d.num_text_layers),
            num_layerwise_text_blocks: u("num_layerwise_text_blocks", d.num_layerwise_text_blocks),
            num_refiner_text_blocks: u("num_refiner_text_blocks", d.num_refiner_text_blocks),
            text_hidden_dim: u("text_hidden_dim", d.text_hidden_dim),
            text_intermediate_size: u("text_intermediate_size", d.text_intermediate_size),
            text_num_attention_heads: u("text_num_attention_heads", d.text_num_attention_heads),
            text_num_kv_heads: u("text_num_key_value_heads", d.text_num_kv_heads),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Invariants mirrored from the reference `__init__` + the published shapes.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size != self.num_attention_heads * self.attention_head_dim {
            return Err(Error::Msg(format!(
                "krea: hidden_size ({}) must equal num_attention_heads ({}) * attention_head_dim ({})",
                self.hidden_size, self.num_attention_heads, self.attention_head_dim
            )));
        }
        if self.attention_head_dim != self.axes_dims_rope.iter().sum::<usize>() {
            return Err(Error::Msg(format!(
                "krea: attention_head_dim ({}) must equal sum(axes_dims_rope) ({})",
                self.attention_head_dim,
                self.axes_dims_rope.iter().sum::<usize>()
            )));
        }
        if self.num_kv_heads == 0 || !self.num_attention_heads.is_multiple_of(self.num_kv_heads) {
            return Err(Error::Msg(format!(
                "krea: num_attention_heads ({}) not divisible by num_kv_heads ({})",
                self.num_attention_heads, self.num_kv_heads
            )));
        }
        if self.text_hidden_dim != self.text_num_attention_heads * self.attention_head_dim {
            return Err(Error::Msg(format!(
                "krea: text_hidden_dim ({}) must equal text_num_attention_heads ({}) * attention_head_dim ({})",
                self.text_hidden_dim, self.text_num_attention_heads, self.attention_head_dim
            )));
        }
        Ok(())
    }

    /// Query/KV projection widths (GQA): Q spans all heads, K/V the kv heads.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.attention_head_dim
    }
    /// Number of 6-factor modulation streams in `DoubleSharedModulation` (`time_mod_proj` →
    /// pre{scale,shift,gate} + post{scale,shift,gate}).
    pub const MOD_FACTORS: usize = 6;
    /// `time_mod_proj` output width = `MOD_FACTORS` · `hidden_size`.
    pub fn time_mod_dim(&self) -> usize {
        Self::MOD_FACTORS * self.hidden_size
    }
}

fn read_triple(v: Option<&serde_json::Value>, dflt: [usize; 3]) -> [usize; 3] {
    match v.and_then(serde_json::Value::as_array) {
        Some(a) if a.len() == 3 => {
            let mut out = dflt;
            for (i, x) in a.iter().enumerate() {
                if let Some(n) = x.as_u64() {
                    out[i] = n as usize;
                }
            }
            out
        }
        _ => dflt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turbo_config_invariants() {
        let c = Krea2Config::turbo();
        c.validate().unwrap();
        assert_eq!(c.hidden_size, 6144);
        assert_eq!(c.attention_head_dim, c.axes_dims_rope.iter().sum::<usize>());
        assert_eq!(c.q_dim(), 6144);
        assert_eq!(c.kv_dim(), 1536);
        assert_eq!(c.time_mod_dim(), 36864);
        assert_eq!(
            c.text_hidden_dim,
            c.text_num_attention_heads * c.attention_head_dim
        );
    }

    /// The exact published `krea/Krea-2-Turbo` `transformer/config.json` round-trips to [`turbo`].
    #[test]
    fn from_json_matches_published_turbo() {
        let v: serde_json::Value = serde_json::json!({
            "attention_head_dim": 128,
            "axes_dims_rope": [32, 48, 48],
            "in_channels": 64,
            "intermediate_size": 16384,
            "norm_eps": 1e-05,
            "num_attention_heads": 48,
            "num_key_value_heads": 12,
            "num_layers": 28,
            "num_layerwise_text_blocks": 2,
            "num_refiner_text_blocks": 2,
            "num_text_layers": 12,
            "rope_theta": 1000.0,
            "text_hidden_dim": 2560,
            "text_intermediate_size": 6912,
            "text_num_attention_heads": 20,
            "text_num_key_value_heads": 20,
            "timestep_embed_dim": 256
        });
        let c = Krea2Config::from_json(&v).unwrap();
        assert_eq!(c, Krea2Config::turbo());
    }

    #[test]
    fn bad_rope_sum_rejected() {
        let mut c = Krea2Config::turbo();
        c.axes_dims_rope = [32, 48, 49];
        assert!(c.validate().is_err());
    }

    #[test]
    fn bad_gqa_rejected() {
        let mut c = Krea2Config::turbo();
        c.num_kv_heads = 7; // 48 not divisible by 7
        assert!(c.validate().is_err());
    }
}
