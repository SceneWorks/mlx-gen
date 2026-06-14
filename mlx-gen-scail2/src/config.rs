//! SCAIL-2 model configuration: the shared Wan2.1-14B-I2V dimensions
//! ([`WanModelConfig::scail2_14b`]) plus the SCAIL-2-specific conditioning knobs the base Wan DiT does
//! not carry (the 28-channel mask stem, the i2v binary-mask channels, and the per-source RoPE shifts).

use std::path::Path;

use mlx_gen::{Error, Result};
use mlx_gen_wan::config::WanModelConfig;
use serde_json::Value;

/// SCAIL-2 conditioning knobs (zai-org/SCAIL-2 `wan/modules/model_scail2.py`), layered on top of the
/// shared Wan2.1-14B-I2V [`WanModelConfig`].
#[derive(Clone, Debug)]
pub struct Scail2Config {
    /// Shared Wan2.1-14B-I2V dimensions (dim 5120, 40L/40H, `in_dim` 20, z16 VAE).
    pub wan: WanModelConfig,
    /// Channel count of the color-coded semantic-mask latent fed to `patch_embedding_mask` (28 =
    /// 7 color classes × temporal-pack 4; see `extract_and_compress_mask_to_latent`).
    pub mask_dim: usize,
    /// Binary i2v-mask channels concatenated onto each latent before patch-embed (4): the model's
    /// `in_dim` (20) = VAE-z (16) + 4.
    pub i2v_mask_dim: usize,
    /// RoPE H-shift applied to the reference chunk in REPLACEMENT mode (`replace_flag = true`); the
    /// shift is 0 in animation mode.
    pub replace_h_shift: usize,
    /// RoPE W-shift applied to the spatially-downsampled pose chunk (120).
    pub pose_w_shift: usize,
    /// Max source-id the model was trained with (drives fractional interpolation for >N references).
    pub max_trained_src_id: f64,
}

impl Default for Scail2Config {
    fn default() -> Self {
        Self::scail2_14b()
    }
}

impl Scail2Config {
    /// The shipped SCAIL-2 14B config (zai-org/SCAIL-2, `configs/config-14b.json`).
    pub fn scail2_14b() -> Self {
        Self {
            wan: WanModelConfig::scail2_14b(),
            mask_dim: 28,
            i2v_mask_dim: 4,
            replace_h_shift: 120,
            pose_w_shift: 120,
            max_trained_src_id: 5.0,
        }
    }

    /// Load from a snapshot dir's `config.json` (the upstream `config-14b.json` layout: `in_dim`,
    /// `mask_dim`, `dim`, `ffn_dim`, `num_heads`, `num_layers`, `out_dim`, `model_type`). Any field
    /// absent from the JSON keeps the shipped 14B default.
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let mut cfg = Self::scail2_14b();
        let path = root.join("config.json");
        if path.exists() {
            let text = std::fs::read_to_string(&path)?;
            let v: Value = serde_json::from_str(&text)
                .map_err(|e| Error::Msg(format!("scail2: parse config.json: {e}")))?;
            set_usize(&v, "in_dim", &mut cfg.wan.in_dim);
            set_usize(&v, "out_dim", &mut cfg.wan.out_dim);
            set_usize(&v, "dim", &mut cfg.wan.dim);
            set_usize(&v, "ffn_dim", &mut cfg.wan.ffn_dim);
            set_usize(&v, "num_heads", &mut cfg.wan.num_heads);
            set_usize(&v, "num_layers", &mut cfg.wan.num_layers);
            set_usize(&v, "mask_dim", &mut cfg.mask_dim);
        }
        Ok(cfg)
    }
}

fn set_usize(v: &Value, key: &str, slot: &mut usize) {
    if let Some(n) = v.get(key).and_then(Value::as_u64) {
        *slot = n as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_14b_dims() {
        let c = Scail2Config::scail2_14b();
        assert_eq!(c.wan.dim, 5120);
        assert_eq!(c.wan.num_layers, 40);
        assert_eq!(c.wan.num_heads, 40);
        assert_eq!(c.wan.in_dim, 20);
        assert_eq!(c.wan.out_dim, 16);
        assert_eq!(c.mask_dim, 28);
        assert_eq!(c.wan.head_dim(), 128);
        assert_eq!(c.wan.model_version, "2.1");
        assert!(!c.wan.dual_model);
        assert_eq!(c.wan.vae_z_dim, 16);
    }
}
