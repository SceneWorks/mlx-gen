//! Lens text-encoder (gpt-oss-20b) configuration + the YaRN RoPE frequency derivation.
//!
//! Values are the verified `text_encoder/config.json` of `microsoft/Lens-Turbo` (identical in
//! `microsoft/Lens`): 24 layers, hidden 2880, head_dim 64, **64 query / 8 KV heads**, 32 experts
//! top-4, `attention_bias: true`, rms_norm_eps 1e-5, sliding_window 128, swiglu_limit 7.0, and a
//! YaRN rope (`rope_theta 150000, factor 32, beta_fast 32, beta_slow 1,
//! original_max_position_embeddings 4096, truncate false`).

use std::f64::consts::PI;

/// gpt-oss-20b text-encoder config (the Lens / Lens-Turbo `text_encoder/config.json` values).
#[derive(Clone, Copy, Debug)]
pub struct GptOssConfig {
    pub hidden_size: i32,
    pub num_layers: usize,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    /// Per-expert intermediate size (the SwiGLU inner width).
    pub intermediate: i32,
    pub num_experts: i32,
    pub experts_per_tok: i32,
    pub rms_eps: f32,
    pub sliding_window: i32,
    /// gpt-oss clamped-SwiGLU limit (the `GptOssExperts.limit`, 7.0).
    pub swiglu_limit: f32,
    /// gpt-oss clamped-SwiGLU GLU coefficient (the `GptOssExperts.alpha`, 1.702).
    pub swiglu_alpha: f32,
    // --- YaRN RoPE parameters ---
    pub rope_theta: f64,
    pub yarn_factor: f64,
    pub yarn_beta_fast: f64,
    pub yarn_beta_slow: f64,
    pub yarn_orig_max_pos: f64,
    pub yarn_truncate: bool,
}

impl GptOssConfig {
    /// The `microsoft/Lens-Turbo` (and `microsoft/Lens`) gpt-oss-20b text-encoder values.
    pub fn lens() -> Self {
        Self {
            hidden_size: 2880,
            num_layers: 24,
            num_heads: 64,
            num_kv_heads: 8,
            head_dim: 64,
            intermediate: 2880,
            num_experts: 32,
            experts_per_tok: 4,
            rms_eps: 1e-5,
            sliding_window: 128,
            swiglu_limit: 7.0,
            swiglu_alpha: 1.702,
            rope_theta: 150_000.0,
            yarn_factor: 32.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_orig_max_pos: 4096.0,
            yarn_truncate: false,
        }
    }

    /// `layer_types[i]`: the Lens config alternates **sliding, full, sliding, full, …** starting at
    /// layer 0, so even layers are sliding-window and odd layers are full attention.
    pub fn is_sliding(&self, layer: usize) -> bool {
        layer.is_multiple_of(2)
    }

    /// Derive the YaRN `inv_freq` (length `head_dim/2`) and the `attention_scaling` (mscale) factor,
    /// a faithful port of `transformers.modeling_rope_utils._compute_yarn_parameters` for the
    /// gpt-oss `rope_type = "yarn"`. Computed once in f64; both are RoPE constants (no `attention_factor`,
    /// `mscale`, or `mscale_all_dim` in the Lens config, so the factor reduces to `get_mscale(factor)`).
    pub fn yarn_rope(&self) -> (Vec<f32>, f32) {
        let dim = self.head_dim as f64;
        let base = self.rope_theta;
        let factor = self.yarn_factor;
        let half = (self.head_dim / 2) as usize;

        // attention_factor = get_mscale(factor): 1.0 for factor<=1, else 0.1*ln(factor)+1.
        let attention_factor = if factor <= 1.0 {
            1.0
        } else {
            0.1 * factor.ln() + 1.0
        };

        let find_correction_dim = |num_rotations: f64| -> f64 {
            (dim * (self.yarn_orig_max_pos / (num_rotations * 2.0 * PI)).ln()) / (2.0 * base.ln())
        };
        let mut low = find_correction_dim(self.yarn_beta_fast);
        let mut high = find_correction_dim(self.yarn_beta_slow);
        if self.yarn_truncate {
            low = low.floor();
            high = high.ceil();
        }
        low = low.max(0.0);
        high = high.min(dim - 1.0);
        // linear_ramp_factor guards the min==max singularity by nudging max.
        let (rmin, rmax) = if (low - high).abs() < f64::EPSILON {
            (low, high + 0.001)
        } else {
            (low, high)
        };

        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            let pos_freq = base.powf((2 * i) as f64 / dim);
            let extrapolation = 1.0 / pos_freq;
            let interpolation = 1.0 / (factor * pos_freq);
            // ramp over dim//2 indices; inv_freq_extrapolation_factor = 1 - ramp.
            let ramp = ((i as f64 - rmin) / (rmax - rmin)).clamp(0.0, 1.0);
            let extrap_factor = 1.0 - ramp;
            let f = interpolation * (1.0 - extrap_factor) + extrapolation * extrap_factor;
            inv_freq.push(f as f32);
        }
        (inv_freq, attention_factor as f32)
    }
}
