//! Qwen-Image **2512-Fun-Controlnet-Union** control branch (sc-8267). Port of the alibaba-pai
//! `Qwen-Image-2512-Fun-Controlnet-Union` `QwenImageControlTransformer2DModel` (the `VideoX-Fun`
//! VACE family — same shape as the FLUX.2 / Z-Image Fun-Controlnet-Union branches), which **replaces**
//! the retired InstantX `Qwen-Image-ControlNet-Union` shape on the Qwen control path.
//!
//! Unlike the InstantX ControlNet (an independent mini-transformer with a zero-init
//! `controlnet_x_embedder` ADDed onto `img_in(x)`, emitting per-block residuals the base ADDs at a
//! fixed interval), the 2512-Fun branch is **VACE-style**: a `control_img_in` patch embedder
//! (`132 → inner`) feeds a control state `c` threaded through N control blocks that reuse the base
//! block math (and the base modulation / RoPE / timestep), seeded at block 0 by
//! `c = before_proj(c) + img_embed`. Each control block emits a hint via a zero-init `after_proj`;
//! the base transformer adds `hints[n]·control_context_scale` into its image stream **after** the
//! base block at `control_layers[n]` (`[0, 12, 24, 36, 48]` — 5 hints across the 60-layer MMDiT).
//! `control_context_scale = 0` is byte-identical to the base forward (`+0`).
//!
//! The 132-channel control context is `concat([control_latents(16) | mask(1) | inpaint_latent(16)])`
//! packed 2×2 → `33·4 = 132` (the fork's `pipeline_qwenimage_control._prepare`: VAE-encode the
//! control image, a `1 − mask` channel, and an inpaint latent). v1 is **pose-only** (no mask / no
//! inpaint image), so the layout reduces to `[control_latents | 0 | 0]` — see
//! [`crate::pipeline::encode_fun_control_context`].
//!
//! The control blocks are the *same* [`QwenTransformerBlock`] as the base (identical on-disk keys),
//! so the loader reuses the base block remap. Adapters (the character-identity LoRA) target the
//! **base** transformer only — the control branch is never an adapter target (mirrors the FLUX.2 /
//! Z-Image control ports and the fork, which train LoRA on the base).

use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::transformer::{linear_from, QwenTransformerBlock, QwenTransformerConfig};

/// The alibaba-pai `Qwen-Image-2512-Fun-Controlnet-Union` config (`config/qwenimage_control.yaml`):
/// `control_layers = [0, 12, 24, 36, 48]` (5 full dual-stream control blocks injected across the base
/// 60-layer MMDiT), `control_in_dim = 132`, otherwise the base Qwen-Image shape (24 heads × 128).
pub struct QwenFunControlConfig {
    /// Base double-block indices each control block injects its hint into (`control_layers`).
    pub control_layers: Vec<usize>,
    /// Packed control-context channels (`control_img_in` in-features). 132 for the shipped Union
    /// (`[control_latents(16) | mask(1) | inpaint(16)]` × 2×2 patch).
    pub control_in_dim: i32,
    pub num_heads: i32,
    pub head_dim: i32,
}

impl QwenFunControlConfig {
    /// The shipped 2512-Fun Union: 5 control layers at `[0, 12, 24, 36, 48]`, `control_in_dim = 132`.
    pub fn qwen_image_2512_fun() -> Self {
        let base = QwenTransformerConfig::qwen_image();
        Self {
            control_layers: vec![0, 12, 24, 36, 48],
            control_in_dim: 132,
            num_heads: base.num_heads,
            head_dim: base.head_dim,
        }
    }
}

/// The Qwen 2512-Fun control branch (the trainable VACE branch). Holds the `control_img_in` patch
/// embedder, N control blocks (each a full base dual-stream block) with a zero-init `after_proj`
/// hint projection, and a zero-init `before_proj` on block 0 that seeds the control state from the
/// base image stream. Emits one hint per control layer for the base transformer to inject.
pub struct QwenFunControlBranch {
    /// `control_img_in`: 132 → inner_dim. Bias-carrying patch embedder for the packed control context.
    control_img_in: AdaptableLinear,
    /// The N control blocks (same math as the base dual-stream block; reuse the base RoPE / temb).
    blocks: Vec<QwenTransformerBlock>,
    /// Zero-init per-block hint projection (`inner_dim → inner_dim`), one per control block.
    after_proj: Vec<AdaptableLinear>,
    /// Zero-init `before_proj` on control block 0 (`inner_dim → inner_dim`): `c = before_proj(c) + x`.
    before_proj: AdaptableLinear,
    /// Base block indices each control hint injects into (`control_layers`); `places[n]` is the base
    /// index for hint `n`.
    places: Vec<usize>,
}

impl QwenFunControlBranch {
    /// Load from the 2512-Fun checkpoint (already remapped to the base block's internal key names by
    /// the loader). `prefix` is empty for the real single-file checkpoint; a non-empty prefix is used
    /// by synthetic fixtures. Keys: `control_img_in.{weight,bias}`, `control_blocks.{i}.*` (a base
    /// block + `after_proj` for every `i`, plus `before_proj` on `i == 0`).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &QwenFunControlConfig) -> Result<Self> {
        if !cfg.control_layers.contains(&0) {
            return Err(Error::Msg(
                "qwen 2512-fun control: control_layers must contain 0 (before_proj lives on block 0)"
                    .into(),
            ));
        }
        let p = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };
        let n = cfg.control_layers.len();
        let mut blocks = Vec::with_capacity(n);
        let mut after_proj = Vec::with_capacity(n);
        for i in 0..n {
            blocks.push(QwenTransformerBlock::from_weights(
                w,
                &p(&format!("control_blocks.{i}")),
                cfg.num_heads,
                cfg.head_dim,
            )?);
            after_proj.push(linear_from(
                w,
                &p(&format!("control_blocks.{i}.after_proj")),
                true,
            )?);
        }
        Ok(Self {
            control_img_in: linear_from(w, &p("control_img_in"), true)?,
            blocks,
            after_proj,
            before_proj: linear_from(w, &p("control_blocks.0.before_proj"), true)?,
            places: cfg.control_layers.clone(),
        })
    }

    /// Number of control hints (= control layers); drives the base injection sites.
    pub fn num_hints(&self) -> usize {
        self.blocks.len()
    }

    /// The hint index injected at base block `idx`, or `None`. Mirrors the fork's
    /// `control_layers_mapping`.
    pub fn hint_index(&self, idx: usize) -> Option<usize> {
        self.places.iter().position(|&p| p == idx)
    }

    /// Quantize the control branch Linears to Q4/Q8 (group_size 64), mirroring
    /// [`crate::transformer::QwenTransformer::quantize`]. Same transformer-only scope as T2I/Edit.
    /// `control_img_in` (132 in-features = `% 64 != 0`) stays dense, matching the fork's
    /// `nn.quantize` predicate and the FLUX.2 Fun `control_img_in` (260 in-features) handling.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.before_proj.quantize(bits, None)?;
        for block in &mut self.blocks {
            block.quantize(bits)?;
        }
        for ap in &mut self.after_proj {
            ap.quantize(bits, None)?;
        }
        Ok(())
    }

    /// Run the VACE control stack → the per-block hints (pre-scale), one per control layer. The fork's
    /// `forward_control`: `c = control_img_in(control_context)`; block 0 seeds `c = before_proj(c) +
    /// img_embed`; each control block runs the *base* block math (reusing the base modulation /
    /// RoPE / timestep) and threads its own `encoder_hidden_states` to the next; `hint[i] =
    /// after_proj(c_after_block_i)`.
    ///
    /// `img_embed`: post-`img_in` base image stream `[B, img_seq, inner]`. `control_context`: the
    /// packed 132-ch control context `[B, img_seq, 132]` (constant across steps). `encoder_embed`:
    /// post-`txt_in` base text stream `[B, txt_seq, inner]` (seeds the control stack's text branch).
    /// `text_emb`/RoPE/`mask`/`modulate_index` are the base double-stream conditioning (the control
    /// blocks reuse them). The threaded text is local to the control stack — only the image-stream
    /// hints leave.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_control(
        &self,
        img_embed: &Array,
        encoder_embed: &Array,
        control_context: &Array,
        text_emb: &Array,
        img_cos: &Array,
        img_sin: &Array,
        txt_cos: &Array,
        txt_sin: &Array,
        mask: Option<&Array>,
        modulate_index: Option<&Array>,
    ) -> Result<Vec<Array>> {
        // c = control_img_in(control_context); seed block 0 with `before_proj(c) + img_embed`.
        let mut c = add(
            &self
                .before_proj
                .forward(&self.control_img_in.forward(control_context)?)?,
            img_embed,
        )?;
        let mut encoder = encoder_embed.clone();
        let mut hints = Vec::with_capacity(self.blocks.len());
        for (block, ap) in self.blocks.iter().zip(&self.after_proj) {
            let (e, new_c) = block.forward(
                &c,
                &encoder,
                text_emb,
                img_cos,
                img_sin,
                txt_cos,
                txt_sin,
                mask,
                modulate_index,
            )?;
            encoder = e;
            // hint[i] = after_proj(c_after_block_i) (zero-init projection; the fork's `c_skip`).
            hints.push(ap.forward(&new_c)?);
            c = new_c;
        }
        Ok(hints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_config_matches_fork() {
        // The alibaba-pai `config/qwenimage_control.yaml`: 5 control layers across the 60-block MMDiT
        // (interval 12) + a 132-channel control context. Mirrors the verified checkpoint header
        // (`control_blocks.{0..4}` + `control_img_in` 132→3072).
        let cfg = QwenFunControlConfig::qwen_image_2512_fun();
        assert_eq!(cfg.control_layers, vec![0, 12, 24, 36, 48]);
        assert_eq!(cfg.control_in_dim, 132);
        assert_eq!(cfg.num_heads, 24);
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn from_weights_rejects_layers_without_zero() {
        // `before_proj` lives on control block 0, so `0` must be a control layer — the loader guards
        // this before any tensor lookup (so an empty Weights still trips it).
        let cfg = QwenFunControlConfig {
            control_layers: vec![12, 24],
            ..QwenFunControlConfig::qwen_image_2512_fun()
        };
        let w = Weights::empty();
        let err = QwenFunControlBranch::from_weights(&w, "", &cfg)
            .err()
            .expect("expected a guard error")
            .to_string();
        assert!(err.contains("control_layers must contain 0"), "got: {err}");
    }
}
