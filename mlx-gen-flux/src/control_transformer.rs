//! FLUX.1-dev **Fun-Controlnet-Union** control transformer (sc-8238, epic 8236). Port of the diffusers
//! `FluxControlNetModel` as shipped by `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`: a small
//! partial copy of the base FLUX.1 MMDiT (the checkpoint ships `num_layers = 6` double blocks and
//! `num_single_layers = 0`) that ingests the VAE-encoded control image (a pose skeleton / canny / depth
//! map — input-agnostic, no discrete mode index in 2.0) and emits one per-block residual, injected into
//! the frozen base 19-layer double stream at `interval = ceil(19 / 6) = 4` (see
//! [`crate::transformer::FluxTransformer::forward_control`]).
//!
//! This follows the standard diffusers ControlNet shape (the same one the Qwen-Image ControlNet-Union
//! port uses, `mlx-gen-qwen-image/src/control_transformer.rs`): an **independent** mini-transformer with
//! its own `x_embedder` / `context_embedder` / `time_text_embed` and a zero-init `controlnet_x_embedder`
//! that adds the encoded control image into the image stream; each of its blocks' output is projected by
//! a zero-init `controlnet_blocks[i]` Linear into a residual. The N residuals are returned (pre-scale)
//! for the base transformer to add. It is NOT the VACE-style threaded control branch FLUX.2 uses
//! (`Flux2ControlBranch`) — the Shakker FLUX.1 checkpoint is the diffusers residual-emitter shape, which
//! is why the block math is the *same* [`JointBlock`] as the base (identical on-disk keys).
//!
//! Adapters (the character-identity LoRA) target the **base** transformer only — the control branch is
//! never an adapter target (mirrors the FLUX.2 / Z-Image / Qwen control ports, and the fork, which
//! trains LoRA on the base DiT).

use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::transformer::{
    linear_from, FluxRope, FluxTransformer, JointBlock, RopeTable, TimeTextEmbed,
};

/// The Shakker `FLUX.1-dev-ControlNet-Union-Pro-2.0` config (`config.json`): a 6-block partial copy of
/// the base 19-layer FLUX.1 MMDiT, identical inner dims, `num_single_layers = 0`, guidance-embedded
/// (FLUX.1-dev). The 2.0 checkpoint has NO `num_mode` / condition-type embedding (input-agnostic — the
/// 1.0 InstantX checkpoint with its discrete `control_mode` integer is rejected, per the S0 audit).
pub struct FluxControlNetConfig {
    /// Number of control double blocks shipped in the checkpoint (Shakker 2.0 = 6).
    pub num_layers: usize,
    /// FLUX.1-dev carries an embedded guidance scalar (the control branch mirrors the base).
    pub supports_guidance: bool,
}

impl FluxControlNetConfig {
    /// The shipped Shakker Union-Pro-2.0: `num_layers = 6`, guidance-embedded (dev).
    pub fn shakker_union_pro_2_0() -> Self {
        Self {
            num_layers: 6,
            supports_guidance: true,
        }
    }
}

/// The FLUX.1 ControlNet control transformer (the trainable branch). Holds its own input projections +
/// N double blocks + N zero-init residual projections; emits the per-block residuals for the base
/// transformer (`FluxTransformer::forward_control`).
pub struct FluxControlNet {
    x_embedder: AdaptableLinear,
    context_embedder: AdaptableLinear,
    time_text_embed: TimeTextEmbed,
    /// Zero-init projection of the packed control latent (`64 → inner_dim`), added to `x_embedder(x)`.
    controlnet_x_embedder: AdaptableLinear,
    blocks: Vec<JointBlock>,
    /// Zero-init per-block residual projections (`inner_dim → inner_dim`).
    controlnet_blocks: Vec<AdaptableLinear>,
    pos_embed: FluxRope,
}

impl FluxControlNet {
    /// Load from the Shakker Union-Pro-2.0 checkpoint (standard diffusers layout —
    /// `diffusion_pytorch_model.safetensors`, keys un-prefixed for the real single-file checkpoint;
    /// `prefix` is e.g. `"w"` for a synthetic fixture).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &FluxControlNetConfig) -> Result<Self> {
        let p = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let mut controlnet_blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(JointBlock::from_weights(
                w,
                &p(&format!("transformer_blocks.{i}")),
            )?);
            controlnet_blocks.push(linear_from(w, &p(&format!("controlnet_blocks.{i}")), true)?);
        }
        Ok(Self {
            x_embedder: linear_from(w, &p("x_embedder"), true)?,
            context_embedder: linear_from(w, &p("context_embedder"), true)?,
            time_text_embed: TimeTextEmbed::from_weights(
                w,
                &p("time_text_embed"),
                cfg.supports_guidance,
            )?,
            controlnet_x_embedder: linear_from(w, &p("controlnet_x_embedder"), true)?,
            blocks,
            controlnet_blocks,
            pos_embed: FluxRope::new(),
        })
    }

    /// Number of control residuals (= control layers); drives the base injection interval.
    pub fn num_residuals(&self) -> usize {
        self.controlnet_blocks.len()
    }

    /// Quantize the control transformer's Linears to Q4/Q8 (group_size 64), mirroring
    /// [`crate::transformer::FluxTransformer::quantize`] over the control branch. Same transformer-only
    /// scope as the base txt2img path.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.x_embedder.quantize(bits, None)?;
        self.context_embedder.quantize(bits, None)?;
        self.controlnet_x_embedder.quantize(bits, None)?;
        self.time_text_embed.quantize(bits)?;
        for block in &mut self.blocks {
            block.quantize(bits)?;
        }
        for cb in &mut self.controlnet_blocks {
            cb.quantize(bits, None)?;
        }
        Ok(())
    }

    /// Run the control branch → the per-block residuals (pre-scale), one per control layer.
    ///
    /// `hidden_states`: the current packed **noise** latents `[B, img_seq, 64]` (the controlnet sees the
    /// same latents the base does this step). `control_cond`: the packed VAE-encoded control image
    /// `[B, img_seq, 64]` (constant across steps). `prompt_embeds`/`pooled_prompt_embeds`: the same text
    /// features the base forward uses. `sigma`/`guidance`: the scheduler sigma + embedded guidance (same
    /// as the base forward). The returned residuals align 1:1 with the noise token sequence (the control
    /// pose is a single grid, so the RoPE is built over the identical `[txt, img]` layout as the base).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Array,
        control_cond: &Array,
        prompt_embeds: &Array,
        pooled_prompt_embeds: &Array,
        sigma: f32,
        guidance: f32,
        width: u32,
        height: u32,
    ) -> Result<Vec<Array>> {
        // `x_embedder(x) + controlnet_x_embedder(control_cond)` (diffusers `hidden_states =
        // hidden_states + self.controlnet_x_embedder(controlnet_cond)`). The zero-init
        // `controlnet_x_embedder` means an untrained / scale-0 branch starts as a no-op.
        let mut hidden = add(
            &self.x_embedder.forward(hidden_states)?,
            &self.controlnet_x_embedder.forward(control_cond)?,
        )?;
        let mut encoder = self.context_embedder.forward(prompt_embeds)?;
        let text_embeddings = self.time_text_embed.forward(
            sigma * 1000.0,
            pooled_prompt_embeds,
            guidance * 1000.0,
        )?;
        let rope: RopeTable = self.pos_embed.forward(
            prompt_embeds.shape()[1] as usize,
            (height / 16) as usize,
            (width / 16) as usize,
        )?;

        let mut residuals = Vec::with_capacity(self.blocks.len());
        for (block, cn) in self.blocks.iter().zip(&self.controlnet_blocks) {
            let (e, h) = block.forward(&hidden, &encoder, &text_embeddings, &rope)?;
            encoder = e;
            hidden = h;
            // residual[i] = controlnet_blocks[i](hidden_after_block_i) (diffusers zero-init proj).
            residuals.push(cn.forward(&hidden)?);
        }
        Ok(residuals)
    }
}

/// The FLUX.1-dev base MMDiT + its Fun-Controlnet-Union control branch (sc-8238). Composes the
/// parity-proven [`FluxTransformer`] with a [`FluxControlNet`]; [`forward`](Self::forward) computes the
/// control residuals once and threads them (+ an optional identity injector — compose-ready, constraint
/// 2) into the base double stream, and [`quantize`](Self::quantize) packs both.
pub struct FluxControlTransformer {
    base: FluxTransformer,
    branch: FluxControlNet,
}

impl FluxControlTransformer {
    pub fn new(base: FluxTransformer, branch: FluxControlNet) -> Self {
        Self { base, branch }
    }

    /// Quantize the base + the control branch.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.base.quantize(bits)?;
        self.branch.quantize(bits)?;
        Ok(())
    }

    /// Adapter host = the base DiT (LoRA/LoKr target; the control branch is never an adapter target,
    /// mirroring the FLUX.2 / Z-Image / Qwen control ports).
    pub fn base_mut(&mut self) -> &mut FluxTransformer {
        &mut self.base
    }

    /// Read-only access to the base DiT (e.g. for the txt2img injector composition test).
    pub fn base(&self) -> &FluxTransformer {
        &self.base
    }

    /// Number of control double blocks (residuals); `interval = ceil(num_double / num_residuals)`.
    pub fn num_residuals(&self) -> usize {
        self.branch.num_residuals()
    }

    /// Control forward (no identity injector): the convenience entry E2's generator wires.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Array,
        control_cond: &Array,
        prompt_embeds: &Array,
        pooled_prompt_embeds: &Array,
        sigma: f32,
        guidance: f32,
        width: u32,
        height: u32,
        control_scale: f32,
    ) -> Result<Array> {
        self.forward_composed(
            hidden_states,
            control_cond,
            prompt_embeds,
            pooled_prompt_embeds,
            sigma,
            guidance,
            width,
            height,
            control_scale,
            None,
        )
    }

    /// Control forward THAT ALSO threads an optional identity injector (PuLID / XLabs IP-Adapter) — the
    /// **compose-ready** entry (constraint 2). The control residuals are computed once from the branch,
    /// then injected into the base double stream alongside the injector seam in
    /// [`FluxTransformer::forward_control`]. With `injector = None` this is the plain control forward;
    /// with `injector = Some(..)` a future epic stacks identity + control in one denoise step.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_composed(
        &self,
        hidden_states: &Array,
        control_cond: &Array,
        prompt_embeds: &Array,
        pooled_prompt_embeds: &Array,
        sigma: f32,
        guidance: f32,
        width: u32,
        height: u32,
        control_scale: f32,
        injector: Option<&dyn crate::transformer::DitImageInjector>,
    ) -> Result<Array> {
        let residuals = self.branch.forward(
            hidden_states,
            control_cond,
            prompt_embeds,
            pooled_prompt_embeds,
            sigma,
            guidance,
            width,
            height,
        )?;
        self.base.forward_control(
            hidden_states,
            prompt_embeds,
            pooled_prompt_embeds,
            sigma,
            guidance,
            width,
            height,
            injector,
            Some((&residuals, control_scale)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shakker_union_pro_2_0_config() {
        let cfg = FluxControlNetConfig::shakker_union_pro_2_0();
        // 6 control double blocks (the residual count), guidance-embedded (FLUX.1-dev), 0 single.
        assert_eq!(cfg.num_layers, 6);
        assert!(cfg.supports_guidance);
        // The injection interval over FLUX.1's 19 base double blocks: ceil(19/6) = 4.
        assert_eq!(
            crate::transformer::control_residual_interval(19, cfg.num_layers),
            4
        );
    }

    /// Compose-readiness (constraint 2) is a TYPE-LEVEL guarantee here: `forward_composed` accepts an
    /// `Option<&dyn DitImageInjector>`, so a future epic can stack identity (PuLID / XLabs IP-Adapter)
    /// plus control in one denoise. This test pins that signature so a refactor cannot silently drop
    /// the injector seam from the control path (the real-weight steer test exercises it at runtime).
    #[test]
    fn forward_composed_accepts_identity_injector_for_compose() {
        // A zero-sized function-pointer assertion: the compose entry's injector parameter is the same
        // `DitImageInjector` trait object the base txt2img/PuLID/IP-Adapter path uses.
        fn _assert_compose_signature(
            t: &FluxControlTransformer,
            x: &mlx_rs::Array,
            cc: &mlx_rs::Array,
            pe: &mlx_rs::Array,
            pp: &mlx_rs::Array,
            inj: Option<&dyn crate::transformer::DitImageInjector>,
        ) -> mlx_gen::Result<mlx_rs::Array> {
            t.forward_composed(x, cc, pe, pp, 500.0, 3.5, 512, 512, 0.7, inj)
        }
        // Referencing the fn proves it compiles with the injector seam; never called (no weights).
        let _ = _assert_compose_signature;
    }
}
