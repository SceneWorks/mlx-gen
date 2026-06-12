//! `UNet2DConditionModel` — the SDXL denoising U-Net. Port of the vendored `unet.UNetModel`: a
//! conv stem, sinusoidal timestep + SDXL `text_time` micro-conditioning embeddings, a down /
//! mid / up stack of [`UNetBlock2D`]s with cross-attention to the dual-CLIP text conditioning, and
//! a conv head. Runs entirely in NHWC. Predicts the noise (`eps`) for one denoise step.

mod block;
mod controlnet;
mod embeddings;
mod resnet;
mod transformer;

use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::add;
use mlx_rs::transforms::checkpoint;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableConv2d, AdaptableHost, AdaptableLinear, Adapter};
use mlx_gen::nn::{conv2d, group_norm};
use mlx_gen::train::lora::LoraParams;

use crate::silu_glue;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::UNetConfig;
use block::{BlockSpec, UNetBlock2D};
use embeddings::{text_time_temb, SinusoidalPositionalEncoding, TimestepEmbedding};
use transformer::Transformer2D;

// Shared with the VAE (the vendored VAE reuses the UNet `ResnetBlock2D` without a time embedding).
pub use resnet::ResnetBlock2D;

pub use controlnet::{ControlNet, ControlResiduals};

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-5;

/// Transpose a stored NCHW conv weight `[out, in, kH, kW]` to mlx's NHWC `[out, kH, kW, in]`.
pub(crate) fn nchw_to_nhwc(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 1])?)
}

/// Cast a raw parameter array (GroupNorm/LayerNorm weight or bias) to `dtype` in place if it differs
/// — the building block of the U-Net's `cast_weights` traversal for bf16 mixed-precision training
/// (sc-4941). `AdaptableLinear`/`AdaptableConv2d` carry their own `cast_weights`; this covers the
/// bare `Array`s that the norm ops consume directly.
pub(crate) fn cast_array(a: &mut Array, dtype: mlx_rs::Dtype) -> Result<()> {
    if a.dtype() != dtype {
        *a = a.as_dtype(dtype)?;
    }
    Ok(())
}

/// The SDXL conditional U-Net.
pub struct UNet2DConditionModel {
    /// Input conv stem (NHWC) — a conv-layer LoRA target (sc-2919).
    conv_in: AdaptableConv2d,
    timesteps: SinusoidalPositionalEncoding,
    time_embedding: TimestepEmbedding,
    add_time_proj: SinusoidalPositionalEncoding,
    add_embedding: TimestepEmbedding,
    down_blocks: Vec<UNetBlock2D>,
    mid_resnet0: ResnetBlock2D,
    mid_transformer: Transformer2D,
    mid_resnet1: ResnetBlock2D,
    up_blocks: Vec<UNetBlock2D>,
    conv_norm_out_w: Array,
    conv_norm_out_b: Array,
    /// Output conv head (NHWC) — a conv-layer LoRA target (sc-2919).
    conv_out: AdaptableConv2d,
    /// Optional context projection (diffusers `encoder_hid_proj`). Present only when the checkpoint
    /// carries `encoder_hid_proj.weight` — the **Kolors** U-Net (epic 3090), which projects the
    /// ChatGLM3 context (4096) down to `cross_attention_dim` (2048) before cross-attention. SDXL has
    /// no such key, so this stays `None` and the forward is byte-identical to before.
    encoder_hid_proj: Option<AdaptableLinear>,
}

impl UNet2DConditionModel {
    /// Assemble the U-Net from a diffusers SDXL `unet/` checkpoint (keys read directly; conv weights
    /// transposed to NHWC on load). `cfg` is [`UNetConfig::sdxl_base`].
    pub fn from_weights(w: &Weights, cfg: &UNetConfig) -> Result<Self> {
        let n = cfg.num_blocks();
        let boc = &cfg.block_out_channels;
        let temb_dim_src = boc[0]; // sinusoidal timestep width

        // Down blocks: block i goes block_channels[i] -> block_channels[i+1].
        let mut down_blocks = Vec::with_capacity(n);
        // `i` indexes five parallel config arrays + the block prefix, not just `boc` — an
        // `enumerate()` rewrite would be strictly worse here.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            down_blocks.push(UNetBlock2D::from_weights(
                w,
                &BlockSpec {
                    prefix: &format!("down_blocks.{i}"),
                    num_resnets: cfg.layers_per_block[i],
                    out_channels: boc[i],
                    num_heads: cfg.num_attention_heads[i],
                    transformer_layers: cfg.transformer_layers_per_block[i],
                    add_cross_attention: cfg.down_block_types[i].contains("CrossAttn"),
                    add_downsample: i < n - 1,
                    add_upsample: false,
                },
            )?);
        }

        // Mid: resnet, transformer, resnet (the vendored mid_blocks.0/1/2).
        let mid_resnet0 = ResnetBlock2D::from_weights(w, "mid_block.resnets.0")?;
        let mid_transformer = Transformer2D::from_weights(
            w,
            "mid_block.attentions.0",
            *boc.last().unwrap(),
            *cfg.num_attention_heads.last().unwrap(),
            *cfg.transformer_layers_per_block.last().unwrap(),
        )?;
        let mid_resnet1 = ResnetBlock2D::from_weights(w, "mid_block.resnets.1")?;

        // Up blocks: checkpoint up_blocks.{k} corresponds to config index `n-1-k` (the vendored
        // builds them in reversed order). add_upsample on all but the last config index (0).
        let mut up_blocks = Vec::with_capacity(n);
        for k in 0..n {
            let ci = n - 1 - k;
            up_blocks.push(UNetBlock2D::from_weights(
                w,
                &BlockSpec {
                    prefix: &format!("up_blocks.{k}"),
                    num_resnets: cfg.layers_per_block[ci] + 1,
                    out_channels: boc[ci],
                    num_heads: cfg.num_attention_heads[ci],
                    transformer_layers: cfg.transformer_layers_per_block[ci],
                    add_cross_attention: cfg.up_block_types[ci].contains("CrossAttn"),
                    add_downsample: false,
                    add_upsample: ci > 0,
                },
            )?);
        }

        Ok(Self {
            conv_in: AdaptableConv2d::new(
                nchw_to_nhwc(w.require("conv_in.weight")?)?,
                Some(w.require("conv_in.bias")?.clone()),
            ),
            timesteps: SinusoidalPositionalEncoding::timestep(temb_dim_src)?,
            time_embedding: TimestepEmbedding::from_weights(w, "time_embedding")?,
            add_time_proj: SinusoidalPositionalEncoding::timestep(
                cfg.addition_time_embed_dim.unwrap_or(256),
            )?,
            add_embedding: TimestepEmbedding::from_weights(w, "add_embedding")?,
            down_blocks,
            mid_resnet0,
            mid_transformer,
            mid_resnet1,
            up_blocks,
            conv_norm_out_w: w.require("conv_norm_out.weight")?.clone(),
            conv_norm_out_b: w.require("conv_norm_out.bias")?.clone(),
            conv_out: AdaptableConv2d::new(
                nchw_to_nhwc(w.require("conv_out.weight")?)?,
                Some(w.require("conv_out.bias")?.clone()),
            ),
            // Kolors `encoder_hid_proj` (4096→2048). Auto-detected: absent for SDXL → `None`.
            encoder_hid_proj: w.get("encoder_hid_proj.weight").map(|wt| {
                AdaptableLinear::dense(wt.clone(), w.get("encoder_hid_proj.bias").cloned())
            }),
        })
    }

    /// Quantize the true Linears (resnets' `time_emb_proj`, attention, FFN, proj_in/out, embeddings)
    /// to Q4/Q8. **Convs stay dense** — `conv_in`/`conv_out`, resnet `conv1`/`conv2`, the up/down
    /// samplers, **and the resnet `conv_shortcut`** (a 1×1 conv stored as a Linear; quantizing it
    /// collapses 1024² renders, sc-3329 — see [`ResnetBlock2D::quantize`]).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.time_embedding.quantize(bits)?;
        self.add_embedding.quantize(bits)?;
        if let Some(proj) = &mut self.encoder_hid_proj {
            proj.quantize(bits, None)?; // Kolors context projection (sc-3096 validates)
        }
        for b in &mut self.down_blocks {
            b.quantize(bits)?;
        }
        self.mid_resnet0.quantize(bits)?;
        self.mid_transformer.quantize(bits)?;
        self.mid_resnet1.quantize(bits)?;
        for b in &mut self.up_blocks {
            b.quantize(bits)?;
        }
        Ok(())
    }

    /// Toggle SDPA-segment gradient checkpointing across every cross-attention transformer (sc-4941):
    /// down/up blocks + the mid transformer. Training-only; opt-in (the SDXL first-step working set
    /// fits unified memory without it, so the trainer gates this on `gradient_checkpointing` rather
    /// than forcing it always-on). Grads are bit-identical to the retained backward.
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        for b in &mut self.down_blocks {
            b.set_sdpa_checkpoint(on);
        }
        self.mid_transformer.set_sdpa_checkpoint(on);
        for b in &mut self.up_blocks {
            b.set_sdpa_checkpoint(on);
        }
    }

    /// Cast every frozen base weight to `dtype` (sc-4941 bf16 mixed-precision training): the conv
    /// stem/head, the time + add embeddings, every down/mid/up block (resnets, attention transformers,
    /// samplers), the output GroupNorm, and the Kolors `encoder_hid_proj`. The trainable LoRA factors,
    /// loss, gradients, and optimizer state stay f32 (master-weights); only the base + the activation
    /// stream this casts become bf16. Destructive (f32→bf16 drops mantissa) — reload for f32.
    pub fn cast_weights(&mut self, dtype: mlx_rs::Dtype) -> Result<()> {
        self.conv_in.cast_weights(dtype)?;
        self.time_embedding.cast_weights(dtype)?;
        self.add_embedding.cast_weights(dtype)?;
        for b in &mut self.down_blocks {
            b.cast_weights(dtype)?;
        }
        self.mid_resnet0.cast_weights(dtype)?;
        self.mid_transformer.cast_weights(dtype)?;
        self.mid_resnet1.cast_weights(dtype)?;
        for b in &mut self.up_blocks {
            b.cast_weights(dtype)?;
        }
        cast_array(&mut self.conv_norm_out_w, dtype)?;
        cast_array(&mut self.conv_norm_out_b, dtype)?;
        self.conv_out.cast_weights(dtype)?;
        // The Kolors `encoder_hid_proj` (ChatGLM3 4096→2048 context projection) casts with the rest.
        // sc-4941 carve-out audit: an f32 carve-out on this conditioning entry (with f32
        // cross-attention) was MEASURED to make the bf16 grad direction WORSE, not better (global
        // cosine 0.9946→0.9933, min-large 0.971→0.924), so it is deliberately NOT applied — full bf16
        // is the better config for Kolors (see the Kolors trainer's `bf16_grads_direction` gate).
        if let Some(proj) = &mut self.encoder_hid_proj {
            proj.cast_weights(dtype)?;
        }
        Ok(())
    }

    /// The model's current compute dtype (read off the conv stem weight), or `None` if quantized.
    /// Lets the trainer detect a prior [`cast_weights`](Self::cast_weights) (the cast is destructive,
    /// so a trainer reused across runs must not silently re-cast a bf16 base for an f32 request).
    pub fn compute_dtype(&self) -> Option<mlx_rs::Dtype> {
        Some(self.conv_in.weight_dtype())
    }

    /// Install IP-Adapter decoupled K/V projections (sc-3059) into the cross-attention modules, in
    /// the diffusers `attn_processors` walk order — **down_blocks → up_blocks → mid_block** (the
    /// empirical `ip_adapter.{1,3,…}` numeric order). `pairs` are the `to_k_ip/to_v_ip` weights, one
    /// per cross-attention layer (70 for SDXL), in that numeric order. Errors on a count mismatch.
    pub fn install_ip_adapter(&mut self, pairs: Vec<(Array, Array)>) -> Result<()> {
        let expected = pairs.len();
        let mut it = pairs.into_iter();
        for b in &mut self.down_blocks {
            b.install_ip(&mut it)?;
        }
        for b in &mut self.up_blocks {
            b.install_ip(&mut it)?;
        }
        self.mid_transformer.install_ip(&mut it)?;
        let leftover = it.count();
        if leftover != 0 {
            return Err(mlx_gen::Error::Msg(format!(
                "ip_adapter: {leftover} of {expected} K/V pairs unused (cross-attn count mismatch)"
            )));
        }
        Ok(())
    }

    /// Predict `eps` for one denoise step.
    /// - `x`: NHWC latents `[B, H, W, 4]`.
    /// - `timestep`: the (sigma-space) time, broadcast to the batch.
    /// - `encoder_x`: dual-CLIP text conditioning `[B, S, 2048]`.
    /// - `text_emb`: pooled conditioning `[B, 1280]`; `time_ids`: micro-conditioning `[B, 6]`.
    pub fn forward(
        &self,
        x: &Array,
        timestep: f32,
        encoder_x: &Array,
        text_emb: &Array,
        time_ids: &Array,
    ) -> Result<Array> {
        self.forward_core(x, timestep, encoder_x, text_emb, time_ids, None, None)
    }

    /// Like [`forward`](Self::forward) but adds a ControlNet's residuals (sc-3058): each control
    /// down residual is added to the matching skip connection, the control mid residual to the mid
    /// output. The residuals are already scaled by `conditioning_scale` (see [`ControlNet::forward`]).
    pub fn forward_with_control(
        &self,
        x: &Array,
        timestep: f32,
        encoder_x: &Array,
        text_emb: &Array,
        time_ids: &Array,
        control: &ControlResiduals,
    ) -> Result<Array> {
        self.forward_core(
            x,
            timestep,
            encoder_x,
            text_emb,
            time_ids,
            Some(control),
            None,
        )
    }

    /// Like [`forward`](Self::forward) but injects the IP-Adapter image tokens into every
    /// cross-attention via the decoupled branch (sc-3059). `ip = (tokens [B, N, cross_attention_dim],
    /// scale)`. CFG handling is the caller's: pass a zeros uncond row in the batched tokens so the
    /// negative pass contributes no IP signal.
    pub fn forward_with_ip(
        &self,
        x: &Array,
        timestep: f32,
        encoder_x: &Array,
        text_emb: &Array,
        time_ids: &Array,
        ip: (&Array, f32),
    ) -> Result<Array> {
        self.forward_core(x, timestep, encoder_x, text_emb, time_ids, None, Some(ip))
    }

    /// Combined decoupled-cross-attn IP tokens **and** ControlNet residuals in one forward — the
    /// InstantID path (epic 3109, sc-3113/3114): the face IP tokens drive the cross-attention while
    /// the IdentityNet's residuals add into the skip/mid connections. `encoder_x` is the text
    /// conditioning (to_k/to_v); `ip = (face_tokens, ip_adapter_scale)` feeds the decoupled
    /// to_k_ip/to_v_ip branch; `control` is the IdentityNet output (already `conditioning_scale`d, and
    /// itself conditioned on the face tokens — see [`ControlNet::forward`]). With `ip` scale 0 and a
    /// zero-scale `control` this is identical to [`forward`](Self::forward).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_ip_control(
        &self,
        x: &Array,
        timestep: f32,
        encoder_x: &Array,
        text_emb: &Array,
        time_ids: &Array,
        ip: (&Array, f32),
        control: &ControlResiduals,
    ) -> Result<Array> {
        self.forward_core(
            x,
            timestep,
            encoder_x,
            text_emb,
            time_ids,
            Some(control),
            Some(ip),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_core(
        &self,
        x: &Array,
        timestep: f32,
        encoder_x: &Array,
        text_emb: &Array,
        time_ids: &Array,
        control: Option<&ControlResiduals>,
        ip: Option<(&Array, f32)>,
    ) -> Result<Array> {
        let batch = x.shape()[0];
        let dtype = x.dtype();

        // Kolors: project the ChatGLM3 context (4096) to `cross_attention_dim` (2048) before any
        // cross-attention (diffusers applies `encoder_hid_proj` once, up front). No-op for SDXL.
        let projected;
        let encoder_x = match &self.encoder_hid_proj {
            Some(proj) => {
                projected = proj.forward(encoder_x)?;
                &projected
            }
            None => encoder_x,
        };

        // Timestep + SDXL `text_time` micro-conditioning embedding (shared verbatim with
        // `ControlNet::forward` — the encoder-copy contract requires bit-identity; F-070).
        let temb = text_time_temb(
            &self.timesteps,
            &self.time_embedding,
            &self.add_time_proj,
            &self.add_embedding,
            timestep,
            text_emb,
            time_ids,
            batch,
            dtype,
        )?;

        // Conv stem.
        let mut x = conv2d(x, self.conv_in.weight(), self.conv_in.bias(), 1, 1)?;

        // Down path — collect skip residuals (starting with the stem output).
        let mut residuals: Vec<Array> = vec![x.clone()];
        for block in &self.down_blocks {
            let (out, res) = block.forward_ip(&x, encoder_x, &temb, None, ip)?;
            x = out;
            residuals.extend(res);
        }

        // ControlNet (sc-3058): add the (scaled) control down residuals to the skip connections.
        if let Some(c) = control {
            if c.down.len() != residuals.len() {
                return Err(mlx_gen::Error::Msg(format!(
                    "controlnet produced {} down residuals, UNet expects {}",
                    c.down.len(),
                    residuals.len()
                )));
            }
            for (r, cr) in residuals.iter_mut().zip(&c.down) {
                *r = add(&*r, cr)?;
            }
        }

        // Mid.
        x = self.mid_resnet0.forward(&x, Some(&temb))?;
        x = self.mid_transformer.forward_ip(&x, encoder_x, ip)?;
        x = self.mid_resnet1.forward(&x, Some(&temb))?;
        // ControlNet: add the (scaled) control mid residual to the mid output.
        if let Some(c) = control {
            x = add(&x, &c.mid)?;
        }

        // Up path — each block pops its skip residuals.
        for block in &self.up_blocks {
            let (out, _) = block.forward_ip(&x, encoder_x, &temb, Some(&mut residuals), ip)?;
            x = out;
        }

        // Conv head.
        let x = group_norm(
            &x,
            &self.conv_norm_out_w,
            &self.conv_norm_out_b,
            GN_GROUPS,
            GN_EPS,
        )?;
        let x = silu_glue(&x)?;
        conv2d(&x, self.conv_out.weight(), self.conv_out.bias(), 1, 1)
    }

    /// Training forward with **per-block gradient checkpointing** (sc-4941 — the `gradient_checkpointing`
    /// behavior for LoRA training): identical compute to [`forward`](Self::forward), but each down/up
    /// macro-block runs inside an `mlx::checkpoint` segment whose explicit inputs are the block hidden
    /// state, its up-path skip residuals, and its trainable LoRA factors — so the reverse pass
    /// recomputes the block (recovering the conv-resnet activation memory that dominates the SDXL
    /// first-step peak) instead of retaining it, while gradients still flow to the LoRA params (a
    /// captured param gets no grad through `checkpoint`, hence the explicit-input threading). The conv
    /// stem/head, the timestep + add embeddings, and the **mid block** (low resolution, cheap) run
    /// normally — their LoRA, if any, is installed on `self` by the caller and trains through ordinary
    /// autograd. LoRA-only (the caller falls LoKr back to the dense path, guarded). `target_paths` is
    /// the resolved LoRA target set; `params` the live factor map; `alpha` the LoRA alpha. The compute
    /// dtype is read off the input `x` (the U-Net was cast in `train_impl`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_block_checkpointed(
        &self,
        x: &Array,
        timestep: f32,
        encoder_x: &Array,
        text_emb: &Array,
        time_ids: &Array,
        target_paths: &[String],
        params: &LoraParams,
        alpha: f32,
    ) -> Result<Array> {
        let batch = x.shape()[0];
        let dtype = x.dtype();

        // Kolors context projection + temb + conv stem — not checkpointed (identical to `forward_core`).
        let projected;
        let encoder_x = match &self.encoder_hid_proj {
            Some(proj) => {
                projected = proj.forward(encoder_x)?;
                &projected
            }
            None => encoder_x,
        };
        let temb = text_time_temb(
            &self.timesteps,
            &self.time_embedding,
            &self.add_time_proj,
            &self.add_embedding,
            timestep,
            text_emb,
            time_ids,
            batch,
            dtype,
        )?;
        let mut x = conv2d(x, self.conv_in.weight(), self.conv_in.bias(), 1, 1)?;

        // Down path — each block checkpointed; its output states (skip residuals) are checkpoint
        // OUTPUTS, collected onto the residual stack for the up path.
        let mut residuals: Vec<Array> = vec![x.clone()];
        for (i, block) in self.down_blocks.iter().enumerate() {
            let (locals, factors) =
                collect_block_lora(&format!("down_blocks.{i}"), target_paths, params)?;
            let mut inputs: Vec<Array> = Vec::with_capacity(1 + factors.len());
            inputs.push(x.clone());
            inputs.extend(factors);
            // The closure must OWN its block (the backward recompute runs after this frame is gone);
            // Arrays are refcounted, so the clone is cheap. `install_threaded_lora` replaces whatever
            // adapters the clone carried with the explicit-input factors, routing grads to `inp`.
            let mut blk = block.clone();
            let ex = encoder_x.clone();
            let tb = temb.clone();
            let mut seg = checkpoint(move |inp: &[Array]| -> MlxResult<Vec<Array>> {
                install_threaded_lora(&mut blk, &locals, &inp[1..], alpha, dtype)?;
                // The block's final output IS its last skip state (`out == output_states.last()`), so
                // return only the output states — emitting `out` separately would put the SAME array
                // twice in the checkpoint's output list and corrupt the multi-output VJP (the skip
                // path's cotangent would be dropped, scrambling the down-block grads). The next block
                // takes `x` from the last state.
                let (_out, res) = blk
                    .forward_ip(&inp[0], &ex, &tb, None, None)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                Ok(res)
            });
            let outs = seg(&inputs)?;
            x = outs
                .last()
                .ok_or_else(|| mlx_gen::Error::Msg("down block produced no output states".into()))?
                .clone();
            residuals.extend(outs);
        }

        // Mid — dense (lowest-resolution block, cheap to retain; its LoRA trains via the adapters the
        // caller installed on `self`).
        x = self.mid_resnet0.forward(&x, Some(&temb))?;
        x = self.mid_transformer.forward_ip(&x, encoder_x, None)?;
        x = self.mid_resnet1.forward(&x, Some(&temb))?;

        // Up path — each block checkpointed; its skip residuals are peeled off the stack (in push
        // order) and threaded as explicit inputs so the block's recompute consumes them.
        for (k, block) in self.up_blocks.iter().enumerate() {
            let (locals, factors) =
                collect_block_lora(&format!("up_blocks.{k}"), target_paths, params)?;
            let kskip = block.num_skip_inputs();
            let skips = residuals.split_off(residuals.len() - kskip); // push order
            let mut inputs: Vec<Array> = Vec::with_capacity(1 + kskip + factors.len());
            inputs.push(x.clone());
            inputs.extend(skips);
            inputs.extend(factors);
            let mut blk = block.clone();
            let ex = encoder_x.clone();
            let tb = temb.clone();
            let mut seg = checkpoint(move |inp: &[Array]| -> MlxResult<Vec<Array>> {
                // `forward_ip` pops one skip per resnet from the END — so a push-order Vec yields the
                // last-pushed skip first, matching the dense path.
                let mut skips_v: Vec<Array> = inp[1..1 + kskip].to_vec();
                install_threaded_lora(&mut blk, &locals, &inp[1 + kskip..], alpha, dtype)?;
                let (out, _) = blk
                    .forward_ip(&inp[0], &ex, &tb, Some(&mut skips_v), None)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                Ok(vec![out])
            });
            x = seg(&inputs)?
                .into_iter()
                .next()
                .expect("one up-block output");
        }

        // Conv head — not checkpointed.
        let x = group_norm(
            &x,
            &self.conv_norm_out_w,
            &self.conv_norm_out_b,
            GN_GROUPS,
            GN_EPS,
        )?;
        let x = silu_glue(&x)?;
        conv2d(&x, self.conv_out.weight(), self.conv_out.bias(), 1, 1)
    }

    /// Every LoRA-targetable Linear's diffusers dotted path, matching the vendored `lora.py`
    /// reachable surface (sc-2639): down/up attention (`to_q/k/v`, `to_out.0`), the `proj_in`/`proj_out`
    /// projections, and each resnet's `time_emb_proj`. **`mid_block` is intentionally omitted** — the
    /// vendored mlx-examples UNet names it `mid_blocks.1.…`, so community/diffusers LoRA keys
    /// (`mid_block.attentions.0.…`) never match and the vendored path silently drops them; this port
    /// reproduces that exactly. The correct/complete mid_block + ff coverage (strictly more than the
    /// vendored path) is sc-2671. This list also builds the kohya `flattened→dotted` lookup table.
    pub fn lora_target_paths(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, b) in self.down_blocks.iter().enumerate() {
            b.lora_target_paths(&format!("down_blocks.{i}"), &mut out);
        }
        for (k, b) in self.up_blocks.iter().enumerate() {
            b.lora_target_paths(&format!("up_blocks.{k}"), &mut out);
        }
        out
    }

    /// The **complete** LoRA-targetable surface (sc-2671), strictly larger than the vendored-faithful
    /// [`lora_target_paths`](Self::lora_target_paths): the 515 down/up attention+proj+time_emb paths
    /// **plus** `mid_block.attentions.0` (attention + `proj_in`/`proj_out`) — which the vendored
    /// mlx-examples UNet names `mid_blocks.1.…` and so silently drops — **plus** the GEGLU feed-forward
    /// (`ff.net.0.proj`, `ff.net.2`) of every cross-attention transformer (down + mid + up). Used to
    /// build the kohya lookup table when complete coverage is requested; `mid_block`/`ff` deltas are
    /// reachable through [`AdaptableHost::adaptable_mut`] (the merge layer row-splits a `ff.net.0.proj`
    /// delta into `linear1`/`linear2`). This list is **Linear-only**; the conv-layer LoRA targets are
    /// enumerated separately by [`conv_target_paths`](Self::conv_target_paths) (sc-2919) and folded
    /// into the same complete table by the adapter merge.
    pub fn lora_target_paths_complete(&self) -> Vec<String> {
        let mut out = self.lora_target_paths();
        // mid_block attention + proj (the +82 the vendored path can't reach) and the two mid resnet
        // `time_emb_proj`s (symmetric with the down/up resnet time_emb already in the faithful 515).
        self.mid_resnet0
            .lora_target_paths("mid_block.resnets.0", &mut out);
        self.mid_transformer
            .lora_target_paths("mid_block.attentions.0", &mut out);
        self.mid_resnet1
            .lora_target_paths("mid_block.resnets.1", &mut out);
        // GEGLU feed-forward across every cross-attention transformer.
        for (i, b) in self.down_blocks.iter().enumerate() {
            b.lora_target_paths_ff(&format!("down_blocks.{i}"), &mut out);
        }
        self.mid_transformer
            .lora_target_paths_ff("mid_block.attentions.0", &mut out);
        for (k, b) in self.up_blocks.iter().enumerate() {
            b.lora_target_paths_ff(&format!("up_blocks.{k}"), &mut out);
        }
        out
    }

    /// Every **conv-layer** LoRA target (sc-2919), as diffusers dotted paths: `conv_in`, `conv_out`,
    /// each resnet's `conv1`/`conv2`/`conv_shortcut` (down / mid / up), and each down/up-sampler's
    /// `conv`. These are merged only under [`crate::adapters::LoraCoverage::Complete`] — the
    /// Linear-only vendored coverage drops them. Used to extend the kohya `flattened → dotted`
    /// lookup table so conv keys (`lora_unet_..._conv1`, `..._downsamplers_0_conv`, `conv_in`, …)
    /// resolve; the merge layer dispatches each to [`AdaptableHost::adaptable_conv_mut`] (or, for
    /// the 1×1 `conv_shortcut`, the reshaped Linear merge).
    pub fn conv_target_paths(&self) -> Vec<String> {
        let mut out = vec!["conv_in".to_string(), "conv_out".to_string()];
        for (i, b) in self.down_blocks.iter().enumerate() {
            b.conv_target_paths(&format!("down_blocks.{i}"), &mut out);
        }
        self.mid_resnet0
            .conv_target_paths("mid_block.resnets.0", &mut out);
        self.mid_resnet1
            .conv_target_paths("mid_block.resnets.1", &mut out);
        for (k, b) in self.up_blocks.iter().enumerate() {
            b.conv_target_paths(&format!("up_blocks.{k}"), &mut out);
        }
        out
    }
}

/// Collect the per-block LoRA factor inputs for [`UNet2DConditionModel::forward_block_checkpointed`]
/// (sc-4941): for every trained `target_paths` entry under `prefix` (e.g. `down_blocks.0`), return its
/// LOCAL path (after the prefix) plus its raw `[r,in]`/`[out,r]` factors looked up in `params`,
/// interleaved `[a_0, b_0, a_1, b_1, …]` in `locals` order. Threading these as explicit checkpoint
/// inputs is what lets gradients reach them (a captured param gets no grad through `checkpoint`).
fn collect_block_lora(
    prefix: &str,
    target_paths: &[String],
    params: &LoraParams,
) -> Result<(Vec<String>, Vec<Array>)> {
    let dot = format!("{prefix}.");
    let mut locals = Vec::new();
    let mut factors = Vec::new();
    for path in target_paths {
        let Some(local) = path.strip_prefix(&dot) else {
            continue;
        };
        let ak = format!("{path}.lora_a");
        let bk = format!("{path}.lora_b");
        let a = params
            .get(ak.as_str())
            .ok_or_else(|| mlx_gen::Error::Msg(format!("LoRA param missing: {ak}")))?;
        let b = params
            .get(bk.as_str())
            .ok_or_else(|| mlx_gen::Error::Msg(format!("LoRA param missing: {bk}")))?;
        locals.push(local.to_string());
        factors.push(a.clone());
        factors.push(b.clone());
    }
    Ok((locals, factors))
}

/// Reinstall threaded LoRA factors onto a freshly-cloned block inside a checkpoint segment — the SAME
/// `(transpose, alpha/rank fold, scale=1)` `install_training_lora` applies, so the checkpointed block
/// forward is numerically identical to the installed-adapter path, plus the dtype-follow on the block
/// hidden state (under the bf16 training cast the f32 factors must join the bf16 stream or every
/// adapted Linear re-promotes the block to f32). `factors` is `[a_0, b_0, a_1, b_1, …]` matching
/// `locals`. Grads flow back to the factors (f32) through the `astype` VJP (master-weights).
fn install_threaded_lora<H: AdaptableHost>(
    block: &mut H,
    locals: &[String],
    factors: &[Array],
    alpha: f32,
    dtype: Dtype,
) -> MlxResult<()> {
    for (j, local) in locals.iter().enumerate() {
        let a = factors[2 * j].t().as_dtype(dtype)?; // [r,in] -> [in,r]
        let rank = a.shape()[1] as f32;
        let b = factors[2 * j + 1]
            .t() // [out,r] -> [r,out]
            .multiply(Array::from_slice(&[alpha / rank], &[1]))?
            .as_dtype(dtype)?;
        let segs: Vec<&str> = local.split('.').collect();
        block
            .adaptable_mut(&segs)
            .ok_or_else(|| Exception::custom(format!("checkpoint LoRA target not found: {local}")))?
            .set_adapters(vec![Adapter::Lora { a, b, scale: 1.0 }]);
    }
    Ok(())
}

impl AdaptableHost for UNet2DConditionModel {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["down_blocks", i, rest @ ..] => self
                .down_blocks
                .get_mut(i.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["up_blocks", k, rest @ ..] => self
                .up_blocks
                .get_mut(k.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            // mid_block (sc-2671 complete coverage). Routable here, but the vendored coverage path
            // gates mid_block/ff keys out so the faithful 515-module merge is unaffected; only the
            // opt-in complete coverage actually merges into these.
            ["mid_block", "attentions", "0", rest @ ..] => self.mid_transformer.adaptable_mut(rest),
            ["mid_block", "resnets", "0", rest @ ..] => self.mid_resnet0.adaptable_mut(rest),
            ["mid_block", "resnets", "1", rest @ ..] => self.mid_resnet1.adaptable_mut(rest),
            _ => None,
        }
    }

    /// Conv-layer LoRA routing (sc-2919) — the conv analog of [`adaptable_mut`](Self::adaptable_mut).
    /// `conv_in`/`conv_out` resolve directly; the resnet/sampler convs delegate into the down / up /
    /// mid sub-hosts. (The 1×1 `conv_shortcut` is a Linear, reached through `adaptable_mut`.)
    fn adaptable_conv_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableConv2d> {
        match path {
            ["conv_in"] => Some(&mut self.conv_in),
            ["conv_out"] => Some(&mut self.conv_out),
            ["down_blocks", i, rest @ ..] => self
                .down_blocks
                .get_mut(i.parse::<usize>().ok()?)?
                .adaptable_conv_mut(rest),
            ["up_blocks", k, rest @ ..] => self
                .up_blocks
                .get_mut(k.parse::<usize>().ok()?)?
                .adaptable_conv_mut(rest),
            ["mid_block", "resnets", "0", rest @ ..] => self.mid_resnet0.adaptable_conv_mut(rest),
            ["mid_block", "resnets", "1", rest @ ..] => self.mid_resnet1.adaptable_conv_mut(rest),
            _ => None,
        }
    }
}
