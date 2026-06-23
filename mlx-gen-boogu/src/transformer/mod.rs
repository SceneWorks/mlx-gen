//! The Boogu mixed single/double-stream DiT (`BooguImageTransformer2DModel`) forward.
//!
//! Two entry points share one inner path: [`BooguTransformer::forward`] (text-to-image) and
//! [`BooguTransformer::forward_edit`] (text+image-to-image with one or more reference images). Edit
//! VAE-encodes each reference image, patch-embeds it through `ref_image_patch_embedder` + its own
//! `image_index_embedding[i]` row, refines each independently in `ref_image_refiner`, and prepends
//! those tokens — `[ref₀; …; ref_{N-1}; noise]` — to the image sequence (with each reference and the
//! noise positions shifted by the cumulative `max(ref_h, ref_w)` in the unified RoPE).
//!
//! Text-to-image flow (the reference-image blocks stay dormant):
//! ```text
//!   time_caption_embed:  temb = TimestepEmbedder(sinusoid(t·scale));  caption = Linear(RMSNorm(instr))
//!   patchify(p=2, 16→64) → x_embedder                                 → img tokens  [1, Li, 3360]
//!   context_refiner ×2  (no modulation)        on instruct tokens     [1, Lt, 3360]
//!   noise_refiner   ×2  (modulated)            on img tokens
//!   double_stream   ×8  (joint instruct↔img attn + img self-attn)
//!   fuse → [instruct; img]                                            [1, Lt+Li, 3360]
//!   single_stream   ×32 (modulated)            on the joint sequence
//!   norm_out (LuminaLayerNormContinuous + temb) → Linear(3360→64)
//!   unpatchify(img tokens)                                            → velocity [1, 16, H, W]
//! ```
//!
//! Per-sample `B = 1`: true-CFG runs this twice (cond/uncond) rather than padding a batch, so every
//! attention is full/unmasked and numerically identical to the reference's per-sample slice.

mod block;
pub mod rope;

use mlx_rs::fast::{layer_norm, rms_norm};
use mlx_rs::ops::{concatenate_axis, cos, exp, multiply, sin, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::BooguConfig;
use crate::quant::lin;
use block::{DoubleBlock, ModBlock, PlainBlock};
use rope::RopeTables;

/// The Boogu DiT. Carries the text-to-image modules plus the reference-image conditioning path
/// (`ref_image_patch_embedder` + `ref_image_refiner` + `image_index_embedding`) the Edit (E7) forward
/// exercises; the T2I forward simply leaves those dormant.
pub struct BooguTransformer {
    cfg: BooguConfig,
    x_embedder: AdaptableLinear,
    ref_image_patch_embedder: AdaptableLinear,
    image_index_embedding: Array,
    caption_norm: Array,
    caption_linear: AdaptableLinear,
    time_lin1: AdaptableLinear,
    time_lin2: AdaptableLinear,
    context_refiner: Vec<PlainBlock>,
    noise_refiner: Vec<ModBlock>,
    ref_image_refiner: Vec<ModBlock>,
    double_stream: Vec<DoubleBlock>,
    single_stream: Vec<ModBlock>,
    norm_out_lin1: AdaptableLinear,
    norm_out_lin2: AdaptableLinear,
}

impl BooguTransformer {
    /// Build from a loaded `transformer/` weight set (already validated by [`crate::convert`]).
    pub fn from_weights(w: &Weights, cfg: &BooguConfig) -> Result<Self> {
        let (heads, kv, hd) = (
            cfg.num_attention_heads as i32,
            cfg.num_kv_heads as i32,
            cfg.head_dim() as i32,
        );
        let eps = cfg.norm_eps;

        let plain = |name: String| PlainBlock::from_weights(w, &name, heads, kv, hd, eps);
        let mod_ = |name: String| ModBlock::from_weights(w, &name, heads, kv, hd, eps);
        let dbl = |name: String| DoubleBlock::from_weights(w, &name, heads, kv, hd, eps);

        Ok(Self {
            cfg: cfg.clone(),
            x_embedder: lin(w, "x_embedder", true)?,
            ref_image_patch_embedder: lin(w, "ref_image_patch_embedder", true)?,
            image_index_embedding: w.require("image_index_embedding")?.clone(),
            caption_norm: w
                .require("time_caption_embed.caption_embedder.0.weight")?
                .clone(),
            caption_linear: lin(w, "time_caption_embed.caption_embedder.1", true)?,
            time_lin1: lin(w, "time_caption_embed.timestep_embedder.linear_1", true)?,
            time_lin2: lin(w, "time_caption_embed.timestep_embedder.linear_2", true)?,
            context_refiner: (0..cfg.num_refiner_layers)
                .map(|i| plain(format!("context_refiner.{i}")))
                .collect::<Result<_>>()?,
            noise_refiner: (0..cfg.num_refiner_layers)
                .map(|i| mod_(format!("noise_refiner.{i}")))
                .collect::<Result<_>>()?,
            ref_image_refiner: (0..cfg.num_refiner_layers)
                .map(|i| mod_(format!("ref_image_refiner.{i}")))
                .collect::<Result<_>>()?,
            double_stream: (0..cfg.num_double_stream_layers)
                .map(|i| dbl(format!("double_stream_layers.{i}")))
                .collect::<Result<_>>()?,
            single_stream: (0..cfg.num_single_stream_layers())
                .map(|i| mod_(format!("single_stream_layers.{i}")))
                .collect::<Result<_>>()?,
            norm_out_lin1: lin(w, "norm_out.linear_1", true)?,
            norm_out_lin2: lin(w, "norm_out.linear_2", true)?,
        })
    }

    /// Text-to-image velocity prediction.
    ///
    /// - `latent`: `[1, 16, H, W]` (H, W multiples of `patch_size`),
    /// - `timestep`: `[1]` f32 (raw, pre-scale),
    /// - `instruction_hidden`: `[1, L, 4096]` raw Qwen3-VL `last_hidden_state`,
    /// - `instruction_mask`: `[1, L]` (counts the valid leading tokens).
    ///
    /// Returns the velocity `[1, 16, H, W]`.
    pub fn forward(
        &self,
        latent: &Array,
        timestep: &Array,
        instruction_hidden: &Array,
        instruction_mask: &Array,
    ) -> Result<Array> {
        self.forward_inner(latent, &[], timestep, instruction_hidden, instruction_mask)
    }

    /// Edit (text+image-to-image) velocity prediction with **one or more** reference images. Identical
    /// to [`Self::forward`] but with `ref_latents` (each `[1, 16, rH, rW]`, a VAE-encoded reference)
    /// packed — each through `ref_image_patch_embedder` + its own `image_index_embedding[i]` row +
    /// `ref_image_refiner` — *before* the noise tokens in the combined image sequence
    /// (`[ref₀; …; ref_{N-1}; noise]`). An empty slice is exactly [`Self::forward`] (text-to-image).
    /// The Boogu DiT supports up to 5 references (the `image_index_embedding` row count).
    pub fn forward_edit(
        &self,
        latent: &Array,
        ref_latents: &[Array],
        timestep: &Array,
        instruction_hidden: &Array,
        instruction_mask: &Array,
    ) -> Result<Array> {
        self.forward_inner(
            latent,
            ref_latents,
            timestep,
            instruction_hidden,
            instruction_mask,
        )
    }

    /// Shared T2I / edit forward. With an empty `ref_latents` this is the exact text-to-image path
    /// (no reference block, `combined_image == image`); with one or more it prepends the refined
    /// reference-image tokens and shifts the noise positions per the OmniGen2 unified RoPE.
    fn forward_inner(
        &self,
        latent: &Array,
        ref_latents: &[Array],
        timestep: &Array,
        instruction_hidden: &Array,
        instruction_mask: &Array,
    ) -> Result<Array> {
        let p = self.cfg.patch_size as i32;
        let (h, w) = (latent.shape()[2], latent.shape()[3]);
        let (ht, wt) = (h / p, w / p);
        let img_len = ht * wt;

        // Run in the model (weight) dtype — typically bf16 — to match the reference's compute path;
        // the dense Linear feeds activations to matmul as-is (no upcast).
        let dt = self.caption_norm.dtype();
        let latent = latent.as_dtype(dt)?;

        // Valid instruction length (B = 1): slice off any padding.
        let cap_len = sum(&instruction_mask.as_dtype(Dtype::Float32)?, false)?.item::<f32>() as i32;
        let instruct = slice_axis1(&instruction_hidden.as_dtype(dt)?, 0, cap_len)?;

        // Timestep + caption embedding.
        let temb = self.timestep_embed(timestep)?; // [1, 1, 1024]
        let caption = self.caption_linear.forward(&rms_norm(
            &instruct,
            &self.caption_norm,
            self.cfg.norm_eps,
        )?)?; // [1, cap, 3360]

        // Patchify the noise latent → target image tokens.
        let img = self.x_embedder.forward(&patchify(&latent, p)?)?; // [1, img_len, 3360]

        // Reference images (Edit): patch-embed each + add its per-image index embedding row. The j-th
        // reference's tokens get `image_index_embedding[j]` (OmniGen2 lineage; max 5 references). The
        // patch grids drive the multi-image RoPE; an empty `ref_latents` is the text-to-image path.
        let mut ref_tokens: Vec<(Array, usize)> = Vec::with_capacity(ref_latents.len());
        let mut ref_grids: Vec<(usize, usize)> = Vec::with_capacity(ref_latents.len());
        for (j, rl) in ref_latents.iter().enumerate() {
            let rl = rl.as_dtype(dt)?;
            let (rht, rwt) = (rl.shape()[2] / p, rl.shape()[3] / p);
            let ref_t = self.ref_image_patch_embedder.forward(&patchify(&rl, p)?)?; // [1, ref_len, 3360]
            let idx = self
                .image_index_embedding
                .take_axis(Array::from_slice(&[j as i32], &[1]), 0)?
                .as_dtype(dt)?
                .reshape(&[1, 1, self.cfg.hidden_size as i32])?;
            let ref_t = mlx_rs::ops::add(&ref_t, &idx)?;
            ref_tokens.push((ref_t, (rht * rwt) as usize));
            ref_grids.push((rht as usize, rwt as usize));
        }

        let rope = if ref_grids.is_empty() {
            RopeTables::build_t2i(
                cap_len as usize,
                ht as usize,
                wt as usize,
                self.cfg.axes_dim_rope[0],
                self.cfg.rope_theta,
            )
        } else {
            RopeTables::build_edit(
                cap_len as usize,
                &ref_grids,
                ht as usize,
                wt as usize,
                self.cfg.axes_dim_rope[0],
                self.cfg.rope_theta,
            )
        };

        let (text_cos, text_sin) = rope.text()?;
        let (noise_cos, noise_sin) = rope.image()?; // target (noise) tokens only
        let (comb_cos, comb_sin) = rope.combined_image()?; // [ref; noise] for img self-attn
        let (joint_cos, joint_sin) = rope.joint();

        // Context refinement (instruction stream).
        let mut instruct_h = caption;
        for blk in &self.context_refiner {
            instruct_h = blk.forward(&instruct_h, &text_cos, &text_sin)?;
        }

        // Noise refinement (target image stream).
        let mut img = img;
        for blk in &self.noise_refiner {
            img = blk.forward(&img, &noise_cos, &noise_sin, &temb)?;
        }

        // Reference refinement: refine EACH reference independently — its own RoPE sub-slice, no
        // cross-image attention (the OmniGen2 batched `ref_image_refiner` masks each reference to
        // itself). Then prepend the refined references to the noise tokens to form the combined image
        // sequence `[ref₀; …; ref_{N-1}; noise]` (Edit). T2I leaves the sequence as the noise tokens.
        let mut img = if ref_tokens.is_empty() {
            img
        } else {
            let mut combined: Vec<Array> = Vec::with_capacity(ref_tokens.len() + 1);
            let mut local = 0usize;
            for (mut ref_t, ref_len) in ref_tokens {
                let (ref_cos, ref_sin) = rope.ref_image_slice(local, ref_len)?;
                for blk in &self.ref_image_refiner {
                    ref_t = blk.forward(&ref_t, &ref_cos, &ref_sin, &temb)?;
                }
                combined.push(ref_t);
                local += ref_len;
            }
            combined.push(img);
            let refs: Vec<&Array> = combined.iter().collect();
            concatenate_axis(&refs, 1)?
        };

        // Dual-stream blocks (joint instruct↔combined-image attn + combined-image self-attn).
        for blk in &self.double_stream {
            let (ni, nt) = blk.forward(
                &img,
                &instruct_h,
                &joint_cos,
                &joint_sin,
                &comb_cos,
                &comb_sin,
                &temb,
            )?;
            img = ni;
            instruct_h = nt;
        }

        // Fuse to the joint sequence, then single-stream blocks.
        let mut joint = concatenate_axis(&[&instruct_h, &img], 1)?; // [1, cap+ref+img, 3360]
        for blk in &self.single_stream {
            joint = blk.forward(&joint, &joint_cos, &joint_sin, &temb)?;
        }

        // Continuous-AdaLN output projection (LuminaLayerNormContinuous, eps 1e-6, no affine).
        let scale = self.norm_out_lin1.forward(&silu(&temb)?)?; // [1, 1, 3360]
        let normed = layer_norm(&joint, None, None, 1e-6)?;
        let normed = multiply(&normed, &mlx_rs::ops::add(&scale, Array::from_f32(1.0))?)?;
        let out = self.norm_out_lin2.forward(&normed)?; // [1, cap+ref+img, 64]

        // Unpatchify the trailing target-image tokens into the velocity (the reference tokens, when
        // present, are dropped — only the noise/target slice is the prediction).
        let total = out.shape()[1];
        let img_tokens = slice_axis1(&out, total - img_len, total)?;
        unpatchify(&img_tokens, ht, wt, p, self.cfg.out_channels as i32)
    }

    /// `Lumina2CombinedTimestepCaptionEmbedding` timestep branch:
    /// `sinusoid(timestep · timestep_scale, 256) → Linear → SiLU → Linear` → `[1, 1, 1024]`.
    fn timestep_embed(&self, timestep: &Array) -> Result<Array> {
        let scaled = multiply(
            &timestep.as_dtype(Dtype::Float32)?,
            scalar(self.cfg.timestep_scale),
        )?;
        // Sinusoid in f32 (the reference builds it in f32), then cast to the model dtype like the
        // reference's `timestep_proj.to(dtype)` before the embedder MLP.
        let proj = sinusoidal_timestep(&scaled, 256)?.as_dtype(self.caption_norm.dtype())?; // [1, 256]
        let t = self.time_lin1.forward(&proj)?;
        let t = silu(&t)?;
        let t = self.time_lin2.forward(&t)?; // [1, 1024]
        Ok(t.expand_dims(1)?) // [1, 1, 1024]
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.x_embedder
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.ref_image_patch_embedder
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.caption_linear
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.time_lin1
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.time_lin2
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        for b in &mut self.context_refiner {
            b.quantize(bits)?;
        }
        for b in &mut self.noise_refiner {
            b.quantize(bits)?;
        }
        for b in &mut self.ref_image_refiner {
            b.quantize(bits)?;
        }
        for b in &mut self.double_stream {
            b.quantize(bits)?;
        }
        for b in &mut self.single_stream {
            b.quantize(bits)?;
        }
        self.norm_out_lin1
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.norm_out_lin2
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        Ok(())
    }
}

// ── Shared helpers ──────────────────────────────────────────────────────────────────────────

/// Join a module prefix with a leaf name, tolerating an empty prefix.
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Slice `[b, L, ...]` along the sequence axis (axis 1) to `[start, end)`.
pub(crate) fn slice_axis1(x: &Array, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), 1)?)
}

/// Expand `[b, s, hkv, hd]` → `[b, s, hkv·groups, hd]`, repeating each kv head `groups` times
/// consecutively (= `repeat_interleave` over the head axis, matching the reference).
pub(crate) fn repeat_kv(x: &Array, groups: i32) -> Result<Array> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let sh = x.shape();
    let (b, s, hkv, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let x = x.expand_dims(3)?; // [b, s, hkv, 1, hd]
    let x = mlx_rs::ops::broadcast_to(&x, &[b, s, hkv, groups, hd])?;
    Ok(x.reshape(&[b, s, hkv * groups, hd])?)
}

/// diffusers `get_timestep_embedding(x, dim, flip_sin_to_cos=True, downscale_freq_shift=0,
/// max_period=10000)`: `freq_i = 10000^(−i/half)`, `emb = x·freq`, `concat([cos, sin], -1)` (cos
/// first). `x`: `[N]` → `[N, dim]`. `ln(10000)` in f64 to match `math.log` rounding.
fn sinusoidal_timestep(x: &Array, dim: i32) -> Result<Array> {
    let half = dim / 2;
    let arange: Vec<f32> = (0..half).map(|i| i as f32).collect();
    let arange = Array::from_slice(&arange, &[half]);
    let neg_ln = -(10000f64.ln()) as f32;
    let exponent = mlx_rs::ops::divide(&multiply(&arange, scalar(neg_ln))?, scalar(half as f32))?;
    let freqs = exp(&exponent)?; // [half]
    let axis = x.shape().len() as i32;
    let emb = multiply(&x.expand_dims(axis)?, &freqs)?; // [N, half]
    Ok(concatenate_axis(&[&cos(&emb)?, &sin(&emb)?], -1)?)
}

/// `c (h p1) (w p2) -> (h w) (p1 p2 c)` with batch: `[1, C, H, W] → [1, (H/p)(W/p), p·p·C]`.
fn patchify(latent: &Array, p: i32) -> Result<Array> {
    let sh = latent.shape();
    let (b, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
    let (ht, wt) = (h / p, w / p);
    let x = latent.reshape(&[b, c, ht, p, wt, p])?; // B, C, h, p1, w, p2
    let x = x.transpose_axes(&[0, 2, 4, 3, 5, 1])?; // B, h, w, p1, p2, C
    Ok(x.reshape(&[b, ht * wt, p * p * c])?)
}

/// `(h w) (p1 p2 c) -> c (h p1) (w p2)` with batch: `[1, (h)(w), p·p·C] → [1, C, h·p, w·p]`.
fn unpatchify(tokens: &Array, ht: i32, wt: i32, p: i32, c: i32) -> Result<Array> {
    let b = tokens.shape()[0];
    let x = tokens.reshape(&[b, ht, wt, p, p, c])?; // B, h, w, p1, p2, C
    let x = x.transpose_axes(&[0, 5, 1, 3, 2, 4])?; // B, C, h, p1, w, p2
    Ok(x.reshape(&[b, c, ht * p, wt * p])?)
}
