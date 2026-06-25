//! SD3.5 triple text-encoder aggregator (slice **E2**, sc-7861).
//!
//! SD3.5 conditions on **three** text encoders, all reused unchanged from sibling crates — this
//! module is ONLY the aggregator that combines their outputs into the two tensors the MMDiT reads.
//! There is no net-new encoder here:
//!
//! * **CLIP-L** (`text_encoder`) — [`mlx_gen_sdxl::ClipTextEncoder`] with a 768→768 text projection
//!   (SD3 ships CLIP-L as `CLIPTextModelWithProjection`, unlike SDXL where TE1 has no projection).
//! * **CLIP-G / OpenCLIP-bigG** (`text_encoder_2`) — [`mlx_gen_sdxl::ClipTextEncoder`] with the
//!   1280-wide [`ClipTextConfig::sdxl_te2`] (1280→1280 projection); identical to SDXL's TE2.
//! * **T5-XXL** (`text_encoder_3`) — [`mlx_gen_flux::T5TextEncoder`] (24 blocks, 4096-dim), identical
//!   to FLUX's T5, run over `max_sequence_length = 256` tokens (SD3's default).
//!
//! ## Aggregation (matches diffusers `StableDiffusion3Pipeline.encode_prompt` exactly)
//!
//! For each CLIP encoder diffusers runs with `output_hidden_states=True` and takes:
//! * the **penultimate** hidden state (`hidden_states[-2]`, before the final layer-norm) for the
//!   per-token *context* portion, and
//! * the **projected pooled** output (`prompt_embeds[0]`, the text-projected EOS token) for the
//!   *pooled* vector.
//!
//! Then (diffusers `_get_clip_prompt_embeds` + `encode_prompt`):
//!
//! ```text
//! pooled  = cat([clip_l.pooled (768), clip_g.pooled (1280)], dim=-1)            -> [B, 2048]
//!
//! clip_ctx = cat([clip_l.hidden[-2] (77x768), clip_g.hidden[-2] (77x1280)], -1) -> [B, 77, 2048]
//! clip_ctx = pad(clip_ctx, hidden 2048 -> 4096)  (zero-pad on the RIGHT)        -> [B, 77, 4096]
//! context  = cat([clip_ctx, t5.seq (256x4096)], dim=-2)   (CLIP first, then T5) -> [B, 333, 4096]
//! ```
//!
//! The zero-pad placement (trailing hidden dim), the concat order (CLIP context THEN T5 along the
//! sequence axis), and the CLIP penultimate-vs-pooled split are the named TE parity risks from the
//! spike (sc-7850); they are mirrored verbatim here.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::Result;
use mlx_gen_flux::T5TextEncoder;
use mlx_gen_sdxl::{ClipTextConfig, ClipTextEncoder};

/// SD3's per-encoder padded CLIP sequence length (`tokenizer.model_max_length` = 77).
pub const CLIP_SEQ_LEN: usize = 77;
/// SD3's default T5 sequence length (`StableDiffusion3Pipeline.encode_prompt(max_sequence_length=256)`).
pub const T5_SEQ_LEN: usize = 256;
/// The combined context sequence length: CLIP (77) then T5 (256) along the sequence axis.
pub const CONTEXT_SEQ_LEN: usize = CLIP_SEQ_LEN + T5_SEQ_LEN; // 333
/// CLIP-L hidden width (`text_encoder`).
pub const CLIP_L_DIM: usize = 768;
/// CLIP-G / OpenCLIP-bigG hidden width (`text_encoder_2`).
pub const CLIP_G_DIM: usize = 1280;
/// The concatenated CLIP hidden width before zero-padding (`768 + 1280`).
pub const CLIP_CONTEXT_DIM: usize = CLIP_L_DIM + CLIP_G_DIM; // 2048
/// The MMDiT `joint_attention_dim` / T5 hidden width — the padded context hidden dim.
pub const JOINT_ATTENTION_DIM: usize = 4096;
/// SD3 pooled-projection dim (`pooled_projection_dim`): `clip_l.pooled (768) + clip_g.pooled (1280)`.
pub const POOLED_DIM: usize = CLIP_L_DIM + CLIP_G_DIM; // 2048

/// The CLIP-L (`text_encoder`) config for SD3: a 768-wide, 12-layer CLIP-L **with** the 768→768
/// text projection (SD3 ships it as `CLIPTextModelWithProjection`). This differs from
/// [`ClipTextConfig::sdxl_te1`] (no projection) only in `projection_dim`; everything else — layer
/// count, width, head count, quick-gelu activation — is the identical CLIP-L encoder, reused as-is.
pub fn sd3_clip_l_config() -> ClipTextConfig {
    let mut cfg = ClipTextConfig::sdxl_te1();
    // SD3 CLIP-L is CLIPTextModelWithProjection: a 768->768 text projection over the pooled EOS.
    cfg.projection_dim = Some(CLIP_L_DIM as i32);
    cfg
}

/// The CLIP-G (`text_encoder_2`) config for SD3 — identical to SDXL's OpenCLIP-bigG TE2.
pub fn sd3_clip_g_config() -> ClipTextConfig {
    ClipTextConfig::sdxl_te2()
}

/// The two tensors the SD3.5 MMDiT conditions on, produced by [`Sd3TextEncoders::encode`].
pub struct Sd3Conditioning {
    /// Per-token context `[B, 333, 4096]` — `context_embedder` projects this to `[B, 333, hidden]`.
    pub context: Array,
    /// Pooled text vector `[B, 2048]` — fed (with the timestep) into `time_text_embed`.
    pub pooled: Array,
}

/// The three reused SD3.5 text encoders. This crate does NOT reimplement them — it loads the
/// existing SDXL CLIP encoder twice (CLIP-L + CLIP-G) and the FLUX T5 encoder, then aggregates.
pub struct Sd3TextEncoders {
    /// CLIP-L (`text_encoder`), with a 768→768 projection.
    pub clip_l: ClipTextEncoder,
    /// CLIP-G / OpenCLIP-bigG (`text_encoder_2`), with a 1280→1280 projection.
    pub clip_g: ClipTextEncoder,
    /// T5-XXL (`text_encoder_3`).
    pub t5: T5TextEncoder,
}

impl Sd3TextEncoders {
    /// Construct the three encoders. Convention mirrors the diffusers SD3 checkpoint layout where
    /// `clip_l_prefix`/`clip_g_prefix` are the CLIP `text_model` namespaces and `t5_prefix` is the
    /// T5 encoder namespace (the loader/E5 story wires the actual on-disk prefixes).
    pub fn from_weights(
        clip_l_w: &mlx_gen::weights::Weights,
        clip_l_prefix: &str,
        clip_g_w: &mlx_gen::weights::Weights,
        clip_g_prefix: &str,
        t5_w: &mlx_gen::weights::Weights,
        t5_prefix: &str,
    ) -> Result<Self> {
        Ok(Self {
            clip_l: ClipTextEncoder::from_weights(clip_l_w, clip_l_prefix, &sd3_clip_l_config())?,
            clip_g: ClipTextEncoder::from_weights(clip_g_w, clip_g_prefix, &sd3_clip_g_config())?,
            t5: T5TextEncoder::from_weights(t5_w, t5_prefix)?,
        })
    }

    /// Quantize every Linear in the three encoders to Q4/Q8.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.clip_l.quantize(bits)?;
        self.clip_g.quantize(bits)?;
        self.t5.quantize(bits)?;
        Ok(())
    }

    /// Run all three encoders and aggregate into SD3.5 conditioning.
    ///
    /// * `clip_l_ids` / `clip_g_ids` — `[B, 77]` int32 CLIP token ids.
    /// * `t5_ids` — `[B, 256]` int32 T5 token ids.
    /// * `t5_mask` — optional additive T5 key-padding mask (SD3 runs T5 unmasked by default, so
    ///   `None` matches diffusers; the masked path exists for parity experiments).
    pub fn encode(
        &self,
        clip_l_ids: &Array,
        clip_g_ids: &Array,
        t5_ids: &Array,
        t5_mask: Option<&Array>,
    ) -> Result<Sd3Conditioning> {
        let clip_l = self.clip_l.forward(clip_l_ids)?;
        let clip_g = self.clip_g.forward(clip_g_ids)?;
        let t5_seq = self.t5.forward_masked(t5_ids, t5_mask)?;

        // diffusers `_get_clip_prompt_embeds`: the per-token context is the PENULTIMATE hidden state
        // (`hidden_states[-2]`, before the final layer-norm), not `last_hidden_state`.
        let clip_l_ctx = penultimate(&clip_l.hidden_states)?;
        let clip_g_ctx = penultimate(&clip_g.hidden_states)?;

        build_sd3_conditioning(
            clip_l_ctx,
            &clip_l.pooled,
            clip_g_ctx,
            &clip_g.pooled,
            &t5_seq,
        )
    }
}

/// `hidden_states[-2]` — the penultimate per-layer hidden state.
fn penultimate(hidden_states: &[Array]) -> Result<&Array> {
    let n = hidden_states.len();
    if n < 2 {
        return Err(mlx_gen::Error::Msg(format!(
            "sd3 text: CLIP encoder produced {n} hidden states; need >=2 for hidden_states[-2]"
        )));
    }
    Ok(&hidden_states[n - 2])
}

/// The pure aggregation: combine the three encoders' already-selected outputs into SD3.5
/// conditioning. Split out from [`Sd3TextEncoders::encode`] so it is unit-testable with synthetic
/// tensors (no multi-GB weights). Mirrors diffusers `encode_prompt` step-for-step.
///
/// * `clip_l_ctx` — CLIP-L penultimate hidden `[B, 77, 768]`.
/// * `clip_l_pooled` — CLIP-L projected pooled `[B, 768]`.
/// * `clip_g_ctx` — CLIP-G penultimate hidden `[B, 77, 1280]`.
/// * `clip_g_pooled` — CLIP-G projected pooled `[B, 1280]`.
/// * `t5_seq` — T5 sequence `[B, 256, 4096]`.
pub fn build_sd3_conditioning(
    clip_l_ctx: &Array,
    clip_l_pooled: &Array,
    clip_g_ctx: &Array,
    clip_g_pooled: &Array,
    t5_seq: &Array,
) -> Result<Sd3Conditioning> {
    // pooled = concat(CLIP-L pooled 768, CLIP-G pooled 1280) -> [B, 2048].
    let pooled = concatenate_axis(&[clip_l_pooled, clip_g_pooled], -1)?;

    // CLIP context = concat along hidden of [77x768, 77x1280] -> [B, 77, 2048].
    let clip_ctx = concatenate_axis(&[clip_l_ctx, clip_g_ctx], -1)?;

    // Zero-pad the hidden dim 2048 -> 4096 on the RIGHT (diffusers `F.pad(x, (0, 4096-2048))`).
    let sh = clip_ctx.shape(); // [B, 77, 2048]
    let pad_width = JOINT_ATTENTION_DIM as i32 - sh[sh.len() - 1];
    let clip_ctx = if pad_width > 0 {
        let mut zeros_shape = sh.to_vec();
        let last = zeros_shape.len() - 1;
        zeros_shape[last] = pad_width;
        let zeros = Array::zeros::<f32>(&zeros_shape)?.as_dtype(clip_ctx.dtype())?;
        concatenate_axis(&[&clip_ctx, &zeros], -1)?
    } else {
        clip_ctx
    };

    // context = concat along SEQUENCE of [CLIP 77x4096, T5 256x4096] -> [B, 333, 4096].
    // CLIP first, then T5 (diffusers `torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)`).
    let context = concatenate_axis(&[&clip_ctx, t5_seq], -2)?;

    Ok(Sd3Conditioning { context, pooled })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic ramp tensor of the given shape, so concat ordering is observable per-element.
    fn ramp(shape: &[i32], start: f32) -> Array {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| start + i as f32).collect();
        Array::from_slice(&data, shape)
    }

    /// A constant-valued tensor of the given shape.
    fn filled(shape: &[i32], value: f32) -> Array {
        let n: i32 = shape.iter().product();
        Array::from_slice(&vec![value; n as usize], shape)
    }

    #[test]
    fn pooled_concat_768_plus_1280_is_2048() {
        let l = ramp(&[2, CLIP_L_DIM as i32], 0.0);
        let g = ramp(&[2, CLIP_G_DIM as i32], 1000.0);
        let ctx_l = Array::zeros::<f32>(&[2, 77, CLIP_L_DIM as i32]).unwrap();
        let ctx_g = Array::zeros::<f32>(&[2, 77, CLIP_G_DIM as i32]).unwrap();
        let t5 = Array::zeros::<f32>(&[2, T5_SEQ_LEN as i32, JOINT_ATTENTION_DIM as i32]).unwrap();

        let out = build_sd3_conditioning(&ctx_l, &l, &ctx_g, &g, &t5).unwrap();
        assert_eq!(out.pooled.shape(), &[2, POOLED_DIM as i32]);
        // CLIP-L pooled occupies [0..768], CLIP-G pooled [768..2048] (concat order).
        let p = out.pooled.as_slice::<f32>();
        assert_eq!(p[0], 0.0); // first CLIP-L element
        assert_eq!(p[CLIP_L_DIM - 1], (CLIP_L_DIM - 1) as f32); // last CLIP-L element
        assert_eq!(p[CLIP_L_DIM], 1000.0); // first CLIP-G element
    }

    #[test]
    fn context_shape_is_333_by_4096() {
        let ctx_l = ramp(&[1, CLIP_SEQ_LEN as i32, CLIP_L_DIM as i32], 0.0);
        let ctx_g = ramp(&[1, CLIP_SEQ_LEN as i32, CLIP_G_DIM as i32], 0.0);
        let l_pooled = Array::zeros::<f32>(&[1, CLIP_L_DIM as i32]).unwrap();
        let g_pooled = Array::zeros::<f32>(&[1, CLIP_G_DIM as i32]).unwrap();
        let t5 = ramp(&[1, T5_SEQ_LEN as i32, JOINT_ATTENTION_DIM as i32], 5.0);

        let out = build_sd3_conditioning(&ctx_l, &l_pooled, &ctx_g, &g_pooled, &t5).unwrap();
        assert_eq!(
            out.context.shape(),
            &[1, CONTEXT_SEQ_LEN as i32, JOINT_ATTENTION_DIM as i32]
        );
        assert_eq!(out.context.shape(), &[1, 333, 4096]);
    }

    #[test]
    fn clip_context_hidden_is_zero_padded_on_the_right() {
        // CLIP-L all 1.0, CLIP-G all 2.0; the padded hidden region [2048..4096] must be exactly 0.
        let ctx_l = filled(&[1, CLIP_SEQ_LEN as i32, CLIP_L_DIM as i32], 1.0);
        let ctx_g = filled(&[1, CLIP_SEQ_LEN as i32, CLIP_G_DIM as i32], 2.0);
        let l_pooled = Array::zeros::<f32>(&[1, CLIP_L_DIM as i32]).unwrap();
        let g_pooled = Array::zeros::<f32>(&[1, CLIP_G_DIM as i32]).unwrap();
        let t5 = filled(&[1, T5_SEQ_LEN as i32, JOINT_ATTENTION_DIM as i32], 9.0);

        let out = build_sd3_conditioning(&ctx_l, &l_pooled, &ctx_g, &g_pooled, &t5).unwrap();
        let c = out.context.as_slice::<f32>();
        let hidden = JOINT_ATTENTION_DIM;
        // Inspect the first CLIP token (row 0 of the sequence): [0..768]=1, [768..2048]=2,
        // [2048..4096]=0 (the zero-pad).
        assert_eq!(c[0], 1.0);
        assert_eq!(c[CLIP_L_DIM - 1], 1.0);
        assert_eq!(c[CLIP_L_DIM], 2.0);
        assert_eq!(c[CLIP_CONTEXT_DIM - 1], 2.0);
        assert_eq!(c[CLIP_CONTEXT_DIM], 0.0); // first padded element
        assert_eq!(c[hidden - 1], 0.0); // last padded element of token 0
    }

    #[test]
    fn t5_follows_clip_along_the_sequence_axis() {
        // CLIP context all 1.0 (then zero-padded), T5 all 7.0. Token 77 onward must be T5 (=7.0).
        let ctx_l = filled(&[1, CLIP_SEQ_LEN as i32, CLIP_L_DIM as i32], 1.0);
        let ctx_g = filled(&[1, CLIP_SEQ_LEN as i32, CLIP_G_DIM as i32], 1.0);
        let l_pooled = Array::zeros::<f32>(&[1, CLIP_L_DIM as i32]).unwrap();
        let g_pooled = Array::zeros::<f32>(&[1, CLIP_G_DIM as i32]).unwrap();
        let t5 = filled(&[1, T5_SEQ_LEN as i32, JOINT_ATTENTION_DIM as i32], 7.0);

        let out = build_sd3_conditioning(&ctx_l, &l_pooled, &ctx_g, &g_pooled, &t5).unwrap();
        let hidden = JOINT_ATTENTION_DIM;
        let c = out.context.as_slice::<f32>();
        // Last CLIP token (index 76): first hidden element is CLIP (1.0).
        let clip_last_row = (CLIP_SEQ_LEN - 1) * hidden;
        assert_eq!(c[clip_last_row], 1.0);
        // First T5 token (sequence index 77): every hidden element is T5 (7.0).
        let t5_first_row = CLIP_SEQ_LEN * hidden;
        assert_eq!(c[t5_first_row], 7.0);
        assert_eq!(c[t5_first_row + hidden - 1], 7.0);
    }

    #[test]
    fn dimension_constants_match_diffusers() {
        assert_eq!(CLIP_CONTEXT_DIM, 2048);
        assert_eq!(POOLED_DIM, 2048);
        assert_eq!(JOINT_ATTENTION_DIM, 4096);
        assert_eq!(CONTEXT_SEQ_LEN, 333);
        // The CLIP-L SD3 config gains a 768 projection (vs SDXL TE1's None).
        assert_eq!(sd3_clip_l_config().projection_dim, Some(768));
        assert_eq!(sd3_clip_g_config().projection_dim, Some(1280));
    }
}
