//! Ideogram 4 text-to-image pipeline: Qwen3-VL text encode → two-DiT asymmetric-CFG flow-matching
//! denoise → latent de-normalize + unpatchify + VAE decode. Port of `Ideogram4Pipeline.__call__`.
//!
//! The conditional DiT runs over the full `[text ; image]` sequence; the unconditional DiT runs
//! over the **image-only** slice with zeroed conditioning (asymmetric CFG). Per step the velocities
//! combine `v = g·pos_v + (1−g)·neg_v` and Euler-step `z += v·(s−t)`. Tokenization (the Qwen3-VL
//! chat template) is the caller's job — `generate` takes `input_ids`.

use std::path::Path;

use mlx_rs::ops::{add, concatenate_axis, multiply};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{CancelFlag, Error, Progress, Result};
use mlx_gen_flux2::Flux2Vae;

use crate::config::Ideogram4DitConfig;
use crate::latent_norm::{LATENT_SCALE, LATENT_SHIFT};
use crate::loader::{
    load_text_encoder, load_tokenizer, load_transformer, load_unconditional_transformer, load_vae,
};
use crate::scheduler::{make_step_intervals, LogitNormalSchedule};
use crate::text_encoder::Ideogram4TextEncoder;
use crate::transformer::Ideogram4Transformer;

/// `patch_size * ae_scale_factor` — height/width must be a multiple of this (=16).
pub const PATCH: u32 = 2;
pub const AE_SCALE: u32 = 8;
const IMAGE_POSITION_OFFSET: i32 = 65536;
const LLM_TOKEN_INDICATOR: i32 = 3;
const OUTPUT_IMAGE_INDICATOR: i32 = 2;
/// Reference `Ideogram4PipelineConfig.max_text_tokens` — a longer prompt is rejected by `_tokenize`.
const MAX_TEXT_TOKENS: usize = 2048;

pub struct Ideogram4Pipeline {
    cond: Ideogram4Transformer,
    uncond: Ideogram4Transformer,
    te: Ideogram4TextEncoder,
    vae: Flux2Vae,
    tok: TextTokenizer,
    dit: Ideogram4DitConfig,
}

impl Ideogram4Pipeline {
    /// Load all components (2 DiTs + Qwen3-VL text encoder + VAE + tokenizer) from a converted
    /// snapshot dir.
    pub fn load(root: &Path) -> Result<Self> {
        Ok(Self {
            cond: load_transformer(root)?,
            uncond: load_unconditional_transformer(root)?,
            te: load_text_encoder(root)?,
            vae: load_vae(root)?,
            tok: load_tokenizer(root)?,
            dit: Ideogram4DitConfig::v4(),
        })
    }

    /// Tokenize a prompt to `input_ids` exactly as the reference `_tokenize`: wrap it in the
    /// Qwen3-VL single-user chat template ([`ChatTemplate::QwenInstruct`](mlx_gen::tokenizer::ChatTemplate::QwenInstruct))
    /// and encode with `add_special_tokens=false`. Rejects a prompt longer than `MAX_TEXT_TOKENS`.
    /// The prompt is the model's native **JSON caption** string (SceneWorks builds it); plain text
    /// is out-of-distribution.
    pub fn tokenize(&self, prompt: &str) -> Result<Vec<i32>> {
        let ids = self.tok.encode_chat_ids(prompt, false)?;
        if ids.len() > MAX_TEXT_TOKENS {
            return Err(Error::Msg(format!(
                "prompt has {} tokens, exceeds max_text_tokens={MAX_TEXT_TOKENS}",
                ids.len()
            )));
        }
        Ok(ids)
    }

    /// [`tokenize`](Self::tokenize) the prompt, then [`generate`](Self::generate) — the top-level
    /// text-to-image entry point.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_from_prompt(
        &self,
        prompt: &str,
        height: u32,
        width: u32,
        num_steps: usize,
        guidance: f32,
        mu: f64,
        seed: u64,
    ) -> Result<Array> {
        let ids = self.tokenize(prompt)?;
        self.generate(&ids, height, width, num_steps, guidance, mu, seed)
    }

    /// Generate one image. `input_ids`: the chat-templated prompt tokens. Returns an RGB `[H, W, 3]`
    /// `uint8` array. No progress/cancellation — see
    /// [`generate_with_progress`](Self::generate_with_progress).
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        input_ids: &[i32],
        height: u32,
        width: u32,
        num_steps: usize,
        guidance: f32,
        mu: f64,
        seed: u64,
    ) -> Result<Array> {
        self.generate_with_progress(
            input_ids,
            height,
            width,
            num_steps,
            guidance,
            mu,
            seed,
            &CancelFlag::new(),
            &mut |_| {},
        )
    }

    /// [`generate`](Self::generate) with cooperative cancellation + step/decode progress — the path
    /// the [`Generator`](mlx_gen::Generator) registry adapter uses. `cancel` is checked at each step
    /// boundary (returns `Err(Error::Canceled)` on trip); `on_progress` receives a
    /// [`Progress::Step`] per denoise step and [`Progress::Decoding`] before the VAE decode. The
    /// per-step `eval` makes the cancel check able to interrupt mid-render (MLX is lazy).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_progress(
        &self,
        input_ids: &[i32],
        height: u32,
        width: u32,
        num_steps: usize,
        guidance: f32,
        mu: f64,
        seed: u64,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let patch = PATCH * AE_SCALE;
        assert!(
            height.is_multiple_of(patch) && width.is_multiple_of(patch),
            "height/width must be multiples of {patch}"
        );
        let grid_h = (height / patch) as i32;
        let grid_w = (width / patch) as i32;
        let num_img = grid_h * grid_w;
        let num_text = input_ids.len() as i32;
        let seq = num_text + num_img;
        let llm_dim = self.dit.llm_features_dim;
        let ch = self.dit.in_channels;

        // ── Text encode (single prompt, no padding → positions 0..num_text) ──
        let ids = Array::from_slice(input_ids, &[1, num_text]);
        let attn = Array::from_slice(&vec![1i32; num_text as usize], &[1, num_text]);
        let te_out = self.te.prompt_embeds(&ids, &attn)?; // [1, num_text, llm_dim]
        let llm_features = concatenate_axis(&[&te_out, &zeros(&[1, num_img, llm_dim])], 1)?; // [1, seq, llm_dim]

        // ── Packed positions / segments / role indicators (host-built) ──
        let pack = Packing::build(num_text, grid_h, grid_w);
        let position_ids = Array::from_slice(&pack.position_ids, &[1, seq, 3]);
        let segment_ids = Array::from_slice(&pack.segment_ids, &[1, seq]);
        let indicator = Array::from_slice(&pack.indicator, &[1, seq]);
        // Negative branch = image-only slice.
        let neg_position_ids = Array::from_slice(&pack.neg_position_ids, &[1, num_img, 3]);
        let neg_segment_ids = Array::from_slice(&pack.neg_segment_ids, &[1, num_img]);
        let neg_indicator = Array::from_slice(&pack.neg_indicator, &[1, num_img]);
        let neg_llm = zeros(&[1, num_img, llm_dim]);

        // ── Init noise + the text-position latent padding ──
        let key = random::key(seed)?;
        let mut z = random::normal::<f32>(&[1, num_img, ch], None, None, Some(&key))?;
        let text_z_padding = zeros(&[1, num_text, ch]);
        let img_range = Array::from_slice(&(num_text..seq).collect::<Vec<i32>>(), &[num_img]);

        // ── Flow-matching Euler denoise (high → low noise) ──
        let schedule = LogitNormalSchedule::for_resolution(height, width, mu, 1.0);
        let si = make_step_intervals(num_steps);
        for i in (0..num_steps).rev() {
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let t_val = schedule.eval(si[i + 1]);
            let s_val = schedule.eval(si[i]);
            let t = Array::from_slice(&[t_val as f32], &[1]);

            let pos_z = concatenate_axis(&[&text_z_padding, &z], 1)?;
            let pos_out = self.cond.forward(
                &llm_features,
                &pos_z,
                &t,
                &position_ids,
                &segment_ids,
                &indicator,
            )?;
            let pos_v = pos_out.take_axis(&img_range, 1)?; // image-token velocities

            let neg_v = self.uncond.forward(
                &neg_llm,
                &z,
                &t,
                &neg_position_ids,
                &neg_segment_ids,
                &neg_indicator,
            )?;

            // v = g·pos_v + (1−g)·neg_v ; z += v·(s−t)
            let v = add(
                &multiply(&pos_v, Array::from_f32(guidance))?,
                &multiply(&neg_v, Array::from_f32(1.0 - guidance))?,
            )?;
            z = add(&z, &multiply(&v, Array::from_f32((s_val - t_val) as f32))?)?;
            eval([&z])?;
            on_progress(Progress::Step {
                current: (num_steps - i) as u32,
                total: num_steps as u32,
            });
        }

        on_progress(Progress::Decoding);
        self.decode(&z, grid_h, grid_w)
    }

    /// De-normalize → unpatchify → VAE decode → RGB u8 `[H, W, 3]`.
    fn decode(&self, z: &Array, grid_h: i32, grid_w: i32) -> Result<Array> {
        let scale = Array::from_slice(&LATENT_SCALE, &[1, 1, 128]);
        let shift = Array::from_slice(&LATENT_SHIFT, &[1, 1, 128]);
        let denorm = add(&multiply(z, &scale)?, &shift)?; // [1, L, 128]

        // Unpatchify to NHWC: [1,gh,gw,2,2,32] → [1,gh,2,gw,2,32] → [1, gh·2, gw·2, 32].
        let latent = denorm
            .reshape(&[1, grid_h, grid_w, 2, 2, 32])?
            .transpose_axes(&[0, 1, 3, 2, 4, 5])?
            .reshape(&[1, grid_h * 2, grid_w * 2, 32])?;

        let decoded = self.vae.decode(&latent)?; // [1, H, W, 3] f32, ~[-1,1]
        let sh = decoded.shape();
        let (h, w) = (sh[1], sh[2]);
        let clamped = mlx_rs::ops::clip(&decoded, (&Array::from_f32(-1.0), &Array::from_f32(1.0)))?;
        let scaled = multiply(
            &add(&clamped, Array::from_f32(1.0))?,
            Array::from_f32(127.5),
        )?;
        let u8img = mlx_rs::ops::round(&scaled, None)?.as_dtype(Dtype::Uint8)?;
        Ok(u8img.reshape(&[h, w, 3])?)
    }
}

/// f32 zeros of the given shape.
fn zeros(shape: &[i32]) -> Array {
    let n: i32 = shape.iter().product();
    Array::from_slice(&vec![0f32; n as usize], shape)
}

/// Host-built packed sequence metadata: text tokens (`LLM`) then image tokens (`IMAGE`).
struct Packing {
    position_ids: Vec<i32>,
    segment_ids: Vec<i32>,
    indicator: Vec<i32>,
    neg_position_ids: Vec<i32>,
    neg_segment_ids: Vec<i32>,
    neg_indicator: Vec<i32>,
}

impl Packing {
    fn build(num_text: i32, grid_h: i32, grid_w: i32) -> Self {
        let num_img = grid_h * grid_w;
        let mut position_ids = Vec::new();
        let mut indicator = Vec::new();
        for i in 0..num_text {
            position_ids.extend_from_slice(&[i, i, i]);
            indicator.push(LLM_TOKEN_INDICATOR);
        }
        let mut neg_position_ids = Vec::new();
        for j in 0..num_img {
            let (h, w) = (j / grid_w, j % grid_w);
            let p = [0, h + IMAGE_POSITION_OFFSET, w + IMAGE_POSITION_OFFSET];
            position_ids.extend_from_slice(&p);
            neg_position_ids.extend_from_slice(&p);
            indicator.push(OUTPUT_IMAGE_INDICATOR);
        }
        let seq = num_text + num_img;
        Self {
            position_ids,
            segment_ids: vec![1; seq as usize],
            indicator,
            neg_position_ids,
            neg_segment_ids: vec![1; num_img as usize],
            neg_indicator: vec![OUTPUT_IMAGE_INDICATOR; num_img as usize],
        }
    }
}
