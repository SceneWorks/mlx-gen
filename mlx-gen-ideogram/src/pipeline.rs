//! Ideogram 4 text-to-image pipeline: Qwen3-VL text encode → flow-matching denoise → latent
//! de-normalize + unpatchify + VAE decode. Port of `Ideogram4Pipeline.__call__`.
//!
//! Two denoise modes share this pipeline, selected by whether the **unconditional** DiT is present:
//! * **Quality (asymmetric CFG, default)** — the conditional DiT runs over the full `[text ; image]`
//!   sequence; the unconditional DiT runs over the **image-only** slice with zeroed conditioning.
//!   Per step the velocities combine `v = g·pos_v + (1−g)·neg_v`.
//! * **Turbo (CFG-free single DiT, issue #488)** — `uncond` is `None` and the conditional DiT
//!   carries the ostris **TurboTime** LoRA; per step `v = pos_v` (no negative branch, guidance off),
//!   so a render costs one DiT forward over ~8 steps instead of two over ~48.
//!
//! Both Euler-step `z += v·(s−t)`. Tokenization (the Qwen3-VL chat template) is the caller's job —
//! `generate` takes `input_ids`.
//!
//! **Edit (img2img / mask inpaint, sc-6303/6330).** When an [`EditInit`] is supplied, the denoise
//! starts from a VAE-encoded source latent noised to a strength-derived step instead of pure noise
//! (img2img / Remix); an optional latent-grid mask additionally pins the keep region (mask 0) to the
//! source re-noised to each step's σ while regenerating the white region (mask 1) — the classic
//! masked-img2img inpaint on this same flow-match loop (no dedicated inpaint UNet). With no
//! `EditInit`, the path is byte-identical to the original text-to-image render.

use std::path::Path;

use mlx_rs::ops::{add, concatenate_axis, divide, multiply, subtract};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use mlx_gen::image::{resize_lanczos_u8, resize_nearest_u8};
use mlx_gen::media::Image;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{CancelFlag, Error, Progress, Result};
use mlx_gen_flux2::Flux2Vae;

use crate::adapters::apply_ideogram_adapters;
use crate::config::Ideogram4DitConfig;
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
/// Per-step guidance schedule tail: the reference `DEFAULT_GUIDANCE_SCHEDULE` drops to 3.0 for the
/// final `POLISH_STEPS` low-noise steps (the rest use the base guidance, typically 7.0).
const POLISH_STEPS: usize = 3;
const POLISH_GUIDANCE: f32 = 3.0;
/// Ideogram 4 reference scheduler presets — `(mu, std)` are tuned PER step-count (the V4 presets),
/// not constants: TURBO_12 `(0.5, 1.75)`, DEFAULT_20 `(0.0, 1.75)`, QUALITY_48 `(0.0, 1.5)`. An
/// arbitrary step count picks the nearest preset. The earlier hardcoded `std=1.0` starved the
/// low-noise detail steps and smeared every render.
fn preset_mu_std(num_steps: usize) -> (f64, f64) {
    match num_steps {
        s if s <= 15 => (0.5, 1.75), // V4_TURBO_12
        s if s <= 33 => (0.0, 1.75), // V4_DEFAULT_20
        _ => (0.0, 1.5),             // V4_QUALITY_48
    }
}

/// Edit (img2img / mask inpaint) conditioning prepared **once per request** (seed-independent) by
/// [`Ideogram4Pipeline::prepare_edit`]: the BN-normalized packed source latent `[1, num_img, 128]`,
/// an optional latent-grid inpaint mask `[1, num_img, 1]` (1 = repaint, 0 = keep), and the img2img
/// strength. Passed to [`Ideogram4Pipeline::generate_edit_with_progress`] per seed.
pub struct EditInit {
    /// BN-normalized packed source latent `[1, num_img, 128]` (same space as the running `z`).
    pub z0: Array,
    /// Latent-grid inpaint mask `[1, num_img, 1]` (1.0 = repaint/white, 0.0 = keep/black). `None`
    /// for plain img2img (regenerate everywhere from the noised source).
    pub mask: Option<Array>,
    /// img2img strength in `(0, 1]` — fraction of the denoise executed from the noised source.
    pub strength: f32,
}

/// img2img start step (the flux2/fork `init_time_step`): `max(1, floor(num_steps·strength))` for a
/// positive strength clamped to `[0,1]`, else `0`. The denoise executes the lowest `num_run` steps
/// (the high-noise prefix is skipped) over the source noised to `schedule.eval(si[num_run])`.
fn init_time_step(num_steps: usize, strength: f32) -> usize {
    if strength > 0.0 {
        let s = strength.clamp(0.0, 1.0);
        // Python `int(num_steps * strength)` truncates toward zero == floor for s >= 0.
        ((num_steps as f32 * s) as usize).max(1)
    } else {
        0
    }
}

/// Flow-matching interpolation `z = σ·clean + (1−σ)·noise`. **Ideogram's [`LogitNormalSchedule`] is
/// inverted from the usual flow-match σ**: `eval(0) ≈ t_max ≈ 0.999` is the *clean* end and
/// `eval(1) ≈ t_min ≈ 0.0001` is *pure noise* (the denoise loop inits from `random::normal` and the
/// first executed step's `t_val = eval(si[num_run])` is the small-σ noisy end). So a larger σ weights
/// the clean source more, the mirror of the fork's `add_noise_by_interpolation`.
fn add_noise_by_interpolation(clean: &Array, noise: &Array, sigma: f32) -> Result<Array> {
    Ok(add(
        &multiply(clean, Array::from_f32(sigma))?,
        &multiply(noise, Array::from_f32(1.0 - sigma))?,
    )?)
}

/// Preprocess a source image onto the model's input grid: resize (Lanczos) to `width×height`,
/// normalize to `[-1,1]`, NHWC `[1,H,W,3]` f32. Mirrors flux2 `preprocess_ref_image`.
fn preprocess_source_image(image: &Image, width: u32, height: u32) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (width as usize, height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(Error::Msg(format!(
            "ideogram edit: source pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let resized: Vec<f32> = if (ih, iw) == (th, tw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)
    };
    let norm: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    Ok(Array::from_slice(&norm, &[1, th as i32, tw as i32, 3]))
}

/// Build the latent-grid inpaint mask `[1, num_img, 1]` (f32; 1.0 = repaint/white, 0.0 = keep/black)
/// from a mask image: PIL-"L" luma → binarize at image res → nearest `patch·ae = 16×` downsample
/// (top-left of each 16×16 block, torch `nearest`'s `floor(dst·scale)`), flattened row-major to
/// match the image-token order (`j = h·grid_w + w`). Ideogram's token grid is `H/16 × W/16`, so the
/// downsample factor is 16 (vs 8 in sdxl `preprocess_mask`).
fn preprocess_mask_packed(mask: &Image, width: u32, height: u32) -> Result<Array> {
    let (w, h) = (width as usize, height as usize);
    let patch = (PATCH * AE_SCALE) as usize; // 16
                                             // Nearest (not bicubic): a mask must not gain interpolated grays that flip the 0.5 binarize.
    let luma: Vec<u8> = if (mask.width as usize, mask.height as usize) == (w, h) {
        rgb_to_luma(&mask.pixels)
    } else {
        let resized = resize_nearest_u8(
            &mask.pixels,
            mask.height as usize,
            mask.width as usize,
            h,
            w,
        );
        let u8s: Vec<u8> = resized
            .iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect();
        rgb_to_luma(&u8s)
    };
    let (gh, gw) = (h / patch, w / patch);
    let mut packed = Vec::with_capacity(gh * gw);
    for ly in 0..gh {
        for lx in 0..gw {
            let v = luma[(ly * patch) * w + (lx * patch)]; // top-left of the block
            packed.push(if v as f32 / 255.0 >= 0.5 { 1.0f32 } else { 0.0 });
        }
    }
    Ok(Array::from_slice(&packed, &[1, (gh * gw) as i32, 1]))
}

/// PIL "L" grayscale luma: `round(R·299/1000 + G·587/1000 + B·114/1000)` per RGB pixel.
fn rgb_to_luma(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .map(|p| {
            let l = (p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114 + 500) / 1000;
            l.min(255) as u8
        })
        .collect()
}

pub struct Ideogram4Pipeline {
    cond: Ideogram4Transformer,
    /// The unconditional DiT (asymmetric-CFG negative branch). `None` in the **turbo** mode
    /// ([`load_turbo`](Self::load_turbo)) — the CFG-free single-DiT path runs the conditional DiT
    /// alone, halving resident memory.
    uncond: Option<Ideogram4Transformer>,
    te: Ideogram4TextEncoder,
    vae: Flux2Vae,
    tok: TextTokenizer,
    dit: Ideogram4DitConfig,
}

impl Ideogram4Pipeline {
    /// Load all components (2 DiTs + Qwen3-VL text encoder + VAE + tokenizer) from a converted
    /// snapshot dir — the quality (asymmetric-CFG) mode.
    pub fn load(root: &Path) -> Result<Self> {
        Ok(Self {
            cond: load_transformer(root)?,
            uncond: Some(load_unconditional_transformer(root)?),
            te: load_text_encoder(root)?,
            vae: load_vae(root)?,
            tok: load_tokenizer(root)?,
            dit: Ideogram4DitConfig::v4(),
        })
    }

    /// Load the **turbo** pipeline (issue #488): the conditional DiT + Qwen3-VL TE + VAE + tokenizer,
    /// **without** the unconditional DiT. The caller applies the TurboTime LoRA via
    /// [`apply_adapters`](Self::apply_adapters) and runs the CFG-free [`generate`](Self::generate)
    /// path (single forward per step, guidance off). Skips loading the ~half-of-weights uncond DiT.
    pub fn load_turbo(root: &Path) -> Result<Self> {
        Ok(Self {
            cond: load_transformer(root)?,
            uncond: None,
            te: load_text_encoder(root)?,
            vae: load_vae(root)?,
            tok: load_tokenizer(root)?,
            dit: Ideogram4DitConfig::v4(),
        })
    }

    /// `true` when this pipeline runs the CFG-free single-DiT turbo path (no unconditional DiT).
    pub fn is_turbo(&self) -> bool {
        self.uncond.is_none()
    }

    /// Install adapters (the TurboTime LoRA, or a user Ideogram LoRA) onto the **conditional** DiT
    /// via the shared strict loader — errors on any unmatched target. Apply **after**
    /// [`quantize`](Self::quantize) (the residual is computed and added on top of the possibly
    /// quantized base, fork-faithfully). No-op for an empty spec list.
    pub fn apply_adapters(&mut self, specs: &[AdapterSpec]) -> Result<()> {
        if !specs.is_empty() {
            apply_ideogram_adapters(&mut self.cond, specs)?;
        }
        Ok(())
    }

    /// Quantize the whole model in place (group-wise affine Q4/Q8) after the dense load — the
    /// conditional DiT, the unconditional DiT (quality mode only), the Qwen3-VL text encoder, and the
    /// VAE — matching the flux2 family's `spec.quantize` semantics. Norms / tiny embeddings stay
    /// dense (each module's `quantize` decides). Done once at load; runtime is unchanged.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.cond.quantize(bits)?;
        if let Some(uncond) = &mut self.uncond {
            uncond.quantize(bits)?;
        }
        self.te.quantize(bits)?;
        self.vae.quantize(bits)?;
        Ok(())
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

    /// VAE-encode a source image into the BN-normalized packed latent `[1, num_img, 128]` the
    /// denoise operates on — the exact inverse of [`decode`](Self::decode)'s de-normalize +
    /// unpatchify: resize → `encode_mean` → 2×2 patchify (Ideogram's `(ph,pw,c)` c-innermost order)
    /// → BN-normalize `(x − mean)/std`. Seed-independent; encode once per request.
    pub fn encode_init_latents(&self, image: &Image, height: u32, width: u32) -> Result<Array> {
        let patch = PATCH * AE_SCALE;
        let grid_h = (height / patch) as i32;
        let grid_w = (width / patch) as i32;
        let pre = preprocess_source_image(image, width, height)?; // [1, H, W, 3]
        let enc = self.vae.encode_mean(&pre)?; // [1, H/8, W/8, 32] = [1, gh·2, gw·2, 32]
                                               // Patchify to packed [1, L, 128] — inverse of decode's unpatchify (channels (ph, pw, c)).
        let packed = enc
            .reshape(&[1, grid_h, 2, grid_w, 2, 32])?
            .transpose_axes(&[0, 1, 3, 2, 4, 5])?
            .reshape(&[1, grid_h * grid_w, 128])?;
        // BN-normalize in packed NHWC — inverse of decode's `z·std + mean`.
        let (bn_std, bn_mean) = self.vae.bn_stats();
        let normed = divide(
            &subtract(&packed, &bn_mean.reshape(&[1, 1, 128])?)?,
            &bn_std.reshape(&[1, 1, 128])?,
        )?;
        Ok(normed)
    }

    /// Prepare the per-request [`EditInit`] (img2img / inpaint): VAE-encode the source once and
    /// build the optional latent-grid mask. The result is reused across the per-seed count loop.
    pub fn prepare_edit(
        &self,
        source: &Image,
        mask: Option<&Image>,
        strength: f32,
        height: u32,
        width: u32,
    ) -> Result<EditInit> {
        let z0 = self.encode_init_latents(source, height, width)?;
        let mask = match mask {
            Some(m) => Some(preprocess_mask_packed(m, width, height)?),
            None => None,
        };
        Ok(EditInit { z0, mask, strength })
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
    /// per-step `eval` makes the cancel check able to interrupt mid-render (MLX is lazy). Pure
    /// text-to-image (no edit conditioning) — byte-identical to the original render path.
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
        self.run_denoise(
            input_ids,
            height,
            width,
            num_steps,
            guidance,
            mu,
            seed,
            None,
            cancel,
            on_progress,
        )
    }

    /// [`generate_with_progress`](Self::generate_with_progress) for an **edit** (img2img / mask
    /// inpaint, sc-6303/6330): the denoise starts from the [`EditInit`]'s noised source latent at a
    /// strength-derived step and (if a mask is present) pins the keep region per step. Same
    /// asymmetric-CFG / turbo denoise loop as the text-to-image path — only the initial latent,
    /// start step, and optional per-step mask blend differ.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_edit_with_progress(
        &self,
        input_ids: &[i32],
        height: u32,
        width: u32,
        num_steps: usize,
        guidance: f32,
        mu: f64,
        seed: u64,
        edit: &EditInit,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        self.run_denoise(
            input_ids,
            height,
            width,
            num_steps,
            guidance,
            mu,
            seed,
            Some(edit),
            cancel,
            on_progress,
        )
    }

    /// The shared flow-matching denoise behind [`generate_with_progress`] (edit `None`) and
    /// [`generate_edit_with_progress`] (edit `Some`). With `edit == None` the body is byte-identical
    /// to the original text-to-image render (pure-noise init, full step range, no mask blend).
    #[allow(clippy::too_many_arguments)]
    fn run_denoise(
        &self,
        input_ids: &[i32],
        height: u32,
        width: u32,
        num_steps: usize,
        guidance: f32,
        mu: f64,
        seed: u64,
        edit: Option<&EditInit>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let patch = PATCH * AE_SCALE;
        // Request-derived: reject as a typed error rather than abort the worker (F-020/L-A).
        if !height.is_multiple_of(patch) || !width.is_multiple_of(patch) {
            return Err(Error::Msg(format!(
                "ideogram: height/width must be multiples of {patch} (got {height}x{width})"
            )));
        }
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
        // The unconditional (negative) branch runs over the image-only slice. Built only in quality
        // mode — turbo (`uncond=None`) skips it entirely, avoiding both the second DiT forward and
        // the large `neg_llm` zero tensor (num_img × 53248 f32, ~0.9 GB at 1024²).
        let neg = self.uncond.as_ref().map(|uncond| {
            (
                uncond,
                Array::from_slice(&pack.neg_position_ids, &[1, num_img, 3]),
                Array::from_slice(&pack.neg_segment_ids, &[1, num_img]),
                Array::from_slice(&pack.neg_indicator, &[1, num_img]),
                zeros(&[1, num_img, llm_dim]),
            )
        });

        // ── Flow-matching schedule (mu/std from the V4 preset for this step count) ──
        // (mu, std) come from the reference V4 preset for this step count (NOT constants); the
        // passed-in `mu` is superseded by the preset.
        let _ = mu;
        let (mu_eff, std_eff) = preset_mu_std(num_steps);
        let schedule = LogitNormalSchedule::for_resolution(height, width, mu_eff, std_eff);
        let si = make_step_intervals(num_steps);

        // Edit (img2img / inpaint): run only `num_run = floor(steps·strength)` of the reversed loop,
        // which skips the noisiest (smallest-σ) leading steps and starts from the source noised to
        // σ = schedule.eval(si[num_run]) (Ideogram's schedule is inverted — larger σ = cleaner, so a
        // larger `num_run`/strength → a smaller start σ → more change). T2I runs the full range from
        // pure noise. `init_time_step` floors a positive strength to ≥1 step.
        let num_run = match edit {
            Some(e) => init_time_step(num_steps, e.strength),
            None => num_steps,
        };

        // ── Init: always draw the noise (identical RNG stream); blend with the source for an edit ──
        let key = random::key(seed)?;
        let noise = random::normal::<f32>(&[1, num_img, ch], None, None, Some(&key))?;
        let mut z = match edit {
            Some(e) => {
                add_noise_by_interpolation(&e.z0, &noise, schedule.eval(si[num_run]) as f32)?
            }
            None => noise.clone(),
        };
        let text_z_padding = zeros(&[1, num_text, ch]);
        let img_range = Array::from_slice(&(num_text..seq).collect::<Vec<i32>>(), &[num_img]);

        // ── Flow-matching Euler denoise (high → low noise) ──
        for i in (0..num_run).rev() {
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

            let v = match &neg {
                Some((uncond, neg_position_ids, neg_segment_ids, neg_indicator, neg_llm)) => {
                    let neg_v = uncond.forward(
                        neg_llm,
                        &z,
                        &t,
                        neg_position_ids,
                        neg_segment_ids,
                        neg_indicator,
                    )?;
                    // Per-step asymmetric CFG: `v = gw·pos_v + (1−gw)·neg_v`. The reference uses a
                    // per-step guidance schedule (DEFAULT_GUIDANCE_SCHEDULE = 7.0 for the main steps,
                    // dropping to 3.0 for the final 3 "polish" steps) — a CONSTANT high guidance
                    // over-cooks the low-noise detail steps and "splatters" the image. The loop runs
                    // i = num_steps-1 → 0, so the final 3 polish steps are i ∈ {0,1,2}. Base guidance
                    // (`guidance`) drives the main steps.
                    let gw_i = if i < POLISH_STEPS {
                        POLISH_GUIDANCE
                    } else {
                        guidance
                    };
                    add(
                        &multiply(&pos_v, Array::from_f32(gw_i))?,
                        &multiply(&neg_v, Array::from_f32(1.0 - gw_i))?,
                    )?
                }
                // Turbo (issue #488): CFG-free single DiT. The TurboTime LoRA distilled the guided
                // velocity into the conditional DiT, so the velocity is `pos_v` directly — no
                // negative branch, `guidance` unused.
                None => pos_v,
            };
            z = add(&z, &multiply(&v, Array::from_f32((s_val - t_val) as f32))?)?;
            // Inpaint: pin the keep region (mask 0) to the source re-noised to this step's σ
            // (= s_val, the post-step time) and regenerate the white region (mask 1). At the final
            // step s≈0 the keep region is the clean source. Mirrors sdxl `InpaintBlend`; draws no
            // RNG, so an all-white mask reduces to plain img2img.
            if let Some(e) = edit {
                if let Some(mask) = &e.mask {
                    let init_noised = add_noise_by_interpolation(&e.z0, &noise, s_val as f32)?;
                    z = add(
                        &multiply(mask, &z)?,
                        &multiply(&subtract(Array::from_f32(1.0), mask)?, &init_noised)?,
                    )?;
                }
            }
            eval([&z])?;
            on_progress(Progress::Step {
                current: (num_run - i) as u32,
                total: num_run as u32,
            });
        }

        on_progress(Progress::Decoding);
        self.decode(&z, grid_h, grid_w)
    }

    /// De-normalize → unpatchify → VAE decode → RGB u8 `[H, W, 3]`.
    fn decode(&self, z: &Array, grid_h: i32, grid_w: i32) -> Result<Array> {
        // De-normalize the packed latent with the VAE's BatchNorm stats — `z * bn_std + bn_mean`,
        // exactly the reference (`pipeline_ideogram4`), NOT a separate latent_norm. The earlier
        // hardcoded LATENT_SCALE/LATENT_SHIFT did not match the bn stats and distorted the decode.
        let (bn_std, bn_mean) = self.vae.bn_stats();
        let denorm = add(
            &multiply(z, &bn_std.reshape(&[1, 1, 128])?)?,
            &bn_mean.reshape(&[1, 1, 128])?,
        )?; // [1, L, 128]

        // Unpatchify to NHWC: [1,gh,gw,2,2,32] → [1,gh,2,gw,2,32] → [1, gh·2, gw·2, 32]. The 128
        // packed channels are ordered (ph, pw, c) — c innermost — for this DiT (verified: the
        // FLUX-family (c, ph, pw) split produces a 2px grid).
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
            // Reference `_prepare_ids`: image positions are `[t, h, w] + IMAGE_POSITION_OFFSET` on
            // ALL THREE axes (`t_idx` is 0, so t = OFFSET). The offset keeps image positions disjoint
            // from the text positions (0..num_text) — leaving t=0 collides with the text t-axis and
            // corrupts the text→image MRoPE cross-attention (first 24 dims).
            let p = [
                IMAGE_POSITION_OFFSET,
                h + IMAGE_POSITION_OFFSET,
                w + IMAGE_POSITION_OFFSET,
            ];
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
