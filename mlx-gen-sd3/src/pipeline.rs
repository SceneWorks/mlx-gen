//! SD3.5 text-to-image sampling pipeline (E5, sc-7864): tokenization → triple-TE conditioning →
//! seeded latent noise → flow-match Euler denoise (with true-CFG) → VAE decode → RGB8.
//!
//! ## Sampler / shift / CFG
//!
//! * **Flow-match Euler with static shift 3.0.** SD3.5-Large's `scheduler/scheduler_config.json`
//!   pins `FlowMatchEulerDiscreteScheduler { shift: 3.0 }` with no dynamic shifting, so the schedule
//!   is [`FlowMatchEuler::for_static_shift(steps, 3.0)`] — identical to the Z-Image-Turbo path. An
//!   unset `req.scheduler` keeps that native schedule byte-exact; a curated name re-shapes σ over the
//!   same `mu = ln(3)` (epic 7114).
//! * **Timestep convention.** The MMDiT embeds the diffusers-scale timestep `sigma * 1000` (the
//!   scheduler's `num_train_timesteps`). The unified flow sampler hands the predict closure
//!   `ms.timestep(σ) = σ` (the `Sigma` convention); the closure scales it to `σ·1000` before the
//!   forward. The Euler update itself stays in σ-space (`x += (σ_{t+1}-σ_t)·v`).
//! * **True CFG.** SD3.5-Large is a true-CFG model: each step runs TWO forwards (cond + uncond) and
//!   combines `pred = uncond + scale·(cond − uncond)`. The uncond branch conditions on the
//!   (empty/negative) prompt's triple-TE embedding. `guidance_scale` defaults to 3.5.

use mlx_gen::img2img::{add_noise_by_interpolation, init_time_step, preprocess_init_image};
use mlx_gen::{
    run_flow_sampler, CancelFlag, FlowMatchEuler, Image, Progress, Result, TimestepConvention,
};
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use mlx_gen_sdxl::tokenizer::ClipBpeTokenizer;
use mlx_gen_z_image::vae::Vae;

use crate::loader::{Sd3ClipPad, CLIP_MAX_LENGTH};
use crate::text::{Sd3Conditioning, Sd3TextEncoders};
use crate::transformer::Sd3Transformer;

/// SD3.5 latent channel count.
pub const LATENT_CHANNELS: i32 = 16;
/// VAE spatial downsample (latent edge is image/8).
pub const SPATIAL_SCALE: u32 = 8;
/// diffusers `num_train_timesteps` — the MMDiT embeds `sigma * 1000`.
pub const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;
/// SD3.5-Large static flow-match shift (`scheduler_config.json` `shift = 3.0`, no dynamic shifting).
pub const SCHEDULE_SHIFT: f32 = 3.0;

/// Seeded txt2img latent noise — shape `[1, 16, height/8, width/8]`, f32. diffusers
/// `randn_tensor([B, 16, H/8, W/8])`; we draw f32 via `mx.random.normal` keyed on `seed`.
pub fn create_noise(seed: u64, width: u32, height: u32) -> Result<Array> {
    let key = random::key(seed)?;
    let shape = [
        1,
        LATENT_CHANNELS,
        (height / SPATIAL_SCALE) as i32,
        (width / SPATIAL_SCALE) as i32,
    ];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
}

/// Tokenize one prompt for CLIP into the raw (unpadded, capped-at-77) int32 id sequence. The
/// **empty** prompt is NOT special-cased: `ClipBpeTokenizer::tokenize("")` returns `[BOS, EOS]` (BOS
/// is always prepended, EOS always appended), which after padding is exactly diffusers
/// `tokenizer("", padding="max_length")`. This is load-bearing for the true-CFG uncond branch of
/// every default (unset-negative) render — an earlier `is_empty() → Vec::new()` shortcut produced
/// 77×EOS with NO BOS, changing every hidden state and shifting the pooled-at-argmax EOS selection
/// from index 1 to 0 (F-004; same bug family as z-image sc-8958).
fn clip_token_ids(tokenizer: &ClipBpeTokenizer, prompt: &str) -> Result<Vec<i32>> {
    let mut ids = tokenizer.tokenize(prompt)?;
    if ids.len() > CLIP_MAX_LENGTH {
        ids.truncate(CLIP_MAX_LENGTH);
    }
    Ok(ids)
}

/// Right-pad a raw CLIP id sequence to a fixed `[1, 77]` int32 row with `pad_id`
/// (diffusers `padding="max_length", max_length=77`). The pad token DIFFERS per encoder — CLIP-L
/// pads with eos (49407), OpenCLIP-bigG with `!` (0) — see [`Sd3ClipPad`] (sc-9581).
fn pad_clip_row(ids: &[i32], pad_id: i32) -> Array {
    let mut row = ids.to_vec();
    row.resize(CLIP_MAX_LENGTH, pad_id);
    Array::from_slice(&row, &[1, CLIP_MAX_LENGTH as i32])
}

/// Encode one prompt into SD3.5 conditioning (`pooled [1,2048]`, `context [1,333,4096]`) via the
/// triple-TE aggregator. CLIP ids are padded to 77; T5 ids to 256 (the gen-core T5 tokenizer's
/// `pad_to_max_length`). T5 runs unmasked (diffusers default).
///
/// CLIP-L and bigG share ONE BPE tokenizer (identical token sequence), but SD3.5 pads them with
/// DIFFERENT pad tokens: L with eos (49407), bigG with `!` (0). Tokenize once, then pad each row with
/// its encoder's pad id (`clip_pad`) — padding bigG with eos corrupts its penultimate hidden on every
/// pad slot and thus the joint context for any sub-77-token prompt (sc-9581, mirrors candle-gen-sd3).
pub fn encode_prompt(
    encoders: &Sd3TextEncoders,
    clip_tokenizer: &ClipBpeTokenizer,
    clip_pad: Sd3ClipPad,
    t5_tokenizer: &mlx_gen::tokenizer::TextTokenizer,
    prompt: &str,
) -> Result<Sd3Conditioning> {
    let clip_ids = clip_token_ids(clip_tokenizer, prompt)?;
    let clip_l_row = pad_clip_row(&clip_ids, clip_pad.pad_l);
    let clip_g_row = pad_clip_row(&clip_ids, clip_pad.pad_g);
    let t5 = t5_tokenizer.tokenize(prompt)?;
    let (t5_ids, _t5_mask) = mlx_gen::tokenizer::to_arrays(&t5);
    encoders.encode(&clip_l_row, &clip_g_row, &t5_ids, None)
}

/// The shared flow-match Euler denoise core over an explicit `sigmas` slice — the true-CFG predict
/// closure + the unified sampler. Runs the MMDiT once (cond) or twice (cond + uncond → `uncond +
/// scale·(cond − uncond)`) per step; the Euler step advances the latents in σ-space; the MMDiT
/// timestep is `σ·1000`. txt2img passes the full schedule ([`denoise_cfg`]); img2img passes the tail
/// `sigmas[start..]` from a noised init latent ([`denoise_img2img_cfg`]).
#[allow(clippy::too_many_arguments)]
fn denoise_over_sigmas(
    transformer: &Sd3Transformer,
    sigmas: &[f32],
    sampler_name: Option<&str>,
    seed: u64,
    latents: Array,
    cond: &Sd3Conditioning,
    uncond: Option<&Sd3Conditioning>,
    guidance_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let predict = |x: &Array, timestep: f32| -> Result<Array> {
        // The unified flow sampler hands `timestep = σ`; the MMDiT embeds `σ·1000`.
        let t = Array::from_slice(&[timestep * NUM_TRAIN_TIMESTEPS], &[1]);
        let pred_cond = transformer.forward(x, &cond.context, &cond.pooled, &t)?;
        match uncond {
            Some(uc) if guidance_scale != 1.0 => {
                let pred_uncond = transformer.forward(x, &uc.context, &uc.pooled, &t)?;
                // pred = uncond + scale·(cond − uncond).
                let delta = subtract(&pred_cond, &pred_uncond)?;
                Ok(add(
                    &pred_uncond,
                    &multiply(&delta, Array::from_slice(&[guidance_scale], &[1]))?,
                )?)
            }
            _ => Ok(pred_cond),
        }
    };
    run_flow_sampler(
        sampler_name,
        TimestepConvention::Sigma,
        sigmas,
        latents,
        seed,
        cancel,
        on_progress,
        predict,
    )
}

/// One flow-match Euler denoise with **true CFG** + progress + cooperative cancellation. Each step
/// runs the MMDiT twice (cond + uncond) and combines `uncond + scale·(cond − uncond)`; the Euler
/// step then advances the latents in σ-space. The MMDiT timestep is `σ·1000`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_cfg(
    transformer: &Sd3Transformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    seed: u64,
    latents: Array,
    cond: &Sd3Conditioning,
    uncond: Option<&Sd3Conditioning>,
    guidance_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    denoise_over_sigmas(
        transformer,
        &scheduler.sigmas,
        sampler_name,
        seed,
        latents,
        cond,
        uncond,
        guidance_scale,
        cancel,
        on_progress,
    )
}

/// **img2img latent-init** (epic 8588 slice A4, sc-10189) — reference-guided generation on SD3.5.
/// VAE-encode `init` into the same normalized 16-ch latent space as [`create_noise`] (SD3.5's VAE
/// `encode` returns `(mean − shift)·scale`, matching diffusers' `StableDiffusion3Img2ImgPipeline`),
/// blend `(1 − σ_k)·clean + σ_k·noise` at the start sigma `σ_k = sigmas[k]`, and run the true-CFG
/// flow-match Euler sampler over the tail `sigmas[k..]`. `strength` is reference fidelity in the fork's
/// [`init_time_step`] convention (`k = max(1, ⌊num_steps·strength⌋)`): higher strength → later start →
/// fewer denoise steps → the output stays closer to the reference; `strength ≤ 0` degenerates to a full
/// txt2img (`k = 0`, identical to [`denoise_cfg`]). Unlike the packed Qwen-Image / Z-Image path, the
/// SD3.5 MMDiT patchifies internally, so the clean latent is used **unpacked** `[1, 16, H/8, W/8]`
/// (matching `create_noise`) — no pre-pack. Shares the true-CFG predict core with [`denoise_cfg`], so
/// Large/Medium run two forwards/step and the distilled Large-Turbo runs one (`guidance == 1.0`).
#[allow(clippy::too_many_arguments)]
pub fn denoise_img2img_cfg(
    transformer: &Sd3Transformer,
    scheduler: &FlowMatchEuler,
    sampler_name: Option<&str>,
    seed: u64,
    vae: &Vae,
    init: &Image,
    strength: f32,
    width: u32,
    height: u32,
    steps: usize,
    cond: &Sd3Conditioning,
    uncond: Option<&Sd3Conditioning>,
    guidance_scale: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    // Reference → clean latent [1, 16, H/8, W/8]. `Vae::encode` returns the normalized `(mean−shift)·
    // scale` latent (the same space as `create_noise`); SD3.5's MMDiT patchifies internally, so keep it
    // unpacked.
    let image_nchw = preprocess_init_image(init, width, height)?;
    let clean = vae.encode(&image_nchw)?;
    let noise = create_noise(seed, width, height)?;

    // Start step from strength; blend clean⊕noise at σ_k, then denoise sigmas[k..]. The schedule has
    // `steps + 1` sigmas, so clamp the start index inside it (strength ≥ 1 → the last usable step).
    let start = init_time_step(steps, Some(strength)).min(scheduler.sigmas.len().saturating_sub(1));
    let x_start = add_noise_by_interpolation(&clean, &noise, scheduler.sigmas[start])?;
    denoise_over_sigmas(
        transformer,
        &scheduler.sigmas[start..],
        sampler_name,
        seed,
        x_start,
        cond,
        uncond,
        guidance_scale,
        cancel,
        on_progress,
    )
}

/// VAE-decode the final `[1,16,H/8,W/8]` latent → an RGB8 [`Image`]. The de-norm (`z/scale + shift`)
/// is applied inside [`Vae::decode`] (the reused Z-Image VAE with SD3.5's factors), so the raw latent
/// is handed straight through.
pub fn decode_to_image(vae: &Vae, latents: &Array) -> Result<Image> {
    let decoded = vae.decode(latents)?.as_dtype(Dtype::Float32)?;
    mlx_gen::image::decoded_to_image(&decoded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

    use crate::loader::{resolve_clip_pad_id, CLIP_EOS_ID};

    /// Build a tiny synthetic CLIP BPE tokenizer (no real weights) whose special tokens match the
    /// real CLIP vocab ids so [`CLIP_EOS_ID`] (= EOS = 49407) behaves identically. Enough vocab to
    /// tokenize a short ASCII prompt; the empty prompt needs only BOS/EOS. Also writes a `!` (0)
    /// entry + a `tokenizer_config.json` (`pad_token`) so [`resolve_clip_pad_id`] can be exercised.
    /// Written to a unique temp dir and loaded through the real [`ClipBpeTokenizer::from_dir`] path so
    /// this exercises production code (matching the crate's `std::env::temp_dir()` test convention).
    /// `pad_token` selects the config's pad string (`"!"` = bigG, `"<|endoftext|>"` = L).
    fn synthetic_clip_tokenizer_dir(tag: &str, pad_token: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mlx_gen_sd3_clip_tok_{}_{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Vocab: the two specials at their real CLIP ids, `!` at 0 (bigG's pad), plus a few
        // SINGLE-character `</w>` word tokens. The synthetic `merges.txt` has NO merges, so the
        // char-level BPE leaves each word as its per-character sub-tokens; only single-char words
        // (which become one `<char></w>` unigram) map to a vocab entry. A multi-char word like
        // `"fox"` would BPE to `["f", "o", "x</w>"]` — none in this vocab — and error. So the
        // non-empty test prompt below uses only single-char words (`"a b"`).
        let vocab = serde_json::json!({
            "!": 0,
            "<|startoftext|>": 49406,
            "<|endoftext|>": 49407,
            "a</w>": 320,
            "b</w>": 321,
        });
        std::fs::write(dir.join("vocab.json"), vocab.to_string()).unwrap();
        // merges.txt: a header line + no merges (single-token words need none).
        std::fs::write(dir.join("merges.txt"), "#version: 0.2\n").unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::json!({ "pad_token": pad_token }).to_string(),
        )
        .unwrap();
        dir
    }

    fn synthetic_clip_tokenizer() -> ClipBpeTokenizer {
        ClipBpeTokenizer::from_dir(synthetic_clip_tokenizer_dir("l", "<|endoftext|>")).unwrap()
    }

    #[test]
    fn empty_prompt_clip_ids_keep_bos_and_match_tokenize_path() {
        // F-004 (default-run, no real weights): the empty (uncond) prompt must NOT be special-cased.
        // The padded row must equal the padded `tokenize("")` path and begin with BOS (49406) then
        // EOS (49407) — NOT 77×EOS-with-no-BOS as the removed `is_empty() → Vec::new()` shortcut did.
        let tok = synthetic_clip_tokenizer();

        // tokenize("") is [BOS, EOS].
        assert_eq!(tok.tokenize("").unwrap(), vec![49406, 49407]);

        // The L path pads with eos (49407).
        let ids = pad_clip_row(&clip_token_ids(&tok, "").unwrap(), CLIP_EOS_ID);
        eval([&ids]).unwrap();
        let row = ids.as_slice::<i32>();
        assert_eq!(row.len(), CLIP_MAX_LENGTH);
        // First slot is BOS, second is EOS, remainder padded with EOS/pad.
        assert_eq!(row[0], 49406, "empty-prompt uncond row must START with BOS");
        assert_eq!(row[1], CLIP_EOS_ID, "second id is EOS");
        assert!(
            row[2..].iter().all(|&x| x == CLIP_EOS_ID),
            "the tail is EOS/pad"
        );
        // The buggy path would have produced row[0] == EOS (no BOS) — assert we are NOT that.
        assert_ne!(
            row[0], CLIP_EOS_ID,
            "regression: empty-prompt row must not be 77×EOS (missing BOS)"
        );

        // Equivalence with the general tokenize(...) → pad path applied by hand.
        let mut expected = tok.tokenize("").unwrap();
        expected.resize(CLIP_MAX_LENGTH, CLIP_EOS_ID);
        assert_eq!(
            row,
            expected.as_slice(),
            "padded tokenize(\"\") matches by-hand"
        );
    }

    #[test]
    fn resolve_clip_pad_reads_per_encoder_pad_token() {
        // sc-9581: L resolves `<|endoftext|>` (49407); bigG resolves `!` (0). A `tokenizer_config.json`
        // with no `pad_token` (or an unknown token) falls back to eos.
        let l_dir = synthetic_clip_tokenizer_dir("padl", "<|endoftext|>");
        let g_dir = synthetic_clip_tokenizer_dir("padg", "!");
        assert_eq!(resolve_clip_pad_id(&l_dir), 49407, "CLIP-L pad = eos");
        assert_eq!(resolve_clip_pad_id(&g_dir), 0, "CLIP-bigG pad = `!` = 0");

        // Fallback: a dir whose config lacks `pad_token` -> eos.
        let f_dir = std::env::temp_dir().join(format!(
            "mlx_gen_sd3_clip_tok_{}_nofallback",
            std::process::id()
        ));
        std::fs::create_dir_all(&f_dir).unwrap();
        std::fs::write(f_dir.join("tokenizer_config.json"), "{}").unwrap();
        assert_eq!(
            resolve_clip_pad_id(&f_dir),
            CLIP_EOS_ID,
            "missing pad_token -> eos fallback"
        );
    }

    #[test]
    fn bigg_pads_with_bang_not_eos() {
        // sc-9581 core regression: with a sub-77-token prompt, the bigG row must be padded with `!`
        // (0), NOT eos (49407). The pre-fix code shared one eos-padded row for both encoders.
        let tok = synthetic_clip_tokenizer();
        // Single-char words only (`"a b"` -> [BOS, 320, 321, EOS], len 4) so the no-merges synthetic
        // BPE tokenizes without an OOV error; still a sub-77 prompt with a real pad region.
        let ids = clip_token_ids(&tok, "a b").unwrap();
        assert_eq!(
            ids,
            vec![49406, 320, 321, 49407],
            "synthetic tokenize(\"a b\")"
        );
        let l_row = pad_clip_row(&ids, 49407);
        let g_row = pad_clip_row(&ids, 0);
        eval([&l_row, &g_row]).unwrap();
        let (l, g) = (l_row.as_slice::<i32>(), g_row.as_slice::<i32>());
        // Both share the leading content + BOS/EOS.
        assert_eq!(l[0], 49406);
        assert_eq!(g[0], 49406);
        // The pad region (after the real tokens) differs: L=eos, bigG=`!`(0).
        let pad_start = ids.len();
        assert!(
            pad_start < CLIP_MAX_LENGTH,
            "prompt must be shorter than 77"
        );
        assert!(
            l[pad_start..].iter().all(|&x| x == 49407),
            "L pads with eos"
        );
        assert!(g[pad_start..].iter().all(|&x| x == 0), "bigG pads with `!`");
        assert_ne!(l, g, "the two encoder rows must differ on the pad region");
    }

    #[test]
    fn noise_shape_is_batch1_16ch() {
        let n = create_noise(0, 1024, 1024).unwrap();
        assert_eq!(n.shape(), &[1, 16, 128, 128]);
        let n = create_noise(0, 512, 768).unwrap();
        assert_eq!(n.shape(), &[1, 16, 96, 64]);
    }

    #[test]
    fn noise_is_seed_deterministic() {
        let a = create_noise(42, 256, 256).unwrap();
        let b = create_noise(42, 256, 256).unwrap();
        let c = create_noise(43, 256, 256).unwrap();
        eval([&a, &b, &c]).unwrap();
        let av = a.as_slice::<f32>();
        let bv = b.as_slice::<f32>();
        let cv = c.as_slice::<f32>();
        assert_eq!(av, bv, "same seed must reproduce the same noise");
        assert_ne!(av, cv, "a different seed must differ");
    }

    #[test]
    fn static_shift_schedule_matches_diffusers() {
        // SD3.5-Large: FlowMatchEulerDiscreteScheduler shift=3.0, no dynamic shifting.
        let s = FlowMatchEuler::for_static_shift(4, SCHEDULE_SHIFT);
        let expected = [1.0_f32, 0.9, 0.75, 0.5, 0.0];
        assert_eq!(s.sigmas.len(), 5);
        for (got, want) in s.sigmas.iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "got {got} want {want}");
        }
    }
}
