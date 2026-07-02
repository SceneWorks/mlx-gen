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

use mlx_gen::{
    run_flow_sampler, CancelFlag, FlowMatchEuler, Image, Progress, Result, TimestepConvention,
};
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use mlx_gen_sdxl::tokenizer::ClipBpeTokenizer;
use mlx_gen_z_image::vae::Vae;

use crate::loader::{CLIP_MAX_LENGTH, CLIP_PAD_ID};
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

/// Tokenize one prompt for CLIP into a fixed `[1, 77]` int32 id row, padded with the EOS/pad token
/// (diffusers `padding="max_length", max_length=77`). The **empty** prompt is NOT special-cased:
/// `ClipBpeTokenizer::tokenize("")` returns `[BOS, EOS]` (BOS is always prepended, EOS always
/// appended), which after padding is exactly diffusers `tokenizer("", padding="max_length")`. This
/// is load-bearing for the true-CFG uncond branch of every default (unset-negative) render — an
/// earlier `is_empty() → Vec::new()` shortcut produced 77×EOS with NO BOS, changing every hidden
/// state and shifting the pooled-at-argmax EOS selection from index 1 to 0 (F-004; same bug family
/// as z-image sc-8958).
fn clip_ids(tokenizer: &ClipBpeTokenizer, prompt: &str) -> Result<Array> {
    let mut ids = tokenizer.tokenize(prompt)?;
    if ids.len() > CLIP_MAX_LENGTH {
        ids.truncate(CLIP_MAX_LENGTH);
    }
    ids.resize(CLIP_MAX_LENGTH, CLIP_PAD_ID);
    Ok(Array::from_slice(&ids, &[1, CLIP_MAX_LENGTH as i32]))
}

/// Encode one prompt into SD3.5 conditioning (`pooled [1,2048]`, `context [1,333,4096]`) via the
/// triple-TE aggregator. CLIP ids are padded to 77; T5 ids to 256 (the gen-core T5 tokenizer's
/// `pad_to_max_length`). T5 runs unmasked (diffusers default).
pub fn encode_prompt(
    encoders: &Sd3TextEncoders,
    clip_tokenizer: &ClipBpeTokenizer,
    t5_tokenizer: &mlx_gen::tokenizer::TextTokenizer,
    prompt: &str,
) -> Result<Sd3Conditioning> {
    // clip_l and clip_g share ONE BPE tokenizer, so their padded id rows are identical — tokenize
    // once and reuse for both encoders (F-094b).
    let clip_row = clip_ids(clip_tokenizer, prompt)?;
    let t5 = t5_tokenizer.tokenize(prompt)?;
    let (t5_ids, _t5_mask) = mlx_gen::tokenizer::to_arrays(&t5);
    encoders.encode(&clip_row, &clip_row, &t5_ids, None)
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
        &scheduler.sigmas,
        latents,
        seed,
        cancel,
        on_progress,
        predict,
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

    /// Build a tiny synthetic CLIP BPE tokenizer (no real weights) whose special tokens match the
    /// real CLIP vocab ids so [`CLIP_PAD_ID`] (= EOS = 49407) behaves identically. Enough vocab to
    /// tokenize a short ASCII prompt; the empty prompt needs only BOS/EOS. Written to a unique temp
    /// dir and loaded through the real [`ClipBpeTokenizer::from_dir`] path so this exercises
    /// production code (matching the crate's `std::env::temp_dir()` test convention — no new dep).
    fn synthetic_clip_tokenizer() -> ClipBpeTokenizer {
        let dir = std::env::temp_dir().join(format!("mlx_gen_sd3_clip_tok_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Vocab: the two specials at their real CLIP ids, plus a few `</w>`-terminated word tokens so
        // a non-empty prompt also tokenizes without an OOV error.
        let vocab = serde_json::json!({
            "<|startoftext|>": 49406,
            "<|endoftext|>": 49407,
            "a</w>": 320,
            "fox</w>": 3363,
        });
        std::fs::write(dir.join("vocab.json"), vocab.to_string()).unwrap();
        // merges.txt: a header line + no merges (single-token words need none).
        std::fs::write(dir.join("merges.txt"), "#version: 0.2\n").unwrap();
        ClipBpeTokenizer::from_dir(&dir).unwrap()
    }

    #[test]
    fn empty_prompt_clip_ids_keep_bos_and_match_tokenize_path() {
        // F-004 (default-run, no real weights): the empty (uncond) prompt must NOT be special-cased.
        // `clip_ids("")` must equal the padded `tokenize("")` path and begin with BOS (49406) then
        // EOS (49407) — NOT 77×EOS-with-no-BOS as the removed `is_empty() → Vec::new()` shortcut did.
        let tok = synthetic_clip_tokenizer();

        // tokenize("") is [BOS, EOS].
        assert_eq!(tok.tokenize("").unwrap(), vec![49406, 49407]);

        let ids = clip_ids(&tok, "").unwrap();
        eval([&ids]).unwrap();
        let row = ids.as_slice::<i32>();
        assert_eq!(row.len(), CLIP_MAX_LENGTH);
        // First slot is BOS, second is EOS, remainder padded with EOS/pad.
        assert_eq!(row[0], 49406, "empty-prompt uncond row must START with BOS");
        assert_eq!(row[1], CLIP_PAD_ID, "second id is EOS");
        assert!(
            row[2..].iter().all(|&x| x == CLIP_PAD_ID),
            "the tail is EOS/pad"
        );
        // The buggy path would have produced row[0] == EOS (no BOS) — assert we are NOT that.
        assert_ne!(
            row[0], CLIP_PAD_ID,
            "regression: empty-prompt row must not be 77×EOS (missing BOS)"
        );

        // Equivalence with the general tokenize(...) → pad path applied by hand.
        let mut expected = tok.tokenize("").unwrap();
        expected.resize(CLIP_MAX_LENGTH, CLIP_PAD_ID);
        assert_eq!(
            row,
            expected.as_slice(),
            "clip_ids(\"\") == padded tokenize(\"\")"
        );
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
