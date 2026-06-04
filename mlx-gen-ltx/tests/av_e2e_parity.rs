//! S5 AudioVideo e2e pipeline parity vs the reference joint denoise + decode (sc-2684).
//!
//! `#[ignore]`d: needs the real `ltx_2_3_base_q8` weights (~20 GB transformer + upsampler + VAEs +
//! vocoder). The committed golden (`tests/fixtures/ltx_av_e2e_golden.safetensors`, from
//! `tools/dump_ltx_av_e2e_golden.py`) holds the reference video latents/frames + audio latents/waveform
//! for injected synthetic conditioning. This composes the (individually bit-exact) stages end-to-end:
//! the Rust `generate_av_latents` (joint 2-stage denoise) + `decode_to_frames` + `decode_audio_track`
//! must reproduce the frames (px>8) + waveform.
//!
//! Run: `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test av_e2e_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, gt, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_ltx::audio_vae::AudioDecoder;
use mlx_gen_ltx::config::{AudioVaeConfig, LtxConfig, LtxVaeConfig, SplitModel, VocoderConfig};
use mlx_gen_ltx::pipeline::{decode_audio_track, decode_to_frames, generate_av_latents};
use mlx_gen_ltx::positions::{create_audio_position_grid, create_position_grid};
use mlx_gen_ltx::transformer::{AvDiT, Precision};
use mlx_gen_ltx::upsampler::LatentUpsampler;
use mlx_gen_ltx::vae::LtxVideoVae;
use mlx_gen_ltx::vocoder::LtxVocoder;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_av_e2e_golden.safetensors"
);

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

fn f32(x: &Array) -> Array {
    x.as_dtype(Dtype::Float32).unwrap()
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32(got), want).unwrap()).unwrap();
    let denom = max_op(abs(want).unwrap(), None).unwrap().item::<f32>();
    max_op(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

fn mean_rel(got: &Array, want: &Array) -> f32 {
    let num = sum(abs(subtract(f32(got), want).unwrap()).unwrap(), None).unwrap();
    let den = sum(abs(want).unwrap(), None).unwrap();
    num.item::<f32>() / den.item::<f32>().max(1e-12)
}

/// Fraction of uint8 pixels differing by > 8 (the LTX pixel-parity metric).
fn px_gt8(got: &Array, want: &Array) -> f32 {
    let a = got.as_dtype(Dtype::Float32).unwrap();
    let b = want.as_dtype(Dtype::Float32).unwrap();
    let diff = abs(subtract(&a, &b).unwrap()).unwrap();
    let over = gt(&diff, Array::from_int(8))
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    sum(&over, None).unwrap().item::<f32>() / (a.size() as f32) * 100.0
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 weights (~20 GB) + tools/golden"]
fn av_e2e_matches_reference() {
    let dir = base_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("config");
    let split = SplitModel::from_model_dir(&dir).expect("split_model.json");
    let tw = Weights::from_file(dir.join("transformer.safetensors")).expect("transformer");
    let dit =
        AvDiT::from_weights(&tw, &cfg, Precision::quant_f32(split.bits, split.group)).expect("dit");
    let upsampler = LatentUpsampler::from_weights(
        &Weights::from_file(dir.join("upsampler.safetensors")).expect("upsampler"),
    )
    .expect("upsampler");
    let vae_w = Weights::from_file(dir.join("vae_decoder.safetensors")).expect("vae");
    let vae = LtxVideoVae::from_weights(&vae_w, None, &LtxVaeConfig::from_model_dir(&dir).unwrap())
        .expect("vae");
    let mean = vae_w.require("per_channel_statistics.mean").unwrap();
    let std = vae_w.require("per_channel_statistics.std").unwrap();
    let audio_decoder = AudioDecoder::from_weights(
        &Weights::from_file(dir.join("audio_vae.safetensors")).expect("audio_vae"),
        &AudioVaeConfig::from_model_dir(&dir).unwrap(),
    )
    .expect("audio decoder");
    let vcfg = VocoderConfig::from_model_dir(&dir).unwrap();
    let vocoder = LtxVocoder::from_weights(
        &Weights::from_file(dir.join("vocoder.safetensors")).expect("vocoder"),
        &vcfg,
    )
    .expect("vocoder");

    let g = Weights::from_file(GOLDEN).expect("golden");
    // 256×256, 9 frames → video stage dims, audio 9 frames (positions match the dump exactly).
    let pos1 = create_position_grid(1, 2, 4, 4);
    let pos2 = create_position_grid(1, 2, 8, 8);
    let apos = create_audio_position_grid(1, 9);

    let mut steps = 0;
    let (vlat, alat) = generate_av_latents(
        &dit,
        &upsampler,
        g.require("video_s1").unwrap(),
        &pos1,
        g.require("video_s2").unwrap(),
        &pos2,
        g.require("audio_s1").unwrap(),
        g.require("audio_s2").unwrap(),
        &apos,
        g.require("video_ctx").unwrap(),
        g.require("audio_ctx").unwrap(),
        mean,
        std,
        None, // T2V+A (no I2V conditioning in this gate)
        &mut |_| steps += 1,
    )
    .expect("generate_av_latents");

    let pr_v = peak_rel(&vlat, g.require("video_latents").unwrap());
    let pr_a = peak_rel(&alat, g.require("audio_latents").unwrap());
    eprintln!("av e2e latents: video peak_rel {pr_v:.3e} | audio peak_rel {pr_a:.3e}");

    let frames = decode_to_frames(&vae, &vlat).expect("decode video");
    let px = px_gt8(&frames, g.require("frames").unwrap());
    let track = decode_audio_track(
        &audio_decoder,
        &vocoder,
        &alat,
        vcfg.final_sample_rate() as u32,
    )
    .expect("decode audio");
    // Interleave the golden waveform (1, C, S) → (S, C) → flat, to match the AudioTrack layout.
    let wav = g.require("waveform").unwrap();
    let wsh = wav.shape();
    let want_inter = wav
        .reshape(&[wsh[1], wsh[2]])
        .unwrap()
        .transpose_axes(&[1, 0])
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    let got_wav = Array::from_slice(&track.samples, &[track.samples.len() as i32]);
    let wav_pr = peak_rel(&got_wav, &want_inter);
    let wav_mr = mean_rel(&got_wav, &want_inter);
    eprintln!(
        "av e2e: video px>8 {px:.4}% | audio waveform peak_rel {wav_pr:.3e} mean_rel {wav_mr:.3e} (sr {})",
        track.sample_rate
    );

    // Video: pixel-parity (the latents are bit-exact, the f32 VAE matches). Audio: the latents are
    // bit-exact (0.0) and the vocoder is bit-exact on identical input, but the audio-VAE decode sits
    // at its f32 round-off floor (~5e-7, see audio_vae_parity) and the vocoder's deep nonlinear BWE
    // stack (log-mel STFT + SnakeBeta) amplifies that to ~6e-3 in the waveform — irreducible f32
    // accumulation, not a math gap (every stage is individually verified exact-or-f32-floor).
    assert!(px < 0.5, "video px>8 {px:.4}% too high");
    assert!(
        wav_pr < 1.5e-2,
        "audio waveform peak_rel {wav_pr:.3e} too high"
    );
    assert!(
        wav_mr < 5e-3,
        "audio waveform mean_rel {wav_mr:.3e} too high"
    );
    assert_eq!(track.channels, 2);
    assert_eq!(track.sample_rate, 48000);
}
