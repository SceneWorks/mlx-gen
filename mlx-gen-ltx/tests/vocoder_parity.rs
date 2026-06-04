//! S4 vocoder parity vs the reference `VocoderWithBWE` (sc-2684).
//!
//! `#[ignore]`d: needs the real `ltx_2_3_base_q8` `vocoder.safetensors` (~246 MB). The committed
//! golden (`tests/fixtures/ltx_vocoder_golden.safetensors`, from `tools/dump_ltx_vocoder_golden.py`)
//! holds the reference **f32** 48 kHz waveform for a synthetic mel; the Rust `LtxVocoder` loads the
//! SAME weights (config-selected = VocoderWithBWE) and must reproduce it.
//!
//! Run: `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test vocoder_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::VocoderConfig;
use mlx_gen_ltx::vocoder::{LtxVocoder, VocoderWithBwe};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_vocoder_golden.safetensors"
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

#[test]
#[ignore = "needs ltx_2_3_base_q8 vocoder.safetensors (~246 MB)"]
fn vocoder_waveform_matches_reference() {
    let dir = base_dir();
    let cfg = VocoderConfig::from_model_dir(&dir).expect("embedded_config.json");
    assert!(cfg.bwe.is_some(), "shipped vocoder is VocoderWithBWE");
    assert!(cfg.core.is_bigvgan(), "shipped core is BigVGAN (snakebeta)");
    let w = Weights::from_file(dir.join("vocoder.safetensors")).expect("vocoder.safetensors");
    let voc = LtxVocoder::from_weights(&w, &cfg).expect("build LtxVocoder");

    let g = Weights::from_file(GOLDEN).expect("golden (run tools/dump_ltx_vocoder_golden.py)");
    let wav = voc
        .forward(g.require("mel").unwrap())
        .expect("vocoder forward");
    let want = g.require("waveform").unwrap();
    assert_eq!(wav.shape(), want.shape(), "waveform shape");
    let (pr, mr) = (peak_rel(&wav, want), mean_rel(&wav, want));
    eprintln!("vocoder waveform peak_rel = {pr:.3e} mean_rel = {mr:.3e}");
    // f32 Rust vs f32 reference (same ops/weights). The kaiser-sinc filters + STFT-basis matmuls are
    // deterministic; small f32 round-off only.
    assert!(pr < 1e-3, "vocoder waveform peak_rel {pr:.3e} too high");
    assert!(mr < 1e-3, "vocoder waveform mean_rel {mr:.3e} too high");
}

#[test]
#[ignore = "diagnostic: stage bisection of the VocoderWithBwe pipeline"]
fn vocoder_stage_bisection() {
    let dir = base_dir();
    let cfg = VocoderConfig::from_model_dir(&dir).expect("embedded_config.json");
    let w = Weights::from_file(dir.join("vocoder.safetensors")).expect("vocoder.safetensors");
    let voc = match LtxVocoder::from_weights(&w, &cfg).expect("build") {
        LtxVocoder::Bwe(v) => v,
        _ => panic!("expected VocoderWithBwe"),
    };
    let _ = std::any::type_name::<VocoderWithBwe>();
    let g = Weights::from_file(GOLDEN).expect("golden");
    let (low, mel_from_low, residual, skip) =
        voc.stages(g.require("mel").unwrap()).expect("stages");
    let (cp, cu, ca) = voc
        .core_forward_stages(g.require("mel").unwrap())
        .expect("core stages");
    // Isolation on golden inputs: ups[0] alone (ConvTranspose1d) and act_post alone (SnakeBeta).
    let up0 = voc
        .core_debug_up(0, g.require("core_after_conv_pre").unwrap())
        .expect("up0");
    let acton = voc
        .core_debug_act_post(g.require("core_after_up").unwrap())
        .expect("act_on_up");
    let rb0 = voc
        .core_debug_resblock(0, g.require("up0_only").unwrap())
        .expect("rb0");
    for (name, got) in [
        ("up0_only", &up0),
        ("act_on_up", &acton),
        ("rb0_on_up0", &rb0),
        ("core_after_conv_pre", &cp),
        ("core_after_up", &cu),
        ("core_after_act", &ca),
        ("low", &low),
        ("mel_from_low", &mel_from_low),
        ("residual", &residual),
        ("skip", &skip),
    ] {
        let want = g.require(name).unwrap();
        let (pr, mr) = (peak_rel(got, want), mean_rel(got, want));
        eprintln!(
            "  {name:20} shape {:?} peak_rel {pr:.3e} mean_rel {mr:.3e}",
            got.shape()
        );
    }
}
