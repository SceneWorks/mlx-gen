//! S2 AudioVideo-DiT (velocity) parity vs the reference joint `LTXModel(video, audio)` (sc-2684).
//!
//! `#[ignore]`d: needs the real `ltx_2_3_base_q8` `transformer.safetensors` (~20 GB). The committed
//! goldens (`tests/fixtures/ltx_av_dit_golden{,_bf16}.safetensors`, from
//! `tools/dump_ltx_av_dit_golden.py`) hold the reference video + audio velocities over synthetic
//! joint inputs; this test loads the SAME Q8 weights into the Rust `AvDiT` and checks BOTH velocities
//! reproduce — bit-exact, like the video-only gate (the distilled sampler is chaos-sensitive, so the
//! cross-modal dual-stream forward must be as tight as the video-only one).
//!
//! **The goldens MUST be mlx 0.31.2** (the Rust build): `quantized_matmul` changed 0.31.0→0.31.2.
//!
//! Run: `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test av_dit_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::{LtxConfig, SplitModel};
use mlx_gen_ltx::transformer::{AvDiT, Precision};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_av_dit_golden.safetensors"
);
const GOLDEN_BF16: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_av_dit_golden_bf16.safetensors"
);

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home)
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

fn run(bf16: bool, golden: &str) {
    let dir = base_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("embedded_config.json");
    // Quant geometry (bits/group) rides on `split_model.json` (sc-2686).
    let split = SplitModel::from_model_dir(&dir).expect("split_model.json");
    let prec = if bf16 {
        Precision::quant_bf16(split.bits, split.group)
    } else {
        Precision::quant_f32(split.bits, split.group)
    };
    let w =
        Weights::from_file(dir.join("transformer.safetensors")).expect("transformer.safetensors");
    let dit = AvDiT::from_weights(&w, &cfg, prec).expect("build AvDiT");
    let g = Weights::from_file(golden).expect("golden (run tools/dump_ltx_av_dit_golden.py)");

    let (v_vel, a_vel) = dit
        .forward(
            g.require("video_latent").unwrap(),
            g.require("video_timestep").unwrap(),
            g.require("video_context").unwrap(),
            None,
            g.require("video_positions").unwrap(),
            g.require("audio_latent").unwrap(),
            g.require("audio_timestep").unwrap(),
            g.require("audio_context").unwrap(),
            None,
            g.require("audio_positions").unwrap(),
        )
        .expect("av dit forward");

    let want_v = g.require("video_velocity").unwrap();
    let want_a = g.require("audio_velocity").unwrap();
    assert_eq!(v_vel.shape(), want_v.shape(), "video velocity shape");
    assert_eq!(a_vel.shape(), want_a.shape(), "audio velocity shape");
    let (pvr, mvr) = (peak_rel(&v_vel, want_v), mean_rel(&v_vel, want_v));
    let (par, mar) = (peak_rel(&a_vel, want_a), mean_rel(&a_vel, want_a));
    eprintln!("av dit ({prec:?}): video peak_rel {pvr:.3e} mean_rel {mvr:.3e} | audio peak_rel {par:.3e} mean_rel {mar:.3e}");
    // Bit-exact, matching the video-only sc-2842 gate.
    assert!(
        pvr == 0.0 && mvr == 0.0,
        "video velocity not bit-exact (peak {pvr:.3e} mean {mvr:.3e})"
    );
    assert!(
        par == 0.0 && mar == 0.0,
        "audio velocity not bit-exact (peak {par:.3e} mean {mar:.3e})"
    );
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer.safetensors (~20 GB)"]
fn av_dit_velocity_matches_reference() {
    run(false, GOLDEN);
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer.safetensors (~20 GB)"]
fn av_dit_velocity_matches_reference_bf16() {
    run(true, GOLDEN_BF16);
}
