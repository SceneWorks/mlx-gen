//! S3 audio VAE decoder parity vs the reference `AudioDecoder` (sc-2684).
//!
//! `#[ignore]`d: needs the real `ltx_2_3_base_q8` `audio_vae.safetensors` (~61 MB). The committed
//! golden (`tests/fixtures/ltx_audio_vae_golden.safetensors`, from `tools/dump_ltx_audio_vae_golden.py`)
//! holds the reference **f32** mel output for a synthetic latent (built with the config-correct
//! `mid_block_add_attention=False` — see the dump-script / `AudioVaeConfig` note on the reference's
//! random-attn bug). The Rust `AudioDecoder` loads the SAME weights and must reproduce the mel.
//!
//! Run: `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test audio_vae_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_ltx::audio_vae::AudioDecoder;
use mlx_gen_ltx::config::AudioVaeConfig;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_audio_vae_golden.safetensors"
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
#[ignore = "needs ltx_2_3_base_q8 audio_vae.safetensors (~61 MB)"]
fn audio_vae_decode_matches_reference() {
    let dir = base_dir();
    let cfg = AudioVaeConfig::from_model_dir(&dir).expect("embedded_config.json");
    assert!(
        !cfg.mid_block_add_attention,
        "shipped config disables mid attention"
    );
    let w = Weights::from_file(dir.join("audio_vae.safetensors")).expect("audio_vae.safetensors");
    let dec = AudioDecoder::from_weights(&w, &cfg).expect("build AudioDecoder");

    let g = Weights::from_file(GOLDEN).expect("golden (run tools/dump_ltx_audio_vae_golden.py)");
    let mel = dec.decode(g.require("latent").unwrap()).expect("decode");
    let want = g.require("mel").unwrap();
    assert_eq!(mel.shape(), want.shape(), "mel shape");
    let (pr, mr) = (peak_rel(&mel, want), mean_rel(&mel, want));
    eprintln!("audio vae mel peak_rel = {pr:.3e} mean_rel = {mr:.3e}");
    // f32 Rust vs f32 reference (same ops, same weights) → bit-tight.
    assert!(pr < 5e-4, "audio vae mel peak_rel {pr:.3e} too high");
    assert!(mr < 5e-4, "audio vae mel mean_rel {mr:.3e} too high");
}
