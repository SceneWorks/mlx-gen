//! S1 connector parity vs the reference `Embeddings1DConnector` (sc-2679 S1).
//!
//! `#[ignore]`d: needs the real eros `connector.safetensors` (~6.3 GB). The committed golden
//! (`tests/fixtures/ltx_connector_golden.safetensors`, from `tools/dump_ltx_connector_golden.py`)
//! holds the reference f32 input/mask/output; this test loads the SAME connector weights and
//! checks the Rust `Connector` reproduces the video embeddings.
//!
//! Run: `LTX_EROS_DIR=… cargo test -p mlx-gen-ltx --test connector_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::LtxConfig;
use mlx_gen_ltx::connector::Connector;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_connector_golden.safetensors"
);

fn eros_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_EROS_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_eros")
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max(abs(want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

#[test]
#[ignore = "needs eros connector.safetensors (~6.3 GB)"]
fn connector_matches_reference() {
    let dir = eros_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("embedded_config.json");
    let w = Weights::from_file(dir.join("connector.safetensors")).expect("connector.safetensors");
    let conn = Connector::from_weights(
        &w,
        "video_embeddings_connector.",
        &cfg,
        mlx_rs::Dtype::Float32,
    )
    .expect("build");

    let g = Weights::from_file(GOLDEN).expect("golden");
    let features = g.require("features").unwrap();
    let mask01 = g.require("mask01").unwrap();
    let want = g.require("video_embeddings").unwrap();

    let got = conn.forward(features, mask01).expect("forward");
    assert_eq!(got.shape(), want.shape());
    let pr = peak_rel(&got, want);
    eprintln!("connector peak_rel = {pr:.3e}");
    // f32 Rust vs f32 reference (both f64 rope → f32, f32 sdpa) → tight.
    assert!(pr < 5e-3, "connector peak_rel {pr:.3e} too high");
}

#[test]
#[ignore = "needs eros connector.safetensors (~6.3 GB)"]
fn audio_connector_matches_reference() {
    // sc-2684: the audio connector is the same architecture at audio dims (32 × 64 = 2048).
    let dir = eros_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("embedded_config.json");
    let w = Weights::from_file(dir.join("connector.safetensors")).expect("connector.safetensors");
    let conn = Connector::from_weights_dims(
        &w,
        "audio_embeddings_connector.",
        cfg.connector_num_layers,
        cfg.audio_connector_num_attention_heads,
        cfg.audio_connector_attention_head_dim,
        cfg.positional_embedding_theta,
        cfg.connector_positional_embedding_max_pos,
        mlx_rs::Dtype::Float32,
    )
    .expect("build audio connector");

    let g = Weights::from_file(GOLDEN).expect("golden");
    let features = g.require("audio_features").unwrap();
    let mask01 = g.require("mask01").unwrap();
    let want = g.require("audio_embeddings").unwrap();

    let got = conn.forward(features, mask01).expect("forward");
    assert_eq!(got.shape(), want.shape());
    let pr = peak_rel(&got, want);
    eprintln!("audio connector peak_rel = {pr:.3e}");
    assert!(pr < 5e-3, "audio connector peak_rel {pr:.3e} too high");
}
