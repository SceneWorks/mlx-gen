//! S2 video-VAE parity vs the reference `LTX2VideoDecoder` + `VideoEncoder` (sc-2679 S2).
//!
//! `#[ignore]`d: needs the real `ltx_2_3_base_q8` `vae_decoder.safetensors` (~800 MB) +
//! `vae_encoder.safetensors` (~640 MB). The committed golden
//! (`tests/fixtures/ltx_vae_golden.safetensors`, from `tools/dump_ltx_vae_golden.py`) holds the
//! reference **f32** decode/encode I/O; this test loads the SAME bf16 weights, upcasts to f32, and
//! checks the Rust `LtxVideoVae` reproduces both. Honors "divergence is not rounding": the only
//! expected gap is f32 conv summation ordering (mlx conv3d is the shared op → near bit-exact).
//!
//! Run: `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test vae_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max as max_op, subtract};
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::LtxVaeConfig;
use mlx_gen_ltx::tiling::{TilingConfig, VaeTiling};
use mlx_gen_ltx::vae::LtxVideoVae;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_vae_golden.safetensors"
);

const TILING_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_vae_tiling_golden.safetensors"
);

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

/// `max|Δ| / max|ref|`.
fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max_op(abs(want).unwrap(), None).unwrap().item::<f32>();
    max_op(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

fn build() -> (LtxVideoVae, Weights) {
    let dir = base_dir();
    let cfg = LtxVaeConfig::from_model_dir(&dir).expect("embedded_config.json vae block");
    let dec = Weights::from_file(dir.join("vae_decoder.safetensors")).expect("vae_decoder");
    let enc = Weights::from_file(dir.join("vae_encoder.safetensors")).expect("vae_encoder");
    let vae = LtxVideoVae::from_weights(&dec, Some(&enc), &cfg).expect("build LtxVideoVae");
    let golden = Weights::from_file(GOLDEN).expect("golden (run tools/dump_ltx_vae_golden.py)");
    (vae, golden)
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 vae_decoder.safetensors (~800 MB)"]
fn decode_matches_reference() {
    let (vae, g) = build();
    let dec_in = g.require("dec_in").unwrap();
    let want = g.require("dec_out").unwrap();
    let got = vae.decode(dec_in).expect("decode");
    assert_eq!(got.shape(), want.shape(), "decode output shape");
    let pr = peak_rel(&got, want);
    eprintln!("decode peak_rel = {pr:.3e} shape={:?}", got.shape());
    assert!(pr < 5e-3, "decode peak_rel {pr:.3e} too high");
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 vae_encoder.safetensors (~640 MB)"]
fn encode_matches_reference() {
    let (vae, g) = build();
    let enc_in = g.require("enc_in").unwrap();
    let want = g.require("enc_out").unwrap();
    let got = vae.encode(enc_in).expect("encode");
    assert_eq!(got.shape(), want.shape(), "encode output shape");
    let pr = peak_rel(&got, want);
    eprintln!("encode peak_rel = {pr:.3e} shape={:?}", got.shape());
    assert!(pr < 5e-3, "encode peak_rel {pr:.3e} too high");
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 vae_decoder.safetensors (~800 MB)"]
fn decode_tiled_matches_reference() {
    let (vae, _) = build();
    let g = Weights::from_file(TILING_GOLDEN)
        .expect("tiling golden (run tools/dump_ltx_vae_tiling_golden.py)");

    // Spatial-tiled: 64px tile / 32px overlap → 3×3 tiles over the 4×4 latent.
    let sp_in = g.require("sp_in").unwrap();
    let sp_want = g.require("sp_out").unwrap();
    let sp_cfg = TilingConfig::spatial_only(64, 32);
    assert!(
        sp_cfg.needs_tiling(VaeTiling::LTX, 1, 4, 4),
        "spatial cfg should tile 4×4"
    );
    let sp_got = vae
        .decode_tiled(sp_in, &sp_cfg)
        .expect("decode_tiled spatial");
    assert_eq!(sp_got.shape(), sp_want.shape(), "spatial tiled shape");
    let sp_pr = peak_rel(&sp_got, sp_want);
    eprintln!(
        "decode_tiled spatial peak_rel = {sp_pr:.3e} shape={:?}",
        sp_got.shape()
    );
    assert!(sp_pr < 5e-3, "spatial tiled peak_rel {sp_pr:.3e} too high");

    // Temporal-tiled: 16f tile / 8f overlap → 2 causal tiles over the 3 latent frames.
    let tp_in = g.require("tp_in").unwrap();
    let tp_want = g.require("tp_out").unwrap();
    let tp_cfg = TilingConfig::temporal_only(16, 8);
    assert!(
        tp_cfg.needs_tiling(VaeTiling::LTX, 3, 2, 2),
        "temporal cfg should tile 3 frames"
    );
    let tp_got = vae
        .decode_tiled(tp_in, &tp_cfg)
        .expect("decode_tiled temporal");
    assert_eq!(tp_got.shape(), tp_want.shape(), "temporal tiled shape");
    let tp_pr = peak_rel(&tp_got, tp_want);
    eprintln!(
        "decode_tiled temporal peak_rel = {tp_pr:.3e} shape={:?}",
        tp_got.shape()
    );
    assert!(tp_pr < 5e-3, "temporal tiled peak_rel {tp_pr:.3e} too high");
}
