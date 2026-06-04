//! S1 full text-encoder parity vs the reference (sc-2679 S1) — end-to-end.
//!
//! `#[ignore]`d: needs the `gemma-3-12b-it-bf16` shards (~24 GB) + a converted BASE
//! `ltx_2_3_base_q8/connector.safetensors` + the golden from `tools/dump_ltx_te_golden.py`
//! (gitignored). Runs the Rust `LtxTextEncoder` in bf16 (Gemma → feature extractor → connector)
//! and checks `video_features` + `video_embeddings` reproduce the reference.
//!
//! Run: `cargo test -p mlx-gen-ltx --test te_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::LtxConfig;
use mlx_gen_ltx::gemma::GemmaConfig;
use mlx_gen_ltx::text_encoder::LtxTextEncoder;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/ltx_te_golden.safetensors"
);

fn gemma_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_GEMMA_DIR") {
        return d.into();
    }
    let base = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--mlx-community--gemma-3-12b-it-bf16/snapshots");
    std::fs::read_dir(&base)
        .expect("gemma snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max()
        .expect("a gemma snapshot")
}

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max(abs(want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

#[test]
#[ignore = "needs gemma-3-12b-it-bf16 (~24 GB) + ltx_2_3_base_q8 + tools/golden/ltx_te_golden.safetensors"]
fn full_text_encoder_matches_reference() {
    let base = base_dir();
    let cfg = LtxConfig::from_model_dir(&base).expect("embedded_config.json");
    let gemma_w = Weights::from_dir(gemma_dir()).expect("gemma shards");
    let conn_w =
        Weights::from_file(base.join("connector.safetensors")).expect("connector.safetensors");
    let te = LtxTextEncoder::from_weights(
        &gemma_w,
        &conn_w,
        GemmaConfig::gemma_3_12b(),
        &cfg,
        Dtype::Bfloat16,
    )
    .expect("build TE");

    let g = Weights::from_file(GOLDEN).expect("te golden");
    let input_ids = g.require("input_ids").unwrap();
    let attention_mask = g.require("attention_mask").unwrap();

    let (feats, emb) = te
        .encode_with_features(input_ids, attention_mask)
        .expect("encode");

    let pr_feats = peak_rel(&feats, g.require("video_features").unwrap());
    let pr_emb = peak_rel(&emb, g.require("video_embeddings").unwrap());
    eprintln!("TE: video_features peak_rel {pr_feats:.3e}  video_embeddings peak_rel {pr_emb:.3e}");
    // `video_features` (after the per-token-RMS feature extractor) gates feat-extract + Gemma
    // tightly — the RMS-norm contracts the Gemma bf16 cross-build delta (~9e-3) rather than
    // amplifying it.
    assert!(
        pr_feats < 1.5e-2,
        "video_features peak_rel {pr_feats:.3e} too high"
    );
    // `video_embeddings` is looser: the 8-layer connector amplifies that bf16 cross-build delta
    // (pmetal mlx 0.31.1 vs the reference's wheel 0.31.0) ~5×. NOT a bug — the connector is proven
    // correct (bit-exact in f32 via connector_parity; f32-vs-bf16-except-SDPA agree to ~3e-3 on
    // identical input). The real end-to-end gate is the S6 video px>8.
    assert!(
        pr_emb < 6e-2,
        "video_embeddings peak_rel {pr_emb:.3e} too high"
    );
}

#[test]
#[ignore = "needs gemma-3-12b-it-bf16 (~24 GB) + ltx_2_3_base_q8 + tools/golden/ltx_te_golden.safetensors"]
fn full_text_encoder_av_matches_reference() {
    let base = base_dir();
    let cfg = LtxConfig::from_model_dir(&base).expect("embedded_config.json");
    let gemma_w = Weights::from_dir(gemma_dir()).expect("gemma shards");
    let conn_w =
        Weights::from_file(base.join("connector.safetensors")).expect("connector.safetensors");
    let te = LtxTextEncoder::from_weights_av(
        &gemma_w,
        &conn_w,
        GemmaConfig::gemma_3_12b(),
        &cfg,
        Dtype::Bfloat16,
    )
    .expect("build AV TE");

    let g = Weights::from_file(GOLDEN).expect("te golden");
    let input_ids = g.require("input_ids").unwrap();
    let attention_mask = g.require("attention_mask").unwrap();

    let (vf, af, ve, ae) = te
        .encode_av_with_features(input_ids, attention_mask)
        .expect("encode_av");

    // Video half identical to the video-only gate (shared normed_hidden).
    let pr_vf = peak_rel(&vf, g.require("video_features").unwrap());
    let pr_ve = peak_rel(&ve, g.require("video_embeddings").unwrap());
    // Audio half: same per-token-RMS feature path → tight on features, looser through the connector.
    let pr_af = peak_rel(&af, g.require("audio_features").unwrap());
    let pr_ae = peak_rel(&ae, g.require("audio_embeddings").unwrap());
    eprintln!(
        "TE/AV: video_features {pr_vf:.3e} video_emb {pr_ve:.3e} | audio_features {pr_af:.3e} audio_emb {pr_ae:.3e}"
    );
    assert!(pr_vf < 1.5e-2, "video_features peak_rel {pr_vf:.3e}");
    assert!(pr_ve < 6e-2, "video_embeddings peak_rel {pr_ve:.3e}");
    assert!(pr_af < 1.5e-2, "audio_features peak_rel {pr_af:.3e}");
    assert!(pr_ae < 6e-2, "audio_embeddings peak_rel {pr_ae:.3e}");
}
