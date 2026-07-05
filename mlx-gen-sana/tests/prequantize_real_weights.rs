//! Build a packed SANA **Q4/Q8** tier from a dense `Sana_*_mlx` snapshot (sc-8489, epic 8506).
//! `#[ignore]`d — needs the ~9 GB dense snapshot; produces a packed `transformer/` + `text_encoder/`
//! (+ dense `vae/`) tier that [`mlx_gen_sana::model::load`] packed-detects. The **bf16 tier** is the
//! dense source itself (mirror it — no convert). Verify a built tier renders by pointing the existing
//! `pipeline_contract::real_weight_1024_e2e` at it (its `from_weights` packed-detects the same dir):
//!
//!   SANA_PIPELINE_WEIGHTS=$SANA_OUT cargo test -p mlx-gen-sana --release \
//!     --test pipeline_contract -- --ignored --nocapture real_weight_1024_e2e
//!
//! Build:
//!   SANA_SRC=~/.cache/huggingface/hub/models--SceneWorks--Sana_1600M_1024px_mlx/snapshots/<hash> \
//!   SANA_OUT=~/sana-staging/1600m/q4 SANA_BITS=4 \
//!   cargo test -p mlx-gen-sana --release --test prequantize_real_weights \
//!     -- --ignored --nocapture build_tier

use mlx_gen_sana::convert::prequantize_turnkey;
use std::path::PathBuf;

#[test]
#[ignore = "builds a packed SANA tier from a dense snapshot; set SANA_SRC/SANA_OUT/SANA_BITS"]
fn build_tier() {
    let src = PathBuf::from(std::env::var("SANA_SRC").expect("set SANA_SRC"));
    let out = PathBuf::from(std::env::var("SANA_OUT").expect("set SANA_OUT"));
    let bits: i32 = std::env::var("SANA_BITS")
        .expect("set SANA_BITS")
        .parse()
        .expect("SANA_BITS must be an integer");
    assert!(
        bits == 4 || bits == 8,
        "SANA_BITS must be 4 or 8 (the bf16 tier is the dense source mirrored, no convert)"
    );
    assert!(
        src.join("transformer").is_dir(),
        "SANA_SRC must be a dense snapshot root (transformer/ text_encoder/ vae/): {}",
        src.display()
    );
    prequantize_turnkey(&src, &out, bits).expect("prequantize_turnkey");
    eprintln!("BUILT sana q{bits} tier at {}", out.display());
}
