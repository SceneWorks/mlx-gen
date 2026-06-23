//! sc-7568 — real-weight full-DiT parity for the Krea 2 single-stream DiT against the reference
//! `mmdit.py` `SingleStreamDiT` loaded with the published `krea/Krea-2-Turbo` weights.
//!
//! `#[ignore]` — needs the real snapshot + the golden (`tools/dump_krea_dit_real_golden.py`):
//! ```sh
//! KREA_TURBO_DIR=~/.cache/huggingface/hub/models--krea--Krea-2-Turbo/snapshots/<rev> \
//!   cargo test -p mlx-gen-krea --release --test dit_real_weights -- --ignored --nocapture
//! ```
//! Cross-backend bf16 (the real weights are bf16; MLX runs reduced-precision matmul on Metal), so the
//! bar is a high cosine rather than bit-exactness.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_krea::{Krea2Config, Krea2Transformer};
use mlx_rs::ops::{multiply, sqrt, sum};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/krea_dit_real.safetensors"
);

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("KREA_TURBO_DIR").expect("set KREA_TURBO_DIR to the snapshot root"))
}

fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let dot = sum(multiply(&a, &b).unwrap(), false).unwrap();
    let na = sqrt(sum(multiply(&a, &a).unwrap(), false).unwrap()).unwrap();
    let nb = sqrt(sum(multiply(&b, &b).unwrap(), false).unwrap()).unwrap();
    (dot / (na * nb)).item::<f32>()
}

#[test]
#[ignore = "needs real weights (KREA_TURBO_DIR) + golden (tools/dump_krea_dit_real_golden.py)"]
fn dit_matches_real_reference() {
    let g = Weights::from_file(GOLDEN)
        .expect("golden — run tools/dump_krea_dit_real_golden.py with KREA_TURBO_DIR set");
    let root = snapshot();
    let cfg = Krea2Config::from_snapshot(&root).unwrap();
    assert_eq!(
        cfg,
        Krea2Config::turbo(),
        "real config should be the published Turbo"
    );

    let w = Weights::from_dir(root.join("transformer")).expect("load real transformer/");
    let dit = Krea2Transformer::from_weights(&w, &cfg).expect("build DiT from real weights");

    let velocity = dit
        .forward(
            g.require("in.latent").unwrap(),
            g.require("in.timestep").unwrap(),
            g.require("in.context").unwrap(),
            None,
        )
        .unwrap();
    let want = g.require("out.velocity").unwrap();
    assert_eq!(velocity.shape(), want.shape(), "velocity shape");

    let c = cosine(&velocity, want);
    println!("Krea 2 real-weight DiT parity cosine = {c:.7}");
    assert!(c > 0.98, "real-weight DiT cosine {c:.7} <= 0.98");
}
