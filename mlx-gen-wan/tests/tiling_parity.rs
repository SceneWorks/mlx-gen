//! sc-2808 gate: the Wan z16 VAE **tiled** decode ([`WanVae::decode_tiled`], non-causal `T→4T`,
//! spatial ×8) must reproduce the `mlx_video` reference `WanVAE.decode_tiled` — the overlapping
//! spatial/temporal tiles, trapezoidally blended, must match the reference's tiled output bit-for-bit
//! (up to the conv float-ordering gap, like the S2 decode gate).
//!
//! Why match the *reference tiled* output and not a single-pass decode: tiling is **not** identical
//! to a one-shot decode (each tile's causal conv sees zero-pad at its boundary instead of neighbour
//! data — the residual lives at the seams, hidden by overlap+blend). On the tiny **random**-weight
//! fixture that residual is ~40% (no learned smoothness), so tiled-vs-untiled is only meaningful on a
//! real VAE (`wan_tiled_close_to_single_pass_real`, `#[ignore]`). The exact gate is tiled-vs-
//! reference-tiled, both carrying the same seam effects.
//!
//! Self-contained committed golden (`tools/dump_s2_tiling_fixtures.py`, the tiny `dim=4` z16 VAE +
//! the reference tiled IO). Runs on Metal in CI — no real weights. Shared geometry: `mlx_gen::tiling`.

use std::path::PathBuf;

use mlx_gen::tiling::{SpatialTiling, TemporalTiling, TilingConfig, VaeTiling};
use mlx_gen::weights::Weights;
use mlx_gen_wan::WanVae;
use mlx_rs::random;

/// The dump's tiling config (`dump_s2_tiling_fixtures.py`): spatial 64px/32, temporal 16f/8.
fn golden_cfg() -> TilingConfig {
    TilingConfig {
        spatial: Some(SpatialTiling {
            tile_px: 64,
            overlap_px: 32,
        }),
        temporal: Some(TemporalTiling {
            tile_frames: 16,
            overlap_frames: 8,
        }),
    }
}

fn diff(got: &[f32], exp: &[f32]) -> (f32, f64) {
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut sum_ref = 0f64;
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sum_ref += e.abs() as f64;
    }
    (max_abs, sum_abs / sum_ref.max(1e-9))
}

#[test]
fn wan_tiled_decode_matches_reference() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/s2_tiling.safetensors"
    );
    let w = Weights::from_file(path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (run tools/dump_s2_tiling_fixtures.py)"));
    let vae = WanVae::from_weights(&w).expect("build tiny WanVae");

    let tiled_in = w.require("tiled_in").expect("tiled_in");
    let exp = w.require("tiled_out").expect("tiled_out");

    let cfg = golden_cfg();
    let sh = tiled_in.shape();
    assert!(
        cfg.needs_tiling(VaeTiling::WAN, sh[2], sh[3], sh[4]),
        "golden latent must actually tile"
    );
    let got = vae.decode_tiled(tiled_in, &cfg).expect("tiled decode");
    assert_eq!(got.shape(), exp.shape(), "tiled decode shape");

    let (max_abs, mean_rel) = diff(got.as_slice::<f32>(), exp.as_slice::<f32>());
    println!(
        "[tiled vs reference] shape={:?} max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}",
        got.shape()
    );
    // Same conv float-ordering envelope as the S2 single-pass decode gate (mean_rel < 1e-3).
    assert!(
        mean_rel < 1e-3,
        "tiled decode diverged from reference: mean_rel={mean_rel:.3e} max|Δ|={max_abs:.3e}"
    );
}

#[test]
fn wan_tiled_fallback_is_single_pass() {
    // When the config doesn't fire for the dims, decode_tiled must equal a single-pass decode.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/s2_vae.safetensors"
    );
    let w = Weights::from_file(path).expect("read s2_vae (run tools/dump_s2_fixtures.py)");
    let vae = WanVae::from_weights(&w).expect("build tiny WanVae");
    let dec_in = w.require("dec_in").expect("dec_in"); // [1,16,2,4,4] — below any real tile

    let untiled = vae.decode(dec_in).expect("single-pass");
    // Huge tiles → needs_tiling is false → fallback to the single pass.
    let big = TilingConfig::spatial_only(4096, 64);
    let got = vae.decode_tiled(dec_in, &big).expect("fallback");
    let (max_abs, mean_rel) = diff(got.as_slice::<f32>(), untiled.as_slice::<f32>());
    println!("[fallback] max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}");
    assert!(
        mean_rel < 1e-6,
        "no-tiling fallback must equal single-pass decode"
    );
}

/// Real-weight equivalence: on the **converted A14B z16 VAE** (a smooth learned decoder), tiled
/// decode must match a single-pass decode within blend tolerance — the sc-2808 "tiled-vs-untiled"
/// check, only meaningful on real (non-random) weights. `#[ignore]` (needs the converted VAE).
///
/// ```text
/// WAN_A14B_MODEL_DIR=~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16 \
///   cargo test -p mlx-gen-wan --test tiling_parity -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs the converted Wan2.2-T2V-A14B vae.safetensors (WAN_A14B_MODEL_DIR)"]
fn wan_tiled_close_to_single_pass_real() {
    let dir = match std::env::var_os("WAN_A14B_MODEL_DIR") {
        Some(s) => {
            let s = s.to_string_lossy();
            let s = s.strip_prefix("~/").map_or(s.to_string(), |rest| {
                format!("{}/{rest}", std::env::var("HOME").unwrap())
            });
            PathBuf::from(s)
        }
        None => {
            eprintln!("skip: set WAN_A14B_MODEL_DIR");
            return;
        }
    };
    let w = Weights::from_file(dir.join("vae.safetensors")).expect("real vae.safetensors");
    let vae = WanVae::from_weights(&w).expect("build real WanVae");

    // Latent → 192×192 / 12-frame output: spatial 64px tile (8 latent) fires on h=w=24>8.
    let key = random::key(3).unwrap();
    let z = random::normal::<f32>(&[1, 16, 3, 24, 24], None, None, Some(&key)).unwrap();
    let untiled = vae.decode(&z).expect("single-pass");
    let got = vae.decode_tiled(&z, &golden_cfg()).expect("tiled");
    assert_eq!(got.shape(), untiled.shape());
    let (max_abs, mean_rel) = diff(got.as_slice::<f32>(), untiled.as_slice::<f32>());
    println!("[real tiled vs single-pass] max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}");
    // Smooth learned decoder: seams blend to within a few %.
    assert!(
        mean_rel < 5e-2,
        "real tiled decode diverged from single-pass: mean_rel={mean_rel:.3e}"
    );
}
