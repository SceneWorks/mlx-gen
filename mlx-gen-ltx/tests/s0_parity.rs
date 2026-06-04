//! S0 parity vs the `mlx-video-with-audio` reference (sc-2679 S0).
//!
//! Compares the Rust foundation — position grid, SPLIT double-precision RoPE cos/sin, the
//! `apply_split_rotary_emb` rotation, and the distilled sigma schedules — against golden tensors
//! dumped from the *actual* reference functions by `tools/dump_ltx_s0_golden.py`.
//!
//! The golden is small, synthetic, and weight-free (committed under `tests/fixtures/`), so this
//! runs in the default `cargo test` — no model weights needed. Regenerate with:
//!   `MLX_VIDEO_SRC=… ~/Repos/mflux/.venv/bin/python tools/dump_ltx_s0_golden.py`

use mlx_rs::Array;

use mlx_gen::weights::Weights;

use mlx_gen_ltx::positions::create_position_grid;
use mlx_gen_ltx::rope::{apply_split_rotary_emb, precompute_split_freqs_cis};
use mlx_gen_ltx::schedule::{DEFAULT_STAGE_1_SIGMAS, DEFAULT_STAGE_2_SIGMAS};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_s0_golden.safetensors"
);

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn meta_usize(g: &Weights, key: &str) -> usize {
    g.metadata(key)
        .unwrap_or_else(|| panic!("golden missing metadata `{key}`"))
        .parse()
        .unwrap()
}

#[test]
fn position_grid_matches_reference() {
    let g = Weights::from_file(GOLDEN).expect("load golden");
    let (b, f, h, w) = (
        meta_usize(&g, "batch"),
        meta_usize(&g, "latent_frames"),
        meta_usize(&g, "latent_h"),
        meta_usize(&g, "latent_w"),
    );
    let grid = create_position_grid(b, f, h, w);
    let want = g.require("positions").unwrap();
    let d = max_abs_diff(&grid, want);
    eprintln!("position grid max abs diff {d:.3e}");
    // Pure f32 arithmetic (integer scale + one f32 ÷fps) → expect bit-exact.
    assert!(d < 1e-6, "position grid max abs diff {d:.3e}");
}

#[test]
fn split_rope_cos_sin_match_reference() {
    let g = Weights::from_file(GOLDEN).expect("load golden");
    let positions = g.require("positions").unwrap();
    let dim = meta_usize(&g, "dim") as i32;
    let heads = meta_usize(&g, "heads") as i32;
    let theta: f64 = g.metadata("theta").unwrap().parse().unwrap();
    // max_pos from metadata "20,2048,2048".
    let mp: Vec<i32> = g
        .metadata("max_pos")
        .unwrap()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let max_pos = [mp[0], mp[1], mp[2]];

    let (cos, sin) =
        precompute_split_freqs_cis(positions, dim, theta, &max_pos, heads).expect("rope");

    let dc = max_abs_diff(&cos, g.require("rope_cos").unwrap());
    let ds = max_abs_diff(&sin, g.require("rope_sin").unwrap());
    eprintln!("rope cos max abs diff {dc:.3e}  sin {ds:.3e}");
    // f64 grid built in Rust vs numpy, both down-cast to f32 → ~1 ULP f32 differences only.
    assert!(dc < 1e-5, "rope cos max abs diff {dc:.3e}");
    assert!(ds < 1e-5, "rope sin max abs diff {ds:.3e}");
}

#[test]
fn apply_split_rotary_matches_reference() {
    let g = Weights::from_file(GOLDEN).expect("load golden");
    let apply_in = g.require("apply_in").unwrap();
    let cos = g.require("rope_cos").unwrap();
    let sin = g.require("rope_sin").unwrap();
    let out = apply_split_rotary_emb(apply_in, cos, sin).expect("apply");
    let d = max_abs_diff(&out, g.require("apply_out").unwrap());
    eprintln!("apply_split_rotary max abs diff {d:.3e}");
    assert!(d < 1e-5, "apply_split_rotary max abs diff {d:.3e}");
}

#[test]
fn sigma_schedules_match_reference() {
    let g = Weights::from_file(GOLDEN).expect("load golden");
    let s1 = g
        .require("stage1_sigmas")
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let s2 = g
        .require("stage2_sigmas")
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert_eq!(s1, DEFAULT_STAGE_1_SIGMAS.to_vec());
    assert_eq!(s2, DEFAULT_STAGE_2_SIGMAS.to_vec());
}
