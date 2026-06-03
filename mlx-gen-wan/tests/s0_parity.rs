//! S0 parity gate: the deterministic foundations (shifted-sigma schedule + integer timesteps,
//! 3-axis RoPE cos/sin, 3-D patchify/unpatchify reordering) must match the `mlx_video` Wan
//! reference exactly. Fixtures are dumped by `tools/dump_s0_fixtures.py` (committed under
//! `tests/fixtures/s0.json`). Honors the "divergence is not rounding" discipline — gate against the
//! real reference, not a re-derivation.

use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::patchify::{patchify, unpatchify};
use mlx_gen_wan::rope::RopeTable;
use mlx_gen_wan::scheduler::{compute_sigmas, SolverKind};

use mlx_rs::Array;
use serde_json::Value;

fn fixtures() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/s0.json");
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (run tools/dump_s0_fixtures.py)"));
    serde_json::from_str(&text).expect("parse s0.json")
}

fn as_f32_vec(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn sigmas_and_timesteps_match_reference() {
    let fx = fixtures();
    let cases = fx["sigmas"].as_object().unwrap();
    for (name, c) in cases {
        let num_steps = c["num_steps"].as_u64().unwrap() as usize;
        let shift = c["shift"].as_f64().unwrap() as f32;
        let ref_sigmas = as_f32_vec(&c["sigmas"]);
        let ref_ts = as_f32_vec(&c["timesteps"]);

        let sigmas = compute_sigmas(num_steps, shift, 1000);
        let nt = 1000.0_f32;
        let ts: Vec<f32> = sigmas[..sigmas.len() - 1]
            .iter()
            .map(|&s| (s * nt).trunc())
            .collect();

        let sd = max_abs_diff(&sigmas, &ref_sigmas);
        assert!(sd < 1e-6, "[{name}] sigma max|Δ| = {sd}");
        // Integer timesteps must match the reference exactly (model is trained on integer t).
        assert_eq!(ts, ref_ts, "[{name}] integer timesteps differ");
    }
}

#[test]
fn rope_cos_sin_matches_reference() {
    let fx = fixtures();
    let r = &fx["rope"];
    let head_dim = r["head_dim"].as_u64().unwrap() as usize;
    let grid = r["grid"].as_array().unwrap();
    let grid = (
        grid[0].as_u64().unwrap() as usize,
        grid[1].as_u64().unwrap() as usize,
        grid[2].as_u64().unwrap() as usize,
    );
    let ref_cos = as_f32_vec(&r["cos"]);
    let ref_sin = as_f32_vec(&r["sin"]);

    let table = RopeTable::new(head_dim);
    let (cos, sin) = table.precompute_cos_sin(grid).unwrap();
    let cos = cos.as_slice::<f32>().to_vec();
    let sin = sin.as_slice::<f32>().to_vec();

    let cd = max_abs_diff(&cos, &ref_cos);
    let sd = max_abs_diff(&sin, &ref_sin);
    // cos/sin built in f64 then cast f32 in both impls → expect ≤ a couple f32 ULPs.
    assert!(cd < 1e-6, "rope cos max|Δ| = {cd}");
    assert!(sd < 1e-6, "rope sin max|Δ| = {sd}");
}

#[test]
fn patchify_matches_reference() {
    let fx = fixtures();
    let p = &fx["patchify"];
    let in_shape: Vec<i32> = p["in_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect();
    let n: i32 = in_shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let x = Array::from_slice(&data, &in_shape);

    let (tokens, grid) = patchify(&x, (1, 2, 2)).unwrap();
    let ref_tokens = as_f32_vec(&p["tokens"]);
    let ref_grid = p["grid"].as_array().unwrap();
    assert_eq!(grid.0 as u64, ref_grid[0].as_u64().unwrap());
    assert_eq!(grid.1 as u64, ref_grid[1].as_u64().unwrap());
    assert_eq!(grid.2 as u64, ref_grid[2].as_u64().unwrap());
    assert_eq!(
        tokens.as_slice::<f32>(),
        ref_tokens.as_slice(),
        "patchify tokens differ"
    );
}

#[test]
fn unpatchify_matches_reference() {
    let fx = fixtures();
    let u = &fx["unpatchify"];
    let ts: Vec<i32> = u["tokens_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect();
    let n: i32 = ts.iter().product();
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let tokens = Array::from_slice(&data, &ts);

    let out_dim = u["out_dim"].as_u64().unwrap() as usize;
    let grid = u["grid"].as_array().unwrap();
    let grid = (
        grid[0].as_u64().unwrap() as usize,
        grid[1].as_u64().unwrap() as usize,
        grid[2].as_u64().unwrap() as usize,
    );
    let video = unpatchify(&tokens, grid, out_dim, (1, 2, 2)).unwrap();
    let ref_video = as_f32_vec(&u["video"]);
    assert_eq!(
        video.as_slice::<f32>(),
        ref_video.as_slice(),
        "unpatchify video differs"
    );
}

#[test]
fn ti2v_5b_config_dims() {
    // Sanity: the preset the crate targets matches the reference 5B dimensions.
    let c = WanModelConfig::wan22_ti2v_5b();
    assert_eq!(
        (c.dim, c.num_layers, c.num_heads, c.head_dim()),
        (3072, 30, 24, 128)
    );
    assert_eq!(SolverKind::from_name("unipc"), SolverKind::UniPC);
    assert_eq!(SolverKind::from_name("nonsense"), SolverKind::UniPC); // default fallback
}
