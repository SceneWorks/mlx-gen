//! sc-5139: the connector + clip-diff head match the reference (near-bit, f32).
//!
//! Synthetic-fixture parity (the repo's weight-free golden pattern): tiny `MLPConnector` +
//! `SimpleMLPAdaLN` + `FlowMatchScheduler` with random weights, dumped from the reference by
//! `tools/dump_bernini_clip_diff_golden.py`. Exercises `for_gen`/`for_vit`, the net forward
//! (TimestepEmbedder, adaLN ResBlocks, FinalLayer), and a full triple-CFG `sample()` denoise — all
//! f32. Tolerances reflect the MLX-Metal-vs-torch f32 matmul floor accumulated over each path.
//!
//! Run: `cargo test -p mlx-gen-bernini --test clip_diff_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_bernini::clip_diff::DiffLossFm;
use mlx_gen_bernini::connector::MlpConnector;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/clip_diff_golden.safetensors"
);

fn errors(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    (max_diff, max_diff / peak)
}

fn check(name: &str, got: &Array, want: &Array, tol: f32) {
    let (abs, rel) = errors(got, want);
    println!("{name:>10}: peak|Δ|={abs:.3e}  peak-rel={rel:.3e}");
    assert!(rel < tol, "{name} peak-rel {rel:.3e} exceeds {tol:.1e}");
}

#[test]
fn clip_diff_matches_reference_f32() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let depth: usize = w.metadata("depth").unwrap().parse().unwrap();
    let hidden: i32 = w.metadata("hidden").unwrap().parse().unwrap();
    let shift: f32 = w.metadata("shift").unwrap().parse().unwrap();
    let steps: usize = w.metadata("steps").unwrap().parse().unwrap();
    let txt_cfg: f32 = w.metadata("txt_cfg").unwrap().parse().unwrap();
    let img_cfg: f32 = w.metadata("img_cfg").unwrap().parse().unwrap();

    // --- connector ---
    let conn = MlpConnector::from_weights(&w, "conn").expect("connector");
    let cx = w.require("io.conn_x").unwrap().clone();
    // ~1e-3 f32 cross-backend floor (erf-GELU + 2–3 matmuls); a wrong projection is O(0.1+).
    check(
        "for_gen",
        &conn.for_gen(&cx).unwrap(),
        w.require("out.for_gen").unwrap(),
        5e-3,
    );
    check(
        "for_vit",
        &conn.for_vit(&cx).unwrap(),
        w.require("out.for_vit").unwrap(),
        5e-3,
    );

    // --- clip-diff net forward ---
    let mut head = DiffLossFm::from_weights(&w, "net", depth, hidden, shift).expect("head");
    let nx = w.require("io.net_x").unwrap().clone();
    let nt = w.require("io.net_t").unwrap().clone();
    let nc = w.require("io.net_c").unwrap().clone();
    check(
        "net",
        &head.forward(&nx, &nt, &nc).unwrap(),
        w.require("out.net").unwrap(),
        5e-3,
    );

    // --- full triple-CFG sample() ---
    let z = w.require("io.z").unwrap().clone();
    let noise = w.require("io.noise_base").unwrap().clone();
    let sample = head
        .sample(&z, txt_cfg, steps, Some(img_cfg), &noise)
        .expect("sample");
    check("sample", &sample, w.require("out.sample").unwrap(), 5e-3);
}
