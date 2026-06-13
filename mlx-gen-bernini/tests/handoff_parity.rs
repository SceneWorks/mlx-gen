//! sc-5140: the planner→renderer handoff matches the reference (near-bit, f32).
//!
//! Synthetic-fixture parity (`tools/dump_bernini_handoff_golden.py`): a tiny `MLPConnector` +
//! `mask_tokens` with random f32 weights, dumped from the reference `post_process_input_embeds` /
//! `feat_from_planner_to_renderer` + the 4-stream extraction. Validates the mask selection + the
//! `for_gen` integration end-to-end. Tolerance reflects the f32 `for_gen` floor (erf-GELU + 2 matmuls
//! + RMSNorm); the mask selection itself is exact (a wrong stream is O(1)).
//!
//! Run: `cargo test -p mlx-gen-bernini --test handoff_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_bernini::connector::MlpConnector;
use mlx_gen_bernini::mar::{four_streams, post_process_input_embeds};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/handoff_golden.safetensors"
);

fn bools(w: &Weights, key: &str) -> Vec<bool> {
    w.require(key)
        .unwrap()
        .as_slice::<i32>()
        .iter()
        .map(|&x| x != 0)
        .collect()
}

fn check(name: &str, got: &Array, want: &Array, tol: f32) {
    assert_eq!(got.shape(), want.shape(), "{name} shape");
    let n = want.shape().iter().product::<i32>();
    let g = got.reshape(&[n]).unwrap();
    let wv = want.reshape(&[n]).unwrap();
    let (g, wv) = (g.as_slice::<f32>(), wv.as_slice::<f32>());
    let peak = wv.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = g
        .iter()
        .zip(wv)
        .fold(0f32, |m, (&a, &b)| m.max((a - b).abs()));
    println!(
        "{name:>14}: peak|Δ|={max_diff:.3e} peak-rel={:.3e}",
        max_diff / peak
    );
    assert!(
        max_diff / peak < tol,
        "{name} peak-rel {} exceeds {tol:.1e}",
        max_diff / peak
    );
}

#[test]
fn handoff_matches_reference_f32() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let conn = MlpConnector::from_weights(&w, "model.connector").expect("connector");

    // mask_token = mask_tokens[:, :1] -> [1, 1, H].
    let mask_tokens = w.require("model.mask_tokens").unwrap();
    let h = mask_tokens.shape()[2];
    let mask_token = mask_tokens
        .take_axis(Array::from_slice(&[0i32], &[1]), 1)
        .unwrap()
        .reshape(&[1, 1, h])
        .unwrap();

    let cond_gen = bools(&w, "io.cond_gen_mask");
    let uncond_gen = bools(&w, "io.uncond_gen_mask");

    // --- post_process: gen slots set to mask_token ---
    let cond_input = w.require("io.cond_input").unwrap().clone();
    let pp = post_process_input_embeds(&cond_input, &cond_gen, &mask_token).unwrap();
    check(
        "post_process",
        &pp,
        w.require("out.post_processed").unwrap(),
        1e-5,
    );

    // --- 4 streams ---
    let cond_hidden = w.require("io.cond_hidden").unwrap().clone();
    let uncond_hidden = w.require("io.uncond_hidden").unwrap().clone();
    let s = four_streams(&cond_hidden, &cond_gen, &uncond_hidden, &uncond_gen, &conn).unwrap();
    check(
        "wtxt_wvit",
        &s.wtxt_wvit,
        w.require("out.wtxt_wvit").unwrap(),
        5e-3,
    );
    check(
        "wtxt_wovit",
        &s.wtxt_wovit,
        w.require("out.wtxt_wovit").unwrap(),
        5e-3,
    );
    check(
        "wotxt_wvit",
        &s.wotxt_wvit,
        w.require("out.wotxt_wvit").unwrap(),
        5e-3,
    );
    check(
        "wotxt_wovit",
        &s.wotxt_wovit,
        w.require("out.wotxt_wovit").unwrap(),
        5e-3,
    );
}
