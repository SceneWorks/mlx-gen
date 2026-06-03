//! sc-2346 S0: parity for the FLUX.2-klein scaffold math vs the fork.
//! Fixture `tests/fixtures/s0_golden.safetensors` ← `tools/dump_flux2_s0_golden.py`.
//!
//! All pure math (no weights): the flow-match schedule, 2×2 pack/unpack/patchify, the 4-axis
//! RoPE table, and the integer id builders. Trig + arithmetic only, so tolerances are tight;
//! the integer id builders must match exactly.

use mlx_gen::weights::Weights;
use mlx_gen_flux2::{
    pack_latents, patchify_latents, prepare_grid_ids, prepare_text_ids, schedule, timesteps_x1000,
    unpack_latents, Flux2PosEmbed,
};
use mlx_rs::ops::all_close;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/s0_golden.safetensors"
);

fn close(a: &Array, b: &Array, rtol: f64, atol: f64) -> bool {
    all_close(a, b, rtol, atol, false).unwrap().item::<bool>()
}

/// Exact integer-array equality (cast to f32 for `all_close` with zero tolerance — exact for the
/// small ids these builders emit).
fn int_eq(a: &Array, b: &Array) -> bool {
    use mlx_rs::Dtype;
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    all_close(&a, &b, 0.0, 0.0, false).unwrap().item::<bool>()
}

#[test]
fn schedule_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    // (width, height, steps) — must match SCHED_CONFIGS in the dump script.
    for (width, height, steps) in [
        (256, 256, 4),
        (1024, 1024, 4),
        (1024, 560, 4),
        (512, 512, 20),
    ] {
        let tag = format!("{width}x{height}x{steps}");
        let s = schedule(steps as usize, width, height);

        let got_sigmas = Array::from_slice(&s.sigmas, &[s.sigmas.len() as i32]);
        let want_sigmas = w.require(&format!("sched.{tag}.sigmas")).unwrap();
        assert_eq!(
            got_sigmas.shape(),
            want_sigmas.shape(),
            "sigmas shape {tag}"
        );
        assert!(
            close(&got_sigmas, want_sigmas, 1e-4, 1e-4),
            "sigmas diverged for {tag}"
        );

        let ts = timesteps_x1000(&s);
        let got_ts = Array::from_slice(&ts, &[ts.len() as i32]);
        let want_ts = w.require(&format!("sched.{tag}.timesteps")).unwrap();
        assert_eq!(got_ts.shape(), want_ts.shape(), "timesteps shape {tag}");
        assert!(
            close(&got_ts, want_ts, 1e-4, 1e-2),
            "timesteps diverged for {tag}"
        );
    }
}

#[test]
fn rope_table_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let pos = Flux2PosEmbed::new(2000.0, [32, 32, 32, 32]);
    let (cos, sin) = pos.forward(w.require("rope.ids").unwrap()).unwrap();
    let want_cos = w.require("rope.cos").unwrap();
    let want_sin = w.require("rope.sin").unwrap();
    assert_eq!(cos.shape(), want_cos.shape());
    assert_eq!(sin.shape(), want_sin.shape());
    assert!(close(&cos, want_cos, 1e-4, 1e-4), "RoPE cos diverged");
    assert!(close(&sin, want_sin, 1e-4, 1e-4), "RoPE sin diverged");
}

#[test]
fn patchify_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let out = patchify_latents(w.require("patch.in").unwrap()).unwrap();
    let want = w.require("patch.out").unwrap();
    assert_eq!(out.shape(), want.shape());
    assert!(close(&out, want, 1e-5, 1e-5), "patchify diverged");
}

#[test]
fn pack_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let out = pack_latents(w.require("pack.in").unwrap()).unwrap();
    let want = w.require("pack.out").unwrap();
    assert_eq!(out.shape(), want.shape());
    assert!(close(&out, want, 1e-5, 1e-5), "pack diverged");
}

#[test]
fn unpack_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    // Dump used height=48, width=32 -> lat 3x2.
    let out = unpack_latents(w.require("unpack.in").unwrap(), 32, 48).unwrap();
    let want = w.require("unpack.out").unwrap();
    assert_eq!(out.shape(), want.shape());
    assert!(close(&out, want, 1e-5, 1e-5), "unpack diverged");
}

#[test]
fn pack_unpack_round_trips() {
    // pack∘unpack is identity on a packed sequence (independent of the fork).
    let w = Weights::from_file(FIXTURE).unwrap();
    let packed = w.require("unpack.in").unwrap();
    let spatial = unpack_latents(packed, 32, 48).unwrap();
    let repacked = pack_latents(&spatial).unwrap();
    assert!(
        close(&repacked, packed, 1e-6, 1e-6),
        "pack∘unpack != identity"
    );
}

#[test]
fn grid_ids_match_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    // Dump used a [1,C,3,2] latent with t_coord=10 -> lat_h=3, lat_w=2.
    let out = prepare_grid_ids(3, 2, 10);
    let want = w.require("gridids.out").unwrap();
    assert_eq!(out.shape(), want.shape());
    assert!(int_eq(&out, want), "grid ids diverged");
}

#[test]
fn text_ids_match_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let out = prepare_text_ids(5);
    let want = w.require("textids.out").unwrap();
    assert_eq!(out.shape(), want.shape());
    assert!(int_eq(&out, want), "text ids diverged");
}
