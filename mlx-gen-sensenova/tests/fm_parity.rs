//! sc-3184: the flow-matching head, timestep embedder, and sampler math match the reference.
//!
//! Synthetic fixture (`tools/dump_sensenova_fm_golden.py`): the shallow `fm_head`
//! (Linear→GELU→Linear), the GLIDE `TimestepEmbedder`, the standard time schedule, the Euler step +
//! velocity formula, and patchify/unpatchify — all dumped from the reference. f32.
//!
//! Run: `cargo test -p mlx-gen-sensenova --test fm_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_sensenova::{
    apply_time_schedule, euler_step, patchify, unpatchify, velocity, FmHead, TimestepEmbedder,
};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/fm_golden.safetensors"
);

fn rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    a.iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
        / peak
}

fn check(name: &str, got: &Array, want: &Array) {
    assert_eq!(got.shape(), want.shape(), "{name} shape");
    let r = rel(got, want);
    println!("{name:>20}: peak-rel={r:.3e}");
    assert!(r < 5e-3, "{name} peak-rel {r:.3e} exceeds 5e-3");
}

#[test]
fn fm_head_matches_reference() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let head = FmHead::from_weights(&w, "fm_modules.fm_head").unwrap();
    let got = head.forward(w.require("fm.in").unwrap()).unwrap();
    check("fm.out", &got, w.require("fm.out").unwrap());
}

#[test]
fn timestep_embedder_matches_reference() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let te = TimestepEmbedder::from_weights(&w, "fm_modules.timestep_embedder").unwrap();
    let got = te.forward(w.require("ts.in").unwrap()).unwrap();
    check("ts.out", &got, w.require("ts.out").unwrap());
}

#[test]
fn time_schedule_matches_reference() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let t = w.require("sched.t").unwrap();
    check(
        "sched.shift1",
        &apply_time_schedule(t, 1.0).unwrap(),
        w.require("sched.standard_shift1").unwrap(),
    );
    check(
        "sched.shift3",
        &apply_time_schedule(t, 3.0).unwrap(),
        w.require("sched.standard_shift3").unwrap(),
    );
    check(
        "sched.shift05",
        &apply_time_schedule(t, 0.5).unwrap(),
        w.require("sched.standard_shift05").unwrap(),
    );
}

#[test]
fn euler_and_velocity_match_reference() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let z = w.require("euler.z").unwrap();
    let x_pred = w.require("euler.x_pred").unwrap();
    // t_scalar=0.3, t_next=0.55, t_eps=0.05 (fixture metadata).
    let v = velocity(x_pred, z, 0.3, 0.05).unwrap();
    check("v_pred", &v, w.require("euler.v_pred").unwrap());
    let zn = euler_step(&v, z, 0.3, 0.55).unwrap();
    check("z_next", &zn, w.require("euler.z_next").unwrap());
}

#[test]
fn patchify_roundtrip_matches_reference() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let img = w.require("patch.img").unwrap();
    let patches = patchify(img, 2).unwrap();
    check("patches", &patches, w.require("patch.patches").unwrap());
    let recon = unpatchify(&patches, 2, Some(4), Some(4)).unwrap();
    check("recon", &recon, w.require("patch.recon").unwrap());
}
