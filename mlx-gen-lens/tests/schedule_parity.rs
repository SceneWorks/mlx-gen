//! sc-3170 — Lens schedule + CFG parity vs `LensPipeline` (diffusers `FlowMatchEulerDiscreteScheduler`).
//!
//! Weight-free: checks, against `tools/dump_lens_schedule_golden.py`, that for both the Turbo (4-step)
//! and base (20-step) counts the Rust [`mlx_gen_lens::schedule`] reproduces (a) the shifted sigmas,
//! (b) the per-step transformer timesteps, (c) a single flow-match Euler `step`, and (d) the
//! norm-rescaled CFG — all near-bit (f32).
//!
//! Run: `cargo test -p mlx-gen-lens --test schedule_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::guidance::cfg_rescale;
use mlx_gen::weights::Weights;
use mlx_gen::MlxLatentOps;
use mlx_gen_lens::schedule::{lens_schedule, timesteps};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/lens_schedule_golden.safetensors"
);

const LATENT: usize = 64; // 64×64 = 4096 seq_len (matches the dump)

fn max_abs(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    max(abs(subtract(&a, &b).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>()
}

fn peak_rel(a: &Array, b: &Array) -> f32 {
    let b32 = b.as_dtype(Dtype::Float32).unwrap();
    let denom = max(abs(&b32).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .max(1e-12);
    max_abs(a, b) / denom
}

#[test]
#[ignore = "needs tools/golden/lens_schedule_golden.safetensors"]
fn lens_schedule_matches_reference() {
    let g = Weights::from_file(GOLDEN).expect("schedule golden");

    for n in [4usize, 20] {
        let sched = lens_schedule(n, LATENT, LATENT);

        // (a) shifted sigmas (n+1, trailing 0).
        let got_sigmas = Array::from_slice(&sched.sigmas, &[sched.sigmas.len() as i32]);
        let want_sigmas = g.require(&format!("sigmas_{n}")).unwrap();
        let d_sig = max_abs(&got_sigmas, want_sigmas);

        // (b) per-step timesteps = shifted sigma; golden stores sigma·1000.
        let ts = timesteps(&sched);
        let got_ts = Array::from_slice(&ts, &[ts.len() as i32]);
        let want_ts = g.require(&format!("timesteps_{n}")).unwrap();
        let want_ts_div = mlx_rs::ops::multiply(
            want_ts.as_dtype(Dtype::Float32).unwrap(),
            Array::from_f32(1.0 / 1000.0),
        )
        .unwrap();
        let d_ts = max_abs(&got_ts, &want_ts_div);

        // (c) one flow-match Euler step at index 0.
        let step_in = g.require(&format!("step_in_{n}")).unwrap().clone();
        let step_noise = g.require(&format!("step_noise_{n}")).unwrap().clone();
        let got_step = sched.step(&step_in, &step_noise, 0).unwrap();
        let d_step = peak_rel(&got_step, g.require(&format!("step_out_{n}")).unwrap());

        eprintln!(
            "n={n}: sigmas Δ {d_sig:.3e} | timesteps Δ {d_ts:.3e} | step peak_rel {d_step:.3e}"
        );
        assert!(d_sig < 1e-5, "n={n} sigmas Δ {d_sig:.3e}");
        assert!(d_ts < 1e-5, "n={n} timesteps Δ {d_ts:.3e}");
        assert!(d_step < 1e-4, "n={n} step peak_rel {d_step:.3e}");
    }

    // (d) norm-rescaled CFG (guidance 5.0) — the migrated path: the shared per-token
    // `gen_core::guidance::cfg_rescale` over the MLX ops (epic 7434 P3, sc-7441), the exact call the
    // pipeline now makes. Still gated against the vendor `LensPipeline` golden `cfg_out`.
    let cond = g.require("cfg_cond").unwrap().clone();
    let uncond = g.require("cfg_uncond").unwrap().clone();
    let got_cfg = cfg_rescale(&MlxLatentOps, &cond, &uncond, 5.0, &[], &[-1]).unwrap();
    let d_cfg = peak_rel(&got_cfg, g.require("cfg_out").unwrap());
    eprintln!("cfg: peak_rel {d_cfg:.3e}");
    assert!(d_cfg < 1e-4, "cfg peak_rel {d_cfg:.3e}");
    eprintln!("ALL PASS");
}

/// The exact bespoke Lens `cfg_rescale` retired in sc-7441 (was `schedule.rs:58-74`), inlined here as
/// the N1 byte-equivalence reference for [`migrated_cfg_rescale_is_byte_identical`].
fn legacy_cfg_rescale(cond: &Array, uncond: &Array, guidance: f32) -> Array {
    use mlx_rs::ops::{
        add, divide, gt, maximum, multiply, ones_like, sqrt, subtract, sum_axes, which,
    };
    let g = Array::from_f32(guidance);
    let diff = subtract(cond, uncond).unwrap();
    let scaled = multiply(&diff, &g).unwrap();
    let comb = add(uncond, &scaled).unwrap();
    let norm = |x: &Array| {
        let sq = multiply(x, x).unwrap();
        let summed = sum_axes(&sq, &[-1], true).unwrap();
        sqrt(&summed).unwrap()
    };
    let cond_norm = norm(cond);
    let comb_norm = norm(&comb);
    let denom = maximum(&comb_norm, Array::from_f32(1e-12)).unwrap();
    let ratio = divide(&cond_norm, &denom).unwrap();
    let positive = gt(&comb_norm, Array::from_f32(0.0)).unwrap();
    let ones = ones_like(&comb_norm).unwrap();
    let scale = which(&positive, &ratio, &ones).unwrap();
    multiply(&comb, &scale).unwrap()
}

/// sc-7441 (epic 7434 P3) — the Lens CFG migration is a **bit-identical** drop-in: the shared
/// `gen_core::guidance::cfg_rescale` over the MLX [`MlxLatentOps`] (per-token `[-1]`) must reproduce
/// the retired bespoke `schedule::cfg_rescale` exactly on representative `[B, seq, C]` predictions.
/// Weight-free (no golden needed) — proves N1 over the op graph itself, not just the vendor parity.
#[test]
fn migrated_cfg_rescale_is_byte_identical() {
    let (b, seq, c) = (2usize, 37usize, 128usize);
    let n = b * seq * c;
    // Deterministic, non-trivial cond/uncond (spans signs/magnitudes; no RNG → reproducible).
    let cond_v: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.013).sin() * 1.7 - 0.3)
        .collect();
    let uncond_v: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.021 + 1.0).cos() * 0.9)
        .collect();
    let shape = [b as i32, seq as i32, c as i32];
    let cond = Array::from_slice(&cond_v, &shape);
    let uncond = Array::from_slice(&uncond_v, &shape);

    for &g in &[1.0f32, 5.0] {
        let legacy = legacy_cfg_rescale(&cond, &uncond, g);
        let migrated = cfg_rescale(&MlxLatentOps, &cond, &uncond, g, &[], &[-1]).unwrap();
        let d = max_abs(&migrated, &legacy);
        eprintln!("g={g}: migrated-vs-legacy max|Δ| = {d:e}");
        assert_eq!(d, 0.0, "g={g} migration not bit-identical: max|Δ| = {d:e}");
    }
}
