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

use mlx_gen::weights::Weights;
use mlx_gen_lens::schedule::{cfg_rescale, lens_schedule, timesteps};

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

    // (d) norm-rescaled CFG (guidance 5.0).
    let cond = g.require("cfg_cond").unwrap().clone();
    let uncond = g.require("cfg_uncond").unwrap().clone();
    let got_cfg = cfg_rescale(&cond, &uncond, 5.0).unwrap();
    let d_cfg = peak_rel(&got_cfg, g.require("cfg_out").unwrap());
    eprintln!("cfg: peak_rel {d_cfg:.3e}");
    assert!(d_cfg < 1e-4, "cfg peak_rel {d_cfg:.3e}");
    eprintln!("ALL PASS");
}
