//! sc-2963 compile-mechanism microbenchmark (rollout of the Wan sc-2957 template) — does
//! `mx.compile` fuse the Z-Image DiT's fusable elementwise *glue* into faster kernels?
//!
//! No weights — it times the fusable chains in isolation at Z-Image-turbo production shapes (dim 3840
//! = 30×128, SwiGLU hidden ≈10240 (the Lumina 2/3-of-4× rule; override with ZIMAGE_FFN), 30 layers,
//! ~1024² → seq 4096), eager vs `compile`d. Measured f32 here (the base txt2img path also runs bf16;
//! the fusion win is shape-driven). The chains:
//!   * **swiglu** — `silu(h1)·h3` (the FFN activation; w1/w3 GEMMs stay eager).
//!   * **gated** — gated residual `x + gate·normed` (the `mx.fast` RMSNorm stays eager).
//!   * **rope_rotate** — the complex rotation on `[B, S, H, head_dim/2]` (q and k).
//! The control-only `add_hint` (`x + hint·scale`) fuses identically to `gated` (2 ops → 1 kernel).
//!
//! Run it:
//! ```text
//! cargo test --release -p mlx-gen-z-image --test compile_micro -- --ignored --nocapture
//! ```

use std::time::Instant;

use mlx_rs::error::Exception;
use mlx_rs::ops::{add, multiply, sigmoid, subtract};
use mlx_rs::transforms::compile::{compile, CallMut, Compile};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array};

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn bench(warmup: usize, iters: usize, mut f: impl FnMut() -> Array) -> f64 {
    let mut times = Vec::new();
    for i in 0..(warmup + iters) {
        let t0 = Instant::now();
        let out = f();
        eval([&out]).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if i >= warmup {
            times.push(dt);
        }
    }
    median(times)
}

fn swiglu_body((h1, h3): (&Array, &Array)) -> Result<Array, Exception> {
    multiply(&multiply(h1, &sigmoid(h1)?)?, h3)
}

fn gated_body((x, g, n): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    add(x, &multiply(g, n)?)
}

fn rope_body(inp: &[Array]) -> Result<Vec<Array>, Exception> {
    let (r, i, c, s) = (&inp[0], &inp[1], &inp[2], &inp[3]);
    let out0 = subtract(&multiply(r, c)?, &multiply(i, s)?)?;
    let out1 = add(&multiply(r, s)?, &multiply(i, c)?)?;
    Ok(vec![out0, out1])
}

fn normal(shape: &[i32]) -> Array {
    let key = random::key(0).unwrap();
    let x = random::normal::<f32>(shape, None, None, Some(&key)).unwrap();
    eval([&x]).unwrap();
    x
}

fn max_abs_diff(a: &Array, b: &Array) -> f64 {
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
    mlx_rs::ops::max(&d, None).unwrap().item::<f32>() as f64
}

#[test]
#[ignore = "perf microbenchmark (no weights) — run with --ignored --nocapture"]
fn compile_glue_micro() {
    let b = env_usize("ZIMAGE_PERF_BATCH", 1) as i32;
    let s = env_usize("ZIMAGE_PERF_SEQ", 4096) as i32;
    let dim = env_usize("ZIMAGE_DIM", 3840) as i32;
    let ffn = env_usize("ZIMAGE_FFN", 10240) as i32; // ≈ Lumina 2/3-of-4× SwiGLU hidden (approx)
    let heads = env_usize("ZIMAGE_HEADS", 30) as i32;
    let half = env_usize("ZIMAGE_HALF", 64) as i32;
    let warmup = 3usize;
    let iters = 12usize;
    println!("shapes: B={b} S={s} dim={dim} ffn={ffn} heads={heads} half={half}  (warmup={warmup} iters={iters})");

    // ---- swiglu silu(h1)·h3 on [B, S, ffn] f32 (~34 / step) ----
    {
        let h1 = normal(&[b, s, ffn]);
        let h3 = normal(&[b, s, ffn]);
        let eager = bench(warmup, iters, || swiglu_body((&h1, &h3)).unwrap());
        let oneshot = bench(warmup, iters, || compile(swiglu_body, true)((&h1, &h3)).unwrap());
        let mut held = swiglu_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut((&h1, &h3)).unwrap());
        let diff = max_abs_diff(
            &swiglu_body((&h1, &h3)).unwrap(),
            &compile(swiglu_body, true)((&h1, &h3)).unwrap(),
        );
        println!(
            "[swiglu f32 {b}x{s}x{ffn}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×34={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 34.0
        );
    }

    // ---- gated residual on [B, S, dim] f32 (~60 / step) ----
    {
        let x = normal(&[b, s, dim]);
        let n = normal(&[b, s, dim]);
        let g = normal(&[b, 1, dim]);
        let eager = bench(warmup, iters, || gated_body((&x, &g, &n)).unwrap());
        let oneshot = bench(warmup, iters, || compile(gated_body, true)((&x, &g, &n)).unwrap());
        let mut held = gated_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut((&x, &g, &n)).unwrap());
        let diff = max_abs_diff(
            &gated_body((&x, &g, &n)).unwrap(),
            &compile(gated_body, true)((&x, &g, &n)).unwrap(),
        );
        println!(
            "[gated f32 {b}x{s}x{dim}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×60={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 60.0
        );
    }

    // ---- rope_rotate on [B, S, H, half] f32 (q and k, ~68 / step) ----
    {
        let r = normal(&[b, s, heads, half]);
        let im = normal(&[b, s, heads, half]);
        let c = normal(&[1, s, 1, half]);
        let sn = normal(&[1, s, 1, half]);
        let args = [r.clone(), im.clone(), c.clone(), sn.clone()];
        let eager = bench(warmup, iters, || rope_body(&args).unwrap().pop().unwrap());
        let oneshot = bench(warmup, iters, || {
            compile(rope_body, true)(&args).unwrap().pop().unwrap()
        });
        let mut held = rope_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut(&args).unwrap().pop().unwrap());
        let diff = max_abs_diff(
            &rope_body(&args).unwrap()[0],
            &compile(rope_body, true)(&args).unwrap()[0],
        );
        println!(
            "[rope_rotate f32 {b}x{s}x{heads}x{half}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×68={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 68.0
        );
    }
}
