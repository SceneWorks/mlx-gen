//! sc-2963 compile-mechanism microbenchmark (rollout of the Wan sc-2957 template) — does
//! `mx.compile` fuse the Qwen-Image MMDiT's fusable elementwise *glue* into faster kernels?
//!
//! No weights — it times the fusable chains in isolation at production shapes (60 dual-stream layers,
//! inner dim 3072 = 24×128, FFN 12288 = 4×dim, ~1024² → seq 4096), eager vs `compile`d. All f32
//! (Qwen runs f32 latents; the affine promotes the bf16 modulation to f32). The chains:
//!   * **gelu_tanh** — the tanh-GELU FFN activation on `[B, S, ffn]` (img + txt FFNs).
//!   * **modulate** — adaLN affine `x·(1+scale)+shift` on `[B, S, dim]`.
//!   * **gated** — gated residual `x + gate·y` on `[B, S, dim]`.
//!   * **rope_rotate** — the complex rotation on `[B, S, H, head_dim/2]` (img/txt q and k).
//!
//! Run it:
//! ```text
//! cargo test --release -p mlx-gen-qwen-image --test compile_micro -- --ignored --nocapture
//! ```

use std::time::Instant;

use mlx_gen::array::scalar;
use mlx_rs::error::Exception;
use mlx_rs::ops::{add, multiply, power, subtract, tanh};
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

fn gelu_body(x: &Array) -> Result<Array, Exception> {
    let dt = x.dtype();
    let s = |v: f32| -> Result<Array, Exception> { scalar(v).as_dtype(dt) };
    let c = (2.0_f64 / std::f64::consts::PI).sqrt() as f32;
    let x3 = power(x, Array::from_int(3))?;
    let inner = multiply(&add(x, &multiply(&x3, &s(0.044_715)?)?)?, &s(c)?)?;
    let gate = add(&tanh(&inner)?, &s(1.0)?)?;
    multiply(&multiply(x, &s(0.5)?)?, &gate)
}

fn modulate_body((x, sc, sh): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    add(&multiply(x, &add(sc, scalar(1.0))?)?, sh)
}

fn gated_body((x, g, y): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    add(x, &multiply(g, y)?)
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
    let b = env_usize("QWEN_PERF_BATCH", 1) as i32;
    let s = env_usize("QWEN_PERF_SEQ", 4096) as i32; // 1024² image tokens
    let dim = env_usize("QWEN_DIM", 3072) as i32;
    let ffn = env_usize("QWEN_FFN", 12288) as i32; // 4 × 3072
    let heads = env_usize("QWEN_HEADS", 24) as i32;
    let half = env_usize("QWEN_HALF", 64) as i32; // head_dim/2
    let warmup = 3usize;
    let iters = 12usize;
    println!("shapes: B={b} S={s} dim={dim} ffn={ffn} heads={heads} half={half}  (warmup={warmup} iters={iters})");

    // ---- gelu_tanh FFN on [B, S, ffn] f32 (~120 / step: img+txt FFN × 60 layers) ----
    {
        let x = normal(&[b, s, ffn]);
        let eager = bench(warmup, iters, || gelu_body(&x).unwrap());
        let oneshot = bench(warmup, iters, || compile(gelu_body, true)(&x).unwrap());
        let mut held = gelu_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut(&x).unwrap());
        let diff = max_abs_diff(&gelu_body(&x).unwrap(), &compile(gelu_body, true)(&x).unwrap());
        println!(
            "[gelu_tanh f32 {b}x{s}x{ffn}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×120={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 120.0
        );
    }

    // ---- modulate (adaLN affine) on [B, S, dim] f32 (~240 / step) ----
    {
        let m = normal(&[b, s, dim]);
        let sc = normal(&[b, 1, dim]);
        let sh = normal(&[b, 1, dim]);
        let eager = bench(warmup, iters, || modulate_body((&m, &sc, &sh)).unwrap());
        let oneshot = bench(warmup, iters, || {
            compile(modulate_body, true)((&m, &sc, &sh)).unwrap()
        });
        let mut held = modulate_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut((&m, &sc, &sh)).unwrap());
        let diff = max_abs_diff(
            &modulate_body((&m, &sc, &sh)).unwrap(),
            &compile(modulate_body, true)((&m, &sc, &sh)).unwrap(),
        );
        println!(
            "[modulate f32 {b}x{s}x{dim}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×240={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 240.0
        );
    }

    // ---- gated residual on [B, S, dim] f32 (~240 / step) ----
    {
        let x = normal(&[b, s, dim]);
        let y = normal(&[b, s, dim]);
        let g = normal(&[b, 1, dim]);
        let eager = bench(warmup, iters, || gated_body((&x, &g, &y)).unwrap());
        let oneshot = bench(warmup, iters, || compile(gated_body, true)((&x, &g, &y)).unwrap());
        let mut held = gated_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut((&x, &g, &y)).unwrap());
        let diff = max_abs_diff(
            &gated_body((&x, &g, &y)).unwrap(),
            &compile(gated_body, true)((&x, &g, &y)).unwrap(),
        );
        println!(
            "[gated f32 {b}x{s}x{dim}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×240={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 240.0
        );
    }

    // ---- rope_rotate on [B, S, H, half] f32 (img/txt q and k, ~240 / step) ----
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
             | held speedup={:.2}× saved/call={:.3}ms ×240={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 240.0
        );
    }
}
