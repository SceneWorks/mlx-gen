//! sc-7843: real-weight smoke for the qwenimage PiD student. `#[ignore]`d (needs the licensed
//! `nvidia/PiD` checkpoint converted via `tools/convert_pid.py`).
//!
//! Validates that the converted 1.36 B checkpoint loads into `PidNet::from_weights(sr4x())` (every
//! key name + shape matches) and that a forward + the 4-step sampler run end-to-end producing a
//! finite, in-range `[B,3,H,W]` pixel tensor. Pixel parity vs the CUDA reference (sc-7931) is a
//! separate gate; this proves the plumbing on real weights.
//!
//! ```sh
//! PID_QWEN_SAFETENSORS=tools/golden/pid/qwenimage_2kto4k.safetensors \
//!   cargo test -p mlx-gen-pid --release --test e2e_real_weights -- --ignored --nocapture
//! ```

use mlx_gen::weights::Weights;
use mlx_gen_pid::{Gemma2, Gemma2Config, PidConfig, PidNet, Sampler, SamplerConfig};
use mlx_rs::ops::{abs, max};
use mlx_rs::{Array, Dtype};

fn ckpt_path() -> String {
    std::env::var("PID_QWEN_SAFETENSORS")
        .unwrap_or_else(|_| "tools/golden/pid/qwenimage_2kto4k.safetensors".to_string())
}

fn max_abs(a: &Array) -> f32 {
    max(abs(a).unwrap(), None).unwrap().item::<f32>()
}

/// bf16 dummy of `shape` (the reference runs the student in bf16).
fn bf16(shape: &[i32], fill: f32) -> Array {
    let n: i32 = shape.iter().product();
    Array::from_slice(&vec![fill; n as usize], shape)
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
}

#[test]
#[ignore = "needs the converted nvidia/PiD qwenimage checkpoint"]
fn qwenimage_loads_and_runs() {
    let w = Weights::from_file(ckpt_path()).unwrap();
    let cfg = PidConfig::sr4x();
    let net = PidNet::from_weights(&w, "", &cfg).unwrap();
    eprintln!("PidNet loaded from real qwenimage checkpoint (sr4x)");

    // Small smoke resolution: H=W=64 -> patch grid 4x4, latent (z_to_patch_ratio 2) -> 2x2.
    let (b, h, wd) = (1, 64, 64);
    let x = bf16(&[b, 3, h, wd], 0.1);
    let t = Array::from_slice(&[999.0f32], &[b]); // t_cur*timescale ballpark
    let caption = bf16(&[b, 8, cfg.txt_embed_dim], 0.02); // dummy gemma-shaped embeds
    let lq_latent = bf16(&[b, cfg.lq_latent_channels, h / 32, wd / 32], 0.05);
    let sigma = Array::from_slice(&[0.0f32], &[b]);

    // 1) single net forward — shape + finiteness
    let v = net.forward(&x, &t, &caption, &lq_latent, &sigma).unwrap();
    assert_eq!(v.shape(), &[b, 3, h, wd], "net output shape");
    let m = max_abs(&v);
    assert!(m.is_finite(), "net output non-finite (max|·|={m})");
    eprintln!("net.forward ok: shape {:?} max|·|={m:.3e}", v.shape());

    // 2) full 4-step sampler — clamped pixels
    let sampler = Sampler::new(&SamplerConfig::distill_4step());
    let out = sampler
        .sample(&net, &caption, &lq_latent, &sigma, b, h, wd, 0)
        .unwrap();
    assert_eq!(out.shape(), &[b, 3, h, wd], "sampler output shape");
    let om = max_abs(&out);
    assert!(
        om.is_finite() && om <= 1.0 + 1e-3,
        "sampler output out of range (max|·|={om})"
    );
    eprintln!(
        "sampler.sample ok: shape {:?} max|·|={om:.3e} (clamped [-1,1])",
        out.shape()
    );
}

fn gemma_path() -> String {
    std::env::var("PID_GEMMA_SAFETENSORS").unwrap_or_else(|_| {
        format!(
            "{}/.cache/huggingface/hub/models--Efficient-Large-Model--gemma-2-2b-it/snapshots",
            std::env::var("HOME").unwrap()
        )
    })
}

#[test]
#[ignore = "needs the gemma-2-2b-it checkpoint (Efficient-Large-Model mirror)"]
fn gemma2_2b_loads_and_runs() {
    // Resolve the single combined safetensors in the snapshot dir (or an explicit file via env).
    let p = gemma_path();
    let file = if p.ends_with(".safetensors") {
        p
    } else {
        // newest snapshot -> the combined gemma-2-2b-it.safetensors
        let snap = std::fs::read_dir(&p)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|d| d.is_dir())
            .unwrap();
        snap.join("gemma-2-2b-it.safetensors")
            .to_string_lossy()
            .into_owned()
    };
    let w = Weights::from_file(&file).unwrap();
    let model = Gemma2::from_weights(&w, "model.", &Gemma2Config::gemma_2_2b()).unwrap();
    eprintln!("Gemma2 loaded from {file}");

    // small dummy token run — shape + finiteness (parity vs HF is the tiny-fixture gate)
    let ids = Array::from_slice(&[2i32, 651, 1234, 9876, 42, 107], &[1, 6]);
    let h = model.forward(&ids, None).unwrap();
    assert_eq!(h.shape(), &[1, 6, 2304], "gemma last_hidden shape");
    let m = max_abs(&h);
    assert!(m.is_finite(), "gemma output non-finite");
    eprintln!("gemma2.forward ok: shape {:?} max|·|={m:.3e}", h.shape());
}
