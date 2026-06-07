//! Wan-VACE transformer **real-weight** parity (epic 3040 / sc-3388, S1 upgrade) — `#[ignore]` heavy.
//!
//! Loads the real `Wan-AI/Wan2.1-VACE-1.3B-diffusers` transformer (dim 1536, 30 layers, 15 vace
//! blocks) **directly** from its diffusers safetensors shards (the port reads diffusers tensor names
//! natively — no conversion) and compares `WanVaceTransformer::forward_vace` against the diffusers
//! `WanVACETransformer3DModel` forward on the same injected inputs (`tools/dump_wanvace_real_golden.py`
//! → `tests/fixtures/wanvace_real_io.safetensors`, committed — a few MB of I/O, not the 7 GB weights).
//!
//! This upgrades the S1 structural golden (random small-config) to **real 1.3B weights**: it confirms
//! the diffusers-name loader maps every key + dim correctly and that `forward_vace` reproduces the
//! reference. Compared **f32-vs-f32**; the residual is the documented cross-backend matmul-precision
//! floor (mlx Metal f32 matmul vs torch CPU f32 — `wanvace_transformer_parity.rs` root-cause),
//! amplified over the 30-layer + 15-vace-block stack (so a few e-2, not bit-exact — same kind of
//! named floor the base Wan S3 gate carries). The non-trivial monotone `control_hidden_states_scale`
//! in the golden makes a mis-applied / reversed hint scale fail decisively.
//!
//! Run: `WANVACE_DIR=<snapshot> cargo test -p mlx-gen-wan --test wanvace_real_parity -- --ignored
//! --nocapture` (defaults to the HF cache snapshot of the 1.3B repo).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanVaceConfig;
use mlx_gen_wan::WanVaceTransformer;
use mlx_rs::Dtype;

/// Resolve the snapshot dir (the one holding `transformer/`): `WANVACE_DIR`, else the HF cache.
fn snapshot_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("WANVACE_DIR") {
        return Some(PathBuf::from(d));
    }
    let base = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B-diffusers/snapshots");
    std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.join("transformer").is_dir())
}

fn diff(got: &[f32], exp: &[f32]) -> (f32, f64) {
    let (mut ma, mut sa, mut sr) = (0f32, 0f64, 0f64);
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        ma = ma.max(d);
        sa += d as f64;
        sr += e.abs() as f64;
    }
    (ma, sa / sr.max(1e-30))
}

#[test]
#[ignore = "needs the real Wan2.1-VACE-1.3B-diffusers transformer (~7 GB) — set WANVACE_DIR"]
fn vace_real_forward_matches_diffusers() {
    let dir = snapshot_dir().expect("WANVACE_DIR / HF-cache snapshot with transformer/");
    let tdir = dir.join("transformer");

    let cfg =
        WanVaceConfig::from_model_dir(&dir).expect("WanVaceConfig from transformer/config.json");
    assert_eq!(cfg.base.dim, 1536, "1.3B dim");
    assert_eq!(cfg.base.num_layers, 30);
    assert_eq!(cfg.vace_layers.len(), 15);

    let mut w = Weights::from_dir(&tdir).expect("VACE transformer shards");
    // Match the diffusers f32 reference exactly (the matmul weights too) — cast any bf16 to f32.
    w.cast_all(Dtype::Float32).expect("cast weights to f32");
    let model = WanVaceTransformer::from_weights(&w, &cfg, Dtype::Float32)
        .expect("build WanVaceTransformer");

    let io = Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/wanvace_real_io.safetensors"
    ))
    .expect("real-io fixture");

    let hs = io.require("in.hidden_states").unwrap();
    let latent = hs.reshape(&hs.shape()[1..]).unwrap(); // drop batch → [16,T,H,W]
    let ctrl = io.require("in.control_hidden_states").unwrap();
    let control = ctrl.reshape(&ctrl.shape()[1..]).unwrap(); // [96,T,H,W]
    let context = io.require("in.encoder_hidden_states").unwrap().clone(); // [1,L,4096]
    let t = io.require("in.timestep").unwrap().as_slice::<f32>()[0];
    let scales: Vec<f32> = io
        .require("in.control_hidden_states_scale")
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert_eq!(scales.len(), cfg.vace_layers.len());

    let out = model
        .forward_vace(&latent, &control, t, &context, &scales)
        .expect("forward_vace");
    let got = out.as_slice::<f32>().to_vec();
    let exp = io.require("out.sample").unwrap().as_slice::<f32>().to_vec();
    assert_eq!(got.len(), exp.len(), "output length");

    let (max_abs, mean_rel) = diff(&got, &exp);
    println!(
        "[vace real 1.3B] dim={} layers={} vace={} → max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}",
        cfg.base.dim,
        cfg.base.num_layers,
        cfg.vace_layers.len()
    );

    // Cross-backend f32 matmul floor over a 30-layer + 15-vace-block stack — a named precision delta,
    // not a port bug (see wanvace_transformer_parity.rs). Generous like the base Wan S3 bf16 gate; a
    // VACE-logic bug (wrong key map / hint scale / order) blows past this by orders of magnitude.
    assert!(
        mean_rel < 6e-2,
        "VACE real forward diverges past the matmul floor: max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}"
    );
}
