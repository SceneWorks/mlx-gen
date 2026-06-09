//! Wan-VACE LoRA **real-weight** end-to-end parity gate (sc-3439; `#[ignore]` — needs the real
//! `Wan-AI/Wan2.1-VACE-1.3B-diffusers` transformer (~7 GB), so it never runs in CI; the unit merge
//! gate + the diffusers key-map fixture (`adapters.rs`) carry CI).
//!
//! The definitive "vs the diffusers reference" gate for [`merge_vace_adapters`]. It loads the actual
//! 1.3B VACE transformer, **merges** the committed synthetic LoRA
//! (`tests/fixtures/wanvace_real_lora.safetensors` — native-named base blocks + diffusers-named
//! `vace_blocks`, written by `tools/dump_wanvace_lora_real_golden.py`) via the production
//! `merge_vace_adapters` path, and runs `forward_vace` on the committed real-IO inputs. It asserts the
//! merged forward reproduces the golden `out.lora` (which diffusers produced by folding the SAME LoRA
//! through its own `_convert_non_diffusers_wan_lora_to_diffusers` for the base blocks + a direct fold
//! for `vace_blocks`) to the cross-backend f32 matmul floor, that the LoRA **visibly** moves the
//! output vs `out.bare`, and that a strength-0 merge is a bit-exact no-op.
//!
//! Because the base-block factors are native-named, this exercises the native→diffusers rename of
//! `normalize_vace_key` end-to-end on real weights (it must match diffusers' converter), while the
//! `vace_blocks` factors exercise the diffusers passthrough (incl. `proj_in`/`proj_out`).
//!
//! Run: `WANVACE_DIR=<snapshot> cargo test -p mlx-gen-wan --test wanvace_lora_real_parity -- \
//! --ignored --nocapture` (defaults to the HF cache snapshot of the 1.3B repo).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{AdapterKind, AdapterSpec};
use mlx_gen_wan::config::WanVaceConfig;
use mlx_gen_wan::{merge_vace_adapters, WanVaceTransformer};
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

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// Load the real f32 transformer weight map (cast any bf16 to f32, matching the diffusers reference).
fn load_f32_weights(tdir: &std::path::Path) -> Weights {
    let mut w = Weights::from_dir(tdir).expect("VACE transformer shards");
    w.cast_all(Dtype::Float32).expect("cast weights to f32");
    w
}

/// Build the model from a weight map and run `forward_vace` on the committed real-IO inputs.
fn forward(w: &Weights, cfg: &WanVaceConfig, io: &Weights) -> Vec<f32> {
    let model = WanVaceTransformer::from_weights(w, cfg, Dtype::Float32).expect("build");
    let hs = io.require("in.hidden_states").unwrap();
    let latent = hs.reshape(&hs.shape()[1..]).unwrap();
    let ctrl = io.require("in.control_hidden_states").unwrap();
    let control = ctrl.reshape(&ctrl.shape()[1..]).unwrap();
    let context = io.require("in.encoder_hidden_states").unwrap().clone();
    let t = io.require("in.timestep").unwrap().as_slice::<f32>()[0];
    let scales: Vec<f32> = io
        .require("in.control_hidden_states_scale")
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    model
        .forward_vace(&latent, &control, t, &context, &scales)
        .expect("forward_vace")
        .as_slice::<f32>()
        .to_vec()
}

fn lora_spec(scale: f32) -> AdapterSpec {
    AdapterSpec {
        path: PathBuf::from(fixture("wanvace_real_lora.safetensors")),
        scale,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: None,
    }
}

#[test]
#[ignore = "needs the real Wan2.1-VACE-1.3B-diffusers transformer (~7 GB) — set WANVACE_DIR"]
fn vace_lora_real_merge_matches_diffusers() {
    let dir = snapshot_dir().expect("WANVACE_DIR / HF-cache snapshot with transformer/");
    let tdir = dir.join("transformer");
    let cfg = WanVaceConfig::from_model_dir(&dir).expect("WanVaceConfig");
    assert_eq!(cfg.base.dim, 1536, "1.3B dim");

    let io = Weights::from_file(fixture("wanvace_lora_real_io.safetensors"))
        .expect("lora real-io golden (run dump_wanvace_lora_real_golden.py)");
    let inputs =
        Weights::from_file(fixture("wanvace_real_io.safetensors")).expect("real-io inputs");
    let exp_bare = io.require("out.bare").unwrap().as_slice::<f32>().to_vec();
    let exp_lora = io.require("out.lora").unwrap().as_slice::<f32>().to_vec();

    // --- Bare forward (dense, no adapter) — sanity vs the golden's torch dense output ---
    let bare = forward(&load_f32_weights(&tdir), &cfg, &inputs);
    let (b_max, b_mr) = diff(&bare, &exp_bare);
    println!("[bare]  max|Δ|={b_max:.3e} mean_rel={b_mr:.3e}");

    // --- LoRA-merged forward via merge_vace_adapters (strength 1.0) ---
    let mut w = load_f32_weights(&tdir);
    let report = merge_vace_adapters(&mut w, &[lora_spec(1.0)]).expect("merge");
    println!(
        "[merge] applied={} skipped={:?}",
        report.applied, report.skipped
    );
    assert_eq!(
        report.applied, 53,
        "LoRA must merge all 53 targeted modules"
    );
    assert!(
        report.skipped.is_empty(),
        "unexpected skips: {:?}",
        report.skipped
    );
    let lora = forward(&w, &cfg, &inputs);
    let (l_max, l_mr) = diff(&lora, &exp_lora);
    println!("[lora]  max|Δ|={l_max:.3e} mean_rel={l_mr:.3e}");

    // --- Strength-0 merge must be a bit-exact no-op (W + 0·δ == W) ---
    let mut w0 = load_f32_weights(&tdir);
    merge_vace_adapters(&mut w0, &[lora_spec(0.0)]).expect("merge scale 0");
    let zero = forward(&w0, &cfg, &inputs);
    let (z_max, _) = diff(&zero, &bare);
    println!("[scale0 vs bare] max|Δ|={z_max:.3e}");

    // The merged forward reproduces the diffusers-folded golden to the cross-backend f32 matmul floor
    // (same envelope as the bare VACE real gate, ~6e-2 over the 30-layer + 15-vace stack). A rename /
    // routing / fold-order bug blows past this by orders of magnitude.
    assert!(b_mr < 6e-2, "bare diverged: mean_rel={b_mr:.3e}");
    assert!(
        l_mr < 6e-2,
        "lora-merged forward diverged from diffusers golden: mean_rel={l_mr:.3e}"
    );
    // The LoRA must actually move the output (the golden measured ~12% lora-vs-bare).
    let (_, effect_mr) = diff(&exp_lora, &exp_bare);
    println!("[visible effect] golden lora vs bare mean_rel={effect_mr:.4e}");
    assert!(
        effect_mr > 1e-2,
        "golden LoRA had no visible effect (mean_rel={effect_mr:.4e})"
    );
    // Strength 0 is a no-op.
    assert!(
        z_max < 1e-5,
        "strength-0 merge was not a no-op: max|Δ|={z_max:.3e}"
    );
}
