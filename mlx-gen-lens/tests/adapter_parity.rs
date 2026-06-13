//! sc-3174 — Lens DiT LoRA + LoKr adapter parity vs the torch-PEFT reference.
//!
//! Loads the real `transformer/` weights (f32) and applies the **same** adapter files the trainer
//! ships (`tools/dump_lens_adapter_golden.py` → diffusers `save_lora_adapter` for LoRA;
//! `get_peft_model_state_dict` + `networkType=lokr` metadata for LoKr), then asserts the
//! adapter-applied DiT forward matches the torch-PEFT output `tools/golden/lens_adapter_golden.safetensors`.
//!
//! LoRA/LoKr is a **linear-merge** delta, so the f32 gate is tight (the same 48-block f32 matmul
//! floor as the dense `dit_parity` gate). A scale-0 apply must be a **bit-exact** no-op. All
//! `#[ignore]`d — needs the golden + the ~16 GB transformer snapshot.
//!
//! Run: `cargo test -p mlx-gen-lens --test adapter_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, array_eq, max, multiply, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{AdapterKind, AdapterSpec};
use mlx_gen_lens::adapters::apply_lens_adapters;
use mlx_gen_lens::dit::{LensDitConfig, LensTransformer};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/lens_adapter_golden.safetensors"
);
const LORA: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/lens_lora_adapter.safetensors"
);
const LOKR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/lens_lokr_adapter.safetensors"
);

fn transformer_dir() -> std::path::PathBuf {
    let base = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("snapshot dir {}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max()
        .expect("a snapshot")
        .join("transformer")
}

fn meta_usize(g: &Weights, key: &str) -> usize {
    g.metadata(key).unwrap().parse().unwrap()
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max(abs(want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

fn cosine(got: &Array, want: &Array) -> f32 {
    let dot = sum(multiply(got, want).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let na = sum(multiply(got, got).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    let nb = sum(multiply(want, want).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    dot / (na * nb).max(1e-12)
}

/// Build a fresh f32 DiT from the snapshot (adapters mutate it, so each scenario reloads).
fn load_dit() -> LensTransformer {
    let weights = Weights::from_dir(transformer_dir()).expect("load transformer shards");
    LensTransformer::from_weights(&weights, &LensDitConfig::lens(), Dtype::Float32)
        .expect("load DiT")
}

#[test]
#[ignore = "needs tools/golden/lens_adapter_golden.safetensors + adapters + the Lens-Turbo transformer snapshot (~16GB f32)"]
fn lens_adapters_match_reference() {
    let g = Weights::from_file(GOLDEN).expect("adapter golden");
    let (frame, h, w) = (
        meta_usize(&g, "frame"),
        meta_usize(&g, "h_lat"),
        meta_usize(&g, "w_lat"),
    );
    let n_text = meta_usize(&g, "n_text");
    let f32 = |k: &str| g.require(k).unwrap().as_dtype(Dtype::Float32).unwrap();
    let feats: Vec<Array> = (0..n_text).map(|i| f32(&format!("feat_{i}"))).collect();
    let timestep = f32("timestep");
    let hidden = f32("hidden_states");

    let run = |dit: &LensTransformer| -> Array {
        dit.forward(&hidden, &feats, None, &timestep, frame, h, w)
            .expect("forward")
    };

    // --- 1. Base (sanity) + scale-0 no-op (bit-exact) ---
    let mut dit = load_dit();
    let base = run(&dit);
    let base_pr = peak_rel(&base, &f32("base_out"));
    eprintln!(
        "base: peak_rel {base_pr:.3e}  cosine {:.7}",
        cosine(&base, &f32("base_out"))
    );
    assert!(
        base_pr < 1.5e-2,
        "base peak_rel {base_pr:.3e} — DiT load drift"
    );

    // A scale-0 LoRA is a pure no-op: the applied DiT must reproduce the base forward bit-for-bit.
    apply_lens_adapters(
        &mut dit,
        &[AdapterSpec::new(LORA.into(), 0.0, AdapterKind::Lora)],
    )
    .expect("apply scale-0 lora");
    let zero = run(&dit);
    assert!(
        array_eq(&zero, &base, None).unwrap().item::<bool>(),
        "scale-0 LoRA is not a bit-exact no-op"
    );
    eprintln!("scale-0 LoRA: bit-exact no-op ✓");

    // --- 2. LoRA @ scale 1 vs torch-PEFT ---
    let mut dit = load_dit();
    let report = apply_lens_adapters(
        &mut dit,
        &[AdapterSpec::new(LORA.into(), 1.0, AdapterKind::Lora)],
    )
    .expect("apply lora");
    eprintln!("lora applied: {} module(s)", report.applied);
    assert!(report.applied > 0, "no LoRA targets matched");
    let lora = run(&dit);
    let lora_pr = peak_rel(&lora, &f32("lora_out"));
    let lora_cos = cosine(&lora, &f32("lora_out"));
    eprintln!("lora: peak_rel {lora_pr:.3e}  cosine {lora_cos:.7}");
    assert!(
        lora_pr < 1.5e-2 && lora_cos > 0.9999,
        "LoRA diverged from torch-PEFT"
    );

    // --- 3. LoKr @ scale 1 vs torch-PEFT ---
    let mut dit = load_dit();
    let report = apply_lens_adapters(
        &mut dit,
        &[AdapterSpec::new(LOKR.into(), 1.0, AdapterKind::Lokr)],
    )
    .expect("apply lokr");
    eprintln!("lokr applied: {} module(s)", report.applied);
    assert!(report.applied > 0, "no LoKr targets matched");
    let lokr = run(&dit);
    let lokr_pr = peak_rel(&lokr, &f32("lokr_out"));
    let lokr_cos = cosine(&lokr, &f32("lokr_out"));
    eprintln!("lokr: peak_rel {lokr_pr:.3e}  cosine {lokr_cos:.7}");
    assert!(
        lokr_pr < 1.5e-2 && lokr_cos > 0.9999,
        "LoKr diverged from torch-PEFT"
    );

    eprintln!("ALL PASS");
}
