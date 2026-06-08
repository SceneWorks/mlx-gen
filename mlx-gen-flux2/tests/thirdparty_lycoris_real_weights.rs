//! sc-3644 / epic 3641: third-party LyCORIS (LoHa + non-peft LoKr) applies on the REAL FLUX.2-klein
//! module tree via the same `apply_flux2_adapters` path the SceneWorks MLX worker uses.
//!
//! `#[ignore]`d — needs the real FLUX.2-klein weights (env `MLX_GEN_FLUX2_SNAPSHOT` or the HF cache).
//! Validates the link the unit/parity tests can't: a genuine third-party file (kohya/lycoris keys —
//! `<prefix>_<flattened.path>.{lokr_*,hada_*}` + per-module `.alpha`, NO `networkType` metadata)
//! resolves against the real model's module names and installs a forward-time delta. Run:
//!   MLX_GEN_FLUX2_SNAPSHOT=… cargo test -p mlx-gen-flux2 --test thirdparty_lycoris_real_weights -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::{AdapterKind, AdapterSpec};
use mlx_gen_flux2::{apply_flux2_adapters, load_transformer};
use mlx_rs::Array;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9b/snapshots");
    std::fs::read_dir(&snaps)
        .expect("snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// The lokr/lora kind a target installs (we only assert it's a forward `Lokr` residual = the
/// reconstructed-delta path both third-party LoKr and LoHa use).
fn is_lokr_adapter(a: &[Adapter]) -> bool {
    matches!(a, [Adapter::Lokr { .. }])
}

#[test]
#[ignore = "needs real FLUX.2-klein weights"]
fn thirdparty_loha_and_lokr_apply_on_real_tree() {
    // Enumerate the real module tree and take a handful of clean Linear targets.
    let mut probe = load_transformer(&snapshot()).unwrap();
    let mut targets: Vec<String> = probe
        .adaptable_paths()
        .into_iter()
        // Drop the `.0` HF alias when the bare sibling is also enumerated (FLUX.2 `to_out` quirk).
        .filter(|p| !p.ends_with(".0"))
        .collect();
    targets.sort();
    targets.truncate(4);
    assert!(!targets.is_empty(), "no adaptable targets enumerated");
    let shapes: Vec<(String, Vec<i32>)> = targets
        .iter()
        .map(|p| {
            let segs: Vec<&str> = p.split('.').collect();
            let shape = AdaptableHost::adaptable_mut(&mut probe, &segs)
                .unwrap()
                .base_shape();
            (p.clone(), shape)
        })
        .collect();
    println!(
        "targets: {:?}",
        shapes.iter().map(|(p, s)| (p, s)).collect::<Vec<_>>()
    );

    let dir = std::env::temp_dir().join("mlx_gen_flux2_thirdparty_rw");
    std::fs::create_dir_all(&dir).unwrap();
    let r = 2i32;

    // ---- third-party LoHa: lycoris keys, per-module .alpha (scale = alpha/rank = 1), NO metadata.
    let mut loha: Vec<(String, Array)> = Vec::new();
    for (p, shape) in &shapes {
        let (out, inp) = (shape[0], shape[1]);
        let stem = format!("lycoris_{}", p.replace('.', "_"));
        let wa = |seed: i32| {
            Array::from_slice(
                &(0..out * r)
                    .map(|i| (((i + seed) % 7) as f32 - 3.0) * 0.001)
                    .collect::<Vec<_>>(),
                &[out, r],
            )
        };
        let wb = |seed: i32| {
            Array::from_slice(
                &(0..r * inp)
                    .map(|i| (((i + seed) % 5) as f32 - 2.0) * 0.001)
                    .collect::<Vec<_>>(),
                &[r, inp],
            )
        };
        loha.push((format!("{stem}.hada_w1_a"), wa(0)));
        loha.push((format!("{stem}.hada_w1_b"), wb(1)));
        loha.push((format!("{stem}.hada_w2_a"), wa(2)));
        loha.push((format!("{stem}.hada_w2_b"), wb(3)));
        loha.push((
            format!("{stem}.alpha"),
            Array::from_slice(&[r as f32], &[1]),
        ));
    }
    let loha_path = dir.join("thirdparty_loha.safetensors");
    Array::save_safetensors(
        loha.iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
        None as Option<&std::collections::HashMap<String, String>>,
        &loha_path,
    )
    .unwrap();

    // ---- third-party LoKr: full `lokr_w1` ([out,in]) ⊗ 1×1 `lokr_w2` (both-full → scale 1), NO metadata.
    let mut lokr: Vec<(String, Array)> = Vec::new();
    for (p, shape) in &shapes {
        let (out, inp) = (shape[0], shape[1]);
        let stem = format!("lycoris_{}", p.replace('.', "_"));
        let w1 = Array::from_slice(
            &(0..out * inp)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.0005)
                .collect::<Vec<_>>(),
            &[out, inp],
        );
        lokr.push((format!("{stem}.lokr_w1"), w1));
        lokr.push((
            format!("{stem}.lokr_w2"),
            Array::from_slice(&[1.0f32], &[1, 1]),
        ));
        lokr.push((
            format!("{stem}.alpha"),
            Array::from_slice(&[r as f32], &[1]),
        ));
    }
    let lokr_path = dir.join("thirdparty_lokr.safetensors");
    Array::save_safetensors(
        lokr.iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
        None as Option<&std::collections::HashMap<String, String>>,
        &lokr_path,
    )
    .unwrap();

    // ---- apply each onto a fresh real transformer, assert full resolution + a Lokr residual.
    for (label, path) in [("LoHa", &loha_path), ("LoKr", &lokr_path)] {
        let mut t = load_transformer(&snapshot()).unwrap();
        let report = apply_flux2_adapters(
            &mut t,
            &[AdapterSpec {
                path: path.clone(),
                scale: 1.0,
                kind: AdapterKind::Lora, // mislabeled on purpose — detection-by-keys must override
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap_or_else(|e| panic!("{label}: apply failed: {e}"));
        assert_eq!(
            report.applied,
            shapes.len(),
            "{label}: expected all {} targets applied, got {}",
            shapes.len(),
            report.applied
        );
        assert!(
            report.unmatched_paths.is_empty(),
            "{label}: unmatched {:?}",
            report.unmatched_paths
        );
        for (p, _) in &shapes {
            let segs: Vec<&str> = p.split('.').collect();
            let adapters = AdaptableHost::adaptable_mut(&mut t, &segs)
                .unwrap()
                .adapters();
            assert!(
                is_lokr_adapter(adapters),
                "{label}: {p} did not install a reconstructed-delta residual"
            );
        }
        println!(
            "✓ third-party {label} resolved + installed on all {} real FLUX.2 modules",
            shapes.len()
        );
    }
}
