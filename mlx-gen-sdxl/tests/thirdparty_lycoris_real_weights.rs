//! sc-3644 / epic 3641: third-party LyCORIS (LoHa + non-peft LoKr) merges onto the REAL SDXL UNet via
//! the same `apply_sdxl_adapters` path the SceneWorks MLX worker uses — the in-place **merge** path
//! (shared in shape by Wan/LTX). Complements the FLUX.2 residual-path real-weights check.
//!
//! `#[ignore]`d — needs the real SDXL snapshot (`SDXL_SNAPSHOT` or the HF cache). Validates that a
//! genuine third-party file (`lycoris_<flattened.path>.{lokr_*,hada_*}` + per-module `.alpha`, NO
//! `networkType`) resolves against the real UNet module names and folds a delta into the weight. Run:
//!   SDXL_SNAPSHOT=… cargo test -p mlx-gen-sdxl --test thirdparty_lycoris_real_weights -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::{AdapterKind, AdapterSpec};
use mlx_gen_sdxl::{apply_sdxl_adapters, load_unet};
use mlx_rs::Array;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

#[test]
#[ignore = "needs real SDXL weights"]
fn thirdparty_loha_and_lokr_merge_on_real_unet() {
    let unet = load_unet(&snapshot()).unwrap();
    // The vendored LoKr surface (Linear attention/proj) is what the merge resolves against.
    let mut targets: Vec<String> = unet.lora_target_paths();
    targets.sort();
    targets.truncate(4);
    assert!(!targets.is_empty(), "no SDXL LoRA targets enumerated");

    // Resolve each target's [out,in] by re-loading a probe (adaptable_mut needs &mut).
    let mut probe = load_unet(&snapshot()).unwrap();
    let shapes: Vec<(String, Vec<i32>)> = targets
        .iter()
        .map(|p| {
            let segs: Vec<&str> = p.split('.').collect();
            let shape = mlx_gen::adapters::AdaptableHost::adaptable_mut(&mut probe, &segs)
                .expect("target resolves")
                .base_shape();
            (p.clone(), shape)
        })
        .collect();
    println!("SDXL targets: {:?}", shapes);

    let dir = std::env::temp_dir().join("mlx_gen_sdxl_thirdparty_rw");
    std::fs::create_dir_all(&dir).unwrap();
    let r = 2i32;

    let mut loha: Vec<(String, Array)> = Vec::new();
    let mut lokr: Vec<(String, Array)> = Vec::new();
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
    let none = None as Option<&std::collections::HashMap<String, String>>;
    let loha_path = dir.join("thirdparty_loha.safetensors");
    let lokr_path = dir.join("thirdparty_lokr.safetensors");
    Array::save_safetensors(
        loha.iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
        none,
        &loha_path,
    )
    .unwrap();
    Array::save_safetensors(
        lokr.iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
        none,
        &lokr_path,
    )
    .unwrap();

    for (label, path) in [("LoHa", &loha_path), ("LoKr", &lokr_path)] {
        let mut unet = load_unet(&snapshot()).unwrap();
        let report = apply_sdxl_adapters(
            &mut unet,
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
            report.merged,
            shapes.len(),
            "{label}: expected {} merged, got {} (skipped {})",
            shapes.len(),
            report.merged,
            report.skipped_keys
        );
        println!(
            "✓ third-party {label} merged onto all {} real SDXL modules",
            shapes.len()
        );
    }
}
