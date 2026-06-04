//! sc-2602: end-to-end Z-Image LoRA/LoKr adapter consumption against real weights.
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image-Turbo` weights in the HF cache and the
//! adapter goldens produced by `tools/dump_z_image_adapter_golden.py` (gitignored, local). Run:
//!   cargo test -p mlx-gen-z-image --release --test adapter_real_weights -- --ignored --nocapture
//!
//! Three gates: (1) the key→module map resolves the FULL fork `ZImageLoRAMapping` target surface
//! against the real module tree; (2) the public `load(spec.with_adapters(…)).generate()` render
//! matches the fork's LoRA *and* LoKr golden (px>8); (3) a scale-0 adapter is a bit-exact no-op.

use std::path::PathBuf;

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen_z_image::{apply_z_image_adapters, decoded_to_image, load_transformer};
use mlx_rs::ops::array_eq;
use mlx_rs::Array;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("ZIMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden")
}

/// (1) The top-level `AdaptableHost` resolves every fork `ZImageLoRAMapping` target against the
/// real module tree (30 layers + 2 noise_refiner + 2 context_refiner + 6 globals), and rejects
/// off-surface paths — no real weights for the *render*, but needs them to build the transformer.
#[test]
#[ignore = "needs real Z-Image weights"]
fn routing_map_covers_full_fork_surface() {
    let mut t = load_transformer(&snapshot()).unwrap();
    let resolves = |t: &mut _, p: &str| -> bool {
        let segs: Vec<&str> = p.split('.').collect();
        AdaptableHost::adaptable_mut(t, &segs).is_some()
    };

    let block_targets = [
        "attention.to_q",
        "attention.to_k",
        "attention.to_v",
        "attention.to_out.0",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
        "adaLN_modulation.0",
    ];
    // Main + noise_refiner blocks expose all eight targets at every index.
    for (stack, n) in [("layers", 30usize), ("noise_refiner", 2)] {
        for i in 0..n {
            for tgt in block_targets {
                let p = format!("{stack}.{i}.{tgt}");
                assert!(resolves(&mut t, &p), "expected {p} to resolve");
            }
        }
    }
    // Context-refiner blocks have no timestep → attention + feed_forward only (adaLN is correctly
    // absent, mirroring the fork: the file never populates it).
    for i in 0..2 {
        assert!(resolves(
            &mut t,
            &format!("context_refiner.{i}.attention.to_q")
        ));
        assert!(resolves(
            &mut t,
            &format!("context_refiner.{i}.feed_forward.w2")
        ));
        assert!(
            !resolves(&mut t, &format!("context_refiner.{i}.adaLN_modulation.0")),
            "context blocks carry no adaLN"
        );
    }
    // The six global targets (trained-file naming).
    for p in [
        "all_x_embedder.2-1",
        "cap_embedder.1",
        "t_embedder.mlp.0",
        "t_embedder.mlp.2",
        "all_final_layer.2-1.linear",
        "all_final_layer.2-1.adaLN_modulation.1",
    ] {
        assert!(resolves(&mut t, p), "expected global {p} to resolve");
    }
    // Off-surface paths must not resolve.
    for p in [
        "layers.30.attention.to_q",               // out of range
        "layers.0.attention.to_x",                // unknown proj
        "all_final_layer.2-1.adaLN_modulation.0", // final layer uses index 1, not 0
        "cap_embedder.0",                         // the RMSNorm, not a Linear
        "t_embedder.mlp.1",                       // the SiLU slot
    ] {
        assert!(!resolves(&mut t, p), "expected {p} NOT to resolve");
    }
    println!("✓ routing map covers the full fork ZImageLoRAMapping surface");
}

fn render_with_adapter(adapter: Option<(&str, AdapterKind, f32)>, golden_kind: &str) -> Vec<u8> {
    let g =
        Weights::from_file(golden_dir().join(format!("z_image_{golden_kind}_golden.safetensors")))
            .unwrap();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();

    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    if let Some((file, kind, scale)) = adapter {
        spec = spec.with_adapters(vec![AdapterSpec {
            path: golden_dir().join(file),
            scale,
            kind,
            pass_scales: None,
        }]);
    }
    let generator = mlx_gen::load("z_image_turbo", &spec).unwrap();
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    let out = generator.generate(&req, &mut |_| {}).unwrap();
    match out {
        GenerationOutput::Images(mut v) => v.pop().unwrap().pixels,
        other => panic!("expected Images, got {other:?}"),
    }
}

/// Count RGB8 pixels differing by >8 between two buffers.
fn px_gt8(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() > 8)
        .count()
}

/// The base (no-adapter) render's px>8 vs the fork base golden, at the SAME config as the `kind`
/// adapter golden — the inherited bf16 toolchain drift floor the adapter render sits on. Z-Image is
/// fully bf16, so this floor is inherently higher than the mixed-precision Qwen's. The base golden
/// (`z_image_golden.safetensors`) MUST be dumped at the adapter golden's (seed, steps, size); a
/// mismatch is a hard error (it would yield a bogus floor).
fn base_floor_px(kind: &str) -> usize {
    let ag = Weights::from_file(golden_dir().join(format!("z_image_{kind}_golden.safetensors")))
        .unwrap();
    let bg = Weights::from_file(golden_dir().join("z_image_golden.safetensors"))
        .expect("base golden z_image_golden.safetensors (dump it at the adapter config)");
    for k in ["seed", "steps", "w", "h"] {
        assert_eq!(
            ag.metadata(k),
            bg.metadata(k),
            "base golden {k} != adapter golden {k} — regenerate z_image_golden.safetensors at the adapter config"
        );
    }
    let pixels = render_with_adapter(None, kind);
    let bimg = decoded_to_image(bg.require("decoded").unwrap()).unwrap();
    px_gt8(&pixels, &bimg.pixels)
}

fn assert_matches_golden(kind: &str, my_kind: AdapterKind) {
    let pixels = render_with_adapter(
        Some((&format!("z_image_{kind}_adapter.safetensors"), my_kind, 1.0)),
        kind,
    );
    let g = Weights::from_file(golden_dir().join(format!("z_image_{kind}_golden.safetensors")))
        .unwrap();
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = px_gt8(&pixels, &gimg.pixels);

    // Floor-relative gate (sc-2718): the adapter render must not diverge from the fork by materially
    // more than the BASE render does — the inherited bf16 toolchain drift floor — i.e. the adapter
    // itself adds ~zero divergence (the residual is fork-faithful; scale-0 is bit-exact). The
    // `2×floor + 0.5%` cap allows the stronger-perturbation adapter image's larger content floor
    // while staying FAR tighter than a flat %. Measured @512²: Z-Image base 0.82% / LoRA 0.39% /
    // LoKr 1.13% (Z-Image is fully bf16, so its floor is ~0.8%, unlike mixed-precision Qwen's
    // ~0.05%). (Replaces the old flat 5% guard, sized for the inflated 256² floor — a small latent
    // lets bf16 drift flip a large *fraction* of the few pixels; the floor collapses ~2× at 512²,
    // so the goldens are now dumped at 512².)
    let base = base_floor_px(kind);
    let cap = base * 2 + pixels.len() / 200;
    let pct = |n: usize| n as f64 / pixels.len() as f64 * 100.0;
    println!(
        "✓ {kind} adapter render: {differ} px>8 ({:.4}%); base floor {base} ({:.4}%); cap {cap} ({:.4}%)",
        pct(differ),
        pct(base),
        pct(cap),
    );
    assert!(
        differ <= cap,
        "{kind} adapter render diverges beyond the base floor: {differ} px ({:.3}%) > cap {cap} px (base {base})",
        pct(differ),
    );
}

#[test]
#[ignore = "needs real Z-Image weights + adapter & base goldens (same config)"]
fn lora_render_matches_fork_golden() {
    assert_matches_golden("lora", AdapterKind::Lora);
}

#[test]
#[ignore = "needs real Z-Image weights + adapter & base goldens (same config)"]
fn lokr_render_matches_fork_golden() {
    assert_matches_golden("lokr", AdapterKind::Lokr);
}

/// A scale-0 adapter must be a bit-exact no-op vs the no-adapter render (no regression).
#[test]
#[ignore = "needs real Z-Image weights + adapter golden"]
fn scale_zero_adapter_is_noop() {
    let base = render_with_adapter(None, "lora");
    let zero = render_with_adapter(
        Some(("z_image_lora_adapter.safetensors", AdapterKind::Lora, 0.0)),
        "lora",
    );
    let differ = base.iter().zip(&zero).filter(|(a, b)| a != b).count();
    println!("✓ scale-0 adapter no-op: {differ} px differ from the no-adapter render");
    assert_eq!(differ, 0, "scale-0 adapter must be a bit-exact no-op");
}

/// The single installed LoRA's `(a, b)` arrays, or panic — the equivalence comparison expects
/// exactly one adapter per target.
fn lora_arrays(adapters: &[Adapter]) -> (Array, Array) {
    match adapters {
        [Adapter::Lora { a, b, .. }] => (a.clone(), b.clone()),
        _ => panic!("expected exactly one LoRA adapter, got {}", adapters.len()),
    }
}

/// sc-2618: a kohya `lora_unet_` file resolves the SAME modules and installs the byte-identical
/// adapter as the equivalent PEFT file, on the REAL Z-Image module tree. Also guards against
/// enumerator/matcher drift (every kohya path resolves), kohya-stem collisions, and confirms global
/// targets are excluded from the kohya surface (matching the fork) and surfaced loudly.
#[test]
#[ignore = "needs real Z-Image weights"]
fn kohya_matches_peft_on_real_tree() {
    let none = None as Option<&std::collections::HashMap<String, String>>;
    let mut probe = load_transformer(&snapshot()).unwrap();
    let paths = probe.adaptable_paths();
    assert!(!paths.is_empty(), "no kohya targets enumerated");
    for p in &paths {
        let segs: Vec<&str> = p.split('.').collect();
        assert!(
            AdaptableHost::adaptable_mut(&mut probe, &segs).is_some(),
            "drift: enumerated {p} does not resolve via adaptable_mut"
        );
    }
    let flat: std::collections::BTreeSet<String> =
        paths.iter().map(|p| p.replace('.', "_")).collect();
    assert_eq!(
        flat.len(),
        paths.len(),
        "two paths collide when flattened to a kohya stem"
    );

    // One on-disk spelling per module: a Linear may be reachable under aliases (e.g. FLUX.2's
    // `…to_out` and `…to_out.0`), but a real adapter file uses one. Drop a `.0` alias when its bare
    // sibling is also enumerated (a no-op where there are no aliases, as here for Z-Image).
    let targets: Vec<String> = paths
        .iter()
        .filter(|p| match p.strip_suffix(".0") {
            Some(base) => !paths.iter().any(|q| q.as_str() == base),
            None => true,
        })
        .cloned()
        .collect();

    // Identical (down=A, up=B, alpha) factors expressed in both conventions, sized per module.
    let r = 2i32;
    let mut kohya: Vec<(String, Array)> = Vec::new();
    let mut peft: Vec<(String, Array)> = Vec::new();
    for p in &targets {
        let segs: Vec<&str> = p.split('.').collect();
        let shape = AdaptableHost::adaptable_mut(&mut probe, &segs)
            .unwrap()
            .base_shape();
        let (out, inp) = (shape[0], shape[1]);
        let a = Array::from_slice(
            &(0..r * inp)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.001)
                .collect::<Vec<_>>(),
            &[r, inp],
        );
        let b = Array::from_slice(
            &(0..out * r)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.001)
                .collect::<Vec<_>>(),
            &[out, r],
        );
        let alpha = Array::from_slice(&[4.0f32], &[1]); // ≠ rank=2 → exercises the alpha/rank fold
        let stem = p.replace('.', "_");
        kohya.push((format!("lora_unet_{stem}.lora_down.weight"), a.clone()));
        kohya.push((format!("lora_unet_{stem}.lora_up.weight"), b.clone()));
        kohya.push((format!("lora_unet_{stem}.alpha"), alpha.clone()));
        peft.push((format!("transformer.{p}.lora_A.weight"), a));
        peft.push((format!("transformer.{p}.lora_B.weight"), b));
        peft.push((format!("transformer.{p}.alpha"), alpha));
    }
    let dir = std::env::temp_dir().join("mlx_gen_z_image_kohya_test");
    std::fs::create_dir_all(&dir).unwrap();
    let (kpath, ppath) = (dir.join("kohya.safetensors"), dir.join("peft.safetensors"));
    Array::save_safetensors(
        kohya
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
        none,
        &kpath,
    )
    .unwrap();
    Array::save_safetensors(
        peft.iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
        none,
        &ppath,
    )
    .unwrap();

    let mut tk = load_transformer(&snapshot()).unwrap();
    let rk = apply_z_image_adapters(
        &mut tk,
        &[AdapterSpec {
            path: kpath,
            scale: 0.8,
            kind: AdapterKind::Lora,
            pass_scales: None,
        }],
    )
    .unwrap();
    assert_eq!(rk.applied, targets.len(), "kohya: not all targets applied");
    assert!(
        rk.unmatched_paths.is_empty(),
        "kohya unmatched: {:?}",
        rk.unmatched_paths
    );

    let mut tp = load_transformer(&snapshot()).unwrap();
    apply_z_image_adapters(
        &mut tp,
        &[AdapterSpec {
            path: ppath,
            scale: 0.8,
            kind: AdapterKind::Lora,
            pass_scales: None,
        }],
    )
    .unwrap();

    for p in &targets {
        let segs: Vec<&str> = p.split('.').collect();
        let (ka, kb) = lora_arrays(
            AdaptableHost::adaptable_mut(&mut tk, &segs)
                .unwrap()
                .adapters(),
        );
        let (pa, pb) = lora_arrays(
            AdaptableHost::adaptable_mut(&mut tp, &segs)
                .unwrap()
                .adapters(),
        );
        assert!(
            array_eq(&ka, &pa, false).unwrap().item::<bool>()
                && array_eq(&kb, &pb, false).unwrap().item::<bool>(),
            "kohya and peft installed different adapters at {p}"
        );
    }
    println!(
        "✓ kohya ≡ peft across {} Z-Image modules (byte-identical adapters)",
        targets.len()
    );

    // A kohya key for a GLOBAL target (`t_embedder.mlp.0`, excluded from the kohya surface per the
    // fork) is surfaced and errors — never silently dropped.
    let small = Array::from_slice(&[0.01f32], &[1, 1]);
    let gpath = dir.join("kohya_global.safetensors");
    Array::save_safetensors(
        vec![
            ("lora_unet_t_embedder_mlp_0.lora_down.weight", &small),
            ("lora_unet_t_embedder_mlp_0.lora_up.weight", &small),
        ],
        none,
        &gpath,
    )
    .unwrap();
    let mut tg = load_transformer(&snapshot()).unwrap();
    assert!(
        apply_z_image_adapters(
            &mut tg,
            &[AdapterSpec {
                path: gpath,
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
            }],
        )
        .is_err(),
        "a kohya key for a global target must error (globals are excluded from the kohya surface)"
    );
}
