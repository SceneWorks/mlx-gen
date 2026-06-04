//! sc-2646: end-to-end FLUX.2-klein-9b LoRA/LoKr adapter consumption against real weights.
//!
//! `#[ignore]`d — needs the real FLUX.2-klein-9b snapshot (env `MLX_GEN_FLUX2_SNAPSHOT` or the HF
//! cache) and the adapter goldens from `tools/dump_flux2_adapter_golden.py` (gitignored, local):
//!   cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_adapter_golden.py
//!   cargo test -p mlx-gen-flux2 --test adapter_real_weights -- --ignored --nocapture
//!
//! Gates: (1) the key→module map resolves the FULL fork `Flux2LoRAMapping` surface (globals + 8
//! double × 12 + 24 single × 2) against the real module tree, and rejects off-surface; (2) the
//! public `load(spec.with_adapters(…)).generate()` render matches the fork's LoRA *and* LoKr golden
//! (px>8, below the cross-build f32 floor — the crate + the golden both run f32); (3) a scale-0
//! adapter is a bit-exact no-op; (4) scale-1 has a visible effect vs the no-adapter render.

use std::path::PathBuf;

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::image::decoded_to_image;
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen_flux2::{apply_flux2_adapters, load_transformer};
use mlx_rs::ops::array_eq;
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

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden")
}

/// (1) The top-level `AdaptableHost` resolves every fork `Flux2LoRAMapping` diffusers target across
/// the real module tree (globals + 8 double blocks × 12 + 24 single blocks × 2), and rejects
/// off-surface paths (out-of-range blocks, klein-absent guidance linears, internal field names).
#[test]
#[ignore = "needs real FLUX.2-klein-9b weights"]
fn routing_map_covers_full_fork_surface() {
    let mut t = load_transformer(&snapshot()).unwrap();
    let resolves = |t: &mut _, p: &str| -> bool {
        let segs: Vec<&str> = p.split('.').collect();
        AdaptableHost::adaptable_mut(t, &segs).is_some()
    };

    for p in [
        "x_embedder",
        "context_embedder",
        "proj_out",
        "norm_out.linear",
        "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.linear",
        "single_stream_modulation.linear",
        "time_guidance_embed.linear_1",
        "time_guidance_embed.linear_2",
    ] {
        assert!(resolves(&mut t, p), "global {p} should resolve");
    }
    let double_targets = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.linear_in",
        "ff.linear_out",
        "ff_context.linear_in",
        "ff_context.linear_out",
    ];
    for i in 0..8 {
        for tgt in double_targets {
            let p = format!("transformer_blocks.{i}.{tgt}");
            assert!(resolves(&mut t, &p), "expected {p} to resolve");
        }
    }
    for i in 0..24 {
        for tgt in ["attn.to_qkv_mlp_proj", "attn.to_out"] {
            let p = format!("single_transformer_blocks.{i}.{tgt}");
            assert!(resolves(&mut t, &p), "expected {p} to resolve");
        }
    }
    for p in [
        "transformer_blocks.8.attn.to_q", // out of range (8 double blocks: 0..7)
        "single_transformer_blocks.24.attn.to_out", // out of range (24 single blocks: 0..23)
        "time_guidance_embed.guidance_linear_1", // klein has no guidance embedding
        "transformer_blocks.0.attn.add_q", // internal field, not the file's add_q_proj
        "transformer_blocks.0.attn.qkv",  // not a FLUX.2 module
        "norm_out_linear",                // internal field name, not the dotted path
    ] {
        assert!(!resolves(&mut t, p), "expected {p} NOT to resolve");
    }
    println!("✓ routing covers the full Flux2LoRAMapping surface (globals + 8×12 + 24×2) and rejects off-surface");
}

fn meta_u32(g: &Weights, k: &str) -> u32 {
    g.metadata(k).unwrap().parse().unwrap()
}

/// Render `flux2_klein_9b` txt2img with an optional adapter, at the golden's config.
fn render(adapter: Option<(&str, AdapterKind, f32)>, golden_kind: &str) -> Vec<u8> {
    let g =
        Weights::from_file(golden_dir().join(format!("flux2_{golden_kind}_golden.safetensors")))
            .unwrap();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let (seed, steps) = (meta_u32(&g, "seed") as u64, meta_u32(&g, "steps"));
    let (w, h) = (meta_u32(&g, "width"), meta_u32(&g, "height"));

    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    if let Some((file, kind, scale)) = adapter {
        spec = spec.with_adapters(vec![AdapterSpec {
            path: golden_dir().join(file),
            scale,
            kind,
            pass_scales: None,
            moe_expert: None,
        }]);
    }
    let generator = mlx_gen::load("flux2_klein_9b", &spec).unwrap();
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    match generator.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.pop().unwrap().pixels,
        other => panic!("expected Images, got {other:?}"),
    }
}

fn px_gt8(a: &[u8], b: &[u8]) -> (usize, f64) {
    assert_eq!(a.len(), b.len(), "image size mismatch");
    let differ = a
        .iter()
        .zip(b)
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() > 8)
        .count();
    (differ, differ as f64 / a.len() as f64 * 100.0)
}

/// (2) + (4): the public `load(adapter).generate()` render matches the fork golden (parity, below
/// the cross-build f32 floor) AND visibly differs from the no-adapter render (the adapter is real).
fn assert_matches_golden(kind: &str, my_kind: AdapterKind) {
    let adapter_file = format!("flux2_{kind}_adapter.safetensors");
    let pixels = render(Some((&adapter_file, my_kind, 1.0)), kind);
    let g =
        Weights::from_file(golden_dir().join(format!("flux2_{kind}_golden.safetensors"))).unwrap();
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let (differ, frac) = px_gt8(&pixels, &gimg.pixels);
    println!(
        "flux2 {kind} adapter render vs fork f32: {differ}/{} px>8 ({frac:.3}%)",
        pixels.len()
    );
    assert!(
        frac < 5.0,
        "flux2 {kind} adapter render diverges from the fork: {frac:.3}% px>8 (cross-build f32 floor is ~0.9%)"
    );

    // The adapter must actually change the image (guards a silently-dropped/no-op application).
    let base = render(None, kind);
    let (_, effect) = px_gt8(&pixels, &base);
    println!("flux2 {kind} adapter effect vs no-adapter: {effect:.2}% px>8");
    assert!(
        effect > 3.0,
        "flux2 {kind} adapter had no visible effect ({effect:.2}% px>8) — silently dropped?"
    );
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b weights + adapter golden"]
fn lora_render_matches_fork_golden() {
    assert_matches_golden("lora", AdapterKind::Lora);
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b weights + adapter golden"]
fn lokr_render_matches_fork_golden() {
    assert_matches_golden("lokr", AdapterKind::Lokr);
}

/// (3) A scale-0 adapter is a bit-exact no-op vs the no-adapter render.
#[test]
#[ignore = "needs real FLUX.2-klein-9b weights + adapter golden"]
fn scale_zero_adapter_is_noop() {
    let base = render(None, "lora");
    let zero = render(
        Some(("flux2_lora_adapter.safetensors", AdapterKind::Lora, 0.0)),
        "lora",
    );
    let differ = base.iter().zip(&zero).filter(|(a, b)| a != b).count();
    println!("flux2 scale-0 adapter no-op: {differ} px differ from the no-adapter render");
    assert_eq!(differ, 0, "scale-0 adapter must be a bit-exact no-op");
}

/// The single installed LoRA's `(a, b)` arrays, or panic.
fn lora_arrays(adapters: &[Adapter]) -> (Array, Array) {
    match adapters {
        [Adapter::Lora { a, b, .. }] => (a.clone(), b.clone()),
        _ => panic!("expected exactly one LoRA adapter, got {}", adapters.len()),
    }
}

/// sc-2618: a kohya `lora_unet_` file resolves the SAME diffusers-named modules and installs the
/// byte-identical adapter as the equivalent PEFT file, on the REAL FLUX.2-klein tree. Drift guard +
/// collision-free flattening + a BFL fused-qkv key (sc-2743) errors loudly.
#[test]
#[ignore = "needs real FLUX.2-klein-9b weights"]
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

    // One on-disk spelling per module: FLUX.2 exposes the `…attn.to_out` Linear under both
    // `to_out` and the HF `to_out.0` alias, but a real kohya file uses one. Drop the `.0` alias when
    // its bare sibling is also enumerated.
    let targets: Vec<String> = paths
        .iter()
        .filter(|p| match p.strip_suffix(".0") {
            Some(base) => !paths.iter().any(|q| q.as_str() == base),
            None => true,
        })
        .cloned()
        .collect();
    assert!(
        targets.len() < paths.len(),
        "the FLUX.2 to_out.0 alias should be deduped"
    );

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
        let alpha = Array::from_slice(&[4.0f32], &[1]);
        let stem = p.replace('.', "_");
        kohya.push((format!("lora_unet_{stem}.lora_down.weight"), a.clone()));
        kohya.push((format!("lora_unet_{stem}.lora_up.weight"), b.clone()));
        kohya.push((format!("lora_unet_{stem}.alpha"), alpha.clone()));
        peft.push((format!("transformer.{p}.lora_A.weight"), a));
        peft.push((format!("transformer.{p}.lora_B.weight"), b));
        peft.push((format!("transformer.{p}.alpha"), alpha));
    }
    let dir = std::env::temp_dir().join("mlx_gen_flux2_kohya_rw_test");
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
    let rk = apply_flux2_adapters(
        &mut tk,
        &[AdapterSpec {
            path: kpath,
            scale: 0.8,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
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
    apply_flux2_adapters(
        &mut tp,
        &[AdapterSpec {
            path: ppath,
            scale: 0.8,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
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
        "✓ kohya ≡ peft across {} FLUX.2 modules (byte-identical adapters)",
        targets.len()
    );
    // (BFL fused→split is now supported — see `bfl_resolves_and_matches_diffusers_split_on_real_tree`.)
}

/// sc-2743: the FULL BFL / ComfyUI surface resolves on the real FLUX.2-klein tree, and a BFL *fused*
/// qkv LoRA reconstructs the BYTE-IDENTICAL `to_q/to_k/to_v` adapters as the equivalent diffusers
/// split-target LoRA — proven for the `lora_unet_` AND the `base_model.model.` spellings (so all three
/// prefix conventions agree). The diffusers path is fork-verified (sc-2646), so this transitively
/// matches the fork's `LoRALoader` + `Flux2LoRAMapping._get_bfl_*` byte-for-byte.
#[test]
#[ignore = "needs real FLUX.2-klein-9b weights"]
fn bfl_resolves_and_matches_diffusers_split_on_real_tree() {
    let none = None as Option<&std::collections::HashMap<String, String>>;

    // (1) Full BFL surface resolves + count: 8 globals + 8 double×12 + 24 single×2 = 152.
    let mut probe = load_transformer(&snapshot()).unwrap();
    let targets = probe.bfl_targets();
    assert_eq!(targets.len(), 152, "full BFL target count (8 + 96 + 48)");
    for tg in &targets {
        let segs: Vec<&str> = tg.target_path.split('.').collect();
        assert!(
            AdaptableHost::adaptable_mut(&mut probe, &segs).is_some(),
            "BFL target {} does not resolve on the real tree",
            tg.target_path
        );
    }

    // (2) Reconstruct-equivalence for a fused img-qkv (block 0) at REAL shapes: fused up [3·inner, r],
    // shared down [r, in]. Build the fused up as the dim-0 concat of three per-head [inner, r] blocks
    // so the diffusers split file can use those blocks directly.
    let q_shape =
        AdaptableHost::adaptable_mut(&mut probe, &["transformer_blocks", "0", "attn", "to_q"])
            .unwrap()
            .base_shape();
    let (inner, inp) = (q_shape[0], q_shape[1]);
    let r = 2i32;
    let head = |seed: i32| -> Vec<f32> {
        (0..inner * r)
            .map(|i| (((i + seed) % 17) as f32 - 8.0) * 0.001)
            .collect()
    };
    let (hq, hk, hv) = (head(0), head(5), head(11));
    let mut fused = hq.clone();
    fused.extend_from_slice(&hk);
    fused.extend_from_slice(&hv);
    let b_fused = Array::from_slice(&fused, &[3 * inner, r]);
    let b_q = Array::from_slice(&hq, &[inner, r]);
    let b_k = Array::from_slice(&hk, &[inner, r]);
    let b_v = Array::from_slice(&hv, &[inner, r]);
    let a = Array::from_slice(
        &(0..r * inp)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.002)
            .collect::<Vec<_>>(),
        &[r, inp],
    );
    let alpha = Array::from_slice(&[4.0f32], &[1]);

    let dir = std::env::temp_dir().join("mlx_gen_flux2_bfl_rw_test");
    std::fs::create_dir_all(&dir).unwrap();

    // Equivalent diffusers split-target file (per-head up, SHARED down, same alpha).
    let ppath = dir.join("bfl_split_peft.safetensors");
    Array::save_safetensors(
        vec![
            (
                "transformer.transformer_blocks.0.attn.to_q.lora_B.weight",
                &b_q,
            ),
            (
                "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
                &a,
            ),
            ("transformer.transformer_blocks.0.attn.to_q.alpha", &alpha),
            (
                "transformer.transformer_blocks.0.attn.to_k.lora_B.weight",
                &b_k,
            ),
            (
                "transformer.transformer_blocks.0.attn.to_k.lora_A.weight",
                &a,
            ),
            ("transformer.transformer_blocks.0.attn.to_k.alpha", &alpha),
            (
                "transformer.transformer_blocks.0.attn.to_v.lora_B.weight",
                &b_v,
            ),
            (
                "transformer.transformer_blocks.0.attn.to_v.lora_A.weight",
                &a,
            ),
            ("transformer.transformer_blocks.0.attn.to_v.alpha", &alpha),
        ],
        none,
        &ppath,
    )
    .unwrap();
    let mut tp = load_transformer(&snapshot()).unwrap();
    apply_flux2_adapters(
        &mut tp,
        &[AdapterSpec {
            path: ppath,
            scale: 0.8,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .unwrap();

    // The same fused qkv in BOTH the kohya and the base_model.model. spellings must agree with it.
    for (label, up_key, down_key, alpha_key) in [
        (
            "lora_unet_",
            "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight",
            "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight",
            "lora_unet_double_blocks_0_img_attn_qkv.alpha",
        ),
        (
            "base_model.model.",
            "base_model.model.double_blocks.0.img_attn.qkv.lora_B.weight",
            "base_model.model.double_blocks.0.img_attn.qkv.lora_A.weight",
            "base_model.model.double_blocks.0.img_attn.qkv.alpha",
        ),
    ] {
        let bpath = dir.join(format!("bfl_qkv_{}.safetensors", label.replace('.', "_")));
        Array::save_safetensors(
            vec![(up_key, &b_fused), (down_key, &a), (alpha_key, &alpha)],
            none,
            &bpath,
        )
        .unwrap();
        let mut tb = load_transformer(&snapshot()).unwrap();
        let rb = apply_flux2_adapters(
            &mut tb,
            &[AdapterSpec {
                path: bpath,
                scale: 0.8,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();
        assert_eq!(rb.applied, 3, "{label}: fused qkv → 3 split targets");
        assert!(
            rb.unmatched_paths.is_empty(),
            "{label}: unmatched {:?}",
            rb.unmatched_paths
        );

        for tgt in ["to_q", "to_k", "to_v"] {
            let segs = ["transformer_blocks", "0", "attn", tgt];
            let (ba, bb) = lora_arrays(
                AdaptableHost::adaptable_mut(&mut tb, &segs)
                    .unwrap()
                    .adapters(),
            );
            let (pa, pb) = lora_arrays(
                AdaptableHost::adaptable_mut(&mut tp, &segs)
                    .unwrap()
                    .adapters(),
            );
            assert!(
                array_eq(&ba, &pa, false).unwrap().item::<bool>()
                    && array_eq(&bb, &pb, false).unwrap().item::<bool>(),
                "{label}: BFL split ≠ diffusers split at {tgt}"
            );
        }
    }
    println!("✓ BFL fused-qkv ≡ diffusers split on the real FLUX.2 tree (lora_unet_ + base_model.model.)");
}
