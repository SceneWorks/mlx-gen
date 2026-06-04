//! sc-2657: end-to-end FLUX.1-dev LoRA/LoKr adapter consumption against real weights.
//!
//! `#[ignore]`d — needs the real `black-forest-labs/FLUX.1-dev` snapshot (env `MLX_GEN_FLUX_DEV_SNAPSHOT`
//! or the HF cache), the real LoRA (`~/repos/test-files/zhibi_flux.safetensors`, env `FLUX_LORA`), and
//! the goldens from `tools/dump_flux_lora_golden.py` (gitignored, local):
//!   cd ~/Repos/mflux && .venv-0312/bin/python ~/Repos/mlx-gen/tools/dump_flux_lora_golden.py
//!   cargo test -p mlx-gen-flux --test adapter_real_weights -- --ignored --nocapture
//!
//! Gates: (1) the key→module map resolves the FULL fork `FluxLoRAMapping` surface (joint + single
//! blocks incl. the adaLN modulation linears) on the real module tree, and rejects off-surface; (2) the
//! real zhibi BFL/kohya file resolves the WHOLE surface with zero unmatched keys (= 494 targets); (3)
//! the public `load(spec.with_adapters(…)).generate()` render matches the fork's LoRA *and* LoKr golden
//! (px>8 ≤ the measured base floor — adapter-only divergence — per sc-2528/sc-2602); (4) a scale-0
//! adapter is a bit-exact no-op; (5) scale-1 has a visible effect vs the no-adapter render.

use std::path::PathBuf;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::image::decoded_to_image;
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen_flux::{apply_flux_adapters, FluxVariant};

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX_DEV_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots");
    std::fs::read_dir(&snaps)
        .expect("FLUX.1-dev snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden")
}

fn lora_file() -> PathBuf {
    std::env::var("FLUX_LORA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME"))
                .join("repos/test-files/zhibi_flux.safetensors")
        })
}

fn golden() -> Weights {
    Weights::from_file(golden_dir().join("flux1_dev_adapter_golden.safetensors"))
        .expect("run tools/dump_flux_lora_golden.py first")
}

fn meta_u32(g: &Weights, k: &str) -> u32 {
    g.metadata(k).unwrap().parse().unwrap()
}

/// Render `flux1_dev` txt2img with an optional adapter, at the golden's config.
fn render(adapter: Option<(PathBuf, AdapterKind, f32)>) -> Vec<u8> {
    let g = golden();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let (seed, steps) = (meta_u32(&g, "seed") as u64, meta_u32(&g, "steps"));
    let (w, h) = (meta_u32(&g, "width"), meta_u32(&g, "height"));
    let guidance: f32 = g.metadata("guidance").unwrap().parse().unwrap();

    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    if let Some((path, kind, scale)) = adapter {
        spec = spec.with_adapters(vec![AdapterSpec {
            path,
            scale,
            kind,
            pass_scales: None,
            moe_expert: None,
        }]);
    }
    let generator = mlx_gen::load("flux1_dev", &spec).unwrap();
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        guidance: Some(guidance),
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

fn golden_pixels(key: &str) -> Vec<u8> {
    decoded_to_image(golden().require(key).unwrap())
        .unwrap()
        .pixels
}

/// (1) The full fork `FluxLoRAMapping` surface resolves on the REAL FLUX.1-dev module tree (joint +
/// single block linears incl. the adaLN modulation linears), and off-surface paths reject.
#[test]
#[ignore = "needs real FLUX.1-dev weights"]
fn routing_map_covers_full_fork_surface() {
    let mut t = mlx_gen_flux::load_transformer(&snapshot(), FluxVariant::Dev).unwrap();
    let resolves = |t: &mut _, p: &str| -> bool {
        let segs: Vec<&str> = p.split('.').collect();
        AdaptableHost::adaptable_mut(t, &segs).is_some()
    };
    let double = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
        "norm1.linear",
        "norm1_context.linear",
    ];
    for i in 0..19 {
        for tgt in double {
            let p = format!("transformer_blocks.{i}.{tgt}");
            assert!(resolves(&mut t, &p), "expected {p} to resolve");
        }
    }
    for i in 0..38 {
        for tgt in [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "proj_mlp",
            "proj_out",
            "norm.linear",
        ] {
            let p = format!("single_transformer_blocks.{i}.{tgt}");
            assert!(resolves(&mut t, &p), "expected {p} to resolve");
        }
    }
    for p in [
        "x_embedder",
        "context_embedder",
        "norm_out.linear",
        "proj_out",
        "transformer_blocks.19.attn.to_q",
        "single_transformer_blocks.38.proj_out",
        "transformer_blocks.0.attn.add_q",
    ] {
        assert!(!resolves(&mut t, p), "expected {p} NOT to resolve");
    }
    println!(
        "✓ routing covers the full FluxLoRAMapping surface (19×14 + 38×6) and rejects off-surface"
    );
}

/// (2) The real zhibi BFL/kohya LoRA resolves the WHOLE surface with ZERO unmatched keys — every fork
/// `lora_unet_` source (incl. `*_mod_lin`/`modulation_lin` and the fused qkv / 4-way linear1) lands on
/// a distinct target. This is the structural proof the BFL table is complete (494 targets).
#[test]
#[ignore = "needs real FLUX.1-dev weights + the zhibi LoRA"]
fn real_zhibi_resolves_full_surface_no_unmatched() {
    let mut t = mlx_gen_flux::load_transformer(&snapshot(), FluxVariant::Dev).unwrap();
    let report = apply_flux_adapters(
        &mut t,
        &[AdapterSpec {
            path: lora_file(),
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .unwrap();
    assert_eq!(
        report.applied,
        19 * 14 + 38 * 6,
        "zhibi BFL file should fan out onto the full 494-target surface"
    );
    assert!(
        report.unmatched_paths.is_empty(),
        "zhibi left unmatched keys: {:?}",
        report.unmatched_paths
    );
    println!("✓ real zhibi BFL LoRA resolves all 494 targets, zero unmatched (bf16 alpha read OK)");
}

/// Shared (2)+(5) body: the public load(adapter).generate() render matches the fork golden within the
/// base floor AND visibly differs from the no-adapter render.
fn assert_matches_golden(kind: &str, adapter: (PathBuf, AdapterKind, f32)) {
    let base = render(None);
    let (_, base_floor) = px_gt8(&base, &golden_pixels("base_decoded"));
    println!("flux1 base floor (no adapter) vs fork: {base_floor:.3}% px>8");

    let pixels = render(Some(adapter));
    let (differ, frac) = px_gt8(&pixels, &golden_pixels(&format!("{kind}_decoded")));
    println!(
        "flux1 {kind} render vs fork: {differ}/{} px>8 ({frac:.3}%)",
        pixels.len()
    );
    // Adapter-only divergence: at or below the base floor (+ a small absolute margin for the extra
    // residual ops), i.e. the adapter itself contributes ~zero net divergence (sc-2528/sc-2602).
    assert!(
        frac <= base_floor + 0.5,
        "flux1 {kind} render diverges beyond the base floor: {frac:.3}% px>8 (floor {base_floor:.3}%)"
    );

    let (_, effect) = px_gt8(&pixels, &base);
    println!("flux1 {kind} effect vs no-adapter: {effect:.2}% px>8");
    assert!(
        effect > 3.0,
        "flux1 {kind} adapter had no visible effect ({effect:.2}% px>8) — silently dropped?"
    );
}

#[test]
#[ignore = "needs real FLUX.1-dev weights + the zhibi LoRA + golden"]
fn lora_render_matches_fork_golden() {
    assert_matches_golden("lora", (lora_file(), AdapterKind::Lora, 1.0));
}

#[test]
#[ignore = "needs real FLUX.1-dev weights + the LoKr golden"]
fn lokr_render_matches_fork_golden() {
    let scale: f32 = golden().metadata("lokr_scale").unwrap().parse().unwrap();
    let lokr = golden_dir().join("flux1_dev_lokr_adapter.safetensors");
    assert_matches_golden("lokr", (lokr, AdapterKind::Lokr, scale));
}

/// (4) A scale-0 adapter is a bit-exact no-op vs the no-adapter render.
#[test]
#[ignore = "needs real FLUX.1-dev weights + the zhibi LoRA"]
fn scale_zero_adapter_is_noop() {
    let base = render(None);
    let zero = render(Some((lora_file(), AdapterKind::Lora, 0.0)));
    let differ = base.iter().zip(&zero).filter(|(a, b)| a != b).count();
    println!("flux1 scale-0 adapter no-op: {differ} px differ from the no-adapter render");
    assert_eq!(differ, 0, "scale-0 adapter must be a bit-exact no-op");
}
