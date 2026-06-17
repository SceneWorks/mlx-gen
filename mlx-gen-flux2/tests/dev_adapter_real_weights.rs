//! sc-5920: FLUX.2-**dev** LoRA / LoKr adapters on real weights. `#[ignore]`d — needs the real
//! `black-forest-labs/FLUX.2-dev` snapshot (~60 GB DiT + ~45 GB TE); assembles a pre-quantized Q4
//! snapshot in TMPDIR (shared with the other dev real-weight tests):
//!
//!   cargo test -p mlx-gen-flux2 --release --test dev_adapter_real_weights -- --ignored --nocapture
//!
//! LoRA/LoKr fall out of the family-agnostic adapter engine (sc-2343) onto the dev `Flux2Transformer`
//! exactly as they did for klein (sc-2646) — the key→module map is config-driven, so the wider/deeper
//! dev graph (8 double + **48** single blocks) is covered automatically (pinned in the crate's
//! `transformer.rs` unit tests). dev has no mflux fork reference (mflux is klein-only), so — like the
//! dev T2I/edit e2e — this is a **behavioral** check, not a bit-parity claim:
//!   (1) the routing map resolves the full dev surface on the REAL module tree (globals + 8×13 +
//!       48×2 + the dev embedded-guidance embedder), and rejects off-surface;
//!   (2) a real-shaped dev LoRA *and* a real-shaped dev LoKr (synthesized at the actual block dims,
//!       spanning early/mid/deep blocks of both block types) load through the public
//!       `load(spec.with_adapters(…)).generate()` path and VISIBLY change the render vs no-adapter;
//!   (3) a scale-0 adapter is a bit-exact no-op.

use std::path::{Path, PathBuf};

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen_flux2::{load_transformer_dev, quantize_flux2_dit, quantize_flux2_text_encoder_dir};
use mlx_rs::Array;

const BITS: i32 = 4;
const GROUP_SIZE: i32 = 64;
const RANK: i32 = 4;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_DEV_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots");
    std::fs::read_dir(&snaps)
        .expect("snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir under models--black-forest-labs--FLUX.2-dev/snapshots")
}

/// Assemble (or reuse) a pre-quantized Q4 dev snapshot at a stable temp path — identical to the dev
/// T2I/edit e2e tests, so a prior run's output is reused (DiT + Mistral TE pre-quantized, VAE +
/// tokenizer symlinked from the source).
fn prequantized_dev_snapshot() -> PathBuf {
    let src = snapshot();
    let dst = std::env::temp_dir().join(format!("mlx_gen_flux2_dev_prequant_q{BITS}"));

    if !dst
        .join("transformer/diffusion_pytorch_model.safetensors")
        .exists()
    {
        println!("pre-quantizing dev DiT → Q{BITS}…");
        quantize_flux2_dit(
            &src.join("transformer"),
            &dst.join("transformer"),
            BITS,
            GROUP_SIZE,
        )
        .expect("pre-quantize dev DiT");
    }
    if !dst.join("text_encoder/model.safetensors").exists() {
        println!("pre-quantizing dev Mistral TE → Q{BITS}…");
        quantize_flux2_text_encoder_dir(
            &src.join("text_encoder"),
            &dst.join("text_encoder"),
            BITS,
            GROUP_SIZE,
        )
        .expect("pre-quantize dev TE");
    }
    for sub in ["vae", "tokenizer"] {
        let link = dst.join(sub);
        if !link.exists() {
            std::os::unix::fs::symlink(std::fs::canonicalize(src.join(sub)).unwrap(), &link)
                .expect("symlink component");
        }
    }
    dst
}

/// A spread of real dev targets across BOTH block types and early/mid/deep indices — enough to make
/// a clearly visible (compounded over the denoise) change, while exercising the wide+deep graph.
fn lora_targets() -> Vec<String> {
    let mut out = Vec::new();
    let double_tgts = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
        "ff.linear_in",
        "ff.linear_out",
        "ff_context.linear_in",
        "ff_context.linear_out",
    ];
    for i in [0usize, 3, 7] {
        for t in double_tgts {
            out.push(format!("transformer_blocks.{i}.{t}"));
        }
    }
    for i in [0usize, 12, 24, 36, 47] {
        for t in ["attn.to_qkv_mlp_proj", "attn.to_out"] {
            out.push(format!("single_transformer_blocks.{i}.{t}"));
        }
    }
    out
}

/// Probe each target's logical `[out, in]` on the (packed) real dev transformer.
fn probe_shapes(targets: &[String]) -> Vec<(String, i32, i32)> {
    let mut probe = load_transformer_dev(&prequantized_dev_snapshot()).unwrap();
    targets
        .iter()
        .map(|p| {
            let segs: Vec<&str> = p.split('.').collect();
            let shape = AdaptableHost::adaptable_mut(&mut probe, &segs)
                .unwrap_or_else(|| panic!("target {p} does not resolve on the real dev tree"))
                .base_shape();
            (p.clone(), shape[0], shape[1])
        })
        .collect()
}

fn det(seed: i32, n: i32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) % 13) as f32 - 6.0) * scale)
        .collect()
}

/// Write a peft-form dev LoRA (`transformer.‹path›.lora_A/lora_B`, `[r,in]`/`[out,r]`, alpha) at the
/// real block dims. Returns the file path.
fn write_lora(shapes: &[(String, i32, i32)]) -> PathBuf {
    let none = None as Option<&std::collections::HashMap<String, String>>;
    let alpha = Array::from_slice(&[RANK as f32], &[1]);
    let mut arrays: Vec<(String, Array)> = Vec::new();
    for (i, (p, out, inp)) in shapes.iter().enumerate() {
        let a = Array::from_slice(&det(i as i32, RANK * inp, 0.01), &[RANK, *inp]);
        let b = Array::from_slice(&det(i as i32 + 7, out * RANK, 0.01), &[*out, RANK]);
        arrays.push((format!("transformer.{p}.lora_A.weight"), a));
        arrays.push((format!("transformer.{p}.lora_B.weight"), b));
        arrays.push((format!("transformer.{p}.alpha"), alpha.clone()));
    }
    let dir = std::env::temp_dir().join("mlx_gen_flux2_dev_adapter");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("dev_lora.safetensors");
    Array::save_safetensors(
        arrays
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
        none,
        &path,
    )
    .unwrap();
    path
}

/// Write a peft-form dev LoKr (bare `‹path›.lokr_w1`/`lokr_w2_a`/`lokr_w2_b`, `networkType=lokr`) at
/// the real block dims: `w1=[1,1]`, low-rank `w2 = w2_a@w2_b = [out,in]` → `kron(w1,w2)=[out,in]`.
fn write_lokr(shapes: &[(String, i32, i32)]) -> PathBuf {
    let mut md = std::collections::HashMap::new();
    md.insert("networkType".to_string(), "lokr".to_string());
    md.insert("rank".to_string(), RANK.to_string());
    md.insert("alpha".to_string(), RANK.to_string());
    let w1 = Array::from_slice(&[1.0f32], &[1, 1]);
    let mut arrays: Vec<(String, Array)> = Vec::new();
    for (i, (p, out, inp)) in shapes.iter().enumerate() {
        let w2a = Array::from_slice(&det(i as i32, out * RANK, 0.02), &[*out, RANK]);
        let w2b = Array::from_slice(&det(i as i32 + 3, RANK * inp, 0.02), &[RANK, *inp]);
        arrays.push((format!("{p}.lokr_w1"), w1.clone()));
        arrays.push((format!("{p}.lokr_w2_a"), w2a));
        arrays.push((format!("{p}.lokr_w2_b"), w2b));
    }
    let dir = std::env::temp_dir().join("mlx_gen_flux2_dev_adapter");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("dev_lokr.safetensors");
    Array::save_safetensors(
        arrays
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
        Some(&md),
        &path,
    )
    .unwrap();
    path
}

fn render(adapter: Option<(&Path, AdapterKind, f32)>, size: u32, steps: u32, seed: u64) -> Vec<u8> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(prequantized_dev_snapshot()));
    if let Some((path, kind, scale)) = adapter {
        spec = spec.with_adapters(vec![AdapterSpec {
            path: path.to_path_buf(),
            scale,
            kind,
            pass_scales: None,
            moe_expert: None,
        }]);
    }
    let gen = mlx_gen::load("flux2_dev", &spec).expect("dev loads through the registry");
    let req = GenerationRequest {
        prompt: "a red fox resting in fresh snow under soft winter light".into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    match gen.generate(&req, &mut |_| {}).expect("dev generate") {
        GenerationOutput::Images(mut v) => v.pop().unwrap().pixels,
        other => panic!("expected Images, got {other:?}"),
    }
}

fn px_gt8(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "image size mismatch");
    let differ = a
        .iter()
        .zip(b)
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() > 8)
        .count();
    differ as f64 / a.len() as f64 * 100.0
}

/// (1) The routing map resolves the FULL dev surface on the REAL packed module tree (globals + 8
/// double × 13 + 48 single × 2 + the dev embedded-guidance embedder), and rejects off-surface.
#[test]
#[ignore = "needs real FLUX.2-dev weights (~105 GB); assembles a Q4 snapshot in TMPDIR"]
fn dev_routing_map_covers_real_tree() {
    let mut t = load_transformer_dev(&prequantized_dev_snapshot()).unwrap();
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
        // The dev embedded-guidance embedder — present on dev, absent on klein (sc-5920).
        "time_guidance_embed.guidance_embedder.linear_1",
        "time_guidance_embed.guidance_embedder.linear_2",
    ] {
        assert!(resolves(&mut t, p), "global {p} should resolve");
    }
    let double_tgts = [
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
        for t2 in double_tgts {
            let p = format!("transformer_blocks.{i}.{t2}");
            assert!(resolves(&mut t, &p), "expected {p} to resolve");
        }
    }
    for i in 0..48 {
        for t2 in ["attn.to_qkv_mlp_proj", "attn.to_out"] {
            let p = format!("single_transformer_blocks.{i}.{t2}");
            assert!(resolves(&mut t, &p), "expected {p} to resolve");
        }
    }
    for p in [
        "transformer_blocks.8.attn.to_q", // dev has 8 double blocks (0..7)
        "single_transformer_blocks.48.attn.to_out", // dev has 48 single blocks (0..47)
        "transformer_blocks.0.attn.add_q", // internal field, not the file's add_q_proj
        "vae.encoder",
    ] {
        assert!(!resolves(&mut t, p), "expected {p} NOT to resolve");
    }
    println!("✓ dev routing covers the full surface (globals + 8×13 + 48×2 + guidance embedder) and rejects off-surface");
}

/// (2) + (3): a real-shaped dev LoRA AND a real-shaped dev LoKr load through the public generate path
/// and VISIBLY change the render vs no-adapter; a scale-0 LoRA is a bit-exact no-op.
#[test]
#[ignore = "needs real FLUX.2-dev weights (~105 GB); assembles a Q4 snapshot in TMPDIR"]
fn dev_lora_and_lokr_visibly_affect_render() {
    let size: u32 = std::env::var("MLX_GEN_FLUX2_DEV_ADAPTER_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let steps: u32 = std::env::var("MLX_GEN_FLUX2_DEV_ADAPTER_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);

    let targets = lora_targets();
    let shapes = probe_shapes(&targets);
    println!(
        "probed {} dev targets across double {{0,3,7}} + single {{0,12,24,36,47}}",
        shapes.len()
    );
    let lora = write_lora(&shapes);
    let lokr = write_lokr(&shapes);

    let base = render(None, size, steps, 0);

    let lora_px = render(Some((&lora, AdapterKind::Lora, 1.0)), size, steps, 0);
    let lora_effect = px_gt8(&lora_px, &base);
    println!("dev LoRA effect vs no-adapter: {lora_effect:.2}% px>8");
    assert!(
        lora_effect > 0.5,
        "dev LoRA had no visible effect ({lora_effect:.2}% px>8) — silently dropped?"
    );

    let lokr_px = render(Some((&lokr, AdapterKind::Lokr, 1.0)), size, steps, 0);
    let lokr_effect = px_gt8(&lokr_px, &base);
    println!("dev LoKr effect vs no-adapter: {lokr_effect:.2}% px>8");
    assert!(
        lokr_effect > 0.5,
        "dev LoKr had no visible effect ({lokr_effect:.2}% px>8) — silently dropped?"
    );

    // A scale-0 LoRA is a bit-exact no-op (the residual is multiplied by 0).
    let zero = render(Some((&lora, AdapterKind::Lora, 0.0)), size, steps, 0);
    let differ = base.iter().zip(&zero).filter(|(a, b)| a != b).count();
    println!("dev scale-0 LoRA no-op: {differ} px differ from the no-adapter render");
    assert_eq!(differ, 0, "scale-0 adapter must be a bit-exact no-op");
}
