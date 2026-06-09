//! Kolors dispatch-path validation (sc-3874) — exercises the model **through the engine registry**
//! (`mlx_gen::load("kolors", spec).generate(req)`), the SceneWorks worker's in-process entry, rather
//! than the `Kolors` struct API the per-mode parity tests use. Proves the `LoadSpec` → `load` and
//! `GenerationRequest` → `generate` mapping (incl. the count loop + per-conditioning routing) for
//! every wired mode. The per-mode numeric parity is already covered by the dedicated `*_parity`
//! tests; here the gate is "the dispatch path runs each mode and renders coherently."
//!
//! `#[ignore]`d: needs the Kolors snapshot (+ tokenizer.json) and, for the control/IP tests, the
//! Kolors-ControlNet-Pose / Kolors-IP-Adapter-Plus snapshots.
//!
//! Run: `cargo test -p mlx-gen-kolors --release --test registry_parity -- --ignored --nocapture`

use std::path::PathBuf;

// Force-link the provider crate so its `inventory::submit!` registration is included in this test
// binary. Without a reference to *some* symbol of `mlx-gen-kolors`, the linker dead-strips the whole
// crate and `mlx_gen::load("kolors", …)` finds no registration. The same applies to the SceneWorks
// worker — the consumer must `use mlx_gen_kolors as _;` (or otherwise reference it) to register it.
use mlx_gen_kolors as _;

use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, Precision,
    Progress, Quant, WeightsSource,
};

fn snap(env: &str, repo: &str) -> PathBuf {
    if let Ok(p) = std::env::var(env) {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let base = PathBuf::from(home).join(format!(".cache/huggingface/hub/{repo}/snapshots"));
    std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("snapshots dir for {repo}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn kolors_snap() -> PathBuf {
    snap("KOLORS_SNAPSHOT", "models--Kwai-Kolors--Kolors-diffusers")
}
fn cn_snap() -> PathBuf {
    snap(
        "KOLORS_CONTROLNET",
        "models--Kwai-Kolors--Kolors-ControlNet-Pose",
    )
}
fn ip_snap() -> PathBuf {
    snap(
        "KOLORS_IP_ADAPTER",
        "models--Kwai-Kolors--Kolors-IP-Adapter-Plus",
    )
}

fn base_spec() -> LoadSpec {
    LoadSpec {
        weights: WeightsSource::Dir(kolors_snap()),
        quantize: None,
        precision: Precision::Bf16,
        control: None,
        ip_adapter: None,
        adapters: Vec::new(),
        extra_controls: Vec::new(),
    }
}

/// A deterministic 512² test image (the engine never sees a real photo in these gates).
fn test_image() -> Image {
    let (h, w) = (512usize, 512usize);
    let mut px = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            px[i] = (x * 255 / (w - 1)) as u8;
            px[i + 1] = (y * 255 / (h - 1)) as u8;
            px[i + 2] = ((x ^ y) % 256) as u8;
        }
    }
    Image {
        width: w as u32,
        height: h as u32,
        pixels: px,
    }
}

fn t2i_req() -> GenerationRequest {
    GenerationRequest {
        prompt: "a cat playing a grand piano on a city rooftop at sunset".into(),
        negative_prompt: Some("blurry, low quality".into()),
        width: 512,
        height: 512,
        count: 1,
        steps: Some(6),
        guidance: Some(5.0),
        seed: Some(0),
        ..Default::default()
    }
}

fn assert_coherent(out: GenerationOutput, expect: usize) {
    let imgs = match out {
        GenerationOutput::Images(v) => v,
        _ => panic!("expected Images"),
    };
    assert_eq!(imgs.len(), expect, "image count");
    for img in &imgs {
        assert_eq!(img.pixels.len(), (img.width * img.height * 3) as usize);
        assert!(
            img.pixels.iter().any(|&p| p > 16) && img.pixels.iter().any(|&p| p < 239),
            "degenerate render"
        );
    }
}

fn run(spec: &LoadSpec, req: &GenerationRequest) -> GenerationOutput {
    let gen = mlx_gen::load("kolors", spec).expect("registry load");
    gen.generate(req, &mut |_p: Progress| {}).expect("generate")
}

#[test]
#[ignore = "needs the Kolors snapshot + tokenizer.json"]
fn registry_t2i_and_count() {
    // T2I via the registry + the count loop (2 images, distinct seeds).
    let mut req = t2i_req();
    req.count = 2;
    assert_coherent(run(&base_spec(), &req), 2);
    println!("✓ mlx_gen::load(\"kolors\").generate T2I (count=2) renders coherently");
}

#[test]
#[ignore = "needs the Kolors snapshot + tokenizer.json"]
fn registry_img2img() {
    let mut req = t2i_req();
    req.conditioning = vec![Conditioning::Reference {
        image: test_image(),
        strength: Some(0.6),
    }];
    assert_coherent(run(&base_spec(), &req), 1);
    println!("✓ registry img2img (Reference) renders coherently");
}

#[test]
#[ignore = "needs the Kolors snapshot + tokenizer.json"]
fn registry_quant_q8() {
    let mut spec = base_spec();
    spec.quantize = Some(Quant::Q8);
    assert_coherent(run(&spec, &t2i_req()), 1);
    println!("✓ registry Q8 load + T2I renders coherently");
}

#[test]
#[ignore = "needs the Kolors + Kolors-ControlNet-Pose snapshots"]
fn registry_controlnet() {
    let mut spec = base_spec();
    spec.control = Some(WeightsSource::Dir(cn_snap()));
    let mut req = t2i_req();
    req.conditioning = vec![Conditioning::Control {
        image: test_image(),
        kind: ControlKind::Pose,
        scale: 0.7,
    }];
    assert_coherent(run(&spec, &req), 1);
    println!("✓ registry ControlNet (Control/Pose) renders coherently");
}

#[test]
#[ignore = "needs the Kolors + Kolors-IP-Adapter-Plus snapshots"]
fn registry_ip_adapter() {
    let mut spec = base_spec();
    spec.ip_adapter = Some(WeightsSource::Dir(ip_snap()));
    let mut req = t2i_req();
    // In IP mode the Reference is the image prompt (not an img2img init).
    req.conditioning = vec![Conditioning::Reference {
        image: test_image(),
        strength: Some(0.7),
    }];
    assert_coherent(run(&spec, &req), 1);
    println!("✓ registry IP-Adapter (Reference = image prompt) renders coherently");
}
