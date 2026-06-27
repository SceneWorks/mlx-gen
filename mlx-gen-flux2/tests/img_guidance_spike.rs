//! sc-8273 SPIKE harness: render one FLUX.2-klein EDIT image from a real reference + prompt, so an
//! external scorer (the SceneWorks `lora_eval_harness`) can measure ArcFace identity at a given
//! image-guidance scale. The scale itself is read by the engine from `FLUX2_IMG_GUIDANCE` (off / ≤1
//! = the byte-identical baseline). Run once per scale, scoring the saved PNGs out-of-band.
//!
//!   REF_PNG=~/.../anya_sq.png OUT_PNG=/tmp/imgcfg/s2.png FLUX2_IMG_GUIDANCE=2.0 \
//!   PROMPT="a candid photo of a woman hiking on a rocky mountain trail at golden hour, red windbreaker" \
//!   cargo test -p mlx-gen-flux2 --test img_guidance_spike -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::{Conditioning, GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};

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

fn load_ref(path: &str) -> Image {
    let rgb = image::open(path)
        .unwrap_or_else(|e| panic!("open REF_PNG {path}: {e}"))
        .to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Image {
        width,
        height,
        pixels: rgb.into_raw(),
    }
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

/// Load the klein-edit model ONCE, then render the same (reference, prompt, seed) at each scale in
/// `SCALES` (comma list; `off`/`1`/≤1 = baseline, the env var is unset for that arm). Saves
/// `OUT_DIR/s_<scale>.png` per arm so the external ArcFace scorer can diff identity vs scale.
#[test]
#[ignore = "spike: needs the FLUX.2-klein-9b snapshot + REF_PNG/OUT_DIR/PROMPT; SCALES sweeps FLUX2_IMG_GUIDANCE"]
fn flux2_klein_edit_image_guidance_sweep() {
    let ref_png = std::env::var("REF_PNG").expect("REF_PNG");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    // PROMPTS is a `;;`-separated list (falls back to PROMPT); each entry rendered at every SCALE.
    let prompts_raw = std::env::var("PROMPTS")
        .or_else(|_| std::env::var("PROMPT"))
        .expect("PROMPTS or PROMPT");
    let prompts: Vec<&str> = prompts_raw
        .split(";;")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    let size = env_u32("SIZE", 1024);
    let steps = env_u32("STEPS", 4);
    let seed = env_u32("SEED", 12345) as u64;
    let scales = std::env::var("SCALES").unwrap_or_else(|_| "off,1.5,2.0,3.0,4.0".into());

    let image = load_ref(&ref_png);
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    spec.quantize = Some(Quant::Q8);
    // Direct loader (not the registry `load(id)`) so the binary force-links mlx-gen-flux2.
    let gen = mlx_gen_flux2::load_klein_9b_edit(&spec).expect("load klein edit");

    for (pi, prompt) in prompts.iter().enumerate() {
        for raw in scales.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            // Drive the production `image_guidance` REQUEST FIELD (not the env override); `off`/≤1 → None.
            let image_guidance = match raw.parse::<f32>() {
                Ok(s) if s > 1.0 => Some(s),
                _ => None,
            };
            let req = GenerationRequest {
                prompt: (*prompt).to_owned(),
                width: size,
                height: size,
                count: 1,
                seed: Some(seed),
                steps: Some(steps),
                conditioning: vec![Conditioning::Reference {
                    image: image.clone(),
                    strength: None,
                }],
                image_guidance,
                ..Default::default()
            };
            let GenerationOutput::Images(mut images) = gen.generate(&req, &mut |_| {}).unwrap()
            else {
                panic!("expected images");
            };
            let img = images.pop().expect("one image");
            let out_png = format!("{out_dir}/p{pi}_s_{raw}.png");
            image::save_buffer(
                &out_png,
                &img.pixels,
                img.width,
                img.height,
                image::ColorType::Rgb8,
            )
            .unwrap();
            println!("SWEEP p{pi} scale={raw} -> {out_png}  [{prompt}]");
        }
    }
}
