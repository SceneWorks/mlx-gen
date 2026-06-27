//! epic 3401 / sc-8267: real-weights validation of the Qwen-Image **2512-Fun-Controlnet-Union**
//! (strict pose) port — the VACE-style alibaba-pai control branch that **replaces** the retired
//! InstantX `Qwen-Image-ControlNet-Union` shape.
//!
//! `#[ignore]`d — needs the real `Qwen/Qwen-Image-2512` base snapshot (env `QWEN_IMAGE_SNAPSHOT`,
//! else the HF cache) and the alibaba-pai `Qwen-Image-2512-Fun-Controlnet-Union` checkpoint (env
//! `QWEN_CONTROL_WEIGHTS`, else the HF cache — the `Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors`).
//! Gates, smallest-footprint first:
//!  - **control load + forward** (`control_loads_and_emits_hints`): loads ONLY the control branch,
//!    drives it through the base `forward_control` (the branch computes its hints inline), and
//!    asserts a finite, non-degenerate output of the right shape. Validates the loader (sc-8267) +
//!    the VACE control forward cheaply (still needs the base for the inline injection).
//!  - **scale-0 self-consistency** (`scale_zero_matches_base`): asserts
//!    `forward_control(branch, scale = 0)` is **bit-identical** to the plain `forward` — proving the
//!    VACE injection seam is inert at scale 0 (the zero-init `after_proj` + `+0` injection) and the
//!    base parity path is untouched.
//!  - **scale-1 changes output** (`scale_one_changes_output`): with the real branch at scale 1 the
//!    output differs from base — the pose actually takes effect.
//!  - **public pose generate** (`public_generate_runs`): end-to-end smoke of the public
//!    `qwen_image_control` API on a synthetic pose skeleton (encode prompt, build the 132-ch control
//!    context, run the control denoise loop, decode → a valid non-degenerate image).
//!
//! A numeric residual/image golden vs the fork's `pipeline_qwenimage_control` (a
//! `dump_qwen_fun_control_*.py` analog of the InstantX dump tooling) is a follow-up (tracked on
//! sc-8267); this suite proves the loader/forward/injection seam + pose effect end-to-end.
//!
//! Run (the scale gates load the ~40 GB base transformer):
//!   cargo test -p mlx-gen-qwen-image --release --test control_real_weights -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress,
    WeightsSource,
};
use mlx_gen_qwen_image::loader;
use mlx_rs::{random, Array, Dtype};

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
const TXT_SEQ: i32 = 64;
/// Packed control-context channels (`control_in_dim`): `[control_latents(16) | mask(1) | inpaint(16)]`
/// × 2×2 patch = 132.
const CONTROL_IN_DIM: i32 = 132;

/// Base `Qwen/Qwen-Image-2512` snapshot dir (env override, else the HF cache).
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_IMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    // Prefer the 2512 base (the sc-8271 default); fall back to the legacy Qwen-Image snapshot dir.
    for repo in ["models--Qwen--Qwen-Image-2512", "models--Qwen--Qwen-Image"] {
        let snaps = PathBuf::from(&home)
            .join(".cache/huggingface/hub")
            .join(repo)
            .join("snapshots");
        if let Ok(rd) = std::fs::read_dir(&snaps) {
            if let Some(p) = rd
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.is_dir())
            {
                return p;
            }
        }
    }
    panic!("no Qwen-Image base snapshot in the HF cache (set QWEN_IMAGE_SNAPSHOT)");
}

/// alibaba-pai `Qwen-Image-2512-Fun-Controlnet-Union` checkpoint (env `QWEN_CONTROL_WEIGHTS`, else
/// the HF cache — the `Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors`).
fn control_source() -> WeightsSource {
    if let Ok(p) = std::env::var("QWEN_CONTROL_WEIGHTS") {
        return WeightsSource::File(PathBuf::from(p));
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home).join(
        ".cache/huggingface/hub/models--alibaba-pai--Qwen-Image-2512-Fun-Controlnet-Union/snapshots",
    );
    let file = std::fs::read_dir(&snaps)
        .expect("control HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .flat_map(|d| {
            std::fs::read_dir(d)
                .unwrap()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
        })
        // Prefer the -2602 distilled checkpoint when present, else any control .safetensors.
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .min_by_key(|p| {
            // -2602 sorts first (a 0 key); otherwise 1.
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.contains("2602") {
                0
            } else {
                1
            }
        })
        .expect("a control .safetensors");
    WeightsSource::File(file)
}

fn randn(shape: &[i32], seed: u64) -> Array {
    let k = random::key(seed).unwrap();
    random::normal::<f32>(shape, None, None, Some(&k)).unwrap()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
    mlx_rs::ops::max(&d, None)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .item::<f32>()
}

fn max_abs(a: &Array) -> f32 {
    let abs = mlx_rs::ops::abs(a).unwrap();
    mlx_rs::ops::max(&abs, None)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .item::<f32>()
}

/// The latent grid + packed-token sequence for the test geometry.
fn geom() -> (usize, usize, i32) {
    let (lh, lw) = ((HEIGHT / 16) as usize, (WIDTH / 16) as usize);
    (lh, lw, (lh * lw) as i32)
}

#[test]
#[ignore = "needs the base Qwen snapshot (~40 GB) + the 2512-Fun control checkpoint in the HF cache"]
fn control_loads_and_emits_hints() {
    let (lh, lw, seq) = geom();
    let cn = loader::load_controlnet(&control_source()).expect("load control branch");
    assert_eq!(cn.num_hints(), 5, "2512-Fun Union ships 5 control layers");

    // The VACE branch computes its hints inline inside `base.forward_control`, so drive it through
    // the base (also exercises the injection seam). control_context is the packed 132-ch tensor.
    let base = loader::load_transformer(&snapshot()).expect("load base transformer");
    let latents = randn(&[1, seq, 64], 1);
    let control_ctx = randn(&[1, seq, CONTROL_IN_DIM], 2);
    let embeds = randn(&[1, TXT_SEQ, 3584], 3)
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

    let out = base
        .forward_control(
            &latents,
            &embeds,
            None,
            0.7,
            lh,
            lw,
            &[],
            Some((&cn, &control_ctx)),
            0.8,
        )
        .expect("control forward");
    assert_eq!(out.shape(), &[1, seq, 64], "velocity shape");
    let m = max_abs(&out);
    assert!(
        m.is_finite() && m > 0.0,
        "output must be finite + non-zero, got {m}"
    );
}

#[test]
#[ignore = "needs the base Qwen-Image snapshot (~40 GB) + the control checkpoint"]
fn scale_zero_matches_base() {
    let (lh, lw, seq) = geom();
    let base = loader::load_transformer(&snapshot()).expect("load base transformer");
    let cn = loader::load_controlnet(&control_source()).expect("load control branch");

    let latents = randn(&[1, seq, 64], 10);
    let control_ctx = randn(&[1, seq, CONTROL_IN_DIM], 11);
    let embeds = randn(&[1, TXT_SEQ, 3584], 12)
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let sigma = 0.7;

    let base_out: Array = base
        .forward(&latents, &embeds, None, sigma, lh, lw, &[])
        .expect("base forward");
    let ctrl_out = base
        .forward_control(
            &latents,
            &embeds,
            None,
            sigma,
            lh,
            lw,
            &[],
            Some((&cn, &control_ctx)),
            0.0,
        )
        .expect("control forward scale 0");

    // scale 0 ⇒ `hidden + hint*0 == hidden`: bit-identical to the base T2I forward.
    assert_eq!(
        max_abs_diff(&base_out, &ctrl_out),
        0.0,
        "control scale 0 must be bit-identical to base forward"
    );
}

#[test]
#[ignore = "needs the base Qwen-Image snapshot (~40 GB) + the control checkpoint"]
fn scale_one_changes_output() {
    let (lh, lw, seq) = geom();
    let base = loader::load_transformer(&snapshot()).expect("load base transformer");
    let cn = loader::load_controlnet(&control_source()).expect("load control branch");

    let latents = randn(&[1, seq, 64], 20);
    let control_ctx = randn(&[1, seq, CONTROL_IN_DIM], 21);
    let embeds = randn(&[1, TXT_SEQ, 3584], 22)
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let sigma = 0.7;

    let base_out: Array = base
        .forward(&latents, &embeds, None, sigma, lh, lw, &[])
        .expect("base forward");
    let ctrl_out = base
        .forward_control(
            &latents,
            &embeds,
            None,
            sigma,
            lh,
            lw,
            &[],
            Some((&cn, &control_ctx)),
            1.0,
        )
        .expect("control forward scale 1");

    assert!(
        max_abs_diff(&base_out, &ctrl_out) > 0.0,
        "control scale 1 must change the output vs base (pose takes effect)"
    );
}

#[test]
#[ignore = "needs the base Qwen snapshot (~40 GB) + control checkpoint + text encoder (~14 GB)"]
fn public_generate_runs() {
    // End-to-end smoke of the public `qwen_image_control` API (sc-8267): encode prompt, VAE-encode +
    // build the 132-ch control context from the (synthetic) skeleton, run the control denoise loop,
    // decode. No golden — asserts a valid, non-degenerate image at the requested size. 2 steps to
    // keep it runnable.
    let (w, h) = (512u32, 512u32);
    let skeleton = Image {
        width: w,
        height: h,
        pixels: (0..(w * h * 3)).map(|i| (i % 256) as u8).collect(),
    };
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_control(control_source());
    let gen = mlx_gen::load("qwen_image_control", &spec).expect("load qwen_image_control");
    let req = GenerationRequest {
        prompt: "a person standing, photorealistic".into(),
        seed: Some(7),
        width: w,
        height: h,
        count: 1,
        steps: Some(2),
        conditioning: vec![Conditioning::Control {
            image: skeleton,
            kind: ControlKind::Pose,
            scale: 1.0,
        }],
        ..Default::default()
    };
    let out = gen
        .generate(&req, &mut |_p: Progress| {})
        .expect("generate");
    let GenerationOutput::Images(images) = out else {
        panic!("expected images")
    };
    assert_eq!(images.len(), 1);
    let img = &images[0];
    assert_eq!((img.width, img.height), (w, h));
    assert_eq!(img.pixels.len(), (w * h * 3) as usize);
    // Not a flat/degenerate image: more than one distinct pixel value.
    let first = img.pixels[0];
    assert!(
        img.pixels.iter().any(|&p| p != first),
        "decoded image is flat (degenerate render)"
    );
}
