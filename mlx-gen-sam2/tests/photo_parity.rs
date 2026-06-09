//! sc-3708 — SAM2 box-prompt **GO gate**: the MLX `Sam2Segmenter` on real photos vs the spike's
//! quality baseline (PyTorch `transformers Sam2Model` + onnxruntime, sc-3635). This is the
//! engine-side GO before SceneWorks wiring (sc-3709).
//!
//! Golden: `tools/dump_sam2_photo_golden.py` (packages the spike's real RGB photos + box prompts +
//! PyTorch/ONNX baseline masks). Both photos are run end-to-end through `segment(rgb, box)`
//! (preprocess → encode → box prompt → decode → upsample → threshold) and the binary-mask IoU is
//! compared to the baselines. Target: the spike's ort-vs-PyTorch band (zidane ~0.99, bus ~0.93).
//!
//! Run (weights from the SceneWorks mirror or tools/convert_sam2_to_mlx.py):
//!   SCENEWORKS_SAM2_WEIGHTS=/path/to/sam2.1_hiera_large.safetensors \
//!     cargo test -p mlx-gen-sam2 --release --test photo_parity -- --ignored --nocapture

use std::time::Instant;

use mlx_gen::weights::Weights;
use mlx_gen_sam2::{Sam2ImageEncoderConfig, Sam2Segmenter};
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sam2_photo_golden.safetensors"
);

fn u8s(a: &Array) -> Vec<u8> {
    a.as_dtype(mlx_rs::Dtype::Uint8)
        .unwrap()
        .as_slice::<u8>()
        .to_vec()
}

/// IoU of two binary masks (`got` is 0/255, `base` is 0/1).
fn iou(got: &[u8], base: &[u8]) -> f32 {
    let (mut inter, mut union) = (0usize, 0usize);
    for (&g, &b) in got.iter().zip(base) {
        let (gg, bb) = (g > 127, b > 0);
        if gg && bb {
            inter += 1;
        }
        if gg || bb {
            union += 1;
        }
    }
    if union == 0 {
        1.0
    } else {
        inter as f32 / union as f32
    }
}

#[test]
#[ignore = "needs SCENEWORKS_SAM2_WEIGHTS + tools/dump_sam2_photo_golden.py golden"]
fn box_prompt_go_vs_spike_baseline() {
    let wpath = std::env::var("SCENEWORKS_SAM2_WEIGHTS")
        .expect("set SCENEWORKS_SAM2_WEIGHTS to a converted large checkpoint");
    let w = Weights::from_file(&wpath).expect("load weights");
    let g = Weights::from_file(GOLDEN)
        .unwrap_or_else(|e| panic!("missing {GOLDEN}: {e}\nRun tools/dump_sam2_photo_golden.py"));
    let seg = Sam2Segmenter::from_weights(&w, &Sam2ImageEncoderConfig::large()).unwrap();

    // (name, min IoU vs PyTorch) — the spike's ort-vs-PyTorch band: zidane 0.99, bus 0.93.
    let cases = [("zidane", 0.95f32), ("bus", 0.90f32)];
    let mut all_go = true;
    for (name, min_iou) in cases {
        let rgb_arr = g.require(&format!("rgb_{name}")).unwrap();
        let sh = rgb_arr.shape(); // [H, W, 3]
        let (h, wd) = (sh[0] as u32, sh[1] as u32);
        let rgb = u8s(rgb_arr);
        let bx = g
            .require(&format!("box_{name}"))
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let bx = [bx[0], bx[1], bx[2], bx[3]];

        // Warm once (graph/kernel compile), then time a clean run.
        let _ = seg.segment(&rgb, h, wd, bx).unwrap();
        let t = Instant::now();
        let mask = seg.segment(&rgb, h, wd, bx).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let got = u8s(&mask);

        let pt = u8s(g.require(&format!("pt_{name}")).unwrap());
        let ort = u8s(g.require(&format!("ort_{name}")).unwrap());
        let iou_pt = iou(&got, &pt);
        let iou_ort = iou(&got, &ort);
        let fg = got.iter().filter(|&&v| v > 127).count();
        println!(
            "{name} ({h}x{wd}): IoU vs PyTorch {iou_pt:.4}, vs ONNX {iou_ort:.4} | mask fg {fg} | {ms:.0} ms/frame"
        );
        if iou_pt < min_iou {
            all_go = false;
            eprintln!("  ✗ {name} below target {min_iou}");
        }
    }
    assert!(
        all_go,
        "one or more images below the spike-band IoU target — see output"
    );
    println!("GO ✓ — MLX box-prompt segmentation matches the spike quality baseline.");
}
