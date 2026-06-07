//! SVD CLIP-image antialiased-resize parity vs diffusers `_resize_with_antialiasing` (epic 3040 /
//! sc-3412). Gates `mlx_gen_svd::resize_with_antialiasing_unit` (gaussian-blur + align-corners
//! bicubic, in `[-1,1]`) against a golden dumped straight from the diffusers pipeline function
//! (`tools/dump_svd_clip_preprocess_golden.py`) — no checkpoint needed, so this runs by default.
//!
//! Run: `cargo test -p mlx-gen-svd --test clip_preprocess_parity -- --nocapture`

use mlx_rs::ops::{abs, max as max_op, mean, subtract};

use mlx_gen::weights::Weights;
use mlx_gen_svd::resize_with_antialiasing_unit;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/svd_clip_preprocess_golden.safetensors"
);

#[test]
fn svd_clip_antialiased_resize_matches_diffusers() {
    let g = Weights::from_file(GOLDEN).expect("clip preprocess golden");

    // input_image: HWC [448,800,3], integer-valued f32 in [0,255] → RGB8.
    let input = g.require("input_image").unwrap();
    let sh = input.shape();
    let (in_h, in_w) = (sh[0] as usize, sh[1] as usize);
    let flat = input.reshape(&[(in_h * in_w * 3) as i32]).unwrap();
    let rgb8: Vec<u8> = flat.as_slice::<f32>().iter().map(|&v| v as u8).collect();

    // The Rust antialiased preprocess: HWC [224,224,3] in [0,1].
    let out_h = 224usize;
    let out_w = 224usize;
    let hwc = resize_with_antialiasing_unit(&rgb8, in_h, in_w, out_h, out_w);
    let got = mlx_rs::Array::from_slice(&hwc, &[1, out_h as i32, out_w as i32, 3])
        .transpose_axes(&[0, 3, 1, 2]) // NHWC → NCHW to match the golden
        .unwrap();

    let want = g.require("resized_unit").unwrap(); // NCHW [1,3,224,224]
    assert_eq!(got.shape(), want.shape(), "resized_unit shape");

    let diff = abs(subtract(&got, want).unwrap()).unwrap();
    let max_abs = max_op(&diff, None).unwrap().item::<f32>();
    let mean_abs = mean(&diff, None).unwrap().item::<f32>();
    println!("clip antialiased resize parity: max|Δ| {max_abs}, mean|Δ| {mean_abs}");

    // Output is in [0,1]; this is an absolute gate. The blur + cubic + reflect/clamp math matches
    // diffusers structurally — the only residual is f32 op-ordering (host accumulation vs torch's
    // fused kernels). ~1e-4 max is the cross-implementation float floor, far tighter than the
    // previous PIL-bicubic approximation (a different cubic kernel + antialias model entirely).
    assert!(max_abs < 1e-3, "max|Δ| {max_abs} too large");
    assert!(mean_abs < 1e-4, "mean|Δ| {mean_abs} too large");
}
