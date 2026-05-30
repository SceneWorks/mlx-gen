//! sc-2341: Qwen2-VL image-processor parity vs the fork's `QwenImageProcessor`.
//!
//! Fixture `tests/fixtures/qwen_image_processor.safetensors` ← `tools/dump_image_processor.py`
//! (deterministic synthetic images; input dumped as uint8 HWC so the Rust side feeds identical
//! pixels with no image-decode dependency). Three cases: no-resize (exact), downscale, upscale.
//!
//! `grid_thw` and `pixel_values` are bit-exact for the no-resize and upscale cases. The
//! antialiased downscale path agrees to a measured max of 0.0150 (= exactly one uint8
//! quantization level, 1/255, after CLIP-normalization) — PIL's fixed-point resampler isn't
//! bit-reproduced, so that case uses a 1.7e-2 tolerance (just above the measured 1/255).

use mlx_gen::models::qwen::{ImageInput, QwenImageProcessor};
use mlx_gen::weights::Weights;
use mlx_rs::ops::{all_close, array_eq};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen_image_processor.safetensors"
);

fn run_case(w: &Weights, name: &str, exact: bool) {
    let input = w.require(&format!("{name}.input")).unwrap(); // uint8 (H, W, 3)
    let shape = input.shape();
    let (h, wd) = (shape[0] as usize, shape[1] as usize);
    let bytes: Vec<u8> = input.as_slice::<u8>().to_vec();

    let proc = QwenImageProcessor::default();
    let out = proc
        .preprocess(ImageInput {
            data: &bytes,
            height: h,
            width: wd,
        })
        .unwrap();

    let want_pv = w.require(&format!("{name}.pixel_values")).unwrap();
    let want_thw = w.require(&format!("{name}.grid_thw")).unwrap();

    assert!(
        array_eq(&out.grid_thw, want_thw, false)
            .unwrap()
            .item::<bool>(),
        "{name}: grid_thw diverged"
    );
    assert_eq!(
        out.pixel_values.shape(),
        want_pv.shape(),
        "{name}: pixel_values shape"
    );

    // Antialiased downscale differs from PIL's fixed-point bicubic by ≤1/255 (measured
    // 0.0150 after CLIP-normalize); upscale/no-resize are bit-exact.
    let (rtol, atol) = if exact { (1e-4, 1e-4) } else { (1e-2, 1.7e-2) };
    assert!(
        all_close(&out.pixel_values, want_pv, rtol, atol, false)
            .unwrap()
            .item::<bool>(),
        "{name}: pixel_values diverged (exact={exact})"
    );
}

#[test]
fn no_resize_case_is_exact() {
    // dims already multiples of 28 -> bicubic skipped -> isolates normalize + patchify.
    run_case(&Weights::from_file(FIXTURE).unwrap(), "a", true);
}

#[test]
fn downscale_case_matches_pil_bicubic() {
    run_case(&Weights::from_file(FIXTURE).unwrap(), "b", false);
}

#[test]
fn upscale_case_matches_pil_bicubic() {
    run_case(&Weights::from_file(FIXTURE).unwrap(), "c", false);
}

#[test]
fn input_uint8_dtype_roundtrips() {
    // Guard: the fixture input is genuinely uint8 (HWC), as the API expects.
    let w = Weights::from_file(FIXTURE).unwrap();
    let a: &Array = w.require("a.input").unwrap();
    assert_eq!(a.dtype(), mlx_rs::Dtype::Uint8);
    assert_eq!(a.shape(), &[56, 84, 3]);
}
