//! sc-5136: ViT image preprocessing matches `Qwen2VLImageProcessor` on the exactly-matchable pieces.
//!
//! Golden (`tools/dump_bernini_vit_preprocess_golden.py`):
//!   - `smart_resize` over identity / up-clamp / down-clamp / banker's-round cases — **bit-exact**
//!     (integer dims feed `grid_thw`).
//!   - the rescale + CLIP-normalize + temporal-pad + 9-axis pack on a fixed uint8 image (the
//!     reference run with `do_resize=False`, so the non-bit-identical PIL resize is excluded) —
//!     elementwise affine + an exact reshape, so it matches to ~1e-5.
//!
//! Run: `cargo test -p mlx-gen-bernini --test vit_preprocess_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_bernini::vit_preprocess::{pack_patches, smart_resize, smart_video_nframes};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/vit_preprocess_golden.safetensors"
);

fn floats(meta: &str) -> [f32; 3] {
    let v: Vec<f32> = meta.split(',').map(|s| s.parse().unwrap()).collect();
    [v[0], v[1], v[2]]
}

#[test]
fn vit_preprocess_matches_reference() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let factor: i64 = w.metadata("factor").unwrap().parse().unwrap();
    let min_pixels: i64 = w.metadata("min_pixels").unwrap().parse().unwrap();
    let max_pixels: i64 = w.metadata("max_pixels").unwrap().parse().unwrap();
    let mean = floats(w.metadata("image_mean").unwrap());
    let std = floats(w.metadata("image_std").unwrap());
    let patch: i64 = w.metadata("patch").unwrap().parse().unwrap();
    let temporal: i64 = w.metadata("temporal").unwrap().parse().unwrap();
    let merge: i64 = w.metadata("merge").unwrap().parse().unwrap();

    // --- smart_resize: bit-exact dims ---
    let sin = w
        .require("smart_resize.in")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let sout = w
        .require("smart_resize.out")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let n = sin.len() / 2;
    for i in 0..n {
        let (h, wd) = (sin[i * 2] as i64, sin[i * 2 + 1] as i64);
        let got = smart_resize(h, wd, factor, min_pixels, max_pixels);
        let want = (sout[i * 2] as i64, sout[i * 2 + 1] as i64);
        assert_eq!(got, want, "smart_resize({h},{wd})");
    }
    println!("smart_resize: {n} cases bit-exact");

    // --- rescale + normalize + pack: build a [1,3,H,W] from the uint8 image and pack ---
    let pack_h: i64 = w.metadata("pack_h").unwrap().parse().unwrap();
    let pack_w: i64 = w.metadata("pack_w").unwrap().parse().unwrap();
    let img_i32 = w
        .require("pack.image_hwc_u8")
        .unwrap()
        .as_slice::<i32>()
        .to_vec(); // [H,W,3]
    let (hu, wu) = (pack_h as usize, pack_w as usize);
    let mut data = vec![0f32; 3 * hu * wu];
    for c in 0..3usize {
        for y in 0..hu {
            for x in 0..wu {
                let u = img_i32[(y * wu + x) * 3 + c] as f32;
                data[(c * hu + y) * wu + x] = (u / 255.0 - mean[c]) / std[c];
            }
        }
    }
    let frame = Array::from_slice(&data, &[1, 3, pack_h as i32, pack_w as i32]);
    let (pv, grid) = pack_patches(&frame, patch, temporal, merge).unwrap();

    let want_pv = w.require("pack.pixel_values").unwrap();
    let want_grid = w
        .require("pack.grid_thw")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    assert_eq!(grid.to_vec(), want_grid, "grid_thw");
    assert_eq!(pv.shape(), want_pv.shape(), "pixel_values shape");

    let got: Vec<f32> = pv.flatten(None, None).unwrap().as_slice::<f32>().to_vec();
    let want: Vec<f32> = want_pv
        .flatten(None, None)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let peak = want.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = got
        .iter()
        .zip(&want)
        .fold(0f32, |m, (&a, &b)| m.max((a - b).abs()));
    println!(
        "pack: grid {grid:?} pixel_values {:?} peak|Δ|={max_diff:.3e} peak-rel={:.3e}",
        pv.shape(),
        max_diff / peak
    );
    assert!(
        max_diff / peak < 1e-5,
        "pack peak-rel {} exceeds 1e-5",
        max_diff / peak
    );

    // --- smart_video_nframes: bit-exact frame indices ---
    let cases: Vec<Vec<f64>> = w
        .metadata("nframes_cases")
        .unwrap()
        .split(';')
        .map(|c| c.split(',').map(|x| x.parse().unwrap()).collect())
        .collect();
    for (i, c) in cases.iter().enumerate() {
        // (total_frames, video_fps, fps, frame_factor, max_frames, add_one)
        let got = smart_video_nframes(
            c[0] as i64,
            c[1],
            c[2],
            Some(c[3] as i64),
            None,
            Some(c[4] as i64),
            c[5] != 0.0,
        );
        let want: Vec<i64> = w
            .require(&format!("nframes.{i}"))
            .unwrap()
            .as_slice::<i32>()
            .iter()
            .map(|&x| x as i64)
            .collect();
        assert_eq!(got, want, "smart_video_nframes case {i}");
    }
    println!("smart_video_nframes: {} cases bit-exact", cases.len());
}
