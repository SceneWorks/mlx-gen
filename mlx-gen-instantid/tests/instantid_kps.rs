//! sc-3111: InstantID kps control-image renderer — pixel-for-pixel parity vs cv2 `draw_kps`.
//!
//! `#[ignore]`d — needs the golden from `tools/dump_instantid_kps_golden.py` (cv2 4.13 ground truth).
//! Run:
//!   cargo test -p mlx-gen-instantid --release --test instantid_kps -- --ignored --nocapture
//!
//! Four cases (square+view-angle, non-square+detected, extreme profile, tiny 64²) — each compared
//! byte-for-byte against the OpenCV output (zero differing pixels required).

use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen_instantid::{draw_kps, letterbox, view_angle_kps};
use mlx_rs::Dtype;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/instantid_kps_golden.safetensors"
);

fn check_case(g: &Weights, name: &str) {
    let wh = g.require(&format!("{name}_wh")).unwrap();
    let wh = wh.as_dtype(Dtype::Int32).unwrap();
    let wh = wh.as_slice::<i32>();
    let (w, h) = (wh[0] as u32, wh[1] as u32);

    let kps_arr = g.require(&format!("{name}_kps")).unwrap();
    let kps_arr = kps_arr.as_dtype(Dtype::Float32).unwrap();
    let kps_flat = kps_arr.as_slice::<f32>();
    let kps: Vec<(f32, f32)> = kps_flat.chunks_exact(2).map(|c| (c[0], c[1])).collect();

    let golden = g.require(&format!("{name}_img")).unwrap();
    let golden = golden.as_dtype(Dtype::Uint8).unwrap();
    let golden = golden.as_slice::<u8>();

    let img = draw_kps(w, h, &kps).unwrap();
    assert_eq!(
        img.pixels.len(),
        golden.len(),
        "case {name}: buffer len {} != golden {}",
        img.pixels.len(),
        golden.len()
    );

    let mut diff = 0usize;
    let mut first: Option<(usize, u8, u8)> = None;
    for (i, (&a, &b)) in img.pixels.iter().zip(golden).enumerate() {
        if a != b {
            diff += 1;
            if first.is_none() {
                first = Some((i, a, b));
            }
        }
    }
    if diff != 0 {
        let (i, a, b) = first.unwrap();
        let (px, ch) = (i / 3, i % 3);
        let (yy, xx) = (px / w as usize, px % w as usize);
        panic!(
            "case {name} ({w}x{h}): {diff} differing bytes; first @ (x={xx},y={yy},ch={ch}) mine={a} golden={b}"
        );
    }
    println!(
        "case {name} ({w}x{h}): pixel-for-pixel match ({} bytes)",
        golden.len()
    );
}

#[test]
#[ignore = "needs the instantid_kps golden (tools/dump_instantid_kps_golden.py)"]
fn draw_kps_matches_opencv() {
    let g = Weights::from_file(GOLDEN).unwrap_or_else(|e| panic!("load {GOLDEN:?}: {e}"));
    for name in ["a", "b", "c", "d"] {
        check_case(&g, name);
    }
}

#[test]
#[ignore = "needs the instantid_kps golden (tools/dump_instantid_kps_golden.py)"]
fn letterbox_matches_pil() {
    let g = Weights::from_file(GOLDEN).unwrap_or_else(|e| panic!("load {GOLDEN:?}: {e}"));

    let swh = g
        .require("lb_src_wh")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    let swh = swh.as_slice::<i32>();
    let (sw, sh) = (swh[0] as u32, swh[1] as u32);
    let src_px = g
        .require("lb_src_img")
        .unwrap()
        .as_dtype(Dtype::Uint8)
        .unwrap();
    let src = Image {
        width: sw,
        height: sh,
        pixels: src_px.as_slice::<u8>().to_vec(),
    };

    let owh = g
        .require("lb_out_wh")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    let owh = owh.as_slice::<i32>();
    let (ow, oh) = (owh[0] as u32, owh[1] as u32);
    let golden = g
        .require("lb_out_img")
        .unwrap()
        .as_dtype(Dtype::Uint8)
        .unwrap();
    let golden = golden.as_slice::<u8>();

    let out = letterbox(&src, ow, oh);
    assert_eq!((out.width, out.height), (ow, oh), "letterbox output dims");
    let diff = out
        .pixels
        .iter()
        .zip(golden)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff, 0,
        "letterbox vs PIL _letterbox: {diff} differing bytes ({sw}x{sh} -> {ow}x{oh})"
    );
    println!("letterbox ({sw}x{sh} -> {ow}x{oh}): pixel-for-pixel match");
}

#[test]
#[ignore = "needs the instantid_kps golden (tools/dump_instantid_kps_golden.py)"]
fn view_angle_kps_scaling_matches() {
    // Golden case "a" is `VIEW_ANGLE_KPS["front"] * 512` computed in numpy float32; the renderer's
    // `view_angle_kps` helper must produce the identical scaled landmarks.
    let g = Weights::from_file(GOLDEN).unwrap_or_else(|e| panic!("load {GOLDEN:?}: {e}"));
    let kps_arr = g
        .require("a_kps")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let golden = kps_arr.as_slice::<f32>();
    let mine = view_angle_kps("front", 512).expect("front view angle");
    for (i, (&(mx, my), gc)) in mine.iter().zip(golden.chunks_exact(2)).enumerate() {
        assert_eq!(mx, gc[0], "view_angle_kps front x mismatch at {i}");
        assert_eq!(my, gc[1], "view_angle_kps front y mismatch at {i}");
    }
    assert!(view_angle_kps("nonexistent", 512).is_none());
    println!("view_angle_kps front*512 matches numpy float32 scaling");
}

/// F-020/L-A: `draw_kps` (pub, re-exported) now returns a typed error rather than panicking when given
/// fewer than the 5 required keypoints.
#[test]
fn draw_kps_rejects_fewer_than_5_keypoints() {
    let four = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
    let err = draw_kps(64, 64, &four).unwrap_err().to_string();
    assert!(
        err.contains("5 keypoints") && err.contains("got 4"),
        "got: {err}"
    );
    let five = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)];
    assert!(draw_kps(64, 64, &five).is_ok());
}
