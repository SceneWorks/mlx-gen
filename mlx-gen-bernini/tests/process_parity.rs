//! sc-5136: MRoPE position ids + 4-D flex attention mask match the reference, bit-exact.
//!
//! Golden (`tools/dump_bernini_process_golden.py`) copies `get_rope_index` +
//! `build_custom_attention_mask` verbatim and dumps, for four task mixes (t2i / i2i / r2v / rv2v)
//! built with the exact `BerniniTemplate` token layout: `input_ids`, `image_grid_thw`,
//! `video_grid_thw`, `token_type`, `token_segment_ids`, and the reference `position_ids` (3, L) +
//! mask visibility (L, L). These are integer / boolean outputs, so the match is **exact**.
//!
//! Run: `cargo test -p mlx-gen-bernini --test process_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_bernini::process::{build_attention_mask_4d, mrope_position_ids, MRopeConfig};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/process_golden.safetensors"
);

fn i64s(w: &Weights, key: &str) -> Vec<i64> {
    w.require(key)
        .unwrap()
        .as_slice::<i32>()
        .iter()
        .map(|&x| x as i64)
        .collect()
}

fn grids(w: &Weights, key: &str) -> Vec<[i64; 3]> {
    match w.get(key) {
        Some(a) => {
            let s = a.as_slice::<i32>();
            (0..a.shape()[0] as usize)
                .map(|i| [s[i * 3] as i64, s[i * 3 + 1] as i64, s[i * 3 + 2] as i64])
                .collect()
        }
        None => Vec::new(),
    }
}

fn i32s(w: &Weights, key: &str) -> Vec<i32> {
    w.require(key).unwrap().as_slice::<i32>().to_vec()
}

#[test]
fn process_matches_reference() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let cfg = MRopeConfig {
        spatial_merge_size: w.metadata("spatial_merge_size").unwrap().parse().unwrap(),
        tokens_per_second: w.metadata("tokens_per_second").unwrap().parse().unwrap(),
        image_token_id: w.metadata("image_token_id").unwrap().parse().unwrap(),
        video_token_id: w.metadata("video_token_id").unwrap().parse().unwrap(),
        vision_start_token_id: w
            .metadata("vision_start_token_id")
            .unwrap()
            .parse()
            .unwrap(),
    };
    let tasks: Vec<&str> = w.metadata("tasks").unwrap().split(',').collect();

    for task in tasks {
        let input_ids = i64s(&w, &format!("{task}.input_ids"));
        let image_grid = grids(&w, &format!("{task}.image_grid_thw"));
        let video_grid = grids(&w, &format!("{task}.video_grid_thw"));
        let l = input_ids.len() as i32;

        // --- position ids ---
        let pos = mrope_position_ids(&input_ids, &image_grid, &video_grid, &cfg).unwrap();
        let want_pos = w.require(&format!("{task}.position_ids")).unwrap();
        assert_eq!(pos.shape(), want_pos.shape(), "{task} position shape");
        let got: Vec<i32> = pos.flatten(None, None).unwrap().as_slice::<i32>().to_vec();
        let want: Vec<i32> = want_pos
            .flatten(None, None)
            .unwrap()
            .as_slice::<i32>()
            .to_vec();
        let pos_mismatch = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(
            pos_mismatch, 0,
            "{task} position_ids: {pos_mismatch} mismatched"
        );

        // --- 4-D flex mask (compare visibility) ---
        let token_type = i32s(&w, &format!("{task}.token_type"));
        let token_seg = i32s(&w, &format!("{task}.token_segment_ids"));
        let mask = build_attention_mask_4d(&token_type, &token_seg).unwrap();
        assert_eq!(mask.shape(), &[1, l, l], "{task} mask shape");
        let mvis: Vec<f32> = mask.flatten(None, None).unwrap().as_slice::<f32>().to_vec();
        let want_vis: Vec<i8> = w
            .require(&format!("{task}.mask_vis"))
            .unwrap()
            .as_slice::<i8>()
            .to_vec();
        let mask_mismatch = (0..(l * l) as usize)
            .filter(|&i| (mvis[i].is_finite() as i8) != want_vis[i])
            .count();
        assert_eq!(
            mask_mismatch, 0,
            "{task} mask: {mask_mismatch} mismatched cells"
        );

        println!("{task}: L={l} position_ids + {l}x{l} mask exact");
    }
}
