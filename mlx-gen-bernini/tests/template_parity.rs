//! sc-5136: `BerniniTemplate.encode_messages` matches the reference, bit-exact, on the real
//! Qwen2.5-VL tokenizer.
//!
//! Golden (`tools/dump_bernini_template_golden.py`) runs the reference `encode_messages` (verbatim,
//! indexed-pad → plain-pad remap) on the snapshot tokenizer for four task mixes (t2i / i2i / r2v /
//! rv2v). This test reproduces the same conversations via `generate_unified_inputs`, encodes them with
//! the native [`BerniniTemplate`] (plain-pad-during-assembly), and asserts `input_ids`, `token_type`,
//! `token_segment_ids`, `flex_token_types`, and the vit/vae/target-mask lists are **bit-exact** —
//! proving the two assembly strategies are equivalent on the real tokenizer.
//!
//! Requires the converted snapshot's `mllm/tokenizer.json`; `#[ignore]` otherwise (the tokenizer is
//! ~11 MB, not committed). Run:
//!   `cargo test -p mlx-gen-bernini --test template_parity -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_bernini::process::generate_unified_inputs;
use mlx_gen_bernini::template::{BerniniTemplate, TemplateOutput};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/template_golden.safetensors"
);

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/mlx-gen-models/bernini_planner_mlx_bf16")
}

fn want_i32(w: &Weights, key: &str) -> Vec<i32> {
    match w.get(key) {
        // Guard 0-length tensors (empty lists dumped as zeros(0)): `as_slice` errors on a null buffer.
        Some(a) if a.shape().iter().product::<i32>() > 0 => a.as_slice::<i32>().to_vec(),
        _ => Vec::new(),
    }
}

/// (task, conversation, image_token_nums, video_token_nums) — grids match the process golden:
/// token_num = t·(h/2)·(w/2).
#[allow(clippy::type_complexity)]
fn cases() -> Vec<(&'static str, Vec<serde_json::Value>, Vec<i64>, Vec<i64>)> {
    vec![
        (
            "t2i",
            generate_unified_inputs("a cat", &[], 0, 1, 64, 64),
            vec![4],
            vec![],
        ),
        (
            "i2i",
            generate_unified_inputs("edit", &[(48, 72)], 0, 1, 64, 64),
            vec![6, 4],
            vec![],
        ),
        (
            "r2v",
            generate_unified_inputs("subj", &[(72, 48)], 0, 9, 64, 64),
            vec![6],
            vec![12],
        ),
        (
            "rv2v",
            generate_unified_inputs("edit v", &[], 1, 9, 64, 64),
            vec![],
            vec![12, 20],
        ),
    ]
}

#[test]
#[ignore = "needs the converted snapshot's mllm/tokenizer.json (~11 MB, not committed)"]
fn template_matches_reference() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let tmpl = BerniniTemplate::from_tokenizer_file(snapshot().join("mllm/tokenizer.json"))
        .expect("tokenizer");

    for (task, conv, img_nums, vid_nums) in cases() {
        let o: TemplateOutput = tmpl
            .encode_messages(&conv, &img_nums, &vid_nums, task)
            .expect("encode_messages");

        let ids32: Vec<i32> = o.input_ids.iter().map(|&x| x as i32).collect();
        let checks: [(&str, &Vec<i32>); 9] = [
            ("input_ids", &ids32),
            ("token_type", &o.token_type),
            ("token_segment_ids", &o.token_segment_ids),
            ("flex_token_types", &o.flex_token_types),
            ("vit_type_list", &o.vit_type_list),
            ("vit_img_and_vid_id_list", &o.vit_img_and_vid_id_list),
            ("image_target_mask", &o.image_target_mask),
            ("video_target_mask", &o.video_target_mask),
            ("vae_type_list", &o.vae_type_list),
        ];
        for (field, got) in checks {
            let want = want_i32(&w, &format!("{task}.{field}"));
            assert_eq!(got, &want, "{task}.{field}");
        }
        println!("{task}: L={} all 9 fields bit-exact", o.input_ids.len());
    }
}
