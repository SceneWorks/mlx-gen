//! sc-5140: the planner-input assembly glue matches the reference (bit-exact).
//!
//! Synthetic-fixture parity (`tools/dump_bernini_assembly_golden.py`):
//!   - `format_mllm_inputs_embeds` — token embedding + `masked_scatter` of the ViT visual features
//!     into the visual slots (input-ViT ∪ gen-ViT).
//!   - `concat_with_zero_init` — prepend the T5 prompt embeds, then zero-pad / truncate to
//!     `max_sequence_length` (both branches).
//!
//! These are exact host/array ops (gather, scatter, concat, pad, slice) → bit-for-bit equality.
//!
//! Run: `cargo test -p mlx-gen-bernini --test assembly_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_bernini::assembly::{concat_with_zero_init, format_mllm_inputs_embeds};
use mlx_gen_bernini::qwen2_5_vl::{Qwen25VlText, QwenVlTextConfig};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/assembly_golden.safetensors"
);

fn bools(w: &Weights, key: &str) -> Vec<bool> {
    w.require(key)
        .unwrap()
        .as_slice::<i32>()
        .iter()
        .map(|&x| x != 0)
        .collect()
}

fn assert_exact(name: &str, got: &Array, want: &Array) {
    assert_eq!(got.shape(), want.shape(), "{name} shape");
    let n = want.shape().iter().product::<i32>();
    let g = got.reshape(&[n]).unwrap();
    let wv = want.reshape(&[n]).unwrap();
    let max_diff = g
        .as_slice::<f32>()
        .iter()
        .zip(wv.as_slice::<f32>())
        .fold(0f32, |m, (&a, &b)| m.max((a - b).abs()));
    println!("{name:>14}: max|Δ|={max_diff:.3e}");
    assert!(max_diff < 1e-6, "{name} max|Δ| {max_diff} not bit-exact");
}

#[test]
fn assembly_matches_reference() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");

    // Minimal backbone (0 layers) — only the token embedding is exercised.
    let cfg = QwenVlTextConfig {
        hidden_size: 16,
        num_layers: 0,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 8,
        intermediate_size: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        mrope_section: [1, 2, 1],
    };
    let backbone = Qwen25VlText::from_weights(&w, cfg, "model").expect("backbone");

    // --- format_mllm_inputs_embeds ---
    let input_ids = w
        .require("io.input_ids")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let visual_embeds = w.require("io.visual_embeds").unwrap().clone();
    let vin = bools(&w, "io.visual_input_mask");
    let vout = bools(&w, "io.visual_output_mask");
    let got = format_mllm_inputs_embeds(&backbone, &input_ids, Some(&visual_embeds), &vin, &vout)
        .expect("format_mllm");
    assert_exact("format_mllm", &got, w.require("out.format_mllm").unwrap());

    // no-visual path == plain token embedding (sanity: scatter of nothing is a no-op).
    let none =
        format_mllm_inputs_embeds(&backbone, &input_ids, None, &vin, &vout).expect("no visual");
    assert_eq!(none.shape(), &[1, input_ids.len() as i32, 16]);

    // --- concat_with_zero_init (pad + truncate) ---
    let t5 = w.require("io.t5").unwrap();
    let pad = concat_with_zero_init(t5, w.require("io.stream_short").unwrap(), 10).expect("pad");
    assert_exact("concat_pad", &pad, w.require("out.concat_pad").unwrap());
    let trunc = concat_with_zero_init(t5, w.require("io.stream_long").unwrap(), 10).expect("trunc");
    assert_exact(
        "concat_trunc",
        &trunc,
        w.require("out.concat_trunc").unwrap(),
    );
}
