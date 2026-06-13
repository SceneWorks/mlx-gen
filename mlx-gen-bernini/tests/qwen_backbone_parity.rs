//! sc-5132: the native Qwen2.5-VL-7B planner backbone matches the reference forward (near-bit, f32).
//!
//! Synthetic-fixture parity (the repo's weight-free golden pattern, like sensenova's
//! `backbone_parity`): a tiny structurally-faithful Qwen2.5-VL text decoder with random weights,
//! dumped from the reference math by `tools/dump_bernini_qwen_backbone_golden.py`. This exercises the
//! full forward — the 3D MRoPE channel stitch, QKV-bias projections, GQA repeat, the external
//! additive 4D mask, the residual stack, and the HF `hidden_states[-2]` tap — without the 14 GB
//! checkpoint. f32 throughout; the penultimate tolerance reflects the MLX-Metal-vs-torch f32 matmul
//! floor over the 2-layer stack, while the MRoPE table (trig only) matches far tighter.
//!
//! Run: `cargo test -p mlx-gen-bernini --test qwen_backbone_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_bernini::qwen2_5_vl::{Qwen25VlText, QwenVlTextConfig};
use mlx_rs::{Array, Dtype};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen_backbone_golden.safetensors"
);

fn config_from_meta(w: &Weights) -> QwenVlTextConfig {
    let m = |k: &str| {
        w.metadata(k)
            .unwrap_or_else(|| panic!("missing metadata {k}"))
    };
    let sec: Vec<usize> = m("mrope_section")
        .split(',')
        .map(|s| s.parse::<usize>().unwrap())
        .collect();
    QwenVlTextConfig {
        hidden_size: m("hidden_size").parse().unwrap(),
        num_layers: m("num_hidden_layers").parse().unwrap(),
        num_heads: m("num_attention_heads").parse().unwrap(),
        num_kv_heads: m("num_key_value_heads").parse().unwrap(),
        head_dim: m("head_dim").parse().unwrap(),
        intermediate_size: m("intermediate_size").parse().unwrap(),
        rms_norm_eps: m("rms_norm_eps").parse().unwrap(),
        rope_theta: m("rope_theta").parse().unwrap(),
        mrope_section: [sec[0], sec[1], sec[2]],
    }
}

/// (peak abs diff, peak-relative `max|Δ|/max|b|`).
fn errors(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    (max_diff, max_diff / peak)
}

#[test]
fn qwen_backbone_matches_reference_f32() {
    let w = Weights::from_file(FIXTURE).expect("load golden fixture");
    let cfg = config_from_meta(&w);

    // f32 throughout (the dumped weights are f32); load under the fixture's `w.model` namespace.
    let backbone = Qwen25VlText::from_weights(&w, cfg.clone(), "w.model").expect("backbone");

    let embeds = w.require("io.embeds").unwrap().clone();
    let position_ids = w.require("io.position_ids").unwrap().clone();
    let mask = w.require("io.mask").unwrap().clone();

    // 1. MRoPE table golden — the net-new 3D rotary stitch, compared to torch's assembled cos/sin.
    let (cos, sin) = backbone
        .mrope_cos_sin(&position_ids, Dtype::Float32)
        .unwrap();
    let (cos_abs, cos_rel) = errors(&cos, w.require("out.cos").unwrap());
    let (sin_abs, sin_rel) = errors(&sin, w.require("out.sin").unwrap());
    println!("mrope cos: peak|Δ|={cos_abs:.3e} rel={cos_rel:.3e}  sin: peak|Δ|={sin_abs:.3e} rel={sin_rel:.3e}");
    // ~1e-4 cross-backend f32 floor (pow(1e6,·) + outer product + trig); a wrong axis stitch is O(1).
    assert!(
        cos_rel < 1e-3 && sin_rel < 1e-3,
        "MRoPE table must match torch"
    );

    // 2. Penultimate hidden state golden — the full forward through both layers.
    let all = backbone.forward(&embeds, &position_ids, &mask).unwrap();
    assert_eq!(
        all.len(),
        cfg.num_layers as usize + 1,
        "hidden-state count = N+1"
    );
    let penult = &all[all.len() - 2];
    let (abs, rel) = errors(penult, w.require("out.penultimate").unwrap());
    println!("penultimate: peak|Δ|={abs:.3e}  peak-rel={rel:.3e}");
    assert!(
        rel < 5e-3,
        "penultimate hidden state within the f32 matmul floor (rel {rel:.3e})"
    );

    // 3. The convenience accessor returns the same [-2] tensor.
    let p2 = backbone.penultimate(&embeds, &position_ids, &mask).unwrap();
    let (_, rel2) = errors(&p2, penult);
    assert!(rel2 < 1e-6, "penultimate() == forward()[-2]");
}
