//! sc-5132: real-weight load + forward smoke for the Qwen2.5-VL-7B planner backbone (`#[ignore]`).
//!
//! Loads the converted planner snapshot's `qwen2_5_vl.safetensors` (14 GB bf16, all 728 tensors) via
//! `from_weights` — proving the sc-5144 converter keys match the module's expected names exactly — and
//! runs a forward on a small synthetic `(inputs_embeds, position_ids, causal mask)`, asserting the
//! penultimate hidden state has the right shape and is finite (no NaN/Inf at 7B scale). Numeric
//! correctness is covered by the f32 synthetic golden (`qwen_backbone_parity`); this is the scale gate.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_bernini::qwen2_5_vl::{Qwen25VlText, QwenVlTextConfig};
use mlx_rs::{random, Array, Dtype};

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/mlx-gen-models/bernini_planner_mlx_bf16")
}

/// Additive causal mask `[1,1,L,L]` (0 on/below the diagonal, -inf above).
fn causal_mask(l: usize) -> Array {
    let neg = f32::NEG_INFINITY;
    let mut data = vec![0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            if j > i {
                data[i * l + j] = neg;
            }
        }
    }
    Array::from_slice(&data, &[1, 1, l as i32, l as i32])
}

#[test]
#[ignore = "real weights: loads the 14 GB Qwen2.5-VL planner backbone and runs a forward"]
fn qwen_backbone_real_weight_smoke() {
    let snap = snapshot();
    let cfg = QwenVlTextConfig::from_config_json(&snap.join("qwen2_5_vl_config.json"))
        .expect("read qwen2_5_vl_config.json");
    assert_eq!(cfg.num_layers, 28);
    assert_eq!(cfg.hidden_size, 3584);

    let w = Weights::from_file(snap.join("qwen2_5_vl.safetensors")).expect("open backbone weights");
    let backbone = Qwen25VlText::from_weights(&w, cfg.clone(), "model").expect("load backbone");

    let l = 16i32;
    let key = random::key(0).unwrap();
    let shape = [1, l, cfg.hidden_size];
    let embeds = random::normal::<f32>(&shape[..], None, None, Some(&key))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    // Text positions: all three MRoPE rows = arange(L).
    let mut pos = Vec::with_capacity(3 * l as usize);
    for _ in 0..3 {
        pos.extend(0..l);
    }
    let position_ids = Array::from_slice(&pos, &[3, l]);
    let mask = causal_mask(l as usize).as_dtype(Dtype::Bfloat16).unwrap();

    let penult = backbone
        .penultimate(&embeds, &position_ids, &mask)
        .expect("forward");
    assert_eq!(penult.shape(), &[1, l, cfg.hidden_size]);
    // max|·| is NaN/Inf if any element is non-finite, and >0 iff non-trivial — one check covers both.
    let m = penult.abs().unwrap().max(None).unwrap().item::<f32>();
    assert!(
        m.is_finite() && m > 0.0,
        "penultimate must be finite & non-trivial (max abs {m})"
    );
    println!("qwen2.5-vl backbone real-weight forward ok: penultimate max|·| = {m:.4}");
}
