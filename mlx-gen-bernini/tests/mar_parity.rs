//! sc-5140: the MAR semantic-planning loop (`sample_vit_embed`) matches the reference (f32).
//!
//! Synthetic-fixture parity (`tools/dump_bernini_mar_golden.py`): a tiny but structurally-faithful
//! Qwen2.5-VL backbone + `MLPConnector` + `SimpleMLPAdaLN` driven through the **reference**
//! `sample_vit_embed` loop + `feat_from_planner_to_renderer`, with the two RNG consumers injected (the
//! reveal `order` and the per-step FM noise). Validates the full orchestration end-to-end: the cosine
//! reveal schedule, the `nonzero().sum()==0` skip quirk (token 0 stays the mask token), the
//! gather/scatter write-back across 3 streams, the triple-CFG clip-diff sampling, and the 4-stream
//! handoff. Tolerance reflects the f32 floor compounded over 4 steps × 3 backbone forwards + the final
//! handoff; the mask mechanics themselves are exact (a wrong reveal/scatter is O(1)).
//!
//! Run: `cargo test -p mlx-gen-bernini --test mar_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_bernini::clip_diff::DiffLossFm;
use mlx_gen_bernini::connector::MlpConnector;
use mlx_gen_bernini::mar::{sample_vit_embed, StreamState, VitCfg};
use mlx_gen_bernini::qwen2_5_vl::{Qwen25VlText, QwenVlTextConfig};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/mar_golden.safetensors"
);

fn idx(w: &Weights, key: &str) -> Vec<i32> {
    w.require(key).unwrap().as_slice::<i32>().to_vec()
}

fn check(name: &str, got: &Array, want: &Array, tol: f32) {
    assert_eq!(got.shape(), want.shape(), "{name} shape");
    let n = want.shape().iter().product::<i32>();
    let g = got.reshape(&[n]).unwrap();
    let wv = want.reshape(&[n]).unwrap();
    let (g, wv) = (g.as_slice::<f32>(), wv.as_slice::<f32>());
    let peak = wv.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = g
        .iter()
        .zip(wv)
        .fold(0f32, |m, (&a, &b)| m.max((a - b).abs()));
    println!(
        "{name:>16}: peak|Δ|={max_diff:.3e} peak-rel={:.3e}",
        max_diff / peak
    );
    assert!(
        max_diff / peak < tol,
        "{name} peak-rel {} exceeds {tol:.1e}",
        max_diff / peak
    );
}

fn stream(w: &Weights, name: &str) -> StreamState {
    StreamState {
        input_embeds: w.require(&format!("io.{name}.input")).unwrap().clone(),
        position_ids: w.require(&format!("io.{name}.pos")).unwrap().clone(),
        mask: w.require(&format!("io.{name}.mask4d")).unwrap().clone(),
        gen_idx: idx(w, &format!("io.{name}.gen_idx")),
    }
}

#[test]
fn mar_loop_matches_reference_f32() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");

    // Tiny config — mirrors the dumper's structurally-faithful Qwen2.5-VL text decoder.
    let cfg = QwenVlTextConfig {
        hidden_size: 16,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 8,
        intermediate_size: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        mrope_section: [1, 2, 1],
    };
    let backbone = Qwen25VlText::from_weights(&w, cfg, "w.model").expect("backbone");
    let connector = MlpConnector::from_weights(&w, "conn").expect("connector");
    let mut clip_diff = DiffLossFm::from_weights(&w, "net", 2, 16, 2.0).expect("clip_diff");

    let mask_token = w.require("io.mask_token").unwrap().clone(); // [1, 1, H]
    let order = idx(&w, "io.order");

    let vit = VitCfg {
        planning_step: 4,
        vit_denoising_step: 2,
        vit_txt_cfg: 1.4,
        vit_img_cfg: 1.2,
    };

    // step_noise indexed by step; the skipped step-0 entry is a placeholder (never read).
    let step_noise: Vec<Array> = (0..vit.planning_step)
        .map(|s| w.require(&format!("io.noise.{s}")).unwrap().clone())
        .collect();

    let cond = stream(&w, "cond");
    let uncond = stream(&w, "uncond");
    let imgcond = stream(&w, "imgcond");

    let out = sample_vit_embed(
        &backbone,
        &connector,
        &mut clip_diff,
        &cond,
        &uncond,
        &imgcond,
        &vit,
        &order,
        &step_noise,
        &CancelFlag::default(),
        &mut |_step| {},
        &mask_token,
    )
    .expect("sample_vit_embed");

    // The filled target ViT embeds — token 0 stays the mask token (skip quirk); the f32 floor is set
    // by the compounded backbone+clip-diff path.
    check(
        "pred_vit_embed",
        &out.pred_vit_embed,
        w.require("out.pred_vit_embed").unwrap(),
        1e-2,
    );
    check(
        "wtxt_wvit",
        &out.wtxt_wvit,
        w.require("out.wtxt_wvit").unwrap(),
        1e-2,
    );
    check(
        "wtxt_wovit",
        &out.wtxt_wovit,
        w.require("out.wtxt_wovit").unwrap(),
        1e-2,
    );
    check(
        "wotxt_wvit",
        &out.wotxt_wvit,
        w.require("out.wotxt_wvit").unwrap(),
        1e-2,
    );
    check(
        "wotxt_wovit",
        &out.wotxt_wovit,
        w.require("out.wotxt_wovit").unwrap(),
        1e-2,
    );
}
