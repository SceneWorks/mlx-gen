//! ChatGLM3-6B text-encoder parity vs the diffusers `KolorsPipeline` reference (sc-3091).
//!
//! `#[ignore]`d: needs the `Kwai-Kolors/Kolors-diffusers` `text_encoder/` fp16 shards (~12.5 GB) +
//! the golden from `tools/dump_kolors_chatglm_golden.py` (gitignored, real-weights). Runs the Rust
//! `ChatGlmModel` on the golden's fixed input_ids/attention_mask and checks all 29 hidden states +
//! the extracted context/pooled reproduce the reference, for the `packed` (pure-causal) and `padded`
//! (causal+padding) cases, in BOTH f32 (cross-backend Metal-vs-CPU floor ~1e-3, flat over depth) and
//! fp16 (production-dtype floor).
//!
//! Run: `cargo test -p mlx-gen-kolors --test chatglm_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_kolors::chatglm3::{ChatGlmConfig, ChatGlmModel};

fn te_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("KOLORS_TE_DIR") {
        return d.into();
    }
    let base = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--Kwai-Kolors--Kolors-diffusers/snapshots");
    let snap = std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("snapshot dir {}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max()
        .expect("a snapshot");
    snap.join("text_encoder")
}

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/kolors_chatglm_golden.safetensors"
);

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max(abs(want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

/// For one (dtype, case): run the forward + encode_prompt on the golden's ids/mask and assert every
/// hidden state, plus the extracted context/pooled, is within `tol`.
fn check_case(model: &ChatGlmModel, g: &Weights, prefix: &str, num_hidden: usize, tol: f32) {
    let input_ids = g.require(&format!("{prefix}input_ids")).unwrap();
    let attention_mask = g.require(&format!("{prefix}attention_mask")).unwrap();

    let hiddens = model.forward(input_ids, attention_mask).expect("forward");
    assert_eq!(
        hiddens.len(),
        num_hidden,
        "{prefix}: expected {num_hidden} hidden states"
    );

    let mut worst = 0f32;
    let mut worst_i = 0usize;
    let mut profile = Vec::new();
    for (i, h) in hiddens.iter().enumerate() {
        let want = g.require(&format!("{prefix}h_{i:02}")).unwrap();
        let pr = peak_rel(h, want);
        profile.push(pr);
        if pr > worst {
            worst = pr;
            worst_i = i;
        }
    }
    let trace: Vec<String> = [0usize, 1, 2, 8, 16, 24, 26, 27, 28]
        .iter()
        .filter(|&&i| i < profile.len())
        .map(|&i| format!("h{i}={:.2e}", profile[i]))
        .collect();
    eprintln!("{prefix} per-layer peak_rel: {}", trace.join(" "));

    // The Kolors extraction: context = hidden_states[-2], pooled = hidden_states[-1] last token.
    let (context, pooled) = model
        .encode_prompt(input_ids, attention_mask)
        .expect("encode_prompt");
    let ctx_pr = peak_rel(&context, g.require(&format!("{prefix}context")).unwrap());
    let pooled_pr = peak_rel(&pooled, g.require(&format!("{prefix}pooled")).unwrap());
    eprintln!(
        "{prefix} worst hidden peak_rel {worst:.3e} @ h{worst_i} | context {ctx_pr:.3e} | pooled {pooled_pr:.3e}"
    );

    assert!(
        worst < tol,
        "{prefix} hidden peak_rel {worst:.3e} @ h{worst_i} exceeds {tol:.1e}"
    );
    assert!(
        ctx_pr < tol,
        "{prefix} context peak_rel {ctx_pr:.3e} exceeds {tol:.1e}"
    );
    assert!(
        pooled_pr < tol,
        "{prefix} pooled peak_rel {pooled_pr:.3e} exceeds {tol:.1e}"
    );
}

fn run_gate(dtype: Dtype, prefix: &str, tol: f32) {
    let w = Weights::from_dir(te_dir()).expect("load Kolors text_encoder shards");
    let model =
        ChatGlmModel::from_weights(&w, ChatGlmConfig::chatglm3_6b(), None, dtype).expect("build");

    let g = Weights::from_file(GOLDEN).expect("chatglm golden");
    let num_hidden: usize = g.metadata("num_hidden").unwrap().parse().unwrap();

    check_case(&model, &g, &format!("{prefix}packed_"), num_hidden, tol);
    check_case(&model, &g, &format!("{prefix}padded_"), num_hidden, tol);
}

#[test]
#[ignore = "needs Kolors-diffusers text_encoder (~12.5 GB) + tools/golden/kolors_chatglm_golden.safetensors"]
fn chatglm_f32_hidden_states_match_reference() {
    // f32 Rust (MLX/Metal) vs f32 reference (torch/CPU). The half-dim interleaved RoPE + MQA + the
    // 28-layer GLMBlock stack are reproduced op-for-op: the per-layer peak_rel is FLAT at ~1.1e-3
    // across all 28 layers (no depth growth), context ~1.1e-3, pooled ~1.3–2.3e-3. That flat ~1e-3 is
    // the cross-backend f32 floor (different Metal vs CPU GEMM/SDPA kernels — cf. the repo's ~2.4e-3
    // Metal f32-matmul floor), NOT bf16-style accumulation; a structural bug (wrong RoPE pairing,
    // swapped gate/up, bad mask) diverges orders of magnitude past this. The padded case's pooled is
    // marginally higher because right-padding lands the "last token" on a PAD position (a deterministic
    // but noisier pad-query-row state). 5e-3 sits above the floor, far below any real defect.
    run_gate(Dtype::Float32, "", 5e-3);
}

#[test]
#[ignore = "needs Kolors-diffusers text_encoder (~12.5 GB) + tools/golden/kolors_chatglm_golden.safetensors"]
fn chatglm_f16_hidden_states_match_reference() {
    // fp16 Rust vs fp16 reference — the production dtype (Kolors `torch_dtype=float16`). fp16 drift
    // accumulates over 28 layers; the floor tolerance mirrors the Gemma bf16 gate.
    run_gate(Dtype::Float16, "f16_", 2e-2);
}
