//! SANA Linear-DiT **trunk** parity gate vs diffusers `SanaTransformer2DModel` (story sc-8487).
//!
//! Two tests:
//!
//!  * [`trunk_matches_diffusers_tiny`] — DEFAULT (not `#[ignore]`d). Loads a SMALL, committed golden
//!    (`tests/fixtures/sana_transformer_golden.safetensors`, ~74 KB) produced by
//!    `tools/dump_sana_transformer_golden.py`: a faithful random-init `SanaTransformer2DModel`
//!    (ReLU linear self-attn + cross-attn + GLUMBConv Mix-FFN + adaLN-single + NoPE) at a reduced
//!    dim/depth, plus its inputs and reference noise prediction. The Rust trunk loads those exact
//!    weights and must reproduce the diffusers output. This keeps the parity gate reproducible in CI
//!    without the ~1.6B-param real weights.
//!
//!  * [`trunk_matches_diffusers_real`] — `#[ignore]`d, gated behind `SANA_TRANSFORMER_WEIGHTS` +
//!    `SANA_TRANSFORMER_GOLDEN` (a large fp16 single-step golden from `--real`). Characterises
//!    full-model parity against the real `Sana_1600M_1024px_diffusers` transformer.
//!
//! Parity metrics mirror `decode_parity.rs`: `mean_rel = Σ|Δ|/Σ|ref|`, `peak_rel = max|Δ|/max|ref|`.
//! A real port bug (wrong transpose / op order / modulation chunk order) wrecks `mean_rel` by orders
//! of magnitude; Metal's reduced-precision matmul (~1e-3/op) only nudges it. The committed tiny
//! golden runs in f32, so the bar is tight (`mean_rel < 5e-3`).

use mlx_rs::ops::{abs, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_sana::{SanaTransformer, SanaTransformerConfig};

fn f32(x: &Array) -> Array {
    x.as_dtype(Dtype::Float32).unwrap()
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32(got), f32(want)).unwrap()).unwrap();
    let denom = max_op(abs(f32(want)).unwrap(), None).unwrap().item::<f32>();
    max_op(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

fn mean_rel(got: &Array, want: &Array) -> f32 {
    let num = sum(abs(subtract(f32(got), f32(want)).unwrap()).unwrap(), None).unwrap();
    let den = sum(abs(f32(want)).unwrap(), None).unwrap();
    num.item::<f32>() / den.item::<f32>().max(1e-12)
}

/// Strip the dump tool's `w.` weight prefix so the trunk loader sees the diffusers key names.
fn weights_with_w_prefix(golden: &Weights) -> Weights {
    let mut w = Weights::empty();
    for key in golden.keys() {
        if let Some(rest) = key.strip_prefix("w.") {
            w.insert(rest, golden.require(key).unwrap().clone());
        }
    }
    w
}

/// Tiny config matching `dump_sana_transformer_golden.py`'s tiny instance.
fn tiny_config() -> SanaTransformerConfig {
    SanaTransformerConfig {
        in_channels: 4,
        out_channels: 4,
        num_attention_heads: 2,
        attention_head_dim: 8, // inner = 16
        num_layers: 2,
        num_cross_attention_heads: 2,
        cross_attention_head_dim: 8,
        caption_channels: 24,
        mlp_ratio: 2.5,
        patch_size: 1,
        norm_eps: 1e-6,
        caption_norm_eps: 1e-5,
        attn_qk_norm_eps: 1e-5,
        attn_eps: 1e-15,
        // Base SANA tiny config — guidance embedder + qk-norm OFF (these are the Sprint deltas).
        guidance_embeds: false,
        guidance_embeds_scale: 0.1,
        qk_norm: false,
    }
}

#[test]
fn trunk_matches_diffusers_tiny() {
    let golden_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sana_transformer_golden.safetensors"
    );
    let golden = Weights::from_file(golden_path).expect("load tiny golden");

    let latent = golden.require("input.latent").expect("latent"); // [1,4,4,4] NCHW
    let caption = golden.require("input.caption").expect("caption"); // [1,5,24]
    let timestep = golden.require("input.timestep").expect("timestep"); // [1]
    let want = golden.require("output.sample").expect("output"); // [1,4,4,4] NCHW

    let weights = weights_with_w_prefix(&golden);
    let model = SanaTransformer::from_weights(&weights, tiny_config()).expect("build trunk");
    let got = model.forward(latent, caption, timestep).expect("forward");

    assert_eq!(got.shape(), want.shape(), "shape");
    let peak = peak_rel(&got, want);
    let mean = mean_rel(&got, want);
    println!("SANA trunk parity (tiny f32): mean_rel={mean:.6}  peak_rel={peak:.6}");

    // f32 path with a 2-block tiny model: a port bug diverges by orders of magnitude. The clean port
    // sits at MLX reduced-precision-matmul noise; the per-step band Clark Labs observed (~3.4%) is the
    // far-looser ceiling — committed gate is much tighter.
    assert!(
        mean < 5e-3,
        "mean_rel {mean} too high — that IS a port bug, not rounding"
    );
    assert!(peak < 5e-2, "peak_rel {peak} above the precision ceiling");
}

/// Tiny SANA-**Sprint** config matching `dump_sana_sprint_golden.py`'s tiny instance (guidance
/// embedder + qk-norm ON, cross_attention_dim = inner = 16).
fn tiny_sprint_config() -> SanaTransformerConfig {
    SanaTransformerConfig {
        guidance_embeds: true,
        guidance_embeds_scale: 0.1,
        qk_norm: true,
        ..tiny_config()
    }
}

/// SANA-Sprint **guidance-embed trunk** parity vs diffusers `SanaTransformer2DModel(guidance_embeds=
/// True, qk_norm="rms_norm_across_heads")` (sc-8490). Loads the committed tiny golden
/// (`tests/fixtures/sana_sprint_trunk_golden.safetensors`, ~92 KB from `dump_sana_sprint_golden.py`):
/// the random-init Sprint trunk + its inputs (latent, caption, the SCM conditioning timestep, the
/// embedded guidance scalar) + the reference noise prediction. The Rust trunk loaded with the Sprint
/// config and run through `forward_with_guidance` must reproduce it — a wrong guidance-embed sum or
/// missing qk-norm wrecks `mean_rel` by orders of magnitude. f32, so the bar is tight.
#[test]
fn sprint_trunk_matches_diffusers_tiny() {
    let golden_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sana_sprint_trunk_golden.safetensors"
    );
    let golden = Weights::from_file(golden_path).expect("load tiny Sprint golden");

    let latent = golden.require("input.latent").expect("latent");
    let caption = golden.require("input.caption").expect("caption");
    let timestep = golden.require("input.timestep").expect("scm timestep");
    let guidance = golden.require("input.guidance").expect("guidance scalar");
    let want = golden.require("output.sample").expect("output");

    let weights = weights_with_w_prefix(&golden);
    let model = SanaTransformer::from_weights(&weights, tiny_sprint_config())
        .expect("build Sprint trunk (guidance embedder + qk-norm keys)");
    let got = model
        .forward_with_guidance(latent, caption, timestep, Some(guidance))
        .expect("forward_with_guidance");

    assert_eq!(got.shape(), want.shape(), "shape");
    let peak = peak_rel(&got, want);
    let mean = mean_rel(&got, want);
    println!("SANA-Sprint trunk parity (tiny f32): mean_rel={mean:.6}  peak_rel={peak:.6}");
    assert!(
        mean < 5e-3,
        "mean_rel {mean} too high — that IS a port bug in the guidance-embed / qk-norm path"
    );
    assert!(peak < 5e-2, "peak_rel {peak} above the precision ceiling");
}

#[test]
#[ignore = "needs Sana_1600M_1024px_diffusers transformer + dump_sana_transformer_golden.py --real golden"]
fn trunk_matches_diffusers_real() {
    let weights_dir =
        std::env::var("SANA_TRANSFORMER_WEIGHTS").expect("set SANA_TRANSFORMER_WEIGHTS");
    let golden_path =
        std::env::var("SANA_TRANSFORMER_GOLDEN").expect("set SANA_TRANSFORMER_GOLDEN");

    let golden = Weights::from_file(&golden_path).expect("load real golden");
    let latent = golden.require("input.latent").expect("latent");
    let caption = golden.require("input.caption").expect("caption");
    let timestep = golden.require("input.timestep").expect("timestep");
    let want = golden.require("output.sample").expect("output");

    let weights = Weights::from_dir(&weights_dir).expect("load real weights");
    let model = SanaTransformer::from_weights(&weights, SanaTransformerConfig::sana_1600m())
        .expect("build");
    let got = model.forward(latent, caption, timestep).expect("forward");

    assert_eq!(got.shape(), want.shape(), "shape");
    let peak = peak_rel(&got, want);
    let mean = mean_rel(&got, want);
    println!("SANA trunk parity (real fp16): mean_rel={mean:.6}  peak_rel={peak:.6}");

    // Real fp16 single step. The per-step drift band Clark Labs observed across a 20-step gen is
    // ~3.4%/step; a faithful single-step forward should land at or below that.
    assert!(
        mean < 3.4e-2,
        "mean_rel {mean} above the per-step drift band"
    );
}
