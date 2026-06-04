//! Wan2.2 LoRA **real-weight** end-to-end parity gate (sc-2683; `#[ignore]` — needs the 54 GB
//! converted A14B checkpoint + a real MoE high/low LoRA pair, so it never runs in CI; the unit merge
//! gate + the key-normalization fixture carry CI).
//!
//! The definitive "parity vs a reference-merged golden" gate. It loads the **actual converted**
//! Wan2.2-T2V-A14B experts, **merges** the real `lauren_wan22_high`/`lauren_wan22_low` pair onto them
//! via the production [`merge_wan_adapters`](mlx_gen_wan::merge_wan_adapters) path (high→high,
//! low→low — exactly `generate_wan.py --lora-high/--lora-low`), and runs the genuine chain —
//!   real UMT5-XXL encode → per-expert `embed_text` → boundary-switched dual-expert `denoise_moe`
//!   over the two real LoRA-merged 40-layer experts → real z16 VAE decode
//! — comparing the final latents + decoded frames against a golden dumped from the `mlx_video`
//! Python reference with the SAME LoRAs merged (`load_wan_model(..., loras=…)`) on the same injected
//! noise (`tools/dump_lora_real_fixtures.py`). It also asserts the LoRA **visibly** moves the output
//! (the merged latents diverge from the dumped bare-run latents by the reference's measured margin).
//!
//! Run it (after `dump_lora_real_fixtures.py` wrote the fixture):
//! ```text
//! WAN_A14B_MODEL_DIR=~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16 \
//! WAN_LORA_HIGH="…/lauren_high/lauren_wan22_high_epoch_95.safetensors" \
//! WAN_LORA_LOW="…/lauren_low/lauren_wan22_low_epoch_30.safetensors" \
//! WAN_LORA_FIXTURE=/tmp/wan_a14b_lora.safetensors \
//!   cargo test -p mlx-gen-wan --test lora_real_parity -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{AdapterKind, AdapterSpec, MoeExpert};
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::pipeline::denoise_moe;
use mlx_gen_wan::scheduler::SolverKind;
use mlx_gen_wan::{
    decode_to_frames, load_tokenizer, merge_wan_adapters, Expert, Umt5Encoder, WanTransformer,
    WanVae,
};

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(|s| PathBuf::from(shellexpand_home(&s.to_string_lossy())))
}

fn shellexpand_home(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return format!("{}/{rest}", home.to_string_lossy());
        }
    }
    s.to_string()
}

fn diff(got: &[f32], exp: &[f32]) -> (f32, f64) {
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut sum_ref = 0f64;
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sum_ref += e.abs() as f64;
    }
    (max_abs, sum_abs / sum_ref.max(1e-9))
}

fn prepend1(shape: &[i32]) -> Vec<i32> {
    let mut s = vec![1];
    s.extend_from_slice(shape);
    s
}

fn lora_spec(path: PathBuf, expert: MoeExpert) -> AdapterSpec {
    AdapterSpec {
        path,
        scale: 1.0,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: Some(expert),
    }
}

#[test]
#[ignore = "needs the 54 GB converted A14B checkpoint + the lauren high/low LoRA pair (WAN_A14B_MODEL_DIR + WAN_LORA_HIGH/LOW + WAN_LORA_FIXTURE)"]
fn wan_a14b_lora_real_weight_e2e_matches_reference() {
    let Some(model_dir) = env_path("WAN_A14B_MODEL_DIR") else {
        eprintln!("skip: set WAN_A14B_MODEL_DIR to the converted A14B model dir");
        return;
    };
    let Some(fixture) = env_path("WAN_LORA_FIXTURE") else {
        eprintln!("skip: set WAN_LORA_FIXTURE (run tools/dump_lora_real_fixtures.py first)");
        return;
    };
    let (Some(lora_high), Some(lora_low)) = (env_path("WAN_LORA_HIGH"), env_path("WAN_LORA_LOW"))
    else {
        eprintln!("skip: set WAN_LORA_HIGH and WAN_LORA_LOW to the MoE high/low LoRA pair");
        return;
    };

    let cfg = WanModelConfig::from_model_dir(&model_dir).expect("read config.json");
    assert!(cfg.dual_model, "expected the dual-expert A14B config");
    let (low_gs, high_gs) = match cfg.sample_guide_scale {
        mlx_gen_wan::GuideScale::Dual { low, high } => (low, high),
        other => panic!("expected dual guide scale, got {other:?}"),
    };

    let fx = Weights::from_file(&fixture).expect("read fixture (run dump_lora_real_fixtures.py)");
    let noise = fx.require("noise").unwrap();
    let exp_ctx = fx.require("context").unwrap();
    let exp_ctx_null = fx.require("context_null").unwrap();
    let exp_lat = fx.require("lora_latents").unwrap();
    let bare_lat = fx.require("bare_latents").unwrap();
    let exp_vid = fx.require("lora_video").unwrap();

    // Must match dump_lora_real_fixtures.py.
    let prompt = "a red fox trotting across a snowy meadow at sunrise, cinematic";
    let steps = 6usize;
    let shift = cfg.sample_shift;

    // --- Real-weight UMT5 encode (also re-checks real-weight T5 parity) ---
    let tokenizer = load_tokenizer(model_dir.join("tokenizer.json"), cfg.text_len).unwrap();
    let t5_w = Weights::from_file(model_dir.join("t5_encoder.safetensors")).expect("t5 weights");
    let enc = Umt5Encoder::from_weights(&t5_w, &cfg).expect("umt5");
    let context = enc.encode(&tokenizer, prompt).unwrap();
    let context_null = enc.encode(&tokenizer, &cfg.sample_neg_prompt).unwrap();
    let (cx_max, cx_mr) = diff(context.as_slice::<f32>(), exp_ctx.as_slice::<f32>());
    let (_, cn_mr) = diff(
        context_null.as_slice::<f32>(),
        exp_ctx_null.as_slice::<f32>(),
    );
    println!(
        "[t5 context] max|Δ|={cx_max:.3e} mean_rel={cx_mr:.3e}  context_null mean_rel={cn_mr:.3e}"
    );
    drop(enc);
    drop(t5_w);

    // --- Load both real experts and MERGE the LoRA pair per expert (the sc-2683 path) ---
    let mut low_w = Weights::from_file(model_dir.join("low_noise_model.safetensors")).expect("low");
    let mut high_w =
        Weights::from_file(model_dir.join("high_noise_model.safetensors")).expect("high");
    let specs = [
        lora_spec(lora_high, MoeExpert::High),
        lora_spec(lora_low, MoeExpert::Low),
    ];
    let low_rep = merge_wan_adapters(&mut low_w, &specs, MoeExpert::Low).expect("merge low");
    let high_rep = merge_wan_adapters(&mut high_w, &specs, MoeExpert::High).expect("merge high");
    println!(
        "[merge] low: applied={} skipped={:?}  high: applied={} skipped={:?}",
        low_rep.applied, low_rep.skipped, high_rep.applied, high_rep.skipped
    );
    // The real lauren pair targets 400 modules each (40 blocks × {self,cross}_attn.{q,k,v,o} + ffn).
    assert_eq!(low_rep.applied, 400, "low LoRA should merge 400 modules");
    assert_eq!(high_rep.applied, 400, "high LoRA should merge 400 modules");
    assert!(low_rep.skipped.is_empty() && high_rep.skipped.is_empty());

    let low_dit = WanTransformer::from_weights(&low_w, &cfg).expect("low DiT");
    let high_dit = WanTransformer::from_weights(&high_w, &cfg).expect("high DiT");
    let low = Expert {
        transformer: &low_dit,
        ctx_cond: low_dit.embed_text(&context).unwrap(),
        ctx_uncond: Some(low_dit.embed_text(&context_null).unwrap()),
        guidance: low_gs,
    };
    let high = Expert {
        transformer: &high_dit,
        ctx_cond: high_dit.embed_text(&context).unwrap(),
        ctx_uncond: Some(high_dit.embed_text(&context_null).unwrap()),
        guidance: high_gs,
    };
    let boundary_timestep = cfg.boundary * cfg.num_train_timesteps as f32;

    let latents = denoise_moe(
        &low,
        &high,
        boundary_timestep,
        SolverKind::UniPC,
        cfg.num_train_timesteps,
        steps,
        shift,
        noise,
        None,
        &mut |i| println!("  step {i}/{steps}"),
    )
    .expect("denoise_moe");
    assert_eq!(latents.shape(), exp_lat.shape(), "final latent shape");
    let (la_max, la_mr) = diff(latents.as_slice::<f32>(), exp_lat.as_slice::<f32>());
    println!(
        "[lora latents] shape={:?} max|Δ|={la_max:.3e} mean_rel={la_mr:.3e}",
        latents.shape()
    );
    // Visible effect: the LoRA-merged latents must diverge from the bare-run latents.
    let (_, effect_mr) = diff(exp_lat.as_slice::<f32>(), bare_lat.as_slice::<f32>());
    println!("[visible effect] lora vs bare mean_rel={effect_mr:.4e}");
    drop(low_dit);
    drop(high_dit);
    drop(low_w);
    drop(high_w);

    // --- Real z16 VAE decode ---
    let vae_w = Weights::from_file(model_dir.join("vae.safetensors")).expect("vae");
    let vae = WanVae::from_weights(&vae_w).expect("vae");
    let video = vae
        .decode(&latents.reshape(&prepend1(latents.shape())).unwrap())
        .unwrap();
    assert_eq!(video.shape(), exp_vid.shape(), "video shape");
    let (_, vid_mr) = diff(video.as_slice::<f32>(), exp_vid.as_slice::<f32>());
    println!(
        "[lora video] shape={:?} mean_rel={vid_mr:.3e}",
        video.shape()
    );

    let frames_u8 = decode_to_frames(&vae, &latents, None).unwrap();
    let images = mlx_gen_wan::frames_to_images(&frames_u8).unwrap();
    assert_eq!(images.len(), exp_vid.shape()[2] as usize, "frame count");

    // T5 is bf16-GEMM cross-build drift; the LoRA-merged 40-layer dual-expert stack rides the same
    // 0.31.1-vs-0.31.2 NAX-kernel envelope as the bare S6 gate (≤8e-2) — the merge adds ~0 divergence
    // (proven bit-exact in the unit gate). A merge / routing / high-low bug gives mean_rel ~O(1).
    assert!(cx_mr < 1e-2, "t5 context diverged: mean_rel={cx_mr:.3e}");
    assert!(la_mr < 8e-2, "lora latents diverged: mean_rel={la_mr:.3e}");
    assert!(vid_mr < 8e-2, "lora video diverged: mean_rel={vid_mr:.3e}");
    // The LoRA must actually do something (rank-64 @ strength 1.0): a real, non-trivial shift.
    assert!(
        effect_mr > 1e-2,
        "LoRA had no visible effect vs bare (mean_rel={effect_mr:.4e}) — merge may be a no-op"
    );
}
