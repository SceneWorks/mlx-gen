//! LTX-2.3 LoRA-in-generate parity (sc-2687) vs the reference forward-time residual
//! (`mlx_video/lora/apply.py::LoRALinear`).
//!
//! Golden: `tools/dump_ltx_lora_golden.py` applies a real LTX LoRA to the `ltx_2_3_base_q8`
//! transformer via the reference `LoRALinear` (residual over the Q8 base, strength 1.0) and runs the
//! **native bf16 + Q8** 2-stage distilled denoise from the committed e2e golden's injected inputs →
//! `tests/fixtures/ltx_lora_golden.safetensors`. The Rust port stacks the same residuals
//! ([`apply_ltx_adapters`] → [`LtxDiT`] forward) at `Precision::quant_bf16`, so the chaos-sensitive
//! stage-1 must reproduce the reference per-forward bit-for-bit. The residual math is a pure
//! per-Linear forward add, identical for the video-only [`LtxDiT`] and the production [`AvDiT`], so a
//! bit-exact `LtxDiT` gate + a full-surface `AvDiT` routing gate together cover the production path.
//!
//! Default video LoRA = `LTX2.3_Crisp_Enhance` (attn + ff + gate, 576 video targets, no audio/adaLN);
//! default multi-surface LoRA = `Samantha_ltx2.3` (also trains audio + cross-modal). Override with
//! `LTX_LORA` / `LTX_LORA_MULTI` (regenerate the golden the same way for a different `LTX_LORA`).
//!
//! Gates (all `#[ignore]`, ~20 GB model + the LoRA):
//!  1. `lora_routing_resolves_full_crisp_surface` — every Crisp target resolves (applied 576, 0 skipped).
//!  2. `lora_frames_match_reference` — bit-exact final latents + frames px>8 < 1% vs the golden.
//!  3. `lora_multi_surface_skips_audio_without_error` — on the video-only `LtxDiT` building block, a
//!     LoRA's audio targets are reported skipped, never errored.
//!  4. `lora_full_surface_routes_on_avdit` — on the production `AvDiT`, the same LoRA's audio +
//!     cross-modal targets all resolve (1632, 0 skipped).
//!  5. `lora_per_pass_strength_changes_output` — a `[1.0, 0.0]` per-pass schedule (stage-2 LoRA off)
//!     diverges from the uniform golden, proving the per-pass wiring drives the pipeline.
//!
//! Run: `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test lora_real_weights -- --ignored --nocapture`

use mlx_rs::ops::{abs, gt, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen_ltx::adapters::apply_ltx_adapters;
use mlx_gen_ltx::config::{LtxConfig, LtxVaeConfig, SplitModel};
use mlx_gen_ltx::pipeline::{decode_to_frames, generate_t2v_latents, NUM_DENOISE_PASSES};
use mlx_gen_ltx::positions::create_position_grid;
use mlx_gen_ltx::transformer::{AvDiT, LtxDiT, Precision};
use mlx_gen_ltx::upsampler::LatentUpsampler;
use mlx_gen_ltx::vae::LtxVideoVae;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_lora_golden.safetensors"
);

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

fn lora_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("LTX_LORA") {
        return p.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home).join(
        "Library/Application Support/SceneWorks/data/loras/crisp_enhance/LTX2.3_Crisp_Enhance.safetensors",
    )
}

/// A multi-surface LoRA that also trains the audio + cross-modal stacks (default `Samantha_ltx2.3`).
fn multi_lora_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("LTX_LORA_MULTI") {
        return p.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home).join(
        "Library/Application Support/SceneWorks/data/loras/samantha/Samantha_ltx2.3.safetensors",
    )
}

fn f32(x: &Array) -> Array {
    x.as_dtype(Dtype::Float32).unwrap()
}

fn mean_rel(got: &Array, want: &Array) -> f32 {
    let num = sum(abs(subtract(f32(got), f32(want)).unwrap()).unwrap(), None).unwrap();
    let den = sum(abs(f32(want)).unwrap(), None).unwrap();
    num.item::<f32>() / den.item::<f32>().max(1e-12)
}

fn px_gt8(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32(got), f32(want)).unwrap()).unwrap();
    let over = gt(&diff, Array::from_int(8)).unwrap();
    sum(over.as_dtype(Dtype::Float32).unwrap(), None)
        .unwrap()
        .item::<f32>()
        / (got.size() as f32)
}

/// A uniform LoRA spec at `scale` (matching the reference golden's strength 1.0).
fn lora_spec(scale: f32) -> AdapterSpec {
    AdapterSpec::new(lora_path(), scale, AdapterKind::Lora)
}

/// Build the Bf16Q8 DiT and install `specs`, returning the adapted DiT + the apply report.
fn dit_with_adapters(
    dir: &std::path::Path,
    specs: &[AdapterSpec],
) -> (LtxDiT, mlx_gen_ltx::LtxLoraReport) {
    let cfg = LtxConfig::from_model_dir(dir).expect("config");
    let split = SplitModel::from_model_dir(dir).expect("split_model.json");
    let tw = Weights::from_file(dir.join("transformer.safetensors")).expect("transformer");
    let mut dit = LtxDiT::from_weights(&tw, &cfg, Precision::quant_bf16(split.bits, split.group))
        .expect("dit");
    let report = apply_ltx_adapters(&mut dit, specs, NUM_DENOISE_PASSES).expect("apply adapters");
    (dit, report)
}

/// Run the bf16 2-stage pipeline from the golden's injected inputs → (final_latents, frames).
fn render(dir: &std::path::Path, dit: &LtxDiT, g: &Weights) -> (Array, Array) {
    let uw = Weights::from_file(dir.join("upsampler.safetensors")).expect("upsampler");
    let up = LatentUpsampler::from_weights(&uw).expect("upsampler");
    let vcfg = LtxVaeConfig::from_model_dir(dir).expect("vae cfg");
    let dec = Weights::from_file(dir.join("vae_decoder.safetensors")).expect("vae");
    let vae = LtxVideoVae::from_weights(&dec, None, &vcfg).expect("vae");
    let mean = dec
        .require("per_channel_statistics.mean")
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let std = dec
        .require("per_channel_statistics.std")
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

    let pos1 = create_position_grid(1, 2, 4, 4);
    let pos2 = create_position_grid(1, 2, 8, 8);
    let latents = generate_t2v_latents(
        dit,
        &up,
        g.require("stage1_noise").unwrap(),
        &pos1,
        g.require("stage2_noise").unwrap(),
        &pos2,
        g.require("video_embeddings").unwrap(),
        &mean,
        &std,
        &mlx_gen::CancelFlag::default(),
        &mut |_| {},
    )
    .expect("generate_t2v_latents");
    let frames = decode_to_frames(&vae, &latents).expect("decode");
    (latents, frames)
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 (~20 GB) + the LTX LoRA (~705 MB)"]
fn lora_routing_resolves_full_crisp_surface() {
    let dir = base_dir();
    let (_dit, report) = dit_with_adapters(&dir, &[lora_spec(1.0)]);
    eprintln!(
        "Crisp routing: applied={} skipped={}",
        report.applied,
        report.skipped.len()
    );
    // Crisp targets attn1/attn2 {to_q,to_k,to_v,to_out,to_gate_logits} + ff {proj_in,proj_out} on all
    // 48 blocks = 48·12 = 576, every one resolved (no audio / adaLN / globals in this file).
    assert_eq!(
        report.applied, 576,
        "expected the full 576-target Crisp surface"
    );
    assert!(
        report.skipped.is_empty(),
        "no Crisp target should be skipped: {:?}",
        report.skipped
    );
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 (~20 GB) + the LTX LoRA (~705 MB)"]
fn lora_frames_match_reference() {
    let dir = base_dir();
    let g = Weights::from_file(GOLDEN).expect("lora golden");
    let (dit, report) = dit_with_adapters(&dir, &[lora_spec(1.0)]);
    assert_eq!(report.applied, 576);

    let (latents, frames) = render(&dir, &dit, &g);
    let fmr = mean_rel(&latents, g.require("final_latents").unwrap());
    let want_frames = g.require("frames").unwrap();
    assert_eq!(frames.shape(), want_frames.shape(), "frame shape");
    let px = px_gt8(&frames, want_frames);
    eprintln!(
        "LoRA e2e: final latents mean_rel = {fmr:.3e}, frames px>8 = {:.2}%",
        px * 100.0
    );
    // The residual matches the reference `LoRALinear` op-for-op, so the per-forward is bit-exact and
    // the chaos-sensitive stage-1 stays deterministic-identical → bit-exact latents, pixel-exact frames.
    assert!(
        fmr == 0.0,
        "LoRA final latents must be bit-exact to the reference residual: {fmr:.3e}"
    );
    assert!(
        px < 1e-2,
        "LoRA frames px>8 {:.2}% exceeds the 1% acceptance",
        px * 100.0
    );
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 (~20 GB) + a multi-surface LTX LoRA with audio (default Samantha)"]
fn lora_multi_surface_skips_audio_without_error() {
    // A real character LoRA (`Samantha_ltx2.3`) trains the **audio** stack too. The video-only port
    // must apply every video attn/ff/gate target (576) and *skip* every audio target (1056) — reported,
    // never errored (the reference `apply_loras_to_weights` likewise only counts skips). This guards
    // the "full capability, no silent error" property: loading a production LoRA must not blow up.
    let dir = base_dir();
    let (_dit, report) = dit_with_adapters(
        &dir,
        &[AdapterSpec::new(multi_lora_path(), 1.0, AdapterKind::Lora)],
    );
    eprintln!(
        "video-only (LtxDiT) routing: applied={} skipped={}",
        report.applied,
        report.skipped.len()
    );
    assert_eq!(
        report.applied, 576,
        "all 576 video attn/ff/gate targets apply"
    );
    assert_eq!(
        report.skipped.len(),
        1056,
        "every audio target is skipped (reported, not errored)"
    );
    // Every skipped path is an audio / cross-modal target (none of the video surface leaked in).
    assert!(
        report
            .skipped
            .iter()
            .all(|p| p.contains("audio") || p.contains("av_ca") || p.contains("a2v")),
        "skipped set must be exactly the audio/cross-modal targets"
    );
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 (~20 GB) + a multi-surface LTX LoRA with audio (default Samantha)"]
fn lora_full_surface_routes_on_avdit() {
    // On the **production** dual-modality AvDiT (sc-2684), the same Samantha LoRA's audio + cross-modal
    // targets now RESOLVE (vs skipped on the video-only LtxDiT building block above): 576 video +
    // 1056 audio/cross-modal (audio_attn1/2 + audio_ff + audio_to_video_attn + video_to_audio_attn,
    // 22 per block × 48) = 1632, no skips. Proves the full-surface routing (sc-2687 over sc-2684).
    let dir = base_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("config");
    let split = SplitModel::from_model_dir(&dir).expect("split_model.json");
    let tw = Weights::from_file(dir.join("transformer.safetensors")).expect("transformer");
    let mut dit = AvDiT::from_weights(&tw, &cfg, Precision::quant_bf16(split.bits, split.group))
        .expect("avdit");
    let report = apply_ltx_adapters(
        &mut dit,
        &[AdapterSpec::new(multi_lora_path(), 1.0, AdapterKind::Lora)],
        NUM_DENOISE_PASSES,
    )
    .expect("apply adapters");
    eprintln!(
        "AvDiT full-surface routing: applied={} skipped={}",
        report.applied,
        report.skipped.len()
    );
    assert_eq!(
        report.applied, 1632,
        "576 video + 1056 audio/cross-modal targets all resolve on AvDiT"
    );
    assert!(
        report.skipped.is_empty(),
        "no Samantha target should be skipped on AvDiT: {:?}",
        report.skipped
    );
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 (~20 GB) + the LTX LoRA (~705 MB)"]
fn lora_per_pass_strength_changes_output() {
    let dir = base_dir();
    let g = Weights::from_file(GOLDEN).expect("lora golden");
    // Per-pass [stage1=1.0, stage2=0.0]: the LoRA is active in stage-1 but OFF in stage-2, so the
    // final latents must diverge from the uniform (strength-1.0-both-passes) golden — proving the
    // per-pass schedule actually drives the pipeline (sc-2687).
    let spec = lora_spec(1.0).with_pass_scales(vec![1.0, 0.0]);
    let (dit, report) = dit_with_adapters(&dir, &[spec]);
    assert_eq!(report.applied, 576);

    let (latents, _frames) = render(&dir, &dit, &g);
    let mr = mean_rel(&latents, g.require("final_latents").unwrap());
    eprintln!("per-pass [1.0, 0.0] vs uniform golden: final latents mean_rel = {mr:.3e}");
    assert!(
        mr > 1e-3,
        "per-pass [1.0, 0.0] should differ from the uniform golden, got mean_rel {mr:.3e}"
    );
}
