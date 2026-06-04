//! sc-2671 + sc-2919: SDXL **complete-coverage** LoRA — mid_block + GEGLU feed-forward + the
//! **conv-layer** LoRAs (resnet convs, samplers, conv_in/out), strictly beyond the vendored
//! 515-module Linear-only surface (sc-2639).
//!
//! `#[ignore]`d — needs the real SDXL snapshot + a real kohya LoRA (`latent-consistency/lcm-lora-sdxl`
//! in the HF cache). **No golden file is needed:** the per-module merge math is the sc-2639-proven
//! primitive (`lora_delta`, bit-exact vs the vendored f16 matmul ×515), so what sc-2671 adds and what
//! these gates check is purely **routing + the GEGLU row-split + an exhaustive count** — all validated
//! *build-independently* (byte-exact within this build, so immune to the pmetal-vs-wheel f32 residual).
//!
//! Run: cargo test -p mlx-gen-sdxl --release --test lora_complete_real_weights -- --ignored --nocapture
//!
//! Gates:
//! - `complete_strictly_exceeds_vendored_and_count_matches_table` — complete coverage merges strictly
//!   more than the vendored 515, and the exact merge count equals an independent path-level derivation
//!   from the LoRA file (Linear + conv stems; proving nothing reachable is dropped and the GEGLU 2×
//!   split is counted). LCM-LoRA: vendored 515 → complete 858 (809 Linear + 49 conv).
//! - `complete_deltas_match_reference_byte_exact` — the merged mid_block + GEGLU weights byte-match an
//!   independently computed reference (`base + lora_delta`, with the value/gate row-split), proving the
//!   delta lands on the right Linear and the right half.
//! - `complete_conv_deltas_match_reference_byte_exact` (sc-2919) — the merged conv weights byte-match
//!   `base + conv_lora_delta` for a 3×3 resnet conv, a down/up-sampler conv (NHWC), and a 1×1
//!   conv_shortcut (Linear reshape), proving the conv fold + NCHW→NHWC routing land correctly.
//! - `complete_scale_zero_is_bit_exact_noop` / `complete_conv_scale_zero_is_bit_exact_noop` — a scale-0
//!   complete merge leaves the Linear / conv weights untouched.
//! - `complete_render_differs_from_vendored_and_is_sane` — the extra mid/ff/conv signal actually
//!   changes the rendered image (and the render is a full, non-empty buffer).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use mlx_rs::ops::{add, array_eq};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{conv_lora_delta, AdaptableHost};
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource,
};
use mlx_gen_sdxl::{
    apply_sdxl_adapters, apply_sdxl_adapters_with, load_unet, lora_delta, LoraCoverage,
    UNet2DConditionModel,
};
// Force-link the provider so its `inventory::submit!` registers `"sdxl"`.
use mlx_gen_sdxl as _;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn lora_path() -> PathBuf {
    if let Ok(p) = std::env::var("SDXL_LORA") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--latent-consistency--lcm-lora-sdxl/snapshots");
    let dir = std::fs::read_dir(&snaps)
        .expect("LCM-LoRA HF cache")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir");
    dir.join("pytorch_lora_weights.safetensors")
}

fn lora_spec(scale: f32) -> AdapterSpec {
    AdapterSpec {
        path: lora_path(),
        scale,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: None,
    }
}

/// Read (a clone of) the current dense weight of the Linear at `dotted` — used to snapshot bases and
/// read back merged weights. Panics if the path is not routable or the base is already quantized.
fn weight_at(unet: &mut UNet2DConditionModel, dotted: &str) -> Array {
    let parts: Vec<&str> = dotted.split('.').collect();
    unet.adaptable_mut(&parts)
        .unwrap_or_else(|| panic!("{dotted} should be routable under complete coverage"))
        .dense_weight()
        .expect("dense base")
        .0
        .clone()
}

/// Rows `[lo, hi)` of a 2-D array (mirrors the private `adapters::rows` slicer).
fn rows(a: &Array, lo: i32, hi: i32) -> Array {
    let idx = Array::from_slice(&(lo..hi).collect::<Vec<i32>>(), &[hi - lo]);
    a.take_axis(&idx, 0).unwrap()
}

/// Read the LoRA-file alpha for a flattened stem (defaults to `rank` when absent), matching the
/// merge code's `t.alpha.unwrap_or(rank)`.
fn alpha_for(lora: &Weights, flat: &str, rank: f32) -> f32 {
    lora.get(&format!("{flat}.alpha"))
        .map(|a| {
            a.as_dtype(Dtype::Float32)
                .unwrap()
                .reshape(&[1])
                .unwrap()
                .as_slice::<f32>()[0]
        })
        .unwrap_or(rank)
}

/// The full LoRA delta for one kohya module stem (flattened, incl. the `lora_unet_` prefix), scale 1.
fn delta_for(lora: &Weights, flat: &str) -> Array {
    let down = lora.require(&format!("{flat}.lora_down.weight")).unwrap();
    let up = lora.require(&format!("{flat}.lora_up.weight")).unwrap();
    let rank = down.shape()[0] as f32;
    lora_delta(down, up, alpha_for(lora, flat, rank), rank, 1.0).unwrap()
}

/// The fused **conv** LoRA delta (trained-file NCHW `[out, in, kH, kW]`) for one kohya conv stem.
fn conv_delta_for(lora: &Weights, flat: &str) -> Array {
    let down = lora.require(&format!("{flat}.lora_down.weight")).unwrap();
    let up = lora.require(&format!("{flat}.lora_up.weight")).unwrap();
    let rank = down.shape()[0] as f32;
    conv_lora_delta(down, up, alpha_for(lora, flat, rank), rank, 1.0).unwrap()
}

/// Read (a clone of) the current NHWC `[out, kH, kW, in]` weight of the conv at `dotted` (sc-2919).
fn conv_weight_at(unet: &mut UNet2DConditionModel, dotted: &str) -> Array {
    let parts: Vec<&str> = dotted.split('.').collect();
    unet.adaptable_conv_mut(&parts)
        .unwrap_or_else(|| panic!("{dotted} should be a routable conv under complete coverage"))
        .weight()
        .clone()
}

/// Logical (stride-respecting) element-wise equality. Conv weights are loaded as a strided
/// `nchw_to_nhwc` *transpose view*, while a merged weight is the contiguous output of `add`; their
/// raw `as_slice()` buffers are in different physical orders even when the logical tensors are equal,
/// so conv comparisons must use `array_eq` (mlx compares by logical index, not buffer order).
fn logical_eq(a: &Array, b: &Array) -> bool {
    array_eq(a, b, false).unwrap().item::<bool>()
}

#[test]
#[ignore = "needs the real SDXL snapshot + LCM-LoRA in the HF cache"]
fn complete_strictly_exceeds_vendored_and_count_matches_table() {
    let lora = Weights::from_file(lora_path()).unwrap();

    // Distinct kohya module stems present in the LoRA file.
    let mut stems: BTreeSet<String> = BTreeSet::new();
    for k in lora.keys() {
        let stem = k
            .strip_suffix(".lora_down.weight")
            .or_else(|| k.strip_suffix(".lora_up.weight"))
            .or_else(|| k.strip_suffix(".alpha"))
            .unwrap_or(k);
        stems.insert(stem.to_string());
    }

    let unet0 = load_unet(&snapshot()).unwrap();
    let vendored: BTreeSet<String> = unet0.lora_target_paths().into_iter().collect();
    // The complete surface is the Linear targets (sc-2671) PLUS the conv-layer targets (sc-2919);
    // `apply_sdxl_adapters_with(Complete)` merges both, so the count derivation must too.
    let mut complete_paths = unet0.lora_target_paths_complete();
    complete_paths.extend(unet0.conv_target_paths());
    let complete_set: BTreeSet<String> = complete_paths.iter().cloned().collect();
    let complete_table: BTreeMap<String, String> = complete_paths
        .into_iter()
        .map(|p| (p.replace('.', "_"), p))
        .collect();

    // Independent, path-level derivation of the expected complete merge count: every LoRA stem that
    // maps into the complete surface (Linear or conv) contributes 1 weight update, except a GEGLU
    // `ff.net.0.proj`, which row-splits across `linear1`+`linear2` (2). No matmul — purely routing.
    let mut expected = 0usize;
    for s in &stems {
        let flat = s.strip_prefix("lora_unet_").unwrap_or(s);
        if let Some(dotted) = complete_table.get(flat) {
            expected += if dotted.ends_with(".ff.net.0.proj") {
                2
            } else {
                1
            };
        }
    }

    let mut uv = load_unet(&snapshot()).unwrap();
    let rv = apply_sdxl_adapters(&mut uv, &[lora_spec(1.0)]).unwrap();
    assert_eq!(
        rv.merged, 515,
        "vendored coverage must stay at the faithful 515"
    );

    let mut uc = load_unet(&snapshot()).unwrap();
    let rc = apply_sdxl_adapters_with(&mut uc, &[lora_spec(1.0)], LoraCoverage::Complete).unwrap();
    assert_eq!(
        rc.merged, expected,
        "complete merge count {} must equal the table-derived expectation {expected}",
        rc.merged
    );
    assert!(
        rc.merged > rv.merged,
        "complete coverage must reach strictly more than the vendored 515 (got {})",
        rc.merged
    );
    assert!(
        vendored.is_subset(&complete_set),
        "the faithful surface must be a strict subset of the complete surface"
    );
    println!(
        "✓ vendored {} → complete {} merges (+{}); table-derived expected {}",
        rv.merged,
        rc.merged,
        rc.merged - rv.merged,
        expected
    );
}

#[test]
#[ignore = "needs the real SDXL snapshot + LCM-LoRA"]
fn complete_deltas_match_reference_byte_exact() {
    let lora = Weights::from_file(lora_path()).unwrap();
    let blk = "down_blocks.1.attentions.0.transformer_blocks.0";
    let blk_flat = format!("lora_unet_{}", blk.replace('.', "_"));
    let mid_attn = "mid_block.attentions.0.transformer_blocks.0.attn1.to_q";
    let mid_proj = "mid_block.attentions.0.proj_in";
    let mid_resnet = "mid_block.resnets.0.time_emb_proj";

    // Snapshot the base weights, then build the expected merged weights (base + delta) — the GEGLU
    // proj row-split is applied here independently of the merge code under test.
    let mut base = load_unet(&snapshot()).unwrap();
    let base_l1 = weight_at(&mut base, &format!("{blk}.ff.linear1"));
    let base_l2 = weight_at(&mut base, &format!("{blk}.ff.linear2"));
    let base_l3 = weight_at(&mut base, &format!("{blk}.ff.linear3"));
    let base_mid_attn = weight_at(&mut base, mid_attn);
    let base_mid_proj = weight_at(&mut base, mid_proj);
    let base_mid_resnet = weight_at(&mut base, mid_resnet);

    let full = delta_for(&lora, &format!("{blk_flat}_ff_net_0_proj")); // [2h, D]
    let two_h = full.shape()[0];
    let h = two_h / 2;
    let exp_l1 = add(&base_l1, rows(&full, 0, h)).unwrap();
    let exp_l2 = add(&base_l2, rows(&full, h, two_h)).unwrap();
    let exp_l3 = add(&base_l3, delta_for(&lora, &format!("{blk_flat}_ff_net_2"))).unwrap();
    let exp_mid_attn = add(
        &base_mid_attn,
        delta_for(&lora, &format!("lora_unet_{}", mid_attn.replace('.', "_"))),
    )
    .unwrap();
    let exp_mid_proj = add(
        &base_mid_proj,
        delta_for(&lora, &format!("lora_unet_{}", mid_proj.replace('.', "_"))),
    )
    .unwrap();
    let exp_mid_resnet = add(
        &base_mid_resnet,
        delta_for(
            &lora,
            &format!("lora_unet_{}", mid_resnet.replace('.', "_")),
        ),
    )
    .unwrap();

    let mut unet = load_unet(&snapshot()).unwrap();
    apply_sdxl_adapters_with(&mut unet, &[lora_spec(1.0)], LoraCoverage::Complete).unwrap();

    let cases: [(&str, &Array); 6] = [
        (&format!("{blk}.ff.linear1"), &exp_l1),
        (&format!("{blk}.ff.linear2"), &exp_l2),
        (&format!("{blk}.ff.linear3"), &exp_l3),
        (mid_attn, &exp_mid_attn),
        (mid_proj, &exp_mid_proj),
        (mid_resnet, &exp_mid_resnet),
    ];
    for (path, expected) in cases {
        let got = weight_at(&mut unet, path);
        assert_eq!(
            got.as_slice::<f32>(),
            expected.as_slice::<f32>(),
            "merged weight at {path} must byte-match base + lora_delta (routing/split bug otherwise)"
        );
    }
    // The two GEGLU halves must receive *different* deltas (sanity: not the same rows twice).
    assert_ne!(
        exp_l1.as_slice::<f32>(),
        exp_l2.as_slice::<f32>(),
        "value and gate halves must differ"
    );
    println!("✓ mid_block (attn/proj/resnet) + GEGLU (linear1/2/3) merged weights are byte-exact");
}

#[test]
#[ignore = "needs the real SDXL snapshot + LCM-LoRA"]
fn complete_conv_deltas_match_reference_byte_exact() {
    // sc-2919: the conv-layer LoRA merge. Three representative conv kinds, each checked
    // build-independently (no torch golden): the merged conv weight must byte-match an
    // independently computed `base + δ`, where δ = `conv_lora_delta` (NCHW) routed into the conv's
    // own layout. Covers (a) a 3×3 resnet conv1, (b) a 3×3 down-sampler conv (both NHWC
    // `AdaptableConv2d`), and (c) a 1×1 `conv_shortcut` (a Linear, merged via a reshaped 2-D delta).
    let lora = Weights::from_file(lora_path()).unwrap();

    // (a)/(b): NHWC convs. expected = base_nhwc + transpose([out,in,kH,kW]→[out,kH,kW,in]) of δ.
    let nhwc_cases = [
        "down_blocks.0.resnets.0.conv1",
        "down_blocks.0.downsamplers.0.conv",
        "up_blocks.0.upsamplers.0.conv",
    ];
    let mut base = load_unet(&snapshot()).unwrap();
    let nhwc_expected: Vec<Array> = nhwc_cases
        .iter()
        .map(|dotted| {
            let b = conv_weight_at(&mut base, dotted);
            let flat = format!("lora_unet_{}", dotted.replace('.', "_"));
            let d_nhwc = conv_delta_for(&lora, &flat)
                .transpose_axes(&[0, 2, 3, 1])
                .unwrap();
            add(&b, &d_nhwc).unwrap()
        })
        .collect();

    // (c): conv_shortcut is a Linear `[out,in]`; the 1×1 δ folds `[out,in,1,1]→[out,in]`.
    let sc = "down_blocks.1.resnets.0.conv_shortcut";
    let base_sc = weight_at(&mut base, sc);
    let sc_flat = format!("lora_unet_{}", sc.replace('.', "_"));
    let d_sc = conv_delta_for(&lora, &sc_flat);
    let ss = d_sc.shape();
    let exp_sc = add(&base_sc, d_sc.reshape(&[ss[0], ss[1]]).unwrap()).unwrap();

    let mut unet = load_unet(&snapshot()).unwrap();
    apply_sdxl_adapters_with(&mut unet, &[lora_spec(1.0)], LoraCoverage::Complete).unwrap();

    for (dotted, expected) in nhwc_cases.iter().zip(&nhwc_expected) {
        let got = conv_weight_at(&mut unet, dotted);
        assert_eq!(
            got.shape(),
            expected.shape(),
            "conv {dotted} shape changed under merge"
        );
        assert!(
            logical_eq(&got, expected),
            "merged conv weight at {dotted} must byte-match base + transpose(conv_lora_delta)"
        );
    }
    let got_sc = weight_at(&mut unet, sc);
    assert!(
        logical_eq(&got_sc, &exp_sc),
        "merged conv_shortcut at {sc} must byte-match base + reshaped conv δ"
    );
    println!(
        "✓ conv merge byte-exact: resnet conv1 + down/up-sampler conv (NHWC) + conv_shortcut (1×1 Linear)"
    );
}

#[test]
#[ignore = "needs the real SDXL snapshot + LCM-LoRA"]
fn complete_conv_scale_zero_is_bit_exact_noop() {
    // sc-2919: a scale-0 complete merge must leave the conv weights bit-identical (δ·0 ⇒ W+0).
    let mut base = load_unet(&snapshot()).unwrap();
    let conv = "down_blocks.0.resnets.0.conv1";
    let sc = "down_blocks.1.resnets.0.conv_shortcut";
    let base_conv = conv_weight_at(&mut base, conv);
    let base_sc = weight_at(&mut base, sc);

    let mut unet = load_unet(&snapshot()).unwrap();
    apply_sdxl_adapters_with(&mut unet, &[lora_spec(0.0)], LoraCoverage::Complete).unwrap();
    assert!(
        logical_eq(&conv_weight_at(&mut unet, conv), &base_conv),
        "scale-0 conv merge must be a bit-exact no-op"
    );
    assert!(
        logical_eq(&weight_at(&mut unet, sc), &base_sc),
        "scale-0 conv_shortcut merge must be a bit-exact no-op"
    );
    println!("✓ scale-0 complete conv merge is a bit-exact no-op (conv1 + conv_shortcut)");
}

#[test]
#[ignore = "needs the real SDXL snapshot + LCM-LoRA"]
fn complete_scale_zero_is_bit_exact_noop() {
    let mut base = load_unet(&snapshot()).unwrap();
    let blk = "down_blocks.1.attentions.0.transformer_blocks.0";
    let base_l1 = weight_at(&mut base, &format!("{blk}.ff.linear1"));
    let base_mid = weight_at(&mut base, "mid_block.attentions.0.proj_in");

    let mut unet = load_unet(&snapshot()).unwrap();
    apply_sdxl_adapters_with(&mut unet, &[lora_spec(0.0)], LoraCoverage::Complete).unwrap();
    let got_l1 = weight_at(&mut unet, &format!("{blk}.ff.linear1"));
    let got_mid = weight_at(&mut unet, "mid_block.attentions.0.proj_in");
    assert_eq!(
        base_l1.as_slice::<f32>(),
        got_l1.as_slice::<f32>(),
        "scale-0 GEGLU merge must be a bit-exact no-op"
    );
    assert_eq!(
        base_mid.as_slice::<f32>(),
        got_mid.as_slice::<f32>(),
        "scale-0 mid_block merge must be a bit-exact no-op"
    );
    println!("✓ scale-0 complete merge is a bit-exact no-op (mid + ff)");
}

#[test]
#[ignore = "needs the real SDXL snapshot + LCM-LoRA"]
fn complete_render_differs_from_vendored_and_is_sane() {
    let req = GenerationRequest {
        prompt: "a red fox in a forest, highly detailed".into(),
        negative_prompt: Some("blurry, low quality".into()),
        width: 512,
        height: 512,
        seed: Some(42),
        steps: Some(8),
        guidance: Some(7.0),
        ..Default::default()
    };
    let render = |complete: bool| -> Image {
        // Complete is now the load default; the vendored render opts back via SDXL_LORA_VENDORED.
        if complete {
            std::env::remove_var("SDXL_LORA_VENDORED");
        } else {
            std::env::set_var("SDXL_LORA_VENDORED", "1");
        }
        let spec =
            LoadSpec::new(WeightsSource::Dir(snapshot())).with_adapters(vec![lora_spec(1.0)]);
        let model = mlx_gen::load("sdxl", &spec).unwrap();
        let img = match model.generate(&req, &mut |_| {}).unwrap() {
            GenerationOutput::Images(mut v) => v.pop().unwrap(),
            other => panic!("expected Images, got {other:?}"),
        };
        std::env::remove_var("SDXL_LORA_VENDORED");
        img
    };

    let vend = render(false);
    let comp = render(true);

    assert_eq!(
        comp.pixels.len(),
        (512 * 512 * 3) as usize,
        "complete render must be a full RGB buffer"
    );
    assert!(
        comp.pixels.iter().any(|&p| p != 0),
        "complete render must not be all-black"
    );
    let diff = vend
        .pixels
        .iter()
        .zip(&comp.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    assert!(
        diff > 0,
        "complete coverage must change the render (the extra mid_block/ff signal is applied)"
    );
    println!("✓ complete vs vendored render differs at {diff} px (>8) — mid/ff signal is live");
}

/// Eyeball helper (not a parity gate): render the SAME prompt/seed at vendored (515) vs complete
/// (809) coverage and write both PNGs + an amplified abs-diff to `tools/golden/`, for a side-by-side
/// look at what signal the vendored path drops. Config overridable via `SDXL_{PROMPT,W,H,STEPS,CFG,SEED}`.
/// Run: cargo test -p mlx-gen-sdxl --release --test lora_complete_real_weights -- --ignored --nocapture dump_vendored_vs_complete_pngs
#[test]
#[ignore = "needs the real SDXL snapshot + LCM-LoRA — writes comparison PNGs"]
fn dump_vendored_vs_complete_pngs() {
    let env = |k: &str| std::env::var(k).ok();
    let prompt = env("SDXL_PROMPT").unwrap_or_else(|| {
        "a red fox in a snowy forest at dawn, highly detailed, photographic".into()
    });
    let w: u32 = env("SDXL_W").and_then(|s| s.parse().ok()).unwrap_or(1024);
    let h: u32 = env("SDXL_H").and_then(|s| s.parse().ok()).unwrap_or(1024);
    let steps: u32 = env("SDXL_STEPS").and_then(|s| s.parse().ok()).unwrap_or(8);
    let cfg: f32 = env("SDXL_CFG").and_then(|s| s.parse().ok()).unwrap_or(2.0);
    let seed: u64 = env("SDXL_SEED").and_then(|s| s.parse().ok()).unwrap_or(42);

    let req = GenerationRequest {
        prompt,
        negative_prompt: Some("blurry, low quality, deformed".into()),
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        guidance: Some(cfg),
        ..Default::default()
    };
    let render = |complete: bool| -> Image {
        // Complete is now the load default; the vendored render opts back via SDXL_LORA_VENDORED.
        if complete {
            std::env::remove_var("SDXL_LORA_VENDORED");
        } else {
            std::env::set_var("SDXL_LORA_VENDORED", "1");
        }
        let spec =
            LoadSpec::new(WeightsSource::Dir(snapshot())).with_adapters(vec![lora_spec(1.0)]);
        let model = mlx_gen::load("sdxl", &spec).unwrap();
        let img = match model.generate(&req, &mut |_| {}).unwrap() {
            GenerationOutput::Images(mut v) => v.pop().unwrap(),
            other => panic!("expected Images, got {other:?}"),
        };
        std::env::remove_var("SDXL_LORA_VENDORED");
        img
    };

    let vend = render(false);
    let comp = render(true);

    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/golden");
    std::fs::create_dir_all(&out).unwrap();
    let save = |name: &str, img: &Image| {
        let p = out.join(name);
        image::save_buffer(
            &p,
            &img.pixels,
            img.width,
            img.height,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
        p
    };
    let pv = save("sdxl_lora_vendored.png", &vend);
    let pc = save("sdxl_lora_complete.png", &comp);
    // Amplified (×4, clamped) per-pixel abs diff so the dropped signal is visible.
    let diff: Vec<u8> = vend
        .pixels
        .iter()
        .zip(&comp.pixels)
        .map(|(a, b)| (((*a as i32 - *b as i32).abs() * 4).min(255)) as u8)
        .collect();
    let pd = out.join("sdxl_lora_diff_x4.png");
    image::save_buffer(
        &pd,
        &diff,
        vend.width,
        vend.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();

    let n = vend
        .pixels
        .iter()
        .zip(&comp.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    println!("VENDORED (515): {}", pv.display());
    println!("COMPLETE (809): {}", pc.display());
    println!("DIFF ×4       : {}", pd.display());
    println!(
        "{w}x{h} {steps} steps cfg {cfg} seed {seed} — px>8 differing: {n}/{} ({:.1}%)",
        vend.pixels.len(),
        100.0 * n as f32 / vend.pixels.len() as f32
    );
}
