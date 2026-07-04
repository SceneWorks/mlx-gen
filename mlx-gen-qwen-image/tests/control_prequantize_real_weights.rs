//! sc-9517 (epic 9083): maintainer's on-device proof that a **pre-quantized packed** 2512-Fun
//! control tier — built by [`mlx_gen_qwen_image::convert::quantize_qwen_control_branch`] — packs the
//! right scope and loads via the packed-detect loader
//! ([`mlx_gen_qwen_image::loader::load_controlnet`]). This is the SHARED MLX tier the candle lane
//! (epic 9083) also consumes via `candle_gen::quant`.
//!
//! The pack is verified **byte-for-byte against the load-time quantize** (the sc-8670 round-trip:
//! pre-quantize-on-disk == quantize-at-load), so no ~40 GB Qwen-Image-2512 base and no full render is
//! needed here — byte-identity to the load-time control path transitively inherits its already
//! on-device-validated pose/canny/depth render (sc-8267 / sc-8350).
//!
//! `#[ignore]`d — needs the alibaba-pai `Qwen-Image-2512-Fun-Controlnet-Union` checkpoint (the
//! `-2602.safetensors` overlay, ~3.3 GB bf16). Run with:
//!   cargo test -p mlx-gen-qwen-image --release --test control_prequantize_real_weights -- --ignored --nocapture
//!
//! Env knobs:
//!   SC9517_CONTROL  the `-2602.safetensors` overlay (default: the HF-cache copy)
//!   SC9517_OUT      output dir for the packed tier (default: a scratch dir in the temp dir)
//!   SC9517_BITS     4 (Q4, default) or 8 (Q8)
//!   SC9517_KEEP     if set, keep the built tier (else it is removed after the checks)

use mlx_gen::quant::packed_bits;
use mlx_gen::weights::Weights;
use mlx_gen::WeightsSource;
use mlx_gen_qwen_image::convert::quantize_qwen_control_branch;
use mlx_gen_qwen_image::loader::load_controlnet;
use mlx_rs::ops::{array_eq, quantize};
use mlx_rs::Dtype;
use std::path::PathBuf;

/// The canonical MLX overlay file inside the alibaba-pai repo (sc-8350: the repo ships two overlays —
/// the `-2602` variant is the one the MLX engine loads, so it is the one we pack).
const OVERLAY_FILE: &str = "Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors";
/// Codebase-default group size (== `crate::quant::GROUP_SIZE`).
const GROUP_SIZE: i32 = 64;

/// Resolve the 2512-Fun `-2602` overlay: `SC9517_CONTROL` if set, else the HF-cache copy.
fn control_overlay() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SC9517_CONTROL") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home).join(
        ".cache/huggingface/hub/\
         models--alibaba-pai--Qwen-Image-2512-Fun-Controlnet-Union/snapshots",
    );
    let snap = std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())?;
    let f = snap.join(OVERLAY_FILE);
    f.is_file().then_some(f)
}

fn bits_env() -> i32 {
    std::env::var("SC9517_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
}

/// Build-only harness for producing the **hostable** control tier (epic 9083 rollout): pack the
/// overlay into `SC9517_OUT` and keep it — no load (fast, variant-agnostic). Run per tier:
///   SC9517_CONTROL=<overlay> SC9517_OUT=<staging/control-q4> SC9517_BITS=4 \
///     cargo test -p mlx-gen-qwen-image --release --test control_prequantize_real_weights \
///     -- --ignored build_control_tier_only --nocapture
#[test]
#[ignore = "build-only control-tier producer for hosting; set SC9517_CONTROL/OUT/BITS"]
fn build_control_tier_only() {
    let src = control_overlay().expect("SC9517_CONTROL (or the HF-cache -2602 overlay) required");
    let out =
        PathBuf::from(std::env::var("SC9517_OUT").expect("SC9517_OUT (tier output dir) required"));
    let bits = bits_env();
    println!(
        "building Q{bits} control tier: {} -> {}",
        src.display(),
        out.display()
    );
    quantize_qwen_control_branch(&src, &out, bits).expect("quantize_qwen_control_branch succeeds");
    let mf = out.join("model.safetensors");
    let sz = std::fs::metadata(&mf)
        .expect("packed model.safetensors present")
        .len();
    println!(
        "✓ built {} ({:.3} GB packed control branch)",
        out.display(),
        sz as f64 / 1e9
    );
}

#[test]
#[ignore = "needs the 2512-Fun control checkpoint; builds + verifies a packed control tier"]
fn control_tier_packs_scope_and_loads_packed() {
    let Some(src) = control_overlay() else {
        eprintln!("skip: no 2512-Fun overlay (set SC9517_CONTROL or populate the HF cache)");
        return;
    };
    let bits = bits_env();
    let out = std::env::var("SC9517_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join(format!("qwen-2512-fun-control-q{bits}")));

    println!(
        "building Q{bits} control tier: {} -> {}",
        src.display(),
        out.display()
    );
    quantize_qwen_control_branch(&src, &out, bits).expect("quantize_qwen_control_branch succeeds");

    // The packed branch must be a single `model.safetensors` (the loader globs `*.safetensors`).
    let mf = out.join("model.safetensors");
    assert!(mf.is_file(), "missing packed model.safetensors");
    println!(
        "  model.safetensors = {:.3} GB packed",
        std::fs::metadata(&mf).unwrap().len() as f64 / 1e9
    );

    // (1) Pack SCOPE — the same dense/quantized split as `QwenFunControlBranch::quantize`:
    // `control_img_in` (132-in, `% 64 != 0`) + the 1-D attn RMSNorms stay dense; every control-block
    // Linear + before/after_proj packs (a `.scales` sibling, u32 codes, inferred bits == requested).
    let packed = Weights::from_dir(&out).expect("load packed tier");
    assert!(
        packed.get("control_img_in.scales").is_none(),
        "control_img_in must stay dense (132 in-features)"
    );
    assert!(
        packed.get("control_blocks.0.attn.norm_q.scales").is_none(),
        "the 1-D attn RMSNorm must stay dense"
    );
    for base in [
        "control_blocks.0.before_proj",
        "control_blocks.4.after_proj",
        "control_blocks.0.attn.to_q",
        "control_blocks.0.attn.add_k_proj",
        "control_blocks.0.img_mlp.net.0.proj",
        "control_blocks.0.img_mod.1",
    ] {
        let scales = packed
            .get(&format!("{base}.scales"))
            .unwrap_or_else(|| panic!("{base} must be packed (missing .scales)"));
        let wq = packed.get(&format!("{base}.weight")).unwrap();
        assert_eq!(wq.dtype(), Dtype::Uint32, "{base}: packed codes are u32");
        assert_eq!(
            packed_bits(wq, scales, GROUP_SIZE).unwrap(),
            bits,
            "{base}: inferred bit-width"
        );
    }

    // (2) Byte-identity to the load-time quantize (sc-8670) on a real block Linear: the packed
    // before_proj triple equals `quantize(source.bf16, 64, bits)` exactly — so the tier renders
    // identically to the (validated) load-time control path, no golden / base render required.
    let dense = Weights::from_file(&src).expect("load dense overlay");
    let src_w = dense.get("control_blocks.0.before_proj.weight").unwrap();
    let (ewq, esc, ebi) =
        quantize(src_w.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, bits).unwrap();
    let pwq = packed.get("control_blocks.0.before_proj.weight").unwrap();
    let psc = packed.get("control_blocks.0.before_proj.scales").unwrap();
    let pbi = packed.get("control_blocks.0.before_proj.biases").unwrap();
    assert!(
        array_eq(pwq, &ewq, false).unwrap().item::<bool>(),
        "packed weight != load-time quantize (sc-8670 round-trip broken)"
    );
    assert!(
        array_eq(psc, &esc, false).unwrap().item::<bool>(),
        "packed scales != load-time quantize"
    );
    assert!(
        array_eq(pbi, &ebi, false).unwrap().item::<bool>(),
        "packed biases != load-time quantize"
    );

    // (3) Packed-detect fires at the BRANCH level: `load_controlnet` on the tier reports the packed
    // bits (F-076 accessor), and the dense overlay + load-time quantize reports the same — parity.
    let branch =
        load_controlnet(&WeightsSource::Dir(out.clone())).expect("load packed control tier");
    assert_eq!(
        branch.packed_bits(),
        Some(bits),
        "packed-detect must fire on the loaded control tier"
    );
    let mut dense_branch =
        load_controlnet(&WeightsSource::File(src.clone())).expect("load dense overlay");
    assert_eq!(
        dense_branch.packed_bits(),
        None,
        "dense overlay is not packed"
    );
    dense_branch.quantize(bits).expect("load-time quantize");
    assert_eq!(
        dense_branch.packed_bits(),
        Some(bits),
        "load-time quantize reports the requested bits"
    );

    println!(
        "✓ packed Q{bits} 2512-Fun control tier: scope + byte-identity + packed-detect all pass"
    );

    if std::env::var("SC9517_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&out);
        println!("  removed {} (set SC9517_KEEP to retain)", out.display());
    }
}
