//! Wan-VACE end-to-end real-weight smoke (epic 3040 / sc-3388 S3 → sc-3467) — **weight-gated**
//! (`#[ignore]`, heavy: loads the real 1.3B transformer + 11 GB UMT5 + z16 VAE).
//!
//! Exercises the full native `WanVACEPipeline` path — UMT5 text encode → 96-ch control latent
//! (clip + mask + refs) → CFG VACE denoise → z16-VAE decode → RGB frames — on a real assembled
//! `wan_vace` snapshot. The numeric core is already locked component-wise: the transformer at real
//! 1.3B weights (`wanvace_real_parity.rs`, S1, max|Δ|≈9.5e-3 vs diffusers) and the conditioning host
//! ops byte-exact (`wanvace_cond_parity.rs`, S2); the shared UMT5/z16-VAE carry the base-Wan goldens
//! (`s1_t5_golden`, `convert_vae_parity`). So this gate's job is the **integration**: that the
//! assembled snapshot loads and the staged pipeline yields a coherent, correctly-shaped video — the
//! one thing component tests can't see (it is what surfaced the frame-count contract below).
//!
//! **Frame count is the engine's documented *non-causal* z16 decode (`vae.rs`: "T latent → 4·T
//! frames"), engine-wide for Wan T2V/I2V/VACE** (the `mlx_video` reference it ports). For an
//! `n`-frame control clip → `t_lat = (n-1)/4 + 1` latent frames → `4·t_lat` output frames (n=13 →
//! 4 → 16). diffusers `WanVACEPipeline` uses a *causal* VAE and returns exactly `n` (= `4·t_lat - 3`);
//! that ±3 boundary delta is the pre-existing engine-wide non-causal convention (shared with base Wan,
//! accepted in epic 3018), **not** a VACE-specific defect — tracked separately if causal output is
//! ever wanted.
//!
//! Snapshot: assembled by `mlx_gen_wan::convert::assemble_wan_vace_snapshot` (the diffusers VACE
//! `transformer/` + a converted base-Wan snapshot's shared `t5_encoder.safetensors` +
//! `vae.safetensors` + `tokenizer.json`). Point at an explicit dir with `WANVACE_DIR=/path/to/wan_vace`,
//! or let the test assemble from the local HF cache + a `~/.cache/mlx-gen-models/wan2_2_*_a14b_*` base
//! snapshot when `WANVACE_DIR` is unset. Run: `cargo test -p mlx-gen-wan --release --test wanvace_e2e
//! -- --ignored`.

use std::path::PathBuf;

use mlx_gen::{
    registry, Conditioning, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec,
    Progress, Quant, ReplacementMode, WeightsSource,
};
use mlx_gen_wan::convert::assemble_wan_vace_snapshot;
use mlx_gen_wan::MODEL_ID_VACE;

/// The VACE `transformer/` dir in the local HF cache (the one snapshot under `snapshots/<hash>/`).
fn cached_vace_transformer() -> Option<PathBuf> {
    let base = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub/models--Wan-AI--Wan2.1-VACE-1.3B-diffusers/snapshots");
    std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path().join("transformer")))
        .find(|p| p.join("config.json").is_file())
}

/// Resolve the snapshot: `WANVACE_DIR` if set, else assemble one from the local HF cache (the VACE
/// `transformer/`) + a converted base-Wan snapshot (the shared UMT5/z16-VAE/tokenizer). Returns
/// `None` (→ skip) when the ingredients aren't present locally.
fn resolve_or_assemble_snapshot() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("WANVACE_DIR") {
        return Some(PathBuf::from(d));
    }
    let tf = cached_vace_transformer()?;
    // Any converted base-Wan 14B dir carries the shared native components (t5/vae/tokenizer).
    let base = [
        ".cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16",
        ".cache/mlx-gen-models/wan2_2_i2v_a14b_mlx_bf16",
    ]
    .into_iter()
    .map(|r| PathBuf::from(std::env::var("HOME").unwrap()).join(r))
    .find(|p| p.join("t5_encoder.safetensors").is_file())?;
    let out = PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/mlx-gen-models/wan_vace_1_3b_mlx_bf16");
    assemble_wan_vace_snapshot(&out, &tf, &base, true).ok()
}

/// A solid mid-gray RGB frame.
fn frame(w: u32, h: u32) -> Image {
    Image {
        width: w,
        height: h,
        pixels: vec![128u8; (w * h * 3) as usize],
    }
}

/// A white (fully-active) mask frame — VACE regenerates the whole frame (pose/depth-control style).
fn mask_frame(w: u32, h: u32) -> Image {
    Image {
        width: w,
        height: h,
        pixels: vec![255u8; (w * h * 3) as usize],
    }
}

/// Build a small control-clip request (mid-gray clip + fully-active mask → regenerate the whole
/// frame), generate, and assert the non-causal frame count + coherence. `label` tags the log line.
fn run_vace_smoke(g: &dyn Generator, label: &str) {
    let (w, h, n) = (256u32, 256u32, 13usize); // n=13 = 1 + 4·3 → t_lat = 4
    let req = GenerationRequest {
        prompt: "a person walking".into(),
        width: w,
        height: h,
        frames: Some(n as u32),
        steps: Some(8),
        conditioning: vec![Conditioning::ControlClip {
            frames: (0..n).map(|_| frame(w, h)).collect(),
            mask: (0..n).map(|_| mask_frame(w, h)).collect(),
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::FaceOnly,
        }],
        ..Default::default()
    };

    let mut on_progress = |_p: Progress| {};
    let out = g.generate(&req, &mut on_progress).expect("vace generate");
    let GenerationOutput::Video { frames, .. } = out else {
        panic!("expected Video output, got {out:?}");
    };

    // Engine non-causal z16 decode: t_lat = (n-1)/4 + 1, output = 4·t_lat (see module docs).
    let t_lat = (n - 1) / 4 + 1;
    let expected = t_lat * 4;
    assert_eq!(
        frames.len(),
        expected,
        "[{label}] expected {expected} output frames (4·t_lat, non-causal), got {}",
        frames.len()
    );
    for (i, f) in frames.iter().enumerate() {
        assert_eq!((f.width, f.height), (w, h), "[{label}] frame {i} size");
        assert_eq!(
            f.pixels.len(),
            (w * h * 3) as usize,
            "[{label}] frame {i} pixel buffer"
        );
    }

    // Coherence (mirrors the base-Wan generate smoke): a real decode, not a degenerate constant.
    let all: Vec<u8> = frames
        .iter()
        .flat_map(|f| f.pixels.iter().copied())
        .collect();
    let mean = all.iter().map(|&b| b as f64).sum::<f64>() / all.len() as f64;
    assert!(
        (2.0..=253.0).contains(&mean),
        "[{label}] decoded pixel mean {mean:.1} is pinned at an extreme — VAE decode looks degenerate"
    );
    let identical = frames.windows(2).all(|p| p[0].pixels == p[1].pixels);
    assert!(
        !identical,
        "[{label}] all {} frames are byte-identical — temporal decode is degenerate",
        frames.len()
    );
    eprintln!(
        "wan_vace e2e [{label}] ok: {} frames @ {w}x{h}, pixel mean {mean:.1}",
        frames.len()
    );
}

#[test]
#[ignore = "needs a wan_vace snapshot — set WANVACE_DIR or have the VACE transformer + a base-Wan snapshot in the cache"]
fn wan_vace_generate_smoke() {
    let Some(dir) = resolve_or_assemble_snapshot() else {
        eprintln!(
            "skipping: no wan_vace snapshot (set WANVACE_DIR or populate the HF/mlx-gen cache)"
        );
        return;
    };
    let g = registry::load(MODEL_ID_VACE, &LoadSpec::new(WeightsSource::Dir(dir)))
        .expect("load wan_vace");
    run_vace_smoke(g.as_ref(), "dense-bf16");
}

/// sc-3440: the Q4/Q8 `WanVaceTransformer::quantize` path loads + generates a coherent video on real
/// weights (catches a broken quantize cascade / packed-Linear shape bug end-to-end).
#[test]
#[ignore = "needs a wan_vace snapshot — set WANVACE_DIR or have the VACE transformer + a base-Wan snapshot in the cache"]
fn wan_vace_quantized_generate_smoke() {
    let Some(dir) = resolve_or_assemble_snapshot() else {
        eprintln!(
            "skipping: no wan_vace snapshot (set WANVACE_DIR or populate the HF/mlx-gen cache)"
        );
        return;
    };
    let g = registry::load(
        MODEL_ID_VACE,
        &LoadSpec::new(WeightsSource::Dir(dir)).with_quant(Quant::Q8),
    )
    .expect("load wan_vace Q8");
    run_vace_smoke(g.as_ref(), "q8");
}
