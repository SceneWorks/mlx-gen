//! sc-7570 — real-weight parity for the Krea 2 VAE decode. Krea 2's VAE is the Qwen-Image
//! `AutoencoderKLQwenImage`; this loads the published `krea/Krea-2-Turbo` `vae/` through the reused
//! [`mlx_gen_krea::load_vae`] and checks `decode` against the diffusers reference on the Krea snapshot's
//! own f32 weights.
//!
//! `#[ignore]` — needs the real snapshot + the golden (`tools/dump_krea_vae_golden.py`):
//! ```sh
//! KREA_TURBO_DIR=~/.cache/huggingface/hub/models--krea--Krea-2-Turbo/snapshots/<rev> \
//!   cargo test -p mlx-gen-krea --release --test vae_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_krea::load_vae;
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/krea_vae_real.safetensors"
);

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("KREA_TURBO_DIR").expect("set KREA_TURBO_DIR to the snapshot root"))
}

/// Peak- and mean-relative error vs the golden (mirrors `mlx-gen-qwen-image`'s VAE gates). Peak
/// `max|a-b|/max|b|`; mean `mean|a-b|/mean|b|`. Peak ≫ mean ⇒ a localized bug; peak ≈ mean ⇒
/// distributed f32 reduction-order accumulation (expected to grow with conv-net depth).
fn rel_errors(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let sum_abs_b: f64 = b.iter().map(|&v| v.abs() as f64).sum();
    let sum_abs_diff: f64 = a.iter().zip(b).map(|(&x, &y)| (x - y).abs() as f64).sum();
    (max_diff / peak, (sum_abs_diff / sum_abs_b) as f32)
}

#[test]
#[ignore = "needs real weights (KREA_TURBO_DIR) + golden (tools/dump_krea_vae_golden.py)"]
fn vae_decode_matches_real_reference() {
    let g = Weights::from_file(GOLDEN)
        .expect("golden — run tools/dump_krea_vae_golden.py with KREA_TURBO_DIR set");
    let vae = load_vae(snapshot()).expect("load real vae/ (diffusers→internal remap)");

    // diffusers' `AutoencoderKLQwenImage.decode().sample` (the golden) is **internally clamped to
    // [-1,1]**; the fork-faithful `QwenVae::decode` returns the raw (unclamped) decoder output, and the
    // Krea pipeline applies the clamp *after* decode (reference `sampling.py`: `img.clamp(-1,1)·0.5 +
    // 0.5` — the ideogram-pipeline precedent). Mirror that here so the comparison is apples-to-apples:
    // unclamped, this VAE diverges only on the ~7% of these random-latent pixels that saturate (peak
    // ~28%), which the clamp the pipeline applies regardless erases (→ peak ~5e-3 / mean ~9e-4).
    let out = vae.decode(g.require("in.latent").unwrap()).unwrap();
    let out = mlx_rs::ops::clip(&out, (&Array::from_f32(-1.0), &Array::from_f32(1.0))).unwrap();
    let want = g.require("out.image").unwrap();
    assert_eq!(
        out.shape(),
        want.shape(),
        "decode output shape (NCTHW, T=1)"
    );

    let (peak, mean) = rel_errors(&out, want);
    println!(
        "Krea 2 real-weight VAE decode (clamped): peak-rel = {peak:.3e}, mean-rel = {mean:.3e}"
    );
    // Same gates as the Qwen-Image VAE decode parity (the reused code): mean is the structural gate,
    // the looser peak tolerates the few pixels that diverge in f32 after upsample+conv.
    assert!(mean < 2e-3, "VAE decode mean-rel regressed: {mean:.3e}");
    assert!(peak < 1.5e-2, "VAE decode peak-rel regressed: {peak:.3e}");
}
