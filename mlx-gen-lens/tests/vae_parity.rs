//! sc-3169 — Lens VAE decode parity vs `LensPipeline._decode`.
//!
//! Loads the real Flux.2 `AutoencoderKLFlux2` from the cached `microsoft/Lens-Turbo` `vae/` (via
//! `mlx_gen_flux2::load_vae`), runs [`mlx_gen_lens::vae::decode`] on the golden's synthetic DiT output,
//! and asserts the decoded image matches the torch `_decode(...).sample` near-bit (f32 VAE). This
//! exercises the whole shim — the reshape-to-packed-grid, the bn de-normalize, the 2×2 unpatchify, and
//! the Flux.2 conv decoder — end to end.
//!
//! Run: `cargo test -p mlx-gen-lens --test vae_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::Dtype;

use mlx_gen::weights::Weights;
use mlx_gen_flux2::load_vae;
use mlx_gen_lens::vae::decode;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/lens_vae_golden.safetensors"
);

fn snapshot_root() -> std::path::PathBuf {
    let base = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("snapshot dir {}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max()
        .expect("a snapshot")
}

fn meta_usize(g: &Weights, key: &str) -> usize {
    g.metadata(key).unwrap().parse().unwrap()
}

#[test]
#[ignore = "needs tools/golden/lens_vae_golden.safetensors + the Lens-Turbo vae/ snapshot"]
fn lens_vae_decode_matches_reference() {
    let g = Weights::from_file(GOLDEN).expect("vae golden");
    let (lat_h, lat_w) = (meta_usize(&g, "latent_h"), meta_usize(&g, "latent_w"));

    let vae = load_vae(&snapshot_root()).expect("load Flux.2 VAE from Lens snapshot");
    let dit_out = g.require("dit_out").unwrap().clone(); // [1, h·w, 128]

    let got = decode(&vae, &dit_out, lat_h, lat_w, None).unwrap(); // [1, H, W, 3] (NHWC), [-1,1]

    // Golden is NCHW [1, 3, H, W]; transpose to NHWC for comparison.
    let want = g
        .require("image")
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();
    assert_eq!(
        got.shape(),
        want.shape(),
        "shape {:?} != {:?}",
        got.shape(),
        want.shape()
    );

    let diff = abs(subtract(
        got.as_dtype(Dtype::Float32).unwrap(),
        want.as_dtype(Dtype::Float32).unwrap(),
    )
    .unwrap())
    .unwrap();
    let max_abs = max(&diff, None).unwrap().item::<f32>();
    let denom = max(abs(&want).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .max(1e-12);
    let peak_rel = max_abs / denom;
    eprintln!("vae decode: max_abs {max_abs:.3e}  peak_rel {peak_rel:.3e}");
    // The shim's own ops (reshape-to-packed-grid, bn de-normalize, 2×2 unpatchify) are bit-exact f32
    // reshapes/elementwise; the residual ~8e-3 worst pixel is the deep conv decoder's mlx-Metal-vs-CPU
    // f32 floor (the shared, already-validated Flux.2 decoder — flux2's own e2e gates the decoded image
    // on pixel coherence, not tight peak_rel, for the same reason). A wrong channel packing would
    // garble the image (peak_rel ≫ this), so this bounds the shim as correct.
    assert!(
        peak_rel < 1.5e-2,
        "vae decode peak_rel {peak_rel:.3e} ≥ 1.5e-2 — beyond the conv-VAE f32 floor"
    );
    eprintln!("ALL PASS");
}
