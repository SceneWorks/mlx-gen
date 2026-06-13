//! sc-4706 seam: the Bernini renderer's packed forward path
//! ([`WanTransformer::patch_embed_tokens`] → [`WanTransformer::forward_packed`] → `unpatchify`) must
//! be **bit-identical** to the standard single-latent [`WanTransformer::forward`] for a target-only
//! (source_id 0, plain spatial RoPE) latent. This pins the seam: at batch 1 with one token segment
//! the packed forward reduces exactly to the validated dense forward, so any Bernini divergence comes
//! only from the added conditioning sources / source-id RoPE / APG — not from the forward plumbing.
//!
//! Runs in CI on the tiny seeded S5 fixture (no real checkpoint).

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::patchify::unpatchify;
use mlx_gen_wan::WanTransformer;

fn tiny_cfg() -> WanModelConfig {
    let mut c = WanModelConfig::wan21_t2v_1_3b();
    c.dim = 128;
    c.num_heads = 1;
    c.num_layers = 2;
    c.ffn_dim = 256;
    c.freq_dim = 256;
    c.text_dim = 32;
    c.text_len = 8;
    c.in_dim = 16;
    c.out_dim = 16;
    c.vae_z_dim = 16;
    c.boundary = 0.875;
    c.num_train_timesteps = 1000;
    c
}

fn load(name: &str) -> Weights {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    Weights::from_file(&path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (run dump_s5_fixtures.py)"))
}

fn max_abs(got: &[f32], exp: &[f32]) -> f32 {
    got.iter()
        .zip(exp.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max)
}

#[test]
fn forward_packed_matches_dense_forward_for_target_only() {
    let w = load("s5_low.safetensors");
    let cfg = tiny_cfg();
    let dit = WanTransformer::from_weights(&w, &cfg).expect("DiT");

    let latent = w.require("init_noise").unwrap(); // [16, 2, 2, 2]
    let ctx = dit.embed_text(w.require("ctx_cond").unwrap()).unwrap(); // [1, text_len, dim]
    let t = 833.0f32;

    // Reference: the validated single-latent forward → [out, F, H, W].
    let dense = dit.forward(latent, t, &ctx).unwrap();

    // Packed seam: patch-embed → packed forward (full self-attention over one segment) → unpatchify.
    let (tokens, grid) = dit.patch_embed_tokens(latent).unwrap();
    let (cos, sin) = dit.prepare_rope(grid).unwrap();
    let cross_kv = dit.prepare_cross_kv(&ctx).unwrap();
    let packed = dit
        .forward_packed(&tokens, t, &cross_kv, &cos, &sin)
        .unwrap(); // [1, L, out·∏patch]
    let l = (grid.0 * grid.1 * grid.2) as i32;
    let op = packed.shape()[2];
    let packed = packed.reshape(&[l, op]).unwrap();
    let packed_out = unpatchify(&packed, grid, cfg.out_dim, cfg.patch_size).unwrap();

    assert_eq!(packed_out.shape(), dense.shape());
    let d = max_abs(packed_out.as_slice::<f32>(), dense.as_slice::<f32>());
    println!("[forward_packed vs forward] max|Δ| = {d:.3e}");
    assert_eq!(
        d, 0.0,
        "packed forward must be bit-identical to the dense forward at source_id 0"
    );
}
