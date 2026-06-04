//! sc-2963 invariant (rollout of the Wan sc-2957 template): the **compiled elementwise glue**
//! ([`set_compile_glue(true)`]) produces a forward **bit-identical** to the eager forward. `mx.compile`
//! fuses the SwiGLU activation, the gated residuals, the complex RoPE rotation, and (control only) the
//! hint injection into single kernels; the fusion must not perturb the result. Gated on the committed
//! tiny synthetic models — the **base** DiT forward AND the **control** DiT forward (control ON, so
//! `add_hint` is exercised) — in CI, no real checkpoint.

use mlx_gen::weights::Weights;
use mlx_gen_z_image::{
    set_compile_glue, ZImageControlTransformer, ZImageTransformer, ZImageTransformerConfig,
};
use mlx_rs::Array;

fn max_abs(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    a.as_slice::<f32>()
        .iter()
        .zip(b.as_slice::<f32>())
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
}

fn base_cfg() -> ZImageTransformerConfig {
    ZImageTransformerConfig {
        patch_size: 2,
        f_patch_size: 1,
        in_channels: 4,
        dim: 96,
        n_layers: 2,
        n_refiner_layers: 1,
        n_heads: 4,
        norm_eps: 1e-5,
        cap_feat_dim: 32,
        rope_theta: 256.0,
        t_scale: 1000.0,
        axes_dims: vec![8, 8, 8],
        axes_lens: vec![64, 64, 64],
        frequency_embedding_size: 256,
    }
}

fn control_cfg() -> ZImageTransformerConfig {
    ZImageTransformerConfig {
        dim: 64,
        n_layers: 4,
        n_refiner_layers: 2,
        axes_dims: vec![8, 4, 4],
        ..base_cfg()
    }
}

#[test]
fn compiled_glue_bit_identical_to_eager_base() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/z_transformer.safetensors"
    );
    let w = Weights::from_file(path).unwrap();
    let model = ZImageTransformer::from_weights(&w, "w", base_cfg()).unwrap();
    let x = w.require("in.x").unwrap();
    let cap = w.require("in.cap_feats").unwrap();

    set_compile_glue(false);
    let eager = model.forward(x, 0.7, cap).unwrap();
    set_compile_glue(true);
    let compiled = model.forward(x, 0.7, cap).unwrap();
    set_compile_glue(false);

    let d = max_abs(&compiled, &eager);
    println!("[z-image base compiled vs eager] max|Δ|={d:.3e}");
    assert_eq!(d, 0.0, "Z-Image base compiled glue diverged from eager");
}

#[test]
fn compiled_glue_bit_identical_to_eager_control() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/z_control_transformer.safetensors"
    );
    let w = Weights::from_file(path).unwrap();
    let base = ZImageTransformer::from_weights(&w, "w", control_cfg()).unwrap();
    let model = ZImageControlTransformer::from_weights(base, &w, "w").unwrap();
    let x = w.require("in.x").unwrap();
    let cap = w.require("in.cap_feats").unwrap();
    let cc = w.require("in.control_context").unwrap();

    // Control ON (scale 1.0) so the control branch + `add_hint` are exercised.
    set_compile_glue(false);
    let eager = model.forward(x, 0.7, cap, Some(cc), 1.0).unwrap();
    set_compile_glue(true);
    let compiled = model.forward(x, 0.7, cap, Some(cc), 1.0).unwrap();
    set_compile_glue(false);

    let d = max_abs(&compiled, &eager);
    println!("[z-image control compiled vs eager] max|Δ|={d:.3e}");
    assert_eq!(d, 0.0, "Z-Image control compiled glue diverged from eager");
}
