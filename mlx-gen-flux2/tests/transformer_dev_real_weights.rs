//! sc-5916: real-weights smoke for the FLUX.2-**dev** MMDiT transformer. `#[ignore]`d — needs the
//! real `black-forest-labs/FLUX.2-dev` snapshot:
//!
//!   cargo test -p mlx-gen-flux2 --test transformer_dev_real_weights -- --ignored --nocapture
//!
//! The parametric transformer math is already proven bit-tight in f32 by the committed
//! `transformer_parity.rs` (tiny config) and is dimension-agnostic, so this proves the **dev
//! loader** on the real checkpoint — every key + the diffusers→internal remaps resolve at the dev
//! dims (8 double / 48 single blocks, 48 heads, `joint_attention_dim` 15360), and a real-dimension
//! forward runs finite. End-to-end numerical parity on real dev weights is closed by the dev T2I
//! e2e pixel test (sc-2365).

use std::path::PathBuf;

use mlx_gen_flux2::{create_noise, load_transformer_dev, prepare_grid_ids, prepare_text_ids};
use mlx_rs::{random, Dtype};

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_DEV_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots");
    std::fs::read_dir(&snaps)
        .expect("snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir under models--black-forest-labs--FLUX.2-dev/snapshots")
}

#[test]
#[ignore = "needs real FLUX.2-dev snapshot (~32B / ~64 GB transformer)"]
fn dev_transformer_loads_and_forwards() {
    // Proves the dev key map + diffusers remaps resolve at the dev dims (joint 15360, 48 heads, 48
    // single blocks) and a real-dimension forward runs finite.
    let t = load_transformer_dev(&snapshot()).expect("all dev transformer keys + remaps resolve");

    // 64×64 → lat 4×4 = 16 image tokens; a short 8-token prompt at the dev joint width 15360.
    let (w, h) = (64u32, 64u32);
    let hidden = create_noise(0, w, h, 128).unwrap(); // [1,16,128]
    let key = random::key(1).unwrap();
    let encoder = random::normal::<f32>(&[1, 8, 15360][..], None, None, Some(&key)).unwrap();
    let img_ids = prepare_grid_ids((h / 16) as usize, (w / 16) as usize, 0);
    let txt_ids = prepare_text_ids(8);

    let out = t
        .forward(&hidden, &encoder, &img_ids, &txt_ids, 500.0)
        .unwrap();
    assert_eq!(out.shape(), &[1, 16, 128], "velocity shape");
    let total = out
        .as_dtype(Dtype::Float32)
        .unwrap()
        .sum(None)
        .unwrap()
        .item::<f32>();
    assert!(
        total.is_finite(),
        "dev transformer output is non-finite: {total}"
    );
    println!(
        "flux2-dev transformer real-weights forward OK: shape {:?}, finite",
        out.shape()
    );
}
