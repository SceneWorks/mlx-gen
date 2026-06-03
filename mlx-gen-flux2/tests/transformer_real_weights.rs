//! sc-2346 S3: real-weights smoke for the FLUX.2 MMDiT transformer. `#[ignore]`d — needs the real
//! `black-forest-labs/FLUX.2-klein-9b` snapshot:
//!
//!   cargo test -p mlx-gen-flux2 --test transformer_real_weights -- --ignored --nocapture
//!
//! The committed `transformer_parity.rs` proves the forward *math* bit-tight in f32 on a tiny
//! config; this proves the *loader* on the real checkpoint — every key + the diffusers→internal
//! remaps (`attn.to_out.0`→`to_out`, `timestep_embedder.linear_{1,2}`→`linear_{1,2}`) resolve, and
//! a real-dimension forward runs finite. End-to-end numerical parity on real weights is closed by
//! the S4 e2e pixel test.

use std::path::PathBuf;

use mlx_gen_flux2::{create_noise, load_transformer, prepare_grid_ids, prepare_text_ids};
use mlx_rs::{random, Dtype};

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9b/snapshots");
    std::fs::read_dir(&snaps)
        .expect("snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot (~17 GB transformer)"]
fn transformer_loads_and_forwards() {
    let t = load_transformer(&snapshot()).expect("all transformer keys + remaps resolve");

    // 64×64 → lat 4×4 = 16 image tokens; a short 8-token prompt.
    let (w, h) = (64u32, 64u32);
    let hidden = create_noise(0, w, h, 128).unwrap(); // [1,16,128]
    let key = random::key(1).unwrap();
    let encoder = random::normal::<f32>(&[1, 8, 12288][..], None, None, Some(&key)).unwrap();
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
        "transformer output is non-finite: {total}"
    );
    println!(
        "flux2 transformer real-weights forward OK: shape {:?}, finite",
        out.shape()
    );
}
