//! sc-4705: Bernini renderer weight conversion — the full `ByteDance/Bernini-Diffusers` combined
//! `bernini/` index → a native MLX dual-expert snapshot the existing `wan2_2_t2v_14b` provider loads.
//!
//! Real weights (~168 GB package + a converted `wan2_2_t2v_a14b` base snapshot) live outside CI, so
//! the end-to-end assembly is `#[ignore]`. It proves the diffusers→internal bijection on the actual
//! finetuned weights: each emitted expert's internal key set must equal the native conversion's, the
//! 16-channel patch-embed IO must hold, the dtype must be bf16, the weights must load into
//! `WanTransformer`, and a forward must produce finite output. (Value-level torch parity is sc-4706.)

use std::collections::BTreeSet;
use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::convert::assemble_bernini_renderer_snapshot;
use mlx_gen_wan::WanTransformer;
use mlx_rs::{random, Dtype};

/// The newest snapshot dir of an HF-cached repo, or `None` if it is not cached.
fn hf_snapshot(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(snaps)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir())
}

fn key_set(w: &Weights) -> BTreeSet<String> {
    w.keys().map(str::to_string).collect()
}

#[test]
#[ignore = "real weights: ~168 GB Bernini-Diffusers + a converted wan2_2_t2v_a14b base snapshot"]
fn assemble_real_bernini_renderer() {
    let pkg = hf_snapshot("ByteDance/Bernini-Diffusers")
        .expect("ByteDance/Bernini-Diffusers snapshot in the HF cache");
    let base = PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16");
    assert!(
        base.join("high_noise_model.safetensors").is_file(),
        "converted base Wan2.2-T2V-A14B snapshot required at {}",
        base.display()
    );

    let out = std::env::temp_dir().join("bernini_renderer_mlx_test");
    let _ = std::fs::remove_dir_all(&out);
    assemble_bernini_renderer_snapshot(&out, &pkg, &base, None, true).expect("assemble");

    // Sidecar + loadable config present.
    assert!(out.join("config.json").is_file());
    assert!(out.join("bernini_renderer.json").is_file());

    let cfg = WanModelConfig::wan22_t2v_14b();
    for name in [
        "high_noise_model.safetensors",
        "low_noise_model.safetensors",
    ] {
        let w = Weights::from_file(out.join(name)).expect("load converted expert");
        let want = key_set(&Weights::from_file(base.join(name)).expect("load base expert"));
        let got = key_set(&w);
        assert_eq!(
            got, want,
            "{name}: internal key set must match the native conversion exactly"
        );
        assert_eq!(got.len(), 1095, "{name}: expected 1095 tensors per expert");

        // 16-channel latent IO (patch embed [dim, in·patch = 16·4 = 64]) + bf16.
        let pe = w.require("patch_embedding_proj.weight").unwrap();
        assert_eq!(pe.shape(), [5120, 64], "{name}: 16-ch patch-embed IO");
        assert_eq!(pe.dtype(), Dtype::Bfloat16, "{name}: bf16");

        // Loads into the model: every required key present, every shape accepted.
        WanTransformer::from_weights(&w, &cfg).expect("from_weights");
    }

    // Forward smoke on the high-noise expert: the Bernini-finetuned weights compute finite output
    // through the existing Wan2.2 DiT forward.
    let w = Weights::from_file(out.join("high_noise_model.safetensors")).unwrap();
    let dit = WanTransformer::from_weights(&w, &cfg).unwrap();
    let key = random::key(0).unwrap();
    let latent = random::normal::<f32>(&[16, 2, 4, 4], None, None, Some(&key)).unwrap();
    let raw_ctx = random::normal::<f32>(&[4, cfg.text_dim as i32], None, None, Some(&key)).unwrap();
    let ctx = dit.embed_text(&raw_ctx).unwrap();
    let pred = dit.forward(&latent, 875.0, &ctx).unwrap();
    assert_eq!(
        pred.shape(),
        latent.shape(),
        "forward preserves latent shape"
    );
    let pred = pred.as_dtype(Dtype::Float32).unwrap();
    mlx_rs::transforms::eval([&pred]).unwrap();
    assert!(
        pred.as_slice::<f32>().iter().all(|x| x.is_finite()),
        "forward output must be finite"
    );
}
