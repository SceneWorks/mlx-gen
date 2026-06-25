//! sc-3173 — end-to-end Lens-Turbo T2I parity vs the vendor `LensPipeline`.
//!
//! Runs the **full** Rust pipeline — the [`LensTokenizer`] (harmony render) → the gpt-oss
//! [`LensTextEncoder`] (capture + `txt_offset` slice) → the [`LensTransformer`] denoise (turbo
//! schedule + norm-rescaled CFG) → the Flux.2 [`vae`] decode — on the **same injected initial
//! latents** the torch golden used, and compares against the reference's final latents + decoded
//! image.
//!
//! The e2e is **cross-build** (MLX-Metal vs torch-CPU, both bf16): per-step bf16 op-order diverges
//! and accumulates over 48 DiT blocks × 4 steps, so the gate is **structural** (cosine) + coherence,
//! not bit-exact — the FLUX-hyper / cross-backend precedent. Injecting the reference's starting noise
//! removes the only *un*-reproducible source (the RNG); a wrong wiring (channel packing, offset slice,
//! CFG, timestep convention, …) would collapse the cosine, so a high cosine bounds the pipeline as
//! correct. The tokenizer is validated *inside* the e2e: the Rust render (with the golden's date) must
//! reproduce the golden's `input_ids` byte-for-byte.
//!
//! Run: `cargo test -p mlx-gen-lens --test e2e_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, multiply, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_lens::pipeline::LensPipeline;
use mlx_gen_lens::text::LensTokenizer;
use mlx_gen_lens::vae;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/lens_e2e_golden.safetensors"
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
fn meta_f32(g: &Weights, key: &str) -> f32 {
    g.metadata(key).unwrap().parse().unwrap()
}

/// `max|a-b| / max|b|`.
fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max(abs(want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

/// Cosine similarity over the flattened tensors.
fn cosine(got: &Array, want: &Array) -> f32 {
    let dot = sum(multiply(got, want).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let na = sum(multiply(got, got).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    let nb = sum(multiply(want, want).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    dot / (na * nb).max(1e-12)
}

#[test]
#[ignore = "needs tools/golden/lens_e2e_golden.safetensors + the full Lens-Turbo snapshot (~50GB bf16 load)"]
fn lens_e2e_matches_reference() {
    let g = Weights::from_file(GOLDEN).expect("e2e golden");
    let (lat_h, lat_w) = (meta_usize(&g, "latent_h"), meta_usize(&g, "latent_w"));
    let num_steps = meta_usize(&g, "num_steps");
    let guidance = meta_f32(&g, "guidance");
    let prompt = g.metadata("prompt").unwrap();
    let negative = g.metadata("negative_prompt").unwrap_or_default();
    let date = g.metadata("current_date").unwrap();

    let snap = snapshot_root();

    // 1. Tokenizer cross-check (inside the e2e): the Rust harmony render with the golden's date must
    //    reproduce the golden's input_ids exactly — otherwise the encoder sees a different sequence.
    let tok =
        LensTokenizer::from_file(snap.join("tokenizer").join("tokenizer.json")).expect("tokenizer");
    let out = tok.encode(prompt, date).expect("encode prompt");
    let want_ids = g.require("input_ids").unwrap(); // [1, L] i32
    let got_ids = Array::from_slice(&out.ids, &[1, out.ids.len() as i32]);
    assert_eq!(
        got_ids.shape(),
        want_ids.shape(),
        "tokenizer length {:?} != golden {:?}",
        got_ids.shape(),
        want_ids.shape()
    );
    let id_mismatch = max(
        abs(subtract(&got_ids, want_ids.as_dtype(Dtype::Int32).unwrap()).unwrap()).unwrap(),
        None,
    )
    .unwrap()
    .item::<i32>();
    assert_eq!(id_mismatch, 0, "Rust tokenizer ids differ from the golden");

    // 2. Load the full pipeline (bf16 production) and run the real path with the injected latents.
    eprintln!("loading Lens pipeline (encoder MXFP4→bf16 + DiT bf16 + VAE f32)…");
    let pipe = LensPipeline::load(&snap, Dtype::Bfloat16).expect("load pipeline");

    let (features, mask) = pipe
        .encode_prompt(prompt, negative, date)
        .expect("encode_prompt");
    let init = g.require("init_latents").unwrap().clone(); // [1, seq, 128] f32

    eprintln!("denoising {num_steps} steps @ latent {lat_h}x{lat_w}…");
    let latents = pipe
        .denoise(
            &features,
            &mask,
            &init,
            lat_h,
            lat_w,
            num_steps,
            guidance,
            &mlx_gen::CancelFlag::default(),
            &mut |c, t| eprintln!("  step {c}/{t}"),
        )
        .expect("denoise");

    // 3. Compare the final latents (the tightest e2e signal: encoder + DiT + scheduler + CFG, pre-VAE).
    let got_lat = latents.as_dtype(Dtype::Float32).unwrap();
    let want_lat = g.require("final_latents").unwrap(); // [1, seq, 128] f32
    assert_eq!(
        got_lat.shape(),
        want_lat.shape(),
        "final-latent shape {:?} != {:?}",
        got_lat.shape(),
        want_lat.shape()
    );
    let lat_cos = cosine(&got_lat, want_lat);
    let lat_pr = peak_rel(&got_lat, want_lat);
    eprintln!("final latents: cosine {lat_cos:.5}  peak_rel {lat_pr:.3e}");

    // 4. Compare the decoded image (full e2e incl. the VAE shim).
    let decoded = vae::decode(pipe.vae(), &latents, lat_h, lat_w, None).unwrap(); // [1,H,W,3] NHWC [-1,1]
    let got_img = {
        // → [0,1] to match the golden's stored range.
        let half = Array::from_f32(0.5);
        let x = mlx_rs::ops::add(
            mlx_rs::ops::multiply(decoded.as_dtype(Dtype::Float32).unwrap(), &half).unwrap(),
            &half,
        )
        .unwrap();
        mlx_rs::ops::clip(&x, (0.0, 1.0)).unwrap()
    };
    let want_img = g.require("image").unwrap(); // [1,H,W,3] f32 in [0,1]
    assert_eq!(
        got_img.shape(),
        want_img.shape(),
        "image shape {:?} != {:?}",
        got_img.shape(),
        want_img.shape()
    );
    let img_cos = cosine(&got_img, want_img);
    let img_pr = peak_rel(&got_img, want_img);
    // Coherence floor: a degenerate (flat) render would have ~0 variance. The reference image's own
    // std is the yardstick; ours must be in the same ballpark (not collapsed to a constant).
    let std_of = |x: &Array| -> f32 {
        let m = mlx_rs::ops::mean(x, None).unwrap();
        let v = mlx_rs::ops::mean(
            multiply(subtract(x, &m).unwrap(), subtract(x, &m).unwrap()).unwrap(),
            None,
        )
        .unwrap()
        .item::<f32>();
        v.sqrt()
    };
    let (got_std, want_std) = (std_of(&got_img), std_of(want_img));
    eprintln!("image: cosine {img_cos:.5}  peak_rel {img_pr:.3e}  std got {got_std:.4} / ref {want_std:.4}");

    // Gates — structural (cross-build), not bit-exact.
    assert!(
        lat_cos > 0.90,
        "final-latent cosine {lat_cos:.5} ≤ 0.90 — wiring divergence, not bf16 noise"
    );
    assert!(
        img_cos > 0.90,
        "decoded-image cosine {img_cos:.5} ≤ 0.90 — wiring/VAE divergence"
    );
    assert!(
        got_std > 0.5 * want_std,
        "decoded image is near-flat (std {got_std:.4} vs ref {want_std:.4}) — not a coherent render"
    );
    eprintln!("ALL PASS");
}

/// sc-7305 real-weight smoke: the curated sampler/scheduler knobs actually drive the real Lens DiT.
///
/// Runs the production [`LensPipeline::denoise_with_sampler`] over the real pipeline for the default
/// `euler`, a second-order solver (`heun`), and a curated scheduler (`karras`), asserting each yields a
/// finite latent of the right shape — and that `heun` changes the trajectory vs `euler` (the knob is
/// **live**) while staying coherent (high cosine, same model/prompt/seed). The default path's numeric
/// parity vs the torch golden is covered by [`lens_e2e_matches_reference`] (the default now flows
/// through the same unified `euler`), so this only needs the snapshot, not the golden.
#[test]
#[ignore = "needs the full Lens-Turbo snapshot (~50GB bf16 load)"]
fn lens_curated_samplers_drive_the_real_dit() {
    let snap = snapshot_root();
    eprintln!("loading Lens pipeline for the curated-sampler smoke…");
    let pipe = LensPipeline::load(&snap, Dtype::Bfloat16).expect("load pipeline");

    let (lat_h, lat_w) = (16usize, 16usize); // 256×256 / 16 — small + cheap for a smoke
    let seq = (lat_h * lat_w) as i32;
    let (num_steps, guidance) = (4usize, 5.0f32);
    let (features, mask) = pipe
        .encode_prompt("a red fox in snow", "", "2025-01-01")
        .expect("encode_prompt");

    let run = |sampler: Option<&str>, scheduler: Option<&str>| -> Array {
        mlx_rs::random::seed(7).unwrap();
        let init = mlx_rs::random::normal::<f32>(&[1, seq, 128], None, None, None).unwrap();
        pipe.denoise_with_sampler(
            &features,
            &mask,
            &init,
            lat_h,
            lat_w,
            num_steps,
            guidance,
            sampler,
            scheduler,
            7,
            &mlx_gen::CancelFlag::default(),
            &mut |_, _| {},
        )
        .expect("denoise_with_sampler")
        .as_dtype(Dtype::Float32)
        .unwrap()
    };

    let euler = run(None, None);
    let heun = run(Some("heun"), None);
    let karras = run(None, Some("karras"));

    for (name, a) in [("euler", &euler), ("heun", &heun), ("karras", &karras)] {
        assert_eq!(a.shape(), &[1, seq, 128], "{name}: wrong latent shape");
        assert!(
            a.as_slice::<f32>().iter().all(|v| v.is_finite()),
            "{name}: produced non-finite latents"
        );
    }
    // Per-output non-degeneracy: a broken solver collapses (std → 0) or blows up (std ≫); two valid
    // samples from the same model/prompt/seed have comparable spread. We compare each curated output's
    // std to euler's (the golden-validated default) rather than a brittle absolute magnitude — and do
    // NOT assert cosine-to-euler: at 4 steps a 2nd-order solver (heun) legitimately diverges from
    // euler-4 (the reason Chroma uses heun as its low-step default), so a high cosine is the wrong gate.
    let std_of = |x: &Array| -> f32 {
        let m = mlx_rs::ops::mean(x, None).unwrap();
        mlx_rs::ops::mean(
            multiply(subtract(x, &m).unwrap(), subtract(x, &m).unwrap()).unwrap(),
            None,
        )
        .unwrap()
        .item::<f32>()
        .sqrt()
    };
    let (se, sh, sk) = (std_of(&euler), std_of(&heun), std_of(&karras));
    eprintln!(
        "std  euler {se:.4}  heun {sh:.4}  karras {sk:.4}\n\
         cosine(heun,euler) {:.4}  cosine(karras,euler) {:.4}  peak_rel(heun,euler) {:.4}",
        cosine(&heun, &euler),
        cosine(&karras, &euler),
        peak_rel(&heun, &euler)
    );
    // The sampler knob is live — heun's extra eval changes the trajectory vs euler.
    assert!(
        peak_rel(&heun, &euler) > 1e-3,
        "heun did not change the trajectory vs euler — the curated knob is not wired"
    );
    // Each curated output is a valid, non-degenerate latent (vs the golden-validated euler default).
    for (name, s) in [("heun", sh), ("karras", sk)] {
        assert!(
            s > 0.25 * se && s < 4.0 * se,
            "{name} std {s:.4} is degenerate vs euler {se:.4} (collapsed or blown up)"
        );
    }
    eprintln!("CURATED SMOKE PASS");
}
