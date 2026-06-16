//! sc-5915: real-weights smoke for the FLUX.2-**dev** Mistral text encoder.
//! `#[ignore]`d — needs the real `black-forest-labs/FLUX.2-dev` snapshot in the HF cache and the
//! golden produced by `tools/dump_flux2_te_dev_real_golden.py` (gitignored, large, regenerable):
//!
//!   ~/mlx-flux-venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_te_dev_real_golden.py
//!   cargo test -p mlx-gen-flux2 --test te_dev_real_weights -- --ignored --nocapture
//!
//! The committed `te_dev_parity.rs` proves the encoder *math* bit-tight in f32 on a tiny Mistral
//! config; this proves the *loader* (sharded `language_model.model.*` keys, no-qk-norm path, the
//! [10,20,30] concat) on the real 24B checkpoint. The Rust TE runs f32 activations; the fork-side
//! golden is the production **bf16**, so the gate is a generous mean-relative bound (a gross
//! loader/key bug diverges ~100%; the residual is the expected bf16-vs-f32 accumulation over 30
//! layers, with Rust f32 the more-accurate side). `FLUX2_TE_DEV_F32=1` dumps an f32 golden for a
//! tight gate.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_flux2::{load_text_encoder_dev, load_tokenizer_dev};
use mlx_rs::Array;
use mlx_rs::Dtype;

/// Must match `PROMPT` in `tools/dump_flux2_te_dev_real_golden.py`.
const PROMPT: &str = "a red fox resting in fresh snow under soft winter light";

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
        .expect("a snapshot under models--black-forest-labs--FLUX.2-dev/snapshots")
}

/// The f32 golden (`FLUX2_TE_DEV_F32=1`, same precision as the Rust port) if present, else bf16.
fn golden() -> (Weights, bool) {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden");
    let f32 = base.join("flux2_te_dev_real_f32.safetensors");
    if let Ok(w) = Weights::from_file(&f32) {
        return (w, true);
    }
    let bf16 = base.join("flux2_te_dev_real.safetensors");
    let w = Weights::from_file(&bf16).unwrap_or_else(|_| {
        panic!(
            "missing {} — run tools/dump_flux2_te_dev_real_golden.py (FLUX2_TE_DEV_F32=1 for the f32 ref)",
            bf16.display()
        )
    });
    (w, false)
}

/// Mean-relative error vs golden `b`.
fn mean_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let mabs = (ys.iter().map(|y| y.abs()).sum::<f32>() / ys.len() as f32).max(1e-12);
    let mean_diff = xs.iter().zip(ys).map(|(x, y)| (x - y).abs()).sum::<f32>() / xs.len() as f32;
    mean_diff / mabs
}

/// Mean-relative error over the REAL (non-pad) token positions only. The prompt pads to 512 but is
/// ~44 real tokens, so ~91% of positions are pad — their TE outputs are don't-cares (masked out in
/// the downstream DiT cross-attention) whose bf16-vs-f32 values diverge freely. Averaging over the
/// full 512 swamps the metric; the conditioning that matters is the real-token rows. `mask`: golden
/// `[1, S]` attention mask.
fn mean_rel_masked(out: &Array, want: &Array, mask: &Array) -> f32 {
    let s = *mask.shape().last().unwrap();
    let mh = mlx_gen::array::host_i32(mask).unwrap();
    let idx: Vec<i32> = (0..s).filter(|&j| mh[j as usize] == 1).collect();
    let idx = Array::from_slice(&idx, &[idx.len() as i32]);
    let out_r = out.take_axis(&idx, 1).unwrap();
    let want_r = want.take_axis(&idx, 1).unwrap();
    mean_rel(&out_r, &want_r)
}

#[test]
#[ignore = "needs real FLUX.2-dev snapshot + tools/golden/flux2_te_dev_real.safetensors"]
fn dev_tokenizer_ids_match_reference() {
    let tok = load_tokenizer_dev(&snapshot()).unwrap();
    let out = tok.tokenize(PROMPT).unwrap();
    let (input_ids, _) = mlx_gen::tokenizer::to_arrays(&out);
    let (g, _) = golden();
    let want = g.require("input_ids").unwrap();
    assert_eq!(input_ids.shape(), want.shape(), "input_ids shape (1,512)");
    // Exact integer match: the dev chat template + Mistral BPE + BOS post-processor + max-length
    // padding must reproduce the `PixtralProcessor` byte-for-byte.
    let got = input_ids.as_dtype(Dtype::Float32).unwrap();
    let want_f = want.as_dtype(Dtype::Float32).unwrap();
    let eq = mlx_rs::ops::all_close(&got, &want_f, 0.0, 0.0, false)
        .unwrap()
        .item::<bool>();
    assert!(
        eq,
        "dev tokenizer input_ids diverged from the PixtralProcessor"
    );
}

#[test]
#[ignore = "needs real FLUX.2-dev snapshot + tools/golden/flux2_te_dev_real.safetensors"]
fn dev_text_encoder_prompt_embeds_match_reference() {
    let te = load_text_encoder_dev(&snapshot()).unwrap();
    let (g, is_f32) = golden();
    let out = te
        .prompt_embeds(
            g.require("input_ids").unwrap(),
            g.require("attention_mask").unwrap(),
        )
        .unwrap();
    let want = g.require("prompt_embeds").unwrap();
    assert_eq!(
        out.shape(),
        want.shape(),
        "prompt_embeds shape (1,512,15360)"
    );
    let mask = g.require("attention_mask").unwrap();
    let mean = mean_rel_masked(&out, want, mask);
    let mean_all = mean_rel(&out, want);
    let ref_kind = if is_f32 { "ref f32" } else { "ref bf16" };
    println!(
        "flux2-dev TE real-weights: mean_rel(real)={mean:.5} mean_rel(all+pad)={mean_all:.5} \
         (Rust f32 vs {ref_kind})"
    );
    let bound = if is_f32 { 5e-3 } else { 3.5e-2 };
    assert!(
        mean < bound,
        "dev TE prompt_embeds diverged (real tokens): mean_rel={mean} (ref={ref_kind})"
    );
}
