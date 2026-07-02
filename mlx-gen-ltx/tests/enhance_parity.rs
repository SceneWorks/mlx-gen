//! sc-2845 — LTX-2.3 prompt-enhancement behavioral/smoke gate.
//!
//! There is **no numeric parity** here (stochastic text rewrite; mlx-rs RNG isn't portable to
//! mlx-python). The gate is behavioral:
//! - **off ⇒ passthrough**: `GenerationRequest` defaults `enhance_prompt = false`, so the diffusion
//!   path (and the e2e parity seams) are untouched.
//! - **on ⇒ non-empty, cleaned rewrite** (the heavy `#[ignore]` test below, on the local
//!   gemma-3-12b-it-bf16 snapshot): exercises the whole new infra end to end — the Gemma causal-LM
//!   KV cache + `decode_logits`, the sampler (temperature + repetition penalty), the detokenizer, the
//!   stop tokens, and `clean_response`.
//!
//! The **uncensored** 4-bit path (`use_uncensored_enhancer`) shares this exact loop and is covered by
//! the same logic; validating it end to end needs the separate `TheCluster/amoral-gemma-3-12B-v2-mlx-4bit`
//! download (set `$LTX_UNCENSORED_GEMMA_DIR`) — see `uncensored_enhancer_smoke` (ignored).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::GenerationRequest;
use mlx_gen_ltx::enhance::{self, EnhanceConfig, SampleParams};
use mlx_gen_ltx::gemma::{GemmaConfig, GemmaModel};
use mlx_gen_ltx::tokenizer::LtxTokenizer;

/// Passthrough-when-off is a default-driven guarantee — no weights needed.
#[test]
fn enhancement_is_off_by_default() {
    let req = GenerationRequest::default();
    assert!(
        !req.enhance_prompt,
        "enhancement must default off (passthrough)"
    );
    assert!(!req.use_uncensored_enhancer);
    assert!(req.enhance_max_tokens.is_none());
    assert!(req.enhance_temperature.is_none());
}

/// Resolve the gemma-3-12b-it-bf16 snapshot: `$LTX_GEMMA_DIR` wins, else the newest HF-cache snapshot.
fn gemma_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("LTX_GEMMA_DIR") {
        return Some(d.into());
    }
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--mlx-community--gemma-3-12b-it-bf16/snapshots");
    std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .max_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
}

fn run_smoke(dir: &PathBuf, sampler: SampleParams) -> String {
    let w = Weights::from_dir(dir).expect("load gemma weights");
    let gemma =
        GemmaModel::from_weights(&w, GemmaConfig::gemma_3_12b(), None).expect("build gemma");
    let tok = LtxTokenizer::from_dir(dir).expect("load gemma tokenizer");
    let cfg = EnhanceConfig {
        max_tokens: 96, // keep the smoke quick; full default is 512
        seed: 42,
    };
    // F-018: a pre-tripped cancel aborts before the ~12B prefill, returning Error::Canceled.
    let tripped = mlx_gen::CancelFlag::new();
    tripped.cancel();
    let canceled = enhance::enhance(
        &gemma,
        &tok,
        enhance::T2V_SYSTEM_PROMPT,
        "a red fox running through a snowy forest at dawn",
        &cfg,
        &sampler,
        Some(&tripped),
    );
    assert!(
        matches!(canceled, Err(mlx_gen::Error::Canceled)),
        "pre-tripped cancel must abort enhance with Error::Canceled"
    );

    let out = enhance::enhance(
        &gemma,
        &tok,
        enhance::T2V_SYSTEM_PROMPT,
        "a red fox running through a snowy forest at dawn",
        &cfg,
        &sampler,
        None,
    )
    .expect("enhance");
    eprintln!("enhanced ({:?}): {out}", sampler);
    // Cleaned: no leading/trailing whitespace, no leading punctuation.
    assert!(!out.trim().is_empty(), "rewrite must be non-empty");
    assert_eq!(
        out,
        out.trim(),
        "clean_response strips surrounding whitespace"
    );
    let first = out.chars().next().unwrap();
    assert!(
        first.is_alphanumeric() || first == '_',
        "cleaned output starts with a word char, got {first:?}"
    );
    out
}

/// Heavy: censored path on the loaded TE backbone (the default enhancer). Run with
/// `cargo test -p mlx-gen-ltx --test enhance_parity -- --ignored --nocapture`.
#[test]
#[ignore = "heavy: needs the local gemma-3-12b-it-bf16 snapshot (~24GB) + autoregressive generation"]
fn censored_enhancer_produces_nonempty_cleaned_rewrite() {
    let dir = gemma_dir().expect("set $LTX_GEMMA_DIR or have the HF gemma-3-12b-it-bf16 snapshot");
    run_smoke(&dir, SampleParams::censored(0.7));
}

/// Heavy: exercises the **uncensored** sampler shape (pure temperature, no repetition penalty) on the
/// bf16 TE snapshot. The real uncensored model (`TheCluster/amoral-gemma-3-12B-v2-mlx-4bit`, `model.`
/// key prefix + 4-bit quant) is loaded by the production `model::load_uncensored_enhancer` path; this
/// test only validates the loop under the uncensored `SampleParams`, not the specific weights.
#[test]
#[ignore = "heavy: needs the local gemma-3-12b-it-bf16 snapshot + autoregressive generation"]
fn uncensored_sampler_shape_smoke() {
    let dir = gemma_dir().expect("set $LTX_GEMMA_DIR or have the HF gemma-3-12b-it-bf16 snapshot");
    run_smoke(&dir, SampleParams::uncensored(0.7));
}
