//! Build the dense **bf16** LTX-2.3 tier (sc-8513, epic 8506) from the cached distilled-1.1
//! checkpoint. `#[ignore]`d — it needs the ~43 GB single-file source and produces a ~47 GB `bf16/`
//! turnkey. Mirrors the Q4/Q8 tier builds (`convert_parity.rs`, `convert_ltx_native` in the worker)
//! but with [`LtxConvertOpts::default`] (`quantize: false`) so the transformer stays dense bf16 —
//! the connector / VAE / upsampler / vocoder are already bf16 in every tier.
//!
//! Stage an upscaler dir holding ONLY `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (the worker's
//! `ensure_ltx_upscaler_cached` fetches exactly that), matching the hosted q4/q8 `upsampler.safetensors`.
//!
//! Run:
//!   LTX_BF16_SRC=~/.cache/huggingface/.../ltx-2.3-22b-distilled-1.1.safetensors \
//!   LTX_BF16_UPSCALER_DIR=~/ltx-bf16-staging/upscaler \
//!   LTX_BF16_OUT=~/ltx-bf16-staging/out/bf16 \
//!   cargo test -p mlx-gen-ltx --test build_bf16_tier -- --ignored --nocapture

use mlx_gen_ltx::convert::{convert_and_assemble, LtxConvertOpts};
use std::path::PathBuf;

#[test]
#[ignore = "builds the ~47 GB LTX-2.3 bf16 tier from the ~43 GB distilled source"]
fn build_bf16_tier() {
    let src = PathBuf::from(std::env::var("LTX_BF16_SRC").expect("set LTX_BF16_SRC"));
    let updir =
        PathBuf::from(std::env::var("LTX_BF16_UPSCALER_DIR").expect("set LTX_BF16_UPSCALER_DIR"));
    let out = PathBuf::from(std::env::var("LTX_BF16_OUT").expect("set LTX_BF16_OUT"));
    assert!(
        src.is_file(),
        "source checkpoint missing: {}",
        src.display()
    );

    // Dense: default opts are `include_audio: true, quantize: false` → the transformer is cast to
    // bf16 and saved without quantization (no `quantize_config.json`, `split_model.json` quantized:false).
    let opts = LtxConvertOpts::default();
    assert!(
        !opts.quantize,
        "the bf16 tier must be dense (quantize:false)"
    );
    assert!(
        opts.include_audio,
        "the bf16 tier must include the audio components"
    );

    let produced = convert_and_assemble(&src, Some(&updir), &out, &opts)
        .unwrap_or_else(|e| panic!("bf16 convert_and_assemble failed: {e}"));
    eprintln!("BUILT bf16 tier at {}", produced.display());
}
