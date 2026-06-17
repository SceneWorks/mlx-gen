//! Token-layout parity for FLUX.2-dev **caption upsampling** (sc-6030): the Rust
//! `build_upsample_input_ids` + `expand_pixtral_image_tokens` must reproduce the EXACT `input_ids`
//! the reference dev `PixtralProcessor` + diffusers `upsample_prompt` helpers produce — including the
//! `[IMG]`/`[IMG_BREAK]`/`[IMG_END]` image-token layout. The generate path then runs the Mistral
//! tower over these ids, so getting them byte-exact is what makes the rewrite coherent.
//!
//! `#[ignore]`d — needs the real `black-forest-labs/FLUX.2-dev` snapshot in the HF cache (only the
//! `tokenizer/` — no diffusion weights) and the committed golden
//! `tests/fixtures/caption_upsample_golden.safetensors`
//! (← `tools/dump_flux2_dev_caption_upsample_golden.py`):
//!
//!   cargo test -p mlx-gen-flux2 --test caption_upsample_golden -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_flux2::{
    build_upsample_input_ids, load_tokenizer_dev, SYSTEM_MESSAGE_UPSAMPLING_I2I,
    SYSTEM_MESSAGE_UPSAMPLING_T2I,
};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/caption_upsample_golden.safetensors"
);
/// Must match the dump tool's `PROMPT`.
const PROMPT: &str = "a red fox in fresh snow";
/// `patch_size · spatial_merge_size` — the dev image processor resizes to a multiple of this, so the
/// merged grid is `(H/28, W/28)`.
const MERGE_PATCH: i32 = 28;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_DEV_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots");
    std::fs::read_dir(&base)
        .expect("snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot under models--black-forest-labs--FLUX.2-dev/snapshots")
}

fn golden_ids(w: &Weights, key: &str) -> Vec<i32> {
    w.require(key).unwrap().as_slice::<i32>().to_vec()
}

#[test]
#[ignore = "needs the real FLUX.2-dev snapshot (tokenizer.json) + the committed golden"]
fn t2i_input_ids_match_reference() {
    let tok = load_tokenizer_dev(&snapshot()).unwrap();
    let g = Weights::from_file(GOLDEN).unwrap();
    let want = golden_ids(&g, "t2i_input_ids");
    let got = build_upsample_input_ids(&tok, SYSTEM_MESSAGE_UPSAMPLING_T2I, PROMPT, None).unwrap();
    assert_eq!(
        got, want,
        "T2I caption-upsample input_ids must match the reference PixtralProcessor exactly"
    );
}

#[test]
#[ignore = "needs the real FLUX.2-dev snapshot (tokenizer.json) + the committed golden"]
fn i2i_input_ids_and_image_token_layout_match_reference() {
    let tok = load_tokenizer_dev(&snapshot()).unwrap();
    let g = Weights::from_file(GOLDEN).unwrap();
    let want = golden_ids(&g, "i2i_input_ids");
    let sizes = golden_ids(&g, "i2i_image_sizes"); // [H, W] the reference resized to.
    let (h, w) = (sizes[0], sizes[1]);
    // The merged (post-2×2) grid the projector / token-expansion use.
    let merged = (h / MERGE_PATCH, w / MERGE_PATCH);
    let got = build_upsample_input_ids(&tok, SYSTEM_MESSAGE_UPSAMPLING_I2I, PROMPT, Some(merged))
        .unwrap();
    assert_eq!(
        got, want,
        "I2I caption-upsample input_ids (incl. the [IMG]/[IMG_BREAK]/[IMG_END] layout) must match the reference"
    );
    // The `[IMG]`(10) count equals the merged-grid product = the projector's projected-feature rows.
    assert_eq!(
        got.iter().filter(|&&t| t == 10).count() as i32,
        merged.0 * merged.1,
        "[IMG] token count must equal the projector merged-token count"
    );
}
