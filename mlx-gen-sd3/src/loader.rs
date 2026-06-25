//! Snapshot-layout loader for SD3.5 (E5, sc-7864). Assembles the full text-to-image stack from a
//! `stabilityai/stable-diffusion-3.5-large` diffusers snapshot directory:
//!
//! ```text
//! <root>/transformer/      diffusion_pytorch_model{,-00001-of-00002}.safetensors  (MMDiT)
//! <root>/text_encoder/     model.safetensors                                       (CLIP-L)
//! <root>/text_encoder_2/   model.safetensors                                       (CLIP-G / bigG)
//! <root>/text_encoder_3/   model-0000{1,2}-of-00002.safetensors                    (T5-XXL)
//! <root>/tokenizer{,_2}/   vocab.json + merges.txt                                 (CLIP BPE)
//! <root>/tokenizer_3/      tokenizer.json                                          (T5)
//! <root>/vae/              diffusion_pytorch_model.safetensors                     (16-ch VAE)
//! ```
//!
//! This crate ships NO net-new encoder/VAE: it REUSES the SDXL CLIP encoder (loaded twice), the FLUX
//! T5 encoder, and the Z-Image 16-ch VAE (via E4's [`crate::vae::load_sd3_vae`]). The loader's job is
//! the snapshot-layout glue + the diffusers→MLX weight-key remap each reused module expects.

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::Dtype;

use mlx_gen_sdxl::tokenizer::ClipBpeTokenizer;
use mlx_gen_sdxl::ClipTextEncoder;

use crate::config::Sd3Arch;
use crate::text::{sd3_clip_g_config, sd3_clip_l_config, Sd3TextEncoders};
use crate::transformer::Sd3Transformer;
use crate::vae::load_sd3_vae;

/// CLIP context window (diffusers `tokenizer(..., max_length=77, padding="max_length")`).
pub const CLIP_MAX_LENGTH: usize = 77;
/// CLIP pad token id — `<|endoftext|>` (49407), the SD3 CLIP tokenizer's `pad_token`. diffusers pads
/// CLIP to 77 with the EOS token; the encoder's pooled-at-argmax then still selects the FIRST EOS.
pub const CLIP_PAD_ID: i32 = 49407;
/// T5 sequence length for SD3 (diffusers `max_sequence_length=256`).
pub const T5_MAX_LENGTH: usize = 256;
/// T5 pad token id — `<pad>` (0).
pub const T5_PAD_ID: i32 = 0;

/// Resolve a component's weight file inside `subdir`, preferring the f32 master (`model.safetensors`)
/// over the `model.fp16.safetensors` variant when both are cached (the loader casts to the load dtype
/// regardless). T5 ships sharded — handled by [`load_text_encoder_3`] via [`Weights::from_dir`].
fn clip_weight_file(root: &Path, subdir: &str) -> Result<std::path::PathBuf> {
    let plain = root.join(subdir).join("model.safetensors");
    let fp16 = root.join(subdir).join("model.fp16.safetensors");
    if plain.exists() {
        Ok(plain)
    } else if fp16.exists() {
        Ok(fp16)
    } else {
        Err(Error::Msg(format!(
            "sd3: missing {subdir}/model.safetensors (and no .fp16 variant)"
        )))
    }
}

/// Load CLIP-L (`text_encoder`) at f32 — the SD3 CLIP-L config (768-wide, with a 768 text projection).
fn load_clip_l(root: &Path) -> Result<ClipTextEncoder> {
    let file = clip_weight_file(root, "text_encoder")?;
    let mut w = Weights::from_file(&file)?;
    w.cast_all(Dtype::Float32)?;
    ClipTextEncoder::from_weights(&w, "text_model", &sd3_clip_l_config())
}

/// Load CLIP-G / OpenCLIP-bigG (`text_encoder_2`) at f32 — 1280-wide with the 1280 pooled projection.
fn load_clip_g(root: &Path) -> Result<ClipTextEncoder> {
    let file = clip_weight_file(root, "text_encoder_2")?;
    let mut w = Weights::from_file(&file)?;
    w.cast_all(Dtype::Float32)?;
    ClipTextEncoder::from_weights(&w, "text_model", &sd3_clip_g_config())
}

/// Load the three text encoders. CLIP-L + CLIP-G via the SDXL encoder at the `text_model` prefix; the
/// T5-XXL via the FLUX encoder at the empty prefix (sharded `text_encoder_3/` loaded as a dir). Loaded
/// dense at f32 (the CLIP path runs f32; the T5 promotes internally) — `quantize` is applied after.
pub fn load_text_encoders(root: &Path) -> Result<Sd3TextEncoders> {
    let clip_l = load_clip_l(root)?;
    let clip_g = load_clip_g(root)?;
    let t5_w = load_t5_weights(&root.join("text_encoder_3"))?;
    let t5 = mlx_gen_flux::T5TextEncoder::from_weights(&t5_w, "")?;
    Ok(Sd3TextEncoders { clip_l, clip_g, t5 })
}

/// Load the sharded T5 (`text_encoder_3/`) weights, selecting ONLY the f32 shards. The SD3.5 snapshot
/// ships the f32 master shards (`model-0000N-of-0000M.safetensors`) AND the fp16 variant shards
/// (`model.fp16-0000N-of-0000M.safetensors`) side-by-side in the same dir, so a plain
/// [`Weights::from_dir`] sees both copies of every key and errors on the duplicate. The f32
/// `model.safetensors.index.json` enumerates exactly the f32 shards, so drive the merge from it.
fn load_t5_weights(dir: &Path) -> Result<Weights> {
    let index_path = dir.join("model.safetensors.index.json");
    let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)
        .map_err(|e| Error::Msg(format!("sd3 t5: parse {index_path:?}: {e}")))?;
    let weight_map = index
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| Error::Msg(format!("sd3 t5: {index_path:?} has no object `weight_map`")))?;
    // The distinct shard filenames referenced by the f32 index (sorted for determinism).
    let mut shards: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    shards.sort();
    shards.dedup();
    if shards.is_empty() {
        return Err(Error::Msg(format!(
            "sd3 t5: {index_path:?} `weight_map` references no shards"
        )));
    }
    let mut merged = Weights::empty();
    for shard in shards {
        let w = Weights::from_file(dir.join(&shard))?;
        for k in w.keys().map(String::from).collect::<Vec<_>>() {
            merged.insert(k.clone(), w.require(&k)?.clone());
        }
    }
    Ok(merged)
}

/// Load the CLIP BPE tokenizer (one instance serves both CLIP encoders — `tokenizer/` and
/// `tokenizer_2/` ship byte-identical `vocab.json` + `merges.txt`).
pub fn load_clip_tokenizer(root: &Path) -> Result<ClipBpeTokenizer> {
    ClipBpeTokenizer::from_dir(root.join("tokenizer"))
}

/// Load the T5 tokenizer from `tokenizer_3/tokenizer.json`, configured to pad to SD3's 256-token T5
/// window with the `<pad>` (0) token — diffusers `padding="max_length", max_length=256`.
pub fn load_t5_tokenizer(root: &Path) -> Result<TextTokenizer> {
    let config = TokenizerConfig {
        max_length: T5_MAX_LENGTH,
        pad_token_id: T5_PAD_ID,
        chat_template: ChatTemplate::None,
        pad_to_max_length: true,
    };
    Ok(TextTokenizer::from_file(
        root.join("tokenizer_3").join("tokenizer.json"),
        config,
    )?)
}

/// Load the MMDiT transformer from `transformer/` (sharded; auto dense-vs-prequantized per Linear).
pub fn load_transformer(root: &Path, arch: &Sd3Arch) -> Result<Sd3Transformer> {
    Sd3Transformer::from_dir(&root.join("transformer"), arch)
}

/// Load the 16-channel VAE (decoder + encoder) from `vae/` via the E4 reuse path.
pub fn load_vae(root: &Path) -> Result<mlx_gen_z_image::vae::Vae> {
    load_sd3_vae(&root.join("vae"))
}
