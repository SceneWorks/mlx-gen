//! Real-checkpoint loading for Z-Image-turbo: assemble the tokenizer, Qwen text encoder, DiT
//! transformer, and VAE decoder from a `Tongyi-MAI/Z-Image-Turbo` snapshot directory, applying
//! the diffusers-checkpoint → internal-name remaps (the fork's `z_image_weight_mapping`).
//!
//! The snapshot layout is the standard diffusers multi-component tree:
//! ```text
//!   <root>/tokenizer/tokenizer.json
//!   <root>/text_encoder/*.safetensors
//!   <root>/transformer/*.safetensors
//!   <root>/vae/*.safetensors
//! ```
//! These loaders were validated stage-by-stage against the Python fork on real bf16 weights
//! (sc-2352); they are the production path behind [`crate::model::load`].

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Result, WeightsSource};

use crate::control_transformer::ZImageControlTransformer;
use crate::text_encoder::{TextEncoder, ZTextEncoderConfig};
use crate::transformer::{ZImageTransformer, ZImageTransformerConfig};
use crate::vae::{Vae, VaeDecoderConfig, VaeEncoderConfig};

/// Qwen3 pad token id (`<|endoftext|>`).
const PAD_TOKEN_ID: i32 = 151643;
/// Prompts pad to this length (the fork's `padding="max_length"`).
const MAX_LENGTH: usize = 512;

/// Load the Qwen tokenizer with the Z-Image chat-template + padding policy.
pub fn load_tokenizer(root: &Path) -> Result<TextTokenizer> {
    TextTokenizer::from_file(
        root.join("tokenizer/tokenizer.json"),
        TokenizerConfig {
            max_length: MAX_LENGTH,
            pad_token_id: PAD_TOKEN_ID,
            chat_template: ChatTemplate::QwenInstruct,
            pad_to_max_length: true,
        },
    )
}

/// Load the Qwen3-style text encoder (prompt → `cap_feats`). The checkpoint keys are prefixed
/// `model.` (the encoder's own modules); no other remap is needed.
pub fn load_text_encoder(root: &Path) -> Result<TextEncoder> {
    let w = Weights::from_dir(root.join("text_encoder"))?;
    TextEncoder::from_weights(&w, "model", &ZTextEncoderConfig::z_image())
}

/// Load the DiT transformer, applying the timestep-embedder + final-layer key remaps.
///
/// KEEP-F32 (sc-2609): Z-Image-Turbo ships f32 weights on disk and we deliberately keep them f32
/// (the fork downcasts to bf16). The dense render is then *sharper / more contrast* than the fork's
/// bf16 — Michael's preferred output — and it costs nothing today (activations already run f32, so
/// the matmul is f32 either way). Do NOT downcast these to bf16 for parity. The only place we cast
/// to bf16 is the quantize path (`AdaptableLinear::quantize`, tagged PARITY-BF16) to byte-match the
/// fork's Q8/Q4 golden; that too is a flip-to-f32 candidate once parity stops being the goal.
pub fn load_transformer(root: &Path) -> Result<ZImageTransformer> {
    let mut w = Weights::from_dir(root.join("transformer"))?;
    remap_transformer_keys(&mut w);
    ZImageTransformer::from_weights(&w, "", ZImageTransformerConfig::turbo())
}

/// Load the ControlNet transformer (sc-2349): the base DiT from the snapshot `root`, overlaid with
/// the Fun-Controlnet-Union control branch from `control` (a single `.safetensors` `File`, or a
/// `Dir` of them — the HF cache stores the checkpoint as one file inside a snapshot dir). The
/// control keys map 1:1 onto the control branch's param tree, so no remap is needed.
pub fn load_control_transformer(
    root: &Path,
    control: &WeightsSource,
) -> Result<ZImageControlTransformer> {
    let base = load_transformer(root)?;
    let control_weights = match control {
        WeightsSource::File(p) => Weights::from_file(p)?,
        WeightsSource::Dir(p) => Weights::from_dir(p)?,
    };
    ZImageControlTransformer::from_weights(base, &control_weights, "")
}

/// Load the full VAE (decoder + encoder), remapping both diffusers trees to the internal naming
/// and transposing conv weights to NHWC. The encoder powers img2img (`Conditioning::Reference`).
pub fn load_vae(root: &Path) -> Result<Vae> {
    let mut w = Weights::from_dir(root.join("vae"))?;
    remap_vae_decoder(&mut w)?;
    remap_vae_encoder(&mut w)?;
    Vae::from_weights(&w, "", &VaeDecoderConfig::default_z_image())?.with_encoder(
        &w,
        "encoder",
        &VaeEncoderConfig::default_z_image(),
    )
}

/// diffusers DiT checkpoint → internal names: the timestep embedder is `t_embedder.mlp.{0,2}` on
/// disk but `linear{1,2}` internally, and the final layer's adaLN is `Sequential(SiLU, Linear)`
/// (Linear at index 1 on disk, index 0 internally). Everything else matches directly.
pub fn remap_transformer_keys(w: &mut Weights) {
    for (from, to) in [
        ("t_embedder.mlp.0.weight", "t_embedder.linear1.weight"),
        ("t_embedder.mlp.0.bias", "t_embedder.linear1.bias"),
        ("t_embedder.mlp.2.weight", "t_embedder.linear2.weight"),
        ("t_embedder.mlp.2.bias", "t_embedder.linear2.bias"),
        (
            "all_final_layer.2-1.adaLN_modulation.1.weight",
            "all_final_layer.2-1.adaLN_modulation.0.weight",
        ),
        (
            "all_final_layer.2-1.adaLN_modulation.1.bias",
            "all_final_layer.2-1.adaLN_modulation.0.bias",
        ),
    ] {
        w.alias(from, to);
    }
}

/// Map one diffusers VAE key *suffix* (the part after the `decoder.`/`encoder.` prefix) to its
/// internal target suffix and whether the tensor is an NCHW→NHWC conv weight needing transpose.
/// Pure (no `Weights`), so the remap's regression-prone naming/transpose logic is unit-testable
/// without a checkpoint fixture. `sampler_substr` is the resample-conv path that also carries a
/// conv weight: `.upsamplers.0.conv.` for the decoder up path, `.downsamplers.0.conv.` for the
/// encoder down path.
fn vae_key_target(rest: &str, sampler_substr: &str) -> (String, bool) {
    match rest {
        "conv_in.weight" => ("conv_in.conv.weight".into(), true),
        "conv_in.bias" => ("conv_in.conv.bias".into(), false),
        "conv_out.weight" => ("conv_out.conv.weight".into(), true),
        "conv_out.bias" => ("conv_out.conv.bias".into(), false),
        "conv_norm_out.weight" => ("conv_norm_out.norm.weight".into(), false),
        "conv_norm_out.bias" => ("conv_norm_out.norm.bias".into(), false),
        _ => {
            let is_conv_w = rest.ends_with(".weight")
                && (rest.contains(".conv1.")
                    || rest.contains(".conv2.")
                    || rest.contains(".conv_shortcut.")
                    || rest.contains(sampler_substr));
            (rest.to_string(), is_conv_w)
        }
    }
}

/// Remap a diffusers VAE sub-tree (`{prefix}*`, flat conv names, NCHW conv weights) → the crate's
/// internal `conv.`/`norm.` wrapper naming, conv weights transposed to NHWC `[out,kH,kW,in]`.
/// Inserts the remapped keys alongside the originals. `keep_prefix` controls whether `{prefix}` is
/// retained on the target key — the encoder keeps it, the decoder drops it, and both must coexist in
/// one `Weights`. Shared by [`remap_vae_decoder`]/[`remap_vae_encoder`] (F-037).
fn remap_vae_tree(
    w: &mut Weights,
    prefix: &str,
    keep_prefix: bool,
    sampler_substr: &str,
) -> Result<()> {
    let keys: Vec<String> = w
        .keys()
        .filter(|k| k.starts_with(prefix))
        .map(String::from)
        .collect();
    for k in keys {
        let rest = k.strip_prefix(prefix).unwrap();
        let (suffix, transpose) = vae_key_target(rest, sampler_substr);
        let target = if keep_prefix {
            format!("{prefix}{suffix}")
        } else {
            suffix
        };
        let t = w.require(&k)?.clone();
        let t = if transpose {
            t.transpose_axes(&[0, 2, 3, 1])? // NCHW -> NHWC conv weight
        } else {
            t
        };
        w.insert(target, t);
    }
    Ok(())
}

/// diffusers VAE checkpoint (`decoder.*`) → internal decoder naming, conv weights transposed to
/// NHWC. Drops the `decoder.` prefix (the encoder remap keeps its prefix; both coexist in one
/// `Weights`). Up path → `upsamplers`.
pub fn remap_vae_decoder(w: &mut Weights) -> Result<()> {
    remap_vae_tree(w, "decoder.", false, ".upsamplers.0.conv.")
}

/// diffusers VAE checkpoint (`encoder.*`) → internal encoder naming, conv weights transposed to
/// NHWC. Keeps the `encoder.` prefix. Down path → `downsamplers`.
pub fn remap_vae_encoder(w: &mut Weights) -> Result<()> {
    remap_vae_tree(w, "encoder.", true, ".downsamplers.0.conv.")
}

#[cfg(test)]
mod vae_remap_tests {
    use super::vae_key_target;

    const UP: &str = ".upsamplers.0.conv.";
    const DOWN: &str = ".downsamplers.0.conv.";

    #[test]
    fn special_conv_and_norm_keys_map_and_transpose() {
        assert_eq!(
            vae_key_target("conv_in.weight", UP),
            ("conv_in.conv.weight".to_string(), true)
        );
        assert_eq!(
            vae_key_target("conv_in.bias", UP),
            ("conv_in.conv.bias".to_string(), false)
        );
        assert_eq!(
            vae_key_target("conv_out.weight", UP),
            ("conv_out.conv.weight".to_string(), true)
        );
        // conv_norm_out is a norm, never transposed.
        assert_eq!(
            vae_key_target("conv_norm_out.weight", UP),
            ("conv_norm_out.norm.weight".to_string(), false)
        );
    }

    #[test]
    fn resnet_conv_weights_transpose_but_keep_name() {
        let (t, tr) = vae_key_target("mid_block.resnets.1.conv2.weight", UP);
        assert_eq!(t, "mid_block.resnets.1.conv2.weight");
        assert!(tr, "resnet conv2 weight transposes");
        // The matching bias is not a conv weight.
        assert_eq!(
            vae_key_target("mid_block.resnets.1.conv2.bias", UP),
            ("mid_block.resnets.1.conv2.bias".to_string(), false)
        );
        // A norm weight inside a resnet is not a conv → no transpose.
        assert!(!vae_key_target("up_blocks.0.resnets.0.norm1.weight", UP).1);
    }

    #[test]
    fn sampler_substr_is_path_specific() {
        // The up path's upsampler conv counts as a conv weight only under the decoder substr...
        assert!(vae_key_target("up_blocks.0.upsamplers.0.conv.weight", UP).1);
        assert!(!vae_key_target("up_blocks.0.upsamplers.0.conv.weight", DOWN).1);
        // ...and symmetrically for the down path's downsampler conv under the encoder substr.
        assert!(vae_key_target("down_blocks.2.downsamplers.0.conv.weight", DOWN).1);
        assert!(!vae_key_target("down_blocks.2.downsamplers.0.conv.weight", UP).1);
    }
}
