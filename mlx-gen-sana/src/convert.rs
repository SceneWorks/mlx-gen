//! Offline pre-quantization: read a dense SANA MLX snapshot (`transformer/ text_encoder/ vae/`) and
//! write a packed Q4/Q8 turnkey that [`crate::quant`] (via [`crate::model::load`] /
//! [`crate::model::load_sprint`]) loads with no dense bf16/f32 transient. Mirrors the other Group-B
//! converters (`mlx_gen_chroma::convert` / `mlx_gen_sdxl::convert`) — the same
//! `mlx_gen::quant::quantize_map`, byte-equal to the load-time `.quantize` seam — differing in the
//! SANA key layout and the **two** quantized components.
//!
//! SANA quantizes:
//!  * the Linear-DiT `transformer/` — every trunk Linear ([`is_transformer_target`]): the self /
//!    cross attention `to_q/k/v` + `to_out.0`, the timestep / (Sprint) guidance / modulation MLPs, the
//!    caption-projection MLP, and `proj_out`. The `patch_embed` conv and the GLUMBConv depth/point/
//!    inverted convs stay dense (they are conv weights, not a quant target).
//!  * the Gemma-2 CHI `text_encoder/` — every decoder projection ([`is_te_target`]): `self_attn`
//!    `q/k/v/o_proj` and the `mlp` `gate/up/down_proj`. `embed_tokens` + all RMSNorms stay dense.
//!    (This is the biggest component, so packing it is where the low-RAM win is.)
//!
//! The **DC-AE `vae/`** decoder stays dense in every tier (all-conv; its Linear-attention is a
//! measurably-0% memory win and is not routed through the quant seam), so it is mirrored verbatim.
//!
//! Each per-component predicate matches the loader's `crate::quant::lin` scope exactly — a missed site
//! (or a wrongly-packed dense tensor) loads u32 codes as dense floats → a garbage render. The
//! completeness gate is the real-weight render in `tests/prequantize_real_weights.rs`. There is no
//! `config.json` in `transformer/` (the config is the hard-coded `SanaTransformerConfig`), and the
//! packed bit-width self-describes via the `.scales` shapes, so no quant-config sidecar is written.
//!
//! Group-B per-crate converter template (sc-8669).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;

use mlx_gen::quant::{copy_dir, copy_turnkey_assets, quantize_map, save_map};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::quant::GROUP_SIZE;

/// The single packed weight file each component ships (the source is already one flat file per
/// component: `transformer/diffusion_pytorch_model.safetensors`, `text_encoder/gemma-2-2b-it.safetensors`).
const TRANSFORMER_FILE: &str = "diffusion_pytorch_model.safetensors";
const TE_FILE: &str = "gemma-2-2b-it.safetensors";

// ============================================================================================
// Pack predicates (operate on the **base** = the on-disk key minus its `.weight`).
// ============================================================================================

/// Whether a `transformer/` key's `base` is a trunk **Linear** the DiT quantizes — matching the
/// `crate::quant::lin` sites in [`crate::transformer`] exactly. Everything else (the `patch_embed` +
/// GLUMBConv convs, `scale_shift_table`, `caption_norm`, the per-block affine-free norms) stays dense.
fn is_transformer_target(base: &str) -> bool {
    if let Some(rest) = base.strip_prefix("transformer_blocks.") {
        // rest = `{i}.<tail>`
        let Some((_i, tail)) = rest.split_once('.') else {
            return false;
        };
        return matches!(
            tail,
            "attn1.to_q"
                | "attn1.to_k"
                | "attn1.to_v"
                | "attn1.to_out.0"
                | "attn2.to_q"
                | "attn2.to_k"
                | "attn2.to_v"
                | "attn2.to_out.0"
        );
    }
    // Trunk-level MLPs: timestep + (Sprint) guidance embedders (both the base `time_embed.emb.…` and
    // the Sprint `time_embed.…` key layouts), the modulation Linear, the caption-projection MLP, and
    // the final `proj_out`.
    matches!(
        base,
        "time_embed.emb.timestep_embedder.linear_1"
            | "time_embed.emb.timestep_embedder.linear_2"
            | "time_embed.timestep_embedder.linear_1"
            | "time_embed.timestep_embedder.linear_2"
            | "time_embed.guidance_embedder.linear_1"
            | "time_embed.guidance_embedder.linear_2"
            | "time_embed.linear"
            | "caption_projection.linear_1"
            | "caption_projection.linear_2"
            | "proj_out"
    )
}

/// Whether a `text_encoder/` key's `base` is a Gemma-2 decoder **projection** the TE quantizes —
/// matching `mlx_gen_pid::gemma2`'s `lin` sites exactly (`self_attn` q/k/v/o + `mlp` gate/up/down).
/// `model.embed_tokens`, `model.norm`, and every `*_layernorm` stay dense.
fn is_te_target(base: &str) -> bool {
    let Some(rest) = base.strip_prefix("model.layers.") else {
        return false;
    };
    let Some((_i, tail)) = rest.split_once('.') else {
        return false;
    };
    matches!(
        tail,
        "self_attn.q_proj"
            | "self_attn.k_proj"
            | "self_attn.v_proj"
            | "self_attn.o_proj"
            | "mlp.gate_proj"
            | "mlp.up_proj"
            | "mlp.down_proj"
    )
}

/// Load a component dir's safetensors into one key→`Array` map (a duplicate key is a corrupt
/// snapshot → error). SANA ships one flat file per component, but `from_dir` handles a sharded set too.
fn load_map(dir: &Path) -> Result<HashMap<String, Array>> {
    let w = Weights::from_dir(dir)?;
    let mut map: HashMap<String, Array> = HashMap::new();
    for k in w.keys().map(str::to_string).collect::<Vec<_>>() {
        let v = w.get(&k).expect("listed key").clone();
        if map.insert(k.clone(), v).is_some() {
            return Err(Error::Msg(format!(
                "sana convert: duplicate key `{k}` in {}",
                dir.display()
            )));
        }
    }
    Ok(map)
}

/// Copy every non-`.safetensors` sidecar (e.g. `text_encoder/tokenizer.json`) from `src` → `dst`,
/// dereferencing HF-cache symlinks (`std::fs::copy` follows the link to the blob).
fn copy_sidecars(src: &Path, dst: &Path) -> Result<()> {
    for entry in std::fs::read_dir(src)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors")
            || mlx_gen::gen_core::weightsmeta::is_hidden_file(&path)
        {
            continue;
        }
        if let Some(name) = path.file_name() {
            std::fs::copy(&path, dst.join(name))?;
        }
    }
    Ok(())
}

/// Assemble a full pre-quantized turnkey SANA snapshot in `dst_root`: pack the Linear-DiT
/// `transformer/` and the Gemma-2 `text_encoder/`, mirror the dense DC-AE `vae/`, and copy the
/// tokenizer + any root license/notice verbatim (deref symlinks). The result loads via
/// [`crate::model::load`] / [`load_sprint`] (packed weights auto-detect) with no dense transient.
/// `bits` = 4 (Q4 tier) or 8 (Q8 tier). The **bf16 tier** is the dense source itself (no conversion —
/// mirror it; see the tier builder in `tests/prequantize_real_weights.rs`).
pub fn prequantize_turnkey(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    std::fs::create_dir_all(dst_root)?;

    // Transformer: pack the trunk Linears into one flat file.
    let tr_src = src_root.join("transformer");
    if !tr_src.is_dir() {
        return Err(Error::Msg(format!(
            "sana convert: source snapshot {} has no transformer/ dir",
            src_root.display()
        )));
    }
    let tr_dst = dst_root.join("transformer");
    std::fs::create_dir_all(&tr_dst)?;
    let tr = quantize_map(load_map(&tr_src)?, bits, GROUP_SIZE, is_transformer_target)?;
    save_map(&tr_dst.join(TRANSFORMER_FILE), &tr)?;

    // Text encoder: pack the Gemma-2 projections, keep the tokenizer.json sidecar.
    let te_src = src_root.join("text_encoder");
    if !te_src.is_dir() {
        return Err(Error::Msg(format!(
            "sana convert: source snapshot {} has no text_encoder/ dir",
            src_root.display()
        )));
    }
    let te_dst = dst_root.join("text_encoder");
    std::fs::create_dir_all(&te_dst)?;
    let te = quantize_map(load_map(&te_src)?, bits, GROUP_SIZE, is_te_target)?;
    save_map(&te_dst.join(TE_FILE), &te)?;
    copy_sidecars(&te_src, &te_dst)?;

    // DC-AE decoder stays dense (all-conv) — mirror verbatim.
    copy_dir(&src_root.join("vae"), &dst_root.join("vae"))?;

    // Root license / notice / etc.
    copy_turnkey_assets(src_root, dst_root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{eq, quantize};
    use mlx_rs::{Array, Dtype};

    #[test]
    fn transformer_predicate_matches_trunk_linears_only() {
        for base in [
            "transformer_blocks.0.attn1.to_q",
            "transformer_blocks.0.attn1.to_k",
            "transformer_blocks.0.attn1.to_v",
            "transformer_blocks.0.attn1.to_out.0",
            "transformer_blocks.19.attn2.to_q",
            "transformer_blocks.19.attn2.to_k",
            "transformer_blocks.19.attn2.to_v",
            "transformer_blocks.19.attn2.to_out.0",
            "time_embed.emb.timestep_embedder.linear_1",
            "time_embed.emb.timestep_embedder.linear_2",
            "time_embed.timestep_embedder.linear_1",
            "time_embed.guidance_embedder.linear_2",
            "time_embed.linear",
            "caption_projection.linear_1",
            "caption_projection.linear_2",
            "proj_out",
        ] {
            assert!(is_transformer_target(base), "{base} should pack");
        }
        // Convs + norms + tables stay dense.
        for base in [
            "patch_embed.proj",
            "transformer_blocks.0.ff.conv_inverted",
            "transformer_blocks.0.ff.conv_depth",
            "transformer_blocks.0.ff.conv_point",
            "transformer_blocks.0.attn1.norm_q",
            "transformer_blocks.0.attn2.norm_k",
            "transformer_blocks.0.scale_shift_table",
            "caption_norm",
            "scale_shift_table",
        ] {
            assert!(!is_transformer_target(base), "{base} should stay dense");
        }
    }

    #[test]
    fn te_predicate_matches_gemma_projections_only() {
        for base in [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.25.mlp.gate_proj",
            "model.layers.25.mlp.up_proj",
            "model.layers.25.mlp.down_proj",
        ] {
            assert!(is_te_target(base), "{base} should pack");
        }
        for base in [
            "model.embed_tokens",
            "model.norm",
            "model.layers.0.input_layernorm",
            "model.layers.0.post_attention_layernorm",
            "model.layers.0.pre_feedforward_layernorm",
        ] {
            assert!(!is_te_target(base), "{base} should stay dense");
        }
    }

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    /// A packed trunk Linear is byte-identical to the op the load-time `.quantize` runs (bf16 cast,
    /// group 64) — the sc-8669 round-trip guarantee. A dense embedder-style key (predicate miss) and a
    /// 1-D norm (shape guard) stay dense.
    #[test]
    fn quantize_map_packs_trunk_linear_byte_identical() {
        let w = Array::from_slice(
            &(0..64 * 128).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
            &[64, 128],
        );
        let mut map: HashMap<String, Array> = HashMap::new();
        map.insert("transformer_blocks.0.attn1.to_q.weight".into(), w.clone());
        map.insert("model.layers.0.self_attn.q_proj.weight".into(), w.clone());
        map.insert(
            "patch_embed.proj.weight".into(),
            Array::from_slice(
                &(0..64 * 128).map(|i| (i as f32).cos()).collect::<Vec<_>>(),
                &[64, 128],
            ),
        );
        map.insert(
            "transformer_blocks.0.attn1.norm_q.weight".into(),
            Array::ones::<f32>(&[128]).unwrap(),
        );

        let out = quantize_map(map, 4, GROUP_SIZE, is_transformer_target).unwrap();

        let base = "transformer_blocks.0.attn1.to_q";
        let wq = out.get(&format!("{base}.weight")).expect("packed");
        assert_eq!(wq.dtype(), Dtype::Uint32, "Q4 codes are u32-packed");
        let (ewq, esc, ebi) =
            quantize(w.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, 4).unwrap();
        assert!(byte_equal(wq, &ewq), "packed weight != load-time quantize");
        assert!(byte_equal(
            out.get(&format!("{base}.scales")).unwrap(),
            &esc
        ));
        assert!(byte_equal(
            out.get(&format!("{base}.biases")).unwrap(),
            &ebi
        ));

        // The is_transformer_target predicate does NOT pack a TE key or the conv → both stay dense.
        assert_eq!(
            out.get("patch_embed.proj.weight").unwrap().dtype(),
            Dtype::Float32
        );
        assert!(!out.contains_key("model.layers.0.self_attn.q_proj.scales"));
        assert_eq!(
            out.get("transformer_blocks.0.attn1.norm_q.weight")
                .unwrap()
                .dtype(),
            Dtype::Float32
        );
    }
}
