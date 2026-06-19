//! Boogu transformer "conversion" = **architecture validation**.
//!
//! The published diffusers checkpoint uses dotted keys that map 1:1 onto the
//! `BooguImageTransformer2DModel` module tree, so [`mlx_gen::weights::Weights::from_dir`] loads
//! them directly — there is no fork-style key remap (unlike FLUX.2's `to_out.0`→`to_out`). What we
//! *do* need is to prove the on-disk tensor set exactly matches the architecture implied by
//! [`BooguConfig`] before the DiT forward (E3) trusts it: every expected key present, no stray
//! extras, and the shape-bearing entry-points (patch embedders, caption embedder, FFN, out proj)
//! sized as the config says. This catches a wrong variant / truncated download / config-weight
//! mismatch loudly at load instead of as garbage latents.

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::BooguConfig;

/// A non-modulated transformer block (context refiner): plain `norm1`/`norm2` RMSNorm.
fn block_keys_no_mod(prefix: &str) -> Vec<String> {
    let mut k = attn_keys(&format!("{prefix}.attn"));
    k.extend(ffn_keys(&format!("{prefix}.feed_forward")));
    for n in ["ffn_norm1", "ffn_norm2", "norm1", "norm2"] {
        k.push(format!("{prefix}.{n}.weight"));
    }
    k
}

/// A modulated single-stream / refiner block: `norm1` is a `LuminaRMSNormZero`
/// (`linear.{weight,bias}` + `norm.weight`), `norm2` a plain RMSNorm.
fn block_keys_mod(prefix: &str) -> Vec<String> {
    let mut k = attn_keys(&format!("{prefix}.attn"));
    k.extend(ffn_keys(&format!("{prefix}.feed_forward")));
    k.push(format!("{prefix}.ffn_norm1.weight"));
    k.push(format!("{prefix}.ffn_norm2.weight"));
    k.extend(lumina_rms_zero_keys(&format!("{prefix}.norm1")));
    k.push(format!("{prefix}.norm2.weight"));
    k
}

/// A double-stream (dual-stream) block: a joint instruct↔img attention whose QKV lives on the
/// processor, an image self-attention, two FFNs, three img modulations + two instruct modulations,
/// and the per-sublayer output RMSNorms.
fn double_block_keys(prefix: &str) -> Vec<String> {
    let mut k = Vec::new();
    // Joint attention: per-head q/k norm + the processor's own projections + the shared output.
    k.push(format!("{prefix}.img_instruct_attn.norm_q.weight"));
    k.push(format!("{prefix}.img_instruct_attn.norm_k.weight"));
    for side in ["img", "instruct"] {
        for p in ["to_q", "to_k", "to_v", "out"] {
            k.push(format!(
                "{prefix}.img_instruct_attn.processor.{side}_{p}.weight"
            ));
        }
    }
    k.push(format!("{prefix}.img_instruct_attn.to_out.0.weight"));
    // Image self-attention.
    k.extend(attn_keys(&format!("{prefix}.img_self_attn")));
    // FFNs.
    k.extend(ffn_keys(&format!("{prefix}.img_feed_forward")));
    k.extend(ffn_keys(&format!("{prefix}.instruct_feed_forward")));
    // Modulations.
    for n in [
        "img_norm1",
        "img_norm2",
        "img_norm3",
        "instruct_norm1",
        "instruct_norm2",
    ] {
        k.extend(lumina_rms_zero_keys(&format!("{prefix}.{n}")));
    }
    // Output RMSNorms.
    for n in [
        "img_attn_norm",
        "img_self_attn_norm",
        "img_ffn_norm1",
        "img_ffn_norm2",
        "instruct_attn_norm",
        "instruct_ffn_norm1",
        "instruct_ffn_norm2",
    ] {
        k.push(format!("{prefix}.{n}.weight"));
    }
    k
}

/// GQA attention with per-head q/k RMSNorm and a `to_out` Sequential (`to_out.0`).
fn attn_keys(prefix: &str) -> Vec<String> {
    ["norm_q", "norm_k", "to_q", "to_k", "to_v", "to_out.0"]
        .iter()
        .map(|p| format!("{prefix}.{p}.weight"))
        .collect()
}

/// SwiGLU feed-forward (`linear_1`/`linear_3` in, `linear_2` out), all bias-free.
fn ffn_keys(prefix: &str) -> Vec<String> {
    ["linear_1", "linear_2", "linear_3"]
        .iter()
        .map(|p| format!("{prefix}.{p}.weight"))
        .collect()
}

/// `LuminaRMSNormZero`: a SiLU→Linear modulation (`linear.weight`+`linear.bias`) plus the RMSNorm
/// (`norm.weight`).
fn lumina_rms_zero_keys(prefix: &str) -> Vec<String> {
    vec![
        format!("{prefix}.linear.weight"),
        format!("{prefix}.linear.bias"),
        format!("{prefix}.norm.weight"),
    ]
}

/// The complete set of transformer tensor keys implied by `cfg`.
pub fn expected_transformer_keys(cfg: &BooguConfig) -> Vec<String> {
    let mut keys = Vec::new();

    // Embedders.
    for e in ["x_embedder", "ref_image_patch_embedder"] {
        keys.push(format!("{e}.weight"));
        keys.push(format!("{e}.bias"));
    }
    keys.push("image_index_embedding".to_string());

    // Time + caption embedding.
    for n in ["linear_1", "linear_2"] {
        keys.push(format!("time_caption_embed.timestep_embedder.{n}.weight"));
        keys.push(format!("time_caption_embed.timestep_embedder.{n}.bias"));
    }
    keys.push("time_caption_embed.caption_embedder.0.weight".to_string()); // RMSNorm
    keys.push("time_caption_embed.caption_embedder.1.weight".to_string()); // Linear
    keys.push("time_caption_embed.caption_embedder.1.bias".to_string());

    // Refiners (context = no modulation; noise + ref-image = modulated).
    for i in 0..cfg.num_refiner_layers {
        keys.extend(block_keys_no_mod(&format!("context_refiner.{i}")));
        keys.extend(block_keys_mod(&format!("noise_refiner.{i}")));
        keys.extend(block_keys_mod(&format!("ref_image_refiner.{i}")));
    }

    // Dual-stream then single-stream stacks.
    for i in 0..cfg.num_double_stream_layers {
        keys.extend(double_block_keys(&format!("double_stream_layers.{i}")));
    }
    for i in 0..cfg.num_single_stream_layers() {
        keys.extend(block_keys_mod(&format!("single_stream_layers.{i}")));
    }

    // Continuous-AdaLN output projection (LuminaLayerNormContinuous).
    for n in ["linear_1", "linear_2"] {
        keys.push(format!("norm_out.{n}.weight"));
        keys.push(format!("norm_out.{n}.bias"));
    }

    keys
}

/// Validate a loaded transformer against `cfg`: exact key coverage (no missing, no extra) and the
/// shapes of the dimension-bearing entry points.
pub fn validate_transformer(w: &Weights, cfg: &BooguConfig) -> Result<()> {
    use std::collections::BTreeSet;

    let expected: BTreeSet<String> = expected_transformer_keys(cfg).into_iter().collect();
    // A pre-quantized snapshot (E8) replaces each Linear `{base}.weight` with the packed triple
    // `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`. The `{base}.weight` key is
    // still present, so drop the two quant-only artifacts before the coverage diff.
    let actual: BTreeSet<String> = w
        .keys()
        .filter(|k| !k.ends_with(".scales") && !k.ends_with(".biases"))
        .map(str::to_string)
        .collect();

    let missing: Vec<&String> = expected.difference(&actual).collect();
    let extra: Vec<&String> = actual.difference(&expected).collect();
    if !missing.is_empty() || !extra.is_empty() {
        let head = |v: &[&String]| {
            v.iter()
                .take(8)
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        return Err(Error::Msg(format!(
            "boogu transformer key mismatch vs config: {} missing [{}], {} extra [{}]",
            missing.len(),
            head(&missing),
            extra.len(),
            head(&extra),
        )));
    }

    // Shape checks on the dimension-bearing tensors (Linear weight = [out, in]).
    let h = cfg.hidden_size as i32;
    check_shape(w, "x_embedder.weight", &[h, cfg.patch_in_dim() as i32])?;
    check_shape(
        w,
        "norm_out.linear_2.weight",
        &[cfg.patch_out_dim() as i32, h],
    )?;
    check_shape(
        w,
        "time_caption_embed.caption_embedder.1.weight",
        &[h, cfg.preprocessed_instruction_feat_dim() as i32],
    )?;
    check_shape(
        w,
        "time_caption_embed.timestep_embedder.linear_1.weight",
        &[cfg.modulation_dim() as i32, 256],
    )?;
    // A representative SwiGLU FFN: validates the multiple_of rounding.
    check_shape(
        w,
        "single_stream_layers.0.feed_forward.linear_1.weight",
        &[cfg.ffn_inner_dim() as i32, h],
    )?;
    // GQA: q projects to all heads, k/v to kv heads.
    let head_dim = cfg.head_dim() as i32;
    check_shape(
        w,
        "single_stream_layers.0.attn.to_q.weight",
        &[cfg.num_attention_heads as i32 * head_dim, h],
    )?;
    check_shape(
        w,
        "single_stream_layers.0.attn.to_k.weight",
        &[cfg.num_kv_heads as i32 * head_dim, h],
    )?;
    Ok(())
}

fn check_shape(w: &Weights, key: &str, expected: &[i32]) -> Result<()> {
    // A packed (quantized) `{base}.weight` is u32-codes with a different on-disk shape; skip the
    // dense-shape check when a sibling `{base}.scales` marks it pre-quantized (E8).
    if let Some(base) = key.strip_suffix(".weight") {
        if w.get(&format!("{base}.scales")).is_some() {
            return Ok(());
        }
    }
    let t = w.require(key)?;
    if t.shape() != expected {
        return Err(Error::Msg(format!(
            "boogu: {key} shape {:?}, expected {:?}",
            t.shape(),
            expected
        )));
    }
    Ok(())
}

// ===================================================================================================
// E8-1 — on-disk pre-quantization (mirror SCAIL-2 `convert::quantize_scail2_dit`).
//
// Quantizing **at load** (`BooguPipeline::quantize`) still materializes the full dense bf16 stack
// before packing, so the load-time peak (and therefore `minMemoryGb`) is the *bf16* footprint. This
// offline converter moves that transient to a one-off convert: the shipped snapshot is already
// packed, so the consume side (`quant::lin`/`embedding`, which auto-detect `.scales`) never builds a
// dense weight. Quant targets are derived from the SAME structure as `expected_transformer_keys`
// (the exact `quant::lin` call sites) so the predicate can't drift from the load path.
// ===================================================================================================

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::ops::quantize as mlx_quantize;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::text_encoder::BooguTextEncoderConfig;

/// Group size the packing uses — the codebase-wide default ([`crate::quant::GROUP_SIZE`], 64), the
/// same value the auto-detecting loader infers from the packed shapes.
pub const QUANT_GROUP_SIZE: i32 = crate::quant::GROUP_SIZE;

/// The Linear `…​.weight` keys of an attention module that [`crate::quant::lin`] quantizes
/// (`to_q/to_k/to_v/to_out.0`; the per-head `norm_q/norm_k` stay dense).
fn attn_lin_keys(prefix: &str) -> Vec<String> {
    ["to_q", "to_k", "to_v", "to_out.0"]
        .iter()
        .map(|p| format!("{prefix}.{p}.weight"))
        .collect()
}

/// The SwiGLU FFN Linear weights (`linear_1/2/3`).
fn ffn_lin_keys(prefix: &str) -> Vec<String> {
    ["linear_1", "linear_2", "linear_3"]
        .iter()
        .map(|p| format!("{prefix}.{p}.weight"))
        .collect()
}

/// The single `LuminaRMSNormZero` modulation Linear (`linear.weight`; its `linear.bias` + `norm.weight`
/// stay dense).
fn mod_lin_key(prefix: &str) -> String {
    format!("{prefix}.linear.weight")
}

/// The exact set of DiT Linear `…​.weight` keys quantized at load (`BooguTransformer::quantize` →
/// block `quantize` → `quant::lin`). A strict subset of [`expected_transformer_keys`]; everything
/// else (norms, embeddings table, modulation/Linear biases, `image_index_embedding`) stays dense.
pub fn transformer_quant_targets(cfg: &BooguConfig) -> std::collections::BTreeSet<String> {
    let mut t = std::collections::BTreeSet::new();
    // Top-level Linears.
    for e in ["x_embedder", "ref_image_patch_embedder"] {
        t.insert(format!("{e}.weight"));
    }
    t.insert("time_caption_embed.caption_embedder.1.weight".to_string());
    t.insert("time_caption_embed.timestep_embedder.linear_1.weight".to_string());
    t.insert("time_caption_embed.timestep_embedder.linear_2.weight".to_string());
    t.insert("norm_out.linear_1.weight".to_string());
    t.insert("norm_out.linear_2.weight".to_string());

    let block_no_mod = |prefix: &str, t: &mut std::collections::BTreeSet<String>| {
        t.extend(attn_lin_keys(&format!("{prefix}.attn")));
        t.extend(ffn_lin_keys(&format!("{prefix}.feed_forward")));
    };
    // Refiners: context = PlainBlock (attn+ffn); noise/ref-image = ModBlock (attn+ffn+norm1.linear).
    for i in 0..cfg.num_refiner_layers {
        block_no_mod(&format!("context_refiner.{i}"), &mut t);
        for name in ["noise_refiner", "ref_image_refiner"] {
            block_no_mod(&format!("{name}.{i}"), &mut t);
            t.insert(mod_lin_key(&format!("{name}.{i}.norm1")));
        }
    }
    // Double-stream blocks.
    for i in 0..cfg.num_double_stream_layers {
        let p = format!("double_stream_layers.{i}");
        for side in ["img", "instruct"] {
            for proj in ["to_q", "to_k", "to_v", "out"] {
                t.insert(format!(
                    "{p}.img_instruct_attn.processor.{side}_{proj}.weight"
                ));
            }
        }
        t.insert(format!("{p}.img_instruct_attn.to_out.0.weight"));
        t.extend(attn_lin_keys(&format!("{p}.img_self_attn")));
        t.extend(ffn_lin_keys(&format!("{p}.img_feed_forward")));
        t.extend(ffn_lin_keys(&format!("{p}.instruct_feed_forward")));
        for n in [
            "img_norm1",
            "img_norm2",
            "img_norm3",
            "instruct_norm1",
            "instruct_norm2",
        ] {
            t.insert(mod_lin_key(&format!("{p}.{n}")));
        }
    }
    // Single-stream blocks (ModBlock).
    for i in 0..cfg.num_single_stream_layers() {
        let p = format!("single_stream_layers.{i}");
        block_no_mod(&p, &mut t);
        t.insert(mod_lin_key(&format!("{p}.norm1")));
    }
    t
}

/// The exact set of Qwen3-VL **text tower** Linear `…​.weight` keys quantized at load
/// (`BooguTextEncoder::quantize`): each decoder layer's `self_attn` `q/k/v/o_proj` and `mlp`
/// `gate/up/down_proj`. The `embed_tokens` table stays **dense** (see `BooguTextEncoder::quantize`),
/// as do the **vision tower** (`model.visual.*`, runs f32), the RMSNorms, and `lm_head`.
pub fn mllm_quant_targets(num_layers: i32) -> std::collections::BTreeSet<String> {
    let mut t = std::collections::BTreeSet::new();
    for i in 0..num_layers {
        for p in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            t.insert(format!("model.language_model.layers.{i}.{p}.weight"));
        }
    }
    t
}

/// Group-wise affine Q4/Q8-pack every `targets` `{base}.weight` in `map` (cast to bf16 first so the
/// packing is byte-identical to the load-time [`mlx_gen::adapters::AdaptableLinear::quantize`] and to
/// mflux), emitting the packed triple `{base}.{weight,scales,biases}`; every other tensor passes
/// through unchanged. Result = the exact key layout `quant::{lin,embedding}` read back.
fn quantize_targets(
    map: HashMap<String, Array>,
    targets: &std::collections::BTreeSet<String>,
    bits: i32,
    group_size: i32,
) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::with_capacity(map.len());
    for (k, v) in map {
        if targets.contains(&k) {
            let base = k
                .strip_suffix(".weight")
                .expect("quant target ends with .weight");
            let wbf16 = v.as_dtype(Dtype::Bfloat16)?;
            let (wq, scales, biases) = mlx_quantize(&wbf16, group_size, bits)?;
            out.insert(format!("{base}.weight"), wq);
            out.insert(format!("{base}.scales"), scales);
            out.insert(format!("{base}.biases"), biases);
        } else {
            out.insert(k, v);
        }
    }
    Ok(out)
}

/// Read every tensor of a component dir (all shards) into an owned key→`Array` map (MLX arrays are
/// ref-counted, so the clone is a handle copy).
fn load_dir_map(dir: &Path) -> Result<HashMap<String, Array>> {
    let w = Weights::from_dir(dir)?;
    Ok(w.keys()
        .map(|k| (k.to_string(), w.get(k).expect("listed key").clone()))
        .collect())
}

/// Materialize + write a key→`Array` map to a single consolidated `.safetensors` (the loader globs
/// all `*.safetensors` in the dir, so one file replaces the source's shards).
fn save_map(path: &Path, map: &HashMap<String, Array>) -> Result<()> {
    eval(map.values().collect::<Vec<_>>())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Array::save_safetensors(
        map.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        path,
    )?;
    Ok(())
}

/// Copy `src_dir/config.json` to `dst_dir/config.json` with a `"quantization": {bits, group_size}`
/// manifest block added (informational + the E9 turnkey marker; the loader itself auto-detects packed
/// weights per-key, so it does not depend on this block). A missing source config starts empty.
fn write_quantized_config(
    src_dir: &Path,
    dst_dir: &Path,
    bits: i32,
    group_size: i32,
) -> Result<()> {
    let src_cfg = src_dir.join("config.json");
    let mut v: serde_json::Value = if src_cfg.exists() {
        serde_json::from_str(&std::fs::read_to_string(&src_cfg)?)
            .map_err(|e| Error::Msg(format!("boogu: parse {}: {e}", src_cfg.display())))?
    } else {
        serde_json::json!({})
    };
    v["quantization"] = serde_json::json!({ "bits": bits, "group_size": group_size });
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| Error::Msg(format!("boogu: serialize config.json: {e}")))?;
    std::fs::create_dir_all(dst_dir)?;
    std::fs::write(dst_dir.join("config.json"), text)?;
    Ok(())
}

/// Offline one-shot: read the dense `{src_root}/transformer/` (all shards) and write a pre-quantized
/// `{dst_root}/transformer/diffusion_pytorch_model.safetensors` (packed Q4/Q8) + `config.json` (with
/// the `quantization` manifest). `group_size` is the mflux/reference default of 64
/// ([`QUANT_GROUP_SIZE`]).
pub fn quantize_transformer(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    let cfg = BooguConfig::from_snapshot(src_root)?;
    let src = src_root.join("transformer");
    let dst = dst_root.join("transformer");
    let map = load_dir_map(&src)?;
    let quantized = quantize_targets(
        map,
        &transformer_quant_targets(&cfg),
        bits,
        QUANT_GROUP_SIZE,
    )?;
    save_map(&dst.join("diffusion_pytorch_model.safetensors"), &quantized)?;
    write_quantized_config(&src, &dst, bits, QUANT_GROUP_SIZE)?;
    Ok(())
}

/// Offline one-shot: read the dense `{src_root}/mllm/` and write a pre-quantized
/// `{dst_root}/mllm/model.safetensors` (the Qwen3-VL **text tower** packed Q4/Q8; the vision tower +
/// norms + lm_head pass through dense) + `config.json` (with the `quantization` manifest). The
/// tokenizer / processor JSONs are not touched here — the E9 turnkey assembly copies them alongside.
pub fn quantize_mllm(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    let num_layers = BooguTextEncoderConfig::qwen3_vl_8b().num_layers;
    let src = src_root.join("mllm");
    let dst = dst_root.join("mllm");
    let map = load_dir_map(&src)?;
    let quantized = quantize_targets(map, &mllm_quant_targets(num_layers), bits, QUANT_GROUP_SIZE)?;
    save_map(&dst.join("model.safetensors"), &quantized)?;
    write_quantized_config(&src, &dst, bits, QUANT_GROUP_SIZE)?;
    Ok(())
}

/// Copy a single file `src → dst` (creating `dst`'s parent).
fn copy_file(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(src, dst).map_err(|e| {
        Error::Msg(format!(
            "boogu turnkey copy {} → {}: {e}",
            src.display(),
            dst.display()
        ))
    })?;
    Ok(())
}

/// Copy every regular file in `src_dir` (non-recursive — the VAE dir is flat) into `dst_dir`.
fn copy_dir_flat(src_dir: &Path, dst_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dst_dir)?;
    for entry in std::fs::read_dir(src_dir)? {
        let path = entry?.path();
        if path.is_file() {
            let name = path.file_name().expect("dir entry has a name");
            copy_file(&path, &dst_dir.join(name))?;
        }
    }
    Ok(())
}

/// Assemble a **complete, `from_snapshot`-loadable turnkey** at `dst_root` from the dense source
/// snapshot `src_root`, at `bits` (E9). Pre-quantizes the two big stacks on disk and copies the
/// small files the loaders need verbatim:
///   - `transformer/` — packed DiT + `config.json` (`quantize_transformer`),
///   - `mllm/` — packed Qwen3-VL text tower (vision tower / embedding / norms dense) + `config.json`
///     (`quantize_mllm`) + `tokenizer.json` (what [`crate::BooguTokenizer`] reads),
///   - `vae/` — the dense FLUX.1 `AutoencoderKL`, copied unchanged.
///
/// The result is lean (no source `.bin`/extra pickles) and loads through the exact
/// [`crate::BooguPipeline::from_snapshot`] path — the published `SceneWorks/boogu-image-mlx/<variant>`.
pub fn assemble_quantized_snapshot(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    quantize_transformer(src_root, dst_root, bits)?;
    quantize_mllm(src_root, dst_root, bits)?;
    copy_file(
        &src_root.join("mllm").join("tokenizer.json"),
        &dst_root.join("mllm").join("tokenizer.json"),
    )?;
    copy_dir_flat(&src_root.join("vae"), &dst_root.join("vae"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_key_count_matches_published_base() {
        let cfg = BooguConfig::base();
        let keys = expected_transformer_keys(&cfg);
        let unique: std::collections::BTreeSet<_> = keys.iter().collect();
        assert_eq!(keys.len(), unique.len(), "no duplicate expected keys");
        // 26 (context×2) + 30 (noise×2) + 30 (ref×2) + 360 (double×8) + 480 (single×32) + 16 top-level.
        assert_eq!(keys.len(), 942);
    }

    /// Real-weight architecture validation against the published Base snapshot.
    /// `BOOGU_BASE_DIR=<snapshot root>` (the dir containing `transformer/`).
    #[test]
    #[ignore = "needs real weights: set BOOGU_BASE_DIR"]
    fn validate_real_base_snapshot() {
        let root = std::env::var("BOOGU_BASE_DIR").expect("set BOOGU_BASE_DIR");
        let cfg = BooguConfig::from_snapshot(&root).unwrap();
        assert_eq!(cfg, BooguConfig::base());
        let w = Weights::from_dir(format!("{root}/transformer")).unwrap();
        validate_transformer(&w, &cfg).unwrap();
        assert_eq!(w.len(), 942);
    }

    // ── E8-1 quant-target predicate ─────────────────────────────────────────────────────────────
    #[test]
    fn transformer_quant_targets_subset_of_expected() {
        let cfg = BooguConfig::base();
        let expected: std::collections::BTreeSet<String> =
            expected_transformer_keys(&cfg).into_iter().collect();
        let targets = transformer_quant_targets(&cfg);
        // Every quant target is a real, dense Linear weight in the architecture key set.
        for k in &targets {
            assert!(
                k.ends_with(".weight"),
                "quant target must be a .weight: {k}"
            );
            assert!(
                expected.contains(k),
                "quant target not in expected keys: {k}"
            );
        }
        // 7 top-level + 14 context(2×7) + 16 noise(2×8) + 16 ref(2×8) + 192 double(8×24) + 256 single(32×8).
        assert_eq!(targets.len(), 501, "DiT Linear count");
    }

    #[test]
    fn mllm_quant_targets_count() {
        // 36 layers × (4 attn proj + 3 mlp proj); the embedding stays dense.
        assert_eq!(mllm_quant_targets(36).len(), 36 * 7);
    }

    #[test]
    fn quantize_targets_packs_only_targets() {
        let mut map: HashMap<String, Array> = HashMap::new();
        // A target Linear (in-dim a multiple of the group size), its dense bias, and a non-target norm.
        map.insert(
            "x_embedder.weight".into(),
            Array::ones::<f32>(&[128, 64]).unwrap(),
        );
        map.insert(
            "x_embedder.bias".into(),
            Array::zeros::<f32>(&[128]).unwrap(),
        );
        map.insert(
            "single_stream_layers.0.norm2.weight".into(),
            Array::ones::<f32>(&[64]).unwrap(),
        );
        let targets: std::collections::BTreeSet<String> =
            ["x_embedder.weight".to_string()].into_iter().collect();

        let out = quantize_targets(map, &targets, 4, 64).unwrap();

        // The target became the packed triple (u32 codes + scales + biases)…
        let wq = out.get("x_embedder.weight").expect("packed weight");
        assert_eq!(wq.dtype(), Dtype::Uint32, "Q4 codes are u32-packed");
        assert!(out.contains_key("x_embedder.scales"));
        assert!(out.contains_key("x_embedder.biases"));
        // …its dense bias and the non-target norm passed through untouched.
        assert!(out.contains_key("x_embedder.bias"));
        let norm = out
            .get("single_stream_layers.0.norm2.weight")
            .expect("dense norm");
        assert_eq!(norm.dtype(), Dtype::Float32, "non-target weight unchanged");
        assert!(!out.contains_key("single_stream_layers.0.norm2.scales"));
    }

    /// Real-weight convert→reload parity: pre-quantize the Base `transformer/` to Q4 on disk, reload
    /// it through the packed loader (+ quant-aware `validate_transformer`), and assert it produces the
    /// same velocity as the load-time-quantized DiT — i.e. the on-disk converter ≡ `.quantize()`.
    /// `BOOGU_BASE_DIR=<snapshot root>`. Heavy on host RAM (loads the 20.6 GB dense DiT once), light
    /// on GPU (one forward).
    #[test]
    #[ignore = "needs real weights: set BOOGU_BASE_DIR"]
    fn on_disk_quant_matches_load_time_quant() {
        use mlx_rs::ops::{multiply, sqrt, sum};
        let root = std::env::var("BOOGU_BASE_DIR").expect("set BOOGU_BASE_DIR");
        let root = std::path::PathBuf::from(root);
        let cfg = BooguConfig::from_snapshot(&root).unwrap();

        // Load-time-quantized reference DiT.
        let mut dit_load = crate::load_transformer(&root).unwrap();
        dit_load.quantize(4).unwrap();

        // On-disk-quantized DiT: convert to a temp dir, then reload through the packed path.
        let dst = std::env::temp_dir().join("boogu_e8_q4_convert");
        quantize_transformer(&root, &dst, 4).unwrap();
        let w = Weights::from_dir(dst.join("transformer")).unwrap();
        validate_transformer(&w, &cfg).unwrap();
        let dit_disk = crate::transformer::BooguTransformer::from_weights(&w, &cfg).unwrap();

        // Identical inputs → compare velocity.
        let lat = mlx_rs::random::normal::<f32>(&[1, 16, 32, 32], None, None, None).unwrap();
        let t = Array::from_slice(&[0.5f32], &[1]);
        let instr = mlx_rs::random::normal::<f32>(&[1, 8, 4096], None, None, None).unwrap();
        let mask = Array::ones::<i32>(&[1, 8]).unwrap();
        let v_load = dit_load.forward(&lat, &t, &instr, &mask).unwrap();
        let v_disk = dit_disk.forward(&lat, &t, &instr, &mask).unwrap();

        let cosine = |a: &Array, b: &Array| -> f32 {
            let a = a.as_dtype(Dtype::Float32).unwrap();
            let b = b.as_dtype(Dtype::Float32).unwrap();
            let dot = sum(multiply(&a, &b).unwrap(), false).unwrap();
            let na = sqrt(sum(multiply(&a, &a).unwrap(), false).unwrap()).unwrap();
            let nb = sqrt(sum(multiply(&b, &b).unwrap(), false).unwrap()).unwrap();
            (dot / (na * nb)).item::<f32>()
        };
        let c = cosine(&v_load, &v_disk);
        println!("on-disk vs load-time Q4 velocity cosine = {c:.7}");
        assert!(
            c > 0.9999,
            "on-disk Q4 must match load-time Q4 (cosine {c})"
        );
        let _ = std::fs::remove_dir_all(&dst);
    }
}
