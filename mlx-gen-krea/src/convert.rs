//! Krea 2 transformer "conversion" = **architecture validation** + **on-disk Q4/Q8 turnkey assembly**.
//!
//! The published `krea/Krea-2-Turbo` diffusers checkpoint uses dotted keys that map 1:1 onto the
//! `Krea2Transformer2DModel` module tree, so [`mlx_gen::weights::Weights::from_dir`] loads them
//! directly — there is no fork-style key remap (the VAE's NCHW→NHWC conv transpose is applied in the
//! VAE loader at load time, sc-7570, not as a pre-conversion). So the "converter" is two things:
//!
//! 1. **Architecture validation** ([`validate_transformer`]) — prove the on-disk tensor set exactly
//!    matches the architecture implied by [`Krea2Config`] before the DiT forward (sc-7568) trusts it:
//!    every expected key present, no stray extras, and the shape-bearing entry points sized as the
//!    config says. Catches a wrong variant / truncated download / config-weight mismatch loudly at
//!    load instead of as garbage latents.
//! 2. **On-disk pre-quantization** ([`assemble_quantized_snapshot`]) — produce the lean, packed
//!    Q4/Q8 turnkey the worker ships, mirroring `mlx-gen-boogu`'s `assemble_quantized_snapshot`.
//!    Quantizing *at load* still materializes the dense bf16 stack first (so `minMemoryGb` is the
//!    bf16 footprint); the offline pack moves that transient to a one-off convert.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use mlx_gen::quant::{load_dir_map, quantize_map, save_map};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::Array;

use crate::config::Krea2Config;

/// Group size every Krea group-wise-affine quantization uses (pack + load) — the codebase default 64.
/// Every Krea quant-target Linear has an input dim divisible by 64 (hidden 6144, ff 16384, text 2560,
/// text-ff 6912, TE 4096/1024/9728), so 64 packs the whole model (unlike Boogu's 3360 hidden, which
/// forced 32).
pub const QUANT_GROUP_SIZE: i32 = 64;

// ===================================================================================================
// Architecture validation — the exact 430-tensor `Krea2Transformer2DModel` key tree.
// ===================================================================================================

/// GQA / full attention Linear weights: `to_q/to_k/to_v/to_gate/to_out.0` + per-head `norm_q/norm_k`.
/// Identical shape across the text-fusion (full, 20/20) and single-stream (GQA, 48/12) blocks; only
/// the projection widths differ.
fn attn_keys(prefix: &str) -> Vec<String> {
    [
        "norm_q", "norm_k", "to_q", "to_k", "to_v", "to_gate", "to_out.0",
    ]
    .iter()
    .map(|p| format!("{prefix}.{p}.weight"))
    .collect()
}

/// SwiGLU feed-forward (`gate`/`up` in, `down` out), all bias-free.
fn ff_keys(prefix: &str) -> Vec<String> {
    ["gate", "up", "down"]
        .iter()
        .map(|p| format!("{prefix}.{p}.weight"))
        .collect()
}

/// A text-fusion block (`layerwise_blocks` / `refiner_blocks`): RMSNorm-attn(+gate)-RMSNorm-SwiGLU,
/// no per-block modulation table.
fn text_block_keys(prefix: &str) -> Vec<String> {
    let mut k = attn_keys(&format!("{prefix}.attn"));
    k.extend(ff_keys(&format!("{prefix}.ff")));
    k.push(format!("{prefix}.norm1.weight"));
    k.push(format!("{prefix}.norm2.weight"));
    k
}

/// A single-stream `transformer_block`: a text-fusion-shaped block plus the per-block 6-factor
/// `scale_shift_table` (the `DoubleSharedModulation` offset added to the shared `time_mod_proj`).
fn single_block_keys(prefix: &str) -> Vec<String> {
    let mut k = text_block_keys(prefix);
    k.push(format!("{prefix}.scale_shift_table"));
    k
}

/// The complete set of transformer tensor keys implied by `cfg` (= the published 430 for Turbo/Raw).
pub fn expected_transformer_keys(cfg: &Krea2Config) -> Vec<String> {
    let mut keys = Vec::new();

    // Image patch embed.
    keys.push("img_in.weight".into());
    keys.push("img_in.bias".into());

    // Text input projection: RMSNorm(text) → Linear(text→hidden) → Linear(hidden→hidden).
    keys.push("txt_in.norm.weight".into());
    for n in ["linear_1", "linear_2"] {
        keys.push(format!("txt_in.{n}.weight"));
        keys.push(format!("txt_in.{n}.bias"));
    }

    // Timestep embed + the shared 6-factor modulation projection.
    for n in ["linear_1", "linear_2"] {
        keys.push(format!("time_embed.{n}.weight"));
        keys.push(format!("time_embed.{n}.bias"));
    }
    keys.push("time_mod_proj.weight".into());
    keys.push("time_mod_proj.bias".into());

    // text_fusion: layerwise (cross-layer-axis aggregator) → projector(12→1) → refiner (token-axis).
    for i in 0..cfg.num_layerwise_text_blocks {
        keys.extend(text_block_keys(&format!(
            "text_fusion.layerwise_blocks.{i}"
        )));
    }
    keys.push("text_fusion.projector.weight".into());
    for i in 0..cfg.num_refiner_text_blocks {
        keys.extend(text_block_keys(&format!("text_fusion.refiner_blocks.{i}")));
    }

    // The single-stream stack.
    for i in 0..cfg.num_layers {
        keys.extend(single_block_keys(&format!("transformer_blocks.{i}")));
    }

    // Continuous-AdaLN output (2-factor scale/shift table).
    keys.push("final_layer.linear.weight".into());
    keys.push("final_layer.linear.bias".into());
    keys.push("final_layer.norm.weight".into());
    keys.push("final_layer.scale_shift_table".into());

    keys
}

/// Validate a loaded transformer against `cfg`: exact key coverage (no missing, no extra) and the
/// shapes of the dimension-bearing entry points.
pub fn validate_transformer(w: &Weights, cfg: &Krea2Config) -> Result<()> {
    let expected: BTreeSet<String> = expected_transformer_keys(cfg).into_iter().collect();
    // A pre-quantized snapshot replaces each Linear `{base}.weight` with the packed triple
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
            "krea transformer key mismatch vs config: {} missing [{}], {} extra [{}]",
            missing.len(),
            head(&missing),
            extra.len(),
            head(&extra),
        )));
    }

    // Shape checks on the dimension-bearing tensors (Linear weight = [out, in]).
    let h = cfg.hidden_size as i32;
    check_shape(w, "img_in.weight", &[h, cfg.in_channels as i32])?;
    check_shape(w, "final_layer.linear.weight", &[cfg.in_channels as i32, h])?;
    check_shape(w, "final_layer.scale_shift_table", &[2, h])?;
    check_shape(
        w,
        "txt_in.linear_1.weight",
        &[h, cfg.text_hidden_dim as i32],
    )?;
    check_shape(w, "txt_in.linear_2.weight", &[h, h])?;
    check_shape(
        w,
        "time_embed.linear_1.weight",
        &[h, cfg.timestep_embed_dim as i32],
    )?;
    check_shape(w, "time_mod_proj.weight", &[cfg.time_mod_dim() as i32, h])?;
    check_shape(
        w,
        "text_fusion.projector.weight",
        &[1, cfg.num_text_layers as i32],
    )?;
    // A representative text-fusion block (full attention, text width).
    let th = cfg.text_hidden_dim as i32;
    check_shape(
        w,
        "text_fusion.layerwise_blocks.0.attn.to_q.weight",
        &[th, th],
    )?;
    check_shape(
        w,
        "text_fusion.layerwise_blocks.0.ff.gate.weight",
        &[cfg.text_intermediate_size as i32, th],
    )?;
    // A representative single-stream block: GQA + the SwiGLU FFN + the 6-factor modulation table.
    check_shape(
        w,
        "transformer_blocks.0.attn.to_q.weight",
        &[cfg.q_dim() as i32, h],
    )?;
    check_shape(
        w,
        "transformer_blocks.0.attn.to_k.weight",
        &[cfg.kv_dim() as i32, h],
    )?;
    check_shape(
        w,
        "transformer_blocks.0.attn.to_gate.weight",
        &[cfg.q_dim() as i32, h],
    )?;
    check_shape(
        w,
        "transformer_blocks.0.ff.gate.weight",
        &[cfg.intermediate_size as i32, h],
    )?;
    check_shape(
        w,
        "transformer_blocks.0.scale_shift_table",
        &[Krea2Config::MOD_FACTORS as i32, h],
    )?;
    Ok(())
}

fn check_shape(w: &Weights, key: &str, expected: &[i32]) -> Result<()> {
    // A packed (quantized) `{base}.weight` is u32-codes with a different on-disk shape; skip the
    // dense-shape check when a sibling `{base}.scales` marks it pre-quantized.
    if let Some(base) = key.strip_suffix(".weight") {
        if w.get(&format!("{base}.scales")).is_some() {
            return Ok(());
        }
    }
    let t = w.require(key)?;
    if t.shape() != expected {
        return Err(Error::Msg(format!(
            "krea: {key} shape {:?}, expected {:?}",
            t.shape(),
            expected
        )));
    }
    Ok(())
}

// ===================================================================================================
// On-disk pre-quantization — the Q4/Q8 turnkey assembly (mirror boogu `assemble_quantized_snapshot`).
//
// Quant targets are derived from the SAME structure as `expected_transformer_keys` (the exact future
// `quant::lin` call sites — the 256 BF16 Linears) so the predicate can't drift from the load path.
// ===================================================================================================

/// The DiT Linear `….weight` keys quantized at load (the 256 BF16 Linears: per single-stream block and
/// per text-fusion block, the attention `to_q/to_k/to_v/to_gate/to_out.0` + SwiGLU `gate/up/down`). A
/// strict subset of [`expected_transformer_keys`]; everything else (norms, embedders, `time_mod_proj`,
/// `scale_shift_table`s, `projector`, per-head `norm_q/norm_k`) stays dense.
pub fn transformer_quant_targets(cfg: &Krea2Config) -> BTreeSet<String> {
    let mut t = BTreeSet::new();
    let block = |prefix: &str, t: &mut BTreeSet<String>| {
        for p in ["to_q", "to_k", "to_v", "to_gate", "to_out.0"] {
            t.insert(format!("{prefix}.attn.{p}.weight"));
        }
        for p in ["gate", "up", "down"] {
            t.insert(format!("{prefix}.ff.{p}.weight"));
        }
    };
    for i in 0..cfg.num_layerwise_text_blocks {
        block(&format!("text_fusion.layerwise_blocks.{i}"), &mut t);
    }
    for i in 0..cfg.num_refiner_text_blocks {
        block(&format!("text_fusion.refiner_blocks.{i}"), &mut t);
    }
    for i in 0..cfg.num_layers {
        block(&format!("transformer_blocks.{i}"), &mut t);
    }
    t
}

/// The Qwen3-VL **text tower** Linear `….weight` keys quantized at load: each decoder layer's
/// `self_attn` `q/k/v/o_proj` + `mlp` `gate/up/down_proj`. The `embed_tokens` table stays **dense**,
/// as do the **vision tower** (`visual.*`, runs f32 for the edit/VLM path — unused for T2I), the
/// RMSNorms, and `lm_head`. NOTE Krea's TE keys are under `language_model.*` (no `model.` prefix).
pub fn text_encoder_quant_targets(num_layers: usize) -> BTreeSet<String> {
    let mut t = BTreeSet::new();
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
            t.insert(format!("language_model.layers.{i}.{p}.weight"));
        }
    }
    t
}

/// Group-wise affine Q4/Q8-pack every `targets` `{base}.weight` in `map` (cast to bf16 first so the
/// packing is byte-identical to the load-time [`mlx_gen::adapters::AdaptableLinear::quantize`] and to
/// the reference), emitting the packed triple `{base}.{weight,scales,biases}`; every other tensor
/// passes through unchanged.
fn quantize_targets(
    map: HashMap<String, Array>,
    targets: &BTreeSet<String>,
    bits: i32,
    group_size: i32,
) -> Result<HashMap<String, Array>> {
    quantize_map(map, bits, group_size, |base| {
        targets.contains(&format!("{base}.weight"))
    })
}

/// Copy `src_dir/config.json` to `dst_dir/config.json` with a `"quantization": {bits, group_size}`
/// manifest block added (informational; the loader auto-detects packed weights per-key). A missing
/// source config starts empty.
fn write_quantized_config(
    src_dir: &Path,
    dst_dir: &Path,
    bits: i32,
    group_size: i32,
) -> Result<()> {
    let src_cfg = src_dir.join("config.json");
    let mut v: serde_json::Value = if src_cfg.exists() {
        serde_json::from_str(&std::fs::read_to_string(&src_cfg)?)
            .map_err(|e| Error::Msg(format!("krea: parse {}: {e}", src_cfg.display())))?
    } else {
        serde_json::json!({})
    };
    v["quantization"] = serde_json::json!({ "bits": bits, "group_size": group_size });
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| Error::Msg(format!("krea: serialize config.json: {e}")))?;
    std::fs::create_dir_all(dst_dir)?;
    std::fs::write(dst_dir.join("config.json"), text)?;
    Ok(())
}

/// Offline one-shot: read the dense `{src_root}/transformer/` (all shards) and write a pre-quantized
/// `{dst_root}/transformer/diffusion_pytorch_model.safetensors` (packed Q4/Q8) + `config.json`.
pub fn quantize_transformer(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    let cfg = Krea2Config::from_snapshot(src_root)?;
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

/// Offline one-shot: read the dense `{src_root}/text_encoder/` (Qwen3-VL-4B) and write a pre-quantized
/// `{dst_root}/text_encoder/model.safetensors` (text tower packed; vision tower + embedding + norms
/// dense) + `config.json`. The number of decoder layers is read from the source `config.json`.
pub fn quantize_text_encoder(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    let num_layers = read_te_num_layers(src_root)?;
    let src = src_root.join("text_encoder");
    let dst = dst_root.join("text_encoder");
    let map = load_dir_map(&src)?;
    let quantized = quantize_targets(
        map,
        &text_encoder_quant_targets(num_layers),
        bits,
        QUANT_GROUP_SIZE,
    )?;
    save_map(&dst.join("model.safetensors"), &quantized)?;
    write_quantized_config(&src, &dst, bits, QUANT_GROUP_SIZE)?;
    Ok(())
}

/// Read `text_config.num_hidden_layers` from `{src_root}/text_encoder/config.json` (default 36).
fn read_te_num_layers(src_root: &Path) -> Result<usize> {
    let path = src_root.join("text_encoder").join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| Error::Msg(format!("krea: read {}: {e}", path.display())))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Msg(format!("krea: parse {}: {e}", path.display())))?;
    Ok(v.get("text_config")
        .and_then(|t| t.get("num_hidden_layers"))
        .and_then(serde_json::Value::as_u64)
        .map(|n| n as usize)
        .unwrap_or(36))
}

/// Copy a single file `src → dst` (creating `dst`'s parent).
fn copy_file(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(src, dst).map_err(|e| {
        Error::Msg(format!(
            "krea turnkey copy {} → {}: {e}",
            src.display(),
            dst.display()
        ))
    })?;
    Ok(())
}

/// Copy every regular file in `src_dir` (non-recursive) into `dst_dir`. Missing source = no-op.
fn copy_dir_flat(src_dir: &Path, dst_dir: &Path) -> Result<()> {
    if !src_dir.exists() {
        return Ok(());
    }
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
/// snapshot `src_root`, at `bits`. Pre-quantizes the two big stacks on disk and copies the small files
/// the loaders need verbatim:
///   - `transformer/` — packed DiT + `config.json` ([`quantize_transformer`]),
///   - `text_encoder/` — packed Qwen3-VL text tower (vision/embedding/norms dense) + `config.json`
///     ([`quantize_text_encoder`]),
///   - `vae/` — the dense `AutoencoderKLQwenImage`, copied unchanged (F32),
///   - `tokenizer/`, `scheduler/` — copied unchanged,
///   - `model_index.json` — copied unchanged (carries `text_encoder_select_layers` + `patch_size`).
///
/// The result loads through the exact `KreaPipeline::from_snapshot` path (the published
/// `SceneWorks/krea-2-turbo-mlx`, sc-7573).
///
/// **License (Krea 2 Community License):** the re-host (sc-7573) must additionally carry the license,
/// prefix the model name with "Krea", and retain attribution — handled at publish time, not here.
pub fn assemble_quantized_snapshot(src_root: &Path, dst_root: &Path, bits: i32) -> Result<()> {
    quantize_transformer(src_root, dst_root, bits)?;
    quantize_text_encoder(src_root, dst_root, bits)?;
    copy_dir_flat(&src_root.join("vae"), &dst_root.join("vae"))?;
    copy_dir_flat(&src_root.join("tokenizer"), &dst_root.join("tokenizer"))?;
    copy_dir_flat(&src_root.join("scheduler"), &dst_root.join("scheduler"))?;
    let idx = src_root.join("model_index.json");
    if idx.exists() {
        copy_file(&idx, &dst_root.join("model_index.json"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    #[test]
    fn expected_key_count_matches_published_turbo() {
        let cfg = Krea2Config::turbo();
        let keys = expected_transformer_keys(&cfg);
        let unique: BTreeSet<_> = keys.iter().collect();
        assert_eq!(keys.len(), unique.len(), "no duplicate expected keys");
        // 17 top-level + 49 text_fusion (2×12 layerwise + 1 projector + 2×12 refiner) + 364 blocks
        // (28×13) = 430, matching the published safetensors index exactly.
        assert_eq!(keys.len(), 430);
    }

    #[test]
    fn quant_targets_subset_of_expected_and_count() {
        let cfg = Krea2Config::turbo();
        let expected: BTreeSet<String> = expected_transformer_keys(&cfg).into_iter().collect();
        let targets = transformer_quant_targets(&cfg);
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
        // 28 single-stream blocks × 8 + 4 text-fusion blocks × 8 = 256 (== the on-disk BF16 count).
        assert_eq!(targets.len(), 256, "DiT Linear count");
    }

    #[test]
    fn text_encoder_quant_targets_count_and_prefix() {
        let t = text_encoder_quant_targets(36);
        // 36 layers × (4 attn proj + 3 mlp proj); embed_tokens + norms + visual.* stay dense.
        assert_eq!(t.len(), 36 * 7);
        assert!(t.iter().all(|k| k.starts_with("language_model.layers.")));
    }

    #[test]
    fn quantize_targets_packs_only_targets() {
        let mut map: HashMap<String, Array> = HashMap::new();
        // A target Linear (in-dim a multiple of the group size), its dense bias, and a non-target norm.
        map.insert(
            "transformer_blocks.0.attn.to_q.weight".into(),
            Array::ones::<f32>(&[128, 64]).unwrap(),
        );
        map.insert(
            "transformer_blocks.0.norm1.weight".into(),
            Array::ones::<f32>(&[64]).unwrap(),
        );
        let targets: BTreeSet<String> = ["transformer_blocks.0.attn.to_q.weight".to_string()]
            .into_iter()
            .collect();

        let out = quantize_targets(map, &targets, 4, 64).unwrap();

        // The target became the packed triple (u32 codes + scales + biases)…
        let wq = out
            .get("transformer_blocks.0.attn.to_q.weight")
            .expect("packed weight");
        assert_eq!(wq.dtype(), Dtype::Uint32, "Q4 codes are u32-packed");
        assert!(out.contains_key("transformer_blocks.0.attn.to_q.scales"));
        assert!(out.contains_key("transformer_blocks.0.attn.to_q.biases"));
        // …the non-target norm passed through untouched.
        let norm = out
            .get("transformer_blocks.0.norm1.weight")
            .expect("dense norm");
        assert_eq!(norm.dtype(), Dtype::Float32, "non-target weight unchanged");
        assert!(!out.contains_key("transformer_blocks.0.norm1.scales"));
    }

    /// Real-weight architecture validation against the published Turbo snapshot. Set
    /// `KREA_TURBO_DIR=<snapshot root>` (the dir holding `transformer/`, e.g. the HF cache snapshot).
    /// Loads the 24 GB dense DiT once (host RAM), no GPU.
    #[test]
    #[ignore = "needs real weights: set KREA_TURBO_DIR"]
    fn validate_real_turbo_snapshot() {
        let root = std::env::var("KREA_TURBO_DIR").expect("set KREA_TURBO_DIR");
        let cfg = Krea2Config::from_snapshot(&root).unwrap();
        assert_eq!(cfg, Krea2Config::turbo());
        let w = Weights::from_dir(format!("{root}/transformer")).unwrap();
        validate_transformer(&w, &cfg).unwrap();
        assert_eq!(w.len(), 430);
    }

    /// Real-weight converter proof: assemble a **Q8** turnkey transformer from the dense snapshot, then
    /// reload it through the packed-aware [`validate_transformer`] and confirm the 256 target Linears
    /// emitted the packed triple (`.weight` u32 codes + `.scales` + `.biases`) while the dense F32
    /// tensors passed through. Set `KREA_TURBO_DIR=<snapshot root>`. Heavy: loads the 24 GB dense DiT
    /// and writes a packed copy to a temp dir.
    #[test]
    #[ignore = "needs real weights: set KREA_TURBO_DIR"]
    fn assemble_q8_transformer_reloads_packed() {
        let root =
            std::path::PathBuf::from(std::env::var("KREA_TURBO_DIR").expect("set KREA_TURBO_DIR"));
        let cfg = Krea2Config::from_snapshot(&root).unwrap();
        let dst = std::env::temp_dir().join("krea_q8_convert_test");
        let _ = std::fs::remove_dir_all(&dst);
        quantize_transformer(&root, &dst, 8).unwrap();

        let w = Weights::from_dir(dst.join("transformer")).unwrap();
        // The packed-aware coverage diff still passes (it drops `.scales`/`.biases`).
        validate_transformer(&w, &cfg).unwrap();
        // A representative target is now packed; a representative non-target stays dense.
        assert_eq!(
            w.require("transformer_blocks.0.attn.to_q.weight")
                .unwrap()
                .dtype(),
            Dtype::Uint32,
            "Q8 target packed to u32 codes"
        );
        assert!(w.get("transformer_blocks.0.attn.to_q.scales").is_some());
        assert!(w.get("transformer_blocks.0.attn.to_q.biases").is_some());
        assert!(
            w.get("transformer_blocks.0.norm1.scales").is_none(),
            "norm stays dense"
        );
        // Every one of the 256 quant targets emitted a `.scales` sibling.
        let packed = w.keys().filter(|k| k.ends_with(".scales")).count();
        assert_eq!(packed, transformer_quant_targets(&cfg).len());
        let _ = std::fs::remove_dir_all(&dst);
    }
}
