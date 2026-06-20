//! Native (Rust/MLX) Wan2.2 weight converter (sc-3224). Replaces the Python `mlx_video.convert_wan`.
//!
//! Wan native checkpoints ship the transformer as safetensors but the T5 encoder and VAE as torch
//! `.pth` (zip-of-pickle) — read via [`crate::pth`]. This module ports the reference sanitizers that
//! map the native key layout onto the MLX model layout the Wan loaders consume.
//!
//! **sc-3237: the Wan2.2 VAE path.** [`convert_vae22`] reads `Wan2.2_VAE.pth`, applies
//! [`sanitize_wan22_vae`] (the reference `sanitize_wan22_vae_weights`), and writes
//! `vae.safetensors` in f32 (official Wan runs VAE decode in float32).
//!
//! **sc-3238: the TI2V-5B single-model converter.** [`convert_ti2v_5b`] assembles a full MLX dir —
//! the transformer (native safetensors shards → [`sanitize_wan_transformer`], bf16), the T5
//! (`.pth` → [`sanitize_wan_t5`], bf16), the VAE (f32), and `config.json`.
//!
//! **sc-3239: the A14B dual-expert converters.** [`convert_i2v_14b`] (in_dim 36) and
//! [`convert_t2v_14b`] (in_dim 16) share [`convert_dual_a14b`] — both `low_noise_model` +
//! `high_noise_model` experts (optionally Q4/Q8 via [`quantize_wan_transformer`]), the z16 Wan2.1 VAE
//! ([`sanitize_wan_vae_weights`]), the T5, and the respective `config.json` — differing only in that
//! config. Byte-parity validated end-to-end against a Python `convert_wan` reference on the real
//! 126 GB native I2V-A14B checkpoint (both 1095-tensor experts + T5 + z16 VAE byte-identical;
//! `tests/convert_i2v_14b_parity.rs`). T2V-A14B reuses the same byte-proven path; the SceneWorks
//! manifest ships it a turnkey pre-converted repo, so its native conversion is the optional
//! supply-chain-independent alternative.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::{WanModelConfig, WanQuant};
use mlx_rs::ops::quantize;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

/// Channels-last transpose of a PyTorch conv weight: Conv3d `[O,I,D,H,W]→[O,D,H,W,I]`, Conv2d
/// `[O,I,H,W]→[O,H,W,I]`. Other ranks pass through.
fn conv_channels_last(v: &Array) -> Result<Array> {
    match v.ndim() {
        5 => Ok(v.transpose_axes(&[0, 2, 3, 4, 1])?),
        4 => Ok(v.transpose_axes(&[0, 2, 3, 1])?),
        _ => Ok(v.clone()),
    }
}

/// Drop every size-1 axis (`np.squeeze`) — for the RMS_norm `gamma` tensors `(dim,1,1,1)`/`(dim,1,1)`
/// → `(dim,)`.
fn squeeze_all(v: &Array) -> Result<Array> {
    let new_shape: Vec<i32> = v.shape().iter().copied().filter(|&d| d != 1).collect();
    Ok(v.reshape(&new_shape)?)
}

/// Port of `sanitize_wan22_vae_weights` (mlx_video/models/wan/vae22.py): map the native Wan2.2 VAE
/// key layout (PyTorch `nn.Sequential` indices, channels-first convs, 4-D RMS gammas) onto the MLX
/// `WanVae22` layout. With `include_encoder=false` the encoder + `conv1.*` are dropped (decode-only);
/// TI2V/I2V keep them. Conv weights → channels-last; `gamma` → squeezed.
pub fn sanitize_wan22_vae(
    raw: &HashMap<String, Array>,
    include_encoder: bool,
) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::new();
    for (k, src) in raw {
        if !include_encoder && (k.starts_with("encoder.") || k.starts_with("conv1.")) {
            continue;
        }

        // Sequential index → named layer: residual.{0,2,3,6} and head.{0,2}.
        let mut new = k.clone();
        for idx in ["0", "2", "3", "6"] {
            new = new.replace(
                &format!(".residual.{idx}."),
                &format!(".residual.layer_{idx}."),
            );
        }
        for idx in ["0", "2"] {
            new = new.replace(&format!(".head.{idx}."), &format!(".head.layer_{idx}."));
        }
        // Resample Conv2d + AttentionBlock Conv2d renames (first match wins, mirroring the if/elif).
        if new.contains(".resample.1.weight") {
            new = new.replace(".resample.1.weight", ".resample_weight");
        } else if new.contains(".resample.1.bias") {
            new = new.replace(".resample.1.bias", ".resample_bias");
        }
        if new.contains(".to_qkv.weight") {
            new = new.replace(".to_qkv.weight", ".to_qkv_weight");
        } else if new.contains(".to_qkv.bias") {
            new = new.replace(".to_qkv.bias", ".to_qkv_bias");
        } else if new.contains(".proj.weight") && !new.contains("time_projection") {
            new = new.replace(".proj.weight", ".proj_weight");
        } else if new.contains(".proj.bias") && !new.contains("time_projection") {
            new = new.replace(".proj.bias", ".proj_bias");
        }

        // Conv-weight channels-last (keys ending `.weight` OR the renamed `_weight`). Gate on rank-4
        // so the predicate can only ever transpose an actual conv weight: today only conv weights
        // match and `conv_channels_last` no-ops on rank<4, but a future 2-D `_weight` key would
        // otherwise be silently transposed (F-045). The gate is byte-identical for current weights.
        let mut value = if (new.ends_with(".weight") || new.ends_with("_weight")) && src.ndim() >= 4
        {
            conv_channels_last(src)?
        } else {
            src.clone()
        };
        // RMS_norm gamma: squeeze trailing singleton dims.
        if new.contains("gamma") {
            value = squeeze_all(&value)?;
        }
        out.insert(new, value);
    }
    Ok(out)
}

/// Convert a Wan2.2 `Wan2.2_VAE.pth` into `out_file` (`vae.safetensors`), f32. `include_encoder` is
/// `true` for TI2V/I2V (encode path needed), `false` for decode-only T2V.
pub fn convert_vae22(
    vae_pth: impl AsRef<Path>,
    out_file: impl AsRef<Path>,
    include_encoder: bool,
) -> Result<()> {
    let vae_pth = vae_pth.as_ref();
    if !vae_pth.is_file() {
        return Err(Error::Msg(format!(
            "Wan VAE .pth not found: {}",
            vae_pth.display()
        )));
    }
    // Load the native .pth as f32 (mirrors torch.load(...).float()), then sanitize.
    let raw = crate::pth::load_pth_f32(vae_pth)?;
    let sanitized = sanitize_wan22_vae(&raw, include_encoder)?;

    let arrays: Vec<&Array> = sanitized.values().collect();
    eval(arrays)?;
    if let Some(parent) = out_file.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    Array::save_safetensors(
        sanitized.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        out_file.as_ref(),
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// sc-3238: Wan2.2 TI2V-5B full converter (transformer + T5 + config + orchestration)
// ---------------------------------------------------------------------------

/// Collect a [`Weights`] into a plain key→Array map (lazy clones).
fn weights_to_map(w: &Weights) -> HashMap<String, Array> {
    w.keys()
        .map(|k| {
            (
                k.to_string(),
                w.require(k).expect("key from keys()").clone(),
            )
        })
        .collect()
}

/// Cast every tensor in `map` to `dtype` in place.
fn cast_map(map: &mut HashMap<String, Array>, dtype: Dtype) -> Result<()> {
    for v in map.values_mut() {
        if v.dtype() != dtype {
            *v = v.as_dtype(dtype)?;
        }
    }
    Ok(())
}

/// Materialize + write a key→Array map to `path`.
fn save_map(path: PathBuf, map: &HashMap<String, Array>) -> Result<()> {
    let arrays: Vec<&Array> = map.values().collect();
    eval(arrays)?;
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

fn write_json(path: PathBuf, v: &serde_json::Value) -> Result<()> {
    let text = serde_json::to_string_pretty(v)
        .map_err(|e| Error::Msg(format!("serialize {}: {e}", path.display())))?;
    std::fs::write(&path, text)?;
    Ok(())
}

/// Port of `sanitize_wan_transformer_weights`: native Wan transformer keys → MLX `WanTransformer`
/// keys. `patch_embedding.weight` `[dim,in,1,2,2]` flattens to `[dim, in·4]` → `patch_embedding_proj`;
/// the `text/time_embedding` Sequentials (`.0`/`.2`) → `_0`/`_1`; `time_projection.1` → bare;
/// `ffn.0`/`ffn.2` → `ffn.fc1`/`ffn.fc2`; the `freqs` buffer is dropped. Everything else (attn, norms,
/// modulation, head) passes through unchanged.
pub fn sanitize_wan_transformer(raw: &HashMap<String, Array>) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::new();
    for (key, value) in raw {
        if key == "patch_embedding.weight" {
            let s = value.shape();
            let cols: i32 = s[1..].iter().product();
            out.insert(
                "patch_embedding_proj.weight".into(),
                value.reshape(&[s[0], cols])?,
            );
            continue;
        }
        if key == "patch_embedding.bias" {
            out.insert("patch_embedding_proj.bias".into(), value.clone());
            continue;
        }
        let renamed_seq = [
            ("text_embedding.0.", "text_embedding_0."),
            ("text_embedding.2.", "text_embedding_1."),
            ("time_embedding.0.", "time_embedding_0."),
            ("time_embedding.2.", "time_embedding_1."),
            ("time_projection.1.", "time_projection."),
        ]
        .iter()
        .find_map(|(p, r)| key.strip_prefix(p).map(|rest| format!("{r}{rest}")));
        if let Some(new) = renamed_seq {
            out.insert(new, value.clone());
            continue;
        }
        if key == "freqs" {
            continue;
        }
        let new = key
            .replace(".ffn.0.", ".ffn.fc1.")
            .replace(".ffn.2.", ".ffn.fc2.");
        out.insert(new, value.clone());
    }
    Ok(out)
}

/// Port of `sanitize_wan_t5_weights`: the sole rename `.ffn.gate.0.` → `.ffn.gate_proj.` (the gate
/// Linear); every other UMT5 key passes through.
pub fn sanitize_wan_t5(raw: &HashMap<String, Array>) -> HashMap<String, Array> {
    raw.iter()
        .map(|(k, v)| (k.replace(".ffn.gate.0.", ".ffn.gate_proj."), v.clone()))
        .collect()
}

/// The `wan22_ti2v_5b` preset serialized to its config.json (F-027: built from the single preset +
/// `SAMPLE_NEG_PROMPT` in `config.rs`, not a hand-inlined copy). Guarded by the round-trip test below.
fn wan22_ti2v_5b_config() -> serde_json::Value {
    WanModelConfig::wan22_ti2v_5b().to_json()
}

/// Convert a native Wan2.2 **TI2V-5B** checkpoint dir into an MLX model dir at `out_dir`: the
/// transformer (native `diffusion_pytorch_model-*.safetensors` shards) → `model.safetensors` (bf16),
/// the T5 (`models_t5_umt5-xxl-enc-bf16.pth`) → `t5_encoder.safetensors` (bf16), the VAE
/// (`Wan2.2_VAE.pth`) → `vae.safetensors` (f32), plus `config.json`. The UMT5 `tokenizer.json` (at
/// `google/umt5-xxl/tokenizer.json` in the native repo) is copied by the install flow, not emitted
/// here — matching the reference `convert_wan`.
pub fn convert_ti2v_5b(
    checkpoint_dir: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
) -> Result<PathBuf> {
    let checkpoint_dir = checkpoint_dir.as_ref();
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    // 1. Transformer — native single-model safetensors (the 3 shards merge in `from_dir`).
    let w = Weights::from_dir(checkpoint_dir)?;
    let map = weights_to_map(&w);
    let mut transformer = sanitize_wan_transformer(&map)?;
    cast_map(&mut transformer, Dtype::Bfloat16)?;
    save_map(out_dir.join("model.safetensors"), &transformer)?;
    drop((w, map, transformer));

    // 2. Config.
    write_json(out_dir.join("config.json"), &wan22_ti2v_5b_config())?;

    // 3. T5 encoder — native `.pth` (pickle) → f32 → sanitize → bf16.
    let t5_pth = checkpoint_dir.join("models_t5_umt5-xxl-enc-bf16.pth");
    let raw_t5 = crate::pth::load_pth_f32(&t5_pth)?;
    let mut t5 = sanitize_wan_t5(&raw_t5);
    cast_map(&mut t5, Dtype::Bfloat16)?;
    save_map(out_dir.join("t5_encoder.safetensors"), &t5)?;
    drop((raw_t5, t5));

    // 4. VAE — TI2V keeps the encoder; saved f32.
    convert_vae22(
        checkpoint_dir.join("Wan2.2_VAE.pth"),
        out_dir.join("vae.safetensors"),
        true,
    )?;

    Ok(out_dir.to_path_buf())
}

// ---------------------------------------------------------------------------
// sc-3239: Wan2.2 I2V-A14B dual-expert converter (in_dim 36, z16 VAE, optional Q4/Q8)
// ---------------------------------------------------------------------------

/// The reference `_quantize_predicate`: a Wan transformer Linear is quantized iff its weight key
/// (minus `.weight`) ends with one of these — attention q/k/v/o (self + cross) and the FFN fc1/fc2.
/// Norms / modulation / embeddings / head stay dense.
const WAN_QUANT_SUFFIXES: &[&str] = &[
    ".self_attn.q",
    ".self_attn.k",
    ".self_attn.v",
    ".self_attn.o",
    ".cross_attn.q",
    ".cross_attn.k",
    ".cross_attn.v",
    ".cross_attn.o",
    ".ffn.fc1",
    ".ffn.fc2",
];

/// Port of `sanitize_wan_vae_weights` (the Wan2.1 z16 VAE — `convert_wan.py`): channels-last conv
/// transposes only (Conv3d/Conv2d weights gated on `"weight" in key`), **no** key renames. Distinct
/// from the bespoke z48 [`sanitize_wan22_vae`].
pub fn sanitize_wan_vae_weights(raw: &HashMap<String, Array>) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        let value = if k.contains("weight") {
            conv_channels_last(v)?
        } else {
            v.clone()
        };
        out.insert(k.clone(), value);
    }
    Ok(out)
}

/// Selectively Q4/Q8-quantize a (sanitized) Wan transformer expert in place: each
/// [`WAN_QUANT_SUFFIXES`]-matched Linear `{base}.weight` (bf16) becomes the packed triple
/// `{base}.weight` (u32), `{base}.scales`, `{base}.biases` via MLX `quantize` (byte-identical to
/// `nn.quantize`); the bias and all other tensors pass through.
pub fn quantize_wan_transformer(
    map: HashMap<String, Array>,
    bits: i32,
    group_size: i32,
) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::with_capacity(map.len());
    for (k, v) in map {
        let base = k.strip_suffix(".weight");
        let is_q = base.is_some_and(|b| WAN_QUANT_SUFFIXES.iter().any(|s| b.ends_with(s)));
        if let (true, Some(base)) = (is_q, base) {
            let (wq, scales, biases) = quantize(&v, group_size, bits)?;
            out.insert(format!("{base}.weight"), wq);
            out.insert(format!("{base}.scales"), scales);
            out.insert(format!("{base}.biases"), biases);
        } else {
            out.insert(k, v);
        }
    }
    Ok(out)
}

/// The `WanModelConfig.wan22_i2v_14b().to_dict()` config.json (round-trips through
/// `WanModelConfig::from_config_json`; the dual guide scale is a 2-element array).
fn wan22_i2v_14b_config(quantize: Option<(i32, i32)>) -> serde_json::Value {
    let mut cfg = WanModelConfig::wan22_i2v_14b();
    if let Some((bits, group_size)) = quantize {
        cfg.quantization = Some(WanQuant { bits, group_size });
    }
    cfg.to_json()
}

/// The `WanModelConfig.wan22_t2v_14b().to_dict()` config.json (the dual-expert **T2V** preset:
/// in_dim 16, boundary 0.875, sample_shift 12.0, dual guide scale `[3.0, 4.0]`, no resolution cap).
fn wan22_t2v_14b_config(quantize: Option<(i32, i32)>) -> serde_json::Value {
    let mut cfg = WanModelConfig::wan22_t2v_14b();
    if let Some((bits, group_size)) = quantize {
        cfg.quantization = Some(WanQuant { bits, group_size });
    }
    cfg.to_json()
}

/// Convert one native transformer expert dir (`low_noise_model` / `high_noise_model`) → a sanitized,
/// bf16 (optionally quantized) component file.
fn convert_expert(
    expert_dir: &Path,
    out_file: PathBuf,
    quantize: Option<(i32, i32)>,
) -> Result<()> {
    let w = Weights::from_dir(expert_dir)?;
    let map = weights_to_map(&w);
    let mut t = sanitize_wan_transformer(&map)?;
    cast_map(&mut t, Dtype::Bfloat16)?;
    let t = match quantize {
        Some((bits, group)) => quantize_wan_transformer(t, bits, group)?,
        None => t,
    };
    save_map(out_file, &t)?;
    Ok(())
}

/// Shared dual-expert A14B conversion (I2V + T2V differ only in the emitted `config`): both MoE
/// experts (`low_noise_model` / `high_noise_model` → `*.safetensors`, optionally Q4/Q8), the z16
/// Wan2.1 VAE (`Wan2.1_VAE.pth`, falling back to `Wan2.2_VAE.pth`), the T5, and `config.json`.
fn convert_dual_a14b(
    checkpoint_dir: &Path,
    out_dir: &Path,
    config: serde_json::Value,
    quantize: Option<(i32, i32)>,
) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir)?;

    // 1. Dual experts.
    for (sub, out) in [
        ("low_noise_model", "low_noise_model.safetensors"),
        ("high_noise_model", "high_noise_model.safetensors"),
    ] {
        let expert_dir = checkpoint_dir.join(sub);
        if !expert_dir.is_dir() {
            return Err(Error::Msg(format!(
                "missing expert dir: {}",
                expert_dir.display()
            )));
        }
        convert_expert(&expert_dir, out_dir.join(out), quantize)?;
    }

    // 2. Config.
    write_json(out_dir.join("config.json"), &config)?;

    // 3. T5 encoder.
    let t5_pth = checkpoint_dir.join("models_t5_umt5-xxl-enc-bf16.pth");
    let raw_t5 = crate::pth::load_pth_f32(&t5_pth)?;
    let mut t5 = sanitize_wan_t5(&raw_t5);
    cast_map(&mut t5, Dtype::Bfloat16)?;
    save_map(out_dir.join("t5_encoder.safetensors"), &t5)?;
    drop((raw_t5, t5));

    // 4. VAE — prefer the z16 Wan2.1 VAE; fall back to the z48 Wan2.2 VAE (encoder kept).
    let vae21 = checkpoint_dir.join("Wan2.1_VAE.pth");
    let vae22 = checkpoint_dir.join("Wan2.2_VAE.pth");
    if vae21.is_file() {
        let raw = crate::pth::load_pth_f32(&vae21)?;
        let sanitized = sanitize_wan_vae_weights(&raw)?;
        save_map(out_dir.join("vae.safetensors"), &sanitized)?;
    } else if vae22.is_file() {
        convert_vae22(&vae22, out_dir.join("vae.safetensors"), true)?;
    } else {
        return Err(Error::Msg(format!(
            "no VAE (.pth) found in {} — provide Wan2.1_VAE.pth or Wan2.2_VAE.pth",
            checkpoint_dir.display()
        )));
    }

    Ok(out_dir.to_path_buf())
}

/// Convert a native Wan2.2 **I2V-A14B** checkpoint dir into an MLX model dir (in_dim 36 image-concat
/// conditioning, optional Q4/Q8). Byte-parity validated end-to-end against a Python `convert_wan`
/// reference on the real 126 GB native checkpoint (both 1095-tensor experts + T5 + z16 VAE identical).
pub fn convert_i2v_14b(
    checkpoint_dir: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    quantize: Option<(i32, i32)>,
) -> Result<PathBuf> {
    convert_dual_a14b(
        checkpoint_dir.as_ref(),
        out_dir.as_ref(),
        wan22_i2v_14b_config(quantize),
        quantize,
    )
}

/// Convert a native Wan2.2 **T2V-A14B** checkpoint dir into an MLX model dir (in_dim 16, text-only).
/// The same byte-validated dual-expert path as [`convert_i2v_14b`], differing only in the emitted
/// `config.json` (the `wan22_t2v_14b` preset). The SceneWorks manifest ships a turnkey pre-converted
/// MLX repo for this model, so conversion is optional — this is the native path that avoids that
/// third-party dependency and fully covers the reference `convert_wan` capability surface.
pub fn convert_t2v_14b(
    checkpoint_dir: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    quantize: Option<(i32, i32)>,
) -> Result<PathBuf> {
    convert_dual_a14b(
        checkpoint_dir.as_ref(),
        out_dir.as_ref(),
        wan22_t2v_14b_config(quantize),
        quantize,
    )
}

/// Assemble a `wan_vace` snapshot dir (sc-3467) from the diffusers VACE transformer + a converted
/// base-Wan snapshot's shared native components — **packaging, not conversion**.
///
/// Wan-VACE is Wan2.1-based and shares its UMT5 text encoder + z16 Wan VAE + tokenizer with the base
/// Wan2.2 14B (the *same* components), so the only VACE-specific weights are the **transformer**,
/// which [`crate::WanVaceTransformer`] reads directly in diffusers layout (no conversion). This
/// combines:
///   - `vace_transformer_dir` — the diffusers repo's `transformer/` (its `config.json` + the
///     `diffusion_pytorch_model-*.safetensors` shards) → `<out_dir>/transformer/`, and
///   - `base_wan_snapshot/{t5_encoder.safetensors, vae.safetensors, tokenizer.json}` (a
///     [`convert_i2v_14b`]/[`convert_t2v_14b`] output) → `<out_dir>/`,
///
/// producing the dir layout `mlx_gen::load("wan_vace", WeightsSource::Dir(..))` expects
/// (`WanVaceConfig::from_model_dir` reads `transformer/config.json`; `model_vace` reads the three
/// shared files from the dir root).
///
/// `link == true` symlinks each component (fast, zero-copy — the engine's [`Weights::from_dir`]
/// resolves symlinks); `false` copies (a portable, self-contained snapshot for packaging). The 11 GB
/// T5 dominates, so `link` is the right default for local dev / the gated e2e. Existing entries at the
/// targets are replaced, so assembly is idempotent.
pub fn assemble_wan_vace_snapshot(
    out_dir: impl AsRef<Path>,
    vace_transformer_dir: impl AsRef<Path>,
    base_wan_snapshot: impl AsRef<Path>,
    link: bool,
) -> Result<PathBuf> {
    let out_dir = out_dir.as_ref();
    let tf_src = vace_transformer_dir.as_ref();
    let base = base_wan_snapshot.as_ref();

    if !tf_src.join("config.json").is_file() {
        return Err(Error::Msg(format!(
            "assemble_wan_vace_snapshot: no config.json under {} (expected the diffusers VACE \
             `transformer/` dir)",
            tf_src.display()
        )));
    }
    let has_shard = std::fs::read_dir(tf_src)?
        .filter_map(|e| e.ok())
        .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("safetensors"));
    if !has_shard {
        return Err(Error::Msg(format!(
            "assemble_wan_vace_snapshot: no .safetensors shards under {}",
            tf_src.display()
        )));
    }
    const SHARED: [&str; 3] = [
        "t5_encoder.safetensors",
        "vae.safetensors",
        "tokenizer.json",
    ];
    for name in SHARED {
        if !base.join(name).is_file() {
            return Err(Error::Msg(format!(
                "assemble_wan_vace_snapshot: base snapshot {} is missing {name} (point at a converted \
                 base-Wan dir, e.g. a convert_t2v_14b/convert_i2v_14b output)",
                base.display()
            )));
        }
    }

    std::fs::create_dir_all(out_dir)?;
    place_component(&out_dir.join("transformer"), tf_src, link)?;
    for name in SHARED {
        place_component(&out_dir.join(name), &base.join(name), link)?;
    }
    Ok(out_dir.to_path_buf())
}

/// Assemble a `wan2_2_vace_fun_14b` snapshot dir (sc-6604) from the diffusers VACE-Fun **dual-expert**
/// transformers + a converted base-Wan snapshot's shared native components — the dual-expert sibling of
/// [`assemble_wan_vace_snapshot`], still **packaging, not conversion**.
///
/// VACE-Fun (Wan2.2-A14B base) ships TWO diffusers `WanVACETransformer3DModel`s — `transformer/`
/// (high-noise) + `transformer_2/` (low-noise) — both read directly in diffusers layout by
/// [`crate::WanVaceTransformer`] (no conversion), and shares the SAME UMT5 text encoder + z16 Wan VAE +
/// tokenizer as the base Wan 14B (VACE-Fun is z16-VAE like Wan2.1 VACE). This combines:
///   - `high_transformer_dir` (the diffusers repo's `transformer/`) → `<out_dir>/transformer/`,
///   - `low_transformer_dir` (the diffusers repo's `transformer_2/`) → `<out_dir>/transformer_2/`, and
///   - `base_wan_snapshot/{t5_encoder.safetensors, vae.safetensors, tokenizer.json}` → `<out_dir>/`,
///
/// producing the dir `mlx_gen::load("wan2_2_vace_fun_14b", WeightsSource::Dir(..))` expects
/// ([`crate::config::WanVaceConfig::vace_fun_from_model_dir`] reads `transformer/config.json` +
/// `model_index.json`; [`crate::model_vace`]'s `WanVaceFun` loads the two transformer dirs + the three
/// shared files from the root). `link`/replace semantics match [`assemble_wan_vace_snapshot`].
pub fn assemble_wan_vace_fun_snapshot(
    out_dir: impl AsRef<Path>,
    high_transformer_dir: impl AsRef<Path>,
    low_transformer_dir: impl AsRef<Path>,
    base_wan_snapshot: impl AsRef<Path>,
    link: bool,
) -> Result<PathBuf> {
    let out_dir = out_dir.as_ref();
    let base = base_wan_snapshot.as_ref();
    let experts = [
        ("transformer", high_transformer_dir.as_ref()),
        ("transformer_2", low_transformer_dir.as_ref()),
    ];
    for (label, src) in experts {
        if !src.join("config.json").is_file() {
            return Err(Error::Msg(format!(
                "assemble_wan_vace_fun_snapshot: no config.json under {} (expected the diffusers \
                 VACE-Fun `{label}/` expert dir)",
                src.display()
            )));
        }
        let has_shard = std::fs::read_dir(src)?
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("safetensors"));
        if !has_shard {
            return Err(Error::Msg(format!(
                "assemble_wan_vace_fun_snapshot: no .safetensors shards under {}",
                src.display()
            )));
        }
    }
    const SHARED: [&str; 3] = [
        "t5_encoder.safetensors",
        "vae.safetensors",
        "tokenizer.json",
    ];
    for name in SHARED {
        if !base.join(name).is_file() {
            return Err(Error::Msg(format!(
                "assemble_wan_vace_fun_snapshot: base snapshot {} is missing {name} (point at a \
                 converted base-Wan dir, e.g. a convert_t2v_14b/convert_i2v_14b output)",
                base.display()
            )));
        }
    }

    std::fs::create_dir_all(out_dir)?;
    for (label, src) in experts {
        place_component(&out_dir.join(label), src, link)?;
    }
    for name in SHARED {
        place_component(&out_dir.join(name), &base.join(name), link)?;
    }
    Ok(out_dir.to_path_buf())
}

/// Link-or-copy `src` (a file or dir) to `dst`, replacing any existing entry (idempotent assembly).
fn place_component(dst: &Path, src: &Path, link: bool) -> Result<()> {
    if dst.is_symlink() || dst.exists() {
        if dst.is_dir() && !dst.is_symlink() {
            std::fs::remove_dir_all(dst)?;
        } else {
            std::fs::remove_file(dst)?;
        }
    }
    let src_abs = src.canonicalize()?;
    if link {
        std::os::unix::fs::symlink(&src_abs, dst)?;
    } else if src_abs.is_dir() {
        copy_dir_recursive(&src_abs, dst)?;
    } else {
        std::fs::copy(&src_abs, dst)?;
    }
    Ok(())
}

/// Recursively copy a directory, resolving symlinked entries (e.g. the HF-cache blob links) to real
/// files so the result is self-contained.
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path().canonicalize()?;
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// The Bernini renderer inference knobs preserved beside the converted snapshot (consumed by the
/// sc-4706 provider; the loadable `config.json` carries only the Wan2.2 architecture).
const BERNINI_RENDERER_SIDECAR: &str = "bernini_renderer.json";

/// Diffusers `WanTransformer3DModel` keys → mlx-gen **internal** `WanTransformer` keys (sc-4705).
///
/// This is the inverse of [`crate::adapters`]'s internal→diffusers `normalize`, composed with the
/// `patch_embedding` conv→Linear reshape that [`sanitize_wan_transformer`] already applies. It lets a
/// diffusers-layout Wan2.2 transformer (here: a Bernini renderer DiT, shipped diffusers-named in the
/// combined `bernini/` index) load through the validated dual-expert [`crate::pipeline`] /
/// [`WanTransformer::from_weights`] path — the same path the native `convert_t2v_14b` output uses.
///
/// The map is a verified 1:1 bijection (1095 tensors / 42 key patterns) against a native-converted
/// `wan2_2_t2v_a14b` expert:
///   - `attn1.{to_q,to_k,to_v,to_out.0,norm_q,norm_k}` → `self_attn.{q,k,v,o,norm_q,norm_k}`;
///     `attn2.*` → `cross_attn.*`
///   - `ffn.net.0.proj` → `ffn.fc1`, `ffn.net.2` → `ffn.fc2`, `norm2` → `norm3`,
///     per-block `scale_shift_table` → `modulation`
///   - `condition_embedder.{text,time}_embedder.linear_{1,2}` → `{text,time}_embedding_{0,1}`,
///     `condition_embedder.time_proj` → `time_projection`
///   - `patch_embedding.weight` `[dim,16,1,2,2]` → reshape → `patch_embedding_proj.weight` `[dim,64]`;
///     `proj_out` → `head.head`; top-level `scale_shift_table` → `head.modulation`
pub fn sanitize_wan_transformer_diffusers(
    raw: &HashMap<String, Array>,
) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::with_capacity(raw.len());
    for (key, value) in raw {
        // Conv3d patch embed [dim, in, 1, 2, 2] → Linear [dim, in·4] (same as the native path).
        if key == "patch_embedding.weight" {
            let s = value.shape();
            let cols: i32 = s[1..].iter().product();
            out.insert(
                "patch_embedding_proj.weight".into(),
                value.reshape(&[s[0], cols])?,
            );
            continue;
        }
        if key == "patch_embedding.bias" {
            out.insert("patch_embedding_proj.bias".into(), value.clone());
            continue;
        }
        // Final-norm modulation table + output projection (top-level, not per-block).
        if key == "scale_shift_table" {
            out.insert("head.modulation".into(), value.clone());
            continue;
        }
        if let Some(rest) = key.strip_prefix("proj_out.") {
            out.insert(format!("head.head.{rest}"), value.clone());
            continue;
        }
        // Global condition embedder → text/time embeddings + time projection.
        let mut t = key
            .replace(
                "condition_embedder.text_embedder.linear_1",
                "text_embedding_0",
            )
            .replace(
                "condition_embedder.text_embedder.linear_2",
                "text_embedding_1",
            )
            .replace(
                "condition_embedder.time_embedder.linear_1",
                "time_embedding_0",
            )
            .replace(
                "condition_embedder.time_embedder.linear_2",
                "time_embedding_1",
            )
            .replace("condition_embedder.time_proj", "time_projection");
        // Per-block attention / FFN / norms / modulation.
        if t.starts_with("blocks.") {
            t = t
                .replace(".attn1.to_out.0", ".self_attn.o")
                .replace(".attn1.to_q", ".self_attn.q")
                .replace(".attn1.to_k", ".self_attn.k")
                .replace(".attn1.to_v", ".self_attn.v")
                .replace(".attn1.norm_q", ".self_attn.norm_q")
                .replace(".attn1.norm_k", ".self_attn.norm_k")
                .replace(".attn2.to_out.0", ".cross_attn.o")
                .replace(".attn2.to_q", ".cross_attn.q")
                .replace(".attn2.to_k", ".cross_attn.k")
                .replace(".attn2.to_v", ".cross_attn.v")
                .replace(".attn2.norm_q", ".cross_attn.norm_q")
                .replace(".attn2.norm_k", ".cross_attn.norm_k")
                .replace(".ffn.net.0.proj", ".ffn.fc1")
                .replace(".ffn.net.2", ".ffn.fc2")
                .replace(".norm2.", ".norm3.")
                .replace(".scale_shift_table", ".modulation");
        }
        out.insert(t, value.clone());
    }
    Ok(out)
}

/// Pull one renderer expert out of the combined Bernini `bernini/` index: every tensor whose key
/// starts with `prefix` (`diff_dec.transformer.` = high-noise / `diff_dec_low.transformer_2.` = low),
/// the prefix stripped. Reads shard-by-shard and keeps only the matching (lazy, mmap-backed) arrays,
/// so the ~168 GB of non-renderer weights (MLLM / T5 / VAE / connector / vit_decoder) packed in the
/// same shards are never materialized.
fn extract_bernini_expert(bernini_dir: &Path, prefix: &str) -> Result<HashMap<String, Array>> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(bernini_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(Error::Msg(format!(
            "assemble_bernini_renderer_snapshot: no .safetensors shards under {}",
            bernini_dir.display()
        )));
    }
    let mut out = HashMap::new();
    for shard in &shards {
        let w = Weights::from_file(shard)?;
        for k in w.keys() {
            if let Some(rest) = k.strip_prefix(prefix) {
                out.insert(rest.to_string(), w.require(k)?.clone());
            }
        }
    }
    if out.is_empty() {
        return Err(Error::Msg(format!(
            "assemble_bernini_renderer_snapshot: no keys with prefix '{prefix}' under {} \
             (expected a ByteDance/Bernini-Diffusers `bernini/` index)",
            bernini_dir.display()
        )));
    }
    Ok(out)
}

/// The Bernini renderer knobs, read from the package `config.json` where present, else the upstream
/// `BerniniRendererConfig` defaults. The provider (sc-4706) reads these for the source-id rotary,
/// expert-switch boundary, and flow shift.
fn bernini_renderer_knobs(pkg: &Path) -> serde_json::Value {
    use serde_json::json;
    let cfg: serde_json::Value = std::fs::read(pkg.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_else(|| json!({}));
    let f = |k: &str, d: f64| cfg.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d);
    let b = |k: &str, d: bool| cfg.get(k).and_then(serde_json::Value::as_bool).unwrap_or(d);
    let i = |k: &str, d: i64| cfg.get(k).and_then(serde_json::Value::as_i64).unwrap_or(d);
    json!({
        "switch_dit_boundary": f("switch_dit_boundary", 0.875),
        "shift": f("shift", 3.0),
        "use_src_id_rotary_emb": b("use_src_id_rotary_emb", true),
        "interpolate_src_id": b("interpolate_src_id", true),
        "max_trained_src_id": i("max_trained_src_id", 5),
        "max_sequence_length": i("max_sequence_length", 512),
        "flow_shift": f("flow_shift", 5.0),
    })
}

/// Assemble a native MLX **Bernini renderer** snapshot from a `ByteDance/Bernini-Diffusers` package
/// (sc-4705). The renderer is Wan2.2-T2V-A14B verbatim, finetuned; the only Bernini-specific weights
/// are the two dual-expert DiTs, which the full package bundles into one combined `bernini/` index
/// (38 F32 shards) under the `diff_dec.transformer.` (high-noise) and `diff_dec_low.transformer_2.`
/// (low-noise) prefixes. The stock Wan2.2 UMT5 / z16 VAE / tokenizer — which the reference
/// `BerniniRendererModel` itself loads from its `wan22_base` — are reused from a converted base-Wan
/// snapshot (`base_wan_snapshot`, a [`convert_t2v_14b`] output) rather than re-derived.
///
/// Emits the dual-expert layout the existing `wan2_2_t2v_14b` provider loads:
///   - `high_noise_model.safetensors` ← `diff_dec.transformer.*` → internal → bf16 (+ optional Q4/Q8)
///   - `low_noise_model.safetensors`  ← `diff_dec_low.transformer_2.*` → internal → bf16 (+ Q4/Q8)
///   - `t5_encoder.safetensors`, `vae.safetensors`, `tokenizer.json` ← `base_wan_snapshot` (link or copy)
///   - `config.json` (the loadable `wan22_t2v_14b` preset) + `bernini_renderer.json` (Bernini knobs)
///
/// `link == true` symlinks the shared components (zero-copy; the engine resolves symlinks); `false`
/// copies them (a portable, self-contained snapshot). The extracted DiTs are always written fresh
/// (they are F32 in the package and re-saved bf16, halving them). Idempotent: existing targets are
/// replaced.
pub fn assemble_bernini_renderer_snapshot(
    out_dir: impl AsRef<Path>,
    bernini_diffusers_dir: impl AsRef<Path>,
    base_wan_snapshot: impl AsRef<Path>,
    quantize: Option<(i32, i32)>,
    link: bool,
) -> Result<PathBuf> {
    let out_dir = out_dir.as_ref();
    let pkg = bernini_diffusers_dir.as_ref();
    let base = base_wan_snapshot.as_ref();

    let bernini_dir = pkg.join("bernini");
    if !bernini_dir.is_dir() {
        return Err(Error::Msg(format!(
            "assemble_bernini_renderer_snapshot: no `bernini/` dir under {} (point at a \
             ByteDance/Bernini-Diffusers snapshot root)",
            pkg.display()
        )));
    }
    // The stock Wan2.2 components Bernini-R reuses (the reference loads T5/VAE from `wan22_base`).
    const SHARED: [&str; 3] = [
        "t5_encoder.safetensors",
        "vae.safetensors",
        "tokenizer.json",
    ];
    for name in SHARED {
        if !base.join(name).is_file() {
            return Err(Error::Msg(format!(
                "assemble_bernini_renderer_snapshot: base snapshot {} is missing {name} (point at a \
                 convert_t2v_14b output — Bernini-R reuses the stock Wan2.2 T5/VAE/tokenizer)",
                base.display()
            )));
        }
    }

    std::fs::create_dir_all(out_dir)?;

    // 1. The two renderer experts: diffusers → internal → bf16 (+ optional Q4/Q8). Done one at a
    //    time so only one expert's tensors are materialized at the save eval.
    for (prefix, out_name) in [
        ("diff_dec.transformer.", "high_noise_model.safetensors"),
        ("diff_dec_low.transformer_2.", "low_noise_model.safetensors"),
    ] {
        let raw = extract_bernini_expert(&bernini_dir, prefix)?;
        let mut expert = sanitize_wan_transformer_diffusers(&raw)?;
        drop(raw);
        cast_map(&mut expert, Dtype::Bfloat16)?;
        let expert = match quantize {
            Some((bits, group)) => quantize_wan_transformer(expert, bits, group)?,
            None => expert,
        };
        save_map(out_dir.join(out_name), &expert)?;
    }

    // 2. Loadable Wan2.2 config + preserved Bernini renderer knobs.
    write_json(out_dir.join("config.json"), &wan22_t2v_14b_config(quantize))?;
    write_json(
        out_dir.join(BERNINI_RENDERER_SIDECAR),
        &bernini_renderer_knobs(pkg),
    )?;

    // 3. Shared stock-Wan2.2 components (link or copy).
    for name in SHARED {
        place_component(&out_dir.join(name), &base.join(name), link)?;
    }

    Ok(out_dir.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::all_close;

    fn exact_eq(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape() && all_close(a, b, 0.0, 0.0, false).unwrap().item::<bool>()
    }

    fn m(entries: &[(&str, Array)]) -> HashMap<String, Array> {
        entries
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    /// sc-4705: diffusers `WanTransformer3DModel` keys → internal `WanTransformer` keys is a faithful
    /// bijection (one representative tensor per pattern), and the patch-embed conv flattens to a Linear.
    #[test]
    fn bernini_diffusers_key_map() {
        let lin = |o: i32, i: i32| Array::ones::<f32>(&[o, i]).unwrap();
        let vec = |n: i32| Array::ones::<f32>(&[n]).unwrap();
        let raw = m(&[
            (
                "patch_embedding.weight",
                Array::ones::<f32>(&[8, 16, 1, 2, 2]).unwrap(),
            ),
            ("patch_embedding.bias", vec(8)),
            ("scale_shift_table", Array::ones::<f32>(&[1, 2, 8]).unwrap()),
            ("proj_out.weight", lin(64, 8)),
            ("proj_out.bias", vec(64)),
            (
                "condition_embedder.text_embedder.linear_1.weight",
                lin(8, 4096),
            ),
            (
                "condition_embedder.text_embedder.linear_2.weight",
                lin(8, 8),
            ),
            (
                "condition_embedder.time_embedder.linear_1.weight",
                lin(8, 256),
            ),
            (
                "condition_embedder.time_embedder.linear_2.weight",
                lin(8, 8),
            ),
            ("condition_embedder.time_proj.weight", lin(48, 8)),
            ("blocks.0.attn1.to_q.weight", lin(8, 8)),
            ("blocks.0.attn1.to_out.0.weight", lin(8, 8)),
            ("blocks.0.attn1.norm_q.weight", vec(8)),
            ("blocks.0.attn2.to_k.weight", lin(8, 8)),
            ("blocks.0.ffn.net.0.proj.weight", lin(32, 8)),
            ("blocks.0.ffn.net.2.weight", lin(8, 32)),
            ("blocks.0.norm2.weight", vec(8)),
            (
                "blocks.0.scale_shift_table",
                Array::ones::<f32>(&[1, 6, 8]).unwrap(),
            ),
        ]);
        let out = sanitize_wan_transformer_diffusers(&raw).unwrap();
        let mut keys: Vec<&str> = out.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "blocks.0.cross_attn.k.weight",
                "blocks.0.ffn.fc1.weight",
                "blocks.0.ffn.fc2.weight",
                "blocks.0.modulation",
                "blocks.0.norm3.weight",
                "blocks.0.self_attn.norm_q.weight",
                "blocks.0.self_attn.o.weight",
                "blocks.0.self_attn.q.weight",
                "head.head.bias",
                "head.head.weight",
                "head.modulation",
                "patch_embedding_proj.bias",
                "patch_embedding_proj.weight",
                "text_embedding_0.weight",
                "text_embedding_1.weight",
                "time_embedding_0.weight",
                "time_embedding_1.weight",
                "time_projection.weight",
            ]
        );
        // Conv patch embed [8, 16, 1, 2, 2] flattens to the Linear [8, 16·1·2·2 = 64].
        assert_eq!(out["patch_embedding_proj.weight"].shape(), [8, 64]);
    }

    /// Key renames: Sequential index → layer_N, resample/to_qkv/proj conv renames.
    #[test]
    fn vae_key_renames() {
        let ones5 = Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(); // conv3d weight
        let s = sanitize_wan22_vae(
            &m(&[
                ("decoder.middle.0.residual.0.weight", ones5.clone()),
                (
                    "decoder.middle.0.residual.6.bias",
                    Array::ones::<f32>(&[2]).unwrap(),
                ),
                (
                    "decoder.head.0.gamma",
                    Array::ones::<f32>(&[4, 1, 1, 1]).unwrap(),
                ),
                ("decoder.head.2.weight", ones5.clone()),
                (
                    "decoder.upsamples.0.upsamples.0.resample.1.weight",
                    Array::ones::<f32>(&[2, 2, 3, 3]).unwrap(),
                ),
                (
                    "decoder.middle.0.to_qkv.weight",
                    Array::ones::<f32>(&[6, 2, 1, 1]).unwrap(),
                ),
                (
                    "decoder.middle.0.proj.bias",
                    Array::ones::<f32>(&[2]).unwrap(),
                ),
            ]),
            true,
        )
        .unwrap();
        let mut keys: Vec<&str> = s.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "decoder.head.layer_0.gamma",
                "decoder.head.layer_2.weight",
                "decoder.middle.0.proj_bias",
                "decoder.middle.0.residual.layer_0.weight",
                "decoder.middle.0.residual.layer_6.bias",
                "decoder.middle.0.to_qkv_weight",
                "decoder.upsamples.0.upsamples.0.resample_weight",
            ]
        );
    }

    /// `include_encoder=false` drops `encoder.*` and `conv1.*`; `true` keeps them.
    #[test]
    fn vae_encoder_gating() {
        let entries = [
            (
                "encoder.conv1.weight",
                Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(),
            ),
            (
                "conv1.weight",
                Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(),
            ),
            (
                "conv2.weight",
                Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(),
            ),
            (
                "decoder.conv1.weight",
                Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(),
            ),
        ];
        let dec_only = sanitize_wan22_vae(&m(&entries), false).unwrap();
        assert!(!dec_only
            .keys()
            .any(|k| k.starts_with("encoder.") || k.starts_with("conv1.")));
        assert!(dec_only.contains_key("conv2.weight"));
        assert!(dec_only.contains_key("decoder.conv1.weight")); // not a top-level conv1
        let with_enc = sanitize_wan22_vae(&m(&entries), true).unwrap();
        assert!(with_enc.contains_key("conv1.weight"));
        assert!(with_enc.contains_key("encoder.conv1.weight"));
    }

    /// Conv3d weight → channels-last; gamma squeezed; bias untouched.
    #[test]
    fn vae_transpose_and_squeeze() {
        // Conv3d [O=1,I=2,D=1,H=1,W=2] row-major 0..3 → [O,D,H,W,I]=[1,1,1,2,2] values [0,2,1,3].
        let v = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 2, 1, 1, 2]);
        let s = sanitize_wan22_vae(
            &m(&[
                ("conv2.weight", v),
                (
                    "decoder.middle.0.norm.gamma",
                    Array::ones::<f32>(&[3, 1, 1, 1]).unwrap(),
                ),
            ]),
            true,
        )
        .unwrap();
        assert!(exact_eq(
            &s["conv2.weight"],
            &Array::from_slice(&[0.0f32, 2.0, 1.0, 3.0], &[1, 1, 1, 2, 2])
        ));
        assert_eq!(s["decoder.middle.0.norm.gamma"].shape(), &[3]);
    }

    /// Transformer sanitizer: patch_embedding flatten, Sequential→_0/_1, time_projection.1→bare,
    /// ffn.0/2→fc1/fc2, freqs dropped, attn/modulation pass-through.
    #[test]
    fn transformer_renames() {
        let s = sanitize_wan_transformer(&m(&[
            // [dim=2, in=3, 1, 2, 2] → patch_embedding_proj.weight [2, 12]
            (
                "patch_embedding.weight",
                Array::ones::<f32>(&[2, 3, 1, 2, 2]).unwrap(),
            ),
            ("patch_embedding.bias", Array::ones::<f32>(&[2]).unwrap()),
            (
                "text_embedding.0.weight",
                Array::ones::<f32>(&[2, 4]).unwrap(),
            ),
            ("text_embedding.2.bias", Array::ones::<f32>(&[2]).unwrap()),
            (
                "time_embedding.0.weight",
                Array::ones::<f32>(&[2, 4]).unwrap(),
            ),
            (
                "time_embedding.2.weight",
                Array::ones::<f32>(&[2, 2]).unwrap(),
            ),
            (
                "time_projection.1.weight",
                Array::ones::<f32>(&[12, 2]).unwrap(),
            ),
            (
                "blocks.0.ffn.0.weight",
                Array::ones::<f32>(&[8, 2]).unwrap(),
            ),
            (
                "blocks.0.ffn.2.weight",
                Array::ones::<f32>(&[2, 8]).unwrap(),
            ),
            (
                "blocks.0.self_attn.q.weight",
                Array::ones::<f32>(&[2, 2]).unwrap(),
            ),
            (
                "blocks.0.modulation",
                Array::ones::<f32>(&[1, 6, 2]).unwrap(),
            ),
            ("freqs", Array::ones::<f32>(&[2, 2]).unwrap()),
        ]))
        .unwrap();
        let mut keys: Vec<&str> = s.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "blocks.0.ffn.fc1.weight",
                "blocks.0.ffn.fc2.weight",
                "blocks.0.modulation",
                "blocks.0.self_attn.q.weight",
                "patch_embedding_proj.bias",
                "patch_embedding_proj.weight",
                "text_embedding_0.weight",
                "text_embedding_1.bias",
                "time_embedding_0.weight",
                "time_embedding_1.weight",
                "time_projection.weight",
            ]
        );
        assert_eq!(s["patch_embedding_proj.weight"].shape(), &[2, 12]); // 3·1·2·2 = 12
        assert!(!s.contains_key("freqs"));
    }

    /// T5 sanitizer: only `.ffn.gate.0.` → `.ffn.gate_proj.`; everything else unchanged.
    #[test]
    fn t5_gate_rename() {
        let s = sanitize_wan_t5(&m(&[
            (
                "blocks.0.ffn.gate.0.weight",
                Array::ones::<f32>(&[4, 2]).unwrap(),
            ),
            (
                "blocks.0.ffn.fc1.weight",
                Array::ones::<f32>(&[4, 2]).unwrap(),
            ),
            (
                "blocks.0.attn.q.weight",
                Array::ones::<f32>(&[2, 2]).unwrap(),
            ),
            (
                "token_embedding.weight",
                Array::ones::<f32>(&[5, 2]).unwrap(),
            ),
        ]));
        assert!(s.contains_key("blocks.0.ffn.gate_proj.weight"));
        assert!(!s.keys().any(|k| k.contains("gate.0")));
        assert!(s.contains_key("blocks.0.ffn.fc1.weight"));
        assert!(s.contains_key("blocks.0.attn.q.weight"));
        assert!(s.contains_key("token_embedding.weight"));
    }

    /// z16 VAE sanitizer (`sanitize_wan_vae_weights`): conv transpose only, no key renames.
    #[test]
    fn z16_vae_transpose_only() {
        // Conv2d [O=1,I=2,H=2,W=1] 0..3 → [O,H,W,I]=[1,2,1,2] values [0,2,1,3]; keys unchanged.
        let s = sanitize_wan_vae_weights(&m(&[
            (
                "decoder.conv1.weight",
                Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 2, 2, 1]),
            ),
            (
                "decoder.middle.0.residual.0.bias",
                Array::ones::<f32>(&[3]).unwrap(),
            ),
        ]))
        .unwrap();
        // raw keys preserved (NOT renamed to layer_N like the z48 vae22 sanitizer)
        assert!(s.contains_key("decoder.conv1.weight"));
        assert!(s.contains_key("decoder.middle.0.residual.0.bias"));
        assert!(exact_eq(
            &s["decoder.conv1.weight"],
            &Array::from_slice(&[0.0f32, 2.0, 1.0, 3.0], &[1, 2, 1, 2])
        ));
    }

    /// Wan quant predicate: attn q/k/v/o (self + cross) + ffn fc1/fc2; norms/modulation/head dense.
    #[test]
    fn wan_quant_predicate() {
        let q = |k: &str| {
            k.strip_suffix(".weight")
                .is_some_and(|b| WAN_QUANT_SUFFIXES.iter().any(|s| b.ends_with(s)))
        };
        for k in [
            "blocks.0.self_attn.q.weight",
            "blocks.5.cross_attn.o.weight",
            "blocks.0.ffn.fc1.weight",
            "blocks.0.ffn.fc2.weight",
        ] {
            assert!(q(k), "should quantize: {k}");
        }
        for k in [
            "blocks.0.self_attn.q.bias",
            "blocks.0.self_attn.norm_q.weight",
            "blocks.0.modulation",
            "patch_embedding_proj.weight",
            "head.head.weight",
        ] {
            assert!(!q(k), "should stay dense: {k}");
        }
    }

    /// Quantizing a Wan transformer emits packed weight + scales/biases for matched Linears, keeps
    /// the bias, and leaves norms dense.
    #[test]
    fn quantize_wan_transformer_packs() {
        let bf = |a: Array| a.as_dtype(Dtype::Bfloat16).unwrap();
        let q = quantize_wan_transformer(
            m(&[
                (
                    "blocks.0.self_attn.q.weight",
                    bf(Array::ones::<f32>(&[64, 128]).unwrap()),
                ),
                (
                    "blocks.0.self_attn.q.bias",
                    bf(Array::ones::<f32>(&[64]).unwrap()),
                ),
                (
                    "blocks.0.norm1.weight",
                    bf(Array::ones::<f32>(&[64]).unwrap()),
                ),
            ]),
            4,
            64,
        )
        .unwrap();
        assert!(q.contains_key("blocks.0.self_attn.q.scales"));
        assert!(q.contains_key("blocks.0.self_attn.q.biases"));
        assert!(q.contains_key("blocks.0.self_attn.q.bias")); // bias preserved
        assert_ne!(q["blocks.0.self_attn.q.weight"].dtype(), Dtype::Bfloat16); // packed (u32)
        assert!(q.contains_key("blocks.0.norm1.weight")); // dense
        assert!(!q.contains_key("blocks.0.norm1.scales"));
    }

    /// The TI2V-5B config.json round-trips through the loader's parser to the `wan22_ti2v_5b` preset
    /// — the literal it replaced (F-027) had no guard. Confirms `to_json` emits every key the loader
    /// reads (incl. the scalar `sample_guide_scale` and `SAMPLE_NEG_PROMPT`).
    #[test]
    fn ti2v_5b_config_round_trips() {
        use crate::config::WanModelConfig;
        let cfg = WanModelConfig::from_config_json(&wan22_ti2v_5b_config());
        assert_eq!(cfg, WanModelConfig::wan22_ti2v_5b());
        assert_eq!(
            cfg.sample_neg_prompt,
            crate::config::SAMPLE_NEG_PROMPT,
            "5B config must carry the shared negative prompt"
        );
    }

    /// The I2V-14B config.json round-trips through the loader's parser to the `wan22_i2v_14b` preset
    /// (no golden exists, so this is the validation oracle), with the quant block when requested.
    #[test]
    fn i2v_14b_config_round_trips() {
        use crate::config::WanModelConfig;
        let cfg = WanModelConfig::from_config_json(&wan22_i2v_14b_config(None));
        assert_eq!(cfg, WanModelConfig::wan22_i2v_14b());
        let cfgq = WanModelConfig::from_config_json(&wan22_i2v_14b_config(Some((4, 64))));
        assert!(cfgq.quantization.is_some());
    }

    /// The T2V-14B config.json round-trips through the loader's parser to the `wan22_t2v_14b` preset.
    #[test]
    fn t2v_14b_config_round_trips() {
        use crate::config::WanModelConfig;
        let cfg = WanModelConfig::from_config_json(&wan22_t2v_14b_config(None));
        assert_eq!(cfg, WanModelConfig::wan22_t2v_14b());
    }

    /// `assemble_wan_vace_snapshot` (sc-3467) lays out a load-ready `wan_vace` dir: a `transformer/`
    /// plus the three shared base-Wan files, linked (idempotent) and resolvable. No weights needed —
    /// it is pure file packaging.
    #[test]
    fn assemble_wan_vace_snapshot_links_components() {
        let tmp = std::env::temp_dir().join(format!("wanvace_assemble_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let tf = tmp.join("vace_repo/transformer");
        let base = tmp.join("base_wan");
        let out = tmp.join("wan_vace");
        std::fs::create_dir_all(&tf).unwrap();
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(
            tf.join("config.json"),
            b"{\"_class_name\":\"WanVACETransformer3DModel\"}",
        )
        .unwrap();
        std::fs::write(
            tf.join("diffusion_pytorch_model-00001-of-00001.safetensors"),
            b"shard",
        )
        .unwrap();
        for name in [
            "t5_encoder.safetensors",
            "vae.safetensors",
            "tokenizer.json",
        ] {
            std::fs::write(base.join(name), name.as_bytes()).unwrap();
        }

        // Assemble twice to prove idempotency (the second call must replace cleanly).
        assemble_wan_vace_snapshot(&out, &tf, &base, true).unwrap();
        let out = assemble_wan_vace_snapshot(&out, &tf, &base, true).unwrap();

        assert!(out.join("transformer").is_symlink());
        // The dir layout `WanVaceConfig::from_model_dir` + `model_vace::load` resolve.
        assert!(out.join("transformer/config.json").is_file());
        assert!(out
            .join("transformer/diffusion_pytorch_model-00001-of-00001.safetensors")
            .is_file());
        for name in [
            "t5_encoder.safetensors",
            "vae.safetensors",
            "tokenizer.json",
        ] {
            let p = out.join(name);
            assert!(p.is_symlink(), "{name} should be a symlink");
            assert_eq!(
                std::fs::read(&p).unwrap(),
                name.as_bytes(),
                "{name} resolves"
            );
        }

        // A missing shared component is a clear, actionable error (not a silent partial snapshot).
        std::fs::remove_file(base.join("vae.safetensors")).unwrap();
        let err = assemble_wan_vace_snapshot(out.join("again"), &tf, &base, true).unwrap_err();
        assert!(err.to_string().contains("vae.safetensors"), "got: {err}");

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// `assemble_wan_vace_fun_snapshot` (sc-6604) lays out a load-ready dual-expert `wan2_2_vace_fun_14b`
    /// dir: `transformer/` (high) + `transformer_2/` (low) + the three shared base-Wan files, linked and
    /// idempotent. Pure file packaging, no weights.
    #[test]
    fn assemble_wan_vace_fun_snapshot_links_both_experts() {
        let tmp = std::env::temp_dir().join(format!("wanvacefun_assemble_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let high = tmp.join("vace_fun_repo/transformer");
        let low = tmp.join("vace_fun_repo/transformer_2");
        let base = tmp.join("base_wan");
        let out = tmp.join("wan_vace_fun");
        for tf in [&high, &low] {
            std::fs::create_dir_all(tf).unwrap();
            std::fs::write(
                tf.join("config.json"),
                b"{\"_class_name\":\"WanVACETransformer3DModel\"}",
            )
            .unwrap();
            std::fs::write(
                tf.join("diffusion_pytorch_model-00001-of-00001.safetensors"),
                b"shard",
            )
            .unwrap();
        }
        std::fs::create_dir_all(&base).unwrap();
        for name in [
            "t5_encoder.safetensors",
            "vae.safetensors",
            "tokenizer.json",
        ] {
            std::fs::write(base.join(name), name.as_bytes()).unwrap();
        }

        // Idempotent: assemble twice.
        assemble_wan_vace_fun_snapshot(&out, &high, &low, &base, true).unwrap();
        let out = assemble_wan_vace_fun_snapshot(&out, &high, &low, &base, true).unwrap();

        // Both experts land in the layout `WanVaceFun`/`load_vace_fun_expert_weights` resolve.
        assert!(out.join("transformer").is_symlink());
        assert!(out.join("transformer_2").is_symlink());
        assert!(out.join("transformer/config.json").is_file());
        assert!(out.join("transformer_2/config.json").is_file());
        for name in [
            "t5_encoder.safetensors",
            "vae.safetensors",
            "tokenizer.json",
        ] {
            assert!(out.join(name).is_symlink(), "{name} should be a symlink");
        }

        // A missing low-noise expert is a clear error (no silent single-expert snapshot).
        std::fs::remove_file(low.join("config.json")).unwrap();
        let err = assemble_wan_vace_fun_snapshot(out.join("again"), &high, &low, &base, true)
            .unwrap_err();
        assert!(err.to_string().contains("transformer_2"), "got: {err}");

        std::fs::remove_dir_all(&tmp).ok();
    }
}
