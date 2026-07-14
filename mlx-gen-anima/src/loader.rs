//! Assemble the Anima components from the on-disk `split_files/` layout:
//! `diffusion_models/anima-{variant}-v1.0.safetensors` (DiT + bundled `net.llm_adapter.*`
//! conditioner), `text_encoders/qwen_3_06b_base.safetensors`, `vae/qwen_image_vae.safetensors`.
//!
//! The DiT safetensors bundles BOTH the Cosmos DiT (Cosmos naming, `net.*`) and the
//! `AnimaTextConditioner` (`net.llm_adapter.*`). We load it once and build both from the same
//! `Weights` with their respective key prefixes — the `net.llm_adapter.` split is exactly the Anima
//! convert script's `split_anima_transformer_checkpoint`.

use std::path::{Path, PathBuf};

use mlx_rs::Dtype;

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};

use crate::conditioner::AnimaTextConditioner;
use crate::config::{ConditionerConfig, DitConfig, Qwen3Config, Variant};
use crate::text_encoder::AnimaQwen3;
use crate::tokenizer::AnimaTokenizers;
use crate::transformer::CosmosDiT;
use crate::vae::{load_vae, QwenVae};

/// The conditioner-splitting marker (port of the Anima convert script's
/// `split_anima_transformer_checkpoint`, which splits on `llm_adapter.`). The DiT root prefix varies
/// by file — `net` (base) or `model.diffusion_model` (turbo/aesthetic) — so we detect it rather than
/// hardcode it.
const ADAPTER_MARKER: &str = "llm_adapter.";
/// A key that unambiguously fixes the DiT root prefix (present in every Anima DiT file).
const PREFIX_ANCHOR: &str = ".x_embedder.proj.1.weight";

const TEXT_ENCODER_FILE: &str = "text_encoders/qwen_3_06b_base.safetensors";
const VAE_FILE: &str = "vae/qwen_image_vae.safetensors";

/// Detect the DiT root prefix (`net` or `model.diffusion_model`) from the checkpoint keys.
fn detect_dit_prefix(w: &Weights) -> Result<String> {
    w.keys()
        .find(|k| k.ends_with(PREFIX_ANCHOR))
        .map(|k| k[..k.len() - PREFIX_ANCHOR.len()].to_string())
        .ok_or_else(|| {
            Error::Msg(format!(
                "anima: no DiT root prefix found (no key ending in {PREFIX_ANCHOR})"
            ))
        })
}

/// Split a loaded Anima DiT checkpoint's keys into `(dit_keys, adapter_keys)` — any key containing
/// `llm_adapter.` is the conditioner, everything else is the Cosmos DiT (prefix-agnostic).
pub fn split_anima_keys(w: &Weights) -> (Vec<String>, Vec<String>) {
    let mut dit = Vec::new();
    let mut adapter = Vec::new();
    for k in w.keys() {
        if k.contains(ADAPTER_MARKER) {
            adapter.push(k.to_string());
        } else {
            dit.push(k.to_string());
        }
    }
    (dit, adapter)
}

/// Resolve the `split_files/` directory holding `diffusion_models/`, `text_encoders/`, `vae/`.
///
/// - `Dir(p)`: `p` itself if it already contains `diffusion_models/`, else `p/split_files`.
/// - `File(dit)`: the DiT's grandparent (`.../split_files/diffusion_models/x.safetensors` → `split_files`).
fn resolve_split_files(source: &WeightsSource) -> Result<PathBuf> {
    match source {
        WeightsSource::Dir(p) => {
            if p.join("diffusion_models").is_dir() {
                Ok(p.clone())
            } else if p.join("split_files").join("diffusion_models").is_dir() {
                Ok(p.join("split_files"))
            } else {
                Err(Error::Msg(format!(
                    "anima: {} is not an Anima split_files dir (no diffusion_models/ or split_files/diffusion_models/)",
                    p.display()
                )))
            }
        }
        WeightsSource::File(dit) => dit
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "anima: cannot resolve split_files/ from DiT file {}",
                    dit.display()
                ))
            }),
    }
}

/// Per-component on-disk footprint (sc-10894) for the MLX fit-gate's staged-residency split. Anima nests
/// its components under a resolved `split_files/` root ([`resolve_split_files`]), NOT directly under
/// `spec.weights`: the Qwen3-0.6B text encoder in `text_encoders/`, the Cosmos DiT in
/// `diffusion_models/`, and the Qwen-Image VAE in `vae/`. A name-guessing consumer would read the
/// encoder as ZERO (`text_encoders` is not a `text_encoder*` match, and it is a level down inside
/// `split_files/`); this seam reports the real bytes. Shared by anima base/aesthetic/turbo (they differ
/// only in the DiT filename inside `diffusion_models/`).
///
/// PRE-WIRING: this split is computed correctly, but anima is NOT yet in the worker's
/// `SEQUENTIAL_CAPABLE_ENGINES` allowlist, so the fit-gate does not consume it until anima is added
/// there in the fan-out (sc-10840). Until then it is inert (the worker uses its whole-model total).
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    let root = resolve_split_files(&spec.weights)?;
    Ok(mlx_gen::PerComponentBytes::from_root_subdirs(
        &root,
        &["text_encoders"],
        &["diffusion_models"],
        &["vae"],
    ))
}

/// The assembled Anima components for one variant.
pub struct AnimaComponents {
    pub dit: CosmosDiT,
    pub conditioner: AnimaTextConditioner,
    pub text_encoder: AnimaQwen3,
    pub vae: QwenVae,
    pub tokenizers: AnimaTokenizers,
}

impl AnimaComponents {
    /// Load all components for `variant` from a weights source (a `split_files/` dir or a DiT file).
    /// Composes the same two per-phase loaders the sequential-residency seam uses
    /// ([`load_heavy_phase`] + [`load_text_phase`], sc-10840) so the resident struct-API assembly and
    /// the staged-residency generator build byte-identical components (loading is RNG-free, so the
    /// heavy-then-text order here is equivalent to the historical per-file order).
    pub fn load(source: &WeightsSource, variant: Variant) -> Result<Self> {
        let (dit, conditioner, vae) = load_heavy_phase(source, variant)?;
        let (text_encoder, tokenizers) = load_text_phase(source, variant)?;
        Ok(Self {
            dit,
            conditioner,
            text_encoder,
            vae,
            tokenizers,
        })
    }
}

/// Load the phase-A **text-encode** components (sc-10840): the Qwen3-0.6B text encoder
/// (`text_encoders/`) + the tokenizers. This is the component dropped first under
/// [`OffloadPolicy::Sequential`](mlx_gen::OffloadPolicy) — encode → **drop the Qwen3 TE** → run the
/// conditioner + DiT + VAE, bounding peak unified memory to `max(Qwen3-TE, DiT+conditioner+VAE)`. The
/// bundled `AnimaTextConditioner` is NOT here: it is `net.llm_adapter.*` inside the DiT file (part of
/// the checkpoint the adapters strict-apply over), so it rides the heavy phase with the DiT
/// ([`load_heavy_phase`]). The TE file is variant-independent; `variant` only resolves the
/// `split_files/` root uniformly with the heavy loader.
pub fn load_text_phase(
    source: &WeightsSource,
    _variant: Variant,
) -> Result<(AnimaQwen3, AnimaTokenizers)> {
    let root = resolve_split_files(source)?;
    let te_weights = Weights::from_file(root.join(TEXT_ENCODER_FILE))?;
    let text_encoder = AnimaQwen3::from_weights(&te_weights, "model", &Qwen3Config::anima())?;
    let tokenizers = AnimaTokenizers::load()?;
    Ok((text_encoder, tokenizers))
}

/// Load the phase-B **heavy render** components (sc-10840): the Cosmos DiT + the bundled
/// `AnimaTextConditioner` (both from the one `diffusion_models/` file — `net.*` DiT + `net.llm_adapter.*`
/// conditioner) + the Qwen-Image VAE (`vae/`). Held after the Qwen3 TE is dropped under `Sequential`;
/// independent of the TE (separate files, RNG-free), so the `Sequential` path rebuilds a byte-identical
/// bundle. Keeping the conditioner here (not in the text phase) is what lets [`apply_anima_adapters`]
/// strict-apply the whole spec — DiT (`blocks.*`) AND conditioner (`llm_adapter.*`) — in one pass, so a
/// LoRA that spans both never loads at partial strength (sc-10274 / sc-10521).
///
/// [`apply_anima_adapters`]: crate::adapters::apply_anima_adapters
pub fn load_heavy_phase(
    source: &WeightsSource,
    variant: Variant,
) -> Result<(CosmosDiT, AnimaTextConditioner, QwenVae)> {
    let root = resolve_split_files(source)?;
    let dit_path = root.join("diffusion_models").join(variant.dit_filename());
    if !dit_path.is_file() {
        return Err(Error::Msg(format!(
            "anima: DiT file not found: {}",
            dit_path.display()
        )));
    }

    // The DiT file carries both the Cosmos DiT and the bundled conditioner. The root prefix is
    // `net` (base) or `model.diffusion_model` (turbo/aesthetic) — detect it.
    let dit_weights = Weights::from_file(&dit_path)?;
    let prefix = detect_dit_prefix(&dit_weights)?;
    let dit = CosmosDiT::from_weights(&dit_weights, &prefix, DitConfig::anima())?;
    let conditioner = AnimaTextConditioner::from_weights(
        &dit_weights,
        &format!("{prefix}.llm_adapter"),
        ConditionerConfig::anima(),
    )?;

    let vae = load_vae(root.join(VAE_FILE))?;
    Ok((dit, conditioner, vae))
}

/// Build a `Weights` holding an fp-cast copy of ONLY the keys containing `marker` (avoids
/// materializing the full DiT in fp32 just to reach the bundled conditioner).
fn cast_subset(w: &Weights, marker: &str, dtype: Dtype) -> Result<Weights> {
    let keys: Vec<String> = w
        .keys()
        .filter(|k| k.contains(marker))
        .map(String::from)
        .collect();
    let mut out = Weights::empty();
    for k in keys {
        out.insert(k.clone(), w.require(&k)?.as_dtype(dtype)?);
    }
    Ok(out)
}

/// Load ONLY the conditioning stack — the Qwen3 text encoder + the bundled `AnimaTextConditioner` —
/// for `variant` at a chosen **compute dtype** (sc-10577). Anima's on-disk TE + conditioner weights are
/// bf16; passing [`Dtype::Float32`] upcasts every one to fp32, mirroring the diffusers reference's
/// `.float()`, to build the **fp32-TE reference variant** used to isolate the bf16-conditioning parity
/// offset. [`Dtype::Bfloat16`] reproduces the resident production modules (the cast is a no-op).
///
/// This is a **measurement path**, not the production load: an fp32 copy ~doubles the TE + conditioner
/// memory, so [`AnimaComponents::load`] keeps them bf16 and this is only reached from the sc-10577
/// parity harness. Returns `(text_encoder, conditioner)`.
pub fn load_conditioning_at_dtype(
    source: &WeightsSource,
    variant: Variant,
    dtype: Dtype,
) -> Result<(AnimaQwen3, AnimaTextConditioner)> {
    let root = resolve_split_files(source)?;

    // Conditioner: bundled in the DiT file under `{prefix}.llm_adapter.*`. Cast only those keys.
    let dit_path = root.join("diffusion_models").join(variant.dit_filename());
    if !dit_path.is_file() {
        return Err(Error::Msg(format!(
            "anima: DiT file not found: {}",
            dit_path.display()
        )));
    }
    let dit_weights = Weights::from_file(&dit_path)?;
    let prefix = detect_dit_prefix(&dit_weights)?;
    let cond_weights = cast_subset(&dit_weights, ADAPTER_MARKER, dtype)?;
    let conditioner = AnimaTextConditioner::from_weights(
        &cond_weights,
        &format!("{prefix}.llm_adapter"),
        ConditionerConfig::anima(),
    )?;

    // Text encoder: its own file. The whole tower runs at `dtype`, so cast every weight.
    let mut te_weights = Weights::from_file(root.join(TEXT_ENCODER_FILE))?;
    te_weights.cast_all(dtype)?;
    let text_encoder =
        AnimaQwen3::from_weights_dtype(&te_weights, "model", &Qwen3Config::anima(), dtype)?;

    Ok((text_encoder, conditioner))
}
