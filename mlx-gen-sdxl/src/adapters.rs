//! SDXL LoRA application (sc-2639) — faithful Rust port of the vendored SceneWorks `lora.py` merge
//! for the mlx-examples SDXL U-Net.
//!
//! Two on-disk formats, both **merged into the dense f32 U-Net weights at load** (`W += δ`, NOT a
//! forward-time residual): SDXL's ancestral sampler is chaos-sensitive, and a residual's
//! `W·x + δ·x` differs from the merged `(W+δ)·x` by ~1 ULP, which cascades to a visible whole-image
//! divergence. Merging reproduces the vendored merged-weight forward bit-for-bit.
//!
//! - **kohya** (`lora_unet_<diffusers path, `.`→`_`>.lora_down/up.weight` + optional `.alpha`) — what
//!   `pipe.save_lora_weights()` and most HF community SDXL LoRAs (incl. LCM-LoRA) ship. The
//!   `_`-flattening is ambiguous (diffusers names like `down_blocks`/`transformer_blocks` already
//!   contain `_`), so the flattened stem is resolved against a table built by flattening every
//!   routable module path — the Rust equivalent of the vendored `unet.named_modules()` walk.
//! - **PEFT** (`base_model.model.unet.<dotted path>.lora_A/B.default.weight` + optional `.alpha`) —
//!   what `peft.save_pretrained()` / SceneWorks' `_SdxlLoraBackend` emit. The dotted path resolves
//!   directly. (kohya `lora_down`/`lora_up` == PEFT `lora_A`/`lora_B`.)
//!
//! Linear-only and matching the vendored reachable surface **exactly** (515 modules on LCM-LoRA):
//! down/up attention (`to_q/k/v`, `to_out.0`), `proj_in`/`proj_out`, resnet `time_emb_proj`. No
//! `mid_block` (the vendored mlx-examples UNet names it `mid_blocks.1.…` so diffusers keys miss it),
//! no ff/GEGLU, no conv, no text-encoder. Keys outside that surface are counted as skipped and
//! surfaced in the returned [`SdxlLoraReport`] — never silently dropped. The correct/complete
//! superset (mid_block + ff, strictly more than the vendored path) is sc-2671.

use std::collections::BTreeMap;

use mlx_rs::ops::{matmul, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::array::scalar;
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::unet::UNet2DConditionModel;

const KOHYA_PREFIX: &str = "lora_unet_";
const PEFT_PREFIX: &str = "base_model.model.unet.";

#[derive(Clone, Copy)]
enum Role {
    Down,
    Up,
    Alpha,
}

#[derive(Default)]
struct LoraTriple {
    down: Option<Array>, // A: [rank, in]
    up: Option<Array>,   // B: [out, rank]
    alpha: Option<f32>,
}

/// Outcome of applying the SDXL adapter specs: how many module weights were merged, and how many
/// keys fell outside the routable surface (mid_block / ff / conv / text-encoder — surfaced, not
/// silently dropped).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SdxlLoraReport {
    pub merged: usize,
    pub skipped_keys: usize,
}

/// Map one safetensors key to `(diffusers_dotted_path, role)`, or `None` if it targets a module
/// outside the routable surface (mirrors the vendored `_classify_key` returning `(None, None)`).
fn classify_key(key: &str, kohya_to_dotted: &BTreeMap<String, String>) -> Option<(String, Role)> {
    if let Some(rem) = key.strip_prefix(PEFT_PREFIX) {
        // PEFT: the dotted diffusers path resolves directly. Accept the peft `.default.weight`
        // infix and the bare `.weight` form.
        for (suf, role) in [
            (".lora_A.default.weight", Role::Down),
            (".lora_B.default.weight", Role::Up),
            (".lora_A.weight", Role::Down),
            (".lora_B.weight", Role::Up),
            (".alpha", Role::Alpha),
        ] {
            if let Some(path) = rem.strip_suffix(suf) {
                return Some((path.to_string(), role));
            }
        }
        return None;
    }
    if let Some(rem) = key.strip_prefix(KOHYA_PREFIX) {
        // kohya: resolve the flattened stem against the routable-path table.
        for (suf, role) in [
            (".lora_down.weight", Role::Down),
            (".lora_up.weight", Role::Up),
            (".alpha", Role::Alpha),
        ] {
            if let Some(stem) = rem.strip_suffix(suf) {
                return kohya_to_dotted.get(stem).map(|d| (d.clone(), role));
            }
        }
        return None;
    }
    // `lora_te1_`/`lora_te2_`/… text-encoder keys land here — deliberately skipped (UNet-only).
    None
}

/// `δ = (B @ A) · (alpha/rank) · scale`, reproducing the vendored `lora.py` merge bit-for-bit.
///
/// The vendored computes `(b@a)` in the LoRA tensors' on-disk dtype (f16 for community/LCM LoRAs),
/// then `.astype(weight.dtype)` (f32) and `* effective_scale`. On the pmetal NAX build a 16-bit
/// `b@a` (K=rank≤512) would hit the dense GEMM bug, so we run the matmul in **f32** (correct) and
/// round the result back through the source dtype — MLX's f16 matmul equals `round_f16` of the
/// f32-accumulated product, so this is bit-identical to the reference without touching the bug.
fn lora_delta(down: &Array, up: &Array, alpha: f32, rank: f32, scale: f32) -> Result<Array> {
    let src = up.dtype(); // f16 for kohya/community LoRAs; f32 makes the round-trip a no-op.
    let ba = matmul(
        &up.as_dtype(Dtype::Float32)?,
        &down.as_dtype(Dtype::Float32)?,
    )?;
    let ba = ba.as_dtype(src)?.as_dtype(Dtype::Float32)?;
    // effective_scale in f64 then f32, matching the reference's Python-float arithmetic.
    let eff = ((alpha as f64 / rank as f64) * scale as f64) as f32;
    Ok(multiply(&ba, scalar(eff))?)
}

fn read_scalar(a: &Array) -> Result<f32> {
    Ok(a.as_dtype(Dtype::Float32)?.reshape(&[1])?.as_slice::<f32>()[0])
}

/// Merge one LoRA file into `unet` at `scale`, classifying every key (both formats) and folding the
/// complete `(down, up)` pairs into their target weights. Half-pairs and out-of-surface / conv-shaped
/// keys are counted as skipped.
fn merge_one(
    unet: &mut UNet2DConditionModel,
    w: &Weights,
    scale: f32,
    kohya_to_dotted: &BTreeMap<String, String>,
    report: &mut SdxlLoraReport,
) -> Result<()> {
    let mut triples: BTreeMap<String, LoraTriple> = BTreeMap::new();
    for key in w.keys().map(str::to_string).collect::<Vec<_>>() {
        match classify_key(&key, kohya_to_dotted) {
            Some((path, Role::Down)) => {
                triples.entry(path).or_default().down = Some(w.require(&key)?.clone())
            }
            Some((path, Role::Up)) => {
                triples.entry(path).or_default().up = Some(w.require(&key)?.clone())
            }
            Some((path, Role::Alpha)) => {
                triples.entry(path).or_default().alpha = Some(read_scalar(w.require(&key)?)?)
            }
            None => report.skipped_keys += 1,
        }
    }

    for (path, t) in triples {
        let (Some(down), Some(up)) = (t.down, t.up) else {
            // Half-pair (a down/up whose partner targeted a non-routable module) — skip.
            report.skipped_keys += 1;
            continue;
        };
        // Conv-shaped (4-D) LoRAs are not Linear merges (matches the vendored `ndim != 2` skip).
        if down.ndim() != 2 || up.ndim() != 2 {
            report.skipped_keys += 2;
            continue;
        }
        let rank = down.shape()[0] as f32;
        let alpha = t.alpha.unwrap_or(rank);
        let delta = lora_delta(&down, &up, alpha, rank, scale)?;
        let parts: Vec<&str> = path.split('.').collect();
        match unet.adaptable_mut(&parts) {
            Some(lin) => {
                lin.merge_dense_delta(&delta)?;
                report.merged += 1;
            }
            // PEFT keys can name a non-routable path (mid_block/ff); kohya stems always resolve
            // (the table is built from routable paths). Either way: surfaced, not merged.
            None => report.skipped_keys += 1,
        }
    }
    Ok(())
}

/// Merge every LoRA spec in `specs` into `unet` (sc-2639). LoRA only — LoKr (which the vendored SDXL
/// path *rejects*) is sc-2640. Builds the kohya `flattened→dotted` table once from the U-Net's
/// routable surface, then merges each file. Errors if a non-empty spec list merges nothing (a real
/// format/prefix misconfiguration — e.g. an original-SD `lora_unet_input_blocks_*` file).
pub fn apply_sdxl_adapters(
    unet: &mut UNet2DConditionModel,
    specs: &[AdapterSpec],
) -> Result<SdxlLoraReport> {
    if specs.is_empty() {
        return Ok(SdxlLoraReport::default());
    }
    let kohya_to_dotted: BTreeMap<String, String> = unet
        .lora_target_paths()
        .into_iter()
        .map(|p| (p.replace('.', "_"), p))
        .collect();

    let mut report = SdxlLoraReport::default();
    for spec in specs {
        if spec.kind != AdapterKind::Lora {
            return Err(Error::Msg(format!(
                "sdxl: {:?} adapters are not supported here (LoRA only; LoKr is sc-2640)",
                spec.kind
            )));
        }
        let w = Weights::from_file(&spec.path)?;
        merge_one(unet, &w, spec.scale, &kohya_to_dotted, &mut report)?;
    }

    if report.merged == 0 {
        return Err(Error::Msg(format!(
            "sdxl: no LoRA target modules matched across {} adapter file(s) — check the format \
             (expected kohya `lora_unet_` with diffusers block naming, or PEFT \
             `base_model.model.unet.`; original-SD `lora_unet_input_blocks_*` and conv/ff-only \
             LoRAs are not supported)",
            specs.len()
        )));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> BTreeMap<String, String> {
        // A tiny routable surface: one attention leaf + a proj.
        [
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
            "up_blocks.0.attentions.0.proj_in",
        ]
        .into_iter()
        .map(|p| (p.replace('.', "_"), p.to_string()))
        .collect()
    }

    #[test]
    fn classify_kohya_resolves_flattened_stem_incl_to_out_0() {
        let t = table();
        // A kohya `to_q` key resolves through the flattened-stem table to its dotted path.
        let (path, role) = classify_key(
            "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
            &t,
        )
        .expect("kohya to_q should resolve");
        assert_eq!(
            path,
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
        );
        assert!(matches!(role, Role::Down));
        // up + alpha roles classify too.
        assert!(matches!(
            classify_key(
                "lora_unet_up_blocks_0_attentions_0_proj_in.lora_up.weight",
                &t
            )
            .unwrap()
            .1,
            Role::Up
        ));
        assert!(matches!(
            classify_key("lora_unet_up_blocks_0_attentions_0_proj_in.alpha", &t)
                .unwrap()
                .1,
            Role::Alpha
        ));
    }

    #[test]
    fn classify_skips_out_of_surface_and_text_encoder_keys() {
        let t = table();
        // mid_block / ff / conv stems aren't in the table → None (skipped, surfaced upstream).
        assert!(classify_key(
            "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
            &t
        )
        .is_none());
        assert!(classify_key(
            "lora_unet_down_blocks_1_resnets_0_conv1.lora_down.weight",
            &t
        )
        .is_none());
        // text-encoder LoRA keys are never UNet targets.
        assert!(classify_key(
            "lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight",
            &t
        )
        .is_none());
    }

    #[test]
    fn classify_peft_resolves_dotted_path_with_default_infix() {
        let t = table();
        // PEFT keys carry the dotted diffusers path directly (with the peft `.default.` infix).
        let (path, role) = classify_key(
            "base_model.model.unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
            &t,
        )
        .unwrap();
        assert_eq!(
            path,
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
        );
        assert!(matches!(role, Role::Down));
        // The bare `.weight` form (no `.default`) is also accepted.
        assert!(matches!(
            classify_key("base_model.model.unet.foo.bar.lora_B.weight", &t)
                .unwrap()
                .1,
            Role::Up
        ));
    }
}
