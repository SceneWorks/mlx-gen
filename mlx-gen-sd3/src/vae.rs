//! Stable Diffusion 3.5 **16-channel VAE** wiring + diffusers→MLX converter (sc-7863, the SD3.5
//! E4 slice).
//!
//! ## Reuse decision (spike sc-7850)
//!
//! SD3.5's VAE is a diffusers `AutoencoderKL` whose `vae/config.json` is **byte-for-byte identical**
//! to Z-Image's — 16-ch latents, `block_out_channels [128,256,512,512]`, `layers_per_block 2`,
//! `norm_num_groups 32`, mid-block attention, no quant/post-quant conv — differing **only** in the
//! two latent-normalization constants (`scaling_factor 1.5305` vs Z-Image's `0.3611`; `shift_factor
//! 0.0609` vs `0.1159`). So this crate **reuses** the Z-Image 16-ch [`Vae`](mlx_gen_z_image::vae::Vae)
//! (AutoencoderKL: GroupNorm-32 conv encoder/decoder + spatial mid-attention, with an
//! already-parameterized scale/shift de-norm) and its diffusers→MLX VAE key remap, plugging SD3.5's
//! own factors via [`Vae::from_weights_with_factors`](mlx_gen_z_image::vae::Vae::from_weights_with_factors).
//!
//! Z-Image's VAE itself derives from the FLUX 16-ch VAE; the flux2 *32-ch packed* VAE (hardcoded
//! `scale=1.0`/`shift=0.0`) is a **different family** and is deliberately NOT used here.
//!
//! ## Latent de-norm direction (named parity risk — verified vs diffusers SD3 pipeline)
//!
//! diffusers `StableDiffusion3Pipeline` applies, on **decode**:
//! `latents = (latents / scaling_factor) + shift_factor` then `vae.decode(latents)`; and on
//! **encode** (img2img init): `latents = (vae.encode(image).sample() - shift_factor) * scaling_factor`.
//! That is exactly the math the reused [`Vae::decode`]/[`Vae::encode`] already implement (Z-Image
//! shares this convention), so plugging SD3.5's two factors yields the correct direction with no new
//! arithmetic. This is the load-bearing parity guarantee of E4.
//!
//! ## What E4 adds
//!
//! * SD3.5 latent factor constants + [`Sd3VaeArch`] (the dimension-parametric channel topology).
//! * [`expected_vae_tensors`] / [`validate_vae_arch`] — the exhaustive shape-checked diffusers VAE
//!   tensor table (`encoder.conv_out [32,512,3,3]`, `decoder.conv_in [512,16,3,3]`, …), mirroring
//!   the E1 transformer validator.
//! * [`build_vae_state_dict`] — the diffusers→MLX VAE converter (a validated 1:1 selection over the
//!   diffusers VAE key set; the NCHW→NHWC conv-weight transpose is applied at load by the reused
//!   [`remap_vae_decoder`]/[`remap_vae_encoder`]).
//! * [`load_sd3_vae`] — assemble the reused `Vae` (decoder + encoder) from an SD3.5 `vae/` snapshot
//!   dir with the SD3.5 factors.

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::Array;

use mlx_gen_z_image::loader::{remap_vae_decoder, remap_vae_encoder};
use mlx_gen_z_image::vae::{Vae, VaeDecoderConfig, VaeEncoderConfig};

/// SD3.5 latent `scaling_factor` (`vae/config.json`, real-weight confirmed). decode: `z/scale+shift`.
pub const SD3_VAE_SCALING_FACTOR: f32 = 1.5305;
/// SD3.5 latent `shift_factor` (`vae/config.json`, real-weight confirmed). encode: `(mean-shift)*scale`.
pub const SD3_VAE_SHIFT_FACTOR: f32 = 0.0609;

/// Latent channel count (16) — half the encoder `conv_out` channel count (32 = mean ⧺ logvar).
pub const SD3_VAE_LATENT_CHANNELS: i64 = 16;
/// VAE spatial downsample factor (8× = three stride-2 down-blocks).
pub const SD3_VAE_SCALE_FACTOR: usize = 8;

/// The dimension-parametric SD3.5 (diffusers `AutoencoderKL`) channel topology. The default
/// ([`Sd3VaeArch::sd3`]) is the real SD3.5 layout; a test can construct a tiny one to exercise the
/// converter/validator against a synthetic checkpoint without the multi-GB weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sd3VaeArch {
    /// Diffusers `block_out_channels` (encoder order; decoder reverses it). Length = #down/up blocks.
    pub block_out_channels: Vec<i64>,
    /// Diffusers `layers_per_block` — the encoder's resnets-per-down-block (the decoder uses +1).
    pub layers_per_block: usize,
    /// Image (pixel) channel count — `in_channels`/`out_channels` (3 = RGB).
    pub image_channels: i64,
    /// Latent channel count (`latent_channels`, 16). The encoder `conv_out` emits `2×` this.
    pub latent_channels: i64,
}

impl Sd3VaeArch {
    /// The real SD3.5-Large / -Turbo / -Medium VAE (one shared 16-ch AutoencoderKL).
    pub fn sd3() -> Self {
        Self {
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            image_channels: 3,
            latent_channels: SD3_VAE_LATENT_CHANNELS,
        }
    }

    fn num_blocks(&self) -> usize {
        self.block_out_channels.len()
    }

    /// The decoder's `VaeDecoderConfig` for this arch: `layers_per_block + 1` resnets per up-block,
    /// upsampling on all but the last. Matches diffusers' decoder construction.
    pub fn decoder_config(&self) -> VaeDecoderConfig {
        let n = self.num_blocks();
        VaeDecoderConfig {
            up_blocks: (0..n)
                .map(|i| (self.layers_per_block + 1, i + 1 < n))
                .collect(),
        }
    }

    /// The encoder's `VaeEncoderConfig` for this arch: `layers_per_block` resnets per down-block,
    /// downsampling on all but the last. Matches diffusers' encoder construction.
    pub fn encoder_config(&self) -> VaeEncoderConfig {
        let n = self.num_blocks();
        VaeEncoderConfig {
            down_blocks: (0..n).map(|i| (self.layers_per_block, i + 1 < n)).collect(),
        }
    }
}

impl Default for Sd3VaeArch {
    fn default() -> Self {
        Self::sd3()
    }
}

/// An expected diffusers VAE tensor: its key and shape (NCHW for conv weights, as on disk).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpectedVaeTensor {
    pub key: String,
    pub shape: Vec<i64>,
}

impl ExpectedVaeTensor {
    fn new(key: impl Into<String>, shape: impl Into<Vec<i64>>) -> Self {
        Self {
            key: key.into(),
            shape: shape.into(),
        }
    }
}

const CONV_K: i64 = 3; // 3×3 convs throughout (resnets, conv_in/out, samplers)
const ATTN_HEADS: i64 = 1; // VAE mid-block spatial attention is single-head

/// Push a `{prefix}.{weight,bias}` conv pair (NCHW `[out,in,k,k]` + `[out]`).
fn conv(out: &mut Vec<ExpectedVaeTensor>, prefix: &str, oc: i64, ic: i64, k: i64) {
    out.push(ExpectedVaeTensor::new(
        format!("{prefix}.weight"),
        vec![oc, ic, k, k],
    ));
    out.push(ExpectedVaeTensor::new(format!("{prefix}.bias"), vec![oc]));
}

/// Push a `{prefix}.{weight,bias}` norm/scale pair (`[c]` each).
fn norm(out: &mut Vec<ExpectedVaeTensor>, prefix: &str, c: i64) {
    out.push(ExpectedVaeTensor::new(format!("{prefix}.weight"), vec![c]));
    out.push(ExpectedVaeTensor::new(format!("{prefix}.bias"), vec![c]));
}

/// Push a `{prefix}.{weight,bias}` Linear pair (`[out,in]` + `[out]`).
fn linear(out: &mut Vec<ExpectedVaeTensor>, prefix: &str, oc: i64, ic: i64) {
    out.push(ExpectedVaeTensor::new(
        format!("{prefix}.weight"),
        vec![oc, ic],
    ));
    out.push(ExpectedVaeTensor::new(format!("{prefix}.bias"), vec![oc]));
}

/// One diffusers `ResnetBlock2D`: norm1→conv1→norm2→conv2 (+ a 1×1 `conv_shortcut` when in≠out).
fn resnet(out: &mut Vec<ExpectedVaeTensor>, prefix: &str, ic: i64, oc: i64) {
    norm(out, &format!("{prefix}.norm1"), ic);
    conv(out, &format!("{prefix}.conv1"), oc, ic, CONV_K);
    norm(out, &format!("{prefix}.norm2"), oc);
    conv(out, &format!("{prefix}.conv2"), oc, oc, CONV_K);
    if ic != oc {
        conv(out, &format!("{prefix}.conv_shortcut"), oc, ic, 1);
    }
}

/// One diffusers `UNetMidBlock2D` (resnet → spatial attention → resnet), all at `c` channels.
fn mid_block(out: &mut Vec<ExpectedVaeTensor>, prefix: &str, c: i64) {
    resnet(out, &format!("{prefix}.resnets.0"), c, c);
    let a = format!("{prefix}.attentions.0");
    norm(out, &format!("{a}.group_norm"), c);
    let head_dim = c / ATTN_HEADS;
    linear(out, &format!("{a}.to_q"), head_dim, c);
    linear(out, &format!("{a}.to_k"), head_dim, c);
    linear(out, &format!("{a}.to_v"), head_dim, c);
    linear(out, &format!("{a}.to_out.0"), c, head_dim);
    resnet(out, &format!("{prefix}.resnets.1"), c, c);
}

/// The exhaustive expected diffusers VAE tensor set (key + NCHW shape) for the given arch.
///
/// Encoder: `conv_in (img→C0)` → down-blocks (`layers_per_block` resnets each; the first resnet of
/// blocks `1..` changes channels `C_{i-1}→C_i` so carries a `conv_shortcut`; all but the last block
/// add a stride-2 downsampler conv `C_i→C_i`) → mid-block (at `C_last`) → `conv_norm_out` →
/// `conv_out (C_last → 2·latent)`.
///
/// Decoder mirrors it over `reversed(block_out_channels)`: `conv_in (latent→Cr0)` → mid-block (at
/// `Cr0`) → up-blocks (`layers_per_block+1` resnets; the first resnet of blocks `1..` changes
/// `Cr_{i-1}→Cr_i` with a `conv_shortcut`; all but the last add a nearest-2× upsampler conv
/// `Cr_i→Cr_i`) → `conv_norm_out` → `conv_out (Cr_last → img)`.
pub fn expected_vae_tensors(arch: &Sd3VaeArch) -> Vec<ExpectedVaeTensor> {
    let mut out: Vec<ExpectedVaeTensor> = Vec::new();
    let ch = &arch.block_out_channels;
    let n = arch.num_blocks();
    let c0 = ch[0];
    let c_last = ch[n - 1];
    let lpb = arch.layers_per_block;

    // ---- encoder ------------------------------------------------------------------------------
    conv(&mut out, "encoder.conv_in", c0, arch.image_channels, CONV_K);
    for i in 0..n {
        let oc = ch[i];
        let ic = if i == 0 { c0 } else { ch[i - 1] };
        let bp = format!("encoder.down_blocks.{i}");
        for r in 0..lpb {
            // Only the FIRST resnet of a block changes channel count (in→out); the rest stay at out.
            let rin = if r == 0 { ic } else { oc };
            resnet(&mut out, &format!("{bp}.resnets.{r}"), rin, oc);
        }
        if i + 1 < n {
            conv(
                &mut out,
                &format!("{bp}.downsamplers.0.conv"),
                oc,
                oc,
                CONV_K,
            );
        }
    }
    mid_block(&mut out, "encoder.mid_block", c_last);
    norm(&mut out, "encoder.conv_norm_out", c_last);
    // conv_out emits 2·latent channels (mean ⧺ logvar): [32,512,3,3] for SD3.5.
    conv(
        &mut out,
        "encoder.conv_out",
        2 * arch.latent_channels,
        c_last,
        CONV_K,
    );

    // ---- decoder ------------------------------------------------------------------------------
    let rch: Vec<i64> = ch.iter().rev().copied().collect();
    let cr0 = rch[0];
    let cr_last = rch[n - 1];
    // conv_in: latent → first (reversed) block channels: [512,16,3,3] for SD3.5.
    conv(
        &mut out,
        "decoder.conv_in",
        cr0,
        arch.latent_channels,
        CONV_K,
    );
    mid_block(&mut out, "decoder.mid_block", cr0);
    for i in 0..n {
        let oc = rch[i];
        let ic = if i == 0 { cr0 } else { rch[i - 1] };
        let bp = format!("decoder.up_blocks.{i}");
        for r in 0..(lpb + 1) {
            let rin = if r == 0 { ic } else { oc };
            resnet(&mut out, &format!("{bp}.resnets.{r}"), rin, oc);
        }
        if i + 1 < n {
            conv(&mut out, &format!("{bp}.upsamplers.0.conv"), oc, oc, CONV_K);
        }
    }
    norm(&mut out, "decoder.conv_norm_out", cr_last);
    conv(
        &mut out,
        "decoder.conv_out",
        arch.image_channels,
        cr_last,
        CONV_K,
    );

    out
}

/// The total number of VAE tensors the validator expects (244 for the real SD3.5 VAE).
pub fn expected_vae_tensor_count(arch: &Sd3VaeArch) -> usize {
    expected_vae_tensors(arch).len()
}

/// Validate a known VAE tensor set (key → NCHW shape) against the expected SD3.5 VAE arch. Reports
/// missing, extra (non-arch), and shape-mismatched keys. Works for an in-memory converted map or a
/// safetensors-header read. Mirrors the E1 transformer [`crate::convert::validate_arch`].
pub fn validate_vae_arch<'a, I>(arch: &Sd3VaeArch, provided: I) -> Result<()>
where
    I: IntoIterator<Item = (&'a str, &'a [i64])>,
{
    let expected: HashMap<String, Vec<i64>> = expected_vae_tensors(arch)
        .into_iter()
        .map(|e| (e.key, e.shape))
        .collect();
    let provided: HashMap<&str, &[i64]> = provided.into_iter().collect();

    let mut missing: Vec<&String> = expected
        .keys()
        .filter(|k| !provided.contains_key(k.as_str()))
        .collect();
    let mut extra: Vec<&&str> = provided
        .keys()
        .filter(|k| !expected.contains_key(**k))
        .collect();
    let mut bad_shape: Vec<String> = provided
        .iter()
        .filter_map(|(k, shape)| {
            expected.get(*k).and_then(|exp| {
                if exp.len() == shape.len() && exp.iter().zip(*shape).all(|(&e, &g)| e == g) {
                    None
                } else {
                    Some(format!("{k} (expected {exp:?}, got {shape:?})"))
                }
            })
        })
        .collect();

    if missing.is_empty() && extra.is_empty() && bad_shape.is_empty() {
        return Ok(());
    }
    missing.sort();
    extra.sort();
    bad_shape.sort();
    Err(Error::Msg(format!(
        "SD3.5 VAE architecture validation FAILED: {} missing, {} extra, {} shape mismatch. \
         expected {} tensors. missing={:?} extra={:?} shape={:?}",
        missing.len(),
        extra.len(),
        bad_shape.len(),
        expected.len(),
        &missing[..missing.len().min(5)],
        &extra[..extra.len().min(5)],
        &bad_shape[..bad_shape.len().min(5)],
    )))
}

/// Build the MLX-side VAE state dict from a diffusers `AutoencoderKL` tensor set (`src`).
///
/// The SD3.5 VAE diffusers layout is the convention the reused Z-Image `Vae` consumes (after the
/// `remap_vae_*` name/transpose pass applied at load), so this converter is a **validated 1:1
/// selection** over the expected diffusers key set — it drops any non-arch tensor and surfaces a
/// missing one, the single seam should a key ever need a rename. Tensors are returned in their
/// on-disk dtype/layout (NCHW conv weights); the NCHW→NHWC transpose is applied later by
/// [`remap_vae_decoder`]/[`remap_vae_encoder`] at load time, matching the E1 transformer path.
pub fn build_vae_state_dict(src: &Weights, arch: &Sd3VaeArch) -> Result<HashMap<String, Array>> {
    let mut out: HashMap<String, Array> = HashMap::new();
    for ExpectedVaeTensor { key, .. } in expected_vae_tensors(arch) {
        let t = src.require(&key)?;
        out.insert(key, t.clone());
    }
    Ok(out)
}

/// Read a safetensors file's tensor names + NCHW shapes from the JSON header alone (no weights).
/// Shares the E1 reader.
pub fn safetensors_header_shapes(path: &Path) -> Result<HashMap<String, Vec<i64>>> {
    crate::convert::safetensors_header_shapes(path)
}

/// Validate a real on-disk SD3.5 `vae/` directory's tensor set against [`Sd3VaeArch`] using only the
/// safetensors headers (no weight load). Catches a wrong-repo / wrong-shape / truncated VAE before
/// any multi-GB load.
pub fn validate_vae_dir(arch: &Sd3VaeArch, vae_dir: &Path) -> Result<()> {
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();
    let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(vae_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(Error::Msg(format!(
            "no vae safetensors in {}",
            vae_dir.display()
        )));
    }
    for shard in &shards {
        shapes.extend(safetensors_header_shapes(shard)?);
    }
    let provided: Vec<(&str, &[i64])> = shapes
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    validate_vae_arch(arch, provided.iter().copied())
}

/// Load the full SD3.5 VAE (decoder + encoder) from a `vae/` snapshot directory, reusing the
/// Z-Image 16-ch [`Vae`] AutoencoderKL with **SD3.5's** `1.5305` / `0.0609` latent factors.
///
/// Applies the same diffusers→MLX name remaps + NCHW→NHWC conv-weight transposes as Z-Image (the VAE
/// trees are identically named), then constructs the `Vae` via
/// [`Vae::from_weights_with_factors`] and attaches the img2img encoder.
pub fn load_sd3_vae(vae_dir: &Path) -> Result<Vae> {
    let arch = Sd3VaeArch::sd3();
    let mut w = Weights::from_dir(vae_dir)?;
    remap_vae_decoder(&mut w)?;
    remap_vae_encoder(&mut w)?;
    Vae::from_weights_with_factors(
        &w,
        "",
        &arch.decoder_config(),
        SD3_VAE_SCALING_FACTOR,
        SD3_VAE_SHIFT_FACTOR,
    )?
    .with_encoder(&w, "encoder", &arch.encoder_config())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factors_match_sd3_config() {
        // Real-weight-confirmed vae/config.json values (NOT Z-Image's 0.3611 / 0.1159).
        assert_eq!(SD3_VAE_SCALING_FACTOR, 1.5305);
        assert_eq!(SD3_VAE_SHIFT_FACTOR, 0.0609);
        // Guard against accidentally inheriting the Z-Image defaults.
        assert_ne!(SD3_VAE_SCALING_FACTOR, Vae::SCALING_FACTOR);
        assert_ne!(SD3_VAE_SHIFT_FACTOR, Vae::SHIFT_FACTOR);
    }

    #[test]
    fn arch_configs_match_diffusers_construction() {
        let a = Sd3VaeArch::sd3();
        // Encoder: 2 resnets/down-block, downsample on first 3 of 4.
        assert_eq!(
            a.encoder_config().down_blocks,
            vec![(2, true), (2, true), (2, true), (2, false)]
        );
        // Decoder: 3 resnets/up-block, upsample on first 3 of 4.
        assert_eq!(
            a.decoder_config().up_blocks,
            vec![(3, true), (3, true), (3, true), (3, false)]
        );
    }

    #[test]
    fn expected_count_is_244_for_real_sd3_vae() {
        // The real SD3.5 VAE checkpoint is 244 tensors (122 encoder + 122 decoder), confirmed by a
        // header dump of stabilityai/stable-diffusion-3.5-large vae/.
        assert_eq!(expected_vae_tensor_count(&Sd3VaeArch::sd3()), 244);
    }

    /// The two load-bearing conv shapes the AC calls out by name.
    #[test]
    fn special_conv_shapes_match_ac() {
        let t = expected_vae_tensors(&Sd3VaeArch::sd3());
        let find = |k: &str| t.iter().find(|e| e.key == k).map(|e| e.shape.clone());
        assert_eq!(find("encoder.conv_out.weight"), Some(vec![32, 512, 3, 3]));
        assert_eq!(find("decoder.conv_in.weight"), Some(vec![512, 16, 3, 3]));
        assert_eq!(find("encoder.conv_in.weight"), Some(vec![128, 3, 3, 3]));
        assert_eq!(find("decoder.conv_out.weight"), Some(vec![3, 128, 3, 3]));
    }

    /// conv_shortcuts appear exactly where channels change between blocks (real-weight confirmed).
    #[test]
    fn conv_shortcuts_at_channel_transitions() {
        let t = expected_vae_tensors(&Sd3VaeArch::sd3());
        let shortcuts: Vec<(&str, &[i64])> = t
            .iter()
            .filter(|e| e.key.ends_with("conv_shortcut.weight"))
            .map(|e| (e.key.as_str(), e.shape.as_slice()))
            .collect();
        // Encoder: blocks 1 (128→256) and 2 (256→512). Decoder (reversed ch): blocks 2 (512→256)
        // and 3 (256→128). 1×1 convs.
        let mut keys: Vec<&str> = shortcuts.iter().map(|(k, _)| *k).collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "decoder.up_blocks.2.resnets.0.conv_shortcut.weight",
                "decoder.up_blocks.3.resnets.0.conv_shortcut.weight",
                "encoder.down_blocks.1.resnets.0.conv_shortcut.weight",
                "encoder.down_blocks.2.resnets.0.conv_shortcut.weight",
            ]
        );
        for (k, s) in shortcuts {
            assert_eq!(s[2], 1, "{k} shortcut is 1×1");
            assert_eq!(s[3], 1, "{k} shortcut is 1×1");
        }
    }

    #[test]
    fn validate_accepts_exact_set_rejects_perturbations() {
        let arch = Sd3VaeArch::sd3();
        let exp = expected_vae_tensors(&arch);

        // Exact set passes.
        let provided: Vec<(&str, &[i64])> = exp
            .iter()
            .map(|e| (e.key.as_str(), e.shape.as_slice()))
            .collect();
        validate_vae_arch(&arch, provided.iter().copied()).unwrap();

        // A missing key fails.
        let missing: Vec<(&str, &[i64])> = exp[1..]
            .iter()
            .map(|e| (e.key.as_str(), e.shape.as_slice()))
            .collect();
        assert!(validate_vae_arch(&arch, missing.iter().copied()).is_err());

        // A wrong shape fails.
        let bad = vec![1i64, 2, 3, 4];
        let mut perturbed: Vec<(&str, &[i64])> = exp
            .iter()
            .map(|e| (e.key.as_str(), e.shape.as_slice()))
            .collect();
        perturbed[0] = (exp[0].key.as_str(), bad.as_slice());
        assert!(validate_vae_arch(&arch, perturbed.iter().copied()).is_err());

        // An extra key fails.
        let extra_shape = vec![1i64];
        let mut extra: Vec<(&str, &[i64])> = exp
            .iter()
            .map(|e| (e.key.as_str(), e.shape.as_slice()))
            .collect();
        extra.push(("encoder.bogus.weight", extra_shape.as_slice()));
        assert!(validate_vae_arch(&arch, extra.iter().copied()).is_err());
    }

    /// A tiny synthetic arch keeps the converter/validator testable without multi-GB weights and
    /// proves the parametric derivation isn't hardcoded to the 244-tensor real layout.
    #[test]
    fn tiny_arch_is_consistent() {
        let tiny = Sd3VaeArch {
            block_out_channels: vec![8, 16],
            layers_per_block: 1,
            image_channels: 3,
            latent_channels: 4,
        };
        let t = expected_vae_tensors(&tiny);
        let find = |k: &str| t.iter().find(|e| e.key == k).map(|e| e.shape.clone());
        // encoder.conv_out emits 2×latent = 8 channels from C_last = 16.
        assert_eq!(find("encoder.conv_out.weight"), Some(vec![8, 16, 3, 3]));
        // decoder.conv_in maps latent(4) → reversed-C0(16).
        assert_eq!(find("decoder.conv_in.weight"), Some(vec![16, 4, 3, 3]));
        // 2 blocks → 1 downsampler (encoder) + 1 upsampler (decoder).
        assert_eq!(
            t.iter().filter(|e| e.key.contains("downsamplers")).count(),
            2 // weight + bias
        );
        // self-consistent: the validator accepts its own expected set.
        let provided: Vec<(&str, &[i64])> = t
            .iter()
            .map(|e| (e.key.as_str(), e.shape.as_slice()))
            .collect();
        validate_vae_arch(&tiny, provided.iter().copied()).unwrap();
    }
}
