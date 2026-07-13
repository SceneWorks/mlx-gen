//! Kolors IP-Adapter-Plus (sc-3098) — reference-image conditioning, reusing epic 3041's SDXL
//! IP-Adapter primitive. `Kwai-Kolors/Kolors-IP-Adapter-Plus` is the standard IP-Adapter "plus"
//! stack with two Kolors-specific deltas, both expressible as config:
//!
//!  - the image tower is **CLIP-ViT-L/14-336** (1024-d, 336px → 577 tokens), not the ViT-H the SDXL
//!    IP-Adapter uses — [`VisionConfig::vit_l_14_336`];
//!  - the "plus" [`Resampler`] works at width **2048** (latents `[1,16,2048]`, inner 768), projecting
//!    the 1024-d penultimate → 16×2048 image tokens — [`ResamplerConfig::kolors_plus`].
//!
//! The decoupled cross-attention is identical to SDXL: 70 `ip_adapter.{n}.to_k_ip/to_v_ip` pairs
//! (the IP tokens are 2048-d = the U-Net cross-attention width), installed into the U-Net and added
//! at `ip_scale` alongside the (encoder_hid_proj-projected) ChatGLM3 text path. So this module is a
//! thin loader over the SDXL primitive; the denoise wiring lives on [`crate::Kolors`].

use std::path::Path;

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use mlx_gen_sdxl::{
    load_ip_kv_pairs, ClipVisionEncoder, IpImageEncoder, Resampler, ResamplerConfig, VisionConfig,
};

/// The Kolors ViT-L/14-336 CLIP crop size (the IP-Adapter image tower).
pub const KOLORS_IP_IMAGE_SIZE: usize = 336;

/// `true` when a loaded component is a **pre-quantized** (packed Q4/Q8) snapshot — detected by any
/// `{base}.scales` key. A packed component must NOT be `cast_all`-ed: its `.weight` are u32 codes and
/// its `.scales`/`.biases` carry the quantization at a fixed dtype, so a blanket `astype` corrupts the
/// codes/scales. Mirrors `mlx_gen_sdxl::loader::is_packed` — the SDXL F-082 fix guarded all three of
/// its sites but left this clone uncovered (F-143, sc-11129).
fn is_packed(w: &Weights) -> bool {
    w.keys().any(|k| k.ends_with(".scales"))
}

/// Load the Kolors IP-Adapter-Plus from a `Kwai-Kolors/Kolors-IP-Adapter-Plus` snapshot dir:
/// the `image_encoder/` (CLIP-ViT-L/14-336) + `ip_adapter_plus_general.safetensors` (the
/// `image_proj` Resampler + the 70 `ip_adapter.{n}.to_k_ip/to_v_ip` decoupled-attn pairs). Returns
/// the [`IpImageEncoder`] (reference image → 16×2048 tokens) and the K/V pairs to install into the
/// Kolors U-Net via [`crate::Kolors::install_ip_adapter`]. Cast to `dtype`.
pub fn load_kolors_ip_adapter(
    snapshot: &Path,
    dtype: Dtype,
) -> Result<(IpImageEncoder, Vec<(Array, Array)>)> {
    let mut enc_w = Weights::from_file(snapshot.join("image_encoder/model.safetensors"))?;
    // F-143: never cast a pre-quantized packed payload (its u32 codes/scales would be corrupted),
    // matching the SDXL IP-Adapter loader's F-082 guard.
    if !is_packed(&enc_w) {
        enc_w.cast_all(dtype)?;
    }
    let encoder = ClipVisionEncoder::from_weights(&enc_w, &VisionConfig::vit_l_14_336())?;

    let mut ip_w = Weights::from_file(snapshot.join("ip_adapter_plus_general.safetensors"))?;
    if !is_packed(&ip_w) {
        ip_w.cast_all(dtype)?;
    }
    let resampler = Resampler::from_weights(&ip_w, "image_proj", &ResamplerConfig::kolors_plus())?;
    let pairs = load_ip_kv_pairs(&ip_w)?;

    Ok((
        IpImageEncoder::with_image_size(encoder, resampler, KOLORS_IP_IMAGE_SIZE),
        pairs,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_packed_detects_scales_key_like_sdxl_guard() {
        // F-143 (sc-11129): the `cast_all` guard in `load_kolors_ip_adapter` keys off `is_packed`, which
        // mirrors `mlx_gen_sdxl::loader::is_packed` — a pre-quantized payload (a `{base}.scales` key)
        // must be detected as packed so its u32 codes/scales are never blanket-cast (which would corrupt
        // them). A dense payload has no such key and is cast as before.
        let mut packed = Weights::empty();
        packed.insert(
            "ip_adapter.0.to_k_ip.weight",
            Array::from_slice(&[0u32, 1, 2, 3], &[2, 2]),
        );
        packed.insert(
            "ip_adapter.0.to_k_ip.scales",
            Array::from_slice(&[0.1f32, 0.2], &[2, 1]),
        );
        assert!(
            is_packed(&packed),
            "a payload with a .scales key must read as packed"
        );

        let mut dense = Weights::empty();
        dense.insert(
            "ip_adapter.0.to_k_ip.weight",
            Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]),
        );
        assert!(!is_packed(&dense), "a dense payload has no .scales key");
    }
}
