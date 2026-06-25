//! Backbone / checkpoint registry — a 1:1 port of the reference's two source-of-truth tables:
//! - `pid/_src/inference/checkpoint_registry.py` ((backbone, ckpt_type) → checkpoint path + SR scale);
//! - `pid/_src/inference/pipeline_registry.py` (per-latent-space channel count + latent normalization).
//!
//! A PiD decoder is tied to a **latent space**, not a model. Several backbone tags are *aliases* that
//! reuse another space's student (verified against `checkpoint_registry.py`):
//! - `zimage` / `zimage-turbo` reuse **`flux`** (Z-Image ships Flux1-dev's 16-ch VAE — there is **no**
//!   dedicated zimage checkpoint, contra the original sc-7845 scoping; see that story's comments);
//! - `qwenimage-2512` reuses **`qwenimage`** (same VAE, different upstream transformer);
//! - `flux2-klein-4b` / `flux2-klein-9b` reuse **`flux2`**.
//!
//! Out of scope for this epic (vision-encoder latents, not VAE latents): `dinov2`, `siglip`.

/// Which release a checkpoint comes from.
///
/// `Res2k` = 2048-trained, used as a 4× (512→2048) decoder. `Res2kTo4k` = multi-resolution-trained
/// (2048→4096) with an SD3-style dynamic shift, for 1024 LDM → 4K decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CkptType {
    Res2k,
    Res2kTo4k,
}

/// How a backbone's *normalized* sampler latent maps back to the raw VAE latent. PiD consumes the
/// **normalized** latent directly (the same tensor the native VAE decode receives); this enum records
/// the space so the Phase-2 wiring can assert the engine hands PiD the right thing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LatentNorm {
    /// `raw = latent / scale + shift` (Flux1 / SD3 / SDXL).
    Affine { scale: f32, shift: f32 },
    /// `raw = latent * std + mean`, per-channel vectors carried by the VAE (Qwen-Image).
    PerChannelMeanStd,
    /// `raw = latent * bn_std + bn_mean`, BatchNorm running stats carried by the VAE (Flux2).
    BatchNorm,
}

/// A resolved latent-space entry: the PiD student topology is shared (see [`crate::config::PidConfig`]);
/// only these fields and the checkpoint differ per space.
#[derive(Debug, Clone)]
pub struct BackboneSpec {
    /// The canonical (alias-resolved) latent-space tag.
    pub latent_space: &'static str,
    /// Latent channel count fed to the LQ adapter (`net.lq_latent_channels`). **Not** always the
    /// VAE's raw channel count: the flux2 student consumes the *packed* 128-ch latent (32 raw ×
    /// 2×2 patchify), so this is 128 for flux2 even though `AutoencoderKLFlux2` is 32-ch (sc-7847).
    pub latent_channels: i32,
    /// Latent normalization convention.
    pub latent_norm: LatentNorm,
    /// Spatial compression of the latent grid the LQ adapter is fed (`net.latent_spatial_down_factor`):
    /// pixel side ÷ latent-grid side. 8 for the VAE latent spaces (qwen/flux/sd3/sdxl, latent at H/8);
    /// **16 for flux2**, whose PiD student is fed the *packed* latent at H/16 (the 2×2 patchify halves
    /// the grid again). Drives `PidDecoder`'s output-size math and the LQ upsample ratio (sc-7847).
    pub latent_spatial_down_factor: i32,
    /// Spatial SR factor baked into the student (4× for every diffusers backbone).
    pub pid_scale: i32,
    /// SDXL distilled a **variance-preserving-frame** student (`x_t = √(1−σ²)x0 + σε`); every other
    /// space uses the flow-matching frame (`x_t = (1−σ)x0 + σε`). Drives `from_clean` noising and the
    /// x_t-capture frame at wiring time.
    pub vp_frame: bool,
    /// Checkpoint path (relative to the `nvidia/PiD` snapshot root) for the `2k` release, if shipped.
    pub ckpt_2k: Option<&'static str>,
    /// Checkpoint path for the `2kto4k` release, if shipped.
    pub ckpt_2kto4k: Option<&'static str>,
}

impl BackboneSpec {
    /// Resolve the checkpoint path for a release type (mirrors `get_pid_checkpoint`).
    pub fn checkpoint(&self, ckpt: CkptType) -> Option<&'static str> {
        match ckpt {
            CkptType::Res2k => self.ckpt_2k,
            CkptType::Res2kTo4k => self.ckpt_2kto4k,
        }
    }
}

const QWENIMAGE: BackboneSpec = BackboneSpec {
    latent_space: "qwenimage",
    latent_channels: 16,
    latent_norm: LatentNorm::PerChannelMeanStd,
    latent_spatial_down_factor: 8,
    pid_scale: 4,
    vp_frame: false,
    ckpt_2k: None, // qwenimage ships 2kto4k only
    ckpt_2kto4k: Some(
        "checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth",
    ),
};

const FLUX: BackboneSpec = BackboneSpec {
    latent_space: "flux",
    latent_channels: 16,
    // pipeline_registry.py: vae_scale_factor=0.3611, vae_shift_factor=0.1159 (== our z-image VAE).
    latent_norm: LatentNorm::Affine {
        scale: 0.3611,
        shift: 0.1159,
    },
    latent_spatial_down_factor: 8,
    pid_scale: 4,
    vp_frame: false,
    ckpt_2k: Some("checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth"),
    ckpt_2kto4k: Some(
        "checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth",
    ),
};

const SD3: BackboneSpec = BackboneSpec {
    latent_space: "sd3",
    latent_channels: 16,
    latent_norm: LatentNorm::Affine {
        scale: 1.5305,
        shift: 0.0609,
    },
    latent_spatial_down_factor: 8,
    pid_scale: 4,
    vp_frame: false,
    ckpt_2k: Some("checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth"),
    ckpt_2kto4k: Some(
        "checkpoints/PiD_res2kto4k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth",
    ),
};

const SDXL: BackboneSpec = BackboneSpec {
    latent_space: "sdxl",
    latent_channels: 4,
    latent_norm: LatentNorm::Affine {
        scale: 0.13025,
        shift: 0.0,
    },
    latent_spatial_down_factor: 8,
    pid_scale: 4,
    vp_frame: true, // the one VP-frame student
    ckpt_2k: None,  // sdxl ships 2kto4k only
    ckpt_2kto4k: Some(
        "checkpoints/PiD_res2kto4k_sr4x_official_sdxl_distill_4step/model_ema_bf16.pth",
    ),
};

const FLUX2: BackboneSpec = BackboneSpec {
    latent_space: "flux2",
    // RE-VERIFIED + RESOLVED at wiring time (sc-7847): the PiD student is fed the **packed 128-ch**
    // BN-normalized latent (`net.lq_latent_channels=128`), NOT the 32-ch raw VAE latent. The "32" in
    // `pipeline_registry.py::DiffusionPipelineConfig` is the *VAE* channel count (used only to unpack);
    // `experiment/flux2.py` overrides the net to `lq_latent_channels=128, latent_spatial_down_factor=16`
    // ("32 raw × 2×2 patchify / 16× compression"), and the released checkpoint's first LQ conv is
    // `net.lq_proj.latent_proj.0.weight = (512, 128, 3, 3)` — 128 input channels. Confirmed three ways.
    latent_channels: 128,
    latent_norm: LatentNorm::BatchNorm,
    // 16, not 8: the packed latent the student consumes is at H/16 (VAE 8× + the 2×2 patchify).
    latent_spatial_down_factor: 16,
    pid_scale: 4,
    vp_frame: false,
    ckpt_2k: Some("checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth"),
    // NOTE the `_2606` suffix: the un-suffixed 2kto4k flux2 checkpoint has a color-drift bug and must
    // NOT be used (model card warning). The registry intentionally points at the fixed one.
    ckpt_2kto4k: Some(
        "checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step_2606/model_ema_bf16.pth",
    ),
};

/// Resolve a `--backbone` tag (including the alias tags) to its latent-space spec. Returns `None` for
/// unknown / out-of-scope (`dinov2`, `siglip`) tags.
pub fn lookup(backbone: &str) -> Option<BackboneSpec> {
    let spec = match backbone {
        "qwenimage" | "qwenimage-2512" => QWENIMAGE,
        // Z-Image reuses Flux1's latent space + checkpoint — no dedicated zimage student exists.
        "flux" | "zimage" | "zimage-turbo" => FLUX,
        "sd3" => SD3,
        "sdxl" => SDXL,
        "flux2" | "flux2-klein-4b" | "flux2-klein-9b" => FLUX2,
        _ => return None,
    };
    Some(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwenimage_is_2kto4k_only_perchannel() {
        let s = lookup("qwenimage").unwrap();
        assert_eq!(s.latent_channels, 16);
        assert_eq!(s.latent_norm, LatentNorm::PerChannelMeanStd);
        assert!(s.checkpoint(CkptType::Res2k).is_none());
        assert!(s
            .checkpoint(CkptType::Res2kTo4k)
            .unwrap()
            .contains("qwenimage"));
    }

    #[test]
    fn qwenimage_2512_aliases_qwenimage() {
        assert_eq!(
            lookup("qwenimage-2512")
                .unwrap()
                .checkpoint(CkptType::Res2kTo4k),
            lookup("qwenimage").unwrap().checkpoint(CkptType::Res2kTo4k),
        );
    }

    #[test]
    fn zimage_reuses_flux_not_qwenimage() {
        // The load-bearing scope correction from sc-7845: Z-Image is the Flux latent space.
        let z = lookup("zimage").unwrap();
        assert_eq!(z.latent_space, "flux");
        assert_eq!(
            z.latent_norm,
            LatentNorm::Affine {
                scale: 0.3611,
                shift: 0.1159
            }
        );
        assert_eq!(
            z.checkpoint(CkptType::Res2kTo4k),
            lookup("flux").unwrap().checkpoint(CkptType::Res2kTo4k),
        );
        assert_eq!(lookup("zimage-turbo").unwrap().latent_space, "flux");
    }

    #[test]
    fn sdxl_is_vp_frame_4ch() {
        let s = lookup("sdxl").unwrap();
        assert!(s.vp_frame);
        assert_eq!(s.latent_channels, 4);
        assert_eq!(s.latent_spatial_down_factor, 8);
    }

    #[test]
    fn flux2_feeds_packed_128ch_at_16x() {
        // The load-bearing sc-7847 correction: the flux2 student consumes the PACKED 128-ch BN latent
        // at H/16, not the 32-ch raw VAE latent at H/8. (Checkpoint conv = (512,128,3,3); the
        // experiment config sets lq_latent_channels=128, latent_spatial_down_factor=16.)
        let s = lookup("flux2").unwrap();
        assert_eq!(s.latent_channels, 128);
        assert_eq!(s.latent_spatial_down_factor, 16);
        assert_eq!(s.latent_norm, LatentNorm::BatchNorm);
        // The LQ upsample ratio (sr·lsdf)/patch = (4·16)/16 = 4 (vs 2 for the 8× spaces).
        assert_eq!((s.pid_scale * s.latent_spatial_down_factor) / 16, 4);
    }

    #[test]
    fn flux2_klein_aliases_flux2_with_2606_fix() {
        let k = lookup("flux2-klein-9b").unwrap();
        assert_eq!(k.latent_space, "flux2");
        assert_eq!(k.latent_channels, 128);
        assert_eq!(k.latent_spatial_down_factor, 16);
        assert!(k.checkpoint(CkptType::Res2kTo4k).unwrap().contains("_2606"));
    }

    #[test]
    fn out_of_scope_backbones_are_none() {
        assert!(lookup("dinov2").is_none());
        assert!(lookup("siglip").is_none());
        assert!(lookup("nonsense").is_none());
    }
}
