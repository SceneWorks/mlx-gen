//! `mlx-gen-ltx` model entry: the LTX-2.3 **AudioVideo** descriptor, the config-driven `load`, the
//! public `generate`, and registry self-registration.
//!
//! **Scope (sc-2684):** the production path is the full **synchronized audio+video** generation
//! (`generate_av.py`) — prompt → Gemma-3 tokenizer → [`LtxTextEncoder::encode_av`] (video 4096 +
//! audio 2048 embeddings) → seeded noise → the joint 2-stage distilled denoise ([`generate_av_latents`]:
//! both streams through the dual-modality [`AvDiT`] with cross-modal attention every step; the video is
//! 2× upsampled between stages, the audio is not) → [`LtxVideoVae`] decode → uint8 RGB frames **plus**
//! [`AudioDecoder`] → [`LtxVocoder`] → an [`mlx_gen::media::AudioTrack`]. The audio is always denoised
//! (it conditions the video via cross-modal attention), so the video differs from the video-only
//! sc-2679 building block (`LtxDiT`, audio disabled). `--no-audio` (`req.video_mode == "no_audio"`)
//! runs the full A/V denoise but skips the audio decode (`audio: None`).
//!
//! 16-bit-WAV write + peak-normalize + the `ffmpeg -c:v copy -c:a aac -shortest` mux are **host-side**
//! (the `AudioTrack` is the raw vocoder waveform — `generate_av.py`'s `audio_np` before `save_audio`),
//! matching how MP4 video muxing already lives outside the crate (the Wan sibling).
//!
//! The Gemma text-encoder weights are a **separate** snapshot (the base model dir holds only the
//! `connector`/transformer/vae); [`resolve_gemma_dir`] locates them via `$LTX_GEMMA_DIR` or the HF
//! cache (`mlx-community/gemma-3-12b-it-bf16`).
//!
//! **Precision.** Selected by `LoadSpec::precision`: `Bf16` (the default) → the reference's **native**
//! bf16 activations × Q8 ([`Precision::Bf16Q8`]) — the production-speed path; `Fp32` →
//! [`Precision::F32Q8`] (f32 activations × Q8) — the quality target. Both are bit-exact to their
//! reference golden (sc-2842). The latent statistics follow the path dtype (so the upsampler + denoise
//! run in that precision); the VAE decode stays f32 (a post-sampling quality island, pixel-parity
//! either way), and the Gemma backbone runs bf16 as the reference does. Distilled 2-stage → **no CFG**
//! (guidance baked in). Q4/Q8-of-everything, I2V, LoRA/LoKr, and the audio half are sibling slices.

use mlx_rs::{random, Array, Dtype};

use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{
    default_seed, Capabilities, Error, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, Modality, ModelDescriptor, Precision as LoadPrecision, Progress, Result,
    WeightsSource,
};

use crate::audio_vae::AudioDecoder;
use crate::config::{AudioVaeConfig, LtxConfig, LtxVaeConfig, VocoderConfig};
use crate::gemma::GemmaConfig;
use crate::pipeline::{decode_audio_track, decode_to_frames, generate_av_latents};
use crate::positions::{compute_audio_frames, create_audio_position_grid, create_position_grid};
use crate::text_encoder::LtxTextEncoder;
use crate::tokenizer::LtxTokenizer;
use crate::transformer::{AvDiT, Precision};
use crate::upsampler::LatentUpsampler;
use crate::vae::LtxVideoVae;
use crate::vocoder::LtxVocoder;

/// Public registry id: `mlx_gen::load("ltx_2_3", spec)`.
pub const MODEL_ID: &str = "ltx_2_3";

/// Reference text-encoder token budget (`LTX2TextEncoder.encode` default `max_length=1024`).
const MAX_PROMPT_TOKENS: usize = 1024;
/// LTX-2 latent channels.
const LATENT_CHANNELS: i32 = 128;
/// Audio latent channels (pre-patchify) and mel bins — the audio latent is `(1, 8, T, 16)`.
const AUDIO_LATENT_CHANNELS: i32 = 8;
const AUDIO_MEL_BINS: i32 = 16;
/// VAE temporal compression (8×): `latent_frames = 1 + (frames − 1) / 8`.
const TEMPORAL_SCALE: u32 = 8;
/// VAE spatial compression (32×); stage-1 additionally halves resolution.
const SPATIAL_SCALE: u32 = 32;

/// Stable identity + advertised capabilities for the LTX-2.3 AudioVideo model (produces video frames
/// + a synchronized audio track).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "ltx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Distilled 2-stage path: CFG is forced to 1.0, so no guidance / negative prompt.
            // (I2V, LoRA, LoKr, and Q4/Q8-of-everything are sibling slices.)
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            samplers: Vec::new(),
            schedulers: Vec::new(),
            // height/width must be divisible by 64 (stage-1 runs at //2//32).
            min_size: 64,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// The loaded LTX-2.3 model: the assembled **AudioVideo** components + the cached descriptor. The
/// production path is the joint A/V denoise (`generate_av.py`) — the audio latents are always
/// denoised (the cross-modal attention couples them to the video every step), so the video stream
/// differs from the video-only sc-2679 building block. Audio is decoded into the output unless
/// `--no-audio` (`req.video_mode == "no_audio"`).
pub struct Ltx {
    descriptor: ModelDescriptor,
    tokenizer: LtxTokenizer,
    text_encoder: LtxTextEncoder,
    transformer: AvDiT,
    upsampler: LatentUpsampler,
    vae: LtxVideoVae,
    audio_decoder: AudioDecoder,
    vocoder: LtxVocoder,
    latent_mean: Array,
    latent_std: Array,
    audio_sample_rate: u32,
    stat_dt: Dtype,
}

/// Locate the Gemma-3-12B text-encoder snapshot. `$LTX_GEMMA_DIR` wins; otherwise the newest
/// `mlx-community/gemma-3-12b-it-bf16` snapshot in the HF cache.
fn resolve_gemma_dir() -> Result<std::path::PathBuf> {
    if let Ok(d) = std::env::var("LTX_GEMMA_DIR") {
        return Ok(d.into());
    }
    let home = std::env::var("HOME").map_err(|_| Error::Msg("ltx_2_3: HOME unset".into()))?;
    let base = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--mlx-community--gemma-3-12b-it-bf16/snapshots");
    let newest = std::fs::read_dir(&base)
        .map_err(|_| {
            Error::Msg(format!(
                "ltx_2_3: gemma snapshot not found at {} (set $LTX_GEMMA_DIR)",
                base.display()
            ))
        })?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .max_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
    newest.ok_or_else(|| Error::Msg("ltx_2_3: no gemma snapshot in the HF cache".into()))
}

/// Load the model from a split-weight snapshot directory (the `ltx_2_3_base*` tree). Reads
/// `embedded_config.json`, locates the Gemma TE separately, and assembles every component.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p,
            WeightsSource::File(_) => return Err(Error::Msg(
                "ltx_2_3: expected a model directory (split-weight snapshot), not a single file"
                    .into(),
            )),
        };
    // Precision selection. `Bf16` (the [`LoadSpec`] default) → the reference's **native** bf16
    // activations × Q8 — the production-speed path; `Fp32` → f32 activations × Q8 — the quality
    // target. Both are bit-exact to their reference golden (sc-2842; the distilled stage-1 sampler is
    // chaos-sensitive, so each per-forward is bit-exact). The latent statistics (the upsampler's
    // un-/re-normalize) follow the path dtype so the whole denoise stays in that precision; the VAE
    // decode stays f32 in both — a post-sampling quality island (pixel-parity either way).
    let (dit_prec, stat_dt) = match spec.precision {
        LoadPrecision::Bf16 => (Precision::Bf16Q8, Dtype::Bfloat16),
        LoadPrecision::Fp32 => (Precision::F32Q8, Dtype::Float32),
    };
    if spec.quantize.is_some() {
        return Err(Error::Msg(
            "ltx_2_3: Q4/Q8-of-everything is a sibling slice (sc-2686); the transformer is already \
             shipped Q8"
                .into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "ltx_2_3: LoRA/LoKr adapters are sibling slices (sc-2687 / sc-2393), not yet wired"
                .into(),
        ));
    }

    let config = LtxConfig::from_model_dir(root)?;
    let vae_config = LtxVaeConfig::from_model_dir(root)?;
    let audio_vae_config = AudioVaeConfig::from_model_dir(root)?;
    let vocoder_config = VocoderConfig::from_model_dir(root)?;

    let gemma_dir = resolve_gemma_dir()?;
    let gemma_w = Weights::from_dir(&gemma_dir)?;
    let connector_w = Weights::from_file(root.join("connector.safetensors"))?;
    let transformer_w = Weights::from_file(root.join("transformer.safetensors"))?;
    let upsampler_w = Weights::from_file(root.join("upsampler.safetensors"))?;
    let vae_w = Weights::from_file(root.join("vae_decoder.safetensors"))?;
    let audio_vae_w = Weights::from_file(root.join("audio_vae.safetensors"))?;
    let vocoder_w = Weights::from_file(root.join("vocoder.safetensors"))?;

    // The text encoder runs **bf16** end-to-end (the reference TE dtype; S1-validated), producing both
    // the video (4096) and audio (2048) embeddings. Its bf16 embeddings enter the DiT, which upcasts
    // the cross-attn context as the reference transformer does.
    let text_encoder = LtxTextEncoder::from_weights_av(
        &gemma_w,
        &connector_w,
        GemmaConfig::gemma_3_12b(),
        &config,
        Dtype::Bfloat16,
    )?;
    let transformer = AvDiT::from_weights(&transformer_w, &config, dit_prec)?;
    let upsampler = LatentUpsampler::from_weights(&upsampler_w)?;
    let vae = LtxVideoVae::from_weights(&vae_w, None, &vae_config)?;
    // The audio VAE decoder + vocoder run f32 (post-sampling quality islands, gated bit-exact).
    let audio_decoder = AudioDecoder::from_weights(&audio_vae_w, &audio_vae_config)?;
    let vocoder = LtxVocoder::from_weights(&vocoder_w, &vocoder_config)?;
    let audio_sample_rate = vocoder_config.final_sample_rate() as u32;
    // The VAE `per_channel_statistics` double as the upsampler's latent norm, at the path dtype.
    let latent_mean = to_dtype(vae_w.require("per_channel_statistics.mean")?, stat_dt)?;
    let latent_std = to_dtype(vae_w.require("per_channel_statistics.std")?, stat_dt)?;

    Ok(Box::new(Ltx {
        descriptor: descriptor(),
        tokenizer: LtxTokenizer::from_dir(&gemma_dir)?,
        text_encoder,
        transformer,
        upsampler,
        vae,
        audio_decoder,
        vocoder,
        latent_mean,
        latent_std,
        audio_sample_rate,
        stat_dt,
    }))
}

impl Ltx {
    /// Latent dims `(frames, stage1_h, stage1_w, stage2_h, stage2_w)` for a request.
    pub(crate) fn latent_dims(req: &GenerationRequest) -> (usize, usize, usize, usize, usize) {
        let frames = req.frames.unwrap_or(1).max(1);
        let latent_frames = 1 + (frames as usize - 1) / TEMPORAL_SCALE as usize;
        let (h, w) = (req.height, req.width);
        (
            latent_frames,
            (h / 2 / SPATIAL_SCALE) as usize,
            (w / 2 / SPATIAL_SCALE) as usize,
            (h / SPATIAL_SCALE) as usize,
            (w / SPATIAL_SCALE) as usize,
        )
    }

    /// Audio latent-frame count for the request (`compute_audio_frames(num_frames, fps)`).
    pub(crate) fn audio_frames(req: &GenerationRequest) -> usize {
        compute_audio_frames(
            req.frames.unwrap_or(1).max(1) as usize,
            req.fps.unwrap_or(24) as f64,
        )
    }

    /// `--no-audio` toggle: `req.video_mode == "no_audio"` runs the full A/V denoise but skips the
    /// audio decode + returns `audio: None` (the reference `--no-audio`).
    fn no_audio(req: &GenerationRequest) -> bool {
        matches!(
            req.video_mode.as_deref(),
            Some("no_audio") | Some("video_only")
        )
    }

    /// The full A/V path with **injected** stage noise (the deterministic seam `generate` calls with
    /// RNG-drawn noise and the e2e parity test calls with the reference samples). Encodes the prompt
    /// to both video + audio embeddings, then defers to
    /// [`generate_av_from_embeddings`](Self::generate_av_from_embeddings).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_with_noise(
        &self,
        req: &GenerationRequest,
        video_s1: &Array,
        video_s2: &Array,
        audio_s1: &Array,
        audio_s2: &Array,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let (ids, mask) = self.tokenizer.encode(&req.prompt, MAX_PROMPT_TOKENS)?;
        let (video_ctx, audio_ctx) = self.text_encoder.encode_av(&ids, &mask)?;
        self.generate_av_from_embeddings(
            req,
            &video_ctx,
            &audio_ctx,
            video_s1,
            video_s2,
            audio_s1,
            audio_s2,
            on_progress,
        )
    }

    /// The A/V path from **injected** text embeddings + noise — the pipeline-only seam (no Gemma), so
    /// the parity test can gate the joint 2-stage pipeline + video/audio decode against the reference
    /// conditioning. `video_ctx` `(1, ctx, 4096)`, `audio_ctx` `(1, ctx, 2048)`; video noise
    /// `(1,128,F,h,w)` per stage; audio noise `(1,8,T,16)` per stage (`T = audio_frames`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_av_from_embeddings(
        &self,
        req: &GenerationRequest,
        video_ctx: &Array,
        audio_ctx: &Array,
        video_s1: &Array,
        video_s2: &Array,
        audio_s1: &Array,
        audio_s2: &Array,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let (lf, h1, w1, h2, w2) = Self::latent_dims(req);
        let pos1 = create_position_grid(1, lf, h1, w1);
        let pos2 = create_position_grid(1, lf, h2, w2);
        let audio_pos = create_audio_position_grid(1, Self::audio_frames(req));

        let mut step = 0usize;
        let (video_latents, audio_latents) = generate_av_latents(
            &self.transformer,
            &self.upsampler,
            video_s1,
            &pos1,
            video_s2,
            &pos2,
            audio_s1,
            audio_s2,
            &audio_pos,
            video_ctx,
            audio_ctx,
            &self.latent_mean,
            &self.latent_std,
            &mut |_| {
                step += 1;
                on_progress(Progress::Step {
                    current: step as u32,
                    total: 11,
                });
            },
        )?;

        on_progress(Progress::Decoding);
        let frames = decode_to_frames(&self.vae, &video_latents)?;
        let images = frames_to_images(&frames)?;
        // Audio always denoised (it conditions the video); decode it unless `--no-audio`.
        let audio = if Self::no_audio(req) {
            None
        } else {
            Some(decode_audio_track(
                &self.audio_decoder,
                &self.vocoder,
                &audio_latents,
                self.audio_sample_rate,
            )?)
        };
        Ok(GenerationOutput::Video {
            frames: images,
            fps: req.fps.unwrap_or(24),
            audio,
        })
    }
}

/// Capability-driven request validation (weight-free, so it's unit-testable without a load):
/// non-empty prompt, 64-aligned width/height (stage-1 runs at //2//32), `num_frames = 1 + 8·k`.
pub(crate) fn validate_request(req: &GenerationRequest) -> Result<()> {
    if req.prompt.is_empty() {
        return Err(Error::Msg("ltx_2_3: prompt must not be empty".into()));
    }
    if !req.width.is_multiple_of(64) || !req.height.is_multiple_of(64) {
        return Err(Error::Msg(format!(
            "ltx_2_3: width/height must be divisible by 64 (got {}x{})",
            req.width, req.height
        )));
    }
    if let Some(frames) = req.frames {
        if frames % 8 != 1 {
            return Err(Error::Msg(format!(
                "ltx_2_3: num_frames must be 1 + 8·k (got {frames})"
            )));
        }
    }
    Ok(())
}

/// `(F, H, W, 3)` uint8 → one [`Image`] per frame.
pub(crate) fn frames_to_images(frames: &Array) -> Result<Vec<Image>> {
    let sh = frames.shape(); // (F, H, W, 3)
    let (f, h, w) = (sh[0] as usize, sh[1] as u32, sh[2] as u32);
    let data = frames.as_slice::<u8>();
    let per = (h as usize) * (w as usize) * 3;
    Ok((0..f)
        .map(|i| Image {
            width: w,
            height: h,
            pixels: data[i * per..(i + 1) * per].to_vec(),
        })
        .collect())
}

impl Generator for Ltx {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let (lf, h1, w1, h2, w2) = Self::latent_dims(req);
        let af = Self::audio_frames(req) as i32;
        let seed = req.seed.unwrap_or_else(default_seed);
        // Seeded noise at the path dtype (the reference seeds `normal(...).astype(model_dtype)`). RNG
        // is not portable to mlx-python, so the pixel/waveform parity gate injects the reference
        // samples via `generate_with_noise`. Distinct keys per stage/modality.
        let normal = |key: u64, shape: &[i32]| -> Result<Array> {
            let k = random::key(key)?;
            Ok(random::normal::<f32>(shape, None, None, Some(&k))?.as_dtype(self.stat_dt)?)
        };
        let video_s1 = normal(seed, &[1, LATENT_CHANNELS, lf as i32, h1 as i32, w1 as i32])?;
        let video_s2 = normal(
            seed.wrapping_add(1),
            &[1, LATENT_CHANNELS, lf as i32, h2 as i32, w2 as i32],
        )?;
        let audio_s1 = normal(
            seed.wrapping_add(2),
            &[1, AUDIO_LATENT_CHANNELS, af, AUDIO_MEL_BINS],
        )?;
        let audio_s2 = normal(
            seed.wrapping_add(3),
            &[1, AUDIO_LATENT_CHANNELS, af, AUDIO_MEL_BINS],
        )?;
        self.generate_with_noise(req, &video_s1, &video_s2, &audio_s1, &audio_s2, on_progress)
    }
}

inventory::submit! {
    mlx_gen::ModelRegistration { descriptor, load }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_dims_matches_reference_formula() {
        // 256×256, 9 frames: latent_frames = 1+(9-1)/8 = 2; stage1 = H/2/32 = 4; stage2 = H/32 = 8.
        let req = GenerationRequest {
            width: 256,
            height: 256,
            frames: Some(9),
            ..Default::default()
        };
        assert_eq!(Ltx::latent_dims(&req), (2, 4, 4, 8, 8));
        // 512×768, 1 frame: latent_frames = 1; stage1 = 8×12; stage2 = 16×24.
        let req = GenerationRequest {
            width: 768,
            height: 512,
            frames: Some(1),
            ..Default::default()
        };
        assert_eq!(Ltx::latent_dims(&req), (1, 8, 12, 16, 24));
    }

    #[test]
    fn validate_request_enforces_constraints() {
        let base = GenerationRequest {
            prompt: "a".into(),
            width: 512,
            height: 512,
            frames: Some(33),
            ..Default::default()
        };
        assert!(validate_request(&base).is_ok());
        assert!(validate_request(&GenerationRequest {
            prompt: String::new(),
            ..base.clone()
        })
        .is_err());
        assert!(validate_request(&GenerationRequest {
            width: 500,
            ..base.clone()
        })
        .is_err());
        assert!(validate_request(&GenerationRequest {
            frames: Some(32),
            ..base.clone()
        })
        .is_err());
    }

    #[test]
    fn frames_to_images_splits_per_frame() {
        // (F=2, H=1, W=2, 3): each frame = 6 bytes.
        let data: Vec<u8> = (0..12).collect();
        let frames = Array::from_slice(&data, &[2, 1, 2, 3]);
        let imgs = frames_to_images(&frames).unwrap();
        assert_eq!(imgs.len(), 2);
        assert_eq!((imgs[0].width, imgs[0].height), (2, 1));
        assert_eq!(imgs[0].pixels, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(imgs[1].pixels, vec![6, 7, 8, 9, 10, 11]);
    }
}
