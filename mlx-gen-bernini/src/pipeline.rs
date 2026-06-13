//! The native Bernini renderer provider (sc-4706): loads the converted dual-expert snapshot
//! (sc-4705) + the stock Wan2.2 UMT5/VAE/tokenizer, and runs the boundary-switched, APG-guided
//! denoise in **spatial latent space**, decoding to an image (1 frame) or video.
//!
//! Mirrors `mlx_gen_wan::model::Wan14b`'s staging (UMT5 → experts → VAE) to bound peak memory, with
//! the dual-expert CFG `denoise_moe` replaced by [`denoise_bernini`] (per-step
//! [`crate::forward::guided_velocity`] over the resolved [`Mode`]).

use std::path::PathBuf;

use mlx_rs::transforms::eval;
use mlx_rs::{random, Array};

use mlx_gen::gen_core;
use mlx_gen::tiling::TilingConfig;
use mlx_gen::weights::Weights;
use mlx_gen::{
    Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, Modality, ModelDescriptor, Progress, Quant, Result, WeightsSource,
};
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::pipeline::{align_dim, decode_to_frames, frames_to_images, latent_shape};
use mlx_gen_wan::scheduler::{make_scheduler, SolverKind};
use mlx_gen_wan::text_encoder::{load_tokenizer, Umt5Encoder};
use mlx_gen_wan::{WanTransformer, WanVae};

use crate::config::{resolve_mode, BerniniKnobs, Defaults};
use crate::forward::{guided_velocity, num_momentum_buffers, GuidanceParams, Mode, PackedForward};
use crate::guidance::MomentumBuffer;
use crate::preprocess::{encode_image, encode_videoclip};

pub const MODEL_ID: &str = "bernini_renderer";

/// Stable identity + advertised capabilities for the Bernini renderer (Wan2.2-A14B dual-expert with
/// source-id rotary + token-packed conditioning + APG guidance; t2v/t2i/i2i/v2v/r2v/rv2v).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "bernini",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // Source media: a single Reference (i2i) / MultiReference (r2v refs) / VideoClip (v2v/rv2v
            // source video). Text-only modes (t2i/t2v) need no conditioning.
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
                ConditioningKind::VideoClip,
            ],
            // LoRA/quant are follow-ons (sc-5146); the renderer ships dense bf16.
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["unipc"],
            schedulers: Vec::new(),
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: true,
            requires_sigma_shift: false,
        },
    }
}

/// The loaded Bernini renderer: resolved Wan2.2 config + Bernini knobs + the snapshot dir. The heavy
/// components are staged inside [`generate`](BerniniRenderer::generate_impl).
pub struct BerniniRenderer {
    descriptor: ModelDescriptor,
    config: WanModelConfig,
    knobs: BerniniKnobs,
    root: PathBuf,
    quant: Option<Quant>,
}

/// Load the Bernini renderer from a converted MLX snapshot directory
/// ([`mlx_gen_wan::convert::assemble_bernini_renderer_snapshot`] output: `low/high_noise_model` +
/// `t5_encoder` + `vae` + `tokenizer.json` + `config.json` + `bernini_renderer.json`).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "bernini_renderer: expected a model directory (converted MLX snapshot), not a single file".into(),
            ))
        }
    };
    let config = WanModelConfig::from_model_dir(&root)?;
    if !config.dual_model {
        return Err(Error::Msg(format!(
            "bernini_renderer: config.json is not a dual-expert model (model_type={}); expected the \
             assembled Bernini renderer snapshot",
            config.model_type
        )));
    }
    let knobs = BerniniKnobs::from_dir(&root);
    Ok(Box::new(BerniniRenderer {
        descriptor: descriptor(),
        config,
        knobs,
        root,
        quant: spec.quantize,
    }))
}

/// Registry adapter: bridge the crate's rich [`Result`] into the registry's [`gen_core::Result`].
fn load_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load(spec).map_err(Into::into)
}

inventory::submit! {
    mlx_gen::ModelRegistration { descriptor, load: load_registered }
}

/// One expert (high or low) with its prepared per-expert cross-attention K/V for the cond / empty-neg
/// text contexts (text embedding is per-expert, so K/V is built per expert).
struct BExpert<'a> {
    transformer: &'a WanTransformer,
    cross_kv_cond: Vec<(Array, Array)>,
    cross_kv_uncond: Vec<(Array, Array)>,
}

impl<'a> BExpert<'a> {
    fn build(dit: &'a WanTransformer, context: &Array, context_null: &Array) -> Result<Self> {
        let cc = dit.embed_text(context)?;
        let cu = dit.embed_text(context_null)?;
        Ok(Self {
            transformer: dit,
            cross_kv_cond: dit.prepare_cross_kv(&cc)?,
            cross_kv_uncond: dit.prepare_cross_kv(&cu)?,
        })
    }
}

/// The boundary-switched, APG-guided denoise loop (the Bernini analog of
/// `mlx_gen_wan::pipeline::denoise_moe`). Runs in **spatial latent space** `[16, T, H8, W8]`: each
/// step picks the high-noise expert while `t ≥ boundary` and the low-noise expert below it, multiplies
/// all omegas by `omega_scale` once on the first low-noise step, computes the per-mode guided velocity
/// (sigma = this step's flow sigma, for the APG x-conversion), and applies the UniPC flow step.
#[allow(clippy::too_many_arguments)]
fn denoise_bernini(
    pf: &PackedForward,
    mode: Mode,
    low: &BExpert,
    high: &BExpert,
    boundary: f32,
    num_train: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    videos: &[Array],
    images: &[Array],
    base_g: &GuidanceParams,
    omega_scale: f32,
    momentum: f32,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let mut sched = make_scheduler(SolverKind::UniPC, num_train);
    sched.set_timesteps(steps, shift);
    let timesteps = sched.timesteps().to_vec();
    let sigmas = sched.sigmas().to_vec();

    let mut latent = init_noise.clone();
    let mut switched = false;
    let mut g = base_g.clone();
    let mut mbufs: Vec<MomentumBuffer> = (0..num_momentum_buffers(mode))
        .map(|_| MomentumBuffer::new(momentum))
        .collect();

    for (i, &t) in timesteps.iter().enumerate() {
        on_step(i);
        let expert = if t >= boundary {
            high
        } else {
            if !switched {
                switched = true;
                g.omega_vid *= omega_scale;
                g.omega_img *= omega_scale;
                g.omega_txt *= omega_scale;
            }
            low
        };
        let sigma = sigmas[i];
        let v = guided_velocity(
            pf,
            mode,
            expert.transformer,
            &latent,
            videos,
            images,
            t,
            sigma,
            &expert.cross_kv_cond,
            &expert.cross_kv_uncond,
            &g,
            &mut mbufs,
        )?;
        latent = sched.step(&v, &latent)?;
        eval([&latent])?;
    }
    Ok(latent)
}

impl Generator for BerniniRenderer {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.validate_impl(req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

impl BerniniRenderer {
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        if let Some(frames) = req.frames {
            if frames % 4 != 1 {
                return Err(Error::Msg(format!(
                    "bernini_renderer: num_frames must be 1 + 4·k (got {frames})"
                )));
            }
        }
        Ok(())
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let cfg = &self.config;
        let k = &self.knobs;

        // --- Resolve geometry + knobs ---
        let frames = req
            .frames
            .map(|f| f as usize)
            .unwrap_or(Defaults::NUM_FRAMES)
            .max(1);
        let width = align_dim(req.width, cfg.patch_size.2, cfg.vae_stride.2);
        let height = align_dim(req.height, cfg.patch_size.1, cfg.vae_stride.1);
        let steps = req.steps.map(|s| s as usize).unwrap_or(Defaults::STEPS);
        let seed = req.seed.unwrap_or(42);
        let neg = req.negative_prompt.clone().unwrap_or_default();

        // Conditioning split: VideoClip → videos, Reference/MultiReference → images.
        let has_video = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::VideoClip { .. }));
        let has_image = req.conditioning.iter().any(|c| {
            matches!(
                c,
                Conditioning::Reference { .. } | Conditioning::MultiReference { .. }
            )
        });
        let mode = resolve_mode(req.video_mode.as_deref(), has_video, has_image);

        let omega_txt = req.guidance.unwrap_or(Defaults::OMEGA_TXT);
        let base_g = GuidanceParams {
            omega_vid: Defaults::OMEGA_VID,
            omega_img: Defaults::OMEGA_IMG,
            omega_txt,
            eta: Defaults::ETA,
            norm_threshold: [Defaults::NORM_THRESHOLD, Defaults::NORM_THRESHOLD],
        };

        let lat = latent_shape(frames, height, width, cfg.vae_z_dim, cfg.vae_stride);

        // --- Stage 1: UMT5 text encode (loaded → used → freed) ---
        let tokenizer = load_tokenizer(self.root.join("tokenizer.json"), cfg.text_len)?;
        let (context, context_null) = {
            let w = Weights::from_file(self.root.join("t5_encoder.safetensors"))?;
            let enc = Umt5Encoder::from_weights(&w, cfg)?;
            let context = enc.encode(&tokenizer, &req.prompt)?;
            let context_null = enc.encode(&tokenizer, &neg)?;
            eval([&context, &context_null])?;
            (context, context_null)
        };

        // --- Stage 1b: VAE-encode source media → conditioning latents (→ encoder freed) ---
        let (videos, images) = if has_video || has_image {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = WanVae::from_weights(&w)?;
            let mut videos = Vec::new();
            let mut images = Vec::new();
            for c in &req.conditioning {
                match c {
                    Conditioning::VideoClip { frames, .. } => {
                        videos.push(encode_videoclip(&vae, frames, width, height)?)
                    }
                    Conditioning::Reference { image, .. } => {
                        images.push(encode_image(&vae, image, width, height)?)
                    }
                    Conditioning::MultiReference { images: imgs } => {
                        for im in imgs {
                            images.push(encode_image(&vae, im, width, height)?);
                        }
                    }
                    _ => {}
                }
            }
            let all: Vec<&Array> = videos.iter().chain(images.iter()).collect();
            if !all.is_empty() {
                eval(all)?;
            }
            (videos, images)
        } else {
            (Vec::new(), Vec::new())
        };

        // Seeded init noise (spatial, f32). Bit-parity vs torch needs the reference's CPU-MT19937
        // draw injected; the coherence bar uses the MLX RNG.
        let key = random::key(seed)?;
        let init_noise = random::normal::<f32>(&lat[..], None, None, Some(&key))?;

        // --- Stage 2: load both experts, APG denoise (→ experts freed) ---
        let latents = {
            let low_w = Weights::from_file(self.root.join("low_noise_model.safetensors"))?;
            let high_w = Weights::from_file(self.root.join("high_noise_model.safetensors"))?;
            let mut low_dit = WanTransformer::from_weights(&low_w, cfg)?;
            let mut high_dit = WanTransformer::from_weights(&high_w, cfg)?;
            if let Some(q) = self.quant {
                low_dit.quantize(q.bits(), None)?;
                high_dit.quantize(q.bits(), None)?;
            }
            let low = BExpert::build(&low_dit, &context, &context_null)?;
            let high = BExpert::build(&high_dit, &context, &context_null)?;
            let pf = PackedForward::new(
                cfg.dim / cfg.num_heads,
                cfg.out_dim,
                cfg.patch_size,
                k.max_trained_src_id,
                k.interpolate_src_id,
            );
            let boundary = k.switch_dit_boundary * cfg.num_train_timesteps as f32;
            let total = steps as u32;
            let mut on_step = |i: usize| {
                on_progress(Progress::Step {
                    current: i as u32,
                    total,
                })
            };
            denoise_bernini(
                &pf,
                mode,
                &low,
                &high,
                boundary,
                cfg.num_train_timesteps,
                steps,
                k.shift,
                &init_noise,
                &videos,
                &images,
                &base_g,
                Defaults::OMEGA_SCALE,
                Defaults::MOMENTUM,
                &mut on_step,
            )?
        };

        // --- Stage 3: z16 VAE decode → RGB8 frames ---
        on_progress(Progress::Decoding);
        let out_frames = lat[1] * cfg.vae_stride.0 as i32;
        let tiling = TilingConfig::auto(height as i32, width as i32, out_frames);
        let frames_u8 = {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = WanVae::from_weights(&w)?;
            decode_to_frames(&vae, &latents, tiling.as_ref())?
        };
        let images_out = frames_to_images(&frames_u8)?;

        // num_frames == 1 ⇒ a still image (t2i/i2i). A single latent frame (T_lat = 1) still decodes
        // to one VAE temporal chunk (the causal decode emits `vae_stride_t` near-identical frames);
        // the still image is the first of them, matching the reference's single-frame PNG.
        if frames == 1 {
            let first = images_out.into_iter().next().ok_or_else(|| {
                Error::Msg("bernini_renderer: VAE decode produced no frames".into())
            })?;
            Ok(GenerationOutput::Images(vec![first]))
        } else {
            let fps = req.fps.unwrap_or(16);
            Ok(GenerationOutput::Video {
                frames: images_out,
                fps,
                audio: None,
            })
        }
    }
}
