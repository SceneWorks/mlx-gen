//! sc-5145: the **full Bernini** Generator (`mlx_gen::load("bernini")`) — the registered pipeline that
//! finally strings the whole planner → renderer stack together, mirroring `BerniniPipeline.__call__`
//! (`pipeline.py` 887-1181).
//!
//! ```text
//!   preprocess (ViT + VAE on the sources, gen-target ViT grid)
//!     → 3 planner streams (cond / uncond / imgcond)            [build_stream]
//!     → MAR semantic-planning loop                             [crate::mar::sample_vit_embed]
//!     → 4 renderer prompt streams + T5 concat_with_zero_init   [crate::assembly]
//!     → ViT-conditioned dual-expert APG denoise                [crate::pipeline::denoise_bernini_wvitcfg]
//!     → z16 VAE decode → image (1 frame) / video
//! ```
//!
//! The planner ([`BerniniPlanner`]) is Qwen2.5-VL-7B (penultimate extractor) + `MLPConnector` +
//! `DiffLoss_FM` clip-diff head + the MAR mask token; the renderer is the existing
//! [`crate::pipeline`] dual-expert MoE + UMT5 + z16 VAE. Both load from one full-snapshot dir
//! ([`crate::convert::assemble_bernini_snapshot`]).
//!
//! **Three conditioning streams.** The reference runs its data pipeline three times with different
//! dropout (`bernini_process_sample`): `cond` keeps everything; `imgcond` drops the input visuals but
//! keeps the text; `uncond` drops the input visuals and swaps the text for the (qwen) negative prompt.
//! We realise that by building each stream's conversation with the matching visuals/text present — the
//! gen-target slot is always there, and `post_process_input_embeds` masks it before the loop.
//!
//! **Scope notes (surfaced, not silently narrowed).** Image sources (t2i / i2i / r2v / t2v) are
//! faithful end-to-end. Video sources (v2v / mv2v / ads2v / rv2v) reuse the renderer's first-approx
//! resize and assume the worker supplies frames at `target_fps` (16) for the ViT/VAE frame sampling
//! (`smart_video_nframes`) — the same simplification the renderer provider already documents. The
//! gen-target ViT grid (which sizes `n_query`, the MAR token count) is computed analytically from the
//! output geometry rather than by running the HF processor on a fake clip; `n_query` only has to be
//! self-consistent (≤ `num_mask_token`), it does not feed the renderer's latent geometry (that comes
//! from the output H/W/frames). The real-weight e2e is a coherence smoke (the established bar — full
//! trajectory pixel parity is cross-backend-chaos-limited; components/early-step are torch-validated in
//! the per-module parity suites).

use std::path::{Path, PathBuf};

use image::RgbImage;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use mlx_gen::media::Image;
use mlx_gen::tiling::TilingConfig;
use mlx_gen::weights::Weights;
use mlx_gen::{
    Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, Modality, ModelDescriptor, Progress, Quant, Result, WeightsSource,
};

use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::pipeline::{align_dim, decode_to_frames, frames_to_images, latent_shape};
use mlx_gen_wan::text_encoder::{load_tokenizer, Umt5Encoder};
use mlx_gen_wan::{WanTransformer, WanVae};

use crate::assembly::{concat_with_zero_init, format_mllm_inputs_embeds};
use crate::clip_diff::DiffLossFm;
use crate::config::BerniniKnobs;
use crate::connector::MlpConnector;
use crate::forward::{PackedForward, VitGuidanceParams, VitMode};
use crate::mar::{
    mar_schedule, post_process_input_embeds, sample_vit_embed, SampledStreams, StreamState, VitCfg,
};
use crate::pipeline::{denoise_bernini_wvitcfg, BVitExpert};
use crate::process::{
    build_attention_mask_4d, generate_unified_inputs, mrope_position_ids, MRopeConfig,
};
use crate::qwen2_5_vl::{Qwen25VlText, QwenVlTextConfig};
use crate::rope::assign_source_ids;
use crate::template::BerniniTemplate;
use crate::vae_features::{image_vae_latent, video_vae_latent};
use crate::vae_preprocess::{vae_transform_image, VAE_MAX_SIZE, VAE_MIN_SIZE, VAE_STRIDE};
use crate::vision::{VisionConfig, VisionTower};
use crate::vit_preprocess::{
    normalized_frame, pack_patches, preprocess_image, smart_resize, smart_video_nframes, FACTOR,
    IMAGE_MEAN, IMAGE_STD, MERGE_SIZE, PATCH_SIZE, TEMPORAL_PATCH_SIZE,
};

pub const MODEL_ID: &str = "bernini";

/// Full-pipeline CLI defaults (`bernini/cli.py add_common_args` for the `BerniniPipeline` path). A
/// request's `guidance` overrides `omega_txt`; the rest are fixed until the worker surfaces them.
struct FullDefaults;
impl FullDefaults {
    const STEPS: usize = 40;
    const NUM_FRAMES: usize = 81;
    const OMEGA_VID: f32 = 1.25;
    const OMEGA_IMG: f32 = 4.5;
    const OMEGA_TXT: f32 = 4.0;
    const OMEGA_TGT: f32 = 0.5;
    const OMEGA_SCALE: f32 = 0.8;
    const PLANNING_STEP: usize = 25;
    const VIT_TXT_CFG: f32 = 1.2;
    const VIT_IMG_CFG: f32 = 1.0;
    const VIT_DENOISING_STEP: usize = 5;
    const FLOW_SHIFT: f32 = 5.0;
    /// `sample_one_step`'s `v2v_apg` hardcodes eta 1.0 / norm_threshold[0].
    const ETA: f32 = 1.0;
    const NORM_THRESHOLD: f32 = 50.0;
    /// Source-media ViT pixel budget (`preprocess_inputs` `vit_min/max_pixels`).
    const VIT_MIN_PIXELS: i64 = 3136;
    const VIT_MAX_PIXELS: i64 = 50176;
    const FPS: u32 = 16;
}

/// Planner knobs read from the `bernini_planner.json` sidecar (else the package-config defaults).
struct PlannerKnobs {
    max_sequence_length: i32,
    num_mask_token: i32,
    clip_diff_depth: usize,
    clip_diff_in_channels: i32,
    clip_diff_shift: f32,
}

impl PlannerKnobs {
    fn from_dir(root: &Path) -> Self {
        let v: serde_json::Value =
            std::fs::read(root.join(crate::convert::BERNINI_PLANNER_SIDECAR))
                .ok()
                .and_then(|b| serde_json::from_slice(&b).ok())
                .unwrap_or(serde_json::Value::Null);
        let i = |k: &str, d: i64| v.get(k).and_then(serde_json::Value::as_i64).unwrap_or(d);
        let cd = v
            .get("clip_diff_cfg")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let cdi = |k: &str, d: i64| cd.get(k).and_then(serde_json::Value::as_i64).unwrap_or(d);
        let cdf = |k: &str, d: f64| cd.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d);
        Self {
            max_sequence_length: i("max_sequence_length", 512) as i32,
            num_mask_token: i("num_mask_token", 4096) as i32,
            // The `vit_decoder` (`SimpleMLPAdaLN`) depth is fixed at 16 in the released checkpoint; the
            // sidecar carries z_channels / shift verbatim.
            clip_diff_depth: 16,
            clip_diff_in_channels: cdi("z_channels", 3584) as i32,
            clip_diff_shift: cdf("shift", 2.0) as f32,
        }
    }
}

/// Read the planner's MRoPE / token-id config from the Qwen2.5-VL `qwen2_5_vl_config.json`.
fn read_mrope_config(path: &Path) -> MRopeConfig {
    let d = MRopeConfig::default();
    let v: serde_json::Value = std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(serde_json::Value::Null);
    // tokens_per_second / spatial_merge_size live in vision_config on Qwen2.5-VL.
    let vc = v.get("vision_config").unwrap_or(&v);
    let i = |o: &serde_json::Value, k: &str, dv: i64| {
        o.get(k).and_then(serde_json::Value::as_i64).unwrap_or(dv)
    };
    let f = |o: &serde_json::Value, k: &str, dv: f64| {
        o.get(k).and_then(serde_json::Value::as_f64).unwrap_or(dv)
    };
    MRopeConfig {
        spatial_merge_size: i(vc, "spatial_merge_size", d.spatial_merge_size),
        tokens_per_second: f(vc, "tokens_per_second", d.tokens_per_second),
        image_token_id: i(&v, "image_token_id", d.image_token_id),
        video_token_id: i(&v, "video_token_id", d.video_token_id),
        vision_start_token_id: i(&v, "vision_start_token_id", d.vision_start_token_id),
    }
}

/// The loaded Bernini semantic planner: Qwen2.5-VL backbone + vision tower + connector + clip-diff
/// head + MAR mask token + the host-side templating/MRoPE config.
struct BerniniPlanner {
    backbone: Qwen25VlText,
    vision: VisionTower,
    connector: MlpConnector,
    clip_diff: DiffLossFm,
    /// A single MAR mask token `[1, 1, 3584]` (`self.mask_tokens[:, :1]`, broadcast over the target).
    mask_token: Array,
    mrope: MRopeConfig,
    template: BerniniTemplate,
    knobs: PlannerKnobs,
}

impl BerniniPlanner {
    fn load(root: &Path, quant: Option<Quant>) -> Result<Self> {
        let cfg_path = root.join("qwen2_5_vl_config.json");
        let qcfg = QwenVlTextConfig::from_config_json(&cfg_path)?;
        let vcfg = VisionConfig::from_config_json(&cfg_path)?;

        let qw = Weights::from_file(root.join("qwen2_5_vl.safetensors"))?;
        let mut backbone = Qwen25VlText::from_weights(&qw, qcfg, "model")?;
        let vision = VisionTower::from_weights(&qw, vcfg, "visual")?;

        let cw = Weights::from_file(root.join("connector.safetensors"))?;
        // Kept dense (see the quant policy below) — small + the clip_diff runs the MAR flow loop.
        let connector = MlpConnector::from_weights(&cw, "")?;

        let knobs = PlannerKnobs::from_dir(root);
        let vw = Weights::from_file(root.join("vit_decoder.safetensors"))?;
        let clip_diff = DiffLossFm::from_weights(
            &vw,
            "net",
            knobs.clip_diff_depth,
            knobs.clip_diff_in_channels,
            knobs.clip_diff_shift,
        )?;

        let mw = Weights::from_file(root.join("mask_tokens.safetensors"))?;
        // `self.mask_tokens[:, :1]` — a single mask token, broadcast over the n_query target slots.
        let mask_token = mw
            .require("mask_tokens")?
            .take_axis(Array::from_slice(&[0i32], &[1]), 1)?;

        // sc-5146 conservative quant policy: quantize the Qwen2.5-VL **LLM** linears — the planner
        // footprint that matters (~7B params, ~14GB bf16; all dims divisible by the group-64 quant
        // size). Everything else on the planner side stays DENSE *where quant is unsafe or not worth
        // it*: the **vision tower** (~0.6B) has group-64-misaligned linears (MLP intermediate 3420,
        // folded patch-embed 1176) so it cannot be affine-quantized at group 64; the **connector**
        // (~0.2GB) and **clip_diff** flow head (~1.6GB) are small and clip_diff runs ~75× through the
        // MAR planning loop with triple-CFG where 4-bit error would compound. The two heavy renderer
        // experts carry the dominant footprint (~56GB → ~28GB Q8 / ~14GB Q4) and are quantized
        // separately. `quantize` eval-frees the bf16 transient at load (sensenova/lens pattern).
        if let Some(q) = quant {
            backbone.quantize(q.bits())?;
        }

        let template =
            BerniniTemplate::from_tokenizer_file(root.join("mllm").join("tokenizer.json"))?;
        Ok(Self {
            backbone,
            vision,
            connector,
            clip_diff,
            mask_token,
            mrope: read_mrope_config(&cfg_path),
            template,
            knobs,
        })
    }
}

/// One preprocessed source visual: its ViT features (planner conditioning) + grid + VAE latent
/// (renderer conditioning).
struct SourceVisual {
    /// `[merged, 3584]` planner ViT features.
    vit_feat: Array,
    /// `[t, h, w]` ViT grid (drives the token count + MRoPE).
    vit_grid: [i32; 3],
    /// `[16, T, H8, W8]` normalized VAE latent (the renderer source-conditioning latent).
    vae_latent: Array,
    /// Original source `(height, width)` (cosmetic, for the conversation's `image` message).
    hw: (i64, i64),
}

/// Convert a public RGB8 [`Image`] to an `image::RgbImage`.
fn to_rgb(img: &Image) -> Result<RgbImage> {
    RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .ok_or_else(|| Error::Msg("bernini: malformed RGB8 conditioning image".into()))
}

/// ViT-encode one image → `[merged, 3584]` + grid.
fn vit_encode_image(planner: &BerniniPlanner, rgb: &RgbImage) -> Result<(Array, [i32; 3])> {
    let (pixels, grid) = preprocess_image(
        rgb,
        FullDefaults::VIT_MIN_PIXELS,
        FullDefaults::VIT_MAX_PIXELS,
        IMAGE_MEAN,
        IMAGE_STD,
    )?;
    let feat = planner.vision.forward(&pixels, &[grid])?;
    Ok((feat, grid))
}

/// ViT-encode a stack of (already ViT-sampled) video frames → `[merged, 3584]` + grid. All frames are
/// `smart_resize`d to a common size (the HF video processor resizes the clip uniformly), normalized,
/// stacked `[F, 3, H, W]`, then `pack_patches` (temporal 2).
fn vit_encode_video(planner: &BerniniPlanner, frames: &[RgbImage]) -> Result<(Array, [i32; 3])> {
    let (h0, w0) = (frames[0].height() as i64, frames[0].width() as i64);
    let (rh, rw) = smart_resize(
        h0,
        w0,
        FACTOR,
        FullDefaults::VIT_MIN_PIXELS,
        FullDefaults::VIT_MAX_PIXELS,
    );
    let mut chw_t = Vec::with_capacity(frames.len());
    for f in frames {
        let resized = image::imageops::resize(
            f,
            rw as u32,
            rh as u32,
            image::imageops::FilterType::CatmullRom,
        );
        chw_t.push(normalized_frame(
            resized.as_raw(),
            rh,
            rw,
            IMAGE_MEAN,
            IMAGE_STD,
        ));
    }
    let refs: Vec<&Array> = chw_t.iter().collect();
    let frames_t = concatenate_axis(&refs, 0)?; // [F, 3, H, W]
    let (pixels, grid) = pack_patches(&frames_t, PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE)?;
    let feat = planner.vision.forward(&pixels, &[grid])?;
    Ok((feat, grid))
}

/// VAE-encode one image (`.mode()`, the Gaussian mean) → normalized `[16, T, H8, W8]`.
fn vae_encode_image(vae: &WanVae, rgb: &RgbImage) -> Result<Array> {
    let chw = vae_transform_image(rgb, VAE_MAX_SIZE, VAE_MIN_SIZE, VAE_STRIDE); // [3, H, W] in [-1,1]
    drop_batch(&image_vae_latent(vae, &chw)?)
}

/// VAE-encode a video clip (`.sample()`) → normalized `[16, T_lat, H8, W8]`. `eps` is generated for the
/// latent shape so the encode is deterministic given the seed.
fn vae_encode_video(vae: &WanVae, frames: &[RgbImage], z_dim: usize, key: &Array) -> Result<Array> {
    let mut chw_t = Vec::with_capacity(frames.len());
    for f in frames {
        let chw = vae_transform_image(f, VAE_MAX_SIZE, VAE_MIN_SIZE, VAE_STRIDE); // [3, H, W]
        chw_t.push(chw.expand_dims(1)?); // [3, 1, H, W]
    }
    let refs: Vec<&Array> = chw_t.iter().collect();
    let video = concatenate_axis(&refs, 1)?; // [3, T, H, W]
    let s = video.shape();
    let (t, h, w) = (s[1], s[2], s[3]);
    let t_lat = (t - 1) / 4 + 1; // z16 temporal stride 4
    let eps = random::normal::<f32>(
        &[1, z_dim as i32, t_lat, h / 8, w / 8],
        None,
        None,
        Some(key),
    )?;
    drop_batch(&video_vae_latent(vae, &video, &eps)?)
}

/// Drop the leading batch dim of `[1, z, T, H, W]` → `[z, T, H, W]` (what `PackedForward` expects).
fn drop_batch(z: &Array) -> Result<Array> {
    Ok(z.reshape(&z.shape()[1..])?)
}

/// The gen-target ViT grid `[t, h, w]` (sizes `n_query`, the MAR token count). For an image target
/// (`frames == 1`) `t = 1`; for a video target the ViT samples `vit_fps` (= fps/8) frames from the
/// `num_frames` clip (assumed at `target_fps`), `t = vit_frames / temporal`. The spatial grid is the
/// `smart_resize` of the output H/W under the ViT pixel budget.
fn gen_target_grid(height: i64, width: i64, frames: usize, fps: u32) -> [i32; 3] {
    let (rh, rw) = smart_resize(
        height,
        width,
        FACTOR,
        FullDefaults::VIT_MIN_PIXELS,
        FullDefaults::VIT_MAX_PIXELS,
    );
    let gh = (rh / PATCH_SIZE) as i32;
    let gw = (rw / PATCH_SIZE) as i32;
    let t = if frames <= 1 {
        1
    } else {
        let vit_fps = (fps / 8).max(1) as f64;
        let vit_frames = smart_video_nframes(
            frames as i64,
            fps as f64,
            vit_fps,
            Some(TEMPORAL_PATCH_SIZE),
            None,
            Some(frames as i64),
            false,
        )
        .len() as i64;
        (vit_frames / TEMPORAL_PATCH_SIZE).max(1) as i32
    };
    [t, gh, gw]
}

/// Merged-token count of a ViT grid (`t·h·w / merge²`).
fn grid_tokens(grid: [i32; 3]) -> i64 {
    let m2 = (MERGE_SIZE * MERGE_SIZE) as i32;
    (grid[0] * grid[1] * grid[2] / m2) as i64
}

/// Build one planner stream's [`StreamState`] (cond / uncond / imgcond). `images`/`videos` are the
/// **present** input source visuals (empty for uncond/imgcond); `prompt` is the stream's text (raw for
/// cond/imgcond, negative for uncond). `gen_grid` is the gen-target ViT grid; `gen_is_video` selects
/// whether the gen slot is an image or a video. The gen-target ViT features are zeros (they are masked
/// by [`post_process_input_embeds`] before the loop).
#[allow(clippy::too_many_arguments)]
fn build_stream(
    planner: &BerniniPlanner,
    task: &str,
    prompt: &str,
    images: &[SourceVisual],
    videos: &[SourceVisual],
    gen_grid: [i32; 3],
    gen_is_video: bool,
    out_h: i64,
    out_w: i64,
) -> Result<(StreamState, i32)> {
    // Conversation: videos first, then images, then the text prompt + the gen-target slot.
    let image_hw: Vec<(i64, i64)> = images.iter().map(|s| s.hw).collect();
    let output_t = if gen_is_video {
        // any T > 1 selects the video gen path in `generate_unified_inputs`.
        2
    } else {
        1
    };
    let conv = generate_unified_inputs(prompt, &image_hw, videos.len(), output_t, out_h, out_w);

    // Per-type ViT grids in conversation order (input visuals first, gen target last). `image_grids`
    // are the image-type visuals (input images + gen if image), `video_grids` the video-type ones.
    let mut image_grids: Vec<[i32; 3]> = images.iter().map(|s| s.vit_grid).collect();
    let mut video_grids: Vec<[i32; 3]> = videos.iter().map(|s| s.vit_grid).collect();
    if gen_is_video {
        video_grids.push(gen_grid);
    } else {
        image_grids.push(gen_grid);
    }
    let image_token_nums: Vec<i64> = image_grids.iter().map(|&g| grid_tokens(g)).collect();
    let video_token_nums: Vec<i64> = video_grids.iter().map(|&g| grid_tokens(g)).collect();

    let tout =
        planner
            .template
            .encode_messages(&conv, &image_token_nums, &video_token_nums, task)?;
    let l = tout.input_ids.len();

    // visual_embeds: conversation order = [video feats, image feats, gen-target zeros].
    let h_vit = planner.knobs.clip_diff_in_channels;
    let gen_tokens = grid_tokens(gen_grid) as i32;
    let mut feats: Vec<Array> = Vec::new();
    for v in videos {
        feats.push(v.vit_feat.clone());
    }
    for im in images {
        feats.push(im.vit_feat.clone());
    }
    feats.push(Array::zeros::<f32>(&[gen_tokens, h_vit])?);
    let feat_refs: Vec<&Array> = feats.iter().collect();
    // Unify to the snapshot dtype (bf16): the ViT-tower feats are bf16, the gen-target placeholder
    // is f32; a mixed concat would promote the whole planner path to f32 (f32 activations through
    // bf16 weights). The converter always emits bf16, so this is the backbone's embedding dtype.
    let visual_embeds = concatenate_axis(&feat_refs, 0)?.as_dtype(Dtype::Bfloat16)?;

    // MRoPE position ids (`[3, L]`) + the flex 4-D mask.
    let to_i64 = |g: &[[i32; 3]]| -> Vec<[i64; 3]> {
        g.iter()
            .map(|&[a, b, c]| [a as i64, b as i64, c as i64])
            .collect()
    };
    let pos = mrope_position_ids(
        &tout.input_ids,
        &to_i64(&image_grids),
        &to_i64(&video_grids),
        &planner.mrope,
    )?;
    let mask = build_attention_mask_4d(&tout.token_type, &tout.token_segment_ids)?;

    // Visual slot masks (token_type 2 = input-vit, 3 = gen-output).
    let vin: Vec<bool> = tout.token_type.iter().map(|&t| t == 2).collect();
    let vout: Vec<bool> = tout.token_type.iter().map(|&t| t == 3).collect();

    let ids_i32: Vec<i32> = tout.input_ids.iter().map(|&x| x as i32).collect();
    let embeds = format_mllm_inputs_embeds(
        &planner.backbone,
        &ids_i32,
        Some(&visual_embeds),
        &vin,
        &vout,
    )?;
    // Keep the planner's activation dtype consistent (the f32 mask-multiply in post_process would
    // otherwise upcast); match the backbone embed dtype + cast the additive mask to it.
    let dtype = embeds.dtype();
    let embeds = post_process_input_embeds(&embeds, &vout, &planner.mask_token)?.as_dtype(dtype)?;
    let mask = mask.as_dtype(dtype)?;

    let gen_idx: Vec<i32> = (0..l).filter(|&i| vout[i]).map(|i| i as i32).collect();
    let n_query = gen_idx.len() as i32;
    Ok((
        StreamState {
            input_embeds: embeds,
            position_ids: pos,
            mask,
            gen_idx,
        },
        n_query,
    ))
}

/// Resolve the full-Bernini [`VitMode`] from the request's `video_mode` (a guidance-mode name
/// preferred) plus the conditioning + output kind. Defaults: video source ⇒ `v2v_apg`; video+image or
/// image-refs→video ⇒ `rv2v_wapg`; otherwise (t2i/i2i/t2v) ⇒ `vae_txt_vit_wapg`.
fn resolve_vit_mode(
    video_mode: Option<&str>,
    has_video: bool,
    has_image: bool,
    out_video: bool,
) -> VitMode {
    if let Some(s) = video_mode {
        if let Some(m) = VitMode::from_name(s) {
            return m;
        }
        if let Some(m) = task_to_vit_mode(s) {
            return m;
        }
    }
    match (has_video, has_image, out_video) {
        (true, _, _) => {
            if has_image {
                VitMode::Rv2vWapg
            } else {
                VitMode::V2vApg
            }
        }
        (false, true, true) => VitMode::Rv2vWapg, // r2v: image refs → video
        _ => VitMode::VaeTxtVitWapg,              // t2i / i2i / t2v
    }
}

/// Upstream task_type → full-pipeline guidance mode (fallback when `video_mode` is a task name).
fn task_to_vit_mode(task: &str) -> Option<VitMode> {
    Some(match task {
        "t2i" | "t2v" | "i2i" => VitMode::VaeTxtVitWapg,
        "v2v" | "mv2v" | "ads2v" => VitMode::V2vApg,
        "r2v" | "rv2v" => VitMode::Rv2vWapg,
        _ => return None,
    })
}

/// Stable identity + advertised capabilities for the full Bernini pipeline.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "bernini",
        backend: "mlx",
        // Full Bernini covers both still-image (t2i/i2i) and video (t2v/v2v/r2v/rv2v) tasks.
        modality: Modality::Both,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
                ConditioningKind::VideoClip,
            ],
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["unipc"],
            schedulers: Vec::new(),
            supported_guidance_methods: vec![],
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

/// The loaded full Bernini pipeline: the snapshot dir + the resolved renderer config/knobs.
pub struct Bernini {
    descriptor: ModelDescriptor,
    config: WanModelConfig,
    knobs: BerniniKnobs,
    root: PathBuf,
    quant: Option<Quant>,
}

/// Load the full Bernini pipeline from a combined snapshot dir
/// ([`crate::convert::assemble_bernini_snapshot`]): planner components + dual-expert renderer DiTs +
/// UMT5 / z16 VAE / tokenizer + the knob sidecars.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "bernini: expected a model directory (converted full-Bernini snapshot), not a single file"
                    .into(),
            ))
        }
    };
    let config = WanModelConfig::from_model_dir(&root)?;
    if !config.dual_model {
        return Err(Error::Msg(format!(
            "bernini: config.json is not a dual-expert renderer (model_type={}); expected the \
             assembled full-Bernini snapshot",
            config.model_type
        )));
    }
    let knobs = BerniniKnobs::from_dir(&root);
    Ok(Box::new(Bernini {
        descriptor: descriptor(),
        config,
        knobs,
        root,
        quant: spec.quantize,
    }))
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`.
mlx_gen::register_generators! { descriptor => load }

mlx_gen::impl_generator!(Bernini {
    validate: |s, req| s.validate_impl(req),
    generate: generate_impl,
});

impl Bernini {
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        if let Some(frames) = req.frames {
            if frames % 4 != 1 {
                return Err(Error::Msg(format!(
                    "bernini: num_frames must be 1 + 4·k (got {frames})"
                )));
            }
        }
        validate_conditioning_video_clips(req)?;
        Ok(())
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let cfg = &self.config;
        let task = req.video_mode.as_deref().unwrap_or("");

        // --- Geometry + knobs ---
        let frames = req
            .frames
            .map(|f| f as usize)
            .unwrap_or(FullDefaults::NUM_FRAMES)
            .max(1);
        let out_video = frames > 1;
        let width = align_dim(req.width, cfg.patch_size.2, cfg.vae_stride.2);
        let height = align_dim(req.height, cfg.patch_size.1, cfg.vae_stride.1);
        let steps = req.steps.map(|s| s as usize).unwrap_or(FullDefaults::STEPS);
        let seed = req.seed.unwrap_or(42);
        let neg = req.negative_prompt.clone().unwrap_or_default();

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
        let mode = resolve_vit_mode(req.video_mode.as_deref(), has_video, has_image, out_video);

        // --- Stage 1: planner (loaded → 3 streams + MAR loop → freed) ---
        on_progress(Progress::Step {
            current: 0,
            total: steps as u32,
        });
        let planner = BerniniPlanner::load(&self.root, self.quant)?;

        // Preprocess the conditioning: videos first, then images (the conversation / source_id order).
        let mut videos: Vec<SourceVisual> = Vec::new();
        let mut images: Vec<SourceVisual> = Vec::new();
        let (videos_pix, images_pix) = collect_conditioning(req);
        {
            let vae_w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = WanVae::from_weights(&vae_w)?;
            for (vi, clip) in videos_pix.iter().enumerate() {
                let rgb: Vec<RgbImage> = clip.iter().map(to_rgb).collect::<Result<_>>()?;
                let vit_frames = sample_vit_frames(&rgb);
                let (vit_feat, vit_grid) = vit_encode_video(&planner, &vit_frames)?;
                let key = random::key(seed.wrapping_add(0x51d_u64).wrapping_add(vi as u64))?;
                let vae_latent = vae_encode_video(&vae, &rgb, cfg.vae_z_dim, &key)?;
                let hw = (rgb[0].height() as i64, rgb[0].width() as i64);
                videos.push(SourceVisual {
                    vit_feat,
                    vit_grid,
                    vae_latent,
                    hw,
                });
            }
            for img in &images_pix {
                let rgb = to_rgb(img)?;
                let (vit_feat, vit_grid) = vit_encode_image(&planner, &rgb)?;
                let vae_latent = vae_encode_image(&vae, &rgb)?;
                images.push(SourceVisual {
                    vit_feat,
                    vit_grid,
                    vae_latent,
                    hw: (img.height as i64, img.width as i64),
                });
            }
        }

        let gen_grid = gen_target_grid(
            height as i64,
            width as i64,
            frames,
            req.fps.unwrap_or(FullDefaults::FPS),
        );

        // Three streams: cond (full), imgcond (text, no input visuals), uncond (neg text, no visuals).
        let (cond, n_query) = build_stream(
            &planner,
            task,
            &req.prompt,
            &images,
            &videos,
            gen_grid,
            out_video,
            height as i64,
            width as i64,
        )?;
        let (imgcond, _) = build_stream(
            &planner,
            task,
            &req.prompt,
            &[],
            &[],
            gen_grid,
            out_video,
            height as i64,
            width as i64,
        )?;
        let (uncond, _) = build_stream(
            &planner,
            task,
            &neg,
            &[],
            &[],
            gen_grid,
            out_video,
            height as i64,
            width as i64,
        )?;

        // MAR planning loop (seeded reveal order + per-step FM noise, injectable for parity).
        let vit_cfg = VitCfg {
            planning_step: FullDefaults::PLANNING_STEP,
            vit_denoising_step: FullDefaults::VIT_DENOISING_STEP,
            vit_txt_cfg: FullDefaults::VIT_TXT_CFG,
            vit_img_cfg: FullDefaults::VIT_IMG_CFG,
        };
        if n_query > planner.knobs.num_mask_token {
            return Err(Error::Msg(format!(
                "bernini: gen-target needs {n_query} ViT tokens but the planner has only {} mask \
                 tokens — lower the resolution/frames",
                planner.knobs.num_mask_token
            )));
        }
        let max_seq = planner.knobs.max_sequence_length.max(1);
        let order = seeded_permutation(n_query, seed)?;
        let step_noise = seeded_step_noise(
            n_query,
            vit_cfg.planning_step,
            &order,
            planner.knobs.clip_diff_in_channels,
            seed,
        )?;
        let mut planner = planner;
        let streams: SampledStreams = sample_vit_embed(
            &planner.backbone,
            &planner.connector,
            &mut planner.clip_diff,
            &cond,
            &uncond,
            &imgcond,
            &vit_cfg,
            &order,
            &step_noise,
            &req.cancel,
            &planner.mask_token,
        )?;
        eval([
            &streams.wtxt_wvit,
            &streams.wtxt_wovit,
            &streams.wotxt_wvit,
            &streams.wotxt_wovit,
        ])?;

        // VAE-encoded source latents carry into the renderer; free the planner before the renderer.
        let bf16 = |a: &Array| a.as_dtype(Dtype::Bfloat16);
        let s_wtxt_wvit = bf16(&streams.wtxt_wvit)?;
        let s_wtxt_wovit = bf16(&streams.wtxt_wovit)?;
        let s_wotxt_wvit = bf16(&streams.wotxt_wvit)?;
        let s_wotxt_wovit = bf16(&streams.wotxt_wovit)?;
        let src_videos: Vec<Array> = videos.iter().map(|s| s.vae_latent.clone()).collect();
        let src_images: Vec<Array> = images.iter().map(|s| s.vae_latent.clone()).collect();
        drop(planner);

        // --- Stage 2: T5 prompt encode + concat_with_zero_init for the 4 renderer streams ---
        let tokenizer = load_tokenizer(self.root.join("tokenizer.json"), cfg.text_len)?;
        let (t5_pos, t5_neg) = {
            let w = Weights::from_file(self.root.join("t5_encoder.safetensors"))?;
            let enc = Umt5Encoder::from_weights(&w, cfg)?;
            // `encode` returns `[T, 4096]` (no batch); the planner streams are `[1, S, 4096]`, so add
            // the batch axis for the `concat_with_zero_init` sequence-axis concat.
            let pos = enc
                .encode(&tokenizer, &req.prompt)?
                .expand_dims(0)?
                .as_dtype(Dtype::Bfloat16)?;
            let neg = enc
                .encode(&tokenizer, &neg)?
                .expand_dims(0)?
                .as_dtype(Dtype::Bfloat16)?;
            eval([&pos, &neg])?;
            (pos, neg)
        };
        // wtxt streams prepend the positive T5; wotxt streams prepend the negative T5. The renderer's
        // `embed_text` consumes a 2-D `[S, text_dim]` context (it pads/reshapes the batch axis itself),
        // so drop the leading batch dim that `concat_with_zero_init` carries.
        let drop0 = |a: Array| -> Result<Array> {
            let s = a.shape().to_vec();
            Ok(a.reshape(&[s[1], s[2]])?)
        };
        let pe_wtxt_wvit = drop0(concat_with_zero_init(&t5_pos, &s_wtxt_wvit, max_seq)?)?;
        let pe_wtxt_wovit = drop0(concat_with_zero_init(&t5_pos, &s_wtxt_wovit, max_seq)?)?;
        let pe_wotxt_wvit = drop0(concat_with_zero_init(&t5_neg, &s_wotxt_wvit, max_seq)?)?;
        let pe_wotxt_wovit = drop0(concat_with_zero_init(&t5_neg, &s_wotxt_wovit, max_seq)?)?;

        // --- Stage 3: load both experts, ViT-conditioned APG denoise ---
        let key = random::key(seed)?;
        let lat = latent_shape(frames, height, width, cfg.vae_z_dim, cfg.vae_stride)?;
        let init_noise = random::normal::<f32>(&lat[..], None, None, Some(&key))?;

        let base_g = VitGuidanceParams {
            omega_txt: req.guidance.unwrap_or(FullDefaults::OMEGA_TXT),
            omega_img: FullDefaults::OMEGA_IMG,
            omega_vid: FullDefaults::OMEGA_VID,
            omega_tgt: FullDefaults::OMEGA_TGT,
            eta: FullDefaults::ETA,
            norm_threshold: FullDefaults::NORM_THRESHOLD,
        };

        // Source ids (videos first, then images — mirrors `PackedForward::build_combos`/packing_vae).
        let (nv, ni) = (src_videos.len(), src_images.len());
        let sids = assign_source_ids(
            nv + ni,
            self.knobs.max_trained_src_id,
            self.knobs.interpolate_src_id,
        );
        let video_srcs: Vec<(Array, f64)> = src_videos
            .iter()
            .enumerate()
            .map(|(k, v)| (v.clone(), sids[k]))
            .collect();
        let image_srcs: Vec<(Array, f64)> = src_images
            .iter()
            .enumerate()
            .map(|(j, im)| (im.clone(), sids[nv + j]))
            .collect();

        // Load each expert and (if quantizing) quantize-then-free it before loading the next, so only
        // one expert's bf16 transient is resident at a time (sc-5360 — `WanTransformer::quantize`
        // eval-frees the bf16 dequant). Without quant this just loads both bf16.
        let load_expert = |name: &str| -> Result<WanTransformer> {
            let w = Weights::from_file(self.root.join(name))?;
            let mut dit = WanTransformer::from_weights(&w, cfg)?;
            if let Some(q) = self.quant {
                dit.quantize(q.bits(), None)?;
            }
            Ok(dit)
        };
        let latents = {
            let low_dit = load_expert("low_noise_model.safetensors")?;
            let high_dit = load_expert("high_noise_model.safetensors")?;
            let streams4 = [
                &pe_wtxt_wvit,
                &pe_wtxt_wovit,
                &pe_wotxt_wvit,
                &pe_wotxt_wovit,
            ];
            let low = BVitExpert::build(&low_dit, streams4)?;
            let high = BVitExpert::build(&high_dit, streams4)?;
            let pf = PackedForward::new(
                cfg.dim / cfg.num_heads,
                cfg.out_dim,
                cfg.patch_size,
                self.knobs.max_trained_src_id,
                self.knobs.interpolate_src_id,
            );
            let boundary = self.knobs.switch_dit_boundary * cfg.num_train_timesteps as f32;
            let total = steps as u32;
            let mut on_step = |i: usize| {
                on_progress(Progress::Step {
                    current: i as u32,
                    total,
                })
            };
            denoise_bernini_wvitcfg(
                &pf,
                mode,
                &low,
                &high,
                boundary,
                cfg.num_train_timesteps,
                steps,
                FullDefaults::FLOW_SHIFT,
                &init_noise,
                &image_srcs,
                &video_srcs,
                &base_g,
                FullDefaults::OMEGA_SCALE,
                &req.cancel,
                &mut on_step,
            )?
        };

        // --- Stage 4: z16 VAE decode → image / video ---
        on_progress(Progress::Decoding);
        let out_frames = lat[1] * cfg.vae_stride.0 as i32;
        let tiling = TilingConfig::auto(height as i32, width as i32, out_frames);
        let frames_u8 = {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = WanVae::from_weights(&w)?;
            decode_to_frames(&vae, &latents, tiling.as_ref())?
        };
        let images_out = frames_to_images(&frames_u8)?;

        if frames == 1 {
            let first = images_out
                .into_iter()
                .next()
                .ok_or_else(|| Error::Msg("bernini: VAE decode produced no frames".into()))?;
            Ok(GenerationOutput::Images(vec![first]))
        } else {
            let fps = req.fps.unwrap_or(FullDefaults::FPS);
            Ok(GenerationOutput::Video {
                frames: images_out,
                fps,
                audio: None,
            })
        }
    }
}

/// Reject empty / non-`1+4k` conditioning video clips before the full pipeline dereferences
/// `frames[0]` (F-022). The full path does `rgb[0].height()` (generate_impl) and the MAR sampler
/// reads `frames[0]`, and the WanVae temporally packs the clip (`T = 1 + 4·k`), so an empty or
/// mis-counted clip from a malformed worker payload would panic / shape-mismatch deep inside
/// generate_impl instead of erroring cleanly. Mirrors the renderer's `encode_videoclip` guard
/// (preprocess.rs:35-47). Free fn so it is unit-testable without loading the full Bernini weights.
fn validate_conditioning_video_clips(req: &GenerationRequest) -> Result<()> {
    for c in &req.conditioning {
        if let Conditioning::VideoClip { frames, .. } = c {
            if frames.is_empty() {
                return Err(Error::Msg("bernini: empty conditioning video clip".into()));
            }
            if frames.len() % 4 != 1 {
                return Err(Error::Msg(format!(
                    "bernini: conditioning video-clip frame count must be 1 + 4·k (got {})",
                    frames.len()
                )));
            }
        }
    }
    Ok(())
}

/// Collect the conditioning into video clips + reference images, preserving order (videos then images).
fn collect_conditioning(req: &GenerationRequest) -> (Vec<Vec<Image>>, Vec<Image>) {
    let mut videos: Vec<Vec<Image>> = Vec::new();
    let mut images: Vec<Image> = Vec::new();
    for c in &req.conditioning {
        match c {
            Conditioning::VideoClip { frames, .. } => videos.push(frames.clone()),
            Conditioning::Reference { image, .. } => images.push(image.clone()),
            Conditioning::MultiReference { images: imgs } => images.extend(imgs.clone()),
            _ => {}
        }
    }
    (videos, images)
}

/// Sub-sample a decoded clip to the ViT frame set (`smart_video_nframes`, assuming `target_fps`).
fn sample_vit_frames(frames: &[RgbImage]) -> Vec<RgbImage> {
    let fps = FullDefaults::FPS as f64;
    let vit_fps = (FullDefaults::FPS / 8).max(1) as f64;
    let idx = smart_video_nframes(
        frames.len() as i64,
        fps,
        vit_fps,
        Some(TEMPORAL_PATCH_SIZE),
        None,
        Some(frames.len() as i64),
        false,
    );
    idx.iter()
        .map(|&i| frames[(i as usize).min(frames.len() - 1)].clone())
        .collect()
}

/// A deterministic reveal permutation of `[0, n)` from the seed (argsort of seeded normal noise on the
/// host — bit-stable, and injectable in tests for torch parity).
fn seeded_permutation(n: i32, seed: u64) -> Result<Vec<i32>> {
    let key = random::key(seed.wrapping_add(0x4d_a4))?;
    let noise = random::normal::<f32>(&[n], None, None, Some(&key))?;
    let vals = noise.as_slice::<f32>().to_vec();
    let mut idx: Vec<i32> = (0..n).collect();
    idx.sort_by(|&a, &b| vals[a as usize].partial_cmp(&vals[b as usize]).unwrap());
    Ok(idx)
}

/// Per-step base FM noise for the MAR loop — one `[revealed, in]` tensor per planning step (the
/// reference's `torch.randn(n_revealed, in)`, tiled ×3 inside `DiffLossFm::sample`).
fn seeded_step_noise(
    n_query: i32,
    planning_step: usize,
    order: &[i32],
    in_channels: i32,
    seed: u64,
) -> Result<Vec<Array>> {
    let schedule = mar_schedule(n_query, planning_step, order);
    let mut out = Vec::with_capacity(planning_step);
    for (s, revealed) in schedule.iter().enumerate() {
        let np = (revealed.len() as i32).max(1);
        let key = random::key(seed.wrapping_add(0x9e_37).wrapping_add(s as u64))?;
        out.push(random::normal::<f32>(
            &[np, in_channels],
            None,
            None,
            Some(&key),
        )?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guidance-mode resolution: an explicit guidance-mode name wins, then a task-type name, then the
    /// conditioning/output defaults (t2i/t2v → vae_txt_vit_wapg; video src → v2v_apg; refs→video and
    /// video+image → rv2v_wapg).
    #[test]
    fn vit_mode_resolution() {
        // explicit guidance-mode name
        assert_eq!(
            resolve_vit_mode(Some("vae_txt_vit_wapg"), false, false, false),
            VitMode::VaeTxtVitWapg
        );
        assert_eq!(
            resolve_vit_mode(Some("rv2v_wapg"), true, true, true),
            VitMode::Rv2vWapg
        );
        // task-name fallback
        assert_eq!(
            resolve_vit_mode(Some("t2i"), false, false, false),
            VitMode::VaeTxtVitWapg
        );
        assert_eq!(
            resolve_vit_mode(Some("v2v"), true, false, true),
            VitMode::V2vApg
        );
        assert_eq!(
            resolve_vit_mode(Some("r2v"), false, true, true),
            VitMode::Rv2vWapg
        );
        // conditioning/output-driven defaults
        assert_eq!(
            resolve_vit_mode(None, false, false, false),
            VitMode::VaeTxtVitWapg
        ); // t2i
        assert_eq!(resolve_vit_mode(None, true, false, true), VitMode::V2vApg); // v2v
        assert_eq!(resolve_vit_mode(None, true, true, true), VitMode::Rv2vWapg); // rv2v
        assert_eq!(resolve_vit_mode(None, false, true, true), VitMode::Rv2vWapg); // r2v (refs→video)
        assert_eq!(
            resolve_vit_mode(None, false, true, false),
            VitMode::VaeTxtVitWapg
        ); // i2i
    }

    /// F-022: empty / non-`1+4k` conditioning video clips must be rejected up front with a clean
    /// `Error`, not dereferenced as `frames[0]` deep in the full pipeline on a malformed worker payload.
    #[test]
    fn rejects_empty_and_miscounted_conditioning_video_clips() {
        let clip = |n: usize| Conditioning::VideoClip {
            frames: vec![
                Image {
                    width: 2,
                    height: 2,
                    pixels: vec![0u8; 2 * 2 * 3],
                };
                n
            ],
            frame_idx: 0,
            strength: 1.0,
        };
        let req = |conds: Vec<Conditioning>| GenerationRequest {
            conditioning: conds,
            ..Default::default()
        };
        // empty clip → clean error (the panic this guards), not a frames[0] index.
        assert!(validate_conditioning_video_clips(&req(vec![clip(0)])).is_err());
        // 3 frames (not 1 + 4·k) → error.
        assert!(validate_conditioning_video_clips(&req(vec![clip(3)])).is_err());
        // valid temporal counts (1 + 4·k) → ok.
        assert!(validate_conditioning_video_clips(&req(vec![clip(1)])).is_ok());
        assert!(validate_conditioning_video_clips(&req(vec![clip(5)])).is_ok());
        // no conditioning → ok.
        assert!(validate_conditioning_video_clips(&req(vec![])).is_ok());
        // a bad clip among several valid ones is still caught.
        assert!(validate_conditioning_video_clips(&req(vec![clip(5), clip(0)])).is_err());
    }

    /// `grid_tokens` = t·h·w / merge².
    #[test]
    fn grid_token_count() {
        assert_eq!(grid_tokens([1, 12, 20]), 60);
        assert_eq!(grid_tokens([5, 12, 20]), 300);
    }

    /// The gen-target ViT grid: image targets are single-frame (`t = 1`); video targets sample
    /// `vit_fps` frames so `t > 1`; the spatial grid is the `smart_resize` of the output H/W.
    #[test]
    fn gen_target_grid_image_vs_video() {
        let img = gen_target_grid(480, 832, 1, 16);
        assert_eq!(img[0], 1, "image target is single-frame");
        assert_eq!(
            [img[1], img[2]],
            [12, 20],
            "480x832 → 12x20 merged-patch grid"
        );
        let vid = gen_target_grid(480, 832, 81, 16);
        assert!(vid[0] > 1, "video target spans multiple temporal patches");
        assert_eq!([vid[1], vid[2]], [12, 20], "same spatial grid as the image");
    }

    /// The seeded reveal permutation is deterministic for a seed and a valid permutation of `[0, n)`.
    #[test]
    fn seeded_permutation_is_a_permutation() {
        let n = 60;
        let a = seeded_permutation(n, 42).unwrap();
        let b = seeded_permutation(n, 42).unwrap();
        assert_eq!(a, b, "deterministic for a fixed seed");
        let c = seeded_permutation(n, 7).unwrap();
        assert_ne!(a, c, "different seeds → different order");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>(), "covers [0, n) once");
    }

    /// Per-step FM noise: one tensor per planning step, each `[revealed, in_channels]`.
    #[test]
    fn step_noise_shapes_match_schedule() {
        let n = 60;
        let steps = 25;
        let order = seeded_permutation(n, 42).unwrap();
        let noise = seeded_step_noise(n, steps, &order, 3584, 42).unwrap();
        assert_eq!(noise.len(), steps);
        let schedule = mar_schedule(n, steps, &order);
        for (s, arr) in noise.iter().enumerate() {
            let np = (schedule[s].len() as i32).max(1);
            assert_eq!(arr.shape(), &[np, 3584], "step {s} noise shape");
        }
    }
}
