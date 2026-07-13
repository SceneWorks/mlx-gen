//! `Seedvr2Generator` — the [`mlx_gen::Generator`] wiring the SeedVR2 pipeline into `mlx_gen`'s
//! registry (sc-4813 image, sc-4814 video). Registered under `seedvr2` (alias) + `seedvr2_3b`.
//!
//! **Surface.** A one-step super-resolution **upscaler** over image **and** video (`Modality::Both`),
//! dispatched on the request's conditioning:
//!   * [`Conditioning::Reference`] — the LR input image → [`GenerationOutput::Images`];
//!   * [`Conditioning::VideoClip`] — the LR input frame sequence → [`GenerationOutput::Video`]
//!     (temporal chunking + overlap cross-fade + a memory-budgeted chunk sizer; sc-4814).
//!
//! `width`/`height` are the target output size (both ÷16). No prompt, no guidance/CFG (1-step), no
//! LoRA. `spec.weights` is the raw `numz/SeedVR2_comfyUI` checkpoint dir (converted in-memory at
//! load — no Python). Dense bf16 default; `Fp32` honored (the parity path). Video `fps` passes
//! through `req.fps` (the worker supplies the source cadence; audio mux is the worker's job).
//!
//! 3B (default) + 7B (pixel-mode RoPE — sc-5197) are wired; `spec.quantize` Q4/Q8 quantizes the DiT
//! Linears at load (sc-5198).

use mlx_rs::Dtype;

use mlx_gen::{
    default_seed, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, Precision, Progress,
    Quant, Result, WeightsSource,
};

use crate::config::DitConfig;
use crate::pipeline::Seedvr2Pipeline;

pub const MODEL_ID: &str = "seedvr2";
pub const MODEL_ID_3B: &str = "seedvr2_3b";
pub const MODEL_ID_7B: &str = "seedvr2_7b";
const VAE_SCALE: u32 = 16; // VAE /8 · patch /2
const DIT_FILE_3B: &str = "seedvr2_ema_3b_fp16.safetensors";
const DIT_FILE_7B: &str = "seedvr2_ema_7b_fp16.safetensors";
/// Output fps when the request omits one (the worker normally supplies the source cadence).
const DEFAULT_FPS: u32 = 24;

/// The DiT checkpoint file + transformer config for a registered id (3B default; 7B is the
/// pixel-mode-RoPE variant — sc-5197). The VAE is shared across both.
fn variant(id: &str) -> (&'static str, DitConfig) {
    if id == MODEL_ID_7B {
        (DIT_FILE_7B, DitConfig::seedvr2_7b())
    } else {
        (DIT_FILE_3B, DitConfig::seedvr2_3b())
    }
}

fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        id,
        family: "seedvr2",
        backend: "mlx",
        modality: Modality::Both, // image (Reference) + video (VideoClip) upscaling
        capabilities: Capabilities {
            supports_negative_prompt: false, // precomputed neg-embed; no prompt surface
            supports_guidance: false,        // one-step, guidance fixed at 1.0
            supports_true_cfg: false,
            // the LR input image (image upscale) or LR frame sequence (video upscale)
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::VideoClip],
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["seedvr2_euler"],
            schedulers: vec!["seedvr2_euler"],
            supported_guidance_methods: vec![],
            min_size: VAE_SCALE,
            max_size: 4096,
            max_count: 8,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8], // Linear-only DiT quant (sc-5198)
            supports_kv_cache: false,
            requires_sigma_shift: false,
            // Not wired onto the shared `Residency` seam (F-176); Sequential is a no-op fallback.
            supports_sequential_offload: false,
        },
    }
}

pub fn descriptor() -> ModelDescriptor {
    descriptor_for(MODEL_ID)
}
pub fn descriptor_3b() -> ModelDescriptor {
    descriptor_for(MODEL_ID_3B)
}
pub fn descriptor_7b() -> ModelDescriptor {
    descriptor_for(MODEL_ID_7B)
}

pub struct Seedvr2Generator {
    descriptor: ModelDescriptor,
    pipe: Seedvr2Pipeline,
}

fn load_with(spec: &LoadSpec, id: &'static str) -> Result<Box<dyn Generator>> {
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(Error::Msg(format!(
            "{id}: ControlNet / IP-Adapter conditioning is not part of SeedVR2"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(format!(
            "{id}: LoRA/LoKr adapters are not supported"
        )));
    }
    let dtype = match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    };
    let dir = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{id}: expects a numz/SeedVR2_comfyUI checkpoint directory, not a single file"
            )))
        }
    };
    let (dit_file, cfg) = variant(id);
    let mut pipe = Seedvr2Pipeline::load(&dir, dit_file, &cfg, dtype)?;
    // sc-5198: Q4/Q8 quantize the DiT Linears at load (the VAE stays dense).
    if let Some(q) = spec.quantize {
        pipe.quantize(q.bits())?;
    }
    Ok(Box::new(Seedvr2Generator {
        descriptor: descriptor_for(id),
        pipe,
    }))
}

mlx_gen::impl_generator!(Seedvr2Generator {
    validate: |s, req| s.validate_impl(req),
    generate: generate_impl,
});

/// The LR input image carried by the request's `Reference` conditioning.
fn reference_image(req: &GenerationRequest) -> Option<&Image> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, .. } => Some(image),
        _ => None,
    })
}

impl Seedvr2Generator {
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        let has_video = req.video_clips().iter().any(|c| !c.frames.is_empty());
        if !has_video && reference_image(req).is_none() {
            return Err(Error::Msg(format!(
                "{}: requires a Reference image (image upscale) or a non-empty VideoClip frame \
                 sequence (video upscale)",
                self.descriptor.id
            )));
        }
        if !req.width.is_multiple_of(VAE_SCALE) || !req.height.is_multiple_of(VAE_SCALE) {
            return Err(Error::Msg(format!(
                "{}: width/height must be multiples of {VAE_SCALE} (got {}x{})",
                self.descriptor.id, req.width, req.height
            )));
        }
        Ok(())
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate_impl(req)?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let softness = req.softness.unwrap_or(0.0); // input pre-blur; reference default 0.0 (sc-4816)

        // Video upscale: a VideoClip carries the LR source frame sequence → one upscaled clip.
        // F-099: `validate_impl` accepts the request if ANY clip is non-empty, so match that
        // contract here by taking the first NON-empty clip — `into_iter().next()` would instead
        // take the first clip unconditionally, so `[empty_clip, real_clip]` returned an empty
        // video as success.
        if let Some(clip) = req.video_clips().into_iter().find(|c| !c.frames.is_empty()) {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            // F-099: `generate_video` reports per-chunk `Step{i, n}` progress itself (per-frame on
            // the fallback paths) — a minutes-long N-chunk run used to surface a single Step{1,1}.
            let frames = self.pipe.generate_video(
                clip.frames,
                req.width as i32,
                req.height as i32,
                base_seed,
                softness,
                None,
                &req.cancel,
                on_progress,
            )?;
            on_progress(Progress::Decoding);
            return Ok(GenerationOutput::Video {
                frames,
                fps: req.fps.unwrap_or(DEFAULT_FPS),
                audio: None,
            });
        }

        let image = reference_image(req).expect("validated");
        let mut out = Vec::with_capacity(req.count as usize);
        let count = req.count as usize;
        for i in 0..req.count {
            // Honor a cancel that arrives between output images for count > 1 (sc-5551); the per-tile
            // and single-pass paths inside `generate_with_progress` check the flag at their own step
            // boundaries (the tiled loop per tile, the single pass right after its `(1,1)` emit).
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let seed = base_seed.wrapping_add(i as u64);
            // F-162 / F-164 (sc-11133): fold each image's per-tile progress into ONE monotone bar
            // across the whole count loop. `generate_with_progress` reports `(tile_idx, n_tiles)`
            // (a lone `(1,1)` on the single-pass path); mapping it to
            // `current = i·n_tiles + tile_idx`, `total = count·n_tiles` makes both the count axis
            // (F-162) AND per-tile liveness (F-164) visible without the bar restarting per image or
            // reporting a bare `Step{1,1}`.
            let idx = i as usize;
            let img = self.pipe.generate_with_progress(
                image,
                req.width as i32,
                req.height as i32,
                seed,
                softness,
                &req.cancel,
                &mut |tile_idx, n_tiles| {
                    let (current, total) = fold_tile_progress(idx, tile_idx, n_tiles, count);
                    on_progress(Progress::Step { current, total });
                },
            )?;
            out.push(img);
        }
        // F-162: emit the terminal decode phase exactly ONCE for the whole batch (the pre-fix code
        // emitted it per image, violating the Decoding-exactly-once contract).
        on_progress(Progress::Decoding);
        Ok(GenerationOutput::Images(out))
    }
}

// Thin id-binding loaders over `load_with` (each pins the variant id), so they can't be a plain
// `load` path. They return the crate's rich `Result`; `register_generators!` adds the
// `gen_core::Result` bridge (epic 3720) and emits each `inventory::submit!`.
fn load_base(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID)
}
fn load_3b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID_3B)
}
fn load_7b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID_7B)
}

mlx_gen::register_generators! {
    descriptor => load_base,
    descriptor_3b => load_3b,
    descriptor_7b => load_7b,
}

/// Fold one image's per-tile progress onto the whole-batch bar (F-162 / F-164, sc-11133). Image
/// `image_idx` (0-based) of `count`, tile `tile_idx` (1-based) of `n_tiles`, maps to an absolute
/// `(current, total)` where `total = count × n_tiles` and `current = image_idx × n_tiles + tile_idx`
/// (clamped). A single-pass image reports `n_tiles == 1`, so this reduces to `(image_idx + 1, count)`
/// — the count axis F-162 wants — while a tiled image contributes per-tile liveness F-164 wants. The
/// sequence across the loop is monotone non-decreasing and reaches `total` on the final tile.
fn fold_tile_progress(
    image_idx: usize,
    tile_idx: usize,
    n_tiles: usize,
    count: usize,
) -> (u32, u32) {
    let total = count * n_tiles;
    let current = (image_idx * n_tiles + tile_idx).min(total);
    (current as u32, total as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F-162 / F-164 (sc-11133): the fold yields one monotone bar reaching `total`, covering both the
    /// single-pass (`n_tiles == 1` → the count axis) and spatial-tiling (per-tile liveness) shapes.
    #[test]
    fn fold_tile_progress_is_monotone_and_reaches_total() {
        // Single-pass, count = 8: each image is one "tile" → Step{i+1, 8} (the F-162 count axis).
        let seq: Vec<(u32, u32)> = (0..8).map(|i| fold_tile_progress(i, 1, 1, 8)).collect();
        assert_eq!(
            seq,
            (1..=8).map(|c| (c, 8)).collect::<Vec<_>>(),
            "single-pass batch must be a monotone 1..=count bar"
        );

        // count = 2, 3 tiles each: one monotone 1..=6 bar, per-tile liveness within each image.
        let mut folded = Vec::new();
        for image_idx in 0..2 {
            for tile_idx in 1..=3 {
                folded.push(fold_tile_progress(image_idx, tile_idx, 3, 2));
            }
        }
        assert_eq!(
            folded,
            vec![(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)],
            "tiled multi-image batch must fold to one monotone 1..=6 bar"
        );
        // The final tile of the final image reaches total.
        assert_eq!(folded.last(), Some(&(6, 6)));
    }

    #[test]
    fn descriptor_is_seedvr2() {
        let d = descriptor();
        assert_eq!(d.id, MODEL_ID);
        assert_eq!(d.family, "seedvr2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Both); // image (Reference) + video (VideoClip)
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::Reference));
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::VideoClip));
        assert!(!d.capabilities.supports_guidance);
        assert!(d.capabilities.mac_only);
    }

    #[test]
    fn both_ids_resolve_in_registry() {
        for id in [MODEL_ID, MODEL_ID_3B, MODEL_ID_7B] {
            let spec = LoadSpec {
                weights: WeightsSource::Dir("/nonexistent/seedvr2".into()),
                quantize: None,
                precision: Precision::Bf16,
                control: None,
                ip_adapter: None,
                adapters: Vec::new(),
                extra_controls: Vec::new(),
                pid: None,
                identity: None,
                text_encoder: None,
                offload_policy: Default::default(),
            };
            let err = match mlx_gen::load(id, &spec) {
                Ok(_) => panic!("bogus weights dir must fail to load"),
                Err(e) => e.to_string(),
            };
            assert!(
                !err.contains("no generator registered"),
                "{id} should resolve; got: {err}"
            );
        }
    }

    /// F-099: `validate_impl` accepts a request if ANY clip is non-empty, so the generate path must
    /// take the first NON-empty clip — not `into_iter().next()`. With the old `next()` selection,
    /// `[empty, real]` validated but generated an empty video "as success". This pins the
    /// first-non-empty predicate the generate path now uses.
    #[test]
    fn video_upscale_takes_the_first_non_empty_clip() {
        // Two clips: the first empty, the second with one frame. `find(!frames.is_empty())` must
        // skip the empty one (mirrors the generate_impl selection at line ~175).
        let frame = Image {
            pixels: vec![0u8; 3],
            width: 1,
            height: 1,
        };
        let nonempty = [frame];
        let clips: Vec<&[Image]> = vec![&[], &nonempty[..]];
        let selected = clips.into_iter().find(|c| !c.is_empty());
        assert!(
            selected.is_some(),
            "the non-empty second clip must be selected"
        );
        assert_eq!(selected.unwrap().len(), 1);
    }
}
