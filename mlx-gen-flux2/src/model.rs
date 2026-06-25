//! FLUX.2 provider registration + the generation path, shared across the klein and **dev** variants.
//!
//! `load()` assembles the tokenizer, text encoder, MMDiT transformer, and 32-ch VAE from a snapshot
//! directory — klein uses the Qwen3 loaders, dev (sc-2365) the Mistral3 `*_dev` loaders (which load
//! a pre-quantized Q4 snapshot packed, sc-5917); `spec.quantize` (Q4/Q8, sc-2643) then quantizes the
//! dense parts in place (a no-op for already-packed dev weights). `generate()` runs the flow-match
//! denoise loop, then BN-denormalizes + 2×2-unpatchifies + VAE-decodes. Guidance is variant-typed:
//! distilled klein runs CFG-free (1.0 = single forward; a base variant would CFG dual-forward when
//! `guidance > 1`); guidance-distilled **dev** feeds its scale as an embedded scalar into the
//! transformer's guidance embedder (single forward, default ~4.0 over ~28 steps — NOT true-CFG).
//! txt2img (`flux2_klein_9b`, `flux2_dev`) and the single-/multi-reference edit variants share this
//! path.
//!
//! Activations run f32 (matmul(f32, bf16)→f32): dodges the dense 16-bit Metal GEMM bug and is the
//! quality target. Pixel-parity with the fork's bf16 render is therefore not the gate (see the
//! e2e test) — component f32 parity + visual correctness is.

use mlx_gen::array::scalar;
use mlx_gen::image::decoded_to_image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, run_flow_sampler, Error, GenerationOutput, GenerationRequest, Generator,
    LatentDecoder, LoadSpec, ModelDescriptor, Precision, Progress, Result, TimestepConvention,
    WeightsSource,
};
use mlx_gen_pid::{resolve_pid_decoder, PidEngine};
use mlx_rs::ops::{add, concatenate_axis, multiply, pad, subtract};
use mlx_rs::Array;

use crate::caption_upsample;
use crate::chunk::MemoryConfig;
use crate::config::Flux2Variant;
use crate::kv_cache::{CacheMode, Flux2KvCache};
use crate::pipeline::{
    add_noise_by_interpolation, create_noise, init_time_step, pack_latents, patchify_latents,
    prepare_grid_ids, prepare_text_ids, preprocess_ref_image, schedule_with,
};
use crate::text_encoder::Qwen3TextEncoder;
use crate::transformer::{Flux2ForwardInputs, Flux2Transformer};
use crate::vae::Flux2Vae;
use crate::vision::{Mistral3Projector, PixtralVisionTower};
use crate::{loader, Flux2Config};

/// PiD latent-space tag for the FLUX.2 family (epic 7840, sc-7847): the FLUX.2 `AutoencoderKLFlux2`
/// 32-ch / 2×2-patchified / BatchNorm latent. `flux2`, `flux2-klein-4b`, and `flux2-klein-9b` all
/// resolve to the same student + checkpoint in [`mlx_gen_pid::registry`]; Lens and Ideogram 4 reuse
/// this same space.
pub const PID_BACKBONE: &str = "flux2";

/// Joint DiT sequence length (txt + target + reference tokens) above which the gated activation
/// levers (sc-6266) engage. Sits between a single-reference 1024² edit (~8.7K tokens, fits the 96 GB
/// budget) and a 2-reference one (~12.8K tokens, ~104 GB un-bounded, sc-6124) so only the over-budget
/// multi-reference / high-resolution edits take the bounded-memory path; every shipped path (T2I,
/// single-reference edit, strict pose, LoRA) stays on the byte-identical [`MemoryConfig::OFF`].
const LONG_SEQ_TOKEN_THRESHOLD: usize = 10_000;

/// Per-reference stride on the RoPE time axis (the fork's `prepare_reference_image_conditioning`):
/// reference `i` is tagged at `t = REFERENCE_TIME_STRIDE * (i + 1)` (10, 20, 30, …) so each edit
/// reference occupies its own time band, distinct from the target's `t = 0`. The stride must exceed a
/// single reference's t-extent (1, since each ref is one packed grid at a fixed t) to avoid two refs
/// colliding on the same time index; at the `MultiReference` capability cap (`max_count = 8`) the band
/// tops out at `t = 80`, well inside the RoPE t-axis range. Named so the invariant is explicit rather
/// than a bare `10 + 10*i`.
const REFERENCE_TIME_STRIDE: i32 = 10;

/// Sanitize model-generated text for a single-line, machine-parsed log record (the worker consumes
/// the `ENHANCED_PROMPT:` / `ENHANCER_FALLBACK:` prefix): replace every control/whitespace char (incl.
/// embedded newlines that would split the record or forge a second prefix line) with a space, collapse
/// runs, and length-cap to 512 chars. Only the logged copy is touched — never the prompt itself.
fn sanitize_log_text(s: &str) -> String {
    const CAP: usize = 512;
    let collapsed: String = s
        .chars()
        .map(|c| {
            if c.is_control() || c.is_whitespace() {
                ' '
            } else {
                c
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if collapsed.chars().count() > CAP {
        let truncated: String = collapsed.chars().take(CAP).collect();
        format!("{truncated}…")
    } else {
        collapsed
    }
}

/// Walk the request conditioning for reference images (`Reference` + `MultiReference`), flattened in
/// conditioning order then image order (the fork's flat `image_paths`). Shared by the edit and
/// caption-upsample paths; the empty-check is the caller's (edit requires ≥1, upsample's T2I path
/// tolerates none) (F-013/L-dedup).
fn collect_reference_images(req: &GenerationRequest) -> Vec<&mlx_gen::media::Image> {
    let mut refs: Vec<&mlx_gen::media::Image> = Vec::new();
    for c in &req.conditioning {
        match c {
            mlx_gen::Conditioning::Reference { image, .. } => refs.push(image),
            mlx_gen::Conditioning::MultiReference { images } => refs.extend(images.iter()),
            _ => {}
        }
    }
    refs
}

pub fn descriptor_klein_9b() -> ModelDescriptor {
    Flux2Variant::Klein9b.descriptor()
}

pub fn descriptor_klein_9b_edit() -> ModelDescriptor {
    Flux2Variant::Klein9bEdit.descriptor()
}

pub fn descriptor_klein_9b_kv_edit() -> ModelDescriptor {
    Flux2Variant::Klein9bKvEdit.descriptor()
}

pub fn load_klein_9b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9b, spec)
}

pub fn load_klein_9b_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9bEdit, spec)
}

pub fn load_klein_9b_kv_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9bKvEdit, spec)
}

pub fn descriptor_dev() -> ModelDescriptor {
    Flux2Variant::Dev.descriptor()
}

pub fn descriptor_dev_edit() -> ModelDescriptor {
    Flux2Variant::DevEdit.descriptor()
}

/// FLUX.2-dev txt2img (sc-2365): the guidance-distilled 32B flagship. Loads the dev snapshot
/// (Mistral3 TE + dev DiT, pre-quantized Q4 per sc-5917) and runs the embedded-guidance denoise
/// (single forward, default guidance ~4.0 over ~28 steps — NOT true-CFG).
pub fn load_dev(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Dev, spec)
}

/// FLUX.2-dev image-conditioned edit (sc-5919): single + multi reference. Loads the same dev
/// snapshot as [`load_dev`] and runs the shared edit conditioning path — reference images are
/// VAE-encoded, packed, and concatenated to the DiT image stream (the klein edit mechanism, faithful
/// to the diffusers `Flux2Pipeline`; the prompt embeds stay text-only). Embedded-guidance denoise.
pub fn load_dev_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::DevEdit, spec)
}

fn load_variant(variant: Flux2Variant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        // The dense path loads at the on-disk dtype and runs f32 activations; an explicit fp32
        // precision override isn't a separate wired mode. Q4/Q8 (sc-2643) go through `spec.quantize`.
        return Err(Error::Msg(format!(
            "{}: only the default precision is wired; drop the precision override (Q4/Q8 = spec.quantize)",
            variant.id()
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{} expects a FLUX.2 snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/), not a single .safetensors file",
                variant.id()
            )))
        }
    };

    // The dev checkpoint has a different text encoder (Mistral3, not Qwen3) + tokenizer + DiT dims,
    // so it loads through the `*_dev` loaders; a pre-quantized dev snapshot loads packed directly
    // (the loaders read the per-component `quantization` manifest, sc-5917). The VAE is identical.
    // Both dev variants (txt2img + edit) load the same snapshot through these loaders.
    let dev = variant.is_dev();
    let (mut text_encoder, mut transformer) = if dev {
        (
            loader::load_text_encoder_dev(root)?,
            loader::load_transformer_dev(root)?,
        )
    } else {
        (
            loader::load_text_encoder(root)?,
            loader::load_transformer(root)?,
        )
    };
    let mut vae = loader::load_vae(root)?;
    // Q4/Q8 quantizes the **whole model** in place after the dense load — the fork's `nn.quantize`
    // over (transformer, text_encoder, vae), group_size 64, every quantizable Linear (+ the text
    // encoder's token Embedding). Full-model scope like Z-Image (sc-2532), unlike Qwen's
    // transformer-only quant (sc-2565) — quant scope is per-fork. The VAE's quantized surface is
    // just its two mid-block attentions (everything else there is Conv/GroupNorm). The dense load
    // runs f32, but `quantize` casts weights to bf16 before packing so the scales byte-match the
    // fork's bf16 `nn.quantize` (sc-2604).
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        text_encoder.quantize(bits)?;
        vae.quantize(bits)?;
    }

    // LoRA/LoKr (sc-2646): applied AFTER quantization, as forward-time residuals over the
    // (possibly quantized) transformer — fork-faithful, transformer-only. No-op when empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_flux2_adapters(&mut transformer, &spec.adapters)?;
    }

    let tokenizer = if dev {
        loader::load_tokenizer_dev(root)?
    } else {
        loader::load_tokenizer(root)?
    };
    // Caption upsampling (sc-6030) is dev-only: load the Pixtral vision tower + Mistral3 projector
    // (the `text_encoder/` snapshot's `vision_tower.*` / `multi_modal_projector.*`, full precision).
    // The Mistral generation head (final norm + LM head) was loaded into `text_encoder` by
    // `load_text_encoder_dev`. klein has no vision tower → `None` (caption upsampling is unavailable).
    let (vision_tower, projector) = if dev {
        (
            Some(loader::load_vision_tower_dev(root)?),
            Some(loader::load_multimodal_projector_dev(root)?),
        )
    } else {
        (None, None)
    };
    // PiD decoder overlay (epic 7840, sc-7847): load the `flux2` student + Gemma caption encoder once
    // when the spec carries it. The student is shared across the whole FLUX.2 family (klein + dev).
    let pid = spec
        .pid
        .as_ref()
        .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
        .transpose()?;
    Ok(Box::new(Flux2 {
        descriptor: variant.descriptor(),
        variant,
        config: variant.config(),
        parts: Some(Flux2Parts {
            tokenizer,
            text_encoder,
            transformer,
            vae,
        }),
        vision_tower,
        projector,
        pid,
    }))
}

/// The always-present core of a *loaded* FLUX.2 model. Held behind a single `Option` on [`Flux2`]
/// (`Some` once weights are loaded, `None` only for the weightless validation-test instances) so
/// "is the model loaded?" is one decision — distinct from the dev-only [`Flux2`] `vision_tower` /
/// `projector` `Option`s, which encode the unrelated "dev variant vs klein" question. Untangling the
/// two reasons a field used to be `Option` is the point of F-013.
struct Flux2Parts {
    tokenizer: TextTokenizer,
    text_encoder: Qwen3TextEncoder,
    transformer: Flux2Transformer,
    vae: Flux2Vae,
}

/// The FLUX.2-klein generator.
pub struct Flux2 {
    descriptor: ModelDescriptor,
    variant: Flux2Variant,
    config: Flux2Config,
    /// The loaded core (tokenizer / text-encoder / transformer / VAE). `None` only for the weightless
    /// `new_for_tests` instances; the production load path always populates it.
    parts: Option<Flux2Parts>,
    /// FLUX.2-dev caption upsampling (sc-6030): the Pixtral vision tower + Mistral3 projector that
    /// encode reference images for the image-conditioned (I2I) prompt rewrite. `None` for klein and
    /// the weightless test instances — caption upsampling is dev-only and gated on `enhance_prompt`.
    vision_tower: Option<PixtralVisionTower>,
    projector: Option<Mistral3Projector>,
    /// Optional PiD super-resolving decoder overlay (epic 7840, sc-7847): loaded when the request
    /// carries `LoadSpec::pid`. `Some` → a `req.use_pid` generation decodes the packed BN-normalized
    /// latent through the `flux2` PiD student (4× SR) instead of the VAE. `None` for the default
    /// (byte-exact VAE) path and the weightless test instances.
    pid: Option<PidEngine>,
}

impl Flux2 {
    /// Construct a weightless instance for validation tests (`parts: None`).
    pub fn new_for_tests(variant: Flux2Variant) -> Self {
        Self {
            descriptor: variant.descriptor(),
            variant,
            config: variant.config(),
            parts: None,
            vision_tower: None,
            projector: None,
            pid: None,
        }
    }

    /// Borrow the loaded core as the historical `(tokenizer, text_encoder, transformer, vae)` tuple.
    /// One `Option` check now stands for "is the model loaded" (vs the former four), and the dev
    /// extras (`vision_tower`/`projector`) are queried separately by `run_upsample` (F-013).
    fn parts(
        &self,
    ) -> Result<(
        &TextTokenizer,
        &Qwen3TextEncoder,
        &Flux2Transformer,
        &Flux2Vae,
    )> {
        let p = self
            .parts
            .as_ref()
            .ok_or_else(|| Error::Msg(format!("{}: model is not loaded", self.descriptor.id)))?;
        Ok((&p.tokenizer, &p.text_encoder, &p.transformer, &p.vae))
    }

    /// Encode a prompt → `(prompt_embeds [1,512,joint], text_ids [1,512,4])`.
    fn encode(
        &self,
        tokenizer: &TextTokenizer,
        te: &Qwen3TextEncoder,
        prompt: &str,
    ) -> Result<(Array, Array)> {
        let tok = tokenizer.tokenize(prompt)?;
        let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&tok);
        let embeds = te.prompt_embeds(&input_ids, &attention_mask)?;
        let ids = prepare_text_ids(embeds.shape()[1] as usize);
        Ok((embeds, ids))
    }

    /// Edit reference conditioning for **N** images (the fork's `prepare_reference_image_conditioning`):
    /// each image → resize → VAE-encode → crop-to-even → 2×2 patchify → BN-normalize → pack, tagged
    /// with grid ids at `t = 10 + 10·i` (the per-reference time offset), then all refs concatenated
    /// on the sequence axis. Returns `(image_latents [1, Σseq_ref, 128], image_latent_ids
    /// [1, Σseq_ref, 4])`. A single reference (N = 1) reduces to the original `t = 10` path. The
    /// FLUX.2 text encoder is a dense Qwen3 LLM with no vision input, so the prompt embeds are
    /// independent of the references — multi-image conditioning flows ONLY through these tokens.
    fn encode_references(
        &self,
        vae: &Flux2Vae,
        images: &[&mlx_gen::media::Image],
        width: u32,
        height: u32,
    ) -> Result<(Array, Array)> {
        let mut packed: Vec<Array> = Vec::with_capacity(images.len());
        let mut ids: Vec<Array> = Vec::with_capacity(images.len());
        for (i, image) in images.iter().enumerate() {
            let pre = preprocess_ref_image(image, width, height)?; // NHWC [1,H,W,3]
            let enc = vae.encode_mean(&pre)?; // NHWC [1,H/8,W/8,32]
            let enc = enc.transpose_axes(&[0, 3, 1, 2])?; // → NCHW for the pipeline helpers
            let enc = crop_to_even(&enc)?;
            let patchified = patchify_latents(&enc)?; // [1,128,h,w]
            let normed = vae.bn_normalize_nchw(&patchified)?;
            let sh = patchified.shape();
            packed.push(pack_latents(&normed)?); // [1, seq_ref, 128]
            ids.push(prepare_grid_ids(
                sh[2] as usize,
                sh[3] as usize,
                REFERENCE_TIME_STRIDE * (i as i32 + 1),
            ));
        }
        let packed_refs: Vec<&Array> = packed.iter().collect();
        let id_refs: Vec<&Array> = ids.iter().collect();
        Ok((
            concatenate_axis(&packed_refs, 1)?,
            concatenate_axis(&id_refs, 1)?,
        ))
    }

    /// Collect the ordered edit reference images from the request: a single `Reference`, a
    /// `MultiReference { images }` (N images, sc-2645), or several `Reference`s — flattened in
    /// conditioning order then image order (the fork passes a flat `image_paths` list). At least
    /// one reference is required (the empty-check is the edit caller's; the upsample T2I path uses the
    /// shared [`collect_reference_images`] walk directly and tolerates none).
    fn collect_edit_references<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Vec<&'a mlx_gen::media::Image>> {
        let refs = collect_reference_images(req);
        if refs.is_empty() {
            return Err(Error::Msg(format!(
                "{}: edit requires at least one reference image",
                self.descriptor.id
            )));
        }
        Ok(refs)
    }

    /// FLUX.2-dev caption upsampling (sc-6030): rewrite the prompt with the Mistral3 multimodal LLM
    /// before encoding (the diffusers `upsample_prompt`), gated on `req.enhance_prompt` — the
    /// LTX-2.3 prompt-enhancement contract field (sc-2845), reused here for the image-aware analog.
    /// Returns the rewritten prompt, or the original `req.prompt` when the gate is off, the variant
    /// isn't dev, or on **any** upsampler failure / empty output (reference-faithful fallback, like
    /// `generate_av.py`'s try/except). Logs the LTX `ENHANCED_PROMPT:` / `ENHANCER_FALLBACK:` tokens.
    fn maybe_upsample(&self, req: &GenerationRequest) -> String {
        if !req.enhance_prompt || !self.variant.is_dev() {
            return req.prompt.clone();
        }
        match self.run_upsample(req) {
            Ok(p) if !p.trim().is_empty() => {
                // The log record is machine-parsed on the `ENHANCED_PROMPT:` prefix; sanitize the
                // model-generated text so an embedded newline can't split the record or forge a
                // second prefix line (the returned `p` itself is unchanged) (L-log-injection).
                eprintln!("ENHANCED_PROMPT:{}", sanitize_log_text(&p));
                p
            }
            Ok(_) => {
                eprintln!("ENHANCER_FALLBACK:EmptyOutput:caption upsampler returned empty output");
                req.prompt.clone()
            }
            Err(e) => {
                eprintln!("ENHANCER_FALLBACK:{}", sanitize_log_text(&e.to_string()));
                req.prompt.clone()
            }
        }
    }

    /// Run the dev caption upsampler: the Mistral3 multimodal `generate()` over the prompt plus any
    /// reference images (through the Pixtral tower). Errors surface to
    /// [`maybe_upsample`](Self::maybe_upsample)'s fallback.
    fn run_upsample(&self, req: &GenerationRequest) -> Result<String> {
        let id = self.descriptor.id;
        let not_loaded = |what: &str| Error::Msg(format!("{id}: {what} is not loaded"));
        let (tokenizer, te, _, _) = self.parts()?;
        let vision = self
            .vision_tower
            .as_ref()
            .ok_or_else(|| not_loaded("vision tower"))?;
        let projector = self
            .projector
            .as_ref()
            .ok_or_else(|| not_loaded("projector"))?;
        let refs = collect_reference_images(req);
        let temperature = req
            .enhance_temperature
            .unwrap_or(caption_upsample::DEFAULT_TEMPERATURE);
        // Clamp the requested decode length to a hard ceiling (F-012): each step is a full ~32B forward
        // over a growing KV cache, so an unclamped `enhance_max_tokens` is an effectively unbounded job.
        let max_new_tokens = caption_upsample::clamp_max_new_tokens(req.enhance_max_tokens);
        let seed = req.seed.unwrap_or_else(default_seed);
        caption_upsample::upsample_prompt(
            tokenizer,
            te,
            vision,
            projector,
            &req.prompt,
            &refs,
            temperature,
            max_new_tokens,
            seed,
            &req.cancel,
        )
    }

    /// Extract the single img2img init image + its strength from the txt2img request. The
    /// per-reference strength wins over `req.strength`. txt2img img2img conditions on exactly one
    /// init image, so more than one `Reference` is an error (multi-reference is the edit variant +
    /// `MultiReference`, sc-2645). Returns `None` for pure txt2img.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a mlx_gen::media::Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let mlx_gen::Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(format!(
                        "{}: multiple reference images are not supported (single img2img init only)",
                        self.descriptor.id
                    )));
                }
                reference = Some((image, strength.or(req.strength)));
            }
        }
        Ok(reference)
    }

    /// img2img init conditioning: resize → VAE-encode → NCHW → crop-to-even → center-crop/pad to the
    /// target latent grid → 2×2 patchify → BN-normalize → pack. Returns the **clean** packed latents
    /// `[1, lat_h·lat_w, 128]` (seed-independent — blended with the per-seed noise in `generate`).
    /// Mirrors the fork's `_prepare_img2img_latents` (minus the noise blend); same encode chain as
    /// `encode_reference`, plus the `_match_latent_spatial_size` step and the txt2img grid ids.
    fn encode_init_latents(
        &self,
        vae: &Flux2Vae,
        image: &mlx_gen::media::Image,
        width: u32,
        height: u32,
    ) -> Result<Array> {
        let pre = preprocess_ref_image(image, width, height)?; // NHWC [1,H,W,3]
        let enc = vae.encode_mean(&pre)?; // NHWC [1,H/8,W/8,32]
        let enc = enc.transpose_axes(&[0, 3, 1, 2])?; // → NCHW for the pipeline helpers
        let enc = crop_to_even(&enc)?;
        // Target the denoise latent grid: `latent_h·2 × latent_w·2 = H/8 × W/8`. A no-op at the
        // standard multiple-of-16 sizes (encoded H/8 already equals the target).
        let enc = match_latent_spatial_size(&enc, (height / 8) as i32, (width / 8) as i32)?;
        let patchified = patchify_latents(&enc)?; // [1,128,h,w]
        let normed = vae.bn_normalize_nchw(&patchified)?;
        pack_latents(&normed) // [1, lat_h·lat_w, 128]
    }
}

/// Crop a NCHW latent's spatial dims down to even (the fork's `crop_to_even_spatial`), so the 2×2
/// patchify divides cleanly. A no-op at the standard multiple-of-16 sizes.
pub(crate) fn crop_to_even(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let mut x = x.clone();
    if sh[2] % 2 != 0 {
        let idx = Array::from_slice(&(0..sh[2] - 1).collect::<Vec<i32>>(), &[sh[2] - 1]);
        x = x.take_axis(&idx, 2)?;
    }
    if sh[3] % 2 != 0 {
        let idx = Array::from_slice(&(0..sh[3] - 1).collect::<Vec<i32>>(), &[sh[3] - 1]);
        x = x.take_axis(&idx, 3)?;
    }
    Ok(x)
}

/// Center-crop or symmetric-pad a NCHW latent's spatial dims to `(target_h, target_w)` — the fork's
/// `_match_latent_spatial_size`. A no-op at the standard multiple-of-16 sizes (the VAE-encoded H/8
/// already equals the `latent_h·2` target); guards odd / mismatched user images.
pub(crate) fn match_latent_spatial_size(x: &Array, target_h: i32, target_w: i32) -> Result<Array> {
    let mut x = x.clone();
    let (h, w) = (x.shape()[2], x.shape()[3]);
    if h != target_h {
        if h > target_h {
            let off = (h - target_h) / 2;
            let idx = Array::from_slice(&(off..off + target_h).collect::<Vec<i32>>(), &[target_h]);
            x = x.take_axis(&idx, 2)?;
        } else {
            let before = (target_h - h) / 2;
            let after = (target_h - h) - before;
            x = pad(
                &x,
                &[(0, 0), (0, 0), (before, after), (0, 0)][..],
                None,
                None,
            )?;
        }
    }
    if w != target_w {
        if w > target_w {
            let off = (w - target_w) / 2;
            let idx = Array::from_slice(&(off..off + target_w).collect::<Vec<i32>>(), &[target_w]);
            x = x.take_axis(&idx, 3)?;
        } else {
            let before = (target_w - w) / 2;
            let after = (target_w - w) - before;
            x = pad(
                &x,
                &[(0, 0), (0, 0), (0, 0), (before, after)][..],
                None,
                None,
            )?;
        }
    }
    Ok(x)
}

mlx_gen::impl_generator!(Flux2 {
    validate: |s, req| validate_request(&s.descriptor, req),
    generate: generate_impl,
});

impl Flux2 {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let (tokenizer, te, transformer, vae) = self.parts()?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let steps = req.steps.unwrap_or(self.variant.default_steps()) as usize;
        let guidance = req.guidance.unwrap_or(self.variant.default_guidance());
        // dev is guidance-DISTILLED: the scale is an embedded scalar fed into the transformer's
        // guidance embedder (single forward), NOT a true-CFG dual-forward over a negative prompt.
        let embedded_guidance = self.variant.uses_embedded_guidance().then_some(guidance);

        // Edit: build the reference-image conditioning from one `Reference` or one `MultiReference`
        // (sc-2645). The transformer sees the joint sequence `[txt, target, ref0, ref1, …]`; its
        // output keeps the leading `target_seq` image tokens.
        let reference = if self.variant.is_edit() {
            let images = self.collect_edit_references(req)?;
            Some(self.encode_references(vae, &images, req.width, req.height)?)
        } else {
            None
        };

        // img2img (txt2img variant): a single `Reference` init image seeds the latents via the
        // noise blend at `sigmas[start_step]`, with the denoise loop starting at `start_step`
        // (= the fork's `_prepare_img2img_latents` + `Config.init_time_step`). The edit variant
        // consumes its `Reference` above (token concat), so img2img is txt2img-only.
        let img2img = if self.variant.is_edit() {
            None
        } else {
            self.resolve_reference(req)?
        };
        let start_step = match &img2img {
            Some((_, strength)) => init_time_step(steps, *strength),
            None => 0,
        };

        // FLUX.2-dev caption upsampling (sc-6030): optionally rewrite the prompt with the Mistral3
        // multimodal LLM (using any reference images) before encoding, gated on `enhance_prompt`.
        // A no-op (returns `req.prompt`) for klein, when the gate is off, or on any upsampler failure.
        let prompt = self.maybe_upsample(req);
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let (prompt_embeds, text_ids) = self.encode(tokenizer, te, &prompt)?;
        // True-CFG dual-forward only for the (non-embedded-guidance) base path at guidance >1; dev
        // routes its scale through the embedded guidance embedder instead, so it never takes a
        // negative pass, and distilled klein runs at guidance 1.0 (also no negative).
        let negative = if !self.variant.uses_embedded_guidance() && guidance > 1.0 {
            Some(self.encode(tokenizer, te, " ")?)
        } else {
            None
        };

        let sched = schedule_with(steps, req.width, req.height, req.scheduler.as_deref());
        let lat_h = (req.height / 16) as usize;
        let lat_w = (req.width / 16) as usize;
        let latent_ids = prepare_grid_ids(lat_h, lat_w, 0);
        let in_channels = self.config.in_channels as i32;

        // The img2img clean init latents are seed-independent — encode once, blend with per-seed
        // noise below. `None` for pure txt2img (or strength ≤ 0, where `start_step == 0`).
        let clean_init = match &img2img {
            Some((image, _)) if start_step > 0 => {
                Some(self.encode_init_latents(vae, image, req.width, req.height)?)
            }
            _ => None,
        };

        // sc-6266: a multi-reference edit concatenates each reference's latent tokens onto the joint
        // `[txt, target, ref…]` DiT sequence, making the denoise activation-bound — a 2-reference
        // 1024² edit peaks ~104 GB, over the 96 GB budget (sc-6124). Above the single-reference
        // ceiling, bound the per-step activation high-water with `eval_per_block` (bit-exact, so the
        // edit's pixels are unchanged). Shorter sequences (T2I, single-reference edit, pose) stay on
        // `MemoryConfig::OFF` → the shipped forward is byte-identical. Env-overridable (the doc on
        // `MemoryConfig::from_env`) so a deployment can tune chunking without a recompile.
        let total_seq = prompt_embeds.shape()[1] as usize
            + lat_h * lat_w
            + reference
                .as_ref()
                .map(|(r, _)| r.shape()[1] as usize)
                .unwrap_or(0);
        let mem = MemoryConfig::from_env(if total_seq > LONG_SEQ_TOKEN_THRESHOLD {
            MemoryConfig::LONG_SEQ
        } else {
            MemoryConfig::OFF
        });

        // For an edit, the transformer's image input/ids are `[target, ref]` (or `[target]` only on
        // a cached KV step); its output keeps the image stream, of which we take the leading
        // `target_seq` tokens. txt2img has no ref, so the concat + slice are no-ops.
        // `include_ref=false` drops the reference tokens (the 9b-kv cached step); `cache` threads
        // the per-seed KV cache through the transformer.
        let run = |latents: &Array,
                   embeds: &Array,
                   ids: &Array,
                   ts: f32,
                   include_ref: bool,
                   cache: Option<&Flux2KvCache>|
         -> Result<Array> {
            let target_seq = latents.shape()[1];
            let (hidden, img_ids) = match (&reference, include_ref) {
                (Some((ref_lat, ref_ids)), true) => (
                    concatenate_axis(&[latents, ref_lat], 1)?,
                    concatenate_axis(&[&latent_ids, ref_ids], 1)?,
                ),
                _ => (latents.clone(), latent_ids.clone()),
            };
            let out = transformer.forward_with_mem(
                &Flux2ForwardInputs {
                    hidden_states: &hidden,
                    encoder_hidden_states: embeds,
                    img_ids: &img_ids,
                    txt_ids: ids,
                    timestep: ts,
                    guidance: embedded_guidance,
                },
                cache,
                &mem,
            )?;
            let idx = Array::from_slice(&(0..target_seq).collect::<Vec<i32>>(), &[target_seq]);
            Ok(out.take_axis(&idx, 1)?)
        };

        // 9b-kv edit: cache reference K/V on step 0, reuse on steps 1+ (the ~2.4× speedup). The
        // edit path always has a reference, so `num_ref > 0`.
        let kv_enabled = self.variant.is_kv() && reference.is_some();
        let num_ref = reference
            .as_ref()
            .map(|(r, _)| r.shape()[1] as usize)
            .unwrap_or(0);

        // sc-2963 (rollout of sc-2957): run the MMDiT's fusable elementwise glue (adaLN affine,
        // SwiGLU, gated residual, RoPE rotation) through `mx.compile` — bit-exact (`max|Δ|=0`,
        // compile_parity.rs) and a per-step win at production geometry. Scoped to this render by the
        // RAII guard (F-007): the process-global toggle is restored on drop, even on an early `?`.
        let _compile_glue = crate::transformer::CompileGlueGuard::enable();

        // PiD decode overlay (epic 7840, sc-7847): when `req.use_pid` is set and the model was loaded
        // with `LoadSpec::pid`, mint a per-generation decoder (clean σ=0, seeded from `base_seed`) that
        // super-resolves the packed latent 4× in place of the VAE. Errors if requested-but-not-loaded;
        // `None` (the default) → the byte-exact VAE path. One decoder serves the whole count loop.
        let pid_decoder =
            resolve_pid_decoder(self.pid.as_ref(), req, base_seed, self.descriptor.id)?;

        let sampler_name = req.sampler.as_deref();
        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let noise = create_noise(seed, req.width, req.height, self.config.in_channels)?;
            // img2img: `(1-σ)·clean + σ·noise` at `σ = sigmas[start_step]`; txt2img: pure noise.
            let latents = match &clean_init {
                Some(clean) => add_noise_by_interpolation(clean, &noise, sched.sigmas[start_step])?,
                None => noise,
            };
            // Fresh cache per seed — the cached reference K/V depend on the step-0 target latents.
            let cache = kv_enabled.then(|| {
                Flux2KvCache::new(self.config.num_double_layers, self.config.num_single_layers)
            });
            // The curated unified-framework solver owns the loop (epic 7114 P3). KV step role: the
            // first executed forward extracts the reference K/V (the full `[txt, target, ref]` pass);
            // later forwards run `[txt, target]` and splice the cached ref K/V back in. "First executed
            // forward" is tracked by `extracted` (not `t == start_step`) so a multi-eval solver still
            // extracts the ref K/V once and reuses it; the single-eval Euler default is byte-identical
            // to the prior loop. FLUX.2 feeds `sigma · 1000` as the transformer timestep (Sigma
            // convention; the ×1000 is applied here).
            let mut extracted = false;
            let predict = |latents: &Array, sigma: f32| -> Result<Array> {
                let ts = sigma * 1000.0;
                let (include_ref, cache_ref) = match &cache {
                    Some(c) => {
                        let mode = if extracted {
                            CacheMode::Cached
                        } else {
                            CacheMode::Extract
                        };
                        c.configure(mode, num_ref);
                        extracted = true;
                        (mode == CacheMode::Extract, Some(c))
                    }
                    None => (true, None),
                };
                let v = run(
                    latents,
                    &prompt_embeds,
                    &text_ids,
                    ts,
                    include_ref,
                    cache_ref,
                )?;
                match &negative {
                    Some((neg_embeds, neg_ids)) => {
                        // CFG with the cache mirrors the fork: the same cache feeds both forwards
                        // (the negative extract overwrites the positive's slots). Distilled klein
                        // runs guidance 1.0 → no negative pass, so this is the base path in practice.
                        let vn = run(latents, neg_embeds, neg_ids, ts, include_ref, cache_ref)?;
                        // noise = neg + guidance·(pos − neg)
                        Ok(add(&vn, &multiply(&subtract(&v, &vn)?, scalar(guidance))?)?)
                    }
                    None => Ok(v),
                }
            };
            // Cancellation, the per-step `eval` (sc-5522 / sc-5399), and progress live in
            // `run_flow_sampler`. img2img slices the schedule from `start_step` (the fork's
            // `range(init_time_step, n)`).
            let final_latents = run_flow_sampler(
                sampler_name,
                TimestepConvention::Sigma,
                &sched.sigmas[start_step..],
                latents,
                seed,
                &req.cancel,
                on_progress,
                predict,
            )?;
            on_progress(Progress::Decoding);
            let packed = final_latents.reshape(&[1, lat_h as i32, lat_w as i32, in_channels])?;
            let nchw = match &pid_decoder {
                // PiD: `packed` (NHWC [1,h,w,128]) is already the BN-normalized packed latent the
                // student trained on — the exact tensor `decode_packed_latents` BN-de-normalizes
                // (sc-7847). Hand it over as NCHW [1,128,h,w]; the student returns NCHW [1,3,4H,4W].
                Some(d) => d.decode(&packed.transpose_axes(&[0, 3, 1, 2])?)?,
                // Native VAE: BN-de-normalize + 2×2-unpatchify + decode → NHWC [1,H,W,3] → NCHW.
                None => vae
                    .decode_packed_latents(&packed)?
                    .transpose_axes(&[0, 3, 1, 2])?,
            };
            images.push(decoded_to_image(&nchw)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

pub(crate) fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    // Empty-prompt first so it wins over the shared floor for a bare default request.
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!("{}: prompt is required", desc.id)));
    }
    // The shared capability floor (count, size range, negative/guidance/true_cfg, sampler, scheduler,
    // conditioning) — the same check chroma delegates to (F-100; this dedups flux2's near-verbatim
    // copy and adds the previously-missing scheduler validation).
    desc.capabilities.validate_request(desc.id, req)?;
    // FLUX.2-specific: latent dims must be a multiple of 16 (VAE 8× × patch 2).
    if !req.width.is_multiple_of(16) || !req.height.is_multiple_of(16) {
        return Err(Error::Msg(format!(
            "{}: width and height must be multiples of 16, got {}x{}",
            desc.id, req.width, req.height
        )));
    }
    Ok(())
}

// Link-time registration (epic 3720): the macro emits each `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`.
mlx_gen::register_generators! {
    descriptor_klein_9b => load_klein_9b,
    descriptor_klein_9b_edit => load_klein_9b_edit,
    descriptor_klein_9b_kv_edit => load_klein_9b_kv_edit,
    descriptor_dev => load_dev,
    descriptor_dev_edit => load_dev_edit,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        DEFAULT_GUIDANCE_DEV, DEFAULT_STEPS_DEV, FLUX2_DEV_EDIT_ID, FLUX2_DEV_ID,
        FLUX2_KLEIN_9B_EDIT_ID, FLUX2_KLEIN_9B_ID,
    };
    use mlx_gen::media::Image;
    use mlx_gen::Conditioning;

    /// L-log-injection: sanitize collapses embedded newlines/control chars (no second prefix line) and
    /// length-caps, so a model-generated rewrite can't break the machine-parsed `ENHANCED_PROMPT:` record.
    #[test]
    fn sanitize_log_text_collapses_and_caps() {
        let dirty = "a\nb\tc\r\nENHANCED_PROMPT:forged";
        let clean = sanitize_log_text(dirty);
        assert!(!clean.contains('\n') && !clean.contains('\t') && !clean.contains('\r'));
        assert_eq!(clean, "a b c ENHANCED_PROMPT:forged"); // newlines → spaces, but on ONE line
        let long = "x".repeat(1000);
        let capped = sanitize_log_text(&long);
        assert!(
            capped.chars().count() <= 513,
            "capped to ~512 chars + ellipsis"
        );
        assert!(capped.ends_with('…'));
        assert_eq!(sanitize_log_text("   "), "");
    }

    #[test]
    fn validates_basic_txt2img_request() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "a hummingbird".into(),
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    // ---- sc-2365 FLUX.2-dev T2I wiring ---------------------------------------------------------

    #[test]
    fn dev_descriptor_registered_with_t2i_caps() {
        // The dev variant is registered (loadable by id) with the dev id + txt2img/img2img caps.
        assert_eq!(descriptor_dev().id, FLUX2_DEV_ID);
        let d = descriptor_dev();
        assert!(d.capabilities.supports_guidance, "dev consumes guidance");
        assert!(
            !d.capabilities.supports_negative_prompt && !d.capabilities.supports_true_cfg,
            "dev is guidance-distilled, not true-CFG"
        );
        assert!(d.capabilities.mac_only);
        // A single Reference (img2img init), like klein txt2img — no edit conditioning.
        assert_eq!(
            d.capabilities.conditioning,
            vec![mlx_gen::ConditioningKind::Reference]
        );
    }

    #[test]
    fn dev_uses_embedded_guidance_with_dev_defaults() {
        assert!(Flux2Variant::Dev.uses_embedded_guidance());
        assert!(!Flux2Variant::Klein9b.uses_embedded_guidance());
        assert_eq!(Flux2Variant::Dev.default_steps(), DEFAULT_STEPS_DEV);
        assert_eq!(Flux2Variant::Dev.default_guidance(), DEFAULT_GUIDANCE_DEV);
    }

    #[test]
    fn dev_validates_basic_txt2img_request() {
        let model = Flux2::new_for_tests(Flux2Variant::Dev);
        let req = GenerationRequest {
            prompt: "a red fox in fresh snow".into(),
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    #[test]
    fn rejects_empty_prompt() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest::default();
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("prompt is required"));
    }

    #[test]
    fn rejects_unsupported_scheduler() {
        // F-100: flux2 delegated to the shared floor now validates the scheduler (was silently
        // accepted). epic 7114 scheduler axis: the curated names (e.g. "karras") + the "flow_match_euler"
        // native alias now pass; a genuinely unknown name is still rejected.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let err = model
            .validate(&GenerationRequest {
                prompt: "x".into(),
                scheduler: Some("not_a_real_scheduler".into()),
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported scheduler"), "got: {err}");
        for ok in ["flow_match_euler", "karras", "sgm_uniform"] {
            model
                .validate(&GenerationRequest {
                    prompt: "x".into(),
                    scheduler: Some(ok.into()),
                    ..Default::default()
                })
                .unwrap_or_else(|e| panic!("{ok} should validate: {e}"));
        }
    }

    #[test]
    fn rejects_non_multiple_of_16() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            width: 1023,
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("multiples of 16"));
    }

    #[test]
    fn txt2img_accepts_reference_conditioning() {
        // A `Reference` on the txt2img variant is an img2img init image (sc-2644).
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: Some(0.6),
            }],
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    #[test]
    fn txt2img_rejects_multiple_references() {
        // img2img conditions on exactly one init image; the resolver rejects more than one.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![
                Conditioning::Reference {
                    image: Image::default(),
                    strength: Some(0.6),
                },
                Conditioning::Reference {
                    image: Image::default(),
                    strength: Some(0.6),
                },
            ],
            ..Default::default()
        };
        let err = model.resolve_reference(&req).unwrap_err().to_string();
        assert!(err.contains("multiple reference images"));
    }

    #[test]
    fn edit_accepts_single_reference() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9bEdit);
        let req = GenerationRequest {
            prompt: "make it night".into(),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..Default::default()
        };
        model.validate(&req).unwrap();
        assert_eq!(model.collect_edit_references(&req).unwrap().len(), 1);
    }

    #[test]
    fn edit_accepts_multi_reference() {
        // sc-2645: N reference images via `MultiReference`, flattened in order.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9bEdit);
        let req = GenerationRequest {
            prompt: "combine these".into(),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![Image::default(), Image::default(), Image::default()],
            }],
            ..Default::default()
        };
        model.validate(&req).unwrap();
        assert_eq!(model.collect_edit_references(&req).unwrap().len(), 3);
    }

    #[test]
    fn edit_without_reference_errors() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9bEdit);
        let req = GenerationRequest {
            prompt: "make it night".into(),
            ..Default::default()
        };
        let err = model.collect_edit_references(&req).unwrap_err().to_string();
        assert!(err.contains("at least one reference image"));
    }

    // ---- sc-5919 FLUX.2-dev edit (DiT-concat reference conditioning) ---------------------------

    #[test]
    fn dev_edit_registered_with_edit_caps() {
        // Registered (loadable by id) with the dev-edit id + the klein edit conditioning surface.
        assert_eq!(descriptor_dev_edit().id, FLUX2_DEV_EDIT_ID);
        let caps = descriptor_dev_edit().capabilities;
        assert_eq!(
            caps.conditioning,
            vec![
                mlx_gen::ConditioningKind::Reference,
                mlx_gen::ConditioningKind::MultiReference,
            ]
        );
        // Embedded guidance (no negative/true-CFG), no KV cache, mac-only.
        assert!(
            caps.supports_guidance && !caps.supports_negative_prompt && !caps.supports_true_cfg
        );
        assert!(!caps.supports_kv_cache && caps.mac_only);
    }

    #[test]
    fn dev_edit_accepts_single_and_multi_reference() {
        let model = Flux2::new_for_tests(Flux2Variant::DevEdit);
        // Single `Reference`.
        let single = GenerationRequest {
            prompt: "make it a watercolor".into(),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..Default::default()
        };
        model.validate(&single).unwrap();
        assert_eq!(model.collect_edit_references(&single).unwrap().len(), 1);
        // `MultiReference` (N images).
        let multi = GenerationRequest {
            prompt: "combine these".into(),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![Image::default(), Image::default()],
            }],
            ..Default::default()
        };
        model.validate(&multi).unwrap();
        assert_eq!(model.collect_edit_references(&multi).unwrap().len(), 2);
    }

    #[test]
    fn dev_edit_without_reference_errors() {
        let model = Flux2::new_for_tests(Flux2Variant::DevEdit);
        let req = GenerationRequest {
            prompt: "make it night".into(),
            ..Default::default()
        };
        let err = model.collect_edit_references(&req).unwrap_err().to_string();
        assert!(err.contains("at least one reference image"));
    }

    #[test]
    fn txt2img_rejects_multi_reference() {
        // Multi-image editing belongs to the edit variant, not txt2img.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![Image::default(), Image::default()],
            }],
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("conditioning"));
    }

    #[test]
    fn generate_without_weights_errors_not_loaded() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            ..Default::default()
        };
        let mut progress = |_p: Progress| {};
        let err = model.generate(&req, &mut progress).unwrap_err().to_string();
        assert!(err.contains("not loaded"));
    }

    #[test]
    fn ids_match_expected() {
        assert_eq!(descriptor_klein_9b().id, FLUX2_KLEIN_9B_ID);
        assert_eq!(descriptor_klein_9b_edit().id, FLUX2_KLEIN_9B_EDIT_ID);
    }
}
