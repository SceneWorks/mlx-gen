//! Chroma provider registration + txt2img generation (sc-3839).
//!
//! Reuses the flux flow-match machinery (`build_linear_sigmas` for the raw `linspace(1,1/N,N)`;
//! `create_noise`/`unpack_latents`; the shared AutoencoderKL `Vae::decode`) and the core
//! `FlowMatchSampler` (Euler `x + v·Δσ`, `timestep(t)=σ`). Chroma's scheduler is **static-shift**
//! (`use_dynamic_shifting=false`, `σ' = shift·σ/(1+(shift-1)·σ)`), NOT FLUX's resolution-dependent
//! exp-shift, so the shift is applied here (see [`denoise`](Chroma::denoise)).
//! Chroma-specific: T5-only masked encode (sc-3838), the per-step **true CFG** (`neg + g·(pos−neg)`),
//! and the full-sequence MMDiT mask (text mask ++ image ones). The transformer runs f32 activations
//! over the bf16 weights (mlx promotes), matching a diffusers-bf16→f32 reference.
//!
//! Component residency (epic 10834; sc-10840): under [`OffloadPolicy::Sequential`] the T5-XXL text
//! encoder is dropped after the prompt encode so peak unified memory is bounded to
//! `max(T5, DiT+VAE)` instead of the sum; the default [`OffloadPolicy::Resident`] holds every
//! component warm and is byte-for-byte the pre-seam path. The staged encode → drop-T5 → denoise →
//! decode lifecycle is driven by the shared [`mlx_gen::Residency`] seam.

use mlx_gen::array::scalar;
use mlx_gen::image::decoded_to_image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, resolve_flow_schedule, run_flow_sampler, CancelFlag, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LatentDecoder, LoadSpec, ModelDescriptor, OffloadPolicy,
    Precision, Progress, Residency, Result, TimestepConvention, WeightsSource,
};
use mlx_gen_flux::{build_linear_sigmas, create_noise, unpack_latents, T5TextEncoder};
use mlx_gen_pid::{flow_capture_for_request, resolve_pid_decoder_at_sigma, PidEngine};
use mlx_gen_z_image::vae::Vae;
use mlx_rs::ops::{add, concatenate_axis, multiply, subtract};
use mlx_rs::Array;
use std::path::Path;

use crate::config::{ChromaTransformerConfig, ChromaVariant, DEFAULT_SAMPLER, HEUN_SAMPLER};
use crate::loader;
use crate::text::encode_prompt;
use crate::transformer::{ChromaTransformer, RopeTable};

pub fn descriptor_hd() -> ModelDescriptor {
    ChromaVariant::Hd.descriptor()
}

pub fn descriptor_base() -> ModelDescriptor {
    ChromaVariant::Base.descriptor()
}

pub fn descriptor_flash() -> ModelDescriptor {
    ChromaVariant::Flash.descriptor()
}

pub fn load_hd(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(load_chroma(ChromaVariant::Hd, spec)?))
}

pub fn load_base(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(load_chroma(ChromaVariant::Base, spec)?))
}

pub fn load_flash(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(load_chroma(ChromaVariant::Flash, spec)?))
}

/// Construct a [`Chroma`] generator from a [`LoadSpec`], honoring [`LoadSpec::offload_policy`]
/// (sc-10840). `Resident` (default) builds the T5 encoder + DiT + VAE now via [`build_residency`]
/// and holds them warm; `Sequential` keeps only the per-phase loader closures and re-loads per
/// generate in phase order (encode → drop the T5 encoder → denoise/decode) to bound peak memory to
/// `max(T5, DiT+VAE)`. Both use the same per-phase loaders, so the components are byte-identical.
pub fn load_chroma(variant: ChromaVariant, spec: &LoadSpec) -> Result<Chroma> {
    // Precision + snapshot-dir guard up front for BOTH policies (fail fast); the per-phase loaders
    // re-resolve from the (now validated) spec.
    resolve_root(variant, spec)?;
    // F-181: a `Sequential` + `spec.quantize` load over a *dense* snapshot re-quantizes the DiT on
    // every generate. An already-packed turnkey loads packed (no re-quant, `spec.quantize` is `None`);
    // `Resident` quantizes once. So warn only for the Sequential-over-dense combination that pays the
    // repeated cost.
    if let Some(q) = spec.quantize {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential) {
            mlx_gen::residency::warn_sequential_requantize(variant.id(), q.bits());
        }
    }
    Ok(Chroma {
        descriptor: variant.descriptor(),
        variant,
        residency: build_residency(variant, spec)?,
    })
}

/// The policy→[`Residency`] dispatch every Chroma variant shares (sc-10840), routed through the
/// single [`Residency::from_policy`] seam so no variant re-derives the `match offload_policy`.
/// `Resident` eager-loads the T5 encoder + heavy bundle now (the heavy loader with `use_pid = true`,
/// loading any PiD overlay once and reusing it); `Sequential` captures the two per-phase loaders and
/// loads nothing now, deferring each to [`Residency::run`]. Both use the same [`load_text_only`] /
/// [`load_heavy`], so the `Resident` composition is byte-identical to the pre-seam one. The deferral is
/// weight-free-testable: under `Sequential` this touches no component weights, so a dispatch that
/// ignored `offload_policy` would eager-load and fail the "Sequential defers" unit test.
pub(crate) fn build_residency(
    variant: ChromaVariant,
    spec: &LoadSpec,
) -> Result<Residency<ChromaTextOwned, ChromaHeavyOwned>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || load_text_only(variant, &spec_text),
        move |use_pid| load_heavy(variant, &spec_heavy, use_pid),
    )
}

/// Precision guard (only dense bf16 compute is wired) + snapshot-dir resolution (rejecting a
/// single-file source), shared by [`load_chroma`] and [`build_residency`]'s per-phase loaders.
fn resolve_root(variant: ChromaVariant, spec: &LoadSpec) -> Result<&Path> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{}: only bf16 compute is wired for the Chroma port (drop the precision override; Q4/Q8 \
             load via a pre-quantized packed tier or `spec.quantize`, orthogonal to precision)",
            variant.id()
        )));
    }
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(Error::Msg(format!(
            "{} expects a Chroma diffusers snapshot directory (tokenizer/ text_encoder/ \
             transformer/ vae/), not a single .safetensors file",
            variant.id()
        ))),
    }
}

/// Load the (bundled) tokenizer + the T5-XXL text encoder — the phase-A components dropped first under
/// `Sequential`. The tokenizer is small and rides along so a `Sequential` generate re-loads the two
/// together. Factored so the `Resident` and `Sequential` paths build byte-identical encoders.
fn load_text_only(variant: ChromaVariant, spec: &LoadSpec) -> Result<ChromaTextOwned> {
    let root = resolve_root(variant, spec)?;
    Ok(ChromaTextOwned {
        tokenizer: loader::load_tokenizer()?,
        t5: loader::load_t5_encoder(root)?,
    })
}

/// Load the heavy render-phase components — the DiT transformer (+ Q4/Q8 + LoRA/LoKr residuals), the
/// VAE, and the optional PiD overlay — everything but the T5 encoder. Factored so the `Sequential`
/// path loads these AFTER the encoder is dropped (bounding peak to `max(T5, DiT+VAE)`). The
/// quantize-then-adapters order matches the pre-seam composition; the components are independent of the
/// text encoder (separate weight files, deterministic RNG-free quant), so the `Resident` composition is
/// byte-identical.
fn load_heavy(variant: ChromaVariant, spec: &LoadSpec, load_pid: bool) -> Result<ChromaHeavyOwned> {
    let root = resolve_root(variant, spec)?;
    let cfg = ChromaTransformerConfig::default();
    let mut transformer = loader::load_transformer(root, cfg)?;
    let vae = loader::load_vae(root)?;

    // Q4/Q8 over the DiT's heavy block linears (sc-3841 / sc-8777). Two paths, both correct:
    //   * **Pre-quantized packed tier** (the hosted Q4/Q8 turnkeys): the block Linears load already
    //     quantized via `crate::quant::lin` (packed-detect on `{base}.scales`), so `spec.quantize` is
    //     `None` and this `.quantize()` is skipped — no dense transient.
    //   * **Dense snapshot + `spec.quantize`**: the block Linears load dense, then this `.quantize()`
    //     packs them in place (byte-identical to the packed tier). If a packed base ever reaches here
    //     it no-ops (`AdaptableLinear::quantize` only acts on a dense base).
    // T5/VAE stay f32 in every tier (their quant is a measurably-0% memory-only win and not wired).
    if let Some(q) = spec.quantize {
        transformer.quantize(q.bits())?;
    }
    // Install LoRA/LoKr adapters AFTER quantization (forward-time residual over the quantized base;
    // sc-3842). No-op when empty; any unmatched target errors loudly (never silently dropped).
    crate::adapters::apply_chroma_adapters(&mut transformer, &spec.adapters)?;

    // Optional PiD decoder overlay (epic 7840, sc-7846): Chroma's FLUX.1 16-ch VAE latent space has a
    // PiD student (the `flux` backbone), so the final decode can route through `mlx_gen_pid` when
    // `req.use_pid` is set. Loaded only when the spec carries `pid` AND this generate uses it
    // (`load_pid`, F-177): the Resident path passes `true` (loaded once, reused), the Sequential path
    // passes `req.use_pid` so a non-PiD generate skips the student + its caption encoder entirely.
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };

    Ok(ChromaHeavyOwned {
        transformer,
        vae,
        pid,
    })
}

/// Resolve the curated sampler **name** to drive [`run_flow_sampler`] with. An explicit `req.sampler`
/// wins (already gated by `validate_request` against the advertised curated menu); an unset sampler
/// resolves to the variant default — Flash distills toward the second-order **Heun** (sc-5392),
/// everything else to flow-match **Euler**. The legacy `flow_match` alias and any non-solver name fall
/// back to Euler inside [`run_flow_sampler`] (epic 7114 N3), so they need no special-case here.
fn resolve_sampler_name(variant: ChromaVariant, sampler: Option<&str>) -> &str {
    match sampler {
        Some(s) => s,
        None if matches!(variant, ChromaVariant::Flash) => HEUN_SAMPLER,
        None => DEFAULT_SAMPLER,
    }
}

/// The phase-A text components (tokenizer + T5-XXL encoder) dropped first under `Sequential`.
/// `pub(crate)` so [`build_residency`]'s loader closures name it.
pub(crate) struct ChromaTextOwned {
    tokenizer: TextTokenizer,
    t5: T5TextEncoder,
}

/// The heavy render-phase components (the DiT transformer, the VAE, and the optional PiD decoder) —
/// everything but the T5 encoder. Owned by the `Resident` components or by a `Sequential` generate.
pub(crate) struct ChromaHeavyOwned {
    transformer: ChromaTransformer,
    vae: Vae,
    /// Optional PiD super-resolving decoder overlay (epic 7840, sc-7846). `Some` only when the
    /// `LoadSpec` carried `pid`; selected per-generation by `req.use_pid`. Chroma reuses the FLUX.1
    /// 16-ch VAE (`mlx_gen_flux::load_vae`), so it shares the `flux` PiD student.
    pid: Option<PidEngine>,
}

pub struct Chroma {
    descriptor: ModelDescriptor,
    variant: ChromaVariant,
    /// Component-residency strategy (sc-10840), selected from [`LoadSpec::offload_policy`]. `Resident`
    /// (default) holds the T5 encoder + DiT + VAE warm for the whole job and across jobs; `Sequential`
    /// holds only the per-phase loader closures and re-loads per generation in phase order (encode →
    /// **drop the T5 encoder** → denoise/decode). The [`Residency`] seam owns the eval/drop/clear
    /// discipline, the stage-boundary cancel checks, and the error-safe cache flush.
    residency: Residency<ChromaTextOwned, ChromaHeavyOwned>,
}

/// PiD backbone (latent-space) tag for Chroma (epic 7840, sc-7846). Chroma is a FLUX.1-schnell
/// derivative whose VAE is loaded by `mlx_gen_flux::load_vae` (byte-identical FLUX.1 16-ch
/// `AutoencoderKL`, scale 0.3611 / shift 0.1159), so it resolves to the `flux` PiD student. Used only
/// at load time to build the [`PidEngine`]; shared by all three Chroma variants (hd/base/flash).
pub const PID_BACKBONE: &str = "flux";

/// FluxPosEmbed image position ids `[h2·w2, 3]` (axis 1 = row, axis 2 = col), row-major over the
/// packed `(height/16, width/16)` grid — diffusers `_prepare_latent_image_ids`.
fn latent_image_ids(h2: usize, w2: usize) -> Array {
    let mut data = vec![0f32; h2 * w2 * 3];
    for i in 0..h2 {
        for j in 0..w2 {
            let o = (i * w2 + j) * 3;
            data[o + 1] = i as f32;
            data[o + 2] = j as f32;
        }
    }
    Array::from_slice(&data, &[(h2 * w2) as i32, 3])
}

/// Text position ids `[L, 3]` — all zero (FluxPosEmbed places every text token at the origin).
fn zero_text_ids(l: usize) -> Array {
    Array::from_slice(&vec![0f32; l * 3], &[l as i32, 3])
}

impl Chroma {
    /// Borrow the warm-resident components as the historical `(tokenizer, t5, transformer, vae)`
    /// tuple — the accessor the public parity/test helpers (`denoise`, `decode`, `*_ref`) reach through
    /// under the default `Resident` policy. Under `Sequential` no components are held between generates,
    /// so this errors (those helpers drive `Resident`; the production `generate` path threads the
    /// components through the residency seam instead).
    fn parts(&self) -> Result<(&TextTokenizer, &T5TextEncoder, &ChromaTransformer, &Vae)> {
        let (text, heavy) = self.residency.resident_parts().ok_or_else(|| {
            Error::Msg(format!(
                "{}: components are not resident (Sequential offload holds no warm components between \
                 generates; the parity/test accessors require the default Resident policy)",
                self.descriptor.id
            ))
        })?;
        Ok((&text.tokenizer, &text.t5, &heavy.transformer, &heavy.vae))
    }

    /// The full-sequence MMDiT mask `[1, L + Si]` (0/1) = text mask ++ image ones.
    fn full_mask(text_mask: &Array, image_seq: i32) -> Result<Array> {
        let ones = Array::ones::<f32>(&[1, image_seq])?;
        Ok(concatenate_axis(&[text_mask, &ones], 1)?)
    }

    /// Run the true-CFG flow-match denoise from a given **packed** initial latent `[1, Si, 64]` →
    /// final packed latent with the **Euler** sampler. Public so the e2e parity test can inject the
    /// reference's initial latents (mlx and torch RNG differ); [`generate`](Self::generate) seeds it
    /// via `create_noise` and selects the sampler per [`resolve_sampler_name`].
    ///
    /// Thin wrapper over [`Self::denoise_with_sampler`] (the single source of truth for the
    /// encode/sigma/RoPE/mask/CFG setup) forcing Euler — the diffusers reference and every committed
    /// parity golden step with flow-match Euler. The production Flash **Heun** default (sc-5392) is
    /// gated same-backend via [`Self::denoise_with_sampler_name`] (no torch Heun reference exists;
    /// `ChromaPipeline` has no Heun scheduler). Drives the default `Resident` policy (the components are
    /// borrowed warm); a `Sequential`-loaded model errors in [`Self::parts`].
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        prompt: &str,
        negative: &str,
        width: u32,
        height: u32,
        steps: u32,
        guidance: f32,
        latents: Array,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        self.denoise_with_sampler(
            prompt,
            negative,
            width,
            height,
            steps,
            guidance,
            latents,
            DEFAULT_SAMPLER,
            None, // native schedule (parity / test helper path)
            0,
            cancel,
            on_progress,
        )
    }

    /// Build the flow-match σ schedule for a render (native + optional curated re-shape), factored out
    /// of [`Self::denoise_with_sampler`] so the `generate_impl` PiD `from_ldm` early-stop (sc-8048) can
    /// mint the decoder at the achieved capture σ and truncate the denoise — the schedule must be known
    /// where the decoder is resolved. Native schedule is the byte-exact default (epic 7114 N1): Base's
    /// beta re-spacing, or HD/Flash's static-shift linspace; a curated `scheduler_name` then re-shapes σ
    /// over the variant's static shift (HD `ln(3)`; Base/Flash `ln(1) = 0`).
    fn build_schedule(
        &self,
        width: u32,
        height: u32,
        steps: u32,
        scheduler_name: Option<&str>,
    ) -> Result<Vec<f32>> {
        let native = if self.variant.use_beta_sigmas() {
            crate::beta::base_sigmas(steps as usize)
        } else {
            let shift = self.variant.sigma_shift();
            let mut s = build_linear_sigmas(steps as usize, width, height, false)?;
            for v in s.iter_mut().take(steps as usize) {
                *v = shift * *v / (1.0 + (shift - 1.0) * *v);
            }
            s
        };
        let mu = self.variant.sigma_shift().ln();
        Ok(resolve_flow_schedule(
            scheduler_name,
            mu,
            steps as usize,
            &native,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn denoise_with_sampler(
        &self,
        prompt: &str,
        negative: &str,
        width: u32,
        height: u32,
        steps: u32,
        guidance: f32,
        latents: Array,
        sampler_name: &str,
        scheduler_name: Option<&str>,
        seed: u64,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let sigmas = self.build_schedule(width, height, steps, scheduler_name)?;
        self.denoise_with_schedule(
            prompt,
            negative,
            width,
            height,
            guidance,
            latents,
            sampler_name,
            sigmas,
            seed,
            cancel,
            on_progress,
        )
    }

    /// [`Self::denoise_with_sampler`] over a pre-built σ schedule — the parity/test helper the e2e
    /// suite drives (it re-encodes the prompt from the warm-resident T5 each call). The production
    /// `generate` path does not go through here: it encodes once in the residency seam's phase-A closure
    /// (so the T5 can be dropped under `Sequential`) and then calls [`Self::denoise_prepared`] with the
    /// hoisted embeds. Both share [`Self::denoise_prepared`], so the render is identical. Passing the
    /// full schedule is byte-identical to the prior inline path (chroma is flow-match `vp_frame=false`,
    /// so the schedule σ *is* the degrade σ); a *truncated* schedule stops the denoise at the achieved
    /// capture σ (sc-8048).
    #[allow(clippy::too_many_arguments)]
    fn denoise_with_schedule(
        &self,
        prompt: &str,
        negative: &str,
        width: u32,
        height: u32,
        guidance: f32,
        latents: Array,
        sampler_name: &str,
        sigmas: Vec<f32>,
        seed: u64,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let (tok, t5, tr, _) = self.parts()?;
        let encoded = encode_cfg(tok, t5, prompt, negative, guidance)?;
        self.denoise_prepared(
            tr,
            &encoded,
            width,
            height,
            guidance,
            latents,
            sampler_name,
            sigmas,
            seed,
            cancel,
            on_progress,
        )
    }

    /// The transformer-side denoise from pre-encoded conditioning — the single body shared by the
    /// public [`Self::denoise_with_schedule`] parity helper (which encodes from the warm-resident T5)
    /// and the production `generate` render closure (which received the phase-A embeds after the T5 was
    /// dropped). Builds the RoPE tables + full-sequence masks (needs the transformer) and runs the
    /// true-CFG denoise loop. The compiled-glue enable is scoped here (per call), matching the pre-seam
    /// per-image scope.
    #[allow(clippy::too_many_arguments)]
    fn denoise_prepared(
        &self,
        tr: &ChromaTransformer,
        encoded: &ChromaEncoded,
        width: u32,
        height: u32,
        guidance: f32,
        latents: Array,
        sampler_name: &str,
        sigmas: Vec<f32>,
        seed: u64,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let (pos_embeds, pos_mask) = (&encoded.pos_embeds, &encoded.pos_mask);

        let h2 = (height / 16) as usize;
        let w2 = (width / 16) as usize;
        let si = (h2 * w2) as i32;
        let img_ids = latent_image_ids(h2, w2);
        let txt_ids_pos = zero_text_ids(pos_embeds.shape()[1] as usize);
        let mask_pos = Self::full_mask(pos_mask, si)?;

        // Scoped compiled-glue enable (F-007): restored on drop instead of leaking the global on.
        let _compile_glue = crate::transformer::CompileGlueGuard::enable();

        let rope_pos = tr.build_rope_table(&txt_ids_pos, &img_ids)?;
        let mask_pos2d = ChromaTransformer::attention_mask2d(Some(&mask_pos))?;
        let neg_prepared = match &encoded.neg {
            Some((neg_embeds, neg_mask)) => {
                let txt_ids_neg = zero_text_ids(neg_embeds.shape()[1] as usize);
                let mask_neg = Self::full_mask(neg_mask, si)?;
                Some((
                    neg_embeds,
                    tr.build_rope_table(&txt_ids_neg, &img_ids)?,
                    ChromaTransformer::attention_mask2d(Some(&mask_neg))?,
                ))
            }
            None => None,
        };

        self.denoise_loop(
            tr,
            latents,
            sigmas,
            sampler_name,
            seed,
            guidance,
            pos_embeds,
            &rope_pos,
            mask_pos2d.as_ref(),
            neg_prepared.as_ref(),
            cancel,
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn denoise_loop(
        &self,
        tr: &ChromaTransformer,
        latents: Array,
        sigmas: Vec<f32>,
        sampler_name: &str,
        seed: u64,
        guidance: f32,
        pos_embeds: &Array,
        rope_pos: &RopeTable,
        mask_pos2d: Option<&Array>,
        neg_prepared: Option<&(&Array, RopeTable, Option<Array>)>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        // The true-CFG velocity field `v = neg + g·(pos − neg)` (single forward when CFG is off).
        // Chroma is rectified-flow (FLOW prediction), so the model is fed the raw schedule sigma as
        // its timestep ([`TimestepConvention::Sigma`]) and `predict` returns the velocity directly.
        let predict = |latents: &Array, sigma: f32| -> Result<Array> {
            let ts = Array::from_slice(&[sigma], &[1]);
            let pooled = tr.pooled_temb(&ts)?;
            let pos = tr.forward_prepared(latents, pos_embeds, &pooled, rope_pos, mask_pos2d)?;
            match neg_prepared {
                Some((neg_embeds, rope_neg, mask_neg2d)) => {
                    let neg = tr.forward_prepared(
                        latents,
                        neg_embeds,
                        &pooled,
                        rope_neg,
                        mask_neg2d.as_ref(),
                    )?;
                    Ok(add(
                        &neg,
                        &multiply(&subtract(&pos, &neg)?, scalar(guidance))?,
                    )?)
                }
                None => Ok(pos),
            }
        };

        // Route through the unified curated-sampler framework (epic 7114 P3): `euler` reproduces the
        // legacy flow-match step within the N1 parity tolerance, `heun` is the second-order refinement
        // (identical to the previous hand-rolled velocity-average arm — for FLOW the k-diffusion
        // derivative `d = (x − x0)/σ` equals the velocity `v`), and the rest of the curated menu
        // (dpmpp_2m / uni_pc / …) becomes available. Cancellation, the per-step `eval` (sc-5514 /
        // sc-5399 — bounds the lazy graph so a mid-render cancel lands within ~1 model eval), and
        // progress are handled inside `run_flow_sampler`.
        run_flow_sampler(
            Some(sampler_name),
            TimestepConvention::Sigma,
            &sigmas,
            latents,
            seed,
            cancel,
            on_progress,
            predict,
        )
    }

    /// Test accessors (real-weight e2e, sc-3839). Reach the warm-resident components (the e2e suite
    /// drives the default `Resident` policy); a `Sequential`-loaded model errors in [`Self::parts`].
    #[doc(hidden)]
    pub fn transformer_ref(&self) -> &ChromaTransformer {
        self.parts().expect("components resident").2
    }
    #[doc(hidden)]
    pub fn tokenizer_ref(&self) -> &TextTokenizer {
        self.parts().expect("components resident").0
    }
    #[doc(hidden)]
    pub fn t5_ref(&self) -> &T5TextEncoder {
        self.parts().expect("components resident").1
    }

    /// Test accessor (real-weight e2e, sc-6903): run the denoise with an explicit sampler **name**,
    /// routed through the production [`resolve_sampler_name`] selection. Lets the e2e gate
    /// drive the Flash **Heun** path that `generate` runs by default (sc-5392) but the Euler-forced
    /// [`Self::denoise`] goldens never exercise. `sampler = None` reproduces the variant default
    /// (Heun for Flash, Euler otherwise).
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_with_sampler_name(
        &self,
        prompt: &str,
        negative: &str,
        width: u32,
        height: u32,
        steps: u32,
        guidance: f32,
        latents: Array,
        sampler: Option<&str>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let name = resolve_sampler_name(self.variant, sampler);
        self.denoise_with_sampler(
            prompt,
            negative,
            width,
            height,
            steps,
            guidance,
            latents,
            name,
            None, // native schedule (e2e sampler-name accessor)
            0,
            cancel,
            on_progress,
        )
    }

    /// Unpack + decode a packed latent `[1, Si, 64]` → an [`Image`]. `decoder` overrides the native
    /// FLUX.1 VAE when `Some` — a PiD super-resolving decode (epic 7840, sc-7846) that consumes the
    /// same normalized latent and 4× upscales; `None` is the byte-exact VAE default (the warm-resident
    /// VAE, reached through [`Self::parts`] — the parity/test helper path).
    pub fn decode(
        &self,
        latents: &Array,
        width: u32,
        height: u32,
        decoder: Option<&dyn LatentDecoder>,
    ) -> Result<Image> {
        let (_, _, _, vae) = self.parts()?;
        Self::decode_with_vae(vae, latents, width, height, decoder)
    }

    /// Decode against an explicit VAE — the body shared by the public [`Self::decode`] (warm-resident
    /// VAE) and the `generate` render closure (the heavy bundle's just-loaded VAE under `Sequential`).
    fn decode_with_vae(
        vae: &Vae,
        latents: &Array,
        width: u32,
        height: u32,
        decoder: Option<&dyn LatentDecoder>,
    ) -> Result<Image> {
        let unpacked = unpack_latents(latents, width, height)?;
        let decoder: &dyn LatentDecoder = match decoder {
            Some(d) => d,
            None => vae,
        };
        let decoded = decoder
            .decode(&unpacked)?
            .as_dtype(mlx_rs::Dtype::Float32)?;
        decoded_to_image(&decoded)
    }
}

/// Pre-encoded T5 conditioning (the residency seam's phase-A output): the positive prompt embeds +
/// transformer mask, and — only when true CFG is active (`guidance > 1`) — the negative pair. The
/// masks are tokenizer-derived (host int arrays), so they carry no reference to the T5 weights; only
/// the embeds do, and the seam's `materialize` step evals those before the T5 is dropped.
pub(crate) struct ChromaEncoded {
    pos_embeds: Array,
    pos_mask: Array,
    neg: Option<(Array, Array)>,
}

/// Encode the positive (and, for `guidance > 1`, the negative) prompt into [`ChromaEncoded`] — the
/// phase-A body shared by the production `generate` (its residency phase-A closure) and the public
/// parity helpers via [`Chroma::denoise_with_schedule`]. Seed-independent, so hoisting it out of the
/// per-image loop (as `generate` now does) is byte-identical to the pre-seam per-image re-encode.
fn encode_cfg(
    tok: &TextTokenizer,
    t5: &T5TextEncoder,
    prompt: &str,
    negative: &str,
    guidance: f32,
) -> Result<ChromaEncoded> {
    let (pos_embeds, pos_mask) = encode_prompt(tok, t5, prompt)?;
    let neg = if guidance > 1.0 {
        Some(encode_prompt(tok, t5, negative)?)
    } else {
        None
    };
    Ok(ChromaEncoded {
        pos_embeds,
        pos_mask,
        neg,
    })
}

mlx_gen::impl_generator!(Chroma {
    validate: |s, req| s.validate_impl(req),
    generate: generate_impl,
});

impl Chroma {
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        if req.prompt.trim().is_empty() {
            return Err(Error::Msg(format!(
                "{}: prompt must not be empty",
                self.descriptor.id
            )));
        }
        if !req.width.is_multiple_of(16) || !req.height.is_multiple_of(16) {
            return Err(Error::Msg(format!(
                "{}: width and height must be multiples of 16, got {}x{}",
                self.descriptor.id, req.width, req.height
            )));
        }
        // `base_sigmas`/`build_linear_sigmas` clamp `steps.max(1)`, so steps==0 would silently run one
        // denoise step instead of erroring (sibling families like Kolors reject it). Match them (F-074).
        if req.steps == Some(0) {
            return Err(Error::Msg(format!(
                "{}: steps must be >= 1",
                self.descriptor.id
            )));
        }
        Ok(())
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let steps = req.steps.unwrap_or_else(|| self.variant.default_steps());
        let guidance = req
            .true_cfg
            .unwrap_or_else(|| self.variant.default_true_cfg());
        let negative = req.negative_prompt.as_deref().unwrap_or("").to_string();
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let name = resolve_sampler_name(self.variant, req.sampler.as_deref());
        // Build the flow-match σ schedule once — it is seed-independent (same steps/dims/scheduler for
        // every image), and the PiD `from_ldm` early-stop below needs it to mint the decoder at the
        // achieved capture σ and truncate the denoise.
        let sigmas = self.build_schedule(req.width, req.height, steps, req.scheduler.as_deref())?;

        // Staged residency lifecycle (sc-10840): under `Sequential` the seam loads the T5 encoder,
        // encodes pos (+neg), materializes, then DROPS it + `clear_cache()` so it frees before the
        // DiT/VAE load below — the peak-bounding win. Under `Resident` it borrows the warm encoder and
        // runs the identical encode/denoise/decode with no eval/clear. The prompt encode is
        // seed-independent, so hoisting it here (out of the per-image loop) is byte-identical to the
        // pre-seam per-image re-encode.
        self.residency.run(
            &req.cancel,
            req.use_pid,
            on_progress,
            // ── Phase A: prompt → T5 conditioning (pos + optional neg).
            |text: &ChromaTextOwned| {
                encode_cfg(&text.tokenizer, &text.t5, &req.prompt, &negative, guidance)
            },
            // Materialize the embeds while the T5 is still alive (Sequential only) — MLX is lazy, so an
            // un-evaluated embed keeps the encoder referenced through the graph and the drop would free
            // nothing. The masks are tokenizer-derived (independent of T5), so evaling the embeds is
            // sufficient.
            |encoded| {
                match &encoded.neg {
                    Some((neg_embeds, _)) => {
                        mlx_rs::transforms::eval([&encoded.pos_embeds, neg_embeds])?
                    }
                    None => mlx_rs::transforms::eval([&encoded.pos_embeds])?,
                }
                Ok(())
            },
            // ── Phase B: denoise/decode from the heavy bundle. Runs identically for both residencies.
            |heavy, encoded, on_progress| {
                // PiD decode overlay (epic 7840, sc-7846) + `from_ldm` early-stop (sc-8048): Chroma
                // shares the FLUX.1 VAE latent space (the `flux` student), so the decode can route
                // through PiD when `use_pid` is set + an overlay was loaded; errors loudly if requested
                // without one. `None` → the native VAE. When `pid_capture_sigma` asks for an early exit
                // on this flow-match schedule (Chroma is `vp_frame=false`, so the schedule σ *is* the
                // degrade σ), stop the denoise at the achieved-σ step and hand PiD the partially-denoised
                // x_k; else the clean σ=0 full-denoise path (`capture_sigma = 0`, full schedule).
                // txt2img here → `start_step = 0`. Shared across the count loop (same prompt → same
                // caption embeds); per-image variation is the per-seed denoised latent.
                let (capture_sigma, keep) = flow_capture_for_request(req, &sigmas, 0);
                let pid_decoder = resolve_pid_decoder_at_sigma(
                    heavy.pid.as_ref(),
                    req,
                    base_seed,
                    self.descriptor.id,
                    capture_sigma,
                )?;
                let denoise_sigmas = &sigmas[..keep];

                let mut images = Vec::with_capacity(req.count as usize);
                for i in 0..req.count {
                    // Cancel between images too, so a multi-image batch stops promptly (F-096).
                    if req.cancel.is_cancelled() {
                        return Err(Error::Canceled);
                    }
                    let seed = base_seed.wrapping_add(i as u64);
                    let latents = create_noise(seed, req.width, req.height)?;
                    let final_latents = self.denoise_prepared(
                        &heavy.transformer,
                        &encoded,
                        req.width,
                        req.height,
                        guidance,
                        latents,
                        name,
                        denoise_sigmas.to_vec(),
                        seed,
                        &req.cancel,
                        on_progress,
                    )?;
                    on_progress(Progress::Decoding);
                    images.push(Self::decode_with_vae(
                        &heavy.vae,
                        &final_latents,
                        req.width,
                        req.height,
                        pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder),
                    )?);
                }
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

// Link-time registration (epic 3720): the macro emits each `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`.
/// Per-component on-disk footprint (sc-10894) for the MLX fit-gate's staged-residency split — the
/// T5-XXL text encoder (`text_encoder/` — Chroma keeps T5 here, not flux's `text_encoder_2/`), the DiT
/// (`transformer/`), and the VAE (`vae/`), summed from the exact snapshot subdirs [`crate::loader`]
/// loads. Shared by chroma hd/base/flash.
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder"],
        &["transformer"],
        &["vae"],
    )
}

mlx_gen::register_generators! {
    descriptor_hd => load_hd,
    descriptor_base => load_base,
    descriptor_flash => load_flash,
    ; footprint = component_footprint
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core;

    /// A Chroma with no loadable components — enough to exercise the request-boundary paths
    /// (`validate`, pre-run cancellation) that run before any tensor is touched. The residency is
    /// `Sequential` over loader closures that would error if invoked, but the pre-run cancel check in
    /// `Residency::run` returns before either fires (and the `*_ref`/`denoise` accessors are the only
    /// other callers — untested here).
    fn weightless(variant: ChromaVariant) -> Chroma {
        Chroma {
            descriptor: variant.descriptor(),
            variant,
            residency: Residency::sequential(
                || Err(Error::Msg("weightless: text encoder not loadable".into())),
                |_use_pid| Err(Error::Msg("weightless: heavy bundle not loadable".into())),
            ),
        }
    }

    #[test]
    fn generate_honors_pre_cancellation() {
        // F-096: an already-cancelled request must abort before any forward, returning the typed
        // `Error::Canceled` (displays as "cancelled") the cancellation contract mandates — epic 3720
        // / sc-4481. The residency seam's first action is a cancel check, so this returns before either
        // per-phase loader is invoked (no loaded weights needed).
        let model = weightless(ChromaVariant::Hd);
        let req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        req.cancel.cancel();
        let mut nop = |_p: Progress| {};
        let err = model.generate(&req, &mut nop).unwrap_err();
        assert!(matches!(err, gen_core::Error::Canceled), "got: {err}");
    }

    // ── sc-10840: weight-free, default-run proof that Chroma's dispatch HONORS `offload_policy`.
    // `build_residency` points at a non-existent snapshot *directory* (so the up-front precision/
    // single-file guard passes) and the discriminator is deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the T5 encoder from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident`) would eager-load under a `Sequential`
    // request and fail the first assertion. The A/B real-weight test is `#[ignore]`d; this runs by
    // default.
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/chroma-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        let res = build_residency(
            ChromaVariant::Hd,
            &missing_snapshot_spec(OffloadPolicy::Sequential),
        )
        .expect("Sequential must defer loads and not touch the (missing) snapshot dir");
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential (deferred) residency"
        );
    }

    #[test]
    fn build_residency_resident_eager_loads_and_fails_on_missing_snapshot() {
        let err = build_residency(
            ChromaVariant::Hd,
            &missing_snapshot_spec(OffloadPolicy::Resident),
        )
        .err()
        .expect("Resident must eager-load and fail on a missing snapshot dir");
        let msg = err.to_string();
        assert!(
            !msg.contains("single .safetensors file") && !msg.contains("precision override"),
            "expected an eager-load failure, not the up-front guard: {msg}"
        );
    }
}
