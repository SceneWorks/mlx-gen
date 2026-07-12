//! PuLID-FLUX end-to-end generate (sc-3074) + native face wiring (sc-3073).
//!
//! Assembles the full face-identity path on top of the FLUX.1-dev backbone:
//!   1. **Face analysis** (native MLX, epic 3079): the reference face (`Conditioning::Reference`) →
//!      `mlx_gen_face::FaceAnalysis` → largest face's ArcFace embedding (512-d) + `face_features_image`
//!      (512² aligned, background-whitened grayscale). No Python/onnx.
//!   2. **EVA-CLIP** (sc-3070): `face_features_image` → resize/normalize → `id_cond_vit` (768-d,
//!      L2-normalized) + 5 hidden states.
//!   3. **IDFormer** (sc-3071): `id_cond = cat(arcface 512, id_cond_vit 768)` + hidden → `id_embedding`
//!      [1,32,2048].
//!   4. **CA injection** (sc-3072): build `PulidCa` and run the FLUX flow-match denoise through
//!      `Flux1::generate_with_injector` (fake-CFG, true_cfg=1.0) → AE decode.
//!
//! The whole conditioning path runs in **f32**: mlx-gen-flux keeps the DiT image stream in f32 (mixed
//! precision, sc-2787), so f32 CA weights/id_embedding inject cleanly into the f32 hidden tokens (no
//! dtype mismatch) and at higher accuracy than the reference's bf16 — the e2e gate is ArcFace-cosine
//! (cross-encoder, loose), so this is strictly safe. Real-CFG / uncond-id is sc-3075; quant is sc-3076.

use std::path::{Path, PathBuf};

use mlx_rs::ops::{concatenate_axis, divide, maximum, sqrt, square, sum_axes};
use mlx_rs::{Array, Dtype};

use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, gen_core, Capabilities, Conditioning,
    ConditioningKind, Error, GenerationOutput, GenerationRequest, Generator, IdentityWeights,
    LoadSpec, Modality, ModelDescriptor, Progress, Quant, Result, WeightsSource,
};
use mlx_gen_face::FaceAnalysis;
use mlx_gen_flux::config::FluxVariant;
use mlx_gen_flux::model::{load_flux1, Flux1};

use crate::ca::PulidCa;
use crate::eva_clip::{transform, EvaConfig, EvaVisionTransformer};
use crate::idformer::{IdFormer, IdFormerConfig};

/// FLUX.1-dev DiT block counts (the PuLID injection schedule is defined over these).
const NUM_DOUBLE_BLOCKS: usize = 19;
const NUM_SINGLE_BLOCKS: usize = 38;
/// Default step from which the real-CFG (and uncond-id) branch engages when the request leaves
/// `timestep_to_start_cfg` unset — the upstream PuLID default (the photoreal preset overrides to 4
/// via `req.timestep_to_start_cfg`).
const DEFAULT_TIMESTEP_TO_START_CFG: usize = 1;
/// ArcFace (antelopev2) face embedding width — the first half of the IdFormer `id_cond`
/// (`cat(arcface, id_cond_vit)`). The id_cond_vit half is the EVA head's `proj_dim`.
const ARCFACE_DIM: i32 = 512;

pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: "pulid_flux",
        family: "pulid",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // PuLID drives its OWN real-CFG (sc-3075): `generate` reads `req.true_cfg` (>1 enables the
            // pos/neg identity branches) and `req.negative_prompt` (the negative branch text). The
            // descriptor must advertise what `generate` actually honors, else the shared floor (which
            // `validate` now delegates to, F-026) would reject the very request that drives the feature.
            supports_negative_prompt: true, // honored in the real-CFG branch (sc-3075)
            supports_guidance: true,        // FLUX.1-dev guidance-distilled CFG (default ~4.0)
            supports_true_cfg: true, // >1 enables real-CFG pos/neg identity branches (sc-3075)
            conditioning: vec![ConditioningKind::Reference], // the reference face
            supports_lora: false,
            supports_lokr: false,
            // Epic 7114 (sc-7297): PuLID delegates its denoise to the FLUX.1-dev backbone
            // (`generate_with_injector_cfg` → `run_denoise` → `run_flow_sampler`), which already
            // honors the full curated integrator menu over the flow-match σ schedule AND the curated
            // scheduler axis (it threads `req.sampler` / `req.scheduler`). The descriptor previously
            // advertised only `flow_match`, silently N3-dropping every curated name; mirror the
            // backbone's menu so the curated samplers are selectable. `hyper` is intentionally NOT
            // advertised — it needs the dev Hyper-FLUX LoRA loaded at scale, which PuLID does not load.
            // Pure advertised-name change: the curated names route through the SAME, already-tested
            // FLUX flow denoise (no new code path, parity by construction). `flow_match` kept as the
            // legacy default alias (== Euler); `linear` kept as the native default scheduler.
            samplers: {
                let mut s = curated_sampler_names();
                s.push("flow_match");
                s
            },
            schedulers: {
                let mut s = curated_scheduler_names();
                s.push("linear");
                s
            },
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: true, // dev
            // Not wired onto the shared `Residency` seam (F-176); Sequential is a no-op fallback.
            supports_sequential_offload: false,
            supported_quants: &[Quant::Q4, Quant::Q8],
        },
    }
}

/// L2-normalize each row of `[B, D]` over the feature axis (the PuLID `id_cond_vit` normalization).
fn l2_normalize_rows(x: &Array) -> Result<Array> {
    let sumsq = sum_axes(&square(x)?, &[1], true)?; // [B, 1]
                                                    // Clamp the norm to a tiny epsilon (torch `F.normalize`'s default eps) so a degenerate zero-norm
                                                    // row — e.g. a bad ArcFace crop — yields a zero vector instead of NaN-poisoning the entire
                                                    // identity-conditioned generation. Byte-identical for real embeddings (norm >> eps) (F-078).
    let norm = maximum(&sqrt(&sumsq)?, Array::from_f32(1e-12))?;
    Ok(divide(x, &norm)?)
}

pub struct PulidFlux {
    descriptor: ModelDescriptor,
    flux: Flux1,
    eva: EvaVisionTransformer,
    idformer: IdFormer,
    /// The PuLID checkpoint weights (f32) — kept to build a per-generate [`PulidCa`] bound to the
    /// computed id_embedding. `pulid_encoder.*` already consumed by `idformer`; `pulid_ca.*` here.
    pulid: Weights,
    face: FaceAnalysis,
}

impl PulidFlux {
    /// Build from already-loaded sub-models. `pulid` must hold both `pulid_encoder.*` and
    /// `pulid_ca.*` (cast to f32); `eva`/`idformer` must likewise be f32 (the conditioning path).
    /// `face` must have a parser attached (`with_parser`) for `face_features_image`.
    pub fn new(
        flux: Flux1,
        eva: EvaVisionTransformer,
        pulid: Weights,
        face: FaceAnalysis,
    ) -> Result<Self> {
        let idformer = IdFormer::from_weights(&pulid, "pulid_encoder", IdFormerConfig::default())?;
        Ok(Self {
            descriptor: descriptor(),
            flux,
            eva,
            idformer,
            pulid,
            face,
        })
    }

    /// Face image (RGB, row-major, `h×w`) → `id_embedding` `[1,32,2048]`. Mirrors PuLID's
    /// `get_id_embedding` (the conditional side; cal_uncond is sc-3075).
    pub fn compute_id_embedding(&self, pixels: &[u8], h: usize, w: usize) -> Result<Array> {
        let faces = self.face.analyze(pixels, h, w)?;
        let face = faces.first().ok_or_else(|| {
            Error::Msg("pulid_flux: no face detected in the reference image".into())
        })?;
        // ArcFace 512-d (id_ante_embedding) — raw, un-normalized, matching the reference.
        let arcface = Array::from_slice(&face.embedding, &[1, face.embedding.len() as i32]);
        // face_features_image (512² aligned, bg-whitened gray) → EVA 336² transform → tower.
        let ffi = self.face.face_features_image(pixels, h, w, face)?;
        let eva_in = transform::eva_transform(&ffi, self.eva_image_size())?;
        let eva_out = self.eva.forward(&eva_in)?;
        let id_cond_vit = l2_normalize_rows(&eva_out.id_cond_vit)?; // [1,768]
        let id_cond = concatenate_axis(&[&arcface, &id_cond_vit], 1)?; // [1,1280]
        self.idformer.forward(&id_cond, &eva_out.hidden)
    }

    /// The unconditional id_embedding — IDFormer over **zeroed** id_cond + zeroed hidden states (the
    /// PuLID `get_id_embedding(cal_uncond=True)` path), injected on the negative real-CFG branch.
    pub fn compute_uncond_id_embedding(&self) -> Result<Array> {
        // Derive the EVA token geometry from the loaded tower's config, not the default-tower
        // constants (F-082): seq = grid²+1 (CLS + patches) = 577, embed = embed_dim = 1024, and one
        // zeroed hidden state per captured block (5). `id_cond` is the IdFormer input width (ArcFace
        // 512 + the EVA head's proj_dim 768).
        let cfg = self.eva.config();
        let seq = cfg.grid() * cfg.grid() + 1;
        let embed = cfg.embed_dim;
        let id_cond_dim = ARCFACE_DIM + cfg.proj_dim;
        let id_cond = Array::from_slice(&vec![0f32; id_cond_dim as usize], &[1, id_cond_dim]);
        let hidden: Vec<Array> = cfg
            .hidden_capture
            .iter()
            .map(|_| Array::from_slice(&vec![0f32; (seq * embed) as usize], &[1, seq, embed]))
            .collect();
        self.idformer.forward(&id_cond, &hidden)
    }

    fn eva_image_size(&self) -> i32 {
        self.eva.config().image_size
    }

    fn reference_face<'a>(&self, req: &'a GenerationRequest) -> Result<(&'a Image, f32)> {
        select_reference_face(&req.conditioning)
    }
}

/// Pick the single reference-face conditioning, rejecting any other kind. PuLID-FLUX advertises only
/// `ConditioningKind::Reference`, so a stray Control/Mask/etc. attached by a worker must **error**
/// rather than be silently dropped when `generate` clears `flux_req.conditioning` (F-094).
fn select_reference_face(conditioning: &[Conditioning]) -> Result<(&Image, f32)> {
    let mut found = None;
    for c in conditioning {
        match c {
            Conditioning::Reference { image, strength } => {
                if found.is_some() {
                    return Err(Error::Msg(
                        "pulid_flux: exactly one reference face is supported".into(),
                    ));
                }
                // The reference strength is the PuLID id_weight (0–3, default 1.0).
                found = Some((image, strength.unwrap_or(1.0)));
            }
            other => {
                return Err(Error::Msg(format!(
                    "pulid_flux: unsupported conditioning {:?} — only a reference face \
                     (Conditioning::Reference) is accepted",
                    other.kind()
                )));
            }
        }
    }
    found.ok_or_else(|| {
        Error::Msg(
            "pulid_flux: a reference face image (Conditioning::Reference) is required".into(),
        )
    })
}

impl Generator for PulidFlux {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Validate against PuLID's OWN descriptor, not the FLUX-dev backbone's (F-026). `generate`
        // delegates the denoise to `PulidFlux::flux` (the dev backbone), whose descriptor advertises
        // the `hyper` sampler — but PuLID deliberately does NOT load the dev Hyper LoRA, so a `hyper`
        // request would silently render WITHOUT it. PuLID's descriptor omits `hyper`, so delegating to
        // its own capability floor rejects that (and every other un-advertised sampler/scheduler,
        // negative_prompt, true_cfg, size/count/steps) instead of the backbone waving it through. The
        // `?` keeps the typed `Error::Unsupported` for capability gaps.
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        // PuLID-specific: a reference face image is required (consumed into the identity injector).
        self.reference_face(req)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

impl PulidFlux {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let (image, id_weight) = self.reference_face(req)?;
        let id_embedding =
            self.compute_id_embedding(&image.pixels, image.height as usize, image.width as usize)?;
        let mk_ca = |emb: Array| {
            PulidCa::from_weights(
                &self.pulid,
                "pulid_ca",
                emb,
                id_weight,
                NUM_DOUBLE_BLOCKS,
                NUM_SINGLE_BLOCKS,
            )
        };
        // The reference face is consumed into the injector; hand the FLUX backbone a plain request
        // (it rejects conditioning + negative_prompt it doesn't itself implement — both are handled
        // here / passed to the CFG denoise directly).
        let mut flux_req = req.clone();
        flux_req.conditioning = Vec::new();
        flux_req.negative_prompt = None;
        flux_req.true_cfg = None; // PuLID drives real-CFG itself; the backbone forbids it

        let true_cfg = req.true_cfg.unwrap_or(1.0);
        if true_cfg > 1.0 + 1e-3 {
            // Real-CFG (sc-3075): positive (id) + negative (uncond id) branches + a negative prompt.
            let pos = mk_ca(id_embedding)?;
            let neg = mk_ca(self.compute_uncond_id_embedding()?)?;
            let neg_prompt = req.negative_prompt.as_deref().unwrap_or("");
            let start_cfg = req
                .timestep_to_start_cfg
                .map(|v| v as usize)
                .unwrap_or(DEFAULT_TIMESTEP_TO_START_CFG);
            self.flux.generate_with_injector_cfg(
                &flux_req,
                &pos,
                &neg,
                neg_prompt,
                true_cfg,
                start_cfg,
                on_progress,
            )
        } else {
            // Fake-CFG (true_cfg = 1.0): single forward (sc-3074), bit-identical to that path.
            self.flux
                .generate_with_injector(&flux_req, Some(&mk_ca(id_embedding)?), on_progress)
        }
    }
}

// ---- registration -------------------------------------------------------------------------------

/// Resolve a required file path from an env var, erroring with the var name if unset/missing.
fn env_path(var: &str) -> Result<PathBuf> {
    let p = std::env::var(var)
        .map_err(|_| Error::Msg(format!("pulid_flux: set {var} to the weights path")))?;
    let p = PathBuf::from(p);
    if !p.exists() {
        return Err(Error::Msg(format!(
            "pulid_flux: {var} path does not exist: {}",
            p.display()
        )));
    }
    Ok(p)
}

/// Extract the `PathBuf` from a spec-supplied [`WeightsSource`] override, validating existence with a
/// clear message (F-114). Accepts either `File` or `Dir` — the caller knows which it expects.
fn spec_path(src: &WeightsSource, what: &str) -> Result<PathBuf> {
    let p = match src {
        WeightsSource::File(p) | WeightsSource::Dir(p) => p.clone(),
    };
    if !p.exists() {
        return Err(Error::Msg(format!(
            "pulid_flux: LoadSpec identity {what} path does not exist: {}",
            p.display()
        )));
    }
    Ok(p)
}

/// Resolve an identity sub-model path: prefer the `LoadSpec::identity` override (F-114), else fall
/// back to the historical `PULID_*` env var (its HF-cache-glob resolver, for `encoder`, is handled by
/// [`resolve_pulid_weights`] separately).
fn resolve_identity_path(
    override_src: Option<&WeightsSource>,
    var: &str,
    what: &str,
) -> Result<PathBuf> {
    match override_src {
        Some(src) => spec_path(src, what),
        None => env_path(var),
    }
}

/// Locate `pulid_flux_v0.9.1.safetensors` — the `LoadSpec::identity.encoder` override (F-114), else
/// `PULID_FLUX_WEIGHTS`, else the HF cache.
fn resolve_pulid_weights(override_src: Option<&WeightsSource>) -> Result<PathBuf> {
    // Prefer an explicit spec override; then the env var; then the HF-cache glob.
    if let Some(src) = override_src {
        return spec_path(src, "encoder");
    }
    // Route the override through `env_path` so a typo'd path errors with the var name up front,
    // matching the sibling weight-path helpers (F-093), instead of a bare later I/O error.
    if std::env::var_os("PULID_FLUX_WEIGHTS").is_some() {
        return env_path("PULID_FLUX_WEIGHTS");
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let glob = format!("{home}/.cache/huggingface/hub/models--guozinan--PuLID/snapshots");
    let snaps = std::fs::read_dir(&glob).map_err(|e| {
        Error::Msg(format!(
            "pulid_flux: no PuLID cache ({glob}): {e}; set PULID_FLUX_WEIGHTS"
        ))
    })?;
    for s in snaps.flatten() {
        let cand = s.path().join("pulid_flux_v0.9.1.safetensors");
        if cand.exists() {
            return Ok(cand);
        }
    }
    Err(Error::Msg(
        "pulid_flux: pulid_flux_v0.9.1.safetensors not found; set PULID_FLUX_WEIGHTS".into(),
    ))
}

/// Load EVA weights (f32) from a converted safetensors (tools/convert_eva_clip.py output). Keys are
/// bare mlx-names (no prefix).
fn load_eva(path: &Path) -> Result<EvaVisionTransformer> {
    let mut w = Weights::from_file(path)?;
    w.cast_all(Dtype::Float32)?;
    EvaVisionTransformer::from_weights(&w, "", EvaConfig::default())
}

/// Registered loader for the `pulid_flux` target. Weight sources (each identity sub-model prefers the
/// structured `LoadSpec::identity` override, falling back to its historical env var — F-114):
///   * FLUX.1-dev snapshot dir — `spec.weights` (Dir).
///   * identity encoder — `spec.identity.encoder`, else `PULID_FLUX_WEIGHTS` (else HF cache).
///   * EVA tower — `spec.identity.eva`, else `PULID_EVA_WEIGHTS` (converted EVA02-CLIP-L-14-336).
///   * face dir — `spec.identity.face_dir`, else `PULID_FACE_WEIGHTS_DIR` (scrfd_10g /
///     arcface_iresnet100 / bisenet_parsing).
pub fn load_pulid_flux(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    // F-114: the identity sub-model paths ride the spec when present; env vars are the fallback.
    let id = spec.identity.clone().unwrap_or_default();
    let IdentityWeights {
        encoder,
        eva,
        face_dir,
    } = id;
    // FLUX.1-dev backbone (its loader validates the snapshot dir). Q8/Q4 (sc-3076) composes for free:
    // `spec.quantize` flows through `load_flux1`, quantizing ONLY the FLUX backbone linears. The PuLID
    // conditioning (EVA tower, IDFormer, the 20 CA modules) stays f32 — it runs once per image, not
    // per step, so the memory win is the backbone, and the f32 CA residual injects into the (still
    // f32) DiT image stream unchanged. No quant-specific wiring needed here.
    let flux = load_flux1(FluxVariant::Dev, spec)?;

    // PuLID encoder + CA weights, cast f32 (conditioning path).
    let mut pulid = Weights::from_file(resolve_pulid_weights(encoder.as_ref())?)?;
    pulid.cast_all(Dtype::Float32)?;

    // EVA-CLIP tower (f32).
    let eva = load_eva(&resolve_identity_path(
        eva.as_ref(),
        "PULID_EVA_WEIGHTS",
        "eva",
    )?)?;

    // Native face stack.
    let face_dir = resolve_identity_path(face_dir.as_ref(), "PULID_FACE_WEIGHTS_DIR", "face_dir")?;
    let face = FaceAnalysis::load(
        &Weights::from_file(face_dir.join("scrfd_10g.safetensors"))?,
        &Weights::from_file(face_dir.join("arcface_iresnet100.safetensors"))?,
    )?
    .with_parser(&Weights::from_file(
        face_dir.join("bisenet_parsing.safetensors"),
    )?)?;

    Ok(Box::new(PulidFlux::new(flux, eva, pulid, face)?))
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!` and bridges the
// crate's rich `Result` into the registry's backend-neutral `gen_core::Result`. The `impl
// Generator` above stays hand-written because `validate` adds a reference-face check beyond the
// FLUX backbone's, so it is not the plain delegation `impl_generator!` expresses.
mlx_gen::register_generators! { descriptor => load_pulid_flux }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pulid_weights_override_validates_existence() {
        // F-093: a set-but-nonexistent PULID_FLUX_WEIGHTS errors up front with the var name (routed
        // through env_path), not a bare later I/O error. (RUST_TEST_THREADS=1 makes the env mutation
        // safe — see .cargo/config.toml.)
        std::env::set_var(
            "PULID_FLUX_WEIGHTS",
            "/nonexistent/pulid_flux_v0.9.1.safetensors",
        );
        let err = resolve_pulid_weights(None).unwrap_err().to_string();
        std::env::remove_var("PULID_FLUX_WEIGHTS");
        assert!(err.contains("PULID_FLUX_WEIGHTS"), "got: {err}");
        assert!(err.contains("does not exist"), "got: {err}");
    }

    #[test]
    fn pulid_identity_spec_override_takes_precedence() {
        // F-114: a `LoadSpec::identity.encoder` override is preferred over the env var, and a
        // nonexistent override path errors with the spec-side message (not the env var name).
        let src = WeightsSource::File("/nonexistent/spec_pulid.safetensors".into());
        let err = resolve_pulid_weights(Some(&src)).unwrap_err().to_string();
        assert!(err.contains("LoadSpec identity encoder"), "got: {err}");
        assert!(err.contains("does not exist"), "got: {err}");
    }

    fn img() -> Image {
        Image {
            width: 1,
            height: 1,
            pixels: vec![0u8; 3],
        }
    }

    /// F-094: a single reference face is accepted (its strength becomes the id_weight, default 1.0).
    #[test]
    fn select_reference_accepts_single_reference() {
        let cond = vec![Conditioning::Reference {
            image: img(),
            strength: Some(2.0),
        }];
        let (_, weight) = select_reference_face(&cond).unwrap();
        assert_eq!(weight, 2.0);
    }

    /// F-094: a non-Reference conditioning (e.g. a stray Mask/Control) is rejected rather than
    /// silently dropped when `generate` clears the conditioning.
    #[test]
    fn select_reference_rejects_unsupported_conditioning() {
        let cond = vec![
            Conditioning::Reference {
                image: img(),
                strength: None,
            },
            Conditioning::Mask { image: img() },
        ];
        let err = select_reference_face(&cond).unwrap_err().to_string();
        assert!(err.contains("unsupported"), "got: {err}");
    }

    /// F-026: `validate` delegates to PuLID's OWN descriptor floor, which omits the `hyper` sampler
    /// (PuLID doesn't load the dev Hyper LoRA). A `hyper` request must be rejected — not silently
    /// rendered WITHOUT the LoRA the way the dev backbone descriptor would have allowed.
    #[test]
    fn own_descriptor_floor_rejects_hyper_sampler() {
        let caps = descriptor().capabilities;
        assert!(
            !caps.samplers.contains(&"hyper"),
            "PuLID's descriptor must omit the hyper sampler"
        );
        let base = GenerationRequest {
            prompt: "a portrait".into(),
            width: 1024,
            height: 1024,
            conditioning: vec![Conditioning::Reference {
                image: img(),
                strength: None,
            }],
            ..Default::default()
        };
        // hyper is not advertised → typed Unsupported.
        let hyper = GenerationRequest {
            sampler: Some("hyper".into()),
            ..base.clone()
        };
        assert!(matches!(
            caps.validate_request(descriptor().id, &hyper),
            Err(gen_core::Error::Unsupported(_))
        ));
        // A curated sampler + PuLID's real-CFG knobs (true_cfg / negative_prompt) now validate, since
        // the descriptor advertises them (sc-3075).
        let ok = GenerationRequest {
            sampler: Some("euler".into()),
            true_cfg: Some(2.0),
            negative_prompt: Some("blurry".into()),
            ..base
        };
        assert!(
            caps.validate_request(descriptor().id, &ok).is_ok(),
            "curated sampler + real-CFG should pass PuLID's floor"
        );
    }

    /// Two reference faces are rejected, and an empty request is rejected as missing.
    #[test]
    fn select_reference_rejects_multiple_and_missing() {
        let two = vec![
            Conditioning::Reference {
                image: img(),
                strength: None,
            },
            Conditioning::Reference {
                image: img(),
                strength: None,
            },
        ];
        assert!(select_reference_face(&two)
            .unwrap_err()
            .to_string()
            .contains("exactly one"));
        assert!(select_reference_face(&[])
            .unwrap_err()
            .to_string()
            .contains("required"));
    }
}
