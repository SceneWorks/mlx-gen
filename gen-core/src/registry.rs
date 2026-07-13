//! Model + transform discovery — the link-time registry, i.e. the Rust stand-in for a DI
//! container's resolve-by-id. See `docs/MODEL_ARCHITECTURE.md` §4.
//!
//! A provider crate self-registers just by being linked (`inventory::submit!`); `mlx-gen` has
//! no central match statement to edit, so adding a model is purely additive. A consumer that
//! links one provider sees exactly one registration. Mirrors the worker's `payload.model` →
//! `MODEL_TARGETS` → load.

use crate::caption::{Captioner, CaptionerDescriptor};
use crate::generator::{ConditioningKind, Generator, Modality, ModelDescriptor};
use crate::image_embed::{ImageEmbedder, ImageEmbedderDescriptor};
use crate::runtime::{LoadSpec, WeightsSource};
use crate::text_embed::{TextEmbedder, TextEmbedderDescriptor};
use crate::train::{Trainer, TrainerDescriptor};
use crate::transform::{Transform, TransformDescriptor};
use crate::weightsmeta::safetensors_path_bytes;
use crate::{Error, Result};

use std::path::Path;

/// The per-component on-disk weight footprint (bytes) of a model, the provider-owned staged-residency
/// signal (sc-10894). Each field is the summed `.safetensors` byte size of one component category:
///
/// - `text_encoder` — the phase-A prompt encoder(s) that [`OffloadPolicy::Sequential`](crate::runtime::OffloadPolicy)
///   drops *before* the heavy render bundle loads (one or more, e.g. SDXL's two CLIPs, SD3's three);
/// - `dit` — the heavy transformer / U-Net (the "DiT"), the dominant render-phase component;
/// - `vae` — the autoencoder, co-resident with the DiT through the render.
///
/// Why a provider owns this rather than the consumer inferring it: the Sequential/staged peak is
/// `max(text_encoder, dit + vae)` (the encoder is freed before the renderer materializes — see
/// [`LoadPhase::Renderer`](crate::runtime::LoadPhase::Renderer)), not the resident sum. A consumer that
/// guesses the text-encoder size from `text_encoder*` subdir NAMING reads **zero** for any family whose
/// encoder is not under such a subdir — or has no separable encoder at all (a flat unified checkpoint) —
/// collapsing the staged peak back to the resident peak so no saving is ever selected. Each provider,
/// by contrast, computes the split from the exact subdir paths its own loader resolves.
///
/// All three are tensor-free on-disk sums ([`safetensors_dir_bytes`]) — **zero** MLX allocation, no
/// whole-file reads — so this is safe to call from a pre-load admission gate. A component a model does
/// not have (or cannot separate) is `0`.
///
/// **On-disk byte SUMS, not load-exact.** Each field totals *every* `.safetensors` under the named
/// path(s), which can exceed what a single load materializes: one component dir may ship multiple
/// interchangeable variant files (anima's `diffusion_models/` holds the base/aesthetic/turbo DiTs, but
/// a run loads exactly one — so `dit` over-counts by the unused variants), or side-by-side dtype shards
/// (an SD3 `text_encoder_3/` carrying both f32 and fp16 double-counts). Today the worker consumes only
/// `text_encoder` plus the true whole-model total; `dit` / `vae` are **informational** for a future
/// consumer, which must treat them as an upper-bound on-disk footprint, not the resident size of one load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PerComponentBytes {
    pub text_encoder: u64,
    pub dit: u64,
    pub vae: u64,
}

impl PerComponentBytes {
    /// Best-effort footprint for a diffusers-style snapshot: sum the `.safetensors` bytes under each
    /// named component subdir of the spec's weights DIRECTORY. Each list is the exact subdir(s) the
    /// caller's own loader resolves — `["text_encoder", "text_encoder_2"]` for the two SDXL CLIPs,
    /// `["unet"]` / `["transformer"]` for the DiT, `["vae"]` — so the paths are always correct per
    /// engine. A subdir that is absent contributes `0` ([`safetensors_dir_bytes`]).
    ///
    /// Each name may be a component *subdir* OR a flat component *file* ([`safetensors_path_bytes`]),
    /// so this also covers the bernini / anima flat-file layouts. Errors only when `spec.weights` is a
    /// single [`WeightsSource::File`]: a one-file checkpoint has no component tree to split (the consumer
    /// then falls back to whole-file / resident accounting).
    pub fn from_spec_subdirs(
        spec: &LoadSpec,
        text_encoder: &[&str],
        dit: &[&str],
        vae: &[&str],
    ) -> Result<Self> {
        let root = match &spec.weights {
            WeightsSource::Dir(p) => p.as_path(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "per-component footprint requires a snapshot directory, not a single .safetensors \
                     file"
                    .to_owned(),
            )),
        };
        Ok(Self::from_root_subdirs(root, text_encoder, dit, vae))
    }

    /// Sum each component's `.safetensors` bytes under an already-resolved `root` — for a provider whose
    /// component tree is NOT directly under `spec.weights` (e.g. anima's `split_files/` nesting resolves
    /// the root itself, then names `text_encoders` / `diffusion_models` / `vae` under it). Each name is a
    /// subdir or a flat file ([`safetensors_path_bytes`]); a missing one contributes `0`. Infallible —
    /// the root is the caller's to validate.
    pub fn from_root_subdirs(
        root: &Path,
        text_encoder: &[&str],
        dit: &[&str],
        vae: &[&str],
    ) -> Self {
        let sum = |names: &[&str]| -> u64 {
            names
                .iter()
                .map(|n| safetensors_path_bytes(root.join(n)))
                .sum()
        };
        Self {
            text_encoder: sum(text_encoder),
            dit: sum(dit),
            vae: sum(vae),
        }
    }
}

/// A generator provider's registration — `descriptor` for introspection (no weights loaded),
/// `load` to construct the model, and the optional [`footprint`](Self::footprint) size seam.
/// ≈ `services.AddKeyedSingleton<IGenerator>("id", factory)`.
pub struct ModelRegistration {
    pub descriptor: fn() -> ModelDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Generator>>,
    /// Optional per-component on-disk footprint (sc-10894) — `Some` for a provider that has declared its
    /// [`PerComponentBytes`] split (via `register_generators! { … ; footprint = … }`), `None` otherwise.
    /// `None` is the default so **every** provider that does not set it registers unchanged; a consumer
    /// reaching [`footprint`] then gets `Ok(None)` and falls back to its own accounting. Mirrors the
    /// [`load`](Self::load) fn-pointer shape (a spec in, a `Result` out).
    pub footprint: Option<fn(&LoadSpec) -> Result<PerComponentBytes>>,
}

inventory::collect!(ModelRegistration);

/// A transform provider's registration (parallel to [`ModelRegistration`]).
pub struct TransformRegistration {
    pub descriptor: fn() -> TransformDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Transform>>,
}

inventory::collect!(TransformRegistration);

/// A trainer provider's registration (parallel to [`ModelRegistration`]) — `descriptor` for
/// introspection, `load` to construct the trainer with its (frozen) base model from a [`LoadSpec`].
pub struct TrainerRegistration {
    pub descriptor: fn() -> TrainerDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Trainer>>,
}

inventory::collect!(TrainerRegistration);

/// A captioner provider's registration (parallel to [`ModelRegistration`]).
pub struct CaptionerRegistration {
    pub descriptor: fn() -> CaptionerDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Captioner>>,
}

inventory::collect!(CaptionerRegistration);

/// An image-embedder provider's registration (parallel to [`ModelRegistration`]). Unlike
/// `FaceEmbedder` — a directly-constructed utility — image embedders self-register so the worker's
/// `dataset_analysis` job can `load_image_embedder(id, spec)` by id, exactly like captioners.
pub struct ImageEmbedderRegistration {
    pub descriptor: fn() -> ImageEmbedderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn ImageEmbedder>>,
}

inventory::collect!(ImageEmbedderRegistration);

/// A text-embedder provider's registration (parallel to [`ImageEmbedderRegistration`]). Used by the
/// worker's `dataset_analysis` job for caption/image alignment in CLIP's joint space.
pub struct TextEmbedderRegistration {
    pub descriptor: fn() -> TextEmbedderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn TextEmbedder>>,
}

inventory::collect!(TextEmbedderRegistration);

/// All registered generators (one per linked provider crate).
pub fn generators() -> impl Iterator<Item = &'static ModelRegistration> {
    inventory::iter::<ModelRegistration>.into_iter()
}

/// All registered transforms.
pub fn transforms() -> impl Iterator<Item = &'static TransformRegistration> {
    inventory::iter::<TransformRegistration>.into_iter()
}

/// All registered trainers (one per linked provider crate that supports training).
pub fn trainers() -> impl Iterator<Item = &'static TrainerRegistration> {
    inventory::iter::<TrainerRegistration>.into_iter()
}

/// All registered captioners (one per linked provider crate that supports image-to-text captioning).
pub fn captioners() -> impl Iterator<Item = &'static CaptionerRegistration> {
    inventory::iter::<CaptionerRegistration>.into_iter()
}

/// All registered image embedders (one per linked provider crate).
pub fn image_embedders() -> impl Iterator<Item = &'static ImageEmbedderRegistration> {
    inventory::iter::<ImageEmbedderRegistration>.into_iter()
}

/// All registered text embedders (one per linked provider crate).
pub fn text_embedders() -> impl Iterator<Item = &'static TextEmbedderRegistration> {
    inventory::iter::<TextEmbedderRegistration>.into_iter()
}

/// Define a first-wins `load_*` resolver over a registration iterator (F-056). The six load fns were
/// byte-identical modulo the iterator, the returned trait-object type, and the `$kind` label that
/// appears in both the "no {kind} registered" error and the duplicate-id debug assertion — so they
/// share one body here. **The generated behavior (first-wins, the debug-build duplicate assertion,
/// and the EXACT per-kind error text) is unchanged** — the worker/tests match on the error string.
macro_rules! define_load {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident via $iter:ident as $kind:literal => $trait:ty
    ) => {
        $(#[$meta])*
        $vis fn $name(id: &str, spec: &LoadSpec) -> Result<Box<$trait>> {
            let mut matches = $iter().filter(|r| (r.descriptor)().id == id);
            let reg = matches.next().ok_or_else(|| {
                Error::Msg(format!(
                    concat!("no ", $kind, " registered for id '{id}'"),
                    id = id
                ))
            })?;
            debug_assert!(
                matches.next().is_none(),
                concat!("duplicate ", $kind, " id '{id}' registered (first-wins shadows the rest)"),
                id = id
            );
            (reg.load)(spec)
        }
    };
}

define_load! {
    /// Load a generator by model id (e.g. `"z_image_turbo"`).
    ///
    /// The link-time registry is **first-wins** on duplicate ids; a debug-build assertion surfaces a
    /// duplicate registration (a provider-crate mistake) instead of silently shadowing one (sc-6983).
    pub fn load via generators as "generator" => dyn Generator
}

define_load! {
    /// Load a transform by id.
    pub fn load_transform via transforms as "transform" => dyn Transform
}

/// The per-component on-disk [`PerComponentBytes`] footprint of generator `id`, if its provider declared
/// one (sc-10894). Looks the registration up by id (first-wins, like [`load`]) and calls its optional
/// `footprint` fn:
///
/// - `Ok(Some(bytes))` — the provider computed the split (from the paths its own loader resolves);
/// - `Ok(None)` — the id is registered but declares **no** footprint (the default), so the consumer
///   falls back to its own accounting (e.g. the worker's `text_encoder*` subdir sum);
/// - `Err` — no generator is registered for `id`, or the provider's footprint fn itself failed
///   (e.g. a single-file source with no component tree). A fail-open consumer treats `Err` like `None`.
///
/// Tensor-free and weights-free of allocation — the footprint fns read only on-disk sizes — so this is
/// safe on a pre-load admission path. Sees exactly the registrations the calling binary links (the
/// sc-4482 dead-strip rule), same as [`load`].
pub fn footprint(id: &str, spec: &LoadSpec) -> Result<Option<PerComponentBytes>> {
    let mut matches = generators().filter(|r| (r.descriptor)().id == id);
    let reg = matches
        .next()
        .ok_or_else(|| Error::Msg(format!("no generator registered for id '{id}'")))?;
    debug_assert!(
        matches.next().is_none(),
        "duplicate generator id '{id}' registered (first-wins shadows the rest)"
    );
    match reg.footprint {
        Some(f) => f(spec).map(Some),
        None => Ok(None),
    }
}

define_load! {
    /// Load a trainer by model id (e.g. `"z_image_turbo"`) with its (frozen) base model.
    pub fn load_trainer via trainers as "trainer" => dyn Trainer
}

define_load! {
    /// Load a captioner by model id (e.g. `"joy_caption"`).
    pub fn load_captioner via captioners as "captioner" => dyn Captioner
}

define_load! {
    /// Load an image embedder by id (e.g. `"clip_vit_l14"`).
    pub fn load_image_embedder via image_embedders as "image embedder" => dyn ImageEmbedder
}

define_load! {
    /// Load a text embedder by id (e.g. `"clip_vit_l14_text"`).
    pub fn load_text_embedder via text_embedders as "text embedder" => dyn TextEmbedder
}

// ---------------------------------------------------------------------------------------------
// Descriptor-level conformance sweep (sc-9098, F-009)
// ---------------------------------------------------------------------------------------------

/// An identifier-shaped registry string: non-empty lowercase `a-z0-9` with `_`/`-`/`.`/`/`
/// separators — the shape every shipped id/family/backend uses (`z_image_turbo`, `image-embed`,
/// `mlx`, and HF-repo-style captioner ids like `fancyfeast/llama-joycaption-beta-one-hf-llava`).
/// Rejects whitespace/uppercase/unicode, which would break worker payload routing and log grepping.
fn is_registry_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| {
            c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '_' | '-' | '.' | '/')
        })
}

/// Push an error for every malformed identity field (shared by all descriptor kinds).
fn check_identity(errs: &mut Vec<String>, ctx: &str, fields: &[(&str, &str)]) {
    for (name, value) in fields {
        if !is_registry_ident(value) {
            errs.push(format!(
                "{ctx}: {name} {value:?} is not a valid registry identifier \
                 (non-empty lowercase [a-z0-9_.-/])"
            ));
        }
    }
}

/// Push an error for empty/whitespace/duplicate entries in a descriptor's curated name list
/// (samplers / schedulers / guidance methods).
fn check_name_list(errs: &mut Vec<String>, ctx: &str, list_name: &str, names: &[&str]) {
    for (i, n) in names.iter().enumerate() {
        if n.is_empty() || n.chars().any(char::is_whitespace) {
            errs.push(format!(
                "{ctx}: {list_name}[{i}] {n:?} is empty or contains whitespace"
            ));
        }
        if names[..i].contains(n) {
            errs.push(format!("{ctx}: duplicate {list_name} entry {n:?}"));
        }
    }
}

/// The weights-free invariants a generator [`ModelDescriptor`] must satisfy — everything checkable
/// from `(registration.descriptor)()` alone, with no model load (sc-9098, F-009):
///
/// - `id` / `family` / `backend` are non-empty registry identifiers,
/// - `max_count ≥ 1` and `1 ≤ min_size ≤ max_size` (a `Default` 0 bound rejects every request with
///   a confusing "out of range 0..=0" — the F-084 footgun, enforced here for *every* linked
///   descriptor rather than only when a request happens to reach `validate_request`),
/// - `samplers` / `schedulers` / `supported_guidance_methods` entries are non-empty, whitespace-free
///   and duplicate-free (name *shape* only — resolvability is per-engine: several families advertise
///   native sampler names alongside the gen-core curated set),
/// - `conditioning` is duplicate-free, and the video-clip kinds
///   ([`Keyframe`](ConditioningKind::Keyframe) / [`VideoClip`](ConditioningKind::VideoClip) /
///   [`ControlClip`](ConditioningKind::ControlClip)) are only advertised by `Video`/`Both`-modality
///   models — an `Image` model cannot consume a clip.
///
/// Returns one message per violation (empty = conformant). Public so a provider's own tests can
/// target a single descriptor; [`descriptor_conformance_errors`] sweeps every linked registration.
pub fn model_descriptor_errors(d: &ModelDescriptor) -> Vec<String> {
    let mut errs = Vec::new();
    let ctx = format!("generator '{}'", d.id);
    check_identity(
        &mut errs,
        &ctx,
        &[("id", d.id), ("family", d.family), ("backend", d.backend)],
    );
    let caps = &d.capabilities;
    if caps.max_count == 0 {
        errs.push(format!(
            "{ctx}: max_count is 0 — every request would be rejected"
        ));
    }
    if caps.min_size == 0 || caps.max_size == 0 {
        errs.push(format!(
            "{ctx}: min_size={} max_size={} — size bounds left at the Default 0",
            caps.min_size, caps.max_size
        ));
    } else if caps.min_size > caps.max_size {
        errs.push(format!(
            "{ctx}: min_size {} > max_size {}",
            caps.min_size, caps.max_size
        ));
    }
    check_name_list(&mut errs, &ctx, "sampler", &caps.samplers);
    check_name_list(&mut errs, &ctx, "scheduler", &caps.schedulers);
    check_name_list(
        &mut errs,
        &ctx,
        "guidance_method",
        &caps.supported_guidance_methods,
    );
    for (i, k) in caps.conditioning.iter().enumerate() {
        if caps.conditioning[..i].contains(k) {
            errs.push(format!("{ctx}: duplicate conditioning kind {k:?}"));
        }
        let is_video_kind = matches!(
            k,
            ConditioningKind::Keyframe
                | ConditioningKind::VideoClip
                | ConditioningKind::ControlClip
        );
        if is_video_kind && d.modality == Modality::Image {
            errs.push(format!(
                "{ctx}: advertises video conditioning {k:?} but modality is Image"
            ));
        }
    }
    errs
}

/// Push duplicate-id errors for one registry kind (the link-time registry is first-wins, so a
/// duplicate silently shadows a registration — sc-6983's debug assertion, surfaced for every kind).
fn check_unique_ids(errs: &mut Vec<String>, kind: &str, ids: &[&str]) {
    for (i, id) in ids.iter().enumerate() {
        if ids[..i].contains(id) {
            errs.push(format!(
                "{kind} id '{id}' is registered more than once (first-wins shadows the rest)"
            ));
        }
    }
}

/// Weights-free descriptor-level conformance sweep over **every registration linked into the
/// current binary** (sc-9098, F-009): generators through [`model_descriptor_errors`], plus identity
/// and capability-bound checks and per-kind id uniqueness for trainers, captioners, transforms and
/// image/text embedders. No `load` is ever called, so it runs by default (no weights, no Metal) —
/// each provider crate invokes it from a default test, giving every registered id at least
/// descriptor-level coverage; behavioral conformance (progress/cancel/seed) stays weights-gated in
/// the `gen-core-testkit` suite.
///
/// Returns one message per violation (empty = conformant). The sweep sees exactly the registrations
/// the calling binary links — the same visibility rule as [`load`] (the sc-4482 dead-strip trap),
/// so a caller must force-link its providers (`use mlx_gen_<x> as _;`).
pub fn descriptor_conformance_errors() -> Vec<String> {
    let mut errs = Vec::new();

    let gen_descs: Vec<ModelDescriptor> = generators().map(|r| (r.descriptor)()).collect();
    for d in &gen_descs {
        errs.extend(model_descriptor_errors(d));
    }
    let gen_ids: Vec<&str> = gen_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "generator", &gen_ids);

    let trainer_descs: Vec<TrainerDescriptor> = trainers().map(|r| (r.descriptor)()).collect();
    for d in &trainer_descs {
        let ctx = format!("trainer '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
        if !d.supports_lora && !d.supports_lokr {
            errs.push(format!(
                "{ctx}: supports neither LoRA nor LoKr — a trainer must offer at least one adapter kind"
            ));
        }
    }
    let trainer_ids: Vec<&str> = trainer_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "trainer", &trainer_ids);

    let cap_descs: Vec<CaptionerDescriptor> = captioners().map(|r| (r.descriptor)()).collect();
    for d in &cap_descs {
        let ctx = format!("captioner '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
        let c = &d.capabilities;
        if c.min_image_size == 0 || c.max_image_size < c.min_image_size {
            errs.push(format!(
                "{ctx}: image-size bounds incoherent (min {} max {})",
                c.min_image_size, c.max_image_size
            ));
        }
        if c.max_new_tokens == 0 {
            errs.push(format!(
                "{ctx}: max_new_tokens is 0 — no caption could be produced"
            ));
        }
    }
    let cap_ids: Vec<&str> = cap_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "captioner", &cap_ids);

    let tf_descs: Vec<TransformDescriptor> = transforms().map(|r| (r.descriptor)()).collect();
    for d in &tf_descs {
        let ctx = format!("transform '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
    }
    let tf_ids: Vec<&str> = tf_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "transform", &tf_ids);

    let ie_descs: Vec<ImageEmbedderDescriptor> =
        image_embedders().map(|r| (r.descriptor)()).collect();
    let te_descs: Vec<TextEmbedderDescriptor> =
        text_embedders().map(|r| (r.descriptor)()).collect();
    for (ctx_kind, id, family, backend, dim, space) in ie_descs
        .iter()
        .map(|d| {
            (
                "image embedder",
                d.id,
                d.family,
                d.backend,
                d.embedding_dim,
                d.space,
            )
        })
        .chain(te_descs.iter().map(|d| {
            (
                "text embedder",
                d.id,
                d.family,
                d.backend,
                d.embedding_dim,
                d.space,
            )
        }))
    {
        let ctx = format!("{ctx_kind} '{id}'");
        check_identity(
            &mut errs,
            &ctx,
            &[("id", id), ("family", family), ("backend", backend)],
        );
        if dim == 0 {
            errs.push(format!("{ctx}: embedding_dim is 0"));
        }
        if space.is_empty() {
            errs.push(format!("{ctx}: embedding space is empty"));
        }
    }
    let ie_ids: Vec<&str> = ie_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "image embedder", &ie_ids);
    let te_ids: Vec<&str> = te_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "text embedder", &te_ids);

    errs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caption::{
        CaptionCapabilities, CaptionOutput, CaptionRequest, Captioner, CaptionerDescriptor,
    };
    use crate::generator::{
        Capabilities, GenerationOutput, GenerationRequest, Modality, ModelDescriptor,
    };
    use crate::image_embed::{ImageEmbedder, ImageEmbedderDescriptor};
    use crate::media::Image;
    use crate::runtime::{Progress, WeightsSource};
    use crate::text_embed::{TextEmbedder, TextEmbedderDescriptor};
    use crate::train::{
        Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
    };
    use std::path::PathBuf;

    struct DummyGen {
        desc: ModelDescriptor,
    }

    impl Generator for DummyGen {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.desc
        }
        fn validate(&self, _req: &GenerationRequest) -> Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            _req: &GenerationRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<GenerationOutput> {
            Ok(GenerationOutput::Images(vec![Image::default()]))
        }
    }

    /// Small-but-coherent capabilities for the dummy registrations: the descriptor sweep
    /// ([`descriptor_conformance_errors`]) runs over everything this test binary registers, so the
    /// dummies must carry real bounds (a `Capabilities::default()` has the F-084 all-zero bounds
    /// the sweep exists to reject).
    fn dummy_caps() -> Capabilities {
        Capabilities {
            min_size: 64,
            max_size: 512,
            max_count: 1,
            ..Default::default()
        }
    }

    fn dummy_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_test_model",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: dummy_descriptor(),
        }))
    }

    crate::register_generators! { dummy_descriptor => dummy_load }

    struct DummyDelegatedGen {
        descriptor: ModelDescriptor,
    }

    impl DummyDelegatedGen {
        fn generate_impl(
            &self,
            _req: &GenerationRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<GenerationOutput> {
            Ok(GenerationOutput::Images(vec![Image::default()]))
        }
    }

    crate::impl_generator!(DummyDelegatedGen {
        validate: |_s, _req| Ok::<(), Error>(()),
        generate: generate_impl,
    });

    fn dummy_delegated_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_delegated_test_model",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_delegated_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyDelegatedGen {
            descriptor: dummy_delegated_descriptor(),
        }))
    }

    crate::register_generators! { dummy_delegated_descriptor => dummy_delegated_load }

    // A dummy generator that DECLARES a per-component footprint (sc-10894), exercising the
    // `; footprint = …` macro arm and the [`footprint`] entry point. Its text encoder is under a
    // non-standard `mllm/` subdir (the real boogu layout) — a naming a `text_encoder*` guesser would
    // read as ZERO — so the provider-owned split is what finds it.
    fn dummy_footprint_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_footprint_model",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_footprint_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: dummy_footprint_descriptor(),
        }))
    }

    fn dummy_footprint(spec: &LoadSpec) -> Result<PerComponentBytes> {
        PerComponentBytes::from_spec_subdirs(spec, &["mllm"], &["transformer"], &["vae"])
    }

    crate::register_generators! {
        dummy_footprint_descriptor => dummy_footprint_load ; footprint = dummy_footprint
    }

    struct DummyTrainer {
        desc: TrainerDescriptor,
    }

    impl Trainer for DummyTrainer {
        fn descriptor(&self) -> &TrainerDescriptor {
            &self.desc
        }

        fn validate(&self, _req: &TrainingRequest) -> Result<()> {
            Ok(())
        }

        fn train(
            &mut self,
            _req: &TrainingRequest,
            _on_progress: &mut dyn FnMut(TrainingProgress),
        ) -> Result<TrainingOutput> {
            Ok(TrainingOutput {
                adapter_path: PathBuf::from("/tmp/dummy.safetensors"),
                steps: 0,
                final_loss: 0.0,
            })
        }
    }

    fn dummy_trainer_descriptor() -> TrainerDescriptor {
        TrainerDescriptor {
            id: "dummy_test_trainer",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            supports_lora: true,
            supports_lokr: false,
            supports_control: false,
        }
    }

    fn dummy_trainer_load(_spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
        Ok(Box::new(DummyTrainer {
            desc: dummy_trainer_descriptor(),
        }))
    }

    crate::register_trainer! { dummy_trainer_descriptor => dummy_trainer_load }

    // Multi-arm fixtures: a single `register_generators!` / `register_trainer!` invocation with two
    // `desc => load` arms exercises the `,+` repetition that single-arm callers never reach. This is
    // the path the provider migration sweep (sc-7780) leans on for multi-variant crates like boogu.
    fn dummy_multi_gen_a_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_multi_gen_a",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_multi_gen_b_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_multi_gen_b",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_multi_gen_a_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: dummy_multi_gen_a_descriptor(),
        }))
    }

    fn dummy_multi_gen_b_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: dummy_multi_gen_b_descriptor(),
        }))
    }

    crate::register_generators! {
        dummy_multi_gen_a_descriptor => dummy_multi_gen_a_load,
        dummy_multi_gen_b_descriptor => dummy_multi_gen_b_load,
    }

    fn dummy_multi_trainer_a_descriptor() -> TrainerDescriptor {
        TrainerDescriptor {
            id: "dummy_multi_trainer_a",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            supports_lora: true,
            supports_lokr: false,
            supports_control: false,
        }
    }

    fn dummy_multi_trainer_b_descriptor() -> TrainerDescriptor {
        TrainerDescriptor {
            id: "dummy_multi_trainer_b",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            supports_lora: true,
            supports_lokr: false,
            supports_control: false,
        }
    }

    fn dummy_multi_trainer_a_load(_spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
        Ok(Box::new(DummyTrainer {
            desc: dummy_multi_trainer_a_descriptor(),
        }))
    }

    fn dummy_multi_trainer_b_load(_spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
        Ok(Box::new(DummyTrainer {
            desc: dummy_multi_trainer_b_descriptor(),
        }))
    }

    crate::register_trainer! {
        dummy_multi_trainer_a_descriptor => dummy_multi_trainer_a_load,
        dummy_multi_trainer_b_descriptor => dummy_multi_trainer_b_load,
    }

    struct DummyCaptioner {
        desc: CaptionerDescriptor,
    }

    impl Captioner for DummyCaptioner {
        fn descriptor(&self) -> &CaptionerDescriptor {
            &self.desc
        }
        fn validate(&self, _req: &CaptionRequest) -> Result<()> {
            Ok(())
        }
        fn caption(
            &self,
            _req: &CaptionRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<CaptionOutput> {
            Ok(CaptionOutput {
                text: "caption".to_owned(),
                generated_tokens: Some(1),
                finish_reason: None,
            })
        }
    }

    fn dummy_captioner_descriptor() -> CaptionerDescriptor {
        CaptionerDescriptor {
            id: "dummy_test_captioner",
            family: "test",
            backend: "mlx",
            capabilities: CaptionCapabilities {
                min_image_size: 1,
                max_image_size: 4096,
                max_prompt_chars: 4000,
                max_name_chars: 120,
                max_extra_options: 16,
                max_extra_option_chars: 500,
                max_trigger_words: 32,
                max_trigger_word_chars: 120,
                max_new_tokens: 1024,
                ..Default::default()
            },
        }
    }

    fn dummy_captioner_load(_spec: &LoadSpec) -> Result<Box<dyn Captioner>> {
        Ok(Box::new(DummyCaptioner {
            desc: dummy_captioner_descriptor(),
        }))
    }

    crate::register_captioner! { dummy_captioner_descriptor => dummy_captioner_load }

    #[test]
    fn registry_resolves_by_id() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = load("dummy_test_model", &spec).expect("dummy is registered");
        assert_eq!(g.descriptor().id, "dummy_test_model");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn unknown_id_errors() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(load("no_such_model", &spec).is_err());
    }

    #[test]
    fn dummy_appears_in_iteration() {
        assert!(generators().any(|r| (r.descriptor)().id == "dummy_test_model"));
    }

    #[test]
    fn macro_delegated_generator_resolves_by_id() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = load("dummy_delegated_test_model", &spec).expect("dummy is registered");
        assert_eq!(g.descriptor().id, "dummy_delegated_test_model");
        g.validate(&GenerationRequest::default()).unwrap();
    }

    #[test]
    fn macro_registered_trainer_resolves_by_id() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = load_trainer("dummy_test_trainer", &spec).expect("dummy trainer is registered");
        assert_eq!(t.descriptor().id, "dummy_test_trainer");
        assert!(trainers().any(|r| (r.descriptor)().id == "dummy_test_trainer"));
    }

    #[test]
    fn multi_arm_register_generators_registers_each() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        for id in ["dummy_multi_gen_a", "dummy_multi_gen_b"] {
            let g = load(id, &spec).unwrap_or_else(|_| panic!("{id} is registered"));
            assert_eq!(g.descriptor().id, id);
        }
    }

    #[test]
    fn multi_arm_register_trainer_registers_each() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        for id in ["dummy_multi_trainer_a", "dummy_multi_trainer_b"] {
            let t = load_trainer(id, &spec).unwrap_or_else(|_| panic!("{id} is registered"));
            assert_eq!(t.descriptor().id, id);
        }
    }

    #[test]
    fn captioner_registry_resolves_by_id() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let c =
            load_captioner("dummy_test_captioner", &spec).expect("dummy captioner is registered");
        assert_eq!(c.descriptor().id, "dummy_test_captioner");
    }

    #[test]
    fn unknown_captioner_id_errors() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(load_captioner("no_such_captioner", &spec).is_err());
    }

    #[test]
    fn dummy_captioner_appears_in_iteration() {
        assert!(captioners().any(|r| (r.descriptor)().id == "dummy_test_captioner"));
    }

    struct DummyTextEmbedder {
        desc: TextEmbedderDescriptor,
    }

    impl TextEmbedder for DummyTextEmbedder {
        fn descriptor(&self) -> &TextEmbedderDescriptor {
            &self.desc
        }

        fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            Ok(vec![text.len() as f32, 1.0])
        }
    }

    fn dummy_text_embedder_descriptor() -> TextEmbedderDescriptor {
        TextEmbedderDescriptor {
            id: "dummy_test_text_embedder",
            family: "test",
            backend: "mlx",
            embedding_dim: 2,
            space: "test-space",
            mac_only: true,
        }
    }

    fn dummy_text_embedder_load(_spec: &LoadSpec) -> Result<Box<dyn TextEmbedder>> {
        Ok(Box::new(DummyTextEmbedder {
            desc: dummy_text_embedder_descriptor(),
        }))
    }

    crate::register_text_embedder! { dummy_text_embedder_descriptor => dummy_text_embedder_load }

    #[test]
    fn text_embedder_registry_resolves_by_id() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let e = load_text_embedder("dummy_test_text_embedder", &spec)
            .expect("dummy text embedder is registered");
        assert_eq!(e.descriptor().id, "dummy_test_text_embedder");
        assert_eq!(e.embed_text("clip").unwrap(), vec![4.0, 1.0]);
        assert_eq!(
            e.embed_text_batch(&["a", "abcd"]).unwrap(),
            vec![vec![1.0, 1.0], vec![4.0, 1.0]]
        );
    }

    #[test]
    fn unknown_text_embedder_id_errors() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(load_text_embedder("no_such_text_embedder", &spec).is_err());
    }

    #[test]
    fn dummy_text_embedder_appears_in_iteration() {
        assert!(text_embedders().any(|r| (r.descriptor)().id == "dummy_test_text_embedder"));
    }

    struct DummyImageEmbedder {
        desc: ImageEmbedderDescriptor,
    }

    impl ImageEmbedder for DummyImageEmbedder {
        fn descriptor(&self) -> &ImageEmbedderDescriptor {
            &self.desc
        }

        fn embed(&self, image: &Image) -> Result<Vec<f32>> {
            Ok(vec![image.width as f32, image.height as f32])
        }
    }

    fn dummy_image_embedder_descriptor() -> ImageEmbedderDescriptor {
        ImageEmbedderDescriptor {
            id: "dummy_test_image_embedder",
            family: "test",
            backend: "mlx",
            embedding_dim: 2,
            space: "test-space",
            mac_only: true,
        }
    }

    fn dummy_image_embedder_load(_spec: &LoadSpec) -> Result<Box<dyn ImageEmbedder>> {
        Ok(Box::new(DummyImageEmbedder {
            desc: dummy_image_embedder_descriptor(),
        }))
    }

    crate::register_image_embedder! { dummy_image_embedder_descriptor => dummy_image_embedder_load }

    #[test]
    fn image_embedder_registry_resolves_by_id() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let e = load_image_embedder("dummy_test_image_embedder", &spec)
            .expect("dummy image embedder is registered");
        assert_eq!(e.descriptor().id, "dummy_test_image_embedder");
        let img = Image {
            width: 7,
            height: 3,
            pixels: vec![0; 7 * 3 * 3],
        };
        assert_eq!(e.embed(&img).unwrap(), vec![7.0, 3.0]);
    }

    #[test]
    fn unknown_image_embedder_id_errors() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(load_image_embedder("no_such_image_embedder", &spec).is_err());
    }

    #[test]
    fn dummy_image_embedder_appears_in_iteration() {
        assert!(image_embedders().any(|r| (r.descriptor)().id == "dummy_test_image_embedder"));
    }

    /// The sweep (sc-9098, F-009) is clean over everything this test binary registers — the dummy
    /// generators/trainers/captioner/embedders all carry coherent descriptors.
    #[test]
    fn descriptor_sweep_is_clean_over_registered_dummies() {
        let errs = descriptor_conformance_errors();
        assert!(
            errs.is_empty(),
            "descriptor conformance FAILED:\n  - {}",
            errs.join("\n  - ")
        );
    }

    /// Each per-descriptor invariant fires: identity shape, zero/inverted bounds, duplicate or
    /// malformed curated names, duplicate conditioning, video conditioning on an Image model.
    #[test]
    fn model_descriptor_errors_flags_each_violation() {
        // A fully-coherent descriptor produces no errors.
        assert!(model_descriptor_errors(&dummy_descriptor()).is_empty());

        let broken = ModelDescriptor {
            id: "Bad Id", // uppercase + whitespace
            family: "",   // empty
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities {
                min_size: 512,
                max_size: 256,                                // inverted
                max_count: 0,                                 // zero
                samplers: vec!["euler", "euler", "bad name"], // duplicate + whitespace
                conditioning: vec![
                    ConditioningKind::Reference,
                    ConditioningKind::Reference, // duplicate
                    ConditioningKind::VideoClip, // video kind on an Image model
                ],
                ..Default::default()
            },
        };
        let errs = model_descriptor_errors(&broken);
        let has = |needle: &str| errs.iter().any(|e| e.contains(needle));
        assert!(has("id \"Bad Id\""), "{errs:?}");
        assert!(has("family \"\""), "{errs:?}");
        assert!(has("max_count is 0"), "{errs:?}");
        assert!(has("min_size 512 > max_size 256"), "{errs:?}");
        assert!(has("duplicate sampler entry \"euler\""), "{errs:?}");
        assert!(has("sampler[2] \"bad name\""), "{errs:?}");
        assert!(has("duplicate conditioning kind Reference"), "{errs:?}");
        assert!(has("video conditioning VideoClip"), "{errs:?}");

        // All-zero bounds report the Default-0 message (F-084), not the inverted-bounds one.
        let zeroed = ModelDescriptor {
            id: "zeroed",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities::default(),
        };
        assert!(model_descriptor_errors(&zeroed)
            .iter()
            .any(|e| e.contains("left at the Default 0")));
    }

    /// Build a synthetic diffusers-style snapshot with a `bytes`-sized `model.safetensors` under each
    /// named subdir, returning the root. The caller cleans it up.
    fn synthetic_snapshot(tag: &str, subdirs: &[(&str, usize)]) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "gencore_footprint_{tag}_{}_{}",
            std::process::id(),
            line!()
        ));
        for (sub, bytes) in subdirs {
            let dir = root.join(sub);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), vec![0u8; *bytes]).unwrap();
        }
        root
    }

    /// sc-10894: a provider that declared a footprint returns the per-component on-disk split, resolved
    /// from the exact subdirs its loader uses — including a text encoder under a NON-`text_encoder`
    /// subdir (`mllm/`, the boogu layout) that a name-guessing consumer would read as zero.
    #[test]
    fn footprint_returns_provider_component_split() {
        let root = synthetic_snapshot(
            "split",
            &[("mllm", 1500), ("transformer", 9000), ("vae", 400)],
        );
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));

        let fp = footprint("dummy_footprint_model", &spec)
            .expect("registered + declares a footprint")
            .expect("Some — the provider computed the split");
        assert_eq!(
            fp,
            PerComponentBytes {
                text_encoder: 1500,
                dit: 9000,
                vae: 400,
            }
        );
        // The whole point: the text encoder is NON-zero even though it is not under `text_encoder*`.
        assert!(fp.text_encoder > 0, "mllm/ text encoder must be measured");

        std::fs::remove_dir_all(&root).ok();
    }

    /// A registered generator that declares NO footprint yields `Ok(None)` (the consumer falls back);
    /// an unknown id is an `Err`.
    #[test]
    fn footprint_is_none_without_declaration_and_errs_on_unknown_id() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        // `dummy_test_model` is registered but declares no footprint.
        assert_eq!(footprint("dummy_test_model", &spec).unwrap(), None);
        // Unknown id → Err (a fail-open consumer treats it like None).
        assert!(footprint("no_such_model", &spec).is_err());
    }

    /// sc-10894: `from_spec_subdirs` sums each component's subdir(s) (SD3's three text encoders here),
    /// treats a missing subdir as `0`, and errors on a single-`File` source (no tree to split).
    #[test]
    fn per_component_bytes_from_spec_subdirs_and_file_guard() {
        let root = synthetic_snapshot(
            "sd3",
            &[
                ("text_encoder", 100),
                ("text_encoder_2", 200),
                ("text_encoder_3", 4000),
                ("transformer", 8000),
                ("vae", 300),
            ],
        );
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let fp = PerComponentBytes::from_spec_subdirs(
            &spec,
            &["text_encoder", "text_encoder_2", "text_encoder_3"],
            &["transformer"],
            &["vae"],
        )
        .unwrap();
        assert_eq!(fp.text_encoder, 4300); // 100 + 200 + 4000
        assert_eq!(fp.dit, 8000);
        assert_eq!(fp.vae, 300);

        // A named-but-absent subdir contributes 0 (does not error).
        let fp_missing =
            PerComponentBytes::from_spec_subdirs(&spec, &["nope"], &["transformer"], &["vae"])
                .unwrap();
        assert_eq!(fp_missing.text_encoder, 0);

        // A single-file source has no component tree → Err (consumer falls back to whole-file).
        let file_spec = LoadSpec::new(WeightsSource::File(
            root.join("transformer/model.safetensors"),
        ));
        assert!(
            PerComponentBytes::from_spec_subdirs(&file_spec, &["te"], &["dit"], &["vae"]).is_err()
        );

        std::fs::remove_dir_all(&root).ok();
    }

    /// sc-10894: `from_root_subdirs` sums a component named by a flat FILE (the bernini/anima layout —
    /// `t5_encoder.safetensors` at the root, not a `text_encoder/` subdir) as well as a subdir, against
    /// an already-resolved root.
    #[test]
    fn per_component_bytes_from_root_subdirs_handles_flat_files() {
        let root = std::env::temp_dir().join(format!(
            "gencore_footprint_flat_{}_{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&root).unwrap();
        // bernini-style flat component files at the root.
        std::fs::write(root.join("t5_encoder.safetensors"), vec![0u8; 2000]).unwrap();
        std::fs::write(root.join("low_noise_model.safetensors"), vec![0u8; 6000]).unwrap();
        std::fs::write(root.join("high_noise_model.safetensors"), vec![0u8; 6000]).unwrap();
        std::fs::write(root.join("vae.safetensors"), vec![0u8; 500]).unwrap();

        let fp = PerComponentBytes::from_root_subdirs(
            &root,
            &["t5_encoder.safetensors"],
            &[
                "low_noise_model.safetensors",
                "high_noise_model.safetensors",
            ],
            &["vae.safetensors"],
        );
        assert_eq!(
            fp,
            PerComponentBytes {
                text_encoder: 2000,
                dit: 12000,
                vae: 500,
            }
        );

        std::fs::remove_dir_all(&root).ok();
    }
}
