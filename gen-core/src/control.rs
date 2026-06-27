//! `ControlBranch` — the backend-neutral seam shared by every **Fun-Controlnet-Union** /
//! ControlNet variant (sc-8241, epic 8236).
//!
//! A control variant (Z-Image-turbo, FLUX.2-dev, Qwen-Image — and, by design, the forthcoming
//! FLUX.1-dev branch sc-8238/sc-8239) is the base generator plus a control branch that injects a
//! VAE-encoded skeleton/depth/canny hint into the backbone's own DiT blocks. The **per-model forward
//! pass is NOT shareable**: a Fun-Controlnet-Union branch is a partial copy of each backbone's blocks
//! (FLUX.2 MMDiT / Z-Image no-RoPE MMDiT / Qwen DiT), so the residual/context math lives in the
//! provider crate. What *was* duplicated near-verbatim across all three crates — and is collapsed
//! here — is the **boilerplate** around that forward:
//!
//! - pulling the base-snapshot dir + the (required) control checkpoint out of a [`LoadSpec`], with
//!   consistent error messages,
//! - extracting the single [`Conditioning::Control`] image + its scale from a request (with the
//!   accepted-kind policy),
//! - the `validate()` tail that requires a `Control` conditioning be present.
//!
//! This trait is **tensor-free** (it touches only gen-core contract types), so it lives in gen-core
//! alongside [`Generator`] and is implemented by each provider's loaded control struct. The default
//! message bodies match the canonical Z-Image port; FLUX.2 / Qwen override only the few methods
//! whose wording legitimately differs, so the user-facing error text stays **byte-identical** to the
//! hand-written originals (the `#[ignore]` smokes assert on the substrings these produce).
//!
//! The trait surface is deliberately **input-agnostic**: the Fun-Union family has no discrete
//! control-mode index — pose / canny / depth differ only by the host-side preprocessor, never by a
//! branch parameter — so there is intentionally no "mode" on this seam. A new backbone (FLUX.1) plugs
//! in by implementing the same handful of methods; no trait change is required.

use crate::error::{Error, Result};
use crate::generator::{Conditioning, ControlKind, GenerationRequest};
use crate::media::Image;
use crate::runtime::{LoadSpec, WeightsSource};
use std::path::Path;

/// Which [`ControlKind`]s a control branch accepts. The Fun-Controlnet-Union checkpoints are a
/// *union* of pose/canny/depth over one VAE-encoded path, so most branches accept [`Any`]; a branch
/// wired for a single signal (e.g. Qwen v1 = pose-only) restricts to [`Only`].
///
/// This is the *acceptance policy* for the request-side [`Conditioning::Control`] `kind`, kept on the
/// trait so the validation collapses into the shared [`ControlBranch::resolve_control`] helper rather
/// than re-implementing the `kind` check in each provider.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AcceptedControlKinds {
    /// Any [`ControlKind`] is accepted — the input-agnostic Fun-Union default (pose/canny/depth all
    /// flow through the same VAE-encoded control path; they differ only by host preprocessor).
    Any,
    /// Only these specific kinds are accepted (e.g. Qwen-Image v1 = `[Pose]`); any other is rejected
    /// at `resolve_control` rather than silently treated as the accepted one.
    Only(Vec<ControlKind>),
}

impl AcceptedControlKinds {
    /// Whether `kind` is admitted by this policy.
    pub fn accepts(&self, kind: &ControlKind) -> bool {
        match self {
            AcceptedControlKinds::Any => true,
            AcceptedControlKinds::Only(kinds) => kinds.contains(kind),
        }
    }
}

/// The shared engine seam for a ControlNet / Fun-Controlnet-Union variant.
///
/// A provider's loaded control struct (`ZImageTurboControl`, `Flux2DevControl`, `QwenImageControl`,
/// future `Flux1DevControl`) implements this **in addition to** [`Generator`]. The required methods
/// supply the per-model identity (id, accepted kinds); the **provided methods** supply all the
/// boilerplate that used to be copy-pasted across the crates, plus override points for the few
/// message bodies that legitimately differ per model. Nothing here touches tensors — the per-model
/// forward, weight load, and `generate` loop stay in the provider crate.
///
/// The load-time boilerplate is exposed as the free functions [`require_base_dir`] /
/// [`require_control`] (not trait methods) for use at the **top of a provider's `load(spec)`**,
/// before the struct exists (so there is no `&self` yet).
pub trait ControlBranch: crate::Generator {
    /// The registry id of this control variant (e.g. `"z_image_turbo_control"`). Woven verbatim into
    /// the shared error messages so they read identically to the hand-written originals.
    fn model_id(&self) -> &'static str;

    /// Which control signals this branch admits (see [`AcceptedControlKinds`]). Defaults to the
    /// input-agnostic Fun-Union policy ([`AcceptedControlKinds::Any`]); a single-signal branch
    /// overrides (Qwen v1 → `Only([Pose])`).
    fn accepted_control_kinds(&self) -> AcceptedControlKinds {
        AcceptedControlKinds::Any
    }

    /// The "requires a `Control` conditioning" message (used by both [`resolve_control`] when none is
    /// present and [`require_control_present`]). Defaults to the Z-Image/FLUX.2 wording; Qwen
    /// overrides to its "(pose skeleton)" phrasing.
    fn missing_control_message(&self) -> String {
        format!(
            "{} requires a Control conditioning (the pose/union skeleton)",
            self.model_id()
        )
    }

    /// Extract the single [`Conditioning::Control`] image + its scale from a request, applying
    /// [`accepted_control_kinds`](Self::accepted_control_kinds). Errors when: a control `kind` is not
    /// admitted (rejected, not silently coerced); more than one `Control` is present; or none is
    /// present (uses [`missing_control_message`](Self::missing_control_message)).
    fn resolve_control<'a>(&self, req: &'a GenerationRequest) -> Result<(&'a Image, f32)> {
        let accepted = self.accepted_control_kinds();
        let mut found = None;
        for c in &req.conditioning {
            if let Conditioning::Control { image, kind, scale } = c {
                if !accepted.accepts(kind) {
                    return Err(Error::Msg(self.unsupported_kind_message(kind)));
                }
                if found.is_some() {
                    return Err(Error::Msg(format!(
                        "{}: a single control image is supported",
                        self.model_id()
                    )));
                }
                found = Some((image, *scale));
            }
        }
        found.ok_or_else(|| Error::Msg(self.missing_control_message()))
    }

    /// The message for a control `kind` rejected by [`accepted_control_kinds`](Self::accepted_control_kinds).
    /// Only reached when the policy is [`AcceptedControlKinds::Only`]; the default matches Qwen v1's
    /// "supports pose control only" wording.
    fn unsupported_kind_message(&self, kind: &ControlKind) -> String {
        format!(
            "{} v1 supports pose control only, got {kind:?}",
            self.model_id()
        )
    }

    /// The `validate()` tail shared by every control variant: a [`Conditioning::Control`] must be
    /// present (the capability floor — the provider's `validate_request` — runs just before this).
    fn require_control_present(&self, req: &GenerationRequest) -> Result<()> {
        if !req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Control { .. }))
        {
            return Err(Error::Msg(self.control_present_message()));
        }
        Ok(())
    }

    /// The message for [`require_control_present`](Self::require_control_present) when no `Control`
    /// is present. Defaults to [`missing_control_message`](Self::missing_control_message); Qwen
    /// overrides to its "(pose skeleton) conditioning" phrasing (which differs slightly from its
    /// resolve message — preserved byte-for-byte).
    fn control_present_message(&self) -> String {
        self.missing_control_message()
    }
}

/// Free-function form of "extract the base snapshot dir or error", for use at the top of a provider's
/// `load(spec)` (before the control struct exists, so there is no `&self`).
///
/// Returns the snapshot directory [`Path`], or an [`Error::Msg`] that names `model_id` and reads
/// identically to the hand-written loaders. `dir_label` is the model's own description of the
/// expected directory (Z-Image/Qwen: `"a base snapshot directory"`; FLUX.2: `"a FLUX.2-dev snapshot
/// directory"`), so the wording stays byte-identical. The existing `load_rejects_single_file_base`
/// smokes assert on the `"snapshot directory"` substring this produces.
pub fn require_base_dir<'a>(
    spec: &'a LoadSpec,
    model_id: &str,
    dir_label: &str,
) -> Result<&'a Path> {
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p.as_path()),
        WeightsSource::File(_) => Err(Error::Msg(format!(
            "{model_id} expects {dir_label} (tokenizer/ text_encoder/ transformer/ vae/) as \
             `weights`, not a single .safetensors file"
        ))),
    }
}

/// Free-function form of "the control checkpoint is required", for use in a provider's `load(spec)`.
///
/// Returns the [`WeightsSource`] of [`LoadSpec::control`], or an [`Error::Msg`] that names `model_id`
/// and `control_weights_label` (e.g. `"Fun-Controlnet-Union"`, `"FLUX.2-dev-Fun-Controlnet-Union"`,
/// `"InstantX Qwen-Image-ControlNet-Union"`). The existing `load_rejects_missing_control_weights`
/// smokes assert on the label substring this produces.
pub fn require_control<'a>(
    spec: &'a LoadSpec,
    model_id: &str,
    control_weights_label: &str,
) -> Result<&'a WeightsSource> {
    spec.control.as_ref().ok_or_else(|| {
        Error::Msg(format!(
            "{model_id} requires the {control_weights_label} weights — set LoadSpec::control \
             (e.g. with_control(WeightsSource::File(...)))"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::{Capabilities, GenerationRequest, Generator, Modality, ModelDescriptor};
    use crate::media::Image;
    use std::path::PathBuf;

    fn img() -> Image {
        Image {
            width: 4,
            height: 4,
            pixels: vec![0u8; 4 * 4 * 3],
        }
    }

    fn req_with(conditioning: Vec<Conditioning>) -> GenerationRequest {
        GenerationRequest {
            conditioning,
            ..Default::default()
        }
    }

    /// A minimal control struct to exercise the default trait methods without weights/tensors.
    struct Stub {
        descriptor: ModelDescriptor,
        accepted: AcceptedControlKinds,
    }

    impl Generator for Stub {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }
        fn validate(&self, _req: &GenerationRequest) -> Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            _req: &GenerationRequest,
            _on_progress: &mut dyn FnMut(crate::Progress),
        ) -> Result<crate::GenerationOutput> {
            unreachable!("stub does not generate")
        }
    }

    impl ControlBranch for Stub {
        fn model_id(&self) -> &'static str {
            "stub_control"
        }
        fn accepted_control_kinds(&self) -> AcceptedControlKinds {
            self.accepted.clone()
        }
    }

    fn stub(accepted: AcceptedControlKinds) -> Stub {
        Stub {
            descriptor: ModelDescriptor {
                id: "stub_control",
                family: "stub",
                backend: "mlx",
                modality: Modality::Image,
                capabilities: Capabilities::default(),
            },
            accepted,
        }
    }

    #[test]
    fn accepted_any_admits_everything() {
        let any = AcceptedControlKinds::Any;
        assert!(any.accepts(&ControlKind::Pose));
        assert!(any.accepts(&ControlKind::Canny));
        assert!(any.accepts(&ControlKind::Depth));
        assert!(any.accepts(&ControlKind::Other("seg".into())));
    }

    #[test]
    fn accepted_only_restricts() {
        let pose = AcceptedControlKinds::Only(vec![ControlKind::Pose]);
        assert!(pose.accepts(&ControlKind::Pose));
        assert!(!pose.accepts(&ControlKind::Canny));
        assert!(!pose.accepts(&ControlKind::Depth));
    }

    #[test]
    fn require_base_dir_accepts_dir_rejects_file() {
        let dir = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/snap")));
        assert_eq!(
            require_base_dir(&dir, "m", "a base snapshot directory").unwrap(),
            Path::new("/snap")
        );
        let file = LoadSpec::new(WeightsSource::File(PathBuf::from("/x.safetensors")));
        let err = require_base_dir(&file, "m", "a base snapshot directory")
            .unwrap_err()
            .to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    #[test]
    fn require_control_present_and_missing() {
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/snap")))
            .with_control(WeightsSource::File(PathBuf::from("/c.safetensors")));
        require_control(&spec, "m", "Fun-Controlnet-Union").unwrap();

        let bare = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/snap")));
        let err = require_control(&bare, "m", "Fun-Controlnet-Union")
            .unwrap_err()
            .to_string();
        assert!(err.contains("Fun-Controlnet-Union"), "got: {err}");
    }

    #[test]
    fn resolve_control_kind_dup_missing() {
        let s = stub(AcceptedControlKinds::Only(vec![ControlKind::Pose]));

        // present + admitted
        let r = req_with(vec![Conditioning::Control {
            image: img(),
            kind: ControlKind::Pose,
            scale: 0.7,
        }]);
        let (_, scale) = s.resolve_control(&r).unwrap();
        assert_eq!(scale, 0.7);

        // wrong kind rejected
        let r = req_with(vec![Conditioning::Control {
            image: img(),
            kind: ControlKind::Canny,
            scale: 1.0,
        }]);
        assert!(s.resolve_control(&r).is_err());

        // duplicate rejected
        let r = req_with(vec![
            Conditioning::Control {
                image: img(),
                kind: ControlKind::Pose,
                scale: 1.0,
            },
            Conditioning::Control {
                image: img(),
                kind: ControlKind::Pose,
                scale: 1.0,
            },
        ]);
        let err = s.resolve_control(&r).unwrap_err().to_string();
        assert!(err.contains("single control image"), "got: {err}");

        // missing rejected
        let any = stub(AcceptedControlKinds::Any);
        let r = req_with(vec![]);
        let err = any.resolve_control(&r).unwrap_err().to_string();
        assert!(err.contains("requires a Control"), "got: {err}");
    }

    #[test]
    fn require_control_present_checks_presence() {
        let s = stub(AcceptedControlKinds::Any);
        let r = req_with(vec![Conditioning::Control {
            image: img(),
            kind: ControlKind::Pose,
            scale: 1.0,
        }]);
        s.require_control_present(&r).unwrap();

        let r = req_with(vec![]);
        let err = s.require_control_present(&r).unwrap_err().to_string();
        assert!(err.contains("requires a Control"), "got: {err}");
    }
}
