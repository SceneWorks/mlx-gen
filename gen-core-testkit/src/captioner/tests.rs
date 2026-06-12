//! The captioner testkit verifying itself: a configurable in-crate stub captioner drives each
//! conformance check, and one deliberately-broken variant per check proves the check fires
//! (sc-4895). The stub is pure-host (no tensor library), so these run on the Linux gen-core lane.

use super::*;
use gen_core::registry::CaptionerRegistration;
use gen_core::runtime::LoadSpec;
use gen_core::{
    CaptionCapabilities, CaptionFinishReason, CaptionOutput, CaptionRequest, Captioner,
    CaptionerDescriptor, Error, Progress,
};

const STUB_ID: &str = "testkit_captioner_stub";
const UNREG_ID: &str = "testkit_captioner_unregistered_stub";

/// Which contract guarantees the stub upholds. `good()` upholds all; each broken-stub test flips
/// exactly one to false and asserts the matching check fails.
#[derive(Clone, Copy)]
struct Behavior {
    /// `validate()` enforces the advertised capability surface (vs. rubber-stamping every request).
    honest_validate: bool,
    /// Emits `Progress::Step` events.
    emit_progress: bool,
    /// Checks `CancelFlag` before inference and bails.
    honor_cancel: bool,
    /// On cancel, returns the typed `Error::Canceled` (vs. a stringified `Error::Msg`).
    typed_cancel: bool,
}

impl Behavior {
    fn good() -> Self {
        Self {
            honest_validate: true,
            emit_progress: true,
            honor_cancel: true,
            typed_cancel: true,
        }
    }
}

struct StubCaptioner {
    desc: CaptionerDescriptor,
    behavior: Behavior,
}

fn stub_caps() -> CaptionCapabilities {
    CaptionCapabilities {
        // Both unsupported so the conditional validate-honesty negatives fire.
        supports_custom_prompt: false,
        supports_low_vram: false,
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
    }
}

fn stub_desc(id: &'static str) -> CaptionerDescriptor {
    CaptionerDescriptor {
        id,
        family: "testkit",
        capabilities: stub_caps(),
    }
}

impl StubCaptioner {
    fn new(id: &'static str, behavior: Behavior) -> Self {
        Self {
            desc: stub_desc(id),
            behavior,
        }
    }

    fn boxed(id: &'static str, behavior: Behavior) -> Box<dyn Captioner> {
        Box::new(Self::new(id, behavior))
    }

    fn cancel_err(&self) -> Error {
        if self.behavior.typed_cancel {
            Error::Canceled
        } else {
            Error::Msg("stub captioner: cancelled".to_owned())
        }
    }
}

impl Captioner for StubCaptioner {
    fn descriptor(&self) -> &CaptionerDescriptor {
        &self.desc
    }

    fn validate(&self, req: &CaptionRequest) -> gen_core::Result<()> {
        if self.behavior.honest_validate {
            self.desc.capabilities.validate_request(self.desc.id, req)
        } else {
            Ok(())
        }
    }

    fn caption(
        &self,
        req: &CaptionRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<CaptionOutput> {
        if self.behavior.honest_validate {
            self.validate(req)?;
        }
        // Check the flag up front, before any "inference" (mirrors JoyCaption's prompt_embeds guard).
        if self.behavior.honor_cancel && req.cancel.is_cancelled() {
            return Err(self.cancel_err());
        }
        if self.behavior.emit_progress {
            on_progress(Progress::Step {
                current: 1,
                total: 2,
            });
        }
        if self.behavior.emit_progress {
            on_progress(Progress::Step {
                current: 2,
                total: 2,
            });
        }
        Ok(CaptionOutput {
            text: "a stub caption".to_owned(),
            generated_tokens: Some(2),
            finish_reason: Some(CaptionFinishReason::StopToken),
        })
    }
}

fn stub_descriptor() -> CaptionerDescriptor {
    stub_desc(STUB_ID)
}
fn stub_load(_spec: &LoadSpec) -> gen_core::Result<Box<dyn Captioner>> {
    Ok(StubCaptioner::boxed(STUB_ID, Behavior::good()))
}
inventory::submit! {
    CaptionerRegistration { descriptor: stub_descriptor, load: stub_load }
}

fn cheap() -> CaptionerProfile {
    CaptionerProfile::cheap()
}

#[test]
fn good_stub_passes_full_conformance() {
    captioner_conformance(|| StubCaptioner::boxed(STUB_ID, Behavior::good()), &cheap());
}

#[test]
fn good_stub_passes_every_check_individually() {
    let c = StubCaptioner::new(STUB_ID, Behavior::good());
    check_captioner_validate(&c, &cheap()).unwrap();
    check_captioner_progress(&c, &cheap()).unwrap();
    check_captioner_cancellation(&c, &cheap()).unwrap();
    check_captioner_registry(&c).unwrap();
}

#[test]
fn dishonest_validate_fails_validate_check() {
    let c = StubCaptioner::new(
        STUB_ID,
        Behavior {
            honest_validate: false,
            ..Behavior::good()
        },
    );
    assert!(check_captioner_validate(&c, &cheap()).is_err());
}

#[test]
fn missing_progress_fails_progress_check() {
    let c = StubCaptioner::new(
        STUB_ID,
        Behavior {
            emit_progress: false,
            ..Behavior::good()
        },
    );
    assert!(check_captioner_progress(&c, &cheap()).is_err());
}

#[test]
fn ignoring_cancel_fails_cancellation_check() {
    let c = StubCaptioner::new(
        STUB_ID,
        Behavior {
            honor_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_captioner_cancellation(&c, &cheap()).unwrap_err();
    assert!(err.contains("returned Ok"), "got: {err}");
}

#[test]
fn stringified_cancel_fails_cancellation_check() {
    let c = StubCaptioner::new(
        STUB_ID,
        Behavior {
            typed_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_captioner_cancellation(&c, &cheap()).unwrap_err();
    assert!(err.contains("typed Err(Error::Canceled)"), "got: {err}");
}

#[test]
fn unregistered_id_fails_registry_check() {
    let c = StubCaptioner::new(UNREG_ID, Behavior::good());
    assert!(check_captioner_registry(&c).is_err());
}

#[test]
#[should_panic(expected = "conformance FAILED")]
fn conformance_panics_on_a_broken_stub() {
    captioner_conformance(
        || {
            StubCaptioner::boxed(
                STUB_ID,
                Behavior {
                    honor_cancel: false,
                    ..Behavior::good()
                },
            )
        },
        &cheap(),
    );
}
