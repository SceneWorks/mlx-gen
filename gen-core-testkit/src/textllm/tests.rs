//! The text-LLM testkit verifying itself: a configurable in-crate stub provider drives each
//! conformance check, and one deliberately-broken variant per check proves the check fires
//! (sc-5500). The stub is pure-host (no tensor library), so these run on the Linux gen-core lane.

use super::*;
use gen_core::registry::TextLlmRegistration;
use gen_core::runtime::LoadSpec;
use gen_core::{
    Error, Progress, TextLlm, TextLlmCapabilities, TextLlmDescriptor, TextLlmFinishReason,
    TextLlmOutput, TextLlmRequest,
};

const STUB_ID: &str = "testkit_textllm_stub";
const UNREG_ID: &str = "testkit_textllm_unregistered_stub";

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

struct StubTextLlm {
    desc: TextLlmDescriptor,
    behavior: Behavior,
}

fn stub_caps() -> TextLlmCapabilities {
    TextLlmCapabilities {
        // Unsupported so the conditional validate-honesty negative fires.
        supports_system_prompt: false,
        max_prompt_chars: 8000,
        max_system_chars: 16000,
        max_new_tokens: 1024,
        ..Default::default()
    }
}

fn stub_desc(id: &'static str) -> TextLlmDescriptor {
    TextLlmDescriptor {
        id,
        family: "testkit",
        backend: "stub",
        capabilities: stub_caps(),
    }
}

impl StubTextLlm {
    fn new(id: &'static str, behavior: Behavior) -> Self {
        Self {
            desc: stub_desc(id),
            behavior,
        }
    }

    fn boxed(id: &'static str, behavior: Behavior) -> Box<dyn TextLlm> {
        Box::new(Self::new(id, behavior))
    }

    fn cancel_err(&self) -> Error {
        if self.behavior.typed_cancel {
            Error::Canceled
        } else {
            Error::Msg("stub textllm: cancelled".to_owned())
        }
    }
}

impl TextLlm for StubTextLlm {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.desc
    }

    fn validate(&self, req: &TextLlmRequest) -> gen_core::Result<()> {
        if self.behavior.honest_validate {
            self.desc.capabilities.validate_request(self.desc.id, req)
        } else {
            Ok(())
        }
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<TextLlmOutput> {
        if self.behavior.honest_validate {
            self.validate(req)?;
        }
        // Check the flag up front, before any "inference".
        if self.behavior.honor_cancel && req.cancel.is_cancelled() {
            return Err(self.cancel_err());
        }
        if self.behavior.emit_progress {
            on_progress(Progress::Step {
                current: 1,
                total: 2,
            });
            on_progress(Progress::Step {
                current: 2,
                total: 2,
            });
        }
        Ok(TextLlmOutput {
            text: "a stub rewrite".to_owned(),
            generated_tokens: Some(2),
            finish_reason: Some(TextLlmFinishReason::StopToken),
        })
    }
}

fn stub_descriptor() -> TextLlmDescriptor {
    stub_desc(STUB_ID)
}
fn stub_load(_spec: &LoadSpec) -> gen_core::Result<Box<dyn TextLlm>> {
    Ok(StubTextLlm::boxed(STUB_ID, Behavior::good()))
}
inventory::submit! {
    TextLlmRegistration { descriptor: stub_descriptor, load: stub_load }
}

fn cheap() -> TextLlmProfile {
    TextLlmProfile::cheap()
}

#[test]
fn good_stub_passes_full_conformance() {
    textllm_conformance(|| StubTextLlm::boxed(STUB_ID, Behavior::good()), &cheap());
}

#[test]
fn good_stub_passes_every_check_individually() {
    let c = StubTextLlm::new(STUB_ID, Behavior::good());
    check_textllm_validate(&c, &cheap()).unwrap();
    check_textllm_progress(&c, &cheap()).unwrap();
    check_textllm_cancellation(&c, &cheap()).unwrap();
    check_textllm_registry(&c).unwrap();
}

#[test]
fn dishonest_validate_fails_validate_check() {
    let c = StubTextLlm::new(
        STUB_ID,
        Behavior {
            honest_validate: false,
            ..Behavior::good()
        },
    );
    assert!(check_textllm_validate(&c, &cheap()).is_err());
}

#[test]
fn missing_progress_fails_progress_check() {
    let c = StubTextLlm::new(
        STUB_ID,
        Behavior {
            emit_progress: false,
            ..Behavior::good()
        },
    );
    assert!(check_textllm_progress(&c, &cheap()).is_err());
}

#[test]
fn ignoring_cancel_fails_cancellation_check() {
    let c = StubTextLlm::new(
        STUB_ID,
        Behavior {
            honor_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_textllm_cancellation(&c, &cheap()).unwrap_err();
    assert!(err.contains("returned Ok"), "got: {err}");
}

#[test]
fn stringified_cancel_fails_cancellation_check() {
    let c = StubTextLlm::new(
        STUB_ID,
        Behavior {
            typed_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_textllm_cancellation(&c, &cheap()).unwrap_err();
    assert!(err.contains("typed Err(Error::Canceled)"), "got: {err}");
}

#[test]
fn unregistered_id_fails_registry_check() {
    let c = StubTextLlm::new(UNREG_ID, Behavior::good());
    assert!(check_textllm_registry(&c).is_err());
}

#[test]
#[should_panic(expected = "conformance FAILED")]
fn conformance_panics_on_a_broken_stub() {
    textllm_conformance(
        || {
            StubTextLlm::boxed(
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
