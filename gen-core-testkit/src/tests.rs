//! The testkit verifying itself: a configurable in-crate stub generator drives each conformance
//! check, and one deliberately-broken variant per check proves the check actually fires (the
//! sc-4481 AC). The stub is pure-host (no tensor library), so these run on the Linux gen-core lane.

use super::*;
use gen_core::registry::ModelRegistration;
use gen_core::runtime::LoadSpec;
use gen_core::{
    Capabilities, ConditioningKind, Error, GenerationOutput, GenerationRequest, Generator, Image,
    Modality, ModelDescriptor, Progress,
};
use std::cell::Cell;

/// The registered stub id (round-trips through the registry, see the `inventory::submit!` below).
const STUB_ID: &str = "testkit_stub";
/// A stub id that is deliberately NOT registered — exercises the registry-check failure path.
const UNREG_ID: &str = "testkit_unregistered_stub";

/// Which contract guarantees the stub upholds. `good()` upholds all of them; each broken-stub test
/// flips exactly one to false and asserts the matching check fails.
#[derive(Clone, Copy)]
struct Behavior {
    /// `validate()` enforces the capability floor (vs. rubber-stamping every request).
    honest_validate: bool,
    /// Emits a `Progress::Step` per denoise iteration.
    emit_progress: bool,
    /// Number of `Progress::Decoding` events emitted after the step loop (contract requires exactly 1).
    decoding_events: u32,
    /// Emit `Step.current` up to `2*total` — the F-050 multi-eval-sampler overrun (>100%).
    overrun_steps: bool,
    /// Stop emitting `Step` at `total - 1` while still advertising `total` — the F-030 frozen-below-total
    /// (PiD early-stop) shape.
    freeze_below_total: bool,
    /// Checks `CancelFlag` at each step boundary and bails.
    honor_cancel: bool,
    /// On cancel, returns the typed `Error::Canceled` (vs. a stringified `Error::Msg`).
    typed_cancel: bool,
    /// Output pixels depend only on the seed (vs. drifting per call).
    deterministic: bool,
}

impl Behavior {
    fn good() -> Self {
        Self {
            honest_validate: true,
            emit_progress: true,
            decoding_events: 1,
            overrun_steps: false,
            freeze_below_total: false,
            honor_cancel: true,
            typed_cancel: true,
            deterministic: true,
        }
    }
}

struct Stub {
    desc: ModelDescriptor,
    behavior: Behavior,
    /// Per-instance call counter — the nondeterministic variant fills pixels from this.
    runs: Cell<u32>,
}

fn stub_caps() -> Capabilities {
    Capabilities {
        conditioning: vec![ConditioningKind::Reference],
        min_size: 64,
        max_size: 512,
        max_count: 4,
        ..Default::default()
    }
}

fn stub_desc(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        id,
        family: "testkit",
        backend: "stub",
        modality: Modality::Image,
        capabilities: stub_caps(),
    }
}

impl Stub {
    fn new(id: &'static str, behavior: Behavior) -> Self {
        Self {
            desc: stub_desc(id),
            behavior,
            runs: Cell::new(0),
        }
    }

    fn boxed(id: &'static str, behavior: Behavior) -> Box<dyn Generator> {
        Box::new(Self::new(id, behavior))
    }
}

impl Generator for Stub {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.desc
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        if self.behavior.honest_validate {
            self.desc.capabilities.validate_request(self.desc.id, req)
        } else {
            Ok(())
        }
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        if self.behavior.honest_validate {
            self.validate(req)?;
        }
        let total = req.steps.unwrap_or(2);
        let run = self.runs.get();
        self.runs.set(run + 1);
        // How many Step events actually get emitted: `total` (good), `2*total` (F-050 overrun),
        // or `total - 1` (F-030 frozen below its advertised total).
        let emit_max = if self.behavior.overrun_steps {
            total.saturating_mul(2)
        } else if self.behavior.freeze_below_total {
            total.saturating_sub(1)
        } else {
            total
        };
        for i in 1..=emit_max {
            if self.behavior.honor_cancel && req.cancel.is_cancelled() {
                return Err(if self.behavior.typed_cancel {
                    Error::Canceled
                } else {
                    Error::Msg("generation cancelled".into())
                });
            }
            if self.behavior.emit_progress {
                on_progress(Progress::Step { current: i, total });
            }
        }
        for _ in 0..self.behavior.decoding_events {
            on_progress(Progress::Decoding);
        }
        let fill = if self.behavior.deterministic {
            req.seed.unwrap_or(0) as u8
        } else {
            run as u8
        };
        let img = Image {
            width: req.width,
            height: req.height,
            pixels: vec![fill; req.width as usize * req.height as usize * 3],
        };
        Ok(GenerationOutput::Images(vec![img]))
    }
}

// Register the good stub so the registry round-trip resolves its id. This is the only
// ModelRegistration in the testkit's test binary, so the registry contains exactly this one.
fn stub_descriptor() -> ModelDescriptor {
    stub_desc(STUB_ID)
}
fn stub_load(_spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Stub::boxed(STUB_ID, Behavior::good()))
}
inventory::submit! {
    // `footprint: None` — this stub declares no per-component footprint (sc-10894); the field defaults
    // to `None` for every registration that does not set it (the `register_generators!` macro likewise).
    ModelRegistration { descriptor: stub_descriptor, load: stub_load, footprint: None }
}

fn cheap() -> Profile {
    Profile::cheap()
}

#[test]
fn good_stub_passes_full_conformance() {
    conformance(|| Stub::boxed(STUB_ID, Behavior::good()), &cheap());
}

#[test]
fn good_stub_passes_every_check_individually() {
    let g = Stub::new(STUB_ID, Behavior::good());
    check_validate_honesty(&g, &cheap()).unwrap();
    check_progress(&g, &cheap()).unwrap();
    check_progress_contract(&g, &cheap()).unwrap();
    check_cancellation(&g, &cheap()).unwrap();
    check_precancellation(&g, &cheap()).unwrap();
    check_seed_determinism(&g, &cheap()).unwrap();
    check_registry_roundtrip(&g).unwrap();
}

#[test]
fn ignoring_cancel_fails_precancellation_check() {
    // A provider that never consults the flag runs to completion even on an already-cancelled
    // request — the non-denoise-seam class (sc-11128): it returns Ok instead of typed Canceled.
    let g = Stub::new(
        STUB_ID,
        Behavior {
            honor_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_precancellation(&g, &cheap()).unwrap_err();
    assert!(err.contains("returned Ok"), "got: {err}");
}

#[test]
fn stringified_cancel_fails_precancellation_check() {
    // Honors the flag up front but stringifies the error — must still fail the typed contract.
    let g = Stub::new(
        STUB_ID,
        Behavior {
            typed_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_precancellation(&g, &cheap()).unwrap_err();
    assert!(err.contains("typed Err(Error::Canceled)"), "got: {err}");
}

#[test]
fn dishonest_validate_fails_validate_check() {
    let g = Stub::new(
        STUB_ID,
        Behavior {
            honest_validate: false,
            ..Behavior::good()
        },
    );
    assert!(check_validate_honesty(&g, &cheap()).is_err());
}

#[test]
fn missing_progress_fails_progress_check() {
    let g = Stub::new(
        STUB_ID,
        Behavior {
            emit_progress: false,
            ..Behavior::good()
        },
    );
    assert!(check_progress(&g, &cheap()).is_err());
}

#[test]
fn overrunning_steps_fail_progress_contract() {
    // The F-050 class: a multi-eval sampler double-counts and reports current up to 2*total.
    let g = Stub::new(
        STUB_ID,
        Behavior {
            overrun_steps: true,
            ..Behavior::good()
        },
    );
    let err = check_progress_contract(&g, &cheap()).unwrap_err();
    assert!(err.contains("exceeds total"), "got: {err}");
}

#[test]
fn freezing_below_total_fails_progress_contract() {
    // The F-030 class: an early-stopped schedule never reaches its advertised total.
    let g = Stub::new(
        STUB_ID,
        Behavior {
            freeze_below_total: true,
            ..Behavior::good()
        },
    );
    let err = check_progress_contract(&g, &cheap()).unwrap_err();
    assert!(err.contains("must reach"), "got: {err}");
}

#[test]
fn missing_decoding_fails_progress_contract() {
    // The F-030 class: the decode stage is invisible because Decoding is never emitted.
    let g = Stub::new(
        STUB_ID,
        Behavior {
            decoding_events: 0,
            ..Behavior::good()
        },
    );
    let err = check_progress_contract(&g, &cheap()).unwrap_err();
    assert!(err.contains("emitted 0 times"), "got: {err}");
}

#[test]
fn repeated_decoding_fails_progress_contract() {
    // The F-136/F-162 restarting-bar class: Decoding (or the bar) restarts per output.
    let g = Stub::new(
        STUB_ID,
        Behavior {
            decoding_events: 3,
            ..Behavior::good()
        },
    );
    let err = check_progress_contract(&g, &cheap()).unwrap_err();
    assert!(err.contains("emitted 3 times"), "got: {err}");
}

#[test]
fn ignoring_cancel_fails_cancellation_check() {
    let g = Stub::new(
        STUB_ID,
        Behavior {
            honor_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_cancellation(&g, &cheap()).unwrap_err();
    assert!(err.contains("ran to completion"), "got: {err}");
}

#[test]
fn stringified_cancel_fails_cancellation_check() {
    // The exact pre-sc-4481 family behavior: stops early but returns Error::Msg, not Canceled.
    let g = Stub::new(
        STUB_ID,
        Behavior {
            typed_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_cancellation(&g, &cheap()).unwrap_err();
    assert!(err.contains("typed Err(Error::Canceled)"), "got: {err}");
}

#[test]
fn nondeterministic_fails_seed_check() {
    let g = Stub::new(
        STUB_ID,
        Behavior {
            deterministic: false,
            ..Behavior::good()
        },
    );
    assert!(check_seed_determinism(&g, &cheap()).is_err());
}

#[test]
fn unregistered_id_fails_registry_check() {
    let g = Stub::new(UNREG_ID, Behavior::good());
    assert!(check_registry_roundtrip(&g).is_err());
}

/// The weights-free descriptor sweep (sc-9098, F-009) is clean over this binary's registry (the
/// good stub is its only registration). The per-violation firing is unit-tested next to the checks
/// in `gen_core::registry`.
#[test]
fn registry_sweep_passes_for_the_registered_stub() {
    registry_conformance();
}

/// `check_progress_with` accepts a request-supplied run (the SVD/SeedVR2/renderer shape) and flags
/// a resolved-total mismatch when `expected_total` is pinned.
#[test]
fn progress_with_checks_request_supplied_runs() {
    let g = Stub::new(STUB_ID, Behavior::good());
    let req = GenerationRequest {
        prompt: "a fox".into(),
        width: 128,
        height: 128,
        steps: Some(3),
        seed: Some(7),
        ..Default::default()
    };
    check_progress_with(&g, &req, Some(3)).unwrap();
    check_progress_with(&g, &req, None).unwrap();
    let err = check_progress_with(&g, &req, Some(5)).unwrap_err();
    assert!(err.contains("expected resolved step count"), "got: {err}");
}

#[test]
#[should_panic(expected = "conformance FAILED")]
fn conformance_panics_on_a_broken_stub() {
    conformance(
        || {
            Stub::boxed(
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
