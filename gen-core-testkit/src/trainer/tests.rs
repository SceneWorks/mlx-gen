//! The trainer testkit verifying itself: a configurable in-crate stub trainer drives each
//! conformance check, and one deliberately-broken variant per check proves the check fires
//! (sc-4895). The stub is pure-host (no tensor library), so these run on the Linux gen-core lane.

use std::path::PathBuf;

use super::*;
use gen_core::registry::TrainerRegistration;
use gen_core::runtime::LoadSpec;
use gen_core::{
    Error, Modality, NetworkType, Trainer, TrainerDescriptor, TrainingItem, TrainingOutput,
    TrainingProgress, TrainingRequest,
};

/// The registered stub id (round-trips through the registry, see the `inventory::submit!` below).
const STUB_ID: &str = "testkit_trainer_stub";
/// A stub id deliberately NOT registered — exercises the registry-check failure path.
const UNREG_ID: &str = "testkit_trainer_unregistered_stub";

/// Which contract guarantees the stub upholds. `good()` upholds all; each broken-stub test flips
/// exactly one to false and asserts the matching check fails.
#[derive(Clone, Copy)]
struct Behavior {
    /// `validate()` enforces the dataset/network-type floor (vs. rubber-stamping every request).
    honest_validate: bool,
    /// Emits a `TrainingProgress::Training` per optimizer step.
    emit_progress: bool,
    /// Checks `CancelFlag` in the caching + training loops and bails.
    honor_cancel: bool,
    /// On a cancelled-before-any-step run, returns the typed `Error::Canceled` (vs. a stringified
    /// `Error::Msg`).
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

struct StubTrainer {
    desc: TrainerDescriptor,
    behavior: Behavior,
}

fn stub_desc(id: &'static str) -> TrainerDescriptor {
    TrainerDescriptor {
        id,
        family: "testkit",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
    }
}

impl StubTrainer {
    fn new(id: &'static str, behavior: Behavior) -> Self {
        Self {
            desc: stub_desc(id),
            behavior,
        }
    }

    fn boxed(id: &'static str, behavior: Behavior) -> Box<dyn Trainer> {
        Box::new(Self::new(id, behavior))
    }

    /// The error a cancelled-before-any-step run surfaces — typed `Canceled` for the good stub, a
    /// stringified `Msg` for the broken one (the exact pre-sc-4895 family behavior).
    fn cancel_err(&self) -> Error {
        if self.behavior.typed_cancel {
            Error::Canceled
        } else {
            Error::Msg("stub trainer: training cancelled".to_owned())
        }
    }
}

impl Trainer for StubTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.desc
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        if !self.behavior.honest_validate {
            return Ok(());
        }
        if req.items.is_empty() {
            return Err(Error::Msg("stub trainer: dataset is empty".to_owned()));
        }
        match req.config.network_type {
            NetworkType::Lokr if !self.desc.supports_lokr => {
                Err(Error::Msg("stub trainer: LoKr not supported".to_owned()))
            }
            NetworkType::Lora if !self.desc.supports_lora => {
                Err(Error::Msg("stub trainer: LoRA not supported".to_owned()))
            }
            _ => Ok(()),
        }
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        if self.behavior.honest_validate {
            self.validate(req)?;
        }
        on_progress(TrainingProgress::Preparing);
        on_progress(TrainingProgress::LoadingModel);

        // --- cache (per-item, cancellable) ---
        let total = req.items.len() as u32;
        let mut cached = 0u32;
        for i in 0..req.items.len() {
            if self.behavior.honor_cancel && req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            cached += 1;
        }
        if cached == 0 {
            // Disambiguate (sc-4895): cancelled-during-caching → typed Canceled; otherwise a real
            // "no usable dataset items" error.
            if self.behavior.honor_cancel && req.cancel.is_cancelled() {
                return Err(self.cancel_err());
            }
            return Err(Error::Msg(
                "stub trainer: no usable dataset items".to_owned(),
            ));
        }

        // --- train (per-step, cancellable) ---
        let steps = req.config.steps;
        let mut steps_run = 0u32;
        for step in 1..=steps {
            if self.behavior.honor_cancel && req.cancel.is_cancelled() {
                break;
            }
            steps_run = step;
            if self.behavior.emit_progress {
                on_progress(TrainingProgress::Training {
                    step,
                    total: steps,
                    loss: 1.0 / step as f32,
                });
            }
        }
        if steps_run == 0 {
            // Cancelled before any step → typed Canceled, no adapter written (F-040).
            return Err(self.cancel_err());
        }

        on_progress(TrainingProgress::Saving);
        Ok(TrainingOutput {
            adapter_path: req.output_dir.join(&req.file_name),
            steps: steps_run,
            final_loss: 0.0,
        })
    }
}

// Register the good stub so the registry round-trip resolves its id.
fn stub_descriptor() -> TrainerDescriptor {
    stub_desc(STUB_ID)
}
fn stub_load(_spec: &LoadSpec) -> gen_core::Result<Box<dyn Trainer>> {
    Ok(StubTrainer::boxed(STUB_ID, Behavior::good()))
}
inventory::submit! {
    TrainerRegistration { descriptor: stub_descriptor, load: stub_load }
}

fn item(name: &str) -> TrainingItem {
    TrainingItem {
        image_path: PathBuf::from(format!("/nonexistent/{name}.png")),
        caption: format!("a {name}"),
    }
}

fn profile() -> TrainerProfile {
    // Two dummy items (the stub never reads them); 2 steps via `cheap`.
    TrainerProfile::cheap(
        vec![item("red"), item("blue")],
        std::env::temp_dir().join("gen_core_testkit_trainer_stub"),
    )
}

fn make_good() -> Box<dyn Trainer> {
    StubTrainer::boxed(STUB_ID, Behavior::good())
}

#[test]
fn good_stub_passes_full_conformance() {
    trainer_conformance(make_good, &profile());
}

#[test]
fn good_stub_passes_every_check_individually() {
    let mut g = StubTrainer::new(STUB_ID, Behavior::good());
    check_trainer_validate(&g, &profile()).unwrap();
    check_trainer_progress(&mut g, &profile()).unwrap();
    check_trainer_cancellation(&make_good, &profile()).unwrap();
    check_trainer_registry(&g).unwrap();
}

#[test]
fn dishonest_validate_fails_validate_check() {
    let g = StubTrainer::new(
        STUB_ID,
        Behavior {
            honest_validate: false,
            ..Behavior::good()
        },
    );
    assert!(check_trainer_validate(&g, &profile()).is_err());
}

#[test]
fn missing_progress_fails_progress_check() {
    let mut g = StubTrainer::new(
        STUB_ID,
        Behavior {
            emit_progress: false,
            ..Behavior::good()
        },
    );
    let err = check_trainer_progress(&mut g, &profile()).unwrap_err();
    assert!(err.contains("Training"), "got: {err}");
}

#[test]
fn ignoring_cancel_fails_cancellation_check() {
    let err = check_trainer_cancellation(
        &|| {
            StubTrainer::boxed(
                STUB_ID,
                Behavior {
                    honor_cancel: false,
                    ..Behavior::good()
                },
            )
        },
        &profile(),
    )
    .unwrap_err();
    assert!(err.contains("returned Ok"), "got: {err}");
}

#[test]
fn stringified_cancel_fails_cancellation_check() {
    // The exact pre-sc-4895 family behavior: stops early but returns Error::Msg, not Canceled.
    let err = check_trainer_cancellation(
        &|| {
            StubTrainer::boxed(
                STUB_ID,
                Behavior {
                    typed_cancel: false,
                    ..Behavior::good()
                },
            )
        },
        &profile(),
    )
    .unwrap_err();
    assert!(err.contains("typed Err(Error::Canceled)"), "got: {err}");
}

#[test]
fn unregistered_id_fails_registry_check() {
    let g = StubTrainer::new(UNREG_ID, Behavior::good());
    assert!(check_trainer_registry(&g).is_err());
}

#[test]
#[should_panic(expected = "conformance FAILED")]
fn conformance_panics_on_a_broken_stub() {
    trainer_conformance(
        || {
            StubTrainer::boxed(
                STUB_ID,
                Behavior {
                    honor_cancel: false,
                    ..Behavior::good()
                },
            )
        },
        &profile(),
    );
}
