//! Contract conformance for [`gen_core::Trainer`] providers — the training analog of the
//! [`Generator`](crate::conformance) suite (epic 3720, sc-4895). It exercises the behavioral
//! guarantees the [`Trainer`] contract promises but cannot express in the type system: typed
//! cancellation, `TrainingProgress` monotonicity, capability honesty, and registry discoverability.
//!
//! ## Trainer-specific cancellation semantics
//!
//! Unlike a generator (where any cancel → `Canceled`), a trainer that has already completed ≥1
//! optimizer step on cancel returns a **partial** `Ok` — a legitimately-trained adapter with
//! `TrainingOutput.steps < config.steps` (the documented "stopped early" result). The **typed
//! `Err(Error::Canceled)`** contract therefore covers cancellation *before any step runs*:
//!
//! * cancel tripped during dataset caching ⇒ the cache loop breaks, and (depending on whether any
//!   item was cached first) either the empty-cache disambiguation or the `steps_run == 0` guard
//!   returns `Canceled` — never a valid-looking identity adapter (F-040);
//! * a pre-cancelled request ⇒ the cache loop breaks on the first item, and the empty-cache
//!   disambiguation returns `Canceled`.
//!
//! The check drives both paths. Because `train` is `&mut self` and several families are single-use
//! (LTX/Wan free their text encoder after caching), the cancellation check takes a `make` closure
//! and constructs a **fresh** trainer per sub-check rather than sharing one instance.

use std::cell::Cell;
use std::path::PathBuf;

use gen_core::{
    Error, NetworkType, Trainer, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
};

/// Parameters for a conformance run. Keep `config.steps` and the dataset tiny — the suite trains a
/// real (if minimal) run for the progress check, so the macOS-lane cost is dominated by it.
#[derive(Clone, Debug)]
pub struct TrainerProfile {
    /// The dataset. The progress check trains over it, so the real lane must point these at actual
    /// images; the Linux stub never reads them. Keep it to 1–2 items.
    pub items: Vec<TrainingItem>,
    /// Hyperparameters. The progress check asserts `TrainingProgress::Training.total == config.steps`
    /// and that the run completes `config.steps`, so keep `steps` small (2) and `save_every` at 0.
    pub config: TrainingConfig,
    /// Where the (final/checkpoint) adapter is written. The cancellation checks assert nothing is
    /// written here; the progress check writes one cheap adapter.
    pub output_dir: PathBuf,
    /// Output adapter file name.
    pub file_name: String,
}

impl TrainerProfile {
    /// The cheapest generally-valid profile: a 2-step run at a 64px bucket, rank 8, no intermediate
    /// checkpoints, over the supplied dataset. `output_dir` is where a passing progress run writes
    /// its one adapter (use a temp dir).
    pub fn cheap(items: Vec<TrainingItem>, output_dir: PathBuf) -> Self {
        Self {
            items,
            config: TrainingConfig {
                rank: 8,
                alpha: 8.0,
                learning_rate: 1e-3,
                steps: 2,
                resolution: 64,
                save_every: 0,
                seed: 7,
                ..Default::default()
            },
            output_dir,
            file_name: "conformance_lora.safetensors".to_owned(),
        }
    }
}

/// Build the in-capability request the positive checks train from, with a fresh cancel flag.
fn base_request(profile: &TrainerProfile) -> TrainingRequest {
    TrainingRequest {
        items: profile.items.clone(),
        config: profile.config.clone(),
        output_dir: profile.output_dir.clone(),
        file_name: profile.file_name.clone(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    }
}

/// **Validate honesty.** A declared, in-capability request is accepted; an empty dataset is
/// rejected (the universal floor); and a network type the descriptor does **not** advertise is
/// rejected — all by `validate()`, before any expensive work.
pub fn check_trainer_validate(t: &dyn Trainer, profile: &TrainerProfile) -> Result<(), String> {
    let desc = t.descriptor();
    let id = desc.id;

    // Positive: a declared request using a supported network type must be accepted. Prefer LoRA if
    // supported (every family does), else LoKr.
    let mut ok = base_request(profile);
    ok.config.network_type = if desc.supports_lora {
        NetworkType::Lora
    } else {
        NetworkType::Lokr
    };
    t.validate(&ok).map_err(|e| {
        format!(
            "validate-honesty[{id}]: the declared cheap request was rejected by validate(): {e}"
        )
    })?;

    // Negative: an empty dataset must be rejected before any work.
    let mut empty = base_request(profile);
    empty.items.clear();
    if t.validate(&empty).is_ok() {
        return Err(format!(
            "validate-honesty[{id}]: an empty dataset was accepted by validate()"
        ));
    }

    // Negative: a network type the descriptor does not advertise must be rejected.
    if !desc.supports_lokr {
        let mut lokr = base_request(profile);
        lokr.config.network_type = NetworkType::Lokr;
        if t.validate(&lokr).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: a LoKr request was accepted by validate() despite \
                 supports_lokr == false"
            ));
        }
    }
    if !desc.supports_lora {
        let mut lora = base_request(profile);
        lora.config.network_type = NetworkType::Lora;
        if t.validate(&lora).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: a LoRA request was accepted by validate() despite \
                 supports_lora == false"
            ));
        }
    }
    Ok(())
}

/// **Progress.** A completed (uncancelled) run streams `TrainingProgress::Caching` over exactly
/// `1..=items.len()` and `TrainingProgress::Training` over exactly `1..=config.steps` (monotone,
/// complete, constant `total`), and `TrainingOutput.steps == config.steps`.
pub fn check_trainer_progress(t: &mut dyn Trainer, profile: &TrainerProfile) -> Result<(), String> {
    let id = t.descriptor().id;
    let req = base_request(profile);
    let mut caching: Vec<(u32, u32)> = Vec::new();
    let mut training: Vec<(u32, u32)> = Vec::new();
    let out = t
        .train(&req, &mut |p| match p {
            TrainingProgress::Caching { current, total } => caching.push((current, total)),
            TrainingProgress::Training { step, total, .. } => training.push((step, total)),
            _ => {}
        })
        .map_err(|e| format!("progress[{id}]: train() failed on the cheap request: {e}"))?;

    check_monotone(id, "Caching", &caching, profile.items.len() as u32)?;
    check_monotone(id, "Training", &training, profile.config.steps)?;

    if out.steps != profile.config.steps {
        return Err(format!(
            "progress[{id}]: TrainingOutput.steps ({}) != config.steps ({}) on an uncancelled run",
            out.steps, profile.config.steps
        ));
    }
    Ok(())
}

/// Shared monotonicity assertion for a `(current, total)` event stream: `total` constant and equal
/// to `expected_total`, `current` exactly `1..=expected_total`.
fn check_monotone(
    id: &str,
    band: &str,
    events: &[(u32, u32)],
    expected_total: u32,
) -> Result<(), String> {
    if events.is_empty() {
        return Err(format!(
            "progress[{id}]: train() emitted no TrainingProgress::{band} events"
        ));
    }
    let total = events[0].1;
    if let Some((c, t)) = events.iter().find(|(_, t)| *t != total) {
        return Err(format!(
            "progress[{id}]: {band}.total changed mid-run ({total} then {t} at current={c})"
        ));
    }
    let observed: Vec<u32> = events.iter().map(|(c, _)| *c).collect();
    let expected: Vec<u32> = (1..=total).collect();
    if observed != expected {
        return Err(format!(
            "progress[{id}]: {band}.current must be exactly 1..={total} (monotone, complete, no \
             repeats); got {observed:?}"
        ));
    }
    if total != expected_total {
        return Err(format!(
            "progress[{id}]: {band}.total ({total}) != the expected count ({expected_total})"
        ));
    }
    Ok(())
}

/// **Cancellation.** Cancelling before any optimizer step runs makes `train` return the **typed**
/// `Err(Error::Canceled)` (not a stringified `Msg`) and write **no** adapter (no `Saving` event).
/// Two paths are exercised against fresh trainers:
///
/// 1. **pre-cancelled** — the cache loop breaks on the first item, empty-cache disambiguation;
/// 2. **cancel during caching** — tripped at the first `Caching` event, so ≥1 item caches but the
///    training loop breaks before step 1 (`steps_run == 0` guard).
pub fn check_trainer_cancellation(
    make: &dyn Fn() -> Box<dyn Trainer>,
    profile: &TrainerProfile,
) -> Result<(), String> {
    // Path 1: a request that is already cancelled when train() is called.
    {
        let mut t = make();
        let id = t.descriptor().id;
        let req = base_request(profile);
        req.cancel.cancel();
        let mut saved = false;
        let result = t.train(&req, &mut |p| {
            if matches!(p, TrainingProgress::Saving) {
                saved = true;
            }
        });
        classify_cancel(id, "pre-cancelled", result, saved)?;
    }

    // Path 2: cancellation tripped at the first Caching event (≥1 item cached, then the training
    // loop breaks before any step → the steps_run == 0 guard).
    {
        let mut t = make();
        let id = t.descriptor().id;
        let req = base_request(profile);
        let cancel = req.cancel.clone();
        let tripped = Cell::new(false);
        let mut saved = false;
        let result = t.train(&req, &mut |p| match p {
            TrainingProgress::Caching { .. } => {
                if !tripped.get() {
                    cancel.cancel();
                    tripped.set(true);
                }
            }
            TrainingProgress::Saving => saved = true,
            _ => {}
        });
        if !tripped.get() {
            return Err(format!(
                "cancellation[{id}]: no TrainingProgress::Caching was emitted, so mid-caching \
                 cancellation could not be exercised (a trainer must report caching progress)"
            ));
        }
        classify_cancel(id, "cancel-during-caching", result, saved)?;
    }
    Ok(())
}

/// Turn a cancelled `train` result into a pass/fail verdict: it must be the typed `Canceled` and
/// must not have emitted `Saving`.
fn classify_cancel(
    id: &str,
    path: &str,
    result: gen_core::Result<gen_core::TrainingOutput>,
    saved: bool,
) -> Result<(), String> {
    match result {
        Ok(out) => Err(format!(
            "cancellation[{id}/{path}]: train() returned Ok ({} steps) despite cancellation before \
             any step; it must return Err(Error::Canceled) and write no adapter (F-040)",
            out.steps
        )),
        Err(Error::Canceled) if saved => Err(format!(
            "cancellation[{id}/{path}]: returned Canceled but emitted TrainingProgress::Saving — a \
             cancelled-before-any-step run must not write an adapter (F-040)"
        )),
        Err(Error::Canceled) => Ok(()),
        Err(other) => Err(format!(
            "cancellation[{id}/{path}]: must return the typed Err(Error::Canceled) on cancel, got \
             {other:?} — a stringified Error::Msg breaks the typed-cancellation contract (sc-4895)"
        )),
    }
}

/// **Registry round-trip.** The trainer's descriptor `id` is discoverable through
/// `gen_core::registry::trainers()` — its `inventory::submit!` registration survived linking.
pub fn check_trainer_registry(t: &dyn Trainer) -> Result<(), String> {
    let id = t.descriptor().id;
    if gen_core::registry::trainers().any(|r| (r.descriptor)().id == id) {
        Ok(())
    } else {
        Err(format!(
            "registry[{id}]: descriptor id not found via gen_core::registry::trainers() — the \
             provider crate is not linked/registered (missing inventory::submit! or dead-stripped; \
             gen-core {})",
            gen_core::VERSION
        ))
    }
}

/// Run the full trainer conformance suite. `make` constructs a fresh trainer (it is invoked several
/// times — once for the validate/registry pair, once for the progress run, and once per cancellation
/// path — because `train` is `&mut self` and several families are single-use). Panics with every
/// failure aggregated.
pub fn trainer_conformance(make: impl Fn() -> Box<dyn Trainer>, profile: &TrainerProfile) {
    let mut failures: Vec<String> = Vec::new();

    // validate + registry share one (unconsumed) instance.
    {
        let t = make();
        if let Err(e) = check_trainer_validate(t.as_ref(), profile) {
            failures.push(e);
        }
        if let Err(e) = check_trainer_registry(t.as_ref()) {
            failures.push(e);
        }
    }

    // progress trains a fresh instance to completion.
    {
        let mut t = make();
        if let Err(e) = check_trainer_progress(t.as_mut(), profile) {
            failures.push(e);
        }
    }

    if let Err(e) = check_trainer_cancellation(&make, profile) {
        failures.push(e);
    }

    if !failures.is_empty() {
        let id = make().descriptor().id;
        panic!(
            "gen-core trainer conformance FAILED for `{id}` (gen-core {}):\n  - {}",
            gen_core::VERSION,
            failures.join("\n  - ")
        );
    }
}

#[cfg(test)]
mod tests;
