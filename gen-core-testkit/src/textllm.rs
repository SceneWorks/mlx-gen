//! Contract conformance for [`gen_core::TextLlm`] providers — the text-in/text-out analog of the
//! [`Generator`](crate::conformance) and [`Captioner`](crate::captioner) suites (epic 3720, sc-5500).
//! It exercises typed cancellation, `Progress` monotonicity, capability honesty, and registry
//! discoverability.
//!
//! ## TextLlm cancellation semantics
//!
//! Like a captioner, a text-LLM that has already emitted tokens when cancel trips may return a
//! **partial** `Ok` (a truncated reply marked [`TextLlmFinishReason::Cancelled`]). The **typed
//! `Err(Error::Canceled)`** contract covers cancellation *before inference starts*: a provider handed
//! an already-cancelled request must check the flag up front and return `Canceled` rather than
//! running the decoder to produce text nobody asked for. The check drives that pre-cancelled path.

use gen_core::{Error, Progress, TextLlm, TextLlmRequest, TextLlmSampling};

/// Parameters for a text-LLM conformance run — a single in-capability prompt the positive checks
/// generate from, and from which the negative checks derive out-of-bounds variants.
#[derive(Clone, Debug)]
pub struct TextLlmProfile {
    /// An optional system message (empty = none).
    pub system: String,
    /// A non-empty, in-range prompt.
    pub prompt: String,
    pub sampling: TextLlmSampling,
}

impl TextLlmProfile {
    /// A cheap profile: a short prompt, greedy decoding (`temperature 0.0` → deterministic, seed-free)
    /// and a low `max_new_tokens` so the real-lane generation is fast.
    pub fn cheap() -> Self {
        Self {
            system: String::new(),
            prompt: "Rewrite this prompt for an image model: a cat.".to_owned(),
            sampling: TextLlmSampling {
                temperature: 0.0,
                max_new_tokens: 16,
                ..Default::default()
            },
        }
    }
}

/// Build the in-capability request the positive checks generate, with a fresh cancel flag.
fn base_request(profile: &TextLlmProfile) -> TextLlmRequest {
    TextLlmRequest {
        system: profile.system.clone(),
        prompt: profile.prompt.clone(),
        sampling: profile.sampling,
        cancel: Default::default(),
    }
}

/// **Validate honesty.** The declared in-capability request is accepted; requests that exceed the
/// advertised surface (over-long `max_new_tokens`, an unsupported system prompt) are rejected by
/// `validate()`.
pub fn check_textllm_validate(c: &dyn TextLlm, profile: &TextLlmProfile) -> Result<(), String> {
    let desc = c.descriptor();
    let caps = &desc.capabilities;
    let id = desc.id;

    // Positive: the declared cheap request must be accepted.
    c.validate(&base_request(profile)).map_err(|e| {
        format!("validate-honesty[{id}]: the in-capability cheap request was rejected by validate(): {e}")
    })?;

    // Negative: max_new_tokens above the advertised cap must be rejected.
    if let Some(over) = caps.max_new_tokens.checked_add(1) {
        let mut r = base_request(profile);
        r.sampling.max_new_tokens = over;
        if c.validate(&r).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: max_new_tokens {over} (above the advertised cap {}) was \
                 accepted by validate()",
                caps.max_new_tokens
            ));
        }
    }

    // Negative: a system prompt when not supported must be rejected.
    if !caps.supports_system_prompt {
        let mut r = base_request(profile);
        r.system = "an unsupported system prompt".to_owned();
        if c.validate(&r).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: a system prompt was accepted by validate() despite \
                 supports_system_prompt == false"
            ));
        }
    }
    Ok(())
}

/// **Progress.** A completed generation emits at least one `Progress::Step`, with a constant `total`
/// and a strictly-increasing `current` in `1..=total` — enough to drive a progress bar / observe
/// cooperative cancellation (the decoder's progress is token based, not a fixed step count, so this
/// is intentionally laxer than the generator's exact-`1..=total` check).
pub fn check_textllm_progress(c: &dyn TextLlm, profile: &TextLlmProfile) -> Result<(), String> {
    let id = c.descriptor().id;
    let mut steps: Vec<(u32, u32)> = Vec::new();
    c.generate(&base_request(profile), &mut |p| {
        if let Progress::Step { current, total } = p {
            steps.push((current, total));
        }
    })
    .map_err(|e| format!("progress[{id}]: generate() failed on the cheap request: {e}"))?;

    if steps.is_empty() {
        return Err(format!(
            "progress[{id}]: generate() emitted no Progress::Step events"
        ));
    }
    let total = steps[0].1;
    if total == 0 {
        return Err(format!("progress[{id}]: Progress::Step.total was 0"));
    }
    let mut prev = 0u32;
    for &(current, t) in &steps {
        if t != total {
            return Err(format!(
                "progress[{id}]: Step.total changed mid-run ({total} then {t})"
            ));
        }
        if current < 1 || current > total {
            return Err(format!(
                "progress[{id}]: Step.current {current} out of range 1..={total}"
            ));
        }
        if current <= prev {
            return Err(format!(
                "progress[{id}]: Step.current must strictly increase; saw {prev} then {current}"
            ));
        }
        prev = current;
    }
    Ok(())
}

/// **Cancellation.** A provider handed an already-cancelled request must return the **typed**
/// `Err(Error::Canceled)` (not a stringified `Msg`, and not an `Ok` reply) — it must check the flag
/// before running inference.
pub fn check_textllm_cancellation(c: &dyn TextLlm, profile: &TextLlmProfile) -> Result<(), String> {
    let id = c.descriptor().id;
    let req = base_request(profile);
    req.cancel.cancel();
    match c.generate(&req, &mut |_| {}) {
        Ok(out) => Err(format!(
            "cancellation[{id}]: generate() returned Ok ({:?}) despite an already-cancelled request; \
             it must return Err(Error::Canceled) before running inference",
            out.text
        )),
        Err(Error::Canceled) => Ok(()),
        Err(other) => Err(format!(
            "cancellation[{id}]: must return the typed Err(Error::Canceled) on cancel, got {other:?} \
             — a stringified Error::Msg breaks the typed-cancellation contract (sc-5500)"
        )),
    }
}

/// **Registry round-trip.** The provider's descriptor `id` is discoverable through
/// `gen_core::registry::textllms()` — its `inventory::submit!` registration survived linking.
pub fn check_textllm_registry(c: &dyn TextLlm) -> Result<(), String> {
    let id = c.descriptor().id;
    if gen_core::registry::textllms().any(|r| (r.descriptor)().id == id) {
        Ok(())
    } else {
        Err(format!(
            "registry[{id}]: descriptor id not found via gen_core::registry::textllms() — the \
             provider crate is not linked/registered (missing inventory::submit! or dead-stripped; \
             gen-core {})",
            gen_core::VERSION
        ))
    }
}

/// Run the full text-LLM conformance suite against a freshly-`make`d provider. `generate` is `&self`
/// and stateless across calls, so the whole suite is one load. Panics with every failure aggregated.
pub fn textllm_conformance(make: impl Fn() -> Box<dyn TextLlm>, profile: &TextLlmProfile) {
    let c = make();
    let c: &dyn TextLlm = c.as_ref();

    type Check = fn(&dyn TextLlm, &TextLlmProfile) -> Result<(), String>;
    let checks: [Check; 3] = [
        check_textllm_validate,
        check_textllm_progress,
        check_textllm_cancellation,
    ];
    let mut failures: Vec<String> = checks
        .into_iter()
        .filter_map(|f| f(c, profile).err())
        .collect();
    if let Err(e) = check_textllm_registry(c) {
        failures.push(e);
    }

    if !failures.is_empty() {
        panic!(
            "gen-core textllm conformance FAILED for `{}` (gen-core {}):\n  - {}",
            c.descriptor().id,
            gen_core::VERSION,
            failures.join("\n  - ")
        );
    }
}

#[cfg(test)]
mod tests;
