//! Contract conformance for [`gen_core::Captioner`] providers — the image-to-text analog of the
//! [`Generator`](crate::conformance) suite (epic 3720, sc-4895). It exercises typed cancellation,
//! `Progress` monotonicity, capability honesty, and registry discoverability.
//!
//! ## Captioner cancellation semantics
//!
//! Like a trainer, a captioner that has already emitted tokens when cancel trips may return a
//! **partial** `Ok` (a truncated caption marked [`CaptionFinishReason::Cancelled`]). The **typed
//! `Err(Error::Canceled)`** contract covers cancellation *before inference starts*: a captioner
//! handed an already-cancelled request must check the flag up front and return `Canceled` rather
//! than running the vision/text stack to produce a caption nobody asked for. The check drives that
//! pre-cancelled path.

use gen_core::{
    CaptionOptions, CaptionRequest, CaptionSampling, Captioner, Error, Image, Progress,
};

/// Parameters for a captioner conformance run — a single in-capability image+prompt the positive
/// checks caption, and from which the negative checks derive out-of-bounds variants.
#[derive(Clone, Debug)]
pub struct CaptionerProfile {
    /// An in-range RGB image (size within the advertised `min_image_size..=max_image_size`).
    pub image: Image,
    /// A non-empty, in-range prompt.
    pub prompt: String,
    pub options: CaptionOptions,
    pub sampling: CaptionSampling,
}

impl CaptionerProfile {
    /// A cheap profile: a tiny solid RGB image and a short prompt, with the default options and a
    /// low `max_new_tokens` so the real-lane caption is fast.
    pub fn cheap() -> Self {
        let (w, h) = (64u32, 64u32);
        Self {
            image: Image {
                width: w,
                height: h,
                pixels: vec![0u8; (w * h * 3) as usize],
            },
            prompt: "Write a short description for this image.".to_owned(),
            options: CaptionOptions::default(),
            sampling: CaptionSampling {
                temperature: 0.0, // greedy → deterministic, seed-free
                max_new_tokens: 16,
                ..Default::default()
            },
        }
    }
}

/// Build the in-capability request the positive checks caption, with a fresh cancel flag.
fn base_request(profile: &CaptionerProfile) -> CaptionRequest {
    CaptionRequest {
        image: profile.image.clone(),
        prompt: profile.prompt.clone(),
        options: profile.options.clone(),
        sampling: profile.sampling,
        trigger_words: Vec::new(),
        cancel: Default::default(),
    }
}

/// **Validate honesty.** The declared in-capability request is accepted; requests that exceed the
/// advertised surface (over-long `max_new_tokens`, an unsupported custom prompt / low-vram flag) are
/// rejected by `validate()`.
pub fn check_captioner_validate(
    c: &dyn Captioner,
    profile: &CaptionerProfile,
) -> Result<(), String> {
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

    // Negative: a custom prompt when not supported must be rejected.
    if !caps.supports_custom_prompt {
        let mut r = base_request(profile);
        r.options.custom_prompt = "an unsupported custom prompt".to_owned();
        if c.validate(&r).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: a custom prompt was accepted by validate() despite \
                 supports_custom_prompt == false"
            ));
        }
    }

    // Negative: the low-vram flag when not supported must be rejected.
    if !caps.supports_low_vram {
        let mut r = base_request(profile);
        r.options.low_vram = true;
        if c.validate(&r).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: low_vram was accepted by validate() despite \
                 supports_low_vram == false"
            ));
        }
    }
    Ok(())
}

/// **Progress.** A completed caption emits at least one `Progress::Step`, with a constant `total`
/// and a strictly-increasing `current` in `1..=total` — enough to drive a progress bar / observe
/// cooperative cancellation (the captioner's progress is phase/token based, not a fixed step count,
/// so this is intentionally laxer than the generator's exact-`1..=total` check).
pub fn check_captioner_progress(
    c: &dyn Captioner,
    profile: &CaptionerProfile,
) -> Result<(), String> {
    let id = c.descriptor().id;
    let mut steps: Vec<(u32, u32)> = Vec::new();
    c.caption(&base_request(profile), &mut |p| {
        if let Progress::Step { current, total } = p {
            steps.push((current, total));
        }
    })
    .map_err(|e| format!("progress[{id}]: caption() failed on the cheap request: {e}"))?;

    if steps.is_empty() {
        return Err(format!(
            "progress[{id}]: caption() emitted no Progress::Step events"
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

/// **Cancellation.** A captioner handed an already-cancelled request must return the **typed**
/// `Err(Error::Canceled)` (not a stringified `Msg`, and not an `Ok` caption) — it must check the
/// flag before running inference.
pub fn check_captioner_cancellation(
    c: &dyn Captioner,
    profile: &CaptionerProfile,
) -> Result<(), String> {
    let id = c.descriptor().id;
    let req = base_request(profile);
    req.cancel.cancel();
    match c.caption(&req, &mut |_| {}) {
        Ok(out) => Err(format!(
            "cancellation[{id}]: caption() returned Ok ({:?}) despite an already-cancelled request; \
             it must return Err(Error::Canceled) before running inference",
            out.text
        )),
        Err(Error::Canceled) => Ok(()),
        Err(other) => Err(format!(
            "cancellation[{id}]: must return the typed Err(Error::Canceled) on cancel, got {other:?} \
             — a stringified Error::Msg breaks the typed-cancellation contract (sc-4895)"
        )),
    }
}

/// **Registry round-trip.** The captioner's descriptor `id` is discoverable through
/// `gen_core::registry::captioners()` — its `inventory::submit!` registration survived linking.
pub fn check_captioner_registry(c: &dyn Captioner) -> Result<(), String> {
    let id = c.descriptor().id;
    if gen_core::registry::captioners().any(|r| (r.descriptor)().id == id) {
        Ok(())
    } else {
        Err(format!(
            "registry[{id}]: descriptor id not found via gen_core::registry::captioners() — the \
             provider crate is not linked/registered (missing inventory::submit! or dead-stripped; \
             gen-core {})",
            gen_core::VERSION
        ))
    }
}

/// Run the full captioner conformance suite against a freshly-`make`d captioner. `caption` is
/// `&self` and stateless across calls, so the whole suite is one load. Panics with every failure
/// aggregated.
pub fn captioner_conformance(make: impl Fn() -> Box<dyn Captioner>, profile: &CaptionerProfile) {
    let c = make();
    let c: &dyn Captioner = c.as_ref();

    type Check = fn(&dyn Captioner, &CaptionerProfile) -> Result<(), String>;
    let checks: [Check; 3] = [
        check_captioner_validate,
        check_captioner_progress,
        check_captioner_cancellation,
    ];
    let mut failures: Vec<String> = checks
        .into_iter()
        .filter_map(|f| f(c, profile).err())
        .collect();
    if let Err(e) = check_captioner_registry(c) {
        failures.push(e);
    }

    if !failures.is_empty() {
        panic!(
            "gen-core captioner conformance FAILED for `{}` (gen-core {}):\n  - {}",
            c.descriptor().id,
            gen_core::VERSION,
            failures.join("\n  - ")
        );
    }
}

#[cfg(test)]
mod tests;
