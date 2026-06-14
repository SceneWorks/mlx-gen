//! # gen-core-testkit
//!
//! A **contract conformance suite** for gen-core providers — [`gen_core::Generator`] (this module),
//! [`gen_core::Trainer`](crate::trainer), and [`gen_core::Captioner`](crate::captioner). Given any
//! boxed provider — an MLX family from `mlx-gen` or a future candle-gen provider — it exercises the
//! behavioral guarantees the contract *promises but cannot express in the type system*: typed
//! cancellation, progress monotonicity, seed determinism, and capability honesty. Both backends run
//! it in CI, so a provider that silently ignores `CancelFlag` or reports no progress (the sc-4380
//! class of bug) becomes a CI failure instead of a field report (epic 3720, sc-4481/sc-4895).
//!
//! The testkit has **zero tensor dependencies** — it depends only on `gen-core` and drives the
//! provider purely through the public contract, so it builds and runs on the Linux gen-core lane
//! against an in-crate stub exactly as it does on the macOS lane against a real MLX family.
//!
//! ## Usage
//!
//! ```ignore
//! // macOS lane, real family — generator, trainer, captioner:
//! gen_core_testkit::conformance(
//!     || mlx_gen::load("z_image_turbo", &spec).unwrap(),
//!     &gen_core_testkit::Profile::cheap(),
//! );
//! gen_core_testkit::trainer_conformance(
//!     || mlx_gen::load_trainer("z_image_turbo", &spec).unwrap(),
//!     &gen_core_testkit::TrainerProfile::cheap(items, out_dir),
//! );
//! gen_core_testkit::captioner_conformance(
//!     || mlx_gen::load_captioner("joy_caption", &spec).unwrap(),
//!     &gen_core_testkit::CaptionerProfile::cheap(),
//! );
//! ```
//!
//! The individual `check_*` functions are public so a provider's own tests can target one guarantee
//! at a time; the `*_conformance` entry points run them all and panic with the aggregated failures.

pub mod captioner;
pub mod textllm;
pub mod trainer;

pub use captioner::{
    captioner_conformance, check_captioner_cancellation, check_captioner_progress,
    check_captioner_registry, check_captioner_validate, CaptionerProfile,
};
pub use textllm::{
    check_textllm_cancellation, check_textllm_progress, check_textllm_registry,
    check_textllm_validate, textllm_conformance, TextLlmProfile,
};
pub use trainer::{
    check_trainer_cancellation, check_trainer_progress, check_trainer_registry,
    check_trainer_validate, trainer_conformance, TrainerProfile,
};

use gen_core::{
    Capabilities, Conditioning, Error, GenerationOutput, GenerationRequest, Generator, Image,
    Progress,
};

/// Cheap-request parameters for the conformance run. Keep these at the model's *minimum* valid
/// size and a tiny step count — the suite runs `generate` several times, so the macOS-lane cost is
/// `~4 ×` one cheap render.
#[derive(Clone, Debug)]
pub struct Profile {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    /// Denoise steps the request asks for **and** the value the model is expected to resolve to:
    /// [`check_progress`] asserts `Progress::Step.total == steps`. If a model clamps/transforms
    /// `req.steps`, set this to the resolved count, not the requested one.
    pub steps: u32,
    pub seed: u64,
    /// Steps requested for [`check_cancellation`] only — needs headroom (≥ 3) so that a provider
    /// honoring cancellation visibly stops before completion. Generation is cancelled at the first
    /// step boundary, so only ~1 forward actually runs regardless of this value.
    pub cancel_steps: u32,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            prompt: "a fox".to_owned(),
            width: 256,
            height: 256,
            steps: 2,
            seed: 42,
            cancel_steps: 6,
        }
    }
}

impl Profile {
    /// The cheapest generally-valid profile: 256², 2 steps, fixed seed. 256 is a multiple of the
    /// common VAE×patch alignment (16) and ≥ every current family's `min_size`.
    pub fn cheap() -> Self {
        Self::default()
    }
}

/// The in-capability request the positive checks expect the model to accept (and the
/// progress/seed checks render from). Only the fields the profile pins are set; everything else is
/// the contract default (notably `count: 1`).
fn base_request(profile: &Profile) -> GenerationRequest {
    GenerationRequest {
        prompt: profile.prompt.clone(),
        width: profile.width,
        height: profile.height,
        steps: Some(profile.steps),
        seed: Some(profile.seed),
        ..Default::default()
    }
}

/// The raw output pixels, flattened across images/frames — the unit the seed-determinism check
/// compares byte-for-byte.
fn output_bytes(out: &GenerationOutput) -> Vec<u8> {
    match out {
        GenerationOutput::Images(imgs) => {
            imgs.iter().flat_map(|i| i.pixels.iter().copied()).collect()
        }
        GenerationOutput::Video { frames, .. } => frames
            .iter()
            .flat_map(|f| f.pixels.iter().copied())
            .collect(),
    }
}

/// A `width × height` all-zero RGB image, for building conditioning the model should reject.
fn blank_image(profile: &Profile) -> Image {
    Image {
        width: profile.width,
        height: profile.height,
        pixels: vec![0u8; profile.width as usize * profile.height as usize * 3],
    }
}

/// The first easily-constructed [`Conditioning`] whose kind the model does **not** advertise, or
/// `None` if it accepts all of the candidates (then the negative-conditioning sub-check is skipped).
fn undeclared_conditioning(caps: &Capabilities, profile: &Profile) -> Option<Conditioning> {
    [
        Conditioning::Mask {
            image: blank_image(profile),
        },
        Conditioning::Depth {
            image: blank_image(profile),
        },
        Conditioning::Reference {
            image: blank_image(profile),
            strength: None,
        },
    ]
    .into_iter()
    .find(|c| !caps.accepts(c.kind()))
}

/// **Validate honesty.** Everything the descriptor advertises is accepted by `validate()`, and
/// requests that exceed the advertised surface (oversize, overcount, undeclared conditioning) are
/// rejected by `validate()` — *before* any expensive work, not by `generate()` panicking later.
pub fn check_validate_honesty(g: &dyn Generator, profile: &Profile) -> Result<(), String> {
    let desc = g.descriptor();
    let caps = &desc.capabilities;
    let id = desc.id;

    // Positive: the declared cheap request must be accepted.
    g.validate(&base_request(profile)).map_err(|e| {
        format!("validate-honesty[{id}]: the in-capability cheap request ({}x{}, {} steps) was rejected by validate(): {e}", profile.width, profile.height, profile.steps)
    })?;

    // Positive: every advertised sampler must be accepted.
    for &s in &caps.samplers {
        let mut r = base_request(profile);
        r.sampler = Some(s.to_owned());
        if let Err(e) = g.validate(&r) {
            return Err(format!(
                "validate-honesty[{id}]: advertised sampler {s:?} was rejected by validate(): {e}"
            ));
        }
    }

    // Negative: a size above max_size must be rejected.
    if let Some(big) = caps.max_size.checked_add(64) {
        let mut r = base_request(profile);
        r.width = big;
        r.height = big;
        if g.validate(&r).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: a {big}x{big} request (above max_size {}) was accepted by validate()",
                caps.max_size
            ));
        }
    }

    // Negative: a count above max_count must be rejected.
    if let Some(many) = caps.max_count.checked_add(1) {
        let mut r = base_request(profile);
        r.count = many;
        if g.validate(&r).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: count {many} (above max_count {}) was accepted by validate()",
                caps.max_count
            ));
        }
    }

    // Negative: an undeclared conditioning kind must be rejected.
    if let Some(cond) = undeclared_conditioning(caps, profile) {
        let kind = cond.kind();
        let mut r = base_request(profile);
        r.conditioning = vec![cond];
        if g.validate(&r).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: undeclared {kind:?} conditioning was accepted by validate() \
                 (descriptor advertises {:?})",
                caps.conditioning
            ));
        }
    }

    Ok(())
}

/// **Progress.** `Progress::Step{current,total}` is monotone and complete: `current` runs exactly
/// `1..=total`, `total` is constant, and `total` equals the profile's resolved step count.
pub fn check_progress(g: &dyn Generator, profile: &Profile) -> Result<(), String> {
    let id = g.descriptor().id;
    let req = base_request(profile);
    let mut steps: Vec<(u32, u32)> = Vec::new();
    g.generate(&req, &mut |p| {
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
    if let Some((c, t)) = steps.iter().find(|(_, t)| *t != total) {
        return Err(format!(
            "progress[{id}]: Step.total changed mid-run ({total} then {t} at current={c})"
        ));
    }
    let observed: Vec<u32> = steps.iter().map(|(c, _)| *c).collect();
    let expected: Vec<u32> = (1..=total).collect();
    if observed != expected {
        return Err(format!(
            "progress[{id}]: Step.current must be exactly 1..={total} (monotone, complete, no repeats); got {observed:?}"
        ));
    }
    if total != profile.steps {
        return Err(format!(
            "progress[{id}]: Step.total ({total}) != the profile's resolved step count ({}). \
             Set Profile.steps to the value the model resolves req.steps to.",
            profile.steps
        ));
    }
    Ok(())
}

/// **Cancellation.** Tripping `CancelFlag` at the first step boundary makes `generate` return the
/// **typed** `Err(Error::Canceled)` (not a stringified `Msg`) within a bounded number of further
/// steps (≤ 2), and produces no partial output.
pub fn check_cancellation(g: &dyn Generator, profile: &Profile) -> Result<(), String> {
    let id = g.descriptor().id;
    let mut req = base_request(profile);
    req.steps = Some(profile.cancel_steps);
    let cancel = req.cancel.clone();

    let mut tripped = false;
    let mut steps_after_trip = 0u32;
    let mut last_current = 0u32;
    let result = g.generate(&req, &mut |p| {
        if let Progress::Step { current, .. } = p {
            last_current = current;
            if tripped {
                steps_after_trip += 1;
            } else {
                cancel.cancel();
                tripped = true;
            }
        }
    });

    if !tripped {
        return Err(format!(
            "cancellation[{id}]: no Progress::Step was emitted, so cancellation could not be exercised \
             (a provider must report step progress for cooperative cancellation to be observable)"
        ));
    }
    match result {
        Ok(_) => Err(format!(
            "cancellation[{id}]: generate() ran to completion despite CancelFlag set at step 1 \
             (reached step {last_current}); it must return Err(Error::Canceled)"
        )),
        Err(Error::Canceled) if steps_after_trip > 2 => Err(format!(
            "cancellation[{id}]: returned Canceled but emitted {steps_after_trip} further Progress::Step events \
             after the cancel trip (contract allows at most 2)"
        )),
        Err(Error::Canceled) => Ok(()),
        Err(other) => Err(format!(
            "cancellation[{id}]: must return the typed Err(Error::Canceled) on cancel, got {other:?} \
             — a stringified Error::Msg breaks the typed-cancellation contract (epic 3720 D3)"
        )),
    }
}

/// **Seed determinism (same backend).** Two runs of the identical request+seed produce
/// byte-identical output. Cross-backend equality is *not* a goal (RNG algorithms differ); this is
/// the guarantee that makes the seeded per-step RNG (D6) mandatory.
pub fn check_seed_determinism(g: &dyn Generator, profile: &Profile) -> Result<(), String> {
    let id = g.descriptor().id;
    let req = base_request(profile);
    let a = g
        .generate(&req, &mut |_| {})
        .map_err(|e| format!("seed[{id}]: first generate() failed: {e}"))?;
    let b = g
        .generate(&req, &mut |_| {})
        .map_err(|e| format!("seed[{id}]: second generate() failed: {e}"))?;
    let (ba, bb) = (output_bytes(&a), output_bytes(&b));
    if ba.len() != bb.len() {
        return Err(format!(
            "seed[{id}]: same seed produced different output sizes ({} vs {} bytes)",
            ba.len(),
            bb.len()
        ));
    }
    if let Some(i) = ba.iter().zip(&bb).position(|(x, y)| x != y) {
        return Err(format!(
            "seed[{id}]: same request+seed produced different pixels (first diff at byte {i}: {} vs {}, of {} bytes)",
            ba[i], bb[i], ba.len()
        ));
    }
    // A provider that *ignores* the seed would also pass the identical-twice check above, so verify a
    // DIFFERENT seed actually changes the output (F-085).
    let mut req_alt = base_request(profile);
    req_alt.seed = Some(profile.seed.wrapping_add(0x9E37_79B9));
    let c = g
        .generate(&req_alt, &mut |_| {})
        .map_err(|e| format!("seed[{id}]: alternate-seed generate() failed: {e}"))?;
    let bc = output_bytes(&c);
    if bc.len() == ba.len() && bc.iter().zip(&ba).all(|(x, y)| x == y) {
        return Err(format!(
            "seed[{id}]: a different seed produced byte-identical output ({} bytes) — the provider \
             appears to ignore the seed",
            ba.len()
        ));
    }
    Ok(())
}

/// **Registry round-trip.** The provider's descriptor `id` is discoverable through
/// `gen_core::registry` — i.e. its `inventory::submit!` registration is present in the build graph
/// (a missing/dead-stripped registration is the runtime "engine not found" trap, sc-4482).
pub fn check_registry_roundtrip(g: &dyn Generator) -> Result<(), String> {
    let id = g.descriptor().id;
    if gen_core::registry::generators().any(|r| (r.descriptor)().id == id) {
        Ok(())
    } else {
        Err(format!(
            "registry[{id}]: descriptor id not found via gen_core::registry::generators() — the provider \
             crate is not linked/registered (missing inventory::submit! or dead-stripped; gen-core {})",
            gen_core::VERSION
        ))
    }
}

/// Run the full conformance suite against a freshly-`make`d generator. Panics with every failure
/// aggregated (one bullet per failed guarantee) — the test-helper idiom, like a fat `assert`.
///
/// `make` is `Fn` so callers may hand it a registry loader (`|| mlx_gen::load(id, &spec).unwrap()`)
/// or an in-crate stub; it is invoked once. The generator is shared across checks (`generate` is
/// `&self` and stateless across calls), so the whole suite is one model load.
pub fn conformance(make: impl Fn() -> Box<dyn Generator>, profile: &Profile) {
    let g = make();
    let g: &dyn Generator = g.as_ref();

    type Check = fn(&dyn Generator, &Profile) -> Result<(), String>;
    let checks: [Check; 4] = [
        check_validate_honesty,
        check_progress,
        check_cancellation,
        check_seed_determinism,
    ];

    let mut failures: Vec<String> = checks
        .into_iter()
        .filter_map(|f| f(g, profile).err())
        .collect();
    if let Err(e) = check_registry_roundtrip(g) {
        failures.push(e);
    }

    if !failures.is_empty() {
        panic!(
            "gen-core conformance FAILED for `{}` ({} backend, gen-core {}):\n  - {}",
            g.descriptor().id,
            g.descriptor().backend,
            gen_core::VERSION,
            failures.join("\n  - ")
        );
    }
}

#[cfg(test)]
mod tests;
