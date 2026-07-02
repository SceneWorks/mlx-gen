//! Weights-free, default-run descriptor-level gen-core conformance (sc-9098, F-009): every
//! registration this test binary links (this provider + any reused sibling providers) satisfies
//! the descriptor/capability invariants checkable without loading weights — id/family/backend
//! shape, coherent size/count bounds, duplicate-free curated names and conditioning kinds,
//! modality-consistent conditioning, and per-kind registry id uniqueness. Behavioral conformance
//! (progress/cancel/seed) stays weights-gated in the crate's `#[ignore]`d suites.

// Force-link the provider so its `inventory::submit!` registrations survive the linker (this test
// references no other crate symbol); the worker does the same `as _` import per model crate.
use mlx_gen_lens as _;

#[test]
fn registered_descriptors_conform() {
    // The `as _` link above must have registered at least one generator — guards the sc-4482
    // dead-strip trap (an empty registry would pass the sweep vacuously).
    assert!(
        mlx_gen::registry::generators().next().is_some(),
        "no generator registered — the provider registration was dead-stripped"
    );
    let errs = mlx_gen::registry::descriptor_conformance_errors();
    assert!(
        errs.is_empty(),
        "descriptor conformance FAILED ({} violations):\n  - {}",
        errs.len(),
        errs.join("\n  - ")
    );
}
