//! sc-2346 S0: the FLUX.2-klein variants self-register and are introspectable through the core
//! registry without loading weights; loading is guarded until the model modules land (S1–S3).

use mlx_gen::{ConditioningKind, LoadSpec, WeightsSource};
use mlx_gen_flux2 as _;

#[test]
fn flux2_variants_resolve_through_core_registry() {
    for id in ["flux2_klein_9b", "flux2_klein_9b_edit"] {
        let reg = mlx_gen::registry::generators()
            .find(|r| (r.descriptor)().id == id)
            .unwrap_or_else(|| panic!("{id} provider should self-register"));
        let d = (reg.descriptor)();
        assert_eq!(d.family, "flux2");
        assert!(d.capabilities.requires_sigma_shift);
        assert!(d.capabilities.schedulers.contains(&"flow_match_euler"));
    }
}

#[test]
fn edit_advertises_single_reference_txt2img_does_not() {
    let edit = mlx_gen::registry::generators()
        .find(|r| (r.descriptor)().id == "flux2_klein_9b_edit")
        .map(|r| (r.descriptor)())
        .unwrap();
    assert!(edit.capabilities.accepts(ConditioningKind::Reference));
    // Multi-reference edit is sc-2645, not this story.
    assert!(!edit.capabilities.accepts(ConditioningKind::MultiReference));

    let t2i = mlx_gen::registry::generators()
        .find(|r| (r.descriptor)().id == "flux2_klein_9b")
        .map(|r| (r.descriptor)())
        .unwrap();
    // img2img (Reference) is sc-2644, not this story's txt2img variant.
    assert!(!t2i.capabilities.accepts(ConditioningKind::Reference));
}

#[test]
fn load_resolves_then_fails_on_missing_snapshot() {
    for id in ["flux2_klein_9b", "flux2_klein_9b_edit"] {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let err = mlx_gen::load(id, &spec)
            .err()
            .expect("a missing snapshot dir must error")
            .to_string();
        // The id resolves through the registry and reaches the loader (which then fails to read the
        // snapshot) — i.e. NOT a "no generator registered" miss.
        assert!(
            !err.contains("no generator registered"),
            "id should resolve through the registry, got: {err}"
        );
    }
}

#[test]
fn single_file_spec_is_rejected() {
    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
    let err = mlx_gen::load("flux2_klein_9b", &spec)
        .err()
        .expect("a single-file spec is rejected")
        .to_string();
    assert!(err.contains("snapshot directory"), "got: {err}");
}
