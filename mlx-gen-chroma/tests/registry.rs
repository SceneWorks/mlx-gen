//! sc-3835: the three Chroma variants self-register through the core registry with the expected
//! family + capability surface, and the loader rejects a single-file spec (it wants a diffusers
//! snapshot directory).

use mlx_gen::{GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_chroma as _;

#[test]
fn chroma_variants_resolve_through_core_registry() {
    for id in ["chroma1_hd", "chroma1_base", "chroma1_flash"] {
        let reg = mlx_gen::registry::generators()
            .find(|r| (r.descriptor)().id == id)
            .unwrap_or_else(|| panic!("{id} provider should self-register"));
        let d = (reg.descriptor)();
        assert_eq!(d.family, "chroma");
        // Chroma uses true CFG with a real negative prompt; no distilled guidance scalar.
        assert!(d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_guidance);
        assert!(d.capabilities.samplers.contains(&"euler"));
        assert!(d.capabilities.samplers.contains(&"heun"));
        // v1 is T2I only.
        assert!(d.capabilities.conditioning.is_empty());

        let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
        let err = mlx_gen::load(id, &spec)
            .err()
            .expect("single-file spec is rejected by the loader")
            .to_string();
        assert!(
            err.contains("snapshot directory"),
            "expected the chroma loader's directory error, got: {err}"
        );
    }
}

#[test]
fn chroma_validate_rejects_unsupported_surface() {
    let reg = mlx_gen::registry::generators()
        .find(|r| (r.descriptor)().id == "chroma1_hd")
        .expect("chroma1_hd registered");
    let d = (reg.descriptor)();
    // guidance is not advertised (CFG is true_cfg) — the shared capability check must reject it.
    let req = GenerationRequest {
        prompt: "a cat".into(),
        width: 512,
        height: 512,
        guidance: Some(3.5),
        ..Default::default()
    };
    assert!(d.capabilities.validate_request(d.id, &req).is_err());

    let heun = GenerationRequest {
        prompt: "a cat".into(),
        width: 512,
        height: 512,
        sampler: Some("heun".into()),
        ..Default::default()
    };
    assert!(d.capabilities.validate_request(d.id, &heun).is_ok());

    let bad_sampler = GenerationRequest {
        prompt: "a cat".into(),
        width: 512,
        height: 512,
        sampler: Some("dpmpp_sde".into()),
        ..Default::default()
    };
    assert!(d.capabilities.validate_request(d.id, &bad_sampler).is_err());
}
