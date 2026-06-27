use mlx_gen::{ConditioningKind, LoadSpec, Quant, WeightsSource};
use mlx_gen_flux as _;

#[test]
fn flux1_variants_resolve_through_core_registry() {
    for id in ["flux1_schnell", "flux1_dev"] {
        let reg = mlx_gen::registry::generators()
            .find(|r| (r.descriptor)().id == id)
            .unwrap_or_else(|| panic!("{id} provider should self-register"));
        let d = (reg.descriptor)();
        assert_eq!(d.family, "flux");

        let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
        let err = mlx_gen::load(id, &spec)
            .err()
            .expect("single-file spec is rejected by the loader")
            .to_string();
        assert!(
            err.contains("snapshot directory"),
            "expected the flux loader's error, got: {err}"
        );
    }
}

#[test]
fn flux1_dev_control_resolves_through_core_registry() {
    // E2 (sc-8239): the finalized `flux1_dev_control` is a first-class registered model. It must
    // self-register and resolve through `mlx_gen::load` like the base FLUX.1 variants. With a single
    // base file (and no control checkpoint) the loader rejects it on the missing snapshot dir, proving
    // we reached the control loader rather than a missing-registration error.
    let reg = mlx_gen::registry::generators()
        .find(|r| (r.descriptor)().id == "flux1_dev_control")
        .expect("flux1_dev_control provider should self-register");
    let d = (reg.descriptor)();
    assert_eq!(d.family, "flux");
    assert!(d.capabilities.accepts(ConditioningKind::Control));
    // Mirrors flux2_dev_control: guidance-distilled dev base, no negative/true-CFG/KV-cache, mac-only.
    assert!(d.capabilities.supports_guidance);
    assert!(!d.capabilities.supports_true_cfg);
    assert!(!d.capabilities.supports_kv_cache);
    assert!(d.capabilities.mac_only);

    // A single-file base spec must reach the control loader and be rejected for not being a snapshot
    // dir (it errors on the base shape before the missing-control check).
    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()))
        .with_control(WeightsSource::File("/unused-control.safetensors".into()));
    let err = mlx_gen::load("flux1_dev_control", &spec)
        .err()
        .expect("single-file base spec is rejected by the control loader")
        .to_string();
    assert!(
        err.contains("snapshot directory"),
        "expected the control loader's base-shape error, got: {err}"
    );

    // And without the control checkpoint, the loader fails on the missing control overlay (a hard
    // requirement) rather than silently running control-free.
    let spec_no_control = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
    let err = mlx_gen::load("flux1_dev_control", &spec_no_control)
        .err()
        .expect("missing control checkpoint is rejected")
        .to_string();
    assert!(
        err.contains("FLUX.1-dev-ControlNet-Union-Pro-2.0"),
        "expected the missing-control error, got: {err}"
    );
}

#[test]
fn flux1_variants_accept_quantization_specs() {
    for id in ["flux1_schnell", "flux1_dev"] {
        for quant in [Quant::Q4, Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(quant);
            let err = mlx_gen::load(id, &spec)
                .err()
                .expect("missing snapshot should still error")
                .to_string();
            assert!(
                !err.contains("quantized") && !err.contains("quantization"),
                "quantized FLUX load specs should get past capability gating, got: {err}"
            );
        }
    }
}
