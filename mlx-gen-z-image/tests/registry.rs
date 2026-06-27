//! Proves the architecture's central claim (docs/MODEL_ARCHITECTURE.md §4): linking the
//! provider crate self-registers Z-Image into `mlx-gen`'s link-time `inventory` registry — the
//! core has no central match to edit — so `mlx_gen::load("z_image_turbo", …)` resolves across
//! the crate boundary. This is the Rust stand-in for a DI container's resolve-by-id.
//!
//! NOTE: a provider must actually be *linked* into the consumer for its `inventory::submit!` to
//! take effect — a dependency that is declared but never referenced can have its link-section
//! statics dropped by the linker. The `use … as _` below forces the link (the SceneWorks worker
//! references every provider it serves, so this is automatic there). This is the "DI container
//! must know about the assembly" detail.

use mlx_gen::{LoadSpec, WeightsSource};
use mlx_gen_z_image as _;

#[test]
fn z_image_turbo_resolves_through_core_registry() {
    // The descriptor resolves across the crate boundary without loading weights — proof the
    // provider's `inventory::submit!` fired and the core can find it by id.
    let reg = mlx_gen::registry::generators()
        .find(|r| (r.descriptor)().id == "z_image_turbo")
        .expect("provider self-registered via inventory");
    let d = (reg.descriptor)();
    assert_eq!(d.id, "z_image_turbo");
    assert_eq!(d.family, "z-image");

    // `mlx_gen::load(id, …)` routes to *this* provider's loader: a bogus spec surfaces the
    // provider's own snapshot-layout error, not the registry's "no generator registered".
    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
    let err = mlx_gen::load("z_image_turbo", &spec)
        .err()
        .expect("a single-file spec is rejected by the loader")
        .to_string();
    assert!(
        err.contains("snapshot directory"),
        "expected the z-image loader's error, got: {err}"
    );
}

#[test]
fn z_image_turbo_visible_in_registry_iteration() {
    assert!(mlx_gen::registry::generators().any(|r| (r.descriptor)().id == "z_image_turbo"));
}

#[test]
fn base_z_image_resolves_through_core_registry() {
    // sc-8320: the base (non-Turbo) model registers under its own id and resolves across the crate
    // boundary, alongside `z_image_turbo` — proof the second `inventory::submit!` fired without an
    // id clash.
    let reg = mlx_gen::registry::generators()
        .find(|r| (r.descriptor)().id == "z_image")
        .expect("base provider self-registered via inventory");
    let d = (reg.descriptor)();
    assert_eq!(d.id, "z_image");
    assert_eq!(d.family, "z-image");
    // The base is the full-CFG variant (Turbo is guidance-distilled).
    assert!(d.capabilities.supports_guidance);
    assert!(d.capabilities.supports_negative_prompt);

    // `mlx_gen::load("z_image", …)` routes to the base loader (its own snapshot-layout error).
    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
    let err = mlx_gen::load("z_image", &spec)
        .err()
        .expect("a single-file spec is rejected by the loader")
        .to_string();
    assert!(
        err.contains("snapshot directory"),
        "expected the base z-image loader's error, got: {err}"
    );
}

#[test]
fn base_turbo_and_control_all_coexist() {
    // The three z-image engine ids are distinct and all visible in registry iteration — no id
    // collision when the base (sc-8320) was added to the crate that already hosts turbo + control.
    let ids: Vec<&str> = mlx_gen::registry::generators()
        .map(|r| (r.descriptor)().id)
        .filter(|id| id.starts_with("z_image"))
        .collect();
    for want in [
        "z_image",
        "z_image_turbo",
        "z_image_turbo_control",
        "z_image_control",
    ] {
        assert!(ids.contains(&want), "missing {want} in {ids:?}");
    }
}

#[test]
fn unknown_id_still_errors() {
    let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
    assert!(mlx_gen::load("not_a_model", &spec).is_err());
}
