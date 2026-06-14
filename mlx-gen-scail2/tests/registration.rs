//! Registration + capability-surface smoke for the SCAIL-2 provider (sc-5442). No weights.

use mlx_gen::{LoadSpec, Modality, WeightsSource};
use mlx_gen_scail2::pipeline::{descriptor, MODEL_ID};

#[test]
fn descriptor_is_scail2() {
    let d = descriptor();
    assert_eq!(d.id, "scail2_14b");
    assert_eq!(d.id, MODEL_ID);
    assert_eq!(d.family, "scail2");
    assert_eq!(d.backend, "mlx");
    assert_eq!(d.modality, Modality::Video);
    assert!(d.capabilities.mac_only);
    assert!(d.capabilities.supports_negative_prompt);
    assert!(d.capabilities.supports_guidance);
    assert!(!d.capabilities.supports_true_cfg);
    assert_eq!(d.capabilities.max_count, 1);
}

#[test]
fn registered_in_inventory() {
    assert!(
        mlx_gen::registry::generators().any(|r| (r.descriptor)().id == "scail2_14b"),
        "scail2_14b should self-register via inventory::submit!"
    );
}

#[test]
fn load_by_id_reaches_our_loader() {
    // Registered → `registry::load` resolves us → our loader errors on the missing dir (NOT the
    // registry's "no generator registered" miss). Proves both registration and the load path.
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/scail2".into()));
    let err = mlx_gen::registry::load("scail2_14b", &spec)
        .err()
        .expect("loading from a nonexistent dir must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("does not exist"),
        "expected our loader's missing-dir error, got: {msg}"
    );
    assert!(!msg.contains("no generator registered"), "got: {msg}");
}
