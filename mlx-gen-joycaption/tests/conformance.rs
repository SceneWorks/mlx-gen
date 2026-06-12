//! Real-weight gen-core **Captioner contract** conformance for `joy_caption` (epic 3720, sc-4895).
//!
//! The captioner half of the "one real provider per contract" AC: it drives the actual JoyCaption
//! MLX engine through the backend-neutral checks (capability honesty, progress monotonicity, typed
//! cancellation, registry round-trip) — the guarantees a candle captioner will be held to
//! identically. `#[ignore]` because it needs the real
//! `fancyfeast/llama-joycaption-beta-one-hf-llava` snapshot; run on the self-hosted Apple-Silicon
//! runner or a populated dev box:
//!   cargo test -p mlx-gen-joycaption --test conformance -- --ignored --nocapture

use std::path::PathBuf;

// Force-link the provider so its `inventory::submit!` registration survives the linker (this test
// references no other joycaption symbol) — the registry round-trip check would otherwise fail.
use mlx_gen_joycaption as _;

use gen_core_testkit::CaptionerProfile;
use mlx_gen::{LoadSpec, WeightsSource};

#[test]
#[ignore = "needs the JoyCaption HF snapshot; set MLX_GEN_JOYCAPTION_SNAPSHOT (macos-mlx / dev box only)"]
fn joy_caption_satisfies_gen_core_contract() {
    let root = PathBuf::from(
        std::env::var("MLX_GEN_JOYCAPTION_SNAPSHOT")
            .expect("set MLX_GEN_JOYCAPTION_SNAPSHOT to a JoyCaption snapshot directory"),
    );
    let id = mlx_gen::caption::joycaption::JOY_CAPTION_MODEL_ID;
    gen_core_testkit::captioner_conformance(
        || {
            let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
            mlx_gen::load_captioner(id, &spec).expect("load joy_caption")
        },
        // 64² image / 16 greedy tokens — the cheapest valid caption (min_image_size 1, greedy is
        // seed-free and fast).
        &CaptionerProfile::cheap(),
    );
}
