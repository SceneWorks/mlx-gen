//! E8-2 (sc-6511) — real-weight Q4/Q8 **memory profiling** at 1K → `minMemoryGb`.
//!
//! Each profile loads a pipeline, quantizes at load, runs a small **warmup** generation (which
//! materializes the packed weights and frees the dense bf16 transient), records the **load-transient**
//! peak + the packed **resident** set, then `reset_peak_memory()` and runs a **1024²** generation to
//! record the **runtime** peak. Numbers come from MLX's own device counters (`mlx_rs::memory`), which
//! capture Metal/wired memory that `ps`-RSS misses.
//!
//! Two regimes (the SCAIL-2 lesson):
//!   - **load-transient** = peak while the dense bf16 stack (DiT ~20.6 GB + TE ~17.5 GB) is resident
//!     and being packed — the floor for the *download-bf16-then-quantize-in-app* path.
//!   - **runtime** = packed weights + 1K activations, *after* the dense transient is freed — the floor
//!     for the *download-pre-quantized* path E9 ships. `minMemoryGb` is derived from this.
//!
//! `#[ignore]` (needs a 128 GB Mac + the snapshots). Run one config per process for clean counters:
//!   BOOGU_BASE_DIR=<base> [BOOGU_TURBO_DIR=<turbo>] CARGO_TARGET_DIR=~/Repos/mlx-gen/target \
//!     cargo test -p mlx-gen-boogu --test quant_profile profile_base_q4 -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen_boogu::{BooguPipeline, EditOptions, GenerateOptions, TurboOptions};
use mlx_rs::memory::{get_active_memory, get_peak_memory, reset_peak_memory};

fn base_dir() -> PathBuf {
    PathBuf::from(std::env::var("BOOGU_BASE_DIR").expect("set BOOGU_BASE_DIR"))
}

fn turbo_dir() -> PathBuf {
    PathBuf::from(std::env::var("BOOGU_TURBO_DIR").expect("set BOOGU_TURBO_DIR"))
}

fn gb(bytes: usize) -> f64 {
    bytes as f64 / 1e9
}

/// Print the three footprint numbers for a config in a single grep-friendly line.
fn report(label: &str, transient: usize, resident: usize, runtime: usize) {
    println!(
        "[E8-2] {label:14} load-transient {:6.1} GB | resident(packed) {:6.1} GB | runtime-peak@1K {:6.1} GB",
        gb(transient),
        gb(resident),
        gb(runtime)
    );
}

/// 1024² generation knobs — few-step (peak footprint is per-step, independent of step count, so 4
/// steps measures the same peak as 50 far faster).
fn opts_1k(seed: u64) -> GenerateOptions {
    GenerateOptions {
        height: 1024,
        width: 1024,
        steps: 4,
        text_guidance_scale: 4.0,
        seed,
        ..Default::default()
    }
}

fn warmup(pipe: &BooguPipeline) {
    let _ = pipe
        .generate(
            "warmup",
            &GenerateOptions {
                height: 256,
                width: 256,
                steps: 2,
                text_guidance_scale: 4.0,
                seed: 0,
                ..Default::default()
            },
        )
        .expect("warmup generate");
}

#[test]
#[ignore = "needs real weights (128 GB Mac): set BOOGU_BASE_DIR"]
fn profile_base_q4() {
    reset_peak_memory();
    let mut pipe = BooguPipeline::from_snapshot(base_dir()).expect("load Base");
    pipe.quantize(4).expect("quantize Q4");
    warmup(&pipe);
    let transient = get_peak_memory();
    let resident = get_active_memory();
    reset_peak_memory();
    let _ = pipe
        .generate("a red apple on a wooden table", &opts_1k(1))
        .expect("generate 1K");
    report("Base Q4", transient, resident, get_peak_memory());
}

#[test]
#[ignore = "needs real weights (128 GB Mac): set BOOGU_BASE_DIR"]
fn profile_base_q8() {
    reset_peak_memory();
    let mut pipe = BooguPipeline::from_snapshot(base_dir()).expect("load Base");
    pipe.quantize(8).expect("quantize Q8");
    warmup(&pipe);
    let transient = get_peak_memory();
    let resident = get_active_memory();
    reset_peak_memory();
    let _ = pipe
        .generate("a red apple on a wooden table", &opts_1k(1))
        .expect("generate 1K");
    report("Base Q8", transient, resident, get_peak_memory());
}

#[test]
#[ignore = "needs real Turbo weights (128 GB Mac): set BOOGU_TURBO_DIR"]
fn profile_turbo_q4() {
    reset_peak_memory();
    let mut pipe = BooguPipeline::from_snapshot(turbo_dir()).expect("load Turbo");
    pipe.quantize(4).expect("quantize Q4");
    // Turbo warmup uses the DMD sampler (no CFG).
    let _ = pipe
        .generate_turbo(
            "warmup",
            &TurboOptions {
                height: 256,
                width: 256,
                steps: 2,
                seed: 0,
                conditioning_sigma: 0.001,
            },
        )
        .expect("warmup turbo");
    let transient = get_peak_memory();
    let resident = get_active_memory();
    reset_peak_memory();
    let _ = pipe
        .generate_turbo(
            "a red apple on a wooden table",
            &TurboOptions {
                height: 1024,
                width: 1024,
                steps: 4,
                seed: 1,
                conditioning_sigma: 0.001,
            },
        )
        .expect("generate turbo 1K");
    report("Turbo Q4", transient, resident, get_peak_memory());
}

/// Edit path footprint = Base footprint + the lazily-loaded f32 vision tower + the reference image
/// tokens. Footprint is arch-determined, so the Base snapshot stands in for the Edit checkpoint here
/// (same module sizes); only the reference image is synthetic.
#[test]
#[ignore = "needs real weights (128 GB Mac): set BOOGU_BASE_DIR"]
fn profile_edit_q4() {
    reset_peak_memory();
    let mut pipe = BooguPipeline::from_snapshot(base_dir()).expect("load Base");
    pipe.quantize(4).expect("quantize Q4");
    let reference = Image {
        width: 512,
        height: 512,
        pixels: vec![128u8; 512 * 512 * 3],
    };
    let edit_opts = |h: u32, w: u32, steps: usize| EditOptions {
        height: h,
        width: w,
        steps,
        text_guidance_scale: 4.0,
        seed: 0,
        condition_on_image: true,
        use_input_images_4_neg_instruct: false,
        ..Default::default()
    };
    // Warmup also forces the lazy vision-tower load (f32) so the transient peak reflects it.
    let _ = pipe
        .generate_edit(&reference, "warmup", &edit_opts(256, 256, 2))
        .expect("warmup edit");
    let transient = get_peak_memory();
    let resident = get_active_memory();
    reset_peak_memory();
    let _ = pipe
        .generate_edit(&reference, "make it green", &edit_opts(1024, 1024, 4))
        .expect("generate edit 1K");
    report("Edit Q4", transient, resident, get_peak_memory());
}
