//! SD3.5-Large / Large-Turbo Q8/Q4 memory + load-path profiling harness (E7, sc-7866).
//!
//! This is the **reproducible profiling method** behind the SceneWorks worker manifest's
//! `minMemoryGb` for the four SD3.5 image variants (Large-Q8/Q4, Turbo-Q8/Q4). It exercises BOTH
//! quantization load paths the AC calls for and reports peak memory for a real 1024² generation:
//!
//! 1. **Load-time quantization** — load the dense bf16 snapshot, quantize in-process
//!    ([`LoadSpec::with_quant`]). The default worker path.
//! 2. **Pre-quantized on disk** — convert the dense `transformer/` once to a packed Q4/Q8 artifact
//!    ([`mlx_gen_sd3::quantize_sd3_dir`]) into a temp snapshot overlay, then load that directly (the
//!    loader auto-detects the packed transformer). Faster startup, lower load-time peak.
//!
//! ## Measuring peak memory — TWO complementary numbers
//!
//! MLX's working set is **Metal *wired* memory**, which `ps`/RSS *understates* badly. So this harness
//! reports both:
//!
//!   * **MLX `metal::get_peak_memory()`** (printed here per phase) — the MLX-allocator high-water
//!     mark in bytes. This is the *allocator* (relative) read; it UNDER-counts wired memory by ~2×,
//!     so it is NOT the basis for `minMemoryGb` — use it for relative comparisons across cells.
//!   * **Process "peak memory footprint"** — the wired-inclusive OS figure (from `/usr/bin/time -l`).
//!     MLX's wired buffers do NOT show in RSS, and this OS footprint is what `minMemoryGb` is based on.
//!
//! Capture the OS footprint by running the **compiled test binary directly** under `/usr/bin/time -l`,
//! NOT `cargo test`: `/usr/bin/time` measures its immediate child, and `cargo test` execs the test
//! binary as a *grandchild*, so timing `cargo` reports only cargo's own ~40 MB footprint. Build the
//! test binary with `--no-run`, then run the resolved binary directly:
//!
//! ```sh
//! # 1) Build the test binary WITHOUT running it.
//! cargo test -p mlx-gen-sd3 --release --test profile_memory --no-run
//! # 2) Resolve the binary and run it DIRECTLY under /usr/bin/time -l.
//! BIN=$(ls -t target/release/deps/profile_memory-* | grep -v '\.d$' | head -1)
//! SD3_LARGE_SNAPSHOT=~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-large/snapshots/<rev> \
//! SD3_TURBO_SNAPSHOT=~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-large-turbo/snapshots/<rev> \
//! SD3_PROFILE_VARIANT=large SD3_PROFILE_QUANT=q8 \
//!   /usr/bin/time -l "$BIN" profile_memory_single --ignored --nocapture
//! ```
//!
//! Select the variant / quant / path via env (so each `/usr/bin/time -l` run isolates ONE figure —
//! a single process measures one peak):
//!   * `SD3_PROFILE_VARIANT` = `large` (default) | `turbo`
//!   * `SD3_PROFILE_QUANT`   = `q8` (default) | `q4`
//!   * `SD3_PROFILE_PATH`    = `loadtime` (default) | `prequant`
//!   * `SD3_PROFILE_SIZE`    = `1024` (default) — square edge in px
//!
//! `profile_memory_all` sweeps the matrix in ONE process for the MLX-allocator peaks (handy for a
//! quick relative read) but the per-process `/usr/bin/time -l` wired figure is per `profile_memory_single`.
//!
//! `#[ignore]`d — needs the real snapshots in the HF cache (or the `SD3_*_SNAPSHOT` overrides) + Metal.

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use mlx_rs::memory::{get_peak_memory, reset_peak_memory};

// Keep the SD3.5 generator registrations linked (the CLAUDE.md "Linkage gotcha"): this test reaches
// the generator through the `mlx_gen::load` registry only.
use mlx_gen_sd3 as sd3;

const GB: f64 = 1e9;

#[derive(Clone, Copy, Debug)]
enum Variant {
    Large,
    Turbo,
}

impl Variant {
    fn id(self) -> &'static str {
        match self {
            Variant::Large => sd3::MODEL_ID,
            Variant::Turbo => sd3::TURBO_MODEL_ID,
        }
    }
    /// Reference sampling recipe: Large = 28-step true-CFG, Turbo = 4-step CFG-off.
    fn steps(self) -> u32 {
        match self {
            Variant::Large => 28,
            Variant::Turbo => 4,
        }
    }
    fn guidance(self) -> Option<f32> {
        match self {
            Variant::Large => Some(3.5),
            Variant::Turbo => None, // distilled — CFG off
        }
    }
    fn env_snapshot(self) -> &'static str {
        match self {
            Variant::Large => "SD3_LARGE_SNAPSHOT",
            Variant::Turbo => "SD3_TURBO_SNAPSHOT",
        }
    }
    fn hub_dir(self) -> &'static str {
        match self {
            Variant::Large => "models--stabilityai--stable-diffusion-3.5-large",
            Variant::Turbo => "models--stabilityai--stable-diffusion-3.5-large-turbo",
        }
    }
}

/// Resolve a variant's snapshot dir: the `SD3_*_SNAPSHOT` override, else the first snapshot in the HF
/// hub cache.
fn snapshot(v: Variant) -> PathBuf {
    if let Ok(p) = std::env::var(v.env_snapshot()) {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(v.hub_dir())
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("set {} or populate {snaps:?}", v.env_snapshot()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .unwrap_or_else(|| panic!("no snapshot under {snaps:?}"))
}

/// Build a temp snapshot overlay whose `transformer/` is the PRE-QUANTIZED packed artifact for `bits`
/// and every other component is symlinked back to the dense source — so the loader sees a full
/// snapshot but loads the packed transformer directly (the pre-quantized-on-disk path). Returns the
/// overlay root (the caller should not delete the source).
fn prequant_overlay(src: &std::path::Path, v: Variant, bits: i32) -> PathBuf {
    let arch = match v {
        Variant::Large | Variant::Turbo => sd3::Sd3Arch::large(),
    };
    let overlay =
        std::env::temp_dir().join(format!("sd3_prequant_{}_{bits}", v.id().replace('/', "_")));
    std::fs::create_dir_all(&overlay).ok();
    // Symlink every sibling component (text encoders / vae / tokenizers / *.json) from the source.
    for entry in std::fs::read_dir(src).expect("read source snapshot") {
        let entry = entry.unwrap();
        let name = entry.file_name();
        if name == "transformer" {
            continue; // replaced by the packed dir below
        }
        let link = overlay.join(&name);
        if !link.exists() {
            std::os::unix::fs::symlink(entry.path(), &link).expect("symlink component");
        }
    }
    // Pre-quantize the transformer once (idempotent: skip if already packed).
    let packed_dir = overlay.join("transformer");
    let packed = packed_dir.join("diffusion_pytorch_model.safetensors");
    if !packed.exists() {
        let t = Instant::now();
        sd3::quantize_sd3_dir(&arch, &src.join("transformer"), &packed_dir, bits, 64)
            .expect("pre-quantize transformer to disk");
        eprintln!(
            "  [prequant] converted transformer → Q{bits} packed artifact in {:.1}s ({:.2} GB on disk)",
            t.elapsed().as_secs_f32(),
            std::fs::metadata(&packed).map(|m| m.len() as f64 / GB).unwrap_or(0.0),
        );
    }
    overlay
}

/// One profiled run: load (load-time-quant OR pre-quantized-on-disk) + a 1024² generation, reporting
/// the MLX-allocator peak (`get_peak_memory`) for load and for generate.
fn profile_one(v: Variant, quant: Quant, prequant: bool, size: u32) {
    // Force the registration link + keep the id honest.
    assert_eq!(sd3::MODEL_ID, "sd3_5_large");

    let src = snapshot(v);
    let bits = quant.bits();
    let path_label = if prequant {
        "prequant-on-disk"
    } else {
        "load-time-quant"
    };
    eprintln!(
        "\n=== SD3.5 PROFILE [{}] Q{bits} {path_label} {size}x{size} steps={} ===\n  source: {src:?}",
        v.id(),
        v.steps(),
    );

    let (root, spec) = if prequant {
        let overlay = prequant_overlay(&src, v, bits);
        // No `with_quant`: the packed transformer is auto-detected by the loader.
        (overlay.clone(), LoadSpec::new(WeightsSource::Dir(overlay)))
    } else {
        (
            src.clone(),
            LoadSpec::new(WeightsSource::Dir(src.clone())).with_quant(quant),
        )
    };
    let _ = root;

    reset_peak_memory();
    let t_load = Instant::now();
    let generator = mlx_gen::load(v.id(), &spec).unwrap_or_else(|e| panic!("load {}: {e}", v.id()));
    let load_s = t_load.elapsed().as_secs_f32();
    let load_peak = get_peak_memory() as f64 / GB;
    eprintln!("  loaded in {load_s:.1}s — MLX peak after load: {load_peak:.2} GB");

    let req = GenerationRequest {
        prompt: "a photograph of a red fox sitting in a green meadow, sharp focus, daylight".into(),
        negative_prompt: v
            .guidance()
            .map(|_| "blurry, low quality, distorted".into()),
        width: size,
        height: size,
        count: 1,
        seed: Some(7),
        steps: Some(v.steps()),
        guidance: v.guidance(),
        ..Default::default()
    };

    reset_peak_memory();
    let t_gen = Instant::now();
    let out = generator.generate(&req, &mut |_| {}).expect("generate");
    let gen_s = t_gen.elapsed().as_secs_f32();
    let gen_peak = get_peak_memory() as f64 / GB;

    let GenerationOutput::Images(imgs) = out else {
        panic!("expected Images");
    };
    assert_eq!(imgs.len(), 1);
    assert_eq!((imgs[0].width, imgs[0].height), (size, size));

    eprintln!(
        "  RESULT [{}] Q{bits} {path_label} {size}x{size}: load {load_s:.1}s ({load_peak:.2} GB) · \
         render {gen_s:.1}s ({gen_peak:.2} GB) · MLX gen peak = {gen_peak:.2} GB",
        v.id()
    );
    eprintln!(
        "  NOTE: this is the MLX-allocator peak; the OS wired-inclusive 'peak memory footprint' is \
         captured by running this under /usr/bin/time -l (see file header)."
    );
}

fn variant_from_env() -> Variant {
    match std::env::var("SD3_PROFILE_VARIANT").as_deref() {
        Ok("turbo") => Variant::Turbo,
        _ => Variant::Large,
    }
}

fn quant_from_env() -> Quant {
    match std::env::var("SD3_PROFILE_QUANT").as_deref() {
        Ok("q4") => Quant::Q4,
        _ => Quant::Q8,
    }
}

fn prequant_from_env() -> bool {
    matches!(std::env::var("SD3_PROFILE_PATH").as_deref(), Ok("prequant"))
}

fn size_from_env() -> u32 {
    std::env::var("SD3_PROFILE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024)
}

/// ONE process / ONE figure — the cell selected by the `SD3_PROFILE_*` env. Run UNDER `/usr/bin/time -l`
/// so the printed MLX peak is paired with the OS wired-inclusive 'peak memory footprint' line.
#[test]
#[ignore = "needs the SD3.5 snapshots (set SD3_LARGE_SNAPSHOT / SD3_TURBO_SNAPSHOT) + Metal"]
fn profile_memory_single() {
    profile_one(
        variant_from_env(),
        quant_from_env(),
        prequant_from_env(),
        size_from_env(),
    );
}

/// Sweep the whole Large/Turbo × Q8/Q4 matrix (load-time-quant) in ONE process — a quick RELATIVE
/// read of the MLX-allocator peaks. The authoritative per-cell wired figure is `profile_memory_single`
/// under `/usr/bin/time -l` (a single process measures one peak).
#[test]
#[ignore = "needs the SD3.5 snapshots (set SD3_LARGE_SNAPSHOT / SD3_TURBO_SNAPSHOT) + Metal"]
fn profile_memory_all() {
    let size = size_from_env();
    for v in [Variant::Large, Variant::Turbo] {
        for q in [Quant::Q8, Quant::Q4] {
            profile_one(v, q, false, size);
        }
    }
}
