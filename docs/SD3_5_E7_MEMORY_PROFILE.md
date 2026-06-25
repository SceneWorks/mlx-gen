# SD3.5 E7 — Q8/Q4 memory profile + `minMemoryGb` (sc-7866)

Memory profiling for **SD3.5-Large** (`sd3_5_large`) and **SD3.5-Large-Turbo**
(`sd3_5_large_turbo`) at Q8 and Q4, to derive the SceneWorks worker manifest's `minMemoryGb`
for each variant. (SD3.5-Medium's numbers were derived in M3 / sc-7869 and are out of scope here.)

## Method

- **Harness:** `mlx-gen-sd3/tests/profile_memory.rs` (`#[ignore]`, env-gated; needs the real
  snapshots + Metal). Drives the public registry path (`mlx_gen::load(id, spec).generate(req)`) for a
  real **1024² text-to-image generation** at the variant's reference recipe (Large = 28-step true-CFG
  guidance 3.5; Turbo = 4-step CFG-off).
- **Two peak figures, both reported:**
  1. **MLX `metal::get_peak_memory()`** — the MLX-allocator high-water mark, read in-process per
     phase (load / generate).
  2. **OS "peak memory footprint"** — the wired-inclusive figure. MLX's working set is Metal *wired*
     memory, which `ps`/RSS understate; capture it by running the **compiled test binary** under
     `/usr/bin/time -l` and reading the `peak memory footprint` line.
- **Run the binary directly under `/usr/bin/time -l`, NOT `cargo test`** — `/usr/bin/time` measures
  the immediate child, and `cargo test` execs the test binary as a *grandchild*, so timing `cargo`
  reports only cargo's own ~40 MB footprint. Build with `--no-run`, then:

  ```sh
  BIN=$(ls -t target/release/deps/profile_memory-* | grep -v '\.d$' | head -1)
  SD3_PROFILE_VARIANT=large SD3_PROFILE_QUANT=q8 SD3_PROFILE_PATH=loadtime \
    /usr/bin/time -l "$BIN" profile_memory_single --ignored --nocapture
  ```

  Env knobs: `SD3_PROFILE_VARIANT` (`large`|`turbo`), `SD3_PROFILE_QUANT` (`q8`|`q4`),
  `SD3_PROFILE_PATH` (`loadtime`|`prequant`), `SD3_PROFILE_SIZE` (default 1024). One process measures
  one peak, so isolate each cell in its own `/usr/bin/time -l` run.

## Raw figures (this Mac: M-series, 128 GB unified, 1024², weights cached)

| Variant | Quant | Path | render | MLX gen peak | **OS peak footprint** |
|---------|-------|------|-------:|-------------:|----------------------:|
| Large       | Q8 | load-time      |  88–106 s | 30.18 GB | **58.79 GB** |
| Large       | Q4 | load-time      |  ~95 s    | 23.42 GB | **52.51 GB** |
| Large-Turbo | Q8 | load-time      |  ~9 s     | 30.17 GB | **56.79 GB** |
| Large-Turbo | Q4 | load-time      |  ~8 s     | 23.41 GB | **50.76 GB** |
| Large       | Q8 | prequant-on-disk | ~93 s   | 36.74 GB | **64.25 GB** |

Notes / honest findings:

- **Large and Turbo have essentially identical memory** — same MMDiT backbone + triple-TE + VAE; the
  Turbo checkpoint differs only in the distilled sampling recipe (4 steps, CFG off), so it's ~10× faster
  but the same footprint at the same quant.
- **`load_peak` reads 0.00 GB** because MLX is lazy — weights aren't materialized until the first
  `generate`. So the **generate peak is the whole-pipeline peak** (load + denoise + decode), which is
  the right figure for `minMemoryGb`.
- **The OS wired footprint is ~25–30 GB ABOVE the MLX-allocator peak** (e.g. Q8 30.18 GB MLX vs
  58.79 GB OS). This is exactly why the AC insists on `/usr/bin/time -l "peak memory footprint"`:
  RSS/the MLX allocator number alone would under-provision by ~2×. The OS footprint includes the
  transient dense bf16 weights mmap'd from disk + the f32 text-encoder load + the in-place quantize
  copy, none of which sit in the steady-state MLX allocator peak.
- **Q4 saves ~7 GB MLX / ~6 GB OS vs Q8** — a real but modest win, because only the *quantizable
  Linears* shrink; the f32 text-encoder load, the dense VAE, and the 1024² f32 activations are
  quant-independent and dominate the transient peak.
- **Pre-quantized-on-disk is functional but NOT a footprint win here** (Q8: 64.25 GB OS vs 58.79 GB
  load-time). Only the *transformer* is pre-packed; the text encoders still load f32 and quantize at
  load-time, and the packed transformer's f32 dequant path adds activation transients. Its value is
  faster startup / a smaller on-disk artifact (8.74 GB packed Q8 transformer), not a lower peak.
  → The production worker path is **load-time quantization** (`LoadSpec::with_quant`), so base
  `minMemoryGb` on the load-time OS footprints.

## Derived `minMemoryGb` (for the SceneWorks S-side manifest — NOT edited here)

Base on the **load-time OS peak footprint** + ~10–15 % headroom (background pressure, fragmentation),
rounded up to a sane gate:

| Engine id (variant)        | Quant | OS peak | **Recommended `minMemoryGb`** |
|----------------------------|-------|--------:|------------------------------:|
| `sd3_5_large`              | Q8    | 58.8 GB | **64** |
| `sd3_5_large`              | Q4    | 52.5 GB | **56** (use 64 if a single gate must cover Q8) |
| `sd3_5_large_turbo`        | Q8    | 56.8 GB | **64** |
| `sd3_5_large_turbo`        | Q4    | 50.8 GB | **56** (use 64 if a single gate must cover Q8) |

Practical recommendation: if the manifest carries **one `minMemoryGb` per engine id** (default quant),
set **`sd3_5_large` and `sd3_5_large_turbo` to `64`** (covers the Q8 default with headroom; Q4 runs
comfortably under the same gate). If the manifest can gate per-quant, use 64 (Q8) / 56 (Q4).

This matches the order of the other big-DiT image families on the same Mac budget and leaves the
pipeline safely inside 128 GB with concurrent OS/app pressure.
