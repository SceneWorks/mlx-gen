# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`mlx-gen` is a Rust workspace for **on-device inference of generative image/video models on Apple MLX**, built on the `mlx-rs` crate. It is a from-scratch Rust reimplementation of the (now frozen) SceneWorks Python **mflux fork** — that fork is the **reference spec**, not an upstream; we diverge permanently and never merge back. The goal is a single statically-linked component with no Python sidecar.

Read `ARCHITECTURE.md` (design stance) and `docs/MODEL_ARCHITECTURE.md` (the `Generator`/`Transform` contract + provider/registry model) before non-trivial work. `tools/golden/README.md` explains the parity-golden convention.

## Workspace layout

- **`mlx-gen`** (root `src/`) — THE CORE: shared `nn` primitives, `adapters` (LoRA/LoKr `AdaptableLinear`), `weights`, `quant`, `sampler`/`scheduler`, `image`, `tokenizer`, `train` kernels. Re-exports `gen-core` at the historical `mlx_gen::…` paths. **No model-specific code.**
- **`gen-core`** — backend-neutral contract layer (epic 3720) with **zero tensor deps**: the `Generator`/`Trainer`/`Captioner`/`Transform`/`TextLlm` traits, request/output/conditioning/progress/cancel/error types, the link-time registry, and pure host-side policy math (tokenization, PIL-compatible resize, tiling, LR schedule). Numeric types restricted to `f32`/`f64`/`Vec<f32>`/`&[u8]` — never an `Array`. Builds/tests on Linux.
- **`gen-core-testkit`** — conformance suite (also zero tensor deps); family crates dev-depend on it to run their real model through cancel/progress/seed/capabilities checks.
- **`mlx-gen-<family>`** (~24 crates) — provider crates: `-z-image`, `-flux`, `-flux2`, `-chroma`, `-qwen-image`, `-sdxl`, `-kolors`, `-sensenova`, `-wan`, `-ltx`, `-svd`, `-seedvr2`, `-pulid`, `-instantid`, `-face`, `-joycaption`, `-sam2`, `-sam3`, `-bernini`, `-scail2`, `-lens`, `-boogu`, `-ideogram`, `-prompt-refine`. Most depend **only on `mlx-gen`** and self-register a model via `inventory`. Exceptions to keep straight: a few reuse a sibling family crate rather than building on core alone — `-kolors` and `-instantid` build on `-sdxl`; `-pulid` is **FLUX-family** (builds on `-flux`, NOT `-sdxl`); both identity crates also use `-face`. And `-instantid` is a **struct API** (`InstantId`/`InstantIdRequest`, composing SDXL UNet/ControlNet/Resampler parts) — it does **not** `inventory::submit!` a registered `Generator`.

## Build / lint / test

```sh
cargo build --workspace
cargo fmt --all --check          # CI gate — run fmt --check, not just clippy
cargo clippy --workspace --all-targets -- -D warnings   # warnings are errors

cargo test --workspace                       # whole workspace
cargo test -p mlx-gen-z-image                # one crate
cargo test -p mlx-gen-z-image --test pipeline   # one test file (tests/pipeline.rs)
cargo test -p mlx-gen-z-image some_test_name    # one test by name
```

- **`RUST_TEST_THREADS=1` is forced** via `.cargo/config.toml` (`force = true`). MLX's shared default Metal device is **not thread-safe** and SIGSEGVs under cargo's parallel harness. Do not remove this or run tests with `--test-threads`.
- Workspace types are shared across crates — when changing a public type in `mlx-gen` core or a crate reused by others (e.g. SDXL types reused by kolors/instantid; FLUX types reused by pulid), lint/test with `--workspace`, not crate-scoped, or you'll discover the break in CI.
- `mlx-sys` builds MLX from source via cmake (~5 min, cached) and needs full Xcode + the Metal Toolchain. Apple-Silicon only.

### Real-weight tests vs default tests

- **Default `cargo test` is green on a fresh clone** — it depends only on committed inputs.
- `tests/fixtures/*.safetensors` (per crate): committed, small/synthetic, run by default.
- `tools/golden/*`: **gitignored**, large, regenerable; the tests that read them are **`#[ignore]`d** and need licensed HF weights + Metal. Run with `--ignored` and the right env var pointing at a snapshot, e.g.:
  ```sh
  ZIMAGE_SNAPSHOT=/path/to/Z-Image-Turbo \
    cargo test -p mlx-gen-z-image --release --test e2e_real_weights -- --ignored --nocapture
  ```
  Goldens are dumped by `tools/dump_*.py` run from the frozen `mflux` fork at `~/repos/mflux` (its `.venv`).

## Architecture conventions (the "disciplined hybrid")

The split is deliberate — see `ARCHITECTURE.md`:

- **Numeric leaves** (attention, FFN, norms, RoPE, per-block math) → **faithful mirror** of the fork, so a parity failure localizes to one op.
- **Weight tensor keys** → **compatibility contract**: keys mirror the fork's `tree_flatten` names so fork checkpoints (incl. Q4/Q8) load unchanged. **Keys ≠ struct layout** — a divergent shape carries an explicit remap in `from_weights`.
- **Orchestration** (loaders, dispatch, pipelines, errors, config, adapter/quant application) → **clean Rust**, no Python-isms.

### Module pattern

- A module is a plain struct owning its tensors/sub-modules with `fn forward(&self, …) -> Result<Array>` taking `&self` (NOT `mlx-rs`'s `&mut self` `Module` trait) — so a whole model tree evaluates through shared references, no interior mutability.
- Construction is `fn from_weights(w: &Weights, prefix: &str, …) -> Result<Self>`. Required tensors via `Weights::require` (errors, never panics); optional via `Weights::get` → `Option<Array>`.
- Modules are **dimension-parametric** (shapes from config/weights), so the same code runs the real model and tiny parity fixtures.

### Adapters & quantization

- Every quantizable/adaptable projection is an `AdaptableLinear` (`src/adapters.rs`): base + a stack of forward-time residual adapters → `base(x) + Σ adapter.residual(x)`. The **base is never fused/mutated** (fusing would force re-quantization on adapter swap and break quant-safety; LoRA/LoKr compose with Q4/Q8 for free). Adapters install by **dotted path** (the Rust replacement for Python's dynamic `getattr`).
- Quantization is group-wise affine Q4/Q8 at `group_size = 64`, verified **byte-identical** to the fork's packing.
- Errors: one `thiserror` enum `error::Error` with `Result<T>` at the crate root. At the gen-core contract boundary, `?` on a raw `mlx_rs::Exception` will NOT bridge to `gen_core::Error` — provider `*_registered` adapters wrap `crate::Result` → `gen_core::Result` via `.map_err(Into::into)`.

### Adding a model (additive — no edits to core or other providers)

1. `cargo new --lib mlx-gen-<x>`; depend on `mlx-gen`; add to root `Cargo.toml` `members`.
2. Build on `mlx-gen`'s `nn`/`weights`/`quant`/`tokenizer`/`adapters` primitives.
3. `impl Generator` (or `Transform`/`Captioner`/`Trainer`); provide `descriptor()` + `Capabilities`; `inventory::submit! { ModelRegistration { descriptor, load: load_registered } }`.

**Linkage gotcha:** a provider self-registers only when actually linked. A declared-but-unreferenced dependency has its `inventory::submit!` statics dropped by the linker. A consumer that depends on a provider purely for the registration side-effect must force the link with `use mlx_gen_<x> as _;` (this is why the worker needs one such line per model crate, else "no generator registered").

## Testing philosophy

- **Parity over assertion-of-correctness.** Each ported block has a committed parity test vs a tiny fixture dumped from the fork (`tools/dump_*.py` → `tests/fixtures/*.safetensors`).
- **Tolerance ~`1e-2`.** MLX runs fp32 matmul in reduced precision on Metal (~1e-3); matmul chains agree to 3–4 sig figs, not bit-exactly. A real structural bug diverges by orders of magnitude.
- **Cross-backend full-trajectory pixel parity vs a torch reference is chaos-limited** (per-step ~5e-4 compounds under CFG). Gate on components + early-step + coherence, and prefer a same-backend MLX reference for end-to-end.

## Dependency pins (important, single source of truth)

- All shared deps live in **`[workspace.dependencies]`** in the root `Cargo.toml`; members opt in with `<dep> = { workspace = true }`. Bumping MLX/the fork is a **one-line edit there** — editing per-crate would produce two distinct `mlx-rs` builds and cross-crate `Array` type-mismatch errors.
- `mlx-rs` is pinned to a **GitHub fork by commit SHA** (`pmetal-mlx-rs`, mlx-rs 0.25 whose mlx-sys builds MLX core 0.31.2). Read the long comments in `Cargo.toml` before touching the SHA.
- **`MACOSX_DEPLOYMENT_TARGET = "26.2"`** in `.cargo/config.toml` is load-bearing: below 26.2 the NAX 16-bit Metal kernels miscompile to garbage (bf16/f16 GEMM + fused SDPA). CI lowers it to 15.0 because hosted runners cap at SDK 15 (so CI does NOT exercise the NAX fast path — a macOS-26.2+ self-hosted runner is required for that). `mlx-sys`'s `build.rs` has no `rerun-if-env-changed`, so a clean rebuild of `pmetal-mlx-sys` is required for a change here to take effect.
- Rust toolchain is **pinned in `rust-toolchain.toml`** (single source of truth for CI + local); bump deliberately since each stable can add lints that redden `clippy -D warnings`.
- **Local metallib cache can shadow the build (sc-7889).** The fork loads its Metal kernel library at runtime via a patched resolver (`mlx-sys/patches/metallib-search-path.patch`) in this order: `$PMETAL_METALLIB_PATH` → **`~/.cache/pmetal/lib/mlx.metallib`** → colocated `mlx.metallib`. Local `cargo test`/`run` binaries have **no colocated metallib**, so the user-cache copy is the *sole* working resolution — load-bearing, never delete it without replacing it. `build.rs` only refreshes that cache when it re-runs **and** the build artifact is mtime-newer ("newest-build-wins by mtime"); it is keyed by neither rev nor content. So an **older/incomplete** pmetal build (debug/release × historical rev) can own the cache and shadow a complete build. Older revs instantiate a smaller steel-GEMM NAX tile set lacking the small-M `bm64` tile (`steel_gemm_fused_nax_*_bm64_bn128_bk256_*`): large-M matmuls (inference) pick tiles the cache *has* and work, but small-M matmuls (**every LoRA training step**, M≈64) pick `bm64` → **PANIC "… _bm64_… not found"**. Not a missing-kernel/JIT issue — the pinned fork's metal source *does* instantiate `bm64`; the fault is the stale cache. Cure/guard: `tools/refresh_pmetal_metallib.sh` (installs the newest complete build metallib into the cache; `--check` validates it — run as a preflight before local real-weight training). Standing tripwire in the suite: `cargo test -p mlx-gen-qwen-image --release --test bf16_matmul_sweep` (exercises small-M NAX GEMM; crashes if the cache regresses).

## CI

GitHub Actions, two lanes (`.github/workflows/ci.yml`):
1. **contract** (Linux): `gen-core` + `gen-core-testkit` fmt/clippy/test — proves the contract is backend-independent.
2. **test** (macOS Apple-Silicon): `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.

## Notes for this repo

- This work is tracked in **Shortcut** (epics/stories) and surfaced via the codegraph/shortcut MCP servers — see the user's global CLAUDE.md for the requirements→plan→Shortcut workflow.
- `tools/` holds dev-only Python `dump_*.py`/`convert_*.py`/`build_*_tokenizer.py` scripts (golden dumps, weight converters, tokenizer builders). They run from the frozen `mflux` fork's `.venv`, never as part of the Rust build. Also `tools/refresh_pmetal_metallib.sh` — the local Metal-kernel-library cache cure/guard (see Dependency pins → sc-7889).
- `_vendor/` (gitignored) holds read-only third-party reference checkouts cloned on demand for porting.
