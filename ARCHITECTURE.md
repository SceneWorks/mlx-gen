# mlx-gen architecture

Rust-native inference for generative image/video models on Apple MLX, built on the
[`mlx-rs`](https://crates.io/crates/mlx-rs) API surface but on the SceneWorks `pmetal-mlx-rs`
fork ([`michaeltrefry/mlx-rs`](https://github.com/michaeltrefry/mlx-rs), rev-pinned in the root
`Cargo.toml` — its `mlx-sys` builds MLX core 0.31.2 vs upstream's 0.25.1). It reimplements the (now frozen) SceneWorks
**mflux fork** — the Python MLX inference sidecar — as a single statically-linked Rust
component. The fork is the **reference spec**, not an upstream: we diverge permanently and
do not merge back.

## Guiding principle: disciplined hybrid

We are **neither** a 1:1 structural transliteration of the Python class tree **nor** a
free clean-room rewrite. The split is deliberate:

| Layer | Stance | Why |
|---|---|---|
| **Numeric leaves** — attention, FFN, norms, RoPE, the per-block math | **Faithful mirror** | Lets us dump Python per-stage intermediates and diff 1:1, so a parity failure localizes to a single op instead of "the image looks wrong." |
| **Weight tensor keys** | **Compatibility contract** | Fork checkpoints (incl. Q4/Q8) must load unchanged. Keys mirror the fork's `tree_flatten` names. |
| **Orchestration** — loaders, dispatch, pipelines, error handling, config, adapter/quant application | **Clean Rust** | Python-isms here (`getattr` tree-walks, `[0]`-wrapped singleton modules, predicate/`weight_definition` indirection) are pure noise in permanent Rust and would be copied across 5+ families + video. |

**Corollary — keys ≠ structure.** A tensor key is a load-time contract; it does *not*
dictate struct layout. A module whose clean Rust shape diverges from the fork's key names
carries an explicit remap in its `from_weights`. Today the Z-Image block keys already match
1:1, so no remap exists yet — we don't build unused machinery (see "non-speculative" below).

## Conventions

### Modules
- A module is a plain struct that owns its tensors/sub-modules and exposes
  `fn forward(&self, …) -> Result<Array>` taking `&self` (not the `&mut self` `mlx-rs`
  `Module` trait). Forward calls MLX ops directly, so a whole model tree evaluates through
  shared references — no interior mutability, no `&mut` plumbing through the call graph.
- Construction is `fn from_weights(w: &Weights, prefix: &str, …) -> Result<Self>`. The
  module owns its key layout under `prefix`; keys mirror the fork's `tree_flatten` names.
- Modules are **dimension-parametric** (shapes come from a config struct / weights), never
  hardcoded — so the same code runs the real model and tiny parity fixtures.
- Optional sub-weights (e.g. QK-norm) are `Option<Array>` loaded via `Weights::get`;
  required ones via `Weights::require` (which errors, not panics).

### Linears & adapters
- Every quantizable/adaptable projection is an [`AdaptableLinear`](src/adapters.rs): a base
  (`LinearBase::{Dense(nn::Linear), Quantized(nn::QuantizedLinear)}`) plus a stack of
  forward-time residual adapters — `base(x) + Σ adapter.residual(x)`.
- The base is **never fused/mutated**: fusing would force re-quantization on every adapter
  swap and break the quant-safe property. LoRA/LoKr compose with Q4/Q8 for free.
- Adapters are installed by **dotted path** (`AdaptableHost::adaptable_mut` +
  `install_adapter`) — the Rust replacement for Python's dynamic `getattr`, since `mlx-rs`
  flattens module params to `Array` leaves and can't swap a submodule in place.

### Quantization
- Group-wise affine Q4/Q8 at `group_size = 64` (MLX default; the fork never overrides it).
  `quant::resolve_bits` ports the fork's stored-vs-requested resolution. Verified
  **byte-identical** to the fork's mlx 0.31 packing despite the crate's older mlx-rs 0.25.

### Errors
- One typed enum, [`error::Error`](src/error.rs) (`thiserror`), with `Result<T>` re-exported
  at the crate root. `#[from]` lifts `mlx_rs` / IO errors through `?`; `From<&str>/<String>`
  keep ad-hoc message sites ergonomic.

### Dispatch & pipelines (deferred, non-speculative)
- Model-family dispatch (an enum) and a `Pipeline` trait are introduced **when the second
  family lands**, not before — building them against a single model would be guesswork.
- The adapter-file loader (`networkType=lokr` metadata → `install_adapter`) and a macro to
  generate per-module `adaptable_mut` arms remain in sc-2343.

## Testing
- **Parity over assertion-of-correctness.** Each ported block has a committed parity test vs
  a tiny fixture dumped from the fork (`tools/dump_*.py` → `tests/fixtures/*.safetensors`).
  Fixtures use small dims so they commit cheaply and CI stays fast.
- **Tolerance `1e-2`.** MLX runs fp32 matmul in reduced precision on Metal (~1e-3), so
  matmul chains agree to ~3–4 sig figs, not bit-exactly. This matches mflux's own suite. A
  real structural bug diverges by orders of magnitude.
- **Single-threaded harness.** `.cargo/config.toml` sets `RUST_TEST_THREADS=1`: MLX's shared
  default Metal device is not thread-safe and SIGSEGVs under cargo's parallel harness.

## CI
GitHub Actions (`.github/workflows/ci.yml`), two lanes:
1. **contract** (Linux `ubuntu-latest`): `gen-core` + `gen-core-testkit` fmt/clippy/test — proves
   the contract layer is genuinely backend-independent (zero tensor deps).
2. **test** (`macos-15`, Apple Silicon — MLX is Apple-Silicon-only and tests eval on the Metal GPU,
   which hosted runners do expose). `mlx-sys` builds MLX from source via cmake; the build is cached.
   Hosted runners cap at SDK 15, so CI lowers `MACOSX_DEPLOYMENT_TARGET` to 15.0 (the shipping
   config targets 26.2 for the NAX fast kernels, which needs a macOS-26 self-hosted runner). Gate:
   `fmt --check`, `clippy -D warnings`, `cargo test`. CI runs cargo with `--locked` so the
   committed `Cargo.lock` (rev-pinned `core-llm`/`mlx-llm`/`mlx-rs`) governs, not moving refs.
