#!/usr/bin/env bash
# refresh_pmetal_metallib.sh — keep the local Metal kernel library complete (sc-7889).
#
# WHY THIS EXISTS
# --------------
# The pinned fork (pmetal-mlx-rs) loads its Metal kernel library at runtime via a patched
# resolver (mlx-sys/patches/metallib-search-path.patch -> mlx/backend/metal/device.cpp). The
# resolution order is:
#       $PMETAL_METALLIB_PATH  ->  ~/.cache/pmetal/lib/mlx.metallib  ->  colocated mlx.metallib  -> ...
# For local `cargo test`/`cargo run` binaries (target/{release,debug}/deps/...) there is NO
# colocated metallib, so the user-cache copy is the SOLE working resolution — it is load-bearing,
# not just a fallback. Never delete it without immediately replacing it, or MLX cannot load ANY
# metallib and every Metal op aborts.
#
# THE BUG IT CURES
# ----------------
# mlx-sys/build.rs populates that cache from the just-built metallib, but only when build.rs
# re-runs AND the build artifact is mtime-newer ("newest-build-wins by mtime"); the cache is keyed
# by neither rev nor content. A machine accumulates many pmetal-mlx-sys builds (debug/release x
# historical revs). Older revs instantiate a SMALLER steel-GEMM NAX tile set — in particular they
# may lack the small-M `bm64` tile (steel_gemm_fused_nax_*_bm64_bn128_bk256_*). If such an
# incomplete build's metallib wins the cache, then:
#   * large-M matmuls (inference: 1024^2 -> 4096 tokens) pick tiles the cache HAS -> work fine;
#   * small-M matmuls (EVERY LoRA training step, M~=64) pick `bm64`, which is absent
#     -> PANIC at kernel-load: "... _bm64_bn128_bk256_wm2_wn4 not found".
# That is why Krea Turbo inference renders but the Krea Raw trainer cannot run a single step.
#
# The fix is NOT a source change: the pinned fork's metal source DOES instantiate the bm64 tiles
# (mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.metal). The only fault is a
# stale/incomplete cache shadowing a complete build. This script installs a COMPLETE,
# bm64-bearing metallib into the cache, and verifies it.
#
# USAGE
#   tools/refresh_pmetal_metallib.sh            # install newest complete build metallib into the cache
#   tools/refresh_pmetal_metallib.sh --check    # verify cache only; exit non-zero if bm64 is missing
#   tools/refresh_pmetal_metallib.sh /path/to/mlx.metallib   # install an explicit metallib
#
# Run --check as a preflight before any local real-weight MLX *training* run. The standing
# tripwire in the test suite is `cargo test -p mlx-gen-qwen-image --release --test
# bf16_matmul_sweep` — it exercises small-M NAX GEMM and crashes/fails if the cache regresses.
set -euo pipefail

# The exact tile whose absence blocks small-M (training) matmuls. f32 + bf16 both required.
PROBE_F32="steel_gemm_fused_nax_nn_float32_float32_bm64_bn128_bk256"
PROBE_BF16="steel_gemm_fused_nax_nn_bfloat16_bfloat16_bm64_bn128_bk256"

CACHE_DIR="${HOME}/.cache/pmetal/lib"
CACHE="${CACHE_DIR}/mlx.metallib"

has_tile() { # path, probe-string -> 0 if present
  grep -qa "$2" "$1" 2>/dev/null
}

verify() { # path -> 0 if both f32 and bf16 bm64 tiles present
  local p="$1"
  has_tile "$p" "$PROBE_F32" && has_tile "$p" "$PROBE_BF16"
}

# --- --check mode: validate the live cache, no writes -----------------------------------------
if [[ "${1:-}" == "--check" ]]; then
  if [[ ! -f "$CACHE" ]]; then
    echo "FAIL: no cache at $CACHE — run this script with no args to install one." >&2
    exit 1
  fi
  if verify "$CACHE"; then
    echo "OK: $CACHE has the bm64 NAX GEMM tiles (f32 + bf16). Local MLX training can run."
    exit 0
  fi
  echo "FAIL: $CACHE is missing the bm64 NAX GEMM tile — small-M (training) matmuls will PANIC." >&2
  echo "      Cure: tools/refresh_pmetal_metallib.sh" >&2
  exit 1
fi

# --- explicit-path mode -----------------------------------------------------------------------
SRC=""
if [[ $# -ge 1 && "$1" != --* ]]; then
  SRC="$1"
  [[ -f "$SRC" ]] || { echo "no metallib at: $SRC" >&2; exit 1; }
  verify "$SRC" || { echo "refusing to install: $SRC lacks the bm64 tile." >&2; exit 1; }
fi

# --- auto-discover the newest COMPLETE build metallib -----------------------------------------
if [[ -z "$SRC" ]]; then
  # Workspace root = dir of this script's parent (tools/ lives at the workspace root).
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  TARGET_DIR="${CARGO_TARGET_DIR:-${ROOT}/target}"
  newest_mtime=0
  while IFS= read -r m; do
    [[ -f "$m" ]] || continue
    verify "$m" || continue            # only consider metallibs that actually carry bm64
    mt=$(stat -f '%m' "$m")
    if (( mt > newest_mtime )); then newest_mtime=$mt; SRC="$m"; fi
  done < <(find "$TARGET_DIR" -path '*pmetal-mlx-sys-*/out/build/lib/mlx.metallib' 2>/dev/null)

  if [[ -z "$SRC" ]]; then
    echo "No complete (bm64-bearing) pmetal-mlx-sys metallib found under $TARGET_DIR." >&2
    echo "Build the workspace first (cargo build --release), then re-run this script." >&2
    exit 1
  fi
fi

# --- install --------------------------------------------------------------------------------
echo "source : $SRC"
if [[ -f "$CACHE" ]]; then
  if verify "$CACHE"; then echo "cache  : already complete ($CACHE)"; else echo "cache  : STALE/incomplete ($CACHE)"; fi
else
  echo "cache  : missing ($CACHE)"
fi
mkdir -p "$CACHE_DIR"
cp -f "$SRC" "$CACHE"               # replace atomically-enough; never leaves the cache absent

if verify "$CACHE"; then
  echo "OK: installed complete metallib -> $CACHE"
  echo "    (bm64 NAX GEMM tiles present for f32 + bf16; local MLX training can run)"
else
  echo "ERROR: post-copy cache still lacks the bm64 tile — copy failed?" >&2
  exit 1
fi
