#!/usr/bin/env python
"""Convert a released PiD student checkpoint (`model_ema_bf16.pth`) to safetensors for mlx-gen-pid.

Implements the exact key transform from `pid_distill_model.py::PidDistillModel.load_state_dict`
(the inference source of truth):

  - keep `net.*`  → strip the `net.` prefix  (the student backbone + LQ adapter mlx-gen-pid loads)
  - drop `net_ema.*` / `fake_score.*` / `discriminator.*`  (training-only teacher/GAN machinery)
  - keep anything else verbatim  (handles an already-bare ema export)

Output keys are exactly the bare module paths `mlx-gen-pid`'s `PidNet::from_weights(prefix="")`
expects (`patch_blocks.*`, `pixel_blocks.*`, `lq_proj.*`, …). Dtype is preserved (bf16).

Usage:
  python tools/convert_pid.py <input.pth> <output.safetensors>
  python tools/convert_pid.py <input.pth> --dry-run        # report keys, write nothing
  python tools/convert_pid.py --selftest                   # unit-test the key transform (no torch.load)

Run from the reference venv: /Users/michael/Repos/mlx-gen/_vendor/pid/.venv-pid/bin/python
"""

import argparse
import sys


def transform_keys(state_dict):
    """Apply the `load_state_dict` filter; return (kept: dict, dropped: list[str])."""
    kept, dropped = {}, []
    for k, v in state_dict.items():
        if k.startswith("net.") and not k.startswith("net_ema."):
            kept[k[len("net."):]] = v
        elif k.startswith("net_ema.") or k.startswith("fake_score.") or k.startswith("discriminator."):
            dropped.append(k)
        else:
            kept[k] = v
    return kept, dropped


def _unwrap(obj):
    """Find the tensor state_dict inside a torch checkpoint (handles common nesting wrappers)."""
    import torch

    if isinstance(obj, dict):
        # already a flat tensor dict?
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        for key in ("state_dict", "model", "ema", "module", "net"):
            if key in obj and isinstance(obj[key], dict):
                inner = obj[key]
                if inner and all(isinstance(v, torch.Tensor) for v in inner.values()):
                    return inner
        # fall back: the largest sub-dict of tensors
        best = None
        for v in obj.values():
            if isinstance(v, dict) and v and all(isinstance(t, torch.Tensor) for t in v.values()):
                if best is None or len(v) > len(best):
                    best = v
        if best is not None:
            return best
    raise SystemExit("could not locate a tensor state_dict in the checkpoint; inspect its structure")


def selftest():
    sd = {
        "net.patch_blocks.0.norm_x1.weight": 1,
        "net.lq_proj.output_heads.0.weight": 2,
        "net_ema.patch_blocks.0.norm_x1.weight": 3,
        "fake_score.foo": 4,
        "discriminator.bar": 5,
        "final_layer.linear.weight": 6,  # already-bare key
    }
    kept, dropped = transform_keys(sd)
    assert set(kept) == {
        "patch_blocks.0.norm_x1.weight",
        "lq_proj.output_heads.0.weight",
        "final_layer.linear.weight",
    }, kept
    assert set(dropped) == {
        "net_ema.patch_blocks.0.norm_x1.weight",
        "fake_score.foo",
        "discriminator.bar",
    }, dropped
    print("selftest OK — key transform matches load_state_dict (kept 3, dropped 3)")


def main():
    ap = argparse.ArgumentParser(description="Convert a PiD .pth student checkpoint to safetensors.")
    ap.add_argument("input", nargs="?", help="path to model_ema_bf16.pth")
    ap.add_argument("output", nargs="?", help="path to write the .safetensors")
    ap.add_argument("--dry-run", action="store_true", help="report keys but write nothing")
    ap.add_argument("--selftest", action="store_true", help="unit-test the key transform and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.input:
        ap.error("input checkpoint required (or pass --selftest)")

    import torch
    from safetensors.torch import save_file

    obj = torch.load(args.input, map_location="cpu", weights_only=False)
    sd = _unwrap(obj)
    kept, dropped = transform_keys(sd)

    dtypes = sorted({str(v.dtype) for v in kept.values()})
    print(f"input keys: {len(sd)}  ->  kept: {len(kept)}  dropped: {len(dropped)}  dtypes: {dtypes}")
    for sample in list(kept)[:8]:
        print(f"    {sample}  {tuple(kept[sample].shape)}")

    if args.dry_run:
        print("dry-run: no output written")
        return
    if not args.output:
        ap.error("output path required (or pass --dry-run)")

    # safetensors needs contiguous, non-aliased tensors.
    tensors = {k: v.contiguous().clone() for k, v in kept.items()}
    save_file(tensors, args.output)
    print(f"wrote {args.output}  ({len(tensors)} tensors)")


if __name__ == "__main__":
    sys.exit(main())
