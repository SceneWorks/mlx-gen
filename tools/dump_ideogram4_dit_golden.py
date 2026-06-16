"""sc-5986 — golden dump for the Ideogram 4 DiT parity test.

Runs the upstream torch `Ideogram4Transformer` (from the `ideogram4` reference package) in f32 on
the converted bf16 transformer weights, over a small synthetic packed `[text ; image]` sequence,
and saves all inputs + the velocity output. The Rust parity test (`tests/dit_parity.rs`) feeds the
same tensors. f32-on-both-sides isolates the forward graph (MRoPE, AdaLN sandwich norms, segment
mask, token composition) from bf16 runtime numerics (validated later at the smoke).

The reference package is NOT vendored (Ideogram non-commercial license). Fetch it first:
  mkdir -p /tmp/ideogram4-ref/ideogram4 && cd /tmp/ideogram4-ref/ideogram4 && touch __init__.py
  for f in modeling_ideogram4 constants; do
    gh api repos/ideogram-oss/ideogram4/contents/src/ideogram4/$f.py --jq .content | base64 -d > $f.py
  done

Run:
  ~/mlx-flux-venv/bin/python tools/dump_ideogram4_dit_golden.py \
      --ref-dir /tmp/ideogram4-ref --converted ~/.cache/ideogram4-mlx-convert \
      --out tools/golden/ideogram4_dit.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import torch
from safetensors.torch import load_file

OFF = 65536  # IMAGE_POSITION_OFFSET
LLM, IMG = 3, 2  # LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", type=Path, default=Path("/tmp/ideogram4-ref"))
    ap.add_argument("--converted", type=Path, default=Path.home() / ".cache/ideogram4-mlx-convert")
    ap.add_argument("--component", default="transformer", choices=["transformer", "unconditional_transformer"])
    ap.add_argument("--out", type=Path, default=Path("tools/golden/ideogram4_dit.safetensors"))
    args = ap.parse_args()

    sys.path.insert(0, str(args.ref_dir))
    from ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer

    cfg = Ideogram4Config()
    sd_path = args.converted / args.component / "model.safetensors"
    if not sd_path.exists():
        sys.exit(f"converted weights not found: {sd_path}")

    print(f"building f32 Ideogram4Transformer, loading {sd_path} …")
    model = Ideogram4Transformer(cfg)  # f32 params
    sd = load_file(str(sd_path))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # rotary_emb.inv_freq is a non-persistent buffer (recomputed in __init__) → expected "missing".
    missing = [k for k in missing if not k.endswith("inv_freq")]
    if missing or unexpected:
        sys.exit(f"state_dict mismatch  missing={missing}  unexpected={unexpected}")
    model.eval()

    # Synthetic packing: 5 text tokens + a 2×2 image grid (4 tokens) = L 9, single sample.
    torch.manual_seed(0)
    num_text, grid_h, grid_w = 5, 2, 2
    num_img = grid_h * grid_w
    seq = num_text + num_img

    llm_features = torch.randn(1, seq, cfg.llm_features_dim)
    x = torch.randn(1, seq, cfg.in_channels)
    t = torch.tensor([0.3])

    position_ids = torch.zeros(1, seq, 3, dtype=torch.long)
    for i in range(num_text):
        position_ids[0, i] = torch.tensor([i, i, i])
    for j in range(num_img):
        h, w = j // grid_w, j % grid_w
        position_ids[0, num_text + j] = torch.tensor([0, h + OFF, w + OFF])

    segment_ids = torch.ones(1, seq, dtype=torch.long)
    indicator = torch.tensor([[LLM] * num_text + [IMG] * num_img], dtype=torch.long)

    with torch.no_grad():
        out = model(
            llm_features=llm_features,
            x=x,
            t=t,
            position_ids=position_ids,
            segment_ids=segment_ids,
            indicator=indicator,
        )
    print(f"output: {tuple(out.shape)} {out.dtype}  (expect [1,{seq},{cfg.in_channels}])")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.out),
        {
            "llm_features": mx.array(llm_features.numpy()),
            "x": mx.array(x.numpy()),
            "t": mx.array(t.numpy()),
            "position_ids": mx.array(position_ids.to(torch.int32).numpy()),
            "segment_ids": mx.array(segment_ids.to(torch.int32).numpy()),
            "indicator": mx.array(indicator.to(torch.int32).numpy()),
            "golden": mx.array(out.to(torch.float32).numpy()),
        },
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
