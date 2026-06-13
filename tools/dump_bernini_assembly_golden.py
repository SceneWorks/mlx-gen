"""sc-5140: golden for the planner-input assembly glue (format_mllm_inputs_embeds + T5 concat).

  - **format_mllm_inputs_embeds** (`bernini.py`): `embed_tokens(input_ids)` then `masked_scatter` the
    ViT visual features into the `visual_input_mask | visual_output_mask` slots.
  - **concat_with_zero_init** (`pipeline.__call__`): prepend the T5 prompt embeds to a planner stream,
    then zero-pad / truncate to `max_sequence_length` (both branches exercised).

The reference ops (`nn.Embedding`-equivalent gather, `Tensor.masked_scatter`, `torch.cat` +
`feat.new_zeros` pad / `feat[:, :max]` truncate) are run verbatim; these are exact integer/host ops, so
the Rust port must match bit-for-bit.

Run:
  ~/Repos/mflux/.venv/bin/python tools/dump_bernini_assembly_golden.py
Fixture -> mlx-gen-bernini/tests/fixtures/assembly_golden.safetensors
"""

from __future__ import annotations

import os

import torch
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "mlx-gen-bernini", "tests", "fixtures", "assembly_golden.safetensors")

VOCAB = 20
HIDDEN = 16   # planner hidden
WIDTH = 12    # renderer prompt-embed width
MAX_SEQ = 10


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)

    # ---- format_mllm_inputs_embeds ----
    embed_table = torch.randn(VOCAB, HIDDEN)
    # L=9 tokens: input-ViT block [2,3], a few text tokens, gen-output ViT block [6,7,8].
    input_ids = torch.tensor([5, 1, 9, 9, 3, 7, 0, 0, 0], dtype=torch.long)
    visual_input_mask = torch.tensor([0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool)   # 2 input-ViT slots
    visual_output_mask = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.bool)  # 3 gen-ViT slots
    n_visual = int((visual_input_mask | visual_output_mask).sum())
    visual_embeds = torch.randn(n_visual, HIDDEN)

    inputs_embeds = torch.nn.functional.embedding(input_ids.unsqueeze(0), embed_table)  # [1, L, H]
    vmask = (visual_input_mask | visual_output_mask).unsqueeze(0)  # [1, L]
    vmask_exp = vmask.unsqueeze(-1).expand_as(inputs_embeds)
    scattered = inputs_embeds.masked_scatter(vmask_exp, visual_embeds.to(inputs_embeds.dtype))

    # ---- concat_with_zero_init (pad + truncate branches) ----
    t5 = torch.randn(1, 4, WIDTH)
    stream_short = torch.randn(1, 3, WIDTH)   # 4 + 3 = 7 < MAX_SEQ -> pad to 10
    stream_long = torch.randn(1, 9, WIDTH)    # 4 + 9 = 13 > MAX_SEQ -> truncate to 10

    def pad_and_truncate(feat, max_seq=MAX_SEQ):
        if feat.shape[1] < max_seq:
            feat = torch.cat([feat, feat.new_zeros((1, max_seq - feat.shape[1], feat.shape[-1]))], dim=1)
        if feat.shape[1] > max_seq:
            feat = feat[:, :max_seq, :]
        return feat

    concat_pad = pad_and_truncate(torch.cat([t5, stream_short], dim=1))
    concat_trunc = pad_and_truncate(torch.cat([t5, stream_long], dim=1))

    out = {
        "model.embed_tokens.weight": embed_table.contiguous(),
        "model.norm.weight": torch.ones(HIDDEN).contiguous(),
        "io.input_ids": input_ids.to(torch.int32).contiguous(),
        "io.visual_input_mask": visual_input_mask.to(torch.int32).contiguous(),
        "io.visual_output_mask": visual_output_mask.to(torch.int32).contiguous(),
        "io.visual_embeds": visual_embeds.contiguous(),
        "out.format_mllm": scattered.contiguous(),
        "io.t5": t5.contiguous(),
        "io.stream_short": stream_short.contiguous(),
        "io.stream_long": stream_long.contiguous(),
        "out.concat_pad": concat_pad.contiguous(),
        "out.concat_trunc": concat_trunc.contiguous(),
    }
    meta = {"vocab": str(VOCAB), "hidden": str(HIDDEN), "width": str(WIDTH), "max_seq": str(MAX_SEQ)}
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    save_file(out, FIXTURE, metadata=meta)
    print(f"wrote {FIXTURE}  ({len(out)} tensors)")
    print(f"  format_mllm {tuple(scattered.shape)}  concat_pad {tuple(concat_pad.shape)}  "
          f"concat_trunc {tuple(concat_trunc.shape)}")


if __name__ == "__main__":
    main()
