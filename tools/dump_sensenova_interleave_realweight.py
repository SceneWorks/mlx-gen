"""sc-3190: real-weight (35GB) interleaved-generation (Document Studio) reference dump.

Runs the genuine `interleave_gen` (think-mode) on the actual checkpoint for a fixed prompt, wrapping
`torch.randn` to capture the per-image noise so the MLX port can reproduce each image. cfg_scale=2
(a stable regime — high cfg is precision-chaotic, see sc-3189). Dumps the composed text, the
generated images, and the captured per-image noise; the Rust `#[ignore]` test runs
`T2iModel::interleave_gen(init_noises=…)` and compares the deterministic text prefix + structure +
first-image similarity.

Run (vendored reference env):
  cd _vendor/sensenova_u1 && PYTHONPATH=src .venv/bin/python \
      /abs/path/to/tools/dump_sensenova_interleave_realweight.py
Fixture → mlx-gen-sensenova/tests/fixtures/interleave_realweight_golden.safetensors  (gitignored)
"""

from __future__ import annotations

import os
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel

SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--sensenova--SenseNova-U1-8B-MoT/snapshots/"
    "bfa9b436503cb8aed4f2bc60e3236710cc77468d"
)
PROMPT = "Show me a simple red circle, then briefly describe it."
W, H = 256, 256
NUM_STEPS = 8
CFG, IMG_CFG, TS = 2.0, 1.0, 3.0
MAX_NEW = 96
SEED = 0

DEFAULT_SYSTEM_MESSAGE = (
    "You are a multimodal assistant capable of reasoning with both text and images. You support "
    "two modes:\n\nThink Mode: When reasoning is needed, you MUST start with a <think></think> "
    "block and place all reasoning inside it. You MUST interleave text with generated images using "
    "tags like <image1>, <image2>. Images can ONLY be generated between <think> and </think>, and "
    "may be referenced in the final answer.\n\nNon-Think Mode: When no reasoning is needed, directly "
    "provide the answer without reasoning. Do not use tags like <image1>, <image2>; present any "
    "images naturally alongside the text.\n\nAfter the think block, always provide a concise, "
    "user-facing final answer. The answer may include text, images, or both. Match the user's "
    "language in both reasoning and the final answer."
)


@torch.no_grad()
def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {SNAP} on {device} (bf16)…", flush=True)
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    model = NEOChatModel.from_pretrained(SNAP, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()

    # Capture per-image noise: the only [1,3,H,W] randn in interleave_gen is the image init.
    orig_randn = torch.randn
    captured = []

    def patched_randn(*a, **k):
        out = orig_randn(*a, **k)
        if out.dim() == 4 and out.shape[1] == 3:
            captured.append(out.detach().to("cpu").to(torch.float32).clone())
        return out

    torch.randn = patched_randn
    try:
        text, image_tensors = model.interleave_gen(
            tok, PROMPT, images=[], image_size=(W, H), cfg_scale=CFG, img_cfg_scale=IMG_CFG,
            timestep_shift=TS, cfg_interval=(0.0, 1.0), num_steps=NUM_STEPS,
            system_message=DEFAULT_SYSTEM_MESSAGE, think_mode=True, seed=SEED, verbose=False,
        )
    finally:
        torch.randn = orig_randn

    print(f"  generated {len(image_tensors)} image(s); text len {len(text)}", flush=True)
    o = {}
    for i, img in enumerate(image_tensors):
        o[f"image_{i}"] = img.to("cpu").to(torch.float32)
    for i, n in enumerate(captured[: len(image_tensors)]):
        o[f"noise_{i}"] = n
    if not image_tensors:
        # Still dump something so the fixture is valid.
        o["image_0"] = torch.zeros(1, 3, H, W)
        o["noise_0"] = torch.zeros(1, 3, H, W)

    dst = os.path.join(REPO_ROOT, "mlx-gen-sensenova", "tests", "fixtures", "interleave_realweight_golden.safetensors")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    meta = {
        "prompt": PROMPT, "width": str(W), "height": str(H), "num_steps": str(NUM_STEPS),
        "cfg": repr(CFG), "img_cfg": repr(IMG_CFG), "timestep_shift": repr(TS),
        "max_new": str(MAX_NEW), "seed": str(SEED), "n_images": str(len(image_tensors)),
        "text": text,
    }
    save_file(o, dst, metadata=meta)
    print(f"wrote {dst}")
    print(f"  text={text!r}")


if __name__ == "__main__":
    sys.exit(main())
