"""sc-3191: real-weight (35GB) VQA reference dump for the cross-build parity test.

Loads the actual checkpoint + tokenizer and runs the genuine `chat`/`generate` understanding flow
(greedy, `do_sample=False`) for a fixed image + question, dumping the source image, the prompt token
ids, and the greedy answer token stream + decoded text. The source `pixel_values` use the same
ImageNet-normalize + channel-first patchify the MLX `preprocess_image` uses (256×256, no resize).
The Rust `#[ignore]` test runs `T2iModel::vqa` and compares.

Run (vendored reference env):
  cd _vendor/sensenova_u1 && PYTHONPATH=src .venv/bin/python \
      /abs/path/to/tools/dump_sensenova_vqa_realweight.py
Fixture → mlx-gen-sensenova/tests/fixtures/vqa_realweight_golden.safetensors  (gitignored — large)
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
QUESTION = "What is the dominant color in this image? Answer in one word."
SRC_W, SRC_H = 256, 256
MAX_NEW = 24
SEED = 3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess(src: torch.Tensor, ps: int):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    norm = (src - mean) / std
    c, h, w = norm.shape
    gh, gw = h // ps, w // ps
    patches = norm.view(c, gh, ps, gw, ps).permute(1, 3, 0, 2, 4).reshape(gh * gw, c * ps * ps)
    return patches, torch.tensor([[gh, gw]])


@torch.no_grad()
def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {SNAP} on {device} (bf16)…", flush=True)
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    model = NEOChatModel.from_pretrained(SNAP, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.img_start_token_id = tok.convert_tokens_to_ids("<img>")
    ps = model.patch_size

    g = torch.Generator().manual_seed(SEED)
    src = torch.rand(3, SRC_H, SRC_W, generator=g, dtype=torch.float32)
    pixel_values_f32, grid_hw = preprocess(src, ps)
    pixel_values = pixel_values_f32.to(device).to(torch.bfloat16)
    grid_hw = grid_hw.to(device)

    # Build the chat query exactly (empty system message; one image auto-prepended).
    from sensenova_u1.models.neo_unify.conversation import get_conv_template
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    question = "<image>\n" + QUESTION
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    eos_id = tok.convert_tokens_to_ids(template.sep.strip())
    num_patch = int(grid_hw[0, 0] * grid_hw[0, 1] * model.downsample_ratio ** 2)
    query = query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * num_patch + "</img>", 1)

    model_inputs = tok(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    gen_out = model.generate(
        pixel_values=pixel_values, input_ids=input_ids, grid_hw=grid_hw,
        attention_mask=attention_mask, do_sample=False, max_new_tokens=MAX_NEW, eos_token_id=eos_id,
    )
    answer_ids = gen_out[0].to("cpu").to(torch.int32)  # generated tokens (no prompt)
    answer = tok.decode(answer_ids.tolist(), skip_special_tokens=True).strip()

    o = {
        "src": src.to("cpu"),
        "cond_input_ids": input_ids.to("cpu").to(torch.int32),
        "answer_ids": answer_ids,
    }
    dst = os.path.join(REPO_ROOT, "mlx-gen-sensenova", "tests", "fixtures", "vqa_realweight_golden.safetensors")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    meta = {"question": QUESTION, "max_new": str(MAX_NEW), "answer": answer, "eos_id": str(int(eos_id))}
    save_file(o, dst, metadata=meta)
    print(f"wrote {dst}")
    print(f"  question={QUESTION!r}")
    print(f"  answer_ids={answer_ids.tolist()}")
    print(f"  answer={answer!r}")


if __name__ == "__main__":
    sys.exit(main())
