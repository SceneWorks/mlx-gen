"""sc-3192: real-weight (35GB) **fast 8-step distill** reference dump for the cross-build e2e test.

Same flow as `dump_sensenova_t2i_realweight.py`, but first merges the 8-step distill LoRA into the
checkpoint (`utils/lora.py::load_and_merge_lora_weight_from_safetensors`, exactly as
`examples/t2i/inference.py --lora_path`) and runs the distilled recipe: `cfg_scale=1.0`,
`num_steps=8`, `timestep_shift=3.0` (`docs/base_vs_distill.md`). Injects a fixed noise so the MLX
port reproduces the same input, and dumps prompt ids + raw noise + reference final image for
`tests/fast_realweight.rs`.

Run (vendored reference env):
  cd _vendor/sensenova_u1 && PYTHONPATH=src .venv/bin/python \
      /abs/path/to/tools/dump_sensenova_fast_realweight.py
Override the LoRA path with SENSENOVA_DISTILL_LORA.
Fixture → mlx-gen-sensenova/tests/fixtures/fast_realweight_golden.safetensors  (gitignored — large)
"""

from __future__ import annotations

import glob
import os
import sys

import torch
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from _sensenova_common import load_model_and_tokenizer, lora_glob
from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN
from sensenova_u1.utils.lora import load_and_merge_lora_weight_from_safetensors

LORA_FILE = "SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors"
PROMPT = "a red fox sitting in a snowy forest, soft morning light"
W, H = 256, 256
NUM_STEPS = 8
TIMESTEP_SHIFT = 3.0
SEED = 1234


def _resolve_lora() -> str:
    if "SENSENOVA_DISTILL_LORA" in os.environ:
        return os.environ["SENSENOVA_DISTILL_LORA"]
    pat = lora_glob(LORA_FILE)
    hits = glob.glob(pat)
    if not hits:
        raise FileNotFoundError(f"distill LoRA not found; download it or set SENSENOVA_DISTILL_LORA ({pat})")
    return hits[0]


@torch.no_grad()
def main() -> None:
    lora_path = _resolve_lora()
    model, tok, device = load_model_and_tokenizer(dtype=torch.bfloat16)
    print(f"merging distill LoRA {lora_path}", flush=True)
    model = load_and_merge_lora_weight_from_safetensors(model, lora_path)
    model.config.t_eps = 0.02

    merge = int(1 / model.downsample_ratio)
    ps = model.patch_size
    cell = ps * merge
    token_h, token_w = H // cell, W // cell
    grid_h, grid_w = H // ps, W // ps

    think_content = "<think>\n\n</think>\n\n<img>"
    query = model._build_t2i_query(PROMPT, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text=think_content)
    input_ids, indexes, attn_mask = model._build_t2i_text_inputs(tok, query)
    pkv, _ = model._t2i_prefix_forward(input_ids, indexes, attn_mask)
    text_len = indexes.shape[1]
    indexes_image = model._build_t2i_image_indexes(token_h, token_w, text_len, device=input_ids.device)
    gen_grid_hw = torch.tensor([[grid_h, grid_w]], device=input_ids.device)

    base = float(model.noise_scale_base_image_seq_len)
    noise_scale = (((grid_h * grid_w) / (merge ** 2) / base) ** 0.5) * float(model.noise_scale)
    noise_scale = min(noise_scale, model.noise_scale_max_value)

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    raw_noise = torch.randn(1, 3, H, W, generator=gen, dtype=torch.float32)
    image_prediction = (noise_scale * raw_noise).to(device=input_ids.device, dtype=torch.bfloat16)

    timesteps = torch.linspace(0.0, 1.0, NUM_STEPS + 1, device=input_ids.device)
    timesteps = model._apply_time_schedule(timesteps, token_h * token_w, TIMESTEP_SHIFT)
    attn_cond = {"full_attention": None}

    step_images = []
    for step_i in range(NUM_STEPS):
        t = timesteps[step_i]
        t_next = timesteps[step_i + 1]
        z = model.patchify(image_prediction, cell)
        image_input = model.patchify(image_prediction, ps, channel_first=True)
        image_embeds = model.extract_feature(
            image_input.view(grid_h * grid_w, -1), gen_model=True, grid_hw=gen_grid_hw
        ).view(1, token_h * token_w, -1)
        t_expanded = t.expand(token_h * token_w)
        te = model.fm_modules["timestep_embedder"](t_expanded).view(1, token_h * token_w, -1)
        nst = torch.full_like(t_expanded, noise_scale / model.noise_scale_max_value)
        te = te + model.fm_modules["noise_scale_embedder"](nst).view(1, token_h * token_w, -1)
        image_embeds = image_embeds + te
        v_pred = model._t2i_predict_v(
            image_embeds, indexes_image, attn_cond, pkv, t, z,
            image_token_num=token_h * token_w, timestep_embeddings=te, image_size=(W, H),
        )
        z = z + (t_next - t) * v_pred
        image_prediction = model.unpatchify(z, cell, H, W)
        step_images.append(image_prediction.to("cpu").to(torch.float32))
        print(f"  step {step_i} done", flush=True)

    out = {
        "input_ids": input_ids.to("cpu").to(torch.int32),
        "raw_noise": raw_noise.to("cpu").to(torch.float32),
        "final_image": image_prediction.to("cpu").to(torch.float32),
    }
    # Per-step trajectory (image after each euler step) for the step-by-step parity diagnostic.
    for i, im in enumerate(step_images):
        out[f"step.{i}"] = im
    dst = os.path.join(
        REPO_ROOT, "mlx-gen-sensenova", "tests", "fixtures", "fast_realweight_golden.safetensors"
    )
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    meta = {
        "prompt": PROMPT,
        "width": str(W),
        "height": str(H),
        "num_steps": str(NUM_STEPS),
        "timestep_shift": str(TIMESTEP_SHIFT),
        "seed": str(SEED),
        "noise_scale": repr(noise_scale),
        "n_tokens": str(int(input_ids.shape[1])),
        "lora": os.path.basename(lora_path),
    }
    save_file(out, dst, metadata=meta)
    print(f"wrote {dst}")
    img = out["final_image"]
    print(f"  final image: shape {tuple(img.shape)}  min {img.min():.3f} max {img.max():.3f} mean {img.mean():.3f}")


if __name__ == "__main__":
    sys.exit(main())
