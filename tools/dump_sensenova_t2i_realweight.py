"""sc-3188: real-weight (35GB) T2I reference dump for the cross-build e2e parity test.

Loads the actual `sensenova/SenseNova-U1-8B-MoT` checkpoint + tokenizer and runs the genuine
`t2i_generate` flow (non-think, `cfg_scale=1`) for a fixed prompt at a small resolution / few steps,
but with an **injected fixed noise** (so the MLX port can reproduce it bit-for-bit at the noise
input). Dumps the exact prompt token ids, the raw noise, and the reference final image so
`tests/t2i_realweight.rs` (an `#[ignore]` test, run locally) can load the same 33GB into MLX, run
`T2iModel::generate(..., init_noise=Some(noise))`, and check directional/structural similarity +
coherence.

Loads in bf16 (the on-disk dtype the MLX port runs) for a fair cross-build comparison. Device order:
MPS → CPU.

Run (vendored reference env):
  cd _vendor/sensenova_u1 && PYTHONPATH=src .venv/bin/python \
      /abs/path/to/tools/dump_sensenova_t2i_realweight.py
Fixture → mlx-gen-sensenova/tests/fixtures/t2i_realweight_golden.safetensors  (gitignored — large)
"""

from __future__ import annotations

import os
import sys

import torch
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from _sensenova_common import load_model_and_tokenizer
from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN

PROMPT = "a red fox sitting in a snowy forest, soft morning light"
W, H = 256, 256
NUM_STEPS = 8
SEED = 1234


@torch.no_grad()
def main() -> None:
    model, tok, device = load_model_and_tokenizer(dtype=torch.bfloat16)
    model.config.t_eps = 0.02

    merge = int(1 / model.downsample_ratio)
    ps = model.patch_size
    cell = ps * merge
    token_h, token_w = H // cell, W // cell
    grid_h, grid_w = H // ps, W // ps

    # Build the condition prefix exactly as t2i_generate (non-think).
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
    timesteps = model._apply_time_schedule(timesteps, token_h * token_w, 1.0)
    attn_cond = {"full_attention": None}

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
        print(f"  step {step_i} done", flush=True)

    out = {
        "input_ids": input_ids.to("cpu").to(torch.int32),
        "raw_noise": raw_noise.to("cpu").to(torch.float32),
        "final_image": image_prediction.to("cpu").to(torch.float32),
    }
    dst = os.path.join(
        REPO_ROOT, "mlx-gen-sensenova", "tests", "fixtures", "t2i_realweight_golden.safetensors"
    )
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    meta = {
        "prompt": PROMPT,
        "width": str(W),
        "height": str(H),
        "num_steps": str(NUM_STEPS),
        "seed": str(SEED),
        "noise_scale": repr(noise_scale),
        "n_tokens": str(int(input_ids.shape[1])),
    }
    save_file(out, dst, metadata=meta)
    print(f"wrote {dst}")
    print(f"  prompt={PROMPT!r}  tokens={int(input_ids.shape[1])}  noise_scale={noise_scale:.4f}")
    img = out["final_image"]
    print(f"  final image: shape {tuple(img.shape)}  min {img.min():.3f} max {img.max():.3f} mean {img.mean():.3f}")


if __name__ == "__main__":
    sys.exit(main())
