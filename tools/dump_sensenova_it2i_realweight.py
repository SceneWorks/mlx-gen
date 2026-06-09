"""sc-3189: real-weight (35GB) it2i (edit) reference dump for the cross-build e2e parity test.

Loads the actual checkpoint + tokenizer and runs the genuine `it2i_generate` **edit** flow
(`cfg_scale=4`, `img_cfg_scale=1.0` → condition + image-condition caches) for a fixed prompt and a
deterministic source image, with an **injected fixed noise**. The source `pixel_values` are computed
with the same ImageNet-normalize + channel-first patchify the MLX port's `preprocess_image` uses, so
the two match at the vision input. Dumps the source image, noise, prompt token ids, the reference
`pixel_values` (to check the port's preprocess), and the final image.

Run (vendored reference env):
  cd _vendor/sensenova_u1 && PYTHONPATH=src .venv/bin/python \
      /abs/path/to/tools/dump_sensenova_it2i_realweight.py
Fixture → mlx-gen-sensenova/tests/fixtures/it2i_realweight_golden.safetensors  (gitignored — large)
"""

from __future__ import annotations

import math
import os
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel
from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN

SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--sensenova--SenseNova-U1-8B-MoT/snapshots/"
    "bfa9b436503cb8aed4f2bc60e3236710cc77468d"
)
PROMPT = "make the sky a vivid purple"
W, H = 256, 256
SRC_W, SRC_H = 256, 256
NUM_STEPS = 8
SEED = 7
CFG, IMG_CFG = 2.0, 1.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess(src: torch.Tensor, ps: int):
    """ImageNet-normalize [3,H,W] in [0,1] + channel-first patchify → ([gh*gw, 3*ps^2], grid_hw)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    norm = (src - mean) / std
    c, h, w = norm.shape
    gh, gw = h // ps, w // ps
    patches = (
        norm.view(c, gh, ps, gw, ps).permute(1, 3, 0, 2, 4).reshape(gh * gw, c * ps * ps)
    )
    return patches, torch.tensor([[gh, gw]])


@torch.no_grad()
def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {SNAP} on {device} (f32)…", flush=True)
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    model = NEOChatModel.from_pretrained(SNAP, torch_dtype=torch.float32, trust_remote_code=True).to(device).eval()
    model.config.t_eps = 0.02
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.img_start_token_id = tok.convert_tokens_to_ids("<img>")

    ps, merge = model.patch_size, int(1 / model.downsample_ratio)
    cell = ps * merge
    token_h, token_w = H // cell, W // cell
    grid_h, grid_w = H // ps, W // ps

    # Deterministic source image in [0,1], already bucket-sized (256 = min_pixels, no resize).
    g = torch.Generator().manual_seed(SEED)
    src = torch.rand(3, SRC_H, SRC_W, generator=g, dtype=torch.float32)
    pixel_values_f32, src_grid_hw = preprocess(src, ps)
    pixel_values = pixel_values_f32.to(device).to(torch.float32)
    src_grid_hw = src_grid_hw.to(device)

    # Build cond (prompt+image) and img-cond (image-only) prefixes — the edit flow.
    question = "<image>\n" + PROMPT  # auto-prepend (one image, no marker in prompt)
    think_content = "<think>\n\n</think>\n\n<img>"
    cond_query = model._build_t2i_query(question, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text=think_content)
    img_query = model._build_t2i_query("<image>", append_text="<img>")
    num_patch = int(src_grid_hw[0, 0] * src_grid_hw[0, 1] * model.downsample_ratio ** 2)
    block = "<img>" + "<IMG_CONTEXT>" * num_patch + "</img>"
    cond_query = cond_query.replace("<image>", block, 1)
    img_query = img_query.replace("<image>", block, 1)

    und_vit = model.extract_feature(pixel_values, grid_hw=src_grid_hw)  # [n_ctx, H] understanding
    ic, idx_c, am_c = model._build_it2i_inputs(tok, cond_query, pixel_values, src_grid_hw)
    ii, idx_i, am_i = model._build_it2i_inputs(tok, img_query, pixel_values, src_grid_hw)
    pkv_c, cond_prefill_hidden = model._it2i_prefix_forward(ic, idx_c, am_c)
    pkv_i, _ = model._it2i_prefix_forward(ii, idx_i, am_i)
    idx_img_c = model._build_t2i_image_indexes(token_h, token_w, idx_c[0].max() + 1, device=ic.device)
    idx_img_i = model._build_t2i_image_indexes(token_h, token_w, idx_i[0].max() + 1, device=ii.device)

    base = float(model.noise_scale_base_image_seq_len)
    noise_scale = min(math.sqrt((grid_h * grid_w) / (merge ** 2) / base) * float(model.noise_scale), model.noise_scale_max_value)
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    raw_noise = torch.randn(1, 3, H, W, generator=gen, dtype=torch.float32)
    image_prediction = (noise_scale * raw_noise).to(device=ic.device, dtype=torch.float32)

    timesteps = model._apply_time_schedule(torch.linspace(0.0, 1.0, NUM_STEPS + 1, device=ic.device), token_h * token_w, 1.0)
    am = {"full_attention": None}
    for step_i in range(NUM_STEPS):
        t, t_next = timesteps[step_i], timesteps[step_i + 1]
        z = model.patchify(image_prediction, cell)
        image_input = model.patchify(image_prediction, ps, channel_first=True)
        image_embeds = model.extract_feature(image_input.view(grid_h * grid_w, -1), gen_model=True, grid_hw=torch.tensor([[grid_h, grid_w]], device=ic.device)).view(1, token_h * token_w, -1)
        t_exp = t.expand(token_h * token_w)
        te = model.fm_modules["timestep_embedder"](t_exp).view(1, token_h * token_w, -1)
        nst = torch.full_like(t_exp, noise_scale / model.noise_scale_max_value)
        te = te + model.fm_modules["noise_scale_embedder"](nst).view(1, token_h * token_w, -1)
        image_embeds = image_embeds + te
        out_cond = model._t2i_predict_v(image_embeds, idx_img_c, am, pkv_c, t, z, image_token_num=token_h * token_w, timestep_embeddings=te, image_size=(W, H))
        out_img = model._t2i_predict_v(image_embeds, idx_img_i, am, pkv_i, t, z, image_token_num=token_h * token_w, timestep_embeddings=te, image_size=(W, H))
        v_pred = out_img + CFG * (out_cond - out_img)
        z = z + (t_next - t) * v_pred
        image_prediction = model.unpatchify(z, cell, H, W)
        print(f"  step {step_i} done", flush=True)

    # Cond-only pass (cfg=1: v = out_cond) to isolate the condition path from the CFG blend.
    image_cond_only = (noise_scale * raw_noise).to(device=ic.device, dtype=torch.float32)
    cond_only_traj = []
    for step_i in range(NUM_STEPS):
        t, t_next = timesteps[step_i], timesteps[step_i + 1]
        z = model.patchify(image_cond_only, cell)
        image_input = model.patchify(image_cond_only, ps, channel_first=True)
        image_embeds = model.extract_feature(image_input.view(grid_h * grid_w, -1), gen_model=True, grid_hw=torch.tensor([[grid_h, grid_w]], device=ic.device)).view(1, token_h * token_w, -1)
        t_exp = t.expand(token_h * token_w)
        te = model.fm_modules["timestep_embedder"](t_exp).view(1, token_h * token_w, -1)
        nst = torch.full_like(t_exp, noise_scale / model.noise_scale_max_value)
        te = te + model.fm_modules["noise_scale_embedder"](nst).view(1, token_h * token_w, -1)
        image_embeds = image_embeds + te
        v = model._t2i_predict_v(image_embeds, idx_img_c, am, pkv_c, t, z, image_token_num=token_h * token_w, timestep_embeddings=te, image_size=(W, H))
        z = z + (t_next - t) * v
        image_cond_only = model.unpatchify(z, cell, H, W)
        cond_only_traj.append(image_cond_only.to("cpu").to(torch.float32).clone())

    cond_ids = tok(cond_query, return_tensors="pt")["input_ids"]
    out = {
        "src": src.to("cpu"),
        "pixel_values": pixel_values_f32.to("cpu"),
        "cond_input_ids": cond_ids.to("cpu").to(torch.int32),
        "raw_noise": raw_noise.to("cpu"),
        "final_image": image_prediction.to("cpu").to(torch.float32),
        "final_cond_only": image_cond_only.to("cpu").to(torch.float32),
        "cond_only_traj": torch.cat(cond_only_traj, dim=0),
        "und_vit": und_vit.to("cpu").to(torch.float32),
        "cond_prefill_hidden": cond_prefill_hidden.to("cpu").to(torch.float32),
    }
    dst = os.path.join(REPO_ROOT, "mlx-gen-sensenova", "tests", "fixtures", "it2i_realweight_golden.safetensors")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    meta = {"prompt": PROMPT, "width": str(W), "height": str(H), "num_steps": str(NUM_STEPS), "cfg": repr(CFG), "img_cfg": repr(IMG_CFG), "src_w": str(SRC_W), "src_h": str(SRC_H)}
    save_file(out, dst, metadata=meta)
    img = out["final_image"]
    print(f"wrote {dst}")
    print(f"  final image: shape {tuple(img.shape)} min {img.min():.3f} max {img.max():.3f} mean {img.mean():.3f}")


if __name__ == "__main__":
    sys.exit(main())
