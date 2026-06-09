"""sc-3189: synthetic-fixture golden for the SenseNova-U1 it2i (image-conditioned) spine.

Like the T2I golden but exercises the it2i additions: the **understanding** vision embedder
(`vision_model`), splicing its features into the `<IMG_CONTEXT>` prefix positions, the
`get_thw_indexes` image-block (h,w) positions, and the image-conditioned prefill + denoise. Replays
the genuine reference modules (`extract_feature`/`get_thw_indexes`/`_it2i_prefix_forward`/
`_t2i_predict_v`/`_euler_step`) on a random prefix + source image + fixed noise (cond-only,
cfg_scale=1). The tiny vocab can't hold the real special-token ids, so `img_context_token_id` /
`img_start_token_id` are set to small values (10 / 11); the Rust test mirrors that via
`T2iModel::with_image_token_ids`.

Run (vendored reference env):
  cd _vendor/sensenova_u1 && PYTHONPATH=src .venv/bin/python \
      /abs/path/to/tools/dump_sensenova_it2i_golden.py
Fixture → mlx-gen-sensenova/tests/fixtures/it2i_golden.safetensors
"""

from __future__ import annotations

import os
import sys

import torch
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOChatConfig
from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel
from sensenova_u1.models.neo_unify.modeling_qwen3 import create_block_causal_mask

IMG_CONTEXT_ID = 10
IMG_START_ID = 11
IMG_END_ID = 12


def build_config() -> NEOChatConfig:
    vision_config = dict(
        architectures=["NEOVisionModel"], num_channels=3, patch_size=16, hidden_size=32,
        llm_hidden_size=64, downsample_ratio=0.5, rope_theta_vision=10000.0,
        max_position_embeddings_vision=10000,
    )
    llm_config = dict(
        architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=64,
        intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, rms_norm_eps=1e-6, rope_theta=5_000_000.0, rope_theta_hw=10_000.0,
        max_position_embeddings=4096, max_position_embeddings_hw=4096, vocab_size=64,
        attention_bias=False, hidden_act="silu", use_sliding_window=False, max_window_layers=0,
        tie_word_embeddings=False,
    )
    cfg = NEOChatConfig(
        vision_config=vision_config, llm_config=llm_config, downsample_ratio=0.5, template="neo1_0",
        timestep_shift=1.0, time_schedule="standard", time_shift_type="exponential", base_shift=0.5,
        max_shift=1.15, base_image_seq_len=64, max_image_seq_len=4096, noise_scale_mode="resolution",
        noise_scale=1.0, noise_scale_max_value=8.0, noise_scale_base_image_seq_len=64,
        add_noise_scale_embedding=True, fm_head_dim=1536, fm_head_layers=2, fm_head_mlp_ratio=1,
        use_pixel_head=False, use_adaLN=False, concat_time_token_num=0,
    )
    cfg.llm_config._attn_implementation = "eager"
    return cfg


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    cfg = build_config()
    model = NEOChatModel(cfg).to(torch.float32).eval()
    model.config.t_eps = 0.02
    model.img_context_token_id = IMG_CONTEXT_ID
    model.img_start_token_id = IMG_START_ID

    ps, merge = 16, 2
    cell = ps * merge  # 32
    W, H = 64, 64
    token_h, token_w = H // cell, W // cell
    grid_h, grid_w = H // ps, W // ps
    num_steps = 4

    # ---- Source image (random normalized patches) + understanding features ----
    src_gh, src_gw = 4, 4  # source 64x64 → 16 patches → 4 merged ctx tokens
    n_ctx = (src_gh // merge) * (src_gw // merge)  # 4
    pixel_values = torch.randn(src_gh * src_gw, 3 * ps * ps, dtype=torch.float32)
    src_grid_hw = torch.tensor([[src_gh, src_gw]])

    # ---- Prefix with a source-image block, small ids ----
    # [t, t, <img>, <ctx>*4, </img>, t]  (t = random text in [0,10))
    ids = [3, 5, IMG_START_ID] + [IMG_CONTEXT_ID] * n_ctx + [IMG_END_ID, 7]
    input_ids = torch.tensor([ids], dtype=torch.long)
    indexes = model.get_thw_indexes(input_ids[0], src_grid_hw)
    attn_mask = {"full_attention": create_block_causal_mask(indexes[0])}

    input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
    vit = model.extract_feature(pixel_values, grid_hw=src_grid_hw)  # [n_ctx, H]
    B, N, C = input_embeds.shape
    flat = input_embeds.reshape(B * N, C)
    sel = (input_ids.reshape(B * N) == IMG_CONTEXT_ID)
    flat[sel] = vit.reshape(-1, C).to(flat.dtype)
    input_embeds = flat.reshape(B, N, C)

    pkv, _ = model._it2i_prefix_forward(input_embeds, indexes, attn_mask)
    text_len = int(indexes[0].max().item()) + 1
    indexes_image = model._build_t2i_image_indexes(token_h, token_w, text_len, device="cpu")
    gen_grid_hw = torch.tensor([[grid_h, grid_w]])

    base = float(cfg.noise_scale_base_image_seq_len)
    noise_scale = (((grid_h * grid_w) / (merge ** 2) / base) ** 0.5) * float(cfg.noise_scale)
    noise_scale = min(noise_scale, cfg.noise_scale_max_value)
    raw_noise = torch.randn(1, 3, H, W, dtype=torch.float32)
    image_prediction = noise_scale * raw_noise

    timesteps = model._apply_time_schedule(torch.linspace(0.0, 1.0, num_steps + 1), token_h * token_w, 1.0)
    attn_cond = {"full_attention": None}
    traj = []
    for step_i in range(num_steps):
        t = timesteps[step_i]
        t_next = timesteps[step_i + 1]
        z = model.patchify(image_prediction, cell)
        image_input = model.patchify(image_prediction, ps, channel_first=True)
        image_embeds = model.extract_feature(
            image_input.view(grid_h * grid_w, -1), gen_model=True, grid_hw=gen_grid_hw
        ).view(1, token_h * token_w, -1)
        t_expanded = t.expand(token_h * token_w)
        te = model.fm_modules["timestep_embedder"](t_expanded).view(1, token_h * token_w, -1)
        nst = torch.full_like(t_expanded, noise_scale / cfg.noise_scale_max_value)
        te = te + model.fm_modules["noise_scale_embedder"](nst).view(1, token_h * token_w, -1)
        image_embeds = image_embeds + te
        v_pred = model._t2i_predict_v(
            image_embeds, indexes_image, attn_cond, pkv, t, z,
            image_token_num=token_h * token_w, timestep_embeddings=te, image_size=(W, H),
        )
        z = z + (t_next - t) * v_pred
        image_prediction = model.unpatchify(z, cell, H, W)
        traj.append(image_prediction.clone())

    out = {}
    for k, v in model.language_model.state_dict().items():
        out[f"language_model.{k}"] = v.contiguous().to(torch.float32)
    for k, v in model.fm_modules.state_dict().items():
        out[f"fm_modules.{k}"] = v.contiguous().to(torch.float32)
    for k, v in model.vision_model.state_dict().items():
        out[f"vision_model.{k}"] = v.contiguous().to(torch.float32)
    out["prefix.input_ids"] = input_ids.to(torch.int32)
    out["pixel_values"] = pixel_values
    out["raw_noise"] = raw_noise
    out["traj"] = torch.cat(traj, dim=0).to(torch.float32)

    dst = os.path.join(REPO_ROOT, "mlx-gen-sensenova", "tests", "fixtures", "it2i_golden.safetensors")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    meta = {
        "hidden_size": str(cfg.llm_config.hidden_size),
        "intermediate_size": str(cfg.llm_config.intermediate_size),
        "num_hidden_layers": str(cfg.llm_config.num_hidden_layers),
        "num_attention_heads": str(cfg.llm_config.num_attention_heads),
        "num_key_value_heads": str(cfg.llm_config.num_key_value_heads),
        "head_dim": str(cfg.llm_config.head_dim),
        "rms_norm_eps": repr(cfg.llm_config.rms_norm_eps),
        "rope_theta": repr(cfg.llm_config.rope_theta),
        "rope_theta_hw": repr(cfg.llm_config.rope_theta_hw),
        "vocab_size": str(cfg.llm_config.vocab_size),
        "vision_hidden_size": str(cfg.vision_config.hidden_size),
        "patch_size": str(ps),
        "width": str(W),
        "height": str(H),
        "num_steps": str(num_steps),
        "src_grid_h": str(src_gh),
        "src_grid_w": str(src_gw),
        "img_context_id": str(IMG_CONTEXT_ID),
        "img_start_id": str(IMG_START_ID),
    }
    save_file(out, dst, metadata=meta)
    print(f"wrote {dst}")
    print(f"  prefix ids={ids}  text_len={text_len}  traj {tuple(out['traj'].shape)}  tensors {len(out)}")


if __name__ == "__main__":
    sys.exit(main())
