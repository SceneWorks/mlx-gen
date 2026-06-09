"""sc-3190: synthetic-fixture golden for interleave's `append_generated_image` mechanic.

The interleave loop's one genuinely new numeric piece is re-encoding a generated image back into the
text cache (the reference `interleave_gen`'s inner `append_image_to_cache`): map model-space→[0,1],
ImageNet-normalize, understanding-vision embed, append `</img>`, with image tokens at temporal
`t_idx+1` (merged-grid h/w) and `</img>` at `t_idx+2`, under a mask where image tokens see all past +
each other but NOT the `</img>` position. This replays exactly that on a tiny model + a fixed prefix
cache + a fixed image, then decodes one token; the Rust test drives `T2iModel::append_generated_image`
and matches the next-token logits + greedy pick. Small image-token ids (10/11/12) for the tiny vocab.

Run (vendored reference env):
  cd _vendor/sensenova_u1 && PYTHONPATH=src .venv/bin/python \
      /abs/path/to/tools/dump_sensenova_interleave_golden.py
Fixture → mlx-gen-sensenova/tests/fixtures/interleave_golden.safetensors
"""

from __future__ import annotations

import os
import sys

import torch
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOChatConfig
from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel, build_abs_positions_from_grid_hw
from sensenova_u1.models.neo_unify.modeling_qwen3 import create_block_causal_mask

IMG_CONTEXT_ID, IMG_START_ID, IMG_END_ID = 10, 11, 12


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
    model.img_context_token_id = IMG_CONTEXT_ID
    model.img_start_token_id = IMG_START_ID
    ps, merge = 16, 2
    H = W = 64
    token_h, token_w = H // (ps * merge), W // (ps * merge)  # 2,2
    grid_h, grid_w = H // ps, W // ps  # 4,4

    # ---- A text prefix prefilled into the cache (understanding path) ----
    ids = [3, 5, 7, 9, 6]
    input_ids = torch.tensor([ids], dtype=torch.long)
    indexes = torch.zeros(3, len(ids), dtype=torch.long)
    indexes[0] = torch.arange(len(ids))
    attn = {"full_attention": create_block_causal_mask(indexes[0])}
    out = model.language_model(input_ids=input_ids, indexes=indexes, attention_mask=attn, use_cache=True)
    pkv = out.past_key_values
    t_idx = int(indexes[0].max().item())

    # ---- A fixed "generated" image (model space ~[-1,1]) ----
    image = torch.randn(1, 3, H, W, dtype=torch.float32).clamp(-1, 1)

    # ---- Reference append_image_to_cache (verbatim mechanic) ----
    raw = image[0].unsqueeze(0) * 0.5 + 0.5
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    und_img = (raw - mean) / std
    c, h, w = und_img[0].shape
    flat = und_img[0].view(c, h // ps, ps, w // ps, ps).permute(1, 3, 0, 2, 4).reshape((h // ps) * (w // ps), c * ps ** 2)
    gen_grid_hw = torch.tensor([[grid_h, grid_w]])
    vit = model.extract_feature(flat, grid_hw=gen_grid_hw).unsqueeze(0)  # [1, n_img, C]
    n_img = vit.shape[1]
    end_embed = model.language_model.get_input_embeddings()(torch.tensor([[IMG_END_ID]]))
    inputs_embeds_img = torch.cat([vit, end_embed], dim=1)
    abs_w, abs_h = build_abs_positions_from_grid_hw(gen_grid_hw // merge)

    past_len = pkv.get_seq_length()
    tgt = n_img + 1
    t_indexes = torch.zeros(tgt, dtype=torch.long)
    t_indexes[:n_img] = t_idx + 1
    t_indexes[n_img] = t_idx + 2
    h_indexes = torch.zeros(tgt, dtype=torch.long)
    w_indexes = torch.zeros(tgt, dtype=torch.long)
    h_indexes[:n_img] = abs_h
    w_indexes[:n_img] = abs_w
    idx2 = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)
    mask = torch.zeros(1, 1, tgt, past_len + tgt)
    mask[0, 0, :n_img, past_len + n_img] = float("-inf")
    out2 = model.language_model(inputs_embeds=inputs_embeds_img, indexes=idx2,
                                attention_mask={"full_attention": mask}, past_key_values=pkv, use_cache=True)
    next_logits = out2.logits[:, -1, :].reshape(-1)
    next_token = int(torch.argmax(next_logits).item())

    o = {}
    for k, v in model.language_model.state_dict().items():
        o[f"language_model.{k}"] = v.contiguous().to(torch.float32)
    for k, v in model.fm_modules.state_dict().items():
        o[f"fm_modules.{k}"] = v.contiguous().to(torch.float32)
    for k, v in model.vision_model.state_dict().items():
        o[f"vision_model.{k}"] = v.contiguous().to(torch.float32)
    o["prefix.input_ids"] = input_ids.to(torch.int32)
    o["image"] = image
    o["next_logits"] = next_logits.to(torch.float32)

    dst = os.path.join(REPO_ROOT, "mlx-gen-sensenova", "tests", "fixtures", "interleave_golden.safetensors")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    meta = {
        "hidden_size": str(cfg.llm_config.hidden_size), "intermediate_size": str(cfg.llm_config.intermediate_size),
        "num_hidden_layers": str(cfg.llm_config.num_hidden_layers), "num_attention_heads": str(cfg.llm_config.num_attention_heads),
        "num_key_value_heads": str(cfg.llm_config.num_key_value_heads), "head_dim": str(cfg.llm_config.head_dim),
        "rms_norm_eps": repr(cfg.llm_config.rms_norm_eps), "rope_theta": repr(cfg.llm_config.rope_theta),
        "rope_theta_hw": repr(cfg.llm_config.rope_theta_hw), "vocab_size": str(cfg.llm_config.vocab_size),
        "vision_hidden_size": str(cfg.vision_config.hidden_size), "patch_size": str(ps),
        "token_h": str(token_h), "token_w": str(token_w),
        "img_context_id": str(IMG_CONTEXT_ID), "img_start_id": str(IMG_START_ID), "next_token": str(next_token),
    }
    save_file(o, dst, metadata=meta)
    print(f"wrote {dst}")
    print(f"  prefix={ids}  n_img={n_img}  next_token={next_token}")


if __name__ == "__main__":
    sys.exit(main())
