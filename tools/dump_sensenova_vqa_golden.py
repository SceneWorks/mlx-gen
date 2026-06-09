"""sc-3191: synthetic-fixture golden for the SenseNova-U1 VQA / understanding spine.

Exercises the understanding-path AR decode from an **image-conditioned** prefix (the new VQA
composition): the reference image-conditioned prefill (`extract_feature` understanding +
`get_thw_indexes` + splice into `<IMG_CONTEXT>` + `_it2i_prefix_forward`) followed by a greedy
single-token decode (the same mechanic as `chat`'s `language_model.generate`). Random prefix +
source image, tiny model. Small image-token ids (10/11) so they fit the tiny vocab; the Rust test
mirrors via `T2iModel::with_image_token_ids`.

Run (vendored reference env):
  cd _vendor/sensenova_u1 && PYTHONPATH=src .venv/bin/python \
      /abs/path/to/tools/dump_sensenova_vqa_golden.py
Fixture → mlx-gen-sensenova/tests/fixtures/vqa_golden.safetensors
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
    N = 10  # decode steps

    # Source image + understanding features.
    src_gh, src_gw = 4, 4
    n_ctx = (src_gh // merge) * (src_gw // merge)
    pixel_values = torch.randn(src_gh * src_gw, 3 * ps * ps, dtype=torch.float32)
    src_grid_hw = torch.tensor([[src_gh, src_gw]])

    # Image-conditioned question prefix (small ids): [t, <img>, ctx*4, </img>, question tokens...]
    ids = [3, IMG_START_ID] + [IMG_CONTEXT_ID] * n_ctx + [IMG_END_ID, 5, 7, 9]
    input_ids = torch.tensor([ids], dtype=torch.long)
    indexes = model.get_thw_indexes(input_ids[0], src_grid_hw)
    attn_mask = {"full_attention": create_block_causal_mask(indexes[0])}
    input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
    vit = model.extract_feature(pixel_values, grid_hw=src_grid_hw)
    B, Nseq, C = input_embeds.shape
    flat = input_embeds.reshape(B * Nseq, C)
    sel = input_ids.reshape(B * Nseq) == IMG_CONTEXT_ID
    flat[sel] = vit.reshape(-1, C).to(flat.dtype)
    input_embeds = flat.reshape(B, Nseq, C)

    out = model.language_model(inputs_embeds=input_embeds, indexes=indexes, attention_mask=attn_mask, use_cache=True, output_hidden_states=True)
    pkv = out.past_key_values
    t_idx = int(indexes[0].max().item())
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1)

    decode_tokens = []
    for _ in range(N):
        decode_tokens.append(int(next_token.item()))
        model.language_model.model.current_index = t_idx
        step = model.language_model(input_ids=next_token.unsqueeze(0), past_key_values=pkv, use_cache=True)
        pkv = step.past_key_values
        t_idx += 1
        next_token = torch.argmax(step.logits[:, -1, :], dim=-1)

    o = {}
    for k, v in model.language_model.state_dict().items():
        o[f"language_model.{k}"] = v.contiguous().to(torch.float32)
    for k, v in model.fm_modules.state_dict().items():
        o[f"fm_modules.{k}"] = v.contiguous().to(torch.float32)
    for k, v in model.vision_model.state_dict().items():
        o[f"vision_model.{k}"] = v.contiguous().to(torch.float32)
    o["prefix.input_ids"] = input_ids.to(torch.int32)
    o["pixel_values"] = pixel_values
    o["decode.tokens"] = torch.tensor(decode_tokens, dtype=torch.int32)

    dst = os.path.join(REPO_ROOT, "mlx-gen-sensenova", "tests", "fixtures", "vqa_golden.safetensors")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    meta = {
        "hidden_size": str(cfg.llm_config.hidden_size), "intermediate_size": str(cfg.llm_config.intermediate_size),
        "num_hidden_layers": str(cfg.llm_config.num_hidden_layers), "num_attention_heads": str(cfg.llm_config.num_attention_heads),
        "num_key_value_heads": str(cfg.llm_config.num_key_value_heads), "head_dim": str(cfg.llm_config.head_dim),
        "rms_norm_eps": repr(cfg.llm_config.rms_norm_eps), "rope_theta": repr(cfg.llm_config.rope_theta),
        "rope_theta_hw": repr(cfg.llm_config.rope_theta_hw), "vocab_size": str(cfg.llm_config.vocab_size),
        "vision_hidden_size": str(cfg.vision_config.hidden_size), "patch_size": str(ps),
        "src_grid_h": str(src_gh), "src_grid_w": str(src_gw),
        "img_context_id": str(IMG_CONTEXT_ID), "img_start_id": str(IMG_START_ID),
    }
    save_file(o, dst, metadata=meta)
    print(f"wrote {dst}")
    print(f"  prefix ids={ids}  decode.tokens={decode_tokens}  tensors {len(o)}")


if __name__ == "__main__":
    sys.exit(main())
