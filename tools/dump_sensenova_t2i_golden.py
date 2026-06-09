"""sc-3188: synthetic-fixture golden for the SenseNova-U1 T2I denoise spine.

Instantiates the real `NEOChatModel` with a *small* config (dense dual-path Qwen3 + the NEO vision
embedder + the shallow `fm_head` + timestep/noise-scale embedders), then replays the genuine
`t2i_generate` denoise-loop body — `_t2i_prefix_forward` → per-step `patchify`/`extract_feature`
(gen path)/`timestep_embedder`/`noise_scale_embedder`/`_t2i_predict_v`/`_euler_step`/`unpatchify` —
on a random text prefix and a *fixed* noise tensor (no tokenizer, no 35 GB checkpoint). Every numeric
sub-op goes through the real reference modules; only the loop glue is duplicated (and re-implemented
in Rust). Dumps weights + prefix + the fixed noise + the per-step image trajectory; the Rust parity
test (`tests/t2i_parity.rs`) replays it through `T2iModel::denoise` and matches the trajectory.

float32 throughout; `attention_mask={"full_attention": None}` (the gen flash/SDPA full-attention
path), no `prepare_flash_kv_cache` (plain concat fallback — the layout the MLX cache uses).

Run (vendored reference env):
  cd _vendor/sensenova_u1 && PYTHONPATH=src .venv/bin/python \
      /abs/path/to/tools/dump_sensenova_t2i_golden.py
Fixture → mlx-gen-sensenova/tests/fixtures/t2i_golden.safetensors
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


def build_config() -> NEOChatConfig:
    vision_config = dict(
        architectures=["NEOVisionModel"],
        num_channels=3,
        patch_size=16,
        hidden_size=32,
        llm_hidden_size=64,
        downsample_ratio=0.5,
        rope_theta_vision=10000.0,
        max_position_embeddings_vision=10000,
    )
    llm_config = dict(
        architectures=["Qwen3ForCausalLM"],
        model_type="qwen3",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        rope_theta=5_000_000.0,
        rope_theta_hw=10_000.0,
        max_position_embeddings=4096,
        max_position_embeddings_hw=4096,
        vocab_size=64,
        attention_bias=False,
        hidden_act="silu",
        use_sliding_window=False,
        max_window_layers=0,
        tie_word_embeddings=False,
    )
    cfg = NEOChatConfig(
        vision_config=vision_config,
        llm_config=llm_config,
        downsample_ratio=0.5,
        template="neo1_0",
        # FM / sampler knobs read off the config object in NEOChatModel.__init__.
        timestep_shift=1.0,
        time_schedule="standard",
        time_shift_type="exponential",
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=64,
        max_image_seq_len=4096,
        noise_scale_mode="resolution",
        noise_scale=1.0,
        noise_scale_max_value=8.0,
        noise_scale_base_image_seq_len=64,
        add_noise_scale_embedding=True,
        fm_head_dim=1536,
        fm_head_layers=2,
        fm_head_mlp_ratio=1,
        use_pixel_head=False,
        use_adaLN=False,
        concat_time_token_num=0,
    )
    cfg.llm_config._attn_implementation = "eager"
    return cfg


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    cfg = build_config()
    model = NEOChatModel(cfg).to(torch.float32).eval()
    model.config.t_eps = 0.02

    ps = 16
    merge = 2
    cell = ps * merge  # 32
    W, H = 64, 64
    token_h, token_w = H // cell, W // cell  # 2, 2
    grid_h, grid_w = H // ps, W // ps  # 4, 4
    num_steps = 4
    timestep_shift = 1.0

    # ---- Prefix (random text-like ids on the understanding path) ----
    S = 5
    prefix_ids = torch.randint(0, cfg.llm_config.vocab_size, (1, S), dtype=torch.long)
    indexes = torch.zeros(3, S, dtype=torch.long)
    indexes[0] = torch.arange(S)
    attn_mask = {"full_attention": create_block_causal_mask(indexes[0])}
    pkv, _ = model._t2i_prefix_forward(prefix_ids, indexes, attn_mask)
    text_len = S
    indexes_image = model._build_t2i_image_indexes(token_h, token_w, text_len, device="cpu")
    gen_grid_hw = torch.tensor([[grid_h, grid_w]])

    # ---- noise_scale (resolution mode) + fixed initial noise ----
    base = float(cfg.noise_scale_base_image_seq_len)
    noise_scale = (((grid_h * grid_w) / (merge ** 2) / base) ** 0.5) * float(cfg.noise_scale)
    noise_scale = min(noise_scale, cfg.noise_scale_max_value)
    raw_noise = torch.randn(1, 3, H, W, dtype=torch.float32)
    image_prediction = noise_scale * raw_noise

    timesteps = torch.linspace(0.0, 1.0, num_steps + 1)
    timesteps = model._apply_time_schedule(timesteps, token_h * token_w, timestep_shift)

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
        timestep_embeddings = model.fm_modules["timestep_embedder"](t_expanded).view(1, token_h * token_w, -1)
        noise_scale_tensor = torch.full_like(t_expanded, noise_scale / cfg.noise_scale_max_value)
        timestep_embeddings = timestep_embeddings + model.fm_modules["noise_scale_embedder"](
            noise_scale_tensor
        ).view(1, token_h * token_w, -1)
        image_embeds = image_embeds + timestep_embeddings

        v_pred = model._t2i_predict_v(
            image_embeds,
            indexes_image,
            attn_cond,
            pkv,
            t,
            z,
            image_token_num=token_h * token_w,
            timestep_embeddings=timestep_embeddings,
            image_size=(W, H),
        )
        z = z + (t_next - t) * v_pred
        image_prediction = model.unpatchify(z, cell, H, W)
        traj.append(image_prediction.clone())

    out = {}
    for k, v in model.language_model.state_dict().items():
        out[f"language_model.{k}"] = v.contiguous().to(torch.float32)
    for k, v in model.fm_modules.state_dict().items():
        out[f"fm_modules.{k}"] = v.contiguous().to(torch.float32)
    out["prefix.input_ids"] = prefix_ids.to(torch.int32)
    out["prefix.indexes"] = indexes.to(torch.int32)
    out["raw_noise"] = raw_noise
    out["traj"] = torch.cat(traj, dim=0).to(torch.float32)  # [num_steps, 3, H, W]
    out["final_image"] = image_prediction.to(torch.float32)

    dst = os.path.join(
        REPO_ROOT, "mlx-gen-sensenova", "tests", "fixtures", "t2i_golden.safetensors"
    )
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
        "noise_scale": repr(noise_scale),
    }
    save_file(out, dst, metadata=meta)
    print(f"wrote {dst}")
    print(f"  noise_scale={noise_scale:.4f}  traj {tuple(out['traj'].shape)}  tensors {len(out)}")


if __name__ == "__main__":
    sys.exit(main())
