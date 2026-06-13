"""sc-5134: synthetic-fixture golden for the Bernini planner's Qwen2.5-VL vision tower.

Builds a **tiny** `Qwen2_5_VisionTransformerPretrainedModel` (a few small blocks, realistic grid
geometry so the window-partition / full-vs-windowed mask logic is actually exercised) with random
f32 weights, runs the **reference** forward, and dumps weights + inputs + output to a safetensors
fixture the Rust parity test loads:
  - `visual.*` weights (patch_embed Conv3d, 32→tiny blocks, merger),
  - `io.pixel_values` `[sum_patches, in_chans*temporal*patch*patch]`,
  - `io.grid_thw` `[num_items, 3]` (t, h, w in patches),
  - `out.tokens` `[sum_merged, out_hidden]`.

The classes are copied **verbatim** from `_vendor/bernini/bernini/models/modeling_qwen2_5_vl.py`
(`Qwen2_5_VisionPatchEmbed`, `Qwen2_5_VisionRotaryEmbedding`, `Qwen2RMSNorm`, `Qwen2_5_VLPatchMerger`,
`Qwen2_5_VLMLP`, the **eager** `Qwen2_5_VLVisionAttention`, `Qwen2_5_VLVisionBlock`, and the
`Qwen2_5_VisionTransformerPretrainedModel.{rot_pos_emb,get_window_index,forward}` body), with only the
`PreTrainedModel` plumbing dropped (a `SimpleNamespace` config + `nn.Module` base) and the attention
forced to the eager path (flash/sdpa unavailable / non-deterministic on CPU). So the oracle is the
reference. f32 throughout.

The grid `[[1,6,6],[1,4,4]]` exercises: window padding (3→4, 2→4 merged rows), multiple windows per
image, **and** the block-diagonal full-attention mask across two images (cu_seqlens has two blocks).

Run:
  ~/Repos/mflux/.venv/bin/python tools/dump_bernini_vision_tower_golden.py
Fixture -> mlx-gen-bernini/tests/fixtures/vision_tower_golden.safetensors
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(
    REPO_ROOT, "mlx-gen-bernini", "tests", "fixtures", "vision_tower_golden.safetensors"
)

# tiny dims (fixture well under 1 MB) but realistic grid geometry.
HIDDEN = 32
NUM_HEADS = 2  # head_dim = 16
INTERMEDIATE = 20
DEPTH = 4
FULLATT = [1, 3]
SPATIAL_MERGE = 2
WINDOW_SIZE = 8  # vit_merger_window_size = 8 // 2 // 2 = 2
PATCH_SIZE = 2
TEMPORAL_PATCH = 2
IN_CHANS = 3
OUT_HIDDEN = 24
GRID = [[1, 6, 6], [1, 4, 4]]


# ===== verbatim reference (PreTrainedModel plumbing dropped) =====
class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(self, patch_size, temporal_patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states):
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen):
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, dim, context_dim, spatial_merge_size=2):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x):
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, dim, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / (self.head_dim**0.5)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLVisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VLVisionAttention(config.hidden_size, num_heads=config.num_heads)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size, self.spatial_merge_size,
                w // self.spatial_merge_size, self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = grid_h // self.spatial_merge_size, grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def forward(self, hidden_states, grid_thw):
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for layer_num, blk in enumerate(self.blocks):
            cu_seqlens_now = cu_seqlens if layer_num in self.fullatt_block_indexes else cu_window_seqlens
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    cfg = SimpleNamespace(
        hidden_size=HIDDEN, num_heads=NUM_HEADS, intermediate_size=INTERMEDIATE, depth=DEPTH,
        fullatt_block_indexes=FULLATT, spatial_merge_size=SPATIAL_MERGE, window_size=WINDOW_SIZE,
        patch_size=PATCH_SIZE, temporal_patch_size=TEMPORAL_PATCH, in_channels=IN_CHANS,
        out_hidden_size=OUT_HIDDEN,
    )
    model = VisionTransformer(cfg).to(torch.float32).eval()

    grid_thw = torch.tensor(GRID, dtype=torch.int64)
    seq = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
    in_dim = IN_CHANS * TEMPORAL_PATCH * PATCH_SIZE * PATCH_SIZE
    pixel_values = torch.randn(seq, in_dim, dtype=torch.float32)

    tokens = model(pixel_values, grid_thw)

    out = {}
    for k, v in model.state_dict().items():
        out[f"visual.{k}"] = v.contiguous()
    out["io.pixel_values"] = pixel_values.contiguous()
    out["io.grid_thw"] = grid_thw.to(torch.int32).contiguous()
    out["out.tokens"] = tokens.contiguous()

    meta = {
        "hidden": str(HIDDEN), "heads": str(NUM_HEADS), "intermediate": str(INTERMEDIATE),
        "depth": str(DEPTH), "fullatt": ",".join(str(i) for i in FULLATT),
        "spatial_merge": str(SPATIAL_MERGE), "window": str(WINDOW_SIZE), "patch": str(PATCH_SIZE),
        "temporal_patch": str(TEMPORAL_PATCH), "in_chans": str(IN_CHANS), "out_hidden": str(OUT_HIDDEN),
    }
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, FIXTURE, metadata=meta)
    print(f"wrote {FIXTURE}  ({len(out)} tensors)")
    print(f"  pixel_values {tuple(pixel_values.shape)}  grid {GRID}  tokens {tuple(tokens.shape)}")


if __name__ == "__main__":
    main()
