"""Dump Qwen2.5-VL vision-transformer parity goldens for the Rust port (sc-2465, slice 6a).

Micro-gated, mirroring sc-2348. This file grows one gate at a time:

  **Gate 1 (this commit) — the index/RoPE math, NO WEIGHTS.** `get_window_index`,
  the full-attn `cu_seqlens`, and `rot_pos_emb` are pure functions of `grid_thw` + config —
  the error-prone core of the windowed attention. We dump them for a few grids that exercise
  window padding, the exact-multiple edge (pad == window), and multi-image, so the Rust port
  can be verified byte-exact with zero model download.

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_qwen_vision_golden.py
Output (gitignored): tools/golden/qwen_vision_golden.safetensors
"""

import os

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_transformer import VisionTransformer

# The index/RoPE methods don't touch block weights, so depth=1 keeps construction cheap.
vt = VisionTransformer(depth=1)


def dedup_consecutive(cu) -> list:
    """The fork dedups consecutive-equal cu_window_seqlens in __call__ (drops all-pad windows)."""
    out = [int(cu[0].item())]
    for i in range(1, len(cu)):
        v = int(cu[i].item())
        if v != out[-1]:
            out.append(v)
    return out


def cu_seqlens_full(grid: mx.array) -> list:
    """Full-attention cumulative seqlens: [0, cumulative t*h*w per image] (patch units)."""
    out, offset = [0], 0
    for t, h, w in grid.tolist():
        offset += int(t) * int(h) * int(w)
        out.append(offset)
    return out


# (t, grid_h, grid_w) in patch units. spatial_merge=2 -> llm grid = grid//2; merger window = 4.
GRIDS = {
    "g0": [[1, 12, 12]],              # llm 6x6  -> pad 2 (normal windowing)
    "g1": [[1, 16, 16]],              # llm 8x8  -> pad == window (exact-multiple edge: all-pad windows)
    "g2": [[1, 12, 12], [1, 8, 8]],   # multi-image (2nd is exact-multiple)
}

out = {}
for name, g in GRIDS.items():
    grid = mx.array(g, dtype=mx.int32)
    window_index, cu_window = vt.get_window_index(grid)
    rope = vt.rot_pos_emb(grid)  # [seq_patches, 40], pre window-reorder
    mx.eval(window_index, rope)
    out[f"{name}_grid"] = grid
    out[f"{name}_window_index"] = window_index.astype(mx.int32)
    out[f"{name}_cu_window"] = mx.array(dedup_consecutive(cu_window), dtype=mx.int32)
    out[f"{name}_cu_seqlens"] = mx.array(cu_seqlens_full(grid), dtype=mx.int32)
    out[f"{name}_rope"] = rope.astype(mx.float32)
    print(
        f"{name}: grid={g} seq={rope.shape[0]} groups={window_index.shape[0]} "
        f"cu_window={dedup_consecutive(cu_window)}"
    )

# --- Gate A: a small synthetic VisionTransformer, end-to-end (NO snapshot/weights). ---
# Exercises every weight-bearing path — patch_embed, full + windowed (block-diagonal) SDPA, the
# window reorder/reverse, and the merger — at small dims with random f32 weights. The full
# real-weight parity (gate B / slice 6b) confirms the actual checkpoint + key mapping later.
mx.random.seed(0)
SMALL = dict(
    patch_size=14, temporal_patch_size=2, in_channels=3, embed_dim=64, depth=4,
    num_heads=4, mlp_ratio=2.0, hidden_size=32, spatial_merge_size=2, window_size=112,
    fullatt_block_indexes=[1, 3],  # blocks 0,2 windowed; 1,3 full-attn
)
svt = VisionTransformer(**SMALL)
svt.update(tree_map(lambda a: a.astype(mx.float32), svt.parameters()))

io_grid = mx.array([[1, 12, 12]], dtype=mx.int32)  # llm 6x6 -> 4 windows (windowed path exercised)
io_seq = 1 * 12 * 12
io_pixel = mx.random.normal((io_seq, 3 * 2 * 14 * 14)).astype(mx.float32)
io_out = svt(io_pixel, io_grid)
mx.eval(io_out)

out["io_grid"] = io_grid
out["io_pixel_values"] = io_pixel
out["io_out"] = io_out.astype(mx.float32)
for k, v in tree_flatten(svt.parameters()):
    out[f"vt.{k}"] = v.astype(mx.float32)
print(f"vt(small): pixel={io_pixel.shape} grid={io_grid.tolist()} -> out={io_out.shape}")

# --- Gate B: the REAL depth-32 vision transformer (loads Qwen-Image-Edit-2509 `visual.*` weights). ---
# Confirms the actual checkpoint config + the HF->internal weight map (patch-embed transpose, merger
# mlp.0/2 -> mlp_0/mlp_1) at full scale. The golden holds only inputs + the fork's f32 output; the
# Rust test loads the real weights from the snapshot itself (like text_encoder_real_weights.rs).
import glob

from mlx.utils import tree_unflatten

EDIT_SNAP_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2509/snapshots/*"
)
snap = sorted(p for p in glob.glob(EDIT_SNAP_GLOB) if os.path.isdir(p))[0]
shards = sorted(glob.glob(os.path.join(snap, "text_encoder", "*.safetensors")))

vision_params = {}
for shard in shards:
    for k, v in mx.load(shard).items():  # mmap-lazy: only the visual.* tensors get materialized
        if not k.startswith("visual."):
            continue
        leaf = k[len("visual.") :]
        if leaf == "patch_embed.proj.weight":
            v = v.transpose(0, 2, 3, 4, 1)  # [O,I,kD,kH,kW] -> [O,kD,kH,kW,I]
        leaf = leaf.replace("merger.mlp.0.", "merger.mlp_0.").replace("merger.mlp.2.", "merger.mlp_1.")
        vision_params[leaf] = v.astype(mx.float32)

rvt = VisionTransformer()  # defaults == the 2509 vision_config
rvt.update(tree_unflatten(list(vision_params.items())))

real_grid = mx.array([[1, 12, 12]], dtype=mx.int32)
real_seq = 1 * 12 * 12
real_pixel = mx.random.normal((real_seq, 3 * 2 * 14 * 14)).astype(mx.float32)
real_out = rvt(real_pixel, real_grid)
mx.eval(real_out)
out["real_grid"] = real_grid
out["real_pixel_values"] = real_pixel
out["real_out"] = real_out.astype(mx.float32)
print(
    f"vt(real depth-32): {len(vision_params)} params, pixel={real_pixel.shape} "
    f"grid={real_grid.tolist()} -> out={real_out.shape}"
)

golden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(golden_dir, exist_ok=True)
path_out = os.path.join(golden_dir, "qwen_vision_golden.safetensors")
mx.save_safetensors(path_out, out)
print(f"wrote {path_out} ({len(out)} tensors)")
