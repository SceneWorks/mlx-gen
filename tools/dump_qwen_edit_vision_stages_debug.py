"""Debug bisection: capture the fork vision transformer's per-stage activations (patch_embed,
reordered, block0, blocks_all, merger, final) at grid (1,28,28) on the real pixels, replicating its
`__call__`, to find the first stage that diverges from the Rust port. Loads vision weights (bf16).
sc-2465 slice 7a debugging.

Run: cd ~/repos/mflux && uv run python /path/to/mlx-gen/tools/dump_qwen_edit_vision_stages_debug.py
Output (gitignored): tools/golden/qwen_edit_vision_stages_debug.safetensors
"""

import glob
import os

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_transformer import VisionTransformer

here = os.path.dirname(os.path.abspath(__file__))
tok = mx.load(os.path.join(here, "golden", "qwen_edit_tokenize_debug.safetensors"))
pixel_values = tok["pixel_values"].astype(mx.float32)
grid_thw = tok["image_grid_thw"]

snap = sorted(
    p for p in glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/*")) if os.path.isdir(p)
)[0]
vision_params = {}
for shard in sorted(glob.glob(os.path.join(snap, "text_encoder", "*.safetensors"))):
    for k, v in mx.load(shard).items():
        if not k.startswith("visual."):
            continue
        leaf = k[len("visual.") :]
        if leaf == "patch_embed.proj.weight":
            v = v.transpose(0, 2, 3, 4, 1)
        leaf = leaf.replace("merger.mlp.0.", "merger.mlp_0.").replace("merger.mlp.2.", "merger.mlp_1.")
        vision_params[leaf] = v

vt = VisionTransformer()
vt.update(tree_unflatten(list(vision_params.items())))

caps = {}
# --- replicate VisionTransformer.__call__ with captures ---
n = pixel_values.shape[0]
conv_in = pixel_values.reshape(n, 3, 2, 14, 14).transpose(0, 2, 3, 4, 1)
caps["conv_input"] = conv_in
hidden_states = vt.patch_embed(pixel_values)
caps["patch_embed"] = hidden_states
rotary_pos_emb = vt.rot_pos_emb(grid_thw)
window_index, cu_window_seqlens = vt.get_window_index(grid_thw)
uniq = [cu_window_seqlens[0].item()]
for i in range(1, len(cu_window_seqlens)):
    if cu_window_seqlens[i].item() != uniq[-1]:
        uniq.append(cu_window_seqlens[i].item())
cu_window_seqlens = mx.array(uniq, dtype=mx.int32)
seq_len = hidden_states.shape[0]
cu_seqlens = [0]
off = 0
for t, h, w in grid_thw:
    off += int(t) * int(h) * int(w)
    cu_seqlens.append(off)
cu_seqlens = mx.array(cu_seqlens, dtype=mx.int32)
unit = vt.spatial_merge_unit
num_groups = seq_len // unit
hg = hidden_states.reshape(num_groups, unit, -1)[window_index.astype(mx.int32), :, :]
hidden_states = hg.reshape(seq_len, -1)
caps["reordered"] = hidden_states
rg = rotary_pos_emb.reshape(num_groups, unit, -1)[window_index.astype(mx.int32), :, :]
rotary_pos_emb = rg.reshape(seq_len, -1)
emb = mx.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)
position_embeddings = (mx.cos(emb), mx.sin(emb))
for layer_num, block in enumerate(vt.blocks):
    now = cu_seqlens if layer_num in vt.fullatt_block_indexes else cu_window_seqlens
    hidden_states = block(hidden_states, position_embeddings, now)
    if layer_num == 0:
        caps["block0"] = hidden_states
caps["blocks_all"] = hidden_states
hidden_states = vt.merger(hidden_states, grid_thw)
caps["merger"] = hidden_states
reverse_indices = mx.argsort(window_index.astype(mx.int32))
caps["final"] = hidden_states[reverse_indices.astype(mx.int32), :]

mx.eval(list(caps.values()))
out = {k: v.astype(mx.float32) for k, v in caps.items()}
path_out = os.path.join(here, "golden", "qwen_edit_vision_stages_debug.safetensors")
mx.save_safetensors(path_out, out)
for k, v in out.items():
    print(f"{k}: {v.shape}")
print(f"wrote {path_out}")
