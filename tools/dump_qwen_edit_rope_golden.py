"""Dump the multi-image (dual-latent) RoPE golden for Qwen-Image-Edit (sc-2465, slice 7a).

Weight-free: the fork's `QwenEmbedRopeMLX` with `video_fhw=[noise_grid, cond_grid]` — the noise
latents (image index 0) concatenated with the reference (index 1, frame-axis offset). The Rust
`QwenRope3d::forward_multi` must reproduce the concatenated image cos/sin (+ the text base, which is
the max over both grids). This is the novel core of the Edit transformer path.

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python /path/to/mlx-gen/tools/dump_qwen_edit_rope_golden.py
Output (gitignored): tools/golden/qwen_edit_rope_golden.safetensors
"""

import os

import mlx.core as mx

from mflux.models.qwen.model.qwen_transformer.qwen_rope import QwenEmbedRopeMLX

rope = QwenEmbedRopeMLX(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

# (frame, latent_h, latent_w): a non-square noise grid + a smaller reference grid + a text length.
NOISE = (1, 8, 12)
COND = (1, 6, 6)
TXT = 20

(img_cos, img_sin), (txt_cos, txt_sin) = rope(video_fhw=[NOISE, COND], txt_seq_lens=[TXT])
mx.eval(img_cos, img_sin, txt_cos, txt_sin)

out = {
    "img_cos": img_cos.astype(mx.float32),
    "img_sin": img_sin.astype(mx.float32),
    "txt_cos": txt_cos.astype(mx.float32),
    "txt_sin": txt_sin.astype(mx.float32),
}
golden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(golden_dir, exist_ok=True)
path_out = os.path.join(golden_dir, "qwen_edit_rope_golden.safetensors")
mx.save_safetensors(path_out, out)
print(f"noise={NOISE} cond={COND} txt={TXT} -> img {img_cos.shape}, txt {txt_cos.shape}")
print(f"wrote {path_out}")
