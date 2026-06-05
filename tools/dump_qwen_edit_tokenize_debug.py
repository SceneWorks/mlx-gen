"""Debug dump: the fork's VL tokenize output (input_ids + pixel_values + grid) for the fixed
synthetic edit reference, to compare against the Rust `tokenize_edit`. Light (tokenizer + image
processor only, no model). sc-2465 slice 7a debugging.

Run: cd ~/repos/mflux && uv run python ~/Repos/mlx-gen/.claude/worktrees/musing-mclaren-676094/tools/dump_qwen_edit_tokenize_debug.py
Output (gitignored): tools/golden/qwen_edit_tokenize_debug.safetensors
"""

import glob
import os

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from mflux.models.qwen.tokenizer.qwen_image_processor import QwenImageProcessor
from mflux.models.qwen.tokenizer.qwen_vision_language_processor import QwenVisionLanguageProcessor
from mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer import QwenVisionLanguageTokenizer

PROMPT = "make it autumn"
W0, H0 = 512, 512
base = np.add.outer(np.arange(H0), np.arange(W0)).astype(np.int64) % 256
rgb = np.stack([base, (base * 2) % 256, (base * 3) % 256], axis=-1).astype(np.uint8)
ref_path = "/tmp/qwen_edit_ref.png"
Image.fromarray(rgb).save(ref_path)

snap = sorted(
    p
    for p in glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/*")
    )
    if os.path.isdir(p)
)[0]
hf = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
proc = QwenVisionLanguageProcessor(tokenizer=hf, image_processor=QwenImageProcessor())
vlt = QwenVisionLanguageTokenizer(processor=proc, use_picture_prefix=False)

input_ids, attention_mask, pixel_values, image_grid_thw = vlt.tokenize_with_image(PROMPT, ref_path)
mx.eval(input_ids, pixel_values, image_grid_thw)

out = {
    "input_ids": mx.array(np.array(input_ids)).astype(mx.int32),
    "pixel_values": mx.array(np.array(pixel_values)).astype(mx.float32),
    "image_grid_thw": mx.array(np.array(image_grid_thw)).astype(mx.int32),
}
golden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(golden_dir, exist_ok=True)
path_out = os.path.join(golden_dir, "qwen_edit_tokenize_debug.safetensors")
mx.save_safetensors(path_out, out)
print(
    f"input_ids {out['input_ids'].shape} grid {out['image_grid_thw'].tolist()} "
    f"pixel_values {out['pixel_values'].shape}"
)
print(f"wrote {path_out}")
