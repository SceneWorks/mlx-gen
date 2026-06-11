"""Dump a Qwen-Image-Edit VL-encoder parity golden for the Rust port (sc-2465, slice 6b-3).

Loads the real Edit-2511 weights (LM `model.*` + vision `visual.*`) into the fork's
`QwenVisionLanguageEncoder` via the same `WeightLoader`/`WeightApplier` path as `init_edit`, then runs
it on a fixed constructed input (a 64-token template prefix + a `<|image_pad|>` run sized to the grid
+ a short prompt). Dumps only the inputs + the fork's f32 `prompt_embeds`; the Rust test loads the
same weights from the snapshot (`loader::load_vision_language_encoder`).

The LM is ~7B; in f32 (so the golden matches the Rust f32 forward, where bf16 weights promote per-op)
it is ~28 GB + the ~2.7 GB f32 vision tower — run on a machine with enough RAM (the fork loads it
anyway). Mirrors `dump_qwen_text_encoder_golden.py`.

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python /path/to/mlx-gen/tools/dump_qwen_vl_encoder_golden.py
Output (gitignored): tools/golden/qwen_vl_encoder_golden.safetensors
"""

import os

import mlx.core as mx
from mlx.utils import tree_map

from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_language_encoder import QwenVisionLanguageEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_transformer import VisionTransformer
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

# The frozen fork's model_config still pins 2509 (`ModelConfig.qwen_image_edit().model_name`), but
# 2509 is superseded and gone from the HF cache (sc-2997). 2511 has the same VL/vision arch, so load
# it directly here without touching the frozen fork; override with QWEN_EDIT_REPO if needed.
model_path = os.environ.get("QWEN_EDIT_REPO") or "Qwen/Qwen-Image-Edit-2511"
weights = WeightLoader.load(weight_definition=QwenWeightDefinition, model_path=model_path)

te = QwenTextEncoder()
te.encoder.visual = VisionTransformer()  # init_edit does this before applying weights
WeightApplier.apply_and_quantize(
    weights=weights,
    quantize_arg=None,
    weight_definition=QwenWeightDefinition,
    models={"text_encoder": te},
)
# Cast LM + vision to f32 so the golden matches the Rust f32 forward.
te.update(tree_map(lambda a: a.astype(mx.float32), te.parameters()))
vl = QwenVisionLanguageEncoder(encoder=te.encoder)

mx.random.seed(0)
IMAGE_TOKEN = 151655
VISION_START, VISION_END, IM_END = 151652, 151653, 151645

grid = mx.array([[1, 12, 12]], dtype=mx.int32)  # llm 6x6 -> n_vis = 144//4 = 36
n_vis = int((1 * 12 * 12) // 4)
pixel_values = mx.random.normal((1 * 12 * 12, 3 * 2 * 14 * 14)).astype(mx.float32)

# 64-token template prefix (dropped) + <|vision_start|> + 36 <|image_pad|> + <|vision_end|> + prompt.
ids = [100] * 64 + [VISION_START] + [IMAGE_TOKEN] * n_vis + [VISION_END, 101, 102, 103, 104, IM_END]
input_ids = mx.array([ids], dtype=mx.int32)
attention_mask = mx.ones((1, len(ids)), dtype=mx.int32)

prompt_embeds, _ = vl(input_ids, attention_mask, pixel_values, grid)
mx.eval(prompt_embeds)

out = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "pixel_values": pixel_values,
    "vl_grid": grid,
    "prompt_embeds": prompt_embeds.astype(mx.float32),
}
golden_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(golden_dir, exist_ok=True)
path_out = os.path.join(golden_dir, "qwen_vl_encoder_golden.safetensors")
mx.save_safetensors(path_out, out)
print(f"input_ids={input_ids.shape} n_vis={n_vis} -> prompt_embeds={prompt_embeds.shape}")
print(f"wrote {path_out}")
