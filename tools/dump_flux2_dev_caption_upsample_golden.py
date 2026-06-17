"""Token-layout parity golden for FLUX.2-dev **caption upsampling** (sc-6030).

The PixtralProcessor `[IMG]`/`[IMG_BREAK]`/`[IMG_END]` expansion + the dev Mistral chat template are
model-specific, so the reference is the REAL dev tokenizer/processor driven by the diffusers
`upsample_prompt` helpers (`format_input`, `_validate_and_process_images`, the upsampling system
messages). This dumps the `input_ids` the reference produces for a T2I prompt and an I2I
(image + prompt) request — which the Rust `build_upsample_input_ids` + `expand_pixtral_image_tokens`
must reproduce exactly (the generation path then runs the Mistral tower over them). No diffusion
weights are loaded — only the tokenizer/processor — so this is cheap.

Run from the reference venv (transformers + diffusers + the dev snapshot in the HF cache):

    ~/mlx-flux-venv/bin/python ~/Repos/mlx-gen/tools/dump_flux2_dev_caption_upsample_golden.py
"""

import os

import numpy as np
import PIL.Image
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.flux2.pipeline_flux2 import (
    UPSAMPLING_MAX_IMAGE_SIZE,
    _validate_and_process_images,
    format_input,
)
from diffusers.pipelines.flux2.system_messages import (
    SYSTEM_MESSAGE_UPSAMPLING_I2I,
    SYSTEM_MESSAGE_UPSAMPLING_T2I,
)
from safetensors.numpy import save_file
from transformers import AutoProcessor

from _paths import fixture, hf_hub_cache

PROMPT = "a red fox in fresh snow"
# A deterministic non-square synthetic image (H=96, W=64) so the patch grid is rectangular.
IMG_H, IMG_W = 96, 64


def main() -> None:
    snaps = hf_hub_cache() / "models--black-forest-labs--FLUX.2-dev" / "snapshots"
    snap = snaps / sorted(os.listdir(snaps))[0]
    processor = AutoProcessor.from_pretrained(str(snap / "tokenizer"))

    # --- T2I: text-only template (no padding/truncation — the Rust generate path doesn't pad). ---
    msgs_t2i = format_input(prompts=[PROMPT], system_message=SYSTEM_MESSAGE_UPSAMPLING_T2I)
    t2i = processor.apply_chat_template(
        msgs_t2i, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np"
    )
    t2i_ids = np.asarray(t2i["input_ids"])[0].astype(np.int32)

    # --- I2I: a synthetic reference image; the processor resizes it and expands the `[IMG]`. ---
    rng = np.random.RandomState(0)
    img = PIL.Image.fromarray((rng.rand(IMG_H, IMG_W, 3) * 255).astype(np.uint8))
    flux2_image_processor = Flux2ImageProcessor()
    images = _validate_and_process_images([img], flux2_image_processor, UPSAMPLING_MAX_IMAGE_SIZE)
    msgs_i2i = format_input(
        prompts=[PROMPT], system_message=SYSTEM_MESSAGE_UPSAMPLING_I2I, images=images
    )
    i2i = processor.apply_chat_template(
        msgs_i2i, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np"
    )
    i2i_ids = np.asarray(i2i["input_ids"])[0].astype(np.int32)
    # image_sizes: the resized (H, W) the processor produced (drives the merged grid the Rust test
    # feeds `build_upsample_input_ids`).
    image_sizes = np.asarray(i2i["image_sizes"], dtype=np.int32).reshape(-1, 2)

    save_file(
        {
            "t2i_input_ids": t2i_ids,
            "i2i_input_ids": i2i_ids,
            "i2i_image_sizes": image_sizes,
        },
        fixture("mlx-gen-flux2/tests/fixtures/caption_upsample_golden.safetensors"),
    )
    print(f"t2i ids: {t2i_ids.shape}  i2i ids: {i2i_ids.shape}  image_sizes: {image_sizes.tolist()}")
    print(
        f"i2i [IMG]=10: {int((i2i_ids == 10).sum())}  "
        f"[IMG_BREAK]=12: {int((i2i_ids == 12).sum())}  [IMG_END]=13: {int((i2i_ids == 13).sum())}"
    )
    h, w = int(image_sizes[0][0]), int(image_sizes[0][1])
    print(f"merged grid (H//28, W//28) = ({h // 28}, {w // 28}) -> {(h // 28) * (w // 28)} [IMG]")


if __name__ == "__main__":
    main()
