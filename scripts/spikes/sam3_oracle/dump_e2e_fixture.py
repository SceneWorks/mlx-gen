#!/usr/bin/env python
"""SAM3 end-to-end fixture for the mlx-gen-sam3 SAM3-D parity test (sc-4922).

Full zidane + "person" run → the kept instance masks at native 288² (no resize) + the inputs the
Rust segmenter consumes (pixel_values + input_ids + attention_mask).
    /tmp/sam3ref/.venv/bin/python dump_e2e_fixture.py
"""
import os
import urllib.request
from io import BytesIO

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import Sam3Model, Sam3Processor

OUT = os.path.dirname(os.path.abspath(__file__))
URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg"

model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float32).eval()
processor = Sam3Processor.from_pretrained("facebook/sam3")

req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
img = Image.open(BytesIO(urllib.request.urlopen(req, timeout=30).read())).convert("RGB")
inputs = processor(images=img, text="person", return_tensors="pt")

with torch.no_grad():
    out = model(**inputs)

# native-resolution post-process (target_sizes=None → masks stay at 288²)
res = processor.image_processor.post_process_instance_segmentation(
    out, threshold=0.5, mask_threshold=0.5, target_sizes=None
)[0]
n = int(len(res["scores"]))
print("instances", n, "scores", [round(s, 3) for s in res["scores"].tolist()])
print("masks", tuple(res["masks"].shape))

save_file(
    {
        "pixel_values": inputs["pixel_values"].contiguous(),
        "input_ids": inputs["input_ids"].to(torch.int32).contiguous(),
        "attention_mask": inputs["attention_mask"].to(torch.int32).contiguous(),
        "instance_masks": res["masks"].to(torch.uint8).contiguous(),  # [n,288,288]
        "instance_scores": res["scores"].contiguous(),  # [n]
        "instance_boxes": res["boxes"].contiguous(),  # [n,4] xyxy in [0,1]
    },
    os.path.join(OUT, "e2e_fixture.safetensors"),
)
print("wrote e2e_fixture.safetensors")
