#!/usr/bin/env python
"""Dump a SAM3 vision-encoder fixture for the mlx-gen-sam3 SAM3-A parity test.

Saves the preprocessed input + the 4 FPN feature maps (NCHW) the port must reproduce.
    /tmp/sam3ref/.venv/bin/python dump_vision_fixture.py
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
inputs = processor(images=img, return_tensors="pt")

with torch.no_grad():
    vo = model.vision_encoder(inputs["pixel_values"])

tensors = {"pixel_values": inputs["pixel_values"].contiguous()}
for i, t in enumerate(vo.fpn_hidden_states):
    tensors[f"fpn_{i}"] = t.contiguous()  # [1, 256, H, W] NCHW
    print(f"fpn_{i}: {tuple(t.shape)}  mean={t.mean():.5f} std={t.std():.5f}")
save_file(tensors, os.path.join(OUT, "vision_fixture.safetensors"))
print("wrote vision_fixture.safetensors")
