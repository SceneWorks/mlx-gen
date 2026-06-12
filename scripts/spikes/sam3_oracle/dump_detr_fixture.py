#!/usr/bin/env python
"""SAM3 DETR-detector fixture for the mlx-gen-sam3 SAM3-C parity test (sc-4921).

One consistent zidane + "person" run: saves the detector's inputs (the 72² FPN feature + the
projected text features + attention mask) and outputs (pred_logits / pred_boxes / presence_logits).
    /tmp/sam3ref/.venv/bin/python dump_detr_fixture.py
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
img_inputs = processor(images=img, return_tensors="pt")
text_inputs = processor(text="person", return_tensors="pt")

with torch.no_grad():
    vo = model.vision_encoder(img_inputs["pixel_values"])
    tf = model.get_text_features(
        input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"]
    ).pooler_output  # [1,32,256]
    out = model(
        pixel_values=img_inputs["pixel_values"],
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
    )

fpn_72 = vo.fpn_hidden_states[2].contiguous()  # the level the detector uses: fpn[:-1][-1]
print("fpn_72", tuple(fpn_72.shape), "text", tuple(tf.shape))
print("pred_logits", tuple(out.pred_logits.shape), "pred_boxes", tuple(out.pred_boxes.shape),
      "presence", out.presence_logits.flatten().tolist())

save_file(
    {
        "fpn_72": fpn_72,  # [1,256,72,72] NCHW
        "text_features": tf.contiguous(),  # [1,32,256]
        "attention_mask": text_inputs["attention_mask"].to(torch.int32).contiguous(),
        "pred_logits": out.pred_logits.contiguous(),  # [1,200]
        "pred_boxes": out.pred_boxes.contiguous(),  # [1,200,4] xyxy
        "presence_logits": out.presence_logits.contiguous(),  # [1,1]
    },
    os.path.join(OUT, "detr_fixture.safetensors"),
)
print("wrote detr_fixture.safetensors")
