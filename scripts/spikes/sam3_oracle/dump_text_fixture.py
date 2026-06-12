#!/usr/bin/env python
"""SAM3 text-encoder fixture for the mlx-gen-sam3 SAM3-B parity test (sc-4920).

Saves, per concept, the tokenized ids/mask + the CLIP last_hidden_state [1,32,1024] + the
SAM3-projected text features [1,32,256].
    /tmp/sam3ref/.venv/bin/python dump_text_fixture.py
"""
import json
import os

import torch
from safetensors.torch import save_file
from transformers import Sam3Model, Sam3Processor

OUT = os.path.dirname(os.path.abspath(__file__))
model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float32).eval()
processor = Sam3Processor.from_pretrained("facebook/sam3")

tensors, manifest = {}, {}
for concept in ["person", "car"]:
    inputs = processor(text=concept, return_tensors="pt")
    with torch.no_grad():
        out = model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
    tensors[f"{concept}.text_features"] = out.pooler_output.contiguous()  # [1,32,256]
    tensors[f"{concept}.clip_last_hidden_state"] = out.last_hidden_state.contiguous()  # [1,32,1024]
    tensors[f"{concept}.input_ids"] = inputs["input_ids"].to(torch.int32).contiguous()
    tensors[f"{concept}.attention_mask"] = inputs["attention_mask"].to(torch.int32).contiguous()
    manifest[concept] = {
        "input_ids": inputs["input_ids"][0].tolist(),
        "attention_mask": inputs["attention_mask"][0].tolist(),
        "text_features_shape": list(out.pooler_output.shape),
    }
    print(f"{concept}: ids[:4]={inputs['input_ids'][0][:4].tolist()} mask_sum={int(inputs['attention_mask'].sum())} "
          f"feat {tuple(out.pooler_output.shape)} mean={out.pooler_output.mean():.5f}")

save_file(tensors, os.path.join(OUT, "text_fixture.safetensors"))
json.dump(manifest, open(os.path.join(OUT, "text_fixture_manifest.json"), "w"), indent=2)
print("wrote text_fixture.safetensors + manifest")
