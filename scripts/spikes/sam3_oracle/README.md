# SAM3 → MLX parity oracle (spike sc-4911 / epic 4910)

Torch reference fixtures the `mlx-gen-sam3` port validates against — **no Python at port-validation** (same convention as the SAM2 spike sc-3635). Generated from the public `transformers` SAM3 reference (`facebook/sam3`, F32) on real photos.

## Regenerate
```bash
# one-time env (transformers v5 dev required — SAM3 landed in transformers 5.x)
uv venv --python 3.12 /tmp/sam3ref/.venv
/tmp/sam3ref/.venv/bin/python -m pip install torch torchvision transformers pillow numpy safetensors
# run
/tmp/sam3ref/.venv/bin/python run_oracle.py
```

## Component dumpers (Phase A–F)
Each `dump_*.py` runs one real model component and writes a `*_fixture.safetensors` (gitignored) + a tracked `*_manifest.json`; a matching `#[ignore]` Rust test in `mlx-gen-sam3/tests/` gates parity (cosine):
- `dump_vision_fixture.py` / `dump_text_fixture.py` / `dump_detr_fixture.py` / `dump_e2e_fixture.py` — detector (Phase A–D).
- `dump_tracker_fixture.py` — **F1** single-frame box-prompt tracker (neck → prompt → mask decoder).
- `dump_memory_fixture.py` — **F2** memory encoder: `_encode_new_memory` (1008→1152 bilinear mask prep, sigmoid/binarize ·20−10, mask_downsampler → feature_projection → memory_fuser → projection, sine pos-enc, occlusion add). Gated by `tests/memory_parity.rs`.

## Artifacts
- `oracle_manifest.json` — per-case shapes/stats/sha1 (rounded 5dp) for staged parity.
- `fixture_{car,zidane,bus}.npz` — full tensors: `pixel_values`, `input_ids`, `pred_logits`, `pred_boxes`, `presence_logits`, `semantic_seg`, `instance_masks`.
- `overlay_{car,zidane,bus}.png` — box overlay sanity render.

## Contract (ground truth)
- **Preprocess:** resize **1008×1008** (bilinear), `/255`, normalize **mean/std = [0.5,0.5,0.5]** → ~[-1,1]. channels-first. (NOT ImageNet — the modular default is overridden by `processor_config.json`.)
- **Tokenizer:** `CLIPTokenizer`, lowercased, BOS=49406, EOS=49407, **pad with EOS to len 32**. `"person"`→`[49406,2533,49407,…]`, `"car"`→`[49406,1615,49407,…]`.
- **Outputs:** `pred_logits[1,200]`, `pred_boxes[1,200,4]` xyxy∈[0,1], `pred_masks[1,200,288,288]`, `semantic_seg[1,1,288,288]`, `presence_logits[1,1]`.
- **Vision FPN scales:** `[1,256,288,288] / 144 / 72 / 36`; detector uses the first 3; DETR encoder runs on 72²→**5184 tokens**.
- **Scoring (post-process):** `score = σ(pred_logits)·σ(presence_logits)`, keep `> threshold` (0.3 default; 0.5 used here), masks `σ > 0.5`. **No NMS at image level** (video pipeline adds `det_nms_thresh` 0.1).

## Reference results (threshold 0.5)
| case | text | image | instances | top scores | presence σ |
|---|---|---|---|---|---|
| car | car | 2646×1764 | 1 | 0.906 | 0.920 |
| zidane | person | 1280×720 | **2** | 0.966, 0.963 | 0.996 |
| bus | person | 810×1080 | **4** | 0.977–0.951 | 1.000 |

Verified visually: boxes land on the correct people/car (see overlays).
