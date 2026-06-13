"""sc-5136: golden for the Bernini planner's ChatML templating (`BerniniTemplate.encode_messages`).

Runs the **reference** `encode_messages` (copied verbatim from
`_vendor/bernini/bernini/data/bernini_template.py`, with the `veomni` logger / T5 branch dropped) on
the real Qwen2.5-VL tokenizer loaded from the snapshot `mllm/`, for four task mixes (t2i / i2i / r2v /
rv2v) whose conversations come from `generate_unified_inputs`. Dumps the structural tensors the Rust
port must reproduce: `input_ids` (after the indexed-pad → plain-pad remap), `token_type` (0/2/3),
`token_segment_ids`, `flex_token_types`, plus the vit / vae / target-mask lists.

The reference builds indexed visual pads, tokenizes the whole content string, then remaps; the Rust
port emits plain pad ids and tracks type/segment/flex during assembly — equivalent because special
tokens always split BPE. This golden proves that equivalence on the real tokenizer.

Run (inference path: no dropout, vit_mask_ratio implicit):
  ~/Repos/mflux/.venv/bin/python tools/dump_bernini_template_golden.py
Fixture -> mlx-gen-bernini/tests/fixtures/template_golden.safetensors
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "mlx-gen-bernini", "tests", "fixtures", "template_golden.safetensors")
MLLM_DIR = os.path.join(os.environ["HOME"], ".cache/mlx-gen-models/bernini_planner_mlx_bf16/mllm")

IGNORE_INDEX = -100
SPATIAL_MERGE = 2


# ===== verbatim reference: generate_unified_inputs (data_utils.py) =====
def generate_unified_inputs(prompt, input_image_hw=None, input_video_count=0,
                            output_t=None, output_h=None, output_w=None):
    input_image_hw = input_image_hw or []
    s = [{"type": "special_token", "text": "[CLS]", "has_loss": 0}]
    vi = 0
    for _ in range(input_video_count):
        s.append({"type": "video", "video_index": vi, "decode_mode": "video"})
        vi += 1
    for i, (h, w) in enumerate(input_image_hw):
        s.append({"type": "image", "image_index": i + input_video_count, "height": h, "width": w})
    s.append({"type": "text", "text": prompt, "has_loss": 0})
    if output_t == 1:
        s.append({"type": "special_token", "text": "[SOG]", "has_loss": 1})
        s.append({"type": "image_gen", "image_index": len(input_image_hw),
                  "height": output_h, "width": output_w, "has_loss": 1})
        s.append({"type": "special_token", "text": "[EOG]", "has_loss": 1})
    else:
        s.append({"type": "special_token", "text": "[SOV]", "has_loss": 1})
        s.append({"type": "video_gen", "video_index": vi, "decode_mode": "video"})
        s.append({"type": "special_token", "text": "[EOV]", "has_loss": 1})
    s.append({"type": "special_token", "text": "[EOS]", "has_loss": 1})
    return json.dumps(s, ensure_ascii=False)


# ===== verbatim reference: build_custom_attention_mask (attention_utils.py) =====
def build_custom_attention_mask(token_type, token_segment_ids):
    B, L = token_type.shape
    q_type, k_type = token_type.unsqueeze(2), token_type.unsqueeze(1)
    q_id, k_id = token_segment_ids.unsqueeze(2), token_segment_ids.unsqueeze(1)
    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).unsqueeze(0)
    k_is_ti = (k_type == 0) | (k_type == 2)
    k_is_p, k_is_o = (k_type == 1), (k_type == 3)
    ids_match = q_id == k_id
    visible_base_ti = causal_mask & k_is_ti
    fbm = torch.zeros((B, L, L), dtype=torch.bool)
    fbm = fbm | (((q_type == 0) | (q_type == 2)) & visible_base_ti)
    fbm = fbm | ((q_type == 1) & (visible_base_ti | (k_is_p & ids_match)))
    fbm = fbm | ((q_type == 3) & (visible_base_ti | (k_is_o & ids_match)))
    am = torch.zeros((B, L, L), dtype=torch.float32)
    am.masked_fill_(~fbm, float("-inf"))
    return am


# ===== verbatim reference: SYSTEM_PROMPT + Qwen2VLTemplate + BerniniTemplate.encode_messages =====
SYSTEM_PROMPT = {
    "default": "You are a helpful assistant.",
    "t2i": "You are a helpful assistant specialized in text-to-image generation.",
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "i2i": "You are a helpful assistant specialized in image editing.",
    "v2v": "You are a helpful assistant specialized in video editing.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "rv2v": "You are a helpful assistant specialized in video editing with reference.",
}


class BerniniTemplate:
    system_prompt = SYSTEM_PROMPT

    def __init__(self, tokenizer, max_inter=64):
        self.tokenizer = tokenizer
        self.image_pad_id = 151655
        self.video_pad_id = 151656
        self.max_image_or_video_inter_num = max_inter
        self.visual_input_token_pads = [f"<|visual_input_token_pad_{i}|>" for i in range(max_inter)]
        self.visual_output_token_pads = [f"<|visual_output_token_pad_{i}|>" for i in range(max_inter)]
        add = self.visual_input_token_pads + self.visual_output_token_pads
        self.tokenizer.add_special_tokens({"additional_special_tokens": add})
        self.visual_input_token_pad_ids = self.tokenizer.convert_tokens_to_ids(self.visual_input_token_pads)
        self.visual_output_token_pad_ids = self.tokenizer.convert_tokens_to_ids(self.visual_output_token_pads)

    def visual_input_token_pattern(self, n, item_id):
        return "<|vision_start|>" + self.visual_input_token_pads[item_id] * n + "<|vision_end|>"

    def visual_output_token_pattern(self, n, item_id):
        return "<|vision_start|>" + self.visual_output_token_pads[item_id] * n + "<|vision_end|>"

    def _get_system_mesage(self, task_name):
        if task_name not in self.system_prompt:
            task_name = "default"
        return {"role": "system", "content": self.system_prompt[task_name], "loss_mask": 0}

    def format_message(self, content, has_loss):
        return {"role": "user" if has_loss == 0 else "assistant", "content": content,
                "loss_mask": 0 if has_loss == 0 else 1}

    def encode_messages(self, conversations, num_tokens, task_name="", vit_mask_ratio=1.0):
        sys_msg = self._get_system_mesage(task_name)
        messages = [sys_msg]
        image_token_num_list = iter(num_tokens.get("image", []))
        video_token_num_list = iter(num_tokens.get("video", []))
        content = ""
        pre_has_loss = 0
        visual_id_to_type = {}
        visual_id, img_id, vid_id = 0, 0, 0
        indicator_id = 2
        visual_indicator_maps = {}
        image_target_mask, video_target_mask = [], []
        vae_type_list, vit_type_list = [], []
        vit_img_and_vid_id_list = []
        for message in conversations:
            if message["type"] == "special_token":
                continue
            if "has_loss" not in message:
                message["has_loss"] = 1 if message["type"] == "video_gen" else 0
            if pre_has_loss != message["has_loss"]:
                messages.append(self.format_message(content, pre_has_loss))
                content = ""
                pre_has_loss = message["has_loss"]
            if message["type"] in ["text", "cot_text"]:
                content += message["text"]
            elif message["type"] in ["image", "image_gen"]:
                n = next(image_token_num_list)
                if message["has_loss"] == 1:
                    content += self.visual_output_token_pattern(n, visual_id)
                    vit_img_and_vid_id_list.append(img_id)
                    vit_type_list.append(0)
                    indicator_id += 1
                    visual_indicator_maps[self.tokenizer.convert_tokens_to_ids(self.visual_output_token_pads[visual_id])] = indicator_id
                else:
                    content += self.visual_input_token_pattern(n, visual_id)
                    vit_img_and_vid_id_list.append(img_id)
                    vit_type_list.append(0)
                    visual_indicator_maps[self.tokenizer.convert_tokens_to_ids(self.visual_input_token_pads[visual_id])] = indicator_id
                visual_id_to_type[visual_id] = 0
                img_id += 1
                visual_id += 1
                indicator_id += 1
                image_target_mask.append(message["has_loss"])
                vae_type_list.append(0)
            elif message["type"] in ["video", "frame_gen", "video_gen"]:
                n = next(video_token_num_list)
                if message["has_loss"] == 1:
                    content += self.visual_output_token_pattern(n, visual_id)
                    vit_img_and_vid_id_list.append(vid_id)
                    vit_type_list.append(1)
                    indicator_id += 1
                    visual_indicator_maps[self.tokenizer.convert_tokens_to_ids(self.visual_output_token_pads[visual_id])] = indicator_id
                else:
                    content += self.visual_input_token_pattern(n, visual_id)
                    vit_img_and_vid_id_list.append(vid_id)
                    vit_type_list.append(1)
                    visual_indicator_maps[self.tokenizer.convert_tokens_to_ids(self.visual_input_token_pads[visual_id])] = indicator_id
                visual_id_to_type[visual_id] = 1
                vid_id += 1
                visual_id += 1
                indicator_id += 1
                video_target_mask.append(message["has_loss"])
                vae_type_list.append(1)
        messages.append(self.format_message(content, pre_has_loss))

        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            content_str = message["content"].strip()
            if not content_str:
                continue
            loss_mask = message["loss_mask"]
            message_ids = self.tokenizer.encode("<|im_start|>" + message["role"] + "\n", add_special_tokens=False)
            message_ids += self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += message_ids
            attention_mask += [1] * len(message_ids)
            labels += message_ids if loss_mask == 1 else [IGNORE_INDEX] * len(message_ids)

        ex = {
            "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "vit_type_list": vit_type_list, "vit_img_and_vid_id_list": vit_img_and_vid_id_list,
            "image_target_mask": image_target_mask, "video_target_mask": video_target_mask,
            "vae_type_list": vae_type_list,
        }
        ex = {k: torch.tensor(v, dtype=torch.long) for k, v in ex.items()}

        token_types = torch.zeros_like(ex["labels"], dtype=torch.int)
        flex_token_types = -torch.ones_like(ex["labels"], dtype=torch.int)
        token_segment_ids = torch.tensor(range(len(ex["labels"])), dtype=torch.int)
        vis_in = torch.zeros_like(ex["labels"], dtype=torch.bool)
        vis_out = torch.zeros_like(ex["labels"], dtype=torch.bool)
        for vid, in_id in enumerate(self.visual_input_token_pad_ids):
            m = ex["input_ids"] == in_id
            if m.sum() > 0:
                token_types[m] = 2
                vis_in[m] = True
                token_segment_ids[m] = vid + 1
                ex["input_ids"][m] = self.image_pad_id if visual_id_to_type[vid] == 0 else self.video_pad_id
        for vid, out_id in enumerate(self.visual_output_token_pad_ids):
            m = ex["input_ids"] == out_id
            if m.sum() > 0:
                token_types[m] = 3
                flex_token_types[m] = visual_indicator_maps[out_id]
                vis_out[m] = True
                token_segment_ids[m] = vid + 1
                ex["input_ids"][m] = self.image_pad_id if visual_id_to_type[vid] == 0 else self.video_pad_id

        return {
            "input_ids": ex["input_ids"], "token_type": token_types,
            "token_segment_ids": token_segment_ids, "flex_token_types": flex_token_types,
            "vit_type_list": ex["vit_type_list"], "vit_img_and_vid_id_list": ex["vit_img_and_vid_id_list"],
            "image_target_mask": ex["image_target_mask"], "video_target_mask": ex["video_target_mask"],
            "vae_type_list": ex["vae_type_list"],
        }


# (task_name, prompt, input_image_hw, input_video_count, output_t, num_tokens) — grids match the
# process golden: token_num = t*(h/2)*(w/2).
TASKS = {
    "t2i":  ("a cat",  [],           0, 1, {"image": [4]}),                 # gen img (1,4,4)
    "i2i":  ("edit",   [(48, 72)],   0, 1, {"image": [6, 4]}),             # in (1,4,6), gen (1,4,4)
    "r2v":  ("subj",   [(72, 48)],   0, 9, {"image": [6], "video": [12]}), # ref img (1,6,4), gen vid (3,4,4)
    "rv2v": ("edit v", [],           1, 9, {"video": [12, 20]}),           # ref vid (3,4,4), gen vid (5,4,4)
}


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MLLM_DIR)
    tmpl = BerniniTemplate(tok)

    out = {}
    keys = ["input_ids", "token_type", "token_segment_ids", "flex_token_types",
            "vit_type_list", "vit_img_and_vid_id_list", "image_target_mask", "video_target_mask",
            "vae_type_list"]
    for name, (prompt, imgs, nvid, out_t, num_tokens) in TASKS.items():
        conv = json.loads(generate_unified_inputs(prompt, imgs, nvid, out_t, 64, 64))
        ex = tmpl.encode_messages(conv, defaultdict(list, num_tokens), task_name=name)
        for k in keys:
            v = ex[k]
            if v.numel() == 0:
                v = torch.zeros(0, dtype=torch.int32)
            out[f"{name}.{k}"] = v.to(torch.int32).contiguous()
        L = ex["input_ids"].shape[0]
        print(f"  {name}: L={L} vit_types={ex['vit_type_list'].tolist()} "
              f"img_tgt={ex['image_target_mask'].tolist()} vid_tgt={ex['video_target_mask'].tolist()}")

    meta = {"tasks": ",".join(TASKS)}
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    save_file(out, FIXTURE, metadata=meta)
    print(f"wrote {FIXTURE}  ({len(out)} tensors)")


if __name__ == "__main__":
    main()
