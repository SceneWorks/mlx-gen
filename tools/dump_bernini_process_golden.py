"""sc-5136: golden for the Bernini planner's MRoPE position ids + 4-D flex attention mask.

The two outputs the planner's tokenized-input pipeline must get bit-exact:
  - **position_ids** `(3, L)` — `Qwen2_5_VLModel.get_rope_index` (3-D MRoPE: text runs are equal
    1-D ramps carrying a running `max+1`; each vision block lays out t/h/w with the temporal step
    `second_per_grid_t * tokens_per_second`, images using `second_per_grid_t = 0`).
  - **attention_mask_4d** `(1, L, L)` — `build_custom_attention_mask` (causal over text + input-vit
    keys; gen-output queries additionally attend bidirectionally within their own segment id; nothing
    attends *into* the gen latents from the text side).

`get_rope_index` (the `image_grid_thw is not None` branch) and `build_custom_attention_mask` are
copied **verbatim** from `_vendor/bernini/bernini/models/modeling_qwen2_5_vl.py` /
`_vendor/bernini/bernini/data/utils/attention_utils.py`, so the oracle is the reference. The four
task mixes (t2i / i2i / r2v / rv2v) are constructed with the exact token layout the
`BerniniTemplate.encode_messages` produces (vision_start + pad·N + vision_end, indexed pads remapped
to plain image/video pad ids, token_type 0/2/3, token_segment_ids = visual_id+1 on pads), so the
golden exercises real per-task structure without needing the tokenizer (that is sc-5136's templating
sub-piece, goldened separately).

Run:
  ~/Repos/mflux/.venv/bin/python tools/dump_bernini_process_golden.py
Fixture -> mlx-gen-bernini/tests/fixtures/process_golden.safetensors
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(
    REPO_ROOT, "mlx-gen-bernini", "tests", "fixtures", "process_golden.safetensors"
)

# Qwen2.5-VL / Bernini token-id constants (from the snapshot config).
SPATIAL_MERGE_SIZE = 2
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_ID = 151652
VISION_END_ID = 151653
TOKENS_PER_SECOND = 2

_CFG = SimpleNamespace(
    image_token_id=IMAGE_TOKEN_ID,
    video_token_id=VIDEO_TOKEN_ID,
    vision_start_token_id=VISION_START_ID,
    vision_config=SimpleNamespace(
        spatial_merge_size=SPATIAL_MERGE_SIZE, tokens_per_second=TOKENS_PER_SECOND
    ),
)


# ===== verbatim reference: get_rope_index (image_grid_thw branch) =====
def get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None,
                   second_per_grid_ts=None, vision_start_indices=None, attention_mask=None):
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    vision_start_token_id = self.config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1],
                                  dtype=input_ids.dtype, device=input_ids.device)
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            if vision_start_indices is None:
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (image_grid_thw[image_index][0], image_grid_thw[image_index][1],
                               image_grid_thw[image_index][2])
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (video_grid_thw[video_index][0], video_grid_thw[video_index][1],
                               video_grid_thw[video_index][2])
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(), h.item() // spatial_merge_size, w.item() // spatial_merge_size)
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas


# ===== verbatim reference: build_custom_attention_mask =====
def build_custom_attention_mask(token_type, token_segment_ids):
    B, L = token_type.shape
    device = token_type.device
    q_type = token_type.unsqueeze(2)
    k_type = token_type.unsqueeze(1)
    q_id = token_segment_ids.unsqueeze(2)
    k_id = token_segment_ids.unsqueeze(1)
    causal_mask = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool))
    causal_mask = causal_mask.unsqueeze(0)
    k_is_ti = (k_type == 0) | (k_type == 2)
    k_is_p = (k_type == 1)
    k_is_o = (k_type == 3)
    ids_match = (q_id == k_id)
    visible_base_ti = causal_mask & k_is_ti
    visible_p_bidirectional = k_is_p & ids_match
    visible_o_bidirectional = k_is_o & ids_match
    final_bool_mask = torch.zeros((B, L, L), device=device, dtype=torch.bool)
    q_is_ti = (q_type == 0) | (q_type == 2)
    final_bool_mask = final_bool_mask | (q_is_ti & visible_base_ti)
    q_is_p = (q_type == 1)
    final_bool_mask = final_bool_mask | (q_is_p & (visible_base_ti | visible_p_bidirectional))
    q_is_o = (q_type == 3)
    final_bool_mask = final_bool_mask | (q_is_o & (visible_base_ti | visible_o_bidirectional))
    attention_mask = torch.zeros((B, L, L), device=device, dtype=torch.float32)
    attention_mask.masked_fill_(~final_bool_mask, float("-inf"))
    return attention_mask


# ===== task-mix builders (mirror BerniniTemplate.encode_messages token layout) =====
class Builder:
    """Assemble input_ids + token_type + token_segment_ids with the template's exact layout."""

    def __init__(self):
        self.ids = []
        self.types = []  # 0 text, 2 input-vit, 3 gen-output
        self.segs = []   # filled with range() then overwritten on pad runs
        self.visual_id = 0
        self.image_grid = []
        self.video_grid = []

    def text(self, n):
        for _ in range(n):
            self._push(1000 + len(self.ids), 0, None)

    def _push(self, tid, ttype, seg):
        self.ids.append(tid)
        self.types.append(ttype)
        self.segs.append(seg)

    def _vision(self, pad_id, ttype, t, h, w, grid_list):
        grid_list.append([t, h, w])
        n = t * (h // SPATIAL_MERGE_SIZE) * (w // SPATIAL_MERGE_SIZE)
        self._push(VISION_START_ID, 0, None)
        seg = self.visual_id + 1
        for _ in range(n):
            self._push(pad_id, ttype, seg)
        self._push(VISION_END_ID, 0, None)
        self.visual_id += 1

    def input_image(self, t, h, w):
        self._vision(IMAGE_TOKEN_ID, 2, t, h, w, self.image_grid)

    def input_video(self, t, h, w):
        self._vision(VIDEO_TOKEN_ID, 2, t, h, w, self.video_grid)

    def gen_image(self, t, h, w):
        self._vision(IMAGE_TOKEN_ID, 3, t, h, w, self.image_grid)

    def gen_video(self, t, h, w):
        self._vision(VIDEO_TOKEN_ID, 3, t, h, w, self.video_grid)

    def finish(self):
        # token_segment_ids = range(L), then pad runs overwrite with visual_id+1.
        segs = list(range(len(self.ids)))
        for i, s in enumerate(self.segs):
            if s is not None:
                segs[i] = s
        input_ids = torch.tensor([self.ids], dtype=torch.long)
        token_type = torch.tensor([self.types], dtype=torch.int)
        token_segment_ids = torch.tensor([segs], dtype=torch.int)
        image_grid = torch.tensor(self.image_grid, dtype=torch.long) if self.image_grid else None
        video_grid = torch.tensor(self.video_grid, dtype=torch.long) if self.video_grid else None
        return input_ids, token_type, token_segment_ids, image_grid, video_grid


def build_t2i():
    b = Builder()
    b.text(3)  # system + prompt
    b.gen_image(1, 4, 4)
    b.text(1)  # eos
    return b.finish()


def build_i2i():
    b = Builder()
    b.text(2)
    b.input_image(1, 4, 6)
    b.text(2)
    b.gen_image(1, 4, 4)
    b.text(1)
    return b.finish()


def build_r2v():
    b = Builder()
    b.text(2)
    b.input_image(1, 6, 4)  # reference subject
    b.text(2)
    b.gen_video(3, 4, 4)
    b.text(1)
    return b.finish()


def build_rv2v():
    b = Builder()
    b.text(2)
    b.input_video(3, 4, 4)  # reference video
    b.text(2)
    b.gen_video(5, 4, 4)
    b.text(1)
    return b.finish()


TASKS = {"t2i": build_t2i, "i2i": build_i2i, "r2v": build_r2v, "rv2v": build_rv2v}


@torch.no_grad()
def main() -> None:
    out = {}
    for name, build in TASKS.items():
        input_ids, token_type, token_segment_ids, image_grid, video_grid = build()
        pos, _ = get_rope_index(
            SimpleNamespace(config=_CFG), input_ids=input_ids,
            image_grid_thw=image_grid, video_grid_thw=video_grid,
            attention_mask=torch.ones_like(input_ids),
        )
        mask = build_custom_attention_mask(token_type, token_segment_ids)
        vis = (mask[0] > -1e30).to(torch.int8)  # 1 visible / 0 masked

        out[f"{name}.input_ids"] = input_ids[0].to(torch.int32).contiguous()
        out[f"{name}.token_type"] = token_type[0].to(torch.int32).contiguous()
        out[f"{name}.token_segment_ids"] = token_segment_ids[0].to(torch.int32).contiguous()
        out[f"{name}.position_ids"] = pos[:, 0, :].to(torch.int32).contiguous()  # (3, L)
        out[f"{name}.mask_vis"] = vis.contiguous()  # (L, L)
        if image_grid is not None:
            out[f"{name}.image_grid_thw"] = image_grid.to(torch.int32).contiguous()
        if video_grid is not None:
            out[f"{name}.video_grid_thw"] = video_grid.to(torch.int32).contiguous()

    meta = {
        "tasks": ",".join(TASKS),
        "spatial_merge_size": str(SPATIAL_MERGE_SIZE),
        "tokens_per_second": str(TOKENS_PER_SECOND),
        "image_token_id": str(IMAGE_TOKEN_ID),
        "video_token_id": str(VIDEO_TOKEN_ID),
        "vision_start_token_id": str(VISION_START_ID),
    }
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    save_file(out, FIXTURE, metadata=meta)
    print(f"wrote {FIXTURE}  ({len(out)} tensors)")
    for name in TASKS:
        L = out[f"{name}.input_ids"].shape[0]
        print(f"  {name}: L={L}  pos {tuple(out[f'{name}.position_ids'].shape)}")


if __name__ == "__main__":
    main()
