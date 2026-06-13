"""sc-5140: golden for the Bernini planner→renderer handoff.

The pieces that turn the planner's penultimate hidden states into the renderer's conditioning:
  - **post_process_input_embeds** (inference) — set every gen-ViT slot (`visual_output_mask`) to the
    `mask_token` (the MAR loop starts fully masked).
  - **feat_from_planner_to_renderer** (inference) — `connector.for_gen` over *all* tokens
    (`cond_embed_mask = ¬gen | gen = all`), returning the `diff_mllm_contexts` + the txt (`¬gen`) /
    vit (`gen`) position sub-masks.
  - the **4-stream** extraction in `sample_vit_embed` (the `else` branch, `feature_type =
    masked_tgt_embed_with_qwen_txt_vit_tokens`): `wtxt_wvit` = cond contexts; `wtxt_wovit` = cond[txt];
    `wotxt_wvit` = cond[vit]; `wotxt_wovit` = uncond[txt].

Classes copied **verbatim** from `_vendor/bernini/bernini/models/bernini.py` (RMSNorm + MLPConnector +
the two model methods) with a tiny synthetic connector + random f32 hidden states, so the oracle is the
reference. Validates the mask selection + the `for_gen` integration end-to-end (f32).

Run:
  ~/Repos/mflux/.venv/bin/python tools/dump_bernini_handoff_golden.py
Fixture -> mlx-gen-bernini/tests/fixtures/handoff_golden.safetensors
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "mlx-gen-bernini", "tests", "fixtures", "handoff_golden.safetensors")

HIDDEN = 8       # planner hidden / connector in
GEN = 12         # connector for_gen out (renderer prompt-embed width)
NUM_MASK = 4     # mask_tokens parameter length (only [:, :1] used)
L_COND = 11
L_UNCOND = 9


# ===== verbatim reference: RMSNorm + MLPConnector (bernini.py) =====
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


class MLPConnector(nn.Module):
    def __init__(self, in_dim, out_dim_for_gen, out_dim_for_vit):
        super().__init__()
        self.proj_gen = nn.Sequential(
            nn.Linear(in_dim, out_dim_for_gen), nn.GELU(),
            RMSNorm(out_dim_for_gen), nn.Linear(out_dim_for_gen, out_dim_for_gen),
        )
        self.pred_vit = nn.Sequential(
            nn.Linear(in_dim, out_dim_for_vit), nn.GELU(), nn.Linear(out_dim_for_vit, out_dim_for_vit),
            RMSNorm(out_dim_for_vit), nn.Linear(out_dim_for_vit, out_dim_for_vit),
        )

    def for_gen(self, x):
        return self.proj_gen(x)


# ===== verbatim reference: the two model methods (bernini.py) =====
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.connector = MLPConnector(HIDDEN, GEN, HIDDEN)
        self.mask_tokens = nn.Parameter(torch.randn(1, NUM_MASK, HIDDEN))

    def post_process_input_embeds(self, input_embeds, visual_output_mask, tgt_vit_mask, inference=False):
        target_vit_embed_mask = visual_output_mask.squeeze(0)
        target_vit_embeds = input_embeds[:, target_vit_embed_mask, :]
        target_vit_embeds_gt = target_vit_embeds.clone()
        mask_token = self.mask_tokens[:, :1]
        if inference:
            all_vit_token_num = sum(target_vit_embed_mask).detach().cpu().numpy()
            target_vit_embeds[:, :, :] = mask_token.expand(1, all_vit_token_num, -1)
            input_embeds[:, target_vit_embed_mask, :] = target_vit_embeds
            diff_loss_mask = torch.ones(all_vit_token_num)
        return dict(input_embeds=input_embeds, diff_loss_mask=diff_loss_mask,
                    target_vit_embeds=target_vit_embeds_gt)

    def feat_from_planner_to_renderer(self, hidden_states, tgt_vit_mask, visual_output_mask, inference=False):
        pred_vit_embed_mask = visual_output_mask.squeeze(0)
        pred_vit_embeds = hidden_states[:, pred_vit_embed_mask, :].clone()
        txt_and_vit_token_mask = visual_output_mask.squeeze(0).logical_not()
        cond_embed_mask = (txt_and_vit_token_mask | pred_vit_embed_mask)
        diff_mllm_context_txt_mask = txt_and_vit_token_mask[cond_embed_mask]
        diff_mllm_context_vit_mask = pred_vit_embed_mask[cond_embed_mask]
        diff_mllm_contexts = hidden_states[:, cond_embed_mask, :]
        diff_mllm_contexts = self.connector.for_gen(diff_mllm_contexts)
        return dict(diff_mllm_contexts=diff_mllm_contexts,
                    diff_mllm_context_txt_mask=diff_mllm_context_txt_mask,
                    diff_mllm_context_vit_mask=diff_mllm_context_vit_mask)


def gen_mask(length, gen_positions):
    m = torch.zeros(1, length, dtype=torch.bool)
    for p in gen_positions:
        m[0, p] = True
    return m


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    model = Model().to(torch.float32).eval()

    # cond: gen slots at the tail (a vision_start + pads + vision_end pattern, gen = the pad run).
    cond_gen = gen_mask(L_COND, [7, 8, 9])
    uncond_gen = gen_mask(L_UNCOND, [5, 6, 7])

    cond_hidden = torch.randn(1, L_COND, HIDDEN)
    uncond_hidden = torch.randn(1, L_UNCOND, HIDDEN)
    # input_embeds before masking (post_process overwrites the gen slots).
    cond_input = torch.randn(1, L_COND, HIDDEN)

    pp = model.post_process_input_embeds(cond_input.clone(), cond_gen, None, inference=True)
    cond_out = model.feat_from_planner_to_renderer(cond_hidden, None, cond_gen, inference=True)
    uncond_out = model.feat_from_planner_to_renderer(uncond_hidden, None, uncond_gen, inference=True)

    txt_mask = cond_out["diff_mllm_context_txt_mask"]
    vit_mask = cond_out["diff_mllm_context_vit_mask"]
    uncond_txt_mask = uncond_out["diff_mllm_context_txt_mask"]
    wtxt_wvit = cond_out["diff_mllm_contexts"]
    wtxt_wovit = wtxt_wvit[:, txt_mask]
    wotxt_wvit = wtxt_wvit[:, vit_mask]
    wotxt_wovit = uncond_out["diff_mllm_contexts"][:, uncond_txt_mask]

    out = {}
    for k, v in model.state_dict().items():
        out[f"model.{k}"] = v.contiguous()
    out["io.cond_hidden"] = cond_hidden.contiguous()
    out["io.uncond_hidden"] = uncond_hidden.contiguous()
    out["io.cond_input"] = cond_input.contiguous()
    out["io.cond_gen_mask"] = cond_gen.squeeze(0).to(torch.int32).contiguous()
    out["io.uncond_gen_mask"] = uncond_gen.squeeze(0).to(torch.int32).contiguous()
    out["out.post_processed"] = pp["input_embeds"].contiguous()
    out["out.wtxt_wvit"] = wtxt_wvit.contiguous()
    out["out.wtxt_wovit"] = wtxt_wovit.contiguous()
    out["out.wotxt_wvit"] = wotxt_wvit.contiguous()
    out["out.wotxt_wovit"] = wotxt_wovit.contiguous()

    meta = {"hidden": str(HIDDEN), "gen": str(GEN), "num_mask": str(NUM_MASK),
            "l_cond": str(L_COND), "l_uncond": str(L_UNCOND)}
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    save_file(out, FIXTURE, metadata=meta)
    print(f"wrote {FIXTURE}  ({len(out)} tensors)")
    print(f"  wtxt_wvit {tuple(wtxt_wvit.shape)} wtxt_wovit {tuple(wtxt_wovit.shape)} "
          f"wotxt_wvit {tuple(wotxt_wvit.shape)} wotxt_wovit {tuple(wotxt_wovit.shape)}")


if __name__ == "__main__":
    main()
