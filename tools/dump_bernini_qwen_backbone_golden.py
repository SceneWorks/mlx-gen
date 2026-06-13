"""sc-5132: synthetic-fixture golden for the Bernini planner's Qwen2.5-VL-7B text backbone.

Builds a *small* but structurally-faithful Qwen2.5-VL text decoder (GQA, QKV-bias, no q/k-norm,
3D multimodal RoPE) with random f32 weights, runs the **reference** forward, and dumps weights +
inputs + the penultimate hidden state + the assembled MRoPE cos/sin to a safetensors fixture the
Rust parity test loads. This exercises the full forward math — MRoPE channel stitch, GQA, the
external additive 4D mask, the residual stack, and the HF `hidden_states[-2]` tap — without the
14 GB checkpoint, in float32 for a clean near-bit comparison.

The math functions (`rotate_half`, `apply_multimodal_rotary_pos_emb`, the rotary table, RMSNorm,
`repeat_kv`, the eager attention, SwiGLU MLP) are copied **verbatim** from
`_vendor/bernini/bernini/models/modeling_qwen2_5_vl.py` (architecturally stock HF Qwen2.5-VL), so the
oracle is the reference, not a reinterpretation. Self-contained (torch + safetensors only) so it runs
in a modern env without importing the vendored 4.57-pinned module.

Run:
  ~/Repos/mflux/.venv/bin/python tools/dump_bernini_qwen_backbone_golden.py
Fixture -> mlx-gen-bernini/tests/fixtures/qwen_backbone_golden.safetensors
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(
    REPO_ROOT, "mlx-gen-bernini", "tests", "fixtures", "qwen_backbone_golden.safetensors"
)

# --- tiny but structurally faithful config (fixture stays < 1 MB) ---
HIDDEN = 64
LAYERS = 2
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
INTER = 128
EPS = 1e-6
THETA = 1_000_000.0
MROPE = [2, 3, 3]  # sum*2 == HEAD_DIM
SEQ = 6


# ===== reference math, copied verbatim from modeling_qwen2_5_vl.py =====
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotary_cos_sin(position_ids):
    # Qwen2_5_VLRotaryEmbedding: inv_freq[j] = THETA^(-2j/HEAD_DIM); emb = cat(freqs, freqs).
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    # position_ids: (3, L) -> (3, bs=1, L); follow the reference matmul shaping exactly.
    pos = position_ids[:, None, :].float()  # (3, 1, L)
    inv_exp = inv_freq[None, None, :, None].expand(3, 1, half, 1)  # (3,1,half,1)
    pos_exp = pos[:, :, None, :]  # (3,1,1,L)
    freqs = (inv_exp @ pos_exp).transpose(2, 3)  # (3,1,half,L) -> (3,1,L,half)
    emb = torch.cat((freqs, freqs), dim=-1)  # (3,1,L,HEAD_DIM)
    return emb.cos(), emb.sin()  # each (3,1,L,HEAD_DIM)


def rms_norm(x, w):
    dt = x.dtype
    x = x.to(torch.float32)
    v = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(v + EPS)
    return w * x.to(dt)


def repeat_kv(hidden_states, n_rep):
    b, kvh, s, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, kvh, n_rep, s, d)
    return hidden_states.reshape(b, kvh * n_rep, s, d)


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    weights = {}

    def lin(prefix, out_f, in_f, bias):
        w = rand(out_f, in_f) * 0.05
        weights[f"{prefix}.weight"] = w
        if bias:
            weights[f"{prefix}.bias"] = rand(out_f) * 0.05

    weights["model.embed_tokens.weight"] = rand(HIDDEN, HIDDEN) * 0.05  # present; unused (embeds path)
    weights["model.norm.weight"] = torch.ones(HIDDEN) + rand(HIDDEN) * 0.02
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        weights[f"{p}.input_layernorm.weight"] = torch.ones(HIDDEN) + rand(HIDDEN) * 0.02
        weights[f"{p}.post_attention_layernorm.weight"] = torch.ones(HIDDEN) + rand(HIDDEN) * 0.02
        lin(f"{p}.self_attn.q_proj", HEADS * HEAD_DIM, HIDDEN, True)
        lin(f"{p}.self_attn.k_proj", KV_HEADS * HEAD_DIM, HIDDEN, True)
        lin(f"{p}.self_attn.v_proj", KV_HEADS * HEAD_DIM, HIDDEN, True)
        lin(f"{p}.self_attn.o_proj", HIDDEN, HEADS * HEAD_DIM, False)
        lin(f"{p}.mlp.gate_proj", INTER, HIDDEN, False)
        lin(f"{p}.mlp.up_proj", INTER, HIDDEN, False)
        lin(f"{p}.mlp.down_proj", HIDDEN, INTER, False)

    embeds = rand(1, SEQ, HIDDEN)
    # Vision-like position ids: 2 text tokens (all rows equal) + a 2x2 vision block at temporal 2.
    temporal = torch.tensor([0, 1, 2, 2, 2, 2])
    height = torch.tensor([0, 1, 0, 0, 1, 1])
    width = torch.tensor([0, 1, 0, 1, 0, 1])
    position_ids = torch.stack([temporal, height, width], dim=0)  # (3, SEQ)

    # External additive 4D causal mask [1,1,S,S] (0 / -inf) — a representative non-trivial mask.
    neg = torch.finfo(torch.float32).min
    mask = torch.full((SEQ, SEQ), neg)
    mask = torch.triu(mask, diagonal=1)[None, None]  # causal

    # ----- reference forward (output_hidden_states semantics) -----
    cos3, sin3 = rotary_cos_sin(position_ids)  # (3,1,S,HEAD_DIM)

    # Assembled (stitched) cos/sin the apply uses — dump for the MRoPE golden, shaped [1,S,HEAD_DIM].
    sect = MROPE * 2
    cos_stitch = torch.cat([m[i % 3] for i, m in enumerate(cos3.split(sect, dim=-1))], dim=-1)  # (1,S,HD)
    sin_stitch = torch.cat([m[i % 3] for i, m in enumerate(sin3.split(sect, dim=-1))], dim=-1)

    all_hidden = []
    hidden = embeds
    for i in range(LAYERS):
        all_hidden.append(hidden)
        p = f"model.layers.{i}"
        residual = hidden
        x = rms_norm(hidden, weights[f"{p}.input_layernorm.weight"])
        # attention
        q = F.linear(x, weights[f"{p}.self_attn.q_proj.weight"], weights[f"{p}.self_attn.q_proj.bias"])
        k = F.linear(x, weights[f"{p}.self_attn.k_proj.weight"], weights[f"{p}.self_attn.k_proj.bias"])
        v = F.linear(x, weights[f"{p}.self_attn.v_proj.weight"], weights[f"{p}.self_attn.v_proj.bias"])
        q = q.view(1, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(1, SEQ, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(1, SEQ, KV_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos3, sin3, MROPE)
        k = repeat_kv(k, HEADS // KV_HEADS)
        v = repeat_kv(v, HEADS // KV_HEADS)
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(HEAD_DIM)
        attn = attn + mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(1, SEQ, HEADS * HEAD_DIM)
        out = F.linear(out, weights[f"{p}.self_attn.o_proj.weight"])
        hidden = residual + out
        # mlp
        residual = hidden
        x = rms_norm(hidden, weights[f"{p}.post_attention_layernorm.weight"])
        gate = F.linear(x, weights[f"{p}.mlp.gate_proj.weight"])
        up = F.linear(x, weights[f"{p}.mlp.up_proj.weight"])
        x = F.linear(F.silu(gate) * up, weights[f"{p}.mlp.down_proj.weight"])
        hidden = residual + x
    final = rms_norm(hidden, weights["model.norm.weight"])
    all_hidden.append(final)
    # all_hidden = [embeds, layer0_out, ..., layer_{N-2}_out, final]; [-2] = layer_{N-2}_out.
    penultimate = all_hidden[-2]

    out = {f"w.{k}": v.contiguous() for k, v in weights.items()}
    out["io.embeds"] = embeds.contiguous()
    out["io.position_ids"] = position_ids.to(torch.int32).contiguous()
    out["io.mask"] = mask.contiguous()
    out["out.penultimate"] = penultimate.contiguous()
    out["out.cos"] = cos_stitch.contiguous()
    out["out.sin"] = sin_stitch.contiguous()

    meta = {
        "hidden_size": str(HIDDEN),
        "num_hidden_layers": str(LAYERS),
        "num_attention_heads": str(HEADS),
        "num_key_value_heads": str(KV_HEADS),
        "head_dim": str(HEAD_DIM),
        "intermediate_size": str(INTER),
        "rms_norm_eps": repr(EPS),
        "rope_theta": repr(THETA),
        "mrope_section": ",".join(str(x) for x in MROPE),
        "seq_len": str(SEQ),
    }

    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    save_file(out, FIXTURE, metadata=meta)
    print(f"wrote {FIXTURE}")
    print(f"  tensors: {len(out)}  penultimate {tuple(penultimate.shape)}  "
          f"|penultimate|max {penultimate.abs().max().item():.4f}")


if __name__ == "__main__":
    main()
