"""Dump the FLUX.2-dev **Fun-Controlnet-Union** (VACE) control-branch numeric golden for the Rust
port (sc-8978, epic 8236) — the FLUX.2 sibling of the Qwen sc-8335 harness (mlx-gen #628).

Reference = the **authoritative VideoX-Fun** `Flux2ControlTransformer2DModel`
(`alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union`), the exact upstream `mlx-gen-flux2` cites as its
port source (`src/transformer.rs` ~L901). The mflux **MLX** fork has no FLUX.2 VACE control
transformer (and is frozen), so this reference is the upstream **torch** code — a *cross-framework*
golden (torch-fp32 → MLX), not bit-exact; the Rust test gates on a tolerance that covers Metal's
reduced-precision matmul across the block forwards.

To keep the reference authoritative but importable without VideoX-Fun's heavy runtime deps (`..dist`,
`.attention_utils`'s flash-attn backend, `.cache_utils`), this vendors a **minimal, faithful copy**
of only the pieces the control forward actually touches — verbatim from
`videox_fun/models/flux2_transformer2d{,_control}.py` @ github.com/aigc-apps/VideoX-Fun (main),
Apache-2.0 — with the single flash-attn backend call `attention()` swapped for the mathematically
identical `F.scaled_dot_product_attention` (fp32). Unlike the Qwen sibling, FLUX.2's block / attention
/ FeedForward / RoPE are **custom** classes (`Flux2Attention`, `Flux2FeedForward`, `Flux2SwiGLU`,
`apply_rotary_emb`), so nothing from diffusers is reused — the whole block is vendored here. The math
is otherwise the upstream code, run on a **tiny synthetic** control branch (small dims, non-zero-init
before/after_proj so the control path genuinely contributes).

Dumps (committed, tiny) → `mlx-gen-flux2/tests/fixtures/flux2_fun_control.safetensors`:
  * GAP 1 (forward): the control-branch weights (checkpoint key names — the Rust test drives them
    through the same `attn.to_out.0` → `attn.to_out` alias the production loader applies), the fixed
    `forward_control` inputs (post-embedder img/txt streams, the packed control context, the shared
    double-stream modulation params, the interleaved-RoPE cos/sin), and the reference per-block hints.
  * GAP 2 (context fill): a synthetic (post-patchify+BN) control latent + the reference
    `pipeline_flux2_control` packed 260-analog context (`_pack_latents` + zero-mask/zero-inpaint
    concat) — the Rust test byte-confirms the production `fun_control_context_from_latents`.

RoPE convention (verified vs the MLX `apply_rope`): FLUX.2 uses the diffusers *real* interleaved path
(`apply_rotary_emb(use_real=True, use_real_unbind_dim=-1, sequence_dim=1)`) over repeat-interleaved
cos/sin `[S, head_dim]` — the standard complex-pair rotation, identical to MLX's interleaved
`apply_rope` over cos/sin `[S, head_dim/2]`. So this dumps cos/sin at `[S, head_dim/2]` for MLX and
repeat-interleaves them to `[S, head_dim]` for the reference. (This is NOT Qwen's complex-`torch.polar`
path.)

Note: the MLX qk-RMSNorm eps is 1e-5 vs the upstream 1e-6 (this reference); with O(1) inputs that shifts
the norm denominator by ~5e-6, inconsequential under the 1e-2 cross-framework tolerance (the base
FLUX.2 transformer is already parity-proven at 1e-5). The reference here uses the upstream 1e-6.

Run from a torch venv (the mflux fork's venv works — no diffusers needed):
    /Users/michael/Repos/mflux/.venv/bin/python tools/dump_flux2_fun_control_golden.py

This is a dev-only regeneration step; CI consumes the committed fixture and needs no torch/network.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.numpy import save_file

torch.manual_seed(0)

# ----------------------------------------------------------------------------------------------
# Vendored (verbatim, Apache-2.0) from VideoX-Fun flux2_transformer2d{,_control}.py, trimmed to the
# control forward's dependencies.
# ----------------------------------------------------------------------------------------------


def attention(q, k, v, attn_mask=None):
    """Stand-in for `videox_fun.models.attention_utils.attention` (a flash-attn backend wrapper).

    Same contract as the `Flux2AttnProcessor` call site: q/k/v are `[B, S, H, D]` (heads
    un-transposed); returns `[B, S, H, D]`. The upstream default backend is Flash-Attention with the
    standard `1/sqrt(D)` scale and no causal mask for this DiT — `F.scaled_dot_product_attention` is
    numerically identical for our fp32 fixture. Transpose to `[B, H, S, D]` for SDPA, then back.
    """
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return out.transpose(1, 2)


def apply_rotary_emb(x, freqs_cis, use_real=True, use_real_unbind_dim=-1, sequence_dim=2):
    # Verbatim from flux2_transformer2d.py (the control forward uses use_real=True, unbind=-1, seq_dim=1).
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        if sequence_dim == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif sequence_dim == 1:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        else:
            raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


class Flux2SwiGLU(nn.Module):
    # Verbatim from flux2_transformer2d.py.
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        x = self.gate_fn(x1) * x2
        return x


class Flux2FeedForward(nn.Module):
    # Verbatim from flux2_transformer2d.py (SwiGLU with the gate projection fused into linear_in).
    def __init__(self, dim, dim_out=None, mult=3.0, inner_dim=None, bias=False):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.act_fn = Flux2SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x


class Flux2AttnProcessor:
    # Verbatim from flux2_transformer2d.py, trimmed to the standard (non-parallel-self-attn) joint
    # img+txt path the double block uses (the parallel-self-attn branch is the single-stream block,
    # not on the control path).
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("requires PyTorch 2.0")

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

        encoder_query = attn.norm_added_q(encoder_query)
        encoder_key = attn.norm_added_k(encoder_key)

        # [txt, img] order along the sequence axis (axis 1 in B S H D).
        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = attention(query, key, value, attn_mask=attention_mask)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
            [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
        )
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states, encoder_hidden_states


class Flux2Attention(torch.nn.Module):
    # Verbatim from flux2_transformer2d.py (joint dual-stream attention), trimmed of processor
    # (de)registration plumbing — the double block always uses added_kv_proj_dim + Flux2AttnProcessor.
    def __init__(self, query_dim, heads=8, dim_head=64, bias=False, added_kv_proj_dim=None,
                 added_proj_bias=True, out_bias=True, eps=1e-5, out_dim=None, elementwise_affine=True):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(0.0))

        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        self.processor = Flux2AttnProcessor()

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)


class Flux2TransformerBlock(nn.Module):
    # Verbatim from flux2_transformer2d.py (the base dual-stream MMDiT block).
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=3.0, eps=1e-6, bias=False):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux2Attention(
            query_dim=dim, added_kv_proj_dim=dim, dim_head=attention_head_dim, heads=num_attention_heads,
            out_dim=dim, bias=bias, added_proj_bias=bias, out_bias=bias, eps=eps,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

    def forward(self, hidden_states, encoder_hidden_states, temb_mod_params_img, temb_mod_params_txt,
                image_rotary_emb=None, joint_attention_kwargs=None):
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        return encoder_hidden_states, hidden_states


class Flux2ControlTransformerBlock(Flux2TransformerBlock):
    # Verbatim from flux2_transformer2d_control.py.
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=3.0, eps=1e-6, bias=False, block_id=0):
        super().__init__(dim, num_attention_heads, attention_head_dim, mlp_ratio, eps, bias)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        encoder_hidden_states, c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return encoder_hidden_states, c


def forward_control(control_img_in, control_blocks, x, control_context, kwargs):
    # Verbatim loop from Flux2ControlTransformer2DModel.forward_control.
    c = control_img_in(control_context)
    new_kwargs = dict(x=x)
    new_kwargs.update(kwargs)
    for block in control_blocks:
        encoder_hidden_states, c = block(c, **new_kwargs)
        new_kwargs["encoder_hidden_states"] = encoder_hidden_states
    hints = torch.unbind(c)[:-1]
    return hints


# ---- GAP 2: the packed control context (verbatim from pipeline_flux2_control.py) -----------------
def _patchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
    return latents


def _pack_latents(latents):
    # (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
    return latents


# ----------------------------------------------------------------------------------------------
# Tiny fixture geometry (mirrored by mlx-gen-flux2/tests/fun_control_parity.rs).
# ----------------------------------------------------------------------------------------------
HEADS, HEAD_DIM = 2, 8
DIM = HEADS * HEAD_DIM  # 16 (inner_dim)
CONTROL_LAYERS = [0, 2, 4]  # 3 control blocks (num_double_layers=6 → step_by(2)); block_id 0 → before_proj
IMG_SEQ, TXT_SEQ = 5, 3
# GAP 2: tiny latent geometry. real dev: num_latent_channels=32, in_channels=128, mask=4 → 260.
LAT_C = 4  # tiny num_latent_channels
IN_CH = LAT_C * 4  # 16 packed control-latent channels (= DIM here, coincidental)
MASK_CH = IN_CH // LAT_C  # 4 (the 2×2 patch of one mask channel)
CONTROL_IN = IN_CH + MASK_CH + IN_CH  # 36 — the tiny 260-analog packed control-context width
LH, LW = 2, 3  # (post-patchify) control-latent grid → seq = LH·LW = 6


def randf(*shape):
    return torch.randn(*shape, dtype=torch.float32)


def main():
    # --- Control branch (real reference classes, tiny synthetic weights) ---
    control_img_in = nn.Linear(CONTROL_IN, DIM)
    control_blocks = nn.ModuleList(
        [
            Flux2ControlTransformerBlock(
                dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM,
                mlp_ratio=3.0, eps=1e-6, bias=False, block_id=cl,
            )
            for cl in CONTROL_LAYERS
        ]
    )
    # Perturb the zero-init before/after_proj so the control path genuinely contributes (a fresh VACE
    # branch zero-inits them → hints would be all-zero and the golden vacuous), exactly as a trained
    # Fun-Controlnet checkpoint does. Mirrors the Qwen sibling dump_qwen_fun_control_golden.py.
    for blk in control_blocks:
        with torch.no_grad():
            blk.after_proj.weight.copy_(0.1 * torch.randn_like(blk.after_proj.weight))
            blk.after_proj.bias.copy_(0.1 * torch.randn_like(blk.after_proj.bias))
            if hasattr(blk, "before_proj"):
                blk.before_proj.weight.copy_(0.1 * torch.randn_like(blk.before_proj.weight))
                blk.before_proj.bias.copy_(0.1 * torch.randn_like(blk.before_proj.bias))
    control_img_in.eval()
    control_blocks.eval()

    # --- Fixed forward_control inputs ---
    img_embed = randf(1, IMG_SEQ, DIM)  # post-x_embedder base image stream (x)
    txt_embed = randf(1, TXT_SEQ, DIM)  # post-context_embedder base text stream (seeds control txt)
    control_context = randf(1, IMG_SEQ, CONTROL_IN)  # packed control context (control_img_in input)

    # Shared double-stream modulation params: each of img/txt is 2 sets of (shift, scale, gate), each
    # [1, 1, DIM]. Kept small so (1+scale) stays ~1 and the modulated norms are O(1). The FLUX.2 base
    # modulation is shared across all double blocks (the control blocks reuse it, per the fork), so
    # these are passed straight into forward_control — no modulation weight layer needed.
    def mod_set():
        return tuple((0.2 * randf(1, 1, DIM), 0.2 * randf(1, 1, DIM), 0.2 * randf(1, 1, DIM)) for _ in range(2))

    img_mod = mod_set()
    txt_mod = mod_set()

    # Interleaved real RoPE: random per-position angles θ over the concatenated [txt; img] sequence.
    # MLX consumes cos/sin at [S, head_dim/2]; the reference apply_rotary_emb (use_real=True) consumes
    # them repeat-interleaved to [S, head_dim] — the same interleaved complex-pair rotation.
    half = HEAD_DIM // 2
    theta = randf(TXT_SEQ + IMG_SEQ, half)
    cos_half = torch.cos(theta)  # [S, head_dim/2]  → MLX
    sin_half = torch.sin(theta)
    cos_full = torch.repeat_interleave(cos_half, 2, dim=-1)  # [S, head_dim] → reference
    sin_full = torch.repeat_interleave(sin_half, 2, dim=-1)

    kwargs = dict(
        encoder_hidden_states=txt_embed,
        temb_mod_params_img=img_mod,
        temb_mod_params_txt=txt_mod,
        image_rotary_emb=(cos_full, sin_full),
        joint_attention_kwargs=None,
    )

    with torch.no_grad():
        hints = forward_control(control_img_in, control_blocks, img_embed, control_context, kwargs)
    assert len(hints) == len(CONTROL_LAYERS)
    max_hint = max(float(h.abs().max()) for h in hints)
    assert max_hint > 1e-2, f"hints are ~zero (perturbation too small): {max_hint:.2e}"
    print(f"reference hints: {len(hints)} × {tuple(hints[0].shape)}  max|hint|={max_hint:.4f}")

    # --- GAP 2: reference packed control context (pose: zero mask + zero inpaint) ---
    # control_lat is the (post-patchify + BN) control latent [1, IN_CH, LH, LW] the MLX helper packs.
    control_lat = randf(1, IN_CH, LH, LW)
    control_part = _pack_latents(control_lat)  # [1, LH·LW, IN_CH]
    # Pose has no mask image → mask_condition = 1 - ones = 0 → patchify → pack; all-zero, width MASK_CH.
    mask_zero = torch.zeros(1, 1, 2 * LH, 2 * LW)
    mask_part = _pack_latents(_patchify_latents(mask_zero))  # [1, LH·LW, MASK_CH]
    # Pose has no inpaint image → inpaint latent = zeros → patchify → pack; all-zero, width IN_CH.
    inpaint_zero = torch.zeros(1, LAT_C, 2 * LH, 2 * LW)
    inpaint_part = _pack_latents(_patchify_latents(inpaint_zero))  # [1, LH·LW, IN_CH]
    context_ref = torch.concat([control_part, mask_part, inpaint_part], dim=2)  # [1, seq, CONTROL_IN]
    assert context_ref.shape == (1, LH * LW, CONTROL_IN), context_ref.shape
    assert mask_part.shape[-1] == MASK_CH and inpaint_part.shape[-1] == IN_CH

    # --- Assemble the fixture: control weights (checkpoint key names) + IO + goldens ---
    out = {}
    out["control_img_in.weight"] = control_img_in.weight.detach().numpy()
    out["control_img_in.bias"] = control_img_in.bias.detach().numpy()
    for pos, blk in enumerate(control_blocks):
        for k, v in blk.state_dict().items():
            out[f"control_transformer_blocks.{pos}.{k}"] = v.float().cpu().numpy()

    out["in.img_embed"] = img_embed.numpy()
    out["in.txt_embed"] = txt_embed.numpy()
    out["in.control_context"] = control_context.numpy()
    out["in.cos"] = cos_half.contiguous().numpy()  # [S, head_dim/2]
    out["in.sin"] = sin_half.contiguous().numpy()
    for si, (shift, scale, gate) in enumerate(img_mod):
        out[f"in.img_mod_shift_{si}"] = shift.numpy()
        out[f"in.img_mod_scale_{si}"] = scale.numpy()
        out[f"in.img_mod_gate_{si}"] = gate.numpy()
    for si, (shift, scale, gate) in enumerate(txt_mod):
        out[f"in.txt_mod_shift_{si}"] = shift.numpy()
        out[f"in.txt_mod_scale_{si}"] = scale.numpy()
        out[f"in.txt_mod_gate_{si}"] = gate.numpy()
    for i, h in enumerate(hints):
        out[f"out.hint_{i}"] = h.contiguous().numpy()

    out["ctx.control_lat"] = control_lat.contiguous().numpy()  # [1, IN_CH, LH, LW]
    out["ctx.pack_ref"] = context_ref.contiguous().numpy()  # [1, LH·LW, CONTROL_IN]

    fixtures = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "mlx-gen-flux2", "tests", "fixtures"
    )
    os.makedirs(fixtures, exist_ok=True)
    path = os.path.abspath(os.path.join(fixtures, "flux2_fun_control.safetensors"))
    save_file(
        {k: np.ascontiguousarray(v) for k, v in out.items()},
        path,
        metadata={
            "heads": str(HEADS),
            "head_dim": str(HEAD_DIM),
            "control_layers": ",".join(map(str, CONTROL_LAYERS)),
            "control_in": str(CONTROL_IN),
            "lat_c": str(LAT_C),
            "in_ch": str(IN_CH),
            "lh": str(LH),
            "lw": str(LW),
            "reference": "aigc-apps/VideoX-Fun flux2_transformer2d_control + pipeline_flux2_control (Apache-2.0)",
        },
    )
    print(f"wrote {path} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
