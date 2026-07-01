"""Dump the Qwen-Image **2512-Fun-Controlnet-Union** (VACE) control-branch numeric golden for the
Rust port (sc-8335, epic 8236).

Reference = the **authoritative VideoX-Fun** `QwenImageControlTransformer2DModel` (alibaba-pai's
`Qwen-Image-2512-Fun-Controlnet-Union`), the exact upstream that replaced the retired InstantX
`QwenImageControlNetModel` on the Qwen control path (sc-8267). Unlike the base Qwen goldens (dumped
from the frozen mflux **MLX** fork), the fork has **no Qwen VACE control transformer** and is frozen,
so this reference is the upstream **torch** code — a *cross-framework* golden (torch-fp32 → MLX), not
bit-exact; the Rust test gates on a tolerance that covers Metal's reduced-precision matmul.

To keep the reference authoritative but importable without VideoX-Fun's heavy runtime deps
(`..dist`, `.attention_utils`'s flash-attn backend, `.cache_utils` TeaCache, fp8), this vendors a
**minimal, faithful copy** of the pieces the control forward actually touches — verbatim from
`videox_fun/models/qwenimage_transformer2d{,_control}.py` @ github.com/aigc-apps/VideoX-Fun (main),
Apache-2.0 — with the single flash-attn backend call `attention()` swapped for the mathematically
identical `F.scaled_dot_product_attention`. The block/attention/RoPE math is otherwise the upstream
code, run on a **tiny synthetic** control branch (small dims, non-zero-init before/after_proj so the
control path genuinely contributes) exactly like the Z-Image sibling `dump_z_control_transformer.py`.

Dumps (committed, tiny) → `mlx-gen-qwen-image/tests/fixtures/qwen_fun_control.safetensors`:
  * GAP 1 (forward): the control-branch weights (diffusers/checkpoint key names — the Rust test reuses
    the real `remap_transformer_keys`), the fixed `forward_control` inputs (img/text embeds, packed
    control context, temb, interleaved-RoPE cos/sin), and the reference per-block hints.
  * GAP 2 (context fill): a synthetic 16-ch control latent + the reference
    `pipeline_qwenimage_control._pack_latents([control_latents | mask(1) | inpaint(16)])` packed
    132-ch context — the Rust test byte-confirms the production channel-order/fill + 2×2 pack.

Run from a torch venv with diffusers (the mflux fork's venv works):
    /Users/michael/Repos/mflux/.venv/bin/python tools/dump_qwen_fun_control_golden.py

This is a dev-only regeneration step; CI consumes the committed fixture and needs no torch/network.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from safetensors.numpy import save_file

torch.manual_seed(0)

# ----------------------------------------------------------------------------------------------
# Vendored (verbatim, Apache-2.0) from VideoX-Fun, trimmed to the control forward's dependencies.
# ----------------------------------------------------------------------------------------------


def attention(q, k, v, attn_mask=None, dropout_p=0.0, causal=False):
    """Stand-in for `videox_fun.models.attention_utils.attention` (a flash-attn backend wrapper).

    Same contract: q/k/v are `[B, S, H, D]` (heads un-transposed); returns `[B, S, H, D]`. The
    upstream default backend is Flash-Attention with the standard `1/sqrt(D)` scale and no causal
    mask for this DiT — `F.scaled_dot_product_attention` is numerically identical for our fp32
    fixture. Transpose to `[B, H, S, D]` for SDPA, then back.
    """
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=causal)
    return out.transpose(1, 2)


def apply_rotary_emb_qwen(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
    # Verbatim from qwenimage_transformer2d.py (the control forward uses the `use_real=False` path).
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)
        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
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
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


class QwenDoubleStreamAttnProcessor2_0:
    # Verbatim from qwenimage_transformer2d.py (the joint dual-stream attention processor).
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("requires PyTorch 2.0")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        if encoder_hidden_states is None:
            raise ValueError("requires encoder_hidden_states (text stream)")
        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_hidden_states = attention(
            joint_query, joint_key, joint_value, attn_mask=attention_mask, dropout_p=0.0, causal=False
        )
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output)
        return img_attn_output, txt_attn_output


class QwenImageTransformerBlock(nn.Module):
    # Verbatim from qwenimage_transformer2d.py (the base dual-stream MMDiT block).
    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6, zero_cond_t=False):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.zero_cond_t = zero_cond_t

    def _modulate(self, x, mod_params, index=None):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if index is not None:
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]
            index_expanded = index.unsqueeze(-1)
            shift_result = torch.where(index_expanded == 0, shift_0.unsqueeze(1), shift_1.unsqueeze(1))
            scale_result = torch.where(index_expanded == 0, scale_0.unsqueeze(1), scale_1.unsqueeze(1))
            gate_result = torch.where(index_expanded == 0, gate_0.unsqueeze(1), gate_1.unsqueeze(1))
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)
        return x * (1 + scale_result) + shift_result, gate_result

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        temb,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        modulate_index=None,
    ):
        img_mod_params = self.img_mod(temb)
        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_params = self.txt_mod(temb)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, modulate_index)
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        img_attn_output, txt_attn_output = attn_output

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, modulate_index)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output
        return encoder_hidden_states, hidden_states


class QwenImageControlTransformerBlock(QwenImageTransformerBlock):
    # Verbatim from qwenimage_transformer2d_control.py.
    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6, zero_cond_t=False, block_id=0):
        super().__init__(dim, num_attention_heads, attention_head_dim, qk_norm, eps, zero_cond_t)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
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
    # Verbatim loop from QwenImageControlTransformer2DModel.forward_control.
    c = control_img_in(control_context)
    new_kwargs = dict(x=x)
    new_kwargs.update(kwargs)
    for block in control_blocks:
        encoder_hidden_states, c = block(c, **new_kwargs)
        new_kwargs["encoder_hidden_states"] = encoder_hidden_states
    hints = torch.unbind(c)[:-1]
    return hints


def pack_latents(latents, batch_size, num_channels_latents, height, width, num_frame=None):
    # Verbatim from pipeline_qwenimage_control._pack_latents.
    if num_frame is None:
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    else:
        latents = latents.view(batch_size, num_channels_latents, num_frame, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 3, 5, 1, 4, 6)
        latents = latents.reshape(batch_size, num_frame * (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


# ----------------------------------------------------------------------------------------------
# Tiny fixture geometry (mirrored by mlx-gen-qwen-image/tests/fun_control_parity.rs).
# ----------------------------------------------------------------------------------------------
HEADS, HEAD_DIM = 2, 8
DIM = HEADS * HEAD_DIM  # 16 (inner_dim)
CONTROL_IN_DIM = 132  # the real packed control-context width ([16 | 1 | 16] × 2×2)
CONTROL_LAYERS = [0, 1, 2]  # 3 control blocks: exercises block-0 seed + the stack/unbind threading
IMG_SEQ, TXT_SEQ = 5, 3
LAT_C = 16  # VAE latent channels (context gap-2)
LH, LW = 2, 3  # packed latent grid → control latent spatial H/8=2*LH, W/8=2*LW


def randf(*shape):
    return torch.randn(*shape, dtype=torch.float32)


def main():
    device = "cpu"  # fp32 CPU reference

    # --- Control branch (real reference classes, tiny synthetic weights) ---
    control_img_in = nn.Linear(CONTROL_IN_DIM, DIM)
    control_blocks = nn.ModuleList(
        [QwenImageControlTransformerBlock(dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM, block_id=i)
         for i in CONTROL_LAYERS]
    )
    # Perturb the zero-init before/after_proj so the control path genuinely contributes (a fresh VACE
    # branch zero-inits them → hints would be all-zero and the golden vacuous), exactly as a trained
    # Fun-Controlnet checkpoint does. Mirrors dump_z_control_transformer.py.
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
    img_embed = randf(1, IMG_SEQ, DIM)  # post-img_in base image stream (x)
    encoder_embed = randf(1, TXT_SEQ, DIM)  # post-txt_in base text stream (seeds the control text branch)
    control_context = randf(1, IMG_SEQ, CONTROL_IN_DIM)  # packed 132-ch control context
    temb = randf(1, DIM)  # raw timestep embedding (blocks apply SiLU internally)

    # Interleaved complex RoPE: random per-position angles θ → unit freqs = cos+ i·sin. The reference
    # consumes the complex freqs (`use_real=False`); MLX consumes cos/sin (cos=Re, sin=Im) — the same
    # interleaved rotation. |freq| = 1 (a proper rotation).
    half = HEAD_DIM // 2
    img_theta = randf(IMG_SEQ, half)
    txt_theta = randf(TXT_SEQ, half)
    img_freqs = torch.polar(torch.ones_like(img_theta), img_theta)
    txt_freqs = torch.polar(torch.ones_like(txt_theta), txt_theta)

    kwargs = dict(
        encoder_hidden_states=encoder_embed,
        encoder_hidden_states_mask=None,
        temb=temb,
        image_rotary_emb=(img_freqs, txt_freqs),
        joint_attention_kwargs=None,
        modulate_index=None,
    )

    with torch.no_grad():
        hints = forward_control(control_img_in, control_blocks, img_embed, control_context, kwargs)
    assert len(hints) == len(CONTROL_LAYERS)
    max_hint = max(float(h.abs().max()) for h in hints)
    assert max_hint > 1e-3, f"hints are ~zero (perturbation too small): {max_hint:.2e}"
    print(f"reference hints: {len(hints)} × {tuple(hints[0].shape)}  max|hint|={max_hint:.4f}")

    # --- GAP 2: reference 132-ch control-context fill + 2×2 pack (pose = zero mask + zero inpaint) ---
    control_latents = randf(1, LAT_C, 1, 2 * LH, 2 * LW)  # [B, 16, F=1, H/8, W/8]
    mask_condition = torch.zeros(1, 1, 1, 2 * LH, 2 * LW)  # 1 - ones = 0 (pose has no mask)
    inpaint_latent = torch.zeros(1, LAT_C, 1, 2 * LH, 2 * LW)  # pose has no inpaint image
    ctx = torch.concat([control_latents, mask_condition, inpaint_latent], dim=1)  # [B, 33, F, H/8, W/8]
    b, cc, cf, ch, cw = ctx.size()
    pack_ref = pack_latents(ctx, b, cc, ch, cw, num_frame=cf)  # [B, LH*LW, 132]
    assert pack_ref.shape == (1, LH * LW, CONTROL_IN_DIM), pack_ref.shape

    # --- Assemble the fixture: control weights (diffusers/checkpoint key names) + IO + goldens ---
    out = {}
    for k, v in control_img_in.state_dict().items():
        out[f"control_img_in.{k}"] = v.float().cpu().numpy()
    for i, blk in enumerate(control_blocks):
        for k, v in blk.state_dict().items():
            out[f"control_blocks.{i}.{k}"] = v.float().cpu().numpy()

    out["in.img_embed"] = img_embed.numpy()
    out["in.encoder_embed"] = encoder_embed.numpy()
    out["in.control_context"] = control_context.numpy()
    out["in.temb"] = temb.numpy()
    out["in.img_cos"] = img_freqs.real.contiguous().numpy()
    out["in.img_sin"] = img_freqs.imag.contiguous().numpy()
    out["in.txt_cos"] = txt_freqs.real.contiguous().numpy()
    out["in.txt_sin"] = txt_freqs.imag.contiguous().numpy()
    for i, h in enumerate(hints):
        out[f"out.hint_{i}"] = h.contiguous().numpy()

    out["ctx.control_latents"] = control_latents.squeeze(2).contiguous().numpy()  # [1, 16, H/8, W/8]
    out["ctx.pack_ref"] = pack_ref.contiguous().numpy()  # [1, LH*LW, 132]

    fixtures = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "mlx-gen-qwen-image", "tests", "fixtures"
    )
    os.makedirs(fixtures, exist_ok=True)
    path = os.path.abspath(os.path.join(fixtures, "qwen_fun_control.safetensors"))
    save_file(
        {k: np.ascontiguousarray(v) for k, v in out.items()},
        path,
        metadata={
            "heads": str(HEADS),
            "head_dim": str(HEAD_DIM),
            "control_in_dim": str(CONTROL_IN_DIM),
            "control_layers": ",".join(map(str, CONTROL_LAYERS)),
            "lat_c": str(LAT_C),
            "lh": str(LH),
            "lw": str(LW),
            "reference": "aigc-apps/VideoX-Fun qwenimage_transformer2d_control (Apache-2.0)",
        },
    )
    print(f"wrote {path} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
