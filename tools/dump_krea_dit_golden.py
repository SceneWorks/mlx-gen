"""Generate the committed Krea 2 DiT parity fixtures for the Rust port (sc-7568).

The reference is the Krea-published `github.com/krea-ai/krea-2` `mmdit.py` (`SingleStreamDiT` — the
authoritative, self-contained, dimension-parametric implementation the diffusers `Krea2Transformer2DModel`
is a faithful port of; the published HF checkpoint keys are the diffusers names this script remaps onto).
Run at TINY dims so the fixtures commit cleanly and Metal fp32 matmul agrees tightly. Random norm
`scale`s + modulation tables (both zero-init in the reference) exercise every weight-multiply path.

Three fixtures, each a single safetensors of f32 tensors (weights under their **diffusers** keys, inputs
under `in.*`, reference output under `out.*`):
  - `single_block_golden`   one `SingleStreamBlock` (DoubleSharedModulation + gated GQA attn + RoPE + SwiGLU)
  - `text_fusion_golden`    the `TextFusionTransformer` (layer-axis aggregation → projector 12→1 → refiner)
  - `dit_golden`            the full `SingleStreamDiT` (img_in, temb, txt_in, 3-axis RoPE build, final layer)
plus `rope_golden` (the reference RoPE cos/sin for the DiT positions, to localize the #1 parity risk).

Run from a torch venv (CPU is fine — the CUDNN-only SDPA + `torch.compile` are monkeypatched out):
    ~/Repos/mflux/.venv/bin/python tools/dump_krea_dit_golden.py
"""

from __future__ import annotations

import sys

import torch

from _paths import REPO_ROOT, fixture

# ── Make the reference importable + CPU-runnable ────────────────────────────────────────────────
# `mmdit.py` decorates several forwards with `@torch.compile(fullgraph=True)` and forces the CUDNN
# SDPA backend — both CUDA-only. Neuter `torch.compile` to an identity decorator BEFORE importing
# (the decorators run at import), and replace the module-level `attention()` with a CPU SDPA after.
torch.compile = lambda model=None, **kw: (model if model is not None else (lambda f: f))  # noqa: E731

sys.path.insert(0, str(REPO_ROOT / "_vendor" / "krea2"))
import mmdit  # noqa: E402
from mmdit import (  # noqa: E402
    PositionalEncoding,
    SingleMMDiTConfig,
    SingleStreamBlock,
    SingleStreamDiT,
    TextFusionTransformer,
)
from einops import rearrange  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _cpu_attention(q, k, v, mask=None, scale=None, gqa=False):
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa)
    return rearrange(x, "B H L D -> B L (H D)")


mmdit.attention = _cpu_attention

torch.manual_seed(0)


def randomize_special(module: torch.nn.Module) -> None:
    """Reference `RMSNorm.scale` / `DoubleSharedModulation.lin` / `SimpleModulation.lin` all init to
    ZERO — which would make the norm weight exactly 1 and the modulation identity, hiding bugs. Give
    them small random values so every `(scale+1)` / modulation path is exercised."""
    for name, p in module.named_parameters():
        if name.endswith(".scale") or name.endswith(".lin"):
            p.data = 0.1 * torch.randn_like(p.data)


# ── mmdit → diffusers key remap ─────────────────────────────────────────────────────────────────
# An attention/FFN sub-block's params (relative to the block) map identically across single-stream and
# text-fusion blocks; only `mod.lin` (single-stream) and the block prefixes differ.
_ATTN_FF = {
    "attn.wq.weight": "attn.to_q.weight",
    "attn.wk.weight": "attn.to_k.weight",
    "attn.wv.weight": "attn.to_v.weight",
    "attn.gate.weight": "attn.to_gate.weight",
    "attn.wo.weight": "attn.to_out.0.weight",
    "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
    "attn.qknorm.knorm.scale": "attn.norm_k.weight",
    "mlp.gate.weight": "ff.gate.weight",
    "mlp.up.weight": "ff.up.weight",
    "mlp.down.weight": "ff.down.weight",
    "prenorm.scale": "norm1.weight",
    "postnorm.scale": "norm2.weight",
}


def remap_block(sd: dict, dst_prefix: str) -> dict:
    """Remap one `{Text,SingleStream}Block` state-dict to diffusers keys under `dst_prefix`. The
    single-stream `mod.lin` `[6·d]` becomes `scale_shift_table` `[6, d]` (row-major reshape preserves
    the `chunk(6)` order)."""
    out = {}
    for k, v in sd.items():
        if k == "mod.lin":
            d = v.shape[0] // 6
            out[f"{dst_prefix}.scale_shift_table"] = v.reshape(6, d).contiguous()
        elif k in _ATTN_FF:
            out[f"{dst_prefix}.{_ATTN_FF[k]}"] = v
        else:
            raise KeyError(f"unmapped block param: {k}")
    return out


def remap_text_fusion(tf: TextFusionTransformer, dst_prefix: str = "text_fusion") -> dict:
    out = {}
    for i, blk in enumerate(tf.layerwise_blocks):
        out.update(remap_block(blk.state_dict(), f"{dst_prefix}.layerwise_blocks.{i}"))
    out[f"{dst_prefix}.projector.weight"] = tf.projector.weight.data
    for i, blk in enumerate(tf.refiner_blocks):
        out.update(remap_block(blk.state_dict(), f"{dst_prefix}.refiner_blocks.{i}"))
    return out


def remap_dit(dit: SingleStreamDiT) -> dict:
    out = {}
    out["img_in.weight"] = dit.first.weight.data
    out["img_in.bias"] = dit.first.bias.data
    out["time_embed.linear_1.weight"] = dit.tmlp[0].weight.data
    out["time_embed.linear_1.bias"] = dit.tmlp[0].bias.data
    out["time_embed.linear_2.weight"] = dit.tmlp[2].weight.data
    out["time_embed.linear_2.bias"] = dit.tmlp[2].bias.data
    out["time_mod_proj.weight"] = dit.tproj[1].weight.data
    out["time_mod_proj.bias"] = dit.tproj[1].bias.data
    out["txt_in.norm.weight"] = dit.txtmlp[0].scale.data
    out["txt_in.linear_1.weight"] = dit.txtmlp[1].weight.data
    out["txt_in.linear_1.bias"] = dit.txtmlp[1].bias.data
    out["txt_in.linear_2.weight"] = dit.txtmlp[3].weight.data
    out["txt_in.linear_2.bias"] = dit.txtmlp[3].bias.data
    out.update(remap_text_fusion(dit.txtfusion))
    for i, blk in enumerate(dit.blocks):
        out.update(remap_block(blk.state_dict(), f"transformer_blocks.{i}"))
    out["final_layer.norm.weight"] = dit.last.norm.scale.data
    out["final_layer.linear.weight"] = dit.last.linear.weight.data
    out["final_layer.linear.bias"] = dit.last.linear.bias.data
    out["final_layer.scale_shift_table"] = dit.last.modulation.lin.data  # already [2, d]
    return out


def cos_sin(posemb: PositionalEncoding, pos: torch.Tensor):
    """Reference RoPE cos/sin for `pos` `[1, L, 3]`, extracted from the 2×2 rotation `[[cos,-sin],
    [sin,cos]]` → `[1, L, head_dim/2]` (the exact tables the Rust `apply_interleaved_rope` consumes)."""
    freqs = posemb(pos)  # [1, L, half, 2, 2]
    return freqs[..., 0, 0].contiguous(), freqs[..., 1, 0].contiguous()


def f32(d: dict) -> dict:
    return {k: v.detach().to(torch.float32).contiguous() for k, v in d.items()}


def save(rel: str, tensors: dict) -> None:
    from safetensors.torch import save_file

    path = fixture(rel)
    save_file(f32(tensors), path)
    print(f"wrote {path}  ({len(tensors)} tensors)")


# ── Tiny dims (mmdit derives axes = [hd-12*(hd//16), 6*(hd//16), 6*(hd//16)]; hd=32 → [8,12,12]) ──
FEATURES, HEADS, KVHEADS, HEAD_DIM = 128, 4, 2, 32
TXTDIM, TXTHEADS, TXTKVHEADS = 64, 2, 2
MULT, THETA = 4, 1e3
HALF = HEAD_DIM // 2  # 16


@torch.no_grad()
def dump_single_block():
    blk = SingleStreamBlock(FEATURES, HEADS, MULT, bias=False, kvheads=KVHEADS).eval()
    randomize_special(blk)

    seq = 6  # 2 text (0,0,0) + a 2×2 image grid (0,row,col)
    pos = torch.zeros(1, seq, 3)
    pos[0, 2] = torch.tensor([0.0, 0.0, 0.0])
    grid = [(0.0, r, c) for r in range(2) for c in range(2)]
    for i, (t0, t1, t2) in enumerate(grid):
        pos[0, 2 + i] = torch.tensor([t0, t1, t2])
    posemb = PositionalEncoding(FEATURES, [8, 12, 12], theta=THETA)
    freqs = posemb(pos)
    cos, sin = cos_sin(posemb, pos)

    x = torch.randn(1, seq, FEATURES)
    tvec = torch.randn(1, 1, 6 * FEATURES)
    y = blk(x, tvec, freqs, mask=None)

    out = remap_block(blk.state_dict(), "blk")  # already keyed `blk.<diffusers leaf>`
    out["in.x"] = x
    out["in.tvec"] = tvec
    out["in.cos"] = cos
    out["in.sin"] = sin
    out["out.y"] = y
    save("mlx-gen-krea/tests/fixtures/single_block_golden.safetensors", out)


@torch.no_grad()
def dump_text_fusion():
    n_layers = 3  # tiny stand-in for the 12 selected Qwen3-VL layers
    tf = TextFusionTransformer(n_layers, TXTDIM, TXTHEADS, MULT, bias=False, kvheads=TXTKVHEADS).eval()
    randomize_special(tf)

    n_tok = 5
    x = torch.randn(1, n_tok, n_layers, TXTDIM)  # [b, n_tok, num_layers, txt_dim]
    y = tf(x, mask=None)

    out = remap_text_fusion(tf)
    out["in.x"] = x
    out["out.y"] = y
    save("mlx-gen-krea/tests/fixtures/text_fusion_golden.safetensors", out)


@torch.no_grad()
def dump_dit():
    cfg = SingleMMDiTConfig(
        features=FEATURES,
        tdim=64,
        txtdim=TXTDIM,
        heads=HEADS,
        kvheads=KVHEADS,
        multiplier=MULT,
        layers=2,
        patch=2,
        channels=4,
        txtheads=TXTHEADS,
        txtkvheads=TXTKVHEADS,
        txtlayers=3,
        theta=THETA,
    )
    dit = SingleStreamDiT(cfg).eval()
    randomize_special(dit)

    from mmdit import temb as _temb  # noqa: F401  (sanity: reference temb is reachable)

    n_tok = 5
    p, ch, h, w = cfg.patch, cfg.channels, 8, 8
    ht, wt = h // p, w // p
    latent = torch.randn(1, ch, h, w)
    context = torch.randn(1, n_tok, cfg.txtlayers, cfg.txtdim)
    tval = 0.7
    t = torch.full((1,), tval)

    # Reference `prepare()` → patchified img tokens, 3-axis positions, combined mask (all valid).
    txtmask = torch.ones(1, n_tok, dtype=torch.bool)
    from sampling import prepare  # noqa: E402

    img_tokens, posids, mask = prepare(latent, n_tok, p, txtmask)
    out_tokens = dit(img_tokens, context, t, posids, mask)  # [1, img_len, ch·p²]
    velocity = rearrange(
        out_tokens, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=ht, w=wt, ph=p, pw=p
    )

    out = remap_dit(dit)
    out["in.latent"] = latent
    out["in.timestep"] = t
    out["in.context"] = context
    out["out.velocity"] = velocity
    save("mlx-gen-krea/tests/fixtures/dit_golden.safetensors", out)

    # Standalone RoPE golden for the same DiT positions (localizes the position-assignment risk).
    posemb = PositionalEncoding(cfg.features, [8, 12, 12], theta=THETA)
    cos, sin = cos_sin(posemb, posids)
    save(
        "mlx-gen-krea/tests/fixtures/rope_golden.safetensors",
        {
            "cos": cos,
            "sin": sin,
            "meta": torch.tensor([n_tok, ht, wt, 8, 12, 12], dtype=torch.int32),
            "theta": torch.tensor([THETA], dtype=torch.float32),
        },
    )


if __name__ == "__main__":
    dump_single_block()
    dump_text_fusion()
    dump_dit()
    print("done.")
