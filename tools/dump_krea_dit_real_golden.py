"""Generate the real-weight Krea 2 DiT parity golden (sc-7568, the `#[ignore]` gate).

Loads the published `krea/Krea-2-Turbo` diffusers `transformer/` checkpoint INTO the reference
`mmdit.py` `SingleStreamDiT` (via the diffusers→mmdit key map — the inverse of the dump used for the
tiny fixtures), runs one forward at a modest resolution with random latent/context/timestep, and dumps
the velocity + inputs. The Rust `#[ignore]` test loads the SAME real weights through
`Krea2Transformer::from_weights`, runs the same inputs, and checks cross-backend parity.

Context is random (the real Qwen3-VL conditioning lands in sc-7569); the DiT forward is agnostic to
whether its context is real, so this validates the DiT math at full scale/weights.

    KREA_TURBO_DIR=~/.cache/huggingface/hub/models--krea--Krea-2-Turbo/snapshots/<rev> \
    KREA_DEVICE=mps KREA_DTYPE=bf16 \
      ~/Repos/mflux/.venv/bin/python tools/dump_krea_dit_real_golden.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

from _paths import REPO_ROOT, fixture

torch.compile = lambda model=None, **kw: (model if model is not None else (lambda f: f))  # noqa: E731
sys.path.insert(0, str(REPO_ROOT / "_vendor" / "krea2"))
import mmdit  # noqa: E402
from mmdit import SingleMMDiTConfig, SingleStreamDiT  # noqa: E402
from einops import rearrange  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _cpu_attention(q, k, v, mask=None, scale=None, gqa=False):
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa)
    return rearrange(x, "B H L D -> B L (H D)")


mmdit.attention = _cpu_attention


def _rope_f32(pos, dim, theta=1e4, ntk=1.0):
    """MPS has no float64; the reference `rope` builds its frequencies in float64. f32 angles differ by
    ~1e-6 for these small integer positions — negligible vs the bf16 cross-backend tolerance (and the
    Rust builder still uses f64, so this only relaxes the golden, never the port)."""
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos.float(), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2).float()


mmdit.rope = _rope_f32

# The published Turbo/Raw DiT architecture (reference `inference.py::single_mmdit_large_wide`).
CONFIG = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
    theta=1e3,
)

_ATTN_FF = {  # diffusers leaf -> mmdit leaf (within a block)
    "attn.to_q.weight": "attn.wq.weight",
    "attn.to_k.weight": "attn.wk.weight",
    "attn.to_v.weight": "attn.wv.weight",
    "attn.to_gate.weight": "attn.gate.weight",
    "attn.to_out.0.weight": "attn.wo.weight",
    "attn.norm_q.weight": "attn.qknorm.qnorm.scale",
    "attn.norm_k.weight": "attn.qknorm.knorm.scale",
    "ff.gate.weight": "mlp.gate.weight",
    "ff.up.weight": "mlp.up.weight",
    "ff.down.weight": "mlp.down.weight",
    "norm1.weight": "prenorm.scale",
    "norm2.weight": "postnorm.scale",
}
_TOP = {
    "img_in.weight": "first.weight",
    "img_in.bias": "first.bias",
    "time_embed.linear_1.weight": "tmlp.0.weight",
    "time_embed.linear_1.bias": "tmlp.0.bias",
    "time_embed.linear_2.weight": "tmlp.2.weight",
    "time_embed.linear_2.bias": "tmlp.2.bias",
    "time_mod_proj.weight": "tproj.1.weight",
    "time_mod_proj.bias": "tproj.1.bias",
    "txt_in.norm.weight": "txtmlp.0.scale",
    "txt_in.linear_1.weight": "txtmlp.1.weight",
    "txt_in.linear_1.bias": "txtmlp.1.bias",
    "txt_in.linear_2.weight": "txtmlp.3.weight",
    "txt_in.linear_2.bias": "txtmlp.3.bias",
    "text_fusion.projector.weight": "txtfusion.projector.weight",
    "final_layer.norm.weight": "last.norm.scale",
    "final_layer.linear.weight": "last.linear.weight",
    "final_layer.linear.bias": "last.linear.bias",
    "final_layer.scale_shift_table": "last.modulation.lin",  # [2, d] direct
}


def diffusers_to_mmdit(key: str, tensor: torch.Tensor):
    """Return (mmdit_key, tensor) — pure rename except `transformer_blocks.*.scale_shift_table`
    `[6,d]`→`mod.lin` `[6·d]`."""
    if key in _TOP:
        return _TOP[key], tensor
    for stem, dst in (
        ("transformer_blocks.", "blocks."),
        ("text_fusion.layerwise_blocks.", "txtfusion.layerwise_blocks."),
        ("text_fusion.refiner_blocks.", "txtfusion.refiner_blocks."),
    ):
        if key.startswith(stem):
            rest = key[len(stem):]
            idx, leaf = rest.split(".", 1)
            base = f"{dst}{idx}"
            if leaf == "scale_shift_table":
                return f"{base}.mod.lin", tensor.reshape(-1).contiguous()
            return f"{base}.{_ATTN_FF[leaf]}", tensor
    raise KeyError(f"unmapped diffusers key: {key}")


def load_transformer_state(root: Path) -> dict:
    tdir = root / "transformer"
    index = tdir / "diffusion_pytorch_model.safetensors.index.json"
    from safetensors.torch import load_file

    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        shards = ["diffusion_pytorch_model.safetensors"]
    sd = {}
    for shard in shards:
        sd.update(load_file(str(tdir / shard)))
    return sd


@torch.no_grad()
def main():
    root = Path(os.environ["KREA_TURBO_DIR"])
    device = os.environ.get("KREA_DEVICE", "mps")
    dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[os.environ.get("KREA_DTYPE", "bf16")]

    diff_sd = load_transformer_state(root)
    mmdit_sd = {}
    for k, v in diff_sd.items():
        nk, nv = diffusers_to_mmdit(k, v)
        mmdit_sd[nk] = nv

    with torch.device("meta"):
        dit = SingleStreamDiT(CONFIG)
    dit.load_state_dict(mmdit_sd, strict=True, assign=True)
    dit = dit.to(device=device, dtype=dtype).eval().requires_grad_(False)

    torch.manual_seed(0)
    p, ch = CONFIG.patch, CONFIG.channels
    h, w = 64, 64  # → 32×32 = 1024 image tokens
    n_tok = 16
    ht, wt = h // p, w // p
    latent = torch.randn(1, ch, h, w, dtype=dtype, device=device)
    context = torch.randn(1, n_tok, CONFIG.txtlayers, CONFIG.txtdim, dtype=dtype, device=device)
    t = torch.full((1,), 0.7, dtype=dtype, device=device)

    txtmask = torch.ones(1, n_tok, dtype=torch.bool, device=device)
    from sampling import prepare  # noqa: E402

    img_tokens, posids, mask = prepare(latent, n_tok, p, txtmask)
    out_tokens = dit(img_tokens, context, t, posids, mask)
    velocity = rearrange(
        out_tokens, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=ht, w=wt, ph=p, pw=p
    )

    from safetensors.torch import save_file

    out = {
        "in.latent": latent,
        "in.timestep": t,
        "in.context": context,
        "out.velocity": velocity,
    }
    out = {k: v.detach().to(torch.float32).cpu().contiguous() for k, v in out.items()}
    path = fixture("tools/golden/krea_dit_real.safetensors")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, path)
    print(f"wrote {path}  (device={device} dtype={dtype}, velocity {tuple(velocity.shape)})")


if __name__ == "__main__":
    main()
