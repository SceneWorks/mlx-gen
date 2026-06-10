#!/usr/bin/env python
"""Golden dump for the InstantID face Resampler (sc-3110).

InstantID's `image_proj_model` (`_vendor/instantid/ip_adapter/resampler.py`) is the *same* original
Tencent `Resampler`/`PerceiverAttention` as the SDXL IP-Adapter-plus (already ported at
`mlx-gen-sdxl/src/ip_adapter.rs`); the only difference is the input feature width — a single 512-d
antelopev2 ArcFace embedding instead of the 257-token ViT-H penultimate. So this golden validates the
*existing* Rust Resampler under `ResamplerConfig::instantid_face()` (embedding_dim=512).

Runs in **float32** and saves, in ONE file (the Rust test is self-contained from it):
  - ``image_proj.*``  : the f32 Resampler weights (from `ip-adapter.bin`'s `image_proj` sub-dict),
                        keyed exactly as the Rust ``Resampler::from_weights(.., "image_proj", ..)``
                        expects (``image_proj.`` + the torch state-dict key).
  - ``arcface_embed`` : a deterministic [1, 1, 512] input (seeded; parity is parity regardless of the
                        input distribution — this keeps the face stack out of the model-port test).
  - ``face_tokens``   : Resampler output [1, 16, 2048] — the 16 face tokens fed to the UNet.

The whole file is gitignored (see tools/golden/README.md); it is larger than the other goldens
because it bundles the ~82M-param image_proj weights (so the test needs no separate converted file).

Run from a torch venv (has torch + safetensors):
    ~/repos/mflux/.venv-0312/bin/python ~/Repos/mlx-gen/tools/dump_instantid_resampler_golden.py
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

HUB = Path.home() / ".cache/huggingface/hub"
INSTANTID = (
    HUB
    / "models--InstantX--InstantID/snapshots/57b32dfee076092ad2930c71fd6d439c2c3b1820"
)
IP_BIN = INSTANTID / "ip-adapter.bin"
OUT = Path(__file__).resolve().parent / "golden" / "instantid_resampler_golden.safetensors"


# ---- Faithful vendored InstantID Resampler (matches the image_proj.* checkpoint layout) ----
# Structurally identical to tools/dump_ip_adapter_golden.py's Resampler; only `embed` differs (512).
class PerceiverAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=20):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner = dim_head * heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        b, l, _ = latents.shape
        q = self.to_q(latents)
        kv = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv).chunk(2, dim=-1)

        def rs(t):
            return t.reshape(b, -1, self.heads, self.dim_head).transpose(1, 2)

        q, k, v = rs(q), rs(k), rs(v)
        s = 1.0 / math.sqrt(math.sqrt(self.dim_head))
        w = (q * s) @ (k * s).transpose(-2, -1)
        w = torch.softmax(w.float(), dim=-1).type(w.dtype)
        out = w @ v
        out = out.transpose(1, 2).reshape(b, l, -1)
        return self.to_out(out)


def feed_forward(dim, mult=4):
    inner = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner, bias=False),
        nn.GELU(),
        nn.Linear(inner, dim, bias=False),
    )


class Resampler(nn.Module):
    def __init__(self, dim=1280, depth=4, dim_head=64, heads=20, num_queries=16,
                 embed=512, out=2048):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim))
        self.proj_in = nn.Linear(embed, dim)
        self.proj_out = nn.Linear(dim, out)
        self.norm_out = nn.LayerNorm(out)
        self.layers = nn.ModuleList(
            [nn.ModuleList([PerceiverAttention(dim, dim_head, heads), feed_forward(dim)])
             for _ in range(depth)]
        )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm_out(self.proj_out(latents))


def main():
    torch.manual_seed(0)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # InstantID ip-adapter.bin = {"image_proj": {...}, "ip_adapter": {...}}.
    # weights_only=True: plain tensor dict, so don't execute pickle from a third-party file (F-152).
    state = torch.load(str(IP_BIN), map_location="cpu", weights_only=True)
    image_proj = {k: v.float() for k, v in state["image_proj"].items()}

    res = Resampler()
    missing, unexpected = res.load_state_dict(image_proj, strict=False)
    assert not missing, f"resampler missing keys: {missing}"
    assert not unexpected, f"resampler unexpected keys: {unexpected}"
    res.eval()

    # Deterministic ArcFace-shaped embedding (seeded). Isolates the model port from the face stack.
    arcface_embed = torch.randn(1, 1, 512, dtype=torch.float32)
    with torch.no_grad():
        face_tokens = res(arcface_embed)  # [1, 16, 2048]

    out = {f"image_proj.{k}": v.contiguous() for k, v in image_proj.items()}
    out["arcface_embed"] = arcface_embed.contiguous()
    out["face_tokens"] = face_tokens.contiguous()
    save_file(out, str(OUT))

    print(f"wrote {OUT}")
    print(f"  image_proj weights: {len(image_proj)} tensors")
    print(f"  arcface_embed   {tuple(arcface_embed.shape)}")
    print(f"  face_tokens     {tuple(face_tokens.shape)}  "
          f"mean={face_tokens.mean():.5f} std={face_tokens.std():.5f}")


if __name__ == "__main__":
    main()
