"""sc-5139: synthetic-fixture golden for the Bernini planner's connector + clip-diff head.

Builds tiny `MLPConnector` + `SimpleMLPAdaLN` (DiffLoss_FM net) + `FlowMatchScheduler` with random f32
weights, runs the **reference** forwards, and dumps weights + inputs + outputs to a safetensors
fixture the Rust parity test loads:
  - connector `for_gen` / `for_vit`,
  - the net forward `net(x, t, c)`,
  - a full `DiffLoss_FM.sample` with triple (txt/img) CFG over a 3-step denoise (fixed injected noise).

The classes are copied **verbatim** from `_vendor/bernini/bernini/models/{bernini.py,diffloss_fm.py,
scheduler.py}` (only `.cuda()` calls dropped for CPU), so the oracle is the reference. f32 throughout.

Run:
  ~/Repos/mflux/.venv/bin/python tools/dump_bernini_clip_diff_golden.py
Fixture -> mlx-gen-bernini/tests/fixtures/clip_diff_golden.safetensors
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(
    REPO_ROOT, "mlx-gen-bernini", "tests", "fixtures", "clip_diff_golden.safetensors"
)

# tiny dims (fixture < 1 MB)
HIDDEN = 8          # connector in / net target+z channels
GEN = 12            # connector for_gen out
WIDTH = 16          # net model_channels
DEPTH = 2           # res blocks
SHIFT = 2.0
N = 4               # base batch
STEPS = 3
TXT_CFG = 1.4
IMG_CFG = 1.2


# ===== verbatim reference: bernini.py RMSNorm + MLPConnector =====
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
            nn.Linear(in_dim, out_dim_for_gen),
            nn.GELU(),
            RMSNorm(out_dim_for_gen),
            nn.Linear(out_dim_for_gen, out_dim_for_gen),
        )
        self.pred_vit = nn.Sequential(
            nn.Linear(in_dim, out_dim_for_vit),
            nn.GELU(),
            nn.Linear(out_dim_for_vit, out_dim_for_vit),
            RMSNorm(out_dim_for_vit),
            nn.Linear(out_dim_for_vit, out_dim_for_vit),
        )

    def for_gen(self, x):
        return self.proj_gen(x)

    def for_vit(self, x):
        return self.pred_vit(x)


# ===== verbatim reference: diffloss_fm.py blocks =====
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(t.dtype))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class SimpleMLPAdaLN(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks):
        super().__init__()
        self.in_channels = in_channels
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)

    def forward(self, x, t, c):
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)

    def forward_with_txt_img_cfg(self, x, t, c, txt_cfg_scale, img_cfg_scale):
        part = x[: len(x) // 3]
        combined = torch.cat([part, part, part], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps, imgcond_eps = torch.split(eps, len(eps) // 3, dim=0)
        part_eps = uncond_eps + img_cfg_scale * (imgcond_eps - uncond_eps) + txt_cfg_scale * (cond_eps - imgcond_eps)
        eps = torch.cat([part_eps, part_eps, part_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# ===== verbatim reference: scheduler.py (CPU; .cuda() dropped) =====
class FlowMatchScheduler:
    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0,
                 sigma_max=1.0, sigma_min=0.003 / 1.002, extra_one_step=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, shift=None, dtype=torch.float32):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1, dtype=dtype)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps, dtype=dtype)
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def step(self, model_output, timestep, sample, to_final=False):
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_ - sigma)


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    conn = MLPConnector(HIDDEN, GEN, HIDDEN).to(torch.float32).eval()
    net = SimpleMLPAdaLN(HIDDEN, WIDTH, HIDDEN, HIDDEN, DEPTH).to(torch.float32).eval()
    sched = FlowMatchScheduler(num_inference_steps=STEPS, shift=SHIFT, extra_one_step=True)

    out = {}
    for k, v in conn.state_dict().items():
        out[f"conn.{k}"] = v.contiguous()
    for k, v in net.state_dict().items():
        out[f"net.{k}"] = v.contiguous()

    # 1) connector
    cx = torch.randn(N, HIDDEN)
    out["io.conn_x"] = cx
    out["out.for_gen"] = conn.for_gen(cx)
    out["out.for_vit"] = conn.for_vit(cx)

    # 2) net forward
    nx = torch.randn(N, HIDDEN)
    nt = torch.rand(N) * 1000.0
    nc = torch.randn(N, HIDDEN)
    out["io.net_x"] = nx
    out["io.net_t"] = nt
    out["io.net_c"] = nc
    out["out.net"] = net(nx, nt, nc)

    # 3) full sample() with triple CFG (the planner's vit_txt_cfg/vit_img_cfg path)
    noise_base = torch.randn(N, HIDDEN)
    z = torch.randn(3 * N, HIDDEN)  # pre-tiled cond
    out["io.noise_base"] = noise_base
    out["io.z"] = z
    sched.set_timesteps(STEPS)
    samples = torch.cat([noise_base, noise_base, noise_base], dim=0)
    for t in sched.timesteps:
        timestep = t.unsqueeze(0)
        pred = net.forward_with_txt_img_cfg(samples, timestep, z, TXT_CFG, IMG_CFG)
        samples = sched.step(pred, timestep, samples)
    out["out.sample"] = samples

    meta = {
        "hidden": str(HIDDEN), "gen": str(GEN), "width": str(WIDTH), "depth": str(DEPTH),
        "shift": repr(SHIFT), "steps": str(STEPS), "n": str(N),
        "txt_cfg": repr(TXT_CFG), "img_cfg": repr(IMG_CFG),
    }
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, FIXTURE, metadata=meta)
    print(f"wrote {FIXTURE}  ({len(out)} tensors)")
    print(f"  for_gen {tuple(out['out.for_gen'].shape)}  net {tuple(out['out.net'].shape)}  "
          f"sample {tuple(out['out.sample'].shape)}")


if __name__ == "__main__":
    main()
