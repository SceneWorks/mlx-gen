#!/usr/bin/env python
"""Dump a schedule + CFG golden for the Lens sampler (mlx-gen sc-3170). Weight-free.

Builds the `FlowMatchEulerDiscreteScheduler` exactly as `LensPipeline` does (empirical-mu + custom
`linspace(1, 1/n, n)` sigmas + dynamic shift) for the Turbo (4-step) and base (20-step) counts, and
records the resulting sigmas/timesteps, a single denoise `step`, and the norm-rescaled CFG output, so
the Rust `mlx_gen_lens::schedule` can be checked near-bit.

Golden contents (per `n` in {4, 20}):
  - `sigmas_{n}`    [n+1] — `scheduler.sigmas` (shifted, trailing 0);
  - `timesteps_{n}` [n]   — `scheduler.timesteps` (= shifted_sigma · 1000);
  - `step_in_{n}` / `step_out_{n}` — a single `scheduler.step(noise, t0, latents)` (latents = step_in).
Plus, CFG: `cfg_cond` / `cfg_uncond` / `cfg_out` (norm-rescaled, guidance 5.0).
Metadata: seq_len, guidance.

Run (from repo root):
  ~/Repos/mflux/.venv/bin/python tools/dump_lens_schedule_golden.py
Writes `tools/golden/lens_schedule_golden.safetensors` (gitignored).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import save_file

HOME = os.path.expanduser("~")
SCHED_GLOB = f"{HOME}/.cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots/*/scheduler"
OUT = os.path.join(os.path.dirname(__file__), "golden", "lens_schedule_golden.safetensors")

SEQ_LEN = 4096  # 64×64 latent grid (1024px) — exercises the ≤4300 interpolation branch
GUIDANCE = 5.0
STEP_COUNTS = [4, 20]


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def main() -> None:
    matches = sorted(glob.glob(SCHED_GLOB))
    if not matches:
        raise SystemExit(f"no Lens-Turbo scheduler snapshot at {SCHED_GLOB}")

    tensors: dict[str, torch.Tensor] = {}
    torch.manual_seed(0)

    for n in STEP_COUNTS:
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(matches[-1])
        mu = compute_empirical_mu(SEQ_LEN, n)
        sigmas = np.linspace(1.0, 1.0 / n, n)
        sched.set_timesteps(sigmas=sigmas, device="cpu", mu=mu)

        tensors[f"sigmas_{n}"] = sched.sigmas.to(torch.float32).cpu().contiguous()
        tensors[f"timesteps_{n}"] = sched.timesteps.to(torch.float32).cpu().contiguous()

        # one denoise step at index 0
        latents = torch.randn(1, SEQ_LEN, 128, dtype=torch.float32)
        noise = torch.randn(1, SEQ_LEN, 128, dtype=torch.float32)
        out = sched.step(noise, sched.timesteps[0], latents, return_dict=False)[0]
        tensors[f"step_in_{n}"] = latents.contiguous()
        tensors[f"step_noise_{n}"] = noise.contiguous()
        tensors[f"step_out_{n}"] = out.to(torch.float32).contiguous()

    # CFG (norm-rescaled), guidance 5.0
    cond = torch.randn(1, SEQ_LEN, 128, dtype=torch.float32)
    uncond = torch.randn(1, SEQ_LEN, 128, dtype=torch.float32)
    comb = uncond + GUIDANCE * (cond - uncond)
    cond_norm = torch.norm(cond, dim=-1, keepdim=True)
    comb_norm = torch.norm(comb, dim=-1, keepdim=True)
    scale = torch.where(
        comb_norm > 0, cond_norm / comb_norm.clamp_min(1e-12), torch.ones_like(comb_norm)
    )
    cfg_out = comb * scale
    tensors["cfg_cond"] = cond.contiguous()
    tensors["cfg_uncond"] = uncond.contiguous()
    tensors["cfg_out"] = cfg_out.contiguous()

    meta = {"seq_len": str(SEQ_LEN), "guidance": str(GUIDANCE)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT, metadata=meta)
    print(f"wrote {OUT}  (steps {STEP_COUNTS}, seq_len {SEQ_LEN}, guidance {GUIDANCE})")


if __name__ == "__main__":
    main()
