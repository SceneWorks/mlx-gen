#!/usr/bin/env python
"""Dump a tiny 4-step SDE sampler parity fixture for sc-7843 component 3.

Builds the same tiny PidNet as `dump_pid_lq.py`, then runs a faithful inline reproduction of
`pid_distill_model.py::_student_sample_loop` (SDE / velocity-prediction, `student_t_list`,
`fm_timescale=1000`) in pixel space, capturing the initial noise, each per-step fresh ε, and the
final clamped output. The Rust [`Sampler::run`] is deterministic given (noise, ε) so it parity-checks
the loop math bit-for-bit (the production RNG path is a separate same-backend concern).

Run: /Users/michael/Repos/mlx-gen/_vendor/pid/.venv-pid/bin/python tools/dump_pid_sampler.py
"""

import os
import sys

import torch
from safetensors.torch import save_file

PID_ROOT = "/Users/michael/Repos/mlx-gen/_vendor/pid"
sys.path.insert(0, PID_ROOT)

from pid._src.networks.pid_net import PidNet  # noqa: E402

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mlx-gen-pid", "tests", "fixtures", "sampler_tiny.safetensors",
)

CFG = dict(
    in_channels=3, num_groups=2, hidden_size=32, pixel_hidden_size=8,
    pixel_attn_hidden_size=16, pixel_num_groups=2, patch_depth=4, pixel_depth=2,
    patch_size=2, txt_embed_dim=12, txt_max_length=5, use_text_rope=True,
    text_rope_theta=10000.0, rope_mode="ntk_aware", rope_ref_h=16, rope_ref_w=16,
    repa_encoder_index=-1, enable_ed=False,
    lq_inject_mode="controlnet", lq_in_channels=0, lq_latent_channels=4, lq_hidden_dim=8,
    lq_num_res_blocks=2, lq_gate_type="sigma_aware_per_token_per_dim", lq_interval=2,
    zero_init_lq=True, sr_scale=2, latent_spatial_down_factor=2, pit_lq_inject=False,
)
H, W = 8, 12
T_LIST = [0.999, 0.866, 0.634, 0.342, 0.0]
TIMESCALE = 1000.0


def main():
    torch.manual_seed(0)
    gw = torch.Generator().manual_seed(4321)
    model = PidNet(**CFG)
    model.eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            r = torch.randn(p.shape, generator=gw) * 0.2
            if "norm" in name and p.dim() == 1:
                r = r + 1.0
            p.copy_(r)

    gi = torch.Generator().manual_seed(777)
    caption = torch.randn(1, CFG["txt_max_length"], CFG["txt_embed_dim"], generator=gi)
    lq_latent = torch.randn(1, CFG["lq_latent_channels"], H // CFG["patch_size"] // 2,
                            W // CFG["patch_size"] // 2, generator=gi)
    sigma = torch.tensor([0.3], dtype=torch.float32)
    noise = torch.randn(1, 3, H, W, generator=gi)

    tensors = {}
    for k, v in model.state_dict().items():
        tensors[k] = v.detach().contiguous().float()

    # Faithful inline reproduction of _student_sample_loop (SDE / velocity), capturing noise + eps.
    eps_list, step_xs = [], []
    x = noise
    B = 1
    with torch.no_grad():
        for i in range(len(T_LIST) - 1):
            t_cur, t_next = T_LIST[i], T_LIST[i + 1]
            t_scaled = torch.full((B,), t_cur * TIMESCALE, dtype=torch.float32)
            v = model(x, t_scaled, caption, lq_latent=lq_latent, degrade_sigma=sigma)
            # _velocity_to_x0: x - t_cur * v  (t_cur unscaled)
            x0 = x.float() - t_cur * v.float()
            if t_next > 0:
                eps = torch.randn(x0.shape, generator=gi)
                eps_list.append(eps)
                x = (1.0 - t_next) * x0 + t_next * eps
            else:
                x = x0
            step_xs.append(x.clone())
    out = x.clamp(-1, 1)

    tensors["__io__.noise"] = noise.contiguous().float()
    tensors["__io__.caption"] = caption.contiguous().float()
    tensors["__io__.lq_latent"] = lq_latent.contiguous().float()
    tensors["__io__.sigma"] = sigma.contiguous().float()
    tensors["__io__.output"] = out.contiguous().float()
    for i, e in enumerate(eps_list):
        tensors[f"__io__.eps_{i}"] = e.contiguous().float()
    for i, xs in enumerate(step_xs):
        tensors[f"__io__.step_{i}"] = xs.contiguous().float()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(tensors, OUT, metadata={k: str(v) for k, v in CFG.items()})
    print(f"wrote {OUT}")
    print(f"  steps={len(T_LIST)-1}  eps_draws={len(eps_list)}  output {tuple(out.shape)} "
          f"mean={out.mean():.5f} std={out.std():.5f}")


if __name__ == "__main__":
    main()
