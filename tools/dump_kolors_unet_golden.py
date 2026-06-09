"""Kolors U-Net single-forward golden — reference for mlx-gen-sdxl Kolors wiring (sc-3093).

Runs the diffusers Kolors **U-Net** (`UNet2DConditionModel` from the `Kwai-Kolors/Kolors-diffusers`
snapshot) in **f32** for one forward, feeding the ChatGLM3 conditioning the `KolorsPipeline` produces:
`encoder_hidden_states` = `hidden_states[-2]` (context, 4096-d), `added_cond_kwargs` =
{`text_embeds`=pooled(4096), `time_ids`=(1024,1024,0,0,1024,1024)}. The diffusers U-Net projects the
4096 context to `cross_attention_dim`(2048) via its `encoder_hid_proj` and feeds the 5632-wide
`add_embedding`. Dumps every input + the predicted eps so the Rust `UNet2DConditionModel` (with the
auto-detected `encoder_hid_proj`, loaded via `load_unet_kolors_dtype`) is validated in isolation.

This isolates the **U-Net wiring** (encoder_hid_proj + 5632 add_embedding); the ChatGLM3 encoder
itself is gated separately by sc-3091. Latents/eps stored **NHWC** (the mlx U-Net layout).

Loads the ChatGLM3 TE (~25 GB f32) + the U-Net (~10 GB f32). Run:
    ~/repos/mflux/.venv-0312/bin/python tools/dump_kolors_unet_golden.py
Output (gitignored): tools/golden/kolors_unet_golden.safetensors
"""

import glob
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors.torch import load_file

from _paths import fixture, hf_hub_cache

from diffusers import UNet2DConditionModel
from diffusers.pipelines.kolors.text_encoder import ChatGLMConfig, ChatGLMModel
from diffusers.pipelines.kolors.tokenizer import ChatGLMTokenizer

PROMPT = "A cat playing a grand piano on a city rooftop at sunset."
TIMESTEP = 999.0
H = W = 128  # 1024px image → [1,4,128,128] latent


def snapshot() -> Path:
    base = hf_hub_cache() / "models--Kwai-Kolors--Kolors-diffusers" / "snapshots"
    snaps = sorted(glob.glob(str(base / "*")))
    if not snaps:
        raise SystemExit("Kolors-diffusers snapshot not found in HF cache")
    return Path(snaps[-1])


def load_text_encoder(te: Path) -> ChatGLMModel:
    cfg = ChatGLMConfig(**json.loads((te / "config.json").read_text()))
    model = ChatGLMModel(cfg)
    state = {}
    for shard in sorted(glob.glob(str(te / "*.safetensors"))):
        state.update(load_file(shard))
    missing, _ = model.load_state_dict(state, strict=False)
    assert not missing, f"missing TE weights: {missing[:8]}"
    return model.float().eval()


@torch.no_grad()
def main():
    snap = snapshot()
    tok = ChatGLMTokenizer(vocab_file=str(snap / "tokenizer" / "tokenizer.model"))
    te = load_text_encoder(snap / "text_encoder")
    unet = UNet2DConditionModel.from_pretrained(
        snap / "unet", variant="fp16", torch_dtype=torch.float32
    ).eval()

    enc = tok(PROMPT, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    out = te(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        position_ids=enc["position_ids"],
        output_hidden_states=True,
    )
    context = out.hidden_states[-2].permute(1, 0, 2).contiguous()  # [1, 256, 4096]
    pooled = out.hidden_states[-1][-1, :, :].contiguous()  # [1, 4096]
    time_ids = torch.tensor([[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]], dtype=torch.float32)

    g = torch.Generator().manual_seed(0)
    latents_nchw = torch.randn(1, 4, H, W, generator=g, dtype=torch.float32)
    t = torch.tensor(TIMESTEP, dtype=torch.float32)

    eps_nchw = unet(
        latents_nchw,
        t,
        encoder_hidden_states=context,
        added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
    ).sample  # [1, 4, 128, 128]

    def nhwc(t):  # [B,C,H,W] → [B,H,W,C]
        return mx.array(t.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32))

    def arr(t):
        return mx.array(t.cpu().numpy().astype(np.float32))

    tensors = {
        "latents": nhwc(latents_nchw),
        "conditioning": arr(context),
        "pooled": arr(pooled),
        "time_ids": arr(time_ids),
        "eps": nhwc(eps_nchw),
    }
    mx.eval(list(tensors.values()))
    meta = {"prompt": PROMPT, "timestep": str(TIMESTEP), "h": str(H), "w": str(W)}
    out_path = fixture("tools/golden/kolors_unet_golden.safetensors")
    mx.save_safetensors(out_path, tensors, metadata=meta)
    print(f"wrote {out_path}")
    print(f"  latents {tuple(tensors['latents'].shape)} cond {tuple(tensors['conditioning'].shape)} "
          f"eps {tuple(tensors['eps'].shape)} mean|eps|={float(mx.mean(mx.abs(tensors['eps']))):.5f}")


if __name__ == "__main__":
    main()
