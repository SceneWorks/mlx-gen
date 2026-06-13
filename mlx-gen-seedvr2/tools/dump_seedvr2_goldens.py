"""Dump SeedVR2 parity goldens for the mlx-gen-seedvr2 Rust port (sc-4813).

Runs in the mflux frozen-fork venv (the MLX reference):
    cd ~/Repos/mflux-sc2257
    .venv/bin/python <this> --component vae   --dir ~/.cache/mlx-gen-seedvr2-golden
    .venv/bin/python <this> --component dit   --dir ~/.cache/mlx-gen-seedvr2-golden

For each component it writes, to --dir:
  * `<comp>_f32.safetensors` — the **already-converted** MLX weights (flattened tree, f32). These are
    the MLX-native key/layout the Rust modules load, so the Rust parity test is isolated from the
    weight converter (which is gated separately, byte-exact, against the raw checkpoint).
  * `<comp>_io_f32.safetensors` — small deterministic input/output goldens (f32) for the parity gate.

f32 is used (model + activations cast to f32) so the Rust-vs-MLX comparison is near bit-exact
(both run MLX-Metal); op-order is the only expected source of drift.
"""
import argparse
import os

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.seedvr2.model.seedvr2_text_encoder.text_embeddings import SeedVR2TextEmbeddings
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2


def f32_params(module):
    """Flatten module.parameters() and cast every leaf to f32."""
    return {k: v.astype(mx.float32) for k, v in tree_flatten(module.parameters())}


def save(path, arrays):
    mx.eval(list(arrays.values()))
    mx.save_safetensors(path, arrays)
    print(f"  wrote {path}  ({len(arrays)} tensors)")


def cast_f32(module):
    """Recursively cast a module's parameters to f32 in place (so the forward runs f32)."""
    module.update(tree_map(lambda a: a.astype(mx.float32), module.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", required=True, choices=["vae", "dit"])
    ap.add_argument("--dir", default=os.path.expanduser("~/.cache/mlx-gen-seedvr2-golden"))
    ap.add_argument("--model", default="3b", choices=["3b", "7b"])
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    cfg = ModelConfig.seedvr2_3b() if args.model == "3b" else ModelConfig.seedvr2_7b()
    model = SeedVR2(model_config=cfg)

    if args.component == "vae":
        vae = model.vae
        cast_f32(vae)
        save(os.path.join(args.dir, "vae_f32.safetensors"), f32_params(vae))

        io = {}
        # image mode: T=1 -> latentT 1 ; small spatial for speed/tight numerics
        mx.random.seed(0)
        x_img = (mx.random.normal((1, 3, 1, 64, 64)) * 0.5).astype(mx.float32)
        enc_img = vae.encode(x_img)              # (1,16,1,8,8)
        dec_img = vae.decode(enc_img)            # (1,3,1,64,64)
        io["x_img"], io["enc_img"], io["dec_img"] = x_img, enc_img, dec_img
        # video mode: T=5 -> latentT 2 -> decodedT 8 (exercises temporal conv/upsample)
        x_vid = (mx.random.normal((1, 3, 5, 32, 32)) * 0.5).astype(mx.float32)
        enc_vid = vae.encode(x_vid)              # (1,16,2,4,4)
        dec_vid = vae.decode(enc_vid)            # (1,3,8,32,32)
        io["x_vid"], io["enc_vid"], io["dec_vid"] = x_vid, enc_vid, dec_vid
        # decoder stage-by-stage internals (localisation), fed the image latent
        dec = vae.decoder
        z = enc_img / vae.scaling_factor
        h = dec.conv_in(z); io["d_conv_in"] = h
        h = dec.mid_block(h); io["d_mid"] = h
        for i, ub in enumerate(dec.up_blocks):
            h = ub(h); io[f"d_up{i}"] = h
        # final tail sub-stages
        ht = h.transpose(0, 2, 3, 4, 1)
        nout = dec.conv_norm_out(ht.astype(mx.float32)).transpose(0, 4, 1, 2, 3)
        io["d_normout"] = nout
        import mlx.nn as _nn
        hs = _nn.silu(nout)
        io["d_silu"] = hs
        io["d_convout"] = dec.conv_out(hs)
        for k, v in io.items():
            print(f"   vae io {k}: {list(v.shape)}")
        save(os.path.join(args.dir, "vae_io_f32.safetensors"), io)

    else:  # dit
        tr = model.transformer
        cast_f32(tr)
        save(os.path.join(args.dir, "dit_f32.safetensors"), f32_params(tr))

        # small image-mode input: latentT=1, h=w=8 -> 1*4*4=16 vid tokens
        mx.random.seed(1)
        h = w = 8
        vid = (mx.random.normal((1, 33, 1, h, w)) * 0.3).astype(mx.float32)
        txt = SeedVR2TextEmbeddings.load_positive().astype(mx.float32)   # (1,58,5120)
        timestep = mx.array(float(cfg.num_train_steps or 1000.0))

        # localization intermediates (public submodules, known signatures)
        txt_proj = tr.txt_in(txt)                # (1,58,2560)
        vid_tok, vid_shape = tr.vid_in(vid)      # (1,16,2560), (1,3)
        emb = tr.emb_in(timestep)                # (1,15360)
        out = tr(vid=vid, txt=txt, timestep=timestep)  # (1,16,1,8,8)

        io = {
            "vid": vid, "txt": txt, "timestep": timestep,
            "txt_proj": txt_proj, "vid_tok": vid_tok,
            "vid_shape": vid_shape.astype(mx.float32), "emb": emb,
            "dit_out": out,
        }
        for k, v in io.items():
            print(f"   dit io {k}: {list(v.shape)}")
        save(os.path.join(args.dir, "dit_io_f32.safetensors"), io)


if __name__ == "__main__":
    main()
