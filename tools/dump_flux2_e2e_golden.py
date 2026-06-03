"""Real-weights end-to-end golden for FLUX.2-klein txt2img (sc-2346 S4), for the #[ignore]d Rust
e2e parity test. Forces `ModelConfig.precision = float32` (the Rust pipeline runs f32) and runs the
full pipeline manually so we can capture the seeded noise, the final packed latents, and the
decoded image. Small (256², 4 steps, guidance 1.0) to keep the f32 run feasible.

Gitignored output. Run from the mflux fork venv:
    cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_e2e_golden.py
"""

import mlx.core as mx

from mflux.models.common.config.model_config import ModelConfig

ModelConfig.precision = mx.float32  # the Rust pipeline runs f32 activations

from mflux.models.common.config import ModelConfig as MC  # noqa: E402
from mflux.models.common.config.config import Config  # noqa: E402
from mflux.models.flux2.variants import Flux2Klein  # noqa: E402

from _paths import fixture  # noqa: E402

PROMPT = "a red fox resting in fresh snow under soft winter light"
SEED, STEPS, SIZE, GUIDANCE = 0, 4, 256, 1.0

model = Flux2Klein(model_config=MC.flux2_klein_9b())
config = Config(
    model_config=model.model_config,
    num_inference_steps=STEPS,
    height=SIZE,
    width=SIZE,
    guidance=GUIDANCE,
    scheduler="flow_match_euler_discrete",
)

prompt_embeds, text_ids, neg_embeds, neg_ids = model._encode_prompt_pair(
    prompt=PROMPT, negative_prompt=" ", guidance=GUIDANCE
)
latents, latent_ids, lat_h, lat_w = model._prepare_generation_latents(seed=SEED, config=config)
noise0 = latents

predict = model._predict(model.transformer)
v0 = None
for t in range(config.init_time_step, config.num_inference_steps):
    noise = predict(
        latents=latents,
        latent_ids=latent_ids,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        negative_prompt_embeds=neg_embeds,
        negative_text_ids=neg_ids,
        guidance=GUIDANCE,
        timestep=config.scheduler.timesteps[t],
    )
    if v0 is None:
        v0 = noise  # step-0 velocity — the chaos-free real-weights transformer gate
    latents = config.scheduler.step(noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas)
    mx.eval(latents)

packed = latents.reshape(latents.shape[0], lat_h, lat_w, latents.shape[-1]).transpose(0, 3, 1, 2)
decoded = model.vae.decode_packed_latents(packed)  # NCHW [1,3,H,W]
mx.eval(decoded)

out = {
    "noise0": noise0.astype(mx.float32),  # [1, seq, 128]
    "prompt_embeds": prompt_embeds.astype(mx.float32),  # [1, 512, 12288]
    "text_ids": text_ids.astype(mx.int32),  # [seq_txt, 4] (batch already dropped) or [1,seq,4]
    "latent_ids": latent_ids.astype(mx.int32),
    "v0": v0.astype(mx.float32),  # [1, seq, 128] step-0 velocity
    "latents": latents.astype(mx.float32),  # [1, seq, 128] final
    "decoded": decoded.astype(mx.float32),  # NCHW [1,3,256,256]
    "timestep0": mx.array([float(config.scheduler.timesteps[0])]),
}
path = fixture("tools/golden/flux2_e2e.safetensors")
mx.save_safetensors(path, out)
print(f"wrote {path}")
print(f"  noise0 {tuple(noise0.shape)}  v0 {tuple(v0.shape)}  latents {tuple(latents.shape)}")
print(f"  text_ids {tuple(text_ids.shape)}  latent_ids {tuple(latent_ids.shape)}")
print(f"  timestep0 {float(config.scheduler.timesteps[0]):.4f}")
