# SVD port spec (sc-3054) — captured from the diffusers source

Source-of-truth mapping for the remaining `mlx-gen-svd` slices (S1 VAE / S3 UNet / S4 pipeline),
extracted from diffusers 0.37.1 + transformers 5.10.1 (installed in `~/repos/mflux/.venv-0312`).
**Done + validated:** S0 scheduler (sc-3371), S2 image encoder (sc-3373). This doc unblocks S1/S3/S4.

Reuse: `mlx-gen-sdxl` `ResnetBlock2D` (spatial resnet), the 2D VAE encoder pattern (`vae.rs`),
conv/groupnorm primitives. `mlx_gen::nn::conv3d` exists (per-axis stride/pad) for the temporal convs.
`mlx-gen-wan`/`mlx-gen-ltx` have video-VAE temporal-conv precedent.

## Shared building blocks (used by both VAE decoder and UNet)

### SpatioTemporalResBlock (`models/resnet.py`)
- `spatial_res_block`: `ResnetBlock2D` (spatial, with `time_emb_proj` in the UNet; temb-free in the VAE).
- `temporal_res_block`: `TemporalResnetBlock` — GroupNorm(32) → SiLU → **Conv3d kernel (3,1,1) pad (1,0,0)**
  → (UNet: `time_emb_proj` Linear) → GroupNorm → SiLU → Conv3d (3,1,1) → residual; `conv_shortcut`
  (Conv3d 1×1×1) only if in≠out. Operates on `[B,C,F,H,W]` (frame axis = the temporal conv axis).
- `time_mixer`: **AlphaBlender** — `alpha = sigmoid(mix_factor)` (scalar param `time_mixer.mix_factor`);
  in inference `image_only_indicator` is all-zeros so `out = alpha·x_spatial + (1−alpha)·x_temporal`.
- Flow: spatial pass on `[B*F,C,H,W]` → reshape `[B,F,C,H,W]`→`[B,C,F,H,W]` → temporal pass → blend →
  reshape back to `[B*F,C,H,W]`.
- Keys: `{p}.spatial_res_block.{norm1,conv1,norm2,conv2,time_emb_proj,conv_shortcut?}`,
  `{p}.temporal_res_block.{norm1,conv1,norm2,conv2,time_emb_proj,conv_shortcut?}`, `{p}.time_mixer.mix_factor`.

### TransformerSpatioTemporalModel (UNet only; `models/transformers/transformer_temporal.py`)
- `norm` GroupNorm(32) → `proj_in` Linear → per layer: spatial `BasicTransformerBlock` (self-attn +
  cross-attn to `image_embeds`(1024) + GEGLU ff) then `+ time_pos_embed(arange(F))` then temporal
  `TemporalBasicTransformerBlock` (norm_in/ff_in + self-attn + cross-attn + ff over the frame axis) →
  `time_mixer` AlphaBlender blend → `proj_out` Linear.
- `time_proj` (Timesteps, no params) + `time_pos_embed` (TimestepEmbedding C→C·4→C). Heads: head_dim
  = C/num_heads (e.g. 320/5=64).
- Keys: `{p}.{norm,proj_in,proj_out}`, `{p}.transformer_blocks.{i}.*` (BasicTransformerBlock),
  `{p}.temporal_transformer_blocks.{i}.*` (TemporalBasicTransformerBlock: norm_in, ff_in.net.{0.proj,2},
  norm1/attn1, norm2/attn2, norm3/ff.net.{0.proj,2}), `{p}.time_pos_embed.linear_{1,2}`, `{p}.time_mixer.mix_factor`.

## S1 — AutoencoderKLTemporalDecoder (`models/autoencoders/autoencoder_kl_temporal_decoder.py`)
- **Encoder = standard 2D SD VAE** (block_out [128,256,512,512], latent 4, double_z) + `quant_conv` 1×1
  (8→8). REUSE the sdxl 2D encoder pattern. `encode → moments → DiagonalGaussian.mode()` = first 4 ch.
  `scaling_factor` 0.18215.
- **TemporalDecoder** (net-new): `conv_in` Conv2d(4→512) → `MidBlockTemporalDecoder`
  (SpatioTemporalResBlock×? + `Attention` spatial self-attn, attention_head_dim 512) → 4×
  `UpBlockTemporalDecoder` (SpatioTemporalResBlock×3 + Upsample2D; channels 512→512→256→128) →
  `conv_norm_out` GroupNorm(32,128) → SiLU → `conv_out` Conv2d(128→3) → **`time_conv_out` Conv3d
  kernel (3,1,1) pad (1,0,0)** over `[B,3,F,H,W]`.
- `decode(z, num_frames)`: `image_only_indicator = zeros(B,F)`; chunk by `decode_chunk_size`.
- Keys: `encoder.*`, `quant_conv.*`, `decoder.{conv_in,mid_block,up_blocks.{0..3},conv_norm_out,conv_out,time_conv_out}`.

## S3 — UNetSpatioTemporalConditionModel (`models/unets/unet_spatio_temporal_condition.py`)
- in 8 (4 noise + 4 image-latent concat on channels), out 4, block_out [320,640,1280,1280],
  cross_attn 1024, heads [5,10,20,20], layers_per_block 2, transformer_layers_per_block 1, 25 frames.
- **forward(sample [B,F,8,H,W], timestep, encoder_hidden_states [B,seq,1024], added_time_ids [B,3]):**
  - `time_proj`(320) → `time_embedding`(320→1280→1280) → emb.
  - `added_time_ids` = [fps_id, motion_bucket_id, noise_aug_strength] → `add_time_proj`(256 each) →
    flatten 768 → `add_embedding`(768→1280→1280). **emb = time_emb + add_emb**.
  - flatten sample → `[B*F,8,H,W]`; `conv_in`(8→320); `image_only_indicator = zeros(B,F)`.
  - down (CrossAttnDownBlockSpatioTemporal ×3 + DownBlockSpatioTemporal ×1) → mid
    (UNetMidBlockSpatioTemporal) → up (UpBlockSpatioTemporal + CrossAttnUpBlockSpatioTemporal ×3),
    each block = SpatioTemporalResBlock (+ TransformerSpatioTemporalModel for cross-attn blocks) +
    down/upsample. `conv_norm_out`→SiLU→`conv_out`(320→4); reshape `[B,F,4,H,W]`.
- time_embed_dim = block_out[0]·4 = 1280. emb repeated over frames; encoder_hidden_states repeated over frames.
- Keys: `conv_in`, `time_embedding.linear_{1,2}`, `add_embedding.linear_{1,2}`,
  `down_blocks.{i}.{resnets,attentions,downsamplers}`, `mid_block.{resnets,attentions}`,
  `up_blocks.{i}.{resnets,attentions,upsamplers}`, `conv_norm_out`, `conv_out`. (~1428 keys.)

## S4 — StableVideoDiffusionPipeline (`pipelines/stable_video_diffusion/...`)
- **Image cond**: input image → CLIP `image_embeds` (S2; preprocessing = `image*2−1` →
  `_resize_with_antialiasing(224)` → `(image+1)/2` → CLIP normalize mean/std). AND →
  `image = image + noise_aug_strength·N(0,1)` → VAE-encode → `image_latents` (mode), CFG-concat zeros.
- **added_time_ids** = [fps−1, motion_bucket_id, noise_aug_strength] (`fps_id = fps−1`).
- **init**: `latents = randn(B,F,4,H/8,W/8) · scheduler.init_noise_sigma`.
- **loop** per t: `latent_model_input = cat([latents]*2)` (CFG) → `scale_model_input` (S0) →
  **cat image_latents on channels (dim 2)** → UNet(input, t, image_embeds, added_time_ids) →
  CFG `uncond + guidance_scale[frame]·(cond−uncond)` where **guidance_scale = linspace(min_guidance,
  max_guidance, num_frames)** (frame-wise; default 1.0→3.0) → `scheduler.step` (S0 v-pred euler).
- **decode**: `decode_latents` → `z/scaling_factor` → temporal VAE decode (chunked) → `[B,3,F,H,W]` →
  `(x/2+0.5)` frames. Output via the epic-3018 video runtime.
- Defaults: `num_inference_steps` ~25, `motion_bucket_id` 127, `fps` 7, `noise_aug_strength` 0.02,
  `decode_chunk_size` 8, `min_guidance_scale` 1.0, `max_guidance_scale` 3.0, num_frames 25.
- **Provider**: register `svd_xt` (Modality::Video, image_to_video via `Conditioning::Reference`;
  motion_bucket_id/fps/noise_aug_strength via request advanced fields / new optional fields).
