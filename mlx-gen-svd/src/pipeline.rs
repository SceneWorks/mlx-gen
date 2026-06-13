//! SVD image-to-video pipeline (sc-3375) — the `StableVideoDiffusionPipeline` orchestration over the
//! S0–S3 components: image → CLIP `image_embeds` + noise-augmented VAE `image_latents`; a frame-wise
//! CFG denoise loop (EDM v-prediction Euler, image-latent channel-concat) with `guidance_scale =
//! linspace(min, max, num_frames)`; chunked temporal VAE decode → frames. Port of diffusers
//! `pipeline_stable_video_diffusion.py` `__call__` + `_encode_image`/`_encode_vae_image`/
//! `_get_add_time_ids`/`decode_latents`.

use mlx_rs::ops::{add, concatenate_axis, multiply, subtract, zeros_like};
use mlx_rs::Array;

use mlx_gen::array::scalar;
use mlx_gen::{Error, Result};

use crate::config::SchedulerConfig;
use crate::scheduler::{euler_step, scale_model_input, v_pred_denoised, EdmSchedule};
use crate::{SvdImageEncoder, SvdUnet, SvdVae};

/// Image-to-video generation parameters (the `StableVideoDiffusionPipeline.__call__` knobs).
#[derive(Clone, Debug)]
pub struct SvdParams {
    pub num_frames: i32,
    pub num_inference_steps: usize,
    pub min_guidance_scale: f32,
    pub max_guidance_scale: f32,
    pub fps: i32,
    pub motion_bucket_id: f32,
    pub noise_aug_strength: f32,
    /// Frames decoded per temporal VAE pass (diffusers default = `num_frames`).
    pub decode_chunk_size: i32,
}

impl Default for SvdParams {
    fn default() -> Self {
        Self {
            num_frames: 25,
            num_inference_steps: 25,
            min_guidance_scale: 1.0,
            max_guidance_scale: 3.0,
            fps: 7,
            motion_bucket_id: 127.0,
            noise_aug_strength: 0.02,
            decode_chunk_size: 25,
        }
    }
}

/// The assembled SVD pipeline (the image encoder, VAE, and UNet + scheduler config).
pub struct SvdPipeline {
    pub image_encoder: SvdImageEncoder,
    pub vae: SvdVae,
    pub unet: SvdUnet,
    pub scheduler: SchedulerConfig,
}

impl SvdPipeline {
    pub fn new(
        image_encoder: SvdImageEncoder,
        vae: SvdVae,
        unet: SvdUnet,
        scheduler: SchedulerConfig,
    ) -> Self {
        Self {
            image_encoder,
            vae,
            unet,
            scheduler,
        }
    }

    /// The `added_time_ids` micro-conditioning row `[1, 3]` = `[fps − 1, motion_bucket_id,
    /// noise_aug_strength]` (the SVD pipeline reduces fps by 1 — the model was trained on fps−1).
    pub fn added_time_ids(params: &SvdParams) -> Array {
        Array::from_slice(
            &[
                (params.fps - 1) as f32,
                params.motion_bucket_id,
                params.noise_aug_strength,
            ],
            &[1, 3],
        )
    }

    /// The frame-wise CFG schedule `linspace(min, max, F)` shaped `[1, F, 1, 1, 1]` to broadcast over
    /// the `[1, F, H, W, 4]` latents.
    fn guidance_schedule(num_frames: i32, min_g: f32, max_g: f32) -> Array {
        let f = num_frames.max(1);
        let vals: Vec<f32> = (0..f)
            .map(|i| {
                if f == 1 {
                    min_g
                } else {
                    min_g + (max_g - min_g) * (i as f32) / ((f - 1) as f32)
                }
            })
            .collect();
        Array::from_slice(&vals, &[1, f, 1, 1, 1])
    }

    /// The frame-wise CFG v-prediction Euler denoise loop. Inputs are the **conditional** rows
    /// (`[1, …]`); the uncond CFG branch is the diffusers zeros (`image_embeds`/`image_latents` →
    /// `cat([zeros, cond])`). Returns the final `[1, F, H/8, W/8, 4]` latents.
    /// - `latents`: the seeded init noise already scaled by `init_noise_sigma` (`[1, F, H/8, W/8, 4]`).
    /// - `image_embeds`: CLIP conditioning `[1, ctx, 1024]`.
    /// - `image_latents`: per-frame-repeated VAE conditioning latent `[1, F, H/8, W/8, 4]`.
    /// - `added_time_ids`: `[1, 3]`.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        latents: &Array,
        image_embeds: &Array,
        image_latents: &Array,
        added_time_ids: &Array,
        num_frames: i32,
        steps: usize,
        min_g: f32,
        max_g: f32,
    ) -> Result<Array> {
        let sched = EdmSchedule::karras(steps, &self.scheduler);

        // CFG conditioning batches (constant across steps): row 0 = uncond (zeros), row 1 = cond.
        let embeds2 = concatenate_axis(&[&zeros_like(image_embeds)?, image_embeds], 0)?; // [2, ctx, 1024]
        let img_lat2 = concatenate_axis(&[&zeros_like(image_latents)?, image_latents], 0)?; // [2,F,h,w,4]
        let atid2 = concatenate_axis(&[added_time_ids, added_time_ids], 0)?; // [2, 3]
        let guidance = Self::guidance_schedule(num_frames, min_g, max_g);

        let mut latents = latents.clone();
        for i in 0..steps {
            let sigma = sched.sigmas[i];
            let sigma_next = sched.sigmas[i + 1];
            let t = sched.timesteps[i];

            let scaled = scale_model_input(&latents, sigma)?; // [1,F,h,w,4]
            let lat2 = concatenate_axis(&[&scaled, &scaled], 0)?; // [2,F,h,w,4]
            let inp = concatenate_axis(&[&lat2, &img_lat2], -1)?; // [2,F,h,w,8]

            let pred = self.unet.forward(&inp, t, &embeds2, &atid2, num_frames)?; // [2,F,h,w,4]
            let uncond = pred.take_axis(Array::from_int(0), 0)?.expand_dims(0)?; // [1,F,h,w,4]
            let cond = pred.take_axis(Array::from_int(1), 0)?.expand_dims(0)?;
            // noise_pred = uncond + guidance · (cond − uncond), frame-wise.
            let noise_pred = add(&uncond, &multiply(&guidance, &subtract(&cond, &uncond)?)?)?;

            let denoised = v_pred_denoised(&noise_pred, &latents, sigma)?;
            latents = euler_step(&latents, &denoised, sigma, sigma_next)?;
        }
        Ok(latents)
    }

    /// Chunked temporal VAE decode (diffusers `decode_latents`): divide by `scaling_factor`, decode in
    /// `chunk`-frame windows, concat. `latents` `[1, F, H/8, W/8, 4]` → frames `[1, F, H, W, 3]`
    /// (roughly `[-1, 1]`; the caller maps to `[0, 1]` for display).
    pub fn decode(&self, latents: &Array, num_frames: i32, chunk: i32) -> Result<Array> {
        let sh = latents.shape();
        // The reshape below collapses `[B, F, h, w, 4] → [num_frames, h, w, 4]`, which only preserves
        // frame identity when `B == 1`; a `B > 1` caller would silently interleave `B·F` frames as
        // `F`. The generator caps `max_count = 1`, but `decode` is public — reject `B > 1` (F-029).
        if sh[0] != 1 {
            return Err(Error::Msg(format!(
                "svd decode: batch size must be 1 (got {})",
                sh[0]
            )));
        }
        let (h, w_) = (sh[2], sh[3]);
        let flat = latents.reshape(&[num_frames, h, w_, sh[4]])?; // [F,h,w,4]
        let z = multiply(&flat, scalar(1.0 / self.vae.scaling_factor()))?;
        let chunk = chunk.max(1);

        let mut start = 0;
        let mut chunks: Vec<Array> = Vec::new();
        while start < num_frames {
            let n = chunk.min(num_frames - start);
            let idx = Array::from_slice(&(start..start + n).collect::<Vec<i32>>(), &[n]);
            let zc = z.take_axis(&idx, 0)?; // [n,h,w,4]
            chunks.push(self.vae.decode(&zc, n)?); // [n,H,W,3]
            start += n;
        }
        let refs: Vec<&Array> = chunks.iter().collect();
        let frames = concatenate_axis(&refs, 0)?; // [F,H,W,3]
        let fs = frames.shape();
        Ok(frames.reshape(&[1, num_frames, fs[1], fs[2], fs[3]])?)
    }
}
