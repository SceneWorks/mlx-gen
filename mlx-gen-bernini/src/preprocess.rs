//! Source-media preprocessing — decoded conditioning [`Image`]s (the worker owns file I/O) → resized
//! `[-1,1]` pixel tensors → `WanVae::encode` normalized z16 latents, the `videos`/`images` lists
//! [`crate::forward::guided_velocity`] consumes.
//!
//! Reuses [`mlx_gen_wan::pipeline::preprocess_i2v_image`] (resize to the target W×H, RGB→`[-1,1]`
//! CHW). Conditioning is resized to the **output** geometry here (a faithful first approximation; the
//! reference resizes each source to its own aspect under `max_image_size` — refine if a media-mode
//! parity gap appears).

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::media::Image;
use mlx_gen::{Error, Result};
use mlx_gen_wan::pipeline::preprocess_i2v_image;
use mlx_gen_wan::WanVae;

/// One conditioning image `[1,16,1,H/8,W/8]` (z16, normalized): resize → `[-1,1]` `[3,H,W]` →
/// `[1,3,1,H,W]` → `WanVae::encode`.
pub fn encode_image(vae: &WanVae, image: &Image, width: u32, height: u32) -> Result<Array> {
    let chw = preprocess_i2v_image(image, width, height)?; // [3, H, W] in [-1, 1]
    let video = chw.expand_dims(0)?.expand_dims(2)?; // [1, 3, 1, H, W]
    vae.encode(&video)
}

/// One conditioning video clip `[1,16,T_lat,H/8,W/8]`: each frame resized to `[3,H,W]`, stacked on the
/// temporal axis → `[1,3,T,H,W]` (T must be `1 + 4k`), → `WanVae::encode`.
pub fn encode_videoclip(vae: &WanVae, frames: &[Image], width: u32, height: u32) -> Result<Array> {
    if frames.is_empty() {
        return Err(Error::Msg(
            "bernini_renderer: empty conditioning video clip".into(),
        ));
    }
    if frames.len() % 4 != 1 {
        return Err(Error::Msg(format!(
            "bernini_renderer: video-clip frame count must be 1 + 4·k (got {})",
            frames.len()
        )));
    }
    let mut chw_t = Vec::with_capacity(frames.len());
    for f in frames {
        // [3, H, W] → [3, 1, H, W] for the temporal concat.
        chw_t.push(preprocess_i2v_image(f, width, height)?.expand_dims(1)?);
    }
    let refs: Vec<&Array> = chw_t.iter().collect();
    let video = concatenate_axis(&refs, 1)?.expand_dims(0)?; // [1, 3, T, H, W]
    vae.encode(&video)
}
