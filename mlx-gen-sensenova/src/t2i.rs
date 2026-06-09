//! Text-to-image generation (sc-3188) — the `t2i_generate` spine end to end.
//!
//! Ports `modeling_neo_chat.py::t2i_generate` for the dense 8B-MoT checkpoint. The flow:
//!
//! 1. Build the `neo1_0` query ([`build_neo1_query`] + [`SYSTEM_MESSAGE_FOR_GEN`] + the think
//!    sentinel), tokenize, and **prefill** it into a KV cache on the understanding path
//!    ([`Qwen3Backbone::forward_cached`] append). With CFG (`cfg_scale > 1`) a second, *uncondition*
//!    prefix (`<img>` after an empty prompt) is prefilled into its own cache.
//! 2. (think-mode) run the [`Qwen3Backbone::generate_think`] rollout, extending the cache and
//!    placing the image block after the appended `\n\n<img>`.
//! 3. **Denoise** for `num_steps` over the standard flow-matching schedule
//!    ([`apply_time_schedule`]): each step embeds the current noisy image through the gen-path
//!    [`NeoVisionEmbedder`] (channel-first patches) + the timestep (and noise-scale) embedding, runs
//!    the **generation** path over `[cached prefix ++ image block]` via `forward_cached`
//!    **use-only** (`update_cache=False`), maps the image hidden states through the [`FmHead`] to a
//!    patch latent `x_pred`, forms the [`velocity`], and takes an [`euler_step`]. CFG blends the
//!    condition/uncondition velocities ([`CfgNorm`] variants).
//! 4. [`unpatchify`] the final latent → RGB `[1, 3, H, W]`.
//!
//! The pixel path is the `fm_head` → unpatchify (`use_pixel_head = false`); the conv decoders and the
//! dynamic-μ schedule are dead code for this checkpoint.

use mlx_rs::ops::{add, divide, minimum, multiply, subtract, sum_axes};
use mlx_rs::{Array, Dtype};

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::NeoChatConfig;
use crate::fm::{
    apply_time_schedule, euler_step, patchify, patchify_channel_first, unpatchify, velocity,
    FmHead, TimestepEmbedder,
};
use crate::qwen3::{KvCache, Path, Qwen3Backbone};
use crate::text::{build_neo1_query, image_indexes, text_indexes, tokens, SYSTEM_MESSAGE_FOR_GEN};
use crate::vision::NeoVisionEmbedder;

/// Classifier-free-guidance velocity-blend normalisation (`t2i_generate`'s `cfg_norm`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CfgNorm {
    /// Plain blend `v_uncond + cfg·(v_cond − v_uncond)`.
    #[default]
    None,
    /// Rescale the blended velocity to the condition velocity's global norm.
    Global,
    /// Per-token rescale to the condition velocity's per-token norm.
    Channel,
    /// CFG-Zero* (optimised-scale uncondition + step-0 zeroing).
    CfgZeroStar,
}

/// Knobs for [`T2iModel::generate`] (the `t2i_generate` arguments).
#[derive(Clone, Copy, Debug)]
pub struct T2iOptions {
    pub cfg_scale: f32,
    pub cfg_norm: CfgNorm,
    pub cfg_interval: (f32, f32),
    pub num_steps: usize,
    pub timestep_shift: f32,
    pub enable_timestep_shift: bool,
    pub t_eps: f32,
    pub seed: u64,
    pub think_mode: bool,
    pub max_think_tokens: usize,
}

impl Default for T2iOptions {
    fn default() -> Self {
        Self {
            cfg_scale: 1.0,
            cfg_norm: CfgNorm::None,
            cfg_interval: (0.0, 1.0),
            num_steps: 30,
            timestep_shift: 1.0,
            enable_timestep_shift: true,
            t_eps: 0.02,
            seed: 0,
            think_mode: false,
            max_think_tokens: 1024,
        }
    }
}

/// The result of a [`T2iModel::generate`] run.
pub struct T2iOutput {
    /// The generated image `[1, 3, H, W]` (model space, roughly `[-1, 1]`).
    pub image: Array,
    /// The decoded think-block text, when `think_mode` was set.
    pub think_text: Option<String>,
}

/// The T2I model: the backbone plus the flow-matching generation modules.
pub struct T2iModel {
    backbone: Qwen3Backbone,
    gen_vision: NeoVisionEmbedder,
    fm_head: FmHead,
    timestep_embedder: TimestepEmbedder,
    noise_scale_embedder: Option<TimestepEmbedder>,
    patch_size: i32,
    merge_size: i32,
    noise_scale: f32,
    noise_scale_mode: String,
    noise_scale_base_image_seq_len: f32,
    noise_scale_max_value: f32,
}

impl T2iModel {
    /// Build from a loaded checkpoint (`language_model.*` + `fm_modules.*`).
    pub fn from_weights(w: &Weights, cfg: &NeoChatConfig) -> Result<Self> {
        let noise_scale_embedder = if cfg.add_noise_scale_embedding {
            Some(TimestepEmbedder::from_weights(
                w,
                "fm_modules.noise_scale_embedder",
            )?)
        } else {
            None
        };
        Ok(Self {
            backbone: Qwen3Backbone::from_weights(w, cfg, "language_model")?,
            gen_vision: NeoVisionEmbedder::from_weights(
                w,
                cfg,
                "fm_modules.vision_model_mot_gen.embeddings",
            )?,
            fm_head: FmHead::from_weights(w, "fm_modules.fm_head")?,
            timestep_embedder: TimestepEmbedder::from_weights(w, "fm_modules.timestep_embedder")?,
            noise_scale_embedder,
            patch_size: cfg.patch_size as i32,
            merge_size: (1.0 / cfg.downsample_ratio).round() as i32,
            noise_scale: cfg.noise_scale,
            noise_scale_mode: cfg.noise_scale_mode.clone(),
            noise_scale_base_image_seq_len: cfg.noise_scale_base_image_seq_len as f32,
            noise_scale_max_value: cfg.noise_scale_max_value,
        })
    }

    /// The resolution-mode noise scale for a `grid_h × grid_w` patch grid (the `t2i_generate`
    /// formula). For non-resolution modes the bare `noise_scale` is used; both are clamped to
    /// `noise_scale_max_value`.
    fn noise_scale_for(&self, grid_h: i32, grid_w: i32) -> f32 {
        let mut scale = self.noise_scale;
        if matches!(
            self.noise_scale_mode.as_str(),
            "resolution" | "dynamic" | "dynamic_sqrt"
        ) {
            let seq = (grid_h * grid_w) as f32 / (self.merge_size * self.merge_size) as f32;
            scale = (seq / self.noise_scale_base_image_seq_len).sqrt() * self.noise_scale;
            if self.noise_scale_mode == "dynamic_sqrt" {
                scale = scale.sqrt();
            }
        }
        scale.min(self.noise_scale_max_value)
    }

    /// Run the gen-path velocity prediction for one diffusion step against a prefilled cache
    /// (`_t2i_predict_v`): `forward_cached` (Gen path, use-only) over the image block, `fm_head` →
    /// `x_pred`, then the flow-matching velocity. `image_embeds` is the vision+timestep conditioned
    /// image block `[1, L, hidden]`; `text_len` is the prefix length the block sits after.
    #[allow(clippy::too_many_arguments)]
    fn predict_v(
        &self,
        image_embeds: &Array,
        token_h: i32,
        token_w: i32,
        text_len: usize,
        cache: &mut KvCache,
        z: &Array,
        t: f32,
        t_eps: f32,
    ) -> Result<Array> {
        let (it, ih, iw) = image_indexes(token_h as usize, token_w as usize, text_len);
        let hidden =
            self.backbone
                .forward_cached(image_embeds, &it, &ih, &iw, Path::Gen, cache, false)?;
        let x_pred = self.fm_head.forward(&hidden)?;
        velocity(&x_pred, z, t, t_eps)
    }

    /// Prefill a text query into a fresh cache on the understanding path. Returns the cache, the
    /// last-position logits (for think-mode), and the prefix token length.
    fn prefill(&self, ids: &[i32]) -> Result<(KvCache, Array, usize)> {
        let n = ids.len() as i32;
        let ids_arr = Array::from_slice(ids, &[1, n]);
        let embeds = self.backbone.embed(&ids_arr)?;
        let (t, h, wid) = text_indexes(ids.len());
        let mut cache = self.backbone.new_cache();
        let hidden =
            self.backbone
                .forward_cached(&embeds, &t, &h, &wid, Path::Und, &mut cache, true)?;
        let logits = self.backbone.lm_head(&hidden)?; // [1, S, vocab]
        let vocab = logits.shape()[2];
        let last = logits
            .take_axis(Array::from_slice(&[n - 1], &[1]), 1)?
            .reshape(&[vocab])?;
        Ok((cache, last, ids.len()))
    }

    /// Generate an image for `prompt` at `width × height` (both multiples of `patch·merge = 32`).
    /// `init_noise`, when supplied, is a standard-normal tensor `[1, 3, H, W]` used in place of
    /// fresh sampling (for cross-build parity); it is scaled by the resolution-mode `noise_scale`.
    pub fn generate(
        &self,
        tokenizer: &TextTokenizer,
        prompt: &str,
        width: i32,
        height: i32,
        opts: &T2iOptions,
        init_noise: Option<&Array>,
    ) -> Result<T2iOutput> {
        let cell = self.patch_size * self.merge_size;
        if width % cell != 0 || height % cell != 0 {
            return Err(Error::Msg(format!(
                "sensenova t2i: width/height must be multiples of {cell}, got {width}x{height}"
            )));
        }

        // ---- Condition prefix ----
        let think_sentinel = if opts.think_mode {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n<img>"
        };
        let query_cond = format!(
            "{}{}",
            build_neo1_query(prompt, SYSTEM_MESSAGE_FOR_GEN),
            think_sentinel
        );
        let ids_cond = tokenizer.encode_ids(&query_cond, true)?;
        let (mut cache_cond, last_logits, prefix_len) = self.prefill(&ids_cond)?;

        // think-mode: roll out the reasoning block, then append `\n\n<img>`.
        let mut think_text = None;
        let mut text_len = prefix_len;
        if opts.think_mode {
            let append_ids = tokenizer.encode_ids("\n\n<img>", false)?;
            let roll = self.backbone.generate_think(
                last_logits.as_slice::<f32>(),
                &mut cache_cond,
                (prefix_len - 1) as i32,
                tokens::THINK_END,
                tokens::IM_END,
                &append_ids,
                opts.max_think_tokens,
            )?;
            let ids_u32: Vec<u32> = roll.think_token_ids.iter().map(|&i| i as u32).collect();
            think_text = Some(tokenizer.decode(&ids_u32, false)?);
            text_len = (roll.t_idx + 1) as usize;
        }

        // ---- Uncondition prefix (CFG) ----
        let needs_cfg = opts.cfg_scale > 1.0;
        let mut cache_uncond = None;
        if needs_cfg {
            let query_uncond = format!("{}<img>", build_neo1_query("", ""));
            let ids_uncond = tokenizer.encode_ids(&query_uncond, true)?;
            let (cache, _, plen) = self.prefill(&ids_uncond)?;
            cache_uncond = Some((cache, plen));
        }

        let base_noise = match init_noise {
            Some(n) => n.as_dtype(Dtype::Float32)?,
            None => gaussian(&[1, 3, height, width], opts.seed)?,
        };
        let cond_u = cache_uncond.as_mut().map(|(c, l)| (c, *l));
        let traj = self.denoise(
            &mut cache_cond,
            text_len,
            cond_u,
            width,
            height,
            &base_noise,
            opts,
        )?;
        let image = traj.into_iter().last().expect("at least one step");
        Ok(T2iOutput { image, think_text })
    }

    /// Prefill `ids` into a fresh understanding-path cache; returns the cache and prefix length.
    /// Exposed for tests/callers that drive [`T2iModel::denoise`] with an explicit prefix.
    pub fn prefill_ids(&self, ids: &[i32]) -> Result<(KvCache, usize)> {
        let (cache, _, len) = self.prefill(ids)?;
        Ok((cache, len))
    }

    /// The flow-matching denoise loop. `cache_cond` (and the optional `(cache_uncond, text_len)` for
    /// CFG) are prefilled understanding-path caches; `base_noise` is a standard-normal `[1,3,H,W]`
    /// tensor (scaled here by the resolution-mode noise scale). Returns the per-step image
    /// trajectory `[1,3,H,W]` (the last entry is the final image).
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        cache_cond: &mut KvCache,
        text_len: usize,
        mut cache_uncond: Option<(&mut KvCache, usize)>,
        width: i32,
        height: i32,
        base_noise: &Array,
        opts: &T2iOptions,
    ) -> Result<Vec<Array>> {
        let cell = self.patch_size * self.merge_size;
        let token_h = height / cell;
        let token_w = width / cell;
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        let l = token_h * token_w;

        let noise_scale = self.noise_scale_for(grid_h, grid_w);
        let mut image = multiply(
            &base_noise.as_dtype(Dtype::Float32)?,
            Array::from_f32(noise_scale),
        )?;

        let steps = opts.num_steps;
        let lin: Vec<f32> = (0..=steps).map(|i| i as f32 / steps as f32).collect();
        let lin_arr = Array::from_slice(&lin, &[(steps + 1) as i32]);
        let ts_arr = if opts.enable_timestep_shift {
            apply_time_schedule(&lin_arr, opts.timestep_shift)?
        } else {
            lin_arr
        };
        let timesteps = ts_arr.as_slice::<f32>().to_vec();

        // Constant noise-scale conditioning token (added to every step's timestep embedding).
        let noise_embed = if let Some(emb) = &self.noise_scale_embedder {
            let ns = vec![noise_scale / self.noise_scale_max_value; l as usize];
            Some(
                emb.forward(&Array::from_slice(&ns, &[l]))?
                    .reshape(&[1, l, -1])?,
            )
        } else {
            None
        };

        let needs_cfg = opts.cfg_scale > 1.0 && cache_uncond.is_some();
        let mut traj = Vec::with_capacity(steps);
        for i in 0..steps {
            let t = timesteps[i];
            let t_next = timesteps[i + 1];

            let z = patchify(&image, cell)?;
            let image_input =
                patchify_channel_first(&image, self.patch_size)?.reshape(&[grid_h * grid_w, -1])?;
            let vis = self
                .gen_vision
                .forward(&image_input, &[(grid_h as usize, grid_w as usize)])?
                .reshape(&[1, l, -1])?;
            let t_tok = self
                .timestep_embedder
                .forward(&Array::from_slice(&vec![t; l as usize], &[l]))?
                .reshape(&[1, l, -1])?;
            let mut cond = add(&vis, &t_tok)?;
            if let Some(ne) = &noise_embed {
                cond = add(&cond, ne)?;
            }

            let v_cond = self.predict_v(
                &cond, token_h, token_w, text_len, cache_cond, &z, t, opts.t_eps,
            )?;

            let v_pred = if needs_cfg && t >= opts.cfg_interval.0 && t <= opts.cfg_interval.1 {
                let (cache_u, tlu) = cache_uncond.as_mut().unwrap();
                let v_uncond =
                    self.predict_v(&cond, token_h, token_w, *tlu, cache_u, &z, t, opts.t_eps)?;
                cfg_blend(&v_cond, &v_uncond, opts.cfg_scale, opts.cfg_norm, i)?
            } else {
                v_cond
            };

            image = unpatchify(
                &euler_step(&v_pred, &z, t, t_next)?,
                cell,
                Some(token_h),
                Some(token_w),
            )?;
            traj.push(image.clone());
        }
        Ok(traj)
    }
}

/// Blend condition/uncondition velocities under the chosen [`CfgNorm`].
fn cfg_blend(
    v_cond: &Array,
    v_uncond: &Array,
    scale: f32,
    norm: CfgNorm,
    step: usize,
) -> Result<Array> {
    if norm == CfgNorm::CfgZeroStar {
        // CFG-Zero*: project uncond onto cond (optimised scale), zero step 0.
        if step == 0 {
            return multiply(v_cond, Array::from_f32(0.0)).map_err(Error::from);
        }
        let alpha = optimized_scale(v_cond, v_uncond)?;
        let scaled_u = multiply(v_uncond, Array::from_f32(alpha))?;
        let guided = multiply(&subtract(v_cond, &scaled_u)?, Array::from_f32(scale))?;
        return add(&scaled_u, &guided).map_err(Error::from);
    }

    let diff = subtract(v_cond, v_uncond)?;
    let blended = add(v_uncond, &multiply(&diff, Array::from_f32(scale))?)?;
    match norm {
        CfgNorm::Global => {
            let nc = frobenius(v_cond)?;
            let nb = frobenius(&blended)?;
            let s = (nc / (nb + 1e-8)).clamp(0.0, 1.0);
            multiply(&blended, Array::from_f32(s)).map_err(Error::from)
        }
        CfgNorm::Channel => {
            // Per-token (last-axis) norm rescale, clamped to ≤ 1 (norms are ≥ 0).
            let nc = l2_last(v_cond)?;
            let nb = l2_last(&blended)?;
            let ratio = divide(&nc, &add(&nb, Array::from_f32(1e-8))?)?;
            let s = minimum(&ratio, Array::from_f32(1.0))?;
            multiply(&blended, &s).map_err(Error::from)
        }
        _ => Ok(blended),
    }
}

/// `‖x‖₂` over the whole tensor (the reference `torch.norm(v, dim=(1,2))` for batch 1).
fn frobenius(x: &Array) -> Result<f32> {
    Ok(sum_all(&multiply(x, x)?)?.sqrt())
}

/// Per-token L2 norm over the last axis, keeping dims: `[1,L,D] → [1,L,1]`.
fn l2_last(x: &Array) -> Result<Array> {
    let rank = x.shape().len() as i32;
    sum_axes(&multiply(x, x)?, &[rank - 1], true)?
        .sqrt()
        .map_err(Error::from)
}

/// Sum every element to a scalar.
fn sum_all(x: &Array) -> Result<f32> {
    let axes: Vec<i32> = (0..x.shape().len() as i32).collect();
    Ok(sum_axes(x, &axes, false)?.item::<f32>())
}

/// CFG-Zero* optimised scale `⟨cond,uncond⟩ / ‖uncond‖²` (computed in f32).
fn optimized_scale(v_cond: &Array, v_uncond: &Array) -> Result<f32> {
    let dot = sum_all(&multiply(v_cond, v_uncond)?)?;
    let nrm = sum_all(&multiply(v_uncond, v_uncond)?)?;
    Ok(dot / (nrm + 1e-8))
}

/// `smart_resize` (Qwen2.5-VL, the vendored `utils.smart_resize`): round `height`/`width` to
/// multiples of `factor` (use `patch·merge = 32`) with total pixels held in `[min_pixels,
/// max_pixels]`. Returns `(height, width)`.
pub fn smart_resize(
    height: i32,
    width: i32,
    factor: i32,
    min_pixels: i64,
    max_pixels: i64,
) -> (i32, i32) {
    let round_by = |n: f64| ((n / factor as f64).round() as i32) * factor;
    let floor_by = |n: f64| ((n / factor as f64).floor() as i32) * factor;
    let ceil_by = |n: f64| ((n / factor as f64).ceil() as i32) * factor;
    let (hf, wf) = (height as f64, width as f64);
    let mut h_bar = factor.max(round_by(hf));
    let mut w_bar = factor.max(round_by(wf));
    let area = (h_bar as i64) * (w_bar as i64);
    if area > max_pixels {
        let beta = ((hf * wf) / max_pixels as f64).sqrt();
        h_bar = factor.max(floor_by(hf / beta));
        w_bar = factor.max(floor_by(wf / beta));
    } else if area < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        h_bar = ceil_by(hf * beta);
        w_bar = ceil_by(wf * beta);
    }
    (h_bar, w_bar)
}

/// Standard-normal `[shape]` via Box–Muller over a SplitMix64 stream (deterministic per `seed`).
fn gaussian(shape: &[i32], seed: u64) -> Result<Array> {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next_f = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // (0, 1] to keep ln() finite.
        ((z >> 11) as f64 + 1.0) / ((1u64 << 53) as f64)
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next_f();
        let u2 = next_f();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < n {
            out.push((r * theta.sin()) as f32);
        }
    }
    Ok(Array::from_slice(&out, shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice(a: &Array) -> Vec<f32> {
        let n = a.shape().iter().product::<i32>();
        a.reshape(&[n]).unwrap().as_slice::<f32>().to_vec()
    }

    #[test]
    fn cfg_blend_none_is_linear_extrapolation() {
        // v_uncond + scale·(v_cond − v_uncond): [1,1,2] tensors.
        let v_cond = Array::from_slice(&[2.0f32, 4.0], &[1, 1, 2]);
        let v_uncond = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 2]);
        let out = cfg_blend(&v_cond, &v_uncond, 3.0, CfgNorm::None, 1).unwrap();
        // 1 + 3·(2−1) = 4 ; 1 + 3·(4−1) = 10
        assert_eq!(slice(&out), vec![4.0, 10.0]);
    }

    #[test]
    fn cfg_zero_star_zeroes_first_step() {
        let v_cond = Array::from_slice(&[2.0f32, 4.0], &[1, 1, 2]);
        let v_uncond = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 2]);
        let out = cfg_blend(&v_cond, &v_uncond, 3.0, CfgNorm::CfgZeroStar, 0).unwrap();
        assert_eq!(slice(&out), vec![0.0, 0.0]);
    }

    #[test]
    fn global_norm_never_amplifies() {
        // Blended norm > cond norm → scale clamps to keep ‖blended‖ ≤ ‖cond‖.
        let v_cond = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 2]);
        let v_uncond = Array::from_slice(&[-2.0f32, -2.0], &[1, 1, 2]);
        let out = cfg_blend(&v_cond, &v_uncond, 4.0, CfgNorm::Global, 1).unwrap();
        let on = (slice(&out).iter().map(|x| x * x).sum::<f32>()).sqrt();
        let cn = (2.0f32).sqrt();
        assert!(
            on <= cn + 1e-4,
            "global-norm output {on} exceeds cond norm {cn}"
        );
    }

    #[test]
    fn smart_resize_upscales_to_min_pixels() {
        // 100×100 rounds to 96×96 (< 65536 px) → upscaled to the 256×256 bucket.
        assert_eq!(smart_resize(100, 100, 32, 65536, 4_194_304), (256, 256));
    }

    #[test]
    fn smart_resize_keeps_in_range_multiple() {
        assert_eq!(smart_resize(512, 512, 32, 65536, 4_194_304), (512, 512));
        // Non-multiples round to the nearest factor.
        assert_eq!(smart_resize(500, 500, 32, 65536, 4_194_304), (512, 512));
    }
}
