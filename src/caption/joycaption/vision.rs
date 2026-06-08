//! SigLIP vision preprocessing and tower for JoyCaption.
//!
//! JoyCaption uses `google/siglip2-so400m-patch14-384` as the LLaVA vision tower. This module ports
//! the image preprocessing and hidden-state producing vision transformer only; the multimodal
//! projector and language model are separate epic slices.

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, matmul};
use mlx_rs::Array;

use crate::image::resize_bicubic_u8;
use crate::media::Image;
use crate::nn::{conv2d, gelu_tanh};
use crate::weights::Weights;
use crate::{Error, Result};

pub const SIGLIP_IMAGE_SIZE: usize = 384;
pub const SIGLIP_PATCH_SIZE: usize = 14;
pub const SIGLIP_HIDDEN_SIZE: i32 = 1152;
pub const SIGLIP_INTERMEDIATE_SIZE: i32 = 4304;
pub const SIGLIP_NUM_LAYERS: i32 = 27;
pub const SIGLIP_NUM_HEADS: i32 = 16;
pub const SIGLIP_LAYER_NORM_EPS: f32 = 1e-6;
pub const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];
pub const JOYCAPTION_VISION_FEATURE_LAYER: i32 = -2;
pub const JOYCAPTION_VISION_FEATURE_SELECT_STRATEGY: &str = "full";

#[derive(Clone, Debug)]
pub struct SiglipVisionConfig {
    pub image_size: i32,
    pub patch_size: i32,
    pub num_channels: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub layer_norm_eps: f32,
}

impl Default for SiglipVisionConfig {
    fn default() -> Self {
        Self {
            image_size: SIGLIP_IMAGE_SIZE as i32,
            patch_size: SIGLIP_PATCH_SIZE as i32,
            num_channels: 3,
            hidden_size: SIGLIP_HIDDEN_SIZE,
            intermediate_size: SIGLIP_INTERMEDIATE_SIZE,
            num_hidden_layers: SIGLIP_NUM_LAYERS,
            num_attention_heads: SIGLIP_NUM_HEADS,
            layer_norm_eps: SIGLIP_LAYER_NORM_EPS,
        }
    }
}

impl SiglipVisionConfig {
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }

    pub fn grid(&self) -> i32 {
        self.image_size / self.patch_size
    }

    pub fn num_patches(&self) -> i32 {
        let grid = self.grid();
        grid * grid
    }
}

/// RGB uint8 image preprocessing for SigLIP. Output is NHWC `[1, 384, 384, 3]`.
#[derive(Clone, Debug)]
pub struct SiglipImageProcessor {
    pub size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for SiglipImageProcessor {
    fn default() -> Self {
        Self {
            size: SIGLIP_IMAGE_SIZE,
            mean: SIGLIP_MEAN,
            std: SIGLIP_STD,
        }
    }
}

impl SiglipImageProcessor {
    pub fn preprocess(&self, image: &Image) -> Result<Array> {
        let expected = image.width as usize * image.height as usize * 3;
        if image.pixels.len() != expected {
            return Err(Error::Msg(format!(
                "joycaption siglip: expected {} RGB pixels for {}x{}, got {}",
                expected,
                image.width,
                image.height,
                image.pixels.len()
            )));
        }

        let resized: Vec<f32> =
            if image.width as usize == self.size && image.height as usize == self.size {
                image.pixels.iter().map(|&p| p as f32).collect()
            } else {
                resize_bicubic_u8(
                    &image.pixels,
                    image.height as usize,
                    image.width as usize,
                    self.size,
                    self.size,
                )
            };

        let mut normalized = Vec::with_capacity(self.size * self.size * 3);
        for px in resized.chunks_exact(3) {
            for (ch, &v) in px.iter().enumerate() {
                normalized.push((v / 255.0 - self.mean[ch]) / self.std[ch]);
            }
        }
        Ok(Array::from_slice(
            &normalized,
            &[1, self.size as i32, self.size as i32, 3],
        ))
    }
}

pub struct SiglipVisionOutput {
    /// `[B, seq, hidden]` after final post-layernorm.
    pub last_hidden_state: Array,
    /// HF-style hidden states: embeddings output plus one output per encoder layer, before
    /// post-layernorm. JoyCaption reads layer `-2` from this list.
    pub hidden_states: Vec<Array>,
}

pub struct SiglipVisionTower {
    patch_embedding: Array,
    patch_bias: Option<Array>,
    position_embedding: Array,
    layers: Vec<SiglipEncoderLayer>,
    post_ln_w: Array,
    post_ln_b: Array,
    cfg: SiglipVisionConfig,
}

impl SiglipVisionTower {
    /// Load a SigLIP vision tower. `prefix` should point at the HF `vision_model` module, e.g.
    /// `vision_tower.vision_model` for a full LLaVA checkpoint.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: SiglipVisionConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let patch_nchw = w.require(&p("embeddings.patch_embedding.weight"))?;
        let patch_embedding = patch_nchw.transpose_axes(&[0, 2, 3, 1])?;
        let patch_bias = w.get(&p("embeddings.patch_embedding.bias")).cloned();
        let position_embedding = w
            .require(&p("embeddings.position_embedding.weight"))?
            .clone();
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| SiglipEncoderLayer::from_weights(w, &p(&format!("encoder.layers.{i}")), &cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embedding,
            patch_bias,
            position_embedding,
            layers,
            post_ln_w: w.require(&p("post_layernorm.weight"))?.clone(),
            post_ln_b: w.require(&p("post_layernorm.bias"))?.clone(),
            cfg,
        })
    }

    pub fn embeddings(&self, pixel_values: &Array) -> Result<Array> {
        let b = pixel_values.shape()[0];
        let patches = conv2d(
            pixel_values,
            &self.patch_embedding,
            self.patch_bias.as_ref(),
            self.cfg.patch_size,
            0,
        )?;
        let patches = patches.reshape(&[b, self.cfg.num_patches(), self.cfg.hidden_size])?;
        let pos =
            self.position_embedding
                .reshape(&[1, self.cfg.num_patches(), self.cfg.hidden_size])?;
        Ok(add(&patches, &pos)?)
    }

    pub fn forward(&self, pixel_values: &Array) -> Result<SiglipVisionOutput> {
        let mut hidden = self.embeddings(pixel_values)?;
        let mut hidden_states = Vec::with_capacity(self.layers.len() + 1);
        hidden_states.push(hidden.clone());
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
            hidden_states.push(hidden.clone());
        }
        let last_hidden_state = layer_norm(
            &hidden,
            Some(&self.post_ln_w),
            Some(&self.post_ln_b),
            self.cfg.layer_norm_eps,
        )?;
        Ok(SiglipVisionOutput {
            last_hidden_state,
            hidden_states,
        })
    }
}

pub fn select_vision_feature(output: &SiglipVisionOutput, layer: i32) -> Result<Array> {
    if output.hidden_states.is_empty() {
        return Err(Error::Msg(
            "joycaption siglip: no hidden states available".to_owned(),
        ));
    }
    let len = output.hidden_states.len() as i32;
    let idx = if layer < 0 { len + layer } else { layer };
    if idx < 0 || idx >= len {
        return Err(Error::Msg(format!(
            "joycaption siglip: vision feature layer {layer} is out of range for {len} hidden states"
        )));
    }
    Ok(output.hidden_states[idx as usize].clone())
}

pub fn joycaption_vision_features(output: &SiglipVisionOutput) -> Result<Array> {
    select_vision_feature(output, JOYCAPTION_VISION_FEATURE_LAYER)
}

struct SiglipEncoderLayer {
    ln1_w: Array,
    ln1_b: Array,
    ln2_w: Array,
    ln2_b: Array,
    attn: SiglipAttention,
    mlp: SiglipMlp,
    eps: f32,
}

impl SiglipEncoderLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &SiglipVisionConfig) -> Result<Self> {
        Ok(Self {
            ln1_w: w.require(&join(prefix, "layer_norm1.weight"))?.clone(),
            ln1_b: w.require(&join(prefix, "layer_norm1.bias"))?.clone(),
            ln2_w: w.require(&join(prefix, "layer_norm2.weight"))?.clone(),
            ln2_b: w.require(&join(prefix, "layer_norm2.bias"))?.clone(),
            attn: SiglipAttention::from_weights(w, &join(prefix, "self_attn"), cfg)?,
            mlp: SiglipMlp::from_weights(w, &join(prefix, "mlp"))?,
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let y = layer_norm(x, Some(&self.ln1_w), Some(&self.ln1_b), self.eps)?;
        let x = add(x, &self.attn.forward(&y)?)?;
        let y = layer_norm(&x, Some(&self.ln2_w), Some(&self.ln2_b), self.eps)?;
        Ok(add(&x, &self.mlp.forward(&y)?)?)
    }
}

struct SiglipAttention {
    q_w: Array,
    q_b: Option<Array>,
    k_w: Array,
    k_b: Option<Array>,
    v_w: Array,
    v_b: Option<Array>,
    out_w: Array,
    out_b: Option<Array>,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl SiglipAttention {
    fn from_weights(w: &Weights, prefix: &str, cfg: &SiglipVisionConfig) -> Result<Self> {
        let get_bias = |leaf: &str| w.get(&join(prefix, leaf)).cloned();
        let head_dim = cfg.head_dim();
        Ok(Self {
            q_w: w.require(&join(prefix, "q_proj.weight"))?.clone(),
            q_b: get_bias("q_proj.bias"),
            k_w: w.require(&join(prefix, "k_proj.weight"))?.clone(),
            k_b: get_bias("k_proj.bias"),
            v_w: w.require(&join(prefix, "v_proj.weight"))?.clone(),
            v_b: get_bias("v_proj.bias"),
            out_w: w.require(&join(prefix, "out_proj.weight"))?.clone(),
            out_b: get_bias("out_proj.bias"),
            num_heads: cfg.num_attention_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, n) = (sh[0], sh[1]);
        let to_heads = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[b, n, self.num_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_heads(linear_opt(x, &self.q_w, self.q_b.as_ref())?)?;
        let k = to_heads(linear_opt(x, &self.k_w, self.k_b.as_ref())?)?;
        let v = to_heads(linear_opt(x, &self.v_w, self.v_b.as_ref())?)?;
        let out = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let out =
            out.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, n, self.num_heads * self.head_dim])?;
        linear_opt(&out, &self.out_w, self.out_b.as_ref())
    }
}

struct SiglipMlp {
    fc1_w: Array,
    fc1_b: Option<Array>,
    fc2_w: Array,
    fc2_b: Option<Array>,
}

impl SiglipMlp {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1_w: w.require(&join(prefix, "fc1.weight"))?.clone(),
            fc1_b: w.get(&join(prefix, "fc1.bias")).cloned(),
            fc2_w: w.require(&join(prefix, "fc2.weight"))?.clone(),
            fc2_b: w.get(&join(prefix, "fc2.bias")).cloned(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = linear_opt(x, &self.fc1_w, self.fc1_b.as_ref())?;
        let x = gelu_tanh(&x)?;
        linear_opt(&x, &self.fc2_w, self.fc2_b.as_ref())
    }
}

fn linear_opt(x: &Array, w: &Array, b: Option<&Array>) -> Result<Array> {
    let y = matmul(x, w.t())?;
    Ok(if let Some(b) = b { add(&y, b)? } else { y })
}

fn join(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_owned()
    } else {
        format!("{prefix}.{leaf}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caption::joycaption::IMAGE_SEQ_LENGTH;

    fn host_f32(a: &Array) -> Vec<f32> {
        a.try_as_slice::<f32>().expect("readable f32").to_vec()
    }

    #[test]
    fn default_config_matches_joycaption_siglip() {
        let cfg = SiglipVisionConfig::default();
        assert_eq!(cfg.image_size, 384);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.hidden_size, 1152);
        assert_eq!(cfg.intermediate_size, 4304);
        assert_eq!(cfg.num_hidden_layers, 27);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_patches(), IMAGE_SEQ_LENGTH as i32);
        assert_eq!(cfg.head_dim(), 72);
    }

    #[test]
    fn preprocess_normalizes_rgb_to_minus_one_one() {
        let image = Image {
            width: 384,
            height: 384,
            pixels: vec![0u8, 128, 255].repeat(384 * 384),
        };
        let out = SiglipImageProcessor::default()
            .preprocess(&image)
            .expect("preprocess");
        assert_eq!(out.shape(), &[1, 384, 384, 3]);
        let vals = host_f32(&out);
        assert_eq!(vals[0], -1.0);
        assert!((vals[1] - 0.003_921_628).abs() < 1e-6);
        assert_eq!(vals[2], 1.0);
    }

    #[test]
    fn preprocess_rejects_bad_rgb_buffer() {
        let image = Image {
            width: 2,
            height: 2,
            pixels: vec![0u8; 3],
        };
        assert!(SiglipImageProcessor::default().preprocess(&image).is_err());
    }

    #[test]
    fn feature_layer_negative_index_selects_penultimate() {
        let hidden_states = vec![
            Array::from_slice(&[1.0f32], &[1, 1, 1]),
            Array::from_slice(&[2.0f32], &[1, 1, 1]),
            Array::from_slice(&[3.0f32], &[1, 1, 1]),
        ];
        let output = SiglipVisionOutput {
            last_hidden_state: hidden_states[2].clone(),
            hidden_states,
        };
        let selected = select_vision_feature(&output, -2).expect("selected");
        assert_eq!(host_f32(&selected), vec![2.0]);
        assert!(select_vision_feature(&output, -4).is_err());
    }
}
