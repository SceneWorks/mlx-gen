//! FLUX.2 VAE (`AutoencoderKLFlux2`) — a 32-channel diffusers AutoencoderKL with two FLUX.2
//! additions: a **2×2 patchify** that folds the 32-ch latent into the 128-ch transformer space,
//! and a **BatchNorm-stats** normalization of that packed space (`bn.running_mean/var`). Port of
//! the fork's `models/flux2/model/flux2_vae/`.
//!
//! Structurally identical to the SDXL VAE (encoder/decoder, resnets, single-head mid attention,
//! GroupNorm) but with `block_out_channels = (128, 256, 512, 512)`, `latent_channels = 32`,
//! GroupNorm eps **1e-6** (SDXL uses 1e-5), `scaling_factor = 1.0`, `shift_factor = 0.0`. Runs
//! entirely in NHWC, f32 (the VAE is small; f32 dodges the bf16-GEMM bug in the mid attention and
//! is the quality target).

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, multiply, pad, sqrt};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::nn::{conv2d, group_norm, linear, silu, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-6;
const BN_EPS: f32 = 1e-4;
const LATENT_CHANNELS: i32 = 32;
const BLOCK_OUT: [i32; 4] = [128, 256, 512, 512];
const LAYERS_PER_BLOCK: i32 = 2;

/// `[O, I, H, W]` (PyTorch) → `[O, H, W, I]` (mlx conv2d), cast to f32.
fn conv_w(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?
        .transpose_axes(&[0, 2, 3, 1])?
        .as_dtype(Dtype::Float32)?)
}

fn f32w(w: &Weights, key: &str) -> Result<Array> {
    Ok(w.require(key)?.as_dtype(Dtype::Float32)?)
}

/// A 1×1 conv expressed as a channel-wise Linear `[O, I]` (+ bias), f32.
fn squeeze_linear(w: &Weights, name: &str) -> Result<(Array, Array)> {
    let cw = w.require(&format!("{name}.weight"))?;
    let sh = cw.shape();
    Ok((
        cw.reshape(&[sh[0], sh[1]])?.as_dtype(Dtype::Float32)?,
        f32w(w, &format!("{name}.bias"))?,
    ))
}

/// VAE resnet block (temb-free): `silu(gn1(x)) → conv1 → silu(gn2) → conv2 + shortcut`.
struct ResnetBlock2D {
    norm1_w: Array,
    norm1_b: Array,
    conv1_w: Array,
    conv1_b: Array,
    norm2_w: Array,
    norm2_b: Array,
    conv2_w: Array,
    conv2_b: Array,
    shortcut: Option<(Array, Array)>,
}

impl ResnetBlock2D {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let shortcut = if w.get(&format!("{prefix}.conv_shortcut.weight")).is_some() {
            Some((
                conv_w(w, &format!("{prefix}.conv_shortcut.weight"))?,
                f32w(w, &format!("{prefix}.conv_shortcut.bias"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            norm1_w: f32w(w, &format!("{prefix}.norm1.weight"))?,
            norm1_b: f32w(w, &format!("{prefix}.norm1.bias"))?,
            conv1_w: conv_w(w, &format!("{prefix}.conv1.weight"))?,
            conv1_b: f32w(w, &format!("{prefix}.conv1.bias"))?,
            norm2_w: f32w(w, &format!("{prefix}.norm2.weight"))?,
            norm2_b: f32w(w, &format!("{prefix}.norm2.bias"))?,
            conv2_w: conv_w(w, &format!("{prefix}.conv2.weight"))?,
            conv2_b: f32w(w, &format!("{prefix}.conv2.bias"))?,
            shortcut,
        })
    }

    /// `x`: NHWC.
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = group_norm(x, &self.norm1_w, &self.norm1_b, GN_GROUPS, GN_EPS)?;
        let h = conv2d(&silu(&h)?, &self.conv1_w, Some(&self.conv1_b), 1, 1)?;
        let h = group_norm(&h, &self.norm2_w, &self.norm2_b, GN_GROUPS, GN_EPS)?;
        let h = conv2d(&silu(&h)?, &self.conv2_w, Some(&self.conv2_b), 1, 1)?;
        let res = match &self.shortcut {
            Some((cw, cb)) => conv2d(x, cw, Some(cb), 1, 0)?,
            None => x.clone(),
        };
        Ok(add(&h, &res)?)
    }
}

/// Single-head spatial self-attention used in the mid block (the fork's `Flux2AttentionBlock`).
/// q/k/v/out are `nn.Linear` (with bias) — the only VAE modules the fork's `nn.quantize` hits, so
/// they are core [`AdaptableLinear`]s; the GroupNorm stays full precision (as do all the convs).
struct VaeAttention {
    gn_w: Array,
    gn_b: Array,
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
}

impl VaeAttention {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        // q/k/v/out carry bias; weights are loaded f32 (the VAE runs f32). `quantize` casts to bf16
        // before packing so the scales byte-match the fork's bf16 `nn.quantize` (sc-2604 chokepoint).
        let lin = |n: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                f32w(w, &format!("{prefix}.{n}.weight"))?,
                Some(f32w(w, &format!("{prefix}.{n}.bias"))?),
            ))
        };
        Ok(Self {
            gn_w: f32w(w, &format!("{prefix}.group_norm.weight"))?,
            gn_b: f32w(w, &format!("{prefix}.group_norm.bias"))?,
            q: lin("to_q")?,
            k: lin("to_k")?,
            v: lin("to_v")?,
            o: lin("to_out.0")?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.q.quantize(bits, None)?;
        self.k.quantize(bits, None)?;
        self.v.quantize(bits, None)?;
        self.o.quantize(bits, None)?;
        Ok(())
    }

    /// `x`: NHWC `[B, H, W, C]`. Single-head attention over the H·W positions, residual.
    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, h, w_, c) = (sh[0], sh[1], sh[2], sh[3]);
        let y = group_norm(x, &self.gn_w, &self.gn_b, GN_GROUPS, GN_EPS)?;
        let to_seq = |a: Array| -> Result<Array> { Ok(a.reshape(&[b, 1, h * w_, c])?) };
        let q = to_seq(self.q.forward(&y)?)?;
        let k = to_seq(self.k.forward(&y)?)?;
        let v = to_seq(self.v.forward(&y)?)?;
        let scale = (c as f32).powf(-0.5);
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let o = self.o.forward(&o.reshape(&[b, h, w_, c])?)?;
        Ok(add(x, &o)?)
    }
}

/// A run of resnets, then an optional downsample (asymmetric-pad + stride-2 conv) or upsample
/// (nearest-2× + conv). Port of `Flux2{Down,Up}EncoderBlock2D`.
struct SampleBlock {
    resnets: Vec<ResnetBlock2D>,
    downsample: Option<(Array, Array)>,
    upsample: Option<(Array, Array)>,
}

impl SampleBlock {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        num_resnets: i32,
        down: bool,
        up: bool,
    ) -> Result<Self> {
        let resnets = (0..num_resnets)
            .map(|j| ResnetBlock2D::from_weights(w, &format!("{prefix}.resnets.{j}")))
            .collect::<Result<Vec<_>>>()?;
        let conv = |which: &str| -> Result<(Array, Array)> {
            Ok((
                conv_w(w, &format!("{prefix}.{which}.0.conv.weight"))?,
                f32w(w, &format!("{prefix}.{which}.0.conv.bias"))?,
            ))
        };
        Ok(Self {
            resnets,
            downsample: if down {
                Some(conv("downsamplers")?)
            } else {
                None
            },
            upsample: if up { Some(conv("upsamplers")?) } else { None },
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x)?;
        }
        if let Some((cw, cb)) = &self.downsample {
            // Fork pads (right, bottom) then stride-2, pad-0 conv.
            x = pad(&x, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
            x = conv2d(&x, cw, Some(cb), 2, 0)?;
        }
        if let Some((cw, cb)) = &self.upsample {
            x = conv2d(&upsample_nearest(&x, 2)?, cw, Some(cb), 1, 1)?;
        }
        Ok(x)
    }
}

struct Encoder {
    conv_in_w: Array,
    conv_in_b: Array,
    down_blocks: Vec<SampleBlock>,
    mid_resnet0: ResnetBlock2D,
    mid_attn: VaeAttention,
    mid_resnet1: ResnetBlock2D,
    norm_out_w: Array,
    norm_out_b: Array,
    conv_out_w: Array,
    conv_out_b: Array,
}

impl Encoder {
    fn from_weights(w: &Weights) -> Result<Self> {
        let n = BLOCK_OUT.len();
        let down_blocks = (0..n)
            .map(|i| {
                SampleBlock::from_weights(
                    w,
                    &format!("encoder.down_blocks.{i}"),
                    LAYERS_PER_BLOCK,
                    i < n - 1,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in_w: conv_w(w, "encoder.conv_in.weight")?,
            conv_in_b: f32w(w, "encoder.conv_in.bias")?,
            down_blocks,
            mid_resnet0: ResnetBlock2D::from_weights(w, "encoder.mid_block.resnets.0")?,
            mid_attn: VaeAttention::from_weights(w, "encoder.mid_block.attentions.0")?,
            mid_resnet1: ResnetBlock2D::from_weights(w, "encoder.mid_block.resnets.1")?,
            norm_out_w: f32w(w, "encoder.conv_norm_out.weight")?,
            norm_out_b: f32w(w, "encoder.conv_norm_out.bias")?,
            conv_out_w: conv_w(w, "encoder.conv_out.weight")?,
            conv_out_b: f32w(w, "encoder.conv_out.bias")?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = conv2d(x, &self.conv_in_w, Some(&self.conv_in_b), 1, 1)?;
        for db in &self.down_blocks {
            x = db.forward(&x)?;
        }
        x = self.mid_resnet0.forward(&x)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_resnet1.forward(&x)?;
        let x = group_norm(&x, &self.norm_out_w, &self.norm_out_b, GN_GROUPS, GN_EPS)?;
        conv2d(&silu(&x)?, &self.conv_out_w, Some(&self.conv_out_b), 1, 1)
    }
}

struct Decoder {
    conv_in_w: Array,
    conv_in_b: Array,
    mid_resnet0: ResnetBlock2D,
    mid_attn: VaeAttention,
    mid_resnet1: ResnetBlock2D,
    up_blocks: Vec<SampleBlock>,
    norm_out_w: Array,
    norm_out_b: Array,
    conv_out_w: Array,
    conv_out_b: Array,
}

impl Decoder {
    fn from_weights(w: &Weights) -> Result<Self> {
        let n = BLOCK_OUT.len();
        // decoder resnets = layers_per_block + 1.
        let up_blocks = (0..n)
            .map(|i| {
                SampleBlock::from_weights(
                    w,
                    &format!("decoder.up_blocks.{i}"),
                    LAYERS_PER_BLOCK + 1,
                    false,
                    i < n - 1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in_w: conv_w(w, "decoder.conv_in.weight")?,
            conv_in_b: f32w(w, "decoder.conv_in.bias")?,
            mid_resnet0: ResnetBlock2D::from_weights(w, "decoder.mid_block.resnets.0")?,
            mid_attn: VaeAttention::from_weights(w, "decoder.mid_block.attentions.0")?,
            mid_resnet1: ResnetBlock2D::from_weights(w, "decoder.mid_block.resnets.1")?,
            up_blocks,
            norm_out_w: f32w(w, "decoder.conv_norm_out.weight")?,
            norm_out_b: f32w(w, "decoder.conv_norm_out.bias")?,
            conv_out_w: conv_w(w, "decoder.conv_out.weight")?,
            conv_out_b: f32w(w, "decoder.conv_out.bias")?,
        })
    }

    fn forward(&self, z: &Array) -> Result<Array> {
        let mut x = conv2d(z, &self.conv_in_w, Some(&self.conv_in_b), 1, 1)?;
        x = self.mid_resnet0.forward(&x)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_resnet1.forward(&x)?;
        for ub in &self.up_blocks {
            x = ub.forward(&x)?;
        }
        let x = group_norm(&x, &self.norm_out_w, &self.norm_out_b, GN_GROUPS, GN_EPS)?;
        conv2d(&silu(&x)?, &self.conv_out_w, Some(&self.conv_out_b), 1, 1)
    }
}

/// The FLUX.2 autoencoder. All tensors NHWC, f32.
pub struct Flux2Vae {
    encoder: Encoder,
    decoder: Decoder,
    quant: (Array, Array),
    post_quant: (Array, Array),
    bn_mean: Array,
    bn_std: Array,
}

impl Flux2Vae {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let bn_mean = f32w(w, "bn.running_mean")?;
        let bn_var = f32w(w, "bn.running_var")?;
        let bn_std = sqrt(&add(&bn_var, scalar(BN_EPS))?)?;
        Ok(Self {
            encoder: Encoder::from_weights(w)?,
            decoder: Decoder::from_weights(w)?,
            quant: squeeze_linear(w, "quant_conv")?,
            post_quant: squeeze_linear(w, "post_quant_conv")?,
            bn_mean,
            bn_std,
        })
    }

    /// Quantize the VAE to Q4/Q8 (group_size 64). The fork's `nn.quantize` predicate only hits
    /// `nn.Linear`, which in this VAE is exactly the encoder + decoder mid-block attention
    /// (q/k/v/out). Every Conv2d (incl. `quant_conv`/`post_quant_conv`), GroupNorm, and the
    /// BatchNorm stats are not Linears, so they stay full precision — matching the fork.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.encoder.mid_attn.quantize(bits)?;
        self.decoder.mid_attn.quantize(bits)?;
        Ok(())
    }

    /// Decode latents `[B, h, w, 32]` (NHWC) → image `[B, H, W, 3]` in ~`[-1, 1]`.
    /// `scaling_factor=1.0, shift_factor=0.0`, so the latent passes straight to `post_quant_conv`.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let latents = latents.as_dtype(Dtype::Float32)?;
        let z = linear(&latents, &self.post_quant.0, &self.post_quant.1)?;
        self.decoder.forward(&z)
    }

    /// Test-only (sc-2643 byte-parity gate): the quantized `(wq, scales, biases, group_size, bits)`
    /// of the encoder mid-block attention `to_q` — the unique f32-loaded Linear-with-bias case
    /// (the rest of the VAE is Conv/GroupNorm, never quantized). `None` if the VAE is still dense.
    #[doc(hidden)]
    pub fn probe_quant_enc_q(&self) -> Option<(&Array, &Array, &Array, i32, i32)> {
        let (wq, sc, bi, _bias, gs, b) = self.encoder.mid_attn.q.quantized_params()?;
        Some((wq, sc, bi, gs, b))
    }

    /// Decode the transformer's packed output `[B, lat_h, lat_w, 128]` (NHWC): de-normalize with
    /// the BatchNorm stats, 2×2-unpatchify into `[B, lat_h·2, lat_w·2, 32]`, then `decode`.
    pub fn decode_packed_latents(&self, packed: &Array) -> Result<Array> {
        let packed = packed.as_dtype(Dtype::Float32)?;
        // De-normalize: x·std + mean (bn channel order = the packed 128-ch order).
        let denorm = add(&multiply(&packed, &self.bn_std)?, &self.bn_mean)?;
        let latents = unpatchify(&denorm)?;
        self.decode(&latents)
    }

    /// Encode an image `[B, H, W, 3]` (NHWC, ~`[-1, 1]`) → latent **mean** `[B, H/8, W/8, 32]`.
    /// Mirrors the fork's `encode` (returns the mean; `scaling_factor=1.0, shift_factor=0.0`).
    pub fn encode_mean(&self, x: &Array) -> Result<Array> {
        let x = x.as_dtype(Dtype::Float32)?;
        let moments = linear(&self.encoder.forward(&x)?, &self.quant.0, &self.quant.1)?;
        // split (mean, logvar) along channels; keep the mean.
        let c = moments.shape()[3];
        let half = c / 2;
        let idx = Array::from_slice(&(0..half).collect::<Vec<i32>>(), &[half]);
        Ok(moments.take_axis(&idx, 3)?)
    }

    /// Forward BatchNorm-stats normalization of a **NCHW** patchified `[B, 128, h, w]` latent (the
    /// inverse of `decode_packed_latents`' de-normalize): `(x - mean) / std`, the fork's
    /// `bn_normalize_vae_encoded_latents`. Used by edit / img2img to normalize the reference VAE
    /// latent into the transformer's packed space.
    pub fn bn_normalize_nchw(&self, patchified: &Array) -> Result<Array> {
        let c = self.bn_mean.shape()[0];
        let mean = self.bn_mean.reshape(&[1, c, 1, 1])?;
        let std = self.bn_std.reshape(&[1, c, 1, 1])?;
        let x = patchified.as_dtype(Dtype::Float32)?;
        Ok(mlx_rs::ops::divide(
            &mlx_rs::ops::subtract(&x, &mean)?,
            &std,
        )?)
    }
}

/// 2×2 unpatchify (NHWC): `[B, h, w, 128]` → `[B, h·2, w·2, 32]`. Channel order `c·4 + ph·2 + pw`
/// matches the fork's NCHW `reshape(B, C/4, 2, 2, H, W) → transpose → reshape`.
fn unpatchify(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let (b, h, w_, c) = (sh[0], sh[1], sh[2], sh[3]);
    let c4 = c / 4;
    Ok(x.reshape(&[b, h, w_, c4, 2, 2])?
        .transpose_axes(&[0, 1, 4, 2, 5, 3])?
        .reshape(&[b, h * 2, w_ * 2, c4])?)
}

#[allow(dead_code)]
const _: i32 = LATENT_CHANNELS; // documented; channel counts come from the checkpoint shapes.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpatchify_round_trips_patchify_ordering() {
        // Build [1,2,2,8] where channel = c*4 + ph*2 + pw, c in 0..2.
        let mut data = vec![0f32; 2 * 2 * 8];
        for hi in 0..2 {
            for wi in 0..2 {
                for ch in 0..8 {
                    data[((hi * 2 + wi) * 8) + ch] = (hi * 1000 + wi * 100 + ch) as f32;
                }
            }
        }
        let x = Array::from_slice(&data, &[1, 2, 2, 8]);
        let out = unpatchify(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 4, 2]);
        // out[b, 2*hi+ph, 2*wi+pw, c] == x[b, hi, wi, c*4+ph*2+pw]
        let o = out.as_slice::<f32>();
        let at = |hh: usize, ww: usize, cc: usize| o[((hh * 4 + ww) * 2) + cc];
        for hi in 0..2 {
            for wi in 0..2 {
                for ph in 0..2 {
                    for pw in 0..2 {
                        for c in 0..2 {
                            let got = at(2 * hi + ph, 2 * wi + pw, c);
                            let want = (hi * 1000 + wi * 100 + (c * 4 + ph * 2 + pw)) as f32;
                            assert_eq!(got, want, "mismatch at hi{hi} wi{wi} ph{ph} pw{pw} c{c}");
                        }
                    }
                }
            }
        }
    }
}
