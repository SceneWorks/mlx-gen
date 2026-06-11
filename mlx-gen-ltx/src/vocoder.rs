//! S4 — the LTX-2.3 **vocoder family** (sc-2684): mel/STFT-domain spectrogram → waveform. Port of
//! `models/ltx/audio_vae/vocoder.py`. Three config-selected variants (`load_vocoder`):
//!
//! - [`Generator`] in HiFi-GAN mode (`Vocoder`): `conv_pre` → per-upsample [leaky-ReLU →
//!   ConvTranspose1d → mean of dilated `ResBlock1`/`ResBlock2` outputs] → leaky-ReLU(0.01) →
//!   `conv_post` → tanh.
//! - [`Generator`] in BigVGAN mode (`BigVGANVocoder`): `conv_pre` → per-upsample [ConvTranspose1d →
//!   mean of `AMPBlock1` outputs] → [`SnakeBeta`] `act_post` → `conv_post` → tanh/clip. `AMPBlock1`
//!   uses **anti-aliased SnakeBeta**: 2× upsample (zero-insert conv-transpose + kaiser-sinc filter) →
//!   `x + sin²(αx)/(β+eps)` → kaiser-sinc low-pass + 2× downsample.
//! - [`VocoderWithBWE`] (the shipped 2.3 path → 48 kHz): the BigVGAN core → `_compute_mel` (a
//!   windowed-STFT via the checkpoint's stored `forward_basis`/`mel_basis` matmuls, win 512 / hop 80)
//!   → a BigVGAN bandwidth-extension generator → linear-interp skip-upsample of the core output, sum,
//!   clip.
//!
//! NLC (`B, L, C`) layout; Conv1d weight `[C_out, kernel, C_in]`, ConvTranspose1d `[C_out, kernel,
//! C_in]` (no transpose — the split checkpoint already stores the MLX layout). Runs **f32**.
//!
//! The reference's `_upsample_skip` is itself a linear-interp *approximation* of upstream's Hann-sinc
//! resampler, so parity targets the MLX reference (`generate_av.py`), not PyTorch.

use mlx_rs::ops::{
    add, broadcast_to, clip, concatenate_axis, matmul, maximum, mean_axes, multiply, pad,
    stack_axis, subtract, tanh,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{Error, Result};

use crate::config::{VocoderConfig, VocoderGenConfig};

const LRELU_SLOPE: f32 = 0.1;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

fn f32(w: &Weights, key: &str) -> Result<Array> {
    to_dtype(w.require(key)?, Dtype::Float32)
}

/// A contiguous index range `lo..hi` as an `Array` for `take_axis`.
fn range_idx(lo: i32, hi: i32) -> Array {
    Array::from_slice(&(lo..hi).collect::<Vec<i32>>(), &[(hi - lo).max(0)])
}

/// `max(x, slope·x)` (LeakyReLU).
fn leaky(x: &Array, slope: f32) -> Result<Array> {
    Ok(maximum(x, &multiply(x, scalar(slope))?)?)
}

/// Replicate-pad along axis 2 of `(B, C, L)` (edge values), matching `mx.broadcast_to(x[..., :1/-1:])`.
fn replicate_pad_l(x: &Array, left: i32, right: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, c, l) = (sh[0], sh[1], sh[2]);
    let mut parts: Vec<Array> = Vec::new();
    if left > 0 {
        let first = x.take_axis(range_idx(0, 1), 2)?;
        parts.push(broadcast_to(&first, &[b, c, left])?);
    }
    parts.push(x.clone());
    if right > 0 {
        let last = x.take_axis(range_idx(l - 1, l), 2)?;
        parts.push(broadcast_to(&last, &[b, c, right])?);
    }
    let refs: Vec<&Array> = parts.iter().collect();
    Ok(concatenate_axis(&refs, 2)?)
}

/// 1-D conv (NLC), weight `[C_out, kernel, C_in]`, optional bias.
struct Conv1d {
    w: Array,
    b: Option<Array>,
    stride: i32,
    padding: i32,
    dilation: i32,
}

impl Conv1d {
    fn load(w: &Weights, prefix: &str, stride: i32, padding: i32, dilation: i32) -> Result<Self> {
        let weight = f32(w, &format!("{prefix}.weight"))?;
        let b = match w.get(&format!("{prefix}.bias")) {
            Some(bias) => Some(to_dtype(bias, Dtype::Float32)?),
            None => None,
        };
        Ok(Self {
            w: weight,
            b,
            stride,
            padding,
            dilation,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let y = mlx_rs::ops::conv1d(x, &self.w, self.stride, self.padding, self.dilation, 1)?;
        match &self.b {
            Some(b) => Ok(add(&y, b)?),
            None => Ok(y),
        }
    }
}

/// 1-D transposed conv (NLC), weight `[C_out, kernel, C_in]`, optional bias.
struct ConvT1d {
    w: Array,
    b: Option<Array>,
    stride: i32,
    padding: i32,
}

impl ConvT1d {
    fn load(w: &Weights, prefix: &str, stride: i32, padding: i32) -> Result<Self> {
        let weight = f32(w, &format!("{prefix}.weight"))?;
        let b = match w.get(&format!("{prefix}.bias")) {
            Some(bias) => Some(to_dtype(bias, Dtype::Float32)?),
            None => None,
        };
        Ok(Self {
            w: weight,
            b,
            stride,
            padding,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let y = mlx_rs::ops::conv_transpose1d(x, &self.w, self.stride, self.padding, 1, 0, 1)?;
        match &self.b {
            Some(b) => Ok(add(&y, b)?),
            None => Ok(y),
        }
    }
}

/// `x + sin²(exp(α)·x) / (exp(β) + 1e-6)` (`_SnakeCore`; log-scale α/β over channels).
struct SnakeCore {
    alpha: Array, // (C,)
    beta: Array,
}

impl SnakeCore {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            alpha: f32(w, &format!("{prefix}.alpha"))?,
            beta: f32(w, &format!("{prefix}.beta"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let c = *x.shape().last().unwrap();
        let alpha = self.alpha.exp()?.reshape(&[1, 1, c])?;
        let beta = self.beta.exp()?.reshape(&[1, 1, c])?;
        let s = multiply(&alpha, x)?.sin()?;
        // `sin(αx) ** 2` — the reference's `** 2` is `mx.power(·, 2)`, which is NOT bit-identical to
        // `s·s` (the SDXL `σ_up**2` lesson); the gap, fed through the SnakeBeta of every AMPBlock,
        // compounds over the 18-block BigVGAN stack to ~1% in the waveform.
        let num = mlx_rs::ops::power(&s, scalar(2.0))?;
        let den = add(&beta, scalar(1e-6))?;
        Ok(add(x, &mlx_rs::ops::divide(&num, &den)?)?)
    }
}

/// Kaiser-sinc depth-wise filter (`_SnakeFilter`) — a stored `(1,1,taps)` kernel applied per channel.
struct SnakeFilter {
    filter: Array, // (1, 1, taps)
}

impl SnakeFilter {
    fn load(w: &Weights, key: &str) -> Result<Self> {
        Ok(Self {
            filter: f32(w, key)?,
        })
    }

    /// Depth-wise 1-D conv along the time axis. `x` `(B, L, C)` → `(B, L_out, C)`.
    fn apply_filter(&self, x: &Array, stride: i32) -> Result<Array> {
        let taps = *self.filter.shape().last().unwrap();
        let even = taps % 2 == 0;
        let pad_left = taps / 2 - i32::from(even);
        let pad_right = taps / 2;
        let sh = x.shape();
        let (b, _l, c) = (sh[0], sh[1], sh[2]);
        let x_bct = x.transpose_axes(&[0, 2, 1])?; // (B, C, L)
        let x_padded = replicate_pad_l(&x_bct, pad_left, pad_right)?; // (B, C, L+pad)
        let total = x_padded.shape()[2];
        let x_flat = x_padded
            .reshape(&[b * c, 1, total])?
            .transpose_axes(&[0, 2, 1])?; // (B*C, L+pad, 1)
        let kw = self.filter.reshape(&[1, taps, 1])?;
        let out = mlx_rs::ops::conv1d(&x_flat, &kw, stride, 0, 1, 1)?; // (B*C, T_out, 1)
        let t_out = out.shape()[1];
        Ok(out.reshape(&[b, c, t_out])?.transpose_axes(&[0, 2, 1])?)
    }
}

/// 2× upsample via zero-insert conv-transpose with the kaiser-sinc filter (`_SnakeUpsample`).
struct SnakeUpsample {
    filter: Array, // (1, 1, taps)
}

impl SnakeUpsample {
    fn load(w: &Weights, key: &str) -> Result<Self> {
        Ok(Self {
            filter: f32(w, key)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let taps = *self.filter.shape().last().unwrap();
        let ratio = 2;
        let pad = taps / ratio - 1;
        let pad_left = pad * ratio + (taps - ratio) / 2;
        let pad_right = pad * ratio + (taps - ratio + 1) / 2;
        let sh = x.shape();
        let (b, _l, c) = (sh[0], sh[1], sh[2]);
        let x_bct = x.transpose_axes(&[0, 2, 1])?; // (B, C, L)
        let x_padded = replicate_pad_l(&x_bct, pad, pad)?; // (B, C, L+2pad)
        let total = x_padded.shape()[2];
        let x_flat = x_padded
            .reshape(&[b * c, 1, total])?
            .transpose_axes(&[0, 2, 1])?; // (B*C, L+2pad, 1)
        let kw = self.filter.reshape(&[1, taps, 1])?;
        let out = mlx_rs::ops::conv_transpose1d(&x_flat, &kw, ratio, 0, 1, 0, 1)?; // (B*C, T_up, 1)
        let out = multiply(&out, scalar(ratio as f32))?;
        let t_up = out.shape()[1];
        let mut out = out.reshape(&[b, c, t_up])?; // (B, C, T_up)
                                                   // out[:, :, pad_left:] then [:, :, :-pad_right].
        out = out.take_axis(range_idx(pad_left, t_up), 2)?;
        if pad_right > 0 {
            let cur = out.shape()[2];
            out = out.take_axis(range_idx(0, cur - pad_right), 2)?;
        }
        Ok(out.transpose_axes(&[0, 2, 1])?)
    }
}

/// BigVGAN anti-aliased SnakeBeta activation: 2× upsample → SnakeBeta → low-pass + 2× downsample.
struct SnakeBeta {
    act: SnakeCore,
    up: SnakeUpsample,
    down: SnakeFilter,
}

impl SnakeBeta {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            act: SnakeCore::load(w, &format!("{prefix}.act"))?,
            up: SnakeUpsample::load(w, &format!("{prefix}.upsample.filter"))?,
            down: SnakeFilter::load(w, &format!("{prefix}.downsample.lowpass.filter"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.up.forward(x)?;
        let x = self.act.forward(&x)?;
        self.down.apply_filter(&x, 2)
    }
}

/// A vocoder residual block (HiFi-GAN `ResBlock1`/`ResBlock2` or BigVGAN `AMPBlock1`). The variants
/// differ in field count (AMPBlock1 carries the SnakeBeta activations); only ~18 are ever allocated.
#[allow(clippy::large_enum_variant)]
enum ResBlock {
    /// AMPBlock1: `acts1[i] → convs1[i] → acts2[i] → convs2[i]`, residual.
    Amp1 {
        convs1: Vec<Conv1d>,
        convs2: Vec<Conv1d>,
        acts1: Vec<SnakeBeta>,
        acts2: Vec<SnakeBeta>,
    },
    /// ResBlock1: `leaky → convs1[i] → leaky → convs2[i]`, residual.
    Hifi1 {
        convs1: Vec<Conv1d>,
        convs2: Vec<Conv1d>,
    },
    /// ResBlock2: `leaky → convs[i]`, residual.
    Hifi2 { convs: Vec<Conv1d> },
}

impl ResBlock {
    fn load(w: &Weights, prefix: &str, kind: &str, kernel: i32, dilations: &[i32]) -> Result<Self> {
        let pad = |d: i32| (kernel - 1) * d / 2;
        match kind {
            "amp1" => {
                let mut convs1 = Vec::new();
                let mut convs2 = Vec::new();
                let mut acts1 = Vec::new();
                let mut acts2 = Vec::new();
                for (i, &d) in dilations.iter().enumerate() {
                    convs1.push(Conv1d::load(
                        w,
                        &format!("{prefix}.convs1.{i}"),
                        1,
                        pad(d),
                        d,
                    )?);
                    convs2.push(Conv1d::load(
                        w,
                        &format!("{prefix}.convs2.{i}"),
                        1,
                        pad(1),
                        1,
                    )?);
                    acts1.push(SnakeBeta::load(w, &format!("{prefix}.acts1.{i}"))?);
                    acts2.push(SnakeBeta::load(w, &format!("{prefix}.acts2.{i}"))?);
                }
                Ok(ResBlock::Amp1 {
                    convs1,
                    convs2,
                    acts1,
                    acts2,
                })
            }
            "2" => {
                let mut convs = Vec::new();
                for (i, &d) in dilations.iter().enumerate() {
                    convs.push(Conv1d::load(
                        w,
                        &format!("{prefix}.convs.{i}"),
                        1,
                        pad(d),
                        d,
                    )?);
                }
                Ok(ResBlock::Hifi2 { convs })
            }
            _ => {
                let mut convs1 = Vec::new();
                let mut convs2 = Vec::new();
                for (i, &d) in dilations.iter().enumerate() {
                    convs1.push(Conv1d::load(
                        w,
                        &format!("{prefix}.convs1.{i}"),
                        1,
                        pad(d),
                        d,
                    )?);
                    convs2.push(Conv1d::load(
                        w,
                        &format!("{prefix}.convs2.{i}"),
                        1,
                        pad(1),
                        1,
                    )?);
                }
                Ok(ResBlock::Hifi1 { convs1, convs2 })
            }
        }
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            ResBlock::Amp1 {
                convs1,
                convs2,
                acts1,
                acts2,
            } => {
                let mut x = x.clone();
                for i in 0..convs1.len() {
                    let xt = acts1[i].forward(&x)?;
                    let xt = convs1[i].forward(&xt)?;
                    let xt = acts2[i].forward(&xt)?;
                    let xt = convs2[i].forward(&xt)?;
                    x = add(&xt, &x)?;
                }
                Ok(x)
            }
            ResBlock::Hifi1 { convs1, convs2 } => {
                let mut x = x.clone();
                for i in 0..convs1.len() {
                    let xt = convs1[i].forward(&leaky(&x, LRELU_SLOPE)?)?;
                    let xt = convs2[i].forward(&leaky(&xt, LRELU_SLOPE)?)?;
                    x = add(&xt, &x)?;
                }
                Ok(x)
            }
            ResBlock::Hifi2 { convs } => {
                let mut x = x.clone();
                for c in convs {
                    let xt = c.forward(&leaky(&x, LRELU_SLOPE)?)?;
                    x = add(&xt, &x)?;
                }
                Ok(x)
            }
        }
    }
}

/// A HiFi-GAN / BigVGAN generator (`Vocoder` / `BigVGANVocoder`).
pub struct Generator {
    conv_pre: Conv1d,
    ups: Vec<ConvT1d>,
    resblocks: Vec<ResBlock>,
    act_post: Option<SnakeBeta>, // BigVGAN only
    conv_post: Conv1d,
    num_kernels: usize,
    bigvgan: bool,
    use_tanh_at_final: bool,
    apply_final_activation: bool,
}

impl Generator {
    /// Build from a `Weights` map under `prefix` (`""` for a top-level core vocoder,
    /// `"bwe_generator."` for the BWE stage) + its [`VocoderGenConfig`].
    pub fn load(w: &Weights, prefix: &str, cfg: &VocoderGenConfig) -> Result<Self> {
        let bigvgan = cfg.is_bigvgan();
        let kind = if bigvgan {
            "amp1"
        } else if cfg.resblock == "2" {
            "2"
        } else {
            "1"
        };
        let num_upsamples = cfg.upsample_rates.len();
        let num_kernels = cfg.resblock_kernel_sizes.len();

        let conv_pre = {
            let weight = f32(w, &format!("{prefix}conv_pre.weight"))?;
            let k = weight.shape()[1];
            Conv1d::load(w, &format!("{prefix}conv_pre"), 1, k / 2, 1)?
        };
        let mut ups = Vec::with_capacity(num_upsamples);
        for (i, (&stride, &k)) in cfg
            .upsample_rates
            .iter()
            .zip(cfg.upsample_kernel_sizes.iter())
            .enumerate()
        {
            ups.push(ConvT1d::load(
                w,
                &format!("{prefix}ups.{i}"),
                stride,
                (k - stride) / 2,
            )?);
        }
        let mut resblocks = Vec::with_capacity(num_upsamples * num_kernels);
        let mut idx = 0;
        for _ in 0..num_upsamples {
            for (&k, dil) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(cfg.resblock_dilation_sizes.iter())
            {
                resblocks.push(ResBlock::load(
                    w,
                    &format!("{prefix}resblocks.{idx}"),
                    kind,
                    k,
                    dil,
                )?);
                idx += 1;
            }
        }
        let act_post = if bigvgan {
            Some(SnakeBeta::load(w, &format!("{prefix}act_post"))?)
        } else {
            None
        };
        let conv_post = {
            let weight = f32(w, &format!("{prefix}conv_post.weight"))?;
            let k = weight.shape()[1];
            Conv1d::load(w, &format!("{prefix}conv_post"), 1, k / 2, 1)?
        };
        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            act_post,
            conv_post,
            num_kernels,
            bigvgan,
            use_tanh_at_final: cfg.use_tanh_at_final,
            apply_final_activation: cfg.apply_final_activation,
        })
    }

    /// Diagnostic: run only `ups[i]` on a (NLC) input (isolates ConvTranspose1d).
    #[doc(hidden)]
    pub fn debug_up(&self, i: usize, x: &Array) -> Result<Array> {
        self.ups[i].forward(x)
    }

    /// Diagnostic: run only `act_post` (the BigVGAN SnakeBeta) on a (NLC) input.
    #[doc(hidden)]
    pub fn debug_act_post(&self, x: &Array) -> Result<Array> {
        self.act_post.as_ref().unwrap().forward(x)
    }

    /// Diagnostic: run only `resblocks[idx]` on a (NLC) input.
    #[doc(hidden)]
    pub fn debug_resblock(&self, idx: usize, x: &Array) -> Result<Array> {
        self.resblocks[idx].forward(x)
    }

    /// `(B, C, T, F)` mel/feature input → NLC `(B, T, C·F)` → `conv_pre`. The 4-axis transpose
    /// requires 4-D input, so a non-4-D tensor errors here rather than silently passing through (the
    /// old `if sh.len() == 4 { … } else { x }` fallback was dead — F-056). Shared by `forward` and
    /// `forward_stages`.
    fn pre(&self, x: &Array) -> Result<Array> {
        let x = x.transpose_axes(&[0, 1, 3, 2])?; // (B, C, F, T)
        let sh = x.shape();
        let (b, s, c, t) = (sh[0], sh[1], sh[2], sh[3]);
        let x = x.reshape(&[b, s * c, t])?.transpose_axes(&[0, 2, 1])?; // (B, T, C·F)
        self.conv_pre.forward(&x)
    }

    /// The upsampling loop, shared by `forward` and `forward_stages` so the diagnostic can't drift
    /// from production (F-056): per stage, optional leaky pre-act → transposed-conv upsample → mean
    /// of the `num_kernels` AMP/res-block outputs.
    fn up_loop(&self, mut x: Array) -> Result<Array> {
        for i in 0..self.ups.len() {
            if !self.bigvgan {
                x = leaky(&x, LRELU_SLOPE)?;
            }
            x = self.ups[i].forward(&x)?;
            let start = i * self.num_kernels;
            let outs: Vec<Array> = (start..start + self.num_kernels)
                .map(|idx| self.resblocks[idx].forward(&x))
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<&Array> = outs.iter().collect();
            x = mean_axes(&stack_axis(&refs, 0)?, &[0], false)?;
        }
        Ok(x)
    }

    /// Diagnostic: NLC intermediates `(after_conv_pre, after_up_loop, after_act_post)` for bisection.
    #[doc(hidden)]
    pub fn forward_stages(&self, x: &Array) -> Result<(Array, Array, Array)> {
        let x = self.pre(x)?;
        let after_conv_pre = x.clone();
        let x = self.up_loop(x)?;
        let after_up_loop = x.clone();
        let after_act_post = if self.bigvgan {
            self.act_post.as_ref().unwrap().forward(&x)?
        } else {
            leaky(&x, 0.01)?
        };
        Ok((after_conv_pre, after_up_loop, after_act_post))
    }

    /// `x` is the mel/feature input `(B, C, T, F)` (stereo `C=2`). Returns the waveform `(B, C_out, T)`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.pre(x)?;
        let mut x = self.up_loop(x)?;

        if self.bigvgan {
            x = self.act_post.as_ref().unwrap().forward(&x)?;
            x = self.conv_post.forward(&x)?;
            if self.apply_final_activation {
                x = if self.use_tanh_at_final {
                    tanh(&x)?
                } else {
                    clip(&x, (scalar(-1.0), scalar(1.0)))?
                };
            }
        } else {
            // HiFi-GAN: nn.leaky_relu default slope 0.01, then conv_post + tanh.
            x = leaky(&x, 0.01)?;
            x = self.conv_post.forward(&x)?;
            x = tanh(&x)?;
        }
        Ok(x.transpose_axes(&[0, 2, 1])?) // (B, C_out, T)
    }
}

/// Stored windowed-STFT + mel basis (`_MelSTFT` / `_STFTBasis`) for the BWE mel computation.
struct MelStft {
    forward_basis: Array, // (2·n_freq, 1, win)
    mel_basis: Array,     // (n_mels, n_freq)
}

/// Log-mel of a (left-padded) flattened waveform `(BC, T)` from the stored STFT/mel bases.
/// `forward_basis` is the `(2·n_freq, 1, win)` STFT kernel (stacked real+imag rows), `mel_basis` the
/// `(n_mels, n_freq)` filterbank. F-051: the per-frame slice+matmul+`stack_axis` loop is replaced by
/// one gather of all `win`-length windows `(BC, n_frames, win)` + a single batched matmul against the
/// basis, then a batched magnitude → mel → log. This keeps the same `matmul` reduction the reference
/// uses (so it is bit-identical to the former loop — *not* a `conv1d`, whose Metal kernel differs
/// from matmul by ~0.1%), while collapsing ~`n_frames` graph iterations into a handful of ops.
/// Returns `(BC, n_frames, n_mels)`.
fn stft_log_mel(
    x: &Array,
    forward_basis: &Array,
    mel_basis: &Array,
    hop: i32,
    win: i32,
) -> Result<Array> {
    let (bc, total) = (x.shape()[0], x.shape()[1]);
    let n_freq2 = forward_basis.shape()[0];
    let n_freq = n_freq2 / 2;
    let n_frames = ((total - win) / hop + 1).max(1);

    // Gather every sliding window at once: `idx[i, k] = i·hop + k`, so `x[:, idx]` is
    // `(BC, n_frames, win)` — the stacked frame segments without a per-frame slice.
    let mut idx = Vec::with_capacity((n_frames * win) as usize);
    for i in 0..n_frames {
        for k in 0..win {
            idx.push(i * hop + k);
        }
    }
    let windows = x.take_axis(Array::from_slice(&idx, &[n_frames, win]), 1)?; // (BC, n_frames, win)

    // STFT: `windows @ basisᵀ`, one batched matmul (frames folded into the batch). Same per-window
    // length-`win` reduction as the old loop's `matmul(basis, segᵀ)`.
    let basis_t = forward_basis
        .take_axis(Array::from_int(0), 1)? // (2·n_freq, win)
        .transpose_axes(&[1, 0])?; // (win, 2·n_freq)
    let spec = matmul(&windows.reshape(&[bc * n_frames, win])?, &basis_t)?
        .reshape(&[bc, n_frames, n_freq2])?;

    let real = spec.take_axis(range_idx(0, n_freq), 2)?;
    let imag = spec.take_axis(range_idx(n_freq, n_freq2), 2)?;
    let magnitude = add(&multiply(&real, &real)?, &multiply(&imag, &imag)?)?.sqrt()?; // (BC, n_frames, n_freq)

    let mel_basis_t = mel_basis.transpose_axes(&[1, 0])?; // (n_freq, n_mels)
    let n_mels = mel_basis_t.shape()[1];
    let mel = matmul(&magnitude.reshape(&[bc * n_frames, n_freq])?, &mel_basis_t)?
        .reshape(&[bc, n_frames, n_mels])?;
    Ok(maximum(&mel, scalar(1e-5))?.log()?)
}

/// BigVGAN core + bandwidth-extension (`VocoderWithBWE`). The shipped 2.3 vocoder (48 kHz).
pub struct VocoderWithBwe {
    vocoder: Generator,
    bwe_generator: Generator,
    mel_stft: MelStft,
    input_sr: i32,
    output_sr: i32,
    hop: i32,
    win: i32,
}

impl VocoderWithBwe {
    fn load(w: &Weights, cfg: &VocoderConfig) -> Result<Self> {
        let bwe_cfg = cfg.bwe.as_ref().ok_or_else(|| {
            Error::Msg("ltx vocoder: VocoderWithBwe requires a `bwe` config".into())
        })?;
        Ok(Self {
            vocoder: Generator::load(w, "", &cfg.core)?,
            bwe_generator: Generator::load(w, "bwe_generator.", bwe_cfg)?,
            mel_stft: MelStft {
                forward_basis: f32(w, "mel_stft.stft_fn.forward_basis")?,
                mel_basis: f32(w, "mel_stft.mel_basis")?,
            },
            input_sr: cfg.bwe_input_sample_rate,
            output_sr: cfg.bwe_output_sample_rate,
            hop: cfg.bwe_hop_length,
            win: cfg.bwe_win_length,
        })
    }

    /// Log-mel from a waveform `(B, C, T)` via the stored STFT/mel bases → `(B, C, n_mels, T_frames)`.
    fn compute_mel(&self, audio: &Array) -> Result<Array> {
        let sh = audio.shape();
        let (b, c, t) = (sh[0], sh[1], sh[2]);
        let mut x = audio.reshape(&[b * c, t])?;
        let left_pad = (self.win - self.hop).max(0);
        x = pad(&x, &[(0, 0), (left_pad, 0)][..], None, None)?;
        if x.shape()[1] < self.win {
            x = pad(&x, &[(0, 0), (0, self.win - x.shape()[1])][..], None, None)?;
        }

        // Windowed STFT → magnitude → mel → log, as one strided conv1d + batched matmul (F-051).
        let mel_bt = stft_log_mel(
            &x,
            &self.mel_stft.forward_basis,
            &self.mel_stft.mel_basis,
            self.hop,
            self.win,
        )?; // (B*C, T_frames, n_mels)
        let (n_frames, n_mels) = (mel_bt.shape()[1], mel_bt.shape()[2]);
        let mel = mel_bt.reshape(&[b, c, n_frames, n_mels])?;
        Ok(mel.transpose_axes(&[0, 1, 3, 2])?) // (B, C, n_mels, T_frames)
    }

    /// Linear-interp upsample of the skip connection to the BWE rate (`_upsample_skip`).
    fn upsample_skip(&self, x: &Array) -> Result<Array> {
        let ratio = (self.output_sr / self.input_sr).max(1);
        if ratio <= 1 {
            return Ok(x.clone());
        }
        let x_btc = x.transpose_axes(&[0, 2, 1])?; // (B, T, C)
        let sh = x_btc.shape();
        let (_b, t, _c) = (sh[0], sh[1], sh[2]);
        let t_out = t * ratio;
        // idx = arange(t_out)/ratio; floor/ceil clamp; lerp.
        let idx = mlx_rs::ops::divide(
            &Array::arange::<_, f32>(None, t_out, None)?,
            scalar(ratio as f32),
        )?;
        let idx_floor = clip(
            &idx.as_dtype(Dtype::Int32)?,
            (
                scalar(0.0).as_dtype(Dtype::Int32)?,
                scalar((t - 1) as f32).as_dtype(Dtype::Int32)?,
            ),
        )?;
        let idx_ceil = clip(
            &add(&idx_floor, scalar(1.0).as_dtype(Dtype::Int32)?)?,
            (
                scalar(0.0).as_dtype(Dtype::Int32)?,
                scalar((t - 1) as f32).as_dtype(Dtype::Int32)?,
            ),
        )?;
        let frac = subtract(&idx, &idx_floor.as_dtype(Dtype::Float32)?)?.reshape(&[1, t_out, 1])?;
        let lo = x_btc.take_axis(idx_floor, 1)?; // (B, T_out, C)
        let hi = x_btc.take_axis(idx_ceil, 1)?;
        let out = add(&lo, &multiply(&frac, &subtract(&hi, &lo)?)?)?;
        Ok(out.transpose_axes(&[0, 2, 1])?) // (B, C, T_out)
    }

    /// Diagnostic: the core BigVGAN generator's stage taps (delegates to [`Generator::forward_stages`]).
    #[doc(hidden)]
    pub fn core_forward_stages(&self, mel_spec: &Array) -> Result<(Array, Array, Array)> {
        self.vocoder.forward_stages(mel_spec)
    }

    /// Diagnostic: core `ups[i]` on a NLC input.
    #[doc(hidden)]
    pub fn core_debug_up(&self, i: usize, x: &Array) -> Result<Array> {
        self.vocoder.debug_up(i, x)
    }

    /// Diagnostic: core `act_post` (SnakeBeta) on a NLC input.
    #[doc(hidden)]
    pub fn core_debug_act_post(&self, x: &Array) -> Result<Array> {
        self.vocoder.debug_act_post(x)
    }

    /// Diagnostic: core `resblocks[idx]` (AMPBlock1) on a NLC input.
    #[doc(hidden)]
    pub fn core_debug_resblock(&self, idx: usize, x: &Array) -> Result<Array> {
        self.vocoder.debug_resblock(idx, x)
    }

    /// Diagnostic: the four stage taps `(low, mel_from_low, residual, skip)` for parity bisection.
    #[doc(hidden)]
    pub fn stages(&self, mel_spec: &Array) -> Result<(Array, Array, Array, Array)> {
        let low = self.vocoder.forward(mel_spec)?;
        let mel_from_low = self.compute_mel(&low)?;
        let mel_for_bwe = mel_from_low.transpose_axes(&[0, 1, 3, 2])?;
        let residual = self.bwe_generator.forward(&mel_for_bwe)?;
        let skip = self.upsample_skip(&low)?;
        Ok((low, mel_from_low, residual, skip))
    }

    /// `(B, C, T_low)` low → mel → BWE residual + linear-interp skip, summed and clipped.
    pub fn forward(&self, mel_spec: &Array) -> Result<Array> {
        let low = self.vocoder.forward(mel_spec)?; // (B, C, T_low)
        let mel_from_low = self.compute_mel(&low)?; // (B, C, n_mels, T_frames)
        let mel_for_bwe = mel_from_low.transpose_axes(&[0, 1, 3, 2])?; // (B, C, T, n_mels)
        let residual = self.bwe_generator.forward(&mel_for_bwe)?; // (B, C, T_high)
        let skip = self.upsample_skip(&low)?;
        let target = residual.shape()[2].min(skip.shape()[2]);
        let residual = residual.take_axis(range_idx(0, target), 2)?;
        let skip = skip.take_axis(range_idx(0, target), 2)?;
        clip(&add(&residual, &skip)?, (scalar(-1.0), scalar(1.0))).map_err(Error::from)
    }
}

/// The selected LTX vocoder (`load_vocoder`): HiFi-GAN / BigVGAN core, or the core + BWE wrapper.
pub enum LtxVocoder {
    /// `Vocoder` (HiFi-GAN) or `BigVGANVocoder` standalone.
    Plain(Generator),
    /// `VocoderWithBWE` (BigVGAN + bandwidth extension) — the shipped 2.3 path. Boxed (much larger
    /// than the `Plain` variant: it carries two generators + the STFT/mel bases).
    Bwe(Box<VocoderWithBwe>),
}

impl LtxVocoder {
    /// Build from `vocoder.safetensors` + the [`VocoderConfig`] (variant selected like `load_vocoder`).
    pub fn from_weights(w: &Weights, cfg: &VocoderConfig) -> Result<Self> {
        if cfg.bwe.is_some() {
            Ok(LtxVocoder::Bwe(Box::new(VocoderWithBwe::load(w, cfg)?)))
        } else {
            Ok(LtxVocoder::Plain(Generator::load(w, "", &cfg.core)?))
        }
    }

    /// Mel `(B, C, T, F)` → waveform `(B, C_out, T)`.
    pub fn forward(&self, mel_spec: &Array) -> Result<Array> {
        match self {
            LtxVocoder::Plain(g) => g.forward(mel_spec),
            LtxVocoder::Bwe(v) => v.forward(mel_spec),
        }
    }
}

#[cfg(test)]
mod stft_log_mel_tests {
    use super::*;
    use mlx_rs::random;
    use mlx_rs::transforms::eval;

    /// The vectorized `stft_log_mel` (gather + batched matmul) must reproduce the former per-frame
    /// slice+matmul+stack loop **bit-for-bit** on synthetic data — same `matmul` reductions, just
    /// batched (F-051). (A `conv1d` would not: its Metal kernel differs from matmul by ~0.1%.)
    #[test]
    fn vectorized_stft_log_mel_matches_per_frame_loop() {
        let (bc, total, win, hop, n_freq, n_mels) = (2i32, 40i32, 8i32, 4i32, 3i32, 4i32);
        let key = |s| random::key(s).unwrap();
        let x = random::normal::<f32>(&[bc, total], None, None, Some(&key(0))).unwrap();
        let fb = random::normal::<f32>(&[2 * n_freq, 1, win], None, None, Some(&key(1))).unwrap();
        let mb = random::normal::<f32>(&[n_mels, n_freq], None, None, Some(&key(2)))
            .unwrap()
            .abs()
            .unwrap(); // mel filterbank is non-negative
        eval([&x, &fb, &mb]).unwrap();

        let got = stft_log_mel(&x, &fb, &mb, hop, win).unwrap(); // (bc, n_frames, n_mels)

        // Reference: the prior per-frame loop.
        let basis = fb.take_axis(Array::from_int(0), 1).unwrap(); // (2·n_freq, win)
        let mbt = mb.transpose_axes(&[1, 0]).unwrap();
        let n_frames = (total - win) / hop + 1;
        let mut frames = Vec::new();
        for i in 0..n_frames {
            let start = i * hop;
            let seg = x.take_axis(range_idx(start, start + win), 1).unwrap();
            let spec = matmul(&basis, seg.transpose_axes(&[1, 0]).unwrap())
                .unwrap()
                .transpose_axes(&[1, 0])
                .unwrap();
            let real = spec.take_axis(range_idx(0, n_freq), 1).unwrap();
            let imag = spec.take_axis(range_idx(n_freq, 2 * n_freq), 1).unwrap();
            let magnitude = add(
                multiply(&real, &real).unwrap(),
                multiply(&imag, &imag).unwrap(),
            )
            .unwrap()
            .sqrt()
            .unwrap();
            let mel = matmul(&magnitude, &mbt).unwrap();
            frames.push(maximum(&mel, scalar(1e-5)).unwrap().log().unwrap());
        }
        let refs: Vec<&Array> = frames.iter().collect();
        let want = stack_axis(&refs, 1).unwrap(); // (bc, n_frames, n_mels)

        assert_eq!(got.shape(), want.shape());
        let d = subtract(&got, &want).unwrap();
        eval([&d]).unwrap();
        let max_abs = d
            .as_slice::<f32>()
            .iter()
            .fold(0f32, |m, &v| m.max(v.abs()));
        assert_eq!(
            max_abs, 0.0,
            "vectorized vs per-frame loop max abs diff = {max_abs}"
        );
    }
}
