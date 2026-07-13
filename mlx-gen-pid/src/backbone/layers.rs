//! PixDiT leaf modules + host helpers: RMSNorm, SwiGLU `FeedForward`, GELU `MLP`,
//! `TimestepConditioner`, the patch/pixel token embedders, `FinalLayer`, the 2-D sin/cos pixel
//! position table, and the unfold/fold patchify pair. Faithful port of the `modules.py` /
//! `pixeldit_c2i.py` blocks merged into `pixeldit_official.py`.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{concatenate_axis, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::{gelu_exact, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::memo::{memo, TableCache};

/// All PixDiT RMSNorms use eps = 1e-6.
pub const RMS_EPS: f32 = 1e-6;

/// Load an `[out, in]` Linear as a dense [`AdaptableLinear`], auto-detecting the optional bias
/// (`{prefix}.bias`). Quant/LoRA hang off the `AdaptableLinear` later (no separate code path).
pub fn lin(w: &Weights, prefix: &str) -> Result<AdaptableLinear> {
    let weight = w.require(&format!("{prefix}.weight"))?.clone();
    let bias = w.get(&format!("{prefix}.bias")).cloned();
    Ok(AdaptableLinear::dense(weight, bias))
}

/// `x · rsqrt(mean(x²) + eps) · weight` over the last axis. The reference PixDiT `RMSNorm` computes
/// the normalization in **fp32** (`hidden.to(float32)`) then casts back — load-bearing on the real
/// bf16 decode (a bf16-internal reduction drifts over the stack's ~60 norms), and a no-op on the f32
/// fixtures. We upcast x + weight to f32, normalize, and cast back to the input dtype.
pub fn rms(x: &Array, w: &Array) -> Result<Array> {
    let xf = x.as_dtype(Dtype::Float32)?;
    let wf = w.as_dtype(Dtype::Float32)?;
    Ok(rms_norm(&xf, &wf, RMS_EPS)?.as_dtype(x.dtype())?)
}

/// SwiGLU `FeedForward`: `w2(silu(w1(x)) · w3(x))`. Inner width = `int(2·(dim·mlp_ratio)/3)` is
/// baked into the loaded weight shapes; all three projections are bias-less.
pub struct FeedForward {
    w1: AdaptableLinear,
    w2: AdaptableLinear,
    w3: AdaptableLinear,
}

impl FeedForward {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: lin(w, &format!("{prefix}.w1"))?,
            w2: lin(w, &format!("{prefix}.w2"))?,
            w3: lin(w, &format!("{prefix}.w3"))?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gate = silu(&self.w1.forward(x)?)?;
        self.w2.forward(&multiply(&gate, &self.w3.forward(x)?)?)
    }
}

/// `MLP`: `fc2(gelu(fc1(x)))` with exact (erf) GELU and biased projections (pixel-stream FFN).
pub struct Mlp {
    fc1: AdaptableLinear,
    fc2: AdaptableLinear,
}

impl Mlp {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: lin(w, &format!("{prefix}.fc1"))?,
            fc2: lin(w, &format!("{prefix}.fc2"))?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.fc2.forward(&gelu_exact(&self.fc1.forward(x)?)?)
    }
}

/// `TimestepConditioner`: sinusoidal embedding (size 256, **max_period = 10**) → Linear → SiLU →
/// Linear. The `max_period=10` is a PixDiT-specific value (not the usual 10000).
pub struct TimestepConditioner {
    mlp0: AdaptableLinear,
    mlp2: AdaptableLinear,
    freq_size: i32,
}

impl TimestepConditioner {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            mlp0: lin(w, &format!("{prefix}.mlp.0"))?,
            mlp2: lin(w, &format!("{prefix}.mlp.2"))?,
            freq_size: 256,
        })
    }

    /// `t`: `[N]` → `[N, hidden]`.
    pub fn forward(&self, t: &Array) -> Result<Array> {
        let emb = timestep_embedding(t, self.freq_size, 10.0)?;
        let h = silu(&self.mlp0.forward(&emb)?)?;
        self.mlp2.forward(&h)
    }
}

/// `concat([cos(t·freqs), sin(t·freqs)])`, `freqs[i] = exp(-ln(max_period)·i/half)`, `half = dim/2`.
/// `t`: `[N]` → `[N, dim]` (dim even, so no zero-pad branch). cos-then-sin, matching the reference.
pub fn timestep_embedding(t: &Array, dim: i32, max_period: f32) -> Result<Array> {
    let half = (dim / 2) as usize;
    let lp = (max_period as f64).ln();
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-lp * i as f64 / half as f64).exp() as f32)
        .collect();
    let freqs = Array::from_slice(&freqs, &[1, half as i32]);
    let n = t.shape()[0];
    let t = t.reshape(&[n, 1])?;
    let args = multiply(&t, &freqs)?;
    Ok(concatenate_axis(&[&args.cos()?, &args.sin()?], 1)?)
}

/// `PatchTokenEmbedder`: a Linear `proj` with an optional trailing RMSNorm. `s_embedder` has no
/// norm; `y_embedder` carries `norm.weight`.
pub struct PatchTokenEmbedder {
    proj: AdaptableLinear,
    norm: Option<Array>,
}

impl PatchTokenEmbedder {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            proj: lin(w, &format!("{prefix}.proj"))?,
            norm: w.get(&format!("{prefix}.norm.weight")).cloned(),
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.proj.forward(x)?;
        match &self.norm {
            Some(nw) => rms(&x, nw),
            None => Ok(x),
        }
    }
}

/// `FinalLayer`: RMSNorm then a biased Linear to the output channels.
pub struct FinalLayer {
    norm: Array,
    linear: AdaptableLinear,
}

impl FinalLayer {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: w.require(&format!("{prefix}.norm.weight"))?.clone(),
            linear: lin(w, &format!("{prefix}.linear"))?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.linear.forward(&rms(x, &self.norm)?)
    }
}

/// Reference per-axis pixel extent the 2kto4k SR students were trained within: the top of the
/// `2048→3840` multi-resolution training bucket (`experiment_2kto4k/shared_config.py`). The additive
/// pixel positional signal ([`sincos_2d_pos`]) uses **absolute** output-pixel coordinates, so a decode
/// whose long side exceeds this extrapolates it out of distribution — a 4× SR of a tall image pushes
/// the *height* coordinate past it (e.g. a 1280px-tall gen → 5120px decode > 3840) and the light
/// 2-block pixel refiner biases toward a color cast in exactly those out-of-range (bottom) rows. See
/// [`pixel_pos_axis_scale`].
const PIXEL_POS_TRAIN_MAX: f64 = 3840.0;

/// **Per-axis** positional-interpolation factor for one pixel-pos axis: shrink *this* axis's
/// coordinate only if it alone exceeds [`PIXEL_POS_TRAIN_MAX`]. The sincos table encodes height and
/// width in separate halves with the same frequencies, so each axis must be kept in its trained range
/// **independently** — they do NOT share a scale. An earlier aspect-preserving version scaled both
/// axes by the long-side factor; that needlessly compressed the in-range (short) axis and itself
/// introduced a color cast on it (a wide 5120×2880 decode got its already-in-range 2880 height dragged
/// to 2160 → green bottom), while the height axis is the only one that actually casts when
/// out-of-range. Per-axis clamping fixes the genuinely-OOD axis (tall's 5120 height) and leaves an
/// in-range axis byte-identical to the raw-absolute reference.
///
/// Exact **no-op** (`1.0`) for any axis already ≤ [`PIXEL_POS_TRAIN_MAX`] — so the sc-7843 golden
/// fixtures and every in-distribution decode stay byte-identical. `PID_PIXEL_POS_ABS=1` forces the
/// raw-absolute reference behavior (the pre-fix path) for A/B.
fn pixel_pos_axis_scale(n: i32) -> f64 {
    if std::env::var("PID_PIXEL_POS_ABS").is_ok_and(|v| v == "1") {
        return 1.0;
    }
    let n = n as f64;
    if n > PIXEL_POS_TRAIN_MAX {
        PIXEL_POS_TRAIN_MAX / n
    } else {
        1.0
    }
}

/// Host 2-D sin/cos pixel position table `[H·W, embed_dim]` (row-major over `(H, W)`, `W` fastest),
/// matching `get_2d_sincos_pos_embed_from_grid`. The reference computes it in f64; we do too, then
/// cast f32. First half encodes the **w** coordinate, second half the **h** coordinate (the
/// reference's `emb_h`/`emb_w` naming is swapped relative to the axis it encodes — replicated here).
///
/// Each axis's coordinates are scaled by [`pixel_pos_axis_scale`] first so a super-resolved decode
/// whose height (or width) exceeds the trained pixel extent stays in distribution (fixes the
/// tall-image bottom color cast) without disturbing an in-range axis; a no-op for any axis that
/// already fits, preserving reference/golden parity.
pub fn sincos_2d_pos(embed_dim: i32, h: i32, w: i32) -> Array {
    let d = (embed_dim / 2) as usize; // per-axis dim
    let half = d / 2; // omega length
    let omega: Vec<f64> = (0..half)
        .map(|k| 1.0 / 10000f64.powf(k as f64 / (d as f64 / 2.0)))
        .collect();
    // get_1d(pos): concat([sin(pos·omega), cos(pos·omega)]) -> length d
    let oned = |pos: f64, out: &mut [f32]| {
        for k in 0..half {
            let a = pos * omega[k];
            out[k] = a.sin() as f32;
            out[half + k] = a.cos() as f32;
        }
    };
    let (scale_h, scale_w) = (pixel_pos_axis_scale(h), pixel_pos_axis_scale(w));
    let n = (h * w) as usize;
    let mut buf = vec![0f32; n * embed_dim as usize];
    for i in 0..h {
        let yp = i as f64 * scale_h;
        for j in 0..w {
            let xp = j as f64 * scale_w;
            let p = (i * w + j) as usize;
            let base = p * embed_dim as usize;
            oned(xp, &mut buf[base..base + d]); // first half: w coordinate (j)
            oned(yp, &mut buf[base + d..base + 2 * d]); // second half: h coordinate (i)
        }
    }
    Array::from_slice(&buf, &[n as i32, embed_dim])
}

/// `F.unfold(x, kernel=p, stride=p).transpose(1,2)`: `[B, C, H, W]` → `[B, L, C·p²]`, patch order
/// row-major over `(Hs, Ws)` with `Ws` fastest, inner flatten `(C, pH, pW)` C-outermost.
pub fn unfold_patches(x: &Array, patch: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
    let (hs, ws) = (h / patch, w / patch);
    Ok(x.reshape(&[b, c, hs, patch, ws, patch])?
        .transpose_axes(&[0, 2, 4, 1, 3, 5])?
        .reshape(&[b, hs * ws, c * patch * patch])?)
}

/// Inverse of [`unfold_patches`] for the final fold: `[B, C, P², L]` → `[B, C, H, W]` (non-overlapping
/// stride = kernel, so a pure scatter — no overlap-add). `P²` axis is `(pH, pW)`, `L` is `(Hs, Ws)`.
pub fn fold_patches(x: &Array, c: i32, hs: i32, ws: i32, patch: i32) -> Result<Array> {
    let b = x.shape()[0];
    Ok(x.reshape(&[b, c, patch, patch, hs, ws])?
        .transpose_axes(&[0, 1, 4, 2, 5, 3])?
        .reshape(&[b, c, hs * patch, ws * patch])?)
}

/// `PixelTokenEmbedder` image-mode forward: `[B, C, H, W]` → per-pixel Linear → add the 2-D sin/cos
/// position table → patchify to `[B·L, P², D]`. Port of `PixelTokenEmbedder.forward` (dim==4 branch).
pub struct PixelTokenEmbedder {
    proj: AdaptableLinear,
    dim: i32,
    /// Per-decode cache of the 2-D sin/cos pixel positional table, keyed `(h, w)` (`self.dim` is a
    /// per-decode constant). The `~268 MB` host build + H2D upload is otherwise repeated on every
    /// forward though it is identical across the 4 sampler steps and same-sized tiles (F-153).
    pos_cache: TableCache<(i32, i32), Array>,
}

impl PixelTokenEmbedder {
    pub fn from_weights(w: &Weights, prefix: &str, dim: i32) -> Result<Self> {
        Ok(Self {
            proj: lin(w, &format!("{prefix}.proj"))?,
            dim,
            pos_cache: TableCache::default(),
        })
    }

    pub fn forward(&self, x: &Array, h: i32, w: i32, patch: i32) -> Result<Array> {
        let b = x.shape()[0];
        let (hs, ws) = (h / patch, w / patch);
        // [B,C,H,W] -> [B,H,W,C] -> proj -> [B,H,W,D]
        let xt = x.transpose_axes(&[0, 2, 3, 1])?;
        let xp = self.proj.forward(&xt)?;
        // Memoize the raw f32 pos table per `(h, w)` (F-153); the cheap reshape/dtype cast still runs
        // per call so the added `pos` is byte-identical to the pre-cache path for the activation dtype.
        let pos = memo(&self.pos_cache, (h, w), || sincos_2d_pos(self.dim, h, w))
            .reshape(&[1, h, w, self.dim])?
            .as_dtype(xp.dtype())?;
        let xp = mlx_rs::ops::add(&xp, &pos)?;
        // [B,Hs,p,Ws,p,D] -> [B,Hs,Ws,p,p,D] -> [B*L, P2, D]
        Ok(xp
            .reshape(&[b, hs, patch, ws, patch, self.dim])?
            .transpose_axes(&[0, 1, 3, 2, 4, 5])?
            .reshape(&[b * hs * ws, patch * patch, self.dim])?)
    }
}
