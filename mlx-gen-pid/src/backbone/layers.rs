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

/// Host 2-D sin/cos pixel position table `[H·W, embed_dim]` (row-major over `(H, W)`, `W` fastest),
/// matching `get_2d_sincos_pos_embed_from_grid`. The reference computes it in f64; we do too, then
/// cast f32. First half encodes the **w** coordinate, second half the **h** coordinate (the
/// reference's `emb_h`/`emb_w` naming is swapped relative to the axis it encodes — replicated here).
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
    let n = (h * w) as usize;
    let mut buf = vec![0f32; n * embed_dim as usize];
    for i in 0..h {
        for j in 0..w {
            let p = (i * w + j) as usize;
            let base = p * embed_dim as usize;
            oned(j as f64, &mut buf[base..base + d]); // first half: w coordinate (j)
            oned(i as f64, &mut buf[base + d..base + 2 * d]); // second half: h coordinate (i)
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
}

impl PixelTokenEmbedder {
    pub fn from_weights(w: &Weights, prefix: &str, dim: i32) -> Result<Self> {
        Ok(Self {
            proj: lin(w, &format!("{prefix}.proj"))?,
            dim,
        })
    }

    pub fn forward(&self, x: &Array, h: i32, w: i32, patch: i32) -> Result<Array> {
        let b = x.shape()[0];
        let (hs, ws) = (h / patch, w / patch);
        // [B,C,H,W] -> [B,H,W,C] -> proj -> [B,H,W,D]
        let xt = x.transpose_axes(&[0, 2, 3, 1])?;
        let xp = self.proj.forward(&xt)?;
        let pos = sincos_2d_pos(self.dim, h, w)
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
