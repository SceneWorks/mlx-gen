//! Model-agnostic neural-net primitives — the shared `nn` layer of `mlx-gen` core.
//!
//! These are the genuinely family-independent leaf ops: dense linear, SiLU, NHWC `conv2d`,
//! pytorch-compatible `group_norm`, and nearest `upsample`. Model-specific block assemblies
//! (attention / RoPE / SwiGLU layouts) intentionally stay in their family crates — see
//! `docs/MODEL_ARCHITECTURE.md` §3.2 ("each family crate owns its blocks"). A primitive
//! graduates here only once it is provably model-agnostic; we do not lift a block to a shared
//! abstraction off a single model.

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{
    add, broadcast_to, conv2d as conv2d_op, conv3d as conv3d_op, matmul, multiply, sigmoid,
};
use mlx_rs::Array;

use crate::Result;

/// `y = x · Wᵀ + b` for a stored `[out, in]` weight + bias (PyTorch `nn.Linear` convention).
pub fn linear(x: &Array, w: &Array, b: &Array) -> Result<Array> {
    Ok(add(&matmul(x, w.t())?, b)?)
}

/// SiLU / swish activation: `x · sigmoid(x)`.
pub fn silu(x: &Array) -> Result<Array> {
    Ok(multiply(x, &sigmoid(x)?)?)
}

/// 2-D conv over NHWC `x` with an mlx `[out, kH, kW, in]` weight (+ optional bias).
pub fn conv2d(x: &Array, w: &Array, b: Option<&Array>, stride: i32, padding: i32) -> Result<Array> {
    let mut y = conv2d_op(x, w, (stride, stride), (padding, padding), (1, 1), 1)?;
    if let Some(b) = b {
        y = add(&y, b)?;
    }
    Ok(y)
}

/// 3-D conv over NDHWC `x` with an mlx `[out, kD, kH, kW, in]` weight (+ optional bias).
/// `stride`/`padding` are per-axis `(depth, height, width)`. Qwen's causal-Conv3d VAE applies
/// its asymmetric temporal padding manually and calls this with `padding (0, 0, 0)`; future
/// video families (Wan2.2 / LTX) reuse it directly — hence it lives in shared core `nn`.
pub fn conv3d(
    x: &Array,
    w: &Array,
    b: Option<&Array>,
    stride: (i32, i32, i32),
    padding: (i32, i32, i32),
) -> Result<Array> {
    let mut y = conv3d_op(x, w, stride, padding, (1, 1, 1), 1)?;
    if let Some(b) = b {
        y = add(&y, b)?;
    }
    Ok(y)
}

/// PyTorch-compatible group normalization over NHWC `x` (`weight`/`bias` are per-channel).
/// Mirrors mlx-rs `GroupNorm::pytorch_group_norm` + affine: split channels into `num_groups`,
/// layer-norm each group, then scale/shift by `weight`/`bias`.
pub fn group_norm(
    x: &Array,
    weight: &Array,
    bias: &Array,
    num_groups: i32,
    eps: f32,
) -> Result<Array> {
    let sh = x.shape();
    let batch = sh[0];
    let dims = sh[sh.len() - 1];
    let rest = &sh[1..sh.len() - 1];
    let group_size = dims / num_groups;

    let g = x
        .reshape(&[batch, -1, num_groups, group_size])?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, num_groups, -1])?;
    let g = layer_norm(&g, None, None, eps)?;
    let g = g
        .reshape(&[batch, num_groups, -1, group_size])?
        .transpose_axes(&[0, 2, 1, 3])?;

    let mut shape = vec![batch];
    shape.extend_from_slice(rest);
    shape.push(dims);
    let normed = g.reshape(&shape)?;
    Ok(add(&multiply(&normed, weight)?, bias)?)
}

/// Nearest-neighbor upsample of NHWC `x` by `scale` (broadcast + reshape).
pub fn upsample_nearest(x: &Array, scale: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let x6 = x.reshape(&[b, h, 1, w, 1, c])?;
    let bc = broadcast_to(&x6, &[b, h, scale, w, scale, c])?;
    Ok(bc.reshape(&[b, h * scale, w * scale, c])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv3d_1x1x1_sums_input_channels_with_bias() {
        // NDHWC: a single voxel with 2 input channels [1, 2].
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 1, 2]);
        // weight [out=1, kD=1, kH=1, kW=1, in=2] = ones -> sums over the input channels.
        let w = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 1, 1, 2]);
        let bias = Array::from_slice(&[10.0f32], &[1]);
        let y = conv3d(&x, &w, Some(&bias), (1, 1, 1), (0, 0, 0)).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1, 1, 1]);
        assert_eq!(y.item::<f32>(), 13.0); // 1 + 2 + bias 10
    }
}
