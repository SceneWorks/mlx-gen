//! 3-D patchify / unpatchify — the patch reordering of `WanModel._patchify` / `unpatchify`
//! (`models/wan/model.py`), without the patch-embedding `Linear` (that lives in the DiT, S3).
//!
//! A latent video `[C, F, H, W]` is tiled by `patch_size = (pt, ph, pw)` (`(1, 2, 2)` for every Wan
//! variant) into `L = F'·H'·W'` tokens, each carrying `C·pt·ph·pw` channels in the **C-slowest**
//! order `(C, pt, ph, pw)` (matching the Conv3d weight layout the embedding expects). `unpatchify`
//! is the exact inverse for the head's `out_dim·pt·ph·pw` projection.

use mlx_rs::Array;

use mlx_gen::Result;

/// Reorder a latent video `[C, F, H, W]` into patch tokens `[L, C·pt·ph·pw]` and return the patch
/// grid `(F', H', W')` (`F' = F/pt`, etc.). No projection — just the reshape/transpose.
pub fn patchify(
    x: &Array,
    patch_size: (usize, usize, usize),
) -> Result<(Array, (usize, usize, usize))> {
    let shape = x.shape();
    debug_assert_eq!(shape.len(), 4, "patchify expects [C, F, H, W]");
    let (c, f, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let (pt, ph, pw) = (
        patch_size.0 as i32,
        patch_size.1 as i32,
        patch_size.2 as i32,
    );
    let (f_out, h_out, w_out) = (f / pt, h / ph, w / pw);

    // [C, F', pt, H', ph, W', pw] → [F', H', W', C, pt, ph, pw] → [L, C·pt·ph·pw]
    let tokens = x
        .reshape(&[c, f_out, pt, h_out, ph, w_out, pw])?
        .transpose_axes(&[1, 3, 5, 0, 2, 4, 6])?
        .reshape(&[f_out * h_out * w_out, c * pt * ph * pw])?;

    Ok((tokens, (f_out as usize, h_out as usize, w_out as usize)))
}

/// Inverse of [`patchify`]: reconstruct a latent video `[out_dim, F, H, W]` from head tokens
/// `[L, out_dim·pt·ph·pw]` given the patch grid `(F', H', W')`. `F = F'·pt`, etc.
pub fn unpatchify(
    x: &Array,
    grid: (usize, usize, usize),
    out_dim: usize,
    patch_size: (usize, usize, usize),
) -> Result<Array> {
    let (f, h, w) = (grid.0 as i32, grid.1 as i32, grid.2 as i32);
    let (pt, ph, pw) = (
        patch_size.0 as i32,
        patch_size.1 as i32,
        patch_size.2 as i32,
    );
    let c = out_dim as i32;

    // [F', H', W', pt, ph, pw, C] → [C, F', pt, H', ph, W', pw] → [C, F'·pt, H'·ph, W'·pw]
    Ok(x.reshape(&[f, h, w, pt, ph, pw, c])?
        .transpose_axes(&[6, 0, 3, 1, 4, 2, 5])?
        .reshape(&[c, f * pt, h * ph, w * pw])?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::sum;

    #[test]
    fn patchify_shapes_5b() {
        // 5B patch_size (1,2,2): C=48, latent F=21, H=44, W=80 → L = 21·22·40 = 18480.
        let x = Array::zeros::<f32>(&[48, 21, 44, 80]).unwrap();
        let (tokens, grid) = patchify(&x, (1, 2, 2)).unwrap();
        assert_eq!(grid, (21, 22, 40));
        assert_eq!(tokens.shape(), &[18480, 48 * 2 * 2]); // [18480, 192]
    }

    #[test]
    fn patchify_packs_c_pt_ph_pw_order() {
        // [C=1, F=1, H=2, W=4], patch (1,2,2): two width patches, each [ph,pw] = 2×2.
        // Row-major data: row0 = 0,1,2,3 ; row1 = 4,5,6,7.
        let x = Array::from_slice(&[0.0_f32, 1., 2., 3., 4., 5., 6., 7.], &[1, 1, 2, 4]);
        let (tokens, grid) = patchify(&x, (1, 2, 2)).unwrap();
        assert_eq!(grid, (1, 1, 2)); // L = 2 patches
        assert_eq!(tokens.shape(), &[2, 4]);
        // Patch 0 covers w∈{0,1}: (ph,pw) = (0,0)=0,(0,1)=1,(1,0)=4,(1,1)=5 → [0,1,4,5].
        // Patch 1 covers w∈{2,3}: → [2,3,6,7].
        assert_eq!(tokens.as_slice::<f32>(), &[0., 1., 4., 5., 2., 3., 6., 7.]);
    }

    #[test]
    fn single_channel_round_trips() {
        // For C=1 the (C,pt,ph,pw) pack and the (pt,ph,pw,C) unpack coincide → exact inverse.
        let x = Array::from_slice(&[0.0_f32, 1., 2., 3., 4., 5., 6., 7.], &[1, 1, 2, 4]);
        let (tokens, grid) = patchify(&x, (1, 2, 2)).unwrap();
        let back = unpatchify(&tokens, grid, 1, (1, 2, 2)).unwrap();
        assert_eq!(back.shape(), &[1, 1, 2, 4]);
        let diff: f32 = sum(
            mlx_rs::ops::abs(mlx_rs::ops::subtract(&back, &x).unwrap()).unwrap(),
            None,
        )
        .unwrap()
        .item();
        assert_eq!(diff, 0.0, "C=1 round trip changed values");
    }
}
