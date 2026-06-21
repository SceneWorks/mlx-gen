//! sc-6894 (F-009) — **z16 Wan 2.1 VAE decode anchor sweep.** Measures the real MLX peak GPU
//! allocation of [`WanVae::decode`] / [`WanVae::decode_tiled`] across output sizes and tile sizes, so
//! the budgeted decode cost model (`estimated_z16_decode_peak_gib`) can be **fit from real
//! measurements** — the way sc-4998 fit the z48 `vae22` model from `wedge_sweep.rs`. The z16 VAE is a
//! few hundred MB (only `vae.safetensors`, never the 14B DiT), so this loads fast; the decode itself
//! is the GPU-heavy part. One config **per process** (env-driven) so an OOM on the largest configs
//! kills only this process and the driving shell loop keeps the earlier anchors. Synthetic (random)
//! latents — we measure cost/scaling, not parity.
//!
//! ```text
//! Z16_VAE=~/.cache/huggingface/hub/models--SceneWorks--wan2.2-t2v-a14b-mlx/snapshots/<h>/vae.safetensors \
//! Z16_W=768 Z16_H=768 Z16_FRAMES=16 \
//!   cargo test -p mlx-gen-wan --test vae16_decode_sweep --release -- --ignored --nocapture
//! # add Z16_TILE_PX=384 [Z16_OVERLAP_PX=64 Z16_TILE_FRAMES=.. Z16_OVERLAP_FRAMES=..] for a tiled run
//! # add Z16_LIMIT_GB=64 to simulate a smaller-RAM tier
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_rs::memory::{get_memory_limit, get_peak_memory, reset_peak_memory, set_memory_limit};
use mlx_rs::random;

use mlx_gen::tiling::{SpatialTiling, TemporalTiling, TilingConfig};
use mlx_gen::weights::Weights;
use mlx_gen_wan::WanVae;

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(|s| {
        let s = s.to_string_lossy();
        if let Some(rest) = s.strip_prefix("~/") {
            if let Some(home) = std::env::var_os("HOME") {
                return PathBuf::from(format!("{}/{rest}", home.to_string_lossy()));
            }
        }
        PathBuf::from(s.to_string())
    })
}

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn gb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[test]
#[ignore = "needs the 14B z16 vae.safetensors (Z16_VAE); GPU-heavy"]
fn wan_z16_vae_decode_sweep() {
    let vae_path = match env_path("Z16_VAE") {
        Some(p) => p,
        None => {
            eprintln!("skip: set Z16_VAE to the 14B snapshot's vae.safetensors");
            return;
        }
    };
    let w_out = env_usize("Z16_W", 768) as i32;
    let h_out = env_usize("Z16_H", 768) as i32;
    let frames = env_usize("Z16_FRAMES", 16) as i32;
    // z16 Wan 2.1 VAE: spatial /8, temporal /4 (non-causal, out_f = T_lat·4).
    assert_eq!(
        frames % 4,
        0,
        "Z16_FRAMES must be divisible by 4 (got {frames})"
    );
    assert_eq!(h_out % 8, 0, "Z16_H must be a multiple of 8");
    assert_eq!(w_out % 8, 0, "Z16_W must be a multiple of 8");
    let (z, t_lat, h_lat, w_lat) = (16, frames / 4, h_out / 8, w_out / 8);

    if let Ok(lim) = std::env::var("Z16_LIMIT_GB") {
        if let Ok(g) = lim.parse::<usize>() {
            let prev = set_memory_limit(g << 30);
            println!(
                "[limit] pinned MLX memory limit {g} GB (was {:.0} GB)",
                gb(prev)
            );
        }
    }

    let weights = Weights::from_file(&vae_path).expect("read z16 vae.safetensors");
    let vae = WanVae::from_weights(&weights).expect("WanVae::from_weights");

    // Synthetic latent [B=1, z=16, T_lat, H_lat, W_lat] (random — cost only, not parity).
    let key = random::key(0).unwrap();
    let latent =
        random::normal::<f32>(&[1, z, t_lat, h_lat, w_lat], None, None, Some(&key)).unwrap();

    let (out_f, out_h, out_w) = (t_lat * 4, h_lat * 8, w_lat * 8);

    // Tile selection: Z16_BUDGETED=1 exercises the PRODUCTION `auto_tiling_budgeted_z16` selector
    // (honors Z16_LIMIT_GB) — the real bounded/catchable path. Else Z16_TILE_PX sets a fixed tile (for
    // anchor fitting); else single-pass.
    let tile_px = env_usize("Z16_TILE_PX", 0) as i32;
    let cfg = if env_usize("Z16_BUDGETED", 0) == 1 {
        mlx_gen_wan::pipeline::auto_tiling_budgeted_z16(out_w, out_h, out_f)
            .expect("z16 decode fits the budget (catchable error if not)")
    } else if tile_px > 0 {
        let overlap_px = env_usize("Z16_OVERLAP_PX", 64) as i32;
        let spatial = Some(SpatialTiling {
            tile_px,
            overlap_px,
        });
        let tf = env_usize("Z16_TILE_FRAMES", 0) as i32;
        let temporal = (tf > 0).then(|| TemporalTiling {
            tile_frames: tf,
            overlap_frames: env_usize("Z16_OVERLAP_FRAMES", (tf / 2).max(1) as usize) as i32,
        });
        Some(TilingConfig { spatial, temporal })
    } else {
        None
    };
    let out_vox = (out_f as i64) * (out_h as i64) * (out_w as i64);
    // Largest-tile output voxels in the cost model's convention (min(tile_px, out_dim)).
    let (tile_f, tile_h, tile_w) = match &cfg {
        Some(c) => (
            c.temporal
                .map(|t| (t.tile_frames as i64).min(out_f as i64))
                .unwrap_or(out_f as i64),
            c.spatial
                .map(|s| (s.tile_px as i64).min(out_h as i64))
                .unwrap_or(out_h as i64),
            c.spatial
                .map(|s| (s.tile_px as i64).min(out_w as i64))
                .unwrap_or(out_w as i64),
        ),
        None => (out_f as i64, out_h as i64, out_w as i64),
    };
    let tile_vox = tile_f * tile_h * tile_w;

    println!(
        "\n=== z16 sweep: out {out_w}x{out_h}x{out_f}  latent[z{z},T{t_lat},{h_lat},{w_lat}]  \
         tiled={}  MLX limit={:.0} GB ===",
        cfg.is_some(),
        gb(get_memory_limit())
    );

    reset_peak_memory();
    let t = Instant::now();
    let video = match &cfg {
        Some(c) => vae.decode_tiled(&latent, c).unwrap(),
        None => vae.decode(&latent).unwrap(),
    };
    mlx_rs::transforms::eval([&video]).unwrap();
    let secs = t.elapsed().as_secs_f64();
    let peak_bytes = get_peak_memory();
    let peak = gb(peak_bytes);

    println!(
        "[Z16 decode] -> {:?}  {secs:.1}s  peak={peak:.2} GB",
        video.shape()
    );
    // Parse-friendly anchor line: peak vs output/tile voxels → the two cost coefficients.
    println!(
        "ANCHOR out_vox={out_vox} tile_vox={tile_vox} peak_gb={peak:.4} \
         peak_bytes_per_out_vox={:.1} peak_bytes_per_tile_vox={:.1}",
        peak_bytes as f64 / out_vox as f64,
        peak_bytes as f64 / tile_vox as f64,
    );
}
