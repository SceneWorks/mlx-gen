//! sc-6894 (F-004) — **LTX-2.3 video VAE decode anchor sweep.** Measures the real MLX peak GPU
//! allocation of [`LtxVideoVae::decode`] / [`LtxVideoVae::decode_tiled`] across output sizes and tile
//! sizes, so the budgeted decode cost model (`estimated_ltx_decode_peak_gib`) can be **fit from real
//! measurements** — the way sc-4998 fit the Wan z48 model and sc-6894 fit the z16 model. Decode-only:
//! loads just `vae_decoder.safetensors` (no encoder). One config **per process** (env-driven) so an
//! OOM on the largest configs kills only this process; the driving shell loop keeps earlier anchors.
//! Synthetic (random) latents — we measure cost/scaling, not parity.
//!
//! ```text
//! LTX_VAE_DIR=~/.cache/huggingface/hub/models--SceneWorks--ltx-2.3-mlx/snapshots/<h>/q8 \
//! LTX_W=768 LTX_H=768 LTX_FRAMES=25 \
//!   cargo test -p mlx-gen-ltx --test vae_decode_sweep --release -- --ignored --nocapture
//! # add LTX_TILE_PX=256 [LTX_OVERLAP_PX=32 LTX_TILE_FRAMES=.. LTX_OVERLAP_FRAMES=..] for a tiled run
//! # add LTX_LIMIT_GB=48 to simulate a smaller-RAM tier
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_rs::memory::{get_memory_limit, get_peak_memory, reset_peak_memory, set_memory_limit};
use mlx_rs::random;

use mlx_gen::tiling::{SpatialTiling, TemporalTiling, TilingConfig};
use mlx_gen::weights::Weights;
use mlx_gen_ltx::{LtxVaeConfig, LtxVideoVae};

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
#[ignore = "needs the ltx-2.3-mlx vae_decoder.safetensors (LTX_VAE_DIR); GPU-heavy"]
fn ltx_vae_decode_sweep() {
    let dir = match env_path("LTX_VAE_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skip: set LTX_VAE_DIR to a snapshot dir holding vae_decoder.safetensors");
            return;
        }
    };
    let w_out = env_usize("LTX_W", 768) as i32;
    let h_out = env_usize("LTX_H", 768) as i32;
    let frames = env_usize("LTX_FRAMES", 25) as i32;
    // LTX VAE: spatial /32, temporal /8 **causal** (out_f = 1 + (T_lat−1)·8 ⇒ frames = 1 + 8·k).
    assert_eq!(
        (frames - 1) % 8,
        0,
        "LTX_FRAMES must be 1 + 8·k (got {frames})"
    );
    assert_eq!(h_out % 32, 0, "LTX_H must be a multiple of 32");
    assert_eq!(w_out % 32, 0, "LTX_W must be a multiple of 32");
    let (z, t_lat, h_lat, w_lat) = (128, (frames - 1) / 8 + 1, h_out / 32, w_out / 32);

    if let Ok(lim) = std::env::var("LTX_LIMIT_GB") {
        if let Ok(g) = lim.parse::<usize>() {
            let prev = set_memory_limit(g << 30);
            println!(
                "[limit] pinned MLX memory limit {g} GB (was {:.0} GB)",
                gb(prev)
            );
        }
    }

    let vae_cfg = LtxVaeConfig::from_model_dir(&dir).expect("read LtxVaeConfig");
    let decoder_w =
        Weights::from_file(dir.join("vae_decoder.safetensors")).expect("read vae_decoder");
    let vae =
        LtxVideoVae::from_weights(&decoder_w, None, &vae_cfg).expect("LtxVideoVae::from_weights");

    // Synthetic latent [B=1, 128, T_lat, H_lat, W_lat] (random — cost only, not parity).
    let key = random::key(0).unwrap();
    let latent =
        random::normal::<f32>(&[1, z, t_lat, h_lat, w_lat], None, None, Some(&key)).unwrap();

    let (out_f, out_h, out_w) = (1 + (t_lat - 1) * 8, h_lat * 32, w_lat * 32);

    // Fixed-tile iff LTX_TILE_PX is set; otherwise single-pass. (Budgeted mode is added once the cost
    // model exists — see `auto_tiling_budgeted_ltx`.)
    let tile_px = env_usize("LTX_TILE_PX", 0) as i32;
    let cfg = if env_usize("LTX_BUDGETED", 0) == 1 {
        mlx_gen_ltx::pipeline::auto_tiling_budgeted_ltx(out_w, out_h, out_f)
            .expect("ltx decode fits the budget (catchable error if not)")
    } else if tile_px > 0 {
        let overlap_px = env_usize("LTX_OVERLAP_PX", 32) as i32;
        let spatial = Some(SpatialTiling {
            tile_px,
            overlap_px,
        });
        let tf = env_usize("LTX_TILE_FRAMES", 0) as i32;
        let temporal = (tf > 0).then(|| TemporalTiling {
            tile_frames: tf,
            overlap_frames: env_usize("LTX_OVERLAP_FRAMES", (tf / 2).max(1) as usize) as i32,
        });
        Some(TilingConfig { spatial, temporal })
    } else {
        None
    };

    let out_vox = (out_f as i64) * (out_h as i64) * (out_w as i64);
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
        "\n=== ltx sweep: out {out_w}x{out_h}x{out_f}  latent[z{z},T{t_lat},{h_lat},{w_lat}]  \
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
        "[LTX decode] -> {:?}  {secs:.1}s  peak={peak:.2} GB",
        video.shape()
    );
    println!(
        "ANCHOR out_vox={out_vox} tile_vox={tile_vox} peak_gb={peak:.4} \
         peak_bytes_per_out_vox={:.1} peak_bytes_per_tile_vox={:.1}",
        peak_bytes as f64 / out_vox as f64,
        peak_bytes as f64 / tile_vox as f64,
    );
}
