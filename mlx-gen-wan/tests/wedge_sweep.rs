//! sc-4986 — bounded **wedge-sweep** harness for the Wan2.2 TI2V-5B image_to_video GPU wedge.
//!
//! Drives the *real* converted 5B DiT through the genuine `denoise_ti2v` mask-blend loop at a single
//! caller-chosen resolution / frame-count, timing each step (wall-clock) and recording the MLX peak
//! GPU allocation. One config **per process** (env-driven) so that a Metal command-buffer watchdog
//! abort (SIGABRT) or OOM (SIGKILL) on the largest configs kills only this process — the driving
//! shell loop keeps the prior configs' results. Synthetic (random) latents/context: we measure
//! *cost/scaling*, not parity (parity is `ti2v_real_parity.rs`).
//!
//! ```text
//! WAN_5B_MODEL_DIR=~/.cache/mlx-gen-models/wan_2_2_ti2v_5b_mlx_bf16 \
//! WAN_SWEEP_W=1280 WAN_SWEEP_H=704 WAN_SWEEP_FRAMES=145 WAN_SWEEP_STEPS=2 \
//!   cargo test -p mlx-gen-wan --test wedge_sweep --release -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_rs::memory::{get_memory_limit, get_peak_memory, reset_peak_memory};
use mlx_rs::random;

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::pipeline::{build_ti2v_mask, denoise_ti2v, ti2v_blend_init};
use mlx_gen_wan::scheduler::SolverKind;
use mlx_gen_wan::{Wan22Vae, WanTransformer};

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

/// `WAN_SWEEP_LIMIT_GB` pins the MLX memory limit before tile selection, so a 128 GB machine can
/// simulate the smaller tier the budgeted plan must protect (`auto_tiling_budgeted` reads the
/// limit). The limit is soft, so the decode still runs — but the chosen tile + peak reflect the
/// smaller budget. No-op when the env var is unset.
fn pin_limit_from_env() {
    if let Ok(lim) = std::env::var("WAN_SWEEP_LIMIT_GB") {
        if let Ok(gbs) = lim.parse::<usize>() {
            let prev = mlx_rs::memory::set_memory_limit(gbs << 30);
            println!(
                "[limit] pinned MLX memory limit {gbs} GB (was {:.0} GB) for tile selection",
                gb(prev)
            );
        }
    }
}

#[test]
#[ignore = "needs the ~23 GB converted Wan2.2-TI2V-5B snapshot (WAN_5B_MODEL_DIR); GPU-heavy"]
fn wan_ti2v_5b_wedge_sweep() {
    let model_dir = match env_path("WAN_5B_MODEL_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skip: set WAN_5B_MODEL_DIR to the converted 5B snapshot dir");
            return;
        }
    };
    let w = env_usize("WAN_SWEEP_W", 1280) as i32;
    let h = env_usize("WAN_SWEEP_H", 704) as i32;
    let frames = env_usize("WAN_SWEEP_FRAMES", 145);
    let steps = env_usize("WAN_SWEEP_STEPS", 2);

    let cfg = WanModelConfig::from_model_dir(&model_dir).expect("read config.json");
    assert!(
        !cfg.dual_model && cfg.is_ti2v(),
        "expected the dense TI2V-5B config"
    );

    // Latent geometry (stride 4×16×16, z48). Frames must be 1+4·k for a clean T_lat.
    assert_eq!((frames - 1) % 4, 0, "frames must be 1 + 4·k (got {frames})");
    let z = cfg.vae_z_dim as i32;
    let t_lat = ((frames - 1) / 4 + 1) as i32;
    let h_lat = h / cfg.vae_stride.1 as i32;
    let w_lat = w / cfg.vae_stride.2 as i32;
    // DiT token count = (T/pt)·(H/ph)·(W/pw) — the self-attention sequence length.
    let (pt, ph, pw) = cfg.patch_size;
    let tokens = (t_lat as usize / pt) * (h_lat as usize / ph) * (w_lat as usize / pw);
    let budget = get_memory_limit();

    println!(
        "\n=== sweep: {w}x{h}  frames={frames}  steps={steps} ===\n\
         latent [z{z}, T{t_lat}, {h_lat}, {w_lat}]  DiT tokens(L)={tokens}  L^2={:.3e}\n\
         MLX memory limit = {:.0} GB",
        (tokens as f64) * (tokens as f64),
        gb(budget),
    );

    let key = random::key(0).unwrap();

    // WAN_SWEEP_VAE=1: probe the real z48 vae22 ENCODE (conditioning image, runs *before* the denoise
    // loop at model.rs:344) and DECODE (final latents → video) at full res — the pre-/post-loop stages
    // the DiT-only sweep skips. eval() forces each so the timing/peak is the genuine GPU cost.
    if env_usize("WAN_SWEEP_VAE", 0) == 1 {
        let vae_w = Weights::from_file(model_dir.join("vae.safetensors")).expect("vae weights");
        let vae = Wan22Vae::from_weights(&vae_w).expect("vae22");

        // ENCODE: a single conditioning image [1,1,H,W,3] in [-1,1] (T=1 → one causal chunk).
        let img = random::normal::<f32>(&[1, 1, h, w, 3], None, None, Some(&key)).unwrap();
        reset_peak_memory();
        let t = Instant::now();
        let z_img = vae.encode(&img).unwrap();
        mlx_rs::transforms::eval([&z_img]).unwrap();
        println!(
            "[VAE encode] {w}x{h} img -> {:?}  {:.1}s  peak={:.1} GB",
            z_img.shape(),
            t.elapsed().as_secs_f64(),
            gb(get_peak_memory())
        );

        // DECODE: the full latent stack [z, T_lat, h_lat, w_lat] → video (heaviest post-loop op).
        // Mirror the PRODUCTION path: auto_tiling_budgeted(...) picks the memory-budgeted tiling
        // exactly as model.rs (sc-4998). WAN_SWEEP_NOTILE=1 forces single-pass to confirm it OOMs.
        let latents =
            random::normal::<f32>(&[z, t_lat, h_lat, w_lat], None, None, Some(&key)).unwrap();
        let gen_frames = frames as i32; // model.rs passes gen_frames (= frames + trim·4); trim=0 here

        // WAN_SWEEP_LIMIT_GB: simulate a smaller-RAM tier (sc-4998).
        pin_limit_from_env();
        // Plan the tiling for the dtype under test (sc-5039): bf16's lighter cost coefficient lets the
        // budget fit bigger tiles. The f32 reference decode reuses this tiling for the cosine A/B.
        let bf16 = env_usize("WAN_SWEEP_VAE_BF16", 0) == 1;
        let tiling = mlx_gen_wan::pipeline::auto_tiling_budgeted(h, w, gen_frames, bf16)
            .expect("decode tiling within this machine's budget (sc-4998)");
        let notile = env_usize("WAN_SWEEP_NOTILE", 0) == 1;
        println!(
            "[tiling] auto_budgeted({h},{w},{gen_frames}) = {}  (notile_override={notile})",
            match &tiling {
                Some(c) => format!("Some({c:?})"),
                None => "None(single-pass)".to_string(),
            }
        );
        reset_peak_memory();
        let t = Instant::now();
        let video = match (notile, tiling.as_ref()) {
            (false, Some(cfg)) => vae.decode_tiled(&latents, cfg).unwrap(),
            _ => vae.decode(&latents).unwrap(),
        };
        mlx_rs::transforms::eval([&video]).unwrap();
        println!(
            "[VAE decode f32] latents[z{z},T{t_lat},{h_lat},{w_lat}] -> {:?}  {:.1}s  peak={:.1} GB",
            video.shape(),
            t.elapsed().as_secs_f64(),
            gb(get_peak_memory())
        );

        // WAN_SWEEP_VAE_BF16=1 (sc-5039): decode the SAME latents with a bf16-cast decoder, then
        // report cosine(bf16, f32) + finiteness + the bf16 peak/time. Random latents (normal) proxy
        // the real denoised z48 magnitude; this measures the bf16 decode's fidelity + memory/speed.
        if bf16 {
            let mut vae_wb =
                Weights::from_file(model_dir.join("vae.safetensors")).expect("vae weights");
            vae_wb
                .cast_all(mlx_rs::Dtype::Bfloat16)
                .expect("cast vae bf16");
            let vae_bf16 = Wan22Vae::from_weights(&vae_wb).expect("vae22 bf16");
            reset_peak_memory();
            let t = Instant::now();
            let video_bf16 = match (notile, tiling.as_ref()) {
                (false, Some(cfg)) => vae_bf16.decode_tiled(&latents, cfg).unwrap(),
                _ => vae_bf16.decode(&latents).unwrap(),
            };
            mlx_rs::transforms::eval([&video_bf16]).unwrap();
            let bf16_secs = t.elapsed().as_secs_f64();
            let bf16_peak = gb(get_peak_memory());
            let (g, f) = (video_bf16.as_slice::<f32>(), video.as_slice::<f32>());
            let finite = g.iter().all(|v| v.is_finite());
            let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
            for (x, y) in g.iter().zip(f.iter()) {
                dot += *x as f64 * *y as f64;
                na += *x as f64 * *x as f64;
                nb += *y as f64 * *y as f64;
            }
            let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
            println!(
                "[VAE decode bf16] {:.1}s  peak={bf16_peak:.1} GB  finite={finite}  \
                 cosine(bf16,f32)={cos:.6}",
                bf16_secs
            );
        }
        return;
    }

    // Synthetic inputs of the exact production shapes (random — cost only, not parity).
    let lat_shape = [z, t_lat, h_lat, w_lat];
    let noise = random::normal::<f32>(&lat_shape, None, None, Some(&key)).unwrap();
    let z_img = random::normal::<f32>(&[z, 1, h_lat, w_lat], None, None, Some(&key)).unwrap();
    let (mask, mask_tokens) = build_ti2v_mask(
        cfg.vae_z_dim,
        t_lat as usize,
        h_lat as usize,
        w_lat as usize,
        cfg.patch_size,
    );
    let init = ti2v_blend_init(&z_img, &mask, &noise).unwrap();

    // Real DiT weights (the genuine 23 GB bf16 stack — load is part of the cost profile).
    let load_t = Instant::now();
    let dit_w = Weights::from_file(model_dir.join("model.safetensors")).expect("dit weights");
    let dit = WanTransformer::from_weights(&dit_w, &cfg).expect("DiT");
    println!(
        "[load] DiT from_weights in {:.1}s",
        load_t.elapsed().as_secs_f64()
    );

    // Embedded contexts: a random raw UMT5 context [text_len, text_dim] through the DiT's own
    // text-embedding (avoids loading the T5 encoder; shape-faithful for the forward).
    let raw_ctx = random::normal::<f32>(
        &[cfg.text_len as i32, cfg.text_dim as i32],
        None,
        None,
        Some(&key),
    )
    .unwrap();
    let ctx_cond = dit.embed_text(&raw_ctx).unwrap();
    let ctx_uncond = dit.embed_text(&raw_ctx).unwrap();
    // WAN_SWEEP_NOCFG=1 → CFG off (ctx_uncond=None → B=1, one forward/step) = the Lightning fast path.
    let nocfg = env_usize("WAN_SWEEP_NOCFG", 0) == 1;
    let (uncond, guidance) = if nocfg {
        (None, 1.0)
    } else {
        (Some(&ctx_uncond), cfg.sample_guide_scale.effective())
    };
    println!(
        "[cfg] {}",
        if nocfg {
            "OFF (1 forward/step)"
        } else {
            "ON (2 forwards/step)"
        }
    );

    reset_peak_memory();
    let mut step_times: Vec<f64> = Vec::with_capacity(steps);
    let loop_t = Instant::now();
    let mut last = Instant::now();
    let _latents = denoise_ti2v(
        &dit,
        SolverKind::UniPC,
        cfg.num_train_timesteps,
        steps,
        cfg.sample_shift,
        guidance,
        &ctx_cond,
        uncond,
        &init,
        &z_img,
        &mask,
        &mask_tokens,
        &mlx_gen::CancelFlag::default(),
        &mut |i| {
            let dt = last.elapsed().as_secs_f64();
            last = Instant::now();
            step_times.push(dt);
            println!(
                "[step {i}/{steps}] {dt:.1}s   peak={:.1} GB",
                gb(get_peak_memory())
            );
        },
    )
    .expect("denoise_ti2v");

    let total = loop_t.elapsed().as_secs_f64();
    let per_step = step_times.iter().copied().fold(f64::MAX, f64::min);
    println!(
        "\n[RESULT] {w}x{h} frames={frames} L={tokens}  best_step={per_step:.1}s  \
         peak={:.1} GB / {:.0} GB  ({steps} steps in {total:.1}s)\n",
        gb(get_peak_memory()),
        gb(budget),
    );
}
