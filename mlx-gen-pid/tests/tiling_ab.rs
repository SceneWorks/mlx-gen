//! sc-10087 A/B: whole-image vs spatially-tiled PiD decode of an **identical** real latent.
//!
//! The tiled path ([`PidDecoder::decode_tiled`]) is an approximation (the PixDiT is globally
//! self-attentive, so per-tile decode drops cross-tile attention). This test quantifies how close it
//! lands to the whole-image reference on a real production latent, to decide the go/no-go on tiling.
//!
//! Both paths are driven from ONE reconstructed decoder, so the seed (hence the seeded noise + ε) is
//! identical — the ONLY difference between the two outputs is the tiling. The real latent + caption come
//! from a capture (`PID_CAPTURE_LATENT`) of a Krea/Qwen generation; see `maybe_capture`.
//!
//! ```sh
//! # 1) capture a real 1024-native latent (→4096² output) from a Krea run:
//! PID_CAPTURE_LATENT=/tmp/pid_ab.safetensors KREA_PID_SIZE=1024 \
//!   PID_QWEN_SAFETENSORS=.../pid_qwenimage_2kto4k.safetensors PID_GEMMA_DIR=.../snapshots/<snap> \
//!   cargo test -p mlx-gen-krea --release --test pid_decode_real_weights -- --ignored --nocapture
//! # 2) run the A/B on the capture:
//! PID_AB_CAPTURE=/tmp/pid_ab.safetensors PID_QWEN_SAFETENSORS=.../pid_qwenimage_2kto4k.safetensors \
//!   cargo test -p mlx-gen-pid --release --test tiling_ab -- --ignored --nocapture
//! ```

use mlx_gen::decoder::LatentDecoder;
use mlx_gen::weights::Weights;
use mlx_gen_pid::{PidConfig, PidDecoder, PidNet, Sampler, SamplerConfig};
use mlx_rs::ops::{abs, max, mean, multiply, subtract};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

fn env_or(name: &str, default: String) -> String {
    std::env::var(name).unwrap_or(default)
}

/// `[1,3,H,W]` in [-1,1] → RGB8 PNG (logical-order contiguous copy before slicing; see from_clean).
fn save_png(out: &Array, path: &str) {
    let sh = out.shape();
    let (h, w) = (sh[2], sh[3]);
    let hwc = out
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[3, h, w])
        .unwrap()
        .transpose_axes(&[1, 2, 0])
        .unwrap()
        .reshape(&[h * w * 3])
        .unwrap();
    let v: Vec<f32> = hwc.as_slice::<f32>().to_vec();
    let buf: Vec<u8> = v
        .iter()
        .map(|x| (((x + 1.0) * 127.5).clamp(0.0, 255.0)) as u8)
        .collect();
    image::save_buffer(path, &buf, w as u32, h as u32, image::ColorType::Rgb8).unwrap();
}

/// Per-pixel max-over-channels |whole−tiled|, scaled ×`gain` and clamped → grayscale heatmap PNG.
fn save_diff_heatmap(diff_abs: &Array, path: &str, gain: f32) {
    let sh = diff_abs.shape();
    let (h, w) = (sh[2], sh[3]);
    // max over channel axis 1 → [1,1,H,W] → [H,W]
    let m = diff_abs
        .as_dtype(Dtype::Float32)
        .unwrap()
        .max_axis(1, true)
        .unwrap()
        .reshape(&[h * w])
        .unwrap();
    let v: Vec<f32> = m.as_slice::<f32>().to_vec();
    let buf: Vec<u8> = v
        .iter()
        .map(|x| (x * gain * 255.0).clamp(0.0, 255.0) as u8)
        .collect();
    image::save_buffer(path, &buf, w as u32, h as u32, image::ColorType::L8).unwrap();
}

fn scalar_of(a: &Array) -> f32 {
    a.as_dtype(Dtype::Float32).unwrap().item::<f32>()
}

#[test]
#[ignore = "needs a PID_CAPTURE_LATENT dump + the qwenimage PiD checkpoint"]
fn tiled_vs_whole_decode() {
    let cap = env_or("PID_AB_CAPTURE", "/tmp/pid_ab.safetensors".to_string());
    let ckpt = env_or(
        "PID_QWEN_SAFETENSORS",
        format!(
            "{}/.cache/huggingface/hub/models--SceneWorks--pid-qwenimage/snapshots",
            std::env::var("HOME").unwrap()
        ),
    );
    let tile: i32 = env_or("PID_AB_TILE", "2048".into()).parse().unwrap();
    let overlap: i32 = env_or("PID_AB_OVERLAP", "256".into()).parse().unwrap();
    let out_dir = env_or("PID_AB_OUT", "/tmp".into());

    // --- captured real latent + caption ---
    let capw = Weights::from_file(&cap).expect("capture safetensors (run the krea capture first)");
    let latent = capw.require("latent").unwrap().clone();
    let caption = capw
        .require("caption")
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    eprintln!(
        "capture: latent {:?} caption {:?}",
        latent.shape(),
        caption.shape()
    );

    // --- PiD net ---
    let w = Weights::from_file(&ckpt).expect("pid checkpoint");
    let cfg = PidConfig::sr4x();

    // One decoder → identical seed/noise for both paths; tiling is the only variable. σ=0 (clean
    // decode), scale 4, vae_compression 8 — the released qwenimage student geometry.
    let mk = || {
        PidDecoder::new(
            PidNet::from_weights(&w, "", &cfg).unwrap(),
            Sampler::new(&SamplerConfig::distill_4step()),
            caption.clone(),
            0.0,
            4,
            8,
            1234,
        )
    };
    let decoder = mk();
    let (th, tw) = decoder.target_hw(&latent);
    eprintln!(
        "decoding {:?} -> {th}x{tw}  (tile={tile} overlap={overlap})",
        latent.shape()
    );

    // --- whole-image reference ---
    let t = std::time::Instant::now();
    let whole = decoder.decode(&latent).unwrap();
    eval([&whole]).unwrap();
    let whole_dt = t.elapsed().as_secs_f32();
    eprintln!("whole-image decode: {:?} in {whole_dt:.1}s", whole.shape());

    // --- tiled ---
    let t = std::time::Instant::now();
    let tiled = decoder.decode_tiled(&latent, tile, overlap).unwrap();
    eval([&tiled]).unwrap();
    let tiled_dt = t.elapsed().as_secs_f32();
    eprintln!("tiled decode:       {:?} in {tiled_dt:.1}s", tiled.shape());

    assert_eq!(whole.shape(), tiled.shape(), "same geometry");

    // --- metrics over [-1,1] (peak-to-peak 2.0) ---
    let diff = subtract(&whole, &tiled).unwrap();
    let dabs = abs(&diff).unwrap();
    let max_abs = scalar_of(&max(&dabs, None).unwrap());
    let mean_abs = scalar_of(&mean(&dabs, None).unwrap());
    let mse = scalar_of(&mean(multiply(&diff, &diff).unwrap(), None).unwrap());
    let rmse = mse.sqrt();
    let psnr = if rmse > 0.0 {
        20.0 * (2.0f32 / rmse).log10()
    } else {
        f32::INFINITY
    };
    eprintln!(
        "A/B (whole vs tiled): max|Δ|={max_abs:.4}  mean|Δ|={mean_abs:.4}  RMSE={rmse:.4}  PSNR={psnr:.2} dB  (range [-1,1])"
    );

    save_png(&whole, &format!("{out_dir}/pid_ab_whole_{th}.png"));
    save_png(&tiled, &format!("{out_dir}/pid_ab_tiled_{th}.png"));
    save_diff_heatmap(&dabs, &format!("{out_dir}/pid_ab_diff_{th}.png"), 5.0);
    eprintln!(
        "wrote {out_dir}/pid_ab_whole_{th}.png + pid_ab_tiled_{th}.png + pid_ab_diff_{th}.png (diff ×5)"
    );

    assert!(max_abs.is_finite(), "non-finite diff");
}
