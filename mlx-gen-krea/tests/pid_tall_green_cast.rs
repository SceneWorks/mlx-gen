//! A/B repro for the PiD tall-image bottom green-cast (SceneWorks bug): render the exact baseline
//! recipe (Krea 2 Turbo Q4, 720×1280 portrait, seed 333941069) with the PiD 4× decoder, which
//! super-resolves to 2880×5120 — the height (5120) exceeds the 2kto4k training extent (3840), driving
//! the absolute pixel positional embedding out of distribution in the bottom rows.
//!
//! Renders twice: `PID_PIXEL_POS_ABS=1` (raw-absolute, the pre-fix reference behavior) and the default
//! (aspect-preserving positional interpolation into the trained extent). A per-band greenness metric
//! `mean(G − (R+B)/2)` over the top vs bottom strip quantifies the cast; the fix should collapse the
//! bottom-vs-top excess. Both PNGs are written for visual inspection.
//!
//! ```sh
//! cargo test -p mlx-gen-krea --release --test pid_tall_green_cast -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
use mlx_gen_krea::load;

fn cache_root() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/huggingface/hub")
}

/// First (only) snapshot dir under a `models--…` cache entry.
fn snapshot(model_dir: &str) -> PathBuf {
    let base = cache_root().join(model_dir).join("snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("missing HF cache snapshots: {}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .unwrap_or_else(|| panic!("no snapshot dir under {}", base.display()))
}

/// `mean(G − (R+B)/2)` over the row band `[r0, r1)` — positive ⇒ greenish, ~0 ⇒ neutral.
fn band_greenness(img: &Image, r0: u32, r1: u32) -> f64 {
    let (w, _h) = (img.width, img.height);
    let mut sum = 0f64;
    let mut n = 0u64;
    for row in r0..r1 {
        for col in 0..w {
            let i = ((row * w + col) * 3) as usize;
            let r = img.pixels[i] as f64;
            let g = img.pixels[i + 1] as f64;
            let b = img.pixels[i + 2] as f64;
            sum += g - 0.5 * (r + b);
            n += 1;
        }
    }
    sum / n.max(1) as f64
}

fn report(tag: &str, img: &Image) -> (f64, f64) {
    let h = img.height;
    let band = (h / 10).max(1); // top / bottom 10%
    let top = band_greenness(img, 0, band);
    let bottom = band_greenness(img, h - band, h);
    eprintln!(
        "[{tag}] {}x{}  top-greenness={top:+.2}  bottom-greenness={bottom:+.2}  bottom−top={:+.2}",
        img.width,
        img.height,
        bottom - top
    );
    (top, bottom)
}

fn save_png(img: &Image, path: &str) {
    image::save_buffer(
        path,
        &img.pixels,
        img.width,
        img.height,
        image::ColorType::Rgb8,
    )
    .unwrap();
    eprintln!("wrote {path}");
}

fn render_pid() -> Image {
    let spec = LoadSpec::new(WeightsSource::Dir(
        snapshot("models--SceneWorks--krea-2-turbo-mlx").join("q4"),
    ))
    .with_pid(
        WeightsSource::File(
            snapshot("models--SceneWorks--pid-qwenimage").join("pid_qwenimage_2kto4k.safetensors"),
        ),
        WeightsSource::Dir(snapshot("models--SceneWorks--gemma-2-2b-it")),
    );

    let t = Instant::now();
    let model = load(&spec).expect("load Krea 2 Turbo Q4 + PiD");
    eprintln!("loaded in {:.1}s", t.elapsed().as_secs_f32());

    let req = GenerationRequest {
        prompt:
            "An old barn full of rectangular bails of hay. The floor is weathered wooden slats \
                 with loose hay laying around. Golden hour light streams in."
                .into(),
        width: 720,
        height: 1280,
        count: 1,
        seed: Some(333_941_069),
        steps: Some(8),
        use_pid: true,
        ..Default::default()
    };
    let t = Instant::now();
    let img = match model.generate(&req, &mut |_| {}).expect("pid generate") {
        GenerationOutput::Images(v) => v.into_iter().next().unwrap(),
        _ => panic!("expected images"),
    };
    eprintln!("rendered in {:.1}s", t.elapsed().as_secs_f32());
    img
}

/// Render the raw-absolute (pre-fix) arm by forcing `PID_PIXEL_POS_ABS=1`, then the fixed arm.
/// Success criterion: the fix brings the bottom band's greenness excess (`bottom − top`) down to a
/// small fraction of the absolute arm's.
#[test]
#[ignore = "needs Krea Turbo Q4 + pid-qwenimage + gemma-2-2b-it (GPU, ~minutes)"]
fn krea_720x1280_pid_bottom_green_cast() {
    let out = concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/golden/pid");
    let _ = std::fs::create_dir_all(out);

    std::env::set_var("PID_PIXEL_POS_ABS", "1");
    let abs = render_pid();
    let (abs_top, abs_bottom) = report("absolute (pre-fix)", &abs);
    save_png(&abs, &format!("{out}/krea_720x1280_pid_absolute.png"));
    let abs_excess = abs_bottom - abs_top;

    std::env::remove_var("PID_PIXEL_POS_ABS");
    let fixed = render_pid();
    let (fixed_top, fixed_bottom) = report("fixed (pos-interp)", &fixed);
    save_png(&fixed, &format!("{out}/krea_720x1280_pid_fixed.png"));
    let fixed_excess = fixed_bottom - fixed_top;

    eprintln!(
        "\n=== A/B: bottom−top greenness  absolute={abs_excess:+.2}  fixed={fixed_excess:+.2} ==="
    );
    assert!(
        fixed_excess.abs() < abs_excess.abs() * 0.5,
        "fix should at least halve the bottom green-cast excess (absolute={abs_excess:.2}, fixed={fixed_excess:.2})"
    );
}
