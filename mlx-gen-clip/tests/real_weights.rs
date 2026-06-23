//! Real-weights test (sc-6535): loads the cached `openai/clip-vit-large-patch14` snapshot and checks
//! the CLIP image embedder produces a sane 768-d vector with the right cosine geometry. `#[ignore]`d
//! (multi-GB weights), per the mlx-gen convention. Run with:
//!
//! ```sh
//! CLIP_VIT_L_SNAPSHOT=/path/to/clip-vit-large-patch14 \
//!   cargo test -p mlx-gen-clip --test real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::gen_core::runtime::{LoadSpec, WeightsSource};
use mlx_gen::media::Image;
use mlx_gen_clip::{load, load_text};

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("CLIP_VIT_L_SNAPSHOT")
            .expect("set CLIP_VIT_L_SNAPSHOT to the openai/clip-vit-large-patch14 snapshot dir"),
    )
}

/// A uniform-colour image. Center-crop→resize to 224² makes it size-invariant, so two solids of the
/// same colour preprocess byte-identically → identical embedding (a clean determinism check).
fn solid(w: u32, h: u32, rgb: [u8; 3]) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h) {
        pixels.extend_from_slice(&rgb);
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
#[ignore = "loads openai/clip-vit-large-patch14 (~1.7GB); set CLIP_VIT_L_SNAPSHOT"]
fn embeds_real_images_with_sane_cosine_geometry() {
    let embedder =
        load(&LoadSpec::new(WeightsSource::Dir(snapshot()))).expect("load clip embedder");
    assert_eq!(embedder.descriptor().embedding_dim, 768);

    let red = embedder.embed(&solid(64, 64, [220, 30, 30])).unwrap();
    let red_big = embedder.embed(&solid(96, 96, [220, 30, 30])).unwrap();
    let blue = embedder.embed(&solid(64, 64, [30, 30, 220])).unwrap();

    // Right dimensionality + a non-degenerate vector.
    assert_eq!(red.len(), 768, "CLIP ViT-L/14 embedding is 768-d");
    assert!(red.iter().any(|&x| x != 0.0), "embedding is not all-zero");

    // Determinism / size-invariance: the same colour at two sizes → identical embedding.
    let self_cos = cosine(&red, &red_big);
    assert!(
        self_cos > 0.999,
        "same colour at two sizes should match (cos={self_cos})"
    );

    // Colour sensitivity: a different colour is measurably less similar than an identical image.
    let cross_cos = cosine(&red, &blue);
    assert!(
        cross_cos < self_cos,
        "red·blue ({cross_cos}) should be < red·red ({self_cos})"
    );
    println!(
        "clip ok: dim={}, red·red={self_cos:.5}, red·blue={cross_cos:.5}",
        red.len()
    );
}

#[test]
#[ignore = "loads openai/clip-vit-large-patch14 (~1.7GB); set CLIP_VIT_L_SNAPSHOT"]
fn text_and_image_embeds_rank_matching_colours_higher() {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    let image_embedder = load(&spec).expect("load clip image embedder");
    let text_embedder = load_text(&spec).expect("load clip text embedder");

    let red = image_embedder.embed(&solid(64, 64, [220, 30, 30])).unwrap();
    let blue = image_embedder.embed(&solid(64, 64, [30, 30, 220])).unwrap();
    let red_text = text_embedder.embed_text("a red square").unwrap();
    let blue_text = text_embedder.embed_text("a blue square").unwrap();

    assert_eq!(red_text.len(), 768);
    assert_eq!(blue_text.len(), 768);
    assert!(red_text.iter().all(|v| v.is_finite()) && red_text.iter().any(|&v| v != 0.0));
    assert!(blue_text.iter().all(|v| v.is_finite()) && blue_text.iter().any(|&v| v != 0.0));

    let red_text_red = cosine(&red_text, &red);
    let red_text_blue = cosine(&red_text, &blue);
    let blue_text_blue = cosine(&blue_text, &blue);
    let blue_text_red = cosine(&blue_text, &red);

    assert!(
        red_text_red > red_text_blue,
        "red text should rank red image higher ({red_text_red:.5} <= {red_text_blue:.5})"
    );
    assert!(
        blue_text_blue > blue_text_red,
        "blue text should rank blue image higher ({blue_text_blue:.5} <= {blue_text_red:.5})"
    );
    println!(
        "clip text/image ok: red_text·red={red_text_red:.5}, red_text·blue={red_text_blue:.5}, \
         blue_text·blue={blue_text_blue:.5}, blue_text·red={blue_text_red:.5}"
    );
}
