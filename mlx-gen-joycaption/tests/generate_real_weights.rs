//! Ignored real-weights smoke for JoyCaption.
//!
//! Run with a cached `fancyfeast/llama-joycaption-beta-one-hf-llava` snapshot:
//!   cargo test -p mlx-gen-joycaption --test generate_real_weights -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::{CaptionRequest, CaptionSampling, LoadSpec, WeightsSource};
use mlx_gen_joycaption as _;

#[test]
#[ignore = "needs the JoyCaption HF snapshot; set MLX_GEN_JOYCAPTION_SNAPSHOT"]
fn joycaption_generates_short_caption_from_real_weights() {
    let root = PathBuf::from(
        std::env::var("MLX_GEN_JOYCAPTION_SNAPSHOT")
            .expect("set MLX_GEN_JOYCAPTION_SNAPSHOT to a JoyCaption snapshot directory"),
    );
    let id = mlx_gen::caption::joycaption::JOY_CAPTION_MODEL_ID;
    let model = mlx_gen::load_captioner(id, &LoadSpec::new(WeightsSource::Dir(root))).unwrap();
    let req = CaptionRequest {
        image: Image {
            width: 384,
            height: 384,
            pixels: vec![127; 384 * 384 * 3],
        },
        prompt: "Write a very short caption.".to_owned(),
        sampling: CaptionSampling {
            temperature: 0.0,
            top_p: 1.0,
            max_new_tokens: 8,
            seed: None,
        },
        ..Default::default()
    };
    let out = model.caption(&req, &mut |_| {}).unwrap();
    assert!(out.generated_tokens.unwrap_or(0) <= 8);
    assert!(!out.text.trim().is_empty(), "caption should produce text");
}
