//! sc-2837: `trim_first_frames` — generate `trim` extra leading temporal chunks, then discard them
//! after decode (port of `generate_wan.py`). Two gates:
//!  - CI: the **geometry invariant** — trimming preserves the final output frame count (the extra
//!    `trim·4` decoded frames are exactly the ones discarded), via the public `latent_shape` math.
//!  - `#[ignore]`: the live `wan2_2_t2v_14b` path with the real A14B checkpoint — trim runs, yields
//!    the same frame count as no-trim, and stays coherent.

use std::path::PathBuf;

use mlx_gen::{registry, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_wan::pipeline::latent_shape;
use mlx_gen_wan::MODEL_ID_T2V_14B;

/// The Wan z16 geometry: non-causal decode maps `t_lat` latent frames → `t_lat · vae_stride_t`
/// output frames. With `trim` extra leading chunks, `t_lat` grows by `trim` and the decode adds
/// `trim · vae_stride_t` frames — exactly the count discarded — so the kept length is invariant.
#[test]
fn trim_preserves_output_frame_count() {
    let z = 16;
    let stride = (4usize, 8usize, 8usize); // Wan z16
    let stride_t = stride.0 as i32;

    for frames in [5usize, 49, 81] {
        let base_out = latent_shape(frames, 256, 256, z, stride).unwrap()[1] * stride_t;
        for trim in [0usize, 1, 2, 4] {
            let gen_frames = frames + trim * stride.0; // requested + trim·4
            let gen_out = latent_shape(gen_frames, 256, 256, z, stride).unwrap()[1] * stride_t;
            let kept = gen_out - (trim as i32) * stride_t; // drop the leading trim·4 output frames
            assert_eq!(
                kept, base_out,
                "frames={frames} trim={trim}: kept {kept} != no-trim {base_out}"
            );
            // gen_frames stays a valid 1+4k count (frames is, and trim·4 is a multiple of 4).
            assert_eq!(gen_frames % 4, frames % 4, "gen_frames must stay 1+4k");
        }
    }
}

fn env_dir(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(|s| {
        let s = s.to_string_lossy();
        s.strip_prefix("~/").map_or_else(
            || PathBuf::from(s.to_string()),
            |rest| PathBuf::from(format!("{}/{rest}", std::env::var("HOME").unwrap())),
        )
    })
}

fn gen_frame_count(model_dir: &std::path::Path, trim: Option<u32>) -> usize {
    let g = registry::load(
        MODEL_ID_T2V_14B,
        &LoadSpec::new(WeightsSource::Dir(model_dir.to_path_buf())),
    )
    .expect("load wan2_2_t2v_14b");
    let req = GenerationRequest {
        prompt: "a red fox trotting across a snowy meadow at sunrise, cinematic".into(),
        width: 128,
        height: 128,
        frames: Some(5),
        steps: Some(4),
        seed: Some(7),
        sampler: Some("unipc".into()),
        trim_first_frames: trim,
        ..Default::default()
    };
    let mut noop = |_p| {};
    match g.generate(&req, &mut noop).expect("generate") {
        GenerationOutput::Video { frames, .. } => {
            // Coherence: no flat frames (min==max → decode bug).
            for img in &frames {
                let min = *img.pixels.iter().min().unwrap();
                let max = *img.pixels.iter().max().unwrap();
                assert!(max > min, "trim={trim:?}: flat frame");
            }
            frames.len()
        }
        _ => panic!("expected Video"),
    }
}

/// Real-weight: `trim_first_frames` runs end-to-end and preserves the output frame count (the extra
/// leading chunk is generated then discarded). `#[ignore]` — needs the converted A14B checkpoint.
///
/// ```text
/// WAN_A14B_MODEL_DIR=~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16 \
///   cargo test -p mlx-gen-wan --test trim -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs the converted Wan2.2-T2V-A14B checkpoint (WAN_A14B_MODEL_DIR)"]
fn wan_trim_first_frames_runs_and_preserves_count() {
    let dir = match env_dir("WAN_A14B_MODEL_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skip: set WAN_A14B_MODEL_DIR");
            return;
        }
    };
    let n0 = gen_frame_count(&dir, None);
    let n1 = gen_frame_count(&dir, Some(1));
    println!("[trim] frames: trim=0 → {n0}, trim=1 → {n1}");
    assert_eq!(
        n0, n1,
        "trim_first_frames must preserve the output frame count"
    );
    assert!(n0 > 0, "no frames produced");
}
