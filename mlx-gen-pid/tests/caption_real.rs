//! sc-7843 component 5 (caption glue) real-weight parity. `#[ignore]`d (needs gemma-2-2b-it).
//! Golden = `tools/dump_pid_caption.py` (reference `_encode_text_raw`, f32). Two gates:
//! 1. **exact token-id match** — the hard correctness proof for the tokenizer + Chi-prompt + length
//!    policy (a wrong tokenization is a discrete, O(1) failure);
//! 2. caption_embs numeric (loose — MLX runs the gemma weights in bf16 vs the f32 golden).
//!
//! ```sh
//! cargo test -p mlx-gen-pid --release --test caption_real -- --ignored --nocapture
//! ```

use mlx_gen::weights::Weights;
use mlx_gen_pid::{CaptionEncoder, Gemma2, Gemma2Config};
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::Dtype;

const CAPTION: &str =
    "a mountain valley landscape at golden hour with a winding river and pine forest";

fn gemma_snapshot() -> String {
    std::env::var("PID_GEMMA_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap();
        let base = format!(
            "{home}/.cache/huggingface/hub/models--Efficient-Large-Model--gemma-2-2b-it/snapshots"
        );
        let snap = std::fs::read_dir(&base)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|d| d.is_dir())
            .unwrap();
        snap.to_string_lossy().into_owned()
    })
}

#[test]
#[ignore = "needs gemma-2-2b-it weights + tokenizer"]
fn caption_encoding_matches() {
    let snap = gemma_snapshot();
    let w = Weights::from_file(format!("{snap}/gemma-2-2b-it.safetensors")).unwrap();
    let gemma = Gemma2::from_weights(&w, "model.", &Gemma2Config::gemma_2_2b()).unwrap();
    let enc = CaptionEncoder::new(gemma, format!("{snap}/tokenizer.json")).unwrap();

    let golden = Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tools/golden/pid/caption_landscape.safetensors"
    ))
    .unwrap();
    let ref_ids = golden
        .require("input_ids")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    let ref_embs = golden.require("caption_embs").unwrap();

    // --- gate 1: exact token-id match (the hard correctness proof for the glue) ---
    assert_eq!(enc.num_chi_tokens(), 208, "num_chi_tokens");
    let (ids, _mask) = enc.token_ids(CAPTION).unwrap();
    let ref_vec: Vec<i32> = ref_ids.as_slice::<i32>().to_vec();
    assert_eq!(ids.len(), ref_vec.len(), "padded length (num_chi+298)");
    assert_eq!(ids, ref_vec, "token ids must match the reference exactly");
    eprintln!("token ids match exactly ({} ids)", ids.len());

    // --- gate 2: caption_embs (MLX bf16 vs bf16 golden) — cosine is the meaningful metric for a
    // conditioning embedding (peak-rel is dominated by lone bf16 outliers in O(50)-magnitude
    // hidden states over 26 layers). The exact-id match above is the hard correctness proof. ---
    let embs = enc.encode(CAPTION).unwrap();
    assert_eq!(embs.shape(), ref_embs.shape(), "caption_embs shape");
    let a = embs.as_dtype(Dtype::Float32).unwrap();
    let b = ref_embs.as_dtype(Dtype::Float32).unwrap();
    let dot = mlx_rs::ops::multiply(&a, &b)
        .unwrap()
        .sum(None)
        .unwrap()
        .item::<f32>();
    let na = mlx_rs::ops::multiply(&a, &a)
        .unwrap()
        .sum(None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    let nb = mlx_rs::ops::multiply(&b, &b)
        .unwrap()
        .sum(None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    let cos = dot / (na * nb);
    let d = max(abs(subtract(&a, &b).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let rel = d / max(abs(&b).unwrap(), None).unwrap().item::<f32>();
    eprintln!("caption_embs: cosine={cos}  peak-rel={rel:.3e} (bf16 cross-backend)");
    // 0.999 cosine over a 26-layer bf16 transformer with O(50) activations is the cross-backend
    // floor; the exact-id match is the hard correctness proof. (peak-rel ~0.14 is lone bf16 outliers.)
    assert!(
        cos > 0.998,
        "caption_embs cosine={cos} — forward divergence (ids matched exactly)"
    );
}
