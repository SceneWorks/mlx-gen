//! SANA text-conditioning reuse tests (story sc-8488).
//!
//! SANA's text conditioning REUSES PiD's gemma-2-2b-it CHI caption encoder
//! ([`mlx_gen_pid::CaptionEncoder`]); the only divergence is the CHI prompt text. These tests pin:
//!
//!  * [`sana_chi_prompt_is_pid_chi_with_single_quotes`] — DEFAULT (no weights). The hard contract on
//!    the CHI template: SANA's `SANA_CHI_PROMPT` equals PiD's `CHI_PROMPT` in every character except
//!    the quoting around `Enhanced prompt` (single- vs double-quote), and equals the joined diffusers
//!    `complex_human_instruction` list. A wrong CHI string is a discrete, O(1) failure that silently
//!    wrecks conditioning, so it is asserted exactly.
//!
//!  * [`caption_encoding_matches_reference`] — `#[ignore]`d (needs gemma-2-2b-it weights + the SANA
//!    golden from `tools/dump_sana_caption.py`). Two gates mirroring `mlx-gen-pid`'s `caption_real`:
//!    exact token-id match (the tokenizer + CHI-prompt + length-policy correctness proof) and a
//!    cosine gate on the `[1, 300, 2304]` caption embedding vs the diffusers reference.
//!
//!  * [`caption_feeds_trunk_cross_attn`] — `#[ignore]`d (same weights, plus the real SANA transformer
//!    via `SANA_TRANSFORMER_WEIGHTS`). The byte/shape-compatibility proof: the TE's `[1, 300, 2304]`
//!    output is consumed by [`SanaTransformer::forward`]'s `caption` argument and produces a finite
//!    `[1, 32, H, W]` noise prediction.
//!
//! ```sh
//! cargo test -p mlx-gen-sana --test text_encoder            # default (CHI contract)
//! PID_GEMMA_DIR=/path/to/gemma-2-2b-it \
//!   cargo test -p mlx-gen-sana --release --test text_encoder -- --ignored --nocapture
//! ```

use mlx_gen_sana::{SanaTextEncoder, SANA_CHI_PROMPT};

const CAPTION: &str =
    "a mountain valley landscape at golden hour with a winding river and pine forest";

/// SANA's exact `complex_human_instruction` list (diffusers `pipeline_sana.py` / NVlabs Sana),
/// joined by `"\n"` — the reference definition `SANA_CHI_PROMPT` must equal.
fn sana_chi_joined() -> String {
    [
        "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        "Here are examples of how to transform or refine prompts:",
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
        "User Prompt: ",
    ]
    .join("\n")
}

#[test]
fn sana_chi_prompt_is_pid_chi_with_single_quotes() {
    // 1. SANA_CHI_PROMPT is exactly the joined diffusers complex_human_instruction list.
    assert_eq!(
        SANA_CHI_PROMPT,
        sana_chi_joined(),
        "SANA_CHI_PROMPT must equal `\"\\n\".join(complex_human_instruction)`"
    );

    // 2. It differs from PiD's CHI in EXACTLY the quoting around `Enhanced prompt` — the load-bearing
    //    divergence that forced parameterizing rather than reusing PiD's text. Both contain the
    //    single-quote form (resp. double-quote), are the same length, and are otherwise identical.
    let pid = mlx_gen_pid::caption::CHI_PROMPT;
    assert_eq!(
        SANA_CHI_PROMPT.len(),
        pid.len(),
        "same length — divergence is only the quote glyph, not the wording"
    );
    assert!(SANA_CHI_PROMPT.contains("an 'Enhanced prompt'"));
    assert!(pid.contains("an \"Enhanced prompt\""));
    assert_ne!(
        SANA_CHI_PROMPT, pid,
        "SANA and PiD CHI prompts must differ (quote style) — do not reuse PiD's text"
    );
    // Replacing SANA's single-quotes with PiD's double-quotes recovers PiD's string exactly: proves
    // the quote glyph is the SOLE difference.
    assert_eq!(
        SANA_CHI_PROMPT.replacen("an 'Enhanced prompt'", "an \"Enhanced prompt\"", 1),
        pid,
        "the only difference between the SANA and PiD CHI prompts is the Enhanced-prompt quoting"
    );

    // Both end with the trailing "User Prompt: " the caption is appended after.
    assert!(SANA_CHI_PROMPT.ends_with("User Prompt: "));
}

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
#[ignore = "needs gemma-2-2b-it weights + tools/golden/sana/caption_landscape.safetensors"]
fn caption_encoding_matches_reference() {
    use mlx_gen::weights::Weights;
    use mlx_rs::ops::{abs, max, multiply, subtract};
    use mlx_rs::Dtype;

    let snap = gemma_snapshot();
    let enc = SanaTextEncoder::from_snapshot(&snap).unwrap();

    let golden = Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tools/golden/sana/caption_landscape.safetensors"
    ))
    .unwrap();
    let ref_ids = golden
        .require("input_ids")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap();
    let ref_embs = golden.require("caption_embs").unwrap();

    // --- gate 1: exact token-id match (the hard correctness proof for tokenizer + CHI + length) ---
    let (ids, _mask) = enc.token_ids(CAPTION).unwrap();
    let ref_vec: Vec<i32> = ref_ids.as_slice::<i32>().to_vec();
    assert_eq!(ids.len(), ref_vec.len(), "padded length (num_chi + 298)");
    assert_eq!(
        ids, ref_vec,
        "token ids must match the SANA reference exactly"
    );
    eprintln!(
        "token ids match exactly ({} ids, num_chi_tokens={})",
        ids.len(),
        enc.num_chi_tokens()
    );

    // --- gate 2: caption_embs cosine (MLX bf16 vs bf16 golden; see mlx-gen-pid caption_real) ---
    let embs = enc.encode(CAPTION).unwrap();
    assert_eq!(
        embs.shape(),
        &[1, 300, 2304],
        "caption_embs shape [1,300,2304]"
    );
    assert_eq!(embs.shape(), ref_embs.shape(), "caption_embs shape vs ref");
    let a = embs.as_dtype(Dtype::Float32).unwrap();
    let b = ref_embs.as_dtype(Dtype::Float32).unwrap();
    let dot = multiply(&a, &b).unwrap().sum(None).unwrap().item::<f32>();
    let na = multiply(&a, &a)
        .unwrap()
        .sum(None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    let nb = multiply(&b, &b)
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
    assert!(
        cos > 0.998,
        "caption_embs cosine={cos} — forward divergence (ids matched exactly)"
    );
}

#[test]
#[ignore = "needs gemma-2-2b-it weights + SANA_TRANSFORMER_WEIGHTS (real Sana_1600M trunk)"]
fn caption_feeds_trunk_cross_attn() {
    use mlx_gen::weights::Weights;
    use mlx_gen_sana::{SanaTransformer, SanaTransformerConfig};
    use mlx_rs::ops::is_nan;
    use mlx_rs::Array;

    // TE → [1, 300, 2304] caption embedding.
    let snap = gemma_snapshot();
    let enc = SanaTextEncoder::from_snapshot(&snap).unwrap();
    let caption = enc.encode(CAPTION).unwrap();
    assert_eq!(
        caption.shape(),
        &[1, 300, 2304],
        "TE output must be [1, 300, 2304] (SANA caption_channels)"
    );

    // Real SANA-1.6B trunk; its cross-attn caption_channels == 2304 must accept the TE output.
    let weights_dir =
        std::env::var("SANA_TRANSFORMER_WEIGHTS").expect("set SANA_TRANSFORMER_WEIGHTS");
    let weights = Weights::from_dir(&weights_dir).expect("load real trunk weights");
    let trunk = SanaTransformer::from_weights(&weights, SanaTransformerConfig::sana_1600m())
        .expect("build trunk");

    // A small DC-AE f32c32 latent grid (8×8 = 256px @ 32× compression) is enough to exercise cross-attn.
    let latent = Array::from_slice(&vec![0f32; 32 * 8 * 8], &[1, 32, 8, 8]);
    let timestep = Array::from_slice(&[500.0f32], &[1]);
    let out = trunk
        .forward(&latent, &caption, &timestep)
        .expect("trunk must accept the TE caption embedding");

    assert_eq!(out.shape(), &[1, 32, 8, 8], "noise prediction shape");
    let any_nan = is_nan(&out).unwrap().sum(None).unwrap().item::<f32>();
    assert_eq!(any_nan, 0.0, "noise prediction must be finite");
    eprintln!(
        "TE [1,300,2304] accepted by trunk cross-attn → {:?}",
        out.shape()
    );
}
