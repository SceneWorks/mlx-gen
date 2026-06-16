//! Shared test fixtures for the Ideogram 4 tests.

/// The canonical JSON-caption prompt — the model's native format. Byte-identical to
/// `json.dumps(CAPTION)` in `tools/dump_ideogram4_prompt_ids.py`, so native tokenization of this
/// string reproduces `tools/golden/ideogram4_prompt_ids.safetensors` (the `tokenizer_parity` test)
/// and the `smoke` test generates the same validated red-fox image. Keep the two in lockstep — if
/// you change the caption, re-run the dumper to refresh the golden.
pub const CAPTION_JSON: &str = r#"{"high_level_description": "A photograph of a red fox sitting in a snowy forest at golden hour.", "style_description": {"aesthetics": "serene, warm, naturalistic", "lighting": "golden hour, soft warm backlight, long shadows", "photo": "telephoto, shallow depth of field, sharp focus, eye-level", "medium": "photograph"}, "compositional_deconstruction": {"background": "A snowy forest of tall pine trees, soft golden sunlight filtering through the branches, snow on the ground.", "elements": [{"type": "obj", "bbox": [250, 320, 950, 760], "desc": "A red fox with vivid orange fur, white chest and a thick bushy tail, sitting upright in the snow and facing the camera."}]}}"#;
