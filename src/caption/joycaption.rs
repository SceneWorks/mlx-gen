//! JoyCaption prompt, chat-template, and tokenizer helpers.
//!
//! This module ports the request-facing text path before the model provider exists: SceneWorks'
//! caption prompt map, the model's LLaVA/Llama chat wrapper, and tokenizer loading from a cached
//! Hugging Face snapshot. Vision preprocessing, image-token embedding splice, and generation live
//! in later epic slices.

use std::path::Path;

use crate::caption::{CaptionCapabilities, CaptionOptions};
use crate::runtime::{LoadSpec, WeightsSource};
use crate::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use crate::{Error, Result};

pub mod vision;

pub const JOY_CAPTION_MODEL_ID: &str = "fancyfeast/llama-joycaption-beta-one-hf-llava";
pub const JOY_CAPTION_FAMILY: &str = "joycaption";

pub const SYSTEM_PROMPT: &str = "You are a helpful image captioner.";
pub const DEFAULT_DATE_STRING: &str = "26 July 2024";
pub const CUTTING_KNOWLEDGE_DATE: &str = "December 2023";

pub const BEGIN_OF_TEXT_TOKEN_ID: i32 = 128000;
pub const END_OF_TEXT_TOKEN_ID: i32 = 128001;
pub const START_HEADER_TOKEN_ID: i32 = 128006;
pub const END_HEADER_TOKEN_ID: i32 = 128007;
pub const EOM_TOKEN_ID: i32 = 128008;
pub const EOT_TOKEN_ID: i32 = 128009;
pub const PAD_TOKEN_ID: i32 = 128004;
pub const IMAGE_TOKEN_ID: i32 = 128077;
pub const IMAGE_SEQ_LENGTH: usize = 729;
pub const DEFAULT_MAX_CONTEXT_TOKENS: usize = 4096;

pub const IMAGE_TOKEN: &str = "<|reserved_special_token_69|>";
pub const IMAGE_START_TOKEN: &str = "<|reserved_special_token_70|>";
pub const IMAGE_END_TOKEN: &str = "<|reserved_special_token_71|>";

pub const JOY_NAME_OPTION: &str =
    "If there is a person/character in the image you must refer to them as {name}.";

pub const CAPTION_TYPES: &[&str] = &[
    "Descriptive",
    "Descriptive (Casual)",
    "Straightforward",
    "Stable Diffusion Prompt",
    "MidJourney",
    "Danbooru tag list",
    "e621 tag list",
    "Rule34 tag list",
    "Booru-like tag list",
    "Art Critic",
    "Product Listing",
    "Social Media Post",
];

pub const CAPTION_LENGTHS: &[&str] = &["any", "very short", "short", "medium-length", "long"];

const PROMPT_TEMPLATES: &[(&str, [&str; 3])] = &[
    (
        "Descriptive",
        [
            "Write a detailed description for this image.",
            "Write a detailed description for this image in {word_count} words or less.",
            "Write a {length} detailed description for this image.",
        ],
    ),
    (
        "Descriptive (Casual)",
        [
            "Write a descriptive caption for this image in a casual tone.",
            "Write a descriptive caption for this image in a casual tone within {word_count} words.",
            "Write a {length} descriptive caption for this image in a casual tone.",
        ],
    ),
    (
        "Straightforward",
        [
            "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is...\" or similar phrasing.",
            "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is...\" or similar phrasing.",
            "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements-people, objects, scenery-using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Never mention what is absent, resolution, watermarks, signatures, compression artifacts, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is...\" or similar phrasing.",
        ],
    ),
    (
        "Stable Diffusion Prompt",
        [
            "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
            "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
            "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        ],
    ),
    (
        "MidJourney",
        [
            "Write a MidJourney prompt for this image.",
            "Write a MidJourney prompt for this image within {word_count} words.",
            "Write a {length} MidJourney prompt for this image.",
        ],
    ),
    (
        "Danbooru tag list",
        [
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
        ],
    ),
    (
        "e621 tag list",
        [
            "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags.",
            "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags. Keep it under {word_count} words.",
            "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags, if any, prefixed by artist:, copyright:, character:, species:, meta:, and lore:. Then all the general tags.",
        ],
    ),
    (
        "Rule34 tag list",
        [
            "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags.",
            "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags. Keep it under {word_count} words.",
            "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags, if any, prefixed by artist:, copyright:, character:, and meta:. Then all the general tags.",
        ],
    ),
    (
        "Booru-like tag list",
        [
            "Write a list of Booru-like tags for this image.",
            "Write a list of Booru-like tags for this image within {word_count} words.",
            "Write a {length} list of Booru-like tags for this image.",
        ],
    ),
    (
        "Art Critic",
        [
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
        ],
    ),
    (
        "Product Listing",
        [
            "Write a caption for this image as though it were a product listing.",
            "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
            "Write a {length} caption for this image as though it were a product listing.",
        ],
    ),
    (
        "Social Media Post",
        [
            "Write a caption for this image as if it were being used for a social media post.",
            "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
            "Write a {length} caption for this image as if it were being used for a social media post.",
        ],
    ),
];

pub fn capabilities() -> CaptionCapabilities {
    CaptionCapabilities {
        caption_types: CAPTION_TYPES.to_vec(),
        caption_lengths: CAPTION_LENGTHS.to_vec(),
        supports_custom_prompt: true,
        supports_low_vram: true,
        min_image_size: 1,
        max_image_size: 8192,
        max_prompt_chars: 4000,
        max_name_chars: 120,
        max_extra_options: 16,
        max_extra_option_chars: 500,
        max_trigger_words: 32,
        max_trigger_word_chars: 120,
        max_new_tokens: 1024,
        mac_only: true,
    }
}

pub fn build_prompt(options: &CaptionOptions) -> String {
    let custom = options.custom_prompt.trim();
    if !custom.is_empty() {
        return custom.to_owned();
    }

    let caption_length = options.caption_length.as_str();
    let template_index = if caption_length == "any" {
        0
    } else if caption_length.chars().all(|c| c.is_ascii_digit()) {
        1
    } else {
        2
    };
    let mut prompt = templates_for(&options.caption_type)[template_index].to_owned();
    if !options.extra_options.is_empty() {
        prompt.push(' ');
        prompt.push_str(&options.extra_options.join(" "));
    }
    prompt
        .replace("{name}", name_or_placeholder(options))
        .replace("{length}", caption_length)
        .replace("{word_count}", caption_length)
}

pub fn apply_trigger_words(caption: &str, trigger_words: &[String]) -> String {
    let cleaned = caption.split_whitespace().collect::<Vec<_>>().join(" ");
    let lower_caption = cleaned.to_lowercase();
    let mut parts: Vec<String> = trigger_words
        .iter()
        .map(|word| word.trim())
        .filter(|word| !word.is_empty())
        .filter(|word| !lower_caption.contains(&word.to_lowercase()))
        .map(ToOwned::to_owned)
        .collect();
    if !cleaned.is_empty() {
        parts.push(cleaned);
    }
    parts.join(", ")
}

pub fn build_chat_text(prompt: &str) -> String {
    build_chat_text_with_system(prompt, SYSTEM_PROMPT, DEFAULT_DATE_STRING, true)
}

pub fn build_chat_text_with_system(
    prompt: &str,
    system_prompt: &str,
    date_string: &str,
    add_generation_prompt: bool,
) -> String {
    let user_prompt = prompt.replace(IMAGE_TOKEN, "");
    let user_prompt = user_prompt.trim_start();
    let mut text = format!(
        "<|start_header_id|>system<|end_header_id|>\n\n\
         Cutting Knowledge Date: {CUTTING_KNOWLEDGE_DATE}\n\
         Today Date: {date_string}\n\n\
         {system_prompt}<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\n\
         {IMAGE_START_TOKEN}{IMAGE_TOKEN}{IMAGE_END_TOKEN}{user_prompt}<|eot_id|>"
    );
    if add_generation_prompt {
        text.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    text
}

pub fn tokenizer_config(max_length: usize) -> TokenizerConfig {
    TokenizerConfig {
        max_length,
        pad_token_id: PAD_TOKEN_ID,
        chat_template: ChatTemplate::None,
        pad_to_max_length: false,
    }
}

pub fn load_tokenizer(root: impl AsRef<Path>) -> Result<TextTokenizer> {
    TextTokenizer::from_file(
        root.as_ref().join("tokenizer.json"),
        tokenizer_config(DEFAULT_MAX_CONTEXT_TOKENS),
    )
}

pub fn load_tokenizer_from_spec(spec: &LoadSpec) -> Result<TextTokenizer> {
    match &spec.weights {
        WeightsSource::Dir(root) => load_tokenizer(root),
        WeightsSource::File(_) => Err(Error::Msg(
            "joycaption: tokenizer loading requires a snapshot directory".to_owned(),
        )),
    }
}

pub fn encode_chat_prompt(tokenizer: &TextTokenizer, prompt: &str) -> Result<Vec<i32>> {
    tokenizer.encode_ids(&build_chat_text(prompt), false)
}

pub fn decode_generated(tokenizer: &TextTokenizer, ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(ids, true)
        .map(|text| text.trim().to_owned())
}

fn templates_for(caption_type: &str) -> &'static [&'static str; 3] {
    PROMPT_TEMPLATES
        .iter()
        .find(|(kind, _)| *kind == caption_type)
        .map(|(_, templates)| templates)
        .unwrap_or(&PROMPT_TEMPLATES[0].1)
}

fn name_or_placeholder(options: &CaptionOptions) -> &str {
    if options.name_input.is_empty() {
        "{NAME}"
    } else {
        &options.name_input
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;

    use super::*;

    fn options(kind: &str, length: &str) -> CaptionOptions {
        CaptionOptions {
            caption_type: kind.to_owned(),
            caption_length: length.to_owned(),
            ..Default::default()
        }
    }

    #[test]
    fn prompt_defaults_match_sceneworks() {
        assert_eq!(
            build_prompt(&CaptionOptions::default()),
            "Write a long detailed description for this image."
        );
        assert_eq!(
            build_prompt(&options("Descriptive", "any")),
            "Write a detailed description for this image."
        );
        assert_eq!(
            build_prompt(&options("Descriptive", "85")),
            "Write a detailed description for this image in 85 words or less."
        );
    }

    #[test]
    fn prompt_falls_back_to_descriptive_for_unknown_type() {
        assert_eq!(
            build_prompt(&options("Not a real type", "short")),
            "Write a short detailed description for this image."
        );
    }

    #[test]
    fn prompt_appends_options_and_replaces_name() {
        let prompt = build_prompt(&CaptionOptions {
            caption_type: "Descriptive".to_owned(),
            caption_length: "long".to_owned(),
            extra_options: vec![
                JOY_NAME_OPTION.to_owned(),
                "Mention the setting.".to_owned(),
            ],
            name_input: "Mika".to_owned(),
            ..Default::default()
        });
        assert_eq!(
            prompt,
            "Write a long detailed description for this image. If there is a person/character in the image you must refer to them as Mika. Mention the setting."
        );
    }

    #[test]
    fn custom_prompt_overrides_template() {
        let prompt = build_prompt(&CaptionOptions {
            custom_prompt: "  Describe only the outfit.  ".to_owned(),
            ..Default::default()
        });
        assert_eq!(prompt, "Describe only the outfit.");
    }

    #[test]
    fn chat_template_matches_single_turn_hf_shape() {
        let text = build_chat_text("Write a caption.");
        assert_eq!(
            text,
            "<|start_header_id|>system<|end_header_id|>\n\n\
             Cutting Knowledge Date: December 2023\n\
             Today Date: 26 July 2024\n\n\
             You are a helpful image captioner.<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\n\
             <|reserved_special_token_70|><|reserved_special_token_69|><|reserved_special_token_71|>Write a caption.<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn chat_template_strips_user_image_token_and_left_space() {
        let text = build_chat_text("  <|reserved_special_token_69|>  Describe.");
        assert!(text.contains(
            "<|reserved_special_token_70|><|reserved_special_token_69|><|reserved_special_token_71|>Describe."
        ));
    }

    #[test]
    fn trigger_words_are_prepended_only_when_missing() {
        let trigger_words = vec!["mika_token".to_owned(), "hat".to_owned()];
        assert_eq!(
            apply_trigger_words("A portrait of Mika wearing a hat.", &trigger_words),
            "mika_token, A portrait of Mika wearing a hat."
        );
        assert_eq!(
            apply_trigger_words("   ", &trigger_words),
            "mika_token, hat"
        );
    }

    #[test]
    fn load_tokenizer_from_file_spec_errors() {
        let spec = LoadSpec::new(WeightsSource::File(PathBuf::from("model.safetensors")));
        assert!(load_tokenizer_from_spec(&spec).is_err());
    }

    #[test]
    #[ignore = "needs the JoyCaption snapshot; set MLX_GEN_JOYCAPTION_SNAPSHOT"]
    fn tokenizer_encodes_image_marker_from_snapshot() {
        let root =
            env::var("MLX_GEN_JOYCAPTION_SNAPSHOT").expect("set MLX_GEN_JOYCAPTION_SNAPSHOT");
        let tokenizer = load_tokenizer(root).expect("tokenizer loads");
        let ids = encode_chat_prompt(&tokenizer, "Write a caption.").expect("prompt encodes");
        assert!(ids.contains(&IMAGE_TOKEN_ID));
        assert!(ids.contains(&START_HEADER_TOKEN_ID));
        assert!(ids.contains(&END_HEADER_TOKEN_ID));
        assert!(ids.contains(&EOT_TOKEN_ID));
    }
}
