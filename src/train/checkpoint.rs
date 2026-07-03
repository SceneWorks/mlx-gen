//! Intermediate-adapter checkpoint filename convention for training.
//!
//! The family trainers share the intermediate-adapter filename convention (driven by
//! `config.save_every`); the trained adapter itself is written by the family trainer's `save_*`
//! (PEFT/LoKr safetensors).
//!
//! NOTE: mid-schedule *resume* (optimizer-state snapshot/restore) is NOT implemented — no trainer
//! wires it, and it would need real per-optimizer state snapshotting (incl. Prodigy/Rose), not a
//! generic moment-buffer dump. Tracked in sc-9560; do not assume resume works from these helpers.

/// `{stem}-step{step:06}.safetensors` — the intermediate adapter checkpoint filename (matches the
/// Python kernel's `save_every` naming). Zero-padded so a lexical sort is a step-order sort.
pub fn checkpoint_filename(stem: &str, step: u32) -> String {
    format!("{stem}-step{step:06}.safetensors")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_filename_is_zero_padded_and_sortable() {
        assert_eq!(
            checkpoint_filename("my_style", 250),
            "my_style-step000250.safetensors"
        );
        assert_eq!(
            checkpoint_filename("lora", 0),
            "lora-step000000.safetensors"
        );
        // Lexical order == step order.
        assert!(checkpoint_filename("a", 9) < checkpoint_filename("a", 10));
        assert!(checkpoint_filename("a", 999) < checkpoint_filename("a", 1000));
    }
}
