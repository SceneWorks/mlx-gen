//! Checkpoint + resume helpers for training (sc-3043).
//!
//! Two pieces the family trainers share: the intermediate-adapter filename convention (driven by
//! `config.save_every`) and optimizer-state snapshot/restore (the Adam/AdamW moment buffers) so an
//! interrupted run can resume mid-schedule rather than from step 0. The trained adapter itself is
//! written by the family trainer's `save_*` (PEFT/LoKr safetensors).

use std::path::Path;

use mlx_rs::optimizers::{Optimizer, OptimizerState};

use crate::Result;

/// `{stem}-step{step:06}.safetensors` — the intermediate adapter checkpoint filename (matches the
/// Python kernel's `save_every` naming). Zero-padded so a lexical sort is a step-order sort.
pub fn checkpoint_filename(stem: &str, step: u32) -> String {
    format!("{stem}-step{step:06}.safetensors")
}

/// Snapshot the optimizer's per-parameter state (Adam/AdamW first/second moments, etc.) to
/// safetensors, for resume.
pub fn save_optimizer_state(opt: &impl Optimizer, path: impl AsRef<Path>) -> Result<()> {
    opt.state().save_safetensors(path)?;
    Ok(())
}

/// Restore optimizer state previously written by [`save_optimizer_state`].
pub fn load_optimizer_state(opt: &mut impl Optimizer, path: impl AsRef<Path>) -> Result<()> {
    opt.state_mut().load_safetensors(path)?;
    Ok(())
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
