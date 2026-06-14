//! The `TextLlm` contract: text-in, text-out instruction-LLM inference (epic 3720 Phase 6).
//!
//! A generic instruction/chat LLM seam — the caller supplies an optional `system` message and a
//! `user` prompt, the provider runs an autoregressive decoder and returns generated text. It is
//! intentionally separate from [`Generator`](crate::generator::Generator) (which synthesizes media)
//! and [`Captioner`](crate::caption::Captioner) (which consumes an image): a `TextLlm` provider has
//! no image input and no diffusion schedule, just a token-by-token text decode.
//!
//! The first consumer is **prompt refinement** (sc-5500, retiring the worker's Python
//! `prompt_refine.py`): the caller folds its prompt-rewrite rules + the model prompt-guide into the
//! `system` message and passes the user's prompt as `prompt`, keeping all product-specific prompt
//! assembly at the caller's edge so the contract stays a reusable instruction-LLM surface.

use crate::runtime::{CancelFlag, Progress};
use crate::{Error, Result};

/// A text-in, text-out instruction-LLM provider.
pub trait TextLlm {
    /// Stable identity + capability metadata, constructible without loading weights through the
    /// registry.
    fn descriptor(&self) -> &TextLlmDescriptor;

    /// Reject a request this provider cannot serve before running model inference.
    fn validate(&self, req: &TextLlmRequest) -> Result<()>;

    /// Generate text for one request. Long-running implementations should check
    /// [`TextLlmRequest::cancel`] and report token/progress events through `on_progress`. A provider
    /// handed an already-cancelled request must return [`Error::Canceled`] *before* running inference.
    fn generate(
        &self,
        req: &TextLlmRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<TextLlmOutput>;
}

/// A single text-generation request.
#[derive(Clone, Debug, Default)]
pub struct TextLlmRequest {
    /// Optional system / instruction message. Empty = no system turn (some chat templates reject a
    /// system role, so a provider may fold a non-empty system message into the first user turn). The
    /// caller owns all product-specific prompt assembly (e.g. prompt-rewrite rules + a model guide).
    pub system: String,
    /// The user turn — the actual prompt the model responds to. Required (non-empty after trim).
    pub prompt: String,
    /// Sampling controls for the autoregressive decoder.
    pub sampling: TextLlmSampling,
    pub cancel: CancelFlag,
}

/// Autoregressive sampling knobs for text generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextLlmSampling {
    pub temperature: f32,
    pub top_p: f32,
    pub max_new_tokens: u32,
    /// RNG seed for stochastic sampling (`temperature > 0`). `None` draws a fresh per-call seed via
    /// [`default_seed`](crate::generator::default_seed) so repeated calls vary; pass `Some(seed)` to
    /// reproduce an exact generation. (At `temperature == 0` decoding is greedy and the seed is unused.)
    pub seed: Option<u64>,
}

impl Default for TextLlmSampling {
    /// The prompt-refinement defaults the Python `PromptRefiner` used: `temperature 0.7`, `top_p 0.9`,
    /// `max_new_tokens 512`.
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            max_new_tokens: 512,
            seed: None,
        }
    }
}

/// Text generation result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextLlmOutput {
    pub text: String,
    pub generated_tokens: Option<u32>,
    pub finish_reason: Option<TextLlmFinishReason>,
}

/// Why generation stopped, when the provider can report it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextLlmFinishReason {
    StopToken,
    MaxTokens,
    Cancelled,
}

/// A text-LLM provider's stable identity + advertised capabilities.
#[derive(Clone, Debug)]
pub struct TextLlmDescriptor {
    pub id: &'static str,
    pub family: &'static str,
    /// Tensor backend that registered this provider ("mlx" | "candle"); used by the worker's
    /// per-backend capability advertisement (sc-4906, epic 3720).
    pub backend: &'static str,
    pub capabilities: TextLlmCapabilities,
}

/// The shared text-LLM capability surface. Provider-specific constraints are layered on top by each
/// provider's own `validate`.
#[derive(Clone, Debug, Default)]
pub struct TextLlmCapabilities {
    pub max_prompt_chars: usize,
    pub max_system_chars: usize,
    pub supports_system_prompt: bool,
    pub max_new_tokens: u32,
    pub mac_only: bool,
}

impl TextLlmCapabilities {
    /// Reject request fields that exceed the advertised shared capability surface.
    pub fn validate_request(&self, id: &str, req: &TextLlmRequest) -> Result<()> {
        // Footgun guard (F-084): a provider that leaves its bounds at the `Default` 0 would reject
        // every request. Catch the descriptor mistake in debug/test builds.
        debug_assert!(
            self.max_new_tokens > 0 && self.max_prompt_chars > 0,
            "{id}: TextLlmCapabilities bounds left at Default 0 (max_new_tokens={}, \
             max_prompt_chars={}) — descriptor forgot its bounds",
            self.max_new_tokens,
            self.max_prompt_chars
        );
        if req.prompt.trim().is_empty() {
            return Err(Error::Msg(format!("{id}: prompt is required")));
        }
        if req.prompt.chars().count() > self.max_prompt_chars {
            return Err(Error::Msg(format!(
                "{id}: prompt is longer than {} characters",
                self.max_prompt_chars
            )));
        }
        if !req.system.trim().is_empty() && !self.supports_system_prompt {
            return Err(Error::Msg(format!(
                "{id}: a system prompt was supplied but this provider does not support one"
            )));
        }
        if req.system.chars().count() > self.max_system_chars {
            return Err(Error::Msg(format!(
                "{id}: system prompt is longer than {} characters",
                self.max_system_chars
            )));
        }
        if req.sampling.temperature < 0.0 || req.sampling.temperature > 2.0 {
            return Err(Error::Msg(format!(
                "{id}: temperature must be between 0 and 2"
            )));
        }
        if req.sampling.top_p < 0.0 || req.sampling.top_p > 1.0 {
            return Err(Error::Msg(format!("{id}: top_p must be between 0 and 1")));
        }
        if req.sampling.max_new_tokens == 0 || req.sampling.max_new_tokens > self.max_new_tokens {
            return Err(Error::Msg(format!(
                "{id}: max_new_tokens {} out of range 1..={}",
                req.sampling.max_new_tokens, self.max_new_tokens
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> TextLlmCapabilities {
        TextLlmCapabilities {
            max_prompt_chars: 8000,
            max_system_chars: 16000,
            supports_system_prompt: true,
            max_new_tokens: 1024,
            mac_only: false,
        }
    }

    fn base_req() -> TextLlmRequest {
        TextLlmRequest {
            prompt: "Rewrite this prompt for an image model.".to_owned(),
            ..Default::default()
        }
    }

    #[test]
    fn sampling_defaults_match_prompt_refine_surface() {
        let s = TextLlmSampling::default();
        assert_eq!(s.temperature, 0.7);
        assert_eq!(s.top_p, 0.9);
        assert_eq!(s.max_new_tokens, 512);
        assert_eq!(s.seed, None);
    }

    #[test]
    fn validate_request_accepts_supported_surface() {
        let c = caps();
        assert!(c.validate_request("textllm", &base_req()).is_ok());
        assert!(c
            .validate_request(
                "textllm",
                &TextLlmRequest {
                    system: "You are a prompt rewriter.".to_owned(),
                    prompt: "a cat".to_owned(),
                    sampling: TextLlmSampling {
                        temperature: 0.0, // greedy
                        max_new_tokens: 1024,
                        ..Default::default()
                    },
                    ..Default::default()
                }
            )
            .is_ok());
    }

    #[test]
    fn validate_request_enforces_shared_surface() {
        let c = caps();
        let cases = [
            // empty prompt
            TextLlmRequest {
                prompt: String::new(),
                ..base_req()
            },
            // prompt too long
            TextLlmRequest {
                prompt: "x".repeat(8001),
                ..base_req()
            },
            // system too long
            TextLlmRequest {
                system: "x".repeat(16001),
                ..base_req()
            },
            // temperature out of range
            TextLlmRequest {
                sampling: TextLlmSampling {
                    temperature: 2.1,
                    ..Default::default()
                },
                ..base_req()
            },
            // top_p out of range
            TextLlmRequest {
                sampling: TextLlmSampling {
                    top_p: 1.1,
                    ..Default::default()
                },
                ..base_req()
            },
            // max_new_tokens zero
            TextLlmRequest {
                sampling: TextLlmSampling {
                    max_new_tokens: 0,
                    ..Default::default()
                },
                ..base_req()
            },
            // max_new_tokens above cap
            TextLlmRequest {
                sampling: TextLlmSampling {
                    max_new_tokens: 1025,
                    ..Default::default()
                },
                ..base_req()
            },
        ];
        for (i, req) in cases.iter().enumerate() {
            assert!(
                c.validate_request("textllm", req).is_err(),
                "case {i} should have been rejected"
            );
        }
    }

    #[test]
    fn validate_request_rejects_unsupported_system_prompt() {
        let c = TextLlmCapabilities {
            supports_system_prompt: false,
            ..caps()
        };
        let req = TextLlmRequest {
            system: "You are helpful.".to_owned(),
            ..base_req()
        };
        assert!(c.validate_request("textllm", &req).is_err());
    }
}
