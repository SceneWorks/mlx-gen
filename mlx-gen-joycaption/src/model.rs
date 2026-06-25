//! JoyCaption captioner registration, served by the mlx-llm JoyCaption VLM provider.
//!
//! The model — SigLIP vision tower + LLaVA projector + image splice + Llama-3.1 decode + the LLaVA
//! chat-input format — lives in [`mlx-llm`](https://github.com/SceneWorks/mlx-llm) as the
//! `mlx-joycaption` [`core_llm::TextLlm`] vision provider (story 7157). This crate is the thin
//! consumer seam (story 7265): it keeps the SceneWorks caption **product policy** (caption types /
//! length templates / capability bounds, in [`mlx_gen::caption::joycaption`]) and adapts the
//! backend-neutral [`gen_core::Captioner`] contract the worker calls onto the unified engine —
//! building the prompt text + image request and streaming the provider's tokens back.

use mlx_gen::caption::joycaption::{self, JOY_CAPTION_FAMILY, JOY_CAPTION_MODEL_ID};
use mlx_gen::gen_core::{
    core_llm::{
        self, Content, ImageRef, LoadSpec as CoreLoadSpec, Message, ModelRequirements, Role,
        Sampling, StreamEvent, TextLlm, TextLlmRequest,
    },
    Error, Result,
};
use mlx_gen::runtime::Precision;
use mlx_gen::{
    CaptionFinishReason, CaptionOutput, CaptionRequest, Captioner, CaptionerDescriptor, LoadSpec,
    Progress, WeightsSource,
};

// Force-link the mlx-llm engine so the `mlx-joycaption` provider's `inventory::submit!` into
// core-llm's registry survives the linker — this crate resolves the provider through
// `core_llm::load_for_model` and never names another mlx-llm symbol.
use mlx_llm as _;

pub fn descriptor() -> CaptionerDescriptor {
    CaptionerDescriptor {
        id: JOY_CAPTION_MODEL_ID,
        family: JOY_CAPTION_FAMILY,
        backend: "mlx",
        capabilities: joycaption::capabilities(),
    }
}

pub fn load(spec: &LoadSpec) -> Result<Box<dyn Captioner>> {
    Ok(Box::new(load_joycaption(spec)?))
}

pub fn load_joycaption(spec: &LoadSpec) -> Result<JoyCaption> {
    validate_load_spec(spec)?;

    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "joycaption expects a Hugging Face snapshot directory with config.json, \
                 tokenizer.json, and sharded .safetensors, not a single .safetensors file"
                    .to_owned(),
            ))
        }
    };

    // Model-first resolution: the `mlx-joycaption` vision provider's weightless `can_load` claims the
    // LLaVA snapshot; `with_vision()` disambiguates it from the text-only `mlx-llama` provider.
    let provider = core_llm::load_for_model_with(
        &CoreLoadSpec {
            source: root.to_string_lossy().into_owned(),
            quantize: None,
        },
        &ModelRequirements::default().with_vision(),
    )
    .map_err(map_core_err)?;

    Ok(JoyCaption {
        descriptor: descriptor(),
        provider,
    })
}

fn validate_load_spec(spec: &LoadSpec) -> Result<()> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "joycaption: only dense bf16 loading is validated".to_owned(),
        ));
    }
    if spec.quantize.is_some() {
        // Quantized loading is genuinely unsupported here, not merely "not validated" — state that
        // plainly so the message doesn't read as a pending/temporary gap (F-086).
        return Err(Error::Msg(
            "joycaption: quantized loading is not supported".to_owned(),
        ));
    }
    if spec.control.is_some() {
        return Err(Error::Msg(
            "joycaption: control weights are not supported".to_owned(),
        ));
    }
    if spec.ip_adapter.is_some() {
        return Err(Error::Msg(
            "joycaption: IP-Adapter weights are not supported".to_owned(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "joycaption: LoRA/LoKr adapters are not supported".to_owned(),
        ));
    }
    Ok(())
}

/// The JoyCaption captioner: a thin adapter from the [`gen_core::Captioner`] contract onto the
/// mlx-llm `mlx-joycaption` vision provider.
pub struct JoyCaption {
    descriptor: CaptionerDescriptor,
    provider: Box<dyn TextLlm>,
}

fn normalized_request(req: &CaptionRequest) -> CaptionRequest {
    let mut out = req.clone();
    if out.prompt.trim().is_empty() {
        out.prompt = joycaption::build_prompt(&out.options);
    }
    out
}

impl Captioner for JoyCaption {
    fn descriptor(&self) -> &CaptionerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &CaptionRequest) -> Result<()> {
        let req = normalized_request(req);
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, &req)
    }

    fn caption(
        &self,
        req: &CaptionRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CaptionOutput> {
        let req = normalized_request(req);
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, &req)?;
        // Contract: an already-cancelled request errors before inference (the typed cancellation the
        // conformance suite checks).
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // One user turn carrying the image + the (product-policy) prompt text. The model's default
        // system prompt + LLaVA chat-input format are applied inside the provider, so the consumer
        // passes plain text and an image and nothing model-specific.
        let image = ImageRef::new(req.image.width, req.image.height, req.image.pixels.clone())
            .map_err(Error::Msg)?;
        let user = Message {
            role: Role::User,
            content: vec![Content::Image(image), Content::Text(req.prompt.clone())],
            thinking: None,
            // sc-7898 mlx-llm bump: core_llm::Message gained `tool_calls` (Qwen3.6 tool-calling);
            // a caption turn carries none.
            tool_calls: Vec::new(),
        };

        // The provider polls its own `core_llm::CancelFlag`; bridge the gen-core flag onto it by
        // mirroring on each streamed token so the provider's next-step cancel check trips promptly.
        let core_cancel = core_llm::CancelFlag::new();
        let request = TextLlmRequest {
            messages: vec![user],
            sampling: Sampling {
                temperature: req.sampling.temperature,
                top_p: req.sampling.top_p,
                // CaptionSampling exposes no top-k; disabled (0) matches the prior engine sampler.
                top_k: 0,
                repetition_penalty: req.sampling.repetition_penalty,
                repetition_context: req.sampling.repetition_context,
            },
            max_new_tokens: req.sampling.max_new_tokens,
            seed: req.sampling.seed,
            cancel: core_cancel.clone(),
            ..Default::default()
        };

        on_progress(Progress::Step {
            current: 1,
            total: 2,
        });

        let gen_cancel = req.cancel.clone();
        let bridge_cancel = core_cancel;
        let mut on_event = move |ev: StreamEvent| {
            if let StreamEvent::Token { .. } = ev {
                if gen_cancel.is_cancelled() {
                    bridge_cancel.cancel();
                }
            }
        };
        let out = self
            .provider
            .generate(&request, &mut on_event)
            .map_err(map_core_err)?;

        on_progress(Progress::Step {
            current: 2,
            total: 2,
        });

        Ok(CaptionOutput {
            text: out.text.trim().to_owned(),
            generated_tokens: Some(out.usage.generated_tokens),
            finish_reason: out.finish_reason.map(map_finish),
        })
    }
}

/// Map a core-llm engine error onto the gen-core captioner error, preserving the **typed**
/// cancellation the contract (and the conformance suite) require.
fn map_core_err(e: core_llm::Error) -> Error {
    match e {
        core_llm::Error::Canceled => Error::Canceled,
        other => Error::Msg(other.to_string()),
    }
}

fn map_finish(f: core_llm::FinishReason) -> CaptionFinishReason {
    match f {
        // JoyCaption stops on an EOS/stop token or exhausts its token budget; it ships no content
        // filter, so that arm is unreachable but kept total (treated as a model-initiated stop).
        core_llm::FinishReason::Stop | core_llm::FinishReason::ContentFilter => {
            CaptionFinishReason::StopToken
        }
        core_llm::FinishReason::Length => CaptionFinishReason::MaxTokens,
        core_llm::FinishReason::Cancelled => CaptionFinishReason::Cancelled,
    }
}

// Link-time registration (epic 3720): the macro emits the `inventory::submit!`; `load` already
// returns `gen_core::Result`, so the macro's `Into::into` bridge is the identity here (sc-7970).
mlx_gen::register_captioner! { descriptor => load }

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use mlx_gen::caption::CaptionOptions;
    use mlx_gen::media::Image;
    use mlx_gen::runtime::{AdapterKind, AdapterSpec, Quant};

    fn image() -> Image {
        Image {
            width: 384,
            height: 384,
            pixels: vec![127; 384 * 384 * 3],
        }
    }

    fn request() -> CaptionRequest {
        CaptionRequest {
            image: image(),
            prompt: "Write a short caption.".to_owned(),
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_advertises_joycaption_limits() {
        let d = descriptor();
        assert_eq!(d.id, JOY_CAPTION_MODEL_ID);
        assert_eq!(d.family, JOY_CAPTION_FAMILY);
        assert!(d.capabilities.supports_custom_prompt);
        assert!(d.capabilities.supports_low_vram);
        assert!(d.capabilities.mac_only);
        assert_eq!(d.capabilities.max_new_tokens, 1024);
        assert!(d.capabilities.caption_types.contains(&"Straightforward"));
        assert!(d.capabilities.caption_lengths.contains(&"medium-length"));
    }

    #[test]
    fn validation_accepts_empty_prompt_when_options_can_render_it() {
        let req = CaptionRequest {
            prompt: String::new(),
            options: CaptionOptions {
                caption_type: "Straightforward".to_owned(),
                caption_length: "short".to_owned(),
                ..Default::default()
            },
            ..request()
        };
        let req = normalized_request(&req);
        assert_eq!(
            req.prompt,
            joycaption::build_prompt(&CaptionOptions {
                caption_type: "Straightforward".to_owned(),
                caption_length: "short".to_owned(),
                ..Default::default()
            })
        );
        assert!(descriptor()
            .capabilities
            .validate_request(JOY_CAPTION_MODEL_ID, &req)
            .is_ok());
    }

    #[test]
    fn load_rejects_unsupported_specs_before_touching_disk() {
        let root = PathBuf::from("/nonexistent/joycaption");

        let mut fp32 = LoadSpec::new(WeightsSource::Dir(root.clone()));
        fp32.precision = Precision::Fp32;
        assert!(load_joycaption(&fp32)
            .err()
            .expect("fp32 specs are rejected before disk access")
            .to_string()
            .contains("dense bf16"));

        let q4 = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q4);
        assert!(load_joycaption(&q4)
            .err()
            .expect("quantized specs are rejected before disk access")
            .to_string()
            .contains("quantized"));

        let adapters =
            LoadSpec::new(WeightsSource::Dir(root)).with_adapters(vec![AdapterSpec::new(
                PathBuf::from("adapter.safetensors"),
                1.0,
                AdapterKind::Lora,
            )]);
        assert!(load_joycaption(&adapters)
            .err()
            .expect("adapter specs are rejected before disk access")
            .to_string()
            .contains("adapters"));
    }

    #[test]
    fn load_rejects_single_file_snapshot() {
        let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
        let err = load_joycaption(&spec)
            .err()
            .expect("file spec rejected")
            .to_string();
        assert!(err.contains("snapshot directory"), "{err}");
    }
}
