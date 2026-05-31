//! Model + transform discovery — the link-time registry, i.e. the Rust stand-in for a DI
//! container's resolve-by-id. See `docs/MODEL_ARCHITECTURE.md` §4.
//!
//! A provider crate self-registers just by being linked (`inventory::submit!`); `mlx-gen` has
//! no central match statement to edit, so adding a model is purely additive. A consumer that
//! links one provider sees exactly one registration. Mirrors the worker's `payload.model` →
//! `MODEL_TARGETS` → load.

use crate::generator::{Generator, ModelDescriptor};
use crate::runtime::LoadSpec;
use crate::transform::{Transform, TransformDescriptor};
use crate::{Error, Result};

/// A generator provider's registration — `descriptor` for introspection (no weights loaded),
/// `load` to construct the model. ≈ `services.AddKeyedSingleton<IGenerator>("id", factory)`.
pub struct ModelRegistration {
    pub descriptor: fn() -> ModelDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Generator>>,
}

inventory::collect!(ModelRegistration);

/// A transform provider's registration (parallel to [`ModelRegistration`]).
pub struct TransformRegistration {
    pub descriptor: fn() -> TransformDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Transform>>,
}

inventory::collect!(TransformRegistration);

/// All registered generators (one per linked provider crate).
pub fn generators() -> impl Iterator<Item = &'static ModelRegistration> {
    inventory::iter::<ModelRegistration>.into_iter()
}

/// All registered transforms.
pub fn transforms() -> impl Iterator<Item = &'static TransformRegistration> {
    inventory::iter::<TransformRegistration>.into_iter()
}

/// Load a generator by model id (e.g. `"z_image_turbo"`).
pub fn load(id: &str, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let reg = generators()
        .find(|r| (r.descriptor)().id == id)
        .ok_or_else(|| Error::Msg(format!("no generator registered for id '{id}'")))?;
    (reg.load)(spec)
}

/// Load a transform by id.
pub fn load_transform(id: &str, spec: &LoadSpec) -> Result<Box<dyn Transform>> {
    let reg = transforms()
        .find(|r| (r.descriptor)().id == id)
        .ok_or_else(|| Error::Msg(format!("no transform registered for id '{id}'")))?;
    (reg.load)(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::{
        Capabilities, GenerationOutput, GenerationRequest, Modality, ModelDescriptor,
    };
    use crate::media::Image;
    use crate::runtime::{Progress, WeightsSource};

    struct DummyGen {
        desc: ModelDescriptor,
    }

    impl Generator for DummyGen {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.desc
        }
        fn validate(&self, _req: &GenerationRequest) -> Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            _req: &GenerationRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<GenerationOutput> {
            Ok(GenerationOutput::Images(vec![Image::default()]))
        }
    }

    fn dummy_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_test_model",
            family: "test",
            modality: Modality::Image,
            capabilities: Capabilities::default(),
        }
    }

    fn dummy_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: dummy_descriptor(),
        }))
    }

    inventory::submit! {
        ModelRegistration { descriptor: dummy_descriptor, load: dummy_load }
    }

    #[test]
    fn registry_resolves_by_id() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = load("dummy_test_model", &spec).expect("dummy is registered");
        assert_eq!(g.descriptor().id, "dummy_test_model");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn unknown_id_errors() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(load("no_such_model", &spec).is_err());
    }

    #[test]
    fn dummy_appears_in_iteration() {
        assert!(generators().any(|r| (r.descriptor)().id == "dummy_test_model"));
    }
}
