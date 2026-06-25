/// Internal: expand each `$desc => $load` arm into an `inventory::submit!` of the given
/// registration struct `$reg`, bridging the loader's `Result` into `gen_core::Result` via
/// `Into::into` (identity when the loader already returns `gen_core::Result`).
///
/// The public `register_*!` macros below are thin wrappers that fix `$reg` to a specific
/// `*Registration` struct. Sharing one rule keeps the link-time wiring identical across every
/// registration kind (generators, trainers, captioners, image/text embedders) and mirrors what
/// candle-gen needs — the sc-7779 cross-backend note anticipated folding all kinds onto one rule.
#[doc(hidden)]
#[macro_export]
macro_rules! __register_kind {
    ( $reg:path, $( $desc:path => $load:path ),+ $(,)? ) => {
        $(
            $crate::inventory::submit! {
                $reg {
                    descriptor: $desc,
                    load: |spec| $load(spec).map_err(::core::convert::Into::into),
                }
            }
        )+
    };
}

/// Register one or more generator descriptors and loaders with the link-time registry.
#[macro_export]
macro_rules! register_generators {
    ( $( $desc:path => $load:path ),+ $(,)? ) => {
        $crate::__register_kind! { $crate::registry::ModelRegistration, $( $desc => $load ),+ }
    };
}

/// Register one or more trainer descriptors and loaders with the link-time registry.
#[macro_export]
macro_rules! register_trainer {
    ( $( $desc:path => $load:path ),+ $(,)? ) => {
        $crate::__register_kind! { $crate::registry::TrainerRegistration, $( $desc => $load ),+ }
    };
}

/// Register one or more captioner descriptors and loaders with the link-time registry.
#[macro_export]
macro_rules! register_captioner {
    ( $( $desc:path => $load:path ),+ $(,)? ) => {
        $crate::__register_kind! { $crate::registry::CaptionerRegistration, $( $desc => $load ),+ }
    };
}

/// Register one or more image-embedder descriptors and loaders with the link-time registry.
#[macro_export]
macro_rules! register_image_embedder {
    ( $( $desc:path => $load:path ),+ $(,)? ) => {
        $crate::__register_kind! {
            $crate::registry::ImageEmbedderRegistration, $( $desc => $load ),+
        }
    };
}

/// Register one or more text-embedder descriptors and loaders with the link-time registry.
#[macro_export]
macro_rules! register_text_embedder {
    ( $( $desc:path => $load:path ),+ $(,)? ) => {
        $crate::__register_kind! {
            $crate::registry::TextEmbedderRegistration, $( $desc => $load ),+
        }
    };
}

/// Implement the standard delegation-pattern [`Generator`] wrapper for provider structs.
#[macro_export]
macro_rules! impl_generator {
    (
        $ty:ty {
            validate: |$self_arg:ident, $req_arg:ident| $validate:expr,
            generate: $generate:ident $(,)?
        }
    ) => {
        impl $crate::Generator for $ty {
            fn descriptor(&self) -> &$crate::ModelDescriptor {
                &self.descriptor
            }

            fn validate(&self, req: &$crate::GenerationRequest) -> $crate::Result<()> {
                let validate = |$self_arg: &Self, $req_arg: &$crate::GenerationRequest| $validate;
                validate(self, req).map_err(::core::convert::Into::into)
            }

            fn generate(
                &self,
                req: &$crate::GenerationRequest,
                on_progress: &mut dyn FnMut($crate::Progress),
            ) -> $crate::Result<$crate::GenerationOutput> {
                self.$generate(req, on_progress)
                    .map_err(::core::convert::Into::into)
            }
        }
    };
}
