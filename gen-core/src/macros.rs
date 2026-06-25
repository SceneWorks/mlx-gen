/// Register one or more generator descriptors and loaders with the link-time registry.
#[macro_export]
macro_rules! register_generators {
    ( $( $desc:path => $load:path ),+ $(,)? ) => {
        $(
            $crate::inventory::submit! {
                $crate::registry::ModelRegistration {
                    descriptor: $desc,
                    load: |spec| $load(spec).map_err(::core::convert::Into::into),
                }
            }
        )+
    };
}

/// Register one trainer descriptor and loader with the link-time registry.
#[macro_export]
macro_rules! register_trainer {
    ( $desc:path => $load:path $(,)? ) => {
        $crate::inventory::submit! {
            $crate::registry::TrainerRegistration {
                descriptor: $desc,
                load: |spec| $load(spec).map_err(::core::convert::Into::into),
            }
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
