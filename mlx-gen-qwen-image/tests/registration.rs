//! sc-7780 dead-strip gate: the three Qwen-Image variants self-register through the core registry
//! once the crate is linked (`use mlx_gen_qwen_image as _;`). This proves the macro-emitted
//! `inventory::submit!` (register_generators!) still fires; it needs no weights.

use mlx_gen_qwen_image as _;

#[test]
fn qwen_image_variants_resolve_through_core_registry() {
    for id in ["qwen_image", "qwen_image_control", "qwen_image_edit"] {
        let reg = mlx_gen::registry::generators()
            .find(|r| (r.descriptor)().id == id)
            .unwrap_or_else(|| panic!("{id} provider should self-register"));
        assert_eq!((reg.descriptor)().family, "qwen-image");
    }
}
