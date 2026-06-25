//! SD3.5-Large / Large-Turbo arch + variant config, asserted against the sc-7850 real-weight facts.

use mlx_gen_sd3::{
    Sd3Arch, Sd3Variant, LARGE_CAPTION_PROJECTION_DIM, LARGE_HEAD_DIM, LARGE_HIDDEN,
    LARGE_IN_CHANNELS, LARGE_JOINT_ATTENTION_DIM, LARGE_NUM_HEADS, LARGE_NUM_LAYERS,
    LARGE_OUT_CHANNELS, LARGE_PATCH_SIZE, LARGE_POOLED_PROJECTION_DIM, LARGE_POS_EMBED_LEN,
    LARGE_POS_EMBED_MAX_SIZE, LARGE_TIME_PROJ_DIM, SD3_5_LARGE_ID, SD3_5_LARGE_TURBO_ID,
};

#[test]
fn large_arch_matches_real_weights() {
    // sc-7850 empirically confirmed: 38 layers, hidden 2432 (= 38×64), 38 heads, head_dim 64,
    // patch 2, 16-ch in/out, joint 4096, pooled 2048, caption 2432, pos_embed_max 96 ⇒ table len
    // 36864, timestep proj dim 256.
    assert_eq!(LARGE_NUM_LAYERS, 38);
    assert_eq!(LARGE_HEAD_DIM, 64);
    assert_eq!(LARGE_NUM_HEADS, 38);
    assert_eq!(LARGE_HIDDEN, 2432);
    assert_eq!(LARGE_HIDDEN, LARGE_NUM_HEADS * LARGE_HEAD_DIM);
    assert_eq!(LARGE_PATCH_SIZE, 2);
    assert_eq!(LARGE_IN_CHANNELS, 16);
    assert_eq!(LARGE_OUT_CHANNELS, 16);
    assert_eq!(LARGE_JOINT_ATTENTION_DIM, 4096);
    assert_eq!(LARGE_POOLED_PROJECTION_DIM, 2048);
    assert_eq!(LARGE_CAPTION_PROJECTION_DIM, 2432);
    assert_eq!(LARGE_POS_EMBED_MAX_SIZE, 192);
    assert_eq!(LARGE_POS_EMBED_LEN, 36864);
    assert_eq!(LARGE_TIME_PROJ_DIM, 256);
}

#[test]
fn arch_struct_derived_dims() {
    let a = Sd3Arch::large();
    assert_eq!(a.hidden(), 2432);
    assert_eq!(a.pos_embed_len(), 36864);
    // proj_out width = patch*patch*out_channels = 2*2*16 = 64.
    assert_eq!(a.patch_out_dim(), 64);
    assert_eq!(Sd3Arch::default(), Sd3Arch::large());
}

#[test]
fn variant_ids_and_schedules() {
    assert_eq!(Sd3Variant::Large.id(), SD3_5_LARGE_ID);
    assert_eq!(
        Sd3Variant::Large.hf_model(),
        "stabilityai/stable-diffusion-3.5-large"
    );
    assert_eq!(Sd3Variant::Large.default_steps(), 28);
    assert!((Sd3Variant::Large.default_guidance() - 3.5).abs() < 1e-6);
    assert!(Sd3Variant::Large.supports_true_cfg());

    assert_eq!(Sd3Variant::LargeTurbo.id(), SD3_5_LARGE_TURBO_ID);
    assert_eq!(
        Sd3Variant::LargeTurbo.hf_model(),
        "stabilityai/stable-diffusion-3.5-large-turbo"
    );
    assert_eq!(Sd3Variant::LargeTurbo.default_steps(), 4);
    assert!((Sd3Variant::LargeTurbo.default_guidance() - 1.0).abs() < 1e-6);
    // Turbo is distilled, guidance-free (no true CFG / negative prompt).
    assert!(!Sd3Variant::LargeTurbo.supports_true_cfg());

    // Both variants share one MMDiT arch.
    assert_eq!(Sd3Variant::Large.arch(), Sd3Variant::LargeTurbo.arch());
}

#[test]
fn descriptor_capabilities() {
    let large = Sd3Variant::Large.descriptor();
    assert_eq!(large.id, SD3_5_LARGE_ID);
    assert_eq!(large.family, "sd3");
    assert_eq!(large.backend, "mlx");
    assert!(large.capabilities.supports_true_cfg);
    assert!(large.capabilities.supports_negative_prompt);
    assert!(large.capabilities.supports_lora);
    assert!(large.capabilities.supports_lokr);
    assert!(large.capabilities.mac_only);
    assert!(large.capabilities.requires_sigma_shift);

    let turbo = Sd3Variant::LargeTurbo.descriptor();
    assert_eq!(turbo.id, SD3_5_LARGE_TURBO_ID);
    assert!(!turbo.capabilities.supports_true_cfg);
    assert!(!turbo.capabilities.supports_negative_prompt);
}
