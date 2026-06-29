//! SANA pipeline **composition contract** (epic 8485, story sc-8489 — Phase A).
//!
//! Proves the end-to-end native SANA wiring — caption embedding → Linear-DiT trunk denoise over the
//! unified flow-match Euler scheduler (with true CFG) → DC-AE decode → RGB image tensor — composes
//! correctly, WITHOUT the ~1.6B-param real weights. Both the trunk and the DC-AE decoder are built
//! from tiny random-init weights at reduced dim/depth (mirroring the tiny-golden style the trunk and
//! DC-AE parity tests use), so the test runs in CI on committed inputs only.
//!
//! It exercises the real composition seams:
//!  * a synthetic `[1, M, caption_channels]` caption embedding (the [`SanaTextEncoder`] output shape)
//!    drives the trunk's `attn2` cross-attention — the text-encoder forward itself needs the gemma
//!    weights and is covered by `text_encoder.rs` / the `#[ignore]`d e2e test, so it is replaced here
//!    by a random embedding of the exact shape;
//!  * [`pipeline::denoise_cfg`] runs a couple of flow-match Euler steps with CFG over the trunk;
//!  * [`pipeline::decode_to_image`] applies the DC-AE `scaling_factor` un-scale + decode.
//!
//! The non-`#[ignore]`d e2e wiring test ([`pipeline_wires_trunk_scheduler_decode`]) asserts the final
//! image tensor shape and finiteness. The `#[ignore]`d real-weight test
//! ([`real_weight_1024_e2e`]) is gated behind `SANA_PIPELINE_WEIGHTS` and runs a real 1024px gen.

use mlx_rs::ops::{max as max_op, min as min_op};
use mlx_rs::random::{key, normal};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use mlx_gen::image::decoded_to_image;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, FlowMatchEuler, Progress};

use mlx_gen_sana::pipeline::{self, SCHEDULE_SHIFT};
use mlx_gen_sana::{BlockType, DcAeConfig, DcAeDecoder, SanaTransformer, SanaTransformerConfig};

/// Deterministic random tensor for a synthetic weight.
fn rand(shape: &[i32], seed: u64) -> Array {
    normal::<f32>(shape, None, None, Some(&key(seed).unwrap())).unwrap()
}

// ---------------------------------------------------------------------------------------------
// Tiny trunk weights (the diffusers `SanaTransformer2DModel` key names the loader requires).
// ---------------------------------------------------------------------------------------------

fn tiny_trunk_config() -> SanaTransformerConfig {
    SanaTransformerConfig {
        in_channels: 4,
        out_channels: 4, // == DC-AE latent_channels below
        num_attention_heads: 2,
        attention_head_dim: 8, // inner = 16
        num_layers: 2,
        num_cross_attention_heads: 2,
        cross_attention_head_dim: 8,
        caption_channels: 24,
        mlp_ratio: 2.5,
        patch_size: 1,
        norm_eps: 1e-6,
        caption_norm_eps: 1e-5,
        attn_qk_norm_eps: 1e-5,
        attn_eps: 1e-15,
        guidance_embeds: false,
        guidance_embeds_scale: 0.1,
        qk_norm: false,
    }
}

fn linear(w: &mut Weights, prefix: &str, inn: i32, out: i32, bias: bool, seed: &mut u64) {
    w.insert(format!("{prefix}.weight"), rand(&[out, inn], *seed));
    *seed += 1;
    if bias {
        w.insert(format!("{prefix}.bias"), rand(&[out], *seed));
        *seed += 1;
    }
}

/// Conv weight in PyTorch `[O, I/groups, H, W]` layout (the loader transposes to NHWC).
fn conv(
    w: &mut Weights,
    prefix: &str,
    o: i32,
    i_per_group: i32,
    k: i32,
    bias: bool,
    seed: &mut u64,
) {
    w.insert(
        format!("{prefix}.weight"),
        rand(&[o, i_per_group, k, k], *seed),
    );
    *seed += 1;
    if bias {
        w.insert(format!("{prefix}.bias"), rand(&[o], *seed));
        *seed += 1;
    }
}

fn tiny_trunk_weights(cfg: &SanaTransformerConfig) -> Weights {
    let mut w = Weights::empty();
    let mut s = 1_u64;
    let inner = cfg.inner_dim();
    let hidden = (cfg.mlp_ratio * inner as f32) as i32;

    // patch_embed.proj: in → inner (1×1 conv).
    conv(
        &mut w,
        "patch_embed.proj",
        inner,
        cfg.in_channels,
        cfg.patch_size,
        true,
        &mut s,
    );

    // timestep path.
    linear(
        &mut w,
        "time_embed.emb.timestep_embedder.linear_1",
        256,
        inner,
        true,
        &mut s,
    );
    linear(
        &mut w,
        "time_embed.emb.timestep_embedder.linear_2",
        inner,
        inner,
        true,
        &mut s,
    );
    linear(&mut w, "time_embed.linear", inner, 6 * inner, true, &mut s);

    // caption path.
    linear(
        &mut w,
        "caption_projection.linear_1",
        cfg.caption_channels,
        inner,
        true,
        &mut s,
    );
    linear(
        &mut w,
        "caption_projection.linear_2",
        inner,
        inner,
        true,
        &mut s,
    );
    w.insert("caption_norm.weight", rand(&[inner], s));
    s += 1;

    // per-block weights.
    let cross_inner = cfg.num_cross_attention_heads * cfg.cross_attention_head_dim;
    for i in 0..cfg.num_layers {
        let p = format!("transformer_blocks.{i}");
        w.insert(format!("{p}.scale_shift_table"), rand(&[6, inner], s));
        s += 1;
        // attn1 (linear self-attn): q/k/v bias-free, to_out.0 bias.
        linear(
            &mut w,
            &format!("{p}.attn1.to_q"),
            inner,
            inner,
            false,
            &mut s,
        );
        linear(
            &mut w,
            &format!("{p}.attn1.to_k"),
            inner,
            inner,
            false,
            &mut s,
        );
        linear(
            &mut w,
            &format!("{p}.attn1.to_v"),
            inner,
            inner,
            false,
            &mut s,
        );
        linear(
            &mut w,
            &format!("{p}.attn1.to_out.0"),
            inner,
            inner,
            true,
            &mut s,
        );
        // attn2 (cross-attn): all bias.
        linear(
            &mut w,
            &format!("{p}.attn2.to_q"),
            inner,
            cross_inner,
            true,
            &mut s,
        );
        linear(
            &mut w,
            &format!("{p}.attn2.to_k"),
            inner,
            cross_inner,
            true,
            &mut s,
        );
        linear(
            &mut w,
            &format!("{p}.attn2.to_v"),
            inner,
            cross_inner,
            true,
            &mut s,
        );
        linear(
            &mut w,
            &format!("{p}.attn2.to_out.0"),
            cross_inner,
            inner,
            true,
            &mut s,
        );
        // GLUMBConv Mix-FFN.
        conv(
            &mut w,
            &format!("{p}.ff.conv_inverted"),
            2 * hidden,
            inner,
            1,
            true,
            &mut s,
        );
        // depthwise 3×3: groups = 2·hidden, so I/groups = 1.
        conv(
            &mut w,
            &format!("{p}.ff.conv_depth"),
            2 * hidden,
            1,
            3,
            true,
            &mut s,
        );
        conv(
            &mut w,
            &format!("{p}.ff.conv_point"),
            inner,
            hidden,
            1,
            false,
            &mut s,
        );
    }

    // output modulated norm + proj_out.
    w.insert("scale_shift_table", rand(&[2, inner], s));
    s += 1;
    linear(
        &mut w,
        "proj_out",
        inner,
        cfg.patch_size * cfg.patch_size * cfg.out_channels,
        true,
        &mut s,
    );
    w
}

// ---------------------------------------------------------------------------------------------
// Tiny DC-AE weights (all-`Res` stages → only conv1/conv2/norm + conv_in/out + up_block conv).
// ---------------------------------------------------------------------------------------------

fn tiny_dcae_config() -> DcAeConfig {
    DcAeConfig {
        in_channels: 3,
        latent_channels: 4, // == trunk out_channels
        attention_head_dim: 4,
        block_out_channels: vec![6, 8], // 2 stages → 1 upsample → 2× spatial scale
        layers_per_block: vec![1, 1],
        block_types: vec![BlockType::Res, BlockType::Res],
        qkv_multiscales: vec![5],
        upsample_interpolate: true,
        norm_eps: 1e-5,
        attn_eps: 1e-15,
        scaling_factor: 0.41407,
    }
}

fn res_block(w: &mut Weights, prefix: &str, ch: i32, seed: &mut u64) {
    conv(w, &format!("{prefix}.conv1"), ch, ch, 3, true, seed);
    conv(w, &format!("{prefix}.conv2"), ch, ch, 3, false, seed);
    w.insert(format!("{prefix}.norm.weight"), rand(&[ch], *seed));
    *seed += 1;
    w.insert(format!("{prefix}.norm.bias"), rand(&[ch], *seed));
    *seed += 1;
}

fn tiny_dcae_weights(cfg: &DcAeConfig) -> Weights {
    let mut w = Weights::empty();
    let mut s = 5000_u64;
    let n = cfg.num_stages();
    let deepest = cfg.block_out_channels[n - 1];

    // conv_in: latent → deepest (the decoder enters at the deepest stage channel count).
    conv(
        &mut w,
        "decoder.conv_in",
        deepest,
        cfg.latent_channels,
        3,
        true,
        &mut s,
    );

    for i in 0..n {
        let ch = cfg.block_out_channels[i];
        let has_up = i + 1 < n;
        let mut slot = 0;
        if has_up {
            // up_blocks.{i}.0.conv: in = next-deeper channels, out = this stage's channels.
            conv(
                &mut w,
                &format!("decoder.up_blocks.{i}.0.conv"),
                ch,
                cfg.block_out_channels[i + 1],
                3,
                true,
                &mut s,
            );
            slot = 1;
        }
        for j in 0..cfg.layers_per_block[i] {
            res_block(
                &mut w,
                &format!("decoder.up_blocks.{i}.{}", j + slot),
                ch,
                &mut s,
            );
        }
    }

    let shallow = cfg.block_out_channels[0];
    w.insert("decoder.norm_out.weight", rand(&[shallow], s));
    s += 1;
    w.insert("decoder.norm_out.bias", rand(&[shallow], s));
    s += 1;
    conv(
        &mut w,
        "decoder.conv_out",
        cfg.in_channels,
        shallow,
        3,
        true,
        &mut s,
    );
    w
}

#[test]
fn pipeline_wires_trunk_scheduler_decode() {
    let tcfg = tiny_trunk_config();
    let trunk = SanaTransformer::from_weights(&tiny_trunk_weights(&tcfg), tcfg.clone())
        .expect("build tiny trunk");
    let dcfg = tiny_dcae_config();
    let decoder = DcAeDecoder::from_weights(&tiny_dcae_weights(&dcfg), dcfg.clone())
        .expect("build tiny dc-ae");

    // Latent grid: NON-SQUARE on purpose (latent_h != latent_w) so the NHWC→NCHW transpose
    // `[0,3,1,2]` in `decode_to_image` is genuinely exercised — a square grid would let a wrong
    // permutation (e.g. `[0,3,2,1]`) pass undetected. patch_size = 1, so any latent edge is valid.
    // DC-AE scale here = 2^(stages-1) = 2.
    let latent_h = 6;
    let latent_w = 4;
    let scale: i32 = 1 << (dcfg.num_stages() - 1); // 2
    let latents = normal::<f32>(
        &[1, tcfg.out_channels, latent_h, latent_w],
        None,
        None,
        Some(&key(0).unwrap()),
    )
    .unwrap();

    // Synthetic caption embeddings (the SanaTextEncoder output shape [1, M, caption_channels]); M is
    // arbitrary (the trunk's cross-attn is full-softmax over all caption tokens). Cond ≠ uncond so CFG
    // genuinely combines two distinct forwards.
    let m = 7;
    let cond = rand(&[1, m, tcfg.caption_channels], 100);
    let uncond = rand(&[1, m, tcfg.caption_channels], 200);

    let scheduler = FlowMatchEuler::for_static_shift(2, SCHEDULE_SHIFT);
    let cancel = CancelFlag::default();
    let mut steps_seen = 0_usize;
    let mut on_progress = |p: Progress| {
        if matches!(p, Progress::Step { .. }) {
            steps_seen += 1;
        }
    };

    let denoised = pipeline::denoise_cfg(
        &trunk,
        &scheduler,
        None,
        0,
        latents,
        &cond,
        Some(&uncond),
        4.5,
        &cancel,
        &mut on_progress,
    )
    .expect("denoise");

    assert_eq!(
        denoised.shape(),
        &[1, tcfg.out_channels, latent_h, latent_w],
        "denoised latent keeps the trunk in/out channel + spatial shape"
    );

    // Decode through the DC-AE (with the scaling_factor un-scale).
    let img = pipeline::decode_to_image(&decoder, &dcfg, &denoised).expect("decode");
    let exp_w = (latent_w * scale) as u32;
    let exp_h = (latent_h * scale) as u32;
    // NON-SQUARE: width and height differ, so an axis swap in the NHWC→NCHW transpose would land
    // these on the wrong fields and fail. `decoded_to_image` reports width = NCHW axis-3 (W),
    // height = NCHW axis-2 (H); with the correct `[0,3,1,2]` they map straight through.
    assert_ne!(exp_w, exp_h, "test grid must be non-square to catch a swap");
    assert_eq!(img.width, exp_w, "decoded width = latent_w · dc_ae_scale");
    assert_eq!(img.height, exp_h, "decoded height = latent_h · dc_ae_scale");
    assert_eq!(
        img.pixels.len(),
        (exp_w * exp_h * 3) as usize,
        "RGB8 pixel buffer is H·W·3"
    );

    // Finiteness/sanity directly on the decoder tensor (pre-RGB8-quantization).
    let scale_arr = Array::from_slice(&[dcfg.scaling_factor], &[1]);
    let unscaled = mlx_rs::ops::divide(&denoised, &scale_arr).unwrap();
    let decoded = decoder.decode(&unscaled).unwrap();
    let lo = min_op(&decoded, None).unwrap();
    let hi = max_op(&decoded, None).unwrap();
    eval([&lo, &hi]).unwrap();
    let (lo, hi) = (lo.item::<f32>(), hi.item::<f32>());
    assert!(
        lo.is_finite() && hi.is_finite(),
        "decoded output non-finite: [{lo}, {hi}]"
    );
    assert!(
        hi - lo > 1e-6,
        "decoded output is constant — graph degenerate"
    );

    // Round-trips through the shared RGB8 converter too (NHWC → NCHW handled in decode_to_image).
    let _ = decoded_to_image(&decoded.transpose_axes(&[0, 3, 1, 2]).unwrap()).expect("rgb8");
    assert!(
        steps_seen >= 2,
        "flow sampler should report per-step progress"
    );
}

/// Real-weight 1024px e2e. `#[ignore]`d: needs a `Sana_1600M_1024px_diffusers` snapshot
/// (transformer/ + vae/ + the gemma TE mirror). Set `SANA_PIPELINE_WEIGHTS` to the snapshot root.
///
/// Expected layout under `$SANA_PIPELINE_WEIGHTS`:
///   transformer/diffusion_pytorch_model.safetensors   (SANA trunk)
///   vae/diffusion_pytorch_model.safetensors           (DC-AE f32c32)
///   text_encoder/gemma-2-2b-it.safetensors + text_encoder/tokenizer.json  (gemma TE mirror)
///
/// Run:
///   SANA_PIPELINE_WEIGHTS=/path/Sana_1600M_1024px_diffusers \
///     cargo test -p mlx-gen-sana --release --test pipeline_contract \
///       -- --ignored --nocapture real_weight_1024_e2e
#[test]
#[ignore = "needs a Sana_1600M_1024px_diffusers snapshot; set SANA_PIPELINE_WEIGHTS"]
fn real_weight_1024_e2e() {
    use mlx_gen_sana::{SanaGenerateRequest, SanaPipeline, SanaTextEncoder};
    use std::path::PathBuf;

    let root =
        PathBuf::from(std::env::var("SANA_PIPELINE_WEIGHTS").expect("set SANA_PIPELINE_WEIGHTS"));

    let trunk_w =
        Weights::from_file(root.join("transformer/diffusion_pytorch_model.safetensors")).unwrap();
    let trunk =
        SanaTransformer::from_weights(&trunk_w, SanaTransformerConfig::sana_1600m()).unwrap();

    let dcfg = DcAeConfig::sana_f32c32();
    let vae_w = Weights::from_file(root.join("vae/diffusion_pytorch_model.safetensors")).unwrap();
    let decoder = DcAeDecoder::from_weights(&vae_w, dcfg.clone()).unwrap();

    let te = SanaTextEncoder::from_snapshot(root.join("text_encoder")).unwrap();

    let pipe = SanaPipeline::new(te, trunk, decoder, dcfg);
    let mut req = SanaGenerateRequest::new(
        "a photorealistic red panda sitting on a mossy log in a misty forest",
    );
    req.steps = Some(20);
    req.guidance_scale = Some(4.5);
    req.seed = Some(42);

    let img = pipe.generate(&req).expect("real-weight generate");
    assert_eq!(img.width, 1024);
    assert_eq!(img.height, 1024);
    assert_eq!(img.pixels.len(), 1024 * 1024 * 3);
    // All RGB8 bytes are finite-by-type; assert the image isn't a constant flat fill.
    let first = img.pixels[0];
    assert!(
        img.pixels.iter().any(|&p| p != first),
        "real-weight 1024² output is a constant fill — pipeline produced no signal"
    );
    println!("SANA real-weight 1024² gen OK: {} px", img.pixels.len());
}
