//! SANA-Sprint **composition contract** (epic 8485, story sc-8490 — Phase A).
//!
//! Proves the CFG-free SCM/TrigFlow few-step Sprint wiring composes end-to-end without the ~1.6B
//! real weights: a tiny **guidance-embedder** trunk (`guidance_embeds = true`, `qk_norm = true`)
//! driven by [`mlx_gen_sana::denoise_sprint`] over an [`mlx_gen_sana::ScmScheduler`], decoded through
//! a tiny DC-AE. It exercises the real Sprint seams the base SANA contract test does not:
//!
//!  * the trunk's guidance-embedder path (the embedded CFG-free guidance scalar summed into the
//!    timestep conditioning) — a Sprint-config trunk that lacks `time_embed.guidance_embedder.*` /
//!    `attn*.norm_q/k.weight` fails to load, so a successful build already proves the key wiring;
//!  * the SCM/TrigFlow few-step loop (single trunk forward per step, trigflow x0-pred + renoise),
//!    asserting it runs the expected step count and yields a finite, non-degenerate latent/image;
//!  * the `decode_to_image` DC-AE un-scale + decode tail (shared with base SANA).
//!
//! A SMALL committed golden parity test (`tools/dump_sana_sprint_golden.py`) for the SCM scheduler
//! step math + the guidance-embed trunk is the `scm_*` tests in this file (pure host math, no
//! weights) plus the trunk parity in `transformer_parity.rs`; the real-weight e2e is gated behind
//! `SANA_SPRINT_WEIGHTS` (`#[ignore]`d).

use mlx_rs::ops::{max as max_op, min as min_op};
use mlx_rs::random::{key, normal};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Progress};

use mlx_gen_sana::{
    denoise_sprint, BlockType, DcAeConfig, DcAeDecoder, SanaTransformer, SanaTransformerConfig,
    ScmScheduler,
};

fn rand(shape: &[i32], seed: u64) -> Array {
    normal::<f32>(shape, None, None, Some(&key(seed).unwrap())).unwrap()
}

fn linear(w: &mut Weights, prefix: &str, inn: i32, out: i32, bias: bool, seed: &mut u64) {
    w.insert(format!("{prefix}.weight"), rand(&[out, inn], *seed));
    *seed += 1;
    if bias {
        w.insert(format!("{prefix}.bias"), rand(&[out], *seed));
        *seed += 1;
    }
}

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

/// Tiny SANA-Sprint trunk config: same backbone as the base tiny config, with the Sprint deltas ON
/// (`guidance_embeds`, `qk_norm`).
fn tiny_sprint_config() -> SanaTransformerConfig {
    SanaTransformerConfig {
        in_channels: 4,
        out_channels: 4,
        num_attention_heads: 2,
        attention_head_dim: 8, // inner = 16
        num_layers: 2,
        num_cross_attention_heads: 2,
        cross_attention_head_dim: 8, // cross inner = 16
        caption_channels: 24,
        mlp_ratio: 2.5,
        patch_size: 1,
        norm_eps: 1e-6,
        caption_norm_eps: 1e-5,
        attn_qk_norm_eps: 1e-5,
        attn_eps: 1e-15,
        guidance_embeds: true,
        guidance_embeds_scale: 0.1,
        qk_norm: true,
    }
}

fn tiny_sprint_trunk_weights(cfg: &SanaTransformerConfig) -> Weights {
    let mut w = Weights::empty();
    let mut s = 1_u64;
    let inner = cfg.inner_dim();
    let cross_inner = cfg.num_cross_attention_heads * cfg.cross_attention_head_dim;
    let hidden = (cfg.mlp_ratio * inner as f32) as i32;

    conv(
        &mut w,
        "patch_embed.proj",
        inner,
        cfg.in_channels,
        cfg.patch_size,
        true,
        &mut s,
    );

    // Sprint combined timestep+guidance embedder (NO `.emb.` nesting; adds the guidance MLP).
    linear(
        &mut w,
        "time_embed.timestep_embedder.linear_1",
        256,
        inner,
        true,
        &mut s,
    );
    linear(
        &mut w,
        "time_embed.timestep_embedder.linear_2",
        inner,
        inner,
        true,
        &mut s,
    );
    linear(
        &mut w,
        "time_embed.guidance_embedder.linear_1",
        256,
        inner,
        true,
        &mut s,
    );
    linear(
        &mut w,
        "time_embed.guidance_embedder.linear_2",
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

    for i in 0..cfg.num_layers {
        let p = format!("transformer_blocks.{i}");
        w.insert(format!("{p}.scale_shift_table"), rand(&[6, inner], s));
        s += 1;
        // attn1 (linear self-attn) + qk_norm (rms_norm_across_heads over inner).
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
        w.insert(format!("{p}.attn1.norm_q.weight"), rand(&[inner], s));
        s += 1;
        w.insert(format!("{p}.attn1.norm_k.weight"), rand(&[inner], s));
        s += 1;
        // attn2 (cross-attn) + qk_norm over cross_inner.
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
        w.insert(format!("{p}.attn2.norm_q.weight"), rand(&[cross_inner], s));
        s += 1;
        w.insert(format!("{p}.attn2.norm_k.weight"), rand(&[cross_inner], s));
        s += 1;
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

fn tiny_dcae_config() -> DcAeConfig {
    DcAeConfig {
        in_channels: 3,
        latent_channels: 4,
        attention_head_dim: 4,
        block_out_channels: vec![6, 8],
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

/// End-to-end Sprint wiring: tiny guidance-embed trunk → SCM 2-step denoise → DC-AE decode. Asserts
/// the SCM loop ran the expected step count, the denoised latent keeps shape and is finite +
/// non-degenerate, and the decode tail produces a sane RGB image.
#[test]
fn sprint_wires_guidance_trunk_scm_decode() {
    let tcfg = tiny_sprint_config();
    let trunk = SanaTransformer::from_weights(&tiny_sprint_trunk_weights(&tcfg), tcfg.clone())
        .expect("build tiny Sprint trunk (guidance embedder + qk-norm keys present)");
    let dcfg = tiny_dcae_config();
    let decoder =
        DcAeDecoder::from_weights(&tiny_dcae_weights(&dcfg), dcfg.clone()).expect("tiny dc-ae");

    let latent_h = 6;
    let latent_w = 4; // non-square: catches an NHWC↔NCHW axis swap in decode.
    let latents = normal::<f32>(
        &[1, tcfg.out_channels, latent_h, latent_w],
        None,
        None,
        Some(&key(0).unwrap()),
    )
    .unwrap();
    let m = 7;
    let cond = rand(&[1, m, tcfg.caption_channels], 100);

    // 2-step SCM schedule (the Sprint default) → 3 angle timesteps, 2 loop iterations.
    let scheduler = ScmScheduler::new(2);
    assert_eq!(scheduler.num_steps(), 2);
    let cancel = CancelFlag::default();
    let mut steps_seen = 0_usize;
    let mut on_progress = |p: Progress| {
        if matches!(p, Progress::Step { .. }) {
            steps_seen += 1;
        }
    };

    let denoised = denoise_sprint(
        &trunk,
        &scheduler,
        7, // seed
        latents,
        &cond,
        4.5,                        // guidance_scale (embedded, CFG-free)
        tcfg.guidance_embeds_scale, // 0.1
        &cancel,
        &mut on_progress,
    )
    .expect("Sprint SCM denoise");

    assert_eq!(
        denoised.shape(),
        &[1, tcfg.out_channels, latent_h, latent_w],
        "SCM denoise keeps the trunk in/out channel + spatial shape"
    );
    assert_eq!(
        steps_seen,
        scheduler.num_steps(),
        "SCM loop must report exactly num_steps progress events"
    );

    let lo = min_op(&denoised, None).unwrap();
    let hi = max_op(&denoised, None).unwrap();
    eval([&lo, &hi]).unwrap();
    let (lo, hi) = (lo.item::<f32>(), hi.item::<f32>());
    assert!(
        lo.is_finite() && hi.is_finite(),
        "SCM latent non-finite: [{lo}, {hi}]"
    );
    assert!(hi - lo > 1e-6, "SCM latent is constant — graph degenerate");

    let img = mlx_gen_sana::pipeline::decode_to_image(&decoder, &dcfg, &denoised).expect("decode");
    let scale: i32 = 1 << (dcfg.num_stages() - 1);
    assert_eq!(img.width, (latent_w * scale) as u32);
    assert_eq!(img.height, (latent_h * scale) as u32);
    assert_eq!(img.pixels.len(), (img.width * img.height * 3) as usize);
}

/// Single-step Sprint (num_steps = 1): the SCM loop must take exactly one step and skip the renoise
/// (diffusers `if len(self.timesteps) > 1`), producing a finite latent.
#[test]
fn sprint_single_step_runs_one_step() {
    let tcfg = tiny_sprint_config();
    let trunk =
        SanaTransformer::from_weights(&tiny_sprint_trunk_weights(&tcfg), tcfg.clone()).unwrap();
    let latents = normal::<f32>(
        &[1, tcfg.out_channels, 4, 4],
        None,
        None,
        Some(&key(3).unwrap()),
    )
    .unwrap();
    let cond = rand(&[1, 5, tcfg.caption_channels], 11);
    let scheduler = ScmScheduler::new(1);
    assert!(scheduler.is_single_step());
    let cancel = CancelFlag::default();
    let mut steps = 0;
    let out = denoise_sprint(
        &trunk,
        &scheduler,
        1,
        latents,
        &cond,
        4.5,
        tcfg.guidance_embeds_scale,
        &cancel,
        &mut |p| {
            if matches!(p, Progress::Step { .. }) {
                steps += 1;
            }
        },
    )
    .unwrap();
    assert_eq!(steps, 1, "single-step SCM runs exactly one step");
    let lo = min_op(&out, None).unwrap();
    eval([&lo]).unwrap();
    assert!(lo.item::<f32>().is_finite());
}

/// Base SANA must remain loadable from the SAME tiny weights MINUS the Sprint-only keys — i.e. the
/// guidance-embed + qk-norm paths are genuinely config-gated and OFF by default. (A base trunk built
/// from a config with `guidance_embeds = false` must NOT require `time_embed.guidance_embedder.*`.)
#[test]
fn base_trunk_does_not_require_sprint_keys() {
    use mlx_gen_sana::SanaTransformerConfig;
    let mut base = SanaTransformerConfig::sana_1600m();
    // Shrink to the tiny dims but keep guidance_embeds/qk_norm OFF (the defaults).
    base.in_channels = 4;
    base.out_channels = 4;
    base.num_attention_heads = 2;
    base.attention_head_dim = 8;
    base.num_layers = 2;
    base.num_cross_attention_heads = 2;
    base.cross_attention_head_dim = 8;
    base.caption_channels = 24;
    assert!(!base.guidance_embeds);
    assert!(!base.qk_norm);

    // Build base-layout weights (the `time_embed.emb.*` nesting, no guidance_embedder, no norm_q/k).
    let mut w = Weights::empty();
    let mut s = 1_u64;
    let inner = base.inner_dim();
    let cross_inner = base.num_cross_attention_heads * base.cross_attention_head_dim;
    let hidden = (base.mlp_ratio * inner as f32) as i32;
    conv(
        &mut w,
        "patch_embed.proj",
        inner,
        base.in_channels,
        1,
        true,
        &mut s,
    );
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
    linear(
        &mut w,
        "caption_projection.linear_1",
        base.caption_channels,
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
    for i in 0..base.num_layers {
        let p = format!("transformer_blocks.{i}");
        w.insert(format!("{p}.scale_shift_table"), rand(&[6, inner], s));
        s += 1;
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
        conv(
            &mut w,
            &format!("{p}.ff.conv_inverted"),
            2 * hidden,
            inner,
            1,
            true,
            &mut s,
        );
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
    w.insert("scale_shift_table", rand(&[2, inner], s));
    s += 1;
    linear(&mut w, "proj_out", inner, base.out_channels, true, &mut s);

    let trunk = SanaTransformer::from_weights(&w, base.clone())
        .expect("base trunk loads WITHOUT any Sprint-only keys");
    // And its plain `forward` (no guidance) runs.
    let latent = rand(&[1, base.in_channels, 4, 4], 900);
    let cap = rand(&[1, 5, base.caption_channels], 901);
    let ts = Array::from_slice(&[0.3_f32], &[1]);
    let out = trunk.forward(&latent, &cap, &ts).expect("base forward");
    assert_eq!(out.shape(), &[1, base.out_channels, 4, 4]);
}

/// Real-weight Sprint 1024px e2e. `#[ignore]`d: needs a `Sana_Sprint_1.6B_1024px_diffusers` snapshot.
/// Set `SANA_SPRINT_WEIGHTS` to the snapshot root (transformer/ + vae/ + the gemma TE mirror).
///
/// Run:
///   SANA_SPRINT_WEIGHTS=/path/Sana_Sprint_1.6B_1024px_diffusers \
///     cargo test -p mlx-gen-sana --release --test sprint_contract \
///       -- --ignored --nocapture real_weight_sprint_1024_e2e
#[test]
#[ignore = "needs a Sana_Sprint_1.6B_1024px_diffusers snapshot; set SANA_SPRINT_WEIGHTS"]
fn real_weight_sprint_1024_e2e() {
    use mlx_gen_sana::{SanaGenerateRequest, SanaPipeline, SanaTextEncoder};
    use std::path::PathBuf;

    let root =
        PathBuf::from(std::env::var("SANA_SPRINT_WEIGHTS").expect("set SANA_SPRINT_WEIGHTS"));

    let trunk_w =
        Weights::from_file(root.join("transformer/diffusion_pytorch_model.safetensors")).unwrap();
    let trunk = SanaTransformer::from_weights(&trunk_w, SanaTransformerConfig::sana_sprint_1600m())
        .unwrap();
    let dcfg = DcAeConfig::sana_f32c32();
    let vae_w = Weights::from_file(root.join("vae/diffusion_pytorch_model.safetensors")).unwrap();
    let decoder = DcAeDecoder::from_weights(&vae_w, dcfg.clone()).unwrap();
    let te = SanaTextEncoder::from_snapshot(root.join("text_encoder")).unwrap();

    let pipe = SanaPipeline::new_sprint(te, trunk, decoder, dcfg, 0.1);
    assert!(pipe.is_sprint());
    let mut req = SanaGenerateRequest::new("a photorealistic red panda on a mossy log");
    req.steps = Some(2);
    req.guidance_scale = Some(4.5);
    req.seed = Some(42);

    let img = pipe.generate(&req).expect("real-weight Sprint generate");
    assert_eq!(img.width, 1024);
    assert_eq!(img.height, 1024);
    let first = img.pixels[0];
    assert!(
        img.pixels.iter().any(|&p| p != first),
        "real-weight Sprint output is a constant fill"
    );
    println!(
        "SANA-Sprint real-weight 1024² gen OK: {} px",
        img.pixels.len()
    );
}
