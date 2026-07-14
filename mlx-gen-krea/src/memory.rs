//! Krea 2 pose-control **memory-adaptation estimators** (sc-11750) — the per-lane peak estimators that
//! plug the Krea control lane into the shared, backend-neutral escalation policy
//! ([`mlx_gen::gen_core::mempolicy`]). This module holds the Krea/Qwen-VAE cost model; the escalation
//! order + budget comparison live in gen-core.
//!
//! Two shape-derived peaks drive the plan (both **excluding** the phase-A Qwen3-VL text encoder, whose
//! footprint the policy carries once as the residency lever):
//!   * the **control-denoise** peak — the resident heavy weights (base DiT + the bf16 pose branch + the
//!     VAE, all held through the heavy phase) plus the per-step activation working set of the
//!     concatenated single-stream forward with the N-block branch injected;
//!   * the **Qwen-VAE decode** peak — the same resident heavy weights plus the full-output decode spike
//!     through the `AutoencoderKLQwenImage` decoder stack (the sc-11747 target); its tiled floor is the
//!     resident weights + the assembled output buffers + one minimal tile.
//!
//! The weight terms are **first-principles** param counts (validated against the published Krea shapes:
//! ~11.1 B base @ 28×6144, ~3.0 B branch @ N=7 → ~6.1 GB bf16, matching the candle #480 profile's
//! ~6.6 GB), padded by a measured [`RESIDENT_OVERHEAD_GIB`] for the terms the block count omits (the
//! VAE, the DiT's non-block params, and the MLX Metal-allocator resident floor).
//!
//! **Measured-MLX calibration (sc-11847).** The activation + decode-spike coefficients and the resident
//! overhead were RE-FIT on real weights on a 128 GB M-series Metal Mac — the story's e2e gate — via
//! `tests/control_memory_calibration_real_weights.rs`, which measures the isolated denoise and decode
//! `mlx_rs::memory::get_peak_memory` high-water of a real `krea_2_turbo_control` render (base tier ∈
//! {bf16, q4}, resolution ∈ {512², 768², 1024²}, pose branch bf16, `Sequential` residency so the peaks
//! are ex-text). The candle #480 CUDA priors it replaced were **wrong for MLX in both directions**: the
//! coefficients ~8–10× (denoise) / ~4× (decode) too high, yet the estimate still UNDER-shot the real
//! peak at 512² because MLX's materialized-weight + framework resident floor (~33.4 GB bf16 / ~15.9 GB
//! q4) is ~4 GB above the bare param count — the CUDA activation coefficient had merely masked that gap
//! at 1024². The measured slopes are ~44 (bf16) / ~61 (q4) B/(token·hidden) for denoise and ~5211 B/px
//! for decode (tier-independent — the VAE decode is the same at every base tier). The constants below
//! keep the **over-predict / never-under-shoot** convention (an under-shoot is an OOM; an over-shoot
//! only tiles/adapts slightly sooner — the Wan sc-4998 / PiD sc-10087 guard): each is rounded up so the
//! estimate stays ≥ the measured peak (within ≤ ~1.16× at every tested point).

use mlx_gen::gen_core::mempolicy::{plan_memory_adaptation, LaneLevers, MemoryPlan, StagePeaks};
use mlx_gen::tiling::{budgeted_plan, TileCandidates, TilingConfig};
use mlx_gen::{Error, Quant, Result};

use crate::config::Krea2Config;

/// 1 GiB in bytes (`1024³`, matching MLX's `metal::malloc` GiB reporting / the core `memory` module).
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Effective bytes-per-parameter for each residency tier, including the group-wise affine quant
/// overhead (a per-64 group `scale` + `bias`, f16 each ⇒ `2·2/64 = 0.0625` B/param). bf16 is the dense
/// default the pose overlay ships as.
fn bytes_per_param(tier: Option<Quant>) -> f64 {
    match tier {
        None => 2.0,                     // bf16 dense
        Some(Quant::Q8) => 1.0 + 0.0625, // 8-bit + group scale/bias
        Some(Quant::Q4) => 0.5 + 0.0625, // 4-bit + group scale/bias
    }
}

/// A small safety multiplier on the counted parameters to cover the terms this first-principles count
/// omits (RMSNorm scales, the shared modulation projections, `img_in`/`time_embed`/final layers) —
/// deliberately over-counting so the resident-weight estimate never under-shoots the real footprint.
const PARAM_MARGIN: f64 = 1.1;

/// Dominant parameter count of ONE single-stream block: GQA attention (`q`,`k`,`v`,`out`) + the SwiGLU
/// FFN's three projections. `q_dim == hidden`; `kv_dim == num_kv_heads·head_dim`.
fn single_stream_block_params(cfg: &Krea2Config) -> f64 {
    let h = cfg.hidden_size as f64;
    let q = cfg.q_dim() as f64;
    let kv = cfg.kv_dim() as f64;
    let inter = cfg.intermediate_size as f64;
    // attn: q(h·q) + k(h·kv) + v(h·kv) + out(q·h); ffn SwiGLU: gate + up + down = 3·h·inter.
    2.0 * h * q + 2.0 * h * kv + 3.0 * h * inter
}

/// Resident weight bytes of the **base DiT** (`num_layers` single-stream blocks) at `tier`.
fn base_dit_bytes(cfg: &Krea2Config, tier: Option<Quant>) -> f64 {
    let params = cfg.num_layers as f64 * single_stream_block_params(cfg) * PARAM_MARGIN;
    params * bytes_per_param(tier)
}

/// Resident weight bytes of the **pose control branch**: `n` copied single-stream blocks, each plus a
/// `proj_out` (`hidden·hidden`) zero-init output projection.
fn branch_bytes(cfg: &Krea2Config, n: usize, tier: Option<Quant>) -> f64 {
    let h = cfg.hidden_size as f64;
    let per_block = single_stream_block_params(cfg) + h * h;
    let params = n as f64 * per_block * PARAM_MARGIN;
    params * bytes_per_param(tier)
}

/// Fixed resident **overhead** (GiB) the first-principles block count does NOT capture, added to both
/// stage peaks: the Qwen-Image VAE (`AutoencoderKLQwenImage`, f32), the DiT's non-block params
/// (`img_in`/`time_embed`/text-fusion aggregator/`final_layer`/modulation), AND the MLX Metal-allocator
/// resident floor (materialized-weight buffers + the retained working set). **Measured-MLX (sc-11847),
/// not the old ~0.4 GiB VAE-only guess:** the real ex-text resident floor is ~33.4 GiB (bf16) / ~15.9
/// GiB (q4), ~4 GiB above `base_dit + branch` alone; `5.5` covers that residual at every tested point
/// (the bf16 decode-512² point binds it) with a small over-predict margin. Tier-independent to first
/// order (the VAE stays f32; the uncounted DiT params + allocator floor barely move with the base tier —
/// bf16 needs ~4.5, q4 ~4.1), so a single constant is both simpler and conservative.
const RESIDENT_OVERHEAD_GIB: f64 = 5.5;

/// Text/vision-encoder (Qwen3-VL-4B, ~4 B params) resident bytes at `tier` — the phase-A footprint the
/// residency lever frees. Packs with the base tier (sc-11727 `load_krea_text`).
fn text_resident_gib(tier: Option<Quant>) -> f64 {
    const TEXT_PARAMS: f64 = 4.0e9;
    TEXT_PARAMS * bytes_per_param(tier) / GIB
}

/// Per-step denoise **activation** bytes per (token · hidden) element — the concatenated-stream
/// activations + the (fused-SDPA) attention working set + the N-block branch forward, on top of the
/// resident weights. **Measured-MLX (sc-11847):** the real slope is ~44 (bf16) / ~61 (q4)
/// B/(token·hidden) — MLX's fused SDPA + the CFG-free single forward keep the denoise peak
/// resident-weight-dominated, far below the candle #480 CUDA prior (470, from ~11 GB @ 1024²). `80`
/// rounds up over the larger (q4) measured slope with headroom for higher resolutions.
const DENOISE_ACT_BYTES_PER_TOKEN_HIDDEN: f64 = 80.0;

/// Decode **spike** bytes per output pixel through the Qwen-VAE decoder conv stack — the transient that
/// tiling (sc-11747) shrinks. **Measured-MLX (sc-11847):** the real slope is ~5211 B/px, tier-independent
/// (the VAE decode is the same at every base tier), far below the candle #480 CUDA prior (22500, from
/// ~22 GB @ 1024²). `6500` rounds up over the measured slope; the fixed VAE-materialization part of the
/// decode peak lives in [`RESIDENT_OVERHEAD_GIB`], so this term is the pure per-pixel conv growth.
const DECODE_SPIKE_BYTES_PER_PIXEL: f64 = 6_500.0;

/// The assembled full-resolution RGB output buffers held across a tiled decode (`output [1,3,H,W]` +
/// blend `weights`), f32 — the term tiling can NOT shrink. ~16 B/px (12 for the 3-channel output + a
/// 1-channel weight accumulator + margin).
const DECODE_ACCUM_BYTES_PER_PIXEL: f64 = 16.0;

/// The working set of ONE minimal decode tile — the least the tiled decode can spike to, on top of the
/// resident weights + output buffers. Resolution-independent (a fixed tile), so it is the decode floor's
/// only non-buffer term. At the measured [`DECODE_SPIKE_BYTES_PER_PIXEL`] a ~256²-tile conv forward is
/// only ~0.4 GiB, so `1.5` is deliberately conservative; the tiled floor is NOT exercised until the
/// decode-tiling lever lands (sc-11747), so a precise re-fit of this floor is deferred to that story
/// (which drives a real tiled decode). Over-predicting the floor is the safe direction (it only makes
/// the policy prefer resolution reduction slightly sooner, never OOM).
const DECODE_MIN_TILE_GIB: f64 = 1.5;

/// Denoise-forward token count for a `width × height` render: the latent is `[16, H/8, W/8]`,
/// patchified 2×2 → `(H/16)·(W/16)` image tokens (the text tokens are a negligible add).
fn denoise_tokens(width: u32, height: u32) -> f64 {
    (width as f64 / 16.0).floor() * (height as f64 / 16.0).floor()
}

/// The **control-denoise** stage peak (GiB, ex-text): resident heavy weights + the activation working
/// set at `width × height`. Pure (shape + config only) → unit-testable.
pub fn control_denoise_peak_ex_text_gib(
    cfg: &Krea2Config,
    branch_blocks: usize,
    base_tier: Option<Quant>,
    branch_tier: Option<Quant>,
    width: u32,
    height: u32,
) -> f64 {
    let heavy = base_dit_bytes(cfg, base_tier) + branch_bytes(cfg, branch_blocks, branch_tier);
    let act =
        DENOISE_ACT_BYTES_PER_TOKEN_HIDDEN * denoise_tokens(width, height) * cfg.hidden_size as f64;
    (heavy / GIB) + RESIDENT_OVERHEAD_GIB + act / GIB
}

/// The single-pass **Qwen-VAE decode** stage peak (GiB, ex-text): resident heavy weights + the
/// full-output decode spike. Pure.
pub fn qwen_vae_decode_peak_ex_text_gib(
    cfg: &Krea2Config,
    branch_blocks: usize,
    base_tier: Option<Quant>,
    branch_tier: Option<Quant>,
    width: u32,
    height: u32,
) -> f64 {
    let heavy = base_dit_bytes(cfg, base_tier) + branch_bytes(cfg, branch_blocks, branch_tier);
    let px = width as f64 * height as f64;
    (heavy / GIB) + RESIDENT_OVERHEAD_GIB + (DECODE_SPIKE_BYTES_PER_PIXEL * px) / GIB
}

/// The **tiled** Qwen-VAE decode floor (GiB, ex-text): resident heavy weights + the un-shrinkable
/// full-output buffers + one minimal tile. The least [`plan_memory_adaptation`]'s decode-tiling lever
/// can drive the decode peak toward (`budgeted_plan` then sizes the actual tile). Pure.
pub fn qwen_vae_decode_tiled_floor_ex_text_gib(
    cfg: &Krea2Config,
    branch_blocks: usize,
    base_tier: Option<Quant>,
    branch_tier: Option<Quant>,
    width: u32,
    height: u32,
) -> f64 {
    let heavy = base_dit_bytes(cfg, base_tier) + branch_bytes(cfg, branch_blocks, branch_tier);
    let px = width as f64 * height as f64;
    (heavy / GIB)
        + RESIDENT_OVERHEAD_GIB
        + (DECODE_ACCUM_BYTES_PER_PIXEL * px) / GIB
        + DECODE_MIN_TILE_GIB
}

/// Candidate spatial tile sizes (OUTPUT px, multiples of the Qwen-Image VAE's ×8 spatial scale, blended
/// with a 64 px overlap) offered to [`budgeted_plan`] for the tiled decode. Ordered largest-first for
/// readability only — the selector keeps the largest-volume tile that fits regardless of position. The
/// grid spans near-full (768) down to a small 256 tile so a constrained budget always has a fitting
/// option; on Metal the decode spike is modest (~5.2 KB/px, sc-11847), so even large tiles shave the
/// peak toward the un-tileable floor.
const QWEN_DECODE_SPATIAL_PX: [i32; 7] = [768, 640, 512, 448, 384, 320, 256];

/// **Budget-gated** tiling for the Krea control-lane Qwen-VAE decode (sc-11747) — the still-image
/// analogue of Wan's `auto_tiling_budgeted_z16`. Estimates the decode peak from the render shape and
/// returns the tiling the render-time decode should use:
///   • `Ok(None)`    — the single-pass decode already fits `safe_gib` (small image / large-memory Mac);
///                     the caller runs [`QwenVae::decode`](mlx_gen_qwen_image::vae::QwenVae::decode), so
///                     single-pass is reached ONLY when safe and a machine with headroom pays ZERO
///                     tiling overhead (the sc-11750 guarantee).
///   • `Ok(Some(c))` — tiling is required; `c` sizes the LARGEST tile whose estimated peak ≤ `safe_gib`
///                     (largest ⇒ fewest tiles ⇒ least overlap recompute ⇒ fastest within budget).
///   • `Err(..)`     — infeasible even tiled (the resident weights + output buffers alone, or every
///                     candidate tile, exceed `safe_gib`): a **catchable** error surfaced BEFORE the
///                     decode, so the caller reports it rather than the OS/GPU killing the process
///                     mid-decode. On the control lane the cheaper residency / branch-quant levers
///                     (applied at load) should already have made this feasible; this is the backstop.
///
/// The decode `peak_cost` is the Krea/Qwen cost model — and unlike Wan's video cost model it MUST carry
/// the resident heavy weights, because on Metal they DOMINATE the decode peak (sc-11847: the ~5.2 KB/px
/// spike is only ~6 GiB @ 1024², the resident weights are the other ~16+ GiB); omitting them would let
/// the single-pass gate under-count and never tile. The model is: `resident heavy (base DiT + branch) +
/// fixed overhead (+ co-resident text) + the full-output f32 accumulators (unshrinkable) + the per-TILE
/// conv spike`. Every constant is shared with the single-pass / tiled-floor estimators above, so this
/// gate agrees with the [`plan_control_adaptation`] policy on whether single-pass fits.
///
/// `text_co_resident` adds the phase-A Qwen3-VL encoder footprint when it stays resident through the
/// decode (the `Resident` path); under `Sequential` it is dropped before the heavy phase, matching the
/// `*_ex_text_gib` semantics. `safe_gib` is injected ([`mlx_gen::memory::safe_budget_gib`] at the call
/// site) so the gate is unit-testable without a device.
#[allow(clippy::too_many_arguments)]
pub fn plan_control_decode_tiling(
    safe_gib: f64,
    cfg: &Krea2Config,
    branch_blocks: usize,
    base_tier: Option<Quant>,
    branch_tier: Option<Quant>,
    width: u32,
    height: u32,
    text_co_resident: bool,
) -> Result<Option<TilingConfig>> {
    let heavy_gib =
        (base_dit_bytes(cfg, base_tier) + branch_bytes(cfg, branch_blocks, branch_tier)) / GIB;
    let resident_text = if text_co_resident {
        text_resident_gib(base_tier)
    } else {
        0.0
    };
    let resident = heavy_gib + RESIDENT_OVERHEAD_GIB + resident_text;

    // peak_cost(out_f, out_h, out_w, tile_f, tile_h, tile_w): the still-image decode has out_f = 1; a
    // zero tile yields the accumulator-only floor (`budgeted_plan`'s AccumulatorsExceedBudget probe).
    let peak_cost = move |_of: i64, oh: i64, ow: i64, _tf: i64, th: i64, tw: i64| {
        let out_px = (oh * ow) as f64;
        let tile_px = (th * tw) as f64;
        resident
            + (DECODE_ACCUM_BYTES_PER_PIXEL * out_px) / GIB
            + (DECODE_SPIKE_BYTES_PER_PIXEL * tile_px) / GIB
    };
    let candidates = TileCandidates {
        spatial_px: &QWEN_DECODE_SPATIAL_PX,
        spatial_overlap_px: 64,
        temporal: &[], // still image — no temporal axis to tile
    };
    budgeted_plan(
        height as i32,
        width as i32,
        1, // out_frames
        safe_gib,
        candidates,
        peak_cost,
    )
    .map_err(|e| Error::Msg(format!("krea_2 control Qwen-VAE decode: {e}")))
}

/// Everything the Krea control lane needs to decide its memory-adaptation plan against a device budget.
/// The caller fills this from the load spec + request; [`plan_control_adaptation`] turns it into a
/// [`MemoryPlan`] via the shared policy.
#[derive(Clone, Copy, Debug)]
pub struct ControlLaneInputs<'a> {
    /// Architecture config (block count, hidden width, FFN width).
    pub cfg: &'a Krea2Config,
    /// Copied branch blocks `N` (`Krea2ControlBranch::num_blocks`; the S0 recipe is 7).
    pub branch_blocks: usize,
    /// The base DiT / text-encoder quant tier — `None` = dense bf16, `Some(Q4/Q8)` = packed (sc-11727).
    pub base_tier: Option<Quant>,
    /// Requested render size.
    pub width: u32,
    pub height: u32,
    /// Whether the lane can drop to `OffloadPolicy::Sequential` (the Krea control lane wires it).
    pub supports_sequential: bool,
    /// Whether the pose branch may be packed to the base tier at load (sc-11748). `true` ⇒ the
    /// branch-quant lever is offered (bf16 → `base_tier`); `false` ⇒ the branch stays bf16.
    pub allow_branch_quant: bool,
    /// Candidate resolution scales (largest-first, in `(0,1)`) offered as the last resort (sc-11749).
    pub resolution_scales: &'a [f64],
}

/// Decide the Krea control lane's memory-adaptation plan (sc-11750): assemble the per-lane
/// [`StagePeaks`] estimator + [`LaneLevers`] and run them through the shared
/// [`plan_memory_adaptation`]. `safe_gib` is the device budget ([`mlx_gen::memory::safe_budget_gib`]);
/// the returned [`MemoryPlan`] tells the caller which levers to apply (residency / decode tiling /
/// branch quant / resolution) and whether the render fits at all.
pub fn plan_control_adaptation(safe_gib: f64, inputs: ControlLaneInputs<'_>) -> MemoryPlan {
    // The branch packs to the base tier when the lever is offered (and the base is itself packed);
    // otherwise it stays bf16. `branch_quant_saves` is the bf16 → base-tier delta on the branch weights.
    let branch_quant_saves_gib = if inputs.allow_branch_quant && inputs.base_tier.is_some() {
        (branch_bytes(inputs.cfg, inputs.branch_blocks, None)
            - branch_bytes(inputs.cfg, inputs.branch_blocks, inputs.base_tier))
            / GIB
    } else {
        0.0
    };

    let levers = LaneLevers {
        text_resident_gib: text_resident_gib(inputs.base_tier),
        supports_sequential: inputs.supports_sequential,
        branch_quant_saves_gib,
        resolution_scales: inputs.resolution_scales,
    };

    plan_memory_adaptation(safe_gib, levers, |scale| {
        // Scale the render dimensions; keep them multiple-of-16 legal (the DiT/VAE alignment) by
        // flooring to the nearest 16 — the same alignment the pipeline validates.
        let w = (((inputs.width as f64 * scale) as u32) / 16 * 16).max(16);
        let h = (((inputs.height as f64 * scale) as u32) / 16 * 16).max(16);
        // The branch is estimated at bf16 here; the branch-quant *saving* is applied by the policy via
        // `branch_quant_saves_gib`, so the peaks must NOT also pre-apply it (double counting).
        StagePeaks {
            denoise_ex_text_gib: control_denoise_peak_ex_text_gib(
                inputs.cfg,
                inputs.branch_blocks,
                inputs.base_tier,
                None,
                w,
                h,
            ),
            decode_ex_text_gib: qwen_vae_decode_peak_ex_text_gib(
                inputs.cfg,
                inputs.branch_blocks,
                inputs.base_tier,
                None,
                w,
                h,
            ),
            decode_tiled_floor_ex_text_gib: Some(qwen_vae_decode_tiled_floor_ex_text_gib(
                inputs.cfg,
                inputs.branch_blocks,
                inputs.base_tier,
                None,
                w,
                h,
            )),
        }
    })
}

/// Map a base-DiT quant width to the [`Quant`] tier the pose branch matches (`4 → Q4`, `8 → Q8`); any
/// other width has no tier (the branch stays bf16).
fn tier_from_bits(bits: i32) -> Option<Quant> {
    match bits {
        4 => Some(Quant::Q4),
        8 => Some(Quant::Q8),
        _ => None,
    }
}

/// Decide whether to pack the pose control branch to the base tier at LOAD time (sc-11748) — the
/// branch-quant lever's mechanism half. The branch is a resident weight that cannot be re-packed
/// mid-render, so the decision must hold for the largest render the loaded model can serve: this runs
/// the shared [`plan_control_adaptation`] policy at the lane's worst-case resolution (`max_size`²) and
/// returns [`MemoryPlan::quantize_branch`].
///
/// - `base_bits` `None` (dense bf16 base) or a non-Q4/Q8 width ⇒ `false`: there is no base tier for the
///   branch to match (the branch only ever packs to the base's tier).
/// - A machine with headroom ⇒ `false`: the branch stays bf16 (no dequant-on-forward) — the sc-11750
///   "large-memory Mac pays zero overhead" guarantee.
/// - A constrained Mac whose projected footprint won't fit even after the cheaper residency /
///   decode-tiling levers ⇒ `true`: pack the branch to the base tier.
///
/// `resolution_scales` is deliberately empty — resolution reduction is a separate lever (sc-11749); this
/// gate stands on residency + decode-tiling + branch-quant, so it never assumes a resolution reduction
/// this call would not apply. `safe_gib` is injected ([`mlx_gen::memory::safe_budget_gib`] at the call
/// site) so the decision is unit-testable without a device.
pub fn should_quantize_control_branch(
    safe_gib: f64,
    cfg: &Krea2Config,
    branch_blocks: usize,
    base_bits: Option<i32>,
    max_size: u32,
) -> bool {
    let Some(base_tier) = base_bits.and_then(tier_from_bits) else {
        return false;
    };
    plan_control_adaptation(
        safe_gib,
        ControlLaneInputs {
            cfg,
            branch_blocks,
            base_tier: Some(base_tier),
            width: max_size,
            height: max_size,
            supports_sequential: true,
            allow_branch_quant: true,
            resolution_scales: &[],
        },
    )
    .quantize_branch
}

/// Candidate render-resolution scales for the LAST-RESORT resolution lever (sc-11749), largest-first.
/// Full resolution (`1.0`) is always tried first by [`plan_control_resolution`], so a machine with
/// headroom pays ZERO resolution cost; these are the fallbacks tried only when the un-tileable denoise
/// activation peak is still over budget after residency + decode tiling + branch quant. `0.75²`/`0.5²`
/// cut the activation area to ~56% / 25% — the sc-11750 ladder's resolution rung.
const CONTROL_RESOLUTION_SCALES: [f64; 2] = [0.75, 0.5];

/// Scale a render dimension by `scale` and floor to a legal multiple of 16 (the DiT/VAE patch/×8-VAE
/// alignment [`crate::pipeline`] validates), never below one 16-px block.
fn scale_render_dim(dim: u32, scale: f64) -> u32 {
    (((dim as f64 * scale) as u32) / 16 * 16).max(16)
}

/// Decide the render resolution for the Krea control lane (sc-11749) — the render-time RESOLUTION lever,
/// the LAST-RESORT rung of the sc-11750 escalation ladder (the only lever that costs image quality, so it
/// engages only after the free/cheap ones). By the time this runs the cheaper levers are already applied:
/// **Sequential residency** was selected at load (the worker fit-gate `apply_residency_policy`, epic
/// 10834 — reflected here by `text_co_resident`), **branch quant** was decided at load (sc-11748 —
/// reflected by `branch_tier`), and **decode tiling** engages right after (sc-11747). This returns the
/// LARGEST 16-aligned resolution ≤ the requested one whose two shape-derived peaks both fit `safe_gib`:
///   * the **un-tileable DENOISE** activation peak — the sc-11749 target (candle #480's ~11 GiB @ 1024²);
///     resolution is the only lever that shrinks it once residency + branch quant are spent, and
///   * the **tiled Qwen-VAE DECODE** floor — the best the decode can reach at this resolution (decode
///     tiling runs next), so the resolution lever never fires for a spike tiling alone would absorb.
///
/// Both peaks are computed at the ACTUAL `base_tier`/`branch_tier` (a resident weight can't be re-packed
/// mid-render), so this never assumes a quant saving the loaded model can't realize — unlike the
/// load-time [`plan_control_adaptation`], which estimates the branch bf16 and treats packing as a future
/// lever. `text_co_resident` adds the phase-A Qwen3-VL encoder footprint when it stays resident (the
/// `Resident` path); under `Sequential` it was dropped before the heavy phase, matching the
/// `*_ex_text_gib` estimators.
///
/// - `Ok((width, height))` **unchanged** when the request already fits — the common case (a large-memory
///   Mac, or a 32 GB Mac where residency + decode tiling suffice at full res per the sc-11847 calibration)
///   — so a machine with headroom renders at exactly the requested size (the sc-11750 zero-overhead
///   guarantee).
/// - `Ok((w, h))` with a smaller 16-aligned size (aspect ratio preserved) when a reduction is forced.
/// - `Err` when even the smallest scale is over budget — a **catchable** pre-render error (like the
///   decode-tiling gate's), surfaced before the render rather than an OS/GPU kill mid-run.
///
/// `safe_gib` is injected ([`mlx_gen::memory::safe_budget_gib`] at the call site) so the decision is
/// unit-testable without a device.
#[allow(clippy::too_many_arguments)]
pub fn plan_control_resolution(
    safe_gib: f64,
    cfg: &Krea2Config,
    branch_blocks: usize,
    base_tier: Option<Quant>,
    branch_tier: Option<Quant>,
    width: u32,
    height: u32,
    text_co_resident: bool,
) -> Result<(u32, u32)> {
    let text = if text_co_resident {
        text_resident_gib(base_tier)
    } else {
        0.0
    };
    // A resolution fits when BOTH the un-tileable denoise peak and the tiled decode floor (decode tiling
    // engages next, sc-11747) are within budget at that size — the text encoder counted only when it
    // stays co-resident (`Resident`; `Sequential` dropped it before the heavy phase).
    let fits = |w: u32, h: u32| {
        let denoise =
            control_denoise_peak_ex_text_gib(cfg, branch_blocks, base_tier, branch_tier, w, h)
                + text;
        let decode = qwen_vae_decode_tiled_floor_ex_text_gib(
            cfg,
            branch_blocks,
            base_tier,
            branch_tier,
            w,
            h,
        ) + text;
        denoise <= safe_gib && decode <= safe_gib
    };
    // Full resolution first: a machine with headroom never reduces (the sc-11750 zero-overhead guarantee).
    if fits(width, height) {
        return Ok((width, height));
    }
    for &scale in &CONTROL_RESOLUTION_SCALES {
        let (w, h) = (
            scale_render_dim(width, scale),
            scale_render_dim(height, scale),
        );
        if fits(w, h) {
            return Ok((w, h));
        }
    }
    Err(Error::Msg(format!(
        "krea_2 control render at {width}×{height} exceeds the ~{safe_gib:.0} GiB unified-memory budget \
         even at the smallest supported resolution (with the text encoder offloaded, the decode tiled, \
         and the pose branch packed). Lower the output resolution or run on a Mac with more memory."
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::mempolicy::Lever;
    use mlx_gen::OffloadPolicy;

    const SCALES: [f64; 2] = [0.75, 0.5];

    fn inputs(base_tier: Option<Quant>) -> ControlLaneInputs<'static> {
        // `Krea2Config` is not `'static`; the tests build one and leak it once (test-only) so the
        // `'a` borrow lives long enough for the `'static` return. Simpler than threading a lifetime
        // through every case.
        static CFG: std::sync::OnceLock<Krea2Config> = std::sync::OnceLock::new();
        let cfg = CFG.get_or_init(Krea2Config::turbo);
        ControlLaneInputs {
            cfg,
            branch_blocks: 7,
            base_tier,
            width: 1024,
            height: 1024,
            supports_sequential: true,
            allow_branch_quant: true,
            resolution_scales: &SCALES,
        }
    }

    /// The first-principles weight counts land on the published anchors: ~11 B base params, and a bf16
    /// branch of ~6 GB (candle #480 ~6.6 GB) — proof the cost model is grounded, not arbitrary.
    #[test]
    fn weight_estimates_match_published_anchors() {
        let cfg = Krea2Config::turbo();
        let base_params = cfg.num_layers as f64 * single_stream_block_params(&cfg);
        assert!(
            (10.5e9..12.5e9).contains(&base_params),
            "base ≈ 11 B params, got {base_params:.3e}"
        );
        let branch_gib = branch_bytes(&cfg, 7, None) / GIB;
        assert!(
            (5.5..7.5).contains(&branch_gib),
            "bf16 branch ≈ 6–7 GiB (candle #480 ~6.6), got {branch_gib:.2}"
        );
        // Packing the branch to q4 saves the bulk of that.
        let saved = (branch_bytes(&cfg, 7, None) - branch_bytes(&cfg, 7, Some(Quant::Q4))) / GIB;
        assert!(saved > 4.0, "q4 branch saving ≈ 4–5 GiB, got {saved:.2}");
    }

    /// Peaks grow with resolution (more tokens, more pixels) — a sanity guard on the shape scaling.
    #[test]
    fn peaks_grow_with_resolution() {
        let cfg = Krea2Config::turbo();
        let dn_512 = control_denoise_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 512, 512);
        let dn_1024 = control_denoise_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 1024, 1024);
        assert!(dn_1024 > dn_512);
        let dc_512 = qwen_vae_decode_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 512, 512);
        let dc_1024 = qwen_vae_decode_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 1024, 1024);
        assert!(dc_1024 > dc_512);
        // The single-pass decode peak exceeds its tiled floor (tiling's whole point). Note: on measured
        // MLX (sc-11847) the decode spike is only ~6 GiB @ 1024² — far smaller than the candle #480 CUDA
        // ~22 GiB — so tiling shaves ~5 GiB (~20%) off the peak, NOT the ~50% the CUDA prior implied. The
        // resident weights + overhead dominate the decode peak on Metal, so tiling is a modest lever here.
        let floor =
            qwen_vae_decode_tiled_floor_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 1024, 1024);
        assert!(
            floor < dc_1024 && (dc_1024 - floor) > 3.0,
            "tiled floor {floor:.1} must be a real reduction below single-pass {dc_1024:.1}"
        );
    }

    /// Large-memory Mac (128 GB → ~100 GiB safe): the fast path — NOTHING engages, full res, Resident,
    /// bf16 branch, single-pass decode. The sc-11750 regression guard, end to end through the Krea
    /// estimators (not just the synthetic gen-core model).
    #[test]
    fn large_memory_mac_engages_nothing() {
        let plan = plan_control_adaptation(100.0, inputs(Some(Quant::Q4)));
        assert!(plan.engaged.is_empty(), "no lever on a big Mac: {plan:?}");
        assert_eq!(plan.residency, OffloadPolicy::Resident);
        assert!(!plan.tile_decode && !plan.quantize_branch);
        assert_eq!(plan.resolution_scale, 1.0);
        assert!(plan.feasible);
    }

    /// A 32 GB Mac (~24 GiB usable) with a q4 base: levers engage in cost order and the render fits.
    /// The single-pass decode peak (~24.5 GiB ex-text on measured MLX, sc-11847) is over budget once the
    /// text phase is dropped → residency then decode tiling must engage.
    #[test]
    fn constrained_mac_engages_levers_in_cost_order_and_fits() {
        let plan = plan_control_adaptation(24.0, inputs(Some(Quant::Q4)));
        assert!(
            plan.feasible,
            "a 32GB Mac must fit a q4 control render: {plan:?}"
        );
        assert!(
            !plan.engaged.is_empty(),
            "something must engage under 24 GiB"
        );
        // Cost order: whatever engaged is ascending (never quant before residency, etc.).
        let mut sorted = plan.engaged.clone();
        sorted.sort();
        assert_eq!(
            plan.engaged, sorted,
            "levers must be cost-ordered: {plan:?}"
        );
        // The single-pass decode peak is the binding peak over budget → decode tiling is engaged.
        assert!(
            plan.tile_decode,
            "decode tiling must engage on a 32GB Mac: {plan:?}"
        );
        assert!(
            plan.projected_peak_gib <= 24.0,
            "projected peak over budget: {plan:?}"
        );
    }

    /// Residency is tried before the quality-costing resolution lever: a mid budget engages the cheap
    /// levers (residency/tiling/quant) but keeps full resolution.
    #[test]
    fn keeps_full_resolution_until_forced() {
        // 24 GiB fits with the cheap levers (proven above) → resolution must stay 1.0.
        let plan = plan_control_adaptation(24.0, inputs(Some(Quant::Q4)));
        assert_eq!(
            plan.resolution_scale, 1.0,
            "must not drop res prematurely: {plan:?}"
        );
        assert!(!plan.engaged.contains(&Lever::ResolutionReduction));
    }

    // ── sc-11747: the render-time decode-tiling gate (`plan_control_decode_tiling`). ───────────────

    /// A large-memory Mac (or any small render) → single-pass fits → `None`, so the caller runs the
    /// untiled decode and pays ZERO tiling overhead (the sc-11750 fast-path guarantee).
    #[test]
    fn decode_tiling_gate_none_when_single_pass_fits() {
        let cfg = Krea2Config::turbo();
        let sp = qwen_vae_decode_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 1024, 1024);
        // A budget comfortably above the single-pass peak → no tiling.
        let plan =
            plan_control_decode_tiling(sp + 2.0, &cfg, 7, Some(Quant::Q4), None, 1024, 1024, false)
                .unwrap();
        assert!(plan.is_none(), "single-pass fits → must not tile: {plan:?}");
    }

    /// A constrained budget under the single-pass peak but above the tiled floor → tiling engages with a
    /// spatial tile, and the selector guarantees the chosen tile's estimated peak ≤ the budget.
    #[test]
    fn decode_tiling_gate_tiles_when_single_pass_over_budget() {
        let cfg = Krea2Config::turbo();
        let sp = qwen_vae_decode_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 1024, 1024);
        // A few GiB under single-pass: the 1024² decode spike (~6 GiB) means a smaller tile fits here.
        let safe = sp - 3.0;
        let plan =
            plan_control_decode_tiling(safe, &cfg, 7, Some(Quant::Q4), None, 1024, 1024, false)
                .unwrap()
                .expect("single-pass over budget must tile");
        assert!(
            plan.spatial.is_some(),
            "a 1024² decode over budget must tile the spatial axis: {plan:?}"
        );
        assert!(
            plan.temporal.is_none(),
            "a still image has no temporal axis to tile: {plan:?}"
        );
    }

    /// Co-resident text (the `Resident` path) RAISES the decode peak, so a budget that fits single-pass
    /// ex-text can tip into tiling once the text encoder is counted. Proves the `text_co_resident` term
    /// is wired.
    #[test]
    fn decode_tiling_gate_counts_coresident_text() {
        let cfg = Krea2Config::turbo();
        let sp = qwen_vae_decode_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 1024, 1024);
        // Just above the ex-text single-pass peak: fits with text dropped, over budget with it resident.
        let safe = sp + 1.0;
        let seq =
            plan_control_decode_tiling(safe, &cfg, 7, Some(Quant::Q4), None, 1024, 1024, false)
                .unwrap();
        assert!(
            seq.is_none(),
            "ex-text single-pass fits under Sequential: {seq:?}"
        );
        let resident =
            plan_control_decode_tiling(safe, &cfg, 7, Some(Quant::Q4), None, 1024, 1024, true)
                .unwrap();
        assert!(
            resident.is_some(),
            "co-resident text pushes the decode over budget → must tile: {resident:?}"
        );
    }

    /// Infeasible even tiled (a budget below the resident weights + output-buffer floor) → a catchable
    /// `Err`, surfaced before the decode rather than an OS/GPU kill mid-decode.
    #[test]
    fn decode_tiling_gate_errs_when_infeasible() {
        let cfg = Krea2Config::turbo();
        let err =
            plan_control_decode_tiling(1.0, &cfg, 7, Some(Quant::Q4), None, 1024, 1024, false)
                .unwrap_err()
                .to_string();
        assert!(
            err.contains("Qwen-VAE decode"),
            "over-budget decode must surface a catchable, tagged error: {err}"
        );
    }

    // ── sc-11748: the load-time branch-quant gate (`should_quantize_control_branch`). ──────────────

    /// A dense bf16 base (or a non-Q4/Q8 width) never packs the branch — there is no base tier to match,
    /// even under a starved budget.
    #[test]
    fn branch_quant_gate_dense_base_never_packs() {
        let cfg = Krea2Config::turbo();
        assert!(!should_quantize_control_branch(2.0, &cfg, 7, None, 1024));
        assert!(!should_quantize_control_branch(
            2.0,
            &cfg,
            7,
            Some(16),
            1024
        ));
    }

    /// The sc-11750 hard requirement: a large-memory Mac keeps the branch bf16 even with a packed base,
    /// at either tier and up to the worst-case resolution — no dequant-on-forward tax on a machine that
    /// does not need it.
    #[test]
    fn branch_quant_gate_large_memory_keeps_bf16() {
        let cfg = Krea2Config::turbo();
        assert!(!should_quantize_control_branch(
            100.0,
            &cfg,
            7,
            Some(4),
            2048
        ));
        assert!(!should_quantize_control_branch(
            100.0,
            &cfg,
            7,
            Some(8),
            2048
        ));
    }

    /// A constrained Mac whose worst-case footprint won't fit even after residency + decode tiling packs
    /// the branch to the base tier (the deeper-escalation lever engages).
    #[test]
    fn branch_quant_gate_constrained_mac_packs() {
        let cfg = Krea2Config::turbo();
        assert!(should_quantize_control_branch(12.0, &cfg, 7, Some(4), 2048));
    }

    // ── sc-11749: the render-time resolution lever (`plan_control_resolution`). ─────────────────────

    /// A large-memory Mac (or any budget above the full-res peaks) keeps the requested resolution — the
    /// sc-11750 zero-overhead guarantee: no reduction, exact requested size returned.
    #[test]
    fn resolution_lever_keeps_full_res_with_headroom() {
        let cfg = Krea2Config::turbo();
        // 100 GiB dwarfs even the 1024² bf16 peaks → return the request unchanged.
        assert_eq!(
            plan_control_resolution(100.0, &cfg, 7, None, None, 1024, 1024, true).unwrap(),
            (1024, 1024)
        );
        // And a q4 render on a 24 GiB (≈32 GB Mac) budget: the sc-11847 calibration has residency +
        // decode tiling absorbing the 1024² peak, so resolution must NOT drop (text already offloaded).
        assert_eq!(
            plan_control_resolution(
                24.0,
                &cfg,
                7,
                Some(Quant::Q4),
                Some(Quant::Q4),
                1024,
                1024,
                false
            )
            .unwrap(),
            (1024, 1024)
        );
    }

    /// A budget just under the full-res DENOISE peak forces a reduction to a smaller 16-aligned scale.
    /// NB (sc-11847): on measured MLX the peak is resident-weight-dominated, so the activation term this
    /// lever shrinks is only ~1.9 GiB @ 1024² (NOT the ~11 GiB of the candle #480 CUDA profile the story
    /// premise cited) — a WEAK lever. This test picks a budget in that narrow denoise-driven window; the
    /// binding overage must be the (resolution-shrinkable) denoise activation, not the ~resolution-flat
    /// decode floor, for a reduction to be reachable at all.
    #[test]
    fn resolution_lever_reduces_when_denoise_bound() {
        let cfg = Krea2Config::turbo();
        let dn_full =
            control_denoise_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), Some(Quant::Q4), 1024, 1024);
        let dc_full = qwen_vae_decode_tiled_floor_ex_text_gib(
            &cfg,
            7,
            Some(Quant::Q4),
            Some(Quant::Q4),
            1024,
            1024,
        );
        // The reduction is only reachable when denoise (not the ~resolution-flat decode floor) is the
        // binding peak — resolution can't shrink the decode floor. Guard that precondition.
        assert!(
            dn_full > dc_full,
            "denoise must be the binding peak for a resolution reduction to help: {dn_full} vs {dc_full}"
        );
        // Just under the full-res denoise peak (and above the decode floor + the smaller-scale denoise):
        // full res fails on denoise, a smaller scale fits.
        let safe = dn_full - 0.01;
        let (w, h) = plan_control_resolution(
            safe,
            &cfg,
            7,
            Some(Quant::Q4),
            Some(Quant::Q4),
            1024,
            1024,
            false,
        )
        .unwrap();
        assert!(
            w < 1024 && h < 1024,
            "a denoise-bound over-budget render must reduce resolution: got {w}×{h}"
        );
        assert!(
            w % 16 == 0 && h % 16 == 0,
            "reduced dims must stay 16-aligned"
        );
    }

    /// Co-resident text (the `Resident` path) raises BOTH peaks — proving `text_co_resident` is wired.
    /// And it documents the lever's real MLX limit: the ~2+ GiB text term is resolution-INDEPENDENT and
    /// exceeds the max activation the resolution lever can recover (~1.4 GiB, 1024²→512²), so a budget
    /// that fits full-res under `Sequential` (text dropped) is INFEASIBLE under `Resident` — this lever
    /// can't rescue co-resident text; only Sequential residency (selected upstream by the worker fit-gate)
    /// can. The resolution lever therefore targets the activation/decode overage that survives residency.
    #[test]
    fn resolution_lever_counts_coresident_text_but_cannot_rescue_it() {
        let cfg = Krea2Config::turbo();
        let dn =
            control_denoise_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), Some(Quant::Q4), 1024, 1024);
        let dc = qwen_vae_decode_tiled_floor_ex_text_gib(
            &cfg,
            7,
            Some(Quant::Q4),
            Some(Quant::Q4),
            1024,
            1024,
        );
        // Just above the full-res ex-text peaks: fits with text dropped (Sequential), not with it resident.
        let safe = dn.max(dc) + 0.25;
        // Sequential (text dropped) → full res fits, no reduction.
        assert_eq!(
            plan_control_resolution(
                safe,
                &cfg,
                7,
                Some(Quant::Q4),
                Some(Quant::Q4),
                1024,
                1024,
                false
            )
            .unwrap(),
            (1024, 1024)
        );
        // Resident (text co-resident) → the resolution-flat ~2 GiB text term is over budget at EVERY
        // scale (resolution can't shrink it), so this is infeasible — the fix is Sequential, upstream.
        assert!(
            plan_control_resolution(safe, &cfg, 7, Some(Quant::Q4), Some(Quant::Q4), 1024, 1024, true)
                .is_err(),
            "co-resident text can't be rescued by resolution — needs Sequential residency (upstream)"
        );
    }

    /// A budget below even the smallest scale's floor → a catchable, tagged `Err` before the render, not
    /// an OS/GPU kill mid-run (mirrors the decode-tiling gate's infeasible path).
    #[test]
    fn resolution_lever_errs_when_infeasible() {
        let cfg = Krea2Config::turbo();
        let err = plan_control_resolution(
            1.0,
            &cfg,
            7,
            Some(Quant::Q4),
            Some(Quant::Q4),
            1024,
            1024,
            true,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("krea_2 control render") && err.contains("resolution"),
            "over-budget render must surface a catchable, tagged error: {err}"
        );
    }

    /// The reduced dims stay legal multiples of 16 for a non-square, odd-multiple request — the pipeline's
    /// `validate_multiple_of_16` must never trip on a resolution-lever output.
    #[test]
    fn resolution_lever_output_is_16_aligned() {
        assert_eq!(scale_render_dim(1024, 0.75), 768);
        assert_eq!(scale_render_dim(1024, 0.5), 512);
        // A non-multiple-of-16 scaled value floors DOWN to the nearest 16 (e.g. 720·0.75 = 540 → 528).
        assert_eq!(scale_render_dim(720, 0.75), 528);
        assert_eq!(scale_render_dim(720, 0.75) % 16, 0);
        // Never below one 16-px block.
        assert_eq!(scale_render_dim(16, 0.5), 16);
    }
}
