# Full Codebase Review — mlx-gen — 2026-07-01

## Executive summary

- **Repository at a glance:** Rust workspace (Apple MLX inference via `mlx-rs`), 31 member crates + root core + `gen-core` contract layer + `gen-core-testkit`; ~896 Rust files, ~233k LOC. Third whole-workspace review (prior: 2026-06-13 / epic 5250, 2026-06-20 / epic 6882 — both remediated and re-verified here).
- **Coverage:** every crate's non-test source read in full by 16 parallel subsystem reviewers (six lenses each: security, anti-patterns, duplication, dead code, efficiency, readability), plus the non-Rust surface (Cargo pins, CI, tools/, docs) and four cross-cutting sweeps (cancellation/progress conformance across all ~35 registered entry points, panic classes, error seam, registration linkage). Test files quick-scanned for dead/duplicated code only. Excluded: `_vendor/`, `tools/golden/` (gitignored by convention), fixture data. All High findings were re-verified against source by the coordinating reviewer, not just agent-reported.
- **Headline:** the numeric core remains solid and the two prior remediation waves held — every previously-High item (cancellation gaps in svd/seedvr2/bernini/scail2, LTX unbudgeted decode, chroma dup denoise, qwen compile-glue leak, sam3 bf16/eviction) is verified fixed, and a workspace panic-class sweep found the historical NaN-sort/FFI-closure/dtype-abort classes essentially eliminated. The new risk is concentrated in the ~189 commits since 2026-06-20: the never-reviewed crates (sd3, sana, depth) and the new seams (PiD decode, control branches, Group-B converters, hand-rolled per-crate validation) each re-introduce one instance of a bug class the older crates already fixed. Zero exploitable security issues; zero `unsafe` in product code.
- **Counts: Critical: 0 | High: 6 | Medium: 45 | Low: 74 | Info: 25 — 150 findings.**

## Critical findings

None.

## High findings

#### [F-001] Validate the SDXL inpaint mask pixel buffer before indexing
- **Category:** security
- **Severity:** High
- **Location:** `mlx-gen-sdxl/src/inpaint.rs:29-61`
- **Finding:** `preprocess_mask` never checks `mask.pixels.len() == width·height·3`. On the same-size path a short buffer makes `rgb_to_luma` return a short `luma` vec and `luma[(ly*8)*w + (lx*8)]` indexes out of bounds; on the resize path a short buffer reaches `resize_nearest_u8`, whose inferred channel count can round to 0, and the luma indexing panics anyway. The sibling `preprocess_image` (init/control) carries exactly this guard (F-071 of the 06-13 review) — the mask entry point was missed.
- **Impact:** a malformed `Conditioning::Mask` image in a generation request panics/aborts the worker process instead of returning a typed error — the repo's historical highest-severity class, on a request-supplied input.
- **Suggested fix:** mirror `preprocess_image`'s guard at the top of `preprocess_mask`: reject `pixels.len() != w*h*3` and zero dimensions with `Error::Msg` before luma conversion/resize. (Verified by coordinating reviewer.)
- **Confidence:** High

#### [F-002] Validate the Ideogram inpaint mask pixel buffer before indexing
- **Category:** security
- **Severity:** High
- **Location:** `mlx-gen-ideogram/src/pipeline.rs:132-161`
- **Finding:** `preprocess_mask_packed` is the same defect as F-001 in the second inpaint implementation: no `pixels.len()` check, while the sibling `preprocess_source_image` twenty lines up (pipeline.rs:112-116) validates its buffer. Short buffer → short `luma` → `luma[(ly*patch)*w + (lx*patch)]` OOB panic (same-size path), or `resize_nearest_u8` inferring `c < 3` (resize path).
- **Impact:** identical to F-001 — request-supplied mask panics the worker. Two independent crates re-implemented the mask preprocessor and both dropped the guard their own image preprocessor has (see theme T2).
- **Suggested fix:** add the `preprocess_source_image` buffer guard to `preprocess_mask_packed`. (Verified by coordinating reviewer.)
- **Confidence:** High

#### [F-003] Validate SAM3 box prompts before host indexing
- **Category:** security
- **Severity:** High
- **Location:** `mlx-gen-sam3/src/geometry.rs:173-198` (reachable from `mlx-gen-sam3/src/model.rs:159-183` `forward_with_boxes`/`segment_with_boxes`); grid bound at `geometry.rs:306-327`
- **Finding:** the public PVS path passes user `boxes`/`box_labels` straight to the geometry encoder. `let n = boxes.shape()[1]` is trusted; `boxes_host` is then indexed as `boxes_cxcywh[bi*4+3]` in `box_pos_encoding`/`roi_align_matrix`, so a `[1, N, 2]`-shaped prompt makes the flat vec shorter than `4·n` → OOB panic. `Array::from_slice(box_labels, &[n])` hard-panics when `box_labels.len() != n`. Additionally, ROI-align grid sizes `ceil(roi/r)` derive directly from unvalidated box extents — a finite oversized `w = 1e6` yields a ~10⁷-wide host triple loop with no cancellation check (hours-long hang).
- **Impact:** a malformed or hostile box prompt (the exact input class this crate is called with) aborts the request path via panic, or hangs the worker thread in a pure host loop.
- **Suggested fix:** at the top of `Sam3GeometryEncoder::forward` (or `forward_with_boxes`), return `Error::Msg` unless `boxes.shape() == [1, n, 4]`, `box_labels.len() == n`, and coordinates are finite and within `[0, 1]` (mirroring sam2's F-170 `preprocess` guard); the range check also bounds the ROI grid. (Verified by coordinating reviewer.)
- **Confidence:** High

#### [F-004] Fix SD3.5 empty-prompt CLIP encoding — missing BOS diverges from the diffusers uncond
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `mlx-gen-sd3/src/pipeline.rs:57-68`
- **Finding:** `clip_ids` special-cases the empty prompt (`if prompt.is_empty() { Vec::new() }`) then pads with `CLIP_PAD_ID` — producing 77×EOS with **no BOS**. `ClipBpeTokenizer::tokenize("")` returns `[BOS, EOS]` (verified: `tokenizer.rs:150-175` unconditionally pushes BOS/EOS), which after padding matches diffusers' `tokenizer("", padding="max_length")` exactly. The special case therefore changes every hidden state and shifts the pooled-at-argmax EOS selection from index 1 to index 0. This is the uncond branch for **every** Large/Medium CFG generation with an unset negative prompt (the default: `req.negative_prompt.as_deref().unwrap_or("")`), and the real-weight e2e uses an explicit negative prompt so the default path is not golden-covered.
- **Impact:** systematic conditioning divergence from the diffusers reference on the default request shape for `sd3_5_large`/`sd3_5_medium` — a quality/parity deviation coherence smokes can't catch. Same bug family as the z-image sc-8958 fix that merged last week.
- **Suggested fix:** drop the `prompt.is_empty()` special case and always call `tokenizer.tokenize(prompt)?`; keep truncate/pad; correct the comment claiming diffusers encodes it the same way; add an empty-negative golden. (Verified by coordinating reviewer.)
- **Confidence:** High

#### [F-005] Kolors CFG-off (guidance ≤ 1.0) breaks at runtime in every mode
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `mlx-gen-kolors/src/model.rs:209-231` (and `denoise_img2img_latents`, `denoise_curated_latents`, `denoise_controlnet_latents`, `denoise_ip_latents`, `denoise_controlnet_ip_latents` — all six assemblies)
- **Finding:** every Kolors denoise assembly unconditionally concatenates `[pos, neg]` into B=2 conditioning (`concatenate_axis(&[&pos.0, &neg.0], 0)` — verified at model.rs:213-214, 290-291, 377), but the shared `mlx_gen_sdxl::denoise_core` only CFG-batches the **latents** when `cfg > 1.0` (verified: `pipeline.rs:335,345`). With `guidance <= 1.0` — valid per capabilities, and documented on the struct API as "cfg ≤ 1 disables guidance" — the UNet receives B=1 latents with B=2 conditioning and the attention reshape fails mid-denoise with an opaque MLX element-count error.
- **Impact:** a capability-valid request (`guidance: Some(1.0)` or lower) has never worked on any Kolors mode; it dies deep in the UNet instead of either rendering CFG-off or being rejected at validation. SDXL and InstantID handle the same input correctly.
- **Suggested fix:** build the conditioning conditionally (`cfg > 1.0` → `[pos, neg]`, else `pos` only with `kolors_time_ids(1, …)`), and skip encoding the negative prompt when CFG is off (also saves a full ChatGLM3-6B forward). Add a CFG-off smoke. (Verified by coordinating reviewer; upgraded from the subsystem reviewer's Medium — a valid request failing on every mode is High per rubric.)
- **Confidence:** High

#### [F-006] Make the PiD 4-step decode cancellable
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `mlx-gen-pid/src/sampler.rs:64-98`; `mlx-gen-pid/src/decoder.rs:64-83`
- **Finding:** `Sampler::run`'s 4-step loop has no cancellation hook and no `eval` between steps, and the `LatentDecoder::decode(&self, latents)` trait carries no `CancelFlag`. The engine's own docs call this "the ~100 s decode" (`engine.rs:35`), and the seam is wired as the optional decode path into ~13 provider crates (epic 7840).
- **Impact:** once any PiD-enabled render reaches the decode stage, a user cancel has zero effect for the entire multi-minute pass — the exact typed-cancellation contract violation class the 06-20 review rated High for svd/seedvr2/bernini/scail2 and that was remediated everywhere else. Because the seam is catalog-wide, one gap regresses cancellation on a dozen models at once.
- **Suggested fix:** thread a cancel check through the loop — e.g. `Sampler::run` takes an optional `&CancelFlag` (or the decoder holds a clone bound at `PidEngine::decoder` time), checks + `eval`s at each of the 4 step boundaries, and returns `Error::Canceled`; the per-step `eval` also bounds latency and transient memory (see F-013).
- **Confidence:** High

## Medium findings

#### [F-007] Add the steps ≥ 1 floor that validate_request comments claim exists
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `gen-core/src/generator.rs:534-604`; false comments at `gen-core/src/sampling.rs:166-169, 390-391`
- **Finding:** `Capabilities::validate_request` never checks `req.steps`, yet two schedule builders justify their `.max(1)` clamps with "the real floor is `validate_request` enforcing steps>=1 upstream (F-037)" — a floor that exists only where providers hand-wrote it (e.g. z-image `model_base.rs:175` has no zero check).
- **Impact:** `steps: Some(0)` reaches provider schedule code guarded only by ad-hoc clamps — the historical steps=0 panic class (six Highs in the 06-09 review) — and the comments mislead future solver code into dropping clamps.
- **Suggested fix:** reject `req.steps == Some(0)` in `Capabilities::validate_request` (mirroring `CaptionCapabilities`'s `max_new_tokens == 0` rejection), or correct the F-037 comments.
- **Confidence:** High

#### [F-008] `Error::Unsupported` is documented as contract-load-bearing but never constructed
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `gen-core/src/error.rs:26-28`; `gen-core/src/generator.rs:560-603`; reverse bridge `src/error.rs:79`
- **Finding:** the variant's doc says "Candle gating depends on this being typed — do **not** stringify it into `Error::Msg`", yet no code in the workspace ever constructs `Error::Unsupported`: the shared `validate_request` and every provider reject capability gaps with `Error::Msg`, and the reverse bridge additionally degrades any `Unsupported` crossing gen-core→mlx-gen into `Msg(format!("unsupported: …"))`. (Contrast: `Canceled` round-trips 1:1 both directions — that half of the seam is correct everywhere.)
- **Impact:** any consumer (SceneWorks worker / candle gating) matching on the typed `Unsupported` will never see it from this backend — capability gaps are indistinguishable from generic failures, defeating the variant's documented purpose.
- **Suggested fix:** return `Error::Unsupported` from the capability-gap branches of `validate_request`; add an `Unsupported` variant to `mlx_gen::Error` (or map losslessly) so the bridge preserves it.
- **Confidence:** High

#### [F-009] Generator conformance testkit is opt-in and covers 2 of ~35 registered ids
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `gen-core-testkit/src/lib.rs:416-445`; per-crate `tests/conformance.rs` (only z-image, krea) and `tests/cancellation_conformance.rs` (svd, seedvr2 base id, scail2, bernini_renderer)
- **Finding:** nothing iterates the registry — full behavioral conformance (validate/progress/cancel/seed) runs for only `z_image_turbo` and `krea_2_turbo`; cancellation-only for four more ids; zero coverage for ~29 registered ids including every new crate (sana ×2, sd3 ×3), all of wan/ltx/flux/flux2/sdxl/qwen/lens/sensenova/ideogram/boogu/chroma/kolors/pulid. All existing conformance tests are `#[ignore]`d (real weights).
- **Impact:** the epic-3720 goal ("a provider that silently ignores CancelFlag becomes a CI failure") holds only for families that remembered the boilerplate; F-038 (bernini's non-conformant progress) is live proof — it would fail the testkit's `check_progress` but no test runs it.
- **Suggested fix:** add a per-crate testkit invocation for every registered id (fixture-weight profile where possible so it isn't `#[ignore]`d), or a registry-iterating descriptor-level sweep in one place; also write the intended-but-missing boogu conformance test (its testkit dev-dep is dangling — see F-102).
- **Confidence:** High

#### [F-010] Guard zero/degenerate-rank third-party LoKr/LoHa factors before scale derivation
- **Category:** security
- **Severity:** Medium
- **Location:** `src/adapters/loader.rs:221-256` (LoKr) and `src/adapters/loader.rs:374-408` (LoHa)
- **Finding:** `ThirdPartyLokr::rank()` reads `a.shape()[1]` (LoHa: `b.shape()[0]`) directly off file-supplied factor tensors, then `scale()` computes `alpha.unwrap_or(r) / r`. A 1-D/0-D factor panics on the shape index; a zero dim yields `r == 0` → inf/NaN scale, multiplied into the reconstructed delta and silently installed. The equivalent LoRA path was hardened (sc-5252/F-002 of the 06-13 review); the third-party LoKr/LoHa paths — including the new sc-8345/sc-8395 BFL/prefix-strip code — were not.
- **Impact:** a corrupt/adversarial third-party LyCORIS `.safetensors` either aborts the worker at load or NaN-poisons every subsequent render on that model while the load reports success.
- **Suggested fix:** validate factor `ndim == 2` in `rank()` (typed error) and reject `r <= 0` in `scale()`/`delta()`, mirroring the F-002 LoRA guard.
- **Confidence:** High

#### [F-011] `packed_bits` panics on malformed pre-quantized checkpoint shapes (core + flux copy)
- **Category:** security
- **Severity:** Medium
- **Location:** `src/quant.rs:38-41` (via `lin`/`embedding` at 49-90); duplicated at `mlx-gen-flux/src/text_encoder.rs:48-63`
- **Finding:** `packed_bits` computes `in_dim = scales.shape()[1] * group_size; wq.shape()[1] * 32 / in_dim`. A checkpoint whose `{base}.scales` is 1-D panics on the shape index; a `[out, 0]` scales tensor is an integer divide-by-zero. `lin`/`embedding` are the shared load seam for every pre-quantized snapshot (all seven new Group-B packed-load converters feed through it), so the shapes come straight off external `.safetensors` files with no validation.
- **Impact:** a truncated/corrupt/mis-converted pre-quantized snapshot aborts the process at model load instead of surfacing a typed load error.
- **Suggested fix:** have `packed_bits` return `Result`, validating 2-D shapes, `in_dim > 0`, and derived bits ∈ {4, 8}; point the flux-local copy at the shared fn.
- **Confidence:** High

#### [F-012] Fixed-prefix `apply_adapter_specs` mis-reconstructs a third-party LoKr declared as `AdapterKind::Lokr`
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `src/adapters/loader.rs:1114-1150`
- **Finding:** in the fixed-prefix path, `AdapterKind::Lokr` routes unconditionally to the peft `apply_lokr`. For a metadata-less third-party LyCORIS LoKr file, `parse_lokr` silently drops the tucker `lokr_t2` factor and ignores per-module `.alpha` tensors (scale defaults to 1.0). The autoprefix path detects this by keys and routes to `apply_lokr_thirdparty`; the fixed-prefix path does not, and its F-035 scope note documents only the BFL/kohya gaps.
- **Impact:** a caller classifying a third-party LoKr as `Lokr` on the fixed-prefix seam gets a wrongly-scaled (with tucker, structurally wrong) adapter with `applied > 0` — silent quality corruption.
- **Suggested fix:** mirror the autoprefix routing (`!is_lokr(&w) && is_lokr_keys(&w)` → `apply_lokr_thirdparty`), or error loudly on a keys-only file; extend the F-035 doc.
- **Confidence:** Medium

#### [F-013] Add a memory budget / refusal path to PiD decode
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-pid/src/decoder.rs:64-83`; `mlx-gen-pid/src/sampler.rs:104-129`
- **Finding:** `decode` allocates noise + per-step ε at the full super-resolved size (`[B,3,zH·32,zW·32]` f32) and runs 4 full PixDiT forwards there, with no analog of seedvr2's `plan_chunk_size`/`OverBudget`/`maxBufferLength` clamp and no `eval` between steps (the whole 4-step graph schedules at once, raising transient peak).
- **Impact:** a `max_size`-legal 2048² request decodes at 8192² — ≈0.8 GB per pixel-space tensor, 262k patch tokens through 14 MMDiT blocks — an uncatchable MLX OOM/SIGKILL on smaller Macs rather than a typed refusal; affects every PiD-enabled family.
- **Suggested fix:** estimate peak from the target `(th, tw)` against `mlx_gen::memory::safe_budget_gib()` at the `resolve_pid_decoder*` seam and return a typed over-budget error (or tile); at minimum `eval` per sampler step.
- **Confidence:** Medium

#### [F-014] Honor cancellation during the Wan VAE decode stage
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-wan/src/vae_common.rs:80-141` (`tile_decode_accumulate`); `mlx-gen-wan/src/model.rs:460-476, 883-896`; `mlx-gen-wan/src/model_vace.rs:360-371, 621-632`
- **Finding:** every denoise loop checks cancel per step (all modes), but the decode stage does not — `tile_decode_accumulate` takes no `CancelFlag` and no generate path re-checks between denoise and decode. The crate's own sc-4998 comments state z48 decode is "~95% of wall-clock" once Lightning makes the DiT trivial.
- **Impact:** a cancel issued after the last denoise step (or during a tiled decode of a 1280×704×145 video) is ignored for minutes — the dominant-cost stage of a 5B render is uncancellable.
- **Suggested fix:** thread `&CancelFlag` into `decode_to_frames[_22]`/`decode_tiled` → `tile_decode_accumulate`, returning `Error::Canceled` between tiles (per-tile `eval` already bounds latency); check once before starting decode in each `generate_impl`.
- **Confidence:** High

#### [F-015] 5B TI2V/keyframe conditioning is silently broken by `trim_first_frames`
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-wan/src/model.rs:266-270, 320-367, 473-477` (contrast the I2V guard at 700-707)
- **Finding:** the 5B `generate_impl` extends the latent by `trim` frames and later drains `0..trim_out` output frames. With a `Reference` the mask-blend pins latent frame 0 — which decodes to output frame 0 and is then discarded by the trim; with `Keyframe`s the pinned indices land trim-shifted relative to the delivered video. `Wan14b::validate_impl` explicitly rejects `trim_first_frames` for I2V for exactly this mismatch class; the 5B `validate_impl` allows the combination.
- **Impact:** a request combining `trim_first_frames > 0` with Reference/Keyframe silently violates the user-visible contract (image = first frame / pinned frame k) with no error.
- **Suggested fix:** reject the combination in `Wan::validate_impl` (mirroring the I2V guard), or offset mask/keyframe indices by the trim extension.
- **Confidence:** Medium

#### [F-016] Wan dual-expert trainer trains each expert on a disjoint half of an even-sized dataset
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-wan/src/training.rs:584-590`
- **Finding:** the MoE loop routes `ei = if dual && step % 2 == 1 { 1 } else { 0 }` and picks the item as `cache[((step-1) as usize) % cache.len()]`. Step parity and item-index parity are locked together, so with an even `cache.len()` the high expert only ever sees even-indexed items and the low expert odd-indexed ones, for the entire run.
- **Impact:** for any even dataset size, each expert LoRA trains on exactly half the images — silent quality degradation that varies with dataset-size parity.
- **Suggested fix:** decouple expert alternation from item selection (e.g. index by `((step-1) / n_experts) % cache.len()` or a per-expert micro counter) so both experts sweep the full dataset.
- **Confidence:** High

#### [F-017] Port the F-069 partial-window grad-accum fix to the LTX and Wan trainers
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-ltx/src/training.rs:451-460`; `mlx-gen-wan/src/training.rs:629-655`
- **Finding:** both trainers' final flush divides by the full `accum` even when the last window is partial (`cfg.steps % accum != 0`), down-scaling the final update — the exact defect fixed in z-image/lens as F-069 ("Divide by the actual in-window count instead"). Wan additionally discards the *other* expert's partial accumulation entirely on loop exit.
- **Impact:** with e.g. `steps=10, accum=4` the last optimizer step's effective LR is halved (LTX), and on Wan dual-expert runs one expert's tail gradients are silently thrown away — silent training-dynamics skew already recognized as a bug elsewhere in the workspace.
- **Suggested fix:** port the z-image/lens window computation (`if step % accum == 0 { accum } else { step % accum }`); on Wan, flush each expert whose accumulator is `Some` at loop exit, averaging by the actual count.
- **Confidence:** High

#### [F-018] Honor request cancellation in the LTX prompt-enhance decode loop
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-ltx/src/enhance.rs:141-182`; call site `mlx-gen-ltx/src/model.rs:813-852`
- **Finding:** `enhance()` runs up to `max_tokens` (≤2048) autoregressive Gemma-12B forwards with no `CancelFlag` check; `Ltx::run_enhance` never passes `req.cancel`. The doc comment at enhance.rs:45 claims "(only cooperative `cancel` breaks it)" — no cancel hook exists.
- **Impact:** a cancel during prompt enhancement (potentially minutes of 12B decode plus the on-demand uncensored-enhancer load) is ignored until the enhancer finishes, while the denoise loops honor the per-step contract.
- **Suggested fix:** thread `&CancelFlag` into `enhance()`, check per decode step, return `Error::Canceled`; fix the enhance.rs:45 comment.
- **Confidence:** High

#### [F-019] Make the Lens 20B MoE text encode cancellable
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-lens/src/pipeline.rs:413-414`; `mlx-gen-lens/src/text_encoder/encoder.rs:125-152`
- **Finding:** `generate_with_progress` checks `cancel` only inside `run_flow_sampler` and the optional reasoner; `encode_prompt` — up to two full 24-layer forwards of the 20B gpt-oss MoE — never consults the flag, and the layer loop has no hook.
- **Impact:** cancel latency is worst exactly where it matters: on a 4-step turbo run the MoE encode can dominate wall time and is entirely uncancellable.
- **Suggested fix:** thread `&CancelFlag` into `encode_prompt`/`LensTextEncoder::encode`, checking per decoder layer (cheap and effective — `routing_weights` already forces a host sync per layer), plus a check between the positive and negative encodes.
- **Confidence:** High

#### [F-020] Skip the dead uncond half on the Lens turbo default (guidance = 1.0)
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-lens/src/pipeline.rs:319-335` (and `render_sample` 486-501)
- **Finding:** the `predict` closure always concatenates `[latents, latents]` and runs the 48-block DiT at B=2, then `cfg_rescale(cond, uncond, g)`. At `guidance_scale == 1.0` — the `lens_turbo` default — the combination is exactly `cond` (rescale factor 1), so the uncond half is mathematically dead.
- **Impact:** ~2× DiT compute and activation memory per step on the crate's most common configuration.
- **Suggested fix:** when `guidance_scale == 1.0`, skip the batch duplication and negative conditioning (B=1, raw prediction) — bit-identical output, gate with the existing e2e parity test.
- **Confidence:** High

#### [F-021] Lens MoE forward evaluates all 32 experts densely with a host sync per layer
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-lens/src/text_encoder/gpt_oss.rs:684-760`
- **Finding:** `GptOssMoe::forward` runs every expert and zero-weights the non-routed ones, and `routing_weights` pulls logits to host (a blocking eval) once per layer. The code marks this "correctness-first … a gather/grouped-GEMM path can follow".
- **Impact:** ~8× the routed expert FLOPs (top-4 of 32) across 24 layers plus 24 host barriers per encode — the dominant cost of every prompt encode and reasoner decode step (per token). Deliberately deferred in-code, but it is the single biggest perf lever in the crate; it needs a tracked story, not just a comment.
- **Suggested fix:** route on-device (top-k mask or gather + grouped GEMM over selected experts), keeping the host path for tiny parity fixtures; file the story if none exists.
- **Confidence:** High

#### [F-022] Restore the true_cfg/guidance_method checks SDXL's hand-rolled validate dropped
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-sdxl/src/model.rs:833-905`
- **Finding:** SDXL's `validate_request` is a hand-rolled copy of the shared contract and omits `req.true_cfg` (advertised unsupported) and `req.guidance_method` membership (`["cfg","cfg_pp"]`, added by sc-8256). Kolors was remediated for exactly this drift (F-132) and now delegates to core; SDXL did not.
- **Impact:** `true_cfg` and any bogus `guidance_method` (e.g. a `cfg_pp` typo) are silently ignored — the request renders plain CFG, contradicting the crate's own "reject loudly, never silently downgrade" stance.
- **Suggested fix:** delegate to `caps.validate_request(MODEL_ID, req)` (as Kolors does) and keep only the SDXL-specific checks on top.
- **Confidence:** High

#### [F-023] SDXL/Kolors trainers misreport steps == 0 as a cancellation
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-sdxl/src/training/mod.rs:243-259`; `mlx-gen-kolors/src/training.rs:244-260`; `mlx-gen-sdxl/src/training/family.rs:580-587`
- **Finding:** `train_family`'s comment says "`steps == 0` is rejected upstream by `validate`", but neither trainer's `validate` checks it (z-image's does). With `steps: 0` the loop never runs and the function returns `Error::Canceled`.
- **Impact:** a bad config fires downstream cancel semantics (retry/no-artifact) instead of a validation error; the taxonomy comment is false for both consumers.
- **Suggested fix:** add the z-image-style `steps > 0` check to both `validate` fns (or enforce inside `train_family`).
- **Confidence:** High

#### [F-024] Add F-020-style request guards to the InstantID struct API
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-instantid/src/model.rs:412-472` (`generate_with`), `684-767` (`generate_pose_with`)
- **Finding:** the public struct API validates embedding length and kps count but never `req.steps` or dims. `steps: 0` builds an empty ancestral schedule, `denoise_core` returns the prior unchanged, and the pipeline VAE-decodes pure scaled noise as a "successful" image; non-multiple-of-8/zero dims fail deep in convs with opaque errors.
- **Impact:** the same public-surface gap class remediated on Kolors (F-020, `validate_dims`) but never applied to InstantID — degenerate worker requests render garbage as success.
- **Suggested fix:** a `validate_request`-style guard at the top of both entry points: `steps >= 1`, positive multiple-of-8 dims.
- **Confidence:** High

#### [F-025] Complete request validation on the new flux1_dev_control path
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-flux/src/model_control.rs:281-309, 236-237`
- **Finding:** `validate_capability` checks only prompt/sampler/scheduler/multiple-of-16 — never `min_size`/`max_size`, `count` bounds, or the unsupported `negative_prompt`/`true_cfg`. `count: 0` runs zero iterations and returns an empty `Images([])` as success.
- **Impact:** the new control path (sc-8238/8239) bypasses the advertised capability ceiling; out-of-range resolutions reach the DiT and a negative prompt is silently ignored, contradicting the convention every sibling follows.
- **Suggested fix:** delegate to `Capabilities::validate_request` before the bespoke multiple-of-16 check (as chroma/boogu do), or copy the base flux size/count/negative/true_cfg checks.
- **Confidence:** High

#### [F-026] PuLID validate skips the capability floor and accepts the un-advertised hyper sampler
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-pulid/src/pulid_flux.rs:221-225, 261-264`
- **Finding:** `PulidFlux::validate` only runs the reference-face check; size/count/steps/sampler/scheduler are validated later inside `generate` against the **dev descriptor**, which advertises `hyper` — while `pulid_flux`'s own descriptor deliberately omits it ("it needs the dev Hyper-FLUX LoRA loaded at scale, which PuLID does not load").
- **Impact:** preflight `validate()` returns Ok for requests that fail at generate time, and `sampler: "hyper"` passes end-to-end — an 8-step render without the Hyper LoRA, i.e. the exact "undertrained noise" trap the flux config comment warns about.
- **Suggested fix:** validate against `self.descriptor` (sampler/scheduler membership, size/count/steps) in `PulidFlux::validate`, keeping backbone validation as defense in depth.
- **Confidence:** High

#### [F-027] Cap the number of FLUX.2 edit reference images
- **Category:** security
- **Severity:** Medium
- **Location:** `mlx-gen-flux2/src/model.rs:97-107, 338-368, 375-387`
- **Finding:** `collect_reference_images`/`collect_edit_references` flatten every `Reference`/`MultiReference` with no count cap; `encode_references` VAE-encodes and token-concats all of them. The shared floor only checks conditioning *kind*; `max_count = 8` caps output images, not references — yet the `REFERENCE_TIME_STRIDE` doc claims a stride invariant "at the cap" that nothing enforces.
- **Impact:** each reference adds ~4096 joint-DiT tokens (sc-6124 measured ~104 GB peak with 2 unbounded refs at 1024²); N refs → quadratic SDPA cost and OOM/abort from request input.
- **Suggested fix:** enforce an explicit reference-count ceiling (e.g. 8, matching the documented invariant) in `collect_edit_references` and `maybe_upsample`'s ref walk, with a typed error.
- **Confidence:** High

#### [F-028] Stop building ~872 MB host zero-tensors per Ideogram render
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-ideogram/src/pipeline.rs:449, 459-467, 496, 628-632`
- **Finding:** the local `fn zeros` is `Array::from_slice(&vec![0f32; n], shape)`. `run_denoise` builds `zeros(&[1, num_img, llm_dim])` (num_img·53248 f32 ≈ 872 MB host Vec + H2D copy at 1024²) per rendered image, plus a second for `neg_llm` in quality mode — while the crate already uses device-side `mlx_rs::ops::zeros` elsewhere.
- **Impact:** ~0.9–1.7 GB of pointless host allocation + memcpy per generated image; transient RSS spike on memory-constrained Macs.
- **Suggested fix:** use `mlx_rs::ops::zeros::<f32>(shape)` (lazy, device-side) in `run_denoise`.
- **Confidence:** High

#### [F-029] Hoist Ideogram's step-invariant mask/role/RoPE construction out of the per-step forward
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-ideogram/src/transformer/model.rs` (forward: `role_tensors`/`segment_mask`/`rotary_emb`); `mlx-gen-ideogram/src/pipeline.rs:499-569`
- **Finding:** `forward` recomputes `role_tensors` (a host sync + host loops), `segment_mask` (an O(L²) host fill of a `[B,1,L,L]` f32 tensor — ~77 MB at 1024²), and the MRoPE tables on **every call**, though all three depend only on step-invariant inputs. Worse, `Packing::build` sets `segment_ids = vec![1; seq]`, so the mask is always identically zero — equivalent to no mask — while still forcing SDPA onto the masked (non-fast) path.
- **Impact:** in quality mode (48 steps × 2 DiTs) ~96 rebuilds ≈ 7 GB of host alloc/fill/H2D traffic per image, all for a no-op mask.
- **Suggested fix:** precompute the role tensors, the additive mask (or pass `None` when all segment ids are equal), and the MRoPE cos/sin once per `run_denoise` and thread them into `forward`.
- **Confidence:** High

#### [F-030] Enforce advertised sampler/scheduler names in z-image request validation
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-z-image/src/model.rs:278-334` (shared by all four z-image generators)
- **Finding:** the crate-local `validate_request` never validates `req.sampler`, `req.scheduler`, or `req.true_cfg` against capabilities. Downstream, `run_curated_sampler` does `sampler_by_name(..).unwrap_or_else(|| Box::new(Euler))` and `resolve_schedule` returns `native` for an unknown name.
- **Impact:** a typo'd or unsupported sampler/scheduler on any of the four z-image ids silently renders with Euler/native instead of erroring — violating the advertised capability contract and diverging from sibling crates.
- **Suggested fix:** delegate to `caps.validate_request(id, req)` (as sana does) and keep only the z-image extras; also fixes the hardcoded model-id error strings (F-089).
- **Confidence:** High

#### [F-031] Correct the sc-8958 encode_uncond rationale — the claimed `[1,0]` trap cannot occur here
- **Category:** readability
- **Severity:** Medium
- **Location:** `mlx-gen-z-image/src/pipeline.rs:402-427` (commit 0d13f5a)
- **Finding:** the `encode_uncond` doc claims gen-core short-circuits an empty prompt to `[1, 0]` "before the chat template is applied (`pad_to_max_length = false`)" — but z-image's tokenizer is built with `pad_to_max_length: true` (loader.rs:39) and the short-circuit is gated on `!pad_to_max_length` (gen-core/src/tokenizer.rs:175). `tokenize("")` never returned `[1,0]` here; the fix was ported from candle by symmetry without an MLX repro (per its own commit note).
- **Impact:** the fix is behaviorally fine (and cheaper — ~10-token forward vs 512), but the load-bearing comment documents an impossible mechanism; a future reader auditing the empty-negative path will act on a false premise.
- **Suggested fix:** rewrite the comment: with `pad_to_max_length=true`, `tokenize("")` works but wastes a 512-token forward; `encode_chat_ids("", true)` is the equivalent unpadded encoding matching candle. Note on sc-8958 that the MLX-side error was never reproduced.
- **Confidence:** High

#### [F-032] Add memory-bounded tiling (or a size cap) to the SANA DC-AE decoder
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-sana/src/dc_ae.rs:449-464`; `mlx-gen-sana/src/model.rs:57-58`
- **Finding:** `DcAeDecoder::decode` runs the full f32 decode monolithically — no tiling, no budgeting — while the descriptor advertises `max_size = 2048`. At 2048² the shallow 128-channel stage materializes ~2.1 GB tensors with several live simultaneously (conv_inverted expands 8× at stage boundaries).
- **Impact:** a legitimate 2048² request can transiently need tens of GB in decode alone — the OOM/SIGKILL class the workspace already fixed for wan (sc-4998) and seedvr2 (sc-8135/8261) — on sizes the capability manifest says are supported.
- **Suggested fix:** either lower `RES_MAX` to the validated 1024 envelope or port the memory-budgeted spatial tiling pattern from wan's vae22 decode.
- **Confidence:** Medium

#### [F-033] Use the shared `default_seed()` for an unset Krea seed
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-krea/src/model.rs:157`
- **Finding:** `let base_seed = req.seed.unwrap_or(0);` — every other generator uses `req.seed.unwrap_or_else(default_seed)` (gen-core documents 0 as "the 'no seed' sentinel"); qwen's `resolve_run_params` does it correctly.
- **Impact:** a request that omits the seed always renders from seed 0 — identical output on every call, and `count > 1` batches reuse seeds 0..n — violating the random-per-request contract every sibling honors.
- **Suggested fix:** `req.seed.unwrap_or_else(mlx_gen::default_seed)`.
- **Confidence:** High

#### [F-034] Correct SD3.5's `requires_sigma_shift: true` — it is a static-shift schedule
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-sd3/src/config.rs:220-222` (also the stale `DEFAULT_SAMPLER` doc at config.rs:39-42)
- **Finding:** the descriptor sets `requires_sigma_shift: true` ("resolution-aware flow-match shift"), but the pipeline is `FlowMatchEuler::for_static_shift(steps, 3.0)` — explicitly resolution-independent ("identical to the Z-Image-Turbo path"); z-image, the identical-schedule precedent, sets `false` on all four models. The flag is gen-core's resolution-aware-mu loader hint.
- **Impact:** any consumer of the hint (worker capability advertisement / dynamic-mu routing) is told SD3.5 needs resolution-aware shifting when it must not get one; the comment is factually wrong about the model.
- **Suggested fix:** set `requires_sigma_shift: false` matching z-image; fix both comments (the `DEFAULT_SAMPLER` doc describes a training-only logit-normal schedule inference never uses).
- **Confidence:** Medium

#### [F-035] Make the SD3.5 trainer preflight memory guard arch-aware
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-sd3/src/training.rs:658-711` (used at 424-428)
- **Finding:** `projected_dense_peak_gb` is documented "for the 8.1B SD3.5-Large MMDiT" (`PREFLIGHT_BF16 = (16.0, …)`), but the same `train_impl` serves the registered `sd3_5_medium` trainer (2.5B, ~5 GB bf16, 24 vs 38 blocks) with no arch parameter.
- **Impact:** valid dense Medium runs are refused ("needs ~X GB") on machines that fit them — the projection overstates Medium's peak by roughly 3×, forcing checkpointing or lower resolution unnecessarily.
- **Suggested fix:** parameterize the constants by variant (weights term from param count × dtype; linear/quad terms scaled by blocks × hidden) and thread the variant into `preflight_memory_guard`.
- **Confidence:** High

#### [F-036] Stop retaining the full denoise trajectory in SenseNova production paths
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-sensenova/src/t2i.rs:525-572` (`denoise`), `1336-1408` (`it2i_denoise`); consumers at 371, 1001, 1257
- **Finding:** both denoise loops push every step's evaluated `[1,3,H,W]` f32 frame into `traj`, and every production consumer immediately takes `.last()`. Only the diagnostic `t2i_trajectory` needs intermediates.
- **Impact:** at the base recipe (50 steps, up to 2048²) ~2.5 GB of materialized frames held alive through the run on an already 8B-resident path; interleave multiplies it per image.
- **Suggested fix:** track only the current frame (or take a `keep_trajectory` flag / callback used by the diagnostic mode).
- **Confidence:** High

#### [F-037] Add cancellation (and progress) to SenseNova interleave_gen, vqa, and the AR rollout loops
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-sensenova/src/t2i.rs:1131-1271` (`interleave_gen`), `1012-1066` (`vqa`); `mlx-gen-sensenova/src/runtime.rs:169, 184, 217` (AR/think loops)
- **Finding:** F-128 threaded `StepReporter` only through the registry `generate` path. `interleave_gen` passes `reporter: None` into `it2i_denoise` and its AR text loop checks no flag; `vqa`'s decode likewise loops to `max_new_tokens` uncancellable. Per the model.rs docs, the SceneWorks worker consumes both methods directly (Document Studio / VQA). The `runtime.rs` think-mode rollout loops have the same gap (unreachable today only because think-mode is hardcoded off).
- **Impact:** a multi-minute Document Studio interleave or long VQA decode cannot be cancelled or report progress — the exact gap F-128 closed for T2I, persisting on the modes that bypass the Generator contract; wiring think-mode later silently ships another uncancellable rollout.
- **Suggested fix:** add `&CancelFlag` (plus progress) to `interleave_gen`/`vqa`/`Qwen3Backbone::generate`, checking per decoded token; pass a `StepReporter` into the inner `it2i_denoise`.
- **Confidence:** High

#### [F-038] Fix Bernini progress: 0-based steps that never reach total, and a silent planner stage
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-bernini/src/pipeline.rs:180, 289, 459-460`; `mlx-gen-bernini/src/bernini.rs:655-790, 885-889`; `mlx-gen-bernini/src/mar.rs:282-320`
- **Finding:** both registered bernini ids emit `on_step(i)` 0-based — progress runs `0..total-1` and never reaches `total` (every other provider is 1-based; the testkit's `check_progress` requires `1..=total` and bernini would fail it — masked because only cancellation-only conformance runs, see F-009). The full `bernini` id is additionally progress-silent through the entire multi-minute MAR planner stage (75 Qwen2.5-VL-7B forwards).
- **Impact:** progress bars stall one short of completion and show 0/N for minutes during planning — indistinguishable from a hang.
- **Suggested fix:** emit `on_step(i + 1)`; give the planner stage its own monotone emission; add bernini to full testkit conformance.
- **Confidence:** High

#### [F-039] Validate `segment_len`/`segment_overlap` before scail2's `build_segments`
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-scail2/src/generate.rs:214-231` (pub fields at 105-106)
- **Finding:** `let stride = len - overlap;` underflows `usize` when `overlap >= len` (debug panic; release wraps to a huge stride and silently renders only the first window). `Scail2Job` is a public API with public fields and nothing validates them; overlap must additionally be `1 + 4k` or the clean-history VAE encode shape-errors deep in segment 2.
- **Impact:** a worker constructing `Scail2Job` directly panics or silently truncates output; a wrong overlap fails opaquely inside the VAE.
- **Suggested fix:** at the top of `generate()`, reject `segment_len == 0`, `segment_overlap >= segment_len`, and `segment_overlap % TEMPORAL_STRIDE != 1` with typed errors (same pattern as the existing driving-frames checks).
- **Confidence:** High

#### [F-040] Reset per-session state in `Sam3VideoModel::propagate`
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-sam3/src/video.rs:93-105` (state fields), `155-179` (`propagate`)
- **Finding:** `propagate` sets `self.num_frames` and iterates, but `obj_ids`, `banks`, `first_frame`, `keep_alive`, `removed`, `last_occluded` persist on the model. A second `propagate` call (a new clip) gathers memory from banks keyed by the previous clip's frame indices, and the `removed` set permanently suppresses object ids.
- **Impact:** silent cross-clip contamination for any long-lived model instance — exactly what a worker keeps resident (~445M params is expensive to reload).
- **Suggested fix:** clear the per-session fields at the top of `propagate`, or split state into a `VideoSession` object as sam2's `Sam2VideoPredictor`/`VideoState` already does.
- **Confidence:** High

#### [F-041] Port the F-024 memory-bank eviction to SAM2
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-sam2/src/video_predictor.rs:806-811, 862-869` (inserts); no eviction anywhere in the crate
- **Finding:** every propagated frame stores `maskmem_features`/`maskmem_pos_enc` (~2 MB) + `pred_masks` into `state.non_cond` and nothing prunes it; SAM3's F-024 remediation (`evict_stale_bank`/`evict_stale_cond_heavy`, verified real with unit tests) has no counterpart here. `condition_with_memories` only reads windowed entries.
- **Impact:** ~2.3 MB · frames of dead resident growth — a 1000-frame clip holds ~2.3 GB of unused tensors.
- **Suggested fix:** port the sam3 eviction, with the sam2-specific caveat that reverse propagation reads frames ahead and `add_points_internal` re-reads `pred_masks` at arbitrary frames — so null only the `maskmem_*` pair, window-bounded per direction.
- **Confidence:** High

#### [F-042] Clamp or reject out-of-range SAM3 box labels before the embedding gather
- **Category:** security
- **Severity:** Medium
- **Location:** `mlx-gen-sam3/src/geometry.rs:192-197`
- **Finding:** `label_embed.take_axis(&lbl_idx, 0)` gathers from a `[2, C]` table using raw user labels; a label outside `{0, 1}` is an out-of-bounds GPU gather (undefined values). The sibling point path already clamps (`tracker.rs:452` `label.clamp(0, 1)`).
- **Impact:** a stray label silently produces garbage prompt embeddings and nonsense masks; on some backends an OOB gather can read arbitrary buffer memory.
- **Suggested fix:** clamp to `0..=1` as tracker.rs does, or return `Err` (piggybacks on F-003's guard).
- **Confidence:** Medium

#### [F-043] Filter SAM3 detections on-GPU before the 66 MB per-frame mask readback
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-sam3/src/video.rs:336-352`
- **Finding:** `run_detection` reads back **all** 200 query masks every frame (`[200, 288·288]` f32 ≈ 66 MB host copy), then re-`to_vec()`s each kept detection — while typically only a handful pass `SCORE_THRESH_DET`.
- **Impact:** ~66 MB of GPU→host traffic + allocation per video frame, dominated by immediately-discarded masks; seconds of pure copy overhead on long clips.
- **Suggested fix:** read back the tiny `probs` first, build the kept-index list, `take_axis` survivors on-GPU, and read back only those.
- **Confidence:** High

#### [F-044] Reject zero/mismatched image dims in depth preprocessing
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-depth/src/preprocess.rs:30-74`; `mlx-gen-depth/src/lib.rs:98-120`
- **Finding:** `rgb8_to_input`/`rgb8_to_input_sized` (pub) never validate `rgb.len()` against `w·h·3`, and `resize_rgb8_to_unit` computes `.min(in_h - 1)` — with `in_h == 0` a usize underflow (release: wraps, then OOB index). `estimate_control_rgb8` checks exact length but a 0×0 image passes (`0 == 0`) and panics in the resize loop.
- **Impact:** a degenerate or mismatched request-supplied control image panics the depth preprocessor (worker crash) — the brand-new crate (sc-8242) regressed the guard convention face/gen-core already follow, and it now sits on the qwen 2512-Fun control path.
- **Suggested fix:** in `rgb8_to_input_sized` (the shared funnel), reject `width == 0 || height == 0` and `rgb.len() < w·h·3` with `Error::Msg`, mirroring `FaceAnalysis::detect`.
- **Confidence:** High

#### [F-045] Consolidate the Group-B converter glue — real drift already shipped
- **Category:** redundant
- **Severity:** Medium
- **Location:** `mlx-gen-{z-image,lens,flux,sdxl,qwen-image,chroma,sensenova}/src/convert.rs` (e.g. z-image:70/121/163, lens:61/159/205-211, sensenova:49-63/98/129-137); also `mlx-gen-krea/src/convert.rs:308-412`
- **Finding:** the numeric spine is properly shared (`src/quant.rs`), but ~50-90 lines/crate of orchestration glue is cloned: `write_quantized_config` (5 of 6 byte-identical), `copy_dir`, and the turnkey asset-copy tail. Verified drift: (a) z-image's asset list omits `LICENSE.txt`, which all six siblings include; (b) sensenova writes **no** `"quantization"` annotation into its turnkey `config.json` at all; (c) qwen canonicalizes HF blob symlinks on copy, krea does not; (d) the component-quantize job has four structural shapes.
- **Impact:** (a) a z-image snapshot whose license ships as `LICENSE.txt` produces a license-less rehosted tier; (b) sensenova packed snapshots lack the quant marker other loaders/tooling rely on (see also F-076); the 7-way clones will keep drifting.
- **Suggested fix:** hoist `write_quantized_config`, a symlink-resolving `copy_dir`, and one canonical asset list (including `LICENSE.txt`) into `src/quant.rs` or a convert-util module; make sensenova write the config annotation. Fits epic 7778.
- **Confidence:** High

#### [F-046] Pin GitHub Actions to commit SHAs
- **Category:** security
- **Severity:** Medium
- **Location:** `.github/workflows/ci.yml:23, 30, 59, 90`
- **Finding:** all four action references use mutable tags (`actions/checkout@v5`, `Swatinem/rust-cache@v2`); the third-party cache action can be retagged at any time.
- **Impact:** a compromised/retagged action release executes arbitrary code in CI with the repo's GITHUB_TOKEN — the tag-hijack supply-chain vector, in a repo that already treats supply-chain capture seriously for mlx-rs.
- **Suggested fix:** pin to full commit SHAs with trailing version comments; let Dependabot bump.
- **Confidence:** High

#### [F-047] Add a least-privilege `permissions:` block to CI
- **Category:** security
- **Severity:** Medium
- **Location:** `.github/workflows/ci.yml:1-12`
- **Finding:** no `permissions:` at workflow or job level — both jobs run with the default GITHUB_TOKEN grant.
- **Impact:** any compromised step (see F-046) or malicious transitive build script gets a write-capable token; CI only needs read.
- **Suggested fix:** `permissions: { contents: read }` at the top level.
- **Confidence:** High

#### [F-048] Pin the core-llm git dependency by rev, not the mutable `main` branch
- **Category:** security
- **Severity:** Medium
- **Location:** `Cargo.toml:51-57`
- **Finding:** `core-llm = { git = …, branch = "main" }` — the only mutable git ref in the workspace (mlx-rs/mlx-sys/mlx-llm are rev-pinned), and its justifying comment ("matches the mlx-llm convention") is factually wrong — mlx-llm is rev-pinned. Cargo.lock locks it today, but CI never passes `--locked`.
- **Impact:** any push to core-llm `main` silently changes what a re-resolution builds — non-reproducible builds, a mutable supply-chain edge, and red CI with zero local changes.
- **Suggested fix:** pin `rev = "3870ed1c…"` (the SHA already in Cargo.lock); run CI cargo commands with `--locked`.
- **Confidence:** High

#### [F-049] Hoist the mlx-llm git pin into `[workspace.dependencies]`
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-flux2/Cargo.toml:24`; `mlx-gen-joycaption/Cargo.toml:18`; `mlx-gen-sensenova/Cargo.toml:24-29` (deps + dev-deps)
- **Finding:** the same `mlx-llm = { git, rev = "7041411f…" }` pin is copy-pasted four times; the sensenova comment itself states the hazard ("Pin must equal every other mlx-llm pin … so cargo unifies ONE mlx-llm").
- **Impact:** exactly the divergent-per-crate-pin failure mode the root Cargo.toml documents workspace.dependencies as preventing: a bump that misses one copy yields two mlx-llm builds → cross-crate `Array`/`core-llm` type-mismatch errors and a doubled MLX build.
- **Suggested fix:** add it to `[workspace.dependencies]`; switch all four sites to `{ workspace = true }`.
- **Confidence:** High

#### [F-050] Refresh CLAUDE.md: stale crate inventory, dependency map, and metallib resolver order
- **Category:** readability
- **Severity:** Medium
- **Location:** `CLAUDE.md:14-16, 86`; `tools/refresh_pmetal_metallib.sh:5-13`
- **Finding:** (a) the crate list says "~24 crates" and names the deleted `-prompt-refine` (removed sc-7158) while omitting `-clip/-krea/-pid/-sd3/-depth/-sana`; (b) the reuse-exceptions sentence is stale (14 crates now depend on `-pid`; flux→z-image+sdxl, sd3→sdxl+flux, etc.); (c) the gen-core `TextLlm` trait it lists was removed (sc-7189); (d) the sc-7889 metallib bullet and the refresh script header document the **pre-sc-7898** resolver order — the pinned SHA now tries the own-build metallib before the user cache, so "the cache is the sole working resolution" is no longer true.
- **Impact:** CLAUDE.md is the operating manual for agents; a stale crate list, wrong reuse graph, and superseded load-bearing metallib guidance send future sessions down wrong paths — the exact gotchas the file exists to prevent.
- **Suggested fix:** regenerate the crate list from `[workspace] members`; rewrite the reuse-exceptions sentence (near-universal `-pid` dep); point the trait list at `core_llm::TextLlm`; update the sc-7889 bullet and script header to the sc-7898 order.
- **Confidence:** High

#### [F-051] Remove the deleted prompt-refine model and `load_textllm` API from README
- **Category:** dead-code
- **Severity:** Medium
- **Location:** `README.md:15, 74-75`
- **Finding:** the README lists "prompt-refine (Llama-3.2-3B-Instruct prompt rewriting …)" as a supported model and documents `load_textllm` — the crate was deleted (sc-7158) and `load_textllm`/`TextLlmRegistration` were removed from gen-core (sc-7189 Phase 3).
- **Impact:** the public README advertises a capability and an API that no longer exist; a consumer following it hits a compile error.
- **Suggested fix:** remove both mentions (or point at the `core_llm` engine that replaced them).
- **Confidence:** High

## Low findings

#### [F-052] Guard slice-length preconditions in `normalized_guidance_chain` and GuidanceOps axes
- **Category:** bad-pattern · **Severity:** Low · **Location:** `gen-core/src/guidance.rs:251-279, 321-334`
- **Finding:** the chain indexes `scales[i]`/`bufs[i]`/`norm_thresholds[i]` per pred with no length check; `sum_over_broadcast` normalizes axes without bounds-checking.
- **Impact:** mismatched engine-supplied slices panic with an opaque index error instead of a contract error.
- **Suggested fix:** length equality check (early `Err`/`debug_assert_eq!`) at the chain entry; axis-range debug_assert.
- **Confidence:** High

#### [F-053] NaN passes the f32 range validations in gen-core
- **Category:** bad-pattern · **Severity:** Low · **Location:** `gen-core/src/caption.rs:266-273`; `gen-core/src/generator.rs:565-570`
- **Finding:** `if x < 0.0 || x > 2.0` is false for NaN, so `temperature: NAN` validates; `GenerationRequest`'s f32 knobs get no finiteness check.
- **Impact:** NaN parameters flow into decode/guidance math and poison the run instead of a clear rejection.
- **Suggested fix:** NaN-rejecting comparisons (`!(x >= 0.0 && x <= 2.0)`) or explicit `is_finite()` in both validates.
- **Confidence:** High

#### [F-054] Add the leading-terminal-σ guard to Dpmpp2m and Dpmpp2mCfgPp
- **Category:** bad-pattern · **Severity:** Low · **Location:** `gen-core/src/sampling/solvers.rs:157-183`; `gen-core/src/sampling/cfgpp.rs:122-144`
- **Finding:** every sibling solver guards `is_terminal(sigma)` before dividing; the two multistep solvers compute `s_next / sigma` unguarded — `0/0 = NaN` on a degenerate schedule.
- **Impact:** a zero non-final σ silently NaN-poisons latents in exactly two solvers while the rest handle it.
- **Suggested fix:** add the same early-continue to both.
- **Confidence:** Medium

#### [F-055] Make the gen-core resize entry points return Result like their sibling
- **Category:** bad-pattern · **Severity:** Low · **Location:** `gen-core/src/imageops.rs:155-169, 257-260` (vs `union_masks` at 357-366)
- **Finding:** `resize_u8`/`resize_nearest_u8` `assert!` on malformed request images (documented as request-reachable) while `union_masks` in the same file returns `Err` for the same class.
- **Impact:** the same bad request image aborts on one path and errors catchably on the other.
- **Suggested fix:** return `Result` (guards already exist; only the signature changes) or document why panic was chosen.
- **Confidence:** Medium

#### [F-056] Deduplicate the six copy-pasted registry load functions
- **Category:** redundant · **Severity:** Low · **Location:** `gen-core/src/registry.rs:105-180`
- **Finding:** `load`/`load_transform`/`load_trainer`/`load_captioner`/`load_image_embedder`/`load_text_embedder` are byte-identical modulo type and noun.
- **Impact:** six sites to keep in sync (the duplicate-id assertion already had to be replicated into each).
- **Suggested fix:** one private macro or generic helper.
- **Confidence:** High

#### [F-057] Resolve the non-monotone `TilingConfig::auto` tile size
- **Category:** readability · **Severity:** Low · **Location:** `gen-core/src/tiling.rs:150-163`
- **Finding:** `if max_dim > 1024 { 384 } else if max_dim > 768 { 512 } else { 384 }` — first and last arms identical; a 700 px output gets smaller tiles than a 1000 px one. Reads like a transposed threshold.
- **Impact:** either a port typo (mid-size videos pay extra overlap recompute) or an undocumented reference quirk inviting a wrong "fix".
- **Suggested fix:** diff against the reference `TilingConfig.auto`; correct or annotate.
- **Confidence:** Medium

#### [F-058] `AlphaSchedule::scaled_linear` returns a vestigial Result
- **Category:** dead-code · **Severity:** Low · **Location:** `gen-core/src/sampling.rs:121-143`
- **Finding:** no fallible operation in the body; ~10 call sites carry dead `.unwrap()`s. (Alternatively: it's missing the `num_train_timesteps == 0` validation that would justify the signature.)
- **Impact:** noise + a false failure-mode signal to the candle port.
- **Suggested fix:** return `Self`, or add the missing zero-steps validation.
- **Confidence:** High

#### [F-059] Testkit trainer-conformance failure path reloads the full trainer to print an id
- **Category:** efficiency · **Severity:** Low · **Location:** `gen-core-testkit/src/trainer.rs:327-334`
- **Finding:** the aggregated-failure branch calls `make()` a fourth time solely for `descriptor().id` — a multi-GB model load on the real lane.
- **Impact:** a failing conformance run pays an extra multi-minute load before panicking; a flaky load replaces the panic message.
- **Suggested fix:** capture the id from the first `make()`.
- **Confidence:** High

#### [F-060] Guard `Chunk` row slicing against non-divisible fused LoRA factors
- **Category:** bad-pattern · **Severity:** Low · **Location:** `src/adapters/loader.rs:729-737`
- **Finding:** `Chunk { n, index }` never checks `rows % n == 0`; truncating division slices wrong ranges and silently drops trailing rows (contrast `ChunkIfDivisible`).
- **Impact:** an off-spec BFL/ComfyUI LoRA installs shifted q/k/v deltas with `applied > 0` — visually wrong, no error.
- **Suggested fix:** typed error when `rows % n != 0`.
- **Confidence:** High

#### [F-061] Validate the attention-mask length in core `build_mask`
- **Category:** bad-pattern · **Severity:** Low · **Location:** `src/nn.rs:511-526`
- **Finding:** `am[bi*s + j]` indexes a caller-supplied slice against independent `b`/`s` params — a short mask panics in shared core.
- **Impact:** a provider bug aborts the process instead of failing the encode with a typed error.
- **Suggested fix:** check `am.len() == b*s` up front.
- **Confidence:** High

#### [F-062] Add the sibling guards to `window_partition`/`window_unpartition`
- **Category:** bad-pattern · **Severity:** Low · **Location:** `src/nn.rs:531-572`
- **Finding:** no rank-4 guard, no `window > 0` guard (divide-by-zero), and `windows.shape()[0] / num_per_image` can divide by zero for degenerate `pad_hw` — while adjacent `upsample_nearest`/`group_norm` carry exactly these typed-error checks.
- **Impact:** a malformed trunk config aborts the process.
- **Suggested fix:** mirror the F-041 guards.
- **Confidence:** High

#### [F-063] Harden `FlowMatchEuler::timestep`/`num_steps` like the already-guarded step fn
- **Category:** bad-pattern · **Severity:** Low · **Location:** `src/scheduler.rs:91-99`
- **Finding:** `num_steps()` is `sigmas.len() - 1` (underflow on empty, reachable via pub `from_sigmas(vec![])`); `timestep(t)` is an unguarded index — while `flow_match_euler_step` was hardened (F-042).
- **Impact:** a short curated schedule panics denoise setup instead of erroring.
- **Suggested fix:** `len() >= 2` typed error in `from_sigmas`/`new`.
- **Confidence:** High

#### [F-064] Cast defensively in `decoded_to_image` before host readback
- **Category:** bad-pattern · **Severity:** Low · **Location:** `src/image.rs:52-56`
- **Finding:** `flat.as_slice::<f32>()` panics on a dtype mismatch; a provider that forgets the final f32 cast (new bf16 VAE path / PiD impl) aborts at the last step of a render. Also `sh[1..3]` has no rank guard.
- **Impact:** one missed cast in ~20 provider crates becomes a process abort.
- **Suggested fix:** `as_dtype(Float32)?` before readback (no-op when already f32) + a rank check.
- **Confidence:** High

#### [F-065] Reject rank ≤ 0 in the training-side LoRA/LoKr target builders
- **Category:** bad-pattern · **Severity:** Low · **Location:** `src/train/lora.rs:265-334, 340-373`
- **Finding:** `install_training_lokr` passes caller `rank` into `reconstruct_lokr_delta` (`alpha/rank`); `rank == 0` makes the initial "no-op" delta NaN via `0·inf`. The load-side guard (sc-5252) has no training-side counterpart.
- **Impact:** a `rank: 0` training config yields an immediately-NaN run rather than a validation error.
- **Suggested fix:** validate `rank > 0` in `build_lora_targets`/`build_lokr_targets`.
- **Confidence:** Medium

#### [F-066] Batch the Prodigy host syncs
- **Category:** efficiency · **Severity:** Low · **Location:** `src/train/optim.rs:319, 335` (via `sum_all`)
- **Finding:** pass 1 calls `.item::<f32>()` twice per trainable factor — hundreds of serialized Metal round-trips per optimizer step on typical DiT target lists.
- **Impact:** measurable per-step Prodigy overhead growing with target count; defeats lazy batching.
- **Suggested fix:** accumulate lazy scalar Arrays across the loop and `.item()` once each (2 syncs total).
- **Confidence:** High

#### [F-067] Extract the triplicated LyCORIS resolve-and-install skeleton
- **Category:** redundant · **Severity:** Low · **Location:** `src/adapters/loader.rs:167-184, 320-347, 447-471`
- **Finding:** `apply_lokr`/`apply_lokr_thirdparty`/`apply_loha_thirdparty` repeat the identical install body; the sc-8395 prefix-strip fix had to land twice and missed the peft variant (harmless today only because peft paths are bare).
- **Impact:** the next resolution fix lands in three places or silently skews LoKr vs LoHa.
- **Suggested fix:** one `install_lycoris_groups` helper (the `install_bfl_lycoris` shape already exists).
- **Confidence:** High

#### [F-068] Extract the copy-pasted cancel/eval/progress preamble in the three curated-sampler drivers
- **Category:** redundant · **Severity:** Low · **Location:** `src/sampler.rs:308-324, 369-379, 510-520`
- **Finding:** the identical 5-line cancel→eval→progress→delegate block is inlined three times; the explanatory comments exist only on the first copy. (Progress derivation is also O(steps²) — see F-137.)
- **Impact:** a cancel/progress contract change must be replicated thrice.
- **Suggested fix:** a small shared `step_gate` helper.
- **Confidence:** High

#### [F-069] Refactor the 6-flag adapter-format dispatch into a classified enum
- **Category:** readability · **Severity:** Low · **Location:** `src/adapters/loader.rs:1168-1270`
- **Finding:** six interdependent booleans whose precedence is enforced only by comments, with lazily-built caches threaded through `as_ref().unwrap()` at four sites.
- **Impact:** the next adapter format requires re-deriving precedence from prose; a mis-ordered predicate silently misroutes files (F-012 is the live example).
- **Suggested fix:** one `classify() -> AdapterFormat` enum with unit tests per format; flat `match` dispatch.
- **Confidence:** Medium

#### [F-070] Wan `frames_to_images` computes the pixel count in i32 — overflows at request-reachable scales
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-wan/src/pipeline.rs:894-898`
- **Finding:** `let total: i32 = f * h * w * c;` on the decoded full-resolution video. `frames` is validated only as `1+4k` with no upper bound and T2V-14B has `max_area == 0` (uncapped), so `f·h·w·3 > i32::MAX` is reachable (e.g. 1920×1088 @ 349 frames, or 4K @ 89). Release builds wrap (negative/wrong reshape dim → confusing error); debug panics. The four sibling implementations (sdxl/lens/svd/core) were fixed by F-053/F-068/F-076/F-082; this copy was missed.
- **Impact:** an extreme-but-valid request dies at decode with a baffling reshape error instead of a validation message.
- **Suggested fix:** size in `usize`/`i64` and reshape via `&[-1]`, matching the siblings.
- **Confidence:** High

#### [F-071] Clamp the `.pth` reader's pre-allocation to the archive size
- **Category:** security · **Severity:** Low · **Location:** `mlx-gen-wan/src/pth.rs:642, 694`
- **Finding:** `reserve(entry.compressed_size() as usize)` trusts the zip central-directory size — a crafted archive claiming a multi-hundred-GB STORED entry triggers a huge up-front allocation before a byte is read, bypassing the module's own bomb guards (F-015/016/017).
- **Impact:** allocation-abort DoS when converting an untrusted `.pth`.
- **Suggested fix:** clamp the reserve hint to the file length or use `try_reserve`.
- **Confidence:** Medium

#### [F-072] Deduplicate the WanVace validators and the ~90-line VACE generate bodies
- **Category:** redundant · **Severity:** Low · **Location:** `mlx-gen-wan/src/model_vace.rs:215-241` vs `669-695`; `496-562` vs `248-317`
- **Finding:** `validate_vace_clip` is documented "shared" but only `WanVaceFun` uses it — `WanVace` carries a byte-equivalent private copy; the two `generate_impl`s duplicate the entire pre-DiT setup and decode tail verbatim.
- **Impact:** fixes (e.g. F-014's decode cancel) must land twice; the "shared" doc misleads.
- **Suggested fix:** repoint `WanVace::validate_impl` at the shared fn; extract the pre-DiT/post-DiT helpers.
- **Confidence:** High

#### [F-073] Batch or document the VACE 2×B=1 CFG forwards
- **Category:** efficiency · **Severity:** Low · **Location:** `mlx-gen-wan/src/vace.rs:875-895, 955-982`
- **Finding:** `denoise_vace[_moe]` run cond and uncond as two sequential B=1 forwards, while the base pipeline deliberately batches B=2 for a measured win (sc-2853).
- **Impact:** ~2× per-step kernel-launch overhead on the VACE CFG path.
- **Suggested fix:** batch the contexts if VACE parity permits; otherwise document the deliberate divergence at the loop.
- **Confidence:** Medium

#### [F-074] Move the curated-sampler × TI2V rejection into validate, and define Reference+Keyframe precedence
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-wan/src/model.rs:390-397, 322-367`
- **Finding:** the curated×image-conditioned rejection fires in Stage 2, after the ~11 GB UMT5 load and VAE encode, though both inputs are known at validate time; separately, a request with both `Reference` and `Keyframe`s silently drops the Reference (undocumented precedence).
- **Impact:** rejected requests burn tens of seconds first; user conditioning silently ignored.
- **Suggested fix:** move the check into `validate_impl`; either reject the combination or fold Reference in as the frame-0 keyframe, documented.
- **Confidence:** High

#### [F-075] LTX config fields are parsed but never enforced
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-ltx/src/config.rs:66-77, 349-356`
- **Finding:** `rope_type`, `double_precision_rope`, `spatial_padding_mode`, `timestep_conditioning` etc. are parsed but consumed nowhere — the runtime hardcodes SPLIT f64 RoPE, zero padding, no timestep conditioning; the `timestep_conditioning` doc claims a gate that doesn't exist.
- **Impact:** a divergent future checkpoint loads and decodes silently wrong instead of erroring.
- **Suggested fix:** reject unsupported values at load, or drop the unread fields and fix the doc.
- **Confidence:** Medium

#### [F-076] Error (or log) on quant-bits mismatch over pre-packed snapshots
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-qwen-image/src/model.rs:120-123`, `src/model_edit.rs:118-121`, `src/model_control.rs:111-117`; `mlx-gen-krea/src/model.rs:124-126` (pattern shared by all Group-B loaders)
- **Finding:** load paths run `transformer.quantize(q.bits())` — a documented no-op on an already-packed snapshot — with no comparison of requested vs packed bits; `spec.quantize = Q4` over a Q8-packed turnkey silently serves Q8.
- **Impact:** the worker believes it loaded the requested tier; footprint/speed/quality are the other tier's with no signal.
- **Suggested fix:** when `.scales` is detected, read the snapshot's `config.json` `quantization.bits` (both converters write it — except sensenova, see F-045) and error/log on mismatch.
- **Confidence:** Medium

#### [F-077] Thread the request cancel flag into the Krea and Lens training preview renders
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-krea/src/training.rs:888-896, 595-597`; `mlx-gen-lens/src/pipeline.rs:468-507` + `mlx-gen-lens/src/training.rs:614-618`
- **Finding:** both `render_sample`s run their full denoise + decode without consulting `req.cancel` (krea constructs a fresh `CancelFlag::new()`; lens documents "no cancel plumbing"); the trainers check only between preview prompts.
- **Impact:** a cancel during a preview burst blocks for a whole render (tens of seconds on the 12B Raw DiT / high `sample_steps`).
- **Suggested fix:** pass `&req.cancel` into both `render_sample`s and check per step.
- **Confidence:** High

#### [F-078] Deduplicate the qwen/krea text-encoder leaves (`build_mask`, `repeat_kv`, half-split RoPE)
- **Category:** redundant · **Severity:** Low · **Location:** `mlx-gen-qwen-image/src/text_encoder/encoder.rs:136-156` (build_mask, byte-identical to `mlx_gen::nn::build_mask`); `mlx-gen-qwen-image/src/text_encoder/attention.rs:96-124` + `mlx-gen-krea/src/text_encoder/attention.rs` + `mlx-gen-krea/src/transformer/mod.rs:469-478` (repeat_kv/rotate-half ×3)
- **Finding:** parity-critical leaf math re-implemented per crate; krea already imports core `build_mask` while qwen carries a local copy, and krea itself has two `repeat_kv`s.
- **Impact:** drift-prone copies of exactly the class the core hoist targeted.
- **Suggested fix:** hoist `apply_half_split_rope`/`repeat_kv` next to `TextRope` in `mlx_gen::nn`; delete the local `build_mask`. Epic 7778 territory.
- **Confidence:** High

#### [F-079] Cache Krea's step-invariant conditioning and RoPE; use splits over gathers
- **Category:** efficiency · **Severity:** Low · **Location:** `mlx-gen-krea/src/transformer/mod.rs:241-305, 462-465`; `mlx-gen-krea/src/pipeline.rs:152-156`
- **Finding:** `joint_inputs` reruns the text-fusion aggregator, `txt_in`, and the host-loop `RopeTables::build_t2i` (~1M trig ops at 1024²; 4M+ at 2048²) every one of the 8 steps — qwen step-caches the identical tables (F-115); `slice_axis1` materializes arange gathers where `split_sections` suffices (the F-114 fix in qwen).
- **Impact:** redundant per-step GPU work and host trig/upload on every render.
- **Suggested fix:** hoist context projection + RoPE build out of the step closure; implement `slice_axis1` via `split_sections`.
- **Confidence:** High

#### [F-080] Load the qwen edit text_encoder directory once, not twice
- **Category:** efficiency · **Severity:** Low · **Location:** `mlx-gen-qwen-image/src/loader.rs:63-85`
- **Finding:** `load_vision_language_encoder` runs `Weights::from_dir` over the same ~16 GB shard set twice (LM pass + `visual.*` pass).
- **Impact:** doubled shard-header parsing/handles/maps on every `qwen_image_edit` load.
- **Suggested fix:** load once, remap, build both towers from the single `Weights`.
- **Confidence:** High

#### [F-081] Guard `encode_reference_latents` and Krea's TE prefix-drop like their qwen siblings
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-qwen-image/src/vl_tokenizer.rs:156-183`; `mlx-gen-krea/src/text_encoder/encoder.rs` (prefix-drop)
- **Finding:** the pub `encode_reference_latents` indexes an unvalidated buffer (`resized[(y*cw+x)*3+c]`) — the F-020 guard was applied to `QwenImageProcessor::preprocess` but not this equally-public sibling; krea's `(prefix_tokens..n)` gather has no `n > prefix_tokens` check (qwen guards the identical op in both of its encoders).
- **Impact:** direct callers panic OOB / hit an opaque `take_axis` panic instead of the clean errors the qwen twins produce.
- **Suggested fix:** copy the respective qwen guards.
- **Confidence:** High

#### [F-082] SDXL: don't silently drop the scheduler under accel samplers; guard packed CN/IP loads; dedup the pipeline triplication
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-sdxl/src/model.rs:351-359`; `mlx-gen-sdxl/src/loader.rs:144-195` (+ `mlx-gen-kolors/src/ip_adapter.rs:38-44`); `mlx-gen-sdxl/src/pipeline.rs:359-431, 508-568, 611-663`
- **Finding:** three related SDXL-family items: (a) `{sampler: "lcm", scheduler: "karras"}` validates but the scheduler is silently dropped on the accel branch; (b) `load_controlnet`/IP-adapter loaders `cast_all` without the `!is_packed` guard the UNet loader has — a pre-quantized CN snapshot would be corrupted rather than rejected (latent; converter doesn't pack CN today); (c) the ~45-line ControlNet-sum/UNet-dispatch/CFG block is triplicated across `denoise_core`/`denoise_curated`/`denoise_cfgpp`, and `denoise_curated` missed the sc-7443 migration to shared `gen_core::guidance::cfg`.
- **Impact:** silent request downgrade; a packed-CN trap; three copies to keep in lock-step (one already drifted in form).
- **Suggested fix:** (a) error on accel+curated-scheduler; (b) add the `is_packed` guard or reject packed CN; (c) extract a `predict_eps` helper and migrate the curated combine.
- **Confidence:** High

#### [F-083] Hoist the Kolors betas and the per-count VAE re-encode
- **Category:** redundant · **Severity:** Low · **Location:** `mlx-gen-kolors/src/model.rs:354` + `sampler.rs:48-49` + `training.rs:73-74`; `mlx-gen-kolors/src/registry.rs:368-391, 422-424, 473-474`
- **Finding:** the scaled-linear betas (`0.00085, 0.014`) appear as literals in three files; `encode_init_latents` runs inside the count loop though it draws no RNG (the F-068 hoist SDXL already applies).
- **Impact:** schedule drift risk; a full f32 VAE forward wasted per extra image on img2img/pose tiers.
- **Suggested fix:** named consts in one place; hoist the encode above the loop.
- **Confidence:** High

#### [F-084] Give zero-dimension request images typed errors on the SDXL/InstantID paths
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-sdxl/src/pipeline.rs:684-706`; `mlx-gen-instantid/src/kps.rs:559-573`
- **Finding:** a 0×0 image with an empty buffer passes `preprocess_image`'s length check and reaches the core resize `assert!`; `preprocess_clip_image_sized` in the same crate returns a typed error for the identical input.
- **Impact:** inconsistent failure mode — panic vs recoverable error — for the same bad-request class.
- **Suggested fix:** add the `iw == 0 || ih == 0` guard from `preprocess_clip_image_sized` to `preprocess_image` and `letterbox`'s callers.
- **Confidence:** Medium

#### [F-085] Flux-family robustness nits: control-scale 0.0 remap, boogu `build_plan` unwraps
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-flux/src/model_control.rs:199-206`; `mlx-gen-boogu/src/vision/mod.rs:380-383`
- **Finding:** (a) an explicit control `scale: 0.0` is indistinguishable from unset and silently becomes 0.7 (exact float compare on user input; `Conditioning::Control::scale` is a bare f32); (b) `build_plan` ends with three `.unwrap()`s on fallible MLX ops — the only such unwraps on a production generate path in the family.
- **Impact:** (a) users can't express "control inert" for A/B; (b) a Metal exception aborts the process instead of bridging to `gen_core::Error` (undermines the sc-5009 recoverable-command-buffer work).
- **Suggested fix:** (a) `scale: Option<f32>` upstream (matching `Reference::strength`) or reject 0.0; (b) return `Result<Plan>` and `?` the ops.
- **Confidence:** High

#### [F-086] Boogu duplicates flux's ~80-line VAE key remap verbatim
- **Category:** redundant · **Severity:** Low · **Location:** `mlx-gen-boogu/src/loader.rs:77-156` vs `mlx-gen-flux/src/loader.rs:131-208`
- **Finding:** `remap_vae_decoder`/`remap_vae_encoder` are line-for-line copies (diffusers `AutoencoderKL` → z-image remap + NCHW→NHWC transposes); chroma solved the same need by reusing `mlx_gen_flux::load_vae`.
- **Impact:** a fiddly key/transpose map maintained twice.
- **Suggested fix:** hoist into `mlx-gen-z-image` (both already depend on it) next to `Vae`.
- **Confidence:** High

#### [F-087] flux2: surface the kv-edit `image_guidance` no-op and fix the env-var precedence
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-flux2/src/model.rs:717-727, 777-783`
- **Finding:** (a) on `flux2_klein_9b_kv_edit`, `req.image_guidance` validates but the per-step gate (`include_ref && cache_ref.is_none()`) silently disables it whenever the KV cache is active; (b) the debug env var `FLUX2_IMG_GUIDANCE` **overrides** the request value (`.ok().and_then(parse).or(req.image_guidance)`) — a leaked var changes pixels with no trace.
- **Impact:** identity-strength appears broken on the kv variant; env leakage silently alters output.
- **Suggested fix:** reject `image_guidance` on the kv variant at validate; invert env precedence (request wins) or gate behind `debug_assertions`.
- **Confidence:** High

#### [F-088] flux2 edit variants don't require a reference at validate time
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-flux2/src/model.rs:375-387, 827-844` (vs `model_control.rs:335-341`)
- **Finding:** the "at least one reference" requirement is enforced only inside `generate`; the sibling `Flux2DevControl` enforces its required Control in `validate`.
- **Impact:** a worker probing `validate()` gets a false "valid" for a request that fails after model load + prompt encode.
- **Suggested fix:** add the reference-present check to the edit variants' validate closure.
- **Confidence:** High

#### [F-089] z-image polish: per-variant error ids, aligned control kinds, TE hidden-state retention
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-z-image/src/model.rs:278-332`; `model_control.rs:135-139` vs `model_base_control.rs:150-175`; `text_encoder/encoder.rs:99-106`
- **Finding:** (a) the shared `validate_request` hardcodes `"z_image_turbo:"` in errors emitted for all four ids; (b) base control restricts to `Only([Pose, Canny, Depth])` while turbo control accepts `Any` — same checkpoint, inconsistent contract; (c) `TextEncoder::forward` accumulates all 37 layer outputs (~180 MB f32) to return `hidden[len-2]`.
- **Impact:** wrong ids in logs; `Other("scribble")` accepted on turbo and rejected on base; needless activation retention during encode.
- **Suggested fix:** thread the model id; share `accepted_kinds()`; track only prev/cur layers.
- **Confidence:** High

#### [F-090] Deduplicate the four z-image and two sana load bodies
- **Category:** redundant · **Severity:** Low · **Location:** `mlx-gen-z-image/src/{model.rs:119-173, model_base.rs:118-159, model_control.rs:90-128, model_base_control.rs:106-144}`; `mlx-gen-sana/src/model.rs:162-231`
- **Finding:** the four z-image `load` fns repeat the precision guard/snapshot descent/quantize-triple/adapter block nearly verbatim (drift already visible in error prefixes); sana's `load`/`load_sprint` duplicate ~30 lines.
- **Impact:** six places to keep in lockstep for one loading policy.
- **Suggested fix:** crate-private `load_components` helpers.
- **Confidence:** High

#### [F-091] Guard SANA's degenerate scheduler constructions and pipeline dims
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-sana/src/scm.rs:66-95`; `mlx-gen-sana/src/pipeline.rs:73-82, 368-426`
- **Finding:** `with_timesteps(0, ..)` produces a `[NaN]` schedule; `from_timesteps(vec![])` makes `num_steps()` underflow; the public `generate` never validates the multiple-of-32 rule (integer division silently truncates a 1000×1024 request to 992 px) — the boundary check lives only in the Generator adapter.
- **Impact:** direct pipeline/scheduler consumers get raw noise, panics, or silently smaller images — the exact F-033 class z-image fixed at its boundary.
- **Suggested fix:** `max(1)` + `saturating_sub`; validate dims at `generate_with`.
- **Confidence:** High

#### [F-092] Error on SANA guidance-embedder/scalar mismatch; use cfg.norm_eps; gate the final-step noise draw
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-sana/src/transformer.rs:574-581, 601`; `mlx-gen-sana/src/pipeline.rs:245-259`
- **Finding:** (a) `forward_with_guidance`'s catch-all match arm silently drops guidance conditioning on a base/Sprint mixup; (b) the output norm hardcodes `1e-6` where every sibling uses `cfg.norm_eps`; (c) `denoise_sprint` draws and discards a full-latent `randn_like` on the final step of every multi-step run (the comment claims a gate the code lacks).
- **Impact:** silently-wrong distilled output on composition mistakes; a latent config divergence; wasted kernels per generation.
- **Suggested fix:** `Err` on the mismatch arms; `cfg.norm_eps`; gate on `i + 1 < n`.
- **Confidence:** Medium

#### [F-093] Reconcile the CFG-activation gate: z-image `!= 1.0` vs sana/diffusers `> 1.0`
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-z-image/src/model_base.rs:180` (+ `model_base_control.rs:193`, `pipeline.rs:166/279`) vs `mlx-gen-sana/src/pipeline.rs:106, 393`
- **Finding:** sana gates CFG on `guidance > 1.0` (with a diffusers-parity citation); z-image base gates on `!= 1.0`, so `guidance = 0.5` runs the double forward and interpolates toward uncond.
- **Impact:** sub-1.0 guidance behaves differently across families; if the reference `ZImagePipeline` uses the standard gate, z-image diverges for that input range.
- **Suggested fix:** verify the reference gate and align, or document the intended `!= 1.0` surface.
- **Confidence:** Medium

#### [F-094] SD3 nits: silently-ignored negative at guidance==1.0, duplicate tokenize, arch-validator duplication
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-sd3/src/model.rs:203-214`; `mlx-gen-sd3/src/pipeline.rs:79-80`; `mlx-gen-sd3/src/convert.rs:332-380, 453-474` vs `mlx-gen-sd3/src/vae.rs:279-327`
- **Finding:** (a) `guidance: Some(1.0)` + `negative_prompt` validates but the negative is never encoded (the class the Turbo variant explicitly rejects); (b) `encode_prompt` tokenizes the same prompt twice through the same tokenizer for clip_l/clip_g; (c) the transformer and VAE arch validators are ~50-line near-verbatim duplicates (the VAE copy also lacks the `-1` shape wildcard).
- **Impact:** silently-discarded conditioning; wasted BPE; two validators to keep in sync.
- **Suggested fix:** validate the combination; tokenize once; extract a generic `validate_tensor_set`.
- **Confidence:** High

#### [F-095] Route the SenseNova interleave loop through decode_argmax/append_tokens
- **Category:** efficiency · **Severity:** Low · **Location:** `mlx-gen-sensenova/src/t2i.rs:1208-1213, 1233-1238`
- **Finding:** the interleave AR loop still uses `decode_logits` (full-vocab ~600 KB host copy per token) + host argmax — the pattern F-140 already replaced in `Qwen3Backbone::generate`; the two `<img>`-append sites compute and discard full logits where `append_tokens` skips the lm_head entirely.
- **Impact:** per-token host transfer + an unnecessary vocab-width matmul in Document Studio's longest loop.
- **Suggested fix:** `decode_argmax` for the loop; `append_tokens` for the appends.
- **Confidence:** High

#### [F-096] scail2: surface trailing-frame drops, aggregate multi-segment progress, guard the bf16 escape hatch
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-scail2/src/generate.rs:214-231, 477-521, 374-386`
- **Finding:** (a) `build_segments` silently discards driving frames past the last full window (no tests pin the plan); (b) progress restarts at 1 per segment, so multi-segment jobs appear to finish and restart; (c) `SCAIL2_COMPUTE_BF16=1` re-enables the documented NaN-overflow path (sc-5681) with no NaN detection — `pixels_to_u8`'s min/max clamp propagates NaN into garbage pixels silently.
- **Impact:** silently-short output; unusable overall progress; the experimental flag reopens the exact sc-5690 debugging loop.
- **Suggested fix:** emit/handle a final short segment (verify vs upstream) + unit tests; report `current = seg·steps + i + 1`; a per-segment `isnan().any()` typed error when the flag is set.
- **Confidence:** High (a, b verified; a's upstream behavior unverified)

#### [F-097] Bernini robustness: NaN-unsafe sort, swallowed sidecar JSON errors, dup helpers, per-step re-embeds
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-bernini/src/bernini.rs:996-1003, 116-163`; `mlx-gen-bernini/src/config.rs:37-78`; `mlx-gen-bernini/src/forward.rs:115-152, 217-326`; `preprocess.rs:20-22`
- **Finding:** (a) `seeded_permutation` sorts with `partial_cmp().unwrap()` (source is `random::normal`, NaN-free today — belt-and-braces `total_cmp`); (b) `PlannerKnobs::from_dir`/`read_mrope_config`/`BerniniKnobs::from_dir` swallow parse errors (`.ok()…unwrap_or`), so a corrupt sidecar silently reverts every knob to defaults; (c) `drop_batch` is defined twice and `FullDefaults` re-declares 7 shared constants (burying the one real delta, ETA); (d) `PackedForward` re-embeds step-invariant conditioning and rebuilds RoPE tables ~160-200× per generation.
- **Impact:** latent panic; silent quality/geometry degradation on damaged snapshots; drift risk; pure per-step waste.
- **Suggested fix:** `total_cmp`; distinguish NotFound from parse-Err (as `Scail2Config` does); share the helper/constants; cache per-expert source tokens + cos/sin.
- **Confidence:** High

#### [F-098] SVD: decorative configs and a false "override from disk" doc
- **Category:** dead-code · **Severity:** Low · **Location:** `mlx-gen-svd/src/image_encoder.rs:37`; `mlx-gen-svd/src/config.rs:1-114`; `mlx-gen-svd/src/scheduler.rs:54-57`
- **Finding:** `SvdImageEncoder::from_weights(w, _cfg)` ignores its config entirely (hardcodes `vit_h_14()`); several UnetConfig/VaeConfig fields are never read; the module doc claims the loader "can still override from disk" but `load()` parses no JSON; `EdmSchedule::num_steps()` has no caller.
- **Impact:** a checkpoint with different geometry loads wrong shapes with no config-level signal; readers misled about an override path.
- **Suggested fix:** drop the unread fields or actually read the checkpoint JSONs; fix the doc; delete `num_steps()`.
- **Confidence:** High

#### [F-099] seedvr2: empty-clip dispatch, missing per-chunk progress, crop gathers, neg-embed dup
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-seedvr2/src/registry.rs:144-151, 171-188`; `mlx-gen-seedvr2/src/pipeline.rs:160-166, 241-244/290-294/476-480/547-551`
- **Finding:** (a) `validate_impl` accepts when *any* clip is non-empty but `generate_impl` takes the *first* unconditionally — `[empty clip, non-empty clip]` silently returns an empty video; (b) the video path emits one `Step{1,1}` for a potentially minutes-long N-chunk run (no liveness); (c) `decode_crop` builds arange gathers (full-frame copies) where prefix slices suffice, and the style crop is a structural no-op; (d) the neg-embed-missing error block is copy-pasted four times.
- **Impact:** validated requests yielding empty output; frozen progress; wasted copies; message drift.
- **Suggested fix:** select the first non-empty clip; per-chunk `Step{k, plan.len()}`; range slicing; a `require_neg()` helper.
- **Confidence:** High

#### [F-100] PiD contract-boundary guards: latent rank and eps length
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-pid/src/decoder.rs:54-58, 65`; `mlx-gen-pid/src/sampler.rs:84-86`
- **Finding:** `decode` indexes `sh[2]`/`sh[3]` with no ndim check on a cross-crate trait input (a packed-3D-vs-unpacked-4D mixup was the exact sc-7847 confusion); pub `Sampler::run` indexes caller-supplied `eps` with no length check.
- **Impact:** a mis-wired provider panics with an opaque index error at the contract boundary instead of a typed `Error`.
- **Suggested fix:** `ndim != 4` (and channel-count) check at `decode`; `eps.len() >= num_eps()` at `run`.
- **Confidence:** High

#### [F-101] face: extreme aspect ratio yields det_scale = 0 → inf coordinates
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-face/src/face.rs:147-155` (interacting with `scrfd.rs:303-306`)
- **Finding:** `new_h = (det * im_ratio) as usize` truncates to 0 for aspect < 1/640, giving `det_scale = 0.0` and a `1/det_scale = inf` rescale; the F-030 guard covers only exactly-zero dims.
- **Impact:** silent inf/NaN detection coords feeding norm_crop/umeyama rather than a typed rejection.
- **Suggested fix:** clamp `new_w/new_h` to `max(1)` or reject such ratios in `detector_blob`.
- **Confidence:** Medium

#### [F-102] Remove (or exercise) the dangling gen-core-testkit dev-deps in boogu and pid
- **Category:** dead-code · **Severity:** Low · **Location:** `mlx-gen-boogu/Cargo.toml:33`; `mlx-gen-pid/Cargo.toml:22`
- **Finding:** both declare the testkit with zero references anywhere in their tests — boogu's is a tell that conformance coverage was intended and never written.
- **Impact:** stale declarations; feeds the F-009 coverage gap.
- **Suggested fix:** write the boogu conformance test; drop pid's dep.
- **Confidence:** High

#### [F-103] Transform registration kind is dead in this workspace and the docs misdescribe seedvr2
- **Category:** dead-code · **Severity:** Low · **Location:** `gen-core/src/registry.rs:28-33, 77-79, 118`; `gen-core/src/transform.rs:14-19`; `docs/MODEL_ARCHITECTURE.md:57, 138`
- **Finding:** `TransformRegistration`/`transforms()`/`load_transform` have no `register_transform!` macro, zero implementors, zero submitters; seedvr2 — which the docs say is "Transform, not Generator" — registers via `register_generators!`. (Caveat: gen-core is shared with candle-gen, which may use the plumbing.)
- **Impact:** dead contract plumbing plus an architecture doc that would lead a new restorer port into an unregisterable trait.
- **Suggested fix:** add the macro and move seedvr2 onto it as documented, or update §2/§3.3 and mark the kind reserved.
- **Confidence:** Medium

#### [F-104] Replace five crate-local compile_glue clones with the core re-export
- **Category:** redundant · **Severity:** Low · **Location:** `mlx-gen-z-image/src/lib.rs:91-127`; `mlx-gen-sdxl/src/lib.rs:85-120`; `mlx-gen-wan/src/transformer.rs:55-95`; `mlx-gen-ltx/src/lib.rs:111-144`; `mlx-gen-qwen-image/src/transformer/mod.rs:49+` (canonical: `src/nn.rs:113-162`)
- **Finding:** discipline is clean (all 21 production sites use the RAII guard; the F-006 leak class is fully remediated), but five crates carry their own ~40-line `AtomicBool + guard` copies while flux/flux2/chroma correctly re-export the core one.
- **Impact:** six independent toggles for one concept; a core guard fix won't propagate.
- **Suggested fix:** `pub use mlx_gen::nn::{set_compile_glue, compile_glue, CompileGlueGuard};` in the five crates.
- **Confidence:** High

#### [F-105] Lens API hygiene: duplicate selected_layers panic, stale descriptor doc, dead consts, dup defaults/helpers
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-lens/src/text_encoder/encoder.rs:70-90, 143-151`; `mlx-gen-lens/src/registry.rs:58-60, 240`; `mlx-gen-lens/src/reasoner.rs:29` + `text.rs:157-159`; `schedule.rs:20-36` vs `registry.rs:40-56`; `dit/transformer.rs:100-106` + `dit/attention.rs:30-34` + `dit/block.rs:32-36`; `dit/transformer.rs:222-250` vs `306-331`
- **Finding:** (a) `with_selected_layers` docs require uniqueness but only validate non-empty/in-range — duplicates panic later at `c.expect("every selected layer captured")`; (b) the descriptor doc says "no quant (yet), no LoRA" while the body advertises both; (c) `DEFAULT_TEMPERATURE` (no temperature sampling exists — if the vendor path samples, that's an unsurfaced capability divergence) and `HARMONY_END` are unused; (d) the sampling defaults exist twice (`schedule::{TURBO,BASE}` unused; registry re-declares the numbers); (e) the packed-detect load helper is triplicated; (f) `forward` and `forward_with_main_checkpointed` duplicate the ~25-line front-end; (g) dead `let _ = i;` in the generate loop.
- **Impact:** panic on a documented-valid input; misleading docs; drift-prone duplication of parity-critical front-end math.
- **Suggested fix:** reject duplicates at the setter; fix the doc; delete or implement the consts (file a story if the vendor samples); consume `schedule::` from registry; one `load_adaptable`; factor `prepare_streams`; delete the dead line.
- **Confidence:** High

#### [F-106] Emit Progress::Decoding before decoding, not after (lens)
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-lens/src/registry.rs:231-241`; `mlx-gen-lens/src/pipeline.rs:434-437`
- **Finding:** `generate_with_progress` decodes internally and returns the finished image; the registry then emits `Decoding` post-hoc, attributing the multi-second decode to the last denoise step.
- **Impact:** misleading phase display; no correctness impact.
- **Suggested fix:** emit inside the pipeline just before `vae::decode`.
- **Confidence:** High

#### [F-107] Fix the sc-8797 rehost miss in lens trainer_e2e
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-lens/tests/trainer_e2e.rs:26-38` (vs `src/training.rs:963-977`)
- **Finding:** sc-8797 moved the training-base lookup to `models--SceneWorks--Lens` in the src harness but `trainer_e2e.rs`'s fallback still reads `models--microsoft--Lens` (the pulled repo).
- **Impact:** the trainer e2e can no longer find weights via the HF cache on a fresh machine (only the env override works).
- **Suggested fix:** point the fallback at the SceneWorks rehost; ideally share one `snapshot()` helper.
- **Confidence:** High

#### [F-108] SAM polish: dead zip loop, uncached resize matrices, dup helpers, NaN-argmax convention, frame_idx bounds
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-sam3/src/video.rs:228-231`; `mlx-gen-sam3/src/tracker.rs:731-744, 1855-1863`; `mlx-gen-sam3/src/mask.rs:26-29`; `mlx-gen-sam3/src/model.rs:261-268` vs `detr.rs:728-735`; `mlx-gen-sam2/src/video_predictor.rs:562-621`; `mlx-gen-sam2/src/video_predictor.rs:813-815, 871-873` + `mlx-gen-sam3/src/video.rs:174-176`
- **Finding:** (a) a literally-no-op `for … { let _ = (oid, di); }` loop; (b) `bilinear_resize_matrix` (1.2–4.6 MB host build) rebuilt per object per frame — the pattern F-167/sc-6954 already memoized twice elsewhere; (c) `mask.rs` re-copies `util::conv_w_ohwi`, and the additive key-mask builder exists under two names; (d) sam3's local `argmax` uses `x > best` (NaN → arbitrary index 0) instead of sam2's F-169 `total_cmp` convention; (e) `frame_pixels`/`add_new_box` gather with an unvalidated caller `frame_idx`; (f) all three propagation loops report `cb(i, total)` post-frame, so `done == total` is never observed.
- **Impact:** noise, per-frame waste, convention drift, garbage-features-from-bad-index, and progress that never completes.
- **Suggested fix:** delete the loop; memoize by `(in,out)`; use the util helpers; mirror `argmax_f32`; bounds-check `frame_idx`; report `i + 1`.
- **Confidence:** High

#### [F-109] JoyCaption clones the full image buffer twice per caption
- **Category:** efficiency · **Severity:** Low · **Location:** `mlx-gen-joycaption/src/model.rs:112-118, 150`
- **Finding:** `normalized_request` deep-clones the request (pixels included) just to maybe substitute a default prompt, then `ImageRef::new(…, pixels.clone())` copies again.
- **Impact:** two redundant multi-MB copies per caption in a hot worker path.
- **Suggested fix:** compute the effective prompt as `Cow<str>` and pass pixels from the original request.
- **Confidence:** High

#### [F-110] Ideogram polish: private MAX_TEXT_TOKENS shadowing, u8→i32 round-trip, dead entry point
- **Category:** redundant · **Severity:** Low · **Location:** `mlx-gen-ideogram/src/pipeline.rs:51` (vs `config.rs:56`, `loader.rs:68`); `mlx-gen-ideogram/src/model.rs:392-402`; `mlx-gen-ideogram/src/pipeline.rs:302-316`
- **Finding:** the enforced token budget is a private copy of the exported constant (a third `2048` sits in the loader); `array_to_image` casts the already-Uint8 output to Int32 (4× host bytes) then maps back per element; `generate_from_prompt` has no callers and bypasses progress/cancel.
- **Impact:** budget drift risk; per-image waste; dead unexercised API.
- **Suggested fix:** consume `config::MAX_TEXT_TOKENS`; read `as_slice::<u8>()` directly; remove the entry point.
- **Confidence:** High

#### [F-111] flux/flux2: contiguous slices done as arange-index gathers in hot per-step paths
- **Category:** efficiency · **Severity:** Low · **Location:** `mlx-gen-flux2/src/transformer.rs:290-297, 431-438, 690-695` (+ kv_cache.rs:147-148, chunk.rs:143-146); `mlx-gen-flux/src/transformer.rs:797-803`
- **Finding:** the `[:, a:b]` idiom is `Array::from_slice((a..b).collect()) → take_axis` — ~112 gathers per flux2 forward per step (txt/img split ×2/double block, 4-way single-block split, final image slice); flux's `forward_with_ip` rebuilds the same index arrays per joint-attention call (the F-017 hoist covered the single-block loop but not this).
- **Impact:** materialized copies + graph bloat instead of MLX's cheap contiguous split path, repeated per layer per step.
- **Suggested fix:** `split`/`split_axis` at fixed offsets (as `swiglu`/`Modulation` already do); hoist flux's index arrays into `forward_inner`.
- **Confidence:** Medium

#### [F-112] Triple `Weights::from_dir` over the flux2 dev text_encoder
- **Category:** redundant · **Severity:** Low · **Location:** `mlx-gen-flux2/src/loader.rs:105-121, 216-228`; `mlx-gen-flux2/src/model.rs:181-229`
- **Finding:** the dev load path constructs `Weights::from_dir(root.join("text_encoder"))` three times (TE, vision tower, projector), each re-parsing every shard header of the ~45 GB component.
- **Impact:** 3× shard-header parses + tensor-map builds per load.
- **Suggested fix:** load once in `load_variant`, pass `&Weights` into the three constructors.
- **Confidence:** Medium

#### [F-113] LTX robustness/API nits: audio batch guard, vocoder unwrap, connector multiple, sample_every modulo
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-ltx/src/pipeline.rs:1109-1130`; `mlx-gen-ltx/src/vocoder.rs:561`; `mlx-gen-ltx/src/connector.rs:186-196`; `mlx-gen-ltx/src/training.rs:487`
- **Finding:** `decode_audio_track` drops the batch axis with no `B == 1` check (the sibling `to_uint8_frames` got the F-051 guard); the production vocoder forward unwraps `act_post` on a load-time invariant; `replace_with_registers` guards `s < num_reg` but not `s % num_reg != 0`; the preview modulo's zero-guard lives three hops away.
- **Impact:** opaque errors / latent panics on paths the crate elsewhere hardened.
- **Suggested fix:** add the F-051-style guard; `ok_or_else`; extend the connector guard; make the modulo condition self-contained.
- **Confidence:** High

#### [F-114] PuLID resolves sub-model weights via env vars instead of the LoadSpec
- **Category:** bad-pattern · **Severity:** Low · **Location:** `mlx-gen-pulid/src/pulid_flux.rs:296-332, 347-372`
- **Finding:** the registered loader resolves three weight sets via `PULID_FLUX_WEIGHTS`/`PULID_EVA_WEIGHTS`/`PULID_FACE_WEIGHTS_DIR` (with a hardcoded HF-cache scan fallback), outside the `LoadSpec` contract every other provider uses.
- **Impact:** hermetic/multi-tenant deployments must mutate process env; load behavior depends on ambient machine state.
- **Suggested fix:** carry the aux paths on `LoadSpec` (env vars as documented fallback).
- **Confidence:** Medium

#### [F-115] 18 member crates bypass the workspace serde_json pin; 18 duplicate the `image` dev-dep
- **Category:** redundant · **Severity:** Low · **Location:** e.g. `mlx-gen-z-image/Cargo.toml:28` (+17 siblings); `mlx-gen-flux/Cargo.toml:31` (+17 siblings)
- **Finding:** `serde_json = "1"` is declared per-crate despite the workspace entry (only gen-core opts in); the `image` dev-dep is repeated 18× with feature lists already drifting between `["png"]` and `["png","jpeg"]`.
- **Impact:** contradicts the documented single-source-of-truth convention 36 times; version/feature drift is live.
- **Suggested fix:** `{ workspace = true }` everywhere; add `image` to `[workspace.dependencies]`.
- **Confidence:** High

#### [F-116] Docs: wrong mlx-rs org link, CI section contradicts the workflow, MODEL_ARCHITECTURE still a "proposal", stale handoff doc
- **Category:** readability · **Severity:** Low · **Location:** `ARCHITECTURE.md:4, 75-78`; `docs/MODEL_ARCHITECTURE.md:4, 53-58`; `docs/HANDOFF_EPIC3040.md:1-12`
- **Finding:** (a) `github.com/oxiglade/mlx-rs` is a nonexistent org (upstream is `oxideai`; the pin is `michaeltrefry`); (b) the CI section claims `macos-14` — a runner the code comments say cannot build MLX 0.31.2 — and omits the Linux contract lane; (c) the canonical contract doc still opens "Status: proposal for review" and §2's topology shows FLUX.1+FLUX.2 as one crate (5 providers vs 29 actual); (d) HANDOFF_EPIC3040 instructs opening a PR for work merged in #156/#157/#161, without the HISTORICAL banner its sibling got.
- **Impact:** 404/squattable link; inverted CI picture; the doc contradicts the granularity rule it teaches; a future agent could try to re-push merged commits.
- **Suggested fix:** fix the link; describe both lanes on macos-15; mark adopted + annotate §2; add the HISTORICAL banner.
- **Confidence:** High

#### [F-117] tools: metallib cache install is not atomic; bernini mask builder duplicated across dumpers
- **Category:** bad-pattern · **Severity:** Low · **Location:** `tools/refresh_pmetal_metallib.sh:110`; `tools/dump_bernini_process_golden.py:136-162` vs `tools/dump_bernini_template_golden.py:64-90`
- **Finding:** `cp -f` truncates-then-writes the cache metallib (the comment claims "atomically-enough") — a concurrently launching MLX process can read a partial library; the two bernini dumpers each define `build_custom_attention_mask`.
- **Impact:** a rare failure mode exactly while refreshing the thing meant to prevent it; golden-dump drift risk.
- **Suggested fix:** copy to a temp in the same dir + `mv -f` (atomic on APFS); hoist to `tools/_bernini_common.py`.
- **Confidence:** High

#### [F-118] Consolidate the per-test-file snapshot resolvers
- **Category:** redundant · **Severity:** Low · **Location:** `mlx-gen-ltx/tests/*.rs` (18 copies + `src/training.rs:858-865`); `mlx-gen-sdxl/tests/*.rs` (~20); `mlx-gen-kolors/tests/*.rs` (7); `mlx-gen-wan/tests/*` (`max_abs` ×6)
- **Finding:** the env-var + HF-cache snapshot resolver is copy-pasted ~45× workspace-wide with drifting env-var spellings and return types; wan additionally sextuplicates a `max_abs` diff helper.
- **Impact:** a snapshot-layout change needs dozens of edits; the lens F-107 stale path shows the drift is real.
- **Suggested fix:** one `tests/common/mod.rs` helper per crate (standard cargo idiom), or a testkit utility.
- **Confidence:** High

#### [F-119] Boogu converter docs claim group size 64 while the constant is 32
- **Category:** readability · **Severity:** Low · **Location:** `mlx-gen-boogu/src/convert.rs:266-268, 417-419` (vs `src/quant.rs:11-16`)
- **Finding:** `QUANT_GROUP_SIZE` is documented as "the codebase-wide default … 64" but `crate::quant::GROUP_SIZE` is deliberately **32** (3360 % 64 ≠ 0). Behavior is consistent; only the docs lie about a load-bearing pack/load constant.
- **Impact:** a maintainer trusting the doc could "restore" 64 and produce snapshots the DiT rejects.
- **Suggested fix:** correct both comments and reference the quant.rs rationale.
- **Confidence:** High

#### [F-120] Dead flux-family API: control-transformer wrapper and boogu's single-ref edit family
- **Category:** dead-code · **Severity:** Low · **Location:** `mlx-gen-flux/src/control_transformer.rs:216-241`; `mlx-gen-boogu/src/{transformer/rope.rs:53-70,112-116, transformer/mod.rs:131-146, tokenizer.rs:117-127, pipeline.rs:403-419}`
- **Finding:** `FluxControlTransformer::forward` ("the convenience entry E2's generator wires") has zero callers — the generator calls `forward_composed`; boogu's `build_edit`/`ref_image`/`forward_edit`/`encode_edit_with_image`/`generate_edit_with_progress` (~90 lines) have zero callers — production and tests use the `_multi` forms, and `build_edit`'s doc cites call sites that don't exist.
- **Impact:** unexercised wrappers that can silently drift from the multi path they claim to be the N=1 case of.
- **Suggested fix:** delete (or pin wrapper ≡ multi-with-one with a test).
- **Confidence:** High

#### [F-121] Dead wan surface: `GuideScale::cfg_disabled`, `WanScheduler::reset`
- **Category:** dead-code · **Severity:** Low · **Location:** `mlx-gen-wan/src/config.rs:42-49`; `mlx-gen-wan/src/scheduler.rs:52-53, 168-171, 280-283, 526-532`
- **Finding:** `cfg_disabled` has zero callers (all sites re-derive `guidance <= 1.0` inline on the resolved value — correctly); `reset` and its three impls have no callers across wan/bernini/scail2 (`set_timesteps` already reinitializes).
- **Impact:** dead trait/API surface maintained in four places.
- **Suggested fix:** delete both.
- **Confidence:** High

#### [F-122] Dead kv-cache accessor and orphaned doc in flux2 convert
- **Category:** dead-code · **Severity:** Low · **Location:** `mlx-gen-flux2/src/kv_cache.rs:78-80`; `mlx-gen-flux2/src/convert.rs:482-492`
- **Finding:** `Flux2KvCache::mode()` has zero callers; `quantize_flux2_transformer` carries a concatenated orphan doc describing the (now core-hosted) `quantize_map`.
- **Impact:** unused API; rustdoc renders a confusing double description.
- **Suggested fix:** remove the accessor; delete the orphaned paragraph.
- **Confidence:** High

#### [F-123] Chroma stale doc references to the removed `ChromaSamplerKind` and the Base shift "placeholder"
- **Category:** readability · **Severity:** Low · **Location:** `mlx-gen-chroma/src/model.rs:188, 381`; `mlx-gen-chroma/src/config.rs:74-78`
- **Finding:** two docs link a type that no longer exists (selection is `resolve_sampler_name`); `sigma_shift`'s comment still calls Base's `1.0` "a placeholder … handled in sc-3840" though the beta schedule shipped and 1.0 is the real value.
- **Impact:** broken intra-doc links; misleading provenance on the sampler path.
- **Suggested fix:** repoint and reword.
- **Confidence:** High

#### [F-124] Kolors training doc claims the inference registry rejects adapters (it hasn't since sc-4733)
- **Category:** readability · **Severity:** Low · **Location:** `mlx-gen-kolors/src/training.rs:47-50`
- **Finding:** the module doc says inference "today rejects `spec.adapters`", but the registry has applied them since sc-4733 and the descriptor advertises LoRA/LoKr.
- **Impact:** invites re-filing or re-implementing shipped work.
- **Suggested fix:** replace with a pointer to `Kolors::apply_lora`.
- **Confidence:** High

#### [F-125] Core dead code: `LokrFile::delta_scale`, unwired optimizer-state resume
- **Category:** dead-code · **Severity:** Low · **Location:** `src/adapters/loader.rs:74-78`; `src/train/checkpoint.rs:20-31`
- **Finding:** `delta_scale` has zero workspace callers; `save_optimizer_state`/`load_optimizer_state` have zero callers, the module doc promises a resume capability (sc-3043) no trainer wires, and the helpers couldn't snapshot Prodigy/Rose state anyway.
- **Impact:** dead API plus a documented-but-nonexistent capability; if resume is real scope it is silently unimplemented.
- **Suggested fix:** delete + correct the doc, or wire resume (incl. `TrainOptimizer`-level snapshot) and file the story.
- **Confidence:** High

## Informational

#### [F-126] `lcm_style_timesteps` underflows when original_steps > num_train_timesteps
- **Category:** bad-pattern · **Severity:** Info · **Location:** `gen-core/src/sampling.rs:161-185`
- **Finding:** `k = num_train_timesteps / original_steps` truncates to 0 → timesteps of `-1` → `usize::MAX` index downstream. Engine callers hardcode safe values today.
- **Impact:** pub-fn-only panic. **Fix:** clamp `original_steps` (or `k.max(1)`). **Confidence:** High

#### [F-127] `Error::backend`/`From<String>` can silently bury the Canceled variant
- **Category:** bad-pattern · **Severity:** Info · **Location:** `gen-core/src/error.rs:41-59`
- **Finding:** boxing/formatting helpers accept `gen_core::Error::Canceled` itself, hiding cancellation inside `Backend`/`Msg` with no warning; only the (opt-in) testkit would catch it.
- **Impact:** latent contract-loss path. **Fix:** a `debug_assert!(!e.is::<Error>())` tripwire in `Error::backend`. **Confidence:** Medium

#### [F-128] `DiscreteModelSampling::timestep` linear-scans a sorted table
- **Category:** efficiency · **Severity:** Info · **Location:** `gen-core/src/sampling/model_sampling.rs:307-320`
- **Finding:** full argmin over 1000 monotone entries per lookup; some scheduler builds are O(N·M).
- **Impact:** negligible host math, but the candle port will copy it. **Fix:** `partition_point`. **Confidence:** High

#### [F-129] `lambda` helper duplicated between solvers.rs and cfgpp.rs
- **Category:** redundant · **Severity:** Info · **Location:** `gen-core/src/sampling/cfgpp.rs:52-57` (vs `solvers.rs:22-25`)
- **Finding:** a self-described "local copy" instead of `pub(super)`; the 1e-12 floor now lives twice.
- **Impact:** ε-drift between CFG++ and base solvers. **Fix:** share it. **Confidence:** High

#### [F-130] Testkit doc references a text-LLM conformance surface that no longer exists
- **Category:** readability · **Severity:** Info · **Location:** `gen-core-testkit/src/lib.rs:53-56`
- **Finding:** `check_progress_steps` doc cites "the captioner and text-LLM conformance checks"; the TextLlm trait left in sc-7189.
- **Impact:** doc points at removed surface. **Fix:** name the captioner. **Confidence:** High

#### [F-131] Curated-driver progress derivation is O(steps²)
- **Category:** efficiency · **Severity:** Info · **Location:** `src/sampler.rs:319, 374, 515`
- **Finding:** a full-schedule filter-count per denoise eval, triplicated (see F-068).
- **Impact:** negligible at n ≤ 50. **Fix:** carry a counter in the extracted helper. **Confidence:** High

#### [F-132] Wan doc/readability nits
- **Category:** readability · **Severity:** Info · **Location:** `mlx-gen-wan/src/vae22.rs:651-652`; `mlx-gen-wan/src/model.rs:736-737`
- **Finding:** `DownResBlock::forward`'s doc describes a tuple return the signature no longer has; `gen_frames` re-derives `trim·vae_stride` beside the `trim_out` binding the 5B path reuses.
- **Impact:** minor confusion. **Fix:** update the doc; reuse `trim_out`. **Confidence:** High

#### [F-133] LTX micro-helper duplication (`scalar` ×7, `contiguous` ×2, `range_idx` ×2, `f32` loader ×3)
- **Category:** redundant · **Severity:** Info · **Location:** `mlx-gen-ltx/src/{pipeline,transformer,conditioning,vae,audio_vae,vocoder,upsampler}.rs`
- **Finding:** pure duplication of two-line helpers within one crate.
- **Impact:** trivial drift risk. **Fix:** one crate-private util module. **Confidence:** High

#### [F-134] Qwen/krea dead surface: inert `control_in_dim`, unreachable `n == 1` linspace arm, unused sampler constructors
- **Category:** dead-code · **Severity:** Info · **Location:** `mlx-gen-qwen-image/src/control_transformer.rs:44-56`; `mlx-gen-qwen-image/src/pipeline.rs:321-331`; `mlx-gen-krea/src/schedule.rs` (+ `lib.rs:53`)
- **Finding:** the Fun-control config field gates nothing; the F-004 clamp makes the 1-step branch unreachable; `turbo_sampler`/`dynamic_sampler` are exported but unused by production.
- **Impact:** looks load-bearing, isn't. **Fix:** validate against the field or drop it; remove the arm; drop or story-reference the constructors. **Confidence:** High

#### [F-135] z-image `CONTROL_IN_DIM` re-exported but unused internally
- **Category:** dead-code · **Severity:** Info · **Location:** `mlx-gen-z-image/src/control_transformer.rs:50`; `lib.rs:42`
- **Finding:** the 33-channel construction never references the constant it advertises.
- **Impact:** dead unless the worker consumes it. **Fix:** derive/assert the concat count from it, or remove. **Confidence:** Medium

#### [F-136] SD3/sensenova test-oracle twins and hand-formatted JSON marker
- **Category:** dead-code · **Severity:** Info · **Location:** `mlx-gen-sensenova/src/qwen3.rs:392-441, 637-709`; `mlx-gen-sensenova/src/convert.rs:207-224`
- **Finding:** the uncached `forward_path`/`attention` twins exist only for parity tests but nothing marks them test-support-only; `write_merge_marker` hand-formats JSON with an unescaped filename (serde_json is already a dep).
- **Impact:** drift risk; invalid JSON on exotic filenames (provenance-only). **Fix:** `#[doc(hidden)]`/document; `serde_json::json!`. **Confidence:** Medium

#### [F-137] SD3 transformer readability/efficiency nits
- **Category:** readability · **Severity:** Info · **Location:** `mlx-gen-sd3/src/transformer.rs:135-160, 294-307`
- **Finding:** `let s = q.shape()[1]` binds the *head count* (reshape is correct only because `s` is secretly `H`); the joint-attention output split uses host-built index gathers where `split_axis` suffices.
- **Impact:** a future edit reading `s` as seq length produces a wrong-but-type-checking reshape; minor per-block waste ×38 blocks ×2 CFG ×28 steps. **Fix:** rename; `split_axis`. **Confidence:** High

#### [F-138] Bernini dead symbols and silently-ignored sidecar knobs
- **Category:** dead-code · **Severity:** Info · **Location:** `mlx-gen-bernini/src/forward.rs:53-56`; `vision.rs:522-535`; `clip_diff.rs:392-394`; `config.rs:16-19`
- **Finding:** `Mode::is_apg`, `split_vit_features`, `DiffLossFm::in_channels` are uncalled; `use_src_id_rotary_emb`/`max_sequence_length` are parsed from the sidecar and never consulted (source-id RoPE applies unconditionally).
- **Impact:** the ignored knob advertises configurability the runtime doesn't honor. **Fix:** delete; wire or drop the knobs with a comment. **Confidence:** High

#### [F-139] scail2 library-level `eprintln!` diagnostics
- **Category:** readability · **Severity:** Info · **Location:** `mlx-gen-scail2/src/generate.rs:113-122`; `mlx-gen-scail2/src/lora.rs:261-286`
- **Finding:** the alignment notice and diff-patch skip report write to stderr from engine code (acknowledged in-code: no load-time Progress channel exists yet).
- **Impact:** invisible to API consumers. **Fix:** route through a diagnostics channel when gen-core grows one. **Confidence:** High

#### [F-140] seedvr2 output-AdaLN unwraps inconsistent with the crate's F-075 discipline
- **Category:** readability · **Severity:** Info · **Location:** `mlx-gen-seedvr2/src/dit.rs:1112-1118` (also `blocks[0]` at 1082)
- **Finding:** `vid_out_norm.as_ref().unwrap()` etc. ride a load-time invariant in the same file that converted the equivalent block-field invariants to typed errors.
- **Impact:** a future variant toggling `use_output_ada` panics instead of erroring. **Fix:** mirror the `ada_v()` pattern. **Confidence:** High

#### [F-141] SAM2 RoPE `repeat` counts excluded object-pointer tokens
- **Category:** bad-pattern · **Severity:** Info · **Location:** `mlx-gen-sam2/src/memory.rs:414-418`
- **Finding:** `repeat = k_len / q_len` includes the excluded trailing tokens; identical results today only because truncating division absorbs ≤60 tokens against q_len 4096 — an implicit invariant with no guard (contrast sam3's F-016 check).
- **Impact:** latent opaque reshape if geometry changes. **Fix:** compute from `k_len - num_k_exclude_rope`. **Confidence:** High

#### [F-142] SAM3 cond object-pointer growth is unbounded compute (documented follow-up)
- **Category:** efficiency · **Severity:** Info · **Location:** `mlx-gen-sam3/src/video.rs:394-415`
- **Finding:** every cond frame's pointer with `t <= frame_idx` is appended each gather (cond entries accrue every 16 frames per object); heavy tensors are bounded, compute is not. The code defers this as "the F-024 sibling".
- **Impact:** per-frame latency creeps on very long clips. **Fix:** track the deferred cap in its story; verify against the reference budget. **Confidence:** High

#### [F-143] Lens hand-rolled biased linears skip fused addmm; reasoner prefill rebuilds masks 24×
- **Category:** efficiency · **Severity:** Info · **Location:** `mlx-gen-lens/src/text_encoder/gpt_oss.rs:57-60`; `dit/mod.rs:50-56`; `reasoner.rs:88-108`
- **Finding:** `add(matmul(x, wᵀ), b)` instead of the sc-2779 fused helper; the prefill builds a fresh L×L host mask per layer though only two variants exist.
- **Impact:** free wins, negligible next to F-021. **Fix:** fused addmm; hoist the two masks. **Confidence:** Medium

#### [F-144] SenseNova logits readbacks are dtype-safe only via accidental f32 promotion
- **Category:** bad-pattern · **Severity:** Info · **Location:** `mlx-gen-sensenova/src/runtime.rs:115`; `t2i.rs:435, 824, 949, 1117`
- **Finding:** five `as_slice::<f32>()` sites on a bf16-native backbone stay f32 only because the RoPE tables are built at f32 (promoting hidden states in layer 1). A plausible bf16-purity change makes all five process-aborting panics.
- **Impact:** hardening note. **Fix:** `as_dtype(Float32)` before each read, matching every other crate. **Confidence:** High

#### [F-145] PiD σ-capture capability differs silently across families (tracked as sc-7993)
- **Category:** bad-pattern · **Severity:** Info · **Location:** `mlx-gen-qwen-image/src/model.rs:239-249` (σ-capture) vs `mlx-gen-flux/src/model.rs:388-393`, `mlx-gen-sdxl/src/model.rs:501-510` (clean-σ only)
- **Finding:** only qwen/krea wire `resolve_pid_decoder_at_sigma`; flux/sdxl use the σ=0 resolver so capture-style requests are a no-op there. sdxl's deferral is commented and tracked.
- **Impact:** per-family behavioral difference behind one knob; fine while the story is open. **Fix:** verify sc-7993's scope covers flux too. **Confidence:** High

#### [F-146] Boogu accepts whitespace-only prompts
- **Category:** readability · **Severity:** Info · **Location:** `mlx-gen-boogu/src/model.rs:350-352`
- **Finding:** `req.prompt.is_empty()` vs flux/chroma's `trim().is_empty()`.
- **Impact:** `"   "` renders unconditioned instead of erroring. **Fix:** trim. **Confidence:** High

#### [F-147] Root Cargo.toml header counts "16 member crates" (there are 31)
- **Category:** readability · **Severity:** Info · **Location:** `Cargo.toml:5-7`
- **Finding:** comment drift in an otherwise careful header.
- **Impact:** trivial. **Fix:** say "all member crates". **Confidence:** High

#### [F-148] convert_pid.py hardcodes a `/Users/michael` venv path in its docstring
- **Category:** bad-pattern · **Severity:** Info · **Location:** `tools/convert_pid.py:19`
- **Finding:** the sole `/Users/<name>` leak in tools/, against the `_paths.py` portability convention.
- **Impact:** doc-only. **Fix:** reword relative to repo root. **Confidence:** High

#### [F-149] Grad-closure error stringification would swallow a future in-closure cancel
- **Category:** bad-pattern · **Severity:** Info · **Location:** trainer checkpoint/loss closures, e.g. `mlx-gen-z-image/src/training.rs:811-818` (pattern in krea/lens/ltx/sd3/sdxl/wan)
- **Finding:** closures bridge errors via `Exception::custom(e.to_string())` — safe today because cancel is never checked inside them; an in-closure cancel would be stringified and lose its type.
- **Impact:** invariant worth a comment, not a change. **Fix:** note the no-cancel-inside invariant at the sites. **Confidence:** High

#### [F-150] `Ideogram4Pipeline::generate_from_prompt` is dead
- **Category:** dead-code · **Severity:** Info · **Location:** `mlx-gen-ideogram/src/pipeline.rs:302-316`
- **Finding:** zero callers; bypasses progress/cancel.
- **Impact:** unexercised surface. **Fix:** remove or route a smoke through it. **Confidence:** Medium

## Themes and systemic observations

**T1 — New code re-introduces solved bug classes; the fixes don't travel.** The dominant pattern this cycle: a defect fixed in one crate recurs in a sibling or a new crate because the fix lives at a call site, not the shared seam. Concrete chains: the F-020 buffer guard exists on SDXL's init/control preprocessor but not its mask path (F-001) or Ideogram's (F-002) or depth's (F-044); the F-069 grad-accum fix reached z-image/lens but not ltx/wan (F-017); the F-024 eviction reached sam3 but not sam2 (F-041); the F-002 rank-0 guard reached LoRA but not third-party LoKr/LoHa (F-010); the i32-product fix reached four of five `frames_to_images` siblings (F-070); qwen's RoPE step-cache and split-not-gather fixes (F-114/115) didn't reach krea (F-079). When a review fix lands, grep for the pattern's siblings and fix the class, not the instance — or better, hoist the guard into the shared helper so there is only one instance.

**T2 — Hand-rolled per-provider validation drifts below the shared floor.** SDXL (F-022), z-image (F-030), flux-control (F-025), PuLID (F-026), and the flux2 edit variants (F-088) each re-implement `validate_request` and each dropped a different check — while Kolors, remediated last cycle to delegate to `gen_core::Capabilities::validate_request`, is correct. The floor itself is also missing pieces its own comments claim (steps ≥ 1, F-007) and returns the wrong error type (F-008). One fix direction covers a dozen findings: make the shared floor complete and typed, and make providers delegate to it plus family-specific extras.

**T3 — The cancellation contract now holds at denoise loops but not at the other expensive stages.** All ~35 registered entry points check cancel per denoise step (the 06-20 remediation held). Every remaining gap is a non-denoise stage: PiD decode (F-006), wan VAE decode (F-014), lens MoE encode (F-019), ltx prompt-enhance (F-018), sensenova interleave/vqa (F-037), training preview renders (F-077). The contract should be stated as "no unbounded uncancellable stage", not "the denoise loop checks the flag".

**T4 — Conformance enforcement lags the contract.** The testkit exists precisely to catch T3-class regressions but covers 2 of ~35 ids for full conformance (F-009), and the one live conformance violation (bernini progress, F-038) sits exactly in the uncovered set. The boogu dangling dev-dep (F-102) suggests coverage was intended and dropped.

**T5 — Memory-bounded decode is still deployed per-crate, not per-class.** Wan z48/z16 and seedvr2 are budgeted; PiD (F-013) and SANA's DC-AE (F-032) — both new — are not. Same story as T1: the budgeting pattern exists, new decoders don't inherit it.

**T6 — The Group-B/boilerplate wave created a new duplication ring.** Seven near-identical converter glue sets with three verified behavioral drifts (F-045), five compile_glue clones (F-104), duplicated TE leaves (F-078), VAE remaps (F-086), and ~45 test snapshot resolvers (F-118). Epic 7778 (boilerplate reduction) is the right vehicle; this review supplies its worklist.

**T7 — Docs and comments are drifting faster than before.** Load-bearing operational docs (CLAUDE.md crate list and metallib order, F-050; README's removed API, F-051; MODEL_ARCHITECTURE's "proposal" status and seedvr2 contract, F-103/F-116) and in-code rationale comments that assert false mechanisms (sc-8958's impossible `[1,0]` trap, F-031; the F-037 "validated upstream" claims, F-007; boogu's group-size 64, F-119) now mislead the agents and maintainers who treat them as ground truth. Comments that cite a guarantee should be treated as claims to verify at review time.

**T8 — Supply-chain hygiene is one notch below the code's discipline.** The workspace is rigorous about the mlx-rs pin yet ships a branch-pinned core-llm (F-048), tag-pinned CI actions with default token permissions (F-046/047), and a 4×-duplicated mlx-llm pin (F-049). All are one-line fixes.

**Positive observations worth recording:** both prior review waves' remediations survived re-verification (every 06-20 High, the panic-class fixes, compile-glue RAII, sam3 F-023/F-024); the panic-class sweep found zero reachable NaN-sort/FFI-closure/dtype aborts in product code; the error seam preserves `Canceled` 1:1 through all 36 registration seams; cancellation-per-denoise-step is uniform; and the hard reuse (SDXL UNet across kolors/instantid, wan across bernini/scail2, PiD's shared engine, sana reusing PiD's CHI encoder by import) remains exemplary.

## Coverage notes

- **Read in full:** every non-test `.rs` file in all 31 member crates, root `src/`, `gen-core`, `gen-core-testkit` (16 reviewers, file-complete per their coverage notes); root and per-crate `Cargo.toml`s, `.cargo/config.toml`, `rust-toolchain.toml`, CI workflows, `README.md`, `ARCHITECTURE.md`, `CLAUDE.md`, `docs/*`, and `tools/` (206 scripts scanned for secrets/paths/duplication; key scripts read in full).
- **Quick-scanned only:** `tests/` directories (~230 files) — structure, `#[ignore]` gating, helper duplication; test bodies were not line-audited except where a finding required it.
- **Excluded:** `_vendor/` (read-only third-party reference checkouts), `tools/golden/` (gitignored regenerable data), `Cargo.lock` (pins verified, not audited), fixture `.safetensors`.
- **Sweep exhaustiveness caveats:** the dtype-abort sweep risk-ranked ~70 of 222 `as_slice` product sites (all model-output readbacks traced; long-tail host-math sites sampled); duplication findings name the copies found, not a guarantee of no others.
- **Verification:** all six High findings were re-verified against source by the coordinating reviewer; prior-review remediations (06-13/06-20) were explicitly re-checked and their status recorded above. Findings marked Medium/Low confidence should be confirmed before acting (notably F-015 trim semantics vs reference, F-034 downstream flag consumers, F-057 reference tiling thresholds, F-093 reference CFG gate, F-096a upstream segment behavior).
