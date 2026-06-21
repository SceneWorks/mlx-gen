//! SAM3 multi-object video PCS pipeline (`Sam3VideoModel`) — epic 4910, sc-4924, Phase F2.5/F2.6.
//!
//! Pure-host orchestration over the (parity-green) tracker neural primitives in [`crate::tracker`]:
//! per frame, detect concept instances ([`crate::Sam3ImageSegmenter`]), propagate existing identities
//! through the per-object memory bank ([`Sam3Tracker::decode_tracked_frame`]), associate detections to
//! tracklets, seed new identities from unmatched detections
//! ([`Sam3Tracker::decode_mask_conditioning_frame`]), and encode each frame's masks into memory.
//!
//! Mirrors `transformers` `sam3_video/modeling_sam3_video.py` `_det_track_one_frame`. The reference's
//! optional `kernels` post-processing is ported natively (sc-4995), since the `cv-utils` kernel is
//! GPU-only and unavailable on this Mac: detection mask-NMS dedup (`det_nms_thresh`) replicates
//! `generic_nms` with a host greedy mask-IoU pass ([`nms_dedup`]), and hole-fill + sprinkle removal
//! (`fill_holes_in_mask_scores`) replicates the 8-connected `cc_torch`/skimage path with a host flood
//! fill ([`fill_holes_in_mask`]). Masks flow as 288² logits (the processor sigmoids for display).

use std::collections::BTreeMap;
use std::rc::Rc;

use mlx_rs::ops::sigmoid;
use mlx_rs::{Array, Dtype};

use mlx_gen::{CancelFlag, Error, Result};

use crate::config::Sam3VisionConfig;
use crate::tracker::TrackerFrameOutput;
use crate::vision::Backbone;
use crate::{Sam3ImageSegmenter, Sam3Tracker};

// --- config (Sam3VideoConfig defaults) -----------------------------------------------------------
const LOW_RES: i32 = 288; // low_res_mask_size
const SCORE_THRESH_DET: f32 = 0.5; // score_threshold_detection
const DET_NMS_THRESH: f32 = 0.1; // det_nms_thresh (mask-IoU NMS dedup; sc-4995)
const FILL_HOLE_AREA: i32 = 16; // fill_hole_area (hole-fill + sprinkle removal; sc-4995)
const NEW_DET_THRESH: f32 = 0.7;
const ASSOC_IOU_THRESH: f32 = 0.1;
const TRK_ASSOC_IOU_THRESH: f32 = 0.5;
const HIGH_CONF_THRESH: f32 = 0.8;
const HIGH_IOU_THRESH: f32 = 0.8;
const NUM_MASKMEM: i32 = 7;
const MAX_COND_FRAME_NUM: i32 = 4;
const MAX_OBJ_PTRS: i32 = 16; // max_object_pointers_in_encoder
const RECONDITION_EVERY: i32 = 16;
const INIT_KEEP_ALIVE: i32 = 30;
const MAX_KEEP_ALIVE: i32 = 30;
const MIN_KEEP_ALIVE: i32 = -1;
const HOTSTART_DELAY: i32 = 15;
const HOTSTART_UNMATCH: usize = 8;
const HOTSTART_DUP: usize = 8;
const SUPPRESS_OCC_THRESH: f32 = 0.7; // suppress_overlapping_based_on_recent_occlusion_threshold
const NEVER_OCCLUDED: i32 = -1;
const ALWAYS_OCCLUDED: i32 = 100_000;
const NO_OBJ_LOGIT: f32 = -10.0;

/// Gathered spatial memory: `(relative_temporal_offset, maskmem_features, maskmem_pos_enc)` per frame.
type SpatialMem = Vec<(i32, Array, Array)>;
/// Gathered object pointers: `(temporal_offset, pointer [1,256])`.
type ObjPointers = Vec<(i32, Array)>;

/// A detection on a frame: raw 288² mask logits + score + box, plus the prompt that produced it.
struct Detection {
    mask: Vec<f32>, // [288·288] logits
    score: f32,
    prompt_id: i32,
}

/// One stored per-frame output for an object (the memory-bank entry).
#[derive(Clone)]
struct FrameMem {
    maskmem_features: Option<Array>, // [5184,1,64] seq-first (bf16-cast); None until memory-encoded
    maskmem_pos_enc: Option<Array>,  // [5184,1,64]
    object_pointer: Array,           // [1,256]
    object_score: f32,
}

/// Per-object memory bank: conditioning-frame outputs (user/detection-seeded) + tracked-frame outputs.
#[derive(Default, Clone)]
struct ObjectBank {
    cond: BTreeMap<i32, FrameMem>,
    non_cond: BTreeMap<i32, FrameMem>,
}

/// The per-frame segmentation result: object id → 288² mask logits, in id order.
pub struct VideoFrameOutput {
    pub obj_ids: Vec<i32>,
    pub masks: Vec<Vec<f32>>, // each [288·288] logits, parallel to obj_ids
}

/// `Sam3VideoModel`: the detector + the tracker, driving the multi-object PCS pipeline.
pub struct Sam3VideoModel {
    segmenter: Sam3ImageSegmenter,
    tracker: Sam3Tracker,
    // --- per-session state ---
    obj_ids: Vec<i32>,      // ordered; index = obj_idx
    banks: Vec<ObjectBank>, // parallel to obj_ids
    obj_prompt: Vec<i32>,   // prompt id per obj_idx
    max_obj_id: i32,
    num_frames: i32,
    // hotstart metadata (keyed by obj_id)
    first_frame: BTreeMap<i32, i32>,
    unmatched_frames: BTreeMap<i32, Vec<i32>>,
    keep_alive: BTreeMap<i32, i32>,
    overlap_pairs: BTreeMap<(i32, i32), Vec<i32>>,
    removed: std::collections::BTreeSet<i32>,
    last_occluded: BTreeMap<i32, i32>,
}

impl Sam3VideoModel {
    pub fn from_weights(w: &mlx_gen::weights::Weights) -> Result<Self> {
        // One PE backbone, shared between the detector segmenter and the tracker. Both load it from
        // the same `detector_model.vision_encoder.backbone` keys, so loading it twice would carry two
        // identical ~445M-param copies resident at video time (F-028).
        let cfg = Sam3VisionConfig::sam3();
        let backbone = Rc::new(Backbone::from_weights(
            w,
            "detector_model.vision_encoder.backbone",
            &cfg,
        )?);
        Ok(Self {
            segmenter: Sam3ImageSegmenter::from_weights_with_backbone(w, backbone.clone())?,
            tracker: Sam3Tracker::from_weights_with_backbone(w, backbone)?,
            obj_ids: Vec::new(),
            banks: Vec::new(),
            obj_prompt: Vec::new(),
            max_obj_id: -1,
            num_frames: 0,
            first_frame: BTreeMap::new(),
            unmatched_frames: BTreeMap::new(),
            keep_alive: BTreeMap::new(),
            overlap_pairs: BTreeMap::new(),
            removed: std::collections::BTreeSet::new(),
            last_occluded: BTreeMap::new(),
        })
    }

    /// Affine-quantize the whole video model to `bits` (Q8/Q4): the single shared PE backbone plus
    /// the detector segmenter's and the tracker's own heads. Convs/norms/embeddings stay dense
    /// (sc-4925).
    ///
    /// The backbone is shared (one `Rc`) between the segmenter and the tracker (F-028), so it is
    /// quantized **once** and the same quantized `Rc` reinstalled into both — otherwise each side
    /// would quantize into a separate copy and re-duplicate the weights we just deduplicated.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        let mut backbone = (*self.tracker.backbone_rc()).clone();
        backbone.quantize(bits)?;
        let backbone = Rc::new(backbone);
        self.segmenter.set_vision_backbone(backbone.clone());
        self.tracker.set_backbone(backbone);
        self.segmenter.quantize_except_backbone(bits)?;
        self.tracker.quantize_except_backbone(bits)?;
        Ok(())
    }

    /// Process a whole video (forward, non-streaming): `frames[f]` = NCHW `[1,3,1008,1008]`; one text
    /// prompt (`input_ids[1,32]` + `text_mask`). Returns per-frame `obj_id → 288² mask logits`.
    pub fn propagate(
        &mut self,
        frames: &[Array],
        input_ids: &Array,
        text_mask: &[i32],
        cancel: Option<&CancelFlag>,
        mut progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> Result<Vec<VideoFrameOutput>> {
        self.num_frames = frames.len() as i32;
        let total = frames.len();
        let mut outputs = Vec::new();
        for (f, px) in frames.iter().enumerate() {
            // Honor the engine cancellation contract — check before each (seconds-to-minutes) frame.
            if let Some(c) = cancel {
                if c.is_cancelled() {
                    return Err(Error::Canceled);
                }
            }
            outputs.push(self.process_frame(f as i32, px, input_ids, text_mask)?);
            if let Some(cb) = progress.as_deref_mut() {
                cb(f, total);
            }
        }
        Ok(outputs)
    }

    fn process_frame(
        &mut self,
        frame_idx: i32,
        pixels: &Array,
        input_ids: &Array,
        text_mask: &[i32],
    ) -> Result<VideoFrameOutput> {
        // --- Step 1: vision + detection (one shared PE backbone pass feeds both necks, sc-5409) ---
        let features = self.segmenter.backbone_features(pixels)?; // [1,72,72,C], 32-layer ViT once
        let (img_emb, high_res) = self.tracker.encode_frame_from_features(&features)?; // [1,72,72,256], [s0,s1]
        let g = img_emb.shape()[1];
        let cvf = img_emb.reshape(&[g * g, 1, 256])?;
        let cvp = self.tracker.frame_position_encoding(g)?;
        let det = self.run_detection(&features, input_ids, text_mask)?;

        // --- Step 2: propagate existing identities (run_mem_encoder = false) ---
        let num_existing = self.obj_ids.len();
        let mut trk_masks: Vec<Vec<f32>> = Vec::with_capacity(num_existing); // [288²] logits per obj
        let mut trk_scores: Vec<f32> = Vec::with_capacity(num_existing);
        for obj_idx in 0..num_existing {
            let (spatial, pointers, max_optr) = self.gather_memory(obj_idx, frame_idx);
            let conditioned = self
                .tracker
                .prepare_memory_conditioned_features(&cvf, &cvp, &spatial, &pointers, max_optr)?;
            let out = self.tracker.decode_tracked_frame(&conditioned, &high_res)?;
            let mut low = to_vec(&out.low_res)?;
            // Hole-fill the propagated mask here (mirrors `run_tracker_propagation`, sc-4995) so the
            // filled logits flow into association, overlap-suppression, memory encoding, and output.
            fill_holes_in_mask(&mut low, FILL_HOLE_AREA);
            self.banks[obj_idx].non_cond.insert(
                frame_idx,
                FrameMem {
                    maskmem_features: None,
                    maskmem_pos_enc: None,
                    object_pointer: out.object_pointer.clone(),
                    object_score: out.object_score,
                },
            );
            trk_scores.push(out.object_score);
            trk_masks.push(low);
        }

        // --- Step 3: associate + new-object ids + hotstart ---
        let assoc = self.associate(&det, &trk_masks);
        let new_obj_ids: Vec<i32> = (0..assoc.new_det_inds.len() as i32)
            .map(|i| self.max_obj_id + 1 + i)
            .collect();
        for (&oid, &di) in new_obj_ids.iter().zip(&assoc.new_det_inds) {
            // prompt id assigned at creation (recorded when the object is added below)
            let _ = (oid, di);
        }
        let removed_now = self.process_hotstart(frame_idx, &assoc, &new_obj_ids);

        // recondition every Nth frame: confidently re-detected tracks become conditioning frames
        // (recondition_on_trk_masks = True → "validate" mode keeps the tracker mask).
        let mut reconditioned_obj_ids: Vec<i32> = Vec::new();
        if RECONDITION_EVERY > 0
            && frame_idx % RECONDITION_EVERY == 0
            && !assoc.trk_id_to_max_iou_high_conf_det.is_empty()
        {
            for &trk_oid in assoc.trk_id_to_max_iou_high_conf_det.keys() {
                if let Some(obj_idx) = self.obj_ids.iter().position(|&o| o == trk_oid) {
                    if trk_scores.get(obj_idx).copied().unwrap_or(f32::MIN) > HIGH_CONF_THRESH {
                        reconditioned_obj_ids.push(trk_oid);
                    }
                }
            }
        }

        // --- Step 4 (planning tail): suppress overlaps + encode memory for existing objects ---
        if num_existing > 0 {
            self.suppress_overlapping_recent_occlusion(frame_idx, &mut trk_masks, &removed_now);
            self.tracker_update_memories(frame_idx, &img_emb, &trk_masks)?;
            // move reconditioned frames from non_cond → cond so they seed future memory selection.
            for &oid in &reconditioned_obj_ids {
                if let Some(obj_idx) = self.obj_ids.iter().position(|&o| o == oid) {
                    if let Some(fm) = self.banks[obj_idx].non_cond.remove(&frame_idx) {
                        self.banks[obj_idx].cond.insert(frame_idx, fm);
                    }
                }
            }
        }

        // --- Step 5 (execution): add new objects from unmatched detections ---
        for (&oid, &di) in new_obj_ids.iter().zip(&assoc.new_det_inds) {
            self.add_object(oid, det.dets[di].prompt_id);
            let obj_idx = self.obj_ids.len() - 1;
            // binarize the detection logits at 0.5 (reference: det_mask >= 0.5) → mask prompt.
            let mask_bin: Vec<f32> = det.dets[di]
                .mask
                .iter()
                .map(|&v| if v >= 0.5 { 1.0 } else { 0.0 })
                .collect();
            let mask_nhwc = Array::from_slice(&mask_bin, &[1, LOW_RES, LOW_RES, 1]);
            let out: TrackerFrameOutput = self
                .tracker
                .decode_mask_conditioning_frame(&img_emb, &high_res, &mask_nhwc)?;
            let mem =
                self.tracker
                    .encode_new_memory(&img_emb, &out.high_res, out.object_score, true)?;
            self.banks[obj_idx].cond.insert(
                frame_idx,
                FrameMem {
                    maskmem_features: Some(seq_first(&mem.features, true)?),
                    maskmem_pos_enc: Some(seq_first(&mem.pos, false)?),
                    object_pointer: out.object_pointer,
                    object_score: out.object_score,
                },
            );
        }
        // remove objects flagged by hotstart
        for oid in &removed_now {
            self.remove_object(*oid);
        }

        // Bound the per-object memory bank: drop / null `non_cond` entries this frame's writes have
        // pushed out of every future `gather_memory` read window, so a long clip's resident memory
        // stops climbing ~2.65 MB · frames · objects without a ceiling (F-024). Runs after all per-frame
        // writes (the non_cond inserts/fills + the recondition non_cond→cond move) so it never races
        // the current frame's state.
        self.evict_stale_memory(frame_idx);

        // --- build outputs ---
        self.build_outputs(
            &det,
            &assoc,
            &new_obj_ids,
            &trk_masks,
            &reconditioned_obj_ids,
        )
    }

    // ----- detection (run_detection, single prompt) -----
    // Takes the shared backbone features (sc-5409) so the detector FPN reuses the same ViT pass as
    // the tracker neck instead of re-running the backbone. Above-threshold detections are then
    // de-duplicated by greedy mask-IoU NMS (`det_nms_thresh`, sc-4995).
    fn run_detection(
        &self,
        features: &Array,
        input_ids: &Array,
        text_mask: &[i32],
    ) -> Result<DetFrame> {
        let seg = self
            .segmenter
            .forward_from_backbone(features, input_ids, text_mask)?;
        let presence = sigmoid(&seg.presence_logits)?.item::<f32>();
        // Cast to f32 before the host readback (a bf16 detector head would panic in as_slice::<f32>),
        // mirroring the `masks` readback two lines below and tracker.rs's casts (F-023).
        let probs: Vec<f32> = sigmoid(&seg.pred_logits)?
            .as_dtype(Dtype::Float32)?
            .as_slice::<f32>()
            .iter()
            .map(|&s| s * presence)
            .collect();
        let q = probs.len();
        let masks = seg.pred_masks.reshape(&[q as i32, LOW_RES * LOW_RES])?;
        let masks_v = masks.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
        let mut dets = Vec::new();
        for (qi, &p) in probs.iter().enumerate() {
            if p <= SCORE_THRESH_DET {
                continue;
            }
            let m = masks_v
                [qi * (LOW_RES * LOW_RES) as usize..(qi + 1) * (LOW_RES * LOW_RES) as usize]
                .to_vec();
            dets.push(Detection {
                mask: m,
                score: p,
                prompt_id: 0,
            });
        }
        let dets = nms_dedup(dets, DET_NMS_THRESH);
        Ok(DetFrame { dets })
    }

    // ----- memory bank gather (F2.4 selection logic) -----
    fn gather_memory(&self, obj_idx: usize, frame_idx: i32) -> (SpatialMem, ObjPointers, i32) {
        let bank = &self.banks[obj_idx];
        // Spatial memory = two reference windows: up to `MAX_COND_FRAME_NUM` closest conditioning
        // frames (reference `max_cond_frames_in_attn`, offset 0) ++ the `NUM_MASKMEM-1` recent non-cond
        // frames at offsets `[NUM_MASKMEM-1 .. 1]` (reference `num_maskmem`, falling back to an
        // unselected cond frame at that offset). So the assembled spatial count is bounded by their sum.
        let (selected_cond, unselected_cond) =
            select_closest_cond_frames(frame_idx, &bank.cond, MAX_COND_FRAME_NUM);
        let mut spatial: Vec<(i32, Array, Array)> = Vec::new();
        for f in &selected_cond {
            if let Some(m) = bank.cond.get(f) {
                if let (Some(feat), Some(pos)) = (&m.maskmem_features, &m.maskmem_pos_enc) {
                    spatial.push((0, feat.clone(), pos.clone()));
                }
            }
        }
        for rel in (1..NUM_MASKMEM).rev() {
            let prev = frame_idx - rel;
            let out = bank.non_cond.get(&prev).or_else(|| {
                if unselected_cond.contains(&prev) {
                    bank.cond.get(&prev)
                } else {
                    None
                }
            });
            if let Some(m) = out {
                if let (Some(feat), Some(pos)) = (&m.maskmem_features, &m.maskmem_pos_enc) {
                    spatial.push((rel, feat.clone(), pos.clone()));
                }
            }
        }
        debug_assert!(
            spatial.len() <= (MAX_COND_FRAME_NUM + NUM_MASKMEM - 1) as usize,
            "spatial memory exceeds its two reference windows (cond {MAX_COND_FRAME_NUM} + non-cond \
             {})",
            NUM_MASKMEM - 1
        );
        // Object pointers: every eligible conditioning-frame pointer (t <= frame_idx) ++ the recent
        // non-cond pointers within the `max_optr = min(num_frames, MAX_OBJ_PTRS)` window (reference
        // `max_object_pointers_in_encoder`). The non-cond contribution is bounded by `max_optr-1`; the
        // cond contribution tracks the cond-map size (bounded by the recondition cadence, not a fixed
        // cap), so the total is intentionally not hard-asserted here — capping cond growth is the
        // separate `bank.cond` follow-up (the F-024 sibling).
        let max_optr = self.num_frames.min(MAX_OBJ_PTRS);
        let mut pointers: Vec<(i32, Array)> = Vec::new();
        for (&t, m) in &bank.cond {
            if t <= frame_idx {
                pointers.push((frame_idx - t, m.object_pointer.clone()));
            }
        }
        for t_diff in 1..max_optr {
            let r = frame_idx - t_diff;
            if r < 0 || r >= self.num_frames {
                break;
            }
            if let Some(m) = bank.non_cond.get(&r) {
                pointers.push((t_diff, m.object_pointer.clone()));
            }
        }
        (spatial, pointers, max_optr)
    }

    /// Evict `non_cond` bank entries that no future `gather_memory` can read, derived from the same
    /// `NUM_MASKMEM` / `MAX_OBJ_PTRS` windows `gather_memory` uses (single source of truth). After
    /// processing `frame_idx` the next gather is at frame ≥ `frame_idx + 1`; heavy tensors are read
    /// back to `(frame_idx+1) − (NUM_MASKMEM−1)` and object pointers back to
    /// `(frame_idx+1) − (MAX_OBJ_PTRS−1)`, and both windows only slide forward — so any entry older
    /// than that is dead for the rest of the session (F-024). `cond` **entries** are left intact (the
    /// pointer loop reads their lightweight `object_pointer` at arbitrary keys), but their heavy
    /// `maskmem_*` tensors are bounded by [`evict_stale_cond_heavy`] once they fall out of every future
    /// spatial-read window (sc-7060), so a long clip's resident cond memory also stops climbing.
    fn evict_stale_memory(&mut self, frame_idx: i32) {
        let heavy_keep = frame_idx + 1 - (NUM_MASKMEM - 1);
        let ptr_keep = frame_idx + 1 - (MAX_OBJ_PTRS - 1);
        for bank in &mut self.banks {
            evict_stale_bank(bank, heavy_keep, ptr_keep);
            evict_stale_cond_heavy(bank, heavy_keep);
        }
    }

    // ----- association (_associate_det_trk; mask-IoU, no Hungarian) -----
    #[allow(clippy::needless_range_loop)] // parallel-array indexing (iou / obj_ids / dets)
    fn associate(&self, det: &DetFrame, trk_masks: &[Vec<f32>]) -> Assoc {
        let n = det.dets.len();
        let m = trk_masks.len();
        let mut a = Assoc::default();
        if m == 0 {
            a.new_det_inds = (0..n).collect();
            return a;
        }
        let det_bin: Vec<Vec<bool>> = det.dets.iter().map(|d| binarize(&d.mask)).collect();
        let trk_bin: Vec<Vec<bool>> = trk_masks.iter().map(|t| binarize(t)).collect();
        let trk_nonempty: Vec<bool> = trk_bin.iter().map(|t| t.iter().any(|&x| x)).collect();
        // IoU[n][m], zeroed across prompt groups.
        let mut iou = vec![vec![0f32; m]; n];
        for (i, db) in det_bin.iter().enumerate() {
            for (j, tb) in trk_bin.iter().enumerate() {
                if det.dets[i].prompt_id == self.obj_prompt[j] {
                    iou[i][j] = mask_iou(db, tb);
                }
            }
        }
        // tracks: unmatched if non-empty and no det IoU >= trk_assoc; empty if zero-area.
        for j in 0..m {
            let matched = (0..n).any(|i| iou[i][j] >= TRK_ASSOC_IOU_THRESH);
            if !trk_nonempty[j] {
                a.empty_trk.push(self.obj_ids[j]);
            } else if !matched {
                a.unmatched_trk.push(self.obj_ids[j]);
            }
        }
        // detections: new if score >= new_det_thresh and no track IoU >= assoc_iou.
        for i in 0..n {
            let matches_any = (0..m).any(|j| iou[i][j] >= ASSOC_IOU_THRESH);
            if det.dets[i].score >= NEW_DET_THRESH && !matches_any {
                a.new_det_inds.push(i);
            }
            let matched: Vec<i32> = (0..m)
                .filter(|&j| iou[i][j] >= ASSOC_IOU_THRESH)
                .map(|j| self.obj_ids[j])
                .collect();
            // det → max-IoU track for high-conf/high-iou recondition candidates.
            let is_new = det.dets[i].score >= NEW_DET_THRESH && !matches_any;
            let (best_j, best_iou) = (0..m).fold((0usize, -1f32), |(bj, bi), j| {
                if iou[i][j] > bi {
                    (j, iou[i][j])
                } else {
                    (bj, bi)
                }
            });
            if det.dets[i].score >= HIGH_CONF_THRESH
                && !is_new
                && best_iou >= HIGH_IOU_THRESH
                && m > 0
            {
                a.trk_id_to_max_iou_high_conf_det
                    .insert(self.obj_ids[best_j], i);
            }
            a.det_to_matched_trk.push(matched);
        }
        a
    }

    // ----- hotstart (_process_hotstart) -----
    fn process_hotstart(&mut self, frame_idx: i32, a: &Assoc, new_obj_ids: &[i32]) -> Vec<i32> {
        let mut newly_removed = Vec::new();
        let hotstart_diff = frame_idx - HOTSTART_DELAY;
        // log first-appearance + init keep-alive for new objects.
        for &oid in new_obj_ids {
            self.first_frame.entry(oid).or_insert(frame_idx);
            self.keep_alive.insert(oid, INIT_KEEP_ALIVE);
        }
        // matched tracks bump keep-alive; unmatched decrement + log.
        let mut matched: std::collections::BTreeSet<i32> = std::collections::BTreeSet::new();
        for trks in &a.det_to_matched_trk {
            matched.extend(trks.iter().copied());
        }
        for &oid in &matched {
            let k = self
                .keep_alive
                .get(&oid)
                .copied()
                .unwrap_or(INIT_KEEP_ALIVE);
            self.keep_alive.insert(oid, MAX_KEEP_ALIVE.min(k + 1));
        }
        for &oid in &a.unmatched_trk {
            self.unmatched_frames
                .entry(oid)
                .or_default()
                .push(frame_idx);
            let k = self
                .keep_alive
                .get(&oid)
                .copied()
                .unwrap_or(INIT_KEEP_ALIVE);
            self.keep_alive.insert(oid, MIN_KEEP_ALIVE.max(k - 1));
        }
        // removal: unmatched for >= unmatch_thresh frames within hotstart.
        let unmatched_snapshot: Vec<(i32, usize, i32)> = self
            .unmatched_frames
            .iter()
            .map(|(&oid, fs)| (oid, fs.len(), *self.first_frame.get(&oid).unwrap_or(&0)))
            .collect();
        for (oid, count, first) in unmatched_snapshot {
            if self.removed.contains(&oid) || newly_removed.contains(&oid) {
                continue;
            }
            if count >= HOTSTART_UNMATCH && first > hotstart_diff {
                newly_removed.push(oid);
            }
        }
        // duplicate-overlap tracking + removal.
        for trks in &a.det_to_matched_trk {
            if trks.len() < 2 {
                continue;
            }
            let first_appear = *trks
                .iter()
                .min_by_key(|&&o| *self.first_frame.get(&o).unwrap_or(&0))
                .unwrap();
            for &oid in trks {
                if oid != first_appear {
                    self.overlap_pairs
                        .entry((first_appear, oid))
                        .or_default()
                        .push(frame_idx);
                }
            }
        }
        let overlap_snapshot: Vec<(i32, usize, i32)> = self
            .overlap_pairs
            .iter()
            .map(|(&(_f, oid), fs)| (oid, fs.len(), *self.first_frame.get(&oid).unwrap_or(&0)))
            .collect();
        for (oid, count, first) in overlap_snapshot {
            if self.removed.contains(&oid) || newly_removed.contains(&oid) {
                continue;
            }
            if first > hotstart_diff && count >= HOTSTART_DUP {
                newly_removed.push(oid);
            }
        }
        for &oid in &newly_removed {
            self.removed.insert(oid);
        }
        newly_removed
    }

    // ----- occlusion-based overlap suppression (_suppress_overlapping_based_on_recent_occlusion) -----
    #[allow(clippy::needless_range_loop)] // parallel-array indexing (masks / obj_ids / last_occ)
    fn suppress_overlapping_recent_occlusion(
        &mut self,
        frame_idx: i32,
        trk_masks: &mut [Vec<f32>],
        removed_now: &[i32],
    ) {
        let n = trk_masks.len();
        if n == 0 {
            return;
        }
        let bin: Vec<Vec<bool>> = trk_masks.iter().map(|t| binarize(t)).collect();
        // last-occluded per object (NEVER if unseen, ALWAYS if removed this frame).
        let last_occ: Vec<i32> = (0..n)
            .map(|j| {
                let oid = self.obj_ids[j];
                self.last_occluded
                    .get(&oid)
                    .copied()
                    .unwrap_or(if removed_now.contains(&oid) {
                        ALWAYS_OCCLUDED
                    } else {
                        NEVER_OCCLUDED
                    })
            })
            .collect();
        let mut to_suppress = vec![false; n];
        // within each prompt group, suppress the more-recently-occluded of an overlapping pair.
        for pg in unique(&self.obj_prompt[0..n]) {
            let idxs: Vec<usize> = (0..n).filter(|&j| self.obj_prompt[j] == pg).collect();
            if idxs.len() <= 1 {
                continue;
            }
            for ai in 0..idxs.len() {
                for bj in (ai + 1)..idxs.len() {
                    let (i, j) = (idxs[ai], idxs[bj]);
                    if mask_iou(&bin[i], &bin[j]) < SUPPRESS_OCC_THRESH {
                        continue;
                    }
                    // suppress i if it was occluded more recently (and j was previously occluded).
                    if last_occ[i] > last_occ[j] && last_occ[j] > NEVER_OCCLUDED {
                        to_suppress[i] = true;
                    }
                    if last_occ[j] > last_occ[i] && last_occ[i] > NEVER_OCCLUDED {
                        to_suppress[j] = true;
                    }
                }
            }
        }
        // update last-occluded for occluded-or-suppressed objects; zero out suppressed masks.
        for j in 0..n {
            let occluded = !bin[j].iter().any(|&x| x);
            let oid = self.obj_ids[j];
            let new_lo = if occluded || to_suppress[j] {
                frame_idx
            } else {
                last_occ[j]
            };
            self.last_occluded.insert(oid, new_lo);
            if to_suppress[j] {
                for v in trk_masks[j].iter_mut() {
                    *v = NO_OBJ_LOGIT;
                }
            }
        }
    }

    // ----- memory encode for existing objects (_tracker_update_memories) -----
    #[allow(clippy::needless_range_loop)] // index into parallel banks / constrained masks
    fn tracker_update_memories(
        &mut self,
        frame_idx: i32,
        img_emb: &Array,
        trk_masks: &[Vec<f32>],
    ) -> Result<()> {
        let n = trk_masks.len();
        if n == 0 {
            return Ok(());
        }
        // non-overlapping constraints (per prompt group): pixel-wise argmax keep + shrink suppression.
        let constrained = suppress_pw_area_shrinkage(trk_masks, &self.obj_prompt[0..n]);
        for obj_idx in 0..n {
            let mask = &constrained[obj_idx];
            let appearing = mask.iter().any(|&v| v > 0.0);
            let object_score = if appearing { 10.0 } else { -10.0 };
            // high-res mask for the encoder = the 288² logits (encode_new_memory resizes to 1152²).
            let mask_arr = Array::from_slice(mask, &[1, 1, LOW_RES, LOW_RES]);
            let mem = self
                .tracker
                .encode_new_memory(img_emb, &mask_arr, object_score, false)?;
            // store into whichever (cond/non_cond) holds this frame for the object.
            let feat = seq_first(&mem.features, true)?;
            let pos = seq_first(&mem.pos, false)?;
            if let Some(fm) = self.banks[obj_idx].cond.get_mut(&frame_idx) {
                fm.maskmem_features = Some(feat);
                fm.maskmem_pos_enc = Some(pos);
                fm.object_score = object_score;
            } else if let Some(fm) = self.banks[obj_idx].non_cond.get_mut(&frame_idx) {
                fm.maskmem_features = Some(feat);
                fm.maskmem_pos_enc = Some(pos);
                fm.object_score = object_score;
            }
        }
        Ok(())
    }

    // ----- build outputs -----
    #[allow(clippy::needless_range_loop)] // parallel-array indexing (trk_masks / obj_ids)
    fn build_outputs(
        &self,
        det: &DetFrame,
        a: &Assoc,
        new_obj_ids: &[i32],
        trk_masks: &[Vec<f32>],
        reconditioned_obj_ids: &[i32],
    ) -> Result<VideoFrameOutput> {
        let mut obj_ids = Vec::new();
        let mut masks = Vec::new();
        // existing identities → propagated tracker masks, except reconditioned ones use the detection.
        let num_existing = trk_masks.len();
        for j in 0..num_existing {
            let oid = self.obj_ids[j];
            obj_ids.push(oid);
            if reconditioned_obj_ids.contains(&oid) {
                if let Some(&di) = a.trk_id_to_max_iou_high_conf_det.get(&oid) {
                    masks.push(det.dets[di].mask.clone());
                    continue;
                }
            }
            masks.push(trk_masks[j].clone());
        }
        // new identities → detection logits with hole-fill applied to the output mask (sc-4995;
        // reference `build_outputs` Part 2). The raw detection mask is still what seeds the new
        // object's memory (step 5) and what overrides reconditioned objects (Part 3, above).
        for (&oid, &di) in new_obj_ids.iter().zip(&a.new_det_inds) {
            obj_ids.push(oid);
            let mut m = det.dets[di].mask.clone();
            fill_holes_in_mask(&mut m, FILL_HOLE_AREA);
            masks.push(m);
        }
        Ok(VideoFrameOutput { obj_ids, masks })
    }

    fn add_object(&mut self, obj_id: i32, prompt_id: i32) {
        self.obj_ids.push(obj_id);
        self.banks.push(ObjectBank::default());
        self.obj_prompt.push(prompt_id);
        self.max_obj_id = self.max_obj_id.max(obj_id);
    }

    fn remove_object(&mut self, obj_id: i32) {
        if let Some(idx) = self.obj_ids.iter().position(|&o| o == obj_id) {
            self.obj_ids.remove(idx);
            self.banks.remove(idx);
            self.obj_prompt.remove(idx);
        }
    }
}

// --- helpers -------------------------------------------------------------------------------------

struct DetFrame {
    dets: Vec<Detection>,
}

#[derive(Default)]
struct Assoc {
    new_det_inds: Vec<usize>,
    unmatched_trk: Vec<i32>,
    empty_trk: Vec<i32>,
    det_to_matched_trk: Vec<Vec<i32>>,
    trk_id_to_max_iou_high_conf_det: BTreeMap<i32, usize>,
}

/// `_select_closest_cond_frames`: ≤ `max` cond frames closest to `frame_idx`. Returns
/// (selected frame indices, unselected frame indices).
fn select_closest_cond_frames(
    frame_idx: i32,
    cond: &BTreeMap<i32, FrameMem>,
    max: i32,
) -> (Vec<i32>, std::collections::BTreeSet<i32>) {
    let keys: Vec<i32> = cond.keys().copied().collect();
    if max == -1 || keys.len() as i32 <= max {
        return (keys, std::collections::BTreeSet::new());
    }
    let mut selected: std::collections::BTreeSet<i32> = std::collections::BTreeSet::new();
    if let Some(&before) = keys.iter().filter(|&&t| t < frame_idx).max() {
        selected.insert(before);
    }
    if let Some(&after) = keys.iter().filter(|&&t| t >= frame_idx).min() {
        selected.insert(after);
    }
    let mut remaining: Vec<i32> = keys
        .iter()
        .copied()
        .filter(|t| !selected.contains(t))
        .collect();
    remaining.sort_by_key(|&t| (t - frame_idx).abs());
    for t in remaining
        .into_iter()
        .take((max - selected.len() as i32).max(0) as usize)
    {
        selected.insert(t);
    }
    let unselected: std::collections::BTreeSet<i32> = keys
        .iter()
        .copied()
        .filter(|t| !selected.contains(t))
        .collect();
    (selected.into_iter().collect(), unselected)
}

/// Prune one object's `non_cond` bank to the future-readable window (F-024 eviction core, factored
/// out as a free fn so the bound is unit-testable without a loaded model): entries older than
/// `heavy_keep` have their heavy tensors nulled (gather short-circuits on `None`), and entries older
/// than `ptr_keep` are dropped entirely (the object-pointer window has passed). `heavy_keep` and
/// `ptr_keep` are the next gather's read floors; callers derive them from `NUM_MASKMEM`/`MAX_OBJ_PTRS`.
fn evict_stale_bank(bank: &mut ObjectBank, heavy_keep: i32, ptr_keep: i32) {
    for (&k, m) in bank.non_cond.iter_mut() {
        if k < heavy_keep {
            m.maskmem_features = None;
            m.maskmem_pos_enc = None;
        }
    }
    bank.non_cond.retain(|&k, _| k >= ptr_keep);
}

/// Bound the per-object **`cond`** bank's resident memory (sc-7060, the F-024 sibling): null the heavy
/// `maskmem_*` tensors of conditioning frames that no future `gather_memory` can read for spatial
/// memory, keeping the lightweight `object_pointer` (still read by the pointer loop) and the entry
/// itself (so [`select_closest_cond_frames`] still sees the key — it just contributes nothing, exactly
/// as for an unselected frame).
///
/// A cond frame at key `k` can be read for spatial memory by a future gather (frame_idx' ≥ `frame_idx`
/// + 1) iff EITHER:
///  - it is still *selectable* — fewer than [`MAX_COND_FRAME_NUM`] cond frames have a key `> k`. New
///    cond frames only accrue (entries are never removed, only heavy-nulled), so once
///    `MAX_COND_FRAME_NUM` newer keys exist `k` is never among the closest again (for any frame_idx' >
///    all current keys the closest are the largest keys); OR
///  - it is inside the spatial *fallback* window, `k >= heavy_keep` (`= frame_idx + 1 −
///    (NUM_MASKMEM − 1)`), which only slides forward.
///
/// So the heavy tensors are dead exactly when BOTH fail: `k < heavy_keep` AND ≥ `MAX_COND_FRAME_NUM`
/// cond keys exceed `k`. The newest `MAX_COND_FRAME_NUM` keys are therefore always protected. Same
/// single-source-of-truth derivation as [`evict_stale_bank`]; verified a strict no-op against
/// `select_closest_cond_frames` over a long synthetic clip in the tests.
fn evict_stale_cond_heavy(bank: &mut ObjectBank, heavy_keep: i32) {
    let n = bank.cond.len() as i32;
    if n <= MAX_COND_FRAME_NUM {
        return; // every cond frame is always selectable → nothing droppable
    }
    // BTreeMap iterates ascending, so index `i` has `n - 1 - i` newer keys; "at least
    // MAX_COND_FRAME_NUM newer" is `i < n - MAX_COND_FRAME_NUM`. The newest MAX_COND_FRAME_NUM keys
    // stay selectable and are never nulled.
    let droppable_below = n - MAX_COND_FRAME_NUM;
    for (i, (&k, m)) in bank.cond.iter_mut().enumerate() {
        if (i as i32) < droppable_below && k < heavy_keep {
            m.maskmem_features = None;
            m.maskmem_pos_enc = None;
        }
    }
}

/// `_apply_non_overlapping_constraints` + `_suppress_shrinked_masks` per prompt group.
#[allow(clippy::needless_range_loop)] // pixel-wise argmax over parallel grouped masks
fn suppress_pw_area_shrinkage(masks: &[Vec<f32>], prompts: &[i32]) -> Vec<Vec<f32>> {
    let n = masks.len();
    let mut out = masks.to_vec();
    for pg in unique(prompts) {
        let idxs: Vec<usize> = (0..n).filter(|&j| prompts[j] == pg).collect();
        if idxs.len() <= 1 {
            continue;
        }
        let len = masks[0].len();
        // pixel-wise argmax over the group; keep only the max object's logit, clamp others to ≤ -10.
        let mut constrained: Vec<Vec<f32>> = idxs.iter().map(|&j| masks[j].clone()).collect();
        for p in 0..len {
            let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
            for (gi, &j) in idxs.iter().enumerate() {
                if masks[j][p] > bv {
                    bv = masks[j][p];
                    best = gi;
                }
            }
            for gi in 0..idxs.len() {
                if gi != best && constrained[gi][p] > NO_OBJ_LOGIT {
                    constrained[gi][p] = NO_OBJ_LOGIT;
                }
            }
        }
        // shrink suppression: if area drops below 30% after constraints, fully suppress.
        for (gi, &j) in idxs.iter().enumerate() {
            let before = masks[j].iter().filter(|&&v| v > 0.0).count().max(1) as f32;
            let after = constrained[gi].iter().filter(|&&v| v > 0.0).count() as f32;
            if after / before >= 0.3 {
                out[j] = constrained[gi].clone();
            } else {
                out[j] = masks[j].iter().map(|&v| v.min(NO_OBJ_LOGIT)).collect();
            }
        }
    }
    out
}

fn unique(v: &[i32]) -> Vec<i32> {
    let mut s: Vec<i32> = v.to_vec();
    s.sort_unstable();
    s.dedup();
    s
}

/// Threshold a mask (logits or probabilities) at 0 → per-pixel bool. (Tracker masks are logits, det
/// masks are already centered at 0, so the same `> 0` rule applies to both — previously duplicated as
/// a separate `binarize_gt0`, F-071.)
fn binarize(m: &[f32]) -> Vec<bool> {
    m.iter().map(|&v| v > 0.0).collect()
}

fn mask_iou(a: &[bool], b: &[bool]) -> f32 {
    let mut inter = 0u32;
    let mut uni = 0u32;
    for (&x, &y) in a.iter().zip(b) {
        if x && y {
            inter += 1;
        }
        if x || y {
            uni += 1;
        }
    }
    inter as f32 / (uni.max(1) as f32)
}

/// Greedy mask-IoU NMS over above-threshold detections (`det_nms_thresh`, sc-4995). Mirrors the
/// `kernels`-enabled reference `nms_masks` → `generic_nms` (cv-utils, GPU-only) with a host pass:
/// process detections by **descending score**, keep each not-yet-suppressed one, and suppress any
/// lower-scored detection whose binarized mask-IoU with it is **strictly greater** than
/// `iou_threshold` (IoU == threshold is kept, matching `generic_nms_cpu`'s `<= threshold` keep rule).
/// Survivors are returned in their original (query) order, mirroring the reference's
/// `where(pred_probs > threshold)` index-order re-selection after zeroing the suppressed scores.
/// Suppression is confined to detections of the same `prompt_id` (the reference runs NMS per prompt).
/// Ties in score break by ascending original index (stable sort), matching the CPU oracle.
fn nms_dedup(dets: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    let n = dets.len();
    if iou_threshold <= 0.0 || n <= 1 {
        return dets;
    }
    let bins: Vec<Vec<bool>> = dets.iter().map(|d| binarize(&d.mask)).collect();
    // descending score; stable so ties keep ascending original index (matches np stable argsort)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        dets[b]
            .score
            .partial_cmp(&dets[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut suppressed = vec![false; n];
    let mut keep = vec![false; n];
    for pos in 0..n {
        let i = order[pos];
        if suppressed[i] {
            continue;
        }
        keep[i] = true;
        for &j in &order[pos + 1..] {
            if suppressed[j] || dets[j].prompt_id != dets[i].prompt_id {
                continue;
            }
            if mask_iou(&bins[i], &bins[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    dets.into_iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, d)| d)
        .collect()
}

/// 8-connected component sizes over a `LOW_RES²` boolean grid: each `true` pixel gets the pixel-count
/// of its component, each `false` pixel gets 0. Iterative flood fill (matches the reference's
/// `cc_torch`/triton/skimage path, all 8-connectivity). `mask` is row-major `[LOW_RES·LOW_RES]`.
fn component_areas(mask: &[bool]) -> Vec<usize> {
    let w = LOW_RES as usize;
    let h = LOW_RES as usize;
    let n = mask.len();
    let mut area = vec![0usize; n];
    let mut visited = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut comp: Vec<usize> = Vec::new();
    for start in 0..n {
        if !mask[start] || visited[start] {
            continue;
        }
        stack.clear();
        comp.clear();
        stack.push(start);
        visited[start] = true;
        while let Some(p) = stack.pop() {
            comp.push(p);
            let (r, c) = (p / w, p % w);
            let r0 = r.saturating_sub(1);
            let r1 = (r + 1).min(h - 1);
            let c0 = c.saturating_sub(1);
            let c1 = (c + 1).min(w - 1);
            for nr in r0..=r1 {
                for nc in c0..=c1 {
                    let np = nr * w + nc;
                    if np != p && mask[np] && !visited[np] {
                        visited[np] = true;
                        stack.push(np);
                    }
                }
            }
        }
        let sz = comp.len();
        for &p in &comp {
            area[p] = sz;
        }
    }
    area
}

/// Fill small holes / remove small sprinkles in a `LOW_RES²` mask of logits, in place
/// (`fill_holes_in_mask_scores`, sc-4995). Mirrors the reference exactly, using 8-connected
/// components throughout; `max_area <= 0` is a no-op:
///
/// - **fill holes**: background components (logit `<= 0`) of area `<= max_area` become `+0.1`.
/// - **remove sprinkles**: foreground components (logit `> 0`, evaluated *after* the fill) of area `<= min(total_foreground / 2, max_area)` become `-0.1`.
fn fill_holes_in_mask(mask: &mut [f32], max_area: i32) {
    if max_area <= 0 {
        return;
    }
    let max_area = max_area as usize;
    // fill holes: small background components flip to a small positive score.
    let bg: Vec<bool> = mask.iter().map(|&v| v <= 0.0).collect();
    let bg_area = component_areas(&bg);
    for (i, &is_bg) in bg.iter().enumerate() {
        if is_bg && bg_area[i] <= max_area {
            mask[i] = 0.1;
        }
    }
    // remove sprinkles: small foreground components (post-fill) flip to a small negative score. The
    // area threshold is per-mask: half the foreground area, clamped to `max_area`.
    let fg: Vec<bool> = mask.iter().map(|&v| v > 0.0).collect();
    let total_fg = fg.iter().filter(|&&x| x).count();
    let fg_thresh = (total_fg / 2).min(max_area);
    let fg_area = component_areas(&fg);
    for (i, &is_fg) in fg.iter().enumerate() {
        if is_fg && fg_area[i] <= fg_thresh {
            mask[i] = -0.1;
        }
    }
}

fn to_vec(a: &Array) -> Result<Vec<f32>> {
    Ok(a.reshape(&[-1])?
        .as_dtype(Dtype::Float32)?
        .as_slice::<f32>()
        .to_vec())
}

/// Flatten the memory encoder's NHWC `[1,72,72,64]` output to seq-first `[5184,1,64]`. The reference
/// stores `maskmem_features` as **bfloat16** (`bf16 = true`) but `maskmem_pos_enc` stays f32
/// (`to(pred_masks.dtype)`), so the two must round-trip differently.
fn seq_first(a: &Array, bf16: bool) -> Result<Array> {
    let sh = a.shape();
    let (g, c) = (sh[1], sh[3]);
    let flat = a.reshape(&[g * g, 1, c])?;
    if bf16 {
        Ok(flat.as_dtype(Dtype::Bfloat16)?.as_dtype(Dtype::Float32)?)
    } else {
        Ok(flat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::weights::Weights;

    /// Build a detection whose 288² mask is a `+1.0` rectangle (rows `r0..r1`, cols `c0..c1`) on a
    /// `-1.0` background, so `binarize` (`> 0`) recovers exactly that rectangle.
    fn rect_det(
        score: f32,
        prompt_id: i32,
        r0: usize,
        r1: usize,
        c0: usize,
        c1: usize,
    ) -> Detection {
        let w = LOW_RES as usize;
        let mut mask = vec![-1.0f32; w * w];
        for r in r0..r1 {
            for c in c0..c1 {
                mask[r * w + c] = 1.0;
            }
        }
        Detection {
            mask,
            score,
            prompt_id,
        }
    }

    fn scores(dets: &[Detection]) -> Vec<f32> {
        dets.iter().map(|d| d.score).collect()
    }

    fn dummy_fm() -> FrameMem {
        FrameMem {
            maskmem_features: Some(Array::from_slice(&[0.0f32], &[1, 1])),
            maskmem_pos_enc: Some(Array::from_slice(&[0.0f32], &[1, 1])),
            object_pointer: Array::from_slice(&[0.0f32], &[1, 1]),
            object_score: 0.0,
        }
    }

    /// F-024: `evict_stale_bank` drops exactly the `non_cond` entries no future `gather_memory` can
    /// read (older than the pointer window) and nulls the heavy tensors of those past the (tighter)
    /// spatial window, while keeping the rest and never touching `cond`. Derived from frame_idx=20.
    #[test]
    fn evict_stale_bank_prunes_only_unreadable_entries() {
        let mut bank = ObjectBank::default();
        for k in 0..=20 {
            bank.non_cond.insert(k, dummy_fm());
        }
        bank.cond.insert(3, dummy_fm()); // cond must be left intact

        let frame_idx = 20;
        let heavy_keep = frame_idx + 1 - (NUM_MASKMEM - 1);
        let ptr_keep = frame_idx + 1 - (MAX_OBJ_PTRS - 1);
        assert_eq!(
            (heavy_keep, ptr_keep),
            (15, 6),
            "window floors for frame 20"
        );
        evict_stale_bank(&mut bank, heavy_keep, ptr_keep);

        // keys < ptr_keep (0..=5): pointer window passed → entry dropped entirely.
        for k in 0..ptr_keep {
            assert!(!bank.non_cond.contains_key(&k), "key {k} must be evicted");
        }
        // keys in [ptr_keep, heavy_keep) (6..=14): kept, but heavy tensors nulled (object_pointer stays).
        for k in ptr_keep..heavy_keep {
            let m = bank
                .non_cond
                .get(&k)
                .unwrap_or_else(|| panic!("key {k} must be kept"));
            assert!(
                m.maskmem_features.is_none() && m.maskmem_pos_enc.is_none(),
                "heavy tensors must be nulled at key {k}"
            );
        }
        // keys >= heavy_keep (15..=20): still spatially readable → heavy tensors retained.
        for k in heavy_keep..=20 {
            let m = bank.non_cond.get(&k).unwrap();
            assert!(
                m.maskmem_features.is_some() && m.maskmem_pos_enc.is_some(),
                "heavy tensors must be kept at key {k}"
            );
        }
        // cond is never touched by `evict_stale_bank` (its own heavy bound is `evict_stale_cond_heavy`).
        assert!(bank.cond.contains_key(&3), "cond must be left intact");
        let c = bank.cond.get(&3).unwrap();
        assert!(
            c.maskmem_features.is_some() && c.maskmem_pos_enc.is_some(),
            "evict_stale_bank must not touch cond heavy tensors"
        );
    }

    /// sc-7060: `evict_stale_cond_heavy` is a **strict no-op** for spatial memory — it only nulls cond
    /// heavy tensors that no future `gather_memory` can read. Simulate a long clip (cond seeds +
    /// reconditioning cadence), and at every frame assert the spatial read-set (the real
    /// `select_closest_cond_frames` selection ++ the unselected-cond fallback window, exactly as
    /// `gather_memory` reads) still has its heavy tensors present after all prior-frame evictions.
    #[test]
    fn evict_stale_cond_heavy_never_nulls_a_readable_frame() {
        let mut bank = ObjectBank::default();
        // cond frames: an initial seed at 0, a second seed at 5, then reconditioning every
        // RECONDITION_EVERY up to 160 — the long-clip pattern the cond leak comes from.
        let mut cond_frames: Vec<i32> = vec![0, 5];
        let mut f = RECONDITION_EVERY;
        while f <= 160 {
            cond_frames.push(f);
            f += RECONDITION_EVERY;
        }

        let n_frames = 170;
        let mut nulled_total = 0usize;
        for frame_idx in 0..n_frames {
            if cond_frames.contains(&frame_idx) {
                bank.cond.insert(frame_idx, dummy_fm());
            }
            // Read-set exactly as `gather_memory`: selected cond (offset 0) ++ unselected cond in the
            // `[frame_idx-(NUM_MASKMEM-1), frame_idx-1]` fallback window (no non_cond in this fixture).
            let (selected, unselected) =
                select_closest_cond_frames(frame_idx, &bank.cond, MAX_COND_FRAME_NUM);
            let mut read_keys: std::collections::BTreeSet<i32> = selected.into_iter().collect();
            for rel in 1..NUM_MASKMEM {
                let prev = frame_idx - rel;
                if unselected.contains(&prev) {
                    read_keys.insert(prev);
                }
            }
            for k in &read_keys {
                let m = bank.cond.get(k).unwrap();
                assert!(
                    m.maskmem_features.is_some() && m.maskmem_pos_enc.is_some(),
                    "frame {frame_idx}: cond key {k} is in the spatial read-set but its heavy \
                     tensors were nulled by a prior eviction"
                );
            }
            // Evict after the frame's reads, mirroring `evict_stale_memory`.
            let heavy_keep = frame_idx + 1 - (NUM_MASKMEM - 1);
            evict_stale_cond_heavy(&mut bank, heavy_keep);
            nulled_total = bank
                .cond
                .values()
                .filter(|m| m.maskmem_features.is_none())
                .count();
        }
        // The eviction must actually bite (not a vacuous pass): old cond frames' heavy tensors are gone
        // while every entry (and its object_pointer) is retained.
        assert!(
            nulled_total > 0,
            "expected some cond heavy tensors to be nulled over a 170-frame clip"
        );
        assert_eq!(
            bank.cond.len(),
            cond_frames.len(),
            "cond entries (object_pointer) must be retained, only heavy tensors nulled"
        );
        // The newest MAX_COND_FRAME_NUM cond frames are always protected.
        for &k in cond_frames.iter().rev().take(MAX_COND_FRAME_NUM as usize) {
            let m = bank.cond.get(&k).unwrap();
            assert!(
                m.maskmem_features.is_some(),
                "newest cond frame {k} must keep its heavy tensors (always selectable)"
            );
        }
    }

    /// sc-4995: among heavily-overlapping detections, NMS keeps the highest-scored and drops the rest.
    #[test]
    fn nms_dedup_suppresses_overlapping_lower_score() {
        // two identical full-frame rectangles (IoU 1.0) → only the 0.9 survives.
        let dets = vec![
            rect_det(0.6, 0, 0, 100, 0, 100),
            rect_det(0.9, 0, 0, 100, 0, 100),
        ];
        let kept = nms_dedup(dets, DET_NMS_THRESH);
        assert_eq!(scores(&kept), vec![0.9]);
    }

    /// Disjoint detections (IoU 0) are all kept, in original (query) order.
    #[test]
    fn nms_dedup_keeps_disjoint() {
        let dets = vec![
            rect_det(0.8, 0, 0, 10, 0, 10),
            rect_det(0.7, 0, 50, 60, 50, 60),
        ];
        let kept = nms_dedup(dets, DET_NMS_THRESH);
        assert_eq!(scores(&kept), vec![0.8, 0.7]);
    }

    /// Suppression is **strictly** `IoU > threshold`: IoU exactly at the threshold is kept (matches
    /// `generic_nms_cpu`'s `<= threshold` keep rule). A=rows 0..3, B=rows 1..4 (full width) → IoU 0.5.
    #[test]
    fn nms_dedup_threshold_is_strict() {
        let pair = || {
            vec![
                rect_det(0.9, 0, 0, 3, 0, 100),
                rect_det(0.8, 0, 1, 4, 0, 100),
            ]
        };
        // IoU == 0.5 is NOT > 0.5 → both kept.
        assert_eq!(scores(&nms_dedup(pair(), 0.5)), vec![0.9, 0.8]);
        // IoU 0.5 > 0.4 → lower-scored suppressed.
        assert_eq!(scores(&nms_dedup(pair(), 0.4)), vec![0.9]);
    }

    /// Survivors come back in original query order, not score order (mirrors the reference's
    /// `where(pred_probs > threshold)` index-order re-selection after zeroing suppressed scores).
    #[test]
    fn nms_dedup_preserves_query_order() {
        let dets = vec![
            rect_det(0.7, 0, 0, 10, 0, 10),    // disjoint, kept
            rect_det(0.95, 0, 50, 60, 50, 60), // overlaps det2, higher score → kept
            rect_det(0.6, 0, 50, 60, 50, 60),  // suppressed by det1
        ];
        let kept = nms_dedup(dets, DET_NMS_THRESH);
        // query order [0.7, 0.95], NOT score order [0.95, 0.7].
        assert_eq!(scores(&kept), vec![0.7, 0.95]);
    }

    /// NMS is confined to a single prompt group: overlapping detections from different prompts coexist.
    #[test]
    fn nms_dedup_respects_prompt_groups() {
        let dets = vec![
            rect_det(0.9, 0, 0, 100, 0, 100),
            rect_det(0.8, 1, 0, 100, 0, 100), // same mask, different prompt → not suppressed
        ];
        let kept = nms_dedup(dets, DET_NMS_THRESH);
        assert_eq!(kept.len(), 2);
    }

    /// Degenerate inputs pass through unchanged.
    #[test]
    fn nms_dedup_empty_and_single() {
        assert!(nms_dedup(Vec::new(), DET_NMS_THRESH).is_empty());
        let one = nms_dedup(vec![rect_det(0.55, 0, 0, 10, 0, 10)], DET_NMS_THRESH);
        assert_eq!(scores(&one), vec![0.55]);
    }

    // --- hole-fill / sprinkle removal (sc-4995) ---

    fn bg_field(val: f32) -> Vec<f32> {
        vec![val; (LOW_RES * LOW_RES) as usize]
    }
    fn paint(m: &mut [f32], r0: usize, r1: usize, c0: usize, c1: usize, val: f32) {
        let w = LOW_RES as usize;
        for r in r0..r1 {
            for c in c0..c1 {
                m[r * w + c] = val;
            }
        }
    }
    fn at(m: &[f32], r: usize, c: usize) -> f32 {
        m[r * LOW_RES as usize + c]
    }

    /// A small background hole enclosed by foreground (area <= max_area) is filled to +0.1, while the
    /// large outer background and the surrounding foreground are untouched.
    #[test]
    fn fill_holes_fills_small_background_hole() {
        let mut m = bg_field(-1.0);
        paint(&mut m, 10, 40, 10, 40, 1.0); // 30x30 foreground block
        paint(&mut m, 20, 22, 20, 22, -1.0); // 2x2 enclosed hole (area 4)
        fill_holes_in_mask(&mut m, FILL_HOLE_AREA);
        assert_eq!(at(&m, 20, 20), 0.1, "enclosed hole filled");
        assert_eq!(at(&m, 11, 11), 1.0, "foreground untouched");
        assert_eq!(at(&m, 0, 0), -1.0, "outer background untouched");
    }

    /// Hole-fill area threshold is `<= max_area`: a 16-pixel hole is filled, a 20-pixel hole is not.
    #[test]
    fn fill_holes_area_threshold() {
        let mut m = bg_field(-1.0);
        paint(&mut m, 10, 60, 10, 60, 1.0); // 50x50 block
        paint(&mut m, 15, 19, 15, 19, -1.0); // 4x4 = 16 → filled
        paint(&mut m, 15, 19, 40, 45, -1.0); // 4x5 = 20 → kept
        fill_holes_in_mask(&mut m, FILL_HOLE_AREA);
        assert_eq!(at(&m, 15, 15), 0.1, "16-px hole filled (<= max_area)");
        assert_eq!(at(&m, 15, 40), -1.0, "20-px hole not filled (> max_area)");
    }

    /// A small isolated foreground speck (area <= threshold) is removed to -0.1; the dominant blob stays.
    #[test]
    fn fill_holes_removes_small_foreground_speck() {
        let mut m = bg_field(-1.0);
        paint(&mut m, 10, 60, 10, 60, 1.0); // big block (area 2500)
        paint(&mut m, 100, 101, 100, 101, 1.0); // 1-px speck
        fill_holes_in_mask(&mut m, FILL_HOLE_AREA);
        assert_eq!(at(&m, 100, 100), -0.1, "speck removed");
        assert_eq!(at(&m, 30, 30), 1.0, "dominant blob kept");
    }

    /// The sprinkle threshold is clamped to half the total foreground, so a lone small object (area
    /// <= max_area but > half its own area) is NOT wiped out.
    #[test]
    fn fill_holes_half_area_clamp_protects_lone_object() {
        let mut m = bg_field(-1.0);
        paint(&mut m, 50, 53, 50, 54, 1.0); // 3x4 = 12 px, the only foreground
        fill_holes_in_mask(&mut m, FILL_HOLE_AREA);
        // total_fg = 12 → fg_thresh = min(6, 16) = 6; blob area 12 > 6 → kept.
        assert_eq!(at(&m, 50, 50), 1.0, "lone small object preserved");
    }

    /// `max_area <= 0` is a no-op.
    #[test]
    fn fill_holes_noop_when_disabled() {
        let mut m = bg_field(-1.0);
        paint(&mut m, 10, 12, 10, 12, 1.0);
        let before = m.clone();
        fill_holes_in_mask(&mut m, 0);
        assert_eq!(m, before);
    }

    /// F-028: the detector segmenter and the tracker must share **one** PE backbone instance — both
    /// at load and after quantization — rather than each holding its own ~445M-param copy. Checks
    /// `Rc` pointer-identity of the two backbones (the cheapest, most direct proof that the weights
    /// are not duplicated). Weights-gated (no torch fixture needed — only the real `facebook/sam3`
    /// weights).
    #[test]
    #[ignore = "needs SAM3_WEIGHTS=<facebook/sam3 model.safetensors>"]
    fn backbone_is_shared_not_duplicated() {
        let weights_path = std::env::var("SAM3_WEIGHTS")
            .expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
        let w = Weights::from_file(&weights_path).expect("load sam3 weights");

        let mut model = Sam3VideoModel::from_weights(&w).expect("build video model");
        assert!(
            Rc::ptr_eq(
                &model.segmenter.vision_backbone_rc(),
                &model.tracker.backbone_rc(),
            ),
            "at load: segmenter and tracker must point at one shared PE backbone",
        );

        model.quantize(8).expect("quantize q8");
        assert!(
            Rc::ptr_eq(
                &model.segmenter.vision_backbone_rc(),
                &model.tracker.backbone_rc(),
            ),
            "after quantize: the shared backbone must stay a single quantized copy",
        );
    }

    /// sc-5409: running the PE backbone **once** and feeding both necks must be **bit-identical** to
    /// the old two-pass path (`encode_frame` / `segmenter.forward`, each re-running the backbone).
    /// Weights-gated (no torch fixture — only the real `facebook/sam3` weights).
    #[test]
    #[ignore = "needs SAM3_WEIGHTS=<facebook/sam3 model.safetensors>"]
    fn single_backbone_pass_matches_two_pass() {
        use mlx_rs::ops::{abs, max, subtract};

        let weights_path = std::env::var("SAM3_WEIGHTS")
            .expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
        let w = Weights::from_file(&weights_path).expect("load sam3 weights");
        let model = Sam3VideoModel::from_weights(&w).expect("build video model");

        // Deterministic non-constant frame + a fixed text prompt.
        let n = 3 * 1008 * 1008;
        let px: Vec<f32> = (0..n).map(|i| (i % 251) as f32 / 251.0 - 0.5).collect();
        let px = Array::from_slice(&px, &[1, 3, 1008, 1008]);
        let input_ids = Array::from_slice(&[0i32; 32], &[1, 32]);
        let text_mask = vec![1i32; 32];

        let max_abs_diff = |a: &Array, b: &Array| -> f32 {
            let diff = abs(subtract(a, b).unwrap()).unwrap();
            max(diff, None)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap()
                .item::<f32>()
        };

        // Tracker neck: old (backbone+neck) vs new (neck over the shared backbone features).
        let (emb_two_pass, _) = model.tracker.encode_frame(&px).expect("encode_frame");
        let features = model
            .segmenter
            .backbone_features(&px)
            .expect("backbone_features");
        let (emb_one_pass, _) = model
            .tracker
            .encode_frame_from_features(&features)
            .expect("encode_frame_from_features");
        assert_eq!(
            max_abs_diff(&emb_two_pass, &emb_one_pass),
            0.0,
            "tracker neck output must be bit-identical between two-pass and single-pass",
        );

        // Detector: old (forward) vs new (forward over the shared backbone features).
        let seg_two_pass = model
            .segmenter
            .forward(&px, &input_ids, &text_mask)
            .expect("forward");
        let seg_one_pass = model
            .segmenter
            .forward_from_backbone(&features, &input_ids, &text_mask)
            .expect("forward_from_backbone");
        assert_eq!(
            max_abs_diff(&seg_two_pass.pred_masks, &seg_one_pass.pred_masks),
            0.0,
            "detector masks must be bit-identical between two-pass and single-pass",
        );
    }
}
