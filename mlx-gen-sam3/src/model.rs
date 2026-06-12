//! SAM3 still-image concept segmenter — assembles the PE vision encoder (A), CLIP text encoder (B),
//! DETR detector (C), and mask head (D) into the end-to-end `Sam3Model` image path (epic 4910).
//!
//! `pixel_values[1,3,1008,1008] + "person" → per-instance masks`. Mirrors `Sam3Model.forward` for
//! the text-only (no geometry prompt) Promptable Concept Segmentation path.

use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::{Sam3DetrConfig, Sam3TextConfig, Sam3VisionConfig};
use crate::mask::{post_process_instances, Instance, Sam3MaskHead};
use crate::{Sam3Detector, Sam3TextEncoder, Sam3VisionEncoder};

/// Full raw outputs of the image segmenter (pre-post-process).
pub struct SegmentationOutput {
    /// `[1, Q]` concept logits.
    pub pred_logits: Array,
    /// `[1, Q, 4]` boxes xyxy ∈ [0, 1].
    pub pred_boxes: Array,
    /// `[1, 1]` presence logit.
    pub presence_logits: Array,
    /// `[1, Q, 288, 288]` per-query mask logits.
    pub pred_masks: Array,
    /// `[1, 288, 288, 1]` semantic-segmentation logits.
    pub semantic_seg: Array,
}

/// End-to-end SAM3 still-image concept segmenter.
pub struct Sam3ImageSegmenter {
    vision: Sam3VisionEncoder,
    text: Sam3TextEncoder,
    detector: Sam3Detector,
    mask_head: Sam3MaskHead,
}

impl Sam3ImageSegmenter {
    /// Load every stage from a `facebook/sam3` weight map.
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let detr_cfg = Sam3DetrConfig::sam3();
        Ok(Self {
            vision: Sam3VisionEncoder::from_weights(
                w,
                "detector_model.vision_encoder",
                &Sam3VisionConfig::sam3(),
            )?,
            text: Sam3TextEncoder::from_weights(
                w,
                "detector_model.text_encoder.text_model",
                "detector_model.text_projection",
                &Sam3TextConfig::sam3(),
            )?,
            detector: Sam3Detector::from_weights(w, "detector_model", &detr_cfg)?,
            mask_head: Sam3MaskHead::from_weights(w, "detector_model", &detr_cfg)?,
        })
    }

    /// `pixel_values`: NCHW `[1, 3, 1008, 1008]`; `input_ids`: `[1, 32]`; `text_mask`: per-token
    /// validity (`1`/`0`). Runs the full detector + mask head.
    pub fn forward(
        &self,
        pixel_values: &Array,
        input_ids: &Array,
        text_mask: &[i32],
    ) -> Result<SegmentationOutput> {
        let fpn = self.vision.forward(pixel_values)?; // NHWC [288²,144²,72²,36²]
        let text = self.text.forward(input_ids, text_mask)?; // [1,32,256]
        let det = self.detector.forward(&fpn[2], &text, text_mask)?;

        let backbone = [fpn[0].clone(), fpn[1].clone(), fpn[2].clone()];
        let prompt_key_mask = prompt_key_mask(text_mask);
        let masks = self.mask_head.forward(
            &det.query_hidden,
            &backbone,
            &det.encoder_hidden_states,
            &text,
            &prompt_key_mask,
        )?;

        Ok(SegmentationOutput {
            pred_logits: det.pred_logits,
            pred_boxes: det.pred_boxes,
            presence_logits: det.presence_logits,
            pred_masks: masks.pred_masks,
            semantic_seg: masks.semantic_seg,
        })
    }

    /// Convenience: full forward + instance post-process. `target_wh` is the original image size
    /// (for box scaling); masks come back at the native 288² resolution.
    #[allow(clippy::too_many_arguments)]
    pub fn segment(
        &self,
        pixel_values: &Array,
        input_ids: &Array,
        text_mask: &[i32],
        target_wh: (f32, f32),
        threshold: f32,
        mask_threshold: f32,
    ) -> Result<Vec<Instance>> {
        let out = self.forward(pixel_values, input_ids, text_mask)?;
        post_process_instances(
            &out.pred_logits,
            &out.pred_boxes,
            &out.presence_logits,
            &out.pred_masks,
            target_wh,
            threshold,
            mask_threshold,
        )
    }
}

/// Additive key-padding mask `[1, 1, 1, L]` (0 valid, −1e9 padded) for the mask head's prompt attn.
fn prompt_key_mask(text_mask: &[i32]) -> Array {
    let row: Vec<f32> = text_mask
        .iter()
        .map(|&m| if m == 1 { 0.0 } else { -1e9 })
        .collect();
    Array::from_slice(&row, &[1, 1, 1, row.len() as i32])
}
