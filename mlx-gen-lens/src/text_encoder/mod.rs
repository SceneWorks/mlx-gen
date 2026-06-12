//! The Lens text encoder — gpt-oss-20b run encoder-only (forward to layer 23, capture hidden states
//! at layers `[5, 11, 17, 23]`). See epic 3164.
//!
//! Ported so far: the gpt-oss decoder-layer **attention core** (sc-3165), the **MoE** feed-forward +
//! full **decoder-layer** assembly (sc-3166), and the MXFP4 expert dequant ([`mxfp4`]). The 24-layer
//! stack + multi-layer hidden capture (sc-3171) and the encoder front-end projection follow.

pub mod gpt_oss;
pub mod mxfp4;
