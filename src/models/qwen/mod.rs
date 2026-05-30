//! Qwen-Image family. v1 ships the Qwen2-VL image processor (Qwen-Image-Edit reference
//! preprocessing); the transformer / VAE / text encoder land with the model port (sc-2348).

pub mod image_processor;

pub use image_processor::{ImageInput, ProcessedImage, QwenImageProcessor};
