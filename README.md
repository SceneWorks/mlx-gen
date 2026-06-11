# mlx-gen

Rust-native inference for generative **image and video** models on Apple [MLX](https://github.com/ml-explore/mlx), built on [`mlx-rs`](https://crates.io/crates/mlx-rs).

> **Status:** active — 16 model provider crates with merged, parity-validated engines (image, video, identity, and understanding models). Built as a Rust library workspace consumed in-process; not yet published to crates.io. See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

A from-scratch Rust reimplementation of the MLX image/video model stack (a divergence from the Python `mflux` / `mlx-video` lineage), collapsing on-device inference into a single statically-linked component with no Python sidecar. Each model family is its own provider crate registered through the core `mlx-gen` `Generator` contract.

**Supported models**

- **Image:** FLUX.1 (schnell/dev, incl. Hyper few-step), FLUX.2-klein (Qwen3 text encoder + KV-cache), Chroma (`chroma1_hd`/`base`/`flash`), Qwen-Image (+ Qwen-Image-Edit), Stable Diffusion XL (+ inpaint/outpaint, IP-Adapter, tile-ControlNet, LCM/Lightning/Hyper), Kolors (bilingual, ChatGLM3 text encoder), Z-Image (incl. ControlNet), SenseNova-U1 (unified understanding + generation: T2I, image-edit, VQA, interleaved document)
- **Video:** Wan2.2 (text/image-to-video, incl. VACE), LTX-2.3 (text-to-video), Stable Video Diffusion (image-to-video)
- **Identity:** PuLID-FLUX and InstantID, over a native MLX face stack (SCRFD + ArcFace + BiSeNet)
- **Understanding:** JoyCaption (captioning), SAM2 (segmentation)
- **Adapters:** LoRA, LoKr (reconstruct + forward-time residual + stacking, quant-safe), ControlNet, IP-Adapter
- **Quantization:** group-wise affine Q4 / Q8 (byte-identical to the reference packing)

Requires a Mac with full Xcode + the Metal Toolchain (MLX's Metal kernels compile from source).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). You are free to
use, modify, and distribute mlx-gen, **including commercially**, under those terms.

## Acknowledgements

mlx-gen is an independent Rust reimplementation and includes no copied source,
but it stands on the work of others:

- [Apple MLX](https://github.com/ml-explore/mlx) (MIT) and [mlx-rs](https://crates.io/crates/mlx-rs) (Apache-2.0 OR MIT) — the on-device tensor stack
- [mflux](https://github.com/filipstrand/mflux) (MIT) — the MLX diffusion lineage mlx-gen diverged from and validates parity against
- [Apple mlx-examples](https://github.com/ml-explore/mlx-examples) (MIT)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) (Apache-2.0) — the upstream model architectures

See [NOTICE](NOTICE) for full attribution.
