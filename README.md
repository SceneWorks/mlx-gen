# mlx-gen

Rust-native inference for generative **image and video** models on Apple [MLX](https://github.com/ml-explore/mlx), built on [`mlx-rs`](https://crates.io/crates/mlx-rs).

> **Status:** name reserved / work in progress — not yet usable.

A from-scratch Rust reimplementation of the MLX image/video model stack (a divergence from the Python `mflux` / `mlx-video` lineage), collapsing on-device inference into a single statically-linked binary with no Python sidecar.

**Planned scope**

- **Image:** FLUX.1, FLUX.2-klein (incl. KV-cache), Qwen-Image, Z-Image (incl. ControlNet)
- **Video:** Wan2.2, LTX-2.3
- **Adapters:** LoRA, LoKr (reconstruct + residual + stacking), ControlNet
- **Quantization:** Q4 / Q8

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
