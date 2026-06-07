//! Dual-stream MMDiT block. Port of the fork's `QwenTransformerBlock`. Each stream (image, text)
//! gets two AdaLN modulations from the timestep embedding — `mod1` around attention, `mod2` around
//! the feed-forward — with gated residuals; attention itself is joint over `[txt, img]`.

use mlx_rs::error::Exception;
use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{add, multiply, split, which};
use mlx_rs::transforms::compile::compile;
use mlx_rs::Array;

use mlx_gen::adapters::{prefixed_paths, AdaptableHost, AdaptableLinear};
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{compile_glue, join, linear_from, FeedForward, QwenJointAttention};

const LN_EPS: f32 = 1e-6;

pub struct QwenTransformerBlock {
    img_mod: AdaptableLinear,
    txt_mod: AdaptableLinear,
    attn: QwenJointAttention,
    img_ff: FeedForward,
    txt_ff: FeedForward,
}

impl AdaptableHost for QwenTransformerBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // Routes trained-file naming: the Diffusers modulation linears live under
        // `{img,txt}_mod.1` (Sequential[SiLU, Linear]), while the Rust field stores just the Linear.
        match path {
            ["img_mod", "1"] => Some(&mut self.img_mod),
            ["txt_mod", "1"] => Some(&mut self.txt_mod),
            ["attn", rest @ ..] => self.attn.adaptable_mut(rest),
            ["img_mlp", rest @ ..] => self.img_ff.adaptable_mut(rest),
            ["txt_mlp", rest @ ..] => self.txt_ff.adaptable_mut(rest),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = vec!["img_mod.1".to_string(), "txt_mod.1".to_string()];
        out.extend(prefixed_paths("attn", &self.attn));
        out.extend(prefixed_paths("img_mlp", &self.img_ff));
        out.extend(prefixed_paths("txt_mlp", &self.txt_ff));
        out
    }
}

impl QwenTransformerBlock {
    pub fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            img_mod: linear_from(w, &join(prefix, "img_mod_linear"), true)?,
            txt_mod: linear_from(w, &join(prefix, "txt_mod_linear"), true)?,
            attn: QwenJointAttention::from_weights(w, &join(prefix, "attn"), num_heads, head_dim)?,
            img_ff: FeedForward::from_weights(w, &join(prefix, "img_ff"))?,
            txt_ff: FeedForward::from_weights(w, &join(prefix, "txt_ff"))?,
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.img_mod.quantize(bits, None)?;
        self.txt_mod.quantize(bits, None)?;
        self.attn.quantize(bits)?;
        self.img_ff.quantize(bits)?;
        self.txt_ff.quantize(bits)?;
        Ok(())
    }

    /// Returns `(encoder_hidden_states, hidden_states)` (text, image) — matching the fork's order.
    ///
    /// `modulate_index` is `Some` only on the Qwen-Image-Edit-2511 `zero_cond_t` path: then
    /// `text_embeddings` is the doubled temb `[real_t ; zero_t]` (`[2B, dim]`), the image stream
    /// selects modulation per token by the index (`0` = noise → real `t`, `1` = conditioning image →
    /// `t 0`), and the text stream uses only the real-timestep half. `None` (T2I / 2509) is the
    /// original single-`temb` path, byte-for-byte unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Array,         // image [B, img_seq, dim]
        encoder_hidden_states: &Array, // text  [B, txt_seq, dim]
        text_embeddings: &Array,       // [B, dim]  (or [2B, dim] under zero_cond_t)
        img_cos: &Array,
        img_sin: &Array,
        txt_cos: &Array,
        txt_sin: &Array,
        mask: Option<&Array>,
        modulate_index: Option<&Array>,
    ) -> Result<(Array, Array)> {
        // SiLU'd timestep embedding. Under zero_cond_t this is [2B, dim]; the image modulation uses
        // the whole thing, the text modulation only the real-timestep half. SiLU is elementwise, so
        // `act[:B] == silu(temb[:B])` — slicing here is bit-identical to slicing the temb first.
        let act = silu(text_embeddings)?;
        let img_mod = self.img_mod.forward(&act)?;
        let txt_act = match modulate_index {
            Some(_) => split(&act, 2, 0)?.swap_remove(0), // real-timestep half [B, dim]
            None => act.clone(),
        };
        let txt_mod = self.txt_mod.forward(&txt_act)?;
        let img_mod = split(&img_mod, 2, 1)?; // [mod1, mod2], each [2B or B, 3*dim]
        let txt_mod = split(&txt_mod, 2, 1)?;

        let (img_modulated, img_gate1) = modulate_maybe_indexed(
            &layer_norm(hidden_states, None, None, LN_EPS)?,
            &img_mod[0],
            modulate_index,
        )?;
        let (txt_modulated, txt_gate1) = modulate(
            &layer_norm(encoder_hidden_states, None, None, LN_EPS)?,
            &txt_mod[0],
        )?;

        let (img_attn, txt_attn) = self.attn.forward(
            &img_modulated,
            &txt_modulated,
            img_cos,
            img_sin,
            txt_cos,
            txt_sin,
            mask,
        )?;

        let hidden_states = gated(hidden_states, &img_gate1, &img_attn)?;
        let encoder_hidden_states = gated(encoder_hidden_states, &txt_gate1, &txt_attn)?;

        let (img_mod2, img_gate2) = modulate_maybe_indexed(
            &layer_norm(&hidden_states, None, None, LN_EPS)?,
            &img_mod[1],
            modulate_index,
        )?;
        let img_ff = self.img_ff.forward(&img_mod2)?;
        let hidden_states = gated(&hidden_states, &img_gate2, &img_ff)?;

        let (txt_mod2, txt_gate2) = modulate(
            &layer_norm(&encoder_hidden_states, None, None, LN_EPS)?,
            &txt_mod[1],
        )?;
        let txt_ff = self.txt_ff.forward(&txt_mod2)?;
        let encoder_hidden_states = gated(&encoder_hidden_states, &txt_gate2, &txt_ff)?;

        Ok((encoder_hidden_states, hidden_states))
    }
}

/// `(x·(1+scale) + shift, gate)` from a `[B, 3*dim]` modulation (shift, scale, gate). Scale/shift/
/// gate broadcast over the sequence axis. The split + expand run eagerly (a shapeless `mx.compile`
/// can't infer a split's output shapes); when the sc-2963 glue toggle is on, the big affine
/// `x·(1+scale)+shift` is fused into one kernel. Bit-identical to the eager form — the `1.0` stays an
/// f32 scalar (the existing strong-`1` behaviour: `bf16 scale + f32 1` → f32, then `·` the f32 x).
fn modulate(x: &Array, mod_params: &Array) -> Result<(Array, Array)> {
    let p = split(mod_params, 3, 1)?; // shift, scale, gate — each [B, dim]
    let shift = p[0].expand_dims(1)?; // [B, 1, dim]
    let scale = p[1].expand_dims(1)?;
    let gate = p[2].expand_dims(1)?;
    let f = |(x, sc, sh): (&Array, &Array, &Array)| -> std::result::Result<Array, Exception> {
        add(
            &multiply(x, &add(sc, Array::from_slice(&[1.0f32], &[1]))?)?,
            sh,
        )
    };
    let out = if compile_glue() {
        compile(f, true)((x, &scale, &shift))?
    } else {
        f((x, &scale, &shift))?
    };
    Ok((out, gate))
}

/// `modulate` with optional per-token timestep selection (`zero_cond_t`, Qwen-Image-Edit-2511).
/// With `index = None` this is exactly [`modulate`] (T2I / 2509, the compiled fast path). With
/// `index = Some` the `mod_params` carry a doubled batch `[real_t ; zero_t]`; each image token picks
/// the real-`t` half where `index == 0` (noise) and the `t 0` half where `index == 1` (conditioning
/// image) — mirroring the fork's `_modulate(index)` / diffusers `_modulate`.
fn modulate_maybe_indexed(
    x: &Array,
    mod_params: &Array,
    index: Option<&Array>,
) -> Result<(Array, Array)> {
    let Some(index) = index else {
        return modulate(x, mod_params);
    };
    let p = split(mod_params, 3, 1)?; // shift, scale, gate — each [2B, dim]
    let one = Array::from_slice(&[1.0f32], &[1]);
    let index_expanded = index.expand_dims(2)?; // [B, seq, 1]
                                                // `which(cond, a, b)` selects `a` where cond is non-zero, `b` where zero — so with the 0/1 index
                                                // as cond, the `t 0` (cond) half goes to `a` and the real-`t` (noise) half to `b`. Equivalent to
                                                // the fork's `where(index == 0, real, zero)`; the selected values are identical (no arithmetic).
    let pick = |arr: &Array| -> Result<Array> {
        let halves = split(arr, 2, 0)?; // [real_t (B, dim), zero_t (B, dim)]
        let real_t = halves[0].expand_dims(1)?; // [B, 1, dim]
        let zero_t = halves[1].expand_dims(1)?;
        Ok(which(&index_expanded, &zero_t, &real_t)?)
    };
    let shift = pick(&p[0])?;
    let scale = pick(&p[1])?;
    let gate = pick(&p[2])?;
    let out = add(&multiply(x, &add(&scale, &one)?)?, &shift)?;
    Ok((out, gate))
}

/// Gated residual `x + gate·y` — one fused kernel when the sc-2963 glue toggle is on; bit-identical
/// to the eager `add(x, gate·y)`.
fn gated(x: &Array, gate: &Array, y: &Array) -> Result<Array> {
    let f = |(x, g, y): (&Array, &Array, &Array)| -> std::result::Result<Array, Exception> {
        add(x, &multiply(g, y)?)
    };
    if compile_glue() {
        Ok(compile(f, true)((x, gate, y))?)
    } else {
        Ok(f((x, gate, y))?)
    }
}

#[cfg(test)]
mod sc2963 {
    use super::*;
    use crate::transformer::compile_test_util::{max_abs, rnd};
    use crate::transformer::set_compile_glue;
    use mlx_rs::Dtype::{Bfloat16, Float32};

    // sc-2963: the compiled adaLN affine + gated residual are bit-identical to eager (`max|Δ|=0`) at
    // the production dtypes — f32 latents, bf16 modulation (the `mod_params` come from a bf16 Linear).
    #[test]
    fn compiled_modulate_and_gated_bit_identical_to_eager() {
        let (b, s, dim) = (2i32, 16i32, 128i32);
        // modulate: f32 hidden, bf16 mod_params [B, 3·dim].
        let x = rnd(&[b, s, dim], Float32);
        let mod_params = rnd(&[b, 3 * dim], Bfloat16);
        set_compile_glue(false);
        let (eo, eg) = modulate(&x, &mod_params).unwrap();
        set_compile_glue(true);
        let (co, cg) = modulate(&x, &mod_params).unwrap();
        set_compile_glue(false);
        assert_eq!(max_abs(&co, &eo), 0.0, "modulate out");
        assert_eq!(max_abs(&cg, &eg), 0.0, "modulate gate");

        // gated: f32 x + (bf16 gate · f32 y).
        let gate = rnd(&[b, 1, dim], Bfloat16);
        let y = rnd(&[b, s, dim], Float32);
        set_compile_glue(false);
        let e = gated(&x, &gate, &y).unwrap();
        set_compile_glue(true);
        let c = gated(&x, &gate, &y).unwrap();
        set_compile_glue(false);
        assert_eq!(max_abs(&c, &e), 0.0, "gated");
    }
}

// sc-2997: the Qwen-Image-Edit-2511 `zero_cond_t` per-token timestep selection. Weight-free, fast,
// deterministic — proves the core `modulate_maybe_indexed` logic without the 40 GB edit e2e.
#[cfg(test)]
mod zero_cond_t {
    use super::*;
    use crate::transformer::compile_test_util::{max_abs, rnd};
    use mlx_rs::ops::concatenate_axis;
    use mlx_rs::Dtype::Float32;

    // `modulate_maybe_indexed` selects, per token, from a doubled `[real_t ; zero_t]` mod_params:
    // index 0 → the real-timestep half (== plain `modulate` on that half), index 1 → the zero-t half.
    #[test]
    fn all_zero_or_all_one_index_matches_plain_modulate() {
        let (b, seq, dim) = (1i32, 6i32, 16i32);
        let x = rnd(&[b, seq, dim], Float32);
        let real = rnd(&[b, 3 * dim], Float32); // real-timestep modulation params
        let zero = rnd(&[b, 3 * dim], Float32); // zero-timestep modulation params
        let doubled = concatenate_axis(&[&real, &zero], 0).unwrap(); // [2B, 3·dim]

        let idx0 = Array::from_slice(&vec![0i32; (b * seq) as usize], &[b, seq]);
        let (out0, gate0) = modulate_maybe_indexed(&x, &doubled, Some(&idx0)).unwrap();
        let (er, eg) = modulate(&x, &real).unwrap();
        assert_eq!(
            max_abs(&out0, &er),
            0.0,
            "all-zero index out == real-half modulate"
        );
        assert_eq!(
            max_abs(&gate0, &eg),
            0.0,
            "all-zero index gate == real-half gate"
        );

        let idx1 = Array::from_slice(&vec![1i32; (b * seq) as usize], &[b, seq]);
        let (out1, gate1) = modulate_maybe_indexed(&x, &doubled, Some(&idx1)).unwrap();
        let (ez, egz) = modulate(&x, &zero).unwrap();
        assert_eq!(
            max_abs(&out1, &ez),
            0.0,
            "all-one index out == zero-half modulate"
        );
        assert_eq!(
            max_abs(&gate1, &egz),
            0.0,
            "all-one index gate == zero-half gate"
        );
    }

    // A mixed index composes per token: noise rows (0) use the real half, conditioning rows (1) use
    // the zero half — exactly the Edit-2511 layout `[0]*noise_len + [1]*cond_len`.
    #[test]
    fn mixed_index_composes_per_token() {
        let (b, dim) = (1i32, 16i32);
        let x = rnd(&[b, 4, dim], Float32);
        let real = rnd(&[b, 3 * dim], Float32);
        let zero = rnd(&[b, 3 * dim], Float32);
        let doubled = concatenate_axis(&[&real, &zero], 0).unwrap();
        // [noise, noise, cond, cond]
        let idx = Array::from_slice(&[0i32, 0, 1, 1], &[b, 4]);
        let (out, gate) = modulate_maybe_indexed(&x, &doubled, Some(&idx)).unwrap();

        // `er`/`ez` are the modulated activations over the full seq (splittable); the plain gates
        // `egate`/`ezgate` are `[B, 1, dim]` (broadcast over seq) — compared via `max_abs`'s broadcast.
        let (er, egate) = modulate(&x, &real).unwrap(); // all-real over the seq
        let (ez, ezgate) = modulate(&x, &zero).unwrap(); // all-zero over the seq
        let out_parts = split(&out, 2, 1).unwrap(); // [rows 0..2, rows 2..4]
        let gate_parts = split(&gate, 2, 1).unwrap();
        let er_noise = split(&er, 2, 1).unwrap().swap_remove(0); // real, rows 0..2
        let ez_cond = split(&ez, 2, 1).unwrap().swap_remove(1); // zero, rows 2..4
        assert_eq!(
            max_abs(&out_parts[0], &er_noise),
            0.0,
            "noise rows use real half"
        );
        assert_eq!(
            max_abs(&out_parts[1], &ez_cond),
            0.0,
            "cond rows use zero half"
        );
        assert_eq!(
            max_abs(&gate_parts[0], &egate),
            0.0,
            "noise gate uses real half"
        );
        assert_eq!(
            max_abs(&gate_parts[1], &ezgate),
            0.0,
            "cond gate uses zero half"
        );
    }
}
