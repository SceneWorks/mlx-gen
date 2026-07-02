//! Family-agnostic LoRA/LoKr **training** machinery (epic 3039) — the adapter-factor lifecycle the
//! spike (sc-3042) proved, hoisted out of the first family trainer (Z-Image, sc-3044) so every
//! family crate (Z-Image, SDXL sc-3045, Wan sc-3046, LTX sc-3047) shares one implementation.
//!
//! The model crates do NOT use mlx-rs's `Module`/`ModuleParameters` system (hand-rolled `&self`
//! forwards over raw `Array`s, [`crate::adapters`]), so training uses **functional autograd**: the
//! trainable factors live OUTSIDE the model in a [`LoraParams`] map, and each step are re-injected
//! into the target [`AdaptableLinear`](crate::adapters::AdaptableLinear)s as a single
//! [`Adapter`] via [`AdaptableLinear::set_adapters`](crate::adapters::AdaptableLinear::set_adapters).
//! The injection mirrors the inference reload op-for-op — for LoRA: transpose the `[r,in]`/`[out,r]`
//! factors, fold `alpha/rank` into `b`, `scale = 1`; for LoKr: reconstruct the delta with the SAME
//! [`reconstruct_lokr_delta`] the loader uses — so the trained adapter round-trips through the
//! inference path.
//!
//! Everything here is generic over the adapter host ([`AdaptableHost`]); the two genuine per-family
//! differences are passed in by the caller:
//!   * **LoKr reconstruct dtype** — `Bfloat16` for the bf16-residual families (Z-Image/Qwen),
//!     `Float32` for the f32-everywhere SDXL merge path. Training must reconstruct at the dtype the
//!     inference loader uses, so the adapter round-trips.
//!   * **PEFT save-key prefix** — `""` for the DiT families (`{path}.lora_A.weight`),
//!     `"base_model.model.unet."` for SDXL (what `peft.save_pretrained()` / the SceneWorks
//!     `_SdxlLoraBackend` emit, and what the SDXL loader's PEFT classifier expects).
//!
//! The model forward, the noise/target construction, and the text/VAE encoding stay in the family
//! crate (they are model-specific); this module owns only the adapter factors.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::multiply;
use mlx_rs::{random, Array, Dtype};

use crate::adapters::{reconstruct_lokr_delta, AdaptableHost, Adapter};
use crate::Result;

/// The trainable LoRA/LoKr factor map — keyed by `{path}.<factor>` (e.g. `…to_q.lora_a`,
/// `…to_q.lokr_w1`). The autograd arguments (`keyed_value_and_grad`) and the optimizer-stepped
/// state.
pub type LoraParams = HashMap<Rc<str>, Array>;

/// One LoRA-trained Linear: its dotted module path (e.g. `down_blocks.1.…attn1.to_q`) plus the
/// pre-built parameter-map keys and the `[out, in]` dims read off the base weight.
pub struct LoraTarget {
    pub path: String,
    a_key: Rc<str>,
    b_key: Rc<str>,
    pub in_f: i32,
    pub out_f: i32,
}

/// One LoKr-trained Linear: its path, the base `[out,in]` shape, and the factor-map keys —
/// `lokr_w1` always, then either full `lokr_w2` or low-rank `lokr_w2_a`/`lokr_w2_b`.
pub struct LokrTarget {
    path: String,
    base_shape: Vec<i32>,
    w1_key: Rc<str>,
    w2_key: Option<Rc<str>>,
    w2a_key: Option<Rc<str>>,
    w2b_key: Option<Rc<str>>,
}

/// LyCORIS dimension factorization: split `dimension` into `(a, b)`, `a*b = dimension`, `a <= b`,
/// with `a` as close to `factor` (or balanced/√dimension when `factor < 0`) as a divisor allows.
/// The LoKr weight `[out,in]` then factors as `kron(w1, w2)` with `w1 = [fac(out).0, fac(in).0]`,
/// `w2 = [fac(out).1, fac(in).1]`. Port of LyCORIS `factorization`.
pub fn factorization(dimension: i32, factor: i32) -> (i32, i32) {
    if factor > 0 && dimension % factor == 0 {
        let n = dimension / factor;
        return if factor > n { (n, factor) } else { (factor, n) };
    }
    let factor = if factor < 0 { dimension } else { factor };
    let (mut m, mut n) = (1i32, dimension);
    let mut length = m + n;
    while m < n {
        let mut new_m = m + 1;
        while dimension % new_m != 0 {
            new_m += 1;
        }
        let new_n = dimension / new_m;
        if new_m + new_n > length || new_m > factor {
            break;
        }
        m = new_m;
        n = new_n;
        length = m + n;
    }
    if m > n {
        (n, m)
    } else {
        (m, n)
    }
}

/// Resolve each `target_paths` entry on `host`, read its `[out,in]` dims, and initialise the
/// trainable LoRA factors the Python `_MlxLoRALinear` way — `A ~ N(0, 0.02)` `[rank,in]`,
/// `B = 0` `[out,rank]` — keyed `{path}.lora_a` / `{path}.lora_b`. The `B = 0` init makes the
/// adapter start as an exact no-op (it only learns from the gradient).
pub fn build_lora_targets<H: AdaptableHost>(
    host: &mut H,
    target_paths: &[String],
    rank: i32,
    seed: u64,
) -> Result<(Vec<LoraTarget>, LoraParams)> {
    // F-065: training-side twin of the F-002/F-010 load guard. rank <= 0 yields a 0·inf = NaN
    // no-op delta (and a degenerate `[0,in]`/`[out,0]` factor shape); reject it up front rather
    // than training a silently-corrupt adapter.
    if rank <= 0 {
        return Err(crate::Error::Msg(format!(
            "LoRA training: rank must be > 0, got {rank}"
        )));
    }
    let mut targets = Vec::with_capacity(target_paths.len());
    let mut params: LoraParams = HashMap::new();
    for (i, path) in target_paths.iter().enumerate() {
        let segs: Vec<&str> = path.split('.').collect();
        let lin = host.adaptable_mut(&segs).ok_or_else(|| -> crate::Error {
            format!("LoRA target does not resolve on the model: {path}").into()
        })?;
        let shape = lin.base_shape(); // [out, in]
        let (out_f, in_f) = (shape[0], shape[1]);

        let a_key: Rc<str> = Rc::from(format!("{path}.lora_a"));
        let b_key: Rc<str> = Rc::from(format!("{path}.lora_b"));
        // Distinct subkey per target so the RNG init differs per layer.
        let ka = random::key(seed.wrapping_add(2 * i as u64 + 1))?;
        let a = multiply(
            &random::normal::<f32>(&[rank, in_f], None, None, Some(&ka))?,
            Array::from_slice(&[0.02f32], &[1]),
        )?;
        let b = Array::zeros::<f32>(&[out_f, rank])?;
        mlx_rs::transforms::eval([&a, &b])?;
        params.insert(a_key.clone(), a);
        params.insert(b_key.clone(), b);
        targets.push(LoraTarget {
            path: path.clone(),
            a_key,
            b_key,
            in_f,
            out_f,
        });
    }
    Ok((targets, params))
}

/// Inject the current trainable factors as one `Adapter::Lora` per target — EXACTLY as the inference
/// reload (`install_lora_groups`): transpose `[r,in]`→`[in,r]` and `[out,r]`→`[r,out]`, fold
/// `alpha/rank` into `b`, `scale = 1`. Differentiable (the transposes/fold are traced).
pub fn install_training_lora<H: AdaptableHost>(
    host: &mut H,
    params: &LoraParams,
    targets: &[LoraTarget],
    alpha: f32,
) -> MlxResult<()> {
    install_training_lora_as(host, params, targets, alpha, None)
}

/// [`install_training_lora`] with an optional compute-dtype cast: when `dtype` is `Some`, the folded
/// factors are cast to it inside the traced graph (sc-4887 bf16 mixed-precision training — the
/// residual must match the bf16 activation stream or every adapted Linear silently re-promotes the
/// whole chain to f32). The trainable leaves stay f32; the gradient flows back through the `astype`
/// VJP, so the optimizer still sees f32 grads (master-weights pattern). `None` = factor dtype as-is.
pub fn install_training_lora_as<H: AdaptableHost>(
    host: &mut H,
    params: &LoraParams,
    targets: &[LoraTarget],
    alpha: f32,
    dtype: Option<Dtype>,
) -> MlxResult<()> {
    for t in targets {
        // `.get().ok_or_else()?` rather than a direct index: a bookkeeping bug that drops a key from
        // the optimizer-stepped params map must surface as a typed error, not a panic (F-008).
        let a = params
            .get(&t.a_key)
            .ok_or_else(|| Exception::custom(format!("LoRA param missing: {}", t.a_key)))?
            .t(); // [r,in] -> [in,r]
        let b_t = params
            .get(&t.b_key)
            .ok_or_else(|| Exception::custom(format!("LoRA param missing: {}", t.b_key)))?
            .t(); // [out,r] -> [r,out]
        let rank = a.shape()[1] as f32;
        let b = b_t.multiply(Array::from_slice(&[alpha / rank], &[1]))?;
        let (a, b) = match dtype {
            Some(dt) => (a.as_dtype(dt)?, b.as_dtype(dt)?),
            None => (a, b),
        };
        let segs: Vec<&str> = t.path.split('.').collect();
        let lin = host
            .adaptable_mut(&segs)
            .ok_or_else(|| Exception::custom(format!("LoRA target not found: {}", t.path)))?;
        lin.set_adapters(vec![Adapter::Lora { a, b, scale: 1.0 }]);
    }
    Ok(())
}

/// Clear every listed path's adapter stack (back to the bare frozen base).
pub fn clear_adapters<H: AdaptableHost>(host: &mut H, paths: &[String]) {
    for path in paths {
        let segs: Vec<&str> = path.split('.').collect();
        if let Some(lin) = host.adaptable_mut(&segs) {
            lin.set_adapters(Vec::new());
        }
    }
}

/// Write the trainable LoRA factors as PEFT-format safetensors — `{prefix}{path}.lora_A.weight`
/// `[r,in]`, `{prefix}{path}.lora_B.weight` `[out,r]`, scalar `{prefix}{path}.alpha`. `key_prefix`
/// is `""` for the DiT families (bare dotted paths) and `"base_model.model.unet."` for SDXL (what
/// `peft.save_pretrained()` emits and the SDXL loader's PEFT classifier expects). Metadata records
/// the network type/rank/alpha (the epic-2193 reload contract).
pub fn save_lora_peft(
    params: &LoraParams,
    targets: &[LoraTarget],
    alpha: f32,
    rank: u32,
    key_prefix: &str,
    path: impl AsRef<Path>,
) -> Result<()> {
    let alphas: Vec<(String, Array)> = targets
        .iter()
        .map(|t| {
            (
                format!("{key_prefix}{}.alpha", t.path),
                Array::from_slice(&[alpha], &[1]),
            )
        })
        .collect();
    let mut entries: Vec<(String, &Array)> = Vec::with_capacity(targets.len() * 3);
    for t in targets {
        // `.get().ok_or_else()?` over a direct index so a missing param surfaces as a typed error
        // rather than a panic mid-checkpoint (F-008).
        let a = params
            .get(&t.a_key)
            .ok_or_else(|| crate::Error::Msg(format!("LoRA param missing: {}", t.a_key)))?;
        let b = params
            .get(&t.b_key)
            .ok_or_else(|| crate::Error::Msg(format!("LoRA param missing: {}", t.b_key)))?;
        entries.push((format!("{key_prefix}{}.lora_A.weight", t.path), a));
        entries.push((format!("{key_prefix}{}.lora_B.weight", t.path), b));
    }
    for (k, v) in &alphas {
        entries.push((k.clone(), v));
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("networkType".to_string(), "lora".to_string());
    meta.insert("rank".to_string(), rank.to_string());
    meta.insert("alpha".to_string(), alpha.to_string());
    Array::save_safetensors(entries, Some(&meta), path)?;
    Ok(())
}

/// Initialise trainable LoKr factors per target, matching **PEFT `LoKrConfig(init_weights=True)`'s
/// `reset_adapter_parameters`** (sc-5179) — the init the Python LoRA/LoKr trainers
/// (`lens_train_runner.py` and the SDXL/Wan/etc. backends) use, so a *Python LoKr learning rate
/// transfers* to this native path.
///
/// The weight `[out,in]` factors as `kron(w1[out_a,in_a], w2[out_b,in_b])` (`decompose_both` is off,
/// PEFT's default — `w1` is never further low-ranked). `w2` is low-ranked to `rank` when
/// `rank < max(out_b,in_b)/2` (PEFT's `use_w2` predicate). The **first** factor `w1` is
/// **zero-initialised**; the **second** factor `w2` (full, or both `w2_a`/`w2_b` low-rank) is
/// **kaiming-uniform `a=√5`** ⇒ `U(±1/√fan_in)`, `fan_in` = the factor's input dim. Because the zeroed
/// factor is `w1`, the initial delta `kron(0, w2)·(alpha/rank) = 0` (the LoKr analog of LoRA's
/// `B = 0`), while the *non*-zero factor's **fan-in-scaled** magnitude is what makes the LoKr
/// lr-sensitivity track PEFT. `factor` is the decompose knob (`-1` = balanced/auto).
///
/// NOTE (sc-5179): this *replaced* the prior `w1 ~ N(0,0.02)` / zeroed-`w2` init — the **opposite**
/// zeroed factor and a **fixed** 0.02 scale (~4–5× below PEFT's fan-in-scaled kaiming), which is why
/// LoKr formerly needed a ~10× higher, non-transferable lr than LoRA. The save/round-trip format
/// (`lokr_w1` + `lokr_w2`|`lokr_w2_a`/`lokr_w2_b`) and `reconstruct_lokr_delta` are unchanged, so prior
/// adapters still load; only newly-*trained* LoKr dynamics change. (The `factor > 0` factorization
/// sort still differs from PEFT — a separate pre-existing divergence; the default `factor = -1` matches.)
pub fn build_lokr_targets<H: AdaptableHost>(
    host: &mut H,
    target_paths: &[String],
    rank: i32,
    factor: i32,
    seed: u64,
) -> Result<(Vec<LokrTarget>, LoraParams)> {
    // F-065: training-side twin of the F-010 load guard. rank <= 0 (or a degenerate factorization)
    // yields a NaN-scale reconstructed delta; reject it rather than training a corrupt adapter.
    if rank <= 0 {
        return Err(crate::Error::Msg(format!(
            "LoKr training: rank must be > 0, got {rank}"
        )));
    }
    let mut targets = Vec::with_capacity(target_paths.len());
    let mut params = LoraParams::new();
    // `torch.nn.init.kaiming_uniform_(a=√5)` on a 2-D `[d0, d1]` factor ⇒ `U(±1/√fan_in)` with
    // `fan_in = d1` (gain `√(2/(1+5)) = 1/√3`, bound `√3·gain/√fan_in = 1/√fan_in`).
    let kaiming = |shape: [i32; 2], key: &Array| -> Result<Array> {
        let bound = 1.0f32 / (shape[1] as f32).sqrt();
        Ok(random::uniform::<_, f32>(
            -bound,
            bound,
            &shape[..],
            Some(key),
        )?)
    };
    for (i, path) in target_paths.iter().enumerate() {
        let segs: Vec<&str> = path.split('.').collect();
        let lin = host.adaptable_mut(&segs).ok_or_else(|| -> crate::Error {
            format!("LoKr target does not resolve on the model: {path}").into()
        })?;
        let shape = lin.base_shape(); // [out, in]
        let (out_a, out_b) = factorization(shape[0], factor);
        let (in_a, in_b) = factorization(shape[1], factor);

        // w1 = zeros [out_a, in_a] — the zeroed factor, so the initial delta is exactly 0.
        let w1 = Array::zeros::<f32>(&[out_a, in_a])?;
        let w1_key: Rc<str> = Rc::from(format!("{path}.lokr_w1"));
        mlx_rs::transforms::eval([&w1])?;
        params.insert(w1_key.clone(), w1);

        let (mut w2_key, mut w2a_key, mut w2b_key) = (None, None, None);
        // PEFT `use_w2 = not(r < max(out_b,in_b)/2)`: a low-rank w2 = w2_a @ w2_b only when r is below
        // half the larger factor dim, else a full w2. Both factors are kaiming-init (the delta is held
        // at 0 by the zeroed w1, not by zeroing the second factor).
        if rank > 0 && (rank as f32) < (out_b.max(in_b) as f32) / 2.0 {
            let ka = random::key(seed.wrapping_add(7 * i as u64 + 3))?;
            let kb = random::key(seed.wrapping_add(7 * i as u64 + 5))?;
            let w2a = kaiming([out_b, rank], &ka)?; // fan_in = rank
            let w2b = kaiming([rank, in_b], &kb)?; // fan_in = in_b
            let ak: Rc<str> = Rc::from(format!("{path}.lokr_w2_a"));
            let bk: Rc<str> = Rc::from(format!("{path}.lokr_w2_b"));
            mlx_rs::transforms::eval([&w2a, &w2b])?;
            params.insert(ak.clone(), w2a);
            params.insert(bk.clone(), w2b);
            w2a_key = Some(ak);
            w2b_key = Some(bk);
        } else {
            let k2 = random::key(seed.wrapping_add(7 * i as u64 + 3))?;
            let w2 = kaiming([out_b, in_b], &k2)?; // fan_in = in_b
            let wk: Rc<str> = Rc::from(format!("{path}.lokr_w2"));
            mlx_rs::transforms::eval([&w2])?;
            params.insert(wk.clone(), w2);
            w2_key = Some(wk);
        }
        targets.push(LokrTarget {
            path: path.clone(),
            base_shape: shape,
            w1_key,
            w2_key,
            w2a_key,
            w2b_key,
        });
    }
    Ok((targets, params))
}

/// Inject each target's LoKr delta — reconstructed from the trainable factors EXACTLY as the
/// inference loader (`reconstruct_lokr_delta` at `lokr_dtype`; `Adapter::Lokr` residual `x·ΔWᵀ`) —
/// so the trained adapter round-trips. `lokr_dtype` is the dtype the inference loader reconstructs
/// at (`Bfloat16` for Z-Image/Qwen, `Float32` for SDXL). Differentiable (kron/matmul/cast traced).
pub fn install_training_lokr<H: AdaptableHost>(
    host: &mut H,
    params: &LoraParams,
    targets: &[LokrTarget],
    alpha: f32,
    rank: f32,
    lokr_dtype: Dtype,
) -> MlxResult<()> {
    for t in targets {
        let w1 = params.get(t.w1_key.as_ref());
        let w2 = t.w2_key.as_ref().and_then(|k| params.get(k.as_ref()));
        let w2a = t.w2a_key.as_ref().and_then(|k| params.get(k.as_ref()));
        let w2b = t.w2b_key.as_ref().and_then(|k| params.get(k.as_ref()));
        let delta = reconstruct_lokr_delta(
            alpha,
            rank,
            &t.base_shape,
            w1,
            None,
            None,
            w2,
            w2a,
            w2b,
            lokr_dtype,
        )
        .map_err(|e| Exception::custom(e.to_string()))?;
        let segs: Vec<&str> = t.path.split('.').collect();
        let lin = host
            .adaptable_mut(&segs)
            .ok_or_else(|| Exception::custom(format!("LoKr target not found: {}", t.path)))?;
        lin.set_adapters(vec![Adapter::Lokr { delta, scale: 1.0 }]);
    }
    Ok(())
}

/// Write the trainable LoKr factors as safetensors — `{path}.lokr_w1` + (`lokr_w2` | `lokr_w2_a` +
/// `lokr_w2_b`) — with `networkType=lokr` + `rank`/`alpha`/`decomposeFactor` metadata (the epic-2193
/// reload contract). The loaders reconstruct the delta from these shapes (the SDXL LoKr classifier
/// also accepts a `base_model.model.unet.` prefix, but the bare keys resolve directly for every
/// family, so no prefix is written).
pub fn save_lokr(
    params: &LoraParams,
    targets: &[LokrTarget],
    alpha: f32,
    rank: f32,
    decompose_factor: i32,
    path: impl AsRef<Path>,
) -> Result<()> {
    let mut entries: Vec<(String, &Array)> = Vec::with_capacity(targets.len() * 3);
    for t in targets {
        let keys = [
            Some(&t.w1_key),
            t.w2_key.as_ref(),
            t.w2a_key.as_ref(),
            t.w2b_key.as_ref(),
        ];
        for key in keys.into_iter().flatten() {
            // `.get().ok_or_else()?` over a direct index so a missing factor surfaces as a typed
            // error rather than a panic mid-checkpoint (F-008).
            let v = params
                .get(key.as_ref())
                .ok_or_else(|| crate::Error::Msg(format!("LoKr factor missing: {key}")))?;
            entries.push((key.to_string(), v));
        }
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("networkType".to_string(), "lokr".to_string());
    meta.insert("rank".to_string(), (rank as i64).to_string());
    meta.insert("alpha".to_string(), (alpha as i64).to_string());
    meta.insert("decomposeFactor".to_string(), decompose_factor.to_string());
    Array::save_safetensors(entries, Some(&meta), path)?;
    Ok(())
}

/// Sum `grads` into the running accumulator (gradient accumulation across micro-steps).
pub fn accumulate_grads(acc: &mut Option<LoraParams>, grads: LoraParams) -> Result<()> {
    use mlx_rs::ops::add;
    match acc {
        None => *acc = Some(grads),
        Some(a) => {
            for (k, g) in grads {
                let entry = a
                    .get(&k)
                    .ok_or_else(|| crate::Error::Msg(format!("grad key {k} vanished")))?;
                let summed = add(entry, &g)?;
                a.insert(k, summed);
            }
        }
    }
    Ok(())
}

/// Divide accumulated gradients by `accum` (the mean over the accumulation window).
pub fn average_grads(grads: LoraParams, accum: u32) -> Result<LoraParams> {
    if accum <= 1 {
        return Ok(grads);
    }
    let inv = Array::from_slice(&[1.0 / accum as f32], &[1]);
    let mut out = HashMap::with_capacity(grads.len());
    for (k, g) in grads {
        out.insert(k, multiply(&g, &inv)?);
    }
    Ok(out)
}

/// The trainable adapter kind — dispatches the per-step inject and the save the train loop calls,
/// so one loop drives both LoRA and LoKr. `install` takes the LoKr reconstruct dtype and `save` the
/// PEFT key prefix (the two per-family differences); both are no-ops for the other variant.
pub enum TrainAdapter {
    Lora { targets: Vec<LoraTarget> },
    Lokr { targets: Vec<LokrTarget> },
}

impl TrainAdapter {
    /// The dotted paths this adapter trains (for clearing the stack back to the bare base).
    pub fn paths(&self) -> Vec<String> {
        match self {
            TrainAdapter::Lora { targets } => targets.iter().map(|t| t.path.clone()).collect(),
            TrainAdapter::Lokr { targets } => targets.iter().map(|t| t.path.clone()).collect(),
        }
    }

    /// Number of trained target modules.
    pub fn len(&self) -> usize {
        match self {
            TrainAdapter::Lora { targets } => targets.len(),
            TrainAdapter::Lokr { targets } => targets.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn install<H: AdaptableHost>(
        &self,
        host: &mut H,
        params: &LoraParams,
        alpha: f32,
        rank: f32,
        lokr_dtype: Dtype,
    ) -> MlxResult<()> {
        self.install_as(host, params, alpha, rank, None, lokr_dtype)
    }

    /// [`install`](Self::install) with an optional LoRA compute-dtype cast (sc-4887 bf16 training);
    /// LoKr keeps its own `lokr_dtype` (the bf16-residual round-trip contract is independent).
    pub fn install_as<H: AdaptableHost>(
        &self,
        host: &mut H,
        params: &LoraParams,
        alpha: f32,
        rank: f32,
        lora_dtype: Option<Dtype>,
        lokr_dtype: Dtype,
    ) -> MlxResult<()> {
        match self {
            TrainAdapter::Lora { targets } => {
                install_training_lora_as(host, params, targets, alpha, lora_dtype)
            }
            TrainAdapter::Lokr { targets } => {
                install_training_lokr(host, params, targets, alpha, rank, lokr_dtype)
            }
        }
    }

    pub fn save(
        &self,
        params: &LoraParams,
        alpha: f32,
        rank: f32,
        decompose_factor: i32,
        key_prefix: &str,
        path: &Path,
    ) -> Result<()> {
        match self {
            TrainAdapter::Lora { targets } => {
                save_lora_peft(params, targets, alpha, rank as u32, key_prefix, path)
            }
            TrainAdapter::Lokr { targets } => {
                save_lokr(params, targets, alpha, rank, decompose_factor, path)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_lora_peft_errors_on_missing_param() {
        // F-008: a target whose factor key isn't in the (empty) params map must surface a typed
        // error, not a panic from a direct map index.
        let target = LoraTarget {
            path: "blocks.0.attn.to_q".into(),
            a_key: Rc::from("blocks.0.attn.to_q.lora_a"),
            b_key: Rc::from("blocks.0.attn.to_q.lora_b"),
            in_f: 8,
            out_f: 8,
        };
        let params: LoraParams = HashMap::new();
        let err = save_lora_peft(
            &params,
            &[target],
            1.0,
            4,
            "",
            "/tmp/unused_sc4019.safetensors",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("LoRA param missing"), "got: {err}");
    }

    #[test]
    fn factorization_balances_and_respects_factor() {
        // Balanced (factor = -1): the squarest divisor pair, a <= b.
        assert_eq!(factorization(8, -1), (2, 4));
        assert_eq!(factorization(2560, -1), (40, 64));
        // A prime dimension cannot split → (1, p).
        assert_eq!(factorization(7, -1), (1, 7));
        // Explicit factor that divides: the pair straddles `factor`.
        assert_eq!(factorization(64, 8), (8, 8));
        let (a, b) = factorization(2048, 16);
        assert_eq!(a * b, 2048);
        assert!(a <= b);
    }

    #[test]
    fn build_lokr_init_matches_peft_reset_adapter_parameters() {
        // sc-5179 — the native LoKr init must match PEFT `LoKrConfig(init_weights=True)`'s
        // `reset_adapter_parameters` (w1 zero-init, w2 kaiming-uniform a=√5), so a Python LoKr lr
        // transfers. A minimal one-Linear host stands in for a model.
        use crate::adapters::AdaptableLinear;
        struct OneLin(AdaptableLinear);
        impl AdaptableHost for OneLin {
            fn adaptable_mut(&mut self, p: &[&str]) -> Option<&mut AdaptableLinear> {
                if p == ["w"] {
                    Some(&mut self.0)
                } else {
                    None
                }
            }
            fn adaptable_paths(&self) -> Vec<String> {
                vec!["w".into()]
            }
        }
        // out=64 → fac(-1)=(8,8); in=48 → (6,8). out_b = in_b = 8. rank 3 < max(8,8)/2 = 4 → low-rank.
        let host_w = Array::zeros::<f32>(&[64, 48]).unwrap();
        let mut host = OneLin(AdaptableLinear::dense(host_w, None));
        let rank = 3;
        let (targets, params) =
            build_lokr_targets(&mut host, &["w".to_string()], rank, -1, 7).unwrap();
        assert_eq!(targets.len(), 1);
        let t = &targets[0];

        // w1 is the ZEROED factor (PEFT), shape [out_a=8, in_a=6].
        let w1 = &params[&t.w1_key];
        assert_eq!(w1.shape(), &[8, 6]);
        assert_eq!(
            w1.abs().unwrap().sum(None).unwrap().item::<f32>(),
            0.0,
            "w1 must be zero-init (PEFT reset_adapter_parameters), not N(0,0.02)"
        );

        // w2 is low-rank kaiming: no full w2; w2_a [8,3] bound 1/√3, w2_b [3,8] bound 1/√8.
        assert!(
            t.w2_key.is_none() && t.w2a_key.is_some() && t.w2b_key.is_some(),
            "rank 3 < max(8,8)/2 must take the low-rank w2 path"
        );
        let w2a = &params[t.w2a_key.as_ref().unwrap()];
        let w2b = &params[t.w2b_key.as_ref().unwrap()];
        let maxabs = |a: &Array| a.abs().unwrap().max(None).unwrap().item::<f32>();
        let bound_a = 1.0f32 / (rank as f32).sqrt(); // ≈ 0.577
        let bound_b = 1.0f32 / 8f32.sqrt(); // ≈ 0.354
                                            // Within the kaiming bound, and clearly spread (NOT the old fixed-0.02 scale, whose max would
                                            // be ~0.06 — well under bound·0.3 here).
        assert!(
            maxabs(w2a) <= bound_a + 1e-6 && maxabs(w2a) > bound_a * 0.3,
            "w2_a must be kaiming U(±1/√rank): max {} vs bound {bound_a}",
            maxabs(w2a)
        );
        assert!(
            maxabs(w2b) <= bound_b + 1e-6 && maxabs(w2b) > bound_b * 0.3,
            "w2_b must be kaiming U(±1/√in_b): max {} vs bound {bound_b}",
            maxabs(w2b)
        );

        // Initial delta is exactly 0 (w1 = 0) — the adapter starts as a no-op, like LoRA's B = 0.
        let delta = reconstruct_lokr_delta(
            1.0,
            rank as f32,
            &[64, 48],
            Some(w1),
            None,
            None,
            None,
            Some(w2a),
            Some(w2b),
            Dtype::Float32,
        )
        .unwrap();
        assert_eq!(
            delta.abs().unwrap().max(None).unwrap().item::<f32>(),
            0.0,
            "initial LoKr delta must be exactly 0"
        );
    }

    #[test]
    fn average_grads_is_identity_for_unit_accum() {
        let mut p: LoraParams = HashMap::new();
        p.insert(Rc::from("x"), Array::from_slice(&[2.0f32, 4.0], &[2]));
        let out = average_grads(p, 1).unwrap();
        assert_eq!(out["x"].as_slice::<f32>(), &[2.0, 4.0]);
    }

    #[test]
    fn average_grads_divides_by_accum() {
        let mut p: LoraParams = HashMap::new();
        p.insert(Rc::from("x"), Array::from_slice(&[2.0f32, 4.0], &[2]));
        let out = average_grads(p, 2).unwrap();
        assert_eq!(out["x"].as_slice::<f32>(), &[1.0, 2.0]);
    }

    #[test]
    fn accumulate_grads_sums_into_running_total() {
        let mk = |v: f32| {
            let mut p: LoraParams = HashMap::new();
            p.insert(Rc::from("x"), Array::from_slice(&[v], &[1]));
            p
        };
        let mut acc: Option<LoraParams> = None;
        accumulate_grads(&mut acc, mk(1.0)).unwrap();
        accumulate_grads(&mut acc, mk(2.5)).unwrap();
        let acc = acc.unwrap();
        assert_eq!(acc["x"].as_slice::<f32>(), &[3.5]);
    }
}
