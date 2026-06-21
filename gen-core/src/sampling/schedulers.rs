//! The scheduler function library (epic 7114, P1, sc-7116): pure `(ModelSampling, steps) -> sigma
//! array` builders — the σ-schedule half of the unified framework. Each is backend-neutral host math,
//! independent of prediction type, and faithfully ports ComfyUI's `comfy/samplers.py::calculate_sigmas`
//! plus the `comfy/k_diffusion` `get_sigmas_karras` / `get_sigmas_exponential`. The curated set from
//! epic decision 2: `normal`, `simple`, `karras`, `exponential`, `sgm_uniform`, `beta`, `ddim_uniform`.
//!
//! The sampler ([`super::unified::Sampler`]) integrates the latents down the returned schedule. Every
//! schedule is DESCENDING and ends with a trailing `0.0` terminal node (the per-builder note gives the
//! length). The analytic builders (`karras`/`exponential`) take an explicit `[σ_min, σ_max]`; the
//! model-native builders read [`ModelSampling::sigma_table`] / [`ModelSampling::timestep`] /
//! [`ModelSampling::sigma`]. The dispatcher derives `σ_min`/`σ_max` from the model's σ table (its
//! smallest positive / largest entry), matching ComfyUI (for flux the table's smallest positive entry
//! is the real `σ_min`, not 0).

use super::ModelSampling;

/// The curated scheduler vocabulary (epic 7114 decision 2). `Normal`/`Simple` are the model's native
/// schedule sampled two ways; `Karras`/`Exponential` are analytic σ ramps; `SgmUniform` is `Normal`
/// with the SGM endpoint convention; `Beta` biases steps toward the schedule ends; `DdimUniform` is
/// the DDIM stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheduler {
    Normal,
    Simple,
    Karras,
    Exponential,
    SgmUniform,
    Beta,
    DdimUniform,
}

impl Scheduler {
    /// Parse the canonical lowercase name (the UI / recipe vocabulary). Unknown -> `None` (callers
    /// fall back to the model default + emit an event, N3).
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "normal" => Self::Normal,
            "simple" => Self::Simple,
            "karras" => Self::Karras,
            "exponential" => Self::Exponential,
            "sgm_uniform" => Self::SgmUniform,
            "beta" => Self::Beta,
            "ddim_uniform" => Self::DdimUniform,
            _ => return None,
        })
    }

    /// The canonical lowercase name (round-trips with [`Self::from_name`]).
    pub fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Simple => "simple",
            Self::Karras => "karras",
            Self::Exponential => "exponential",
            Self::SgmUniform => "sgm_uniform",
            Self::Beta => "beta",
            Self::DdimUniform => "ddim_uniform",
        }
    }

    /// Every scheduler in the curated vocabulary, in menu order.
    pub const ALL: [Scheduler; 7] = [
        Self::Normal,
        Self::Simple,
        Self::Karras,
        Self::Exponential,
        Self::SgmUniform,
        Self::Beta,
        Self::DdimUniform,
    ];
}

/// Build the σ schedule for `steps` denoise steps under `scheduler` — ComfyUI `calculate_sigmas`. The
/// result is descending with a trailing `0.0`.
pub fn schedule_sigmas(scheduler: Scheduler, ms: &dyn ModelSampling, steps: usize) -> Vec<f32> {
    let steps = steps.max(1);
    match scheduler {
        Scheduler::Karras => {
            let (lo, hi) = table_sigma_range(ms);
            karras_sigmas(steps, lo, hi, 7.0)
        }
        Scheduler::Exponential => {
            let (lo, hi) = table_sigma_range(ms);
            exponential_sigmas(steps, lo, hi)
        }
        Scheduler::Normal => normal_sigmas(ms, steps, false),
        Scheduler::SgmUniform => normal_sigmas(ms, steps, true),
        Scheduler::Simple => simple_sigmas(ms, steps),
        Scheduler::DdimUniform => ddim_uniform_sigmas(ms, steps),
        Scheduler::Beta => beta_sigmas(ms, steps, 0.6, 0.6),
    }
}

/// The (smallest-positive, largest) entries of the model's discrete σ table — ComfyUI feeds these to
/// the analytic schedulers as `σ_min`/`σ_max` (for flux the smallest positive table entry is the real
/// `σ_min`, not the `0.0` the flow ModelSampling reports for the clean end).
fn table_sigma_range(ms: &dyn ModelSampling) -> (f32, f32) {
    let table = ms.sigma_table();
    let hi = table.last().copied().unwrap_or(1.0);
    let lo = table
        .iter()
        .copied()
        .find(|&s| s > 0.0)
        .unwrap_or((hi * 1e-3).max(1e-4));
    (lo, hi)
}

#[inline]
fn lerp(a: f32, b: f32, f: f32) -> f32 {
    a + (b - a) * f
}

// =================================================================================================
// Analytic σ ramps (prediction-independent; take explicit endpoints).
// =================================================================================================

/// Karras et al. (2022) σ schedule: `(σ_max^{1/ρ} + ramp·(σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ` over
/// `ramp = linspace(0, 1, n)`, with a trailing `0.0`. `ρ = 7` is the ComfyUI default. Length `n + 1`.
pub fn karras_sigmas(n: usize, sigma_min: f32, sigma_max: f32, rho: f32) -> Vec<f32> {
    let n = n.max(1);
    let rho = rho as f64;
    let min_inv = (sigma_min.max(0.0) as f64).powf(1.0 / rho);
    let max_inv = (sigma_max as f64).powf(1.0 / rho);
    let mut out: Vec<f32> = (0..n)
        .map(|i| {
            let ramp = if n == 1 {
                0.0
            } else {
                i as f64 / (n - 1) as f64
            };
            (max_inv + ramp * (min_inv - max_inv)).powf(rho) as f32
        })
        .collect();
    out.push(0.0);
    out
}

/// Exponential σ schedule: `exp(linspace(ln σ_max, ln σ_min, n))`, with a trailing `0.0`. Length
/// `n + 1` (geometric spacing in σ).
pub fn exponential_sigmas(n: usize, sigma_min: f32, sigma_max: f32) -> Vec<f32> {
    let n = n.max(1);
    let lmin = (sigma_min.max(1e-12) as f64).ln();
    let lmax = (sigma_max.max(1e-12) as f64).ln();
    let mut out: Vec<f32> = (0..n)
        .map(|i| {
            let f = if n == 1 {
                0.0
            } else {
                i as f64 / (n - 1) as f64
            };
            (lmax + (lmin - lmax) * f).exp() as f32
        })
        .collect();
    out.push(0.0);
    out
}

// =================================================================================================
// Model-native schedules (read the ModelSampling timestep<->sigma map / σ table).
// =================================================================================================

/// ComfyUI `normal_scheduler`: evenly-spaced timesteps between `timestep(σ_max)` and `timestep(σ_min)`,
/// mapped back through [`ModelSampling::sigma`], with a trailing `0.0`. `sgm = true` is the SGM-uniform
/// endpoint convention (`linspace(start, end, steps + 1)[:-1]`). Length `steps + 1`.
pub fn normal_sigmas(ms: &dyn ModelSampling, steps: usize, sgm: bool) -> Vec<f32> {
    let steps = steps.max(1);
    let start = ms.timestep(ms.sigma_max());
    let end = ms.timestep(ms.sigma_min());
    let timesteps: Vec<f32> = (0..steps)
        .map(|i| {
            let f = if sgm {
                // linspace(start, end, steps + 1)[:-1] -> fraction i / steps.
                i as f32 / steps as f32
            } else if steps == 1 {
                0.0
            } else {
                // linspace(start, end, steps) -> fraction i / (steps - 1).
                i as f32 / (steps - 1) as f32
            };
            lerp(start, end, f)
        })
        .collect();
    let mut sigs: Vec<f32> = timesteps.iter().map(|&t| ms.sigma(t)).collect();
    sigs.push(0.0);
    sigs
}

/// ComfyUI `simple_scheduler`: the native σ table sub-sampled by a fixed stride from the noisy end,
/// `table[-(1 + ⌊x·len/steps⌋)]` for `x in 0..steps`, with a trailing `0.0`. Length `steps + 1`.
pub fn simple_sigmas(ms: &dyn ModelSampling, steps: usize) -> Vec<f32> {
    let steps = steps.max(1);
    let table = ms.sigma_table();
    let n = table.len();
    if n == 0 {
        return vec![0.0];
    }
    let ss = n as f64 / steps as f64;
    let mut sigs: Vec<f32> = (0..steps)
        .map(|x| {
            let from_end = 1 + (x as f64 * ss) as usize;
            let idx = n.saturating_sub(from_end);
            table[idx.min(n - 1)]
        })
        .collect();
    sigs.push(0.0);
    sigs
}

/// ComfyUI `ddim_scheduler`: a uniform DDIM stride through the native σ table from index 1 upward,
/// reversed to descending, with a trailing `0.0`. The `table[1] ≈ 0` guard bumps `steps` (a near-zero
/// second node, as some schedules have). Length is `~steps + 1` (stride-dependent).
pub fn ddim_uniform_sigmas(ms: &dyn ModelSampling, steps: usize) -> Vec<f32> {
    let mut steps = steps.max(1);
    let table = ms.sigma_table();
    let n = table.len();
    if n <= 1 {
        return vec![0.0];
    }
    if table[1].abs() < 1e-5 {
        steps += 1;
    }
    let ss = (n / steps).max(1);
    let mut sigs: Vec<f32> = Vec::new();
    let mut x = 1usize;
    while x < n {
        sigs.push(table[x]);
        x += ss;
    }
    sigs.reverse();
    sigs.push(0.0);
    sigs
}

/// ComfyUI `beta_scheduler`: timesteps drawn from the inverse Beta CDF (`α = β = 0.6` default, biasing
/// toward the schedule ends), mapped to native σ-table indices with consecutive-duplicate removal, then
/// a trailing `0.0`. Length is `≤ steps + 1` (after dedup).
pub fn beta_sigmas(ms: &dyn ModelSampling, steps: usize, alpha: f64, beta: f64) -> Vec<f32> {
    let steps = steps.max(1);
    let table = ms.sigma_table();
    let n = table.len();
    if n == 0 {
        return vec![0.0];
    }
    let total = n.saturating_sub(1) as f64;
    let mut sigs: Vec<f32> = Vec::new();
    let mut last_t: i64 = -1;
    for i in 0..steps {
        // ts = 1 - linspace(0, 1, steps, endpoint=False)[i] = 1 - i/steps.
        let ts = 1.0 - i as f64 / steps as f64;
        let t = (beta_ppf(ts, alpha, beta) * total).round() as i64;
        if t != last_t {
            let idx = (t.max(0) as usize).min(n - 1);
            sigs.push(table[idx]);
        }
        last_t = t;
    }
    sigs.push(0.0);
    sigs
}

// =================================================================================================
// Inverse Beta CDF (scipy.stats.beta.ppf) — self-contained host f64, no deps. Regularized incomplete
// beta via Lanczos lnΓ + the Numerical-Recipes `betacf` continued fraction, inverted by bisection.
// =================================================================================================

/// Lanczos approximation of `ln Γ(x)` (g = 7), with the reflection formula for `x < 0.5`.
// The coefficients are the canonical published g=7 Lanczos set, kept verbatim for auditing (the
// extra digits round harmlessly into f64) — same rationale as `sampling.rs::compute_mu`.
#[allow(clippy::excessive_precision)]
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        pi.ln() - (pi * x).sin().abs().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let t = x + G + 0.5;
        let mut a = C[0];
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Continued-fraction core of the regularized incomplete beta (Numerical Recipes `betacf`, modified
/// Lentz).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const FPMIN: f64 = 1e-30;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..300 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-13 {
            break;
        }
    }
    h
}

/// The regularized incomplete beta `I_x(a, b)` (monotone increasing in `x`, `I_0 = 0`, `I_1 = 1`).
fn reg_inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Inverse Beta CDF: the `x ∈ [0, 1]` with `I_x(a, b) = p` (scipy `beta.ppf`), by bisection on the
/// monotone `reg_inc_beta`.
fn beta_ppf(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if reg_inc_beta(a, b, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{
        AlphaSchedule, DiscreteModelSampling, EdmModelSampling, FlowModelSampling,
        TimestepConvention,
    };

    fn sdxl() -> DiscreteModelSampling {
        DiscreteModelSampling::sdxl(&AlphaSchedule::scaled_linear(1000, 0.00085, 0.012).unwrap())
    }

    fn is_descending_to_zero(sigs: &[f32]) -> bool {
        sigs.len() >= 2
            && *sigs.last().unwrap() == 0.0
            && sigs.windows(2).all(|w| w[0] >= w[1])
            && sigs[..sigs.len() - 1].iter().all(|&s| s > 0.0)
    }

    #[test]
    fn karras_endpoints_and_shape() {
        let s = karras_sigmas(5, 0.1, 10.0, 7.0);
        assert_eq!(s.len(), 6); // n + 1
        assert!((s[0] - 10.0).abs() < 1e-4, "first {}", s[0]);
        assert!((s[4] - 0.1).abs() < 1e-4, "last-nonzero {}", s[4]);
        assert_eq!(s[5], 0.0);
        assert!(is_descending_to_zero(&s));
    }

    #[test]
    fn exponential_is_geometric() {
        // linspace(ln10, ln0.1, 3).exp() = [10, 1, 0.1], + trailing 0.
        let s = exponential_sigmas(3, 0.1, 10.0);
        assert_eq!(s.len(), 4);
        for (g, w) in s.iter().zip([10.0_f32, 1.0, 0.1, 0.0]) {
            assert!((g - w).abs() < 1e-4, "got {g} want {w}");
        }
    }

    #[test]
    fn normal_and_sgm_shapes_and_monotonic() {
        let ms = sdxl();
        let normal = normal_sigmas(&ms, 10, false);
        let sgm = normal_sigmas(&ms, 10, true);
        assert_eq!(normal.len(), 11);
        assert_eq!(sgm.len(), 11);
        assert!(is_descending_to_zero(&normal), "normal {normal:?}");
        assert!(is_descending_to_zero(&sgm), "sgm {sgm:?}");
        // normal starts at the model's max sigma; sgm omits the very first endpoint -> starts lower.
        assert!(normal[0] >= sgm[0]);
    }

    #[test]
    fn simple_shape_and_monotonic() {
        let ms = sdxl();
        let s = simple_sigmas(&ms, 8);
        assert_eq!(s.len(), 9);
        assert!(is_descending_to_zero(&s), "{s:?}");
        // First sampled sigma is the noisy-end of the table.
        assert!((s[0] - ms.sigma_max()).abs() / ms.sigma_max() < 1e-3);
    }

    #[test]
    fn ddim_and_beta_monotonic_to_zero() {
        let ms = sdxl();
        let ddim = ddim_uniform_sigmas(&ms, 10);
        let beta = beta_sigmas(&ms, 10, 0.6, 0.6);
        assert!(is_descending_to_zero(&ddim), "ddim {ddim:?}");
        assert!(is_descending_to_zero(&beta), "beta {beta:?}");
        assert!(*beta.first().unwrap() <= ms.sigma_max() * 1.0001);
    }

    #[test]
    fn beta_ppf_matches_known_values() {
        // Symmetric Beta(0.6, 0.6): median at 0.5; CDF round-trips its own ppf.
        assert!((beta_ppf(0.5, 0.6, 0.6) - 0.5).abs() < 1e-6);
        for &p in &[0.1_f64, 0.3, 0.7, 0.9] {
            let x = beta_ppf(p, 0.6, 0.6);
            assert!((reg_inc_beta(0.6, 0.6, x) - p).abs() < 1e-5, "p={p} x={x}");
        }
        // Beta(1,1) is uniform: ppf(p) == p.
        assert!((beta_ppf(0.42, 1.0, 1.0) - 0.42).abs() < 1e-5);
    }

    #[test]
    fn dispatcher_covers_all_schedulers_on_every_model_sampling() {
        let sdxl = sdxl();
        let flow = FlowModelSampling::new(TimestepConvention::Sigma);
        let edm = EdmModelSampling::svd();
        let models: [&dyn ModelSampling; 3] = [&sdxl, &flow, &edm];
        for ms in models {
            for sched in Scheduler::ALL {
                let sigs = schedule_sigmas(sched, ms, 12);
                assert!(
                    is_descending_to_zero(&sigs),
                    "{} produced non-monotone/zero-terminated schedule: {sigs:?}",
                    sched.name()
                );
            }
        }
    }

    #[test]
    fn scheduler_name_roundtrip() {
        for s in Scheduler::ALL {
            assert_eq!(Scheduler::from_name(s.name()), Some(s));
        }
        assert_eq!(Scheduler::from_name("nope"), None);
    }
}
