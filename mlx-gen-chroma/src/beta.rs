//! Beta-distribution sigma schedule for `chroma1_base` (sc-3840).
//!
//! Base's `scheduler_config.json` sets `use_beta_sigmas=true` ("Beta Sampling is All You Need",
//! 2407.12173): the diffusers `FlowMatchEulerDiscreteScheduler._convert_to_beta` maps the linspace
//! sigma range through the inverse beta CDF (α=β=0.6):
//! `σ_i = σ_min + betappf(1 − i/(N−1), 0.6, 0.6) · (σ_max − σ_min)`, with `σ_max=1`, `σ_min=1/N`
//! (the static shift is 1.0 for Base ⇒ identity before the beta conversion).
//!
//! `betappf` is the inverse regularized incomplete beta `I_x⁻¹(0.6,0.6)`, computed by bisection on
//! `I_x` (Lentz continued fraction). α=β=0.6 is fixed, so the log-gamma prefactor is a constant.

/// `lgamma(α+β) − lgamma(α) − lgamma(β)` for α=β=0.6 (the `I_x` prefactor exponent constant).
const LGAMMA_AB: f64 = -0.8818418061417858;
const A: f64 = 0.6;

/// Lentz continued fraction for the incomplete beta, with a=b=`A`.
fn betacf(x: f64) -> f64 {
    let (a, b) = (A, A);
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let tiny = 1e-30;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..300 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
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

/// Regularized incomplete beta `I_x(0.6, 0.6)`.
fn beta_cdf(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (LGAMMA_AB + A * x.ln() + A * (1.0 - x).ln()).exp();
    // symmetric (a=b): pick the faster-converging tail.
    if x < (A + 1.0) / (2.0 * A + 2.0) {
        bt * betacf(x) / A
    } else {
        1.0 - bt * betacf(1.0 - x) / A
    }
}

/// Inverse regularized incomplete beta `I_p⁻¹(0.6, 0.6)` via bisection (monotone CDF).
fn beta_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if beta_cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Base's beta sigma schedule for `steps` denoise steps: length `steps + 1` with a trailing `0.0`.
pub fn base_sigmas(steps: usize) -> Vec<f32> {
    let n = steps.max(1);
    let smin = 1.0 / n as f64;
    let smax = 1.0;
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        let lin = if n == 1 {
            0.0
        } else {
            i as f64 / (n - 1) as f64
        };
        let ppf = beta_ppf(1.0 - lin);
        out.push((smin + ppf * (smax - smin)) as f32);
    }
    out.push(0.0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_ppf_matches_scipy() {
        // scipy.stats.beta.ppf(p, 0.6, 0.6)
        let cases = [
            (0.05, 0.01255446),
            (0.25, 0.17568038),
            (0.5, 0.5),
            (0.75, 0.82431962),
            (0.95, 0.98744554),
        ];
        for (p, want) in cases {
            let got = beta_ppf(p);
            assert!((got - want).abs() < 1e-6, "ppf({p}) = {got}, want {want}");
        }
    }

    #[test]
    fn base_sigmas_match_diffusers() {
        // diffusers FlowMatchEulerDiscreteScheduler(use_beta_sigmas=true), sigmas=linspace(1,1/N,N).
        let s4 = base_sigmas(4);
        let want4 = [1.0, 0.79344, 0.45656, 0.25, 0.0];
        for (g, w) in s4.iter().zip(want4) {
            assert!((g - w).abs() < 1e-4, "N=4: {g} vs {w}");
        }
        let s8 = base_sigmas(8);
        let want8 = [
            1.0, 0.937751, 0.810255, 0.648749, 0.476251, 0.314745, 0.187249, 0.125, 0.0,
        ];
        for (g, w) in s8.iter().zip(want8) {
            assert!((g - w).abs() < 1e-4, "N=8: {g} vs {w}");
        }
    }
}
