//! sc-3072 — PerceiverAttentionCA + FLUX-injection-schedule parity vs the torch f32 reference.
//!
//! Golden: `tools/dump_pulid_ca_golden.py`. Driving the goldens through the `PulidCa` injector
//! validates, in one shot, both the CA cross-attn math and the shared double→single ca_idx schedule
//! (double block i → ca[i/2]; single block i → ca[10 + i/4]). Plus: the index gate (Some/None at the
//! right blocks) and the id_weight=0 ⇒ None property (⇒ `forward_injected` is bit-identical to plain
//! FLUX).
//!
//! Run:
//!   cargo test -p mlx-gen-pulid --release --test pulid_ca_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_flux::transformer::DitImageInjector;
use mlx_gen_pulid::ca::PulidCa;
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/pulid_ca_golden.safetensors"
);

fn golden() -> Weights {
    Weights::from_file(GOLDEN).unwrap_or_else(|e| {
        panic!("missing {GOLDEN}: {e}\nRun tools/dump_pulid_ca_golden.py first (see file header).")
    })
}

fn slice(a: &Array) -> Vec<f32> {
    let n: i32 = a.shape().iter().product();
    a.reshape(&[n])
        .unwrap()
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn cosine(got: &Array, want: &Array) -> f32 {
    let (a, b) = (slice(got), slice(want));
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(&b) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn mean_rel(got: &Array, want: &Array) -> f32 {
    let (a, b) = (slice(got), slice(want));
    let sum_ref: f64 = b.iter().map(|&v| v.abs() as f64).sum();
    let sum_diff: f64 = a.iter().zip(&b).map(|(&x, &y)| (x - y).abs() as f64).sum();
    (sum_diff / sum_ref) as f32
}

fn build(id_weight: f32) -> (Weights, PulidCa, Array) {
    let g = golden();
    let id_embedding = g.require("id_embedding").unwrap().clone();
    let img = g.require("img").unwrap().clone();
    let ca = PulidCa::from_weights(&g, "pulid_ca", id_embedding, id_weight, 19, 38).unwrap();
    (g, ca, img)
}

#[test]
#[ignore = "needs local golden from tools/dump_pulid_ca_golden.py"]
fn ca_forward_and_schedule_match_torch_f32() {
    let (g, ca, img) = build(1.0);
    assert_eq!(ca.num_ca(), 20, "expected 20 CA modules");

    // (block kind, block idx, expected ca module idx)
    let cases = [
        ("double", 0usize, 0),
        ("double", 18, 9),
        ("single", 0, 10),
        ("single", 36, 19),
    ];
    for (kind, blk, ca_idx) in cases {
        let got = match kind {
            "double" => ca.after_double(blk, &img).unwrap(),
            _ => ca.after_single(blk, &img).unwrap(),
        }
        .unwrap_or_else(|| panic!("{kind} block {blk} should inject (ca[{ca_idx}])"));
        let want = g.require(&format!("ca_out_{ca_idx}")).unwrap();
        let cos = cosine(&got, want);
        let mr = mean_rel(&got, want);
        println!("{kind} blk {blk} -> ca[{ca_idx}]: cos {cos:.6} mean-rel {mr:.3e}");
        assert!(
            cos > 0.9999,
            "{kind} blk {blk} ca[{ca_idx}] cosine {cos:.6}"
        );
        assert!(mr < 5e-3, "{kind} blk {blk} ca[{ca_idx}] mean-rel {mr:.3e}");
    }
}

#[test]
#[ignore = "needs local golden from tools/dump_pulid_ca_golden.py"]
fn injection_index_gate() {
    let (_g, ca, img) = build(1.0);
    // double: inject at even block indices only
    for i in 0..19 {
        let some = ca.after_double(i, &img).unwrap().is_some();
        assert_eq!(some, i % 2 == 0, "double block {i} injection gate");
    }
    // single: inject at every 4th block only
    for i in 0..38 {
        assert_eq!(
            ca.injects_after_single(i),
            i % 4 == 0,
            "single block {i} gate"
        );
        let some = ca.after_single(i, &img).unwrap().is_some();
        assert_eq!(some, i % 4 == 0, "single block {i} injection");
    }
}

#[test]
#[ignore = "needs local golden from tools/dump_pulid_ca_golden.py"]
fn id_weight_zero_is_noop() {
    // id_weight = 0 ⇒ every hook returns None ⇒ forward_injected is bit-identical to plain FLUX.
    let (_g, ca, img) = build(0.0);
    for i in 0..19 {
        assert!(
            ca.after_double(i, &img).unwrap().is_none(),
            "double {i} must be None at id_weight=0"
        );
    }
    for i in 0..38 {
        assert!(
            !ca.injects_after_single(i),
            "single {i} gate must be false at id_weight=0"
        );
        assert!(
            ca.after_single(i, &img).unwrap().is_none(),
            "single {i} must be None at id_weight=0"
        );
    }
}
