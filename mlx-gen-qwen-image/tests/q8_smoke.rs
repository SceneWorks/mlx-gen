//! Diagnostic: characterize Q8 forward error across transformer Linear shapes + activation dtypes.
use mlx_gen::adapters::AdaptableLinear;
use mlx_rs::{random, Array, Dtype};

fn rel(a: &Array, b: &Array) -> f64 {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let num: f64 = a.iter().zip(b).map(|(x, y)| (*x - *y).abs() as f64).sum();
    let den: f64 = b.iter().map(|y| y.abs() as f64).sum();
    num / den
}

#[test]
fn q8_forward_close_to_dense() {
    let key = random::key(0).unwrap();
    // (out, in) shapes spanning the transformer's Linears.
    let shapes = [
        (3072, 64),
        (3072, 128),
        (3072, 3072),
        (3072, 3584),
        (64, 3072),
    ];
    let mut worst = 0.0f64;
    for (out, inn) in shapes {
        let w = random::normal::<f32>(&[out, inn], None, None, Some(&key))
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let b = Array::from_slice(&vec![0.1f32; out as usize], &[out])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let x32 = random::normal::<f32>(&[1, 4, inn], None, None, Some(&key)).unwrap();
        let x16 = x32.as_dtype(Dtype::Bfloat16).unwrap();

        let mut lin = AdaptableLinear::dense(w, Some(b));
        let d32 = lin.forward(&x32).unwrap();
        lin.quantize(8, None).unwrap();
        let q32 = lin.forward(&x32).unwrap();
        let q16 = lin.forward(&x16).unwrap();
        // Both quantized outputs are compared to the f32 dense ground truth. The quantized forward
        // now feeds activations to `quantized_matmul` as-is (the bf16→f32 upcast was removed in
        // sc-2719), so q16 carries bf16-rounded inputs and need not match q32 exactly — but both
        // must stay within tolerance of the dense ground truth (`quantized_matmul` accumulates fp32,
        // correct at every activation dtype). `q16-vs-q32` is printed to characterize that gap.
        let r32 = rel(&q32, &d32);
        let r16 = rel(&q16, &d32);
        let r1632 = rel(&q16, &q32);
        println!(
            "[out={out}, in={inn}] q8(f32)={r32:.4}  q8(bf16)={r16:.4}  q16-vs-q32={r1632:.4}"
        );
        worst = worst.max(r32).max(r16);
    }
    assert!(
        worst < 0.05,
        "some Q8 shape/dtype diverged: worst {worst:.4}"
    );
}
