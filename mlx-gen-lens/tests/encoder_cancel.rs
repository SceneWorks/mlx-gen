//! sc-11128 F-029 — the lens gpt-oss MoE encode must honor a cancel that arrives *during* the
//! encode, not only one already tripped at entry.
//!
//! The F-019 remediation added a per-layer `is_cancelled()` check but, under lazy MLX execution
//! (after sc-9500 moved MoE routing on-device), nothing forced a host sync per layer — so all ~24
//! checks executed in microseconds while the graph was *built*, and the entire 20B compute then ran
//! in one uninterruptible `eval`. A cancel arriving during that `eval` was never observed: a
//! false green. F-029 forces `eval([&hidden])` after each layer when a cancel handle is present, so
//! the per-layer check observes real progress and a mid-encode cancel is honored within one layer.
//!
//! This test would **pass against the buggy graph-time-only code only for the already-tripped case**
//! (which both honor); the concurrent mid-encode case below returns `Ok` under the buggy code (the
//! cancel lands during the single monolithic `eval`) and `Err(Canceled)` under the fix.
//!
//! Real-weight + Metal + timing, so `#[ignore]`d. Run:
//! `cargo test -p mlx-gen-lens --test encoder_cancel -- --ignored --nocapture`

use std::sync::Arc;
use std::time::{Duration, Instant};

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_lens::config::GptOssConfig;
use mlx_gen_lens::text_encoder::encoder::LensTextEncoder;

fn text_encoder_dir() -> std::path::PathBuf {
    let base = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots");
    let snap = std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("snapshot dir {}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max()
        .expect("a snapshot");
    snap.join("text_encoder")
}

#[test]
#[ignore = "needs the 12GB Lens-Turbo text_encoder snapshot (~40GB bf16 load) + Metal"]
fn mid_encode_cancel_is_honored_not_false_green() {
    let cfg = GptOssConfig::lens();
    eprintln!("loading text_encoder weights (3 MXFP4 shards → bf16 dequant)…");
    let w = Weights::from_dir(text_encoder_dir()).expect("load text_encoder shards");
    let encoder = LensTextEncoder::from_weights(w, &cfg, Dtype::Bfloat16).expect("build encoder");

    // A non-trivial sequence so the ~24-layer encode is comfortably longer than the cancel delay.
    let l = 256i32;
    let ids = Array::from_slice(&vec![1i32; l as usize], &[1, l]);

    // Sanity: an already-tripped cancel bails immediately (the entry / first per-layer check).
    let pre = CancelFlag::new();
    pre.cancel();
    assert!(
        matches!(
            encoder.encode(&ids, Some(&pre)),
            Err(mlx_gen::Error::Canceled)
        ),
        "already-cancelled encode must return Error::Canceled"
    );

    // The real F-029 guard: trip the flag from another thread AFTER the encode has started. Under the
    // fixed per-layer `eval`, the next layer's check observes the trip → Canceled well before the full
    // encode completes. Under the buggy graph-time-only code, the whole graph builds in microseconds
    // and the cancel lands during the single `eval`, which has no further checks → the encode runs to
    // completion and returns Ok (this assertion then fails, which is the point).
    let flag = CancelFlag::new();
    let flag_bg = flag.clone();
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done_bg = done.clone();
    let canceller = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(40));
        if !done_bg.load(std::sync::atomic::Ordering::Relaxed) {
            flag_bg.cancel();
        }
    });

    let t0 = Instant::now();
    let res = encoder.encode(&ids, Some(&flag));
    let elapsed = t0.elapsed();
    done.store(true, std::sync::atomic::Ordering::Relaxed);
    canceller.join().unwrap();

    eprintln!("mid-encode cancel returned {res:?} after {elapsed:?}");
    assert!(
        matches!(res, Err(mlx_gen::Error::Canceled)),
        "a cancel issued mid-encode must be honored (Error::Canceled), not swallowed by the lazy \
         single-eval false green (F-029); got {res:?} after {elapsed:?}"
    );
}
