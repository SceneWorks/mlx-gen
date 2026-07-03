//! Bernini renderer knobs (the `bernini_renderer.json` sidecar emitted by the sc-4705 converter) +
//! task/guidance-mode resolution + the CLI-default guidance scalars.

use std::path::Path;

use mlx_gen::{Error, Result};

use crate::forward::Mode;

/// Read a knob sidecar JSON, distinguishing "absent" from "corrupt" (F-097). An **absent** file is a
/// valid state — the loader falls back to the built-in defaults — and returns `Value::Null`. A
/// **present but unreadable/invalid** file is a real error: reverting ALL knobs to defaults silently
/// (the old `.ok().and_then(..ok())` behaviour) is worse than surfacing the corruption, so this
/// returns `Err` instead of pretending the sidecar was absent.
pub(crate) fn read_knob_sidecar(path: &Path) -> Result<serde_json::Value> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| {
            Error::Msg(format!(
                "bernini: {} is present but not valid JSON: {e}",
                path.display()
            ))
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(serde_json::Value::Null),
        Err(e) => Err(Error::Msg(format!(
            "bernini: reading {}: {e}",
            path.display()
        ))),
    }
}

/// Bernini renderer inference knobs, read from the converter's `bernini_renderer.json` sidecar (else
/// the upstream `BerniniRendererConfig` defaults).
#[derive(Clone, Debug)]
pub struct BerniniKnobs {
    /// High→low expert switch boundary (× `num_train_timesteps`).
    pub switch_dit_boundary: f32,
    /// UniPC flow-shift (the reference builds the scheduler with `flow_shift = config.shift`).
    pub shift: f32,
    // NOTE: the reference `use_src_id_rotary_emb` toggle is NOT carried here — this port applies the
    // source-id rotary embedding UNCONDITIONALLY (see `rope.rs` / `forward.rs::apply_source_id`),
    // matching the Bernini renderer's shipped config (always `true`). Parsing it into a field the
    // runtime then ignored would advertise a toggle that does nothing, so it is intentionally dropped.
    pub interpolate_src_id: bool,
    pub max_trained_src_id: f64,
    pub max_sequence_length: usize,
}

impl Default for BerniniKnobs {
    fn default() -> Self {
        Self {
            switch_dit_boundary: 0.875,
            shift: 3.0,
            interpolate_src_id: true,
            max_trained_src_id: 5.0,
            max_sequence_length: 512,
        }
    }
}

impl BerniniKnobs {
    /// Read `<root>/bernini_renderer.json`; any missing field falls back to the default. An absent
    /// sidecar yields all-defaults; a **present-but-corrupt** sidecar is an error (F-097), not a
    /// silent revert of every knob to its default.
    pub fn from_dir(root: &Path) -> Result<Self> {
        let d = Self::default();
        let v = read_knob_sidecar(&root.join("bernini_renderer.json"))?;
        let f = |k: &str, dv: f32| {
            v.get(k)
                .and_then(serde_json::Value::as_f64)
                .map(|x| x as f32)
                .unwrap_or(dv)
        };
        let b = |k: &str, dv: bool| v.get(k).and_then(serde_json::Value::as_bool).unwrap_or(dv);
        let i = |k: &str, dv: i64| v.get(k).and_then(serde_json::Value::as_i64).unwrap_or(dv);
        Ok(Self {
            switch_dit_boundary: f("switch_dit_boundary", d.switch_dit_boundary),
            shift: f("shift", d.shift),
            interpolate_src_id: b("interpolate_src_id", d.interpolate_src_id),
            max_trained_src_id: f("max_trained_src_id", d.max_trained_src_id as f32) as f64,
            // Clamp to >=0 before the usize cast: a negative `max_sequence_length` in JSON would wrap
            // to a huge usize and drive an unbounded allocation downstream (F-080).
            max_sequence_length: i("max_sequence_length", d.max_sequence_length as i64).max(0)
                as usize,
        })
    }
}

/// CLI/gradio default guidance scalars (`bernini/cli.py add_common_args` + `run_*.sh`). A request's
/// `guidance` overrides `omega_txt`; the rest are fixed defaults until the worker surfaces them.
pub struct Defaults;
impl Defaults {
    pub const STEPS: usize = 40;
    pub const NUM_FRAMES: usize = 81;
    pub const OMEGA_VID: f32 = 1.25;
    pub const OMEGA_IMG: f32 = 4.5;
    pub const OMEGA_TXT: f32 = 4.0;
    pub const OMEGA_SCALE: f32 = 0.8;
    pub const ETA: f32 = 0.5;
    pub const MOMENTUM: f32 = 0.0;
    pub const NORM_THRESHOLD: f32 = 50.0;
}

/// Resolve the guidance [`Mode`] from the request's `video_mode` (a renderer **guidance mode** name
/// preferred, else a **task_type** name) plus which conditioning is present. With no `video_mode`,
/// default by conditioning: video+images ⇒ `rv2v`, video ⇒ `v2v_apg`, images ⇒ `v2v`, none ⇒ `t2v_apg`.
pub fn resolve_mode(video_mode: Option<&str>, has_video: bool, has_image: bool) -> Mode {
    if let Some(s) = video_mode {
        if let Some(m) = Mode::from_name(s) {
            return m;
        }
        if let Some(m) = task_to_mode(s) {
            return m;
        }
    }
    match (has_video, has_image) {
        (true, true) => Mode::Rv2v,
        (true, false) => Mode::V2vApg,
        (false, true) => Mode::V2v,
        (false, false) => Mode::T2vApg,
    }
}

/// The upstream task_type → guidance_mode table (`gradio_demo.py` RENDERER_TASK_DEFAULTS). Used as a
/// fallback when `video_mode` is a task name rather than a guidance-mode name.
fn task_to_mode(task: &str) -> Option<Mode> {
    Some(match task {
        "t2i" | "t2v" => Mode::T2vApg,
        "i2i" => Mode::V2v,
        "v2v" | "mv2v" | "ads2v" => Mode::V2vApg,
        "r2v" => Mode::R2vApg,
        "rv2v" => Mode::Rv2v,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_resolution_prefers_guidance_then_task_then_conditioning() {
        // Explicit guidance-mode name.
        assert_eq!(resolve_mode(Some("rv2v"), false, false), Mode::Rv2v);
        assert_eq!(resolve_mode(Some("t2v_apg"), false, false), Mode::T2vApg);
        // Task name fallback (t2i/t2v → t2v_apg, r2v → r2v_apg).
        assert_eq!(resolve_mode(Some("t2i"), false, false), Mode::T2vApg);
        assert_eq!(resolve_mode(Some("r2v"), false, true), Mode::R2vApg);
        // Conditioning-driven defaults.
        assert_eq!(resolve_mode(None, false, false), Mode::T2vApg);
        assert_eq!(resolve_mode(None, true, false), Mode::V2vApg);
        assert_eq!(resolve_mode(None, false, true), Mode::V2v);
        assert_eq!(resolve_mode(None, true, true), Mode::Rv2v);
        // "t2v" as a guidance-mode name is the plain mode (from_name wins over the task table).
        assert_eq!(resolve_mode(Some("t2v"), false, false), Mode::T2v);
    }

    #[test]
    fn knobs_default_when_sidecar_missing() {
        // An ABSENT sidecar is fine (defaults), not an error (F-097).
        let k = BerniniKnobs::from_dir(Path::new("/nonexistent")).unwrap();
        assert_eq!(k.switch_dit_boundary, 0.875);
        assert_eq!(k.shift, 3.0);
        assert_eq!(k.max_trained_src_id, 5.0);
    }

    #[test]
    fn corrupt_sidecar_is_an_error_not_a_silent_revert() {
        // A PRESENT-but-corrupt sidecar must surface (F-097) rather than reverting every knob to
        // default. Write a garbage file into a temp dir and confirm `from_dir` errors.
        let dir = std::env::temp_dir().join(format!("bernini_knobs_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("bernini_renderer.json"), b"{ not valid json").unwrap();
        let r = BerniniKnobs::from_dir(&dir);
        std::fs::remove_dir_all(&dir).ok();
        assert!(r.is_err(), "corrupt sidecar must be an error");
    }
}
