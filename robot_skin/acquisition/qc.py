"""Session quality control: stream timing, pressure health, IMU sanity, cameras, labels, sync.

``session_qc(session_dir)`` reads a raw session (manifest + files + events) and returns a JSON-able
report with per-stream statistics and a list of **checks** (``severity`` ``error`` | ``warning``).
The session passes when no *error* check fails. Gates (defaults in :data:`DEFAULT_THRESHOLDS`,
override per call / CLI ``--set key=value``):

- recorder: no source errors, no source without samples (``meta.recorder``; error);
- timing (every stream): measured rate within ±10 % of nominal, dropped samples ≤ 2 % (cameras
  5 %), no gap longer than 0.5 s, monotonic timestamps, the stream covers every recorded phase
  (± 0.5 s; error, warning for offline ``hand_pose`` / ``object_pose``); jitter is reported;
- pressure (only the raw channels the layout maps to taxels — unconnected board channels are
  ignored; the board must have every layout channel, error): saturation inside ``no_contact``
  segments ≤ 0.5 % (error) and overall ≤ 10 %
  (warning); **baseline drift** between the first and last *rest-pose* baseline block (static
  ``no_contact`` phases: D1 ``baseline_start`` → ``baseline_end``; falls back to the first/last
  ``no_contact`` segment for sessions without step kinds) ≤ 3 % (per channel, median raw) — other
  ``no_contact`` blocks are in other poses, and their bending artefact is not drift; dead / stuck
  channels; ΔS noise at rest (reported);
- IMU: quaternion norm error ≤ 0.02, finite values; glove sessions need the IMU calibration in
  ``manifest.calibration`` and a still calibration block (warning);
- cameras: frame count = timestamp count; fps / drops as above;
- hand_pose (if present): fraction of labels with confidence ≥ 0.5 ≥ 0.7 (warning);
- labels: D1 (``motion``) needs ≥ 1 ``no_contact`` segment (error), calibration / self_touch
  (warning); D2 (``task``) needs ``task.task_id`` + ``instruction`` (error) and a success verdict
  (warning); no auto-closed phases (warning);
- sync: if sync blocks exist, every synced stream should have score ≥ 0.3 (warning) and
  |offset| ≤ 0.5 s (error); the 3 taps must be visible in the pressure stream (warning).

CLI: ``python -m robot_skin.acquisition.qc <session_dir> [...] [--write] [--json out.json]``
(exit code 1 if any session fails; ``--strict`` also fails on warnings).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from common.signal import estimate_baseline, relative_change, saturation_mask

from .manifest import SessionManifest
from .recorder import _labels_of, load_events, phases_from_events

__all__ = ["DEFAULT_THRESHOLDS", "QC_NAME", "format_report", "main", "rest_spans", "session_qc", "stream_timing"]

QC_NAME = "qc.json"
DEFAULT_THRESHOLDS: dict[str, float] = {
    "rate_tol": 0.10,
    "gap_factor": 5.0,
    "max_gap_s": 0.5,
    "max_dropped_pct": 2.0,
    "max_dropped_pct_camera": 5.0,
    "max_sat_pct_no_contact": 0.5,
    "max_sat_pct": 10.0,
    "max_baseline_drift_pct": 3.0,
    "max_quat_norm_err": 0.02,
    "hand_min_conf": 0.5,
    "min_hand_coverage": 0.7,
    "min_sync_score": 0.3,
    "max_sync_offset_s": 0.5,
    "max_calib_gyro_rms": 0.15,
    "max_calib_spread_deg": 3.0,
    "min_duration_s": 1.0,
    "sat_max_abs_pct": 90.0,
    "sync_taps": 3,
    "coverage_tol_s": 0.5,
}


class _Checks:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, name: str, passed: bool, *, value: Any = None, limit: Any = None, severity: str = "error",
            stream: str | None = None, message: str = "") -> None:
        self.items.append({"name": name, "stream": stream, "passed": bool(passed), "severity": severity,
                           "value": _num(value), "limit": _num(limit), "message": message})


def _num(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def stream_timing(t, nominal_hz: float | None = None, *, gap_factor: float = 5.0) -> dict:
    """Timing statistics of one timestamp vector: rate, jitter, gaps and dropped samples.

    Dropped samples are counted from the intervals: an interval of ``k`` nominal periods
    (``dt > 1.5/rate``) hides ``round(dt·rate) − 1`` samples.
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    n = int(t.size)
    out: dict[str, Any] = {"n": n, "nominal_hz": nominal_hz}
    if n < 2:
        out.update(duration_s=0.0, rate_hz=None, monotonic=True, jitter_ms=None, max_gap_s=None, n_gaps=0,
                   dropped_pct=None)
        return out
    dt = np.diff(t)
    dur = float(t[-1] - t[0])
    rate = (n - 1) / dur if dur > 0 else None
    ref = float(nominal_hz) if nominal_hz else (1.0 / float(np.median(dt)) if np.median(dt) > 0 else None)
    out.update(duration_s=dur, rate_hz=rate, monotonic=bool(np.all(dt > 0)),
               jitter_ms=float(np.std(dt) * 1e3), max_gap_s=float(dt.max()))
    if ref:
        gaps = dt > gap_factor / ref
        miss = np.where(dt > 1.5 / ref, np.round(dt * ref) - 1.0, 0.0).clip(min=0).sum()
        out["n_gaps"] = int(gaps.sum())
        out["dropped_pct"] = float(100.0 * miss / (n + miss))
    else:
        out["n_gaps"], out["dropped_pct"] = 0, None
    return out


def _mask_spans(t: np.ndarray, spans: Sequence[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(t.shape, dtype=bool)
    for a, b in spans:
        m |= (t >= a) & (t <= b)
    return m


def rest_spans(phases: Sequence[Mapping]) -> list[tuple[float, float]] | None:
    """Rest-pose baseline blocks: phases of step kind ``static`` labelled ``no_contact`` (D1
    ``baseline_start``/``baseline_end``, D2 ``baseline``), sorted by time. ``None`` when the events
    carry no step kinds (sessions not recorded from a protocol) — callers then fall back to the
    ``no_contact`` segments.

    Baseline drift must be measured between blocks of the **same pose**: other ``no_contact``
    blocks (flat-hand calibration, finger flexion …) differ by the pose-dependent bending
    artefact, which is exactly what the baseline predictor has to learn, not drift.
    """
    vals = [p for p in phases if isinstance(p.get("value"), Mapping)]
    if not any("kind" in p["value"] for p in vals):
        return None
    return sorted((float(p["t0"]), float(p["t1"])) for p in vals
                  if p["value"].get("kind") == "static" and "no_contact" in _labels_of(p["value"]))


def _layout_channels(layout: str) -> np.ndarray | None:
    """Sorted raw channel indices used by the session's layout (``None`` if it cannot be loaded)."""
    try:
        from common.layouts import load_layout
        ch = np.unique(np.asarray(load_layout(layout).channels, dtype=np.int64))
    except Exception:                               # unknown / moved layout file: check all channels
        return None
    return ch if ch.size else None


def _pressure_qc(d: Path, m: SessionManifest, name: str, th: Mapping, C: _Checks,
                 phases: Sequence[Mapping] = ()) -> dict:
    with np.load(m.stream_path(d, name)) as z:
        t, raw_all = z["t"], np.asarray(z["raw"], dtype=np.float64)
    C_board = int(raw_all.shape[1])
    out: dict[str, Any] = {"n_channels": C_board}
    # only the channels the layout maps to taxels are checked: a board may have unconnected
    # channels (constant or on a rail) that must not fail the session
    used = _layout_channels(m.layout)
    if used is not None:
        C.add("pressure_channels_layout", int(used.max()) < C_board, value=C_board, limit=int(used.max()) + 1,
              stream=name, message=f"layout {m.layout!r} maps taxels to channels up to {int(used.max())}")
        used = used[used < C_board]
    cols = used if used is not None and used.size else np.arange(C_board)
    out["checked_channels"] = cols.tolist()
    raw = raw_all[:, cols]
    nc = sorted(m.spans("no_contact"))
    if nc:
        first = (t >= nc[0][0]) & (t <= nc[0][1])
        base = np.median(raw[first], axis=0) if first.any() else estimate_baseline(raw, t=t, duration_s=1.0)
    else:
        base = estimate_baseline(raw, t=t, duration_s=1.0)
    stuck = cols[raw.std(0) == 0]
    out["stuck_channels"] = stuck.tolist()                     # board channel numbers
    C.add("pressure_channels_alive", stuck.size == 0, value=stuck.tolist(), stream=name,
          message="channels with zero variance (disconnected / stuck)")
    safe = np.where(base == 0, 1.0, base)
    delta = relative_change(raw, safe)
    sat = saturation_mask(raw, delta, max_abs_pct=float(th["sat_max_abs_pct"]))
    out["saturation_pct"] = float(100.0 * sat.mean())
    C.add("pressure_saturation", out["saturation_pct"] <= th["max_sat_pct"], value=out["saturation_pct"],
          limit=th["max_sat_pct"], severity="warning", stream=name)
    if nc:
        m_nc = _mask_spans(t, nc)
        out["saturation_pct_no_contact"] = float(100.0 * sat[m_nc].mean()) if m_nc.any() else 0.0
        C.add("pressure_saturation_no_contact", out["saturation_pct_no_contact"] <= th["max_sat_pct_no_contact"],
              value=out["saturation_pct_no_contact"], limit=th["max_sat_pct_no_contact"], stream=name,
              message="saturation where no contact is expected (sensor fault or contact during a no_contact block)")
        seg0 = (t >= nc[0][0]) & (t <= nc[0][1])
        out["noise_pct_rest"] = float(np.median(delta[seg0].std(0))) if seg0.sum() > 2 else None
    rest = rest_spans(phases)
    ref = nc if rest is None else rest
    if len(ref) >= 2:
        s0 = (t >= ref[0][0]) & (t <= ref[0][1])
        s1 = (t >= ref[-1][0]) & (t <= ref[-1][1])
        if s0.any() and s1.any():
            b0, b1 = np.median(raw[s0], 0), np.median(raw[s1], 0)
            drift = 100.0 * (b1 - b0) / np.where(b0 == 0, 1.0, b0)
            out["baseline_drift_spans"] = [list(ref[0]), list(ref[-1])]
            out["baseline_drift_pct"] = drift.tolist()
            out["baseline_drift_pct_max"] = float(np.abs(drift).max())
            C.add("pressure_baseline_drift", out["baseline_drift_pct_max"] <= th["max_baseline_drift_pct"],
                  value=out["baseline_drift_pct_max"], limit=th["max_baseline_drift_pct"], stream=name,
                  message="first vs last rest-pose baseline block (warm-up / glove slip / thermal drift)")
    elif m.dataset == "motion":
        C.add("pressure_baseline_drift", False, severity="warning", stream=name,
              message="fewer than two rest-pose baseline blocks: drift not measurable (record a final static baseline)")
    return out


def _imu_qc(d: Path, m: SessionManifest, name: str, th: Mapping, C: _Checks) -> dict:
    with np.load(m.stream_path(d, name)) as z:
        q = np.asarray(z["quat"], dtype=np.float64)
        finite = all(np.isfinite(z[k]).all() for k in ("quat", "gyro", "acc") if k in z.files)
        sites = [str(s) for s in z["sites"]] if "sites" in z.files else []
    err = float(np.abs(np.linalg.norm(q, axis=-1) - 1.0).max()) if q.size else float("nan")
    C.add("imu_finite", finite, stream=name)
    C.add("imu_quat_norm", err <= th["max_quat_norm_err"], value=err, limit=th["max_quat_norm_err"], stream=name)
    out = {"n_sites": int(q.shape[1]) if q.ndim == 3 else 0, "sites": sites, "quat_norm_err_max": err}
    cal = m.calibration or {}
    has_cal = "imu_offsets" in cal
    out["calibrated"] = has_cal
    if m.kind == "glove":
        C.add("imu_calibration_present", has_cal, severity="warning", stream=name,
              message="no imu_offsets in manifest.calibration (record the flat-hand block / run calibrate_session_imu)")
    qual = cal.get("imu_calibration_quality")
    if qual:
        out["calibration_quality"] = qual
        if qual.get("gyro_rms") is not None:
            C.add("imu_calibration_still_gyro", qual["gyro_rms"] <= th["max_calib_gyro_rms"], value=qual["gyro_rms"],
                  limit=th["max_calib_gyro_rms"], severity="warning", stream=name,
                  message="hand moved during the calibration block — repeat it")
        C.add("imu_calibration_still_quat", qual["quat_spread_deg"] <= th["max_calib_spread_deg"],
              value=qual["quat_spread_deg"], limit=th["max_calib_spread_deg"], severity="warning", stream=name)
    return out


def _camera_qc(d: Path, m: SessionManifest, name: str, t: np.ndarray, C: _Checks) -> dict:
    p = m.stream_path(d, name)
    if (p / "frames.npy").exists():
        fr = np.load(p / "frames.npy", mmap_mode="r")
        n_frames, shape = int(fr.shape[0]), list(fr.shape[1:])
    else:
        jpgs = sorted(p.glob("*.jpg"))
        n_frames, shape = len(jpgs), None
        if jpgs:
            try:
                from PIL import Image
                with Image.open(jpgs[0]) as im:
                    shape = [im.height, im.width, 3]
            except ImportError:  # pragma: no cover
                pass
    C.add("camera_frame_count", n_frames == t.size, value=n_frames, limit=int(t.size), stream=name,
          message="frames vs timestamps")
    return {"n_frames": n_frames, "shape": shape}


def _hand_qc(d: Path, m: SessionManifest, name: str, th: Mapping, C: _Checks) -> dict:
    with np.load(m.stream_path(d, name)) as z:
        conf = np.asarray(z["confidence"], dtype=np.float64) if "confidence" in z.files else np.ones(z["t"].shape)
    cov = float((conf >= th["hand_min_conf"]).mean()) if conf.size else 0.0
    C.add("hand_pose_coverage", cov >= th["min_hand_coverage"], value=cov, limit=th["min_hand_coverage"],
          severity="warning", stream=name, message="fraction of labels with confidence ≥ hand_min_conf")
    return {"coverage": cov, "mean_confidence": float(conf.mean()) if conf.size else None}


def _load_times(d: Path, m: SessionManifest, name: str) -> np.ndarray | None:
    p = m.stream_path(d, name)
    if name.startswith("camera_"):
        f = p / "timestamps.npy"
        return np.load(f) if f.exists() else None
    if not p.exists():
        return None
    with np.load(p) as z:
        return np.asarray(z["t"], dtype=np.float64) if "t" in z.files else None


def session_qc(session_dir: str | Path, thresholds: Mapping[str, float] | None = None, *,
               write: bool = False) -> dict:
    """QC report of one raw session (see module docstring). ``write`` saves ``qc.json`` next to
    the manifest."""
    d = Path(session_dir)
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    C = _Checks()
    m = SessionManifest.load(d)
    events = load_events(d)
    rep: dict[str, Any] = {"session_dir": str(d), "session_id": m.session_id, "dataset": m.dataset, "kind": m.kind,
                           "subject": m.subject, "layout": m.layout, "streams": {}, "thresholds": th}
    if m.meta.get("dry_run"):
        C.add("not_dry_run", False, message="dry-run manifest: no data recorded")
    phases = phases_from_events(events)
    rec = m.meta.get("recorder") or {}
    if rec:                                     # sessions written by acquisition.Recorder
        C.add("recorder_errors", not rec.get("errors"), value=list(rec.get("errors") or []),
              message="a source failed while recording (its stream may be truncated)")
        C.add("streams_recorded", not rec.get("empty_streams"), value=list(rec.get("empty_streams") or []),
              message="sources that delivered no samples (device unplugged / not streaming)")
    ph_span = (min(p["t0"] for p in phases), max(p["t1"] for p in phases)) if phases else None
    spans = []
    for name, info in m.streams.items():
        t = _load_times(d, m, name)
        if t is None:
            C.add("stream_file_present", False, stream=name, message=f"missing {info.file}")
            continue
        is_cam = name.startswith("camera_")
        tim = stream_timing(t, info.rate_hz, gap_factor=th["gap_factor"])
        st: dict[str, Any] = {"timing": tim}
        if tim["n"] >= 2:
            spans.append((float(t[0]), float(t[-1])))
        C.add("timestamps_monotonic", tim["monotonic"], stream=name)
        if info.rate_hz and tim["rate_hz"]:
            ratio = tim["rate_hz"] / info.rate_hz
            C.add("rate", abs(ratio - 1.0) <= th["rate_tol"], value=tim["rate_hz"], limit=info.rate_hz, stream=name,
                  message=f"measured/nominal = {ratio:.3f}")
        if tim["dropped_pct"] is not None:
            lim = th["max_dropped_pct_camera"] if is_cam else th["max_dropped_pct"]
            C.add("dropped", tim["dropped_pct"] <= lim, value=tim["dropped_pct"], limit=lim, stream=name)
        if tim["max_gap_s"] is not None:
            C.add("max_gap", tim["max_gap_s"] <= th["max_gap_s"], value=tim["max_gap_s"], limit=th["max_gap_s"],
                  stream=name)
        if ph_span is not None and tim["n"] >= 1:
            # the stream must cover every recorded phase (a stream that started late or died early
            # has no gaps and a normal rate, so the timing checks above cannot see it)
            late = float(t[0]) - ph_span[0]
            early = ph_span[1] - float(t[-1])
            miss = max(late, early, 0.0)
            st["coverage_missing_s"] = miss
            C.add("stream_coverage", miss <= th["coverage_tol_s"], value=miss, limit=th["coverage_tol_s"], stream=name,
                  severity="warning" if name in ("hand_pose", "object_pose") else "error",
                  message=f"stream spans [{float(t[0]):.2f}, {float(t[-1]):.2f}] s, phases [{ph_span[0]:.2f}, "
                          f"{ph_span[1]:.2f}] s")
        try:
            if is_cam:
                st.update(_camera_qc(d, m, name, t, C))
            elif name == "pressure" or info.file.endswith("pressure.npz") or "raw[C]" in info.fields:
                st.update(_pressure_qc(d, m, name, th, C, phases))
            elif name == "imu" or any(f.endswith(".quat") for f in info.fields):
                st.update(_imu_qc(d, m, name, th, C))
            elif name == "hand_pose":
                st.update(_hand_qc(d, m, name, th, C))
        except (KeyError, ValueError) as e:
            C.add("stream_readable", False, stream=name, message=f"{type(e).__name__}: {e}")
        rep["streams"][name] = st
    if not m.streams:
        C.add("has_streams", False, message="manifest lists no streams")
    dur = max((b for _, b in spans), default=0.0) - min((a for a, _ in spans), default=0.0)
    rep["duration_s"] = float(dur)
    C.add("duration", dur >= th["min_duration_s"], value=dur, limit=th["min_duration_s"])

    # labels / events
    auto = [e["name"] for e in events if e["type"] == "phase_end" and isinstance(e.get("value"), dict)
            and e["value"].get("auto_closed")]
    labels: dict[str, int] = {}
    for s in m.segments:
        labels[s["label"]] = labels.get(s["label"], 0) + 1
    rep["events"] = {"n": len(events), "phases": [p["name"] for p in phases], "auto_closed": auto,
                     "segments": labels}
    C.add("phases_closed", not auto, value=auto, severity="warning",
          message="phases closed automatically at stop (recording interrupted?)")
    if m.dataset == "motion":
        C.add("segments_no_contact", labels.get("no_contact", 0) >= 1, value=labels.get("no_contact", 0), limit=1,
              message="D1 needs contact-free segments (baseline training / ΔS baseline)")
        if m.kind == "glove":
            C.add("segments_calibration", labels.get("calibration", 0) >= 1, severity="warning")
            C.add("segments_self_touch", labels.get("self_touch", 0) >= 1, severity="warning")
    if m.dataset == "task":
        task = m.task or {}
        C.add("task_meta", bool(task.get("task_id")) and bool(task.get("instruction")),
              message="D2 episodes need task.task_id and task.instruction")
        C.add("task_success_marked", task.get("success") is not None, severity="warning",
              message="operator success verdict missing")

    # sync
    wins = [p for p in phases if (isinstance(p["value"], dict) and p["value"].get("kind") == "sync")
            or p["name"].startswith("sync")]
    sync = (m.calibration or {}).get("sync")
    rep["sync"] = sync
    if wins:
        if not sync or not sync.get("streams"):
            others = [n for n in m.streams if n != "pressure" and (n.startswith("camera_") or n == "imu")]
            C.add("sync_done", not others, severity="warning", message="sync blocks recorded but sync_session not run")
        else:
            for name, e in sync["streams"].items():
                good = [w for w in e.get("windows", []) if w.get("offset_s") is not None]
                score = min((w["score"] for w in good), default=0.0)
                C.add("sync_score", e.get("status") in ("ok", "drift_rejected") and score >= th["min_sync_score"],
                      value=score, limit=th["min_sync_score"], severity="warning", stream=name)
                if e.get("offset_s") is not None:
                    C.add("sync_offset", abs(e["offset_s"]) <= th["max_sync_offset_s"], value=e["offset_s"],
                          limit=th["max_sync_offset_s"], stream=name)
        if "pressure" in m.streams:
            try:
                from .sync import detect_taps
                with np.load(m.stream_path(d, "pressure")) as z:
                    tp, raw = z["t"], z["raw"]
                counts = [int(detect_taps(tp, raw, n=int(th["sync_taps"]), window=(w["t0"], w["t1"])).size)
                          for w in wins]
            except ValueError:
                counts = [0 for _ in wins]
            rep["sync_taps_detected"] = counts
            C.add("sync_taps_visible", all(c >= int(th["sync_taps"]) for c in counts), value=counts,
                  limit=int(th["sync_taps"]), severity="warning", stream="pressure",
                  message="tap transients found in each sync block")

    rep["checks"] = C.items
    errors = [c for c in C.items if not c["passed"] and c["severity"] == "error"]
    warnings = [c for c in C.items if not c["passed"] and c["severity"] == "warning"]
    rep["n_errors"], rep["n_warnings"] = len(errors), len(warnings)
    rep["passed"] = not errors
    if write:
        (d / QC_NAME).write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=_num))
    return rep


def format_report(rep: Mapping) -> str:
    """Short human-readable summary (failed checks first)."""
    head = (f"{'PASS' if rep['passed'] else 'FAIL'}  {rep['session_dir']}  [{rep['dataset']}/{rep['kind']}] "
            f"{rep.get('duration_s', 0):.1f}s  errors={rep['n_errors']} warnings={rep['n_warnings']}")
    lines = [head]
    for name, st in rep["streams"].items():
        tm = st.get("timing", {})
        rate = tm.get("rate_hz")
        lines.append(f"    {name:<14} n={tm.get('n', 0):<7} rate={rate if rate is None else round(rate, 1)!s:<7}"
                     f" drop={tm.get('dropped_pct') if tm.get('dropped_pct') is None else round(tm['dropped_pct'], 2)!s}%"
                     f" jitter={tm.get('jitter_ms') if tm.get('jitter_ms') is None else round(tm['jitter_ms'], 2)!s}ms")
    for c in rep["checks"]:
        if not c["passed"]:
            tag = "ERROR" if c["severity"] == "error" else "warn "
            where = f"[{c['stream']}] " if c.get("stream") else ""
            lines.append(f"  {tag} {where}{c['name']}: value={c['value']} limit={c['limit']} {c['message']}".rstrip())
    return "\n".join(lines)


def _parse_set(items: Sequence[str]) -> dict[str, float]:
    out = {}
    for it in items or ():
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        if k not in DEFAULT_THRESHOLDS:
            raise SystemExit(f"unknown threshold {k!r} (known: {sorted(DEFAULT_THRESHOLDS)})")
        out[k] = float(v)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m robot_skin.acquisition.qc", description="QC of raw recording sessions.")
    p.add_argument("sessions", nargs="+", type=Path, help="session directories (containing session.json)")
    p.add_argument("--write", action="store_true", help="write qc.json into each session dir")
    p.add_argument("--json", type=Path, default=None, help="write all reports to this JSON file")
    p.add_argument("--strict", action="store_true", help="warnings also fail")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override thresholds")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    th = _parse_set(a.set)
    reports, ok = [], True
    for s in a.sessions:
        rep = session_qc(s, th, write=a.write)
        reports.append(rep)
        ok &= rep["passed"] and (not a.strict or rep["n_warnings"] == 0)
        if not a.quiet:
            print(format_report(rep))
    if a.json:
        a.json.write_text(json.dumps(reports, indent=2, ensure_ascii=False, default=_num))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
