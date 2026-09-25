"""3-tap clock sync, IMU calibration post-processing and session QC (acquisition.sync / .calibration / .qc)."""
import json
import shutil

import numpy as np
import pytest

from robot_skin.acquisition.calibration import calibrate_session_imu, compute_imu_calibration
from robot_skin.acquisition.manifest import SessionManifest
from robot_skin.acquisition.protocol import plan_session
from robot_skin.acquisition.qc import DEFAULT_THRESHOLDS, format_report, main as qc_main, session_qc, stream_timing
from robot_skin.acquisition.session import AutoOperator, fake_source_factory, postprocess_session, run_plan
from robot_skin.acquisition.sources import SimClock
from robot_skin.acquisition.sync import (
    ClockModel, apply_clock_models, apply_offset, change_envelope, detect_taps, estimate_offset, fit_clock_drift,
    sync_session,
)

TAPS = np.array([1.0, 1.6, 2.8])


def _tap_signal(t, taps, *, width=0.05, channels=2, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((t.size, channels))
    for tk in taps:
        x[:, 0] += np.exp(-0.5 * ((t - tk) / width) ** 2)
    return x + rng.normal(0, noise, x.shape)


# ── offsets / envelopes / drift ──────────────────────────────────────────────
@pytest.mark.parametrize("latency", [0.037, -0.052, 0.0])
def test_estimate_offset_recovers_latency(latency):
    ta = np.arange(0, 4, 1 / 200)
    tb = np.arange(0.003, 4, 1 / 100)
    xa = _tap_signal(ta, TAPS, seed=1)
    xb = _tap_signal(tb - latency, TAPS, seed=2, channels=3)   # b stamps events `latency` late
    est = estimate_offset(ta, xa, tb, xb, max_lag_s=0.3, hz=200, window=(0.5, 3.5))
    assert est.offset_s == pytest.approx(-latency, abs=0.003)
    assert est.score > 0.8 and not est.at_limit
    assert float(est) == est.offset_s
    np.testing.assert_allclose(apply_offset(tb, est.offset_s), tb + est.offset_s)


def test_estimate_offset_limits_and_precomputed_envelopes():
    ta = np.arange(0, 4, 1 / 200)
    xa = _tap_signal(ta, TAPS)
    far = estimate_offset(ta, xa, ta + 0.4, xa, max_lag_s=0.2, window=(0.5, 3.5))
    assert far.at_limit
    env_a = np.exp(-0.5 * ((ta[:, None] - TAPS) / 0.03) ** 2).sum(1)
    tb = np.arange(0, 4, 1 / 50)
    env_b = np.exp(-0.5 * ((tb[:, None] - (TAPS + 0.06)) / 0.03) ** 2).sum(1)
    est = estimate_offset(ta, env_a, tb, env_b, max_lag_s=0.2, envelope=False, window=(0.5, 3.5))
    assert est.offset_s == pytest.approx(-0.06, abs=0.004)
    with pytest.raises(ValueError):
        estimate_offset(ta, xa, ta, xa, max_lag_s=0.0)
    with pytest.raises(ValueError):
        estimate_offset(ta, xa, ta, xa, window=(2.0, 1.0))


def test_change_envelope_is_rate_independent_and_ignores_constant_channels():
    t1, t2 = np.arange(0, 4, 1 / 100), np.arange(0, 4, 1 / 400)
    x1 = np.c_[_tap_signal(t1, TAPS, noise=0.0)[:, :1], np.full(t1.size, 5.0)]
    x2 = np.c_[_tap_signal(t2, TAPS, noise=0.0)[:, :1], np.full(t2.size, 5.0)]
    g1, e1, s1 = change_envelope(t1, x1, t_range=(0.5, 3.5))
    g2, e2, s2 = change_envelope(t2, x2, t_range=(0.5, 3.5))
    np.testing.assert_array_equal(g1, g2)
    assert s1.all() and np.corrcoef(e1, e2)[0, 1] > 0.98
    with pytest.raises(ValueError):
        change_envelope(t1[:2], x1[:2])


def test_detect_taps_and_clock_drift():
    t = np.arange(0, 4, 1 / 200)
    found = detect_taps(t, _tap_signal(t, TAPS, width=0.02, seed=4))
    assert found.size == 3 and np.all(np.abs(found - TAPS) < 0.05)
    assert detect_taps(t, _tap_signal(t, TAPS[:1], width=0.02), n=3).size == 1
    ts = np.array([10.0, 250.0, 400.0])
    cm = fit_clock_drift(ts, 1.0002 * ts + 0.05)
    assert cm.scale == pytest.approx(1.0002, abs=1e-9) and cm.offset_s == pytest.approx(0.05, abs=1e-6)
    assert cm.drift_ppm == pytest.approx(200.0, abs=1e-3)
    np.testing.assert_allclose(cm.inverse(cm.apply(ts)), ts)
    assert ClockModel.from_dict(cm.to_dict()) == cm
    one = fit_clock_drift([3.0], [3.02])
    assert one.scale == 1.0 and one.offset_s == pytest.approx(0.02)
    with pytest.raises(ValueError):
        fit_clock_drift([1.0, 2.0], [1.0])


# ── recorded fake session (shared) ───────────────────────────────────────────
@pytest.fixture(scope="module")
def fake_d1(tmp_path_factory):
    """Short fake D1 session: rest, 3-tap sync, flat-hand calibration, free motion, pinch, fist,
    3-tap sync, rest — recorded, synced, calibrated and QC'd."""
    keep = {"baseline_start", "sync_start", "imu_calibration", "free_motion", "pinch", "fist", "sync_end",
            "baseline_end"}
    plan = plan_session("d1_motion", seed=5, time_scale=0.15)
    ep = plan.episodes[0]
    ep.steps = [s for s in ep.steps if s.block in keep and s.id != "pinch_ring" and s.id != "pinch_pinky"]
    fac = fake_source_factory(plan, kind="glove")
    out = tmp_path_factory.mktemp("sync") / "S"
    res = run_plan(plan, kind="glove", source_factory=fac, out=out, subject="S09", clock_factory=SimClock,
                   operator=AutoOperator())
    return {"dir": out, "res": res[0], "scene": fac.scenes[0]}


def test_sync_session_recovers_stream_latencies(fake_d1):
    d, rep, truth = fake_d1["dir"], fake_d1["res"]["sync"], fake_d1["scene"].truth
    assert rep["applied"] and rep["status"] == "ok" and len(rep["windows"]) == 2
    s = rep["streams"]
    assert set(s) == {"imu", "camera_ego", "camera_third"}
    assert s["imu"]["offset_s"] == pytest.approx(-truth["latency_s"]["imu"], abs=0.006)
    for cam in ("camera_ego", "camera_third"):                   # ≤ one frame at 30 Hz
        assert s[cam]["offset_s"] == pytest.approx(-truth["latency_s"][cam], abs=1 / 30)
        assert s[cam]["status"] == "ok" and min(w["score"] for w in s[cam]["windows"]) > 0.3
    with np.load(d / "imu.npz") as z:
        np.testing.assert_allclose(z["t"], ClockModel.from_dict(s["imu"]).apply(z["t_host"]))
        t_before = z["t"].copy()
    th = np.load(d / "camera_ego" / "timestamps_host.npy")
    np.testing.assert_allclose(np.load(d / "camera_ego" / "timestamps.npy"),
                               ClockModel.from_dict(s["camera_ego"]).apply(th))
    with np.load(d / "pressure.npz") as z:                        # reference untouched
        assert "t_host" not in z.files
    again = sync_session(d)                                        # idempotent: estimates on t_host
    with np.load(d / "imu.npz") as z:
        np.testing.assert_allclose(z["t"], t_before, atol=1e-9)
    assert again["streams"]["imu"]["offset_s"] == pytest.approx(s["imu"]["offset_s"], abs=1e-9)
    assert SessionManifest.load(d).calibration["sync"]["applied"]


def test_sync_session_fits_real_drift_only(tmp_path):
    """A device-stamped IMU drifting 800 ppm gets a drift model; a camera without drift keeps a
    constant offset (its start/end difference is below one frame)."""
    from robot_skin.acquisition.fake import FakeConfig
    plan = plan_session("d1_motion", seed=1, time_scale=0.3)
    ep = plan.episodes[0]
    ep.steps = [s for s in ep.steps if s.block in {"baseline_start", "sync_start", "free_motion", "sync_end"}]
    cfg = FakeConfig(clock_ppm={"imu": 800.0, "camera_ego": 0.0})
    fac = fake_source_factory(plan, kind="glove", cameras=("ego",), config=cfg, hand_pose=False)
    (res,) = run_plan(plan, kind="glove", source_factory=fac, out=tmp_path / "S", clock_factory=SimClock,
                      operator=AutoOperator(), calibrate=False, qc=False)
    s = res["sync"]["streams"]
    assert s["imu"]["drift_fitted"] and s["imu"]["scale"] == pytest.approx(1 - 800e-6, abs=1.5e-4)
    assert not s["camera_ego"]["drift_fitted"] and s["camera_ego"]["scale"] == 1.0
    w = s["imu"]["windows"]
    assert w[1]["offset_s"] - w[0]["offset_s"] == pytest.approx(-800e-6 * (w[1]["t"] - w[0]["t"]), abs=0.004)


def test_sync_from_and_manual_models(fake_d1, tmp_path):
    d2 = tmp_path / "copy"
    shutil.copytree(fake_d1["dir"], d2)
    changed = apply_clock_models(d2, {"imu": ClockModel(1.0, 0.5), "nope": ClockModel()})
    assert changed == ["imu"]
    with np.load(d2 / "imu.npz") as z:
        np.testing.assert_allclose(z["t"], z["t_host"] + 0.5)
    out = postprocess_session(d2, sync_from=fake_d1["dir"], calibrate=False, qc=False)
    assert out["sync"]["copied_from"] == str(fake_d1["dir"])
    with np.load(d2 / "imu.npz") as z, np.load(fake_d1["dir"] / "imu.npz") as z0:
        np.testing.assert_allclose(z["t"], z0["t"])


def test_no_sync_phases(tmp_path, fake_d1):
    d = tmp_path / "nosync"
    shutil.copytree(fake_d1["dir"], d)
    (d / "events.jsonl").write_text("")
    rep = sync_session(d)
    assert rep["status"] == "no_sync_phases" and not rep["applied"]


def test_imu_calibration_recovers_mounting(fake_d1):
    from robot_skin.geometry.rotations import quat_mul
    import torch
    from robot_skin.pose.imu_model import imu_calibration_from_dict
    d, scene = fake_d1["dir"], fake_d1["scene"]
    cal = SessionManifest.load(d).calibration
    assert {"imu_offsets", "imu_world", "imu_sites", "imu_calibration_quality"} <= set(cal)
    assert cal["imu_calibration_quality"]["ok"]
    sites = [s.name for s in scene.layout.imu_sites]
    off, world = imu_calibration_from_dict(cal, sites)
    assert off.shape == (7, 4) and world.shape == (4,)
    # offsets = conj(mounting) per site (wrist mounting is identity by construction)
    err = quat_mul(torch.as_tensor(off), torch.as_tensor(scene.imu_mount)).numpy()
    ang = np.degrees(2 * np.arccos(np.clip(np.abs(err[:, 0]), -1, 1)))
    assert ang.max() < 2.0
    again = calibrate_session_imu(d, save=False)
    np.testing.assert_allclose(again["imu_offsets"], cal["imu_offsets"], atol=1e-9)
    with pytest.raises(ValueError):
        compute_imu_calibration(np.ones((2, 7, 4)), sites, scene.layout)
    with pytest.raises(ValueError):
        compute_imu_calibration(np.tile([1.0, 0, 0, 0], (5, 1, 1)), ["elbow"], scene.layout)
    moving = np.random.default_rng(0).normal(size=(20, 7, 4))
    _, q = compute_imu_calibration(moving, sites, scene.layout, gyro=np.ones((20, 7, 3)))
    assert not q["ok"]


# ── QC ───────────────────────────────────────────────────────────────────────
def test_qc_passes_on_fake_session(fake_d1, tmp_path):
    rep = fake_d1["res"]["qc"]
    assert rep["passed"], format_report(rep)
    assert rep["n_warnings"] == 0, format_report(rep)
    assert (fake_d1["dir"] / "qc.json").exists()
    st = rep["streams"]
    assert st["pressure"]["timing"]["rate_hz"] == pytest.approx(200, rel=0.01)
    assert st["pressure"]["baseline_drift_pct_max"] < 1.0
    assert st["pressure"]["saturation_pct_no_contact"] == 0.0
    assert st["imu"]["calibrated"] and st["hand_pose"]["coverage"] > 0.8
    assert rep["sync_taps_detected"] == [3, 3]
    assert rep["events"]["segments"]["no_contact"] >= 2 and rep["events"]["segments"]["self_touch"] >= 1
    out = tmp_path / "all.json"
    assert qc_main([str(fake_d1["dir"]), "--json", str(out), "--quiet"]) == 0
    assert json.loads(out.read_text())[0]["passed"]
    with pytest.raises(SystemExit):
        qc_main([str(fake_d1["dir"]), "--set", "no_such_threshold=1"])


def _corrupt(src, dst, fn):
    shutil.copytree(src, dst)
    fn(dst)
    return session_qc(dst)


def _failed(rep):
    return {c["name"] for c in rep["checks"] if not c["passed"] and c["severity"] == "error"}


def _edit_npz(path, **changes):
    with np.load(path) as z:
        a = {k: z[k] for k in z.files}
    for k, f in changes.items():
        a[k] = f(a)
    np.savez(path, **a)


def test_qc_catches_broken_sessions(fake_d1, tmp_path):
    src = fake_d1["dir"]
    m = SessionManifest.load(src)
    (a0, a1), (b0, b1) = m.spans("no_contact")[0], m.spans("no_contact")[-1]

    def gap(d):
        with np.load(d / "pressure.npz") as z:
            keep = (z["t"] < 5.0) | (z["t"] > 6.0)
            a = {k: (z[k][keep] if z[k].shape[:1] == z["t"].shape else z[k]) for k in z.files}
        np.savez(d / "pressure.npz", **a)
    r = _corrupt(src, tmp_path / "gap", gap)
    assert {"max_gap", "dropped"} <= _failed(r) and not r["passed"]

    def rail(d):
        def f(a):
            raw = a["raw"].copy()
            raw[(a["t"] >= a0) & (a["t"] <= a1), 0] = 0.0
            return raw
        _edit_npz(d / "pressure.npz", raw=f)
    assert "pressure_saturation_no_contact" in _failed(_corrupt(src, tmp_path / "rail", rail))

    def drift(d):
        def f(a):
            raw = a["raw"].copy()
            raw[(a["t"] >= b0) & (a["t"] <= b1)] *= 1.05
            return raw
        _edit_npz(d / "pressure.npz", raw=f)
    assert "pressure_baseline_drift" in _failed(_corrupt(src, tmp_path / "drift", drift))

    assert "imu_quat_norm" in _failed(_corrupt(src, tmp_path / "quat",
                                               lambda d: _edit_npz(d / "imu.npz", quat=lambda a: a["quat"] * 1.1)))

    def frames(d):
        f = np.load(d / "camera_ego" / "frames.npy")
        np.save(d / "camera_ego" / "frames.npy", f[:-5])
    assert "camera_frame_count" in _failed(_corrupt(src, tmp_path / "frames", frames))

    def labels(d):
        mm = SessionManifest.load(d)
        mm.segments = [s for s in mm.segments if s["label"] != "no_contact"]
        mm.save(d)
    assert "segments_no_contact" in _failed(_corrupt(src, tmp_path / "labels", labels))

    def task(d):
        mm = SessionManifest.load(d)
        mm.dataset = "task"
        mm.save(d)
    assert "task_meta" in _failed(_corrupt(src, tmp_path / "task", task))

    def missing(d):
        (d / "imu.npz").unlink()
    assert "stream_file_present" in _failed(_corrupt(src, tmp_path / "missing", missing))
    assert qc_main([str(tmp_path / "missing"), "--quiet"]) == 1


def test_qc_on_dry_run_and_stream_timing(tmp_path):
    from robot_skin.acquisition import glove_logger
    assert glove_logger.main(["--protocol", "d1_motion", "--dry-run", "--out", str(tmp_path / "dry")]) == 0
    rep = session_qc(tmp_path / "dry")
    assert not rep["passed"] and "not_dry_run" in _failed(rep)
    t = np.arange(1000) / 100.0
    t = np.delete(t, np.arange(500, 505))
    st = stream_timing(t, 100.0, gap_factor=5.0)
    assert st["dropped_pct"] == pytest.approx(100 * 5 / 1000)
    assert st["n_gaps"] == 1 and st["max_gap_s"] == pytest.approx(0.06) and st["monotonic"]
    assert stream_timing(t[:1], 100.0)["rate_hz"] is None
    assert set(DEFAULT_THRESHOLDS) >= {"max_gap_s", "max_baseline_drift_pct", "min_hand_coverage"}


def test_qc_baseline_drift_compares_rest_pose_blocks_only(tmp_path, fake_d1):
    """A D2 episode's no_contact blocks are the rest-pose baseline and the flat-hand calibration:
    with a realistic bending artefact they differ by several % although nothing drifted. Drift is
    only compared between rest-pose (static) blocks — D1 baseline_start vs baseline_end."""
    from robot_skin.acquisition.fake import FakeConfig
    from robot_skin.acquisition.qc import rest_spans
    from robot_skin.acquisition.recorder import load_events, phases_from_events
    plan = plan_session("d2_task", seed=0, tasks=["grasp_lift_place"], n_episodes=1, time_scale=0.6)
    fac = fake_source_factory(plan, kind="glove", cameras=(), config=FakeConfig(artifact_pct_per_rad=12.0))
    (res,) = run_plan(plan, kind="glove", source_factory=fac, out=tmp_path / "d2", clock_factory=SimClock,
                      operator=AutoOperator())
    rep = res["qc"]
    assert rep["passed"] and rep["n_warnings"] == 0, format_report(rep)
    assert "baseline_drift_pct_max" not in rep["streams"]["pressure"]
    d = res["session_dir"]
    nc = sorted(SessionManifest.load(d).spans("no_contact"))
    with np.load(f"{d}/pressure.npz") as z:
        t, raw = z["t"], z["raw"]
    med = [np.median(raw[(t >= a) & (t <= b)], 0) for a, b in (nc[0], nc[-1])]
    assert np.abs(100 * (med[1] - med[0]) / med[0]).max() > 3.0      # what the pose change looks like
    assert len(rest_spans(phases_from_events(load_events(d)))) == 1

    ph = {p["name"]: p for p in phases_from_events(load_events(fake_d1["dir"]))}
    st = fake_d1["res"]["qc"]["streams"]["pressure"]
    assert st["baseline_drift_spans"] == [[ph[n]["t0"], ph[n]["t1"]] for n in ("baseline_start", "baseline_end")]
    assert rest_spans([{"name": "x", "t0": 0.0, "t1": 1.0, "value": {"contact": "none"}}]) is None


def test_qc_catches_recorder_failures(fake_d1, tmp_path):
    """A source that errored, never delivered, or stopped early leaves a stream without gaps and
    with a normal rate — the recorder report and the phase coverage catch it."""
    src = fake_d1["dir"]

    def meta(**kw):
        def f(d):
            mm = SessionManifest.load(d)
            mm.meta["recorder"].update(kw)
            mm.save(d)
        return f
    assert "recorder_errors" in _failed(_corrupt(src, tmp_path / "err", meta(errors=["imu: OSError: gone"])))
    assert "streams_recorded" in _failed(_corrupt(src, tmp_path / "empty", meta(empty_streams=["camera_side"])))

    def died(d):
        with np.load(d / "imu.npz") as z:
            half = z["t"] < 0.5 * z["t"][-1]
            a = {k: (z[k][half] if z[k].shape[:1] == z["t"].shape else z[k]) for k in z.files}
        np.savez(d / "imu.npz", **a)
    r = _corrupt(src, tmp_path / "died", died)
    assert "stream_coverage" in _failed(r) and r["streams"]["imu"]["coverage_missing_s"] > 1.0
    assert all(c["passed"] for c in r["checks"] if c["stream"] == "imu" and c["name"] in ("max_gap", "rate"))
    assert fake_d1["res"]["qc"]["streams"]["camera_ego"]["coverage_missing_s"] < 0.1


def test_qc_checks_only_layout_channels(fake_d1, tmp_path):
    """Unconnected board channels (constant / on a rail) are not taxels and must not fail QC; a
    board with fewer channels than the layout maps is an error."""
    src = fake_d1["dir"]

    def extra(d):
        _edit_npz(d / "pressure.npz", raw=lambda a: np.c_[a["raw"], np.zeros((a["raw"].shape[0], 7))])
    r = _corrupt(src, tmp_path / "wide", extra)
    assert r["passed"] and r["n_warnings"] == 0, format_report(r)
    st = r["streams"]["pressure"]
    assert st["n_channels"] == 16 and st["checked_channels"] == list(range(9)) and st["stuck_channels"] == []

    def narrow(d):
        _edit_npz(d / "pressure.npz", raw=lambda a: a["raw"][:, :7])
    r = _corrupt(src, tmp_path / "narrow", narrow)
    assert "pressure_channels_layout" in _failed(r)

    def dead(d):
        def f(a):
            raw = a["raw"].copy()
            raw[:, 3] = 1234.0
            return raw
        _edit_npz(d / "pressure.npz", raw=f)
    r = _corrupt(src, tmp_path / "dead", dead)
    assert "pressure_channels_alive" in _failed(r) and r["streams"]["pressure"]["stuck_channels"] == [3]
