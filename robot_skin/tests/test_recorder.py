"""Recorder, event log, segments and stream sources (robot_skin.acquisition.recorder / .sources)."""
import json
import sys
import types

import numpy as np
import pytest

from robot_skin.acquisition.manifest import SessionManifest
from robot_skin.acquisition.recorder import (
    EVENTS_NAME, EventLog, Recorder, load_events, phases_from_events, segments_from_events, stream_file,
)
from robot_skin.acquisition.sources import (
    CsvLineParser, ImuSource, MonotonicClock, PlaybackSource, RosJointStateSource, SerialPressureSource, SimClock,
    StreamSource, validate_source,
)


def _sources(rng, T=2.0):
    """Deterministic playback sources of every kind, at their own rates / phases."""
    def grid(hz, phase=0.0):
        return phase + np.arange(int(T * hz)) / hz
    tp, ti, tc, tj, th = grid(200), grid(100, 0.003), grid(30, 0.01), grid(100, 0.001), grid(30)
    S = 7
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (ti.size, S, 1))
    return [
        PlaybackSource("pressure", "pressure", tp, {"raw": 1e6 + rng.normal(0, 10, (tp.size, 9))}, rate_hz=200),
        PlaybackSource("imu", "imu", ti, {"quat": quat, "gyro": rng.normal(0, .01, (ti.size, S, 3)),
                                          "acc": rng.normal(0, .01, (ti.size, S, 3))},
                       rate_hz=100, info={"sites": ["wrist", "palm", "thumb", "index", "middle", "ring", "pinky"]}),
        PlaybackSource("camera_ego", "camera", tc, {"frame": rng.integers(0, 255, (tc.size, 6, 8, 3), dtype=np.uint8)},
                       rate_hz=30),
        PlaybackSource("joint_state", "joint_state", tj, {"q": rng.normal(0, 1, (tj.size, 4)),
                                                          "qd": np.zeros((tj.size, 4)), "tau": np.zeros((tj.size, 4))},
                       rate_hz=100, info={"names": ["a", "b", "c", "d"]}),
        PlaybackSource("hand_pose", "hand_pose", th, {"global_orient": np.zeros((th.size, 3)),
                                                      "finger_pose": np.zeros((th.size, 15, 3)),
                                                      "wrist_pos": np.zeros((th.size, 3)),
                                                      "confidence": np.full(th.size, 0.9)}, rate_hz=30),
        PlaybackSource("object_pose", "object_pose", th, {"pos": np.zeros((th.size, 3)),
                                                          "quat": np.tile([1.0, 0, 0, 0], (th.size, 1))}, rate_hz=30),
    ]


def test_recorder_writes_the_raw_formats(tmp_path):
    rng = np.random.default_rng(0)
    srcs = _sources(rng)
    assert all(isinstance(s, StreamSource) for s in srcs)
    clock = SimClock(t0=5000.0)
    m = SessionManifest(kind="glove", layout="glove_template", dataset="motion", subject="S01")
    rec = Recorder(srcs, tmp_path / "S", m, clock=clock, sim_dt=0.01).start()
    with rec.phase("baseline_start", contact="none", labels=["no_contact"]):
        rec.run_until(0.5)
    rec.phase_start("pinch_index", contact="self")
    rec.run_until(1.0)
    rec.marker("tap", {"i": 1})
    rec.phase_end("pinch_index")
    rec.phase_start("free", contact="any")
    rec.run_until(1.5)
    rec.phase_end("free")
    rec.add_segment(1.6, 1.8, "task")
    rec.phase_start("left_open", contact="none")                    # auto-closed at stop
    rec.run_until(2.5)
    out = rec.stop()
    d = tmp_path / "S"

    with np.load(d / "pressure.npz") as z:
        assert set(z.files) == {"t", "raw"}
        assert z["raw"].shape == (400, 9) and z["raw"].dtype == np.float64
        np.testing.assert_allclose(z["t"][:3], [0.0, 0.005, 0.010], atol=1e-9)   # session clock
    with np.load(d / "imu.npz") as z:
        assert z["quat"].shape == (200, 7, 4) and z["gyro"].shape == (200, 7, 3) and z["acc"].shape == (200, 7, 3)
        assert list(z["sites"]) == ["wrist", "palm", "thumb", "index", "middle", "ring", "pinky"]
    with np.load(d / "joint_state.npz") as z:
        assert z["q"].shape == (200, 4) and list(z["names"]) == ["a", "b", "c", "d"] and "tau" in z.files
    from robot_skin.pose.vision_hand import load_hand_labels
    hl = load_hand_labels(d)
    assert hl["finger_pose"].shape == (60, 15, 3) and np.allclose(hl["confidence"], 0.9)
    with np.load(d / "object_pose.npz") as z:
        assert z["pos"].shape == (60, 3) and z["quat"].shape == (60, 4)
    ts = np.load(d / "camera_ego" / "timestamps.npy")
    fr = np.load(d / "camera_ego" / "frames.npy")
    assert fr.shape == (60, 6, 8, 3) and fr.dtype == np.uint8 and ts.shape == (60,)

    m2 = SessionManifest.load(d)
    assert m2 == out
    assert set(m2.streams) == {"pressure", "imu", "camera_ego", "joint_state", "hand_pose", "object_pose"}
    assert m2.streams["camera_ego"].method == "zoh" and m2.streams["camera_ego"].file == "camera_ego"
    assert m2.cameras == ["ego"]
    assert m2.stream_path(d, "imu") == d / "imu.npz"
    assert m2.spans("no_contact") == [(0.0, 0.5), (1.5, 2.5)]
    assert m2.spans("self_touch") == [(0.5, 1.0)]
    assert m2.spans("task") == [(1.6, 1.8)]
    assert not m2.spans("any")
    rs = m2.meta["recorder"]
    assert rs["clock"] == "sim" and rs["streams"]["pressure"]["n"] == 400
    assert rs["streams"]["pressure"]["rate_hz_measured"] == pytest.approx(200.0)

    ev = load_events(d)
    assert [e["type"] for e in ev][:3] == ["phase_start", "phase_end", "phase_start"]
    assert ev[-1] == {"t": 2.5, "type": "phase_end", "name": "left_open", "value": {"auto_closed": True}}
    assert any(e["type"] == "marker" and e["value"] == {"i": 1} for e in ev)
    for line in (d / EVENTS_NAME).read_text().splitlines():
        assert set(json.loads(line)) == {"t", "type", "name", "value"}


def test_camera_jpg_mode_and_overwrite_protection(tmp_path):
    pytest.importorskip("PIL")
    rng = np.random.default_rng(1)
    t = np.arange(10) / 30
    src = PlaybackSource("camera_third", "camera", t, {"frame": rng.integers(0, 255, (10, 16, 16, 3), dtype=np.uint8)})
    m = SessionManifest(kind="glove", layout="glove_template")
    rec = Recorder([src], tmp_path / "S", m, clock=SimClock(), camera_format="jpg").start()
    rec.run_until(0.5)
    rec.stop()
    files = sorted((tmp_path / "S" / "camera_third").glob("*.jpg"))
    assert [f.name for f in files][:2] == ["000000.jpg", "000001.jpg"] and len(files) == 10
    assert not (tmp_path / "S" / "camera_third" / "frames.npy").exists()
    with pytest.raises(FileExistsError):
        Recorder([src], tmp_path / "S", m, clock=SimClock()).start()


def test_source_validation_and_errors(tmp_path):
    t = np.arange(5) / 10.0
    cam_bad = PlaybackSource("ego", "camera", t, {"frame": np.zeros((5, 2, 2, 3), np.uint8)})
    with pytest.raises(ValueError):
        validate_source(cam_bad)
    with pytest.raises(ValueError):
        validate_source(PlaybackSource("camera_x", "pressure", t, {"raw": np.zeros((5, 2))}))
    with pytest.raises(ValueError):
        PlaybackSource("p", "pressure", t, {"frame": np.zeros((5, 2))})            # missing 'raw'
    with pytest.raises(ValueError):
        PlaybackSource("p", "pressure", t[::-1], {"raw": np.zeros((5, 2))})         # decreasing t
    p = PlaybackSource("pressure", "pressure", t, {"raw": np.zeros((5, 2))})
    with pytest.raises(ValueError):
        Recorder([p, p], tmp_path / "x", SessionManifest(kind="glove", layout="glove_template"))
    rec = Recorder([p], tmp_path / "y", SessionManifest(kind="glove", layout="glove_template"), clock=SimClock())
    with pytest.raises(RuntimeError):
        rec.marker("too early")
    rec.start()
    with pytest.raises(ValueError):
        rec.event("teleport", "x")
    with pytest.raises(RuntimeError):
        rec.start()
    rec.stop()
    with pytest.raises(NotImplementedError):
        ImuSource()
    with pytest.raises(NotImplementedError):
        RosJointStateSource("/joint_states")
    assert stream_file("pressure", "pressure") == "pressure.npz"
    assert stream_file("pressure_b", "pressure") == "pressure_b.npz"
    assert stream_file("camera_ego", "camera") == "camera_ego"


def test_threaded_recording_with_host_clock(tmp_path):
    t = np.arange(0, 0.3, 0.005)
    src = PlaybackSource("pressure", "pressure", t, {"raw": np.ones((t.size, 3))}, rate_hz=200)
    m = SessionManifest(kind="bench", layout="sats_4x4")
    with Recorder([src], tmp_path / "S", m, clock=MonotonicClock(), poll_interval_s=0.001) as rec:
        assert rec.running and rec._threaded
        with rec.phase("rest", contact="none"):
            rec.run_for(0.12)
        snap = rec.snapshot("pressure", 0.0, 0.05)
        assert snap["raw"].shape[1] == 3 and snap["t"].max() <= 0.05
        rec.run_for(0.2)
    with pytest.raises(RuntimeError):
        rec.step()
    with np.load(tmp_path / "S" / "pressure.npz") as z:
        assert z["t"].size >= 55 and np.all(np.diff(z["t"]) > 0)
    assert SessionManifest.load(tmp_path / "S").spans("no_contact")


def test_events_to_phases_and_segments(tmp_path):
    log = EventLog(tmp_path / "e" / EVENTS_NAME)
    log.log(0.0, "phase_start", "a", {"contact": "none"})
    log.log(1.0, "phase_start", "a", {"labels": ["x", "y"]})        # nested same name → LIFO
    log.log(2.0, "phase_end", "a")
    log.log(3.0, "phase_end", "a")
    log.log(3.5, "phase_end", "ghost")                             # unmatched → ignored
    log.log(4.0, "phase_start", "open", {"contact": "self"})
    log.log(2.5, "marker", "late write")                           # out of order on disk
    log.close()
    ev = load_events(tmp_path / "e")
    assert [e["t"] for e in ev] == sorted(e["t"] for e in ev)
    ph = phases_from_events(ev, t_end=6.0)
    assert [(p["name"], p["t0"], p["t1"], p["closed"]) for p in ph] == [
        ("a", 0.0, 3.0, True), ("a", 1.0, 2.0, True), ("open", 4.0, 6.0, False)]
    segs = segments_from_events(ev, t_end=6.0)
    assert {(s["label"], s["t0"], s["t1"]) for s in segs} == {
        ("no_contact", 0.0, 3.0), ("x", 1.0, 2.0), ("y", 1.0, 2.0), ("self_touch", 4.0, 6.0)}
    assert load_events(tmp_path / "missing") == []
    (tmp_path / "bad.jsonl").write_text('{"t": 1}\n')
    with pytest.raises(ValueError):
        load_events(tmp_path / "bad.jsonl")


def test_csv_line_parser():
    p = CsvLineParser(3)
    assert p.feed(b"1,2,") == []
    out = p.feed(b"3\n4,5,6\nbad,line,x\n7,8\n\n9,10,11")
    assert [o[1].tolist() for o in out] == [[1, 2, 3], [4, 5, 6]] and out[0][0] is None
    assert p.n_bad == 2
    assert p.feed(b"\n")[0][1].tolist() == [9, 10, 11]
    q = CsvLineParser(2, time_column=True)
    (t, v), = q.feed(b"1500000,1,2\n")
    assert t == pytest.approx(1.5) and v.tolist() == [1, 2]
    with pytest.raises(ValueError):
        CsvLineParser(0)


class _FakeSerial:
    def __init__(self, port, baud, timeout=0.0):
        self.buf = b""
        _FakeSerial.last = self

    def reset_input_buffer(self):
        self.buf = b""

    @property
    def in_waiting(self):
        return len(self.buf)

    def read(self, n):
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def close(self):
        pass


def test_serial_pressure_source_backfills_stamps(monkeypatch):
    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=_FakeSerial))
    clock = SimClock(100.0)
    src = SerialPressureSource("/dev/null", 2, rate_hz=100.0)
    src.start(clock)
    assert src.poll() == []
    _FakeSerial.last.buf = b"1,2\n3,4\n5,6\n7,"
    out = src.poll()
    assert [s["raw"].tolist() for _, s in out] == [[1, 2], [3, 4], [5, 6]]
    np.testing.assert_allclose([t for t, _ in out], [99.98, 99.99, 100.0])
    src.stop()
    assert src.info()["n_channels"] == 2
    # device-time parser: stamps follow the device clock, anchored at the first frame
    src2 = SerialPressureSource("/dev/null", 1, parser=CsvLineParser(1, time_column=True))
    src2.start(clock)
    _FakeSerial.last.buf = b"0,1\n10000,2\n"
    clock.advance(1.0)
    out = src2.poll()
    np.testing.assert_allclose([t for t, _ in out], [101.0, 101.01])


def test_camera_source_requires_cv2():
    try:
        import cv2  # noqa: F401
    except ImportError:
        from robot_skin.acquisition.sources import CameraSource
        with pytest.raises(ImportError):
            CameraSource("ego", 0)
    else:  # pragma: no cover
        pytest.skip("cv2 installed; the guard is not exercised")


def test_fake_sources_are_deterministic_and_consistent():
    from robot_skin.acquisition.fake import FakeScene
    from robot_skin.acquisition.sources import FakeImuSource, FakeJointSource, FakePressureSource, fake_sources
    a, b = FakeScene.default(seed=3), FakeScene.default(seed=3)
    np.testing.assert_array_equal(a.pressure_stream()["raw"], b.pressure_stream()["raw"])
    names = [s.name for s in fake_sources(a)]
    assert names == ["pressure", "imu", "camera_ego", "camera_third", "hand_pose"]
    p = FakePressureSource(a)
    clock = SimClock(0.0)
    p.start(clock)
    clock.advance(1.0)
    got = p.poll()
    assert got and all(t <= 1.0 for t, _ in got) and got[0][1]["raw"].shape == (9,)
    imu = FakeImuSource(a)
    assert imu.info()["sites"][0] == "wrist"
    q = a.imu_stream()["quat"]
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-6)
    # press → raw drops (SATS sign): the pinch step presses thumb & index tips
    ps = a.pressure_stream()
    pinch = next(ts for ts in a.timeline if ts.step.id == "pinch_index")
    m = (ps["tau"] > pinch.t0) & (ps["tau"] < pinch.t1)
    rel = ps["raw"][m] / a.baseline_raw - 1.0
    assert rel[:, 0].min() < -0.1 and rel[:, 1].min() < -0.1 and rel[:, 4].min() > -0.05
    j = FakeJointSource()
    assert j.info()["names"][0] == "thumb_j1" and j.scene.kind == "robot"


def test_fake_d1_air_grasps_are_grasp_shapes_without_press():
    """The fake scene plays the D1 ``air_grasp_*`` steps as the D2 grasp shapes (flexed, per grasp
    type) with no press anywhere — they are no_contact data for the baseline."""
    from robot_skin.acquisition.fake import FakeScene
    from robot_skin.acquisition.protocol import load_protocol, plan_session
    plan = plan_session(load_protocol("d1_motion"), seed=0, time_scale=0.1)
    sc = FakeScene.from_episode(plan.episodes[0], plan.timing, kind="glove", cameras=(), seed=2)
    rest = sc.hand_state(np.array([0.5]))["flex"][0]
    shapes = {}
    for ts in sc.timeline:
        if ts.step.motion.get("type") != "air_grasp":
            continue
        st = sc.hand_state(np.linspace(ts.t0, ts.t1, 200))
        assert not st["press"].any(), ts.step.id
        shapes.setdefault(ts.step.motion["grasp"], []).append(st["flex"].max(0))
    assert set(shapes) == {"power", "precision", "lateral", "tripod"}
    for g, fl in shapes.items():
        assert (np.max(fl, 0) > rest + 0.1).sum() >= 3, g                    # fingers clearly flexed
    assert not np.allclose(shapes["power"][0], shapes["lateral"][0], atol=0.05)


def test_fake_d2_tasks_contact_only_when_expected():
    """Every catalog task: no press while reaching / retreating, clear press while grasping or
    manipulating, and the object moves only if it is grasped (not for press_button)."""
    from robot_skin.acquisition.fake import FakeScene
    from robot_skin.acquisition.protocol import load_protocol, plan_session
    proto = load_protocol("d2_task")
    for tid in proto.tasks:
        plan = plan_session(proto, tasks=[tid], n_episodes=1, seed=0)
        sc = FakeScene.from_episode(plan.episodes[0], plan.timing, kind="glove", cameras=(), seed=1)
        ps = sc.pressure_stream()
        ch = sc.layout.channels
        press = -100.0 * (ps["raw"][:, ch] / sc.baseline_raw[ch] - 1.0)
        spans = {ts.step.id: (ts.t0, ts.t1) for ts in sc.timeline}

        def peak(name):
            a, b = spans[name]
            return press[(ps["tau"] >= a) & (ps["tau"] <= b)].max()
        assert peak("reach") < 3.0 and peak("retreat") < 3.0 and peak("baseline") < 3.0, tid
        assert peak("manipulate") > 15.0, tid
        if "grasp" in spans:
            assert peak("grasp") > 15.0, tid
        moved = np.ptp(sc.object_pose_stream()["pos"], axis=0).max()
        assert (moved < 0.02) if tid == "press_button" else (moved > 0.04), (tid, moved)


def test_serial_stamps_stay_monotonic_on_bursts_and_device_resets(monkeypatch):
    """USB serial delivers frames in bursts: back-filled stamps must never go back before the
    previous frame (QC `timestamps_monotonic`, np.interp downstream); malformed frames do not
    shift the spacing; a device clock that resets is re-anchored instead of jumping back."""
    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(Serial=_FakeSerial))
    clock = SimClock(100.0)
    src = SerialPressureSource("/dev/null", 1, rate_hz=100.0)
    src.start(clock)
    _FakeSerial.last.buf = b"1\n"
    clock.advance(0.01)
    (t1, _), = src.poll()
    _FakeSerial.last.buf = b"2\n3\n9,9\n4\n"                     # 3 good frames within 10 ms + 1 bad
    clock.advance(0.01)
    burst = src.poll()
    ts = [t1] + [t for t, _ in burst]
    assert [s["raw"].tolist() for _, s in burst] == [[2.0], [3.0], [4.0]]
    np.testing.assert_allclose(ts, [100.01, 100.01 + 1 / 300, 100.01 + 2 / 300, 100.02])   # squeezed, ≤ arrival
    assert src.info()["n_bad"] == 1                               # rejected by the CSV parser
    _FakeSerial.last.buf = b"5\n6\n"                              # clock did not advance: still increasing
    same = [t for t, _ in src.poll()]
    assert np.all(np.diff(ts[-1:] + same) > 0) and same[-1] - ts[-1] < 1e-5
    src.stop()

    class WrongSize:                                              # a custom parser emitting a bad frame
        def feed(self, data):
            return [(None, np.ones(1)), (None, np.ones(3)), (None, np.ones(1))]
    ws = SerialPressureSource("/dev/null", 1, parser=WrongSize(), rate_hz=100.0)
    ws.start(clock)
    _FakeSerial.last.buf = b"x"
    got = ws.poll()
    assert len(got) == 2 and ws.n_bad_frames == 1 and ws.info()["n_bad_frames"] == 1
    assert got[1][0] - got[0][0] == pytest.approx(0.01)            # spacing counts good frames only

    dev = SerialPressureSource("/dev/null", 1, parser=CsvLineParser(1, time_column=True))
    dev.start(clock)
    _FakeSerial.last.buf = b"5000000,1\n5010000,2\n"               # device µs clock at 5 s
    clock.advance(1.0)
    a = [t for t, _ in dev.poll()]
    _FakeSerial.last.buf = b"0,3\n10000,4\n"                       # board rebooted: device time restarts
    clock.advance(0.5)
    b = [t for t, _ in dev.poll()]
    assert np.all(np.diff(a + b) > 0) and dev.n_reanchors == 1
    assert b[0] == pytest.approx(clock()) and b[1] - b[0] == pytest.approx(0.01)


def test_overwrite_removes_the_previous_take(tmp_path):
    """Re-recording into a directory must not mix leftovers of the earlier take into the new one:
    extra jpg frames, or a camera ``timestamps_host.npy`` that sync would read as original stamps."""
    pytest.importorskip("PIL")
    rng = np.random.default_rng(2)

    def take(n, **kw):
        t = np.arange(n) / 30
        src = PlaybackSource("camera_ego", "camera", t, {"frame": rng.integers(0, 255, (n, 16, 16, 3), dtype=np.uint8)})
        tp = np.arange(0, n / 30, 0.005)
        p = PlaybackSource("pressure", "pressure", tp, {"raw": np.ones((tp.size, 2))})
        rec = Recorder([src, p], tmp_path / "S", SessionManifest(kind="glove", layout="glove_template"),
                       clock=SimClock(), camera_format="jpg", **kw).start()
        rec.run_until(n / 30)
        rec.stop()

    take(12)
    cam = tmp_path / "S" / "camera_ego"
    np.save(cam / "timestamps_host.npy", np.zeros(12))            # as left by an earlier sync
    (tmp_path / "S" / "qc.json").write_text("{}")
    (tmp_path / "S" / "hand_pose.npz").write_bytes(b"offline labels")   # not ours: kept
    take(5, overwrite=True)
    assert len(list(cam.glob("*.jpg"))) == 5 and np.load(cam / "timestamps.npy").size == 5
    assert not (cam / "timestamps_host.npy").exists() and not (tmp_path / "S" / "qc.json").exists()
    assert (tmp_path / "S" / "hand_pose.npz").exists()
    assert len(load_events(tmp_path / "S")) == 0


def test_fake_imu_quaternions_agree_with_gravity():
    """AHRS convention: the IMU world z is vertical (only the heading is arbitrary), so every site's
    quaternion rotates its accelerometer reading (specific force) onto +z·g at rest."""
    import torch
    from robot_skin.acquisition.fake import FakeScene
    from robot_skin.geometry.rotations import quat_to_matrix
    sc = FakeScene.default(seed=4, cameras=())
    s = sc.imu_stream()
    b = next(ts for ts in sc.timeline if ts.step.id == "baseline_start")
    m = (s["tau"] > b.t0 + 0.2) & (s["tau"] < b.t1 - 0.2)
    R = quat_to_matrix(torch.as_tensor(s["quat"][m])).numpy()
    f = np.einsum("tsij,tsj->tsi", R, s["acc"][m]).mean(0)
    np.testing.assert_allclose(f, np.tile([0.0, 0.0, 9.81], (7, 1)), atol=0.1)
    assert abs(sc.imu_world[1]) < 1e-12 and abs(sc.imu_world[2]) < 1e-12    # rotation about z only


def test_events_jsonl_is_the_source_of_truth_for_segments(tmp_path):
    """Regression: an operator's boundary fix in events.jsonl (DATA_ACQUISITION §9, then re-running the
    post-processing CLI) never reached the manifest segments, so contact labels and the baseline window
    kept the old span. Segments are re-derived from the events; explicit ones (add_segment) are kept."""
    from robot_skin.datasets.build import preprocess_session

    rng = np.random.default_rng(0)
    srcs = [s for s in _sources(rng) if s.name in ("pressure", "imu", "hand_pose")]
    m = SessionManifest(kind="glove", layout="glove_template", dataset="motion", subject="S01")
    rec = Recorder(srcs, tmp_path / "S", m, clock=SimClock(t0=100.0), sim_dt=0.01).start()
    with rec.phase("rest", contact="none"):
        rec.run_until(0.6)
    with rec.phase("pinch", contact="self"):
        rec.run_until(1.0)
    with rec.phase("free", contact="none"):
        rec.run_until(1.9)
    rec.add_segment(1.2, 1.4, "task")
    man = rec.stop()
    d = tmp_path / "S"
    key = lambda ss: sorted((s["t0"], s["t1"], s["label"]) for s in ss)  # noqa: E731
    before, ev0 = key(man.segments), load_events(d)

    # the operator moves the end of "free" from 1.9 to 1.5 s (subject touched the palm afterwards)
    lines = (d / EVENTS_NAME).read_text().splitlines()
    evs = [json.loads(ln) for ln in lines]
    k = next(i for i, ev in enumerate(evs) if (ev["type"], ev["name"]) == ("phase_end", "free"))
    e = json.loads(lines[k])
    e["t"] = 1.5
    lines[k] = json.dumps(e)
    (d / EVENTS_NAME).write_text("\n".join(lines) + "\n")

    # preprocessing labels follow the edited events even before the manifest is refreshed
    ep = preprocess_session(d, None)
    t, lab = ep.t, np.asarray(ep["contact_label"])
    cut = (t >= 1.5) & (t < 1.9)
    assert cut.any() and (np.asarray(ep["phase_id"])[cut] == -1).all()
    assert not (lab[cut] == 0).any() and (lab[(t >= 1.0) & (t < 1.5)] == 0).all()
    assert any("differ from events.jsonl" in n for n in ep.meta.preprocessing["notes"])

    from robot_skin.acquisition.recorder import session_segments
    from robot_skin.acquisition.session import postprocess_session, refresh_segments

    assert man.meta["recorder"]["explicit_segments"] == [{"t0": 1.2, "t1": 1.4, "label": "task"}]
    segs, notes = session_segments(SessionManifest.load(d), load_events(d))
    nc = [(a, b) for a, b, lab_ in key(segs) if lab_ == "no_contact"]
    assert nc[-1] == (pytest.approx(1.0), 1.5) and len(nc) == 2
    assert (1.2, 1.4, "task") in key(segs) and any("differ from events.jsonl" in n for n in notes)
    # a recording made before explicit_segments existed: labels no phase produces are kept
    old = SessionManifest.load(d)
    old.meta["recorder"].pop("explicit_segments")
    assert key(session_segments(old, load_events(d))[0]) == key(segs)
    # a manifest not written by the recorder (synthetic: trimmed no_contact spans) is used as written
    syn = SessionManifest.load(d)
    syn.meta.pop("recorder")
    s2, n2 = session_segments(syn, load_events(d))
    assert key(s2) == key(syn.segments) and any("outside every events.jsonl phase" in n for n in n2)
    # the post-processing CLI brings session.json in line with the events (QC reads it too)
    r = postprocess_session(d, sync=False, calibrate=False, qc=False)
    assert r["segments"]["updated"] and key(SessionManifest.load(d).segments) == key(segs)
    assert not refresh_segments(d)["updated"]
    assert not any("events.jsonl" in n for n in preprocess_session(d, None).meta.preprocessing["notes"])
    # the unedited recording: nothing to re-derive
    segs0, notes0 = session_segments(man, ev0)
    assert key(segs0) == before and notes0 == []
