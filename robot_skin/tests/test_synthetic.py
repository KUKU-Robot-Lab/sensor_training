"""Synthetic raw-session generator: formats, labels, physics-ish model, determinism, speed."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from common.layouts import layout_from_dict, load_layout
from common.signal import ADC_MIN, relative_change, saturation_mask
from robot_skin.acquisition.manifest import SCHEMA_VERSION, SessionManifest
from robot_skin.contact.self_touch import finger_of, point_segment_distance
from robot_skin.datasets import synthetic as S
from robot_skin.pose.imu_model import (
    calibrate_imu_offsets, estimate_world_alignment, imu_calibration_from_dict, imu_reference_rotations,
)
from robot_skin.pose.mano import FINGERS, ManoSkeleton, self_touch_from_hand
from robot_skin.pose.robot_fk import taxel_poses_from_joints
from robot_skin.pose.urdf import URDFModel
from robot_skin.pose.vision_hand import load_hand_labels, smooth_hand_labels


def _gen(root: Path, name: str, **kw) -> tuple[Path, SessionManifest]:
    d = root / name
    return d, S.generate_session(d, **kw)


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    out = {
        "glove_motion": _gen(root, "gm", kind="glove", dataset="motion", seed=3, cameras=("ego", "third")),
        "glove_task": _gen(root, "gt", kind="glove", dataset="task", seed=2, task_id="grasp_lift_place",
                           params={"success_prob": 1.0}),
        "robot_motion": _gen(root, "rm", kind="robot", dataset="motion", seed=4),
        "robot_task": _gen(root, "rt", kind="robot", dataset="task", seed=5, task_id="pour",
                           params={"success_prob": 1.0}),
    }
    return out


def _events(d: Path) -> list[dict]:
    return [json.loads(line) for line in (d / S.EVENTS_FILE).read_text().splitlines() if line.strip()]


def _in_spans(t: np.ndarray, spans) -> np.ndarray:
    m = np.zeros(t.shape, dtype=bool)
    for a, b in spans:
        m |= (t >= a) & (t < b)
    return m


# ── plans ──────────────────────────────────────────────────────────────────
def test_motion_and_task_plans():
    for D in (2.5, 6.0, 30.0):
        for kind in ("glove", "robot"):
            b = S.plan_motion_session(D, kind)
            first = ["imu_calibration", "baseline_start"] if kind == "glove" else ["baseline_start"]
            assert [x.name for x in b[:len(first)]] == first and b[-1].name == "baseline_end" and len(b) >= 5
            assert b[0].t0 == 0.0 and abs(b[-1].t1 - D) < 1e-9
            assert all(abs(x.t1 - y.t0) < 1e-9 and x.dur > 0 for x, y in zip(b, b[1:], strict=False))
            labels = {lab for x in b for lab in x.labels}
            assert labels == ({"no_contact", "self_touch"} | ({"calibration"} if kind == "glove" else set()))
            assert all(x.labels == S.CONTACT_LABELS[x.contact] or x.name == "imu_calibration" for x in b)
            if kind == "robot":
                assert not any(x.gen == "wrist" or x.info.get("finger") in ("ring", "pinky") for x in b)
    gens = {x.gen for x in S.plan_motion_session(30.0, "glove")}
    assert {"sweep", "pinch", "wrist", "free", "fist", "air_grasp"} <= gens
    assert "air_grasp" not in {x.gen for x in S.plan_motion_session(30.0, "robot")}
    t = S.plan_task_session(6.0)
    assert [x.name for x in t] == ["baseline", *S.TASK_PHASES, "baseline_end"]
    assert [x.labels for x in t] == [("no_contact",)] + [()] * 5 + [("no_contact",)]
    assert [x.name for x in t if x.contact == "object"] == ["grasp", "manipulate", "release"]
    with pytest.raises(ValueError):
        S.plan_motion_session(1.0)
    with pytest.raises(ValueError):
        S.plan_task_session(2.0)


# ── file formats ───────────────────────────────────────────────────────────
def test_glove_motion_files_and_formats(sessions):
    d, m = sessions["glove_motion"]
    L = load_layout(m.layout)
    for f in ("session.json", "pressure.npz", "imu.npz", "hand_pose.npz", "events.jsonl", S.GT_FILE,
              "camera_ego/frames.npy", "camera_ego/timestamps.npy", "camera_third/frames.npy"):
        assert (d / f).is_file(), f
    for f in ("joint_state.npz", "object_pose.npz", S.URDF_FILE):
        assert not (d / f).exists()
    with np.load(d / "pressure.npz") as z:
        assert set(z.files) == {"t", "raw"}
        t, raw = z["t"], z["raw"]
    assert t.dtype == np.float64 and t.ndim == 1 and np.all(np.diff(t) > 0) and 0.0 <= t[0] < 0.02
    assert raw.dtype == np.float64 and raw.shape == (len(t), int(L.channels.max()) + 1)
    assert abs(np.mean(np.diff(t)) * 200.0 - 1.0) < 0.01 and t[-1] <= 6.0
    with np.load(d / "imu.npz") as z:
        assert set(z.files) == {"t", "quat", "gyro", "acc", "sites"}
        ti, q, g, a, sites = z["t"], z["quat"], z["gyro"], z["acc"], z["sites"]
    S_ = len(L.imu_sites)
    assert ti.dtype == np.float64 and np.all(np.diff(ti) > 0) and abs(np.mean(np.diff(ti)) * 100.0 - 1.0) < 0.01
    assert q.shape == (len(ti), S_, 4) and q.dtype == np.float32
    assert g.shape == a.shape == (len(ti), S_, 3) and g.dtype == a.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-5)
    assert np.all(q[..., 0] >= 0)                                        # device hemisphere convention
    assert list(sites) == [s.name for s in L.imu_sites]
    calib = _in_spans(ti, m.spans("calibration"))                         # static hold: gravity only
    np.testing.assert_allclose(np.linalg.norm(a[calib], axis=-1), 9.81, atol=0.4)
    assert np.abs(g[calib]).max() < 0.08
    for cam in ("ego", "third"):
        fr = np.load(d / f"camera_{cam}" / "frames.npy")
        ts = np.load(d / f"camera_{cam}" / "timestamps.npy")
        assert fr.dtype == np.uint8 and fr.shape[1:] == (24, 32, 3) and fr.shape[0] == ts.shape[0]
        assert ts.dtype == np.float64 and np.all(np.diff(ts) > 0) and 170 <= len(ts) <= 185
        assert 0.0 <= ts[0] < 0.1 and ts[-1] <= 6.0
        assert abs(m.meta["synthetic"]["camera_offset_s"][cam]) <= 0.03


def test_manifest_v2_roundtrip(sessions):
    d, m = sessions["glove_motion"]
    m2 = SessionManifest.load(d)
    assert m2.to_dict() == m.to_dict()
    assert m2.schema_version == SCHEMA_VERSION == 2
    assert (m2.kind, m2.dataset, m2.subject, m2.task) == ("glove", "motion", "s0", None)
    assert set(m2.streams) == {"pressure", "imu", "hand_pose", "camera_ego", "camera_third"}
    assert m2.cameras == ["ego", "third"] and m2.streams["camera_ego"].method == "zoh"
    for name in m2.streams:
        assert m2.stream_path(d, name).exists()
    assert set(m2.calibration) == {"imu_offsets", "imu_world", "imu_sites"}
    assert m2.meta["generator"] == "robot_skin.datasets.synthetic" and m2.meta["seed"] == 3
    assert m2.meta["synthetic"]["gt_file"] == S.GT_FILE


# ── events / segments / labels ─────────────────────────────────────────────
def test_events_and_segments_consistent(sessions):
    d, m = sessions["glove_motion"]
    ev = _events(d)
    assert all(set(e) == {"t", "type", "name", "value"} for e in ev)
    assert [e["t"] for e in ev] == sorted(e["t"] for e in ev)
    blocks = m.meta["synthetic"]["blocks"]
    starts = [e for e in ev if e["type"] == "phase_start"]
    ends = [e for e in ev if e["type"] == "phase_end"]
    assert [e["name"] for e in starts] == [e["name"] for e in ends] == [b["name"] for b in blocks]
    for s, e, b in zip(starts, ends, blocks, strict=True):
        assert s["t"] == pytest.approx(b["t0"], abs=1e-6) and e["t"] == pytest.approx(b["t1"], abs=1e-6)
        assert s["value"]["labels"] == b["labels"] and s["value"]["contact"] == b["contact"]
    from_events = sorted((round(s["t"], 6), round(e["t"], 6), lab)                 # recorder convention:
                         for s, e in zip(starts, ends, strict=True) for lab in s["value"]["labels"])
    assert from_events == sorted((g["t0"], g["t1"], g["label"]) for g in m.segments)   # one segment per label
    D = m.meta["duration_s"]
    assert all(0.0 <= s["t0"] <= s["t1"] <= D + 1e-9 for s in m.segments)
    assert {s["label"] for s in m.segments} == {"no_contact", "self_touch", "calibration"}
    (c0, c1), = m.spans("calibration")
    assert (c0, c1) == m.spans("no_contact")[0] and c0 == 0.0 and blocks[0]["name"] == "imu_calibration"
    st_blocks = [(b["t0"], b["t1"]) for b in blocks if "self_touch" in b["labels"]]
    assert m.spans("self_touch") == [(pytest.approx(a, abs=1e-6), pytest.approx(b, abs=1e-6)) for a, b in st_blocks]
    gt = S.load_ground_truth(d)
    nc = _in_spans(gt["t"], m.spans("no_contact"))
    assert not gt["contact"][nc].any() and np.abs(gt["press_pct"][nc]).max() < 0.05   # no_contact is truthful
    for a, b in m.spans("self_touch"):                                            # every pinch touches
        assert gt["self_touch"][(gt["t"] >= a) & (gt["t"] < b)].any()
    assert m.meta["synthetic"]["no_contact_trimmed_s"] == 0.0


def test_self_touch_labels_match_preprocessing_geometry(sessions):
    d, m = sessions["glove_motion"]
    gt = S.load_ground_truth(d)
    L = load_layout(m.layout)
    st = self_touch_from_hand(L, ManoSkeleton(), gt["hand_global_orient"], gt["hand_finger_pose"],
                              gt["hand_wrist_pos"])
    np.testing.assert_array_equal(st, gt["self_touch"])
    assert (gt["penetration_m"][gt["contact"]] >= 0).all() and (gt["penetration_m"][~gt["contact"]] == 0).all()


# ── tactile model ──────────────────────────────────────────────────────────
def test_raw_is_channel_major_and_matches_layout(tmp_path):
    base = load_layout("glove_template").to_dict(units="mm")
    perm = [7, 2, 11, 0, 5, 9, 1, 4, 10]                 # layout taxel i → raw column perm[i]
    for tx, ch in zip(base["taxels"], perm, strict=True):
        tx["channel"] = ch
    base["name"] = "glove_permuted"
    L = layout_from_dict(base)
    d = tmp_path / "perm"
    m = S.generate_session(d, kind="glove", seed=11, layout=L, n_channels=13, cameras=())
    assert (d / S.LAYOUT_FILE).is_file() and m.meta["synthetic"]["layout_file"] == S.LAYOUT_FILE
    L2 = load_layout(m.layout)
    np.testing.assert_array_equal(L2.channels, perm)
    raw = np.load(d / "pressure.npz")["raw"]
    assert raw.shape[1] == 13
    gt = S.load_ground_truth(d)
    np.testing.assert_array_equal(gt["channels"], perm)
    lay = L2.by_channel(raw)                                                     # → layout order
    expect = gt["baseline_raw"] * (1.0 + (gt["delta_true_pct"].astype(np.float64) + gt["noise_pct"]) / 100.0)
    ok = ~gt["saturated"]
    np.testing.assert_allclose(lay[ok], np.round(expect[ok]), atol=2.0)         # float32 GT → ≤ 2 counts
    unused = sorted(set(range(13)) - set(perm))
    rel = raw[:, unused] / np.median(raw[:, unused], axis=0) - 1.0
    assert np.abs(rel).max() < 0.015                                             # baseline + drift + noise only
    assert not np.allclose(lay, raw[:, :9])                                      # the permutation matters


def test_press_makes_raw_drop(sessions):
    for key in ("glove_motion", "glove_task", "robot_motion"):
        d, m = sessions[key]
        L = load_layout(m.layout)
        gt = S.load_ground_truth(d)
        raw = L.by_channel(np.load(d / "pressure.npz")["raw"])
        no_press = gt["baseline_raw"] * (1.0 + (gt["artefact_pct"] + gt["drift_pct"] + gt["noise_pct"]) / 100.0)
        deep = gt["penetration_m"] > 0.002
        assert deep.sum() > 20, key
        assert np.all(raw[deep] < 0.98 * no_press[deep]), key                       # ≥ 2 % drop
        near = np.convolve(gt["contact"].any(1).astype(float), np.ones(61), mode="same") > 0   # ±0.15 s
        assert np.abs(gt["press_pct"][~near]).max() < 0.05 and gt["press_pct"].max() <= 1e-6
        ds = relative_change(raw, gt["baseline_raw"])
        assert ds[deep].mean() < ds[~gt["contact"]].mean() - 5.0                    # SATS sign: press → negative
        sat = gt["saturated"]
        assert np.all(raw[sat] == ADC_MIN) and gt["contact"][sat].all()


@pytest.mark.parametrize("kind", ["glove", "robot"])
def test_short_session_no_contact_stays_truthful(tmp_path, kind):
    """At the minimum duration the blocks are short and a pinch's press lag tail can reach into the
    next no_contact block: those spans are trimmed so no_contact never carries a press signal."""
    d = tmp_path / kind
    m = S.generate_session(d, kind=kind, dataset="motion", duration_s=2.5, seed=5, cameras=())
    gt = S.load_ground_truth(d)
    nc = _in_spans(gt["t"], m.spans("no_contact"))
    assert not gt["contact"][nc].any() and np.abs(gt["press_pct"][nc]).max() <= S.PRESS_EPS_PCT
    trimmed = m.meta["synthetic"]["no_contact_trimmed_s"]
    assert 0.0 < trimmed < 0.3                                           # this seed has a tail to trim
    blocks = m.meta["synthetic"]["blocks"]
    nc_total = sum(b["t1"] - b["t0"] for b in blocks if "no_contact" in b["labels"])
    assert sum(b - a for a, b in m.spans("no_contact")) == pytest.approx(nc_total - trimmed, abs=1e-5)
    for a, b in m.spans("no_contact"):                                   # trimmed spans stay inside blocks
        assert any(x["t0"] - 1e-6 <= a and b <= x["t1"] + 1e-6 for x in blocks if "no_contact" in x["labels"])


def test_saturation_dropouts(tmp_path):
    d = tmp_path / "sat"
    m = S.generate_session(d, kind="glove", seed=5, cameras=(), params={"sat_prob": 1.0})
    L = load_layout(m.layout)
    gt = S.load_ground_truth(d)
    raw = L.by_channel(np.load(d / "pressure.npz")["raw"])
    sat = gt["saturated"]
    assert sat.any() and gt["contact"][sat].all()
    np.testing.assert_array_equal(saturation_mask(raw, relative_change(raw, gt["baseline_raw"])), sat)
    dt = np.mean(np.diff(gt["t"]))
    for i in range(L.n):                                  # dropouts are short, around the press peak
        runs = np.flatnonzero(np.diff(np.r_[0, sat[:, i].astype(int), 0]))
        assert all((b - a) * dt <= 0.31 for a, b in zip(runs[::2], runs[1::2], strict=True))
    m0 = S.generate_session(tmp_path / "nosat", kind="glove", seed=5, cameras=(), params={"sat_prob": 0.0})
    assert not S.load_ground_truth(tmp_path / "nosat")["saturated"].any() and m0.session_id == m.session_id


def test_ground_truth_artefact_is_recoverable_and_dynamic(sessions):
    d, m = sessions["glove_motion"]
    L = load_layout(m.layout)
    gt = S.load_ground_truth(d)
    z = np.load(d / "pressure.npz")
    t, raw = z["t"], L.by_channel(z["raw"])
    # what preprocessing does: baseline = median of the first no_contact segment, ΔS = relative change
    base = np.median(raw[_in_spans(t, m.spans("no_contact")[:1])], axis=0)
    np.testing.assert_allclose(base, gt["baseline_raw"], rtol=2e-3)
    nc = _in_spans(t, m.spans("no_contact"))
    err = relative_change(raw, base)[nc] - gt["artefact_pct"][nc]
    assert np.abs(err.mean()) < 0.2 and err.std() < 0.35 * gt["artefact_pct"][nc].std()
    # the artefact is NOT a static function of the joint angle (velocity term + lag)
    u = gt["joint_angle"].astype(np.float64) @ gt["artefact_weights"].T
    static = gt["angle_gain"] * u + gt["quad_gain"] * u * u
    assert np.abs(gt["artefact_pct"] - static).max() > 0.3
    assert np.abs(gt["artefact_pct"][nc]).max() > 1.0                          # motion artefact is sizeable


# ── IMU / hand pose ────────────────────────────────────────────────────────
def test_imu_calibration_is_recorded_and_recoverable(sessions):
    d, m = sessions["glove_motion"]
    L = load_layout(m.layout)
    imu = np.load(d / "imu.npz")
    off_m, G_m = imu_calibration_from_dict(m.calibration, list(imu["sites"]))
    assert off_m.shape == (len(L.imu_sites), 4) and G_m.shape == (4,)
    ang_mount = 2 * np.degrees(np.arccos(np.clip(np.abs(off_m[:, 0]), 0, 1)))
    assert ang_mount[0] < 1e-6 and np.all(ang_mount[1:] > 2.0)               # wrist identity, others mounted
    sel = _in_spans(imu["t"], m.spans("calibration"))
    go_c = m.meta["synthetic"]["calibration_pose"]["global_orient"]
    R_ref = imu_reference_rotations(L, ManoSkeleton(), global_orient=go_c)
    G = estimate_world_alignment(imu["quat"][sel], R_ref, index=0)
    off = calibrate_imu_offsets(imu["quat"][sel], R_ref, world=G)
    err = 2 * np.degrees(np.arccos(np.clip(np.abs((off * off_m).sum(1)), 0, 1)))
    assert err.max() < 0.5
    assert 2 * np.degrees(np.arccos(min(1.0, abs(float(G @ G_m))))) < 0.5


def test_hand_pose_labels(sessions):
    d, m = sessions["glove_motion"]
    lab = load_hand_labels(d)
    T = lab["t"].shape[0]
    assert abs(np.mean(np.diff(lab["t"])) * 30.0 - 1.0) < 0.01 and 170 <= T <= 185
    assert lab["finger_pose"].shape == (T, 15, 3) and lab["confidence"].shape == (T,)
    assert lab["confidence"].min() >= 0.0 and lab["confidence"].max() <= 1.0
    gt = S.load_ground_truth(d)
    good = lab["confidence"] > 0.5
    fp_gt = np.stack([np.interp(lab["t"], gt["t"], gt["hand_finger_pose"][:, j, k])
                      for j in range(15) for k in range(3)], 1).reshape(T, 15, 3)
    assert np.abs(lab["finger_pose"][good] - fp_gt[good]).mean() < 0.03
    sm = smooth_hand_labels(lab["t"], lab["global_orient"], lab["finger_pose"], lab["wrist_pos"], lab["confidence"])
    assert sm["valid"].mean() > 0.8


def test_cameras_encode_hand_flexion(sessions):
    d, m = sessions["glove_motion"]
    gt = S.load_ground_truth(d)
    for cam in m.cameras:
        fr = np.load(d / f"camera_{cam}" / "frames.npy").astype(np.float64) / 255.0
        ts = np.load(d / f"camera_{cam}" / "timestamps.npy")
        y = np.interp(ts - m.meta["synthetic"]["camera_offset_s"][cam], gt["t"], gt["joint_angle"].mean(1))
        X = np.c_[fr.reshape(len(fr), -1), np.ones(len(fr))]
        tr, te = np.arange(len(X)) % 2 == 0, np.arange(len(X)) % 2 == 1
        w = X[tr].T @ np.linalg.solve(X[tr] @ X[tr].T + np.eye(tr.sum()), y[tr])   # ridge (λ=1), dual form
        r2 = 1 - ((X[te] @ w - y[te]) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
        assert r2 > 0.8, (cam, r2)


# ── robot ──────────────────────────────────────────────────────────────────
def test_robot_urdf_parses_and_fk_works(sessions):
    d, m = sessions["robot_motion"]
    assert m.kind == "robot" and m.meta["urdf"] == S.URDF_FILE and (d / S.URDF_FILE).is_file()
    assert set(m.streams) == {"pressure", "joint_state", "camera_ego"} and not m.calibration
    assert not (d / "imu.npz").exists() and not (d / "hand_pose.npz").exists()
    model = URDFModel.from_file(d / m.meta["urdf"])
    assert model.joint_names == S.ROBOT_JOINT_NAMES and model.n_dof == 16
    L = load_layout(m.layout)
    assert set(L.parents) <= set(model.link_names)
    js = np.load(d / "joint_state.npz")
    assert set(js.files) == {"t", "q", "qd", "names"}
    q, names = js["q"], [str(n) for n in js["names"]]
    assert q.dtype == np.float32 and q.shape == (len(js["t"]), 16) and js["qd"].shape == q.shape
    assert abs(np.mean(np.diff(js["t"])) * 100.0 - 1.0) < 0.01
    assert np.all(q >= model.lower - 0.01) and np.all(q <= model.upper + 0.01)
    qm = model.reorder_q(q, names)
    pos, nrm = taxel_poses_from_joints(L, model, qm)
    assert pos.shape == (len(q), L.n, 3) and np.isfinite(pos).all()
    np.testing.assert_allclose(np.linalg.norm(nrm, axis=-1), 1.0, atol=1e-9)
    gt = S.load_ground_truth(d)                                          # FK of joint_state ≈ GT taxel poses
    qi = np.stack([np.interp(gt["t"], js["t"], qm[:, k]) for k in range(16)], 1)
    pos_i, _ = taxel_poses_from_joints(L, model, qi)
    assert np.abs(pos_i - gt["taxel_pos"]).max() < 0.004
    qd_fd = np.gradient(qm, js["t"], axis=0)
    assert np.corrcoef(qd_fd.ravel(), js["qd"].ravel())[0, 1] > 0.95


def test_robot_self_contact_geometry(tmp_path):
    """Robot self-contact from an independent distance computation: fingertip taxels sit on the link
    axis (touch when within 2·r + margin of another finger's axis), palm taxels on the palm surface
    (r + margin); pinches press the pads together without the links passing through each other."""
    d = tmp_path / "robot10"
    m = S.generate_session(d, kind="robot", dataset="motion", duration_s=10.0, seed=4, cameras=())
    assert "fist" in [b["name"] for b in m.meta["synthetic"]["blocks"]]
    L = load_layout(m.layout)
    gt = S.load_ground_truth(d)
    p = S.SynthParams()
    model = URDFModel.from_file(d / m.meta["urdf"])
    names = [n for n in model.link_names if finger_of(n) in FINGERS and n != "thumb_base_link"]
    lens = {n: float(np.linalg.norm(model.joint(j).origin_xyz)) for j in model.joint_names
            for n in [model.joint(j).parent] if n in names}               # link length = next joint offset
    for f in FINGERS:
        lens[f"{f}_distal_link"] = S._ROBOT_DISTAL[f]
    q = gt["q"].astype(np.float64)
    fk = model.fk_numpy(q, links=names)
    p0 = np.stack([fk[n][:, :3, 3] for n in names], 1)
    p1 = p0 + np.stack([fk[n][:, :3, 2] * lens[n] for n in names], 1)
    pos = gt["taxel_pos"].astype(np.float64)
    for i, par in enumerate(L.parents):
        keep = [k for k, n in enumerate(names) if finger_of(n) != finger_of(par)]
        dist = point_segment_distance(pos[:, i, None, :], p0[:, keep], p1[:, keep]).min(1)
        on_axis = finger_of(par) in FINGERS
        thr = p.robot_link_radius * (2.0 if on_axis else 1.0) + p.robot_self_margin
        clear = np.abs(dist - thr) > 2e-4                                   # float32 GT positions
        np.testing.assert_array_equal(gt["self_touch"][clear, i], dist[clear] <= thr, err_msg=par)
    assert gt["self_touch"][:, [0, 1]].any(0).all() and gt["self_touch"][:, 5:].any()   # pinch + fist → palm
    pad = np.linalg.norm(pos[:, 0] - pos[:, 1], axis=1)                    # thumb / index pads (on axis)
    assert pad.min() > 4e-3 and pad.min() < 2 * p.robot_link_radius     # pressed, not interpenetrating


def test_robot_hand_urdf_geometry():
    model = URDFModel.from_string(S.robot_hand_urdf())
    assert model.root_link == "base_link"
    tmpl = load_layout("robot_hand_template")
    assert set(tmpl.parents) <= set(model.link_names)
    q = np.zeros(model.n_dof)
    tip0 = model.fk_numpy(q)["index_distal_link"][:3, 3]
    q[S.ROBOT_JOINT_NAMES.index("index_pip")] = 1.0
    tip1 = model.fk_numpy(q)["index_distal_link"][:3, 3]
    assert tip1[0] > tip0[0] + 0.01                                     # flexion curls toward the palm side (+x)


# ── tasks ──────────────────────────────────────────────────────────────────
def test_glove_task_session(sessions):
    d, m = sessions["glove_task"]
    assert m.dataset == "task" and m.task["task_id"] == "grasp_lift_place" and m.task["success"] is True
    assert set(m.task) == {"task_id", "instruction", "object", "success"}
    assert m.task["object"] in m.task["instruction"] and m.task["object"] in S.OBJECT_RADIUS
    with np.load(d / "object_pose.npz") as z:
        assert set(z.files) == {"t", "pos", "quat"}
        to, pos, quat = z["t"], z["pos"], z["quat"]
    assert pos.shape == (len(to), 3) and quat.shape == (len(to), 4)
    np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-5)
    ev = _events(d)
    assert ev[0]["type"] == "instruction" and ev[0]["value"] == m.task["instruction"]
    assert ev[-1]["type"] == "success" and ev[-1]["value"] is True
    assert (ev[0]["name"], ev[-1]["name"]) == ("instruction", "success")        # = acquisition Recorder
    phases = [e["name"] for e in ev if e["type"] == "phase_start"]
    assert phases == ["baseline", *S.TASK_PHASES, "baseline_end"]
    assert [s["label"] for s in m.segments] == ["no_contact", "task", "no_contact"]
    B = {b["name"]: b for b in m.meta["synthetic"]["blocks"]}
    gt = S.load_ground_truth(d)
    t = gt["t"]
    oc = gt["object_contact"].any(1)

    def ph(n):
        return (t >= B[n]["t0"]) & (t < B[n]["t1"])
    assert not oc[ph("baseline") | ph("reach") | ph("retreat") | ph("baseline_end")].any()
    assert oc[ph("manipulate")].mean() > 0.95 and oc[ph("grasp")].any()
    assert gt["object_contact"][:, [0, 1]].any(0).all()                   # thumb + index hold the object
    moved = np.linalg.norm(pos[to > B["release"]["t0"]].mean(0) - pos[to < B["grasp"]["t0"]].mean(0))
    assert moved > 0.05                                                    # lifted and placed elsewhere
    lifted = pos[(to > B["manipulate"]["t0"]) & (to < B["manipulate"]["t1"]), 2].max()
    assert lifted > pos[0, 2] + 0.03


def test_task_failure_slips(tmp_path):
    d = tmp_path / "fail"
    m = S.generate_session(d, kind="glove", dataset="task", seed=7, task_id="grasp_lift_place",
                           params={"success_prob": 0.0}, cameras=())
    assert m.task["success"] is False and _events(d)[-1]["value"] is False
    gt = S.load_ground_truth(d)
    B = {b["name"]: b for b in m.meta["synthetic"]["blocks"]}
    late = (gt["t"] > B["manipulate"]["t1"] - 0.2 * (B["manipulate"]["t1"] - B["manipulate"]["t0"])) & \
           (gt["t"] < B["manipulate"]["t1"])
    assert gt["object_contact"][(gt["t"] >= B["manipulate"]["t0"]) & (gt["t"] < B["manipulate"]["t1"])].any()
    assert not gt["object_contact"][late].any()                          # dropped before the end
    assert gt["object_pos"][-1, 2] == pytest.approx(S.OBJECT_RADIUS[m.task["object"]], abs=1e-6)   # on the table


def test_robot_task_session(sessions):
    d, m = sessions["robot_task"]
    assert m.meta["synthetic"]["object_frame"] == "hand_base" and m.task["task_id"] == "pour"
    assert (d / "object_pose.npz").is_file() and (d / "joint_state.npz").is_file()
    gt = S.load_ground_truth(d)
    B = {b["name"]: b for b in m.meta["synthetic"]["blocks"]}
    man = (gt["t"] >= B["manipulate"]["t0"]) & (gt["t"] < B["manipulate"]["t1"])
    assert gt["object_contact"][man][:, [0, 1]].all(1).mean() > 0.9     # thumb–index grasp holds
    assert not gt["object_contact"][gt["t"] < B["reach"]["t1"]].any()


@pytest.mark.parametrize("key", ["glove_motion", "glove_task", "robot_motion", "robot_task"])
def test_first_no_contact_segment_is_artefact_free(sessions, key):
    """D1 and D2 (glove and robot) share one ΔS reference: the first no_contact data starts at the
    flat hand / q = 0 (zero artefact), so the preprocessing baseline (median of the first
    no_contact segment) is the true baseline and a D1-trained baseline model carries over to D2."""
    d, m = sessions[key]
    L = load_layout(m.layout)
    gt = S.load_ground_truth(d)
    z = np.load(d / "pressure.npz")
    t, raw = z["t"], L.by_channel(z["raw"])
    (a, b), = m.spans("no_contact")[:1]
    first = (t >= a) & (t < b)
    hold = first & (t < a + 0.5 * (b - a))
    assert np.abs(gt["artefact_pct"][hold]).max() < 1e-6
    np.testing.assert_allclose(np.median(raw[first], axis=0), gt["baseline_raw"], rtol=2e-3)


def test_reach_clears_the_object(tmp_path):
    """Regression: on a straight-line reach the abducted thumb of a precision pre-shape swept
    through the jar (open_jar, seed 3); the reach now travels above and descends onto it."""
    for tid in ("open_jar", "in_hand_rotate", "grasp_lift_place"):
        d = tmp_path / tid
        m = S.generate_session(d, kind="glove", dataset="task", seed=3, task_id=tid, duration_s=3.0, cameras=())
        gt = S.load_ground_truth(d)
        B = {b["name"]: b for b in m.meta["synthetic"]["blocks"]}
        for ph in ("baseline", "reach", "retreat", "baseline_end"):
            sel = (gt["t"] >= B[ph]["t0"]) & (gt["t"] < B[ph]["t1"])
            assert not gt["object_contact"][sel].any(), (tid, ph)


# ── determinism / speed / dataset ──────────────────────────────────────────
def _all_arrays(d: Path) -> dict[str, np.ndarray]:
    out = {}
    for f in sorted(d.rglob("*.np[yz]")):
        if f.suffix == ".npz":
            with np.load(f) as z:
                out.update({f"{f.relative_to(d)}:{k}": z[k] for k in z.files})
        else:
            out[str(f.relative_to(d))] = np.load(f)
    return out


def test_deterministic_under_seed(tmp_path):
    kw = dict(kind="glove", dataset="task", seed=21, cameras=("ego",), duration_s=4.0)
    m1 = S.generate_session(tmp_path / "a", **kw)
    m3 = S.generate_session(tmp_path / "c", **{**kw, "seed": 22})      # in between: no hidden state
    m2 = S.generate_session(tmp_path / "b", **kw)
    a, b = _all_arrays(tmp_path / "a"), _all_arrays(tmp_path / "b")
    assert a.keys() == b.keys() and len(a) > 20
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    assert (tmp_path / "a" / S.EVENTS_FILE).read_text() == (tmp_path / "b" / S.EVENTS_FILE).read_text()
    d1, d2 = m1.to_dict(), m2.to_dict()
    d1.pop("created_utc"), d2.pop("created_utc")
    assert d1 == d2
    assert not np.array_equal(np.load(tmp_path / "c" / "pressure.npz")["raw"], a["pressure.npz:raw"])
    assert m3.session_id != m1.session_id


def test_generation_is_fast(sessions, tmp_path):
    t0 = time.perf_counter()
    S.generate_session(tmp_path / "fast", kind="glove", dataset="motion", duration_s=6.0, seed=99)
    assert time.perf_counter() - t0 < 2.0


def test_generate_dataset_layout(tmp_path):
    dirs = S.generate_dataset(tmp_path, n_motion=2, n_task=3, kind="robot", subjects=("s0", "s1"), seed=1,
                              duration_s=3.0, cameras=())
    assert len(dirs) == 5 and len({p.name for p in dirs}) == 5
    tasks, seeds = [], set()
    for p in dirs:
        m = SessionManifest.load(p)
        assert p == tmp_path / m.dataset / m.subject / m.session_id
        assert m.kind == "robot" and m.meta["duration_s"] == 3.0
        seeds.add(m.meta["seed"])
        if m.dataset == "task":
            tasks.append(m.task["task_id"])
    assert [p.parent.parent.name for p in dirs] == ["motion"] * 2 + ["task"] * 3
    assert [p.parent.name for p in dirs] == ["s0", "s1", "s0", "s1", "s0"]
    assert len(seeds) == 5 and tasks == sorted(S.SYNTH_TASKS)[:3]


def test_shared_glove_physics(tmp_path):
    """One physical glove across sessions: ``glove_seed`` fixes the per-taxel skin parameters (artefact
    gains / signs / lags, press model, baselines, noise levels) while motion, drift and noise stay
    per session; ``generate_dataset`` shares one glove by default."""
    skin = ("angle_gain", "vel_gain", "quad_gain", "tau_s", "press_gain", "press_max_pct", "press_tau_s",
            "noise_std_pct", "baseline_raw_all")
    kw = dict(kind="glove", dataset="motion", duration_s=2.5, cameras=())
    for i, (seed, subj) in enumerate(((1, "s0"), (2, "s1"))):
        S.generate_session(tmp_path / f"g{i}", seed=seed, subject=subj, glove_seed=7, **kw)
    S.generate_session(tmp_path / "own", seed=2, subject="s1", **kw)
    a, b, own = (S.load_ground_truth(tmp_path / n) for n in ("g0", "g1", "own"))
    for k in skin:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    assert not np.array_equal(a["angle_gain"], own["angle_gain"])
    assert not np.array_equal(a["drift_pct"][:100], b["drift_pct"][:100])      # per-session drift / noise
    assert not np.array_equal(a["joint_angle"][:300], b["joint_angle"][:300])  # per-session motion
    # same session seed, legacy (no glove seed) vs a glove seed: identical motion, different skin
    np.testing.assert_array_equal(b["joint_angle"], own["joint_angle"])
    assert SessionManifest.load(tmp_path / "g1").meta["synthetic"]["glove_seed"] == 7
    assert SessionManifest.load(tmp_path / "own").meta["synthetic"]["glove_seed"] is None
    # generate_dataset: one glove (glove_seed = seed) unless shared_glove=False; an explicit glove_seed wins
    dd = dict(n_motion=2, n_task=1, kind="glove", seed=3, duration_s=3.0, cameras=())
    shared = [S.load_ground_truth(p) for p in S.generate_dataset(tmp_path / "ds", **dd)]
    assert all(np.array_equal(shared[0][k], x[k]) for x in shared[1:] for k in skin)
    assert SessionManifest.load(tmp_path / "ds" / "motion" / "s0" / "syn_glove_motion_s0_000_30021"
                                ).meta["synthetic"]["glove_seed"] == 3
    own_ds = [S.load_ground_truth(p) for p in S.generate_dataset(tmp_path / "ds2", shared_glove=False, **dd)]
    assert not np.array_equal(own_ds[0]["angle_gain"], own_ds[1]["angle_gain"])
    pinned = S.generate_dataset(tmp_path / "ds3", glove_seed=7, **{**dd, "n_task": 0})
    np.testing.assert_array_equal(S.load_ground_truth(pinned[0])["angle_gain"], a["angle_gain"])
    with pytest.raises(ValueError, match="glove_seed"):
        S.generate_session(tmp_path / "bad", glove_seed=-1, **kw)
    assert not (tmp_path / "bad").exists()


def test_air_grasp_block_is_contact_free_grasp_shape(tmp_path):
    """Glove D1 mirrors the protocol's ``air_grasp`` blocks: the D2 grasp shape held in the air —
    labelled no_contact, geometrically contact-free, fingers well flexed. Sessions ≤ 6 s keep their
    earlier plan (the block comes after the first four)."""
    assert "air_grasp" not in {b.gen for b in S.plan_motion_session(6.0, "glove")}
    m = S.generate_session(tmp_path / "ag", kind="glove", dataset="motion", duration_s=8.0, seed=4, cameras=())
    gt = S.load_ground_truth(tmp_path / "ag")
    (blk,) = [b for b in m.meta["synthetic"]["blocks"] if b["gen"] == "air_grasp"]
    assert blk["name"] == "air_grasp_slow_power" and blk["contact"] == "none" and blk["labels"] == ["no_contact"]
    inb = (gt["t"] >= blk["t0"]) & (gt["t"] < blk["t1"])
    assert not gt["contact"][inb].any() and not gt["self_touch"][inb].any()
    assert np.all(np.abs(gt["press_pct"][inb]) < S.PRESS_EPS_PCT)
    nc = _in_spans(gt["t"], m.spans("no_contact"))
    assert nc[inb].all()                                             # the whole block is a no_contact segment
    # a grasp, not a rest pose: flexion well above the relaxed start pose, artefact clearly excited
    rest = _in_spans(gt["t"], [(s["t0"], s["t1"]) for s in m.segments if s["label"] == "no_contact"][:1])
    theta = gt["joint_angle"]
    assert theta[inb].max() > theta[rest].max() + 0.5
    assert np.abs(gt["artefact_pct"][inb]).max() > 1.0
    ev = [e for e in _events(tmp_path / "ag") if e["type"] == "phase_start" and e["name"] == blk["name"]]
    assert ev and ev[0]["value"]["contact"] == "none"


def test_input_validation(tmp_path):
    with pytest.raises(ValueError):
        S.generate_session(tmp_path / "x", kind="bench")
    with pytest.raises(ValueError):
        S.generate_session(tmp_path / "x", dataset="other")
    with pytest.raises(ValueError):
        S.generate_session(tmp_path / "x", duration_s=1.0)
    with pytest.raises(ValueError):
        S.generate_session(tmp_path / "x", kind="glove", layout="robot_hand_template")
    with pytest.raises(ValueError):
        S.generate_session(tmp_path / "x", kind="robot", layout="glove_template")
    with pytest.raises(ValueError):
        S.generate_session(tmp_path / "x", dataset="task", task_id="juggle")
    with pytest.raises(ValueError):
        S.generate_session(tmp_path / "x", params={"not_a_knob": 1})
    with pytest.raises(ValueError):
        S.generate_session(tmp_path / "x", cameras=("ego", "ego"))
    with pytest.raises(ValueError):
        S.generate_session(tmp_path / "x", subject="a/b")
    for bad in ({"pressure_hz": 0.0}, {"noise_pct": [0.2, 0.1]}, {"sat_prob": 1.5}, {"artefact_tau_s": [0.0, 0.1]},
                {"drift_pct": -1.0}, {"hand_label_noise": [0.01, 0.02]}, {"jitter": float("nan")}):
        with pytest.raises(ValueError):
            S.SynthParams.from_dict(bad)
    assert S.SynthParams.from_dict({"noise_pct": [0.01, 0.02]}).noise_pct == (0.01, 0.02)
    with pytest.raises(ValueError):
        S.generate_dataset(tmp_path / "ds", n_motion=1, n_task=0, seed=1, session_id="fixed")
    with pytest.raises(ValueError):
        S.generate_dataset(tmp_path / "ds", n_motion=-1)
    assert not (tmp_path / "x").exists() and not (tmp_path / "ds").exists()   # validated before writing
    S.generate_session(tmp_path / "y", duration_s=2.5, cameras=())
    with pytest.raises(FileExistsError):
        S.generate_session(tmp_path / "y", duration_s=2.5, cameras=())
    S.generate_session(tmp_path / "y", duration_s=2.5, cameras=(), overwrite=True)


def test_layout_yaml_path_is_recorded_absolute(tmp_path, monkeypatch):
    """A layout given as a (relative) YAML path is stored absolute, so the session loads from any cwd."""
    import yaml

    (tmp_path / "lay").mkdir()
    (tmp_path / "lay" / "my_glove.yaml").write_text(yaml.safe_dump(load_layout("glove_template").to_dict(units="mm")))
    monkeypatch.chdir(tmp_path / "lay")
    m = S.generate_session(tmp_path / "s", layout="my_glove.yaml", duration_s=2.5, cameras=())
    monkeypatch.chdir(tmp_path)
    assert Path(m.layout).is_absolute() and load_layout(SessionManifest.load(tmp_path / "s").layout).n == 9
    assert S.generate_session(tmp_path / "b", duration_s=2.5, cameras=()).layout == "glove_template"
