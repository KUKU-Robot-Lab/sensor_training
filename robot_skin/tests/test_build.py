"""datasets.build: raw synthetic sessions (datasets.synthetic) → Episodes, checked against the
generator's ground truth (gt_synthetic.npz)."""
import json
import shutil

import numpy as np
import pytest
import torch
import yaml

from common.layouts import layout_from_dict, load_layout
from robot_skin.acquisition.manifest import SessionManifest
from robot_skin.datasets import build as B
from robot_skin.datasets import episode as E
from robot_skin.datasets.synthetic import generate_session, load_ground_truth, robot_hand_urdf
from robot_skin.geometry.rotations import aa_to_matrix, quat_to_matrix
from robot_skin.pose.imu_model import imu_site_quats
from robot_skin.pose.mano import ManoSkeleton
from robot_skin.pose.urdf import URDFModel
from robot_skin.pose.vision_hand import load_hand_labels, save_hand_labels

HZ = 200.0
PERM = [7, 3, 10, 0, 5, 1, 8, 2, 11]          # layout taxel i → raw channel PERM[i] (12 channels)


def _permuted_glove_layout():
    d = load_layout("glove_template").to_dict(units="mm")
    d["name"] = "glove_permuted"
    for tx, ch in zip(d["taxels"], PERM):
        tx["channel"] = ch
    return layout_from_dict(d)


def _nearest(t_src, t):
    k = np.clip(np.searchsorted(t_src, t), 1, len(t_src) - 1)
    return np.where(np.abs(t_src[k - 1] - t) <= np.abs(t_src[k] - t), k - 1, k)


def _gt_at(g, key, t):
    """Ground truth (pressure timestamps) linearly interpolated onto t."""
    x = np.asarray(g[key], dtype=np.float64)
    flat = x.reshape(len(g["t"]), -1)
    out = np.stack([np.interp(t, g["t"], flat[:, j]) for j in range(flat.shape[1])], 1)
    return out.reshape((len(t),) + x.shape[1:])


@pytest.fixture(scope="module")
def glove_motion(tmp_path_factory):
    """Glove D1 session with a channel-permuted Layout object, extra ADC channels and forced
    saturation; the session dir is moved after generation (paths must resolve session-relative)."""
    root = tmp_path_factory.mktemp("gm")
    L = _permuted_glove_layout()
    generate_session(root / "orig", kind="glove", dataset="motion", duration_s=4.0, seed=3,
                     cameras=("ego", "third"), layout=L, n_channels=12, params={"sat_prob": 1.0})
    shutil.move(str(root / "orig"), str(root / "moved"))
    sdir = root / "moved"
    ep = B.preprocess_session(sdir, root / "processed")
    return sdir, ep, L


@pytest.fixture(scope="module")
def robot_motion(tmp_path_factory):
    """Robot D1 session whose joint_state columns are shuffled (driver order ≠ URDF order)."""
    root = tmp_path_factory.mktemp("rm")
    generate_session(root / "orig", kind="robot", dataset="motion", duration_s=4.0, seed=5, cameras=("ego",))
    js = dict(np.load(root / "orig" / "joint_state.npz"))
    perm = np.random.default_rng(0).permutation(js["q"].shape[1])
    np.savez(root / "orig" / "joint_state.npz", t=js["t"], q=js["q"][:, perm], qd=js["qd"][:, perm],
             names=js["names"][perm])
    shutil.move(str(root / "orig"), str(root / "moved"))
    sdir = root / "moved"
    ep = B.preprocess_session(sdir, root / "processed", {"baseline": {"duration_s": 0.3}})
    return sdir, ep


def test_config_yaml_mirrors_defaults_and_rejects_typos():
    assert yaml.safe_load(B.CONFIG_PATH.read_text()) == B.DEFAULTS
    cfg = B.load_preprocess_config(B.CONFIG_PATH, {"baseline": {"duration_s": 2.0}})
    assert cfg["baseline"]["duration_s"] == 2.0 and cfg["qd"]["method"] == "savgol_causal"
    with pytest.raises(ValueError, match="unknown keys"):
        B.load_preprocess_config(overrides={"baseline": {"durration_s": 1.0}})
    with pytest.raises(ValueError, match="unknown preprocess config keys"):
        B.load_preprocess_config(overrides={"basline": {}})
    with pytest.raises(ValueError, match="qd.method"):
        B.load_preprocess_config(overrides={"qd": {"method": "spline"}})


def test_glove_master_clock_layout_order_and_delta(glove_motion):
    sdir, ep, L = glove_motion
    g = load_ground_truth(sdir)
    t = ep.t
    # uniform 200 Hz grid on the session clock, inside the pressure span
    np.testing.assert_allclose(np.diff(t), 1.0 / HZ, atol=1e-9)
    np.testing.assert_allclose(t * HZ, np.round(t * HZ), atol=1e-6)
    assert g["t"][0] <= t[0] and t[-1] <= g["t"][-1]
    assert ep.root == sdir.parent / "processed" / "motion" / ep.meta.episode_id
    # layout order restored from the channel-major raw (custom permuted Layout object)
    np.testing.assert_array_equal(ep.static[E.S_CHANNELS], PERM)
    np.testing.assert_array_equal(g["channels"], PERM)
    raw = dict(np.load(sdir / "pressure.npz"))["raw"]
    assert raw.shape[1] == 12
    k = _nearest(g["t"], t)
    exact = np.abs(g["t"][k] - t) < 1e-9
    np.testing.assert_allclose(ep[E.K_PRESSURE_RAW][exact], raw[k[exact]][:, PERM])
    # baseline from the first no_contact segment (flat hand → artefact-free) ≈ true baseline
    b = ep.meta.preprocessing["baseline"]
    assert b["source"] == "no_contact_segment" and b["segment"][0] == 0.0
    # (within the slow drift + noise of the generator over that segment: < 0.1 %)
    np.testing.assert_allclose(ep.static[E.S_BASELINE_RAW], g["baseline_raw"], rtol=1e-3)
    # ΔS (SATS sign) = ground truth artefact + press + drift + noise, up to the baseline estimate
    sat = np.asarray(ep[E.K_SATURATED])
    gt_sat = _gt_at(g, "saturated", t) > 0
    target = _gt_at(g, "delta_true_pct", t) + _gt_at(g, "noise_pct", t)
    ok = ~sat & ~gt_sat
    err = np.abs(np.asarray(ep[E.K_DELTA], np.float64) - target)[ok]
    assert np.median(err) < 0.02 and np.percentile(err, 99) < 0.1
    rep = ep.meta.preprocessing["synthetic_gt"]
    assert rep["layout_order_match"] and rep["delta_abs_err_pct"]["median"] < 0.02
    # press → negative ΔS; ground-truth rail samples are flagged saturated
    contact = (_gt_at(g, "contact", t) > 0.99) & ok
    assert contact.sum() > 50 and np.asarray(ep[E.K_DELTA])[contact].mean() < -2.0
    assert g["saturated"].any() and sat[gt_sat].all()
    assert rep["saturated_recall"] == 1.0
    assert np.asarray(ep[E.K_DELTA])[sat].min() < -50            # dropouts toward −100 %
    assert ep[E.K_DELTA].dtype == np.float32 and ep[E.K_PRESSURE_RAW].dtype == np.float64
    # optional ground-truth arrays on the master clock
    np.testing.assert_allclose(ep["gt_artefact_pct"], _gt_at(g, "artefact_pct", t), atol=1e-4)
    assert ep["gt_contact"].dtype == bool


def test_glove_imu_calibration_and_quaternions(glove_motion):
    sdir, ep, L = glove_motion
    g = load_ground_truth(sdir)
    info = ep.meta.preprocessing["imu"]
    assert info["calibrated"] and info["calibration_source"] == "manifest" and info["wrist_index"] == 0
    assert ep.meta.imu_sites == [s.name for s in L.imu_sites]
    q = np.asarray(ep[E.K_IMU_QUAT], dtype=np.float64)
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-5)
    assert (np.sum(q[1:] * q[:-1], -1) > 0).all()                       # continuity
    with torch.no_grad():
        q_ref = imu_site_quats(L, ManoSkeleton(), g["hand_global_orient"], g["hand_finger_pose"])
    k = _nearest(g["t"], ep.t)

    def ang(qa):
        return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(qa * q_ref[k], -1)), 0, 1)))

    err = ang(q)
    assert np.median(err) < 1.0 and np.percentile(err, 99) < 4.0
    # without the manifest offsets the quaternions stay in the raw sensor / IMU-world frames
    raw = B.preprocess_session(sdir, None, {"imu": {"apply_calibration": False}, "synthetic_gt": {"check": False}})
    assert not raw.meta.preprocessing["imu"]["calibrated"]
    assert np.median(ang(np.asarray(raw[E.K_IMU_QUAT], dtype=np.float64))) > 3.0
    # gyro/acc rotated into segment frames: static flat-hand acc ≈ Rᵀ·(0, 0, 9.81)
    calib = ep.phase_mask("imu_calibration")
    R = quat_to_matrix(torch.as_tensor(q[calib])).numpy()
    g_s = np.einsum("tsji,j->tsi", R, np.array([0.0, 0.0, 9.81]))
    assert np.median(np.linalg.norm(np.asarray(ep[E.K_IMU_ACC])[calib] - g_s, axis=-1)) < 0.3


def test_glove_hand_pose_q_taxels_self_touch_labels(glove_motion):
    sdir, ep, L = glove_motion
    g = load_ground_truth(sdir)
    t = ep.t
    valid = np.asarray(ep[E.K_HAND_VALID])
    assert valid.mean() > 0.9 and ep.meta.preprocessing["q_source"] == "hand_pose"
    np.testing.assert_array_equal(ep[E.K_Q], np.asarray(ep[E.K_HAND_FINGERS]).reshape(ep.T, 45))
    assert ep.meta.joint_names == list(B.HAND_Q_NAMES) and ep.meta.joint_names[0] == "index1_x"
    fp_gt = _gt_at(g, "hand_finger_pose", t)
    assert np.abs(np.asarray(ep[E.K_HAND_FINGERS]) - fp_gt)[valid].mean() < 0.03
    # qd = smoothed derivative of q
    fd = np.gradient(np.asarray(ep[E.K_Q], np.float64), 1.0 / HZ, axis=0)
    assert np.corrcoef(fd[10:-10].ravel(), np.asarray(ep[E.K_QD])[10:-10].ravel())[0, 1] > 0.95
    # taxel poses from the (noisy, smoothed) vision labels ≈ ground truth, in the hand (MANO wrist)
    # frame of the Episode contract: world poses with the true wrist rotation / position removed
    k = _nearest(g["t"], t)
    R = aa_to_matrix(torch.as_tensor(g["hand_global_orient"][k], dtype=torch.float64)).numpy()
    p_hand = np.einsum("tji,tnj->tni", R, g["taxel_pos"][k] - g["hand_wrist_pos"][k][:, None])
    d = np.linalg.norm(np.asarray(ep[E.K_TAXEL_POS]) - p_hand, axis=-1)
    assert ep.meta.preprocessing["taxel_pose_source"] == "hand_pose" and np.median(d) < 0.005
    assert ep.meta.preprocessing["taxel_frame"] == "mano_wrist"
    np.testing.assert_allclose(np.linalg.norm(ep[E.K_TAXEL_NRM], axis=-1), 1.0, atol=1e-5)
    # geometric self-touch vs generator truth (same capsule model, noisy labels)
    st, gst = np.asarray(ep[E.K_SELF_TOUCH]), _gt_at(g, "self_touch", t) > 0.5
    assert gst.sum() > 50 and (st & gst).sum() / (st | gst).sum() > 0.8
    # contact_label policy: 0 only where truly contact-free, 1 = self-touch, −1 elsewhere
    lab = np.asarray(ep[E.K_CONTACT_LABEL])
    gc = _gt_at(g, "contact", t) > 0
    assert lab.dtype == np.int8 and set(np.unique(lab)) <= {-1, 0, 1}
    assert not gc[lab == 0].any() and gst[lab == 1].mean() > 0.95
    assert (lab[st] == 1).all() or ep.meta.preprocessing["contact_label"]["n_conflict"] > 0
    nc = np.zeros(ep.T, bool)
    for s in ep.meta.preprocessing["segments"]:
        if s["label"] == "no_contact":
            nc |= (t >= s["t0"]) & (t < s["t1"])
    assert ((lab == 0) | st)[nc].all() and (lab[~nc[:, None] & ~st] == -1).all()


def test_glove_phases_cameras_and_files(glove_motion):
    sdir, ep, L = glove_motion
    events = [json.loads(ln) for ln in (sdir / "events.jsonl").read_text().splitlines()]
    starts = [e for e in events if e["type"] == "phase_start"]
    assert ep.meta.phase_names == list(dict.fromkeys(e["name"] for e in starts))
    for e in starts:
        end = next(x for x in events if x["type"] == "phase_end" and x["name"] == e["name"] and x["t"] >= e["t"])
        inside = (ep.t >= e["t"]) & (ep.t < end["t"])
        assert (ep[E.K_PHASE][inside] == ep.meta.phase_names.index(e["name"])).all()
    assert ep[E.K_PHASE].dtype == np.int16
    assert ep.meta.phases[0] == {"name": "imu_calibration", "t0": 0.0, "t1": ep.meta.phases[0]["t1"],
                                 "contact": "none", "labels": ["calibration", "no_contact"]}
    # cameras: latest frame with timestamp ≤ t, −1 before the first frame; dir linked into the episode
    assert ep.meta.cameras == ["ego", "third"]
    loaded = E.Episode.load(ep.root)
    for cam in ("ego", "third"):
        idx = np.asarray(loaded[E.cam_idx_key(cam)])
        ts = np.load(sdir / f"camera_{cam}" / "timestamps.npy")
        assert idx.dtype == np.int32 and (np.diff(idx) >= 0).all()
        np.testing.assert_array_equal(idx, np.searchsorted(ts, ep.t, side="right") - 1)
        assert (idx[ep.t < ts[0]] == -1).all() and (idx[ep.t >= ts[0]] >= 0).all()
        i = int(np.argmax(idx >= 0))
        np.testing.assert_array_equal(loaded.frame_at(cam, i), np.load(sdir / f"camera_{cam}" / "frames.npy")[idx[i]])
    assert (ep.root / "camera_ego").is_symlink()
    # episode is self-contained: layout copy, moved raw session resolved, meta json-clean
    assert B.load_episode_layout(loaded).channels.tolist() == PERM
    assert loaded.meta.layout == str(ep.root / B.LAYOUT_FILE)
    assert loaded.meta.source_session == str(sdir.resolve()) and loaded.meta.task is None
    assert loaded.meta.preprocessing["version"] == B.PREPROCESS_VERSION
    assert set(loaded.arrays) >= {E.K_T, E.K_PRESSURE_RAW, E.K_DELTA, E.K_SATURATED, E.K_TAXEL_POS, E.K_TAXEL_NRM,
                                  E.K_Q, E.K_QD, E.K_IMU_QUAT, E.K_IMU_GYRO, E.K_IMU_ACC, E.K_HAND_GLOBAL,
                                  E.K_HAND_FINGERS, E.K_HAND_WRIST, E.K_HAND_VALID, E.K_PHASE, E.K_SELF_TOUCH,
                                  E.K_CONTACT_LABEL}
    with pytest.raises(FileExistsError):
        B.preprocess_session(sdir, ep.root.parent.parent)


def test_glove_d2_taxel_poses_stay_in_the_hand_frame(tmp_path):
    """Regression (Episode contract K_TAXEL_POS: hand frame): while the wrist travels tens of cm
    through a D2 task, the stored glove taxel centroid moves only a few cm (finger motion); the
    world poses needed for hand–object proximity are R(hand_global_orient)·taxel_pos + hand_wrist_pos."""
    from robot_skin.contact.pseudo_label import taxel_world_positions

    sdir = tmp_path / "task"
    generate_session(sdir, kind="glove", dataset="task", duration_s=4.0, seed=2, cameras=(),
                     task_id="grasp_lift_place")
    ep = B.preprocess_session(sdir, None, {"baseline": {"duration_s": 0.2}, "synthetic_gt": {"check": False}})
    g = load_ground_truth(sdir)
    pre = ep.meta.preprocessing
    assert pre["taxel_frame"] == "mano_wrist" and pre["taxel_pose_source"] == "hand_pose"
    assert pre["version"] == B.PREPROCESS_VERSION == "robot_skin.datasets.build/2"
    centroid = np.asarray(ep[E.K_TAXEL_POS], dtype=np.float64).mean(1)
    wrist = np.asarray(ep[E.K_HAND_WRIST], dtype=np.float64)
    assert np.ptp(wrist, axis=0).max() > 0.25                        # the hand moves through the room …
    assert np.ptp(centroid, axis=0).max() < 0.04                     # … its taxels stay put in the hand frame
    assert np.linalg.norm(centroid - np.median(centroid, 0), axis=1).max() < 0.04
    # self-touch is frame-invariant; world poses are recovered from the hand pose (label noise ≈ mm)
    world, valid = taxel_world_positions(ep)
    assert valid.mean() > 0.9
    err = np.linalg.norm(world - _gt_at(g, "taxel_pos", ep.t), axis=-1)[valid]
    assert np.median(err) < 0.005 and np.percentile(err, 95) < 0.012
    assert SessionManifest.load(sdir).meta["synthetic"]["gt_taxel_frame"] == "world"   # gt taxel_pos: world


def test_glove_task_meta_object_and_labels(tmp_path):
    sdir = tmp_path / "task"
    man = generate_session(sdir, kind="glove", dataset="task", duration_s=4.0, seed=2, cameras=(), task_id="pour")
    ep = B.preprocess_session(sdir, tmp_path / "out",
                              {"baseline": {"duration_s": 0.2}, "cameras": {"copy_frames": "copy"}})
    g = load_ground_truth(sdir)
    assert ep.root == tmp_path / "out" / "task" / man.session_id
    assert ep.meta.task["instruction"] == man.task["instruction"]
    assert ep.meta.task["instruction"].startswith(f"pour the {man.task['object']} into the ")
    assert ep.meta.task["success"] == man.task["success"] and ep.meta.instruction == man.task["instruction"]
    assert ep.meta.phase_names == ["baseline", "reach", "grasp", "manipulate", "release", "retreat", "baseline_end"]
    assert ep.has(E.K_OBJECT_POS) and ep.has(E.K_OBJECT_QUAT) and ep.meta.cameras == []
    np.testing.assert_allclose(ep[E.K_OBJECT_POS], _gt_at(g, "object_pos", ep.t), atol=0.01)
    np.testing.assert_allclose(np.linalg.norm(ep[E.K_OBJECT_QUAT], axis=-1), 1.0, atol=1e-5)
    # D2 object contact is never labelled here: grasp frames without self-touch stay unknown
    lab = np.asarray(ep[E.K_CONTACT_LABEL])
    grasp = ep.phase_mask("grasp") | ep.phase_mask("manipulate")
    assert (lab[grasp][~np.asarray(ep[E.K_SELF_TOUCH])[grasp]] == -1).all()
    assert not (_gt_at(g, "contact", ep.t) > 0)[lab == 0].any()
    # short baseline window at the flat-hand start → ΔS matches the ground truth tightly
    assert ep.meta.preprocessing["synthetic_gt"]["delta_abs_err_pct"]["median"] < 0.02
    # taxel poses stay in the hand frame while the hand travels through the world (reach / pour)
    travel = np.ptp(np.asarray(ep[E.K_TAXEL_POS]).mean(1), axis=0).max()
    assert np.ptp(g["taxel_pos"].mean(1), axis=0).max() > 0.2 and travel < 0.1


def test_robot_joint_reorder_urdf_fk_and_labels(robot_motion, monkeypatch):
    sdir, ep = robot_motion
    g = load_ground_truth(sdir)
    model = URDFModel.from_string(robot_hand_urdf())
    assert ep.meta.joint_names == list(model.joint_names) and ep.meta.kind == "robot"
    assert ep.meta.preprocessing["q_source"] == "joint_state"
    assert ep.meta.preprocessing["urdf"] == str(sdir.resolve() / "robot.urdf")          # session-relative
    np.testing.assert_allclose(ep[E.K_Q], _gt_at(g, "q", ep.t), atol=4e-3)
    fd = np.gradient(_gt_at(g, "q", ep.t), 1.0 / HZ, axis=0)
    assert np.corrcoef(fd.ravel(), np.asarray(ep[E.K_QD]).ravel())[0, 1] > 0.95
    assert ep.meta.preprocessing["taxel_pose_source"] == "urdf"
    d = np.linalg.norm(np.asarray(ep[E.K_TAXEL_POS]) - _gt_at(g, "taxel_pos", ep.t), axis=-1)
    assert np.percentile(d, 99) < 0.002
    assert not ep.has(E.K_SELF_TOUCH) and not ep.has(E.K_IMU_QUAT) and not ep.has(E.K_HAND_VALID)
    lab = np.asarray(ep[E.K_CONTACT_LABEL])
    assert set(np.unique(lab)) == {-1, 0} and not (_gt_at(g, "contact", ep.t) > 0)[lab == 0].any()
    assert ep.meta.preprocessing["synthetic_gt"]["delta_abs_err_pct"]["median"] < 0.02
    # no URDF → q in driver order, static taxel poses (noted)
    (sdir / "robot.urdf").rename(sdir / "robot.urdf.bak")
    try:
        ep2 = B.preprocess_session(sdir, None)
    finally:
        (sdir / "robot.urdf.bak").rename(sdir / "robot.urdf")
    names = [str(n) for n in np.load(sdir / "joint_state.npz")["names"]]
    assert ep2.meta.joint_names == names and ep2.meta.preprocessing["taxel_pose_source"] == "static"
    assert any("URDF" in n for n in ep2.meta.preprocessing["notes"])
    # an explicit cfg robot.urdf: CWD-relative (or absolute) paths resolve; a missing one is an
    # error instead of silently falling back to driver-order q and static taxel poses
    (sdir.parent / "urdfs").mkdir(exist_ok=True)
    shutil.copy(sdir / "robot.urdf", sdir.parent / "urdfs" / "hand.urdf")
    monkeypatch.chdir(sdir.parent)
    ep3 = B.preprocess_session(sdir, None, {"robot": {"urdf": "urdfs/hand.urdf"}, "baseline": {"duration_s": 0.3}})
    assert ep3.meta.preprocessing["urdf"] == str((sdir.parent / "urdfs" / "hand.urdf").resolve())
    assert ep3.meta.joint_names == list(model.joint_names) and ep3.meta.preprocessing["taxel_frame"] == "urdf_root"
    np.testing.assert_array_equal(ep3[E.K_TAXEL_POS], ep[E.K_TAXEL_POS])
    with pytest.raises(FileNotFoundError, match="robot.urdf"):
        B.preprocess_session(sdir, None, {"robot": {"urdf": "urdfs/typo.urdf"}})


def test_hand_pose_confidence_gate(tmp_path):
    sdir = tmp_path / "s"
    generate_session(sdir, kind="glove", dataset="motion", duration_s=3.0, seed=1, cameras=())
    hl = load_hand_labels(sdir)
    conf = hl["confidence"].copy()
    gap = (hl["t"] > 1.0) & (hl["t"] < 1.6)                           # 0.6 s > max_gap_s → invalid
    conf[gap] = 0.05
    conf[(hl["t"] > 2.0) & (hl["t"] < 2.1)] = 0.05                    # short gap → SLERP-filled, valid
    save_hand_labels(sdir, hl["t"], hl["global_orient"], hl["finger_pose"], hl["wrist_pos"], conf)
    ep = B.preprocess_session(sdir, None, {"synthetic_gt": {"check": False, "store": False}})
    v, t = np.asarray(ep[E.K_HAND_VALID]), ep.t
    assert not v[(t > 1.05) & (t < 1.55)].any()
    assert v[(t > 1.95) & (t < 2.15)].all() and v[(t > 0.2) & (t < 0.9)].mean() > 0.9
    assert not np.asarray(ep[E.K_SELF_TOUCH])[~v].any()
    assert "gt_contact" not in ep.arrays and "synthetic_gt" not in ep.meta.preprocessing


def test_layout_object_override_and_empty_hand_labels(tmp_path):
    sdir = tmp_path / "s"
    generate_session(sdir, kind="glove", dataset="motion", duration_s=2.5, seed=6, cameras=())
    L = load_layout("glove_template")
    ep = B.preprocess_session(sdir, tmp_path / "out", {"layout": L, "synthetic_gt": {"store": False}})
    pre = ep.meta.preprocessing
    assert pre["layout_source"] == "override:glove_template" and pre["config"]["layout"]["name"] == "glove_template"
    assert B.load_episode_layout(E.Episode.load(ep.root)).channels.tolist() == L.channels.tolist()
    # a vision pass that found no hand at all: episode without hand pose / q, rest-hand taxel poses
    save_hand_labels(sdir, np.zeros(0), np.zeros((0, 3)), np.zeros((0, 15, 3)), np.zeros((0, 3)))
    ep = B.preprocess_session(sdir, None)
    assert not ep.has(E.K_HAND_VALID) and not ep.has(E.K_Q) and ep.meta.preprocessing["q_source"] is None
    assert ep.meta.preprocessing["taxel_pose_source"] == "rest"
    assert any("no labels" in n for n in ep.meta.preprocessing["notes"])


def test_build_all_detects_episode_id_collisions(tmp_path):
    raw = tmp_path / "raw"
    for subj, seed in (("s0", 1), ("s1", 2)):                          # same session_id, different subjects
        generate_session(raw / "motion" / subj / "x", kind="robot", dataset="motion", duration_s=2.5, seed=seed,
                         cameras=(), session_id="dup")
    reps = B.build_all(raw, tmp_path / "proc", {"synthetic_gt": {"check": False}})
    assert [r["status"] for r in reps] == ["built", "failed"] and "collision" in reps[1]["error"]
    ep = E.Episode.load(tmp_path / "proc" / "motion" / "dup")
    assert ep.meta.source_session == reps[0]["session"]                  # the first one was not overwritten


def test_imu_calibration_from_phase_when_manifest_lacks_it(tmp_path):
    sdir = tmp_path / "s"
    generate_session(sdir, kind="glove", dataset="motion", duration_s=3.0, seed=4, cameras=())
    m = SessionManifest.load(sdir)
    m.calibration = {}
    m.save(sdir)
    g = load_ground_truth(sdir)
    with torch.no_grad():
        q_ref = imu_site_quats(load_layout("glove_template"), ManoSkeleton(), g["hand_global_orient"],
                               g["hand_finger_pose"])

    def rel_err(ep):                                              # site orientation relative to the wrist IMU
        from robot_skin.geometry.rotations import quat_conj, quat_mul

        k = _nearest(g["t"], ep.t)
        a, b = (torch.as_tensor(np.asarray(x, np.float64)) for x in (ep[E.K_IMU_QUAT], q_ref[k]))
        ra = quat_mul(quat_conj(a[:, :1]).expand_as(a), a).numpy()
        rb = quat_mul(quat_conj(b[:, :1]).expand_as(b), b).numpy()
        return np.median(np.degrees(2 * np.arccos(np.clip(np.abs((ra * rb).sum(-1)), 0, 1))))

    raw = B.preprocess_session(sdir, None)
    assert not raw.meta.preprocessing["imu"]["calibrated"] and rel_err(raw) > 5.0
    assert any("IMU calibration" in n for n in raw.meta.preprocessing["notes"])
    ep = B.preprocess_session(sdir, None, {"imu": {"calibrate_if_missing": True}})
    info = ep.meta.preprocessing["imu"]
    assert info["calibration_source"] == "phase:imu_calibration" and info["quality"]["ok"]
    assert rel_err(ep) < 1.0
    assert "imu_offsets" not in SessionManifest.load(sdir).calibration          # raw session untouched


def test_joint_velocity_modes():
    t = np.arange(400) / HZ
    q = np.stack([3.0 * t ** 2 - t, np.sin(2 * t)], 1)
    true = np.stack([6.0 * t - 1.0, 2 * np.cos(2 * t)], 1)
    for m in ("savgol", "savgol_causal"):
        qd = B.joint_velocity(q, HZ, method=m, window_s=0.05, polyorder=2)
        assert qd.dtype == np.float32 and qd.shape == q.shape
        np.testing.assert_allclose(qd[20:, 0], true[20:, 0], atol=1e-3)     # exact for quadratics
        np.testing.assert_allclose(qd[20:-20], true[20:-20], atol=1e-2)
    np.testing.assert_allclose(B.joint_velocity(q, HZ, method="gradient")[20:-20], true[20:-20], atol=1e-2)
    # causal: the value at t only depends on q[:t+1]
    qc = B.joint_velocity(q, HZ, method="savgol_causal")
    qc2 = B.joint_velocity(q[:150], HZ, method="savgol_causal")
    np.testing.assert_allclose(qc[:150], qc2, atol=1e-6)
    with pytest.raises(ValueError):
        B.joint_velocity(q, HZ, method="spline")


def test_camera_index_phase_ids_and_label_policy():
    ts = np.array([0.30, 0.10, 0.20, 0.50])                            # unsorted file order
    t = np.array([0.0, 0.1, 0.15, 0.25, 0.3, 0.45, 0.9])
    np.testing.assert_array_equal(B.camera_frame_index(ts, t), [-1, 1, 1, 2, 0, 0, 3])
    np.testing.assert_array_equal(B.camera_frame_index(ts, t, max_age_s=0.12), [-1, 1, 1, 2, 0, -1, -1])
    np.testing.assert_array_equal(B.camera_frame_index(np.zeros(0), t), -np.ones(7))
    phases = [{"name": "outer", "t0": 0.0, "t1": 1.0}, {"name": "inner", "t0": 0.2, "t1": 0.4},
              {"name": "outer2", "t0": 1.0, "t1": 2.0}]
    pid, names = B.phase_ids(phases, np.array([0.1, 0.3, 0.4, 1.0, 2.0]))
    assert names == ["outer", "inner", "outer2"] and pid.tolist() == [0, 1, 0, 2, -1]
    segs = [{"t0": 0.0, "t1": 1.0, "label": "no_contact"}, {"t0": 1.0, "t1": 2.0, "label": "self_touch"}]
    tt = np.array([0.5, 0.9, 1.5, 1.6])
    st = np.array([[False, False], [True, False], [True, False], [False, False]])
    lab, c = B.contact_labels(tt, segs, st, 2)
    assert lab.tolist() == [[0, 0], [-1, 0], [1, -1], [-1, -1]] and c["n_conflict"] == 1
    assert B.contact_labels(tt, segs, st, 2, conflict="segment")[0][1].tolist() == [0, 0]
    assert B.contact_labels(tt, segs, st, 2, conflict="geometry")[0][1].tolist() == [1, 0]
    assert B.contact_labels(tt, segs, None, 2)[0].tolist() == [[0, 0], [0, 0], [-1, -1], [-1, -1]]


def test_pressure_loader_injection_and_errors(tmp_path):
    sdir = tmp_path / "s"
    generate_session(sdir, kind="robot", dataset="motion", duration_s=2.5, seed=4, cameras=())
    z = dict(np.load(sdir / "pressure.npz"))
    calls = []

    def loader(session_dir, manifest):
        calls.append(session_dir)
        return z["t"], z["raw"]

    ref = B.preprocess_session(sdir, None)
    ep = B.preprocess_session(sdir, None, pressure_loader=loader)
    assert calls == [sdir.resolve()]
    np.testing.assert_array_equal(ep[E.K_DELTA], ref[E.K_DELTA])

    # missing samples (NaN) from a loader: bridged + flagged saturated, never a NaN ΔS or a taxel
    # whose baseline (median over the first no-contact second) is poisoned into a "dead channel"
    def nan_loader(session_dir, manifest):
        raw = z["raw"].copy()
        raw[20, 2] = np.nan                                           # inside the baseline window
        raw[300:305, 4] = np.nan
        return z["t"], raw

    en = B.preprocess_session(sdir, None, pressure_loader=nan_loader)
    d, s = np.asarray(en[E.K_DELTA]), np.asarray(en[E.K_SATURATED])
    assert np.isfinite(d).all() and "dead_taxels" not in en.meta.preprocessing
    near = np.abs(en.t[:, None] - z["t"][300:305][None]).min(1) < 0.006
    assert s[near, 4].all() and s[np.abs(en.t - z["t"][20]) < 0.004, 2].all()
    np.testing.assert_allclose(en.static[E.S_BASELINE_RAW], ref.static[E.S_BASELINE_RAW], rtol=1e-4)
    assert np.abs(d - np.asarray(ref[E.K_DELTA]))[~s].max() < 0.01
    assert any("non-finite" in n for n in en.meta.preprocessing["notes"])
    # a non-npz pressure file needs a loader (mk555 .bin → bin_merge wrapper)
    m = SessionManifest.load(sdir)
    m.streams["pressure"].file = "pressure.bin"
    m.save(sdir)
    with pytest.raises(ValueError, match="pressure loader"):
        B.preprocess_session(sdir, None)
    ep2 = B.preprocess_session(sdir, None, {"pressure": {"loader": f"{__name__}:_npz_bin_loader"}})
    np.testing.assert_array_equal(ep2[E.K_DELTA], ref[E.K_DELTA])


def _npz_bin_loader(session_dir, manifest):
    with np.load(session_dir / "pressure.npz") as z:
        return z["t"], z["raw"]


def test_cli_idempotent_force_and_failures(tmp_path, capsys):
    raw = tmp_path / "raw"
    generate_session(raw / "motion" / "s0" / "a", kind="robot", dataset="motion", duration_s=2.5, seed=1,
                     cameras=("ego",), session_id="a")
    generate_session(raw / "motion" / "s1" / "b", kind="robot", dataset="motion", duration_s=2.5, seed=2,
                     cameras=(), session_id="b")
    out = tmp_path / "proc"
    assert B.main(["--raw", str(raw), "--out", str(out), "-q"]) == 0
    assert sorted(p.name for p in E.list_episodes(out)) == ["a", "b"]
    mtime = (out / "motion" / "a" / E.EPISODE_JSON).stat().st_mtime_ns
    reps = B.build_all(raw, out)
    assert [r["status"] for r in reps] == ["skipped", "skipped"] and not any(r.get("stale") for r in reps)
    assert B.build_all(raw, out, {"qd": {"method": "gradient"}})[0].get("stale")
    (raw / "motion" / "s1" / "b" / "extra.txt").write_text("not a stream")         # ignored by the fingerprint
    assert not any(r.get("stale") for r in B.build_all(raw, out))
    z = dict(np.load(raw / "motion" / "s1" / "b" / "pressure.npz"))
    np.savez(raw / "motion" / "s1" / "b" / "pressure.npz", t_host=z["t"], **z)      # e.g. re-synced raw
    assert [bool(r.get("stale")) for r in B.build_all(raw, out)] == [False, True]
    ep = E.Episode.load(out / "motion" / "a")
    ep.set_derived(E.D_RESIDUAL, np.zeros((ep.T, ep.meta.n_taxels), np.float32))
    assert B.main(["--raw", str(raw), "--out", str(out), "--force", "--set", "qd.method=savgol", "-q"]) == 0
    ep = E.Episode.load(out / "motion" / "a")
    assert ep.meta.preprocessing["config"]["qd"]["method"] == "savgol"
    assert not ep.has_derived(E.D_RESIDUAL)                           # rebuilt from scratch
    assert (out / "motion" / "a" / E.EPISODE_JSON).stat().st_mtime_ns != mtime
    assert not list((out / "motion").glob(".*tmp*"))
    # QC gate and a broken session: reported, the others still build
    (raw / "motion" / "s1" / "b" / "qc.json").write_text(json.dumps({"passed": False}))
    bad = raw / "motion" / "s2" / "c"
    bad.mkdir(parents=True)
    (bad / "session.json").write_text(json.dumps({"kind": "robot", "layout": "robot_hand_template",
                                                  "session_id": "c", "dataset": "motion", "streams": {}}))
    reps = {r["session"].rsplit("/", 1)[-1]: r for r in
            B.build_all(raw, out, {"qc": {"skip_failed": True}}, force=True)}
    assert reps["a"]["status"] == "built" and reps["b"]["status"] == "qc_failed"
    assert reps["c"]["status"] == "failed" and "pressure" in reps["c"]["error"]
    assert B.main(["--raw", str(raw), "--out", str(out), "-q", "--json", str(tmp_path / "r.json")]) == 1
    assert len(json.loads((tmp_path / "r.json").read_text())) == 3
    assert "summary:" in capsys.readouterr().out
    with pytest.raises(ValueError):
        B.build_all(raw, out, force=True, fail_fast=True)
    assert B.main(["--raw", str(tmp_path / "typo"), "--out", str(out), "-q"]) == 2   # typo ≠ empty success
    assert "not found" in capsys.readouterr().err
    with pytest.raises(FileNotFoundError, match="does not exist"):
        B.build_all(tmp_path / "typo", out)


def test_acquisition_fake_d2_session(tmp_path):
    """A session recorded by the acquisition stack (--fake, 3-tap sync, IMU calibration) preprocesses:
    protocol phases (baseline, imu_calibration, sync_start, task phases), rest-pose baseline, calibrated
    IMU, sync taps never labelled no-contact, ACQ task metadata passed through."""
    from robot_skin.acquisition import glove_logger

    out = tmp_path / "raw"
    assert glove_logger.main(["--protocol", "d2_task", "--fake", "--time-scale", "0.05", "--out", str(out),
                              "--subject", "S01", "--cameras", "ego", "--episodes", "1", "--no-qc"]) == 0
    (sdir,) = B.find_sessions(out)
    man = SessionManifest.load(sdir)
    ep = B.preprocess_session(sdir, tmp_path / "proc")
    pre = ep.meta.preprocessing
    assert ep.meta.dataset == "task" and ep.meta.cameras == ["ego"] and ep.has(E.cam_idx_key("ego"))
    assert pre["baseline"]["source"] == "no_contact_segment"
    assert pre["baseline"]["segment"][0] == pytest.approx(man.spans("no_contact")[0][0])
    assert pre["imu"]["calibrated"] and pre["imu"]["calibration_source"] == "manifest"
    names = ep.meta.phase_names
    assert names[:3] == ["baseline", "imu_calibration", "sync_start"] and "manipulate" in names
    sync = next(p for p in ep.meta.phases if p["name"] == "sync_start")
    assert sync["contact"] == "any" and sync["labels"] == ["sync"]
    assert (np.asarray(ep[E.K_CONTACT_LABEL])[ep.phase_mask("sync_start")] != 0).all()
    assert ep.meta.task["instruction"] == man.task["instruction"] and ep.meta.task["template"] == man.task["template"]
    assert np.isfinite(np.asarray(ep[E.K_DELTA])).all() and ep.has(E.K_HAND_VALID)
