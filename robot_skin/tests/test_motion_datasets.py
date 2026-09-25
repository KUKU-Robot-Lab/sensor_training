"""datasets.motion: causal D1 windows (baseline / contact / IMU pose) over processed episodes."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from robot_skin.datasets import build as B
from robot_skin.datasets import episode as E
from robot_skin.datasets import motion as M
from robot_skin.datasets import stats as ST
from robot_skin.datasets.synthetic import generate_session, load_ground_truth
from robot_skin.pose.imu_model import imu_features, imu_windows

T, N, D, S = 40, 3, 2, 2


def _toy(eid="toy", *, kind="glove", q_valid=None):
    """q[t] = (t, 10 t); ΔS[t, n] = 100 t + n; labels: taxel 0 no-contact everywhere, taxel 1
    self-touch on 10..19, taxel 2 unknown; taxel 0 saturated at frame 5."""
    t = np.arange(T) / 200.0
    fr = np.arange(T, dtype=np.float32)
    lab = np.full((T, N), -1, np.int8)
    lab[:, 0] = 0
    lab[10:20, 1] = 1
    lab[20:, 1] = 0
    sat = np.zeros((T, N), bool)
    sat[5, 0] = True
    rng = np.random.default_rng(0)
    quat = rng.normal(size=(T, S, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    arrays = {E.K_T: t, E.K_Q: np.stack([fr, 10 * fr], 1), E.K_QD: np.ones((T, D), np.float32),
              E.K_DELTA: (100 * fr[:, None] + np.arange(N)).astype(np.float32), E.K_SATURATED: sat,
              E.K_CONTACT_LABEL: lab, E.K_TAXEL_POS: np.broadcast_to(fr[:, None, None], (T, N, 3)).astype(np.float32),
              E.K_TAXEL_NRM: np.tile(np.array([0, 0, 1], np.float32), (T, N, 1)),
              E.K_HAND_VALID: np.ones(T, bool) if q_valid is None else q_valid,
              E.K_HAND_FINGERS: np.broadcast_to(fr[:, None, None], (T, 15, 3)).astype(np.float32),
              E.K_HAND_GLOBAL: np.zeros((T, 3), np.float32),
              E.K_IMU_QUAT: quat.astype(np.float32), E.K_IMU_GYRO: rng.normal(size=(T, S, 3)).astype(np.float32),
              E.K_IMU_ACC: rng.normal(size=(T, S, 3)).astype(np.float32)}
    meta = E.EpisodeMeta(episode_id=eid, dataset="motion", kind=kind, layout="x", n_taxels=N, imu_sites=["wrist", "f"],
                         preprocessing={"q_source": "hand_pose" if kind == "glove" else "joint_state",
                                        "imu": {"wrist_index": 0}})
    return E.Episode(meta, arrays)


def test_causal_window_edge_padding():
    np.testing.assert_array_equal(M.causal_window(2, 5), [0, 0, 0, 1, 2])
    np.testing.assert_array_equal(M.causal_window(7, 3), [5, 6, 7])
    with pytest.raises(ValueError):
        M.causal_window(3, 0)


def _at(ds, t):
    """Sample of ``ds`` anchored at frame ``t``."""
    (i,) = np.flatnonzero(ds.index[:, 1] == t)
    return ds[int(i)]


def test_baseline_windows_history_only_labels_and_saturation():
    ds = M.BaselineWindowDataset([_toy()], window=4, stride=1)
    # every frame has the no-contact taxel 0, except frame 5 where it is saturated (taxel 1 unknown)
    assert len(ds) == T - 1 and 5 not in ds.index[:, 1] and ds.joint_dim == D and ds.n_taxels == N
    s = _at(ds, 2)
    np.testing.assert_array_equal(s["q_hist"][:, 0], [0, 0, 1, 2])  # edge-padded history, ends at t
    np.testing.assert_array_equal(s["q_hist"][:, 1], [0, 0, 10, 20])
    assert s["valid"].tolist() == [True, False, False] and s["y"].tolist() == [200.0, 0.0, 0.0]
    assert s["pos"].shape == (N, 3) and float(s["pos"][0, 0]) == 2.0
    assert _at(ds, 15)["valid"].tolist() == [True, False, False]     # taxel 1 self-touch → not a target
    s25 = _at(ds, 25)
    assert s25["valid"].tolist() == [True, True, False] and s25["y"].tolist() == [2500.0, 2501.0, 0.0]
    # min_valid, other label sets, stride, saturation kept
    assert len(M.BaselineWindowDataset([_toy()], window=4, min_valid=2)) == T - 20
    assert len(M.BaselineWindowDataset([_toy()], window=4, only_labels=(1,))) == 10
    assert len(M.BaselineWindowDataset([_toy()], window=4, stride=3)) == len(range(0, T - 1, 3))
    assert len(M.BaselineWindowDataset([_toy()], window=4, exclude_saturated=False)) == T


def test_baseline_windows_skip_invalid_hand_history_and_normalise():
    qv = np.ones(T, bool)
    qv[12] = False
    ds = M.BaselineWindowDataset([_toy(q_valid=qv)], window=4)
    ts = set(int(ds[i]["t_index"]) for i in range(len(ds)))
    # default qd = causal Savitzky–Golay over w = 11 frames (50 ms @ 200 Hz, odd): qd[t] reads
    # q[t−11 … t] (w − 1 plus one frame for the no-scipy fallback) → qd invalid on 12..23, and a
    # 4-frame window then excludes every t in 12..26 (a held label's jump never enters qd_hist)
    assert B.qd_support(200.0) == (11, 0)
    np.testing.assert_array_equal(np.flatnonzero(~M.qd_valid_mask(_toy(q_valid=qv))), np.arange(12, 24))
    assert ts == set(range(T)) - {5} - set(range(12, 27))
    # centred savgol (±5 frames) → qd invalid on 7..17, windows ending on 7..20 excluded
    cen = _toy(q_valid=qv)
    cen.meta.preprocessing["config"] = {"qd": {"method": "savgol"}}
    ts_c = set(int(i) for i in M.BaselineWindowDataset([cen], window=4).index[:, 1])
    assert ts_c == set(range(T)) - {5} - set(range(7, 21))
    robot = _toy("r", kind="robot", q_valid=qv)                     # joint-state q: hand mask irrelevant
    assert len(M.BaselineWindowDataset([robot], window=4)) == T - 1 and M.qd_valid_mask(robot).all()
    stats = ST.compute_stats([_toy()], keys=(E.K_Q, E.K_QD))
    dn = M.BaselineWindowDataset([_toy()], window=4, joint_stats=stats)
    np.testing.assert_allclose(_at(dn, 30)["q_hist"].numpy(),
                               stats[E.K_Q].apply(np.array([[27, 270], [28, 280], [29, 290], [30, 300]])))
    dt = M.BaselineWindowDataset([_toy()], window=4, joint_stats=(stats[E.K_Q], stats[E.K_QD]))
    np.testing.assert_array_equal(_at(dt, 30)["qd_hist"], _at(dn, 30)["qd_hist"])
    batch = next(iter(DataLoader(dn, batch_size=8, shuffle=False)))
    assert batch["q_hist"].shape == (8, 4, D) and batch["y"].dtype == torch.float32
    assert batch["valid"].dtype == torch.bool
    with pytest.raises(ValueError, match="features"):
        M.BaselineWindowDataset([_toy()], window=4, joint_stats={E.K_Q: ST.NormStats(np.zeros(3, np.float32),
                                                                                     np.ones(3, np.float32))})
    bad = _toy()
    del bad.arrays[E.K_Q]
    with pytest.raises(ValueError, match="lacks 'q'"):
        M.BaselineWindowDataset([bad])


def test_contact_windows_need_residual_z():
    ep = _toy()
    with pytest.raises(ValueError, match="residual_z"):
        M.ContactWindowDataset([ep])
    z = np.tile(np.arange(T, dtype=np.float32)[:, None], (1, N))
    z[3, 2] = np.nan
    ep.set_derived(E.D_RESIDUAL_Z, z, save=False)
    ds = M.ContactWindowDataset([ep], window=5)
    assert len(ds) == T - 1                                         # frame 5: only taxel 0 labelled, saturated
    s = _at(ds, 4)
    np.testing.assert_array_equal(s["z_hist"][:, 0], [0, 1, 2, 3, 4])
    assert s["z_hist"][3, 2] == 0.0                                  # non-finite → 0
    assert s["sat_hist"].shape == (5, N) and bool(s["sat_hist"][:, 0].any()) is False
    s15 = _at(ds, 15)
    assert s15["label"].tolist() == [0.0, 1.0, 0.0] and s15["label_mask"].tolist() == [True, True, False]
    assert s15["q"].tolist() == [15.0, 150.0] and s15["qd"].shape == (D,)
    assert len(M.ContactWindowDataset([ep], window=5, exclude_saturated=False)) == T
    assert len(M.ContactWindowDataset([ep], window=5, min_labelled=2)) == T - 10
    assert bool(s15["q_valid"]) is True
    # glove frames whose q / qd are held labels are flagged (not dropped: their labels stay usable)
    qv = np.ones(T, bool)
    qv[30] = False
    gap = _toy("gap", q_valid=qv)
    gap.set_derived(E.D_RESIDUAL_Z, z, save=False)
    dg = M.ContactWindowDataset([gap], window=5)
    flags = {int(dg[i]["t_index"]): bool(dg[i]["q_valid"]) for i in range(len(dg))}
    assert [t for t, f in flags.items() if not f] == list(range(30, 40)) and len(dg) == T - 1
    # consistency checks: derived shape, taxel count across episodes
    bad = _toy("bad")
    bad.set_derived(E.D_RESIDUAL_Z, z[:, :2], save=False)
    with pytest.raises(ValueError, match="shape"):
        M.ContactWindowDataset([bad])


def test_label_key_falls_back_to_derived_pseudo_labels(tmp_path):
    """``label_key`` names an episode array or, when absent, a derived one: the contact stage's
    ``contact_label_pseudo`` (``episode.D_CONTACT_LABEL_PSEUDO``) trains without in-memory views."""
    assert E.D_CONTACT_LABEL_PSEUDO == "contact_label_pseudo"
    ep = _toy("pseudo")
    pseudo = np.full((T, N), -1, np.int8)
    pseudo[:, 2] = 0                                   # taxel 2: unknown in contact_label, 0 in the pseudo labels
    pseudo[25:35, 2] = 1
    z = np.tile(np.arange(T, dtype=np.float32)[:, None], (1, N))
    ep.set_derived(E.D_RESIDUAL_Z, z, save=False)
    ep.set_derived(E.D_CONTACT_LABEL_PSEUDO, pseudo, save=False)
    ds = M.BaselineWindowDataset([ep], window=4, label_key=E.D_CONTACT_LABEL_PSEUDO)
    assert len(ds) == T - 10 and _at(ds, 3)["valid"].tolist() == [False, False, True]
    cw = M.ContactWindowDataset([ep], window=5, label_key=E.D_CONTACT_LABEL_PSEUDO)
    assert len(cw) == T
    s = _at(cw, 30)
    assert s["label"].tolist() == [0.0, 0.0, 1.0] and s["label_mask"].tolist() == [False, False, True]
    # on disk: the derived .npy is found lazily; the preprocessing array still wins when present
    ep.save(tmp_path / "ep")
    disk = E.Episode.load(tmp_path / "ep")
    assert not disk.has(E.D_CONTACT_LABEL_PSEUDO) and disk.has_derived(E.D_CONTACT_LABEL_PSEUDO)
    cw2 = M.ContactWindowDataset([disk], window=5, label_key=E.D_CONTACT_LABEL_PSEUDO)
    np.testing.assert_array_equal(_at(cw2, 30)["label"], s["label"])
    assert _at(M.ContactWindowDataset([disk], window=5), 30)["label_mask"].tolist() == [True, True, False]
    # neither an array nor a derived array; wrong shape
    with pytest.raises(ValueError, match="lacks 'no_such_labels'"):
        M.ContactWindowDataset([ep], label_key="no_such_labels")
    with pytest.raises(ValueError, match="lacks 'no_such_labels'"):
        M.BaselineWindowDataset([ep], label_key="no_such_labels")
    bad = _toy("bad_labels")
    bad.set_derived("labels_2", pseudo[:, :2], save=False)
    with pytest.raises(ValueError, match="labels 'labels_2' have shape"):
        M.BaselineWindowDataset([bad], label_key="labels_2")


def test_datasets_reject_mixed_episodes():
    """Episodes whose q columns, taxel count or IMU sites differ cannot share a dataset (a silent
    column permutation would corrupt training)."""
    a, b = _toy("a"), _toy("b")
    b.meta.joint_names = ["j1", "j0"]
    with pytest.raises(ValueError, match="joint_names"):
        M.BaselineWindowDataset([a, b], window=4)
    c = _toy("c")
    c.meta.n_taxels = 4
    c.arrays = {k: (np.concatenate([v, v[:, :1]], 1) if v.ndim >= 2 and v.shape[1] == N and k != E.K_Q else v)
                for k, v in c.arrays.items()}
    with pytest.raises(ValueError, match="n_taxels"):
        M.BaselineWindowDataset([a, c], window=4)
    d = _toy("d")
    d.meta.imu_sites = ["f", "wrist"]
    with pytest.raises(ValueError, match="IMU sites"):
        M.ImuPoseWindowDataset([a, d], window=4)
    u = _toy("u")
    u.meta.preprocessing["imu"]["calibrated"] = False
    with pytest.warns(UserWarning, match="uncalibrated IMUs"):
        M.ImuPoseWindowDataset([a, u], window=4)


def test_imu_pose_windows_match_pose_imu_model():
    qv = np.ones(T, bool)
    qv[:6] = False
    ep = _toy(q_valid=qv)
    stats = ST.compute_stats([ep], keys=(ST.IMU_FEATURES,))
    ds = M.ImuPoseWindowDataset([ep], window=6, imu_stats=stats)
    assert len(ds) == T - 6 and ds.feature_dim == S * 12
    feat = imu_features(np.array(ep[E.K_IMU_QUAT]), np.array(ep[E.K_IMU_GYRO]), np.array(ep[E.K_IMU_ACC]), 0)
    win = imu_windows(stats[ST.IMU_FEATURES].apply(feat), 6)
    s = ds[4]
    assert s["t_index"] == 10 == _at(ds, 10)["t_index"]
    np.testing.assert_allclose(s["feat"].numpy(), win[10], atol=1e-6)
    assert s["finger_pose"].shape == (15, 3) and float(s["finger_pose"][0, 0]) == 10.0
    assert s["global_orient"].shape == (3,)
    no_vec = M.ImuPoseWindowDataset([ep], window=6, gyro=False, acc=False)
    assert no_vec.feature_dim == S * 6
    with pytest.raises(ValueError, match="no eligible"):
        M.ImuPoseWindowDataset([_toy(q_valid=np.zeros(T, bool))])


def test_datasets_on_preprocessed_synthetic_session(tmp_path):
    """End to end: synthetic D1 glove session → build → train-split stats → the three datasets.
    Baseline targets are only taken on truly contact-free, unsaturated frames."""
    sdir = tmp_path / "raw"
    generate_session(sdir, kind="glove", dataset="motion", duration_s=3.0, seed=11, cameras=(),
                     params={"sat_prob": 1.0})
    ep = B.preprocess_session(sdir, tmp_path / "proc")
    g = load_ground_truth(sdir)
    stats = ST.compute_stats([ep.root], keys=(E.K_Q, E.K_QD, ST.IMU_FEATURES))
    ds = M.BaselineWindowDataset([ep.root], window=16, stride=2, joint_stats=stats)
    k = np.clip(np.searchsorted(g["t"], ep.t), 0, len(g["t"]) - 1)
    contact, sat_gt = g["contact"][k], g["saturated"][k]
    n_valid = 0
    for i in range(len(ds)):
        s = ds[i]
        v, t = s["valid"].numpy(), s["t_index"]
        assert not (contact[t] & v).any() and not (sat_gt[t] & v).any()
        np.testing.assert_allclose(s["y"].numpy()[v], np.asarray(ep[E.K_DELTA][t])[v])
        n_valid += int(v.sum())
    assert n_valid > 0.5 * len(ds) * ep.meta.n_taxels
    q_all = np.concatenate([ds[i]["q_hist"].numpy() for i in range(0, len(ds), 7)])
    assert abs(float(q_all.mean())) < 0.5 and 0.3 < float(q_all.std()) < 2.0
    ip = M.ImuPoseWindowDataset([ep.root], window=16, imu_stats=stats)
    assert ip.feature_dim == 7 * 12 and len(ip) == int(np.asarray(ep[E.K_HAND_VALID]).sum())
    ep2 = E.Episode.load(ep.root)
    rz = -np.asarray(ep2[E.K_DELTA]) / 0.1
    ep2.set_derived(E.D_RESIDUAL_Z, rz.astype(np.float32))
    cw = M.ContactWindowDataset([ep.root], window=8, joint_stats=stats)
    pos = [i for i in range(len(cw)) if cw[i]["label"].sum() > 0]
    assert pos and all(float((cw[i]["z_hist"][-1] * cw[i]["label"]).max()) > 5.0 for i in pos[::5])
