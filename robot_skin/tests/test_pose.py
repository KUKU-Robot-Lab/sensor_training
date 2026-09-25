import numpy as np
import pytest
import torch

from common.layouts import load_layout
from robot_skin.pose import (
    StaticPoseProvider, TaxelPoseProvider, TransformPoseProvider, sample_poses, transform_taxels,
)
from robot_skin.pose.glove_imu2mano import GloveImu2ManoPoseProvider, finetune_vifnet_s, load_vifnet_s
from robot_skin.pose.robot_fk import RobotFKPoseProvider
from robot_skin.pose.vision_hand import (
    HaMeREstimator, VisionHandEstimator, estimate_sequence, load_hand_labels, save_hand_labels,
    smooth_hand_labels,
)


def _rotz(a, p=(0, 0, 0)):
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T[:3, 3] = p
    return T


def test_static_identity_matches_layout():
    L = load_layout("sats_4x4")
    prov = StaticPoseProvider(L)
    assert isinstance(prov, TaxelPoseProvider) and prov.n_taxels == 16
    pos, nrm = prov.pose_at(0.0)
    np.testing.assert_allclose(pos, L.positions)
    np.testing.assert_allclose(nrm, L.normals)


def test_transform_applies_rotation_and_translation():
    L = load_layout("sats_4x4")
    pos, nrm = transform_taxels(L, {"sats_pad": _rotz(np.pi / 2, (1.0, 0.0, 0.0))})
    x0, y0 = L.positions[0, :2]
    np.testing.assert_allclose(pos[0], [1.0 - y0, x0, 0.0], atol=1e-12)
    np.testing.assert_allclose(nrm, L.normals, atol=1e-12)  # z-normals invariant under rotz


def test_missing_parent_raises():
    with pytest.raises(KeyError):
        transform_taxels(load_layout("glove_template"), {"palm": np.eye(4)})


def test_time_varying_provider_and_sampling():
    L = load_layout("sats_4x4")
    prov = TransformPoseProvider(L, lambda t: {"sats_pad": _rotz(0.0, (t, 0.0, 0.0))})
    pos, nrm = sample_poses(prov, np.array([0.0, 0.5, 1.0]))
    assert pos.shape == (3, 16, 3) and nrm.shape == (3, 16, 3)
    np.testing.assert_allclose(pos[2, :, 0] - pos[0, :, 0], 1.0)


def test_provider_constructors_validate_layouts(tmp_path):
    # robot FK: glove layout rejected before the URDF is touched; missing URDF file → FileNotFoundError
    with pytest.raises(ValueError):
        RobotFKPoseProvider(load_layout("glove_template"), "hand.urdf", lambda t: np.zeros(16))
    with pytest.raises(FileNotFoundError):
        RobotFKPoseProvider(load_layout("robot_hand_template"), tmp_path / "hand.urdf", lambda t: np.zeros(16))
    # glove IMU→MANO: non-MANO layouts rejected
    with pytest.raises(ValueError):
        GloveImu2ManoPoseProvider(load_layout("robot_hand_template"), None, None, np.zeros(1),
                                  np.zeros((1, 7, 4)))


def test_external_backbones_are_documented_stubs():
    with pytest.raises(NotImplementedError):
        load_vifnet_s("vifnet_s.pt")
    with pytest.raises(NotImplementedError):
        finetune_vifnet_s([], "out")
    with pytest.raises(NotImplementedError):
        HaMeREstimator()


# ── vision hand labels ─────────────────────────────────────────────────────
def _labels(T=60, hz=30.0, seed=0):
    g = np.random.default_rng(seed)
    t = np.arange(T) / hz
    axis = g.normal(size=(16, 3))
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    rate = g.uniform(0.5, 1.5, size=16)
    rot = axis[None] * (0.2 + rate[None] * t[:, None])[..., None]         # constant-rate rotations
    wp = np.stack([np.sin(t), np.cos(t), 0.1 * t], -1)
    return t, rot[:, 0], rot[:, 1:], wp


def test_hand_labels_save_load_roundtrip(tmp_path):
    t, go, fp, wp = _labels(10)
    p = save_hand_labels(tmp_path / "hand_pose.npz", t, go, fp, wp, np.linspace(0, 1, 10))
    d = load_hand_labels(tmp_path)
    assert p.name == "hand_pose.npz" and set(d) == {"t", "global_orient", "finger_pose", "wrist_pos", "confidence"}
    np.testing.assert_allclose(d["finger_pose"], fp, atol=1e-6)
    np.testing.assert_allclose(d["confidence"], np.linspace(0, 1, 10), atol=1e-7)
    assert load_hand_labels(save_hand_labels(tmp_path / "b.npz", t, go, fp, wp))["confidence"].min() == 1.0
    with pytest.raises(ValueError):
        save_hand_labels(tmp_path / "c.npz", t, go, fp[:, :14], wp)
    with pytest.raises(ValueError):
        save_hand_labels(tmp_path / "d.npz", t[::-1], go, fp, wp)


class _ConstEstimator:
    def estimate(self, frames):
        B = frames.shape[0]
        return {"global_orient": np.zeros((B, 3)), "finger_pose": np.full((B, 15, 3), 0.1),
                "wrist_pos": np.zeros((B, 3)), "confidence": np.full(B, 0.9)}


def test_estimate_sequence_with_protocol_estimator():
    est = _ConstEstimator()
    assert isinstance(est, VisionHandEstimator)
    d = estimate_sequence(est, np.zeros((7, 4, 5, 3), dtype=np.uint8), np.arange(7) / 30.0, batch_size=3)
    assert d["finger_pose"].shape == (7, 15, 3) and np.allclose(d["confidence"], 0.9)


def test_smooth_hand_labels_fills_short_gaps_and_flags_long_ones():
    from robot_skin.geometry.rotations import aa_to_matrix, geodesic_distance
    t, go, fp, wp = _labels()
    conf = np.ones_like(t)
    conf[10:13] = 0.1                                   # 0.1 s gap → SLERP-filled
    conf[30:45] = 0.0                                   # 0.5 s gap → invalid
    conf[:2] = 0.0                                      # leading gap → invalid
    go_n, fp_n = go.copy(), fp.copy()
    go_n[10:13] += 1.0                                  # garbage in low-confidence frames
    fp_n[30:45] = np.nan
    out = smooth_hand_labels(t, go_n, fp_n, wp, conf, max_gap_s=0.2, cutoff_hz=None)
    assert out["valid"].dtype == bool
    assert out["valid"][10:13].all() and not out["valid"][30:45].any() and not out["valid"][:2].any()
    assert out["valid"][2:10].all() and out["valid"][45:].all()
    # constant-rate rotations: SLERP between bracketing frames is exact
    R_hat = aa_to_matrix(torch.as_tensor(out["global_orient"][10:13], dtype=torch.float64))
    R_true = aa_to_matrix(torch.as_tensor(go[10:13]))
    assert float(geodesic_distance(R_hat, R_true).max()) < 2e-3
    np.testing.assert_allclose(out["wrist_pos"][10:13], wp[10:13], atol=5e-3)
    assert np.isfinite(out["finger_pose"]).all()
    np.testing.assert_allclose(out["finger_pose"][35], out["finger_pose"][29], atol=1e-6)   # held


def test_smooth_hand_labels_lowpass_reduces_jitter():
    t, go, fp, wp = _labels(T=120)
    g = np.random.default_rng(1)
    noisy_wp = wp + g.normal(scale=0.01, size=wp.shape)
    noisy_fp = fp + g.normal(scale=0.05, size=fp.shape)
    out = smooth_hand_labels(t, go, noisy_fp, noisy_wp, None, cutoff_hz=3.0)
    assert out["valid"].all()
    err_raw = np.abs(noisy_wp - wp)[10:-10].mean()
    err_f = np.abs(out["wrist_pos"] - wp)[10:-10].mean()
    assert err_f < 0.6 * err_raw
    from robot_skin.geometry.rotations import aa_to_matrix
    from robot_skin.pose.imu_model import rotation_geodesic
    ang = lambda a: rotation_geodesic(aa_to_matrix(torch.as_tensor(a, dtype=torch.float64)),  # noqa: E731
                                      aa_to_matrix(torch.as_tensor(fp)))[10:-10].mean()
    assert float(ang(out["finger_pose"])) < 0.7 * float(ang(noisy_fp))
    none = smooth_hand_labels(t, go, fp, wp, np.zeros_like(t))
    assert not none["valid"].any()


def test_smooth_hand_labels_without_scipy(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "scipy.signal", None)       # force the moving-average fallback
    t, go, fp, wp = _labels(T=120)
    noisy = wp + np.random.default_rng(2).normal(scale=0.01, size=wp.shape)
    out = smooth_hand_labels(t, go, fp, noisy, None, cutoff_hz=3.0)
    assert np.abs(out["wrist_pos"] - wp)[10:-10].mean() < 0.7 * np.abs(noisy - wp)[10:-10].mean()


def test_save_hand_labels_directory_paths(tmp_path):
    t, go, fp, wp = _labels(5)
    p = save_hand_labels(tmp_path / "sess_2026.01.02", t, go, fp, wp)     # new dir with dots in its name
    assert p == tmp_path / "sess_2026.01.02" / "hand_pose.npz" and p.is_file()
    np.testing.assert_allclose(load_hand_labels(p.parent)["wrist_pos"], wp, atol=1e-6)
    (tmp_path / "existing").mkdir()
    assert save_hand_labels(tmp_path / "existing", t, go, fp, wp).name == "hand_pose.npz"
