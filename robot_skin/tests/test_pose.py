import numpy as np
import pytest

from common.layouts import load_layout
from robot_skin.pose import (
    StaticPoseProvider, TaxelPoseProvider, TransformPoseProvider, sample_poses, transform_taxels,
)
from robot_skin.pose.glove_imu2mano import GloveImu2ManoPoseProvider
from robot_skin.pose.robot_fk import RobotFKPoseProvider


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


def test_stubs_raise():
    with pytest.raises(NotImplementedError):
        RobotFKPoseProvider(load_layout("robot_hand_template"), "hand.urdf", lambda t: np.zeros(16))
    with pytest.raises(ValueError):
        RobotFKPoseProvider(load_layout("glove_template"), "hand.urdf", lambda t: np.zeros(16))
    with pytest.raises(NotImplementedError):
        GloveImu2ManoPoseProvider(load_layout("glove_template"), "ckpt.pt", lambda t: np.zeros(7))
