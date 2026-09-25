import math

import numpy as np
import torch

from robot_skin.geometry import rotations as rot


def _rand_R(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return rot.quat_to_matrix(rot.quat_normalize(torch.randn(n, 4, generator=g, dtype=torch.float64)))


def test_aa_matrix_roundtrip_and_orthonormal():
    g = torch.Generator().manual_seed(1)
    aa = torch.randn(64, 3, generator=g, dtype=torch.float64) * 1.2
    R = rot.aa_to_matrix(aa)
    torch.testing.assert_close(R @ R.transpose(-1, -2), torch.eye(3, dtype=torch.float64).expand(64, 3, 3))
    torch.testing.assert_close(rot.aa_to_matrix(rot.matrix_to_aa(R)), R, atol=1e-9, rtol=0)
    torch.testing.assert_close(rot.aa_to_matrix(torch.zeros(3, dtype=torch.float64)), torch.eye(3, dtype=torch.float64))


def test_quat_matrix_consistency():
    R = _rand_R(128)
    q = rot.matrix_to_quat(R)
    assert torch.all(q[:, 0] >= 0)
    torch.testing.assert_close(rot.quat_to_matrix(q), R, atol=1e-9, rtol=0)
    v = torch.randn(128, 3, dtype=torch.float64)
    torch.testing.assert_close(rot.quat_apply(q, v), (R @ v[..., None])[..., 0])
    q2 = rot.matrix_to_quat(_rand_R(128, 2))
    torch.testing.assert_close(rot.quat_to_matrix(rot.quat_mul(q, q2)), R @ rot.quat_to_matrix(q2))
    torch.testing.assert_close(rot.aa_to_quat(rot.quat_to_aa(q)), q, atol=1e-9, rtol=0)


def test_six_d_roundtrip_and_gram_schmidt():
    R = _rand_R(32)
    torch.testing.assert_close(rot.sixd_to_matrix(rot.matrix_to_6d(R)), R)
    Rn = rot.sixd_to_matrix(torch.randn(10, 6, dtype=torch.float64))
    torch.testing.assert_close(torch.linalg.det(Rn), torch.ones(10, dtype=torch.float64))


def test_rpy_urdf_convention():
    Rz = rot.rpy_to_matrix(torch.tensor([0.0, 0.0, math.pi / 2], dtype=torch.float64))
    torch.testing.assert_close(Rz @ torch.tensor([1.0, 0, 0], dtype=torch.float64),
                               torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-12, rtol=0)
    r, p, y = 0.3, -0.4, 1.1
    R = rot.rpy_to_matrix(torch.tensor([r, p, y], dtype=torch.float64))
    Rx = rot.aa_to_matrix(torch.tensor([r, 0, 0], dtype=torch.float64))
    Ry = rot.aa_to_matrix(torch.tensor([0, p, 0], dtype=torch.float64))
    Rz = rot.aa_to_matrix(torch.tensor([0, 0, y], dtype=torch.float64))
    torch.testing.assert_close(R, Rz @ Ry @ Rx)


def test_geodesic_transform_and_continuity():
    R = _rand_R(8)
    assert torch.allclose(rot.geodesic_distance(R, R), torch.zeros(8, dtype=torch.float64), atol=1e-3)
    Rb = R @ rot.aa_to_matrix(torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64))
    torch.testing.assert_close(rot.geodesic_distance(R, Rb), torch.full((8,), 0.5, dtype=torch.float64))
    T = rot.make_transform(R, torch.randn(8, 3, dtype=torch.float64))
    torch.testing.assert_close(T @ rot.invert_transform(T), torch.eye(4, dtype=torch.float64).expand(8, 4, 4))
    p = torch.randn(8, 3, dtype=torch.float64)
    torch.testing.assert_close(rot.transform_points(rot.invert_transform(T), rot.transform_points(T, p)), p)
    q = np.array([[1.0, 0, 0, 0], [-1.0, 0, 0, 0], [-0.9, 0.1, 0, 0]])
    fixed = rot.quat_fix_continuity(q)
    assert np.all(fixed[:, 0] > 0)
