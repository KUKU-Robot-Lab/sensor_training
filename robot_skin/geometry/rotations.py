"""Rotation utilities (torch, batched over any leading dims).

Conventions (used everywhere in robot_skin):
- quaternions are **wxyz**, unit norm;
- axis-angle ``aa[...,3]`` = unit axis × angle (rad), as in MANO;
- 6D rotation = first two **columns** of R concatenated ``[R[:,0], R[:,1]]`` (Zhou et al., CVPR 2019);
- URDF ``rpy`` = fixed-axis roll/pitch/yaw → ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``;
- homogeneous transforms are ``[...,4,4]`` with ``T[..., :3, :3] = R``, ``T[..., :3, 3] = p``.

Numpy inputs are accepted by :func:`as_tensor` wrappers at call sites; these functions take
and return ``torch.Tensor`` (float32/float64 preserved).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_EPS = 1e-8


def as_tensor(x, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x if dtype is None else x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype or torch.float64)


# ── axis-angle ⇄ matrix ────────────────────────────────────────────────────
def skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], -1),
        torch.stack([v[..., 2], z, -v[..., 0]], -1),
        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def aa_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues. Stable near zero (first-order series)."""
    theta = aa.norm(dim=-1, keepdim=True)                     # [...,1]
    small = theta < 1e-6
    safe = torch.where(small, torch.ones_like(theta), theta)
    k = aa / safe
    K = skew(k)
    s = torch.sin(theta)[..., None]
    c = (1 - torch.cos(theta))[..., None]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(K.shape)
    R = eye + s * K + c * (K @ K)
    R_small = eye + skew(aa)
    return torch.where(small[..., None], R_small, R)


def matrix_to_aa(R: torch.Tensor) -> torch.Tensor:
    return quat_to_aa(matrix_to_quat(R))


# ── quaternions (wxyz) ─────────────────────────────────────────────────────
def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], -1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v[...,3]`` by unit quaternions ``q[...,4]``."""
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], -1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_normalize(q).unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Shepperd-style, branch-free via the largest diagonal term. Returns w ≥ 0."""
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    cands = torch.stack([
        torch.stack([1 + tr, R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1),
        torch.stack([R[..., 2, 1] - R[..., 1, 2], 1 + m00 - m11 - m22, R[..., 0, 1] + R[..., 1, 0], R[..., 0, 2] + R[..., 2, 0]], -1),
        torch.stack([R[..., 0, 2] - R[..., 2, 0], R[..., 0, 1] + R[..., 1, 0], 1 - m00 + m11 - m22, R[..., 1, 2] + R[..., 2, 1]], -1),
        torch.stack([R[..., 1, 0] - R[..., 0, 1], R[..., 0, 2] + R[..., 2, 0], R[..., 1, 2] + R[..., 2, 1], 1 - m00 - m11 + m22], -1),
    ], -2)                                                        # [...,4,4]
    diag = torch.stack([tr, m00, m11, m22], -1)
    idx = diag.argmax(-1, keepdim=True)                           # [...,1]
    q = torch.gather(cands, -2, idx[..., None].expand(*idx.shape, 4)).squeeze(-2)
    q = quat_normalize(q)
    return torch.where(q[..., :1] < 0, -q, q)


def quat_to_aa(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = v.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(s, q[..., :1])
    scale = torch.where(s < 1e-7, 2.0 / q[..., :1].clamp_min(_EPS), angle / s.clamp_min(_EPS))
    return v * scale


def aa_to_quat(aa: torch.Tensor) -> torch.Tensor:
    theta = aa.norm(dim=-1, keepdim=True)
    half = 0.5 * theta
    k = torch.where(theta < 1e-8, 0.5 - theta ** 2 / 48.0, torch.sin(half) / theta.clamp_min(_EPS))
    return torch.cat([torch.cos(half), aa * k], -1)


def quat_fix_continuity(q: np.ndarray | torch.Tensor, axis: int = 0):
    """Flip signs along ``axis`` (time) so consecutive quaternions have positive dot product."""
    is_t = isinstance(q, torch.Tensor)
    a = q.detach().cpu().numpy().copy() if is_t else np.array(q, copy=True)
    a = np.moveaxis(a, axis, 0)
    for t in range(1, a.shape[0]):
        flip = np.sum(a[t] * a[t - 1], axis=-1, keepdims=True) < 0
        a[t] = np.where(flip, -a[t], a[t])
    a = np.moveaxis(a, 0, axis)
    return torch.as_tensor(a, dtype=q.dtype) if is_t else a


# ── 6D ─────────────────────────────────────────────────────────────────────
def matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def sixd_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Gram–Schmidt (Zhou et al. 2019). Columns = b1, b2, b1×b2."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=_EPS)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=_EPS)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], -1)


# ── misc ───────────────────────────────────────────────────────────────────
def rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    r, p, y = rpy.unbind(-1)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack([
        torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
        torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
        torch.stack([-sp, cp * sr, cp * cr], -1)], -2)


def geodesic_distance(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Angle (rad) of ``R1ᵀ R2``; numerically safe."""
    tr = (R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7))


def make_transform(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    T = torch.zeros(*R.shape[:-2], 4, 4, dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    T[..., 3, 3] = 1
    return T


def invert_transform(T: torch.Tensor) -> torch.Tensor:
    R, p = T[..., :3, :3], T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    return make_transform(Rt, -(Rt @ p[..., None])[..., 0])


def transform_points(T: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return (T[..., :3, :3] @ p[..., None])[..., 0] + T[..., :3, 3]
