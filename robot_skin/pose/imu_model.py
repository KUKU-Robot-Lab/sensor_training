"""Glove IMUs → MANO finger pose: features, calibration, synthetic IMU, baseline network, loss.

Pipeline (D1 ``motion`` data: vision/mocap MANO labels + 7 glove IMUs)::

    raw quat/gyro/acc ──calibrate_imu_offsets/apply_imu_offsets──▶ segment-frame IMU
      ──imu_features (relative to the wrist IMU)──▶ [T, S·F] ──imu_windows──▶ [T, W, S·F]
      ──ImuHandPoseNet (causal GRU | TCN)──▶ finger pose 15×6D ──to_axis_angle──▶ MANO θ

Paper grounding
- The user-specified pretrained backbone is VIHand **VIFNet-S** (Wang et al., ACM MM 2025,
  DOI 10.1145/3746027.3758215; IMU-only student distilled from a visual-inertial teacher). Its
  weights are external and its exact I/O (IMU count/placement, input/output format) is
  unverified, so :class:`ImuHandPoseNet` is the in-house baseline with the same *role* (IMU window
  → MANO finger pose). ``glove_imu2mano.load_vifnet_s`` is the swap-in point and must wrap an
  input adapter (our 7 sites → VIFNet-S input) and an output conversion (→ finger_pose[15,3]).
- Rotations are regressed in the continuous 6D representation (Zhou et al., *On the Continuity of
  Rotation Representations in Neural Networks*, CVPR 2019) and supervised with the geodesic
  angle, plus an optional fingertip-position term through the MANO skeleton
  (``pose.mano.ManoSkeleton``; MANO: Romero et al., SIGGRAPH Asia 2017).

Conventions
- quaternions wxyz; gyro (rad/s) and acc (m/s², specific force incl. gravity) are in each IMU's
  **sensor frame** (what raw IMUs report). ``vec_frame="world"`` is available for devices that
  output world-frame vectors.
- Calibration model: ``q_meas = G ⊗ q_segment ⊗ M`` (``G`` IMU-world ← model-world, ``M``
  sensor mounting). A static pose with known segment orientations ``q_ref`` gives the per-site
  offset ``q_off = M⁻¹ = q̄_meas⁻¹ ⊗ G ⊗ q_ref`` (``G = I`` → the spec's
  ``mean(q_meas)⁻¹ ⊗ q_ref``) and ``q_segment = G⁻¹ ⊗ q_meas ⊗ q_off``. Features relative to the
  wrist IMU are independent of ``G`` (it cancels in ``q_wrist⁻¹ ⊗ q_site``).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from common.layouts import MANO_SEGMENTS, Layout
from robot_skin.geometry.rotations import (
    aa_to_matrix, as_tensor, matrix_to_6d, matrix_to_aa, matrix_to_quat, quat_conj, quat_fix_continuity,
    quat_mul, quat_normalize, quat_to_matrix, sixd_to_matrix,
)
from robot_skin.pose.mano import ManoSkeleton

ROT_FEATURES = 6
VEC_FEATURES = 3
IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


# ── features ───────────────────────────────────────────────────────────────
def imu_feature_dim(n_sites: int, *, gyro: bool = True, acc: bool = True) -> int:
    """Width of :func:`imu_features` output (per time step)."""
    return n_sites * (ROT_FEATURES + VEC_FEATURES * (int(gyro) + int(acc)))


def imu_features(quat, gyro=None, acc=None, wrist_index: int = 0, *, vec_frame: str = "sensor"):
    """Per-step IMU features relative to the wrist IMU.

    ``quat[...,W,S,4]`` (wxyz), ``gyro``/``acc`` ``[...,W,S,3]`` or None (omitted) →
    ``[...,W,S·F]`` with per site ``[6D(q_wrist⁻¹⊗q_site) | gyro_wrist | acc_wrist]`` (F = 6, 9 or
    12). Vectors are rotated into the wrist IMU frame (``vec_frame="sensor"``: ``R_rel·v``;
    ``"world"``: ``R_wristᵀ·v``). Invariant to a common rotation of the IMU world frame.
    numpy in → float32 numpy out; torch in → torch out.
    """
    if vec_frame not in ("sensor", "world"):
        raise ValueError("vec_frame must be 'sensor' or 'world'")
    is_np = not isinstance(quat, torch.Tensor)
    q = as_tensor(quat)
    if not q.is_floating_point():
        q = q.to(torch.float64)
    if q.ndim < 2 or q.shape[-1] != 4:
        raise ValueError(f"quat must be [..., S, 4], got {tuple(q.shape)}")
    S = q.shape[-2]
    if not -S <= wrist_index < S:
        raise ValueError(f"wrist_index {wrist_index} out of range for {S} sites")
    q = quat_normalize(q)
    qw = q[..., wrist_index:wrist_index + 1, :] if wrist_index != -1 else q[..., -1:, :]
    R_rel = quat_to_matrix(quat_mul(quat_conj(qw), q))                   # [...,S,3,3]
    feats = [matrix_to_6d(R_rel)]
    Rw_T = quat_to_matrix(qw).transpose(-1, -2)                           # [...,1,3,3]
    for v in (gyro, acc):
        if v is None:
            continue
        v = as_tensor(v).to(dtype=q.dtype, device=q.device)
        if v.shape != q.shape[:-1] + (3,):
            raise ValueError(f"gyro/acc must be {tuple(q.shape[:-1]) + (3,)}, got {tuple(v.shape)}")
        M = R_rel if vec_frame == "sensor" else Rw_T
        feats.append(torch.matmul(M, v[..., None])[..., 0])
    out = torch.cat(feats, dim=-1)
    out = out.reshape(*out.shape[:-2], -1)
    return out.detach().cpu().numpy().astype(np.float32) if is_np else out


def imu_windows(feat: np.ndarray, window: int) -> np.ndarray:
    """Causal sliding windows ``feat[T,F]`` → ``[T,W,F]`` (row t = frames t−W+1 … t; the start is
    edge-padded with frame 0). Returns a read-only strided view."""
    feat = np.asarray(feat)
    if feat.ndim != 2 or feat.shape[0] == 0:
        raise ValueError(f"feat must be non-empty [T, F], got {feat.shape}")
    if window < 1:
        raise ValueError("window must be ≥ 1")
    padded = np.concatenate([np.repeat(feat[:1], window - 1, axis=0), feat], axis=0)
    return np.lib.stride_tricks.sliding_window_view(padded, window, axis=0).transpose(0, 2, 1)


# ── calibration ────────────────────────────────────────────────────────────
def mean_quaternion(q: np.ndarray, axis: int = 0) -> np.ndarray:
    """Sign-invariant average (principal eigenvector of Σ q qᵀ) along ``axis``; w ≥ 0."""
    q = np.moveaxis(np.asarray(q, dtype=np.float64), axis, -2)             # [...,T,4]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    M = np.einsum("...ti,...tj->...ij", q, q)
    _, V = np.linalg.eigh(M)
    m = V[..., :, -1]
    return np.where(m[..., :1] < 0, -m, m)


def _np_quat(fn, *xs):
    return fn(*[torch.as_tensor(np.array(x, dtype=np.float64)) for x in xs]).numpy()


def _ref_quats(ref_rot, S: int) -> np.ndarray:
    if ref_rot is None:
        return np.tile([1.0, 0.0, 0.0, 0.0], (S, 1))
    R = np.asarray(ref_rot, dtype=np.float64)
    if R.shape != (S, 3, 3):
        raise ValueError(f"ref_rot must be [S,3,3] with S={S}, got {R.shape}")
    return _np_quat(matrix_to_quat, R)


def calibrate_imu_offsets(quat_calib, ref_rot=None, *, world=None) -> np.ndarray:
    """Per-site mounting offsets from a static calibration pose (e.g. flat hand).

    ``quat_calib[T,S,4]`` raw quaternions recorded while the hand holds a pose whose segment
    orientations ``ref_rot[S,3,3]`` are known (None → identity for every site, i.e. "each sensor
    frame defines its segment frame at the calibration pose"). ``world`` (wxyz, optional) is the
    IMU-world ← model-world rotation ``G`` (see :func:`estimate_world_alignment`).
    Returns ``offsets[S,4]`` with ``q_off = mean(q_meas)⁻¹ ⊗ G ⊗ q_ref``.
    """
    q = np.asarray(quat_calib, dtype=np.float64)
    if q.ndim != 3 or q.shape[-1] != 4:
        raise ValueError(f"quat_calib must be [T,S,4], got {q.shape}")
    S = q.shape[1]
    qm = mean_quaternion(q, axis=0)                                        # [S,4]
    qr = _ref_quats(ref_rot, S)
    lhs = _np_quat(quat_conj, qm)
    if world is not None:
        lhs = _np_quat(quat_mul, lhs, np.broadcast_to(np.asarray(world, dtype=np.float64), (S, 4)))
    off = _np_quat(quat_mul, lhs, qr)
    off /= np.linalg.norm(off, axis=-1, keepdims=True)
    return np.where(off[:, :1] < 0, -off, off)


def estimate_world_alignment(quat_calib, ref_rot=None, *, index: int = 0) -> np.ndarray:
    """``G`` (IMU-world ← model-world, wxyz) assuming site ``index`` (default the wrist) is
    mounted with identity offset: ``G = mean(q_meas[index]) ⊗ q_ref[index]⁻¹``.

    One static pose cannot separate ``G`` from that site's mounting ``M``: if it is not identity,
    wrist-relative features end up expressed in the wrist *sensor* frame (a constant change of
    basis per session) and the IMU-derived global orientation is off by ``M``.
    """
    q = np.asarray(quat_calib, dtype=np.float64)
    if q.ndim != 3 or q.shape[-1] != 4:
        raise ValueError(f"quat_calib must be [T,S,4], got {q.shape}")
    S = q.shape[1]
    if not -S <= index < S:
        raise ValueError(f"index {index} out of range for {S} sites")
    index = int(index) % S
    if ref_rot is not None and np.shape(ref_rot) != (S, 3, 3):
        raise ValueError(f"ref_rot must be [S,3,3] with S={S}, got {np.shape(ref_rot)}")
    qm = mean_quaternion(q[:, index], axis=0)
    qr = _ref_quats(None if ref_rot is None else np.asarray(ref_rot)[index:index + 1], 1)[0]
    g = _np_quat(quat_mul, qm, _np_quat(quat_conj, qr))
    return g / np.linalg.norm(g)


def apply_imu_offsets(quat, offsets, *, world=None):
    """Segment orientations ``G⁻¹ ⊗ q_meas ⊗ q_off`` for ``quat[...,S,4]``, ``offsets[S,4]``."""
    is_np = not isinstance(quat, torch.Tensor)
    q = as_tensor(quat)
    q = q.to(torch.float64) if not q.is_floating_point() else q
    off = as_tensor(offsets).to(dtype=q.dtype, device=q.device)
    if off.shape != (q.shape[-2], 4):
        raise ValueError(f"offsets must be [S={q.shape[-2]}, 4], got {tuple(off.shape)}")
    out = quat_mul(q, off)
    if world is not None:
        G = as_tensor(world).to(dtype=q.dtype, device=q.device)
        out = quat_mul(quat_conj(G).expand_as(out), out)
    out = quat_normalize(out)
    return out.numpy() if is_np else out


def apply_imu_offsets_to_vectors(vec, offsets):
    """Sensor-frame vectors ``[...,S,3]`` (gyro/acc) → segment frame: ``R_offᵀ · v``."""
    is_np = not isinstance(vec, torch.Tensor)
    v = as_tensor(vec)
    v = v.to(torch.float64) if not v.is_floating_point() else v
    R = quat_to_matrix(as_tensor(offsets).to(dtype=v.dtype, device=v.device))  # [S,3,3]
    out = torch.matmul(R.transpose(-1, -2), v[..., None])[..., 0]
    return out.numpy() if is_np else out


#: keys used in ``SessionManifest.calibration`` for the IMU calibration.
CALIB_OFFSETS_KEY = "imu_offsets"      # [[w,x,y,z] per site]  (q_off, IMU site order)
CALIB_WORLD_KEY = "imu_world"          # [w,x,y,z] or absent   (G)
CALIB_SITES_KEY = "imu_sites"          # [site names] (sanity check against imu.npz `sites`)


def imu_calibration_to_dict(offsets, world=None, sites: Sequence[str] | None = None) -> dict:
    """JSON-able calibration entry for ``SessionManifest.calibration``."""
    d = {CALIB_OFFSETS_KEY: np.asarray(offsets, dtype=np.float64).reshape(-1, 4).tolist()}
    if world is not None:
        d[CALIB_WORLD_KEY] = np.asarray(world, dtype=np.float64).reshape(4).tolist()
    if sites is not None:
        d[CALIB_SITES_KEY] = [str(s) for s in sites]
    return d


def imu_calibration_from_dict(calib: dict, sites: Sequence[str] | None = None):
    """``(offsets[S,4] | None, world[4] | None)`` from a manifest calibration dict. When both the
    dict and ``sites`` name the sites, offsets are reordered to ``sites`` (missing → ValueError)."""
    if not calib or CALIB_OFFSETS_KEY not in calib:
        return None, None
    off = np.asarray(calib[CALIB_OFFSETS_KEY], dtype=np.float64).reshape(-1, 4)
    world = calib.get(CALIB_WORLD_KEY)
    world = None if world is None else np.asarray(world, dtype=np.float64).reshape(4)
    saved = calib.get(CALIB_SITES_KEY)
    if sites is not None and saved is not None:
        saved = [str(s) for s in saved]
        missing = [s for s in sites if s not in saved]
        if missing:
            raise ValueError(f"IMU calibration lacks sites {missing} (has {saved})")
        off = off[[saved.index(str(s)) for s in sites]]
    elif sites is not None and len(sites) != off.shape[0]:
        raise ValueError(f"calibration has {off.shape[0]} offsets for {len(sites)} sites")
    return off, world


# ── IMU sites on the skeleton & synthetic IMU ──────────────────────────────
def imu_site_segments(layout: Layout) -> list[int]:
    """Index into ``MANO_SEGMENTS`` of each layout IMU site's parent."""
    if not layout.imu_sites:
        raise ValueError(f"layout {layout.name!r} has no imu_sites")
    return [MANO_SEGMENTS.index(s.parent) for s in layout.imu_sites]


def imu_reference_rotations(layout: Layout, skeleton: ManoSkeleton | None = None, global_orient=None,
                            finger_pose=None) -> np.ndarray:
    """Segment orientation ``[S,3,3]`` of every layout IMU site at a known calibration pose
    (default: flat hand, ``global_orient = 0``) — the ``ref_rot`` for :func:`calibrate_imu_offsets`
    / :func:`estimate_world_alignment`, so that calibrated sensor frames equal the skeleton's
    segment frames (the convention of :func:`synthesize_imu`)::

        R_ref = imu_reference_rotations(layout, skeleton)
        G = estimate_world_alignment(q_calib, R_ref, index=wrist_site)
        q_off = calibrate_imu_offsets(q_calib, R_ref, world=G)
    """
    sk = skeleton if skeleton is not None else ManoSkeleton()
    go = np.zeros(3) if global_orient is None else np.asarray(global_orient, dtype=np.float64).reshape(3)
    fp = np.zeros((15, 3)) if finger_pose is None else np.asarray(finger_pose, dtype=np.float64).reshape(15, 3)
    with torch.no_grad():
        ST = sk.segment_transform_tensor(sk.forward(go, fp))
    return ST[imu_site_segments(layout), :3, :3].numpy().copy()


def imu_site_quats(layout: Layout, skeleton: ManoSkeleton, global_orient, finger_pose) -> np.ndarray:
    """Segment orientation of every IMU site ``[T,S,4]`` (wxyz, continuity-fixed) — the ideal
    (offset-free) IMU reading in the hand-pose world frame."""
    fk = skeleton.forward(global_orient, finger_pose)
    R = skeleton.segment_transform_tensor(fk)[..., imu_site_segments(layout), :3, :3]
    q = matrix_to_quat(R).detach().numpy()
    return quat_fix_continuity(q, axis=0) if q.ndim == 3 else q


def synthesize_imu(layout: Layout, skeleton: ManoSkeleton, t, global_orient, finger_pose, wrist_pos=None,
                   *, gravity: Sequence[float] = (0.0, 0.0, -9.81)) -> dict[str, np.ndarray]:
    """Ideal glove IMU streams from a MANO pose sequence (sensor frame = segment frame).

    Returns ``quat[T,S,4]``, ``gyro[T,S,3]`` (body rate from finite differences of the segment
    rotations), ``acc[T,S,3]`` (specific force ``Rᵀ(p̈ − g)`` at the segment origin; ``gravity`` in
    the pose world frame), ``sites``. Useful for synthetic data and tests; add mounting offsets
    (``q ⊗ M``) and noise to mimic real sensors.
    """
    t = np.asarray(t, dtype=np.float64)
    fp = np.asarray(finger_pose, dtype=np.float64)
    if fp.ndim != 3 or fp.shape[0] != t.shape[0] or t.shape[0] < 3:
        raise ValueError("need t[T] and finger_pose[T,15,3] with T ≥ 3")
    with torch.no_grad():
        fk = skeleton.forward(torch.as_tensor(np.asarray(global_orient, dtype=np.float64)), torch.as_tensor(fp),
                              None if wrist_pos is None else torch.as_tensor(np.asarray(wrist_pos, dtype=np.float64)))
        ST = skeleton.segment_transform_tensor(fk)[:, imu_site_segments(layout)]  # [T,S,4,4]
        R, p = ST[..., :3, :3], ST[..., :3, 3]
        # body rate: log(R_{i-1}ᵀ R_{i+1}) / (t_{i+1} − t_{i-1}) (one-sided at the ends)
        i0 = np.r_[0, np.arange(0, len(t) - 2), len(t) - 2]
        i1 = np.r_[1, np.arange(2, len(t)), len(t) - 1]
        dR = R[i0].transpose(-1, -2) @ R[i1]
        dt = torch.as_tensor(t[i1] - t[i0])[:, None, None]
        # the finite-difference rotation is expressed in frame i0; rotate to the frame at i
        w_i0 = matrix_to_aa(dR) / dt
        mid = R.transpose(-1, -2) @ R[i0]
        gyro = (mid @ w_i0[..., None])[..., 0]
        acc_w = np.gradient(np.gradient(p.numpy(), t, axis=0), t, axis=0)
        f_w = torch.as_tensor(acc_w - np.asarray(gravity, dtype=np.float64))
        acc = (R.transpose(-1, -2) @ f_w[..., None])[..., 0]
        q = quat_fix_continuity(matrix_to_quat(R).numpy(), axis=0)
    return {"t": t, "quat": q, "gyro": gyro.numpy(), "acc": acc.numpy(),
            "sites": np.array([s.name for s in layout.imu_sites])}


# ── network ────────────────────────────────────────────────────────────────
class _CausalTCN(nn.Module):
    """Residual stack of dilated causal 1-D convolutions (dilation 2^i).

    Receptive field = ``1 + (kernel − 1)·(2^n_layers − 1)`` frames (7 for the defaults kernel 3,
    2 layers) — raise ``n_layers`` to cover longer windows."""

    def __init__(self, dim: int, n_layers: int, kernel: int, dropout: float):
        super().__init__()
        self.kernel = kernel
        self.convs = nn.ModuleList(nn.Conv1d(dim, dim, kernel, dilation=2 ** i) for i in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(n_layers))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                   # [B,W,D]
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            pad = (self.kernel - 1) * 2 ** i
            h = norm(x).transpose(1, 2)
            h = conv(nn.functional.pad(h, (pad, 0))).transpose(1, 2)
            x = x + self.drop(nn.functional.gelu(h))
        return x


class ImuHandPoseNet(nn.Module):
    """IMU feature window ``[B,W,F]`` → finger rotations 15×6D (+ optional global 6D).

    In-house baseline filling the VIFNet-S role (see module docstring). ``arch``: ``"gru"``
    (default) or ``"tcn"`` (causal dilated convs). Features are normalised inside the model with
    the ``feat_mean`` / ``feat_std`` buffers (identity until :meth:`set_feature_stats`), so
    checkpoints carry their own normalisation. The output head is zero-initialised around the
    identity rotation → the untrained net predicts the flat hand.
    """

    def __init__(self, in_dim: int, *, hidden: int = 256, n_layers: int = 2, arch: str = "gru",
                 dropout: float = 0.1, kernel: int = 3, predict_global: bool = False):
        super().__init__()
        if arch not in ("gru", "tcn"):
            raise ValueError("arch must be 'gru' or 'tcn'")
        if in_dim < 1 or hidden < 1 or n_layers < 1:
            raise ValueError("in_dim, hidden and n_layers must be positive")
        self.config = dict(in_dim=in_dim, hidden=hidden, n_layers=n_layers, arch=arch, dropout=dropout,
                           kernel=kernel, predict_global=predict_global)
        self.in_dim, self.arch, self.predict_global = in_dim, arch, predict_global
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))
        self.register_buffer("identity6d", torch.tensor(IDENTITY_6D))
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        if arch == "gru":
            self.core = nn.GRU(hidden, hidden, n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0.0)
        else:
            self.core = _CausalTCN(hidden, n_layers, kernel, dropout)
        self.head = self._make_head(hidden, 15 * 6)
        self.global_head = self._make_head(hidden, 6) if predict_global else None

    @staticmethod
    def _make_head(hidden: int, out: int) -> nn.Sequential:
        head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out))
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    @classmethod
    def from_config(cls, cfg: dict) -> "ImuHandPoseNet":
        cfg = dict(cfg)
        return cls(cfg.pop("in_dim"), **cfg)

    def set_feature_stats(self, mean, std, eps: float = 1e-6) -> None:
        """Store input normalisation (e.g. ``NormStats.offset`` / ``.scale``)."""
        m = torch.as_tensor(np.asarray(mean, dtype=np.float32)).reshape(-1)
        s = torch.as_tensor(np.asarray(std, dtype=np.float32)).reshape(-1)
        if m.numel() != self.in_dim or s.numel() != self.in_dim:
            raise ValueError(f"stats must have {self.in_dim} entries")
        self.feat_mean.copy_(m)
        self.feat_std.copy_(s.clamp_min(eps))

    def encode(self, feat: torch.Tensor) -> torch.Tensor:
        """Causal per-step hidden states ``[B,W,H]``."""
        if feat.ndim != 3 or feat.shape[-1] != self.in_dim:
            raise ValueError(f"feat must be [B, W, {self.in_dim}], got {tuple(feat.shape)}")
        x = (feat.to(self.feat_mean.dtype) - self.feat_mean) / self.feat_std   # float64 features OK
        h = self.inp(x)
        return self.core(h)[0] if self.arch == "gru" else self.core(h)

    def forward_all(self, feat: torch.Tensor, *, all_steps: bool = False) -> dict[str, torch.Tensor]:
        """``finger_rot6d[B,15,6]`` (+ ``global_rot6d[B,6]``); ``all_steps`` → per step ``[B,W,…]``."""
        h = self.encode(feat)
        if not all_steps:
            h = h[:, -1]
        out = {"finger_rot6d": self.head(h).unflatten(-1, (15, 6)) + self.identity6d}
        if self.global_head is not None:
            out["global_rot6d"] = self.global_head(h) + self.identity6d
        return out

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """``feat[B,W,F]`` → ``finger_rot6d[B,15,6]`` (last step of the window)."""
        return self.forward_all(feat)["finger_rot6d"]

    @torch.no_grad()
    def predict(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inference helper: adds axis-angle ``finger_pose[B,15,3]`` (+ ``global_orient[B,3]``)."""
        out = self.forward_all(feat)
        out["finger_pose"] = to_axis_angle(out["finger_rot6d"])
        if "global_rot6d" in out:
            out["global_orient"] = to_axis_angle(out["global_rot6d"])
        return out


def to_axis_angle(rot6d: torch.Tensor) -> torch.Tensor:
    """6D ``[...,6]`` → axis-angle ``[...,3]`` (Gram–Schmidt, then log map)."""
    return matrix_to_aa(sixd_to_matrix(rot6d))


# ── loss ───────────────────────────────────────────────────────────────────
def rotation_geodesic(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Angle (rad) of ``R1ᵀR2`` via ``atan2(|sin|, cos)`` — finite gradients at 0 and π."""
    M = R1.transpose(-1, -2) @ R2
    tr = M.diagonal(dim1=-2, dim2=-1).sum(-1)
    v = torch.stack([M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0], M[..., 1, 0] - M[..., 0, 1]], -1)
    return torch.atan2(0.5 * v.norm(dim=-1), 0.5 * (tr - 1.0))


def _masked_mean(x: torch.Tensor, valid) -> torch.Tensor:
    """Weighted mean of ``x[B,...]``; ``valid[B]`` (or ``[B,...]``) broadcasts from the left."""
    if valid is None:
        return x.mean()
    w = as_tensor(valid).to(dtype=x.dtype, device=x.device)
    if w.ndim > x.ndim or tuple(w.shape) != tuple(x.shape[:w.ndim]):
        raise ValueError(f"valid must match the leading dims of {tuple(x.shape)}, got {tuple(w.shape)}")
    w = w.reshape(*w.shape, *([1] * (x.ndim - w.ndim))).expand_as(x)
    return (x * w).sum() / w.sum().clamp_min(1e-8)


def hand_pose_loss(pred6d: torch.Tensor, gt_aa, skeleton: ManoSkeleton | None = None,
                   tip_weight: float = 10.0, *, valid=None, pred_global6d: torch.Tensor | None = None,
                   gt_global=None, global_weight: float = 1.0) -> dict[str, torch.Tensor]:
    """Finger-pose loss: mean geodesic angle (rad) over the 15 joints
    + ``tip_weight`` × mean fingertip distance (m, wrist frame, via ``skeleton``) when a skeleton is
    given + ``global_weight`` × global-orientation geodesic when ``pred_global6d``/``gt_global`` are.

    ``pred6d[B,15,6]``, ``gt_aa[B,15,3]``, ``valid[B]`` (bool/float weights, optional).
    Returns ``{"loss", "rot", ["tip"], ["global"]}`` (``rot``/``global`` in rad, ``tip`` in m).
    Default weight: 1 cm fingertip error ≈ 0.1 rad (5.7°).
    """
    if pred6d.shape[-2:] != (15, 6):
        raise ValueError(f"pred6d must be [...,15,6], got {tuple(pred6d.shape)}")
    gt = as_tensor(gt_aa).to(dtype=pred6d.dtype, device=pred6d.device)
    if tuple(gt.shape) != tuple(pred6d.shape[:-1]) + (3,):
        raise ValueError(f"gt_aa must be {tuple(pred6d.shape[:-1]) + (3,)} (axis-angle), got {tuple(gt.shape)}")
    R_pred = sixd_to_matrix(pred6d)
    R_gt = aa_to_matrix(gt)
    rot = _masked_mean(rotation_geodesic(R_pred, R_gt).mean(-1), valid)
    out = {"rot": rot}
    loss = rot
    if skeleton is not None and tip_weight > 0:
        tp = skeleton.forward_matrices(finger_rot=R_pred)["tip_pos"]
        tg = skeleton.forward_matrices(finger_rot=R_gt)["tip_pos"]
        tip = _masked_mean((tp - tg).norm(dim=-1).mean(-1), valid)
        out["tip"] = tip
        loss = loss + tip_weight * tip
    if pred_global6d is not None and gt_global is not None:
        g = as_tensor(gt_global).to(dtype=pred6d.dtype, device=pred6d.device)
        glob = _masked_mean(rotation_geodesic(sixd_to_matrix(pred_global6d), aa_to_matrix(g)), valid)
        out["global"] = glob
        loss = loss + global_weight * glob
    out["loss"] = loss
    return out


# ── sequence inference ─────────────────────────────────────────────────────
@torch.no_grad()
def predict_finger_pose_sequence(model: ImuHandPoseNet, quat, gyro=None, acc=None, *, window: int,
                                 wrist_index: int = 0, vec_frame: str = "sensor", batch_size: int = 1024,
                                 device: str | torch.device | None = None) -> dict[str, np.ndarray]:
    """Causal per-frame inference over a whole (calibrated) IMU sequence.

    ``quat[T,S,4]`` (+ ``gyro``/``acc`` ``[T,S,3]`` — pass the same streams the model was trained
    with) → ``{"finger_pose": [T,15,3], ("global_orient": [T,3])}`` float32. Frame t sees the
    window ending at t (the start is edge-padded), exactly like training windows from
    :func:`imu_windows`.
    """
    feat = imu_features(np.asarray(quat), None if gyro is None else np.asarray(gyro),
                        None if acc is None else np.asarray(acc), wrist_index, vec_frame=vec_frame)
    if feat.shape[-1] != model.in_dim:
        raise ValueError(f"IMU features have {feat.shape[-1]} dims but the model expects {model.in_dim}; "
                         "pass the same gyro/acc streams used for training")
    win = imu_windows(feat, window)
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    fingers, globs = [], []
    try:
        for s in range(0, win.shape[0], max(1, int(batch_size))):
            x = torch.as_tensor(np.ascontiguousarray(win[s:s + batch_size]), dtype=torch.float32, device=dev)
            out = model.predict(x)
            fingers.append(out["finger_pose"].float().cpu().numpy())
            if "global_orient" in out:
                globs.append(out["global_orient"].float().cpu().numpy())
    finally:
        model.train(was_training)
    res = {"finger_pose": np.concatenate(fingers).astype(np.float32)}
    if globs:
        res["global_orient"] = np.concatenate(globs).astype(np.float32)
    return res
