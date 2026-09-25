"""Glove 7-IMU → MANO pose → taxel poses.

:class:`GloveImu2ManoPoseProvider` runs an IMU→finger-pose network offline over a whole
recording (causal windows, see ``imu_model.predict_finger_pose_sequence``), turns the MANO pose
into segment transforms with ``mano.ManoSkeleton`` and precomputes taxel poses; ``pose_at(t)``
interpolates them. The same poses feed ``contact.self_touch`` via ``mano.self_touch_from_hand``.

Global orientation comes from the (calibrated) wrist IMU unless given; the wrist *position* is not
observable from IMUs and defaults to the origin (hand frame) unless given (e.g. from vision).

Backbone: the intended pretrained model is VIHand **VIFNet-S** (IMU-only variant, user-specified).
Its weights and code are external and not bundled, so :func:`load_vifnet_s` /
:func:`finetune_vifnet_s` are documented stubs; any ``nn.Module`` with the
``imu_model.ImuHandPoseNet`` interface (``in_dim``, ``predict(feat[B,W,F])`` → ``finger_pose``)
can be plugged in. The in-house baseline is ``imu_model.ImuHandPoseNet``
(trained by ``robot_skin.stages.imu_pose``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from common.layouts import MANO_SEGMENTS, Layout
from robot_skin.geometry.rotations import matrix_to_aa, quat_to_matrix
from robot_skin.pose.imu_model import (
    ImuHandPoseNet, apply_imu_offsets, apply_imu_offsets_to_vectors, imu_site_segments,
    predict_finger_pose_sequence,
)
from robot_skin.pose.mano import ManoSkeleton, _parent_index, taxel_poses_from_hand


def load_vifnet_s(checkpoint: str | Path, **kw) -> torch.nn.Module:
    """Load pretrained VIHand VIFNet-S weights (stub).

    Not implemented: VIFNet-S code/weights are external. To plug it in, wrap the released model
    so that it exposes ``in_dim`` and ``predict(feat[B,W,F]) -> {"finger_pose": [B,15,3]}`` on
    features from ``imu_model.imu_features`` (or adapt the feature function to its expected input
    layout: sensor order, units, frames), then pass it as ``model`` to
    :class:`GloveImu2ManoPoseProvider`.
    """
    raise NotImplementedError(
        "VIFNet-S weights are external and not bundled. Wrap the released model to expose "
        "in_dim / predict(feat) -> {'finger_pose'} (see docstring), or train the in-house "
        "ImuHandPoseNet with `robot_skin.stages.imu_pose`.")


def finetune_vifnet_s(train_sessions: list[Path], out_dir: Path, **kw) -> Path:
    """Fine-tune VIFNet-S on glove sessions (stub; external weights).

    Planned recipe: load with :func:`load_vifnet_s`, replace the input adapter for our 7 sites
    (wrist/palm/thumb/index/middle/ring/pinky), fine-tune on D1 ``motion`` episodes with
    vision-derived MANO labels (``hand_pose_valid``) using ``imu_model.hand_pose_loss``. Until the
    weights are available, ``stages.imu_pose`` trains ``ImuHandPoseNet`` (same role; VIFNet-S I/O unverified).
    """
    raise NotImplementedError(
        "VIFNet-S fine-tuning needs the external VIFNet-S weights (see load_vifnet_s); "
        "use robot_skin.stages.imu_pose (ImuHandPoseNet) meanwhile.")


def _find_wrist_site(layout: Layout) -> int:
    for i, s in enumerate(layout.imu_sites):
        if s.parent == "wrist":
            return i
    return 0


class GloveImu2ManoPoseProvider:
    """``TaxelPoseProvider`` for a glove from its IMU recording (offline, precomputed).

    Parameters
    ----------
    layout : glove layout (``parent_frame: mano``, with ``imu_sites`` in the IMU column order).
    model : IMU→pose network (``ImuHandPoseNet`` or a compatible wrapper).
    skeleton : ``ManoSkeleton`` (None → default).
    t : [T] timestamps (s). quat : [T,S,4] raw or calibrated wxyz; gyro / acc : [T,S,3] or None
        (must match the streams the model was trained with).
    window : causal window length (frames) used by the model.
    offsets / world : IMU calibration (``imu_model.calibrate_imu_offsets``); None → already calibrated.
    global_orient : [T,3] override; else from the wrist IMU when ``global_from_imu``; else zeros.
    wrist_pos : [T,3] (None → origin).

    Frame: with the default ``global_from_imu=True`` the taxel poses are in the (calibrated) IMU world
    frame. Processed episodes, the baseline model and VTLA use the **hand frame** (``global_orient =
    0``, wrist at the origin; ``datasets.build`` / ``control.online.glove_pose_fn``) — pass
    ``global_from_imu=False`` (and no ``wrist_pos``) to feed those models.
    """

    def __init__(self, layout: Layout, model: torch.nn.Module, skeleton: ManoSkeleton | None, t, quat,
                 gyro=None, acc=None, *, window: int = 32, wrist_index: int | None = None,
                 offsets=None, world=None, global_orient=None, wrist_pos=None,
                 global_from_imu: bool = True, vec_frame: str = "sensor", batch_size: int = 1024,
                 device: str | torch.device | None = None):
        if layout.parent_frame != "mano":
            raise ValueError("GloveImu2ManoPoseProvider needs a layout with parent_frame: mano")
        _parent_index(layout)
        self.layout = layout
        self.skeleton = skeleton if skeleton is not None else ManoSkeleton()
        self.t = np.asarray(t, dtype=np.float64)
        q = np.asarray(quat, dtype=np.float64)
        if self.t.ndim != 1:
            raise ValueError(f"t must be [T] timestamps (argument order: ..., t, quat, gyro, acc), got {self.t.shape}")
        T = self.t.shape[0]
        if q.ndim != 3 or q.shape[0] != T or q.shape[2] != 4:
            raise ValueError(f"quat must be [T={T}, S, 4], got {q.shape}")
        if T > 1 and np.any(np.diff(self.t) <= 0):
            raise ValueError("t must be strictly increasing")
        if layout.imu_sites and len(layout.imu_sites) != q.shape[1]:
            raise ValueError(f"layout has {len(layout.imu_sites)} imu_sites but quat has {q.shape[1]} sites")
        wi = _find_wrist_site(layout) if wrist_index is None else int(wrist_index)
        if offsets is None and world is not None:          # world alignment only
            offsets = np.tile([1.0, 0.0, 0.0, 0.0], (q.shape[1], 1))
        if offsets is not None:
            q = apply_imu_offsets(q, offsets, world=world)
            # world-frame vectors (vec_frame="world") take G⁻¹, not the per-site mounting offsets
            vkw = {"vec_frame": vec_frame, "world": world}
            gyro = None if gyro is None else apply_imu_offsets_to_vectors(np.asarray(gyro, dtype=np.float64), offsets,
                                                                          **vkw)
            acc = None if acc is None else apply_imu_offsets_to_vectors(np.asarray(acc, dtype=np.float64), offsets,
                                                                        **vkw)
        pred = predict_finger_pose_sequence(model, q, gyro, acc, window=window, wrist_index=wi,
                                            vec_frame=vec_frame, batch_size=batch_size, device=device)
        self.finger_pose = pred["finger_pose"].astype(np.float64)
        if global_orient is not None:
            go = np.asarray(global_orient, dtype=np.float64).reshape(T, 3)
        elif "global_orient" in pred:
            go = pred["global_orient"].astype(np.float64)
        elif global_from_imu and layout.imu_sites:
            # wrist-site segment orientation → joint-0 (MANO global) orientation
            seg = MANO_SEGMENTS[imu_site_segments(layout)[wi]]
            R_site = quat_to_matrix(torch.as_tensor(q[:, wi]))
            R_local = torch.as_tensor(self.skeleton.segment_local_rotation(seg))
            go = matrix_to_aa(R_site @ R_local.T).numpy()
        else:
            go = np.zeros((T, 3))
        self.global_orient = go
        self.wrist_pos = np.zeros((T, 3)) if wrist_pos is None else np.asarray(wrist_pos, dtype=np.float64).reshape(T, 3)
        self._pos, self._nrm = taxel_poses_from_hand(layout, self.skeleton, self.global_orient,
                                                     self.finger_pose, self.wrist_pos)

    @property
    def n_taxels(self) -> int:
        return self.layout.n

    @property
    def poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Precomputed ``(pos[T,N,3], nrm[T,N,3])`` at the IMU timestamps."""
        return self._pos, self._nrm

    def pose_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Linear interpolation between IMU frames (clamped at the ends); normals renormalised."""
        ts = self.t
        if ts.shape[0] == 1 or t <= ts[0]:
            return self._pos[0].copy(), self._nrm[0].copy()
        if t >= ts[-1]:
            return self._pos[-1].copy(), self._nrm[-1].copy()
        i = int(np.searchsorted(ts, t, side="right")) - 1
        a = (t - ts[i]) / (ts[i + 1] - ts[i])
        pos = (1 - a) * self._pos[i] + a * self._pos[i + 1]
        nrm = (1 - a) * self._nrm[i] + a * self._nrm[i + 1]
        return pos, nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)


__all__ = ["GloveImu2ManoPoseProvider", "ImuHandPoseNet", "finetune_vifnet_s", "load_vifnet_s"]
