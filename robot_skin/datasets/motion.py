"""D1 (``motion``) training datasets over processed episodes: causal windows on the master clock.

Every sample is anchored at a master-clock frame ``t`` and looks only at the **history**
``t−W+1 … t`` (frames before the episode start are edge-padded with frame 0), so a model trained on
it can run online with a ring buffer — the same windows ``control.online`` builds live.

- :class:`BaselineWindowDataset` — motion-induced tactile baseline (no-contact artefact) regression:
  joint history ``q_hist[W,D]``, ``qd_hist[W,D]`` + taxel pose at ``t`` → target ``y[N]`` = ΔS at
  ``t``, ``valid[N]`` only for taxels labelled no-contact (``contact_label`` ∈ ``only_labels``) and not
  saturated. The observed ΔS is never an input (lesson of the SATS bending restorer: a model that
  sees ΔS learns to erase contact).
- :class:`ContactWindowDataset` — per-taxel contact classification from the calibrated residual
  history ``z_hist[W,N]`` (derived ``residual_z`` of the baseline/contact stages) + ``q``/``qd`` at
  ``t``; labels from ``contact_label`` ≥ 0 (self-touch = 1, no-contact = 0).
- :class:`ImuPoseWindowDataset` — glove IMUs → MANO finger pose: ``pose.imu_model.imu_features``
  window ``[W,F]`` → ``finger_pose[15,3]`` (axis-angle, MANO order) at ``t`` where
  ``hand_pose_valid``.

Normalisation statistics are **passed in** (fit with ``datasets.stats.compute_stats`` on the train
split: keys ``q``, ``qd``, ``imu_features``), never fit here, so val/test/online share the train
statistics. ``None`` keeps raw units.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from common.signal import NormStats

from .episode import (
    D_RESIDUAL_Z, K_CONTACT_LABEL, K_DELTA, K_HAND_FINGERS, K_HAND_GLOBAL, K_HAND_VALID, K_IMU_ACC, K_IMU_GYRO,
    K_IMU_QUAT, K_Q, K_QD, K_SATURATED, K_TAXEL_NRM, K_TAXEL_POS, Episode,
)
from .stats import IMU_FEATURES, as_episodes, q_valid_mask, qd_valid_mask

__all__ = ["causal_window", "imu_wrist_index", "episode_imu_features", "q_valid_mask", "qd_valid_mask",
           "BaselineWindowDataset", "ContactWindowDataset", "ImuPoseWindowDataset"]


def causal_window(t: int, window: int) -> np.ndarray:
    """Frame indices ``[W]`` of the causal window ending at ``t`` (edge-padded with 0)."""
    if window < 1:
        raise ValueError("window must be ≥ 1")
    return np.clip(np.arange(int(t) - int(window) + 1, int(t) + 1), 0, None)


def _window_all(mask: np.ndarray, window: int) -> np.ndarray:
    """``[T]`` True where every frame of the causal window is True (edge-padded)."""
    bad = (~np.asarray(mask, dtype=bool)).astype(np.int64)
    pad = np.concatenate([np.repeat(bad[:1], window - 1), bad])
    c = np.concatenate([[0], np.cumsum(pad)])
    return (c[window:] - c[:-window]) == 0


def _label_array(ep: Episode, key: str, who: str) -> np.ndarray:
    """Per-taxel label array ``[T,N]``: the episode array ``key`` (preprocessing ``contact_label``) or,
    when absent, the derived array of that name (e.g. the contact stage's ``contact_label_pseudo``,
    ``datasets.episode.D_CONTACT_LABEL_PSEUDO``)."""
    if ep.has(key):
        lab = ep[key]
    elif ep.has_derived(key):
        lab = ep.derived(key)
    else:
        raise ValueError(f"episode {ep.meta.episode_id!r} lacks {key!r} (neither an array nor a derived "
                         f"array; needed by {who})")
    if lab.shape != (ep.T, ep.meta.n_taxels):
        raise ValueError(f"episode {ep.meta.episode_id!r}: labels {key!r} have shape {lab.shape}, "
                         f"expected {(ep.T, ep.meta.n_taxels)}")
    return lab


def imu_wrist_index(ep: Episode) -> int:
    """Index of the wrist IMU in ``ep.meta.imu_sites`` (recorded by preprocessing; else the site
    named ``wrist``; else 0)."""
    imu = (ep.meta.preprocessing or {}).get("imu") or {}
    if "wrist_index" in imu:
        return int(imu["wrist_index"])
    return ep.meta.imu_sites.index("wrist") if "wrist" in ep.meta.imu_sites else 0


def episode_imu_features(ep: Episode, *, gyro: bool = True, acc: bool = True, vec_frame: str = "sensor",
                         wrist_index: int | None = None) -> np.ndarray:
    """``pose.imu_model.imu_features`` of the (calibrated) episode IMU arrays → ``[T,F]`` float32
    (per site: 6D orientation relative to the wrist IMU | gyro | acc in the wrist frame)."""
    from robot_skin.pose.imu_model import imu_features

    if not ep.has(K_IMU_QUAT):
        raise ValueError(f"episode {ep.meta.episode_id!r} has no IMU arrays")
    # np.array copies: the arrays are usually read-only memmaps (torch warns on those)
    g = np.array(ep[K_IMU_GYRO]) if gyro and ep.has(K_IMU_GYRO) else None
    a = np.array(ep[K_IMU_ACC]) if acc and ep.has(K_IMU_ACC) else None
    if gyro and g is None or acc and a is None:
        raise ValueError(f"episode {ep.meta.episode_id!r} lacks imu_gyro/imu_acc requested for the features")
    wi = imu_wrist_index(ep) if wrist_index is None else int(wrist_index)
    return imu_features(np.array(ep[K_IMU_QUAT]), g, a, wi, vec_frame=vec_frame)


def _joint_stats(joint_stats) -> tuple[NormStats | None, NormStats | None]:
    if joint_stats is None:
        return None, None
    if isinstance(joint_stats, Mapping):
        return joint_stats.get(K_Q), joint_stats.get(K_QD)
    q, qd = joint_stats
    return q, qd


def _same(values: Iterable, what: str, cls: str):
    vals = set(values)
    if len(vals) != 1:
        raise ValueError(f"{cls}: episodes have different {what} {sorted(vals, key=str)}")
    return vals.pop()


def _norm(stats: NormStats | None, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if stats is None:
        return np.ascontiguousarray(x)
    if stats.offset.shape[0] != x.shape[-1]:
        raise ValueError(f"stats have {stats.offset.shape[0]} features, data {x.shape[-1]}")
    return stats.apply(x)


class _WindowDataset(Dataset):
    """Shared plumbing: episodes, the ``(episode, frame)`` index, causal windows."""

    def __init__(self, episodes: Iterable[Episode | str | Path], window: int, stride: int):
        if int(window) < 1 or int(stride) < 1:
            raise ValueError("window and stride must be ≥ 1")
        self.episodes: list[Episode] = as_episodes(episodes)
        if not self.episodes:
            raise ValueError("no episodes")
        self.window, self.stride = int(window), int(stride)
        self._index: list[np.ndarray] = []

    def _add(self, e: int, eligible: np.ndarray) -> None:
        idx = np.flatnonzero(eligible)[:: self.stride]
        self._index.append(np.stack([np.full(idx.shape[0], e), idx], 1).astype(np.int64))

    def _finish(self) -> None:
        self.index = np.concatenate(self._index) if self._index else np.zeros((0, 2), dtype=np.int64)
        del self._index
        if self.index.shape[0] == 0:
            raise ValueError(f"{type(self).__name__}: no eligible frames in {len(self.episodes)} episode(s)")

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def _at(self, i: int) -> tuple[int, int, np.ndarray]:
        e, t = (int(v) for v in self.index[i])
        return e, t, causal_window(t, self.window)


class BaselineWindowDataset(_WindowDataset):
    """No-contact baseline windows (see module docstring).

    Sample: ``q_hist[W,D]``, ``qd_hist[W,D]`` (normalised with ``joint_stats``), ``pos[N,3]``,
    ``nrm[N,3]`` (taxel pose at ``t``), ``y[N]`` (ΔS % at ``t``; 0 where not valid), ``valid[N]``,
    ``episode``, ``t_index``. A frame is used when ≥ ``min_valid`` taxels are valid and (with
    ``require_q_valid``) every frame of its window has a measured ``q`` *and* ``qd``
    (:func:`qd_valid_mask`: no held hand label inside the derivative filter's footprint either).
    ``joint_stats``: ``{"q": NormStats, "qd": NormStats}`` (``compute_stats`` output) or a tuple.
    ``label_key``: an episode array (default ``contact_label``) or, when the episode has no such
    array, a derived one (e.g. ``contact_label_pseudo`` of the contact stage).
    All episodes must share ``n_taxels``, the joint dimension and ``meta.joint_names``."""

    def __init__(self, episodes: Iterable[Episode | str | Path], *, window: int = 32, stride: int = 1,
                 only_labels: Sequence[int] = (0,), joint_stats: Mapping[str, NormStats] | tuple | None = None,
                 exclude_saturated: bool = True, min_valid: int = 1, require_q_valid: bool = True,
                 label_key: str = K_CONTACT_LABEL):
        super().__init__(episodes, window, stride)
        self.q_stats, self.qd_stats = _joint_stats(joint_stats)
        self.only_labels = tuple(int(v) for v in only_labels)
        self._q, self._qd, self._valid = [], [], []
        dims = set()
        for e, ep in enumerate(self.episodes):
            for k in (K_Q, K_QD, K_DELTA, K_TAXEL_POS, K_TAXEL_NRM):
                if not ep.has(k):
                    raise ValueError(f"episode {ep.meta.episode_id!r} lacks {k!r} (needed by BaselineWindowDataset)")
            lab = _label_array(ep, label_key, "BaselineWindowDataset")
            dims.add(ep[K_Q].shape[1])
            self._q.append(_norm(self.q_stats, ep[K_Q]))
            self._qd.append(_norm(self.qd_stats, ep[K_QD]))
            valid = np.isin(np.asarray(lab), self.only_labels) & np.isfinite(np.asarray(ep[K_DELTA]))
            if exclude_saturated and ep.has(K_SATURATED):
                valid &= ~np.asarray(ep[K_SATURATED], dtype=bool)
            self._valid.append(valid)
            ok = valid.sum(1) >= int(min_valid)
            if require_q_valid:                               # qd_valid ⊆ q_valid (footprint includes t)
                ok &= _window_all(qd_valid_mask(ep), self.window)
            self._add(e, ok)
        name = type(self).__name__
        self.joint_dim = int(_same(dims, "joint dims", name))
        _same((tuple(ep.meta.joint_names) for ep in self.episodes), "joint_names (q column order)", name)
        self.n_taxels = int(_same((ep.meta.n_taxels for ep in self.episodes), "n_taxels", name))
        self._finish()

    def __getitem__(self, i: int) -> dict[str, Any]:
        e, t, w = self._at(i)
        ep = self.episodes[e]
        valid = self._valid[e][t]
        y = np.where(valid, np.asarray(ep[K_DELTA][t], dtype=np.float32), 0.0).astype(np.float32)
        return {"q_hist": torch.from_numpy(self._q[e][w]), "qd_hist": torch.from_numpy(self._qd[e][w]),
                "pos": torch.from_numpy(np.array(ep[K_TAXEL_POS][t], dtype=np.float32)),
                "nrm": torch.from_numpy(np.array(ep[K_TAXEL_NRM][t], dtype=np.float32)),
                "y": torch.from_numpy(y), "valid": torch.from_numpy(np.array(valid, dtype=bool)),
                "episode": e, "t_index": t}


class ContactWindowDataset(_WindowDataset):
    """Contact-detector windows over the calibrated residual (see module docstring).

    Sample: ``z_hist[W,N]`` (derived ``z_key``, press-positive z; non-finite → 0), ``sat_hist[W,N]``,
    ``q[D]`` / ``qd[D]`` at ``t`` (normalised), ``label[N]`` (1.0 = contact), ``label_mask[N]``
    (``contact_label ≥ 0`` and, with ``exclude_saturated``, not saturated), ``q_valid`` (scalar bool:
    ``q``/``qd`` at ``t`` are measurements, :func:`qd_valid_mask` — a glove frame without a valid hand
    label carries the *held* pose and ``qd`` ≈ 0, which must not read as "hand at rest"),
    ``episode``, ``t_index``. Frames with ≥ ``min_labelled`` labelled taxels are used. Labels come
    from ``label_key``: the episode array (default ``contact_label``) or, when absent, the derived
    array of that name — ``label_key="contact_label_pseudo"`` trains on the contact stage's D2 pseudo
    labels directly. Episodes need ``derived/<z_key>.npy`` ``[T,N]`` and must share ``n_taxels`` /
    joint dims."""

    def __init__(self, episodes: Iterable[Episode | str | Path], *, window: int = 16, stride: int = 1,
                 joint_stats: Mapping[str, NormStats] | tuple | None = None, z_key: str = D_RESIDUAL_Z,
                 label_key: str = K_CONTACT_LABEL, exclude_saturated: bool = True, min_labelled: int = 1):
        super().__init__(episodes, window, stride)
        self.q_stats, self.qd_stats = _joint_stats(joint_stats)
        self._z, self._q, self._qd, self._mask, self._sat, self._qv, self._lab = [], [], [], [], [], [], []
        dims = set()
        for e, ep in enumerate(self.episodes):
            if not ep.has_derived(z_key):
                raise ValueError(f"episode {ep.meta.episode_id!r} has no derived {z_key!r} "
                                 "(run the baseline/contact stages first)")
            for k in (K_Q, K_QD):
                if not ep.has(k):
                    raise ValueError(f"episode {ep.meta.episode_id!r} lacks {k!r} (needed by ContactWindowDataset)")
            lab_arr = _label_array(ep, label_key, "ContactWindowDataset")
            z = ep.derived(z_key)
            if z.shape != (ep.T, ep.meta.n_taxels):
                raise ValueError(f"episode {ep.meta.episode_id!r}: derived {z_key!r} has shape {z.shape}, "
                                 f"expected {(ep.T, ep.meta.n_taxels)}")
            dims.add(ep[K_Q].shape[1])
            self._z.append(z)
            self._qv.append(qd_valid_mask(ep))
            self._q.append(_norm(self.q_stats, ep[K_Q]))
            self._qd.append(_norm(self.qd_stats, ep[K_QD]))
            self._lab.append(lab_arr)
            lab = np.asarray(lab_arr)
            sat = np.asarray(ep[K_SATURATED], dtype=bool) if ep.has(K_SATURATED) else np.zeros(lab.shape, bool)
            m = lab >= 0
            if exclude_saturated:
                m &= ~sat
            self._mask.append(m)
            self._sat.append(sat)
            self._add(e, m.sum(1) >= int(min_labelled))
        self.label_key = label_key
        name = type(self).__name__
        self.joint_dim = int(_same(dims, "joint dims", name))
        self.n_taxels = int(_same((ep.meta.n_taxels for ep in self.episodes), "n_taxels", name))
        self._finish()

    def __getitem__(self, i: int) -> dict[str, Any]:
        e, t, w = self._at(i)
        z = np.nan_to_num(np.asarray(self._z[e][w], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        lab = np.asarray(self._lab[e][t])
        return {"z_hist": torch.from_numpy(z), "sat_hist": torch.from_numpy(np.array(self._sat[e][w], dtype=bool)),
                "q": torch.from_numpy(self._q[e][t].copy()), "qd": torch.from_numpy(self._qd[e][t].copy()),
                "label": torch.from_numpy((lab == 1).astype(np.float32)),
                "label_mask": torch.from_numpy(np.array(self._mask[e][t], dtype=bool)),
                "q_valid": torch.tensor(bool(self._qv[e][t])), "episode": e, "t_index": t}


class ImuPoseWindowDataset(_WindowDataset):
    """IMU → finger-pose windows (see module docstring).

    Sample: ``feat[W,F]`` (``episode_imu_features`` normalised with ``imu_stats`` — a NormStats or the
    ``compute_stats`` dict with key ``imu_features``), ``finger_pose[15,3]`` and ``global_orient[3]``
    (axis-angle labels at ``t``), ``episode``, ``t_index``. Only frames with ``hand_pose_valid``.
    All episodes must list the same ``meta.imu_sites`` (feature columns are per site, in that order)."""

    def __init__(self, episodes: Iterable[Episode | str | Path], *, window: int = 32, stride: int = 1,
                 imu_stats: NormStats | Mapping[str, NormStats] | None = None, gyro: bool = True, acc: bool = True,
                 vec_frame: str = "sensor", wrist_index: int | None = None):
        super().__init__(episodes, window, stride)
        if isinstance(imu_stats, Mapping):
            imu_stats = imu_stats.get(IMU_FEATURES)
        self.imu_stats = imu_stats
        self.feature_kw = {"gyro": gyro, "acc": acc, "vec_frame": vec_frame, "wrist_index": wrist_index}
        self._feat = []
        dims = set()
        for e, ep in enumerate(self.episodes):
            for k in (K_HAND_FINGERS, K_HAND_VALID):
                if not ep.has(k):
                    raise ValueError(f"episode {ep.meta.episode_id!r} lacks {k!r} (needed by ImuPoseWindowDataset)")
            f = _norm(imu_stats, episode_imu_features(ep, **self.feature_kw))
            dims.add(f.shape[1])
            self._feat.append(f)
            self._add(e, np.asarray(ep[K_HAND_VALID], dtype=bool))
        name = type(self).__name__
        _same((tuple(ep.meta.imu_sites) for ep in self.episodes), "IMU sites", name)
        raw = [ep.meta.episode_id for ep in self.episodes
               if ((ep.meta.preprocessing or {}).get("imu") or {}).get("calibrated") is False]
        if raw:                              # wrist-relative features then carry per-session mounting offsets
            warnings.warn(f"{name}: {len(raw)} episode(s) with uncalibrated IMUs (no manifest imu_offsets), "
                          f"e.g. {raw[:3]}; their features include the glove's mounting offsets — rebuild with "
                          "imu.calibrate_if_missing: true or calibrate the raw sessions", stacklevel=2)
        self.feature_dim = int(_same(dims, "IMU feature dims", name))
        self._finish()

    def __getitem__(self, i: int) -> dict[str, Any]:
        e, t, w = self._at(i)
        ep = self.episodes[e]
        return {"feat": torch.from_numpy(self._feat[e][w]),
                "finger_pose": torch.from_numpy(np.array(ep[K_HAND_FINGERS][t], dtype=np.float32)),
                "global_orient": torch.from_numpy(np.array(ep[K_HAND_GLOBAL][t], dtype=np.float32)),
                "episode": e, "t_index": t}
