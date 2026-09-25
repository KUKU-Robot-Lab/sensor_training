"""Camera-based MANO hand labels: estimator protocol, ``hand_pose.npz`` IO, label smoothing.

D1/D2 sessions get their hand-pose *labels* from video: an off-the-shelf monocular MANO
regressor (HaMeR: Pavlakos et al., *Reconstructing Hands in 3D with Transformers*, CVPR 2024;
WiLoR — Potamias et al., CVPR 2025, arXiv:2409.12259 — is a drop-in alternative) is run
**offline** on the egocentric / third-person frames and its
per-frame MANO parameters are written to ``hand_pose.npz`` in the raw session directory
(format in ``acquisition.manifest``)::

    t[T] s (host clock) · global_orient[T,3] · finger_pose[T,15,3] (MANO order, axis-angle,
    relative to the flat template) · wrist_pos[T,3] m · confidence[T] ∈ [0,1]

These labels supervise the IMU→pose model (``imu_model``) and give taxel poses / self-touch
labels for glove sessions (``mano``). Monocular estimates jitter and drop out, so
:func:`smooth_hand_labels` gates by confidence, SLERP-fills short gaps and low-pass filters.

Frame conventions: ``global_orient``/``wrist_pos`` are in the camera (or calibrated world) frame
the estimator reports; if the estimator outputs MANO ``transl``, convert with
``wrist_pos = transl + J_0(β)`` (see ``mano`` module docstring). If it uses MANO's
``hands_mean`` offset (``flat_hand_mean=False``), add ``hands_mean`` to ``finger_pose`` first.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from robot_skin.geometry.rotations import aa_to_quat, quat_fix_continuity, quat_to_aa

HAND_LABEL_KEYS = ("t", "global_orient", "finger_pose", "wrist_pos", "confidence")


@runtime_checkable
class VisionHandEstimator(Protocol):
    """Monocular MANO regressor: ``frames uint8[B,H,W,3]`` (RGB) → dict with
    ``global_orient[B,3]``, ``finger_pose[B,15,3]``, ``wrist_pos[B,3]``, ``confidence[B]``."""

    def estimate(self, frames: np.ndarray) -> dict[str, np.ndarray]: ...


class HaMeREstimator:
    """Placeholder for HaMeR / WiLoR (stub — run them offline).

    These models need their own environments, detector weights and GPU; we do not wrap them
    in-process. Procedure:
      1. export frames per camera (``camera_<name>/frames.npy`` or JPEGs + ``timestamps.npy``);
      2. run the HaMeR (or WiLoR) demo/inference script on them, keeping the right-hand
         detection per frame (highest score) and its MANO outputs (global_orient, hand_pose as
         axis-angle, betas, camera translation) plus the detection score;
      3. convert: ``finger_pose`` = hand_pose (+ hands_mean if the model uses it), ``wrist_pos`` =
         camera-frame wrist joint (``transl + J_0(β)``), ``confidence`` = detection score;
      4. :func:`save_hand_labels` to ``<session>/hand_pose.npz`` with the frame timestamps and
         register the stream in the manifest.
    Any class with ``estimate(frames) -> dict`` satisfying :class:`VisionHandEstimator` can be
    used with :func:`estimate_sequence` instead.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "HaMeR/WiLoR are not bundled: run them offline on the session frames and write "
            "hand_pose.npz with robot_skin.pose.vision_hand.save_hand_labels (see class docstring).")

    def estimate(self, frames: np.ndarray) -> dict[str, np.ndarray]:  # pragma: no cover - stub
        raise NotImplementedError


def _check_labels(t, global_orient, finger_pose, wrist_pos, confidence) -> dict[str, np.ndarray]:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    T = t.shape[0]
    go = np.asarray(global_orient, dtype=np.float32)
    fp = np.asarray(finger_pose, dtype=np.float32)
    wp = np.asarray(wrist_pos, dtype=np.float32)
    conf = np.ones(T, dtype=np.float32) if confidence is None else np.asarray(confidence, dtype=np.float32).reshape(-1)
    for name, a, shape in (("global_orient", go, (T, 3)), ("finger_pose", fp, (T, 15, 3)),
                           ("wrist_pos", wp, (T, 3)), ("confidence", conf, (T,))):
        if a.shape != shape:
            raise ValueError(f"{name} must be {shape}, got {a.shape}")
    if T > 1 and np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    return {"t": t, "global_orient": go, "finger_pose": fp, "wrist_pos": wp, "confidence": conf}


def save_hand_labels(path: str | Path, t, global_orient, finger_pose, wrist_pos, confidence=None) -> Path:
    """Write ``hand_pose.npz`` (validated shapes; confidence defaults to 1).

    ``path`` is a ``.npz`` file or a session directory (created if needed; any path without the
    ``.npz`` suffix is a directory, so names like ``sess_2026.01.02`` are safe).
    """
    d = _check_labels(t, global_orient, finger_pose, wrist_pos, confidence)
    p = Path(path)
    if p.suffix != ".npz":
        p = p / "hand_pose.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **d)
    return p


def load_hand_labels(path: str | Path) -> dict[str, np.ndarray]:
    """Read and validate ``hand_pose.npz`` (a session directory is accepted too)."""
    p = Path(path)
    if p.is_dir():
        p = p / "hand_pose.npz"
    with np.load(p) as z:
        missing = [k for k in ("t", "global_orient", "finger_pose", "wrist_pos") if k not in z.files]
        if missing:
            raise KeyError(f"{p} lacks {missing}")
        return _check_labels(z["t"], z["global_orient"], z["finger_pose"], z["wrist_pos"],
                             z["confidence"] if "confidence" in z.files else None)


def estimate_sequence(estimator: VisionHandEstimator, frames: np.ndarray, t, *,
                      batch_size: int = 16) -> dict[str, np.ndarray]:
    """Run an estimator over ``frames[F,H,W,3]`` in batches → label dict (for :func:`save_hand_labels`)."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be [F,H,W,3], got {frames.shape}")
    outs = [estimator.estimate(frames[s:s + batch_size]) for s in range(0, frames.shape[0], batch_size)]
    cat = {k: np.concatenate([o[k] for o in outs]) for k in ("global_orient", "finger_pose", "wrist_pos")}
    conf = np.concatenate([o.get("confidence", np.ones(len(o["wrist_pos"]))) for o in outs])
    return _check_labels(t, cat["global_orient"], cat["finger_pose"], cat["wrist_pos"], conf)


# ── smoothing ──────────────────────────────────────────────────────────────
def _slerp(q0: np.ndarray, q1: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Shortest-path SLERP between unit quaternions ``[...,4]``; ``tau[...,1]`` ∈ [0,1]."""
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.clip(np.abs(d), 0.0, 1.0)
    theta = np.arccos(d)
    s = np.sin(theta)
    small = s < 1e-6
    s_safe = np.where(small, 1.0, s)
    w0 = np.where(small, 1.0 - tau, np.sin((1.0 - tau) * theta) / s_safe)
    w1 = np.where(small, tau, np.sin(tau * theta) / s_safe)
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` of consecutive True runs."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def _lowpass(x: np.ndarray, fs: float, cutoff_hz: float, order: int) -> np.ndarray:
    """Zero-phase low-pass along axis 0 (scipy Butterworth filtfilt; moving-average fallback)."""
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:  # pragma: no cover - scipy is normally present
        butter = None
    if butter is not None:
        wn = min(0.99, cutoff_hz / (0.5 * fs))
        b, a = butter(order, wn)
        if x.shape[0] > 3 * max(len(a), len(b)):
            return filtfilt(b, a, x, axis=0)
    w = int(round(0.443 * fs / cutoff_hz))                 # boxcar with ≈ the same −3 dB point
    w = min(w + (w + 1) % 2, x.shape[0] - (x.shape[0] + 1) % 2)   # odd → centred (zero phase)
    if w < 3:
        return x
    k = np.ones(w) / w
    pad = np.concatenate([np.repeat(x[:1], w // 2, 0), x, np.repeat(x[-1:], w // 2, 0)], 0)
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, pad)


def smooth_hand_labels(t, global_orient, finger_pose, wrist_pos, confidence=None, *,
                       min_conf: float = 0.5, max_gap_s: float = 0.25, cutoff_hz: float | None = 6.0,
                       order: int = 2) -> dict[str, np.ndarray]:
    """Confidence-gated gap filling + zero-phase low-pass of MANO label sequences.

    1. frames with ``confidence < min_conf`` (or non-finite values) are missing;
    2. interior gaps ≤ ``max_gap_s`` are filled by per-joint quaternion SLERP (rotations) and
       linear interpolation (wrist position) between the bracketing valid frames;
    3. each contiguous valid run is low-pass filtered (``cutoff_hz``; None → skip) on continuity-
       fixed quaternions (renormalised) and positions. Sampling rate = 1 / median Δt.
    Longer gaps hold the previous valid value (leading gaps: the first valid value) and are
    marked invalid.
    Returns the same keys as :func:`load_hand_labels` plus ``valid[T]`` bool.
    """
    d = _check_labels(t, global_orient, finger_pose, wrist_pos, confidence)
    t, T = d["t"], d["t"].shape[0]
    rot = np.concatenate([d["global_orient"][:, None], d["finger_pose"]], axis=1).astype(np.float64)  # [T,16,3]
    pos = d["wrist_pos"].astype(np.float64)
    ok = (d["confidence"] >= min_conf) & np.isfinite(rot).all(axis=(1, 2)) & np.isfinite(pos).all(axis=1)
    out = dict(d)
    if not ok.any():
        out["valid"] = np.zeros(T, dtype=bool)
        return out
    q = aa_to_quat(torch.as_tensor(np.nan_to_num(rot))).numpy()                                   # [T,16,4]
    idx = np.arange(T)
    prev = np.maximum.accumulate(np.where(ok, idx, -1))
    nxt = np.minimum.accumulate(np.where(ok, idx, T)[::-1])[::-1]
    interior = ~ok & (prev >= 0) & (nxt < T)
    fill = np.zeros(T, dtype=bool)
    if interior.any():
        pi, ni = prev[interior], nxt[interior]
        fill[interior] = (t[ni] - t[pi]) <= max_gap_s
    if fill.any():
        pi, ni = prev[fill], nxt[fill]
        tau = ((t[fill] - t[pi]) / (t[ni] - t[pi]))[:, None]
        q[fill] = _slerp(q[pi], q[ni], tau[:, None])
        pos[fill] = (1 - tau) * pos[pi] + tau * pos[ni]
    hold = ~ok & ~fill
    if hold.any():
        src = np.where(prev[hold] >= 0, prev[hold], nxt[hold])
        q[hold], pos[hold] = q[src], pos[src]
    valid = ok | fill
    if cutoff_hz is not None and T > 2:
        if cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be positive (or None)")
        fs = 1.0 / float(np.median(np.diff(t)))
        for s, e in _runs(valid):
            if e - s < 3:
                continue
            qr = quat_fix_continuity(q[s:e], axis=0)
            qr = _lowpass(qr.reshape(e - s, -1), fs, cutoff_hz, order).reshape(e - s, 16, 4)
            q[s:e] = qr / np.linalg.norm(qr, axis=-1, keepdims=True)
            pos[s:e] = _lowpass(pos[s:e], fs, cutoff_hz, order)
    aa = quat_to_aa(torch.as_tensor(q)).numpy().astype(np.float32)
    out.update(global_orient=aa[:, 0], finger_pose=aa[:, 1:], wrist_pos=pos.astype(np.float32), valid=valid)
    return out
