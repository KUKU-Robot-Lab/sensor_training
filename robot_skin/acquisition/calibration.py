"""Per-session glove IMU calibration from the protocol's flat-hand block (``imu_calibration``).

During the calibration block the subject holds a flat hand (fingers together and straight, palm
down, level, in the air). With the known segment orientations of that pose on the MANO skeleton
(``pose.imu_model.imu_reference_rotations``) the per-site mounting offsets follow from
``pose.imu_model.calibrate_imu_offsets`` after estimating the IMU-world alignment ``G`` from the
wrist IMU (``estimate_world_alignment``)::

    R_ref = imu_reference_rotations(layout, skeleton)        # flat hand, global_orient = 0
    G     = estimate_world_alignment(q_calib, R_ref, index=wrist_site)
    q_off = calibrate_imu_offsets(q_calib, R_ref, world=G)   # = M⁻¹ per site
    manifest.calibration |= imu_calibration_to_dict(q_off, G, sites)   # imu_offsets/imu_world/imu_sites

Preprocessing reads it back with ``imu_calibration_from_dict`` and applies ``apply_imu_offsets``.
A single static pose cannot separate ``G`` from the wrist IMU's own mounting (documented in
``estimate_world_alignment``): strap the wrist IMU aligned with the back of the hand.

Stillness is checked (gyro RMS, quaternion spread): a moving hand during the block biases the
offsets, so QC flags it and the operator repeats the block.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from common.layouts import Layout, load_layout

from .manifest import SessionManifest
from .recorder import load_events, phases_from_events

__all__ = ["CALIB_MAX_GYRO_RMS", "CALIB_MAX_SPREAD_DEG", "QUALITY_KEY", "calibrate_session_imu",
           "compute_imu_calibration", "find_calibration_phase"]

#: stillness limits for an accepted calibration block
CALIB_MAX_GYRO_RMS = 0.15     # rad/s (all sites)
CALIB_MAX_SPREAD_DEG = 3.0    # mean angular deviation from the site's mean orientation
QUALITY_KEY = "imu_calibration_quality"


def _angle_spread_deg(q: np.ndarray, qm: np.ndarray) -> np.ndarray:
    """Mean angle (deg) between ``q[T,S,4]`` and the per-site mean ``qm[S,4]``."""
    dots = np.abs(np.einsum("tsi,si->ts", q / np.linalg.norm(q, axis=-1, keepdims=True), qm))
    return np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0))).mean(0)


def compute_imu_calibration(quat, sites: Sequence[str], layout: Layout, *, gyro=None, skeleton=None,
                            wrist_site: str | None = None) -> tuple[dict, dict]:
    """Calibration dict (for ``manifest.calibration``) + quality report from a static flat-hand
    window ``quat[T,S,4]`` (raw sensor quaternions, site order ``sites``)."""
    from robot_skin.pose.imu_model import (
        calibrate_imu_offsets, estimate_world_alignment, imu_calibration_to_dict, imu_reference_rotations,
        mean_quaternion,
    )
    q = np.asarray(quat, dtype=np.float64)
    if q.ndim != 3 or q.shape[-1] != 4 or q.shape[0] < 3:
        raise ValueError(f"quat must be [T≥3, S, 4], got {q.shape}")
    sites = [str(s) for s in sites]
    if len(sites) != q.shape[1]:
        raise ValueError(f"{len(sites)} site names for {q.shape[1]} IMU sites")
    lay_sites = [s.name for s in layout.imu_sites]
    missing = [s for s in sites if s not in lay_sites]
    if missing:
        raise ValueError(f"IMU sites {missing} are not in layout {layout.name!r} ({lay_sites})")
    R_all = imu_reference_rotations(layout, skeleton)                     # layout site order
    R = R_all[[lay_sites.index(s) for s in sites]]
    if wrist_site is None:
        parents = {s.name: s.parent for s in layout.imu_sites}
        wrist_site = next((s for s in sites if parents.get(s) == "wrist"), sites[0])
    wi = sites.index(wrist_site)
    G = estimate_world_alignment(q, R, index=wi)
    off = calibrate_imu_offsets(q, R, world=G)
    calib = imu_calibration_to_dict(off, G, sites)
    spread = _angle_spread_deg(q, mean_quaternion(q, axis=0))
    quality = {"n": int(q.shape[0]), "wrist_site": wrist_site, "pose": "flat_hand",
               "quat_spread_deg": float(spread.max()), "quat_spread_deg_per_site": [float(x) for x in spread]}
    ok = quality["quat_spread_deg"] <= CALIB_MAX_SPREAD_DEG
    if gyro is not None:
        g = np.asarray(gyro, dtype=np.float64)
        rms = float(np.sqrt(np.mean(np.sum(g ** 2, axis=-1))))
        quality["gyro_rms"] = rms
        ok = ok and rms <= CALIB_MAX_GYRO_RMS
    quality["ok"] = bool(ok)
    return calib, quality


def find_calibration_phase(events, name: str | None = "imu_calibration") -> dict | None:
    """Last phase named ``name`` (a repeated block supersedes earlier attempts) — or, if absent,
    the last phase with ``value.kind == "calibration"``."""
    phases = phases_from_events(events)[::-1]
    for p in phases:
        if name is not None and p["name"] == name:
            return p
    for p in phases:
        if isinstance(p["value"], dict) and p["value"].get("kind") == "calibration":
            return p
    return None


def calibrate_session_imu(session_dir: str | Path, *, phase: str | None = "imu_calibration", trim_s: float = 0.5,
                          layout: Layout | str | None = None, stream: str = "imu", skeleton=None,
                          save: bool = True) -> dict | None:
    """Compute the IMU calibration of a recorded session from its calibration phase and store it
    in ``manifest.calibration`` (``save``). Returns the calibration dict (with the quality report
    under ``imu_calibration_quality``) or ``None`` when the session has no IMU / no such phase.

    ``trim_s`` drops the settling time at both ends of the phase (falls back to the whole phase if
    too little remains).
    """
    d = Path(session_dir)
    m = SessionManifest.load(d)
    if stream not in m.streams:
        return None
    ph = find_calibration_phase(load_events(d), phase)
    if ph is None:
        return None
    with np.load(m.stream_path(d, stream)) as z:
        t, quat = z["t"], z["quat"]
        gyro = z["gyro"] if "gyro" in z.files else None
        sites = [str(s) for s in z["sites"]] if "sites" in z.files else None
    sel = (t >= ph["t0"] + trim_s) & (t <= ph["t1"] - trim_s)
    if sel.sum() < 3:
        sel = (t >= ph["t0"]) & (t <= ph["t1"])
    if sel.sum() < 3:
        raise ValueError(f"{d}: fewer than 3 IMU samples inside the calibration phase")
    lay = layout if isinstance(layout, Layout) else load_layout(layout or m.layout)
    sites = sites or [s.name for s in lay.imu_sites]
    calib, quality = compute_imu_calibration(quat[sel], sites, lay, gyro=None if gyro is None else gyro[sel],
                                             skeleton=skeleton)
    quality.update({"phase": ph["name"], "t0": float(ph["t0"]), "t1": float(ph["t1"]), "trim_s": float(trim_s)})
    calib[QUALITY_KEY] = quality
    if save:
        m.calibration.update(calib)
        m.save(d)
    return calib
