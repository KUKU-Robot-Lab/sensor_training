"""Processed-episode format, preprocessing and training datasets.

Only the light Episode contract (:mod:`.episode`, numpy) is imported eagerly. Preprocessing
(:mod:`.build`), splits (:mod:`.splits`), normalisation statistics (:mod:`.stats`) and the D1 window
datasets (:mod:`.motion`, torch) are resolved lazily (PEP 562) on first attribute access, so
``import robot_skin.datasets`` stays cheap and ``python -m robot_skin.datasets.build`` does not import
its own module twice. The synthetic generator (:mod:`.synthetic`, torch + pose code) is never
imported eagerly: use ``from robot_skin.datasets import synthetic`` (or ``robot_skin.datasets.synthetic``).
"""
from __future__ import annotations

import importlib
from typing import Any

from .episode import Episode, EpisodeMeta, list_episodes

#: lazily re-exported names → submodule
_EXPORTS: dict[str, str] = {
    # build (preprocessing)
    "preprocess_session": "build", "build_all": "build", "find_sessions": "build",
    "load_preprocess_config": "build", "resolve_layout": "build", "load_episode_layout": "build",
    "joint_velocity": "build", "PREPROCESS_VERSION": "build",
    # splits
    "make_splits": "splits", "check_splits": "splits", "save_splits": "splits", "load_splits": "splits",
    "episode_group": "splits",
    # stats
    "compute_stats": "stats", "apply_stats": "stats", "invert_stats": "stats", "save_stats": "stats",
    "load_stats": "stats", "q_valid_mask": "stats", "qd_valid_mask": "stats", "IMU_FEATURES": "stats",
    # motion (torch datasets)
    "BaselineWindowDataset": "motion", "ContactWindowDataset": "motion", "ImuPoseWindowDataset": "motion",
}
#: submodules reachable as attributes without an explicit import (never imported eagerly)
_SUBMODULES = ("episode", "build", "splits", "stats", "motion", "synthetic")

__all__ = ["Episode", "EpisodeMeta", "list_episodes", *_EXPORTS]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module 'robot_skin.datasets' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value  # cache
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))
