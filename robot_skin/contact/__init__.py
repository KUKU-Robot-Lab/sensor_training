"""Residual → contact interpretation: calibration, levels, learned detector, hysteresis, pseudo labels.

Light modules (numpy only) are imported eagerly; the torch-based detector is resolved lazily
(PEP 562) so ``robot_skin.contact.ordinal`` / ``saturation_fsm`` users do not pay for torch.
"""
from __future__ import annotations

import importlib
from typing import Any

from .calibration import ResidualCalibrator, residual_levels, robust_sigma, saturation_gate
from .hysteresis import HysteresisFilter
from .ordinal import ContactLevel, OrdinalQuantizer
from .pseudo_label import (D_CONTACT_LABEL_PSEUDO, frame_expectation, phase_expectation, pseudo_label_episode,
                           pseudo_label_metrics, taxel_world_positions)
from .residual import contact_mask, residual
from .saturation_fsm import SatState, SaturationFSM
from .self_touch import finger_of, point_segment_distance, self_touch_labels

_LAZY = {"ContactDetector": "detector", "focal_loss": "detector", "contact_loss": "detector",
         "predict_contact_prob": "detector", "CausalDetectorStream": "detector", "save_detector": "detector",
         "load_detector": "detector"}

__all__ = ["ContactLevel", "OrdinalQuantizer", "SatState", "SaturationFSM", "contact_mask",
           "finger_of", "point_segment_distance", "residual", "self_touch_labels",
           "ResidualCalibrator", "residual_levels", "robust_sigma", "saturation_gate", "HysteresisFilter",
           "D_CONTACT_LABEL_PSEUDO", "frame_expectation", "phase_expectation", "pseudo_label_episode",
           "pseudo_label_metrics", "taxel_world_positions", *_LAZY]


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'robot_skin.contact' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
