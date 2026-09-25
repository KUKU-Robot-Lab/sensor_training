"""Backward-compatible import path of the former stub module.

``project_to_mano`` / ``align_layouts`` are implemented in :mod:`robot_skin.transfer.skeleton` and
:mod:`robot_skin.transfer.align`; import them from :mod:`robot_skin.transfer`.
"""
from __future__ import annotations

from .align import align_layouts
from .skeleton import project_to_mano

__all__ = ["align_layouts", "project_to_mano"]
