"""Glove ⇄ robot-hand transfer: taxels on a shared capsule skeleton, layout correspondence, value
mapping, and the robot → MANO hand-state estimate (``transfer/README.md``)."""
from .align import REDUCE_MODES, LayoutAlignment, align_layouts, layout_rest_poses, map_taxel_values, taxel_groups
from .skeleton import (HAND_GROUPS, CapsuleSkeleton, SkeletonProjection, finger_group, project_to_mano,
                       project_to_skeleton)

__all__ = ["HAND_GROUPS", "CapsuleSkeleton", "SkeletonProjection", "finger_group", "project_to_skeleton",
           "project_to_mano", "REDUCE_MODES", "LayoutAlignment", "taxel_groups", "layout_rest_poses",
           "align_layouts", "map_taxel_values", "RobotToManoEstimator", "RobotToManoTaxelFrame", "mano_tip_fk"]


def __getattr__(name: str):
    # the reverse retargeter pulls in torch + action.retarget: resolve lazily (PEP 562)
    if name in ("RobotToManoEstimator", "RobotToManoTaxelFrame", "mano_tip_fk"):
        from . import reverse

        return getattr(reverse, name)
    raise AttributeError(f"module 'robot_skin.transfer' has no attribute {name!r}")
