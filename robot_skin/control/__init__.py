"""Robot control: deploy a trained VTLA policy bundle on a robot hand with the tactile skin in the loop.

::

    RobotHandInterface ─ read_state / read_pressure ─▶ OnlineTactileProcessor (≡ offline stage 1)
    CameraInterface ─ read ─┐                              │ residual z, levels, features, contact
                            ▼                              ▼
    PolicyRunner ── make_observation / collate_vtla ─▶ VTLAPolicy.predict ─▶ TemporalEnsembler
         │  hand_mano: FingertipRetargeter · robot_joint: direct        ─▶ SafetyFilter ─▶ send
         └─ DeploymentLogger: RAW session (re-ingestible by datasets.build)

Modules: :mod:`.interfaces` (hardware protocols + :class:`FakeRobotHand` / :class:`FakeCamera`),
:mod:`.online` (tactile processor), :mod:`.bundle` (policy bundle loading), :mod:`.runner`
(control loop + logger), :mod:`.safety` (limits, rate limits, tactile stop, watchdog),
:mod:`.latency` (p50/p95, TorchScript export). Stage runner: ``robot_skin.stages.deploy``;
operator guide: ``docs/DEPLOYMENT.md``. Exports resolve lazily (PEP 562) so importing the package
does not import torch.
"""
from __future__ import annotations

import importlib
from typing import Any

_LAZY = {
    # interfaces
    "RobotHandInterface": "interfaces", "CameraInterface": "interfaces", "check_robot": "interfaces",
    "check_camera": "interfaces", "VirtualObject": "interfaces", "FakeRobotHand": "interfaces",
    "FakeCamera": "interfaces", "finger_joint_groups": "interfaces", "taxel_coupling": "interfaces",
    "SYNTHETIC_HAND_HUMAN_TO_ROBOT": "interfaces", "urdf_tip_fk": "interfaces", "layout_tip_offsets": "interfaces",
    "load_urdf_model": "interfaces",
    # online
    "CausalJointVelocity": "online", "TactileFrame": "online", "OnlineTactileProcessor": "online",
    "glove_pose_fn": "online", "robot_pose_fn": "online", "static_pose_fn": "online", "make_pose_fn": "online",
    "load_calibrator": "online", "startup_calibrator": "online", "replay_episode": "online",
    # bundle
    "PolicyBundle": "bundle", "load_policy_bundle": "bundle",
    # runner
    "PolicyRunner": "runner", "DeploymentLogger": "runner", "joint_permutation": "runner",
    "initial_hand_state": "runner", "DEPLOY_LOG_NAME": "runner",
    # safety
    "SafetyFilter": "safety", "SafetyEvent": "safety", "TACTILE_STOP_MODES": "safety",
    "taxel_joint_mask": "safety", "closing_signs": "safety",
    # latency
    "LatencyMeter": "latency", "percentile_summary": "latency", "example_batch": "latency",
    "benchmark_policy": "latency", "export_torchscript": "latency",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'robot_skin.control' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
