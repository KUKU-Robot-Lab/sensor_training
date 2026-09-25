"""Robot-hand logger CLI: tactile taxels + joint states (q, q̇, τ) → raw session dirs.

::

    python -m robot_skin.acquisition.robot_logger --protocol robot_sweep --dry-run --out robot_skin/data/raw/motion/R01/plan
    python -m robot_skin.acquisition.robot_logger --protocol robot_sweep --fake --time-scale 0.1 --subject R01
    python -m robot_skin.acquisition.robot_logger --fake --duration 10 --no-contact --subject R01

``robot_sweep`` (default when neither ``--protocol`` nor ``--duration`` is given) is the scripted
no-contact joint motion that trains the baseline predictor on the robot hand. Task episodes need a
protocol of ``kind: robot`` (or ``any``): ``d2_task`` is glove-only (its per-episode flat-hand IMU
calibration and fingertip sync taps are human-hand blocks) — write a robot task catalog with the
same task/phase schema once teleoperation exists. Real joint states come from ROS 2 ``/joint_states``
(``sources.RosJointStateSource`` — stub until a hand is connected → ``NotImplementedError``); the
taxel board uses the same serial reader as the glove.
"""
from __future__ import annotations

import sys

from common.layouts import load_layout

from ._cli import base_parser, run_logger
from .session import planned_streams
from .sources import RosJointStateSource


def _streams(args, plan, cams):
    return planned_streams("robot", (), plan.rates)


def _real_sources(args, plan, cams):
    RosJointStateSource(args.joint_topic)                      # raises NotImplementedError (stub)
    load_layout(args.layout)
    raise NotImplementedError  # pragma: no cover - unreachable until the joint source exists


def planned_streams_default() -> dict:
    return planned_streams("robot")


def main(argv: list[str] | None = None) -> int:
    p = base_parser("robot_logger", "Record robot-hand taxels + joint states following a protocol.")
    g = p.add_argument_group("robot devices")
    g.add_argument("--joint-topic", default="/joint_states")
    g.add_argument("--port", default="/dev/ttyACM0", help="taxel board serial port")
    args = p.parse_args(argv)
    args.cameras = "none"
    return run_logger(args, kind="robot", default_layout="robot_hand_template", default_protocol="robot_sweep",
                      default_cameras=(), real_sources=_real_sources, streams_fn=_streams)


if __name__ == "__main__":
    sys.exit(main())
