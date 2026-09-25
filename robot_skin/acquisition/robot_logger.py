"""Robot-hand logger CLI (stub): pressure taxels + joint states (q, q̇, τ).

``python -m robot_skin.acquisition.robot_logger --out robot_skin/data/robot/R001 --dry-run``

TODO
- pressure: same reader as the glove (mk555 ``.bin`` → ``bin_merge.py`` is canonical).
- joint_state: subscribe to the hand controller (ROS 2 ``/joint_states`` or vendor SDK);
  store ``t, name[D], q[D], qd[D], tau[D]`` in ``joint_state.npz``.
- scripted no-contact motion sweeps (``--no-contact``) are the baseline predictor's data.
"""
from __future__ import annotations

import sys

from ._cli import base_parser, write_plan
from .manifest import StreamInfo


def planned_streams() -> dict[str, StreamInfo]:
    return {
        "pressure": StreamInfo(file="pressure.bin", rate_hz=200.0, fields=["raw[C]"]),
        "joint_state": StreamInfo(file="joint_state.npz", rate_hz=500.0,
                                  fields=["q[D]", "qd[D]", "tau[D]"]),
    }


def main(argv: list[str] | None = None) -> int:
    p = base_parser("robot_logger", "Record robot-hand pressure + joint states (stub).")
    p.add_argument("--joint-topic", default="/joint_states")
    args = p.parse_args(argv)
    args.layout = args.layout or "robot_hand_template"
    if args.dry_run:
        write_plan(args, "robot", planned_streams())
        return 0
    raise NotImplementedError("robot_logger: device IO not implemented yet (use --dry-run)")


if __name__ == "__main__":
    sys.exit(main())
