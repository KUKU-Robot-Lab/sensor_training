"""Glove logger CLI (stub): pressure taxels + 7 IMUs + camera → one session dir.

``python -m robot_skin.acquisition.glove_logger --out robot_skin/data/glove/S001 --dry-run``
writes the planned ``session.json``. Actual device IO is not implemented yet:

TODO
- pressure: serial reader for the glove board. If it speaks the mk555 protocol, reuse the
  on-disk ``.bin`` format and parse with ``deformable_sats/sats/preprocessing/bin_merge.py``
  (canonical parser — do not copy it here).
- imu: 7 sites (``common/layouts/glove_template.yaml`` → ``imu_sites``), quaternion + gyro + acc.
- camera: frame timestamps (host clock) for MANO / VIFNet-S supervision.
- all streams stamped on one host clock; alignment later via ``common.timeline.align_streams``.
"""
from __future__ import annotations

import sys

from ._cli import base_parser, write_plan
from .manifest import StreamInfo

IMU_SITES = ("wrist", "palm", "thumb", "index", "middle", "ring", "pinky")


def planned_streams() -> dict[str, StreamInfo]:
    return {
        "pressure": StreamInfo(file="pressure.bin", rate_hz=200.0, fields=["raw[C]"]),
        "imu": StreamInfo(file="imu.npz", rate_hz=100.0,
                          fields=[f"{s}.{k}" for s in IMU_SITES for k in ("quat", "gyro", "acc")]),
        "camera": StreamInfo(file="camera/", rate_hz=30.0, fields=["frame"], method="zoh"),
    }


def main(argv: list[str] | None = None) -> int:
    p = base_parser("glove_logger", "Record glove pressure + 7 IMU + camera (stub).")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--camera", default="0")
    args = p.parse_args(argv)
    args.layout = args.layout or "glove_template"
    if args.dry_run:
        write_plan(args, "glove", planned_streams())
        return 0
    raise NotImplementedError("glove_logger: device IO not implemented yet (use --dry-run)")


if __name__ == "__main__":
    sys.exit(main())
