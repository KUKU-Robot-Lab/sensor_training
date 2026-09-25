"""Glove logger CLI: tactile taxels + 7 glove IMUs + cameras (ego, third) → raw session dirs.

::

    # plan only (writes session.json / plan.json, prints the Korean operator script)
    python -m robot_skin.acquisition.glove_logger --protocol d1_motion --subject S01 --dry-run
    # synthetic end-to-end run (record → 3-tap sync → IMU calibration → QC), no hardware
    python -m robot_skin.acquisition.glove_logger --protocol d1_motion --subject S01 --fake --time-scale 0.05
    python -m robot_skin.acquisition.glove_logger --protocol d2_task --task pour --subject S01 --fake --episodes 2
    # real devices (IMU hub not implemented yet → NotImplementedError; --no-imu records pressure + cameras)
    python -m robot_skin.acquisition.glove_logger --protocol d1_motion --subject S01 --port /dev/ttyACM0

Streams: ``pressure`` (serial taxel board; ``--pressure-format csv`` built in, mk555 binary needs an
injected parser — ``deformable_sats/sats/preprocessing/bin_merge.py`` is the canonical parser and is
not copied here), ``imu`` (``sources.ImuSource`` stub), ``camera_<name>`` (OpenCV). Operator
procedure: ``docs/DATA_ACQUISITION.md``.
"""
from __future__ import annotations

import sys

from common.layouts import load_layout

from ._cli import _csv_list, base_parser, run_logger
from .session import IMU_SITES, planned_streams
from .sources import CameraSource, ImuSource, SerialPressureSource

DEFAULT_CAMERAS = ("ego", "third")


def _imu_sites(layout: str) -> tuple[str, ...]:
    try:
        return tuple(s.name for s in load_layout(layout).imu_sites) or IMU_SITES
    except (FileNotFoundError, ValueError):
        return IMU_SITES


def _streams(args, plan, cams):
    return planned_streams("glove", cams, plan.rates, imu_sites=_imu_sites(args.layout), imu=not args.no_imu)


def _real_sources(args, plan, cams):
    """Real devices: build once to fail fast (NotImplementedError / ImportError) before recording."""
    layout = load_layout(args.layout)
    n_ch = args.n_channels or int(layout.channels.max()) + 1
    if not args.no_imu:
        ImuSource()                                           # raises NotImplementedError (device stub)
    devices = {}
    for kv in _csv_list(args.camera_devices) or []:
        name, sep, dev = kv.partition("=")
        if not sep or not name or not dev:
            raise SystemExit(f"--camera-devices expects name=device pairs (e.g. ego=0,third=/dev/video2), got {kv!r}")
        if name not in cams:
            raise SystemExit(f"--camera-devices: camera {name!r} is not recorded (cameras: {list(cams)})")
        devices[name] = dev

    def factory(episode):
        srcs = [SerialPressureSource(args.port, n_ch, baudrate=args.baud, rate_hz=plan.rates.get("pressure", 200.0))]
        for i, c in enumerate(cams):
            dev = devices.get(c, str(i))
            srcs.append(CameraSource(c, int(dev) if dev.isdigit() else dev, rate_hz=plan.rates.get("camera", 30.0)))
        return srcs

    if args.pressure_format != "csv":
        raise NotImplementedError("mk555 binary frames: inject a FrameParser built on "
                                  "deformable_sats/sats/preprocessing/bin_merge.py into SerialPressureSource")
    return factory


def planned_streams_default() -> dict:
    """Streams of a default glove session (pressure, 7-site IMU, ego + third cameras)."""
    return planned_streams("glove", DEFAULT_CAMERAS)


def main(argv: list[str] | None = None) -> int:
    p = base_parser("glove_logger", "Record glove taxels + 7 IMUs + cameras following a protocol.")
    g = p.add_argument_group("glove devices")
    g.add_argument("--cameras", default=None, help="comma list of camera names (default: protocol's, else ego,third; 'none')")
    g.add_argument("--camera-devices", default=None, help="name=device list, e.g. ego=0,third=/dev/video2")
    g.add_argument("--port", default="/dev/ttyACM0", help="taxel board serial port")
    g.add_argument("--baud", type=int, default=115200)
    g.add_argument("--n-channels", type=int, default=None, help="raw channels per frame (default: from layout)")
    g.add_argument("--pressure-format", choices=("csv", "mk555"), default="csv")
    g.add_argument("--no-imu", action="store_true", help="record without the IMU stream")
    args = p.parse_args(argv)
    return run_logger(args, kind="glove", default_layout="glove_template", default_protocol=None,
                      default_cameras=DEFAULT_CAMERAS, real_sources=_real_sources, streams_fn=_streams)


if __name__ == "__main__":
    sys.exit(main())
