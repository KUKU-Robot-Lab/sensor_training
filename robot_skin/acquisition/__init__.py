"""Data acquisition: protocols (D1 motion / D2 task), stream sources, recorder, 3-tap sync, IMU
calibration, QC and the glove / robot logger CLIs. See ``acquisition/README.md`` and
``docs/DATA_ACQUISITION.md``.

Exports are lazy (PEP 562) so ``import robot_skin.acquisition`` stays light; only the synthetic
scene (``fake``) pulls in torch / the MANO skeleton.
"""
from __future__ import annotations

import importlib
from typing import Any

from .manifest import MANIFEST_NAME, SCHEMA_VERSION, SessionManifest, StreamInfo

_EXPORTS = {
    "instructions": ("check_template", "normalize_instruction", "render_instruction", "sample_instruction",
                     "template_slots"),
    "protocol": ("EpisodePlan", "PROTOCOL_DIR", "Protocol", "ProtocolError", "SessionPlan", "Step", "SyncSpec",
                 "TaskSpec", "TimedStep", "Timing", "format_script", "list_protocols", "load_protocol",
                 "make_session_id", "plan_session", "protocol_from_dict", "scale_steps", "scale_timing"),
    "sources": ("CameraSource", "CsvLineParser", "FakeCameraSource", "FakeHandPoseSource", "FakeImuSource",
                "FakeJointSource", "FakeObjectPoseSource", "FakePressureSource", "ImuSource", "MonotonicClock",
                "PlaybackSource", "RosJointStateSource", "STREAM_KINDS", "SerialPressureSource", "SimClock",
                "StreamSource", "fake_sources", "validate_source"),
    "recorder": ("EVENTS_NAME", "EventLog", "Recorder", "load_events", "phases_from_events", "segments_from_events",
                 "session_segments"),
    "sync": ("ClockModel", "OffsetEstimate", "apply_clock_models", "apply_offset", "change_envelope", "detect_taps",
             "estimate_offset", "fit_clock_drift", "sync_session", "sync_windows"),
    "calibration": ("calibrate_session_imu", "compute_imu_calibration"),
    "qc": ("DEFAULT_THRESHOLDS", "format_report", "rest_spans", "session_qc", "stream_timing"),
    "session": ("AutoOperator", "ConsoleOperator", "adhoc_plan", "copy_imu_calibration", "fake_source_factory",
                "planned_streams", "postprocess_session", "record_episode", "refresh_segments", "run_plan",
                "write_dry_run"),
    "fake": ("FakeConfig", "FakeScene"),
}
_WHERE = {name: mod for mod, names in _EXPORTS.items() for name in names}

__all__ = ["MANIFEST_NAME", "SCHEMA_VERSION", "SessionManifest", "StreamInfo", *sorted(_WHERE)]


def __getattr__(name: str) -> Any:
    mod = _WHERE.get(name)
    if mod is None:
        # submodules (e.g. ``from robot_skin.acquisition import glove_logger``) resolve normally
        try:
            return importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f"{__name__}.{mod}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
