import json
import struct

import numpy as np
import pytest

from sats.preprocessing import bin_merge as bm
from sats.preprocessing.merged_bin import open_merged_bin


def _write_header(f, magic: str, header: dict) -> None:
    f.write((magic + "\n").encode("ascii"))
    f.write(json.dumps(header, separators=(",", ":")).encode("ascii"))
    f.write(b"\nEND_HEADER\n")


def _due_payload(base: int) -> bytes:
    values = []
    for sensor_i in range(bm.NUM_SENSORS):
        for frame_i in range(bm.FIFO_FRAMES):
            values.append(base + sensor_i * 10 + frame_i)
    return struct.pack("<" + "I" * len(values), *values)


def _write_due(path, times_ns):
    with open(path, "wb") as f:
        _write_header(
            f,
            bm.DUE_MAGIC,
            {
                "record_bytes": bm.DUE_RECORD_STRUCT.size + bm.DUE_PAYLOAD_SIZE,
                "payload_layout": "sensor-major uint32 little-endian: sensor[16][frame10]",
            },
        )
        for i, ns in enumerate(times_ns):
            f.write(bm.DUE_RECORD_STRUCT.pack(ns))
            f.write(_due_payload(1000 + i * 100))


def _write_ethermotion(path, rows):
    with open(path, "wb") as f:
        _write_header(
            f,
            bm.ETHERMOTION_MAGIC,
            {"record_struct": "<Qdddd", "record_bytes": 40},
        )
        for row in rows:
            f.write(struct.pack("<Qdddd", *row))


def _write_loadcell(path, rows):
    with open(path, "wb") as f:
        _write_header(
            f,
            bm.LOADCELL_MAGIC,
            {"record_struct": "<QI + payload", "record_bytes": "variable"},
        )
        for ns, kg in rows:
            payload = f"{kg:.2f}\n".encode("ascii")
            f.write(bm.LOADCELL_RECORD_STRUCT.pack(ns, len(payload)))
            f.write(payload)


def test_load_due_bin_expands_fifo_frames(tmp_path):
    path = tmp_path / "due_raw_burst_20260601_000000.bin"
    _write_due(path, [0, 10_000_000])

    due = bm.load_due_bin(path)

    assert due.bursts == 2
    assert due.expanded_rows == 20
    assert due.time_s.shape == (20,)
    assert due.sensors.shape == (20, 16)
    assert due.time_s[0] == pytest.approx(0.0)
    assert due.time_s[1] == pytest.approx(0.001)
    assert due.sensors[0, 0] == pytest.approx(1000.0)
    assert due.sensors[1, 0] == pytest.approx(1001.0)
    assert due.sensors[0, 1] == pytest.approx(1010.0)


def test_process_trial_dir_reads_raw_bins_and_writes_merged_bin(tmp_path):
    raw_root = tmp_path / "raw_data"
    trial_dir = raw_root / "ecomesh" / "d5" / "z_2.5mm" / "test1"
    trial_dir.mkdir(parents=True)
    times_ns = [0, 10_000_000, 20_000_000, 30_000_000]

    _write_due(trial_dir / "due_raw_burst_20260601_000000.bin", times_ns)
    _write_ethermotion(
        trial_dir / "ethermotion_encoder_20260601_000000.bin",
        [
            (times_ns[0], 0.0, 0.0, 0.0, 0.0),
            (times_ns[1], 0.0, 0.0, 0.0, 0.0),
            (times_ns[2], -100000.0, -100000.0, 130000.0, 0.0),
            (times_ns[3], -100000.0, -100000.0, 155000.0, 5000.0),
        ],
    )
    _write_loadcell(
        trial_dir / "loadcell_raw_20260601_000000.bin",
        [(times_ns[0], 0.10), (times_ns[1], 0.10), (times_ns[2], 0.10), (times_ns[3], 0.30)],
    )

    summary = bm.process_trial_dir(
        trial_dir,
        raw_root,
        target_hz=100.0,
        max_dt_ms=1.0,
        stable_xy_only=True,
        force_round_dp=None,
    )

    merged_path = trial_dir / "ecomesh_d5_z2.5_test1_merged.bin"
    baseline_path = trial_dir / "ecomesh_d5_z2.5_test1_baseline.json"
    assert merged_path.exists()
    assert baseline_path.exists()
    assert summary["merged_rows"] == 4
    assert summary["due_source_rows"] == 40

    header, rows = open_merged_bin(merged_path)
    assert header["metadata"]["summary"]["trial_id"] == "ecomesh_d5_z2.5_test1"
    assert rows["x_mm"][2] == pytest.approx(-10.0)
    assert rows["z_stage_mm"][2] == pytest.approx(13.0, abs=1e-6)
    assert rows["z_stage_mm"][3] == pytest.approx(15.5, abs=1e-6)
    assert rows["z_depth_mm"][2] == pytest.approx(0.0, abs=1e-6)
    assert rows["z_depth_mm"][3] == pytest.approx(2.5, abs=1e-6)
    assert rows["u_mm"][3] == pytest.approx(0.5, abs=1e-6)
    assert rows["Fz"][3] == pytest.approx((0.30 - 0.10) * bm.GRAVITY, rel=1e-5)

    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    assert baseline["kg_baseline"] == pytest.approx(0.10)
    # DUE payloads are expanded to FIFO frames. The inclusive 0.00..0.01 s
    # baseline window contains burst0 frames 0..9 and burst1 frame 0.
    assert baseline["Skin1_mean"] == pytest.approx((sum(range(1000, 1010)) + 1100) / 11.0)


def test_ethermotion_reader_supports_legacy_40_byte_records(tmp_path):
    path = tmp_path / "ethermotion_encoder_20260601_000000.bin"
    _write_ethermotion(path, [(0, 1.0, 2.0, 3.0, 4.0)])

    data = bm.load_ethermotion_bin(path)

    assert data.rows == 1
    assert data.x_mm[0] == pytest.approx(0.0001)
    assert data.u_mm[0] == pytest.approx(0.0004)


def test_find_bin_set_uses_manifest_when_multiple_sets_exist(tmp_path):
    trial_dir = tmp_path / "test1"
    trial_dir.mkdir()
    for suffix in ["a", "b"]:
        (trial_dir / f"due_raw_burst_{suffix}.bin").write_bytes(b"")
        (trial_dir / f"ethermotion_encoder_{suffix}.bin").write_bytes(b"")
        (trial_dir / f"loadcell_raw_{suffix}.bin").write_bytes(b"")
    (trial_dir / "manifest.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "due": r"C:\x\due_raw_burst_b.bin",
                    "ethermotion": r"C:\x\ethermotion_encoder_b.bin",
                    "loadcell": r"C:\x\loadcell_raw_b.bin",
                }
            }
        )
    )

    paths = bm.find_bin_set(trial_dir)

    assert paths["due"].name == "due_raw_burst_b.bin"
