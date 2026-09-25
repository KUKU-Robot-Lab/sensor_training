from __future__ import annotations

import struct

import numpy as np
import pytest

pytest.importorskip("serial")  # pyserial 미설치(하드웨어 없는 CI/학습 환경)면 이 모듈 skip

from sats.inference.serial_reader import FIFO_FRAMES, NUM_SENSORS, PAYLOAD_BYTES, SensorSerialReader


def _build_sensor_major_payload(base: int = 1000) -> bytes:
    values: list[int] = []
    for sensor_i in range(NUM_SENSORS):
        for frame_j in range(FIFO_FRAMES):
            values.append(base + sensor_i * 100 + frame_j)
    return struct.pack(f"<{NUM_SENSORS * FIFO_FRAMES}L", *values)


def test_binary_payload_contract_sensor_major_frame_minor() -> None:
    reader = SensorSerialReader(window_size=FIFO_FRAMES, baseline_seconds=0.1, protocol="binary")
    reader.baseline_ready = True
    reader._baseline = np.full((NUM_SENSORS,), 1000.0, dtype=np.float64)

    payload = _build_sensor_major_payload(base=1000)
    assert len(payload) == PAYLOAD_BYTES

    reader._process_frame(payload)

    win, seq = reader.get_latest_window_with_seq()
    assert seq == 1
    assert win is not None
    assert win.shape == (FIFO_FRAMES, NUM_SENSORS)

    # reader가 기대하는 매핑: idx = sensor_i * FIFO_FRAMES + frame_j
    # frames[frame_j, sensor_i] == raw(sensor_i, frame_j)
    expected_raw = np.empty((FIFO_FRAMES, NUM_SENSORS), dtype=np.float64)
    for frame_j in range(FIFO_FRAMES):
        for sensor_i in range(NUM_SENSORS):
            expected_raw[frame_j, sensor_i] = 1000 + sensor_i * 100 + frame_j

    expected_s_norm = ((expected_raw - 1000.0) / 1000.0 * 100.0).astype(np.float32)
    assert np.allclose(win, expected_s_norm, atol=1e-6)
