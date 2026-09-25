from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("serial")  # pyserial 미설치(하드웨어 없는 CI/학습 환경)면 이 모듈 skip

from sats.inference.serial_reader import SensorSerialReader


def test_get_latest_window_with_seq_increments_on_new_window() -> None:
    reader = SensorSerialReader(window_size=3, baseline_seconds=0.1, protocol="csv")
    reader.baseline_ready = True
    reader._baseline = np.full((16,), 1000.0, dtype=np.float64)

    # 아직 window_size 미만
    reader._process_csv_values(np.full((16,), 1001.0, dtype=np.float64))
    reader._process_csv_values(np.full((16,), 1002.0, dtype=np.float64))
    win, seq = reader.get_latest_window_with_seq()
    assert win is None
    assert seq == 0

    # 첫 publish
    reader._process_csv_values(np.full((16,), 1003.0, dtype=np.float64))
    win, seq = reader.get_latest_window_with_seq()
    assert win is not None
    assert win.shape == (3, 16)
    assert seq == 1

    # 새 프레임 유입 시 seq 증가
    reader._process_csv_values(np.full((16,), 1004.0, dtype=np.float64))
    win2, seq2 = reader.get_latest_window_with_seq()
    assert win2 is not None
    assert win2.shape == (3, 16)
    assert seq2 == 2
    assert not np.array_equal(win, win2)
