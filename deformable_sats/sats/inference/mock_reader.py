"""
sats/inference/mock_reader.py

센서 없이 시각화/추론 파이프라인을 테스트하기 위한 모의 시리얼 리더.

SensorSerialReader 와 동일한 인터페이스를 제공하므로
run_realtime.py 에서 --mock 플래그로 대체 사용 가능.

모의 데이터 패턴
---------------
  - 시간에 따라 접촉 위치가 원형으로 이동하는 가우시안 압력 분포
  - baseline 측정 시간을 생략하고 즉시 데이터 제공
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class MockSensorReader:
    """
    실제 센서 없이 동작하는 모의 리더.

    Parameters
    ----------
    window_size  : 슬라이딩 윈도우 크기 (학습 window_size 와 동일)
    fps          : 새 윈도우 생성 주기 (Hz)
    n_sensors    : 센서 수 (기본 16)
    """

    def __init__(
        self,
        window_size: int = 10,
        fps: float = 20.0,
        n_sensors: int = 16,
    ) -> None:
        self.window_size      = window_size
        self.fps              = fps
        self.n_sensors        = n_sensors
        self.baseline_ready   = True     # mock 이므로 즉시 준비
        self.baseline_seconds = 0.0
        self.bursts_received  = 0
        self.baseline_progress = 1.0

        self._lock    = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_seq: int = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._t0      = time.time()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[MockReader] 모의 데이터 생성 시작")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def baseline(self):
        """mock 은 s_norm 직접 생성 → flat baseline 개념 없음(theta/bending 비의미)."""
        return None

    def get_latest_window(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest

    def get_latest_window_with_seq(self) -> tuple[Optional[np.ndarray], int]:
        with self._lock:
            return self._latest, self._latest_seq

    def _run(self) -> None:
        interval = 1.0 / self.fps
        while self._running:
            t = time.time() - self._t0
            window = self._make_window(t)
            with self._lock:
                self._latest = window
                self._latest_seq += 1
            self.bursts_received += 1
            time.sleep(interval)

    def _make_window(self, t: float) -> np.ndarray:
        """
        접촉 위치가 시간에 따라 이동하는 가우시안 노이즈 포함 s_norm 윈도우.
        """
        # 접촉 위치: 반경 5mm 원 위를 0.2 Hz 로 이동
        cx = 5.0 * np.cos(2 * np.pi * 0.2 * t)
        cy = 5.0 * np.sin(2 * np.pi * 0.2 * t)

        # 16개 센서 위치 (4×4 grid, 간격 6.5mm, 중심 0)
        offsets = np.array([
            (-9.75, -9.75), (-3.25, -9.75), (3.25, -9.75), (9.75, -9.75),
            (-9.75, -3.25), (-3.25, -3.25), (3.25, -3.25), (9.75, -3.25),
            (-9.75,  3.25), (-3.25,  3.25), (3.25,  3.25), (9.75,  3.25),
            (-9.75,  9.75), (-3.25,  9.75), (3.25,  9.75), (9.75,  9.75),
        ])  # [16, 2]

        # 각 센서에서의 거리에 따른 가상 s_norm
        dx = offsets[:, 0] - cx   # [16]
        dy = offsets[:, 1] - cy
        dist2 = dx**2 + dy**2
        sigma = 5.0
        peak  = 80.0 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
        base_signal = peak * np.exp(-dist2 / (2 * sigma**2))  # [16]

        # window_size 개 프레임 (약간의 시간 변화 포함)
        window = np.stack([
            base_signal * (1.0 + 0.02 * np.random.randn(self.n_sensors))
            for _ in range(self.window_size)
        ], axis=0).astype(np.float32)   # [window_size, 16]

        return window
