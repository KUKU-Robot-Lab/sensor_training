"""
sats/inference/serial_reader.py

실시간 BMP384 센서 시리얼 수신 + 전처리 모듈.
두 가지 프로토콜을 지원한다.
1) 바이너리 burst (0xAA + 640 bytes + 0x55)
2) CSV 라인 (vensor.ino: "v1,v2,...,v16\\n")

프로토콜
--------
  0xAA  |  pressure_burst_data[16][10]  |  0x55
          └─ 16×10×4 = 640 bytes, uint32 LE

  - 200 Hz, 10프레임 묶음 → ~20 burst/sec
"""

from __future__ import annotations

import collections
import struct
import threading
import time
import traceback
from typing import Optional

import numpy as np
import serial


# ─────────────────────────────────────────────────────────────────────────────
NUM_SENSORS   = 16
FIFO_FRAMES   = 10
PAYLOAD_BYTES = NUM_SENSORS * FIFO_FRAMES * 4   # 640
FRAME_HEADER  = 0xAA
FRAME_FOOTER  = 0x55


class SensorSerialReader:
    """
    백그라운드 스레드에서 시리얼 수신, 전처리 후 최신 윈도우를 제공.

    Parameters
    ----------
    port             : 시리얼 포트  (/dev/ttyACM1 등)
    baudrate         : 통신 속도 (기본 250000)
    window_size      : 슬라이딩 윈도우 길이
    baseline_seconds : baseline 자동 측정 시간 (초)
    startup_delay    : 포트 오픈 후 대기 시간(초) — Arduino Due 리셋 대기
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM1",
        baudrate: int = 250_000,
        window_size: int = 10,
        baseline_seconds: float = 5.0,
        startup_delay: float = 3.0,
        protocol: str = "auto",
    ) -> None:
        if protocol not in {"auto", "binary", "csv"}:
            raise ValueError("protocol must be one of: auto, binary, csv")

        self.port             = port
        self.baudrate         = baudrate
        self.window_size      = window_size
        self.baseline_seconds = baseline_seconds
        self.startup_delay    = startup_delay
        self.protocol         = protocol

        self._window: collections.deque[np.ndarray] = collections.deque(maxlen=window_size)
        # ★스트림 연속성 감시: 버스트가 유실되면(프레이밍 오류·버퍼 밀림) 창이 시간을
        #   건너뛴 채 이어 붙는다. LSTM 은 연속 5ms 간격만 학습했으므로 구멍 난 창은
        #   OOD → 순간 오추론. 간극이 감지되면 창을 비워 불연속 창을 모델에 주지 않는다.
        self._last_burst_t: float | None = None
        self.gap_count = 0

        self._baseline: Optional[np.ndarray] = None
        self._baseline_buf: list[np.ndarray] = []
        self._baseline_n_bursts = max(1, int(baseline_seconds * 20))

        self._lock   = threading.Lock()
        self._latest_window: Optional[np.ndarray] = None
        self._latest_window_seq: int = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 진단용 카운터
        self.baseline_ready   = False
        self.dead_channels    = np.empty(0, dtype=int)   # 자동 감지 ∪ 수동 지정
        self._auto_dead       = np.empty(0, dtype=int)   # baseline≈0 자동 감지분
        self._manual_dead     = np.empty(0, dtype=int)   # UI 에서 지정한 파손 taxel
        self.bursts_received  = 0
        self.raw_bytes_in     = 0    # read_until 이 반환한 누적 바이트 수
        self.footer_errors    = 0    # footer 불일치 횟수
        self.parse_errors     = 0    # CSV 파싱 실패 횟수
        self.error_message: Optional[str] = None

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="SerialReader")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def get_latest_window(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_window

    def get_latest_window_with_seq(self) -> tuple[Optional[np.ndarray], int]:
        with self._lock:
            return self._latest_window, self._latest_window_seq

    @property
    def baseline_progress(self) -> float:
        return min(len(self._baseline_buf) / self._baseline_n_bursts, 1.0)

    @property
    def baseline(self):
        """캡처된 flat 무접촉 baseline[16] (밴딩 raw 복원용). 없으면 None."""
        return self._baseline

    # ── 내부 구현 ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._run_inner()
        except Exception:
            self.error_message = traceback.format_exc()
            print(f"\n[SerialReader] 스레드 비정상 종료:\n{self.error_message}")

    def _run_inner(self) -> None:
        # ── 포트 오픈 ─────────────────────────────────────────────────────────
        try:
            ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=2.0)
        except serial.SerialException as e:
            self.error_message = str(e)
            print(f"\n[SerialReader] 포트 오픈 실패: {e}")
            return

        print(f"\n[SerialReader] 연결 완료: {self.port} @ {self.baudrate} baud")

        # Arduino Due 는 포트 오픈 시 리셋됨 → setup() 완료까지 대기
        if self.startup_delay > 0:
            print(f"[SerialReader] Arduino 리셋 대기 {self.startup_delay:.0f}초...")
            time.sleep(self.startup_delay)

        ser.reset_input_buffer()
        print(f"[SerialReader] 버퍼 초기화 완료. 데이터 수신 시작.")
        print(f"[SerialReader] baseline 측정 ({self.baseline_seconds:.0f}초) — 무접촉 유지!")

        protocol = self._detect_protocol(ser) if self.protocol == "auto" else self.protocol
        print(f"[SerialReader] 프로토콜: {protocol}")

        # ── 수신 루프 ─────────────────────────────────────────────────────────
        if protocol == "csv":
            self._run_csv_loop(ser)
        else:
            self._run_binary_loop(ser)

        ser.close()
        print("[SerialReader] 종료")

    def _detect_protocol(self, ser: serial.Serial) -> str:
        """
        자동 감지:
          - 쉼표(,) 기반 16채널 CSV 라인이 3번 이상 연속으로 파싱되면 csv
          - 아니면 binary 로 처리
        """
        csv_hits = 0
        trials = 0

        print("[SerialReader] 프로토콜 자동 감지 중...")
        while self._running and trials < 12:
            line = ser.readline()
            self.raw_bytes_in += len(line)
            if not line:
                trials += 1
                continue

            values = self._parse_csv_line(line)
            trials += 1
            if values is None:
                csv_hits = 0
                continue

            csv_hits += 1
            self._process_csv_values(values)
            if csv_hits >= 3:
                return "csv"

        # auto에서 csv 확정 못 하면 기존 바이너리 방식으로 폴백
        ser.reset_input_buffer()
        return "binary"

    def _run_binary_loop(self, ser: serial.Serial) -> None:
        # ── 수신 루프 (ven_200hz_reader.py 동일 방식) ─────────────────────────
        while self._running:
            try:
                # ── ven_200hz_reader.py 와 동일한 방식 ──────────────────────
                # 헤더(0xAA) 까지 읽기 (반환값 사용 안 함 — ven_reader 동일)
                pre = ser.read_until(expected=bytes([FRAME_HEADER]))
                self.raw_bytes_in += len(pre)

                # payload(640) + footer(1) 읽기
                packet = ser.read(PAYLOAD_BYTES + 1)
                self.raw_bytes_in += len(packet)

                if len(packet) < PAYLOAD_BYTES + 1:
                    # 짧은 read (timeout)
                    continue

                if packet[-1] != FRAME_FOOTER:
                    self.footer_errors += 1
                    # 진단: 처음 20번 footer 오류는 실제 값을 출력
                    if self.footer_errors <= 20:
                        print(f"\n[SerialReader] footer 오류 #{self.footer_errors}: "
                              f"got 0x{packet[-1]:02X} (expected 0x55)")
                    continue

                # 유효 프레임 처리
                self._process_frame(bytes(packet[:-1]))

            except serial.SerialException as e:
                print(f"\n[SerialReader] 시리얼 오류: {e}")
                break

    def _run_csv_loop(self, ser: serial.Serial) -> None:
        while self._running:
            try:
                line = ser.readline()
                self.raw_bytes_in += len(line)
                if not line:
                    continue

                values = self._parse_csv_line(line)
                if values is None:
                    self.parse_errors += 1
                    continue

                self._process_csv_values(values)
            except serial.SerialException as e:
                print(f"\n[SerialReader] 시리얼 오류: {e}")
                break

    def _process_frame(self, payload: bytes) -> None:
        """
        640 bytes → [10, 16] float64.

        ven_200hz_reader.py 동일:
          all_values = struct.unpack('<160L', payload)
          idx = sensor_i * FIFO_FRAMES + frame_j
        """
        try:
            all_values = struct.unpack(f'<{NUM_SENSORS * FIFO_FRAMES}L', payload)
        except struct.error:
            return

        frames = np.empty((FIFO_FRAMES, NUM_SENSORS), dtype=np.float64)
        for j in range(FIFO_FRAMES):
            for i in range(NUM_SENSORS):
                frames[j, i] = all_values[i * FIFO_FRAMES + j]

        self.bursts_received += 1

        if not self.baseline_ready:
            self._collect_baseline(frames)
            return

        now = time.monotonic()
        if self._last_burst_t is not None and now - self._last_burst_t > 0.08:
            self._window.clear()                     # ≥1 버스트 유실 — 불연속 창 방지
            self.gap_count += 1
            if self.gap_count <= 5 or self.gap_count % 50 == 0:
                print(f"\n[SerialReader] 스트림 간극 #{self.gap_count} "
                      f"({(now - self._last_burst_t) * 1000:.0f}ms) — 창 리셋")
        self._last_burst_t = now
        # s_norm = (raw - baseline) / baseline * 100  (파손 채널은 0 고정)
        s_norm = ((frames - self._baseline) / self._baseline * 100.0).astype(np.float32)
        if len(self.dead_channels):
            s_norm[:, self.dead_channels] = 0.0

        for t in range(FIFO_FRAMES):
            self._window.append(s_norm[t])

            if len(self._window) == self.window_size:
                self._publish_latest_window()

    def set_dead_channels(self, idx) -> None:
        """UI 지정 파손 채널(0-based) — 자동 감지분과 합집합으로 유지."""
        self._manual_dead = np.asarray(sorted(set(int(i) for i in idx)), dtype=int)
        self.dead_channels = np.union1d(self._auto_dead, self._manual_dead)

    def _parse_csv_line(self, line: bytes) -> Optional[np.ndarray]:
        text = line.decode("ascii", errors="ignore").strip()
        if not text or "," not in text:
            return None

        values = np.fromstring(text, sep=",", dtype=np.float64)
        if values.size != NUM_SENSORS:
            return None
        return values

    def _process_csv_values(self, values: np.ndarray) -> None:
        self.bursts_received += 1

        frame = values.astype(np.float64, copy=False)
        if not self.baseline_ready:
            self._collect_baseline(frame[np.newaxis, :])
            return

        # s_norm = (raw - baseline) / baseline * 100
        s_norm = ((frame - self._baseline) / self._baseline * 100.0).astype(np.float32)
        self._window.append(s_norm)

        if len(self._window) == self.window_size:
            self._publish_latest_window()

    def _publish_latest_window(self) -> None:
        win = np.stack(list(self._window), axis=0).copy()
        with self._lock:
            self._latest_window = win
            self._latest_window_seq += 1

    def _collect_baseline(self, frames: np.ndarray) -> None:
        self._baseline_buf.append(frames.mean(axis=0))
        n = len(self._baseline_buf)

        if n % 10 == 0:
            pct = n / self._baseline_n_bursts * 100
            print(f"\r[SerialReader] baseline {pct:.0f}%  ({n}/{self._baseline_n_bursts} bursts)", end="", flush=True)

        if n >= self._baseline_n_bursts:
            arr = np.stack(self._baseline_buf, axis=0)
            self._baseline = arr.mean(axis=0)
            # ★파손 taxel 은 raw=0 → baseline=0. 그대로 나누면 NaN 이 창 전체에 퍼져
            #   맵·복원기·미니맵이 모두 무너진다(0/0=NaN, x/0=inf). 해당 채널은 dead 로
            #   기록하고 pct 를 0(=변화 없음)으로 고정한다.
            med = float(np.median(self._baseline[self._baseline > 0])) \
                if (self._baseline > 0).any() else 1.0
            self._auto_dead = np.flatnonzero(self._baseline < 0.5 * med)
            if len(self._auto_dead):
                names = ",".join(f"S{i + 1:02d}" for i in self._auto_dead)
                print(f"\n[SerialReader] ★파손 채널 감지: {names} — pct 0 고정")
                self._baseline = np.where(self._baseline < 0.5 * med, 1.0, self._baseline)
            self.dead_channels = np.union1d(self._auto_dead, self._manual_dead)
            self.baseline_ready = True
            print(f"[SerialReader] baseline 완료!")
            print(f"  raw ADC: min={self._baseline.min():.0f}  max={self._baseline.max():.0f}")
