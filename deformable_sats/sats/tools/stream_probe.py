"""DUE 스트림 실측 프로브 — 데모 PC 에서 펌웨어 타이밍을 그 자리에서 확인한다.

측정 항목:
  1. 버스트 도착 간격(중앙/p95/max) — 50ms 기대. 블로킹 전송이 밀리면 여기서 보인다.
  2. 버스트 내 인접 프레임 동일값 비율 — 펌웨어가 변환 완료를 기다리지 않아 생기는
     묵은 샘플 재독(실측 v6 bin 기준 ~44%). 실효 대역폭 = 200Hz × (1 − 중복률).
  3. 프레이밍 오류(footer 불일치) 횟수.

사용(센서 연결 상태에서):
    python -m sats.tools.stream_probe /dev/ttyACM0 --sec 10
"""
from __future__ import annotations

import argparse
import struct
import time

import numpy as np

PAYLOAD = 16 * 10 * 4


def main() -> int:
    ap = argparse.ArgumentParser(description="DUE 스트림 타이밍 실측")
    ap.add_argument("port")
    ap.add_argument("--baud", type=int, default=250000)
    ap.add_argument("--sec", type=float, default=10.0)
    a = ap.parse_args()
    import serial
    ser = serial.Serial(a.port, a.baud, timeout=0.2)
    ser.reset_input_buffer()
    t0 = time.monotonic()
    arrivals: list[float] = []
    bursts: list[np.ndarray] = []
    bad_footer = 0
    while time.monotonic() - t0 < a.sec:
        h = ser.read(1)
        if not h or h[0] != 0xAA:
            continue
        payload = ser.read(PAYLOAD)
        f = ser.read(1)
        if len(payload) != PAYLOAD or not f or f[0] != 0x55:
            bad_footer += 1
            continue
        arrivals.append(time.monotonic())
        bursts.append(np.frombuffer(payload, dtype="<u4").reshape(16, 10))
    ser.close()
    if len(arrivals) < 10:
        print(f"버스트 {len(arrivals)}개뿐 — 센서/포트 확인")
        return 1
    dt = np.diff(arrivals) * 1000
    v = np.stack(bursts).astype(np.int64)
    dup = float((np.diff(v, axis=2) == 0).mean())
    n_gap = int((dt > 80).sum())
    print(f"버스트 {len(arrivals)}개 / {a.sec:.0f}s  (평균 {len(arrivals)/a.sec:.1f}/s, 기대 20/s)")
    print(f"간격(ms): 중앙 {np.median(dt):.1f}  p95 {np.percentile(dt,95):.1f}  "
          f"max {dt.max():.0f}  | 80ms 초과(유실) {n_gap}회")
    print(f"버스트 내 인접 동일값: {dup*100:.1f}%  → 실효 신규샘플 ≈ {200*(1-dup):.0f}Hz")
    print(f"프레이밍 오류: {bad_footer}회")
    ok = np.median(dt) < 55 and n_gap <= 1 and bad_footer <= 2
    print("판정:", "정상 — 전송이 샘플링을 막지 않음" if ok else "★이상 — 위 수치 확인 필요")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
