#!/usr/bin/env python3
"""무접촉 노이즈 플로어 실측 → 접촉 필터 임계 권장 (4090, 실센서).

필터(fz_on/fz_off/min_peak)는 '무접촉일 때 모델이 내는 fz·peak'(노이즈 플로어)에서
정해야 오검출을 막는다. 센서에서 손 떼고 N프레임 추론해 분포를 재고 임계를 제안한다.

사용(4090, 손 떼고):
  .venv/bin/python scripts/measure_resolution.py --n 300
  .venv/bin/python scripts/measure_resolution.py --n 300 --diameter 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sats.inference.inference_engine import SATSInferenceEngine  # noqa: E402
from sats.inference.run_demo import auto_detect_port             # noqa: E402


def _reader(args, window_size):
    if args.mock:
        from sats.inference.mock_reader import MockSensorReader
        r = MockSensorReader(window_size=window_size); r.start(); return r
    port = auto_detect_port(args.baudrate) if args.port == "auto" else args.port
    if port is None:
        print("[오류] 포트 자동탐지 실패 — --port 지정"); sys.exit(1)
    from sats.inference.serial_reader import SensorSerialReader
    r = SensorSerialReader(port=port, baudrate=args.baudrate, window_size=window_size,
                           baseline_seconds=args.baseline_seconds, startup_delay=args.startup_delay,
                           protocol=args.protocol)
    r.start()
    print(f"[reader] {port}@{args.baudrate} — 손 떼고 baseline 대기...")
    t0 = time.time()
    while not r.baseline_ready:
        if getattr(r, "error_message", None):
            print(f"[오류] {r.error_message}"); r.stop(); sys.exit(1)
        time.sleep(0.3)
        if time.time() - t0 > 30:
            print("[오류] baseline 타임아웃"); r.stop(); sys.exit(1)
    print("[reader] baseline 완료\n")
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description="무접촉 노이즈 플로어 → 필터 임계 권장")
    ap.add_argument("--run-dir", default=str(_ROOT / "sats/training/runs/ecomesh_v6_deploy_all4"))
    ap.add_argument("--diameter", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=300, help="무접촉 추론 프레임 수")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--port", default="auto")
    ap.add_argument("--baudrate", type=int, default=250000)
    ap.add_argument("--baseline-seconds", type=float, default=5.0)
    ap.add_argument("--startup-delay", type=float, default=2.0)
    ap.add_argument("--protocol", choices=["auto", "binary", "csv"], default="binary")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    engine = SATSInferenceEngine(args.run_dir, device=args.device, indenter_diameter_mm=args.diameter)
    reader = _reader(args, engine.window_size)

    print(f"[측정] 무접촉 {args.n}프레임 — 센서에 손대지 말 것...")
    peaks, fzs = [], []
    last_seq = 0
    while len(peaks) < args.n:
        if hasattr(reader, "get_latest_window_with_seq"):
            win, seq = reader.get_latest_window_with_seq()
        else:
            win, seq = reader.get_latest_window(), last_seq + 1
        if win is None or seq == last_seq:
            time.sleep(0.002); continue
        last_seq = seq
        pmap = engine.predict(win)
        peaks.append(float(np.clip(pmap, 0, None).max()))
        fzs.append(engine.get_fz(pmap))
    reader.stop()

    peaks, fzs = np.asarray(peaks), np.asarray(fzs)
    p99_fz, p99_pk = float(np.percentile(fzs, 99)), float(np.percentile(peaks, 99))
    fz_on = round(max(p99_fz * 1.5, 0.20), 3)          # 노이즈 p99 + 마진
    fz_off = round(fz_on * 0.5, 3)
    min_peak = round(p99_pk * 1.2, 2)

    print("\n" + "=" * 56)
    print(f"  무접촉 노이즈 플로어 (D{args.diameter:g}, n={len(peaks)})")
    print("=" * 56)
    print(f"  fz    : mean {fzs.mean():.3f}  std {fzs.std():.3f}  max {fzs.max():.3f}  p99 {p99_fz:.3f}  [N]")
    print(f"  peak  : mean {peaks.mean():.2f}  std {peaks.std():.2f}  max {peaks.max():.2f}  p99 {p99_pk:.2f}")
    print("-" * 56)
    print("  권장 필터 임계(run_dashboard):")
    print(f"    --fz-on {fz_on}  --fz-off {fz_off}  --min-fz {fz_off}")
    print(f"    (min-peak 참고: {min_peak} — 필요 시 extract_contacts min_peak_val)")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
