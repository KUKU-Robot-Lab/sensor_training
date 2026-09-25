#!/usr/bin/env python3
"""SPEC.md 의 [measure@4090] 항목 실측 — 4090(GPU+eval 산출물)에서 실행.

측정:
  1) SATS 추론 처리량(predict Hz) — 실제 배포 엔진·디바이스 기준.
  2) grid/force config 스펙 재확인 출력.

사용:
  .venv/bin/python scripts/measure_demo_spec.py
  .venv/bin/python scripts/measure_demo_spec.py --n 500 --diameter 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:           # 스크립트 직접 실행 시 repo 루트 import 경로 확보
    sys.path.insert(0, str(_ROOT))

from sats.inference.inference_engine import SATSInferenceEngine  # noqa: E402


def measure_infer_hz(engine: SATSInferenceEngine, n: int) -> tuple[float, float]:
    """무작위 윈도우로 predict n회 — (평균 Hz, 프레임당 ms). GPU 워밍업 10회 제외."""
    win = np.random.randn(engine.window_size, 16).astype(np.float32)
    for _ in range(10):                      # 워밍업(CUDA 그래프/캐시)
        engine.predict(win)
    t0 = time.perf_counter()
    for _ in range(n):
        engine.predict(win)
    dt = time.perf_counter() - t0
    return n / dt, dt / n * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser(description="데모 스펙 실측(4090)")
    ap.add_argument("--run-dir", default=str(_ROOT / "sats/training/runs/ecomesh_v6_deploy_all4"))
    ap.add_argument("--diameter", type=float, default=10.0, help="D10 기준")
    ap.add_argument("--n", type=int, default=300, help="추론 반복 수")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    engine = SATSInferenceEngine(args.run_dir, device=args.device,
                                 indenter_diameter_mm=args.diameter)
    hz, ms = measure_infer_hz(engine, args.n)

    area_mm = (engine.grid_max_mm - engine.grid_min_mm)
    print("\n" + "=" * 56)
    print(f"  SATS 데모 스펙 실측 (D{args.diameter:g}, device={engine.device})")
    print("=" * 56)
    print(f"  output map        : {engine.grid_size}x{engine.grid_size} @ {engine.grid_step_mm:g} mm")
    print(f"  sensing area      : {area_mm:g} x {area_mm:g} mm "
          f"({engine.grid_min_mm:g} .. {engine.grid_max_mm:g})")
    print(f"  taxel area (cell) : {engine.taxel_area:g} mm^2")
    print(f"  window size       : {engine.window_size}")
    print(f"  inference rate    : {hz:7.1f} Hz  ({ms:.2f} ms/frame)  ← SPEC.md 추론 출력율")
    print("=" * 56)
    print("  x/y localization err · Fz range/resolution 는 라벨 eval 필요:")
    print("    → 다중접촉 eval(sats/tools/multicontact_metrics) 또는 v6 test 세트로 산출.")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
