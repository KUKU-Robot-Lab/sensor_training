#!/usr/bin/env python3
"""
sats/tools/analyze_gt.py

GT 값 범위 진단 도구.

val_rmse=0.00579 이 좋은 성능인지 판단하기 위해
GT의 실제 값 분포(단위: N/mm²)를 분석한다.

실행:
    cd <repo>
    python3 -m sats.tools.analyze_gt
    python3 -m sats.tools.analyze_gt --trial ecomesh_d5_z1.5_test1
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


GT_DIR = Path("sats/preprocessing/gt_output_v1")
INDEX_PATH = GT_DIR / "dataset_index.json"


def analyze_single(npy_path: Path, trial_id: str) -> dict:
    targets = np.load(str(npy_path), mmap_mode="r")   # [N, 40, 40]
    N = targets.shape[0]

    flat = targets.reshape(N, -1)           # [N, 1600]
    row_max  = flat.max(axis=1)             # [N]  각 timestep의 peak 값
    row_sum  = flat.sum(axis=1)             # [N]  각 timestep의 합계

    # 비접촉 행 = 전체가 0
    active_mask = row_max > 0
    n_active = int(active_mask.sum())
    n_zero   = N - n_active

    active_targets = targets[active_mask]   # [K, 40, 40]
    active_flat    = active_targets.reshape(n_active, -1) if n_active > 0 else np.empty((0, 1600))

    # peak timestep: row_sum가 최대인 row
    peak_idx = int(row_sum.argmax())
    peak_map = targets[peak_idx]           # [40, 40]
    peak_val = float(peak_map.max())

    result = {
        "trial_id":    trial_id,
        "n_total":     N,
        "n_active":    n_active,
        "n_zero":      n_zero,
        "zero_ratio":  n_zero / max(N, 1),
        "global_max":  float(targets.max()),
        "active_peak_mean": float(active_flat.max(axis=1).mean()) if n_active > 0 else 0.0,
        "active_peak_std":  float(active_flat.max(axis=1).std())  if n_active > 0 else 0.0,
        "active_peak_max":  float(active_flat.max()) if n_active > 0 else 0.0,
        "active_mean_all":  float(active_flat.mean()) if n_active > 0 else 0.0,
        "sequence_peak_val": peak_val,
    }
    return result


def summarize(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("GT 값 범위 진단 (단위: N/mm²)")
    print("=" * 70)
    print(f"{'trial_id':<35} {'peak_max':>9} {'peak_mean':>10} {'zero%':>7}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['trial_id']:<35} "
            f"{r['active_peak_max']:>9.5f} "
            f"{r['active_peak_mean']:>10.5f} "
            f"{r['zero_ratio']*100:>6.1f}%"
        )

    all_peak_max  = [r["active_peak_max"]  for r in results if r["n_active"] > 0]
    all_peak_mean = [r["active_peak_mean"] for r in results if r["n_active"] > 0]
    global_max    = max(r["global_max"] for r in results)

    print("=" * 70)
    print(f"  전체 peak 최대값: {global_max:.6f} N/mm²")
    print(f"  trial별 peak_max  평균: {np.mean(all_peak_max):.6f}  std: {np.std(all_peak_max):.6f}")
    print(f"  trial별 peak_mean 평균: {np.mean(all_peak_mean):.6f}  std: {np.std(all_peak_mean):.6f}")

    # val_rmse 해석
    val_rmse = 0.005789   # cnn_v1 best
    print()
    print("─" * 70)
    print(f"  val_rmse (cnn_v1 best): {val_rmse:.6f}")
    if all_peak_max:
        avg_peak = float(np.mean(all_peak_max))
        ratio = val_rmse / avg_peak * 100
        print(f"  RMSE / 평균 peak_max  = {ratio:.1f}%")
        print(f"  → GT peak 대비 {ratio:.1f}% 오차 수준")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="GT 값 범위 진단")
    parser.add_argument("--gt-dir", type=Path, default=GT_DIR)
    parser.add_argument("--trial", default=None, help="특정 trial만 분석 (없으면 전체)")
    args = parser.parse_args()

    if args.trial:
        npy = args.gt_dir / f"{args.trial}_targets.npy"
        if not npy.exists():
            print(f"파일 없음: {npy}")
            return
        results = [analyze_single(npy, args.trial)]
    else:
        npy_files = sorted(args.gt_dir.glob("*_targets.npy"))
        if not npy_files:
            print(f"targets.npy 파일 없음: {args.gt_dir}")
            return
        results = []
        for npy in npy_files:
            tid = npy.stem.replace("_targets", "")
            print(f"  분석 중: {tid}...")
            results.append(analyze_single(npy, tid))

    summarize(results)


if __name__ == "__main__":
    main()
