"""밴딩 raw(due+ethermotion bin) 센서 신호 vs 변형량(δ) 비교 분석.

구·신 센서/취득 비교용. 각 세션의 due_raw_burst.bin(16ch) + ethermotion(Y구동)을
파싱해, flat baseline 대비 상대변화 |Δp|(%)를 변형량 δ(=Y−Y_min) 구간별로 집계.
핵심 질문: 신 데이터가 (a) 더 큰 신호, (b) 저-δ에서 신호, (c) 넓은 관측범위를 갖는가.

출력: JSON(세션·집계). 텍스트/그래프 없음(양 머신 동일 실행 후 수치 대조).
실행: .venv/bin/python scripts/compare_bending_sensor.py --dir skin_ws/raw_data/bending/v6 [--label local]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo 루트 → sats 임포트
from sats.preprocessing.bin_merge import load_due_bin, load_ethermotion_bin

FLAT_TOL_MM = 0.5


def _session_stats(test_dir: Path, y_start: float, y_end: float, z_min: float | None) -> dict | None:
    """밴딩 시작점(y_start=flat baseline)~y_end 유효창만 분석. δ' = y − y_start."""
    due_g = glob.glob(str(test_dir / "due_raw_burst_*.bin"))
    eth_g = glob.glob(str(test_dir / "ethermotion_encoder_*.bin"))
    if not due_g or not eth_g:
        return None
    due = load_due_bin(due_g[0])
    eth = load_ethermotion_bin(eth_g[0])
    s = np.asarray(due.sensors, np.float64)             # [N,16]
    ts = np.asarray(due.time_s, np.float64)
    et = np.asarray(eth.time_s, np.float64)
    idx = np.searchsorted(et, ts).clip(0, len(et) - 1)  # 시간 최근접 정렬
    y = np.asarray(eth.y_mm, np.float64)[idx]
    z = np.asarray(eth.z_mm, np.float64)[idx]
    # 유효창: 밴딩 시작~끝 (+옵션 z 필터)
    win = (y >= y_start) & (y <= y_end)
    if z_min is not None:
        win &= (z >= z_min)
    if win.sum() < 20:
        return None
    s, y = s[win], y[win]
    delta = y - y_start                                # δ' = y − 밴딩시작(0~유효폭)
    flat = delta < FLAT_TOL_MM                         # 밴딩 시작점 = flat baseline
    if flat.sum() < 5:
        flat = delta < np.percentile(delta, 8)
    base = s[flat].mean(0)
    safe = np.where(np.abs(base) < 1e-9, 1e-9, base)
    dp = np.abs((s - base) / safe * 100.0).mean(1)     # 프레임별 16ch 평균 |Δp|%
    span = y_end - y_start
    edges = np.round(np.linspace(0, span, min(11, int(span) + 1)), 1)
    curve = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (delta >= lo) & (delta < hi)
        if m.sum() > 8:
            curve.append({"delta_lo": float(lo), "n": int(m.sum()),
                          "signal_pct": round(float(dp[m].mean()), 3)})
    return {
        "session": test_dir.name,
        "n_frames": int(len(s)),
        "delta_max_mm": round(float(delta.max()), 1),
        "baseline_mean": round(float(base.mean()), 0),
        "flat_noise_pct": round(float(dp[flat].mean()), 3),
        "signal_max_pct": round(float(np.percentile(dp, 99)), 3),
        "curve": curve,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="밴딩 버전 폴더(하위에 test*/ due+ether bin)")
    p.add_argument("--label", default="")
    p.add_argument("--y-start", type=float, default=23.0, help="밴딩 시작 y(mm)=flat baseline (신규 v6=23)")
    p.add_argument("--y-end", type=float, default=33.0, help="밴딩 끝 y(mm) (신규 v6=33)")
    p.add_argument("--z-min", type=float, default=None, help="z 하한 필터(옵션)")
    args = p.parse_args()
    root = Path(args.dir)
    tests = sorted([d for d in root.rglob("*") if d.is_dir() and glob.glob(str(d / "due_raw_burst_*.bin"))])
    sessions = [st for d in tests if (st := _session_stats(d, args.y_start, args.y_end, args.z_min))]
    # 집계: δ구간별 신호 평균(세션 평균)
    agg: dict[float, list] = {}
    for s in sessions:
        for c in s["curve"]:
            agg.setdefault(c["delta_lo"], []).append(c["signal_pct"])
    agg_curve = [{"delta_lo": k, "signal_pct": round(float(np.mean(v)), 3), "n_sess": len(v)}
                 for k, v in sorted(agg.items())]
    out = {
        "label": args.label, "dir": str(root), "n_sessions": len(sessions),
        "baseline_mean_all": round(float(np.mean([s["baseline_mean"] for s in sessions])), 0) if sessions else None,
        "flat_noise_pct_all": round(float(np.mean([s["flat_noise_pct"] for s in sessions])), 3) if sessions else None,
        "signal_max_pct_all": round(float(np.mean([s["signal_max_pct"] for s in sessions])), 3) if sessions else None,
        "delta_max_mm_all": round(float(np.mean([s["delta_max_mm"] for s in sessions])), 1) if sessions else None,
        "signal_vs_delta": agg_curve,
        "sessions": sessions,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
