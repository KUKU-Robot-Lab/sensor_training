#!/usr/bin/env python3
"""밴딩 raw(due+ether CSV) → 학습용 npz + 기하 GT/자가추정 교차검증.

취득: 양단 고정 센서를 스테이지 Y로 밀어 원호 좌굴. 원점(Y=0)이 flat, Y>0 이 (+)밴딩.
파이프라인:
  1) due_data.csv(16ch) + ethermotion_data.csv(Y,Z) 병합 (burst 평균 → Y 시간보간)
  2) baseline = |Y|<tol 인 flat 프레임 평균 → Δp_i = -(p_i - base)/base  [상대변화]
  3) (+)방향 push 세그먼트: 로딩 구간에서 Y∈[0, Ymax]
  4) 기하 GT: δ=Y → 원호모델 κ_geo, R, θ  (sats.bending.geometry, L=--bend-length)
  5) 자가추정: κ̂ = Σz_iΔp_i/Σz_i²
  6) 저장: <out>/<trial>.npz {sensor[N,16], bend_deg, kappa_geo, kappa_hat, delta_mm, y_mm}
  7) 교차검증: κ̂ vs κ_geo 선형회귀(slope·R²) → summary.json (+ --figure 시 png)

sensor 는 dataset.py 컨트랙트대로 raw 16ch. bend_deg = signed θ[deg](여기선 +).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sats.bending.geometry import (
        DEFAULT_BEND_LENGTH_MM, TAXEL_Z_MM,
        curvature_from_compression, self_estimated_curvature, calibrate,
    )
except ImportError:  # 직접 실행 fallback
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from sats.bending.geometry import (  # type: ignore[no-redef]
        DEFAULT_BEND_LENGTH_MM, TAXEL_Z_MM,
        curvature_from_compression, self_estimated_curvature, calibrate,
    )

SKIN_COLS = [f"Skin{i}" for i in range(1, 17)]
COUNTS_PER_MM = 10000.0        # ethermotion pulse: 10000 counts = 1mm
Y_FLAT_TOL_MM = 0.3            # |Y|<tol → flat baseline 프레임
TRIAL_DIR_RE = re.compile(r"^\d{8}_test\d+$")


import glob


def _has_csv(trial_dir: Path) -> bool:
    return (trial_dir / "due_data.csv").exists() and (trial_dir / "ethermotion_data.csv").exists()


def _has_bin(trial_dir: Path) -> bool:
    return bool(glob.glob(str(trial_dir / "due_raw_burst_*.bin"))) and \
        bool(glob.glob(str(trial_dir / "ethermotion_encoder_*.bin")))


def merge_trial_bin(trial_dir: Path) -> pd.DataFrame:
    """신규 취득(raw .bin만): due_raw_burst + ethermotion_encoder → burst df.

    CSV 경로와 동일 스키마(burst_index 평균, y_mm/z_mm 시간보간) 산출. bin_merge 재사용.
    """
    from sats.preprocessing.bin_merge import FIFO_FRAMES, load_due_bin, load_ethermotion_bin
    due = load_due_bin(glob.glob(str(trial_dir / "due_raw_burst_*.bin"))[0])
    eth = load_ethermotion_bin(glob.glob(str(trial_dir / "ethermotion_encoder_*.bin"))[0])
    n = len(due.sensors)
    df = pd.DataFrame(np.asarray(due.sensors, np.float64), columns=SKIN_COLS)
    df["burst_index"] = np.arange(n) // FIFO_FRAMES            # FIFO 10프레임 → 1 burst
    df["time_s"] = np.asarray(due.time_s, np.float64)
    burst = df.groupby("burst_index")[["time_s", *SKIN_COLS]].mean().reset_index()
    et = np.asarray(eth.time_s, np.float64)
    burst["y_mm"] = np.interp(burst["time_s"], et, np.asarray(eth.y_mm, np.float64))  # bin_merge=이미 mm
    burst["z_mm"] = np.interp(burst["time_s"], et, np.asarray(eth.z_mm, np.float64))
    return burst


def merge_trial(trial_dir: Path) -> pd.DataFrame:
    """due(16ch)+ether(Y,Z) 병합. CSV 있으면 CSV, 없으면 raw .bin(신규 취득)."""
    if not _has_csv(trial_dir) and _has_bin(trial_dir):
        return merge_trial_bin(trial_dir)
    due = pd.read_csv(trial_dir / "due_data.csv")
    ether = pd.read_csv(trial_dir / "ethermotion_data.csv")
    for col in ("burst_index", "time_s", *SKIN_COLS):
        if col not in due.columns:
            raise ValueError(f"{trial_dir}: due_data.csv missing column {col!r}")
    if "Y" not in ether.columns or "time_s" not in ether.columns:
        raise ValueError(f"{trial_dir}: ethermotion_data.csv missing Y/time_s")
    burst = due.groupby("burst_index")[["time_s", *SKIN_COLS]].mean().reset_index()
    burst["y_mm"] = np.interp(burst["time_s"], ether["time_s"], ether["Y"] / COUNTS_PER_MM)
    z_src = ether["Z"] / COUNTS_PER_MM if "Z" in ether.columns else 0.0
    burst["z_mm"] = np.interp(burst["time_s"], ether["time_s"], z_src) if "Z" in ether.columns else 0.0
    return burst


def compute_baseline(burst: pd.DataFrame) -> np.ndarray:
    """flat(|δ|<tol) 프레임 평균 = 채널별 baseline [16]. 없으면 |δ| 최소 10프레임.

    δ = burst["delta"] (= y_mm − y_origin). v0는 y_origin=0 이라 δ==y_mm.
    """
    flat = burst[burst["delta"].abs() < Y_FLAT_TOL_MM]
    if len(flat) < 3:
        order = np.argsort(burst["delta"].abs().to_numpy())[:10]
        flat = burst.iloc[order]
    return np.asarray(flat[SKIN_COLS].mean(), dtype=float)


def select_segment(burst: pd.DataFrame, direction: str) -> np.ndarray:
    """밴딩 세그먼트 인덱스(δ = burst["delta"] 기준). direction:

    - positive: 시작~δ양의최대(argmax) 중 δ≥0 (로딩 램프, 복귀 스트로크 배제) — 검증됨.
    - negative: 시작~δ음의최소(argmin) 중 δ≤0 (음방향 로딩 램프).
    - both: 전체 프레임(부호별 signed). ⚠ 대칭 스윕 아니면 음 브랜치 품질 낮음.
    """
    dv = burst["delta"].to_numpy(dtype=float)
    if direction == "positive":
        idx = np.arange(0, int(np.argmax(dv)) + 1)
        return idx[dv[idx] >= 0.0]
    if direction == "negative":
        idx = np.arange(0, int(np.argmin(dv)) + 1)
        return idx[dv[idx] <= 0.0]
    if direction == "both":
        return np.arange(len(dv))
    raise ValueError(f"direction must be positive/negative/both, got {direction!r}")


def process_trial(trial_dir: Path, length_mm: float, delta_max_valid: float,
                  direction: str = "positive", y_origin_mm: float = 0.0,
                  z_active_min_mm: float | None = None) -> dict:
    """밴딩 raw → 곡률 라벨 npz dict.

    y_origin_mm: flat(δ=0) 기준 Y 위치. δ = y_mm − y_origin_mm. v0=0(Y=0 flat),
      v5=18.0(Z=12 도달 후 Y=18 부터 밴딩) — 취득 rig에 따라 지정.
    z_active_min_mm: 지정 시 z_mm ≥ 이 값 프레임만 사용(밴딩 활성 구간). v5=~11.5(Z=12).
    """
    burst = merge_trial(trial_dir)
    if z_active_min_mm is not None:
        burst = burst[burst["z_mm"] >= float(z_active_min_mm)].reset_index(drop=True)
        if len(burst) < 5:
            raise ValueError(f"{trial_dir}: z≥{z_active_min_mm}mm 프레임 부족 (n={len(burst)})")
    burst = burst.copy()
    burst["delta"] = burst["y_mm"] - float(y_origin_mm)           # signed δ (flat 기준)
    baseline = compute_baseline(burst)
    seg = select_segment(burst, direction)
    if seg.size < 5:
        raise ValueError(f"{trial_dir}: {direction} 세그먼트가 너무 짧음 (n={seg.size})")

    raw = burst.loc[seg, SKIN_COLS].to_numpy(dtype=float)          # [N,16] raw
    y_stage = burst.loc[seg, "y_mm"].to_numpy(dtype=float)         # [N] 절대 Y(위치 기록용)
    y_mm = burst.loc[seg, "delta"].to_numpy(dtype=float)          # [N] signed δ(=Y−origin)
    sign = np.sign(y_mm)                                           # 밴딩 방향(±)
    delta_abs = np.abs(y_mm)                                       # 원호 계산용 |δ|
    delta_mm = y_mm                                                # signed δ 저장(=Y)
    # 채널별 상대변화 Δp (baseline 0 나눗셈 보호)
    safe_base = np.where(np.abs(baseline) < 1e-9, 1e-9, baseline)
    delta_p = -(raw - baseline) / safe_base                       # [N,16]

    geo = curvature_from_compression(delta_abs, length_mm)        # |δ| 기반
    kappa_geo = sign * geo["kappa_per_mm"]                        # signed κ
    bend_deg = sign * geo["theta_deg"]                           # signed θ
    k_hat = self_estimated_curvature(delta_p, TAXEL_Z_MM)        # [N] signed, arbitrary scale
    # 유효 마스크: |δ|>delta_max_valid 는 센서 포화(κ̂ 붕괴)로 곡률 라벨 신뢰불가
    valid = delta_abs <= float(delta_max_valid)

    return {
        "trial": trial_dir.name,
        "n": int(seg.size),
        "n_valid": int(valid.sum()),
        "arrays": {
            "sensor": raw.astype(np.float32),
            "baseline": baseline.astype(np.float32),   # flat(δ=0) 채널별 기준 [16] (Phase2 restorer 타깃)
            "bend_deg": bend_deg.astype(np.float32),          # signed θ
            "kappa_geo": kappa_geo.astype(np.float32),         # signed κ
            "kappa_hat": k_hat.astype(np.float32),             # signed κ̂
            "radius_mm": geo["radius_mm"].astype(np.float32),  # |R| (부호 무관)
            "delta_mm": delta_mm.astype(np.float32),           # signed δ (= Y−origin, flat 기준)
            "y_mm": y_stage.astype(np.float32),                # 절대 Y 위치(provenance)
            "valid": valid,  # |δ|≤delta_max_valid (센서 포화 이전, 곡률 라벨 신뢰 구간)
        },
        "y_max_mm": float(np.max(np.abs(y_mm))),
        "delta_max_mm": float(np.max(delta_abs)),
        "theta_max_deg": float(np.nanmax(np.abs(bend_deg))),
        "kappa_geo_max": float(np.nanmax(np.abs(kappa_geo))),
    }


def discover_trials(raw_root: Path, exclude: set[str]) -> list[Path]:
    """raw_root 하위에서 due+ether CSV가 있는 세션 폴더 전부(명명 무관).

    v0는 ``\\d{8}_testN``, v5는 ``v5_bending_0.1mm_10mm_N`` 등 명명이 달라 정규식 대신
    필수 CSV 존재로 판별한다. exclude 로 특정 폴더 배제.
    """
    trials = [p for p in sorted(raw_root.iterdir())
              if p.is_dir() and p.name not in exclude
              and (_has_csv(p) or _has_bin(p))]      # CSV 또는 raw .bin(신규 취득)
    if not trials:
        raise FileNotFoundError(f"밴딩 세션 폴더 없음(due+ether CSV/bin 기준): {raw_root} "
                                f"(exclude={sorted(exclude)})")
    return trials


def write_figure(results: list[dict], calib: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cutoff = float(np.max([r["arrays"]["delta_mm"][r["arrays"]["valid"]].max() for r in results
                           if r["arrays"]["valid"].any()], initial=0.0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for r in results:
        a = r["arrays"]
        v = a["valid"]
        ax1.plot(a["delta_mm"], a["kappa_hat"], ".", ms=2, label=r["trial"])
        ax2.plot(a["kappa_hat"][v], a["kappa_geo"][v], ".", ms=2, label=r["trial"])
    ax1.axvline(cutoff, color="k", ls="--", lw=1, label=f"valid limit {cutoff:g}mm")
    ax1.set_xlabel("delta = Y (mm)"); ax1.set_ylabel("kappa_hat (self-estimate)")
    ax1.set_title("sensor curvature signal vs delta (beyond limit = saturation)")
    ax1.grid(alpha=.3); ax1.legend(fontsize=7)
    ax2.set_xlabel("kappa_hat (self-estimate, arb. scale)"); ax2.set_ylabel("kappa_geo (arc GT, 1/mm)")
    ax2.set_title(f"cross-check (valid): R2={calib['r2']:.3f}, slope={calib['slope']:.3f}")
    ax2.grid(alpha=.3); ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"figure: {out_png}")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", type=Path, default=repo / "skin_ws/raw_data/bending/v0")
    p.add_argument("--out-dir", type=Path, default=repo / "learning_data/bending/v0")
    p.add_argument("--bend-length", type=float, default=DEFAULT_BEND_LENGTH_MM,
                   help="유효 굽힘 길이 L(mm). 측정값으로 교체 시 GT 재산출.")
    p.add_argument("--delta-max-valid", type=float, default=10.0,
                   help="곡률 라벨 신뢰 상한 |δ|(mm). 초과분은 센서 포화로 valid=False. "
                        "실측상 δ≈10mm(κ≈0.076/mm)에서 κ̂ 포화.")
    p.add_argument("--direction", choices=["positive", "negative", "both"], default="positive",
                   help="밴딩 방향. positive=검증된 (+)로딩(기본). both=signed 양방향 "
                        "(⚠ v0는 대칭 스윕 아님 → 음 브랜치 품질 낮음, G1 취득 필요).")
    p.add_argument("--y-origin-mm", type=float, default=0.0,
                   help="flat(δ=0) 기준 Y 위치(mm). δ = Y − origin. v0=0, v5=18.0.")
    p.add_argument("--z-active-min-mm", type=float, default=None,
                   help="지정 시 z_mm≥이 값 프레임만 사용(밴딩 활성 구간). v5=11.5(Z=12 도달 후).")
    p.add_argument("--exclude", nargs="*", default=["20260725_test1"],
                   help="제외할 trial 폴더명 (기본 test1 = 밴딩 스윕 아님).")
    p.add_argument("--figure", action="store_true", help="교차검증 png 저장")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trials = discover_trials(args.raw_root, set(args.exclude))
    print(f"대상 trial {len(trials)}개 (L={args.bend_length}mm): {[t.name for t in trials]}")

    results = [process_trial(t, args.bend_length, args.delta_max_valid, args.direction,
                             y_origin_mm=args.y_origin_mm, z_active_min_mm=args.z_active_min_mm)
               for t in trials]
    for r in results:
        print(f"  {r['trial']}: n={r['n']} (valid≤{args.delta_max_valid:g}mm: {r['n_valid']})  "
              f"Ymax={r['y_max_mm']:.1f}mm  θmax={r['theta_max_deg']:.0f}°  "
              f"κ_geo_max={r['kappa_geo_max']:.4f}/mm")

    # 교차검증은 valid(포화 이전) 프레임에서만 — 여기가 곡률 라벨 신뢰 구간
    def _pool(key: str, valid_only: bool) -> np.ndarray:
        return np.concatenate([
            r["arrays"][key][r["arrays"]["valid"]] if valid_only else r["arrays"][key]
            for r in results])
    calib = calibrate(_pool("kappa_hat", True), _pool("kappa_geo", True))
    calib_all = calibrate(_pool("kappa_hat", False), _pool("kappa_geo", False))
    print(f"교차검증 κ̂↔κ_geo [valid δ≤{args.delta_max_valid:g}mm]: "
          f"slope={calib['slope']:.3f} R²={calib['r2']:.3f} (n={calib['n']})  "
          f"| 전체범위 R²={calib_all['r2']:.3f} (n={calib_all['n']})")

    if args.dry_run:
        print("[dry-run] npz 저장 생략")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        np.savez(args.out_dir / f"{r['trial']}.npz", **r["arrays"],
                 bend_length_mm=np.float32(args.bend_length))
    summary = {
        "bend_length_mm": args.bend_length,
        "delta_max_valid_mm": args.delta_max_valid,
        "excluded": args.exclude,
        "calibration_khat_vs_kgeo_valid": calib,
        "calibration_khat_vs_kgeo_all": calib_all,
        "note": "δ>delta_max_valid 는 센서 포화(κ̂ 붕괴)로 valid=False. 학습·평가는 valid 프레임만 권장.",
        "trials": [{k: r[k] for k in ("trial", "n", "n_valid", "y_max_mm", "delta_max_mm",
                                      "theta_max_deg", "kappa_geo_max")} for r in results],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {args.out_dir}  ({len(results)} npz + summary.json)")
    if args.figure:
        write_figure(results, calib, args.out_dir / "crosscheck.png")


if __name__ == "__main__":
    main()
