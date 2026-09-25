#!/usr/bin/env python3
"""G1 관측성 평가 하네스 — 곡률 관측성(Curvature Observability) 판정.

마스터 게이트 G1 조건을 코드로:
  (a) remounting-held-out: 세션 단위 홀드아웃에서 곡률(θ) 회귀 MAE 가 기준선 대비 개선.
  (b) signed 순서 보존: 예측 θ̂ 와 GT θ 의 Spearman ρ 높음(단조 정렬 유지).
  (c) 이진 붕괴 금지: flat/bent 2-클러스터로 무너지지 않고 연속 회귀(R²) 성립.
  (d) drift: |θ|≈0(무밴딩) 프레임의 예측 분산이 곡률 신호보다 작음.

세션 = trial 이름 앞 8자리(YYYYMMDD) 로 그룹핑(같은 날짜=같은 mounting 가정).
≥2 세션이면 leave-one-session-out, 1 세션이면 trial 홀드아웃 스모크(‼G1 미충족 경고).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from .config import BendingConfig
from .dataset import build_windows
from .train_bending import discover_trials, train_estimator_holdout

SESSION_RE = re.compile(r"^(\d{8})")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def session_of(trial: str) -> str:
    m = SESSION_RE.match(trial)
    return m.group(1) if m else trial


def observability_metrics(model, stats, data_dir: Path, val_trials: list[str],
                          cfg: BendingConfig, device: str) -> dict:
    Xva, yva = build_windows(data_dir, val_trials, window_size=cfg.window_size,
                             valid_only=True, stats=stats)
    X = torch.from_numpy(Xva).to(device)
    L = torch.full((len(X),), cfg.window_size, dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(X, L).cpu().numpy()
    mae = float(np.abs(pred - yva).mean())
    baseline = float(np.abs(yva - yva.mean()).mean())
    rho = _spearman(pred, yva)
    # 연속성(이진붕괴 금지): pred~yva 선형 R²
    if yva.std() > 1e-9:
        sl, ic = np.polyfit(yva, pred, 1)
        r2 = 1.0 - float(np.sum((pred - (sl * yva + ic)) ** 2)) / float(np.sum((pred - pred.mean()) ** 2) + 1e-12)
    else:
        r2 = float("nan")
    # drift: |θ|<10° 프레임 예측 표준편차 vs 전체 곡률 신호 표준편차
    near0 = np.abs(yva) < 10.0
    drift = float(pred[near0].std()) if near0.sum() > 3 else float("nan")
    return {"n": int(len(yva)), "mae_deg": mae, "baseline_mae_deg": baseline,
            "spearman_rho": rho, "linear_r2": r2, "drift_std_deg": drift,
            "signal_std_deg": float(yva.std())}


def run_g1(cfg: BendingConfig, data_dir: str | Path, *, epochs: int = 80,
           device: str | None = None, session_per_trial: bool = False) -> dict:
    """G1 관측성. session_per_trial=True 면 각 trial 폴더 = 독립 세션(remounting)으로 간주.

    기본은 이름 앞 8자리(날짜)로 세션 그룹핑(같은 날=같은 mounting 가정, v0). v5처럼
    같은 날 여러 번 탈착→재장착하면 날짜가 겹치므로 session_per_trial 로 폴더=세션 처리.
    """
    data_dir = Path(data_dir)
    device = device or (cfg.device if torch.cuda.is_available() else "cpu")
    trials = discover_trials(data_dir)
    key = (lambda t: t) if session_per_trial else session_of
    sessions = sorted(set(key(t) for t in trials))
    results = []
    if len(sessions) >= 2:
        mode = "leave-one-session-out"
        folds = [[t for t in trials if key(t) == s] for s in sessions]
    else:
        mode = "trial-holdout (⚠ 단일 세션 — G1 remounting 미충족, 스모크만)"
        folds = [[t] for t in trials]
    for held in folds:
        res = train_estimator_holdout(cfg, data_dir, val_trials=held, epochs=epochs)
        m = observability_metrics(res["model"], res["stats"], data_dir, held, cfg, device)
        m["held_out"] = held
        results.append(m)
        print(f"  홀드아웃 {held}: MAE={m['mae_deg']:.2f}° (기준선 {m['baseline_mae_deg']:.2f}°) "
              f"ρ={m['spearman_rho']:.3f} R²={m['linear_r2']:.3f} drift={m['drift_std_deg']:.2f}°")
    agg = {"mode": mode, "sessions": sessions,
           "mean_mae_deg": float(np.mean([r["mae_deg"] for r in results])),
           "mean_spearman": float(np.nanmean([r["spearman_rho"] for r in results])),
           "folds": results}
    return agg


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="G1 곡률 관측성 평가 하네스.")
    p.add_argument("--data-dir", type=Path, default=repo / "learning_data/bending/v0")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--session-per-trial", action="store_true",
                   help="각 trial 폴더=독립 세션(remounting). 같은 날 여러 remounting(v5 2mm)일 때 사용.")
    p.add_argument("--estimator-arch", choices=["lstm", "cnn2d", "cnn2d_shape", "cnn_lstm", "mlp_frame"], default="lstm",
                   help="곡률 estimator 구조")
    args = p.parse_args()
    print(f"G1 관측성 평가 — {args.data_dir} (arch={args.estimator_arch})")
    agg = run_g1(BendingConfig(estimator_arch=args.estimator_arch), args.data_dir,
                 epochs=args.epochs, device=args.device, session_per_trial=args.session_per_trial)
    print(f"\n모드: {agg['mode']}  세션: {agg['sessions']}")
    print(f"평균 MAE={agg['mean_mae_deg']:.2f}°  평균 Spearman ρ={agg['mean_spearman']:.3f}")
    print(f"G1 판정: {'예비 통과 지표(단 세션≥3 필요)' if agg['mean_spearman'] > 0.9 else '순서상관 부족'}"
          f" — {'세션 3+ 확보 시 정식 판정 가능' if len(agg['sessions']) < 3 else '세션 요건 충족'}")


if __name__ == "__main__":
    main()
