"""밴딩 모듈 학습 드라이버 (Phase 0 스켈레톤).

2단계 (SATS는 항상 동결):
- Phase 1 — BendingEstimator: 밴딩 데이터 → signed deg 회귀(MSE/MAE). ★데이터 오면 실행 가능.
- Phase 2 — BaselineRestorer: flat 등가 복원. 학습 신호 2안(데이터 취득 방식에 따라 선택):
    (A) 오프셋 지도: bending-only(무접촉) 신호 = 순수 밴딩 오프셋 → restorer가 재현하도록
        학습 → bending+contact에서 빼면 contact-only(flat 등가). 중첩 선형성 가정.
    (B) end-to-end: 밴딩+접촉 → restorer → ❄️SATS → 압력맵 손실을 flat 기준 대비 최소화.
        SATS 동결이라 grad는 restorer까지만 전파(pipeline.forward 참조).
- Phase 3 — Pipeline 검증: 밴딩 하 SATS 정확도 vs flat 기준(재학습 0).

여기서는 Phase 1 estimator 학습을 제공(데이터 배열만 있으면 동작). Phase 2/3은 취득 후.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .baseline_restorer import BaselineRestorer
from .bending_estimator import BendingEstimator, build_estimator
from .config import BendingConfig
from .dataset import (BendingTrial, NormStats, build_windows, compute_norm_stats,
                      load_bending_trial, make_windows)


def train_estimator(
    cfg: BendingConfig,
    windows: np.ndarray,      # [M, W, 16]
    degs: np.ndarray,         # [M] signed
    *,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> BendingEstimator:
    """BendingEstimator를 signed deg 회귀로 학습. 지표 = deg MAE."""
    device = cfg.device if torch.cuda.is_available() else "cpu"
    model = build_estimator(cfg).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.from_numpy(windows.astype(np.float32))
    y = torch.from_numpy(degs.astype(np.float32))
    lengths = torch.full((X.shape[0],), X.shape[1], dtype=torch.long)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            xb = X[b].to(device); yb = y[b].to(device); lb = lengths[b].to(device)
            pred = model(xb, lb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        with torch.no_grad():
            mae = (model(X.to(device), lengths.to(device)) - y.to(device)).abs().mean().item()
        print(f"[estimator] ep{ep + 1}/{epochs} mse={tot / n:.4f} deg_MAE={mae:.3f}")
    return model.eval()


TRIAL_GLOB = "*.npz"


def discover_trials(data_dir: Path) -> list[str]:
    """data_dir 의 밴딩 npz(정식 trial) 이름 목록. summary 등 비-trial 제외."""
    names = sorted(p.stem for p in data_dir.glob(TRIAL_GLOB))
    if not names:
        raise FileNotFoundError(f"밴딩 npz 없음: {data_dir}")
    return names


def _mae(model: BendingEstimator, X: torch.Tensor, y: torch.Tensor, win: int) -> float:
    lengths = torch.full((X.shape[0],), win, dtype=torch.long, device=X.device)
    with torch.no_grad():
        return (model(X, lengths) - y).abs().mean().item()


def train_estimator_holdout(
    cfg: BendingConfig,
    data_dir: str | Path,
    *,
    val_trials: list[str],
    valid_only: bool = True,
    epochs: int = 80,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """정식 Phase 1 학습: valid 필터 + train 통계 표준화 + trial-level 홀드아웃.

    train 통계로 표준화(추론까지 동일 적용 위해 stats 반환), val_trials 는 학습에서 제외.
    반환: {model, stats, history[list], val_mae(best), baseline_mae}.
    """
    data_dir = Path(data_dir)
    torch.manual_seed(seed)
    all_trials = discover_trials(data_dir)
    val_set = set(val_trials)
    unknown = val_set - set(all_trials)
    if unknown:
        raise ValueError(f"val_trials 에 없는 trial: {sorted(unknown)} (가용 {all_trials})")
    train_trials = [t for t in all_trials if t not in val_set]
    if not train_trials:
        raise ValueError("train trial 이 0개입니다. val_trials 를 줄이세요.")

    # 표준화 통계는 train 프레임(valid)만으로
    train_raw = np.concatenate([
        load_bending_trial(data_dir / f"{t}.npz", valid_only=valid_only).sensor
        for t in train_trials])
    stats = compute_norm_stats(train_raw)

    Xtr, ytr = build_windows(data_dir, train_trials, window_size=cfg.window_size,
                             valid_only=valid_only, stats=stats)
    Xva, yva = build_windows(data_dir, val_trials, window_size=cfg.window_size,
                             valid_only=valid_only, stats=stats)
    if len(Xtr) == 0 or len(Xva) == 0:
        raise ValueError(f"윈도우 부족: train={len(Xtr)} val={len(Xva)}")

    dev = cfg.device if torch.cuda.is_available() else "cpu"
    model = build_estimator(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr_t, ytr_t = torch.from_numpy(Xtr).to(dev), torch.from_numpy(ytr).to(dev)
    Xva_t, yva_t = torch.from_numpy(Xva).to(dev), torch.from_numpy(yva).to(dev)
    Ltr = torch.full((len(Xtr),), cfg.window_size, dtype=torch.long, device=dev)

    baseline_mae = float(np.abs(yva - ytr.mean()).mean())  # θ 평균 예측 기준선
    history, best = [], {"val_mae": float("inf"), "state": None}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), batch_size):
            b = perm[i:i + batch_size]
            loss = nn.functional.mse_loss(model(Xtr_t[b], Ltr[b]), ytr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        tr_mae = _mae(model, Xtr_t, ytr_t, cfg.window_size)
        va_mae = _mae(model, Xva_t, yva_t, cfg.window_size)
        history.append({"epoch": ep + 1, "train_mae": tr_mae, "val_mae": va_mae})
        if va_mae < best["val_mae"]:
            best = {"val_mae": va_mae, "state": {k: v.cpu().clone()
                                                 for k, v in model.state_dict().items()}}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"ep{ep + 1:3d}  train_MAE={tr_mae:5.2f}°  val_MAE={va_mae:5.2f}°")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"model": model.eval(), "stats": stats, "history": history,
            "val_mae": best["val_mae"], "baseline_mae": baseline_mae,
            "train_trials": train_trials, "val_trials": list(val_trials)}


def save_estimator(result: dict, cfg: BendingConfig, out_path: str | Path) -> None:
    """체크포인트 저장: 모델 + 정규화 통계(추론 동일 적용) + cfg + 지표."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": result["model"].state_dict(),
        "norm_mean": result["stats"].mean,
        "norm_std": result["stats"].std,
        "cfg": asdict(cfg),
        "val_mae": result["val_mae"],
        "baseline_mae": result["baseline_mae"],
        "train_trials": result["train_trials"],
        "val_trials": result["val_trials"],
    }, out_path)
    (out_path.with_suffix(".history.json")).write_text(
        json.dumps({"history": result["history"], "val_mae": result["val_mae"],
                    "baseline_mae": result["baseline_mae"]}, indent=2))
    print(f"saved: {out_path}  (val_MAE={result['val_mae']:.2f}° / baseline={result['baseline_mae']:.2f}°)")


def load_estimator(ckpt_path: str | Path, device: str = "cpu") -> tuple[BendingEstimator, NormStats]:
    """저장된 estimator + 정규화 통계 로드 (추론용)."""
    ck = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    cfg = BendingConfig(**ck["cfg"])
    model = build_estimator(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    stats = NormStats(mean=np.asarray(ck["norm_mean"], np.float32),
                      std=np.asarray(ck["norm_std"], np.float32))
    return model.eval(), stats


def _restorer_windows(
    data_dir: Path, trials: list[str], cfg: BendingConfig, stats: NormStats, valid_only: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """restorer용: 표준화 윈도우 [M,W,16] + deg[M] + trial별 flat baseline(표준화)[M,16]."""
    Xs, ds, bs = [], [], []
    for name in trials:
        z = np.load(data_dir / f"{name}.npz")
        m = np.asarray(z["valid"], bool) if (valid_only and "valid" in z) else np.ones(len(z["sensor"]), bool)
        sensor_n = stats.apply(np.asarray(z["sensor"])[m].astype(np.float32))
        base_n = stats.apply(np.asarray(z["baseline"], np.float32))    # [16]
        w, d = make_windows(BendingTrial(sensor=sensor_n.astype(np.float32),
                                         bend_deg=np.asarray(z["bend_deg"])[m].astype(np.float32)),
                            cfg.window_size)
        if len(w):
            Xs.append(w); ds.append(d); bs.append(np.tile(base_n, (len(w), 1)))
    if not Xs:
        raise ValueError("restorer 윈도우 0개")
    return np.concatenate(Xs), np.concatenate(ds), np.concatenate(bs)


def train_restorer_holdout(
    cfg: BendingConfig,
    data_dir: str | Path,
    *,
    val_trials: list[str],
    valid_only: bool = True,
    epochs: int = 80,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """정식 Phase 2 학습(오프셋 지도, A안): 밴딩(무접촉) 신호 → flat 등가(=baseline) 복원.

    restored = seq − offset(seq,deg) 가 표준화 baseline 에 수렴하도록 학습. GT deg 조건.
    지표 = 오프셋 제거율 = 1 − mean|restored−base| / mean|seq−base| (holdout).
    ※ 접촉 분리 검증은 bending+contact(G2) 데이터 필요 — 여기선 오프셋 복원 메커니즘 확인.
    """
    data_dir = Path(data_dir)
    torch.manual_seed(seed)
    all_trials = discover_trials(data_dir)
    val_set = set(val_trials)
    train_trials = [t for t in all_trials if t not in val_set]
    if not train_trials:
        raise ValueError("train trial 0개")

    train_raw = np.concatenate([
        load_bending_trial(data_dir / f"{t}.npz", valid_only=valid_only).sensor for t in train_trials])
    stats = compute_norm_stats(train_raw)
    Xtr, dtr, btr = _restorer_windows(data_dir, train_trials, cfg, stats, valid_only)
    Xva, dva, bva = _restorer_windows(data_dir, val_trials, cfg, stats, valid_only)

    dev = cfg.device if torch.cuda.is_available() else "cpu"
    model = BaselineRestorer(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tX, td, tb = (torch.from_numpy(a).to(dev) for a in (Xtr, dtr, btr))
    vX, vd, vb = (torch.from_numpy(a).to(dev) for a in (Xva, dva, bva))

    def offset_reduction(X, d, b) -> float:
        with torch.no_grad():
            restored = model(X, d)                                  # [B,W,16]
            tgt = b[:, None, :].expand(-1, X.shape[1], -1)          # [B,W,16]
            before = (X - tgt).abs().mean().item()
            after = (restored - tgt).abs().mean().item()
        return 1.0 - after / before if before > 0 else 0.0

    history, best = [], {"val_red": -1.0, "state": None}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tX), device=dev)
        for i in range(0, len(tX), batch_size):
            bi = perm[i:i + batch_size]
            restored = model(tX[bi], td[bi])
            tgt = tb[bi][:, None, :].expand(-1, cfg.window_size, -1)
            loss = nn.functional.mse_loss(restored, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        tr_red, va_red = offset_reduction(tX, td, tb), offset_reduction(vX, vd, vb)
        history.append({"epoch": ep + 1, "train_reduction": tr_red, "val_reduction": va_red})
        if va_red > best["val_red"]:
            best = {"val_red": va_red, "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"ep{ep + 1:3d}  train_offset제거={tr_red * 100:5.1f}%  val_offset제거={va_red * 100:5.1f}%")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"model": model.eval(), "stats": stats, "history": history,
            "val_reduction": best["val_red"], "train_trials": train_trials, "val_trials": list(val_trials)}


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="밴딩 모듈 학습 드라이버 (Phase 1 estimator / Phase 2 restorer).")
    p.add_argument("--phase", choices=["estimator", "restorer"], default="estimator")
    p.add_argument("--data-dir", type=Path, default=repo / "learning_data/bending/v0")
    p.add_argument("--val-trials", nargs="+", required=True,
                   help="홀드아웃 trial 이름(확장자 제외), 예: 20260727_test4")
    p.add_argument("--out", type=Path, default=None,
                   help="체크포인트 경로. 미지정 시 phase별 기본값.")
    p.add_argument("--valid-only", action=argparse.BooleanOptionalAction, default=True,
                   help="npz valid 마스크(δ≤10mm, 포화 이전)만 사용 (기본 켬).")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--estimator-arch", choices=["lstm", "cnn2d", "cnn2d_shape", "cnn_lstm", "mlp_frame"], default="lstm",
                   help="곡률 estimator 구조")
    p.add_argument("--restorer-mode", choices=["deg_only", "seq_deg", "deg_cnn"], default="deg_only",
                   help="restorer 구조(deg_cnn=곡률→4×4 공간 CNN 오프셋)")
    args = p.parse_args()

    cfg = BendingConfig(estimator_arch=args.estimator_arch, restorer_mode=args.restorer_mode)
    if args.phase == "estimator":
        out = args.out or repo / "sats/bending/runs/estimator_v0/best.pt"
        result = train_estimator_holdout(
            cfg, args.data_dir, val_trials=args.val_trials, valid_only=args.valid_only,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
        print(f"[estimator] val_MAE={result['val_mae']:.2f}° (기준선 {result['baseline_mae']:.2f}°) "
              f"| train={result['train_trials']} val={result['val_trials']}")
        save_estimator(result, cfg, out)
    else:
        out = args.out or repo / "sats/bending/runs/restorer_v0/best.pt"
        result = train_restorer_holdout(
            cfg, args.data_dir, val_trials=args.val_trials, valid_only=args.valid_only,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
        print(f"[restorer] val 오프셋 제거율={result['val_reduction'] * 100:.1f}% "
              f"| train={result['train_trials']} val={result['val_trials']}")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": result["model"].state_dict(),
                    "norm_mean": result["stats"].mean, "norm_std": result["stats"].std,
                    "cfg": asdict(cfg), "val_reduction": result["val_reduction"],
                    "train_trials": result["train_trials"], "val_trials": result["val_trials"]}, out)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
