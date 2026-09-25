#!/usr/bin/env python3
"""Phase 3 검증 — 밴딩 보정이 동결 SATS의 밴딩 환각을 억제하는가.

★ 핵심: SATS 입력은 **상대변화% = (s_raw − baseline)/baseline × 100** (dataset.py:248).
무접촉=0% → SATS 빈맵. 밴딩은 baseline 을 %로 흔들어 **가짜 접촉을 환각**(δ 클수록↑).
restorer 가 밴딩%를 flat(0%)로 되돌리면 SATS 환각이 사라져야 한다.

절차:
  1) pct = (raw−base)/base×100  (SATS 네이티브 표현)
  2) restorer(pct, deg) → restored%  (target=0%=flat), deg=bend_deg(GT), train trial 학습
  3) 홀드아웃에서 환각 억제율 = 1 − mean|SATS(restored)−SATS(0)| / mean|SATS(pct)−SATS(0)|

⚠ 밴딩-only 라 restorer 는 "밴딩%→0"만 학습 → **접촉 보존(진짜 접촉 살리기)은 검증 불가**
(bending+contact=G2 데이터 필요). 여기선 **환각 억제 방향**만 실증 = Phase 3 예비.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .baseline_restorer import BaselineRestorer
from .config import BendingConfig
from .dataset import BendingTrial, make_windows
from .pipeline import load_frozen_sats
from .train_bending import load_estimator

NOMINAL_SIZE_MM = 5.0


def sensor_pct(npz, valid_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """npz → (pct[N,16], bend_deg[N]). pct = SATS 네이티브 상대변화%."""
    m = np.asarray(npz["valid"], bool) if (valid_only and "valid" in npz) else np.ones(len(npz["sensor"]), bool)
    raw = npz["sensor"][m].astype(np.float64); base = npz["baseline"].astype(np.float64)
    pct = ((raw - base) / np.where(np.abs(base) < 1e-9, 1e-9, base)) * 100.0
    return pct.astype(np.float32), npz["bend_deg"][m].astype(np.float32)


def _sats_map(sats, seq: torch.Tensor) -> torch.Tensor:
    L = torch.full((seq.shape[0],), seq.shape[1], dtype=torch.long, device=seq.device)
    size = torch.full((seq.shape[0],), NOMINAL_SIZE_MM, device=seq.device) if getattr(sats, "use_size_input", False) else None
    with torch.no_grad():
        out = sats(seq, L, size)
    return out[0] if isinstance(out, tuple) else out


def _windows(data_dir: Path, trials: list[str], W: int
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """→ (pct 윈도우[M,W,16], gt_deg[M], |δ|[M], raw 윈도우[M,W,16]).

    pct = SATS/restorer 입력. raw = estimator(표준화) 입력용(e2e). δ = 윈도우 끝 압축량.
    """
    Xp, Xr, ds, deltas = [], [], [], []
    for t in trials:
        z = np.load(data_dir / f"{t}.npz")
        pct, deg = sensor_pct(z)
        m = np.asarray(z["valid"], bool) if "valid" in z else np.ones(len(z["sensor"]), bool)
        raw = z["sensor"][m].astype(np.float32)
        dl = np.abs(np.asarray(z["delta_mm"])[m].astype(np.float32))
        wp, d = make_windows(BendingTrial(sensor=pct, bend_deg=deg), W)
        wr, _ = make_windows(BendingTrial(sensor=raw, bend_deg=deg), W)
        if len(wp):
            Xp.append(wp); Xr.append(wr); ds.append(d); deltas.append(dl[W - 1:])
    return (np.concatenate(Xp), np.concatenate(ds), np.concatenate(deltas), np.concatenate(Xr))


def _train_pct_restorer(sats, data_dir: Path, train_trials: list[str], cfg: BendingConfig,
                        device: str, epochs: int, lr: float) -> BaselineRestorer:
    """밴딩%(+deg) → flat(0%) restorer 학습(target=0). GT deg 조건."""
    Xtr, dtr, _, _ = _windows(data_dir, train_trials, cfg.window_size)
    tX, td = torch.from_numpy(Xtr).to(device), torch.from_numpy(dtr).to(device)
    restorer = BaselineRestorer(cfg).to(device)
    opt = torch.optim.Adam(restorer.parameters(), lr=lr)
    for _ in range(epochs):
        restorer.train(); perm = torch.randperm(len(tX), device=device)
        for i in range(0, len(tX), 512):
            b = perm[i:i + 512]
            loss = nn.functional.mse_loss(restorer(tX[b], td[b]), torch.zeros_like(tX[b]))
            opt.zero_grad(); loss.backward(); opt.step()
    return restorer.eval()


def _pipeline_errors(sats, restorer, X: torch.Tensor, deg: torch.Tensor, flat_map, batch: int = 256):
    """프레임별 무보정/보정 오차: flat 기준 편차 + 맵 peak(가짜접촉 크기)."""
    du, dc, pu, pc = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = X[i:i + batch]
            mc_in = restorer(xb, deg[i:i + batch])
            mu, mc = _sats_map(sats, xb), _sats_map(sats, mc_in)
            du.append((mu - flat_map).abs().mean(dim=(1, 2)).cpu().numpy())
            dc.append((mc - flat_map).abs().mean(dim=(1, 2)).cpu().numpy())
            pu.append(mu.amax(dim=(1, 2)).cpu().numpy())
            pc.append(mc.amax(dim=(1, 2)).cpu().numpy())
    return (np.concatenate(du), np.concatenate(dc), np.concatenate(pu), np.concatenate(pc))


def verify_phase3(
    sats_run_dir: str | Path, data_dir: str | Path, val_trials: list[str],
    *, device: str = "cpu", epochs: int = 120, lr: float = 1e-3, seed: int = 42,
) -> dict:
    data_dir = Path(data_dir)
    torch.manual_seed(seed)
    sats = load_frozen_sats(sats_run_dir, device=device)
    cfg = BendingConfig()
    all_trials = sorted(p.stem for p in data_dir.glob("*.npz"))
    train_trials = [t for t in all_trials if t not in set(val_trials)]
    restorer = _train_pct_restorer(sats, data_dir, train_trials, cfg, device, epochs, lr)

    Xva, dva, _, _ = _windows(data_dir, val_trials, cfg.window_size)
    vX, vd = torch.from_numpy(Xva).to(device), torch.from_numpy(dva).to(device)
    flat_map = _sats_map(sats, torch.zeros(1, cfg.window_size, 16, device=device))[0]
    du, dc, _, _ = _pipeline_errors(sats, restorer, vX, vd, flat_map)
    return {"n": int(du.size), "val_trials": list(val_trials),
            "dev_uncorrected": float(du.mean()), "dev_corrected": float(dc.mean()),
            "hallucination_suppression": 1.0 - float(dc.mean()) / float(du.mean()) if du.mean() > 0 else 0.0}


def analyze_by_curvature(
    sats_run_dir: str | Path, data_dir: str | Path, val_trials: list[str],
    *, device: str = "cpu", epochs: int = 120, lr: float = 1e-3, seed: int = 42,
    estimator_ckpt: str | Path | None = None, figure_path: str | Path | None = None,
) -> dict:
    """bent→flat복원→SATS 파이프라인 오차를 **곡률(|δ|)별로** 분해.

    무접촉이므로 정답=flat(빈맵). 오차 = 잔여 가짜접촉. δ 빈별 무보정 vs 보정
    (flat대비 편차, 맵 peak). 반환 dict + (figure_path 지정 시) 그림 저장.

    estimator_ckpt 지정 시 **end-to-end**: restorer 조건 deg 를 GT 가 아니라
    estimator 예측값으로 사용(estimator→restorer→SATS 전체 파이프라인 실측 오차).
    """
    data_dir = Path(data_dir)
    torch.manual_seed(seed)
    sats = load_frozen_sats(sats_run_dir, device=device)
    cfg = BendingConfig()
    all_trials = sorted(p.stem for p in data_dir.glob("*.npz"))
    train_trials = [t for t in all_trials if t not in set(val_trials)]
    restorer = _train_pct_restorer(sats, data_dir, train_trials, cfg, device, epochs, lr)

    Xva, dva, delta, Xraw = _windows(data_dir, val_trials, cfg.window_size)
    vX, vd = torch.from_numpy(Xva).to(device), torch.from_numpy(dva).to(device)
    # e2e: estimator 예측 deg 로 조건(GT 대신). deg_MAE 도 보고.
    e2e = estimator_ckpt is not None
    deg_mae = None
    if e2e:
        estimator, stats = load_estimator(estimator_ckpt, device=device)
        raw_t = torch.from_numpy(Xraw).to(device)
        std_t = (raw_t - torch.tensor(stats.mean, device=device)) / torch.tensor(stats.std, device=device)
        L = torch.full((len(std_t),), cfg.window_size, dtype=torch.long, device=device)
        with torch.no_grad():
            deg_pred = estimator(std_t, L)
        deg_mae = float((deg_pred - vd).abs().mean().cpu())
        cond = deg_pred
    else:
        cond = vd
    flat_ref = _sats_map(sats, torch.zeros(1, cfg.window_size, 16, device=device))[0]
    du, dc, pu, pc = _pipeline_errors(sats, restorer, vX, cond, flat_ref)
    flat_peak = float(flat_ref.max())

    edges = np.arange(0.0, float(np.ceil(delta.max())) + 2.0, 2.0)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (delta >= lo) & (delta < hi)
        if m.sum() < 3:
            continue
        bins.append({"delta_lo": float(lo), "delta_hi": float(hi), "n": int(m.sum()),
                     "dev_uncorr": float(du[m].mean()), "dev_corr": float(dc[m].mean()),
                     "peak_uncorr": float(pu[m].mean()), "peak_corr": float(pc[m].mean())})
    mode = "end-to-end (estimator deg)" if e2e else "GT deg"
    result = {"val_trials": list(val_trials), "n": int(du.size), "flat_peak_ref": flat_peak,
              "mode": mode, "estimator_deg_mae": deg_mae,
              "overall_suppression": 1.0 - float(dc.mean()) / float(du.mean()) if du.mean() > 0 else 0.0,
              "by_curvature": bins}

    if figure_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        c = [(b["delta_lo"] + b["delta_hi"]) / 2 for b in bins]
        tag = f"end-to-end (deg MAE={deg_mae:.1f} deg)" if e2e else "GT deg"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.plot(c, [b["dev_uncorr"] for b in bins], "o-", label="uncorrected")
        ax1.plot(c, [b["dev_corr"] for b in bins], "s-", label="corrected")
        ax1.set_xlabel("|delta| (mm)  (higher = more curvature)")
        ax1.set_ylabel("deviation from flat (error)")
        ax1.set_title(f"bent -> restore -> SATS error vs curvature\n[{tag}]")
        ax1.grid(alpha=.3); ax1.legend()
        ax2.plot(c, [b["peak_uncorr"] for b in bins], "o-", label="uncorrected false-contact peak")
        ax2.plot(c, [b["peak_corr"] for b in bins], "s-", label="corrected peak")
        ax2.axhline(flat_peak, color="k", ls="--", lw=1, label=f"flat reference peak {flat_peak:.2f}")
        ax2.set_xlabel("|delta| (mm)"); ax2.set_ylabel("map peak (false-contact magnitude)")
        ax2.set_title("residual false contact vs curvature"); ax2.grid(alpha=.3); ax2.legend()
        fig.tight_layout(); fig.savefig(figure_path, dpi=120)
        result["figure"] = str(figure_path)
    return result


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Phase 3: 밴딩 보정의 SATS 환각 억제 검증(상대% 표현).")
    p.add_argument("--sats-run", type=Path,
                   default=repo / "sats/training/runs/size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3")
    p.add_argument("--data-dir", type=Path, default=repo / "learning_data/bending/v0")
    p.add_argument("--val-trials", nargs="+", default=["20260727_test4"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--analyze", action="store_true", help="곡률별 오차 분해 + 그림")
    p.add_argument("--estimator", type=Path, default=None,
                   help="estimator 체크포인트. 지정 시 e2e(예측 곡률로 조건).")
    p.add_argument("--figure", type=Path, default=None, help="그림 저장 경로(--analyze 시)")
    args = p.parse_args()
    if args.analyze:
        r = analyze_by_curvature(args.sats_run, args.data_dir, args.val_trials,
                                 device=args.device, estimator_ckpt=args.estimator,
                                 figure_path=args.figure)
        mae_s = f"  deg MAE={r['estimator_deg_mae']:.1f}°" if r["estimator_deg_mae"] is not None else ""
        print(f"[Phase3 곡률별 오차 · {r['mode']}] holdout={r['val_trials']} n={r['n']}  "
              f"flat기준 peak={r['flat_peak_ref']:.3f}  전체 억제={r['overall_suppression'] * 100:.1f}%{mae_s}")
        print(f"  {'|δ|(mm)':>10}  {'n':>4}  {'무보정편차':>9}  {'보정편차':>9}  {'무보정peak':>9}  {'보정peak':>8}")
        for b in r["by_curvature"]:
            print(f"  {b['delta_lo']:.0f}-{b['delta_hi']:.0f}mm{'':>3}  {b['n']:>4}  "
                  f"{b['dev_uncorr']:>9.4f}  {b['dev_corr']:>9.4f}  {b['peak_uncorr']:>9.3f}  {b['peak_corr']:>8.3f}")
        if r.get("figure"):
            print(f"  그림: {r['figure']}")
    else:
        r = verify_phase3(args.sats_run, args.data_dir, args.val_trials, device=args.device)
        print(f"[Phase3] holdout={r['val_trials']} n={r['n']}")
        print(f"  SATS 환각(flat 대비 편차): 무보정={r['dev_uncorrected']:.4f}  보정={r['dev_corrected']:.4f}")
        print(f"  → 밴딩 환각 억제율 {r['hallucination_suppression'] * 100:.1f}%  (접촉 보존은 G2 데이터 필요)")


if __name__ == "__main__":
    main()
