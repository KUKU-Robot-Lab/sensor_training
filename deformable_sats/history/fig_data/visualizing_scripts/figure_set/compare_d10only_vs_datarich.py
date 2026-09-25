#!/usr/bin/env python3
"""d10-only vs datarich(0p5 d5+d10) 결정 실험.

동일한 d10 test3 홀드아웃에서 두 모델의 magnitude 캘리브레이션(peak 비율=pred/gt)을
force 구간별로 비교한다.

- d10-only 가 peak 비율 → 1.0 으로 내려가면  ⇒  d5 희석(9:2 편중)이 원인.
- d10-only 도 여전히 과대예측(>>1)  ⇒  GT(순수 EHS, β 미보정) 자체가 원인.

두 모델을 **같은 val 로더**(datarich cfg 의 val=d5t10+d10t3 → dia>=7.5 필터=d10 t3)에서
평가해 완전 동일 조건으로 비교한다. 재학습·추가취득 없음.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from sats.tools.eval_diagnostics import load_cfg, _load_model  # noqa: E402
from sats.training.gt_gpu import BatchGPUTargetGenerator  # noqa: E402
from sats.training.dataset import build_dataloaders  # noqa: E402

DATARICH = REPO / "sats/training/runs/datarich_probe/ecomesh_xy0p5_datarich_val_d5test10_d10test3"
D10ONLY = REPO / "sats/training/runs/d10_only/ecomesh_xy0p5_d10only_val_test3"
OUT = REPO / "history/fig_data/experiments_archive/pool_diag/compare_d10only_vs_datarich.png"

FORCE_BINS = [(0.1, 0.4), (0.4, 0.8), (0.8, 1.3), (1.3, 2.2)]


def collect(model, tgen, val, device) -> dict:
    """d10(dia>=7.5) 샘플의 (fz, gt_peak, pred_peak, rmse, target_rms) 수집."""
    fz_l, gp_l, pp_l, rmse_l, trms_l = [], [], [], [], []
    with torch.no_grad():
        for sb, mb, ln in val:
            sb, mb, ln = sb.to(device), mb.to(device), ln.to(device)
            tgt = tgen(mb)
            pred, _ = model(sb, ln)
            dia = mb[:, 0].cpu().numpy()
            fz = mb[:, 4].cpu().numpy()
            g = tgt.cpu().numpy()
            p = pred.cpu().numpy()
            for i in np.where(dia >= 7.5)[0]:
                gp_l.append(float(g[i].max()))
                pp_l.append(float(p[i].max()))
                rmse_l.append(float(np.sqrt(((p[i] - g[i]) ** 2).mean())))
                trms_l.append(float(np.sqrt((g[i] ** 2).mean())))
                fz_l.append(float(fz[i]))
    return {
        "fz": np.array(fz_l), "gp": np.array(gp_l), "pp": np.array(pp_l),
        "rmse": np.array(rmse_l), "trms": np.array(trms_l),
    }


def binned(d: dict) -> list[dict]:
    out = []
    for lo, hi in FORCE_BINS:
        m = (d["fz"] >= lo) & (d["fz"] < hi) & (d["gp"] > 1e-6)
        if m.sum() < 5:
            out.append(None)
            continue
        ratio = d["pp"][m] / np.maximum(d["gp"][m], 1e-6)
        rel = d["rmse"][m] / np.maximum(d["trms"][m], 1e-6)
        out.append({
            "n": int(m.sum()),
            "peak_ratio": float(np.median(ratio)),
            "rel_rmse": float(np.median(rel)),
            "abs_rmse": float(np.median(d["rmse"][m])),
        })
    return out


def main() -> None:
    cfg = load_cfg(DATARICH)          # val=d5t10+d10t3, d10 필터해 사용
    device = cfg.effective_device()
    _, val = build_dataloaders(cfg)
    tgen = BatchGPUTargetGenerator(cfg, device)

    m_dr = _load_model(DATARICH, cfg, device)
    cfg_d10 = load_cfg(D10ONLY)
    m_d10 = _load_model(D10ONLY, cfg_d10, device)

    d_dr = collect(m_dr, tgen, val, device)
    d_d10 = collect(m_d10, tgen, val, device)
    print(f"d10 샘플: datarich {d_dr['fz'].size}, d10only {d_d10['fz'].size}")
    print(f"전체 peak 비율(중앙) datarich={np.median(d_dr['pp']/np.maximum(d_dr['gp'],1e-6)):.2f} "
          f"d10only={np.median(d_d10['pp']/np.maximum(d_d10['gp'],1e-6)):.2f}")

    b_dr = binned(d_dr)
    b_d10 = binned(d_d10)
    centers = [(lo + hi) / 2 for lo, hi in FORCE_BINS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    # 좌: peak 비율 vs force
    for b, lab, c in [(b_dr, "datarich (d5+d10, 9:2)", "#8856a7"),
                      (b_d10, "d10-only", "#2ca25f")]:
        xs = [centers[i] for i, v in enumerate(b) if v]
        ys = [v["peak_ratio"] for v in b if v]
        ax1.plot(xs, ys, "o-", color=c, label=lab, lw=2, ms=7)
    ax1.axhline(1.0, color="0.4", ls="--", lw=1, label="calibrated (=1.0)")
    ax1.set_xlabel("fz (N)")
    ax1.set_ylabel("peak ratio (pred / gt)")
    ax1.set_title("d10 magnitude calibration vs force")
    ax1.legend(fontsize=8)
    ax1.grid(ls=":", alpha=0.4)

    # 우: 상대오차 vs force
    for b, lab, c in [(b_dr, "datarich", "#8856a7"), (b_d10, "d10-only", "#2ca25f")]:
        xs = [centers[i] for i, v in enumerate(b) if v]
        ys = [v["rel_rmse"] for v in b if v]
        ax2.plot(xs, ys, "s-", color=c, label=lab, lw=2, ms=7)
    ax2.set_xlabel("fz (N)")
    ax2.set_ylabel("relative RMSE (median)")
    ax2.set_title("d10 relative error vs force")
    ax2.legend(fontsize=8)
    ax2.grid(ls=":", alpha=0.4)

    fig.suptitle("d10 test3 holdout — d10-only vs datarich  "
                 "(ratio→1 means d5-dilution was the cause; still>>1 means GT/EHS cause)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print("saved:", OUT)

    # 콘솔 표
    print("\n force-bin | datarich ratio | d10only ratio | dr rel | d10 rel")
    for i, (lo, hi) in enumerate(FORCE_BINS):
        vd = b_dr[i]; vo = b_d10[i]
        if not vd or not vo:
            continue
        print(f" {lo:.1f}-{hi:.1f}N | {vd['peak_ratio']:.2f} | {vo['peak_ratio']:.2f} "
              f"| {vd['rel_rmse']:.2f} | {vo['rel_rmse']:.2f}  (n={vd['n']})")


if __name__ == "__main__":
    main()
