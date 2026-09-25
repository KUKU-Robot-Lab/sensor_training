#!/usr/bin/env python3
"""(A) 크기 입력 가설 검증: size-input 모델 vs datarich(크기 무입력).

동일 홀드아웃(d5 test10 + d10 test3)에서 두 모델을 force구간·인덴터별로 비교.
가설: 크기 입력이 d10 blend를 풀어 d10 pedestal(+2.0)↓·peak비율→1, d5는 유지.

두 모델 모두 datarich cfg의 val 로더/GT로 평가(공정). size 모델만 forward에 지름 주입.
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
SIZEIN = REPO / "sats/training/runs/size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3"
OUT = REPO / "history/fig_data/experiments_archive/pool_diag/compare_sizeinput_vs_datarich.png"
FORCE_BINS = [(0.1, 0.4), (0.4, 0.8), (0.8, 1.3), (1.3, 2.2)]


def collect(model, cfg, tgen, val, device) -> dict:
    use_size = bool(getattr(cfg, "use_indenter_size_input", False))
    rec = {k: {"gp": [], "pp": [], "fz": []} for k in ("d5", "d10")}
    with torch.no_grad():
        for sb, mb, ln in val:
            sb, mb, ln = sb.to(device), mb.to(device), ln.to(device)
            g = tgen(mb).cpu().numpy()
            size = mb[:, 0] if use_size else None
            p = model(sb, ln, size)[0].cpu().numpy()
            dia = mb[:, 0].cpu().numpy()
            fz = mb[:, 4].cpu().numpy()
            for i in range(len(dia)):
                k = "d5" if dia[i] < 7.5 else "d10"
                rec[k]["gp"].append(float(g[i].max()))
                rec[k]["pp"].append(float(p[i].max()))
                rec[k]["fz"].append(float(fz[i]))
    return {k: {kk: np.array(vv) for kk, vv in d.items()} for k, d in rec.items()}


def binned(d: dict) -> list:
    out = []
    for lo, hi in FORCE_BINS:
        m = (d["fz"] >= lo) & (d["fz"] < hi) & (d["gp"] > 1e-6)
        if m.sum() < 10:
            out.append(None)
            continue
        out.append({
            "gt": float(np.median(d["gp"][m])),
            "pred": float(np.median(d["pp"][m])),
            "abs": float(np.median(d["pp"][m] - d["gp"][m])),
            "ratio": float(np.median(d["pp"][m] / np.maximum(d["gp"][m], 1e-6))),
        })
    return out


def main() -> None:
    cfg = load_cfg(DATARICH)
    device = cfg.effective_device()
    _, val = build_dataloaders(cfg)
    tgen = BatchGPUTargetGenerator(cfg, device)
    cfg_s = load_cfg(SIZEIN)

    m_dr = _load_model(DATARICH, cfg, device)
    m_s = _load_model(SIZEIN, cfg_s, device)
    dr = collect(m_dr, cfg, tgen, val, device)
    si = collect(m_s, cfg_s, tgen, val, device)

    centers = [(lo + hi) / 2 for lo, hi in FORCE_BINS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, ind in zip(axes, ("d10", "d5")):
        for res, lab, c in [(dr, "datarich (no size)", "#8856a7"),
                            (si, "size-input", "#e6550d")]:
            b = binned(res[ind])
            xs = [centers[i] for i, v in enumerate(b) if v]
            ys = [v["ratio"] for v in b if v]
            ax.plot(xs, ys, "o-", color=c, label=lab, lw=2, ms=7)
        ax.axhline(1.0, color="0.4", ls="--", lw=1)
        ax.set_title(f"{ind} peak ratio (pred/gt) vs force")
        ax.set_xlabel("fz (N)")
        ax.set_ylabel("peak ratio")
        ax.legend(fontsize=8)
        ax.grid(ls=":", alpha=0.4)
    fig.suptitle("(A) indenter-size input vs datarich — d10 pedestal fix? d5 preserved?",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print("saved:", OUT)

    for ind in ("d10", "d5"):
        bd, bs = binned(dr[ind]), binned(si[ind])
        print(f"\n=== {ind}  (GTpk | datarich pred/abs/ratio | size pred/abs/ratio) ===")
        for i, (lo, hi) in enumerate(FORCE_BINS):
            if not bd[i] or not bs[i]:
                continue
            print(f" {lo:.1f}-{hi:.1f}N  gt={bd[i]['gt']:.2f} | "
                  f"dr {bd[i]['pred']:.2f}/{bd[i]['abs']:+.2f}/{bd[i]['ratio']:.2f} | "
                  f"si {bs[i]['pred']:.2f}/{bs[i]['abs']:+.2f}/{bs[i]['ratio']:.2f}")


if __name__ == "__main__":
    main()
