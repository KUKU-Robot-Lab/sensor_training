#!/usr/bin/env python3
"""3자 비교: datarich / size-only / size+β. 각 모델을 '자기 학습 GT' 기준 peak 비율로.

- datarich : 크기무입력, plain EHS GT
- size     : 크기입력,   plain EHS GT
- size+β   : 크기입력,   β(p) 물성보정 GT  (자기 GT = β-corrected)

판정: 최종 모델(size+β)이 d5·d10 모두 force 전구간 peak 비율≈1 이면 캘리브레이션 성공.
β는 고압 GT를 키우므로(저압≈1) 고force d10 개선이 주 기대효과.
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

RUNS = {
    "datarich": REPO / "sats/training/runs/datarich_probe/ecomesh_xy0p5_datarich_val_d5test10_d10test3",
    "size": REPO / "sats/training/runs/size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3",
    "size+beta": REPO / "sats/training/runs/size_beta/ecomesh_xy0p5_sizebeta_val_d5t10_d10t3",
    "size+beta_gentle": REPO / "sats/training/runs/size_beta_gentle/ecomesh_xy0p5_sizebeta_gentle_val_d5t10_d10t3",
    "size+beta_physical": REPO / "sats/training/runs/size_beta_physical/ecomesh_xy0p5_sizebeta_physical_val_d5t10_d10t3",
}
OUT = REPO / "history/fig_data/experiments_archive/pool_diag/compare_three_ABbeta.png"
FORCE_BINS = [(0.1, 0.4), (0.4, 0.8), (0.8, 1.3), (1.3, 2.2)]
COLORS = {"datarich": "#8856a7", "size": "#e6550d", "size+beta": "#2ca25f",
          "size+beta_gentle": "#1f78b4", "size+beta_physical": "#d62728"}


CAP = 250_000   # per-indenter sample cap; medians stable well below full 2.9M


def collect(model, cfg, tgen, val, device) -> dict:
    use_size = bool(getattr(cfg, "use_indenter_size_input", False))
    rec = {k: {"gp": [], "pp": [], "fz": []} for k in ("d5", "d10")}
    with torch.no_grad():
        for sb, mb, ln in val:
            sb, mb, ln = sb.to(device), mb.to(device), ln.to(device)
            g = tgen(mb)
            size = mb[:, 0] if use_size else None
            p = model(sb, ln, size)[0]
            gp = g.amax(dim=(1, 2)).cpu().numpy()
            pp = p.amax(dim=(1, 2)).cpu().numpy()
            dia = mb[:, 0].cpu().numpy(); fz = mb[:, 4].cpu().numpy()
            for k, msk in (("d5", dia < 7.5), ("d10", dia >= 7.5)):
                rec[k]["gp"].append(gp[msk]); rec[k]["pp"].append(pp[msk]); rec[k]["fz"].append(fz[msk])
            if all(sum(len(a) for a in rec[k]["fz"]) >= CAP for k in rec):
                break
    return {k: {kk: np.concatenate(vv) if vv else np.array([]) for kk, vv in d.items()}
            for k, d in rec.items()}


def binned(d: dict):
    out = []
    for lo, hi in FORCE_BINS:
        m = (d["fz"] >= lo) & (d["fz"] < hi) & (d["gp"] > 1e-6)
        out.append(None if m.sum() < 10 else {
            "ratio": float(np.median(d["pp"][m] / np.maximum(d["gp"][m], 1e-6))),
            "abs": float(np.median(d["pp"][m] - d["gp"][m])),
        })
    return out


def main() -> None:
    base_cfg = load_cfg(RUNS["datarich"])
    device = base_cfg.effective_device()
    _, val = build_dataloaders(base_cfg)

    results = {}
    for name, run in RUNS.items():
        cfg = load_cfg(run)
        tgen = BatchGPUTargetGenerator(cfg, device)   # β from that run's cfg
        model = _load_model(run, cfg, device)
        results[name] = collect(model, cfg, tgen, val, device)
        del model
        torch.cuda.empty_cache()

    centers = [(lo + hi) / 2 for lo, hi in FORCE_BINS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, ind in zip(axes, ("d10", "d5")):
        for name, res in results.items():
            b = binned(res[ind])
            xs = [centers[i] for i, v in enumerate(b) if v]
            ys = [v["ratio"] for v in b if v]
            ax.plot(xs, ys, "o-", color=COLORS[name], label=name, lw=2, ms=6)
        ax.axhline(1.0, color="0.4", ls="--", lw=1)
        ax.set_title(f"{ind} peak ratio (pred / own-GT) vs force")
        ax.set_xlabel("fz (N)"); ax.set_ylabel("peak ratio"); ax.legend(fontsize=8)
        ax.grid(ls=":", alpha=0.4)
    fig.suptitle("datarich vs size-input vs size+β  (each vs its own training GT)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print("saved:", OUT)

    for ind in ("d10", "d5"):
        print(f"\n=== {ind} peak ratio (abs_err) ===")
        bb = {n: binned(r[ind]) for n, r in results.items()}
        for i, (lo, hi) in enumerate(FORCE_BINS):
            cells = []
            for n in RUNS:
                v = bb[n][i]
                cells.append(f"{n}={v['ratio']:.2f}({v['abs']:+.2f})" if v else f"{n}=--")
            print(f" {lo:.1f}-{hi:.1f}N  " + "  ".join(cells))


if __name__ == "__main__":
    main()
