#!/usr/bin/env python3
"""Fig2D_G(구 Fig3G) — force-matched d10 소재 비교 (분모 교란 제거).

소재별 d10 상대오차는 검증셋의 force 분포(→target_rms=분모)에 좌우돼 직접 비교가 왜곡된다.
force 구간별로 나눠 '같은 force 에서' 소재를 비교하면 교란 없이 소재 우열을 볼 수 있다.
→ 결과: eco-mesh 가 전 force 구간에서 d10 최상(최저 오차). 물리 직관(mesh 우수) 확증.

입력: history/fig_data/experiments_archive/fig3_diag/samples_<run>.npz (대표 fold).
산출: history/fig_data/fig2_material_ablation/panelD_sats/xy1_material/Fig2D_G_forcematched_d10.png (2패널: 상대/절대).

사용::

    .venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_fig3_forcematched_d10.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
DIAG = REPO / "history/fig_data/experiments_archive/fig3_diag"
OUT = REPO / "history/fig_data/fig2_material_ablation/panelD_sats/xy1_material"

REP_RUN = {
    "eco20": "xy1_d5d10_eco20_xy1_fold2_e2e_g05",
    "eco50": "xy1_d5d10_eco50_xy1_fold1_e2e_g05",
    "ecomesh": "xy1_d5d10_ecomesh_xy1_fold3_e2e_g05",
}
COLOR = {"eco20": "#e07b39", "eco50": "#5b8def", "ecomesh": "#2ca25f"}
LABEL = {"eco20": "Eco20", "eco50": "Eco50", "ecomesh": "Eco-mesh"}
ORDER = ["eco20", "eco50", "ecomesh"]
FORCE_EDGES = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
MIN_N = 30


def _d10(run: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = dict(np.load(DIAG / f"samples_{run}.npz"))
    m = (~s["is_d5"]) & np.isfinite(s["rel"])
    return s["fz"][m], s["rel"][m], s["rmse"][m]


def _binned(fz, val):
    """force 구간별 중앙값 + IQR(25~75)."""
    med, lo, hi, labels = [], [], [], []
    for i in range(len(FORCE_EDGES) - 1):
        sel = (fz >= FORCE_EDGES[i]) & (fz < FORCE_EDGES[i + 1])
        if sel.sum() >= MIN_N:
            v = val[sel]
            med.append(float(np.median(v)))
            lo.append(float(np.percentile(v, 25)))
            hi.append(float(np.percentile(v, 75)))
            labels.append(f"{FORCE_EDGES[i]:.2g}–{FORCE_EDGES[i+1]:.2g}")
        else:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            labels.append(f"{FORCE_EDGES[i]:.2g}–{FORCE_EDGES[i+1]:.2g}")
    return np.array(med), np.array(lo), np.array(hi), labels


def main() -> None:
    data = {m: _d10(REP_RUN[m]) for m in ORDER}
    nbin = len(FORCE_EDGES) - 1
    x = np.arange(nbin)
    width = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for ax, metric_idx, title, ylab in [
        (axes[0], 1, "Relative RMSE (rmse / target RMS)", "relative RMSE"),
        (axes[1], 2, "Absolute RMSE  [a.u.]", "absolute RMSE"),
    ]:
        labels = None
        for gi, m in enumerate(ORDER):
            fz, rel, rm = data[m]
            val = rel if metric_idx == 1 else rm
            med, lo, hi, labels = _binned(fz, val)
            off = (gi - 1) * width
            yerr = np.vstack([med - lo, hi - med])
            ax.bar(x + off, med, width, label=LABEL[m], color=COLOR[m],
                   edgecolor="black", linewidth=0.6, alpha=0.9,
                   yerr=yerr, capsize=2, error_kw=dict(lw=0.8, alpha=0.6))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("contact force fz [N]  (force-matched)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", ls=":", alpha=0.4)
        if metric_idx == 1:
            ax.legend(frameon=False, fontsize=9, title="material (d10)")

    fig.suptitle("Fig2D(G) — force-matched d10 material comparison "
                 "(Eco-mesh best at every force; bar=median, whisker=IQR)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "Fig2D_G_forcematched_d10.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)

    # 콘솔 요약
    print("\nforce-matched d10 상대오차 중앙값:")
    hdr = "force".ljust(10) + "".join(LABEL[m].rjust(10) for m in ORDER)
    print(hdr)
    for i in range(len(FORCE_EDGES) - 1):
        row = f"{FORCE_EDGES[i]:.2g}-{FORCE_EDGES[i+1]:.2g}".ljust(10)
        for m in ORDER:
            fz, rel, _ = data[m]
            sel = (fz >= FORCE_EDGES[i]) & (fz < FORCE_EDGES[i + 1])
            row += (f"{np.median(rel[sel]):.3f}" if sel.sum() >= MIN_N else "-").rjust(10)
        print(row)


if __name__ == "__main__":
    main()
