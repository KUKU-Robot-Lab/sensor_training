#!/usr/bin/env python3
"""S19 / Table S2 — ablation 비교. full SATS vs noLSTM/noAttention/noCNN.

입력: history/fig_data/supplementary/S19_ablation/diag_summary.csv
      (eval_diagnostics 로 4개 run 재평가한 결과).
산출: S19_ablation/S19_ablation_ecomesh.png (overall/d5/d10 상대오차 막대).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
DIR = REPO / "history/fig_data/supplementary/S19_ablation"

# run 이름 → 표시 라벨 (표시 순서 = 성능 나쁜→좋은)
LABELS = {
    "noAttention": "noAttention",
    "noLSTM": "noLSTM",
    "noCNN": "noCNN",
    "sizeA_ecomesh_xy1_fold3_e2e_g05": "SATS (full)",
}
ORDER = ["noAttention", "noLSTM", "noCNN", "sizeA_ecomesh_xy1_fold3_e2e_g05"]


def main() -> None:
    with open(DIR / "diag_summary.csv") as f:
        rows = {r["run"]: r for r in csv.DictReader(f)}

    labels = [LABELS[r] for r in ORDER]
    overall = [float(rows[r]["overall_rel_rmse"]) for r in ORDER]
    d5 = [float(rows[r]["d5_rel_rmse"]) for r in ORDER]
    d10 = [float(rows[r]["d10_rel_rmse"]) for r in ORDER]

    x = np.arange(len(labels)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    colors = ["#b0b0b0", "#b0b0b0", "#b0b0b0", "#2ca25f"]  # full 강조
    for off, vals, lab, alpha, hatch in [
        (-w, d5, "d5 rel", 0.5, None), (0.0, d10, "d10 rel", 0.8, "//"),
        (w, overall, "overall rel", 1.0, "xx"),
    ]:
        bars = ax.bar(x + off, vals, w, label=lab, color=colors, alpha=alpha,
                      edgecolor="black", hatch=hatch, linewidth=0.8)
        if lab == "overall rel":
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("relative RMSE  (rmse / target RMS)")
    # 최상단 막대(noAttention d5) 위로 여백을 둬서 주석이 막대에 가리지 않게 함
    ax.set_ylim(0, max(d5) * 1.33)
    ax.set_title("Ablation study (ecomesh, xy 1 mm) — every module contributes")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.text(0.02, 0.98, "removing attention hurts most → spatial aggregation is key",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#c0392b")
    fig.savefig(DIR / "S19_ablation_ecomesh.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved:", DIR / "S19_ablation_ecomesh.png")


if __name__ == "__main__":
    main()
