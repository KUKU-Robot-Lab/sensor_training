"""Fig.2D 소재 비교 — map 품질(loc·peak상관) 버전 (2026-07-20).

기존 Fig2D_B_material_compare 는 d5_rel/d10_rel 을 썼으나:
  - d5 는 무접촉(eco20 d5 fz>0.3N 0%) → 소재 비교 불가
  - d10_rel 은 저force 분모 왜곡
→ rel 대신 **위치(loc)·형태(peak 상관)** 로 소재 서열을 정직하게 표시.
데이터 = `experiments_archive/reeval/map_quality.csv` (reeval_map_quality.py 산출).

실행: .venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_material_mapquality.py
산출: fig2_material_ablation/panelD_sats/xy1_material/Fig2D_B_mapquality.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
CSV = REPO / "history/fig_data/experiments_archive/reeval/map_quality.csv"
OUT = REPO / "history/fig_data/fig2_material_ablation/panelD_sats/xy1_material/Fig2D_B_mapquality.png"

MATS = ["ecomesh xy1", "eco20 xy1", "eco50 xy1"]
LABELS = {"ecomesh xy1": "Eco-mesh", "eco20 xy1": "Eco20", "eco50 xy1": "Eco50"}
COLORS = {"ecomesh xy1": "#2ca25f", "eco20 xy1": "#e07b39", "eco50 xy1": "#5b8def"}


def load() -> dict:
    rows = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows[(r["model"], r["diam"])] = r
    return rows


def main() -> None:
    rows = load()
    fig, (axL, axC) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    x = np.arange(len(MATS))

    # d10 만 사용(d5 는 무접촉이라 소재 비교 불가) — peak 상관은 있으면 d5 도 점으로
    loc = [float(rows[(m, "d10")]["loc_med_mm"]) for m in MATS]
    corr = [float(rows[(m, "d10")]["peak_corr"]) for m in MATS]
    cols = [COLORS[m] for m in MATS]

    axL.bar(x, loc, color=cols)
    for i, v in enumerate(loc):
        axL.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    axL.set_xticks(x); axL.set_xticklabels([LABELS[m] for m in MATS])
    axL.set_ylabel("localization error (mm)  (lower=better)")
    axL.set_title("d10 localization (loc)")
    axL.grid(axis="y", ls=":", alpha=0.4)

    axC.bar(x, corr, color=cols)
    for i, v in enumerate(corr):
        axC.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    axC.set_xticks(x); axC.set_xticklabels([LABELS[m] for m in MATS])
    axC.set_ylabel("GT vs pred peak correlation  (higher=better)")
    axC.set_ylim(0, 1.05)
    axC.set_title("d10 shape/intensity (peak corr)")
    axC.grid(axis="y", ls=":", alpha=0.4)

    fig.suptitle("Fig.2D  Material SATS map quality (d10; rel-artifact free):  Eco-mesh > Eco20 > Eco50", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005, "d5 omitted: near-zero contact (eco20 d5 fz>0.3N = 0%). Source: reeval/map_quality.csv",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print("saved:", OUT)
    for m in MATS:
        print(f"  {LABELS[m]:10s} d10 loc={rows[(m,'d10')]['loc_med_mm']}mm "
              f"peak_corr={rows[(m,'d10')]['peak_corr']}")


if __name__ == "__main__":
    main()
