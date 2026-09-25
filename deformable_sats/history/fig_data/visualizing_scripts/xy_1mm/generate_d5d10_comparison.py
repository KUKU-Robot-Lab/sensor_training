"""Fig.2 — d5 vs d10 통합 비교 (소재 × 인덴터 한 장).

Fig2B_metrics.csv(수용장: peak·σ·active) + Fig2C_metrics.csv(배열: Total|ΔS|·prop σ·entropy)
를 d5/d10 모두 읽어, 메트릭별 그룹막대로 한 그림에 표시.
  - x축 = 소재(eco20/eco50/ecomesh)  → 소재별 차이
  - 막대 hue = 인덴터(d5 빗금 / d10 채움)  → d5/d10 차이
  - 모든 막대에 수치 라벨 → 스케일 압축돼도 차이 판독 가능
출력: Analysis_Results/Fig2_d5d10_comparison.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import generate_2d_heatmap as g2
ROOT = g2.OUT_ROOT   # 출력 일원화: fig2_material_ablation/Analysis_Results
ORDER = ["eco20", "eco50", "ecomesh"]
COLORS = {"eco20": "#3b6ea5", "eco50": "#e08a1e", "ecomesh": "#2e8b57"}

matplotlib.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans",
                            "axes.unicode_minus": False})


def load(dia):
    b = pd.read_csv(os.path.join(ROOT, dia, "Fig2B_metrics.csv"), index_col=0)
    c = pd.read_csv(os.path.join(ROOT, dia, "Fig2C_metrics.csv"), index_col=0)
    return b.join(c)


D = {"d5": load("d5"), "d10": load("d10")}

# (열이름, 표시제목, 값포맷, 출처)
METRICS = [
    ("halfmax_diam_mm", "Mean receptive width (mm)\n(per-taxel, central 4)", "{:.1f}", "receptive"),
    ("sigma_mm", "Receptive-field σ (mm)", "{:.2f}", "receptive"),
    ("active_cells", "Active cells (receptive)", "{:.0f}", "receptive"),
    ("Total |ΔS| (%)", "Total |ΔS| (%)", "{:.1f}", "array"),
    ("Propagation σ (mm)", "Propagation σ (mm)", "{:.2f}", "array"),
    ("Entropy H_norm", "Entropy H_norm", "{:.3f}", "array"),
]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), constrained_layout=True)
    x = np.arange(len(ORDER))
    w = 0.38
    for ax, (col, title, fmt, src) in zip(axes.ravel(), METRICS):
        for k, (dia, off, hatch, alpha) in enumerate(
                [("d5", -w/2, "////", 0.62), ("d10", w/2, None, 1.0)]):
            vals = [D[dia].loc[m, col] for m in ORDER]
            bars = ax.bar(x + off, vals, w, color=[COLORS[m] for m in ORDER],
                          alpha=alpha, hatch=hatch, edgecolor="white", linewidth=0.8)
            for b, v, m in zip(bars, vals, ORDER):
                ax.text(b.get_x() + b.get_width()/2, b.get_height(), fmt.format(v),
                        ha="center", va="bottom", fontsize=8.5, rotation=0)
        # d5/d10 그룹별 소재 단조 추세선
        for dia, off, ls in [("d5", -w/2, "--"), ("d10", w/2, "-")]:
            vals = [D[dia].loc[m, col] for m in ORDER]
            ax.plot(x + off, vals, ls, color="#444", lw=1.0, alpha=0.55, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(ORDER, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.margins(y=0.16)

    # 범례 (인덴터 = 막대 패턴; 소재는 색 = x라벨)
    ind_handles = [
        Patch(facecolor="#9aa6b4", hatch="////", edgecolor="white", label="d5  (∅5 mm — sharp, material contrast)"),
        Patch(facecolor="#5a6675", label="d10  (∅10 mm — saturated)"),
        Patch(facecolor=COLORS["eco20"], label="eco20"),
        Patch(facecolor=COLORS["eco50"], label="eco50"),
        Patch(facecolor=COLORS["ecomesh"], label="ecomesh"),
    ]
    fig.legend(handles=ind_handles, loc="lower center", bbox_to_anchor=(0.5, -0.055),
               ncol=5, frameon=False, fontsize=10.5)

    fig.suptitle("Fig. 2  Material × indenter metric comparison  —  d5 vs d10",
                 fontsize=16, fontweight="bold", y=1.045)
    fig.text(0.5, -0.092,
             "Bars: color = material, pattern = indenter (hatched d5 / solid d10); dashed/solid line = material trend.  "
             "Signal uses a PER-PRESS local baseline (sensor values just before each contact) to remove scan drift; |ΔS| with a 0.5% floor; σ thresholded.  "
             "Receptive width = per-taxel half-max diameter (≥50% of each taxel's own peak) averaged over the 4 CENTRAL taxels (edge taxels truncated by the ±10 mm scan). "
             "Dotted line = taxel pitch 6.5 mm: a receptive field reaching it means neighbouring fields OVERLAP (super-resolution-enabling). "
             "d10 eco50 6.6 ≈ mesh 6.3 ≈ pitch (overlap) while eco20 5.1 < pitch (undersampling); eco20 narrowest at both indenters.  "
             "eco20 is lowest on every metric (localized/undersampling). At d10 ecomesh leads ACTIVE-taxel count (6.9) and keeps eco50-class sensitivity, "
             "but σ_prop·entropy are on par with eco50 (3.58/0.340 ≈ 3.61/0.349) — the earlier 'mesh widest+most uniform' gap was drift inflation, now removed.",
             ha="center", va="top", fontsize=9.2, color="#555")

    out = os.path.join(ROOT, "Fig2_d5d10_comparison.png")
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
