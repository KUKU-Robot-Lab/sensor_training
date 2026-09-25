"""Fig.2 — 6개 메트릭 육각형 레이더 차트 (소재 3종 겹쳐 비교).

Fig2_d5d10_comparison.png 의 6개 메트릭(Half-max 폭·σ·active·Total|ΔS|·prop-σ·entropy)을
육각형 6축에 펼쳐, eco20/eco50/mesh 를 연한 색 다각형으로 겹쳐 그린다.

축마다 단위가 다르므로(mm/%/개수/무차원) **각 축을 그 인덴터의 3소재 최댓값으로 정규화(0~1)**.
→ 다각형이 클수록 그 축에서 우세. 6축 전부 "클수록 SR 에 유리/민감" 방향이라
  **다각형 면적 ≈ SR 적합도**로 직관적으로 읽힌다. 꼭짓점에 실제 원시값 표기.

출력: Analysis_Results/Fig2_radar.png  (d5·d10 두 육각형)
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
# 연한 핑크/파랑/초록 (fill = 연한색, edge = 진한색)
STYLE = {
    "eco20":   {"face": "#bcd4ec", "edge": "#3b6ea5", "label": "eco20"},      # 연한 파랑
    "eco50":   {"face": "#f8c0d4", "edge": "#d6588f", "label": "eco50"},      # 연한 핑크
    "ecomesh": {"face": "#b4e3bd", "edge": "#2e8b57", "label": "mesh20"},     # 연한 초록
}
# (CSV 열, 표시축 이름, 값포맷)
METRICS = [
    ("halfmax_diam_mm",    "Half-max\nreceptive width (mm)", "{:.1f}"),
    ("sigma_mm",           "Receptive-field\nσ (mm)",        "{:.2f}"),
    ("active_cells",       "Active cells\n(receptive)",      "{:.0f}"),
    ("Total |ΔS| (%)",     "Total |ΔS|\n(%)",                "{:.0f}"),
    ("Propagation σ (mm)", "Propagation\nσ (mm)",            "{:.2f}"),
    ("Entropy H_norm",     "Entropy\nH_norm",                "{:.2f}"),
]
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})


def load(dia):
    b = pd.read_csv(os.path.join(ROOT, dia, "Fig2B_metrics.csv"), index_col=0)
    c = pd.read_csv(os.path.join(ROOT, dia, "Fig2C_metrics.csv"), index_col=0)
    return b.join(c)


def draw(ax, dia, vmax):
    df = load(dia)
    cols = [m[0] for m in METRICS]
    raw = {n: np.array([df.loc[n, c] for c in cols], dtype=float) for n in ORDER}

    N = len(METRICS)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)  # offset 은 set_theta_offset 가 처리
    ang_closed = np.concatenate([ang, ang[:1]])

    ax.set_theta_offset(np.pi / 2)      # 첫 축을 위로
    ax.set_theta_direction(-1)          # 시계방향
    for n in ORDER:
        vals = raw[n] / vmax
        vc = np.concatenate([vals, vals[:1]])
        st = STYLE[n]
        ax.plot(ang_closed, vc, color=st["edge"], lw=2.2, zorder=3)
        ax.fill(ang_closed, vc, color=st["face"], alpha=0.4, zorder=2)
        ax.scatter(ang, vals, color=st["edge"], s=24, zorder=4)
        for k, (a, v) in enumerate(zip(ang, vals)):
            ax.text(a, v + 0.07, METRICS[k][2].format(raw[n][k]),
                    color=st["edge"], fontsize=7.5, ha="center", va="center",
                    fontweight="bold", zorder=5)

    ax.set_xticks(ang)
    ax.set_xticklabels([m[1] for m in METRICS], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "50%", "", "max"], fontsize=7, color="#aaa")
    ax.set_rlabel_position(0)
    ax.tick_params(axis="x", pad=14)
    ax.grid(color="#cccccc", alpha=0.6)
    ax.set_title(f"{dia}  ({'⌀5 mm — sharp' if dia=='d5' else '⌀10 mm — SR demo'})",
                 fontsize=13, fontweight="bold", pad=28)
    ax.spines["polar"].set_color("#cccccc")


def main():
    # ★ d5·d10 통합 축별 최댓값 → 두 그림을 같은 스케일로 정규화(절대 크기 차이 보존)
    cols = [m[0] for m in METRICS]
    gmax = np.zeros(len(cols))
    for dia in ["d5", "d10"]:
        df = load(dia)
        for k, c in enumerate(cols):
            gmax[k] = max(gmax[k], max(float(df.loc[n, c]) for n in ORDER))
    gmax[gmax == 0] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6), subplot_kw=dict(polar=True))
    for ax, dia in zip(axes, ["d5", "d10"]):
        draw(ax, dia, gmax)

    handles = [Patch(facecolor=STYLE[n]["face"], edgecolor=STYLE[n]["edge"],
                     lw=2, label=STYLE[n]["label"]) for n in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Fig. 2  Material radar — 6 metrics (eco20 / eco50 / mesh20)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.text(0.5, -0.07,
             "Both charts share ONE scale per axis (max across d5 AND d10), so the d5 hexagon is genuinely smaller — "
             "d10 (⌀10 mm) gives far larger absolute response on every axis (e.g. Total |ΔS| 25→114%, active 22→50). "
             "All 6 axes point 'larger = wider spread / more sensitive' → polygon AREA ≈ SR-favorability; vertex numbers = raw values. "
             "Within d10, mesh20 is the largest, most balanced hexagon.",
             ha="center", va="top", fontsize=9.4, color="#555")

    out = os.path.join(ROOT, "Fig2_radar.png")
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
