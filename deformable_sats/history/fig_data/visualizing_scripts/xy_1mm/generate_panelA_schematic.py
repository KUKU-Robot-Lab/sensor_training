"""Fig.2A - 실험 셋업/격자 모식도 (소재 ablation).

논문 §6 패널 A. 데이터가 아닌 schematic.
  (A1) 센서 평면도: 30×30mm 본체 + 16 taxel(4×4, 6.5mm 간격) + xy 1mm 스캔 격자(±10mm)
       + 인덴터 풋프린트(원형 d5/d10) 실측 스케일
  (A2) 3중층 단면도: Top(Ecoflex+mesh 2.5mm) / Mid(MEMS embedded 2mm) / Bot(base 1mm), 총 5.5mm

출력: visualizing_scripts/xy_1mm/Analysis_Results/Fig2A_schematic.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow

import generate_2d_heatmap as g2  # SENSOR_XY, SKIN_COLS, OUT_ROOT

OUT = os.path.join(g2.OUT_ROOT, "Fig2A_schematic.png")
SENSOR_MM = 30.0       # 센서 외형 30×30
SCAN_LIM = 10.0        # ±10mm 스캔
TAXEL_MM = 2.5         # taxel 표시 사각 크기


def draw_top_view(ax):
    ax.set_aspect("equal")
    half = SENSOR_MM / 2
    # 센서 본체
    ax.add_patch(Rectangle((-half, -half), SENSOR_MM, SENSOR_MM,
                           facecolor="#f0ead6", edgecolor="black", lw=1.5, zorder=0))
    # xy 1mm 스캔 격자
    gx = np.arange(-SCAN_LIM, SCAN_LIM + 1, 1.0)
    XX, YY = np.meshgrid(gx, gx)
    ax.scatter(XX, YY, s=3, c="#9bbbd4", alpha=0.7, zorder=1, label="xy 1mm scan grid")
    # 16 taxel
    for c in g2.SKIN_COLS:
        x, y = g2.SENSOR_XY[c]
        ax.add_patch(Rectangle((x - TAXEL_MM / 2, y - TAXEL_MM / 2), TAXEL_MM, TAXEL_MM,
                               facecolor="#2c3e50", edgecolor="white", lw=0.8, zorder=3))
        ax.text(x, y, c.replace("Skin", "S"), color="white", ha="center", va="center",
                fontsize=7, zorder=4)
    # taxel 간격 치수 표기 (인접 두 taxel)
    ax.annotate("", xy=(3.25, -12.2), xytext=(-3.25, -12.2),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(0, -13.4, "6.5 mm", ha="center", va="top", fontsize=8)
    # 인덴터 풋프린트 (실측 스케일) — 원점 동심원으로 taxel 간격 대비 크기 표시
    for d, col, lab in [(10.0, "#e67e22", "d10"), (5.0, "#c0392b", "d5")]:
        ax.add_patch(Circle((0.0, 0.0), d / 2, fill=False, lw=1.8, ec=col, zorder=2,
                            label=f"indenter {lab}"))
    ax.text(0.0, 5.3, "indenter footprint\n(d5 / d10)", ha="center", va="bottom",
            fontsize=7, color="gray")

    ax.set_xlim(-half - 3, half + 3); ax.set_ylim(-half - 3, half + 3)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("(A1) Sensor top view  (30×30 mm, 16 taxels @ 6.5 mm)", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def draw_cross_section(ax):
    ax.set_aspect("equal")
    W = SENSOR_MM
    x0 = -W / 2
    # 층: (이름, y_bottom, 두께, facecolor, hatch)
    layers = [
        ("Bot: Ecoflex base", 0.0, 1.0, "#d9c9a3", None),
        ("Mid: Ecoflex + MEMS embedded", 1.0, 2.0, "#cfe0c3", None),
        ("Top: Ecoflex + mesh", 3.0, 2.5, "#bcd2e8", "xx"),
    ]
    for name, yb, th, fc, hatch in layers:
        ax.add_patch(Rectangle((x0, yb), W, th, facecolor=fc, edgecolor="black",
                               lw=1.2, hatch=hatch, zorder=1))
        ax.text(x0 + W + 0.6, yb + th / 2, f"{name}\n({th:.1f} mm)",
                va="center", ha="left", fontsize=8)
    # MEMS 칩 (mid 층, taxel x 위치 한 행)
    for x in [-9.75, -3.25, 3.25, 9.75]:
        ax.add_patch(Rectangle((x - 1.0, 1.4), 2.0, 1.2, facecolor="#2c3e50",
                               edgecolor="white", lw=0.6, zorder=2))
    ax.text(0, 2.0, "MEMS barometer", color="white", ha="center", va="center", fontsize=7, zorder=3)
    # 인덴터(구) 압입
    ax.add_patch(Circle((0, 5.5 + 2.5), 2.5, facecolor="#c0392b", alpha=0.85, zorder=4))
    ax.add_patch(FancyArrow(0, 9.2, 0, -0.9, width=0.15, head_width=0.8, head_length=0.5,
                            color="#c0392b", zorder=5))
    ax.text(0, 8.0, "indenter", color="#c0392b", ha="center", va="bottom", fontsize=8)
    # 총 두께 치수
    ax.annotate("", xy=(x0 - 1.2, 5.5), xytext=(x0 - 1.2, 0),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(x0 - 1.6, 2.75, "5.5 mm", rotation=90, va="center", ha="right", fontsize=8)

    ax.set_xlim(x0 - 6, W / 2 + 12); ax.set_ylim(-1, 11)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    ax.set_title("(A2) 3-layer cross-section  (total 5.5 mm)", fontsize=11)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    draw_top_view(axes[0])
    draw_cross_section(axes[1])
    fig.suptitle("Fig.2A  Experimental setup & sensor structure", fontsize=14, fontweight="bold")
    os.makedirs(g2.OUT_ROOT, exist_ok=True)
    fig.savefig(OUT, dpi=150); plt.close(fig)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
