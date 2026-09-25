"""Fig.3A - 전체 학습 알고리즘 구조도 (bending-aware SR, C2).

두 단계를 한 다이어그램으로:
  ① Flat e2e 학습 — raw→flat잔차→[LSTM→Attn→LocalMap→CNN]→map ↔ Boussinesq GT(MSE).
     학습 후 SATS 코어 동결.
  ② Bending-aware SR(재학습 0) — 무접촉 baseline 이동→곡률추정→bending baseline→잔차보정
     →[동결 SATS]→bent map.

설계: sensor_training/docs/superpowers/specs/2026-06-24-bending-aware-sr-design.md
출력: 같은 폴더 Fig3A_algorithm.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# 한글 렌더링 (Windows). 없으면 순서대로 fallback.
for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = _f
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig3A_algorithm.png")

C_FLAT = "#cfe0f3"      # flat 경로
C_SATS = "#d9d2e9"      # SATS 코어
C_NEW = "#d9ead3"       # 신규 bending 모듈
C_GT = "#fce5cd"        # GT/출력
C_EDGE = "#444444"


def box(ax, x, y, w, h, text, fc, fontsize=9, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.25",
                                linewidth=1.4, edgecolor=C_EDGE, facecolor=fc, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal", wrap=True)
    return (x, y, w, h)


def arrow(ax, p0, p1, text=None, color=C_EDGE, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 lw=1.4, color=color, linestyle=ls,
                                 shrinkA=2, shrinkB=2))
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.6, text, ha="center", va="bottom", fontsize=8, color=color)


def right(b):  # 박스 오른쪽 중앙
    x, y, w, h = b; return (x + w, y + h / 2)
def left(b):
    x, y, w, h = b; return (x, y + h / 2)
def top(b):
    x, y, w, h = b; return (x + w / 2, y + h)
def bottom(b):
    x, y, w, h = b; return (x + w / 2, y)


def main():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ── ① Flat 학습 단계 (상단) ─────────────────────────────────────────
    ax.add_patch(Rectangle((1, 60), 98, 36, fill=False, ls="--", ec="#3a6ea5", lw=1.6))
    ax.text(2.5, 93.5, "① Flat e2e 학습  →  학습 후 SATS 코어 동결",
            fontsize=12, fontweight="bold", color="#3a6ea5")

    f_raw = box(ax, 3, 72, 14, 9, "raw window\n[B,T,16]", C_FLAT)
    f_res = box(ax, 21, 72, 16, 9, "flat 잔차\n(raw−baseline)/baseline×100", C_FLAT)
    f_sats = box(ax, 41, 68, 26, 17,
                 "SATS 코어\nLSTM(per-taxel) → Self-Attn(4×4 GAT)\n→ Local-Map → CNN refine",
                 C_SATS, fontsize=10, bold=True)
    f_map = box(ax, 71, 72, 13, 9, "pressure map\n[B,grid,grid]", C_GT)
    f_gt = box(ax, 71, 84, 13, 8, "Boussinesq GT\n(z_depth, Fz)", C_GT, fontsize=8)

    arrow(ax, right(f_raw), left(f_res))
    arrow(ax, right(f_res), left(f_sats), "s_norm")
    arrow(ax, right(f_sats), left(f_map))
    arrow(ax, bottom(f_gt), top(f_map), "MSE", color="#990000")

    # ── 동결 가중치 전달 ────────────────────────────────────────────────
    arrow(ax, bottom(f_sats), (54, 47), "frozen weights\n(재학습 0)", color="#7030a0", ls="--")

    # ── ② Bending-aware SR 단계 (하단) ──────────────────────────────────
    ax.add_patch(Rectangle((1, 6), 98, 38, fill=False, ls="--", ec="#38761d", lw=1.6))
    ax.text(2.5, 41, "② Bending-aware SR  (residual correction, SATS 재학습 0)",
            fontsize=12, fontweight="bold", color="#38761d")

    b_shift = box(ax, 2.5, 26, 15, 9, "무접촉 baseline 이동\nΔp = raw_noload − baseline", C_NEW, fontsize=8)
    b_est = box(ax, 21, 26, 15, 9, "CurvatureEstimator\nΔp → θ̂  (온라인 EMA)", C_NEW, fontsize=8, bold=True)
    b_base = box(ax, 39.5, 26, 15, 9, "BendingBaselineModel\nθ̂ → Δp_bend(16)", C_NEW, fontsize=8, bold=True)
    b_raw = box(ax, 21, 12, 15, 8, "raw window (bent)\n[B,T,16]", C_FLAT, fontsize=8)
    b_corr = box(ax, 58, 19, 17, 11,
                 "ResidualCorrector\n(raw−baseline−Δp_bend)\n/baseline×100", C_NEW, fontsize=8, bold=True)
    b_sats = box(ax, 79, 24, 18, 12, "동결 SATS 코어\n(①과 동일 가중치)", C_SATS, fontsize=9, bold=True)
    b_map = box(ax, 79, 10, 18, 9, "bent pressure map", C_GT, fontsize=9)

    arrow(ax, right(b_shift), left(b_est))
    arrow(ax, right(b_est), left(b_base), "θ̂")
    arrow(ax, right(b_base), (58, 27), "Δp_bend")
    arrow(ax, right(b_raw), (58, 22), "raw")
    arrow(ax, right(b_corr), left(b_sats), "s_corr")
    arrow(ax, bottom(b_sats), top(b_map))

    # 곡률 관측: 무접촉 게이팅 주석
    ax.text(10, 23.5, "(무접촉 판정 프레임에서만\nθ̂ 연속 갱신 — adaptive-baseline gating)",
            ha="center", va="top", fontsize=7, color="#38761d", style="italic")

    # 데이터 취득 후 채울 패널 안내
    ax.text(50, 2.5,
            "취득 후 추가 패널: (B) 각도별 Δbaseline vs z_i · (C) κ·θ 추정 MAE/R² · "
            "(D) flat vs bent SR 보정 전/후 RMSE/R² · (E) 추론 vs GT 맵",
            ha="center", va="center", fontsize=8.5, color="#555555")

    fig.suptitle("Fig.3A  Bending-aware Super-Resolution — 전체 학습 알고리즘 구조",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
