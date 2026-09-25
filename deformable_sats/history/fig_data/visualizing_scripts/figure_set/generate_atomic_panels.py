"""논문 figure set용 atomic 패널 생성 (제목 없는 깔끔한 단위 이미지).

각 패널을 개별 PNG로 렌더링해 build_figure_set_html.py 가 3×3 격자로 조립한다.
데이터 로딩/메트릭은 visualizing_scripts/xy_1mm 의 기존 모듈을 재사용한다.

출력: visualizing_scripts/figure_set/panels/*.png  (모두 제목 없음, tight, 투명 여백 최소)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

# 기존 데이터/메트릭 모듈 재사용
XY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xy_1mm")
sys.path.insert(0, XY_DIR)
import generate_2d_heatmap as g2          # noqa: E402
import generate_panelC_metrics as pc      # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panels")
os.makedirs(OUT, exist_ok=True)

ORDER = ["eco20", "eco50", "ecomesh"]
COLORS = {"eco20": "#3b6ea5", "eco50": "#e08a1e", "ecomesh": "#2e8b57"}
LABEL = {"eco20": "eco20", "eco50": "eco50", "ecomesh": "ecomesh (mesh20)"}

# 공통 폰트 (가독 + 논문풍)
matplotlib.rcParams.update({
    "font.size": 13,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.0,
    "axes.unicode_minus": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, transparent=False, facecolor="white")
    plt.close(fig)
    print(f"[saved] {name}")


# ───────────────────────── 수용장(receptive field) 계산 ─────────────────────────
def compute_rf(diameter):
    """diameter(d5/d10) 의 대표 중심 taxel 수용장 격자/메트릭/데이터 반환."""
    materials = g2.DATASETS[diameter]
    data = {}
    for name in ORDER:
        df, kg_base = g2.load_material(materials[name])
        m = g2.contact_mask(df, kg_base)
        data[name] = (df, m)
    # 대표 중심 taxel = 전 소재 합산 |ΔS| 최대
    score = {t: 0.0 for t in g2.CENTER_TAXELS}
    for name, (df, m) in data.items():
        for t in g2.CENTER_TAXELS:
            score[t] += df.loc[m, f"dS_{t}"].abs().mean()
    rep = max(score, key=score.get)
    grids, metrics = {}, {}
    for name, (df, m) in data.items():
        grids[name] = g2.build_grid(df, df[f"dS_{rep}"])
        metrics[name] = g2.receptive_metrics(grids[name])
    return rep, grids, metrics, data


# 한 번만 계산해 재사용
REP5, GRID5, MET5, DATA5 = compute_rf("d5")
REP10, GRID10, MET10, _ = compute_rf("d10")
print(f"[rep taxel] d5={REP5}  d10={REP10}")


# ───────────────────────── (a) 센서 평면도 ─────────────────────────
def panel_topview():
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.add_patch(Rectangle((-15, -15), 30, 30, facecolor="#f3ecd9",
                           edgecolor="#222", lw=1.6, zorder=0))
    # 1mm 스캔 격자
    gx = np.arange(-10, 10.1, 1.0)
    XX, YY = np.meshgrid(gx, gx)
    ax.scatter(XX, YY, s=2, c="#9bb4d4", alpha=0.55, zorder=1)
    # 16 taxel
    for name, (sx, sy) in g2.SENSOR_XY.items():
        i = int(name.replace("Skin", ""))
        ax.add_patch(FancyBboxPatch((sx - 1.05, sy - 1.05), 2.1, 2.1,
                     boxstyle="round,pad=0.02,rounding_size=0.25",
                     facecolor="#1b2a41", edgecolor="none", zorder=3))
        ax.text(sx, sy, f"{i}", color="white", ha="center", va="center",
                fontsize=8.5, zorder=4)
    # 인덴터 풋프린트 (원점 동심원)
    ax.add_patch(Circle((0, 0), 5.0, fill=False, ec="#e08a1e", lw=2.2, zorder=5))
    ax.add_patch(Circle((0, 0), 2.5, fill=False, ec="#c0392b", lw=2.2, zorder=5))
    ax.text(0, 5.6, "d10", color="#e08a1e", ha="center", fontsize=10, zorder=6)
    ax.text(0, 0, "d5", color="#c0392b", ha="center", va="center", fontsize=9, zorder=6)
    # 6.5mm 스케일
    ax.annotate("", xy=(3.25, -12.3), xytext=(-3.25, -12.3),
                arrowprops=dict(arrowstyle="<->", color="#222", lw=1.3))
    ax.text(0, -13.6, "6.5 mm pitch", ha="center", fontsize=9.5)
    ax.set_xlim(-16, 16); ax.set_ylim(-16, 16)
    ax.set_aspect("equal"); ax.set_xticks([-10, 0, 10]); ax.set_yticks([-10, 0, 10])
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    _save(fig, "fig2_a_topview.png")


# ───────────────────────── (b) 3중층 단면 ─────────────────────────
def panel_xsection():
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    W = 30
    layers = [("Bot: Ecoflex base", 0.0, 1.0, "#d9c7a3"),
              ("Mid: Ecoflex + MEMS", 1.0, 2.0, "#cfe3cf"),
              ("Top: Ecoflex + mesh", 3.0, 2.5, "#bcd3ee")]
    for label, y0, h, fc in layers:
        ax.add_patch(Rectangle((-W / 2, y0), W, h, facecolor=fc, edgecolor="#333", lw=1.3))
    # mesh 해칭
    ax.add_patch(Rectangle((-W / 2, 3.0), W, 2.5, facecolor="none",
                           edgecolor="#33619b", lw=0.8, hatch="xx"))
    # MEMS 챔버
    for sx in [-9.75, -3.25, 3.25, 9.75]:
        ax.add_patch(Rectangle((sx - 1.3, 1.55), 2.6, 0.9,
                     facecolor="#1b2a41", edgecolor="none", zorder=4))
    ax.annotate("MEMS chamber", xy=(-3.25, 2.45), xytext=(-3.0, 6.4),
                fontsize=8.5, color="#1b2a41", ha="center",
                arrowprops=dict(arrowstyle="-", color="#1b2a41", lw=0.8))
    # 인덴터
    ax.add_patch(Circle((6.5, 7.2), 1.1, facecolor="#c0392b", edgecolor="none", zorder=5))
    ax.text(8.0, 7.2, "indenter", color="#c0392b", ha="left", va="center", fontsize=8.5, zorder=6)
    ax.annotate("", xy=(6.5, 5.55), xytext=(6.5, 6.05),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2.4))
    # 층 라벨
    for label, y0, h, _fc in layers:
        ax.text(W / 2 + 0.8, y0 + h / 2, label, va="center", ha="left", fontsize=9)
    # 총 두께 화살표
    ax.annotate("", xy=(-W / 2 - 1.4, 0), xytext=(-W / 2 - 1.4, 5.5),
                arrowprops=dict(arrowstyle="<->", color="#222", lw=1.3))
    ax.text(-W / 2 - 2.0, 2.75, "5.5 mm", rotation=90, va="center", ha="right", fontsize=9.5)
    ax.set_xlim(-W / 2 - 4, W / 2 + 13); ax.set_ylim(-1.2, 9.5)
    ax.set_aspect("equal"); ax.axis("off")
    _save(fig, "fig2_b_xsection.png")


# ───────────────────────── (d,e,f) 수용장 heatmap (d5) ─────────────────────────
def panel_rf_d5():
    vmax = max(MET5[n]["peak"] for n in ORDER)
    rep_xy = g2.SENSOR_XY[REP5]
    C0, C1 = g2.CENTERS[0], g2.CENTERS[-1]
    for name in ORDER:
        fig, ax = plt.subplots(figsize=(3.7, 3.7))
        im = ax.imshow(GRID5[name], origin="lower", cmap="viridis",
                       extent=[C0, C1, C0, C1], vmin=0, vmax=vmax,
                       interpolation="bilinear", aspect="equal")
        for _t, (sx, sy) in g2.SENSOR_XY.items():
            ax.plot(sx, sy, "+", color="white", ms=6, mew=1.0, alpha=0.45)
        ax.plot(*rep_xy, "o", mfc="none", mec="#ff3b3b", ms=13, mew=1.8)
        ax.set_xlabel("indenter x (mm)"); ax.set_ylabel("indenter y (mm)")
        ax.set_xticks([-10, 0, 10]); ax.set_yticks([-10, 0, 10])
        # 좌상단 소재 + 메트릭 배지
        ax.text(0.03, 0.97, f"{LABEL[name]}", transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc=COLORS[name], ec="none", alpha=0.92))
        ax.text(0.03, 0.03,
                f"peak {MET5[name]['peak']:.0f}%\nactive {MET5[name]['active']}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="#000000", ec="none", alpha=0.35))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        _save(fig, f"fig2_rf_d5_{name}.png")


# ───────────────────────── (g) radial 감쇠 (d5) ─────────────────────────
def panel_radial():
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    for name in ORDER:
        rc, rv = MET5[name]["radial"]
        ax.plot(rc, rv, "-o", color=COLORS[name], lw=2.2, ms=5,
                label=f"{LABEL[name]}  (peak {MET5[name]['peak']:.0f}%)")
    ax.set_xlabel("distance from receptive-field centroid (mm)")
    ax.set_ylabel(f"mean ΔS of {REP5} (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, frameon=False)
    ax.set_xlim(left=0)
    _save(fig, "fig2_g_radial.png")


# ───────────────────────── (h) 정량 메트릭 (d5) ─────────────────────────
def panel_metrics_d5():
    results = {n: pc.array_metrics(DATA5[n][0][DATA5[n][1]]) for n in ORDER}
    mks = list(next(iter(results.values())).keys())
    fig, axes = plt.subplots(2, 2, figsize=(4.7, 4.5), constrained_layout=True)
    for ax, mk in zip(axes.ravel(), mks):
        means = np.array([results[n][mk][0] for n in ORDER])
        stds = np.array([results[n][mk][1] for n in ORDER])
        lower = np.minimum(stds, means)
        bars = ax.bar(ORDER, means, yerr=[lower, stds], capsize=3,
                      color=[COLORS[n] for n in ORDER], alpha=0.9)
        ax.set_title(mk, fontsize=10.5)
        ax.set_ylim(bottom=0); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8.5)
        for b, mv in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{mv:.2f}", ha="center", va="bottom", fontsize=8)
    _save(fig, "fig2_h_metrics.png")


# ───────────────────────── (i) d10 수용장 σ (단조 확산) ─────────────────────────
def panel_sigma_d10():
    """d10(포화 인덴터)의 수용장 σ — eco20<eco50<ecomesh 단조 증가.

    d5 σ는 eco20 저SNR로 inflated(신뢰 불가)되어 제외하고, 확산의 단조성은
    포화로 peak이 균등해진 d10 에서 가장 깨끗하게 드러난다.
    """
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    s10 = [MET10[n]["sigma"] for n in ORDER]
    bars = ax.bar(ORDER, s10, 0.6, color=[COLORS[n] for n in ORDER], alpha=0.92)
    for b, v in zip(bars, s10):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f} mm",
                ha="center", va="bottom", fontsize=10)
    # 단조 증가 추세선
    ax.plot(np.arange(len(ORDER)), s10, "--", color="#444", lw=1.4, zorder=3)
    ax.set_ylabel("receptive-field σ (mm),  d10")
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, max(s10) * 1.18)
    ax.annotate("spread  eco20 < eco50 < ecomesh",
                xy=(0.5, 0.93), xycoords="axes fraction", ha="center",
                fontsize=9.5, color="#2e8b57", fontweight="bold")
    _save(fig, "fig2_i_sigma.png")


if __name__ == "__main__":
    panel_topview()
    panel_xsection()
    panel_rf_d5()
    panel_radial()
    panel_metrics_d5()
    panel_sigma_d10()
    print("\n[done] atomic panels ->", OUT)
