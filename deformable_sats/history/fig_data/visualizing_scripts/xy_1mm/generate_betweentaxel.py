"""Fig.2 — between-taxel press: 수용장 overlap 비교 (왜 mesh 가 SR 에 유리한가).

(0,0) center press 는 4개 대각 taxel 에서 멀어(4.6mm) 약하고 비대칭이었다. 대신
**두 인접 taxel 의 중점**(각 taxel 에서 3.25mm 등거리)을 누르면, SR 의 핵심 시나리오인
"물리 taxel 사이의 접촉"을 직접 본다: 양쪽 taxel 이 고르게 같이 반응하면 모델이
그 사이 위치를 보간(super-resolution)할 수 있다.

인접쌍 4개 중점에서 비교:
  - overlap evenness = min(a,b)/max(a,b)  (1=완전 대칭, 둘이 똑같이 반응)
  - weaker-taxel |ΔS| = min(a,b)          (약한 쪽도 반응해야 overlap 성립)
  - pair-sum |ΔS|     = a+b               (절대 민감도)
신호: |ΔS| 절댓값, 국소 drift 제거(=press-induced), 깊이=max(그 점 최대 압입).

출력: Analysis_Results/Fig2_betweentaxel_{d5,d10}.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import generate_2d_heatmap as g2

ORDER = ["eco20", "eco50", "ecomesh"]
LABEL = {"eco20": "eco20", "eco50": "eco50", "ecomesh": "ecomesh (mesh20)"}
COLORS = {"eco20": "#3b6ea5", "eco50": "#e08a1e", "ecomesh": "#2e8b57"}
SK = g2.SKIN_COLS
DS = [f"dS_{c}" for c in SK]
SENSOR_POS = np.array([g2.SENSOR_XY[c] for c in SK])
# 인접 taxel 쌍: (중점 (x,y), taxelA, taxelB)
PAIRS = [((0, -3.25), 6, 7), ((0, 3.25), 10, 11), ((-3.25, 0), 6, 10), ((3.25, 0), 7, 11)]
DISPLAY = ((0, 3.25), 10, 11)   # 맵으로 보여줄 대표 쌍 (좌우 — eco20 쏠림 잘 드러남)
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})


def to_grid(v):
    g = np.full((4, 4), np.nan)
    for i in range(16):
        r, c = divmod(i, 4)
        g[r, c] = v[i]
    return g


def rep_at(df, px, py):
    """press 점 (px,py) 의 max-depth 대표 16-vector (top-k 평균 − 국소 drift)."""
    c = df[((df.x_mm - px).abs() < 0.7) & ((df.y_mm - py).abs() < 0.7)].copy()
    if len(c) == 0:
        return None
    tot = c[DS].abs().sum(axis=1)
    drift = c[tot <= tot.quantile(0.30)][DS].mean().values
    cc = c[c.z_mm > 5]
    zmax = cc.z_mm.max() if len(cc) else c.z_mm.max()
    w = c[c.z_mm >= zmax - 0.6]
    if len(w) == 0:
        w = c
    t = w[DS].abs().sum(axis=1).values
    k = min(10, len(w))
    idx = np.argsort(t)[-k:]
    return w[DS].iloc[idx].mean().values - drift


def main(dia):
    disp, met = {}, {}
    for name in ORDER:
        df, _ = g2.load_material(g2.DATASETS[dia][name])
        # 대표 맵 (display 쌍)
        (dx, dy), _, _ = DISPLAY
        disp[name] = rep_at(df, dx, dy)
        # 4쌍 평균 메트릭
        ev, mn, sm = [], [], []
        for (px, py), ta, tb in PAIRS:
            v = rep_at(df, px, py)
            if v is None:
                continue
            a, b = abs(v[ta - 1]), abs(v[tb - 1])
            ev.append(min(a, b) / max(a, b, 1e-3)); mn.append(min(a, b)); sm.append(a + b)
        met[name] = (np.mean(ev), np.mean(mn), np.mean(sm))

    fig = plt.figure(figsize=(13, 9.5))
    gs = GridSpec(2, 3, height_ratios=[1.15, 0.85], hspace=0.42, wspace=0.28,
                  left=0.07, right=0.91, top=0.88, bottom=0.1)

    (dx, dy), TA, TB = DISPLAY
    last_im = None
    for ci, name in enumerate(ORDER):
        ax = fig.add_subplot(gs[0, ci])
        vec = disp[name]
        a = np.abs(vec)
        peak = a.max() if a.max() > 1e-6 else 1.0
        gridn = np.abs(to_grid(vec)) / peak
        last_im = ax.imshow(gridn, origin="lower", cmap="magma", vmin=0, vmax=1,
                            extent=[-13, 13, -13, 13], interpolation="nearest")
        for i in range(16):
            r, cc = divmod(i, 4)
            sx, sy = SENSOR_POS[i]
            ax.text(sx, sy, f"{a[i]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if gridn[r, cc] < 0.6 else "black")
        # display 쌍 두 taxel 박스 + press 점
        for t in (TA, TB):
            sx, sy = SENSOR_POS[t - 1]
            ax.add_patch(plt.Rectangle((sx - 3.0, sy - 3.0), 6.0, 6.0, fill=False,
                                       ec="#39d0ff", lw=1.8))
        eA, eB = abs(vec[TA - 1]), abs(vec[TB - 1])
        even = min(eA, eB) / max(eA, eB, 1e-3)
        ax.set_title(f"{LABEL[name]}\nSkin{TA}|{TB} = {eA:.1f} | {eB:.1f}%   even {even:.2f}",
                     fontsize=10.5)
        ax.set_xticks([-9.75, -3.25, 3.25, 9.75]); ax.set_yticks([-9.75, -3.25, 3.25, 9.75])
        ax.tick_params(labelsize=7); ax.set_xlabel("sensor x (mm)", fontsize=8)
        if ci == 0:
            ax.set_ylabel("press between two taxels\n\nsensor y (mm)", fontsize=10)

    cb = fig.colorbar(last_im, ax=fig.axes[:3], fraction=0.016, pad=0.015)
    cb.set_label("fraction of own |peak| (shape)", fontsize=9)

    # ── 4쌍 평균 메트릭 막대 ──
    evens = [met[n][0] for n in ORDER]
    mins = [met[n][1] for n in ORDER]
    sums = [met[n][2] for n in ORDER]
    ax1 = fig.add_subplot(gs[1, 0]); ax2 = fig.add_subplot(gs[1, 1]); ax3 = fig.add_subplot(gs[1, 2])
    panels = [(ax1, evens, "Overlap evenness  min/max", "", "{:.2f}", "#2e8b57"),
              (ax2, mins, "Weaker taxel |ΔS|  (overlap)", "%", "{:.1f}", "#2e8b57"),
              (ax3, sums, "Pair-sum |ΔS|  (sensitivity)", "%", "{:.1f}", "#888")]
    for ax, vals, ttl, ylab, fmt, tc in panels:
        bars = ax.bar(ORDER, vals, color=[COLORS[n] for n in ORDER], alpha=0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(ttl + "  (mean of 4 pairs)", fontsize=9.5, color=tc, fontweight="bold")
        ax.set_ylabel(ylab); ax.grid(axis="y", alpha=0.3); ax.margins(y=0.2)
        ax.tick_params(axis="x", labelsize=9)
    if evens[2] == max(evens):
        ax1.annotate("mesh ↑ (evenest)", xy=(2, evens[2]), ha="center", va="bottom",
                     fontsize=9.5, color="#2e8b57", fontweight="bold", xytext=(2, max(evens) * 1.04))

    fig.suptitle(f"Fig. 2  Between-taxel press — receptive-field overlap  ({dia})",
                 fontsize=14.5, fontweight="bold", y=0.965)
    fig.text(0.5, 0.018,
             "Press at the midpoint between two adjacent taxels (3.25 mm from each) — the super-resolution scenario. "
             "Maps normalized to each material's own |peak| (cyan boxes = the two flanking taxels). "
             "Why mesh: both taxels respond EVENLY (overlap → model can interpolate the in-between position) — "
             "eco20 is lopsided (undersampling), mesh is most even; sensitivity (pair-sum) is the separate gray bar.",
             ha="center", va="bottom", fontsize=8.6, color="#555")

    out = os.path.join(g2.OUT_ROOT, f"Fig2_betweentaxel_{dia}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")
    for n in ORDER:
        e, m, s = met[n]
        print(f"   {dia} {n:8s} evenness={e:.2f} weaker={m:.1f}% sum={s:.1f}%")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--diameter", choices=["d5", "d10", "all"], default="all")
    a = ap.parse_args()
    for dia in (["d5", "d10"] if a.diameter == "all" else [a.diameter]):
        main(dia)
