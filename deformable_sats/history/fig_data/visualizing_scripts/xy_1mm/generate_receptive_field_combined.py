"""Fig.2 (supplement) — 한 xy 평면에 16 taxel 수용장을 합친 통합 receptive field.

기존 Fig2B 는 taxel 1개(Skin10) 기준이라, 여기서는 **모든 taxel 의 수용장을 한 평면에**
겹쳐 소재(eco20/eco50/ecomesh)별로 비교한다.

정의: 각 인덴터 위치 (x,y) 에서
  E(x,y) = max_t |ΔS_t|   (그 위치 압입에 가장 강하게 반응한 taxel 의 응답)
→ 16개 taxel 위치마다 봉우리가 서는 'egg-carton' 장. 봉우리 사이 골(valley)의 높이 = 인접
  taxel 수용장이 겹치는 정도. 수용장이 넓으면(mesh) 봉우리가 합쳐져 골이 높고, 좁으면(eco20)
  봉우리가 고립돼 골이 0 에 가깝다 = 수용장 중첩(overlap)을 직접 보여줌(SR 유리 근거).

출력:
  _receptive_depth/combined_<dia>_2d.png : 3소재 통합 히트맵 + 각 taxel 50% 수용장 등고선
  _receptive_depth/combined_<dia>_3d.png : 3소재 3D 표면(파동) — 봉우리=taxel, 골=중첩
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import generate_2d_heatmap as g2

ORDER = ["eco20", "eco50", "ecomesh"]
LABEL = {"eco20": "eco20", "eco50": "eco50", "ecomesh": "ecomesh (mesh20)"}
SK = g2.SKIN_COLS
OUT = os.path.join(g2.REPO, "fig2_material_ablation", "hitmap")   # hitmap 전용 폴더(fig2_material_ablation 안)
os.makedirs(OUT, exist_ok=True)
EXT = [g2.CENTERS[0], g2.CENTERS[-1], g2.CENTERS[0], g2.CENTERS[-1]]
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})


def _blur(a, sigma=1.0):
    """numpy 분리형 가우시안 블러(에지 반사 패딩). scipy 불필요."""
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2)); k /= k.sum()
    b = np.apply_along_axis(lambda m: np.convolve(np.pad(m, r, mode="edge"), k, "valid"), 0, a)
    b = np.apply_along_axis(lambda m: np.convolve(np.pad(m, r, mode="edge"), k, "valid"), 1, b)
    return b


def _upsample(Z, factor=4):
    """선형 업샘플(고해상 매끄러운 표면용). 반환 Z_fine, 그리고 mm 좌표."""
    ny, nx = Z.shape
    xi = np.linspace(0, nx - 1, nx * factor); yi = np.linspace(0, ny - 1, ny * factor)
    Zx = np.array([np.interp(xi, np.arange(nx), row) for row in Z])
    Zf = np.array([np.interp(yi, np.arange(ny), col) for col in Zx.T]).T
    mm = np.linspace(g2.CENTERS[0], g2.CENTERS[-1], nx * factor)
    return Zf, mm


def fields(dia, name):
    """16 taxel 수용장 grid + envelope E(x,y)=max_t |ΔS_t|. dead = 그 녹화에서 죽은 채널."""
    df, _ = g2.load_material(g2.DATASETS[dia][name])
    dead = {t for t in SK if df[f"dS_{t}"].abs().max() < 1.0}
    grids = {t: np.clip(np.nan_to_num(g2.build_grid(df, df[f"dS_{t}"].abs()), nan=0.0), 0, None)
             for t in SK}
    E = np.nanmax(np.stack([grids[t] for t in SK]), axis=0)
    return grids, E, dead


def main(dia):
    data = {n: fields(dia, n) for n in ORDER}
    vmax = max(float(E.max()) for _, E, _ in data.values())

    # ---------- 2D 통합 히트맵 + 각 taxel 50% 수용장 등고선 ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.0), constrained_layout=True)
    yy, xx = np.meshgrid(g2.CENTERS, g2.CENTERS, indexing="ij")
    im = None
    for ax, name in zip(axes, ORDER):
        grids, E, dead = data[name]
        im = ax.imshow(E, origin="lower", cmap="turbo", vmin=0, vmax=vmax,
                       extent=EXT, interpolation="bilinear", aspect="equal")
        for t in SK:
            sx, sy = g2.SENSOR_XY[t]
            if t in dead:                           # 죽은 채널 = 데이터 결손(소재 특성 아님)
                ax.plot(sx, sy, "x", color="#ff3030", ms=11, mew=2.4)
                ax.annotate(f"{t.replace('Skin','s')} DEAD", (sx, sy), color="#ff3030",
                            fontsize=7.5, fontweight="bold", ha="center", va="bottom",
                            xytext=(0, 6), textcoords="offset points")
                continue
            g = grids[t]; pk = g.max()
            if pk > 1.0:
                ax.contour(xx, yy, g, levels=[0.5 * pk], colors="#39d0ff", linewidths=0.8, alpha=0.8)
            ax.plot(sx, sy, "+", color="white", ms=7, mew=1.2)
        ttl = LABEL[name] + (f"   [dead ch: {','.join(sorted(d.replace('Skin','s') for d in dead))}]" if dead else "")
        ax.set_title(ttl, fontsize=11, fontweight="bold",
                     color="#c0392b" if dead else "black")
        ax.set_xlabel("indenter x (mm)"); ax.set_ylabel("indenter y (mm)")
        ax.set_xticks([-9.75, -3.25, 3.25, 9.75]); ax.set_yticks([-9.75, -3.25, 3.25, 9.75])
    cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    cb.set_label("envelope max_t |ΔS| (%)", fontsize=10)
    fig.suptitle(f"Combined receptive field of all 16 taxels on one plane  ({dia})\n"
                 "cyan = each taxel's 50% receptive-field contour, white + = taxel.  Valleys are NOT zero: "
                 "the midpoint BETWEEN two adjacent taxels still responds ~40% (overlap); only the 4-way centre dips to ~10%.",
                 fontsize=12.5, fontweight="bold")
    out2 = os.path.join(OUT, f"combined_{dia}_2d.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {out2}")

    # ---------- 3D 표면(파동): 봉우리=taxel, 골=중첩 ----------
    fig = plt.figure(figsize=(17, 6.2))
    for k, name in enumerate(ORDER):
        _, E, dead = data[name]
        Ef, mm = _upsample(_blur(E, sigma=1.0), factor=4)   # 매끄러운 파동(스파이크 억제)
        Ef = _blur(Ef, sigma=2.0)
        Xf, Yf = np.meshgrid(mm, mm)
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        surf = ax.plot_surface(Xf, Yf, Ef, cmap="turbo", vmin=0, vmax=vmax,
                               rstride=2, cstride=2, linewidth=0, antialiased=True)
        ax.set_zlim(0, vmax * 1.05)
        ax.set_xlabel("indenter x (mm)", fontsize=8)
        ax.set_ylabel("indenter y (mm)", fontsize=8)
        ax.set_zlabel("max_t |ΔS| (%)", fontsize=8)
        ttl = LABEL[name] + (f"  [dead: {','.join(sorted(d.replace('Skin','s') for d in dead))}]" if dead else "")
        ax.set_title(ttl, fontsize=11, fontweight="bold", color="#c0392b" if dead else "black")
        ax.view_init(elev=38, azim=-58)
    fig.colorbar(surf, ax=fig.axes, fraction=0.012, pad=0.04).set_label("max_t |ΔS| (%)", fontsize=9)
    fig.suptitle(f"3D wave — combined receptive field ({dia});  peaks = taxels, valleys filled = overlap "
                 "(mesh fills valleys = wider/overlapping fields; eco20 isolated peaks = gaps)",
                 fontsize=12.5, fontweight="bold")
    out3 = os.path.join(OUT, f"combined_{dia}_3d.png")
    fig.savefig(out3, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {out3}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--diameter", choices=["d5", "d10", "all"], default="all")
    a = ap.parse_args()
    for dia in (["d5", "d10"] if a.diameter == "all" else [a.diameter]):
        main(dia)
