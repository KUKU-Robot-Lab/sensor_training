"""Fig.2 — taxel별 '감지 범위 + 필요 침투깊이' hitmap (소재 × 인덴터).

각 taxel s_i 에 대해, 인덴터를 (x,y) 격자로 누르며 **그 taxel 이 처음 반응(|ΔS_i|>THR)
하는 침투깊이**를 (x,y) 맵으로 그린다.
  - 색 = onset 침투깊이(mm): 0 ≈ 얕게만 닿아도 감지(가까움), 1mm ≈ 깊게 눌러야 감지(멈).
  - 공백(미표시) = 최대압입까지 눌러도 미감지 = 그 taxel 의 감지범위 밖.
→ "가까울수록 얕은 깊이에 찍히고, 멀수록 깊이 눌러야 찍힌다" 를 색으로 직접 보여준다.
   감지범위(공백 아닌 영역)의 넓이 = 그 taxel 의 수용장; 소재별로 비교(mesh 넓고 얕음).

신호: g2.load_material 의 per-press local baseline ΔS. 접촉 구간 z≥ZC 만 사용(travel 노이즈 제외).

출력(소재×인덴터 6조합 각각):
  _receptive_depth/<dia>_<mat>_2d.png   : 4×4 몽타주(센서 배열과 동일 배치)
  _receptive_depth/<dia>_<mat>_3d.png   : 중앙 4 taxel 의 3D 깊이 표면(scatter)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import generate_2d_heatmap as g2

ORDER = ["eco20", "eco50", "ecomesh"]
LABEL = {"eco20": "eco20", "eco50": "eco50", "ecomesh": "ecomesh (mesh20)"}
SK = g2.SKIN_COLS
THR_FLOOR = 0.5     # % |ΔS| 절대 최소 감지 임계 (노이즈~0.1~0.2% 의 ~3배)
THR_REL = 0.20      # 각 taxel 자기 peak 의 20% — 강신호(d10)에서 먼 노이즈가 범위로 잡히는 것 방지
ZC = 13.0           # mm 접촉 구간 시작(travel~12, onset~13.3) → penetration = z - ZC
ZMAX = 14.0
PEN_MAX = ZMAX - ZC  # 1.0 mm
CENTRAL = ["Skin6", "Skin7", "Skin10", "Skin11"]
OUT = os.path.join(g2.REPO, "fig2_material_ablation", "hitmap")   # hitmap 전용 폴더(fig2_material_ablation 안)
os.makedirs(OUT, exist_ok=True)
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})


def onset_map(df, t):
    """taxel t 의 onset 침투깊이 맵 (NBIN×NBIN). 미감지 셀 = NaN.

    감지 임계 = max(절대 0.5%, 그 taxel peak 의 20%) — 상대임계로 강신호(d10)에서 먼
    노이즈가 '감지범위'로 잡혀 범위가 과대해지는 것을 막아 깨끗한 수용장 모양을 얻는다.
    """
    a = df[f"dS_{t}"].abs().values
    ix = np.digitize(df.x_mm.values, g2.EDGES) - 1
    iy = np.digitize(df.y_mm.values, g2.EDGES) - 1
    z = df.z_mm.values
    ok = (ix >= 0) & (ix < g2.NBIN) & (iy >= 0) & (iy < g2.NBIN) & (z >= ZC)
    peak = float(a[ok].max()) if ok.any() else 0.0
    thr = max(THR_FLOOR, THR_REL * peak)
    on = np.full((g2.NBIN, g2.NBIN), np.nan)
    d = pd.DataFrame({"ix": ix[ok], "iy": iy[ok], "z": z[ok], "a": a[ok]})
    for (yy, xx), gg in d.groupby(["iy", "ix"]):
        if gg.a.max() > thr:                       # 그 셀에서 peak 의 20% 이상 반응할 때만
            on[yy, xx] = float(gg[gg.a > thr].z.min() - ZC)   # 첫 감지 침투깊이
    return on


def compute(dia, name):
    df, _ = g2.load_material(g2.DATASETS[dia][name])
    dead = {t for t in SK if df[f"dS_{t}"].abs().max() < 1.0}   # 그 녹화에서 죽은 채널
    if dead:
        print(f"   [{dia} {name}] DEAD channel(s): {sorted(dead)}")
    return {t: onset_map(df, t) for t in SK}, dead


CMAP = cm.get_cmap("turbo").copy()
CMAP.set_bad("#eeeeee")   # 미감지 = 연회색
EXT = [g2.CENTERS[0], g2.CENTERS[-1], g2.CENTERS[0], g2.CENTERS[-1]]


def fig_2d(dia, name, maps, dead=frozenset()):
    """4×4 몽타주: 각 칸 = taxel 의 감지범위+깊이, 센서 물리배치와 동일."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12.6))
    im = None
    for i, t in enumerate(SK):
        r, c = divmod(i, 4)
        ax = axes[3 - r, c]                        # y 위로 증가하게 배치(Skin1 좌하단)
        im = ax.imshow(np.ma.masked_invalid(maps[t]), origin="lower", cmap=CMAP,
                       vmin=0, vmax=PEN_MAX, extent=EXT, interpolation="nearest", aspect="equal")
        sx, sy = g2.SENSOR_XY[t]
        ax.plot(sx, sy, "k+", ms=9, mew=1.6)       # 그 taxel 위치
        hit = int(np.isfinite(maps[t]).sum())
        s = t.replace("Skin", "s")
        if t in dead:                               # 그 녹화에서 죽은 채널 = 데이터 결손(소재 특성 아님)
            ax.set_title(f"{s}  — DEAD channel", fontsize=8.5, color="#c0392b", fontweight="bold")
            for sp in ax.spines.values():
                sp.set_color("#c0392b"); sp.set_linewidth(2.0)
            ax.text(0, 0, "DEAD\nchannel", ha="center", va="center", color="#c0392b",
                    fontsize=9, fontweight="bold")
        else:
            ax.set_title(f"{s}  (hit {hit})", fontsize=8.5)
        ax.set_xticks([-9.75, -3.25, 3.25, 9.75]); ax.set_yticks([-9.75, -3.25, 3.25, 9.75])
        ax.tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("onset penetration depth (mm)  —  0 ≈ near/shallow, 1 ≈ far/deep,  gray = not detected", fontsize=9)
    fig.suptitle(f"Receptive detection range & required depth — {LABEL[name]} · {dia}\n"
                 f"each cell = one taxel's detectable area (indenter x,y);  color = penetration depth at first detection",
                 fontsize=13, fontweight="bold")
    out = os.path.join(OUT, f"{dia}_{name}_2d.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


def fig_3d(dia, name, maps):
    """중앙 4 taxel: x,y=인덴터 위치, z=onset 깊이(낮을수록 가까움). 깊이 cone 시각화."""
    yy, xx = np.meshgrid(g2.CENTERS, g2.CENTERS, indexing="ij")
    fig = plt.figure(figsize=(13, 11))
    for k, t in enumerate(CENTRAL):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        m = maps[t]
        fin = np.isfinite(m)
        X, Y, Z = xx[fin], yy[fin], m[fin]
        p = ax.scatter(X, Y, Z, c=Z, cmap="turbo", vmin=0, vmax=PEN_MAX, s=18, depthshade=False)
        sx, sy = g2.SENSOR_XY[t]
        ax.scatter([sx], [sy], [0], color="k", marker="+", s=120)
        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10); ax.set_zlim(0, PEN_MAX)
        ax.invert_zaxis()                          # 얕음(가까움)=위로
        ax.set_xlabel("indenter x (mm)", fontsize=8)
        ax.set_ylabel("indenter y (mm)", fontsize=8)
        ax.set_zlabel("onset depth (mm)", fontsize=8)
        ax.set_title(f"{t}  @({sx:+.2f},{sy:+.2f})", fontsize=10)
        ax.view_init(elev=22, azim=-60)
    fig.colorbar(p, ax=fig.axes, fraction=0.015, pad=0.04).set_label("onset penetration depth (mm)", fontsize=9)
    fig.suptitle(f"3D — onset depth vs indenter position (central 4 taxels) — {LABEL[name]} · {dia}\n"
                 "higher (z up) = detected at shallow press = near;  lower = needs deeper press = far",
                 fontsize=13, fontweight="bold")
    out = os.path.join(OUT, f"{dia}_{name}_3d.png")
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


def main(dia):
    for name in ORDER:
        maps, dead = compute(dia, name)
        fig_2d(dia, name, maps, dead)   # per-taxel 3D 는 combined wave(generate_receptive_field_combined.py)로 대체


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--diameter", choices=["d5", "d10", "all"], default="all")
    a = ap.parse_args()
    for dia in (["d5", "d10"] if a.diameter == "all" else [a.diameter]):
        main(dia)
