"""CenterLine 3D 시각화 — 소재 × 인덴터별 개별 PNG (축 스케일 통일).

기존 visualize_3d_v3.py 는 test 폴더마다 따로 그려 x/y/z 축 범위가 제각각이라
서로 비교가 불가능했다. 본 스크립트는 combined_centerline.csv 하나에서 모든
(소재 × 인덴터) 조합을 읽어, **x·y·z 축 한계와 시야각·박스비율을 전부 동일**하게
고정한 개별 PNG 를 생성한다 → 6장을 나란히 두면 직접 비교 가능.

축 의미
  x = Skin Sensor index (1~16)        : 16개 barometer(4×4 배열)
  y = Accumulated time (s)            : 인덴터가 센터라인 중앙을 깊이 0~2mm 로 반복 압입한 시간
  z = Sensor Change ΔS (%)            : baseline 대비 raw 변화율 (아래 식)

신호 정의 (기존 스크립트와 동일 부호)
  ΔS_i(t) = ( s_i(t) − baseline_i ) / baseline_i × 100   [%]
  baseline_i = 무접촉('unloaded') 구간 센서 i 평균
  → 압입되면 챔버가 눌려 raw 가 감소하므로 ΔS 는 음수(아래로 깊을수록 강한 접촉응답).

출력: CenterLine/_unified_3d/centerline_<material>_<indenter>.png  (6장, 공통 축)
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "combined_centerline.csv")
OUT = os.path.join(HERE, "_unified_3d")
os.makedirs(OUT, exist_ok=True)

S = [f"s{i}" for i in range(1, 17)]
GROUPS = {"Group A (1-4)": [1, 2, 3, 4], "Group B (5-8)": [5, 6, 7, 8],
          "Group C (9-12)": [9, 10, 11, 12], "Group D (13-16)": [13, 14, 15, 16]}
GCOLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
MAT_LABEL = {"eco20": "eco20", "eco50": "eco50", "eco20+mesh": "mesh20"}

# 공통 시야/비율 (모든 그림 동일)
ELEV, AZIM = 20, -60
BOX_ASPECT = (1.5, 1.7, 0.85)
TARGET_PTS = 3000          # 센서당 표시 포인트(다운샘플)

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})


def main():
    df = pd.read_csv(CSV)
    keys = sorted(df.groupby(["material", "depth"]).groups.keys())

    # ── 1차: ΔS 계산 + 전역 z범위/시간범위 산출 ──
    data, zmin, zmax, tmax = {}, np.inf, -np.inf, 0.0
    for m, d in keys:
        g = df[(df.material == m) & (df.depth == d)].sort_values("timestep")
        base = g[g.phase == "unloaded"][S].mean().values
        ds = (g[S].values - base) / base * 100.0
        t = g.timestep.values - g.timestep.values.min()
        step = max(1, len(g) // TARGET_PTS)
        ds, t = ds[::step], t[::step]
        data[(m, d)] = (t, ds)
        zmin, zmax = min(zmin, ds.min()), max(zmax, ds.max())
        tmax = max(tmax, t.max())

    # 공통 축 한계 (5% 단위로 라운드)
    ZMIN = math.floor(zmin / 5) * 5
    ZMAX = math.ceil(zmax / 5) * 5
    TMAX = math.ceil(tmax / 100) * 100
    print(f"[common axes] x=[1,16]  y=[0,{TMAX}]s  z=[{ZMIN},{ZMAX}]%")

    # ── 2차: 그룹별 동일 축으로 플롯 ──
    for (m, d), (t, ds) in data.items():
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        for (gname, sensors), color in zip(GROUPS.items(), GCOLORS):
            for s in sensors:
                ax.plot(np.full(len(t), s), t, ds[:, s - 1],
                        color=color, linewidth=0.6, alpha=0.85)
            ax.plot([], [], [], color=color, linewidth=2, label=gname)

        ax.set_xlim(0.5, 16.5)
        ax.set_ylim(0, TMAX)
        ax.set_zlim(ZMIN, ZMAX)
        ax.set_xticks(range(1, 17))
        ax.view_init(elev=ELEV, azim=AZIM)
        try:
            ax.set_box_aspect(BOX_ASPECT)
        except Exception:
            pass
        ax.set_xlabel("Skin Sensor", fontsize=12, labelpad=10)
        ax.set_ylabel("Accumulated Time (s)", fontsize=12, labelpad=12)
        ax.set_zlabel("Sensor Change ΔS (%)", fontsize=12, labelpad=10)
        ax.set_title(f"{MAT_LABEL[m]}  ·  {d}   (centerline, common axes)",
                     fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)

        out = os.path.join(OUT, f"centerline_{MAT_LABEL[m]}_{d}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[saved] {os.path.basename(out)}")

    print(f"\n[done] {OUT}\n공통 축: x=Skin1~16, y=0~{TMAX}s, z={ZMIN}~{ZMAX}% (전 그림 동일, 시야각 elev{ELEV}/azim{AZIM}).")


if __name__ == "__main__":
    main()
