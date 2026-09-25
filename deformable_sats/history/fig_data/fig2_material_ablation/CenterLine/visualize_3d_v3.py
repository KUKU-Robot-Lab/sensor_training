import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ======================================================
#  ★ 여기만 수정하세요 ★
#  due_data.csv 와 ethermotion_data.csv 가 들어있는 폴더 경로
DATA_DIR = r"C:\Users\SM\Downloads\CenterLine-20260530T102111Z-3-001\CenterLine\eco20\d10\20260529_test1"

#  Z축 고정 범위 (단위: %, 데이터 보고 필요 시 수정)
Z_MIN = -80.0
Z_MAX =   5.0
# ======================================================

OUT_DIR  = r"C:\Users\SM\Desktop\centerline_ 3D_시각화"

DUE_PATH   = os.path.join(DATA_DIR, "due_data.csv")
ETHER_PATH = os.path.join(DATA_DIR, "ethermotion_data.csv")

# 저장 파일명: 폴더 이름 2개를 합쳐 자동 생성 (예: d10_20260529_test2)
_parts   = DATA_DIR.replace("\\", "/").rstrip("/").split("/")
LABEL    = "_".join(_parts[-2:])
OUT_PATH = os.path.join(OUT_DIR, f"3d_sensor_{LABEL}.png")

SKIN_COLS = [f"Skin{i}" for i in range(1, 17)]

GROUPS = {
    "Group A (1-4)":   [1,  2,  3,  4],
    "Group B (5-8)":   [5,  6,  7,  8],
    "Group C (9-12)":  [9, 10, 11, 12],
    "Group D (13-16)": [13, 14, 15, 16],
}
GROUP_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# ── Step 1: due_data 로드 및 전처리 ──────────────────────────────────
print("Loading due_data.csv ...")
due_df = pd.read_csv(DUE_PATH)

baseline = due_df[SKIN_COLS].iloc[4].values
print(f"  Baseline (row 4): {baseline[:4]} ...")

burst_avg = due_df.groupby("burst_index")[["time_s"] + SKIN_COLS].mean().reset_index()
burst_avg[SKIN_COLS] = (burst_avg[SKIN_COLS].values - baseline) / baseline * 100
burst_avg = burst_avg.sort_values("time_s").reset_index(drop=True)
print(f"  Total bursts: {len(burst_avg)}")

# ── Step 2: ethermotion 시간 범위로 burst 범위 한정 ──────────────────
print("Loading ethermotion_data.csv ...")
ether_time = pd.read_csv(ETHER_PATH, usecols=["time_s"])
t_start = float(ether_time["time_s"].min())
t_end   = float(ether_time["time_s"].max())
del ether_time
print(f"  Ethermotion range: {t_start:.2f}s ~ {t_end:.2f}s")

burst_sel = burst_avg[
    (burst_avg["time_s"] >= t_start) & (burst_avg["time_s"] <= t_end)
].reset_index(drop=True)
print(f"  Bursts in range: {len(burst_sel)}")

if len(burst_sel) < 2:
    print("  WARNING: 데이터 부족 → 전체 burst 사용")
    burst_sel = burst_avg.copy()

# ── Step 3: 가상 연속 시간축 (측정 중단 갭 압축) ──────────────────────
dt = pd.Series(burst_sel["time_s"].values).diff().fillna(0)
sampling_dt   = dt[dt > 0].median()
gap_threshold = sampling_dt * 50
n_gaps        = (dt > gap_threshold).sum()

print(f"  Burst interval: {sampling_dt*1000:.2f} ms")
print(f"  Gap threshold:  {gap_threshold*1000:.2f} ms")
print(f"  Gaps removed:   {n_gaps}")

dt_virtual   = dt.copy()
dt_virtual[dt > gap_threshold] = sampling_dt
virtual_time = dt_virtual.cumsum().values
print(f"  Virtual time:   0 ~ {virtual_time[-1]:.2f}s")

# ── Step 4: 3D 플롯 ───────────────────────────────────────────────────
print("Plotting ...")
Z = burst_sel[SKIN_COLS].values   # (N_bursts, 16)
skin_indices = np.arange(1, 17)

fig = plt.figure(figsize=(16, 10))
ax  = fig.add_subplot(111, projection="3d")

for (group_name, sensors), color in zip(GROUPS.items(), GROUP_COLORS):
    for sensor_num in sensors:
        z_vals = Z[:, sensor_num - 1]
        x_vals = np.full(len(virtual_time), sensor_num)
        ax.plot(x_vals, virtual_time, z_vals,
                color=color, linewidth=0.6, alpha=0.85)
    ax.plot([], [], [], color=color, linewidth=2, label=group_name)

ax.set_xlabel("Skin Sensor", fontsize=12, labelpad=10)
ax.set_ylabel("Accumulated Time (s)", fontsize=12, labelpad=10)
ax.set_zlabel("Sensor Change (%)", fontsize=12, labelpad=10)
ax.set_xticks(skin_indices)
ax.set_ylim(0, virtual_time[-1])
ax.set_zlim(Z_MIN, Z_MAX)
TITLE = f"{_parts[-3]} {_parts[-2]}_centerline"
ax.set_title(TITLE, fontsize=14)
ax.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
plt.show()
