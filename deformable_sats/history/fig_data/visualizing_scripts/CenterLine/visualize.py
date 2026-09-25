#!/usr/bin/env python3
"""
통합 CSV 시각화 스크립트

1) 3D 막대 그래프  : 물리적 센서 위치에 Δsensor (절댓값) 막대 표시
2) 압입자 형상 곡면 : 구형 압입자가 z mm 눌렸을 때의 접촉 3D 곡면

저장: response_png/<ecomesh>_<depth>/  및  response_png/indenter_shape/

사용:
    python scripts/visualize.py
    python scripts/visualize.py --csv <경로> --out <경로>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.patches import Patch

# ── 사용자 설정 ──────────────────────────────────────────────────────────────

# 막대 그래프에서 센서 한 개의 가로/세로 표시 크기 (mm)
SENSOR_SIZE_MM: float = 1.0

# depth 레이블 → 압입자 지름 (mm)
INDENTOR_DIAMETER: dict[str, float] = {
    "d5":  5.0,
    "d10": 10.0,
}

# 그래프 해상도
DPI: int = 150

# 3D 그래프 시점 (두 그래프를 합성할 때 동일하게 유지)
VIEW_ELEV: float = 28.0   # 仰角 (도)
VIEW_AZIM: float = -50.0  # 방위각 (도)

# 출력/입력 기본 경로
CSV_DEFAULT = Path(__file__).resolve().parent.parent / "fig2_material_ablation" / "consolidated.csv"
OUT_DEFAULT  = Path(__file__).resolve().parent.parent / "response_png"

# ── 센서 물리적 위치 (mm) ─────────────────────────────────────────────────────
# 4×4 격자, 6.5mm 간격, 중심 ±9.75mm
_GRID = [-9.75, -3.25, 3.25, 9.75]
_ROWS = ["A", "B", "C", "D"]   # y 오름차순: A=-9.75 → D=9.75

# s1..s16 → (x_mm, y_mm)
SENSOR_XY: dict[str, tuple[float, float]] = {
    f"s{ri * 4 + ci + 1}": (_GRID[ci], _GRID[ri])
    for ri in range(4)
    for ci in range(4)
}
SENSOR_COLS = [f"s{i}" for i in range(1, 17)]

# ── 색상 정의 ────────────────────────────────────────────────────────────────
# 그룹 기본색 (진한 버전)
_GROUP_BASE: dict[str, np.ndarray] = {
    "A": np.array([0.85, 0.10, 0.10]),   # RED
    "B": np.array([0.10, 0.65, 0.10]),   # GREEN
    "C": np.array([0.10, 0.25, 0.85]),   # BLUE
    "D": np.array([0.40, 0.40, 0.40]),   # GREY
}

def _sensor_rgb(sensor: str) -> np.ndarray:
    """센서명 → RGB. 그룹 내 번호 증가(s1→s4)일수록 흰색 방향으로 이동."""
    idx    = int(sensor[1:]) - 1
    group  = _ROWS[idx // 4]
    within = idx % 4                     # 0,1,2,3 → 0%, 20%, 40%, 60% lighter
    base   = _GROUP_BASE[group]
    return base + (1.0 - base) * (within * 0.20)


# ── 공통 유틸 ────────────────────────────────────────────────────────────────

def _safe_name(s: str) -> str:
    """파일명에 사용 가능한 문자열로 변환."""
    return s.replace(" + ", "_plus_").replace(" ", "_")


def _prep_base(grp_df: pd.DataFrame) -> pd.Series:
    """grp_df 내 base phase 센서값 평균 (Series, index=s1..s16)."""
    base = grp_df[grp_df["phase"] == "base"][SENSOR_COLS]
    if base.empty:
        return pd.Series(np.zeros(len(SENSOR_COLS)), index=SENSOR_COLS, dtype=float)
    return base.mean()


# ── 1) 3D 막대 그래프 ─────────────────────────────────────────────────────────

def plot_bar3d(
    deltas: dict[str, float],
    title: str,
    out_path: Path,
    z_max_global: float = 1.0,
) -> None:
    """물리적 센서 위치에 Δsensor 막대를 3D로 그린다."""
    fig = plt.figure(figsize=(7, 6), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.xaxis.pane.set_facecolor("white")
    ax.yaxis.pane.set_facecolor("white")
    ax.zaxis.pane.set_facecolor("white")
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.grid(False)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    s_half = SENSOR_SIZE_MM / 2

    for sname in SENSOR_COLS:
        dz = deltas.get(sname, 0.0)
        if dz < 0:
            dz = 0.0
        sx, sy = SENSOR_XY[sname]
        rgb = _sensor_rgb(sname)

        ax.bar3d(
            sx - s_half, sy - s_half, 0,
            SENSOR_SIZE_MM, SENSOR_SIZE_MM, dz,
            color=tuple(rgb), alpha=0.88, shade=True,
            zsort="average",
        )

    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(0, z_max_global * 1.05)
    ax.set_xlabel("x (mm)", fontsize=8, labelpad=4)
    ax.set_ylabel("y (mm)", fontsize=8, labelpad=4)
    ax.set_zlabel("Δsensor (counts)", fontsize=8, labelpad=4)
    ax.set_title(title, fontsize=9, pad=8)
    ax.tick_params(labelsize=7)

    legend_handles = [
        Patch(color=tuple(_GROUP_BASE[g]),
              label=f"Row {g}  s{i*4+1}–s{i*4+4}")
        for i, g in enumerate(_ROWS)
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=7, framealpha=0.6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def generate_bar_charts(df: pd.DataFrame, out_root: Path) -> None:
    print("=== [1] 3D 막대 그래프 생성 ===")

    for (eco, dep), grp in df.groupby(["ecomesh", "depth"]):
        folder = out_root / f"{_safe_name(eco)}_{dep}"
        base_vals = _prep_base(grp)

        # 이 실험 전체 최대 delta → z축 스케일 통일
        non_base = grp[grp["phase"] != "base"]
        max_delta = 0.0
        for _, row in non_base.groupby(["y", "z", "phase"])[SENSOR_COLS].mean().iterrows():
            d = (row - base_vals).abs().max()
            if d > max_delta:
                max_delta = d
        if max_delta == 0:
            max_delta = 1.0

        # loading + holding phase만 그래프 생성
        for phase in ("loading", "holding"):
            phase_df = grp[grp["phase"] == phase]
            if phase_df.empty:
                continue

            for (y_val, z_val), sub in phase_df.groupby(["y", "z"]):
                y_disp = 0.0 if abs(y_val) < 1e-9 else float(y_val)
                mean_sensors = sub[SENSOR_COLS].mean()
                deltas = {
                    s: float(abs(mean_sensors[s] - base_vals[s]))
                    for s in SENSOR_COLS
                }
                title = (f"{eco} / {dep}\n"
                         f"y = {y_disp:+.1f} mm  |  z = {z_val:.3f} mm  [{phase}]")
                fname = f"bar_y{y_disp:+.1f}_z{z_val:.3f}_{phase}.png"
                plot_bar3d(deltas, title, folder / fname, z_max_global=max_delta)

        print(f"  완료: {eco}/{dep}  →  {folder}")

    print()


# ── 2) 압입자 형상 3D 곡면 ─────────────────────────────────────────────────────

def plot_indenter_shape(
    depth_label: str,
    z_depth: float,
    out_path: Path,
    z_max_global: float | None = None,
) -> None:
    """구형 압입자가 z_depth(mm) 눌렸을 때의 접촉 곡면을 3D surface로 그린다."""
    D = INDENTOR_DIAMETER.get(depth_label, 5.0)
    R = D / 2.0

    if z_depth <= 1e-6:
        return

    # 접촉원 반지름
    r_contact = float(np.sqrt(max(0.0, 2 * R * z_depth - z_depth ** 2)))
    if r_contact <= 0:
        return

    # meshgrid: 접촉 반경 범위로 곡면 생성
    res = 300
    x = np.linspace(-r_contact, r_contact, res)
    y = np.linspace(-r_contact, r_contact, res)
    X, Y = np.meshgrid(x, y)
    R2 = X ** 2 + Y ** 2

    mask = R2 <= r_contact ** 2
    Z = np.full_like(X, np.nan)
    with np.errstate(invalid="ignore"):
        Z[mask] = (z_depth - (R - np.sqrt(np.clip(R ** 2 - R2, 0, None))))[mask]

    fig = plt.figure(figsize=(6, 5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.xaxis.pane.set_facecolor("white")
    ax.yaxis.pane.set_facecolor("white")
    ax.zaxis.pane.set_facecolor("white")
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.grid(False)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    z_top = (z_max_global if z_max_global is not None else z_depth) * 1.1

    surf = ax.plot_surface(
        X, Y, Z,
        cmap="coolwarm_r",
        vmin=0, vmax=z_top,
        alpha=0.90,
        linewidth=0,
        antialiased=True,
        rcount=120, ccount=120,
    )
    cbar = fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08)
    cbar.set_label("indentation depth (mm)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, z_top)
    ax.set_xlabel("x (mm)", fontsize=8, labelpad=4)
    ax.set_ylabel("y (mm)", fontsize=8, labelpad=4)
    ax.set_zlabel("depth (mm)", fontsize=8, labelpad=4)
    ax.set_title(
        f"Indenter shape | {depth_label}  D={D:.0f}mm | z={z_depth:.3f}mm",
        fontsize=9, pad=8,
    )
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def generate_indenter_shapes(df: pd.DataFrame, out_root: Path) -> None:
    print("=== [2] 압입자 형상 곡면 생성 ===")
    folder = out_root / "indenter_shape"

    depth_labels = sorted(df["depth"].unique())
    z_vals = sorted(
        df[df["phase"].isin({"loading", "holding"})]["z"].unique()
    )
    z_vals = [z for z in z_vals if z > 1e-6]   # z=0 제외

    # depth_label별로 같은 z축 스케일 사용
    z_max_global = max(z_vals) if z_vals else 1.0

    for dep in depth_labels:
        for z in z_vals:
            out_path = folder / dep / f"z{z:.3f}.png"
            plot_indenter_shape(dep, z, out_path, z_max_global=z_max_global)
        print(f"  완료: {dep}  →  {folder / dep}")

    print()


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="통합 CSV 시각화")
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT,
                        help="consolidated.csv 경로")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT,
                        help="출력 루트 폴더")
    args = parser.parse_args()

    print(f"CSV : {args.csv}")
    print(f"OUT : {args.out}\n")

    df = pd.read_csv(args.csv)

    generate_bar_charts(df, args.out)

    print(f"모든 그래프 저장 완료: {args.out}")


if __name__ == "__main__":
    main()
