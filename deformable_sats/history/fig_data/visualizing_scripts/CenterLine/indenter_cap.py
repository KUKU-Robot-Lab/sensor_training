#!/usr/bin/env python3
"""
구형 압입자 캡(Spherical Cap) 3D 입체 시각화

구(직경 D)를 평면에 z_depth만큼 눌렀을 때 재료 표면 아래로 들어가는
구형 캡 입체를 3D로 표시한다.

좌표계
  z = 0       : 재료 원래 표면 = 접촉원(바닥면)이 놓이는 평면
  z = z_depth : 캡 꼭짓점 (구의 가장 깊은 지점)

구면 방정식 (극좌표 r 기준)
  z(r) = sqrt(R² - r²) - (R - z_depth)   for r ∈ [0, r_contact]
  r_contact = sqrt(2·R·z_depth - z_depth²)

사용:
    python scripts/indenter_cap.py
    python scripts/indenter_cap.py --D 5 --z 0.1 0.5 1.0 --out response_png/cap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

DPI: int = 150
VIEW_ELEV: float = 28.0
VIEW_AZIM: float = -50.0


def _setup_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_edgecolor("none")
    ax.grid(False)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)


def draw_cap(ax: plt.Axes, D: float, z_depth: float, z_max: float) -> None:
    """구형 캡 입체(구면 + 접촉원 바닥)를 ax에 그린다."""
    R = D / 2.0
    r_contact = float(np.sqrt(max(0.0, 2 * R * z_depth - z_depth ** 2)))
    if r_contact <= 0:
        return

    n_r, n_t = 250, 360

    # ── 구면 (dome) ──────────────────────────────────────────────────────────
    r = np.linspace(0.0, r_contact, n_r)
    theta = np.linspace(0.0, 2 * np.pi, n_t)
    Rg, Th = np.meshgrid(r, theta)
    X = Rg * np.cos(Th)
    Y = Rg * np.sin(Th)
    Z_dome = np.sqrt(np.clip(R ** 2 - Rg ** 2, 0.0, None)) - (R - z_depth)

    ax.plot_surface(
        X, Y, Z_dome,
        cmap="Blues_r",
        vmin=0.0, vmax=z_max,
        alpha=0.88,
        linewidth=0,
        antialiased=True,
        rcount=150, ccount=150,
    )

    # ── 바닥면 — 접촉원 (z = 0) ──────────────────────────────────────────────
    r_d = np.linspace(0.0, r_contact, 80)
    t_d = np.linspace(0.0, 2 * np.pi, 300)
    Rd, Td = np.meshgrid(r_d, t_d)
    ax.plot_surface(
        Rd * np.cos(Td), Rd * np.sin(Td), np.zeros_like(Rd),
        color="steelblue", alpha=0.45, linewidth=0,
    )

    # ── 접촉원 테두리 ─────────────────────────────────────────────────────────
    t_rim = np.linspace(0.0, 2 * np.pi, 500)
    ax.plot(
        r_contact * np.cos(t_rim),
        r_contact * np.sin(t_rim),
        np.zeros(500),
        color="navy", linewidth=1.5,
    )

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0.0, 5)
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([-5, 0, 5])
    ax.set_xlabel("x (mm)", fontsize=8, labelpad=3)
    ax.set_ylabel("y (mm)", fontsize=8, labelpad=3)
    ax.set_zlabel("depth (mm)", fontsize=8, labelpad=3)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"z = {z_depth:.3f} mm | r_contact = {r_contact:.3f} mm",
        fontsize=8, pad=6,
    )


def plot_single(D: float, z_depth: float, z_max: float, out_path: Path) -> None:
    fig = plt.figure(figsize=(6, 5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    _setup_ax(ax)
    draw_cap(ax, D, z_depth, z_max)
    fig.suptitle(f"Spherical cap  D = {D:.0f} mm", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_grid(D: float, z_vals: list[float], z_max: float, out_path: Path) -> None:
    n = len(z_vals)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 4.5 * rows), facecolor="white")
    fig.suptitle(f"Spherical cap  D = {D:.0f} mm", fontsize=11, y=1.01)

    for idx, z in enumerate(z_vals):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        _setup_ax(ax)
        draw_cap(ax, D, z, z_max)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    default_out = Path(__file__).resolve().parent.parent / "response_png" / "indenter_cap"

    parser = argparse.ArgumentParser(description="구형 캡 입체 시각화")
    parser.add_argument("--D", type=float, default=10.0,
                        help="indenter 직경 (mm), 기본 10.0")
    parser.add_argument("--z", type=float, nargs="+",
                        default=[0.1, 0.5, 1.0, 2.0],
                        help="압입 깊이 목록 (mm)")
    parser.add_argument("--out", type=Path, default=default_out,
                        help="출력 폴더")
    args = parser.parse_args()

    D = args.D
    R = D / 2.0
    z_vals = sorted(z for z in args.z if 0 < z < 2 * R)
    if not z_vals:
        print("유효한 z 값 없음 (0 < z < D 이어야 함)")
        return

    z_max = max(z_vals)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"D = {D} mm  /  R = {R} mm")
    print(f"출력 폴더: {args.out}\n")

    # 개별 파일
    for z in z_vals:
        out_file = args.out / f"D{D:.0f}_z{z:.3f}.png"
        plot_single(D, z, z_max, out_file)
        r_c = np.sqrt(max(0.0, 2 * R * z - z ** 2))
        print(f"  z={z:.3f}mm  r_contact={r_c:.3f}mm  → {out_file.name}")

    # 비교용 전체 그리드
    out_all = args.out / f"D{D:.0f}_all.png"
    plot_grid(D, z_vals, z_max, out_all)
    print(f"\n  전체 비교: {out_all}")


if __name__ == "__main__":
    main()
