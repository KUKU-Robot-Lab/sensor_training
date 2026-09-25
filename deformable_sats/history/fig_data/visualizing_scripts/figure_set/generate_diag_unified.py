#!/usr/bin/env python3
"""진단 그림(diag_<run>.png) 소재 통일축 재생성 — samples npz 기반(모델 재추론 없음).

eval_diagnostics 의 per-run 그림은 소재별 자동축이라 비교가 안 된다. 이 스크립트는
이미 최신인 samples_<run>.npz(rmse/rel/x/y/fz/is_d5)에서 위치별 RMSE heatmap + force-RMSE
곡선을 **전 run 공통 축**(heatmap vmax·곡선 y·force x 동일)으로 다시 그려 소재 간 비교를 가능케 한다.

대상: fig3_diag(9 run) · pool_diag · final_xy0p5_diag 의 모든 samples_*.npz.

사용::

    .venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_diag_unified.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import binned_statistic_2d  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
EXP = REPO / "history/fig_data/experiments_archive"
DIRS = [EXP / "sizeA_diag", EXP / "sizeA_final_xy0p5_diag"]  # A(indenter-size) 최종 모델
BINS = 40


def _samples() -> list[tuple[str, Path, dict]]:
    out = []
    for d in DIRS:
        for f in sorted(d.glob("samples_*.npz")):
            run = f.name[len("samples_"):-len(".npz")]
            out.append((run, d, dict(np.load(f))))
    return out


def _posmap(s: dict) -> np.ndarray:
    stat, _, _, _ = binned_statistic_2d(
        s["x"], s["y"], s["rmse"], statistic="mean", bins=BINS, range=[(-10, 10), (-10, 10)])
    return stat.T


def _fz_curve(s: dict) -> tuple[np.ndarray, np.ndarray]:
    fz = s["fz"]; rmse = s["rmse"]
    finite = np.isfinite(fz)
    order = np.argsort(fz[finite])
    win = max(101, (order.size // 200) | 1)
    sm = np.convolve(rmse[finite][order], np.ones(win) / win, mode="same")
    return fz[finite][order], sm


def main() -> None:
    data = _samples()
    print(f"대상 {len(data)} run")

    # 공통 축: heatmap vmax(위치맵 95분위 최대), 곡선 y-max, force x-max
    heat_vmax = 0.0
    curve_ymax = 0.0
    fz_xmax = 0.0
    curves = {}
    maps = {}
    for run, d, s in data:
        pm = _posmap(s)
        maps[run] = pm
        heat_vmax = max(heat_vmax, float(np.nanquantile(pm, 0.95)))
        fz, sm = _fz_curve(s)
        curves[run] = (fz, sm)
        if sm.size:
            curve_ymax = max(curve_ymax, float(np.nanquantile(sm, 0.99)))
            fz_xmax = max(fz_xmax, float(np.nanquantile(fz, 0.99)))

    print(f"공통 축: heatmap vmax={heat_vmax:.3f}  곡선 ymax={curve_ymax:.3f}  fz xmax={fz_xmax:.2f}")

    for run, d, s in data:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        im = axes[0].imshow(maps[run], origin="lower", extent=[-10, 10, -10, 10],
                            aspect="equal", cmap="magma", vmin=0, vmax=heat_vmax)
        axes[0].set_title(f"{run[:38]}\nper-position mean RMSE")
        axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("y (mm)")
        fig.colorbar(im, ax=axes[0], fraction=0.046, label="RMSE [a.u.]")

        fz, sm = curves[run]
        axes[1].plot(fz, sm, lw=1.2, color="#2c7fb8")
        axes[1].set_xlim(0, fz_xmax)
        axes[1].set_ylim(0, curve_ymax * 1.05)
        axes[1].set_title("RMSE vs fz (moving avg)")
        axes[1].set_xlabel("fz (N)"); axes[1].set_ylabel("per-sample RMSE")
        axes[1].grid(ls=":", alpha=0.4)
        fig.suptitle("shared axes (material-unified) — comparable across runs", fontsize=10, y=1.02)
        fig.tight_layout()
        out = d / f"diag_{run}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("saved:", out)


if __name__ == "__main__":
    main()
