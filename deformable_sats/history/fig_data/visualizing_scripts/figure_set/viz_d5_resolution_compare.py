#!/usr/bin/env python3
"""출력 grid 해상도별 d5 성능 비교 — samples_*.npz(진단 덤프) 기반, 재추론 없음.

d5_multires_diag 의 samples_d5only_beta_g{1p0,0p5,0p25,0p1}.npz 를 읽어
해상도(SR 배율)에 따른 상대오차 안정성을 정량 비교한다.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
DIAG = REPO / "history/fig_data/experiments_archive/d5_multires_diag"
OUT = DIAG / "d5_resolution_compare.png"
# (tag, grid mm, grid N, SR)
RES = [("g1p0", 1.0, 21, 27), ("g0p5", 0.5, 41, 105),
       ("g0p25", 0.25, 81, 410), ("g0p1", 0.1, 201, 2525)]
FORCE_BINS = [(0.1, 0.4), (0.4, 0.8), (0.8, 1.3), (1.3, 2.2), (2.2, 4.0)]


def load(tag):
    f = DIAG / f"samples_d5only_beta_{tag}.npz"
    return dict(np.load(f)) if f.exists() else None


def main() -> None:
    data = {tag: load(tag) for tag, *_ in RES}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # 좌: 전체 d5 rel-RMSE vs SR 배율
    srs, rels = [], []
    for tag, mm, N, sr in RES:
        s = data[tag]
        if s is None:
            continue
        d5 = s["is_d5"].astype(bool)
        rel = s["rel"][d5]
        rel = rel[np.isfinite(rel)]
        srs.append(sr); rels.append(float(np.median(rel)))
    ax1.plot(srs, rels, "o-", color="#2ca25f", lw=2, ms=9)
    for sr, r in zip(srs, rels):
        ax1.annotate(f"{r:.3f}", (sr, r), textcoords="offset points", xytext=(0, 8), fontsize=9, ha="center")
    ax1.set_xscale("log")
    ax1.set_xticks(srs); ax1.set_xticklabels([f"{s}x\n{m}mm" for (_, m, _, s) in RES], fontsize=8)
    ax1.set_xlabel("super-resolution factor (output grid)")
    ax1.set_ylabel("d5 relative RMSE (median)")
    ax1.set_ylim(0, max(rels) * 1.5)
    ax1.set_title("d5 error vs SR factor — stable across 100x range")
    ax1.grid(ls=":", alpha=0.4)

    # 우: force 구간별 rel-RMSE, 해상도별 곡선
    colors = ["#e07b39", "#5b8def", "#8856a7", "#d62728"]
    centers = [(lo + hi) / 2 for lo, hi in FORCE_BINS]
    for (tag, mm, N, sr), c in zip(RES, colors):
        s = data[tag]
        if s is None:
            continue
        d5 = s["is_d5"].astype(bool)
        fz = s["fz"][d5]; rel = s["rel"][d5]
        ys = []
        for lo, hi in FORCE_BINS:
            m = (fz >= lo) & (fz < hi) & np.isfinite(rel)
            ys.append(float(np.median(rel[m])) if m.sum() > 20 else np.nan)
        ax2.plot(centers, ys, "o-", color=c, label=f"{mm}mm ({sr}x)", lw=1.8, ms=6)
    ax2.set_xlabel("fz (N)")
    ax2.set_ylabel("d5 relative RMSE (median)")
    ax2.set_title("d5 error vs force, by output resolution")
    ax2.legend(fontsize=8); ax2.grid(ls=":", alpha=0.4)

    fig.suptitle("d5-only + β  output-grid resolution comparison  (16 physical taxels -> virtual grid)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("saved:", OUT)
    print("\n SR factor | d5 rel(median)")
    for sr, r in zip(srs, rels):
        print(f"  {sr:>5}x   | {r:.3f}")


if __name__ == "__main__":
    main()
