#!/usr/bin/env python3
"""FigS20 — 위치추정 오차(localization error). 논문 대표 지표(0.73mm)에 대응.

pred 압력맵의 최대값 위치 = 추정 접촉 위치, GT 맵 최대값 위치 = 실제 접촉 위치.
둘의 유클리드 거리(mm)를 위치추정 오차로 정의한다(논문 정의와 동일).

산출(모델당 1 파일, 2 패널):
  - 좌: 실제 위치별 평균 위치오차 2D 맵 (논문 FigS20A)
  - 우: force 구간별 평균 위치오차 + SEM (논문 FigS20B)

사용::

    .venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_supp_localization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "history/fig_data/supplementary/S20_localization"
CONTACT_HALF_MM = 9.75
CONTACT_EPS = 1e-3  # GT peak 이 이보다 작으면 비접촉으로 제외

# 처리할 모델: label -> run_dir (xy1 대표 3소재 + xy0p5 최종)
RUNS: dict[str, Path] = {
    "eco20_xy1": REPO / "sats/training/runs/size_input_material/sizeA_eco20_xy1_fold2_e2e_g05",
    "eco50_xy1": REPO / "sats/training/runs/size_input_material/sizeA_eco50_xy1_fold1_e2e_g05",
    "ecomesh_xy1": REPO / "sats/training/runs/size_input_material/sizeA_ecomesh_xy1_fold3_e2e_g05",
    "ecomesh_xy0p5_final": REPO / "sats/training/runs/size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3",
}


def _peak_xy(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(B,H,W) 맵의 최대값 픽셀을 mm 좌표로."""
    b, _, w = maps.shape
    coord = np.linspace(-CONTACT_HALF_MM, CONTACT_HALF_MM, w)
    flat = maps.reshape(b, -1).argmax(axis=1)
    return coord[flat % w], coord[flat // w]  # x, y


def collect_localization(run_dir: Path) -> dict[str, np.ndarray]:
    """run 체크포인트로 per-sample 위치오차/실제위치/force 를 수집."""
    import torch

    from sats.tools.eval_diagnostics import load_cfg, _load_model
    from sats.training.gt_gpu import BatchGPUTargetGenerator
    from sats.training.dataset import build_dataloaders

    cfg = load_cfg(run_dir)
    device = cfg.effective_device()
    _, val_loader = build_dataloaders(cfg)
    model = _load_model(run_dir, cfg, device)
    tgen = BatchGPUTargetGenerator(cfg, device)

    gx, gy, err, fz = [], [], [], []
    with torch.no_grad():
        for sensor_b, meta_b, lengths in val_loader:
            sensor_b, meta_b, lengths = (t.to(device) for t in (sensor_b, meta_b, lengths))
            target = tgen(meta_b)
            _sz = meta_b[:, 0] if getattr(cfg, "use_indenter_size_input", False) else None
            pred, _ = model(sensor_b, lengths, _sz)
            g, p = target.cpu().numpy(), pred.cpu().numpy()
            contact = g.max(axis=(1, 2)) > CONTACT_EPS
            if not contact.any():
                continue
            g, p = g[contact], p[contact]
            gxi, gyi = _peak_xy(g)
            pxi, pyi = _peak_xy(p)
            gx.append(gxi); gy.append(gyi)
            err.append(np.hypot(gxi - pxi, gyi - pyi))
            fz.append(meta_b[:, 4].cpu().numpy()[contact])
    return {
        "gx": np.concatenate(gx), "gy": np.concatenate(gy),
        "err": np.concatenate(err), "fz": np.concatenate(fz),
    }


FORCE_EDGES = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0])


def _force_bins(d: dict[str, np.ndarray]) -> tuple[list[str], list[float], list[float]]:
    """force 구간별 평균 위치오차 + SEM."""
    xt, means, sems = [], [], []
    for i in range(len(FORCE_EDGES) - 1):
        sel = (d["fz"] >= FORCE_EDGES[i]) & (d["fz"] < FORCE_EDGES[i + 1])
        if sel.sum() > 20:
            v = d["err"][sel]
            xt.append(f"{FORCE_EDGES[i]:.2g}–{FORCE_EDGES[i+1]:.2g}")
            means.append(float(v.mean())); sems.append(float(v.std() / np.sqrt(v.size)))
    return xt, means, sems


def plot_localization(label: str, d: dict[str, np.ndarray],
                      heat_vmax: float, bar_ymax: float) -> None:
    """heat_vmax(컬러바 상한)·bar_ymax(막대 y상한)는 전 모델 공통값 → 소재 간 비교 가능."""
    from scipy.stats import binned_statistic_2d

    mean_err = float(d["err"].mean())
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)

    # 좌: 실제 위치별 평균 위치오차 2D 맵 (공통 컬러 스케일)
    stat, _, _, _ = binned_statistic_2d(
        d["gx"], d["gy"], d["err"], statistic="mean", bins=20,
        range=[(-10, 10), (-10, 10)])
    im = axes[0].imshow(stat.T, origin="lower", extent=[-10, 10, -10, 10],
                        aspect="equal", cmap="Greens", vmin=0, vmax=heat_vmax)
    axes[0].set_title(f"{label}: localization error by position")
    axes[0].set_xlabel("X [mm]"); axes[0].set_ylabel("Y [mm]")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="position error [mm]")

    # 우: force 구간별 평균 위치오차 + SEM (공통 y축)
    xt, means, sems = _force_bins(d)
    axes[1].bar(range(len(means)), means, yerr=sems, capsize=3,
                color="#2ca25f", edgecolor="black", alpha=0.85)
    axes[1].set_xticks(range(len(xt)))
    axes[1].set_xticklabels(xt, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylim(0, bar_ymax)
    axes[1].set_title(f"{label}: error vs force  (mean={mean_err:.2f} mm)")
    axes[1].set_xlabel("force fz [N]"); axes[1].set_ylabel("position error [mm]")
    axes[1].grid(axis="y", ls=":", alpha=0.4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"S20_localization_{label}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}   (mean loc-error = {mean_err:.3f} mm, n={d['err'].size})")


def main() -> None:
    import argparse
    import csv

    p = argparse.ArgumentParser(description="FigS20 위치추정 오차 생성")
    p.add_argument("--models", nargs="+", default=list(RUNS), choices=list(RUNS))
    p.add_argument("--per-model-scale", action="store_true",
                   help="각 모델 자동 스케일(비교 불가). 기본은 전 모델 공통 스케일.")
    args = p.parse_args()

    # 1) 전 모델 수집 (위치오차는 mm 물리단위라 소재 간 직접 비교 가능)
    data = {label: collect_localization(RUNS[label]) for label in args.models}

    # 2) 공통 스케일: heatmap vmax·막대 y상한 (모델별 값의 최댓값 → 어느 모델도 포화 안 됨)
    heat_vmax = max(float(np.nanquantile(d["err"], 0.95)) for d in data.values())
    bar_ymax = 0.0
    for d in data.values():
        _, means, sems = _force_bins(d)
        bar_ymax = max([bar_ymax, *[m + s for m, s in zip(means, sems)]])
    bar_ymax *= 1.12

    # 3) 렌더
    rows = []
    for label in args.models:
        print(f"--- {label} ---")
        d = data[label]
        vmax = float(np.nanquantile(d["err"], 0.95)) if args.per_model_scale else heat_vmax
        ymax = bar_ymax  # 막대는 항상 공통(비교 목적). --per-model-scale 은 heatmap 만 개별.
        if args.per_model_scale:
            _, mm, ss = _force_bins(d)
            ymax = max([m + s for m, s in zip(mm, ss)], default=1.0) * 1.12
        plot_localization(label, d, vmax, ymax)
        rows.append({"model": label, "mean_loc_error_mm": round(float(d["err"].mean()), 4),
                     "n": int(d["err"].size)})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "loc_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "mean_loc_error_mm", "n"])
        w.writeheader(); w.writerows(rows)
    print("saved:", OUT_DIR / "loc_summary.csv")


if __name__ == "__main__":
    main()
