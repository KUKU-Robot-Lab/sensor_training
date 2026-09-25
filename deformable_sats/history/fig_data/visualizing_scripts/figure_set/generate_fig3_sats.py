#!/usr/bin/env python3
"""Fig3 — SATS(xy_1mm) 학습 결과 시각화 (모든 패널 소재별, 압력·위치오차는 3D).

참고 논문(Super-resolution tactile sensor arrays, Fig.4) 스타일을 따르되,
지표 교란(sats-metric-dataset-confound)을 피하기 위해 **d5/d10 분리 + 상대오차**를 주지표로 쓴다.

입력:
  - per-sample npz : history/fig_data/experiments_archive/fig3_diag/samples_<run>.npz
  - 요약 CSV       : history/fig_data/experiments_archive/fig3_diag/diag_summary.csv
  - run 요약(best_epoch) : history/fig_data/experiments_archive/xy1_material_d5d10/summary_by_run.csv
  - 체크포인트     : sats/training/runs/xy1_material_d5d10/<run>/best_model.pt (Fig3C 전용)

사용::

    .venv/bin/python history/fig_data/visualizing_scripts/figure_set/generate_fig3_sats.py \
        --panels B C D E F
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (projection='3d' 등록용)

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:  # sats 패키지 import (Fig3C 추론용)
    sys.path.insert(0, str(REPO))

PRESSURE_UNIT = "a.u."  # target = norm_kernel × fz × gt_scale(100); kPa 직접환산 아님
CONTACT_HALF_MM = 9.75  # 감지면 반경(맵 extent)

# ---- figset: 어떤 학습 결과 집합을 시각화할지. configure() 로 전역 전환. -----------------
FIGSETS: dict[str, dict] = {
    # xy 1 mm 소재 비교 (eco20/eco50/ecomesh 대표 healthy fold)
    "xy1_material": {
        "diag": REPO / "history/fig_data/experiments_archive/sizeA_diag",
        "run_root": REPO / "sats/training/runs/size_input_material",
        "out": REPO / "history/fig_data/fig2_material_ablation/panelD_sats/xy1_material",
        "order": ["eco20", "eco50", "ecomesh"],
        "colors": {"eco20": "#e07b39", "eco50": "#5b8def", "ecomesh": "#2ca25f"},
        "labels": {"eco20": "Eco20", "eco50": "Eco50", "ecomesh": "Eco-mesh"},
        "rep": {
            "eco20": "sizeA_eco20_xy1_fold2_e2e_g05",
            "eco50": "sizeA_eco50_xy1_fold1_e2e_g05",
            "ecomesh": "sizeA_ecomesh_xy1_fold3_e2e_g05",
        },
        "prefix": "Fig2D_",  # 논문 §6 Fig.2D(소재별 SATS 비교). 구명명 Fig3* → 2026-07-12 변경
        "tag": "xy 1 mm",
    },
    # d5-only 최종 (β 물성보정, 크기입력 불필요). 순수 SATS 구조, xy0.5 d5, 0.5mm 출력.
    "d5_final": {
        "diag": REPO / "history/fig_data/experiments_archive/d5_multires_diag",
        "run_root": REPO / "sats/training/runs/d5_only_multires",
        "out": REPO / "history/fig_data/supplementary/d5_final",
        "order": ["ecomesh"],
        "colors": {"ecomesh": "#2ca25f"},
        "labels": {"ecomesh": "Eco-mesh d5"},
        "rep": {"ecomesh": "d5only_beta_g0p5"},
        "prefix": "D5",
        "tag": "d5-only + beta (0.5mm)",
    },
    # xy 0.5 mm 최종(flat) 데이터 — ecomesh 단일, indenter-size input(A). 최종 성능.
    "xy0p5_final": {
        "diag": REPO / "history/fig_data/experiments_archive/sizeA_final_xy0p5_diag",
        "run_root": REPO / "sats/training/runs/size_input",
        "out": REPO / "history/fig_data/fig3_sats_bending/flat_sr/final_xy0p5",
        "order": ["ecomesh"],
        "colors": {"ecomesh": "#2ca25f"},
        "labels": {"ecomesh": "Eco-mesh"},
        "rep": {"ecomesh": "ecomesh_xy0p5_sizeinput_val_d5t10_d10t3"},
        "prefix": "Final",
        "tag": "xy 0.5 mm final",
    },
}

# configure() 가 채우는 활성 전역 (기본 = xy1_material)
DIAG_DIR: Path = FIGSETS["xy1_material"]["diag"]
RUN_ROOT: Path = FIGSETS["xy1_material"]["run_root"]
OUT_DIR: Path = FIGSETS["xy1_material"]["out"]
MATERIAL_ORDER: list[str] = FIGSETS["xy1_material"]["order"]
MATERIAL_COLOR: dict[str, str] = FIGSETS["xy1_material"]["colors"]
MATERIAL_LABEL: dict[str, str] = FIGSETS["xy1_material"]["labels"]
REP_RUN: dict[str, str] = FIGSETS["xy1_material"]["rep"]
PREFIX: str = FIGSETS["xy1_material"]["prefix"]
TAG: str = FIGSETS["xy1_material"]["tag"]

# --shared-axes: 모든 소재에 동일 축 범위를 적용해 소재 간 비교가 가능하도록 함.
# True 면 산출물을 OUT_DIR/shared_axes/ 에 저장(원본 자동스케일 버전은 그대로 유지).
SHARED_AXES: bool = False

# ref-limits: 다른 figset(예: xy1_material)에서 계산한 축 한계를 주입해 그 figset과 동일 축으로 렌더.
# REF_LIMITS 가 있으면 각 패널은 계산값 대신 이 값을 축 범위로 사용한다.
REF_LIMITS: dict | None = None
# 이번 실행에서 각 패널이 계산한 축 한계(공유 목적). shared-axes 실행 종료 시 JSON 저장.
COMPUTED_LIMITS: dict = {}


def _lim(key: str, computed: float) -> float:
    """패널의 축 한계 반환: 계산값을 저장하고, REF_LIMITS 가 있으면 max(ref, computed).

    ref 를 '하한'으로 써서, 현재 figset 데이터가 ref 범위 안이면 ref 와 동일 축(비교 가능),
    ref 를 넘으면 축을 확장해 데이터가 잘리지 않게 한다(예: final xy0.5 의 d10 오차는 xy1 보다 넓음).
    """
    COMPUTED_LIMITS[key] = float(computed)
    if REF_LIMITS is not None and key in REF_LIMITS:
        return max(float(REF_LIMITS[key]), float(computed))
    return float(computed)


def configure(figset: str) -> None:
    """활성 figset 전역을 전환한다."""
    global DIAG_DIR, RUN_ROOT, OUT_DIR, MATERIAL_ORDER, MATERIAL_COLOR
    global MATERIAL_LABEL, REP_RUN, PREFIX, TAG
    c = FIGSETS[figset]
    DIAG_DIR, RUN_ROOT, OUT_DIR = c["diag"], c["run_root"], c["out"]
    MATERIAL_ORDER, MATERIAL_COLOR, MATERIAL_LABEL = c["order"], c["colors"], c["labels"]
    REP_RUN, PREFIX, TAG = c["rep"], c["prefix"], c["tag"]


# ----------------------------------------------------------------------------- helpers
def _load_summary() -> dict[str, dict[str, float]]:
    with open(DIAG_DIR / "diag_summary.csv") as f:
        return {r["run"]: {k: float(v) for k, v in r.items() if k != "run"}
                for r in csv.DictReader(f)}


def _load_samples(run: str) -> dict[str, np.ndarray]:
    return dict(np.load(DIAG_DIR / f"samples_{run}.npz"))


def _finalize(fig: plt.Figure, name: str) -> None:
    out = OUT_DIR / "shared_axes" if SHARED_AXES else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out / name)


# ----------------------------------------------------------------------------- Fig3B
def panel_material_compare() -> None:
    """소재별 d5/d10 상대오차 막대(대표 fold). 값 라벨 포함."""
    summ = _load_summary()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    width = 0.36
    xpos = np.arange(len(MATERIAL_ORDER))

    for gi, (metric, off, hatch, lab) in enumerate(
        [("d5_rel_rmse", -width / 2, None, "d5 (relative)"),
         ("d10_rel_rmse", width / 2, "//", "d10 (relative)")]
    ):
        vals = [summ[REP_RUN[m]][metric] for m in MATERIAL_ORDER]
        bars = ax.bar(xpos + off, vals, width, label=lab, hatch=hatch,
                      color=[MATERIAL_COLOR[m] for m in MATERIAL_ORDER],
                      edgecolor="black", alpha=0.85 if gi == 0 else 1.0, linewidth=0.8)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(xpos)
    ax.set_xticklabels([MATERIAL_LABEL[m] for m in MATERIAL_ORDER])
    ax.set_ylim(0, max(summ[REP_RUN[m]]["d5_rel_rmse"] for m in MATERIAL_ORDER) * 1.28)
    ax.set_ylabel("relative RMSE  (rmse / target RMS)")
    ax.set_title(f"SATS accuracy by material ({TAG})")
    ax.axhline(summ[REP_RUN["ecomesh"]]["d10_rel_rmse"], ls="--",
               c=MATERIAL_COLOR["ecomesh"], lw=1, alpha=0.7)
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.98, 0.97, "Eco-mesh ≤ Eco50  ✓", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=MATERIAL_COLOR["ecomesh"])
    ax.grid(axis="y", ls=":", alpha=0.4)
    _finalize(fig, f"{PREFIX}B_material_compare.png")


# ----------------------------------------------------------------------------- Fig3C
def _representative_maps(run: str) -> dict[str, tuple]:
    """run 체크포인트로 d5/d10 각각 '중앙 근처 최대 peak' 대표 (pred, gt, fz, x, y) 선택."""
    import torch

    from sats.tools.eval_diagnostics import load_cfg, _load_model
    from sats.training.gt_gpu import BatchGPUTargetGenerator
    from sats.training.dataset import build_dataloaders

    run_dir = RUN_ROOT / run
    cfg = load_cfg(run_dir)
    device = cfg.effective_device()
    _, val_loader = build_dataloaders(cfg)
    model = _load_model(run_dir, cfg, device)
    tgen = BatchGPUTargetGenerator(cfg, device)

    # central(중앙 press) 우선, 없으면 any 로 fallback. 배치별 fallback 금지(가장자리 오염 방지).
    best_c: dict[str, tuple] = {}   # 중앙 후보
    best_a: dict[str, tuple] = {}   # 전체 후보
    with torch.no_grad():
        for sensor_b, meta_b, lengths in val_loader:
            sensor_b, meta_b, lengths = (t.to(device) for t in (sensor_b, meta_b, lengths))
            target = tgen(meta_b)
            _sz = meta_b[:, 0] if getattr(cfg, "use_indenter_size_input", False) else None
            pred, _ = model(sensor_b, lengths, _sz)
            peaks = target.amax(dim=(1, 2)).cpu().numpy()
            dia = meta_b[:, 0].cpu().numpy()
            x = meta_b[:, 1].cpu().numpy(); y = meta_b[:, 2].cpu().numpy()
            fz = meta_b[:, 4].cpu().numpy()
            p, g = pred.cpu().numpy(), target.cpu().numpy()
            # GT 맵의 실제 peak 픽셀이 중앙 40% 박스 안이면 central (좌표계 무관하게 blob 중앙 보장)
            H, W = g.shape[1], g.shape[2]
            flat = g.reshape(g.shape[0], -1).argmax(axis=1)
            pr, pc = flat // W, flat % W
            central = (np.abs(pr - H / 2) < H * 0.2) & (np.abs(pc - W / 2) < W * 0.2)
            for key, lo, hi in [("d5", 0.0, 7.5), ("d10", 7.5, 99.0)]:
                grp = (dia >= lo) & (dia < hi) & (peaks > 0)
                for store, extra in [(best_a, grp), (best_c, grp & central)]:
                    idx = np.where(extra)[0]
                    if not idx.size:
                        continue
                    i = idx[np.argmax(peaks[idx])]
                    score = float(peaks[i])
                    if key not in store or score > store[key][0]:
                        store[key] = (score, p[i], g[i], float(fz[i]), float(x[i]), float(y[i]))
    return {k: best_c.get(k, best_a[k]) for k in best_a}


def _surface(ax, data: np.ndarray, title: str, zmax: float, cmap: str) -> None:
    W = data.shape[1]
    xs = np.linspace(-CONTACT_HALF_MM, CONTACT_HALF_MM, W)
    X, Y = np.meshgrid(xs, xs)
    ax.plot_surface(X, Y, data, cmap=cmap, vmin=0, vmax=zmax,
                    rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [mm]", fontsize=8); ax.set_ylabel("y [mm]", fontsize=8)
    ax.set_zlim(0, zmax)
    ax.set_zlabel(f"pressure [{PRESSURE_UNIT}]", fontsize=8)
    ax.view_init(elev=32, azim=-125)
    ax.tick_params(labelsize=7)


def panel_pressure_maps() -> None:
    """소재별 3D 압력 분포맵(GT/Pred) — d5·d10 대표 press. 소재당 1 파일."""
    # 1) 전 소재 대표맵 수집(모델 추론 소재당 1회)
    gathered = {m: _representative_maps(REP_RUN[m]) for m in MATERIAL_ORDER}
    # 2) 공통 z축: press-type(d5/d10)별로 소재 간 최대 peak 공유
    # C는 대표 샘플 1개의 z-peak 기준이라 figset 간 통일(ref-limits) 부적절(클리핑 유발) → figset 자체 스케일만 공유.
    # 핵심 비교는 같은 행 GT vs Pred(동일 z) 이므로 소재 간 z-공유로 충분.
    gzmax: dict[str, float] = {}
    for key in ("d5", "d10"):
        cand = [max(best[key][0], float(best[key][1].max()))
                for best in gathered.values() if key in best]
        if cand:
            gzmax[key] = max(cand)
    # 3) 렌더
    for m, best in gathered.items():
        keys = [k for k in ("d5", "d10") if k in best]
        fig = plt.figure(figsize=(9.4, 4.6 * len(keys)))
        for row, key in enumerate(keys):
            score, pred_m, gt_m, fz, x, y = best[key]
            zmax = gzmax[key] if SHARED_AXES else max(score, float(pred_m.max()), 1e-9)
            for col, (data, tag) in enumerate([(gt_m, "Ground truth"), (pred_m, "Prediction")]):
                ax = fig.add_subplot(len(keys), 2, row * 2 + col + 1, projection="3d")
                _surface(ax, data, f"{key.upper()}  fz={fz:.2f}N  ·  {tag}", zmax, "hot")
        fig.suptitle(f"{MATERIAL_LABEL[m]} ({TAG}): 3D pressure-map inference", y=1.0)
        _finalize(fig, f"{PREFIX}C_pressure3d_{m}.png")


# ----------------------------------------------------------------------------- Fig3D
def _binned_error(s: dict[str, np.ndarray], mask: np.ndarray, bins: int = 18):
    from scipy.stats import binned_statistic_2d

    m = mask & np.isfinite(s["rel"])
    stat, _, _, _ = binned_statistic_2d(
        s["x"][m], s["y"][m], s["rel"][m], statistic="mean", bins=bins,
        range=[(-10, 10), (-10, 10)])
    centers = np.linspace(-10 + 20 / bins / 2, 10 - 20 / bins / 2, bins)
    X, Y = np.meshgrid(centers, centers)
    return X, Y, np.nan_to_num(stat.T)


def _bar3d_error(ax, X, Y, Z, vmax: float, bins: int) -> None:
    """논문 Fig4B 스타일 파란 3D 막대. 색=높이(steelblue 계열), 시야각 논문 매칭."""
    dx = dy = (20.0 / bins) * 0.92
    xpos, ypos, zh = X.ravel(), Y.ravel(), Z.ravel()
    colors = plt.cm.Blues(0.35 + 0.6 * np.clip(zh / vmax, 0, 1))  # steelblue 그라데이션
    ax.bar3d(xpos - dx / 2, ypos - dy / 2, np.zeros_like(zh), dx, dy, zh,
             color=colors, shade=True, edgecolor="none")
    ax.set_xlabel("X [mm]", fontsize=8); ax.set_ylabel("Y [mm]", fontsize=8)
    ax.set_zlabel("relative RMSE", fontsize=8)
    ax.set_zlim(0, vmax)
    ax.view_init(elev=22, azim=-60)  # 논문 Fig4B 시야각 매칭
    ax.tick_params(labelsize=7)


def panel_position_error() -> None:
    """소재별 감지면 위치별 평균 상대오차 3D 막대(bar3d), d5/d10 분리 — 논문 Fig4B 스타일. 소재당 1 파일."""
    bins = 22
    samples = {m: _load_samples(REP_RUN[m]) for m in MATERIAL_ORDER}
    # 공통 vmax: 전 소재 상대오차 합쳐 0.95 분위(비교축 통일)
    all_rel = np.concatenate([s["rel"][np.isfinite(s["rel"])] for s in samples.values()])
    g_vmax = _lim("d_vmax", float(np.nanquantile(all_rel, 0.95)))
    for m in MATERIAL_ORDER:
        s = samples[m]
        vmax = g_vmax if SHARED_AXES else float(np.nanquantile(s["rel"][np.isfinite(s["rel"])], 0.95))
        fig = plt.figure(figsize=(12.0, 4.8))
        fig.subplots_adjust(left=0.02, right=0.9, wspace=0.05)
        for col, (lab, mask) in enumerate([("d5 contact", s["is_d5"]), ("d10 contact", ~s["is_d5"])]):
            X, Y, Z = _binned_error(s, mask, bins=bins)
            ax = fig.add_subplot(1, 2, col + 1, projection="3d")
            _bar3d_error(ax, X, Y, Z, vmax, bins)
            ax.set_title(f"{MATERIAL_LABEL[m]}  {lab}", fontsize=10)
        mappable = plt.cm.ScalarMappable(cmap="Blues")
        mappable.set_clim(0, vmax)
        cax = fig.add_axes([0.94, 0.25, 0.014, 0.5])  # 오른쪽 끝 전용 컬러바
        fig.colorbar(mappable, cax=cax, label="relative RMSE")
        fig.suptitle(f"{MATERIAL_LABEL[m]} ({TAG}): per-position inference error", y=0.98)
        _finalize(fig, f"{PREFIX}D_poserror3d_{m}.png")


# ----------------------------------------------------------------------------- Fig3E
def panel_error_hist() -> None:
    """소재별 상대오차 히스토그램 + KDE + 평균선(d5/d10) — 소재당 1 파일(논문 Fig4C)."""
    from scipy.stats import gaussian_kde

    samples = {m: _load_samples(REP_RUN[m]) for m in MATERIAL_ORDER}
    # 공통 x-max(비교축): 전 소재·d5/d10 상대오차 0.99 분위의 최댓값
    g_xmax = 0.0
    for s in samples.values():
        for mask in (s["is_d5"], ~s["is_d5"]):
            v = s["rel"][mask & np.isfinite(s["rel"])]
            v = v[(v >= 0)]
            if v.size >= 10:
                g_xmax = max(g_xmax, float(np.quantile(v, 0.99)))
    g_xmax = _lim("e_xmax", g_xmax)
    # 공통 y-max(density): shared 모드에서 세로축도 통일 (렌더와 동일 bins/range 로 산정)
    g_ydens = 0.0
    if SHARED_AXES:
        for s in samples.values():
            for mask in (s["is_d5"], ~s["is_d5"]):
                v = s["rel"][mask & np.isfinite(s["rel"])]
                v = v[(v >= 0) & (v < g_xmax)]
                if v.size < 10:
                    continue
                h, _ = np.histogram(v, bins=55, density=True)
                k = gaussian_kde(v)(np.linspace(0, v.max(), 200))
                g_ydens = max(g_ydens, float(h.max()), float(k.max()))
        g_ydens = _lim("e_ymax", g_ydens)
    for m in MATERIAL_ORDER:
        s = samples[m]
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        for lab, mask, color in [("d5", s["is_d5"], "#e07b39"), ("d10", ~s["is_d5"], "#2ca25f")]:
            v = s["rel"][mask & np.isfinite(s["rel"])]
            if v.size < 10:
                continue
            hi = g_xmax if SHARED_AXES else float(np.quantile(v, 0.99))
            v = v[(v >= 0) & (v < hi)]
            if v.size < 10:
                continue
            ax.hist(v, bins=55, density=True, alpha=0.42, color=color,
                    label=f"{lab} (mean={v.mean():.3f})")
            xs = np.linspace(0, v.max(), 200)
            ax.plot(xs, gaussian_kde(v)(xs), color=color, lw=1.6)
            ax.axvline(v.mean(), color=color, ls="--", lw=1)
        if SHARED_AXES and g_xmax > 0:
            ax.set_xlim(0, g_xmax)
        if SHARED_AXES and g_ydens > 0:
            ax.set_ylim(0, g_ydens * 1.05)
        ax.set_title(f"{MATERIAL_LABEL[m]} ({TAG}): inference error distribution", fontsize=11)
        ax.set_xlabel("relative RMSE  (rmse / target RMS)")
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(ls=":", alpha=0.4)
        _finalize(fig, f"{PREFIX}E_error_hist_{m}.png")


# ----------------------------------------------------------------------------- Fig3F
def panel_force_error() -> None:
    """소재별 force(fz) 구간별 d10 상대오차 바이올린 — 소재당 1 파일(논문 Fig4D)."""
    edges = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0])
    centers = [f"{edges[i]:.2g}–{edges[i+1]:.2g}" for i in range(len(edges) - 1)]
    # 1) 소재별 force-구간 그룹 수집
    gathered: dict[str, tuple[list, list, list]] = {}
    for m in MATERIAL_ORDER:
        s = _load_samples(REP_RUN[m])
        mask = ~s["is_d5"]  # d10 (반복취득 있는 주 조건)
        groups, means, xt = [], [], []
        for i in range(len(edges) - 1):
            sel = mask & np.isfinite(s["rel"]) & (s["fz"] >= edges[i]) & (s["fz"] < edges[i + 1])
            v = s["rel"][sel]
            if v.size > 20:
                v = v[v < np.quantile(v, 0.99)]
                groups.append(v); means.append(float(v.mean())); xt.append(centers[i])
        gathered[m] = (groups, means, xt)
    # 2) 공통 y-max(비교축): 전 소재 그룹의 상대오차 최댓값
    g_ymax = _lim("f_ymax", max((float(v.max()) for grp, _, _ in gathered.values() for v in grp), default=0.0))
    # 3) 렌더
    for m, (groups, means, xt) in gathered.items():
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        if groups:
            parts = ax.violinplot(groups, showmeans=True, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(MATERIAL_COLOR[m]); b.set_alpha(0.6)
            ax.plot(range(1, len(means) + 1), means, "o-", c="0.25", lw=1, ms=4, label="mean")
            ax.set_xticks(range(1, len(xt) + 1))
            ax.set_xticklabels(xt, rotation=30, ha="right", fontsize=8)
            ax.legend(frameon=False, fontsize=9)
        if SHARED_AXES and g_ymax > 0:
            ax.set_ylim(0, g_ymax * 1.05)
        ax.set_title(f"{MATERIAL_LABEL[m]} ({TAG}): error vs. force (d10)", fontsize=11)
        ax.set_xlabel("force fz [N]")
        ax.set_ylabel("relative RMSE")
        ax.grid(axis="y", ls=":", alpha=0.4)
        _finalize(fig, f"{PREFIX}F_force_error_{m}.png")


# ----------------------------------------------------------------------------- Fig3A
def panel_symmetry_line() -> None:
    """소재별 중앙선 압력 프로파일(수용영역 중첩/SR) — 논문 Fig4A. 소재당 1 파일.

    감지면 중앙 행에 peak 가 위치한 press 들의 추론 압력 단면(pressure vs x)을 겹쳐 그려
    force 크기로 색을 매핑한다. 겹치는 종형 곡선 = 초해상도 감지의 근거.
    """
    import torch

    from sats.tools.eval_diagnostics import load_cfg, _load_model
    from sats.training.gt_gpu import BatchGPUTargetGenerator
    from sats.training.dataset import build_dataloaders

    rng = np.random.default_rng(0)
    # 1) 전 소재 프로파일 수집(모델 추론은 소재당 1회)
    gathered: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for m in MATERIAL_ORDER:
        run_dir = RUN_ROOT / REP_RUN[m]
        cfg = load_cfg(run_dir)
        device = cfg.effective_device()
        _, val_loader = build_dataloaders(cfg)
        model = _load_model(run_dir, cfg, device)
        tgen = BatchGPUTargetGenerator(cfg, device)

        profiles: list[np.ndarray] = []   # 추론 압력 단면
        fzs: list[float] = []
        with torch.no_grad():
            for sensor_b, meta_b, lengths in val_loader:
                sensor_b, meta_b, lengths = (t.to(device) for t in (sensor_b, meta_b, lengths))
                target = tgen(meta_b)
                _sz = meta_b[:, 0] if getattr(cfg, "use_indenter_size_input", False) else None
                pred, _ = model(sensor_b, lengths, _sz)
                g = target.cpu().numpy(); p = pred.cpu().numpy()
                fz = meta_b[:, 4].cpu().numpy()
                H, W = g.shape[1], g.shape[2]
                flat = g.reshape(g.shape[0], -1).argmax(axis=1)
                pr, pc = flat // W, flat % W
                on_line = (np.abs(pr - H / 2) < 2) & (g.max(axis=(1, 2)) > 0)  # 중앙 행 press
                for i in np.where(on_line)[0]:
                    profiles.append(p[i, int(pr[i]), :]); fzs.append(float(fz[i]))
        if not profiles:
            continue
        prof = np.array(profiles); fzarr = np.array(fzs)
        if prof.shape[0] > 160:  # 과밀 방지 subsample
            keep = rng.choice(prof.shape[0], 160, replace=False)
            prof, fzarr = prof[keep], fzarr[keep]
        gathered[m] = (prof, fzarr)

    if not gathered:
        return

    # 2) 공통 한계(비교축): 압력 y-max, force 컬러 vmax (ref-limits 있으면 그 값 우선)
    g_ymax = _lim("a_ymax", max(float(p.max()) for p, _ in gathered.values()))
    g_fzmax = _lim("a_fzmax", max(float(np.quantile(f, 0.97)) for _, f in gathered.values()))

    # 3) 렌더
    for m, (prof, fzarr) in gathered.items():
        vmax = g_fzmax if SHARED_AXES else float(np.quantile(fzarr, 0.97))
        xs = np.linspace(-CONTACT_HALF_MM, CONTACT_HALF_MM, prof.shape[1])
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        order = np.argsort(fzarr)  # 저force 먼저(뒤에 고force)
        for i in order:
            ax.plot(xs, prof[i], color=plt.cm.turbo(np.clip(fzarr[i] / vmax, 0, 1)),
                    lw=0.7, alpha=0.55)
        ax.set_title(f"{MATERIAL_LABEL[m]} ({TAG}): pressure profiles along center line", fontsize=11)
        ax.set_xlabel("distance x [mm]")
        ax.set_ylabel(f"inferred pressure [{PRESSURE_UNIT}]")
        if SHARED_AXES:
            ax.set_ylim(top=g_ymax * 1.02)
        ax.grid(ls=":", alpha=0.4)
        mappable = plt.cm.ScalarMappable(cmap="turbo")
        mappable.set_clim(0, vmax)
        fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02, label="contact force fz [N]")
        _finalize(fig, f"{PREFIX}A_lineprofile_{m}.png")


PANELS = {
    "A": panel_symmetry_line,
    "B": panel_material_compare,
    "C": panel_pressure_maps,
    "D": panel_position_error,
    "E": panel_error_hist,
    "F": panel_force_error,
}


def main() -> None:
    p = argparse.ArgumentParser(description="SATS figure 생성 (figset별, 소재별, 3D)")
    p.add_argument("--figset", default="xy1_material", choices=list(FIGSETS),
                   help="시각화할 학습 결과 집합")
    p.add_argument("--panels", nargs="+", default=list(PANELS), choices=list(PANELS))
    p.add_argument("--shared-axes", action="store_true",
                   help="모든 소재에 동일 축 범위 적용(소재 간 비교용). 출력=OUT_DIR/shared_axes/")
    p.add_argument("--ref-limits", type=str, default=None,
                   help="다른 figset이 저장한 axis_limits.json 경로. 그 figset과 동일 축으로 렌더(--shared-axes 필요).")
    args = p.parse_args()
    configure(args.figset)
    global SHARED_AXES, REF_LIMITS
    SHARED_AXES = args.shared_axes
    if args.ref_limits:
        import json
        with open(args.ref_limits) as f:
            REF_LIMITS = json.load(f)
        print(f"*** ref-limits 적용: {args.ref_limits} (동일 축으로 렌더) ***")
    if SHARED_AXES:
        print("*** shared-axes 모드: 동일 축 범위 → shared_axes/ 에 저장 ***")
    for key in args.panels:
        if key == "B" and len(MATERIAL_ORDER) < 2:
            print("--- B 건너뜀 (소재 1종이라 비교 불가) ---")
            continue
        print(f"--- {PREFIX}{key} ---")
        PANELS[key]()

    # shared-axes 이면서 ref 를 안 받았을 때만(=기준 figset) 계산한 축 한계를 저장
    if SHARED_AXES and REF_LIMITS is None and COMPUTED_LIMITS:
        import json
        lim_dir = OUT_DIR / "shared_axes"
        lim_dir.mkdir(parents=True, exist_ok=True)
        with open(lim_dir / "axis_limits.json", "w") as f:
            json.dump(COMPUTED_LIMITS, f, indent=2)
        print("saved:", lim_dir / "axis_limits.json")


if __name__ == "__main__":
    main()
