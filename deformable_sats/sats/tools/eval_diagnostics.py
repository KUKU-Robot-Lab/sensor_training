#!/usr/bin/env python3
"""SATS run 진단 평가: per-sample pressure-map RMSE를 d5/d10·fz·z·위치(xy)로 분해.

reported ``val_rmse``( = sqrt(mean batch MSE) )는 GT target = base_kernel × fz × gt_scale
구조상 고force·d10 샘플에 지배된다. 이 모듈은 저장된 체크포인트를 재평가하여 지표를
diameter/force/depth/위치별로 분리하고, 스케일 불변 상대오차(rmse/target_rms)를 함께 낸다.

사용::

    python3 -m sats.tools.eval_diagnostics \
        --run-dirs sats/training/runs/.../run_a sats/training/runs/.../run_b \
        --out-dir history/fig_data/experiments_archive/diagnostics
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from sats.training.config import SATSConfig
from sats.training.cnn_module import SATSCNNStage
from sats.training.gt_gpu import BatchGPUTargetGenerator
from sats.training.dataset import build_dataloaders

D5_D10_DIAMETER_SPLIT_MM = 7.5  # diameter < 이 값이면 d5, 이상이면 d10


def collect_samples(
    se: np.ndarray, tms: np.ndarray, meta: np.ndarray
) -> dict[str, np.ndarray]:
    """per-sample 배열을 Fig3 시각화용 dict 로 정리한다.

    - ``se``:  per-sample MSE (pred-target 제곱 HxW 평균)
    - ``tms``: per-sample target mean-square (스케일 지표)
    - ``meta``: (N,5) = (diameter, x, y, z_depth, fz)

    반환: rmse(=sqrt(se)), rel(=sqrt(se/tms), target 0 이면 nan),
    dia/x/y/z/fz, is_d5(diameter < 임계).
    """
    dia, x, y, z, fz = (meta[:, i] for i in range(5))
    rmse = np.sqrt(se)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(tms > 0, rmse / np.sqrt(tms), np.nan)
    return {
        "rmse": rmse,
        "rel": rel,
        "dia": dia,
        "x": x,
        "y": y,
        "z": z,
        "fz": fz,
        "is_d5": dia < D5_D10_DIAMETER_SPLIT_MM,
    }


def load_cfg(run_dir: Path) -> SATSConfig:
    """run_dir/config.json 을 SATSConfig 로 복원한다."""
    data = json.loads((run_dir / "config.json").read_text())
    valid = {f.name for f in fields(SATSConfig)}
    return SATSConfig(**{k: v for k, v in data.items() if k in valid})


def _load_model(run_dir: Path, cfg: SATSConfig, device: str) -> SATSCNNStage:
    model = SATSCNNStage(cfg).to(device).eval()
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt.get("state_dict", ckpt)))
    model.load_state_dict(state)
    return model


@torch.no_grad()
def diagnose(run_dir: Path) -> dict:
    """단일 run 의 val holdout 을 per-sample 로 재평가해 분해 지표를 반환한다."""
    cfg = load_cfg(run_dir)
    device = cfg.effective_device()
    _, val_loader = build_dataloaders(cfg)
    model = _load_model(run_dir, cfg, device)
    tgen = BatchGPUTargetGenerator(cfg, device)

    per_se: list[np.ndarray] = []    # per-sample MSE (HxW 평균)
    tgt_ms: list[np.ndarray] = []    # per-sample target mean-square (스케일 지표)
    metas: list[np.ndarray] = []     # (diameter, x, y, z_depth, fz)
    for sensor_b, meta_b, lengths in val_loader:
        sensor_b = sensor_b.to(device, non_blocking=True)
        meta_b = meta_b.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        target = tgen(meta_b)
        size = meta_b[:, 0] if bool(getattr(cfg, "use_indenter_size_input", False)) else None
        pred, _ = model(sensor_b, lengths, size)
        per_se.append(((pred - target) ** 2).mean(dim=(1, 2)).cpu().numpy())
        tgt_ms.append((target ** 2).mean(dim=(1, 2)).cpu().numpy())
        metas.append(meta_b[:, :5].cpu().numpy())

    se = np.concatenate(per_se)
    tms = np.concatenate(tgt_ms)
    meta = np.concatenate(metas)
    dia, x, y, z, fz = (meta[:, i] for i in range(5))

    def rmse(mask: np.ndarray) -> float:
        return math.sqrt(se[mask].mean()) if mask.sum() else float("nan")

    def rel(mask: np.ndarray) -> float:
        denom = tms[mask].mean() if mask.sum() else 0.0
        return (math.sqrt(se[mask].mean()) / math.sqrt(denom)) if denom > 0 else float("nan")

    d5 = dia < D5_D10_DIAMETER_SPLIT_MM
    d10 = ~d5
    all_mask = np.ones_like(se, dtype=bool)

    out: dict = {
        "run": run_dir.name,
        "n": int(se.size),
        "overall_rmse": rmse(all_mask),
        "overall_rel_rmse": rel(all_mask),
        "d5_rmse": rmse(d5), "d5_rel_rmse": rel(d5), "d5_n": int(d5.sum()),
        "d10_rmse": rmse(d10), "d10_rel_rmse": rel(d10), "d10_n": int(d10.sum()),
        "d5_target_rms": math.sqrt(tms[d5].mean()) if d5.sum() else float("nan"),
        "d10_target_rms": math.sqrt(tms[d10].mean()) if d10.sum() else float("nan"),
    }
    out["fz_quartile_rmse"] = _quartile_rmse(fz, se, tms)
    out["z_quartile_rmse"] = _quartile_rmse(z, se, tms)
    out["_xy"] = (x, y, np.sqrt(se), fz)
    out["_samples"] = collect_samples(se, tms, meta)
    return out


def _quartile_rmse(values: np.ndarray, se: np.ndarray, tms: np.ndarray) -> list[tuple]:
    """values 4분위별 (lo, hi, n, rmse, rel_rmse) 목록."""
    valid = values[np.isfinite(values)]
    if not valid.size:
        return []
    qs = np.quantile(valid, [0, 0.25, 0.5, 0.75, 1.0])
    rows: list[tuple] = []
    for i in range(4):
        m = (values >= qs[i]) & (values <= qs[i + 1])
        n = int(m.sum())
        r = math.sqrt(se[m].mean()) if n else float("nan")
        denom = tms[m].mean() if n else 0.0
        rel = (r / math.sqrt(denom)) if denom > 0 else float("nan")
        rows.append((round(float(qs[i]), 3), round(float(qs[i + 1]), 3), n, round(r, 4), round(rel, 4)))
    return rows


def format_report(o: dict) -> str:
    lines = [
        f"### {o['run']}  (N={o['n']})",
        f"  overall RMSE  = {o['overall_rmse']:.4f}   (rel {o['overall_rel_rmse']:.3f})",
        f"  d5-only  RMSE = {o['d5_rmse']:.4f}   (rel {o['d5_rel_rmse']:.3f})  n={o['d5_n']}  target_rms={o['d5_target_rms']:.4f}",
        f"  d10-only RMSE = {o['d10_rmse']:.4f}   (rel {o['d10_rel_rmse']:.3f})  n={o['d10_n']}  target_rms={o['d10_target_rms']:.4f}",
    ]
    if o["fz_quartile_rmse"]:
        lines.append("  fz 4분위 [lo,hi] n rmse rel:")
        lines += [f"    fz[{b[0]:>6},{b[1]:>6}] n={b[2]:>7} rmse={b[3]:.4f} rel={b[4]:.3f}" for b in o["fz_quartile_rmse"]]
    if o["z_quartile_rmse"]:
        lines.append("  z 4분위 [lo,hi] n rmse rel:")
        lines += [f"    z[{b[0]:>6},{b[1]:>6}] n={b[2]:>7} rmse={b[3]:.4f} rel={b[4]:.3f}" for b in o["z_quartile_rmse"]]
    return "\n".join(lines)


def save_position_fig(o: dict, out_dir: Path, bins: int = 40) -> None:
    """좌: per-position(xy) 평균 RMSE, 우: RMSE vs fz(이동평균)."""
    from scipy.stats import binned_statistic_2d

    x, y, rmse_i, fz = o["_xy"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    stat, _, _, _ = binned_statistic_2d(x, y, rmse_i, statistic="mean", bins=bins, range=[(-10, 10), (-10, 10)])
    im = axes[0].imshow(stat.T, origin="lower", extent=[-10, 10, -10, 10], aspect="equal", cmap="magma")
    axes[0].set_title(f"{o['run'][:34]}\nper-position mean RMSE")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    finite = np.isfinite(fz)
    order = np.argsort(fz[finite])
    win = max(101, (order.size // 200) | 1)
    smoothed = np.convolve(rmse_i[finite][order], np.ones(win) / win, mode="same")
    axes[1].plot(fz[finite][order], smoothed, lw=1)
    axes[1].set_title("RMSE vs fz (moving avg)")
    axes[1].set_xlabel("fz (N)")
    axes[1].set_ylabel("per-sample RMSE")
    fig.tight_layout()
    fig.savefig(out_dir / f"diag_{o['run']}.png", dpi=110)
    plt.close(fig)


_CSV_KEYS = [
    "run", "n", "overall_rmse", "overall_rel_rmse",
    "d5_rmse", "d5_rel_rmse", "d5_n", "d5_target_rms",
    "d10_rmse", "d10_rel_rmse", "d10_n", "d10_target_rms",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SATS run 진단: d5/d10·fz·z·위치별 RMSE 분해")
    p.add_argument("--run-dirs", nargs="+", required=True, help="평가할 run 디렉터리들")
    p.add_argument("--out-dir", required=True, help="그림/CSV 출력 디렉터리")
    p.add_argument("--no-fig", action="store_true", help="위치 그림 생성 생략")
    p.add_argument(
        "--dump-samples",
        action="store_true",
        help="run 별 per-sample 배열을 samples_<run>.npz 로 저장 (Fig3 시각화 입력)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for rd in args.run_dirs:
        o = diagnose(Path(rd))
        print(format_report(o), end="\n\n")
        if not args.no_fig:
            try:
                save_position_fig(o, out_dir)
            except Exception as exc:  # 그림 실패는 진단을 막지 않는다
                print(f"  (figure skipped: {exc})")
        if args.dump_samples:
            npz_path = out_dir / f"samples_{o['run']}.npz"
            np.savez_compressed(npz_path, **o["_samples"])
            print("  saved:", npz_path)
        rows.append({k: o[k] for k in _CSV_KEYS})
    if rows:
        with open(out_dir / "diag_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_KEYS)
            writer.writeheader()
            writer.writerows(rows)
        print("saved:", out_dir / "diag_summary.csv")


if __name__ == "__main__":
    main()
