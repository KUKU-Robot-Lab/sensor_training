#!/usr/bin/env python3
"""gpu_on_the_fly run의 대표 GT/pred/error 맵 생성 (d5/d10 각각 상위 peak 샘플)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

from sats.tools.eval_diagnostics import load_cfg, _load_model
from sats.tools.visualize import plot_sample
from sats.training.gt_gpu import BatchGPUTargetGenerator
from sats.training.dataset import build_dataloaders


@torch.no_grad()
def make_maps(run_dir: Path, out_dir: Path, per_group: int = 4) -> None:
    cfg = load_cfg(run_dir)
    device = cfg.effective_device()
    _, val_loader = build_dataloaders(cfg)
    model = _load_model(run_dir, cfg, device)
    tgen = BatchGPUTargetGenerator(cfg, device)
    out_dir.mkdir(parents=True, exist_ok=True)

    # (peak, diameter, fz, pred[41,41], gt[41,41]) 후보를 모아 d5/d10 상위 peak 선택
    cand: list[tuple] = []
    seen = 0
    for sensor_b, meta_b, lengths in val_loader:
        sensor_b = sensor_b.to(device); meta_b = meta_b.to(device); lengths = lengths.to(device)
        target = tgen(meta_b)
        pred, _ = model(sensor_b, lengths)
        peaks = target.amax(dim=(1, 2)).cpu().numpy()
        dia = meta_b[:, 0].cpu().numpy(); fz = meta_b[:, 4].cpu().numpy()
        p = pred.cpu().numpy(); g = target.cpu().numpy()
        # diameter별로 peak 큰 것 몇 개씩 추려 후보에 추가 (d5가 밀리지 않도록)
        for lo, hi in [(0.0, 7.5), (7.5, 99.0)]:
            grp_idx = np.where((dia >= lo) & (dia < hi))[0]
            if grp_idx.size == 0:
                continue
            top = grp_idx[np.argsort(peaks[grp_idx])[-4:]]
            for i in top:
                cand.append((float(peaks[i]), float(dia[i]), float(fz[i]), p[i], g[i]))
        seen += 1

    for label, lo, hi in [("d5", 0.0, 7.5), ("d10", 7.5, 99.0)]:
        grp = [c for c in cand if lo <= c[1] < hi]
        grp.sort(key=lambda c: c[0], reverse=True)
        for k, (peak, dia, fz, pred_m, gt_m) in enumerate(grp[:per_group]):
            title = f"{run_dir.name}  {label}  fz={fz:.2f}N  gt_peak={peak:.2f}"
            rmse = plot_sample(pred_m, gt_m, title, out_dir / f"{label}_{k+1}_fz{fz:.2f}.png")
            print(f"  {label}#{k+1}: fz={fz:.2f} peak={peak:.2f} rmse={rmse:.4f}")


if __name__ == "__main__":
    make_maps(Path(sys.argv[1]), Path(sys.argv[2]))
