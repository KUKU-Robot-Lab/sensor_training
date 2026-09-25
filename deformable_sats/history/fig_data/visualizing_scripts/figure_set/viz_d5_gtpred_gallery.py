#!/usr/bin/env python3
"""d5 GT vs Pred 갤러리 — 감지 표면 전반의 다양한 접촉 위치를 커버.

기존 단일 대표 샘플은 위치가 한쪽(가장자리)에 치우침. 여기서는 표면을 3x3 영역으로
나눠 각 영역에서 뚜렷한 접촉(중간 force) 대표를 골라, 중앙·가장자리·모서리를 고루 보여준다.
행=위치, 열=GT|Pred|Error. 해상도 인자로 grid 선택 가능(기본 0.5mm).

사용: python viz_d5_gtpred_gallery.py [g1p0|g0p5|g0p25|g0p1]
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from sats.tools.eval_diagnostics import load_cfg, _load_model  # noqa: E402
from sats.training.gt_gpu import BatchGPUTargetGenerator  # noqa: E402
from sats.training.dataset import build_dataloaders  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "g0p5"
RUN = REPO / f"sats/training/runs/d5_only_multires/d5only_beta_{TAG}"
OUT = REPO / f"history/fig_data/experiments_archive/d5_multires_diag/d5_gtpred_gallery_{TAG}.png"
FZ_LO, FZ_HI = 2.0, 5.0     # 뚜렷한 접촉 위해 중간~높은 force
# 표면 3x3 영역 중심 (mm). 중앙 반드시 포함.
REGIONS = [(-6, 6), (0, 6), (6, 6), (-6, 0), (0, 0), (6, 0), (-6, -6), (0, -6), (6, -6)]


def main() -> None:
    cfg = load_cfg(RUN)
    device = cfg.effective_device()
    _, val = build_dataloaders(cfg)
    tgen = BatchGPUTargetGenerator(cfg, device)
    model = _load_model(RUN, cfg, device)
    use_size = bool(getattr(cfg, "use_indenter_size_input", False))

    # 각 영역별 최적 후보: GT peak 위치가 영역중심에 가깝고 force 범위 내, peak 큰 것
    best = {i: None for i in range(len(REGIONS))}
    with torch.no_grad():
        for sb, mb, ln in val:
            sb, mb, ln = sb.to(device), mb.to(device), ln.to(device)
            g = tgen(mb)
            size = mb[:, 0] if use_size else None
            p, _ = model(sb, ln, size)
            gnp = g.cpu().numpy(); pnp = p.cpu().numpy()
            fz = mb[:, 4].cpu().numpy()
            H = gnp.shape[1]
            for i in range(len(fz)):
                if not (FZ_LO <= fz[i] < FZ_HI):
                    continue
                gm = gnp[i]
                if gm.max() < 1e-6:
                    continue
                pr, pc = np.unravel_index(gm.argmax(), gm.shape)
                # 픽셀 → mm (grid extent -10..10)
                gx = pc / (H - 1) * 20 - 10
                gy = pr / (H - 1) * 20 - 10
                for ri, (rx, ry) in enumerate(REGIONS):
                    d = (gx - rx) ** 2 + (gy - ry) ** 2
                    cand = best[ri]
                    score = d - 0.01 * gm.max()   # 가까우면서 peak 큰 것 선호
                    if cand is None or score < cand[0]:
                        best[ri] = (score, float(gx), float(gy), float(fz[i]), gm, pnp[i])
            if all(best[i] is not None and best[i][0] < 1.0 for i in best):
                break

    rows = [best[i] for i in range(len(REGIONS)) if best[i] is not None]
    vmax = max(r[4].max() for r in rows)
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.7 * n))
    if n == 1:
        axes = axes[None, :]
    for r, row in enumerate(rows):
        _, gx, gy, fzv, gm, pm = row
        err = np.abs(pm - gm)
        for c, (m, t, vm) in enumerate([(gm, "GT", vmax), (pm, "Pred", vmax),
                                        (err, "|Pred-GT|", vmax * 0.5)]):
            ax = axes[r, c]
            im = ax.imshow(m, origin="lower", extent=[-10, 10, -10, 10],
                           cmap="magma", vmin=0, vmax=vm)
            ax.plot(gx, gy, "c+", ms=10, mew=1.5)   # 접촉 위치 표시
            ax.set_title(t if r == 0 else "", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"({gx:+.0f},{gy:+.0f})mm\nfz={fzv:.1f}N", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
    gmm = {"g1p0": "1.0mm", "g0p5": "0.5mm", "g0p25": "0.25mm", "g0p1": "0.1mm"}[TAG]
    fig.suptitle(f"d5-only + β  GT vs Pred across surface positions  (grid {gmm}, {rows[0][4].shape[0]}x{rows[0][4].shape[1]})",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print("saved:", OUT, f"({n} positions)")


if __name__ == "__main__":
    main()
