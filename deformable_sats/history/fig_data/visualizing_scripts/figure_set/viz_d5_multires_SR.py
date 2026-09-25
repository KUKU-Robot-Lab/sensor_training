#!/usr/bin/env python3
"""d5-only 다해상도 super-resolution 시각화: 같은 대표 접촉을 1.0/0.5/0.25mm 출력으로 렌더.

각 해상도 모델을 로드해 동일 홀드아웃(d5 test10)에서 고force 대표 샘플의 GT|Pred 를
나란히 그려 SR 배율(27×→105×→410×) 향상을 눈으로 보여준다. β 켜진 최종 d5 모델.
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

RUNS = [
    ("1.0mm (27x)", REPO / "sats/training/runs/d5_only_multires/d5only_beta_g1p0"),
    ("0.5mm (105x)", REPO / "sats/training/runs/d5_only_multires/d5only_beta_g0p5"),
    ("0.25mm (410x)", REPO / "sats/training/runs/d5_only_multires/d5only_beta_g0p25"),
    ("0.1mm (2525x)", REPO / "sats/training/runs/d5_only_multires/d5only_beta_g0p1"),
]
OUT = REPO / "history/fig_data/experiments_archive/d5_multires_diag/d5_SR_progression.png"
TARGET_FZ = 3.0   # 대표 힘(N): 이 근처의 중앙 접촉 샘플 선택


def pick_and_render(label, run):
    cfg = load_cfg(run)
    device = cfg.effective_device()
    _, val = build_dataloaders(cfg)
    tgen = BatchGPUTargetGenerator(cfg, device)
    model = _load_model(run, cfg, device)
    best = None
    with torch.no_grad():
        for sb, mb, ln in val:
            sb, mb, ln = sb.to(device), mb.to(device), ln.to(device)
            g = tgen(mb); p, _ = model(sb, ln, None)
            fz = mb[:, 4].cpu().numpy()
            gmax = g.amax(dim=(1, 2)).cpu().numpy()
            for i in np.where((np.abs(fz - TARGET_FZ) < 0.4) & (gmax > 0))[0]:
                score = abs(fz[i] - TARGET_FZ)
                if best is None or score < best[0]:
                    best = (score, float(fz[i]), g[i].cpu().numpy(), p[i].cpu().numpy())
            if best is not None and best[0] < 0.05:
                break
    del model
    torch.cuda.empty_cache()
    return best


def main() -> None:
    cols = []
    for label, run in RUNS:
        b = pick_and_render(label, run)
        cols.append((label, b))
        print(f"{label}: fz={b[1]:.2f}N grid={b[2].shape}")

    vmax = max(c[1][2].max() for c in cols)
    fig, axes = plt.subplots(2, len(cols), figsize=(3.4 * len(cols), 7.2))
    for j, (label, b) in enumerate(cols):
        _, fzv, gmap, pmap = b
        for i, (m, t) in enumerate([(gmap, "GT"), (pmap, "Pred")]):
            ax = axes[i, j]
            im = ax.imshow(m, origin="lower", extent=[-10, 10, -10, 10],
                           cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{label}\n{t}  {m.shape[0]}x{m.shape[1]}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{t}\ny (mm)", fontsize=9)
            ax.set_xlabel("x (mm)", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"d5-only + β  super-resolution progression  (fz≈{cols[0][1][1]:.1f}N, 16 physical taxels → virtual grid)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
