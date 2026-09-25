#!/usr/bin/env python3
"""xy0.5 final 모델의 d10 추론 검증: GT vs 예측이 실제로 안 맞는지(=진짜 약함) 아니면
지표/GT 문제인지 눈+수치로 확인. force별(저/중/고) 대표 샘플 GT|Pred|Error 렌더."""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from sats.tools.eval_diagnostics import load_cfg, _load_model
from sats.training.gt_gpu import BatchGPUTargetGenerator
from sats.training.dataset import build_dataloaders

RUN = REPO / "sats/training/runs/datarich_probe/ecomesh_xy0p5_datarich_val_d5test10_d10test3"
cfg = load_cfg(RUN)
device = cfg.effective_device()
_, val = build_dataloaders(cfg)
model = _load_model(RUN, cfg, device)
tgen = BatchGPUTargetGenerator(cfg, device)

# d10 샘플 수집: (fz, gt_peak, pred_peak, rmse, rel, gt_map, pred_map)
samples = []
with torch.no_grad():
    for sb, mb, ln in val:
        sb, mb, ln = sb.to(device), mb.to(device), ln.to(device)
        tgt = tgen(mb); pred, _ = model(sb, ln)
        dia = mb[:, 0].cpu().numpy(); fz = mb[:, 4].cpu().numpy()
        g = tgt.cpu().numpy(); p = pred.cpu().numpy()
        for i in np.where(dia >= 7.5)[0]:
            gp = float(g[i].max()); pp = float(p[i].max())
            rmse = float(np.sqrt(((p[i]-g[i])**2).mean()))
            trms = float(np.sqrt((g[i]**2).mean()))
            rel = rmse/trms if trms > 1e-6 else np.nan
            samples.append((float(fz[i]), gp, pp, rmse, rel, g[i], p[i]))
        if len(samples) > 4000:
            break

fz_all = np.array([s[0] for s in samples])
print(f"d10 샘플 {len(samples)}개, fz [{fz_all.min():.2f},{fz_all.max():.2f}]")

# peak 상관: 예측 peak이 GT peak을 따라가나 (스케일 맞나)
gp = np.array([s[1] for s in samples]); pp = np.array([s[2] for s in samples])
print(f"GT peak vs Pred peak 상관계수: {np.corrcoef(gp, pp)[0,1]:.3f}")
print(f"peak 비율(pred/gt) 중앙: {np.median(pp/np.maximum(gp,1e-6)):.2f}  (1=스케일일치)")

# force 구간별 대표(중앙값 근처) 샘플 선택
def pick(lo, hi):
    idx = [i for i, s in enumerate(samples) if lo <= s[0] < hi]
    if not idx: return None
    rels = np.array([samples[i][4] for i in idx])
    return idx[int(np.argsort(rels)[len(rels)//2])]  # rel 중앙값 샘플

bins = [(0.1, 0.3, "low fz"), (0.5, 1.0, "mid fz"), (1.5, 2.1, "high fz")]
rows = [(lab, pick(lo, hi)) for lo, hi, lab in bins]
fig, axes = plt.subplots(len([r for r in rows if r[1] is not None]), 3, figsize=(11, 9))
if axes.ndim == 1: axes = axes[None, :]
r = 0
for lab, i in rows:
    if i is None: continue
    fzv, gpk, ppk, rmse, rel, gmap, pmap = samples[i]
    vmax = max(gpk, ppk, 1e-6)
    for c, (m, t) in enumerate([(gmap, "GT"), (pmap, "Pred"), (np.abs(pmap-gmap), "|Pred-GT|")]):
        ax = axes[r, c]
        im = ax.imshow(m, origin="lower", extent=[-10,10,-10,10], cmap="hot",
                       vmin=0, vmax=vmax if c<2 else vmax*0.6)
        ax.set_title(f"{t}" + (f"  peak={m.max():.2f}" if c<2 else f"  rmse={rmse:.2f} rel={rel:.2f}"), fontsize=9)
        if c == 0: ax.set_ylabel(f"{lab}\nfz={fzv:.2f}N", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    r += 1
fig.suptitle("xy0.5 final — d10 추론 검증 (GT vs Pred): 형태 일치하면 지표 문제, 어긋나면 진짜 약함", fontsize=11)
fig.tight_layout()
out = REPO / "history/fig_data/experiments_archive/pool_diag/verify_d10_final.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
