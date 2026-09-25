#!/usr/bin/env python3
"""v6-Test 밴딩+접촉 feasibility — SATS를 jig-bend+접촉으로 가볍게 fine-tune해 재구성 가능성 검증.

배경: jig-bend는 taxel 신호가 작아(30°vs90° 원시차 0.35%) estimator가 곡률을 못 읽음 →
곡률 보정(estimator→restorer) 파이프라인은 부적합. 대신 **SATS 직접 fine-tune**이 올바른 길.

이 스크립트: bent+접촉(접촉 중심(0,0), SATS-flat baseline pct)으로 leave-one-condition-out
fine-tune → ① 홀드아웃 밴딩조건 재구성 ② flat 위치보존(퇴화 아님) 확인.
※ 현 데이터는 접촉이 중심만 → 완전한 off-center 검증엔 위치 다양화 취득 필요.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from sats.bending.pipeline import load_frozen_sats
from sats.preprocessing.merged_bin import merged_bin_to_frame
from sats.tools.v6test_io import load_session
from sats.training.dataset import _load_baseline

W, GRID = 10, 41
GRID_MIN, STEP = -10.0, 0.5
CASES = [("degree_30/d5_degree_30(z_1.2_3.7)", 5.0), ("degree_30/d10_degree_30(z_1.2_3.2)", 10.0),
         ("degree_90/d5_degree_90(z_-3.2_-0.5)", 5.0), ("degree_90/d10_degree_90(z_-3.2_-1.0)", 10.0)]


def sats_flat_baseline(sats_run: Path) -> np.ndarray:
    """SATS 학습 trial들의 baseline.json 평균 = flat 무접촉 기준(cross-session flat 참조)."""
    root = Path("learning_data/sensor_raw_bin/ecomesh_v6_xy1")
    bfiles = glob.glob(str(root / "**/*_baseline.json"), recursive=True)
    bases = [[json.load(open(f))[f"Skin{i}_mean"] for i in range(1, 17)] for f in bfiles]
    return np.mean(bases, axis=0)


def _gauss(cx: float, cy: float, sig: float = 1.5) -> np.ndarray:
    gx = GRID_MIN + np.arange(GRID) * STEP
    xx, yy = np.meshgrid(gx, gx)
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))
    return (g / g.max()).astype(np.float32)


def _fwd(sats, seq, size, dev):
    L = torch.full((seq.shape[0],), W, dtype=torch.long, device=dev)
    sz = torch.full((seq.shape[0],), size, device=dev) if getattr(sats, "use_size_input", False) else None
    o = sats(seq, L, sz)
    return o[0] if isinstance(o, tuple) else o


def _peak(pm) -> np.ndarray:
    out = []
    for m in pm.detach().cpu().numpy():
        r, c = np.unravel_index(m.argmax(), m.shape)
        out.append([GRID_MIN + c * STEP, GRID_MIN + r * STEP])
    return np.asarray(out)


def _bent_windows(rel: str, base: np.ndarray) -> np.ndarray:
    s = load_session(Path("skin_ws/raw_data/v6-Test/bending_contact") / rel)
    pct = ((s.sensor - base) / np.where(np.abs(base) < 1e-9, 1e-9, base) * 100).astype(np.float32)
    mag = np.abs(pct).sum(1)
    ct = np.where(mag > np.quantile(mag, 0.9))[0]
    ct = ct[ct >= W - 1]
    return np.stack([pct[e - W + 1:e + 1] for e in ct[::3]])


def _flat_probe(dev):
    """flat off-center 접촉(v6 SATS holdout d5 test2) — 위치보존 체크용."""
    fb = Path(glob.glob("learning_data/sensor_raw_bin/ecomesh_v6_xy1/**/*d5*test2*_merged.bin", recursive=True)[0])
    df = merged_bin_to_frame(fb)
    base = np.asarray(_load_baseline(Path(glob.glob(str(fb.parent / "*_baseline.json"))[0]), merged_bin=fb), float)
    pct = ((df[[f"s{i}" for i in range(1, 17)]].to_numpy(float) - base)
           / np.where(np.abs(base) < 1e-9, 1e-9, base) * 100).astype(np.float32)
    fz = df["Fz"].to_numpy(); x = df["x_mm"].to_numpy(); y = df["y_mm"].to_numpy()
    e = np.where(fz > 0.3)[0]; e = e[e >= W - 1][::50]
    win = np.stack([pct[i - W + 1:i + 1] for i in e])
    return torch.from_numpy(win).to(dev), np.stack([x[e], y[e]], 1)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="v6 밴딩+접촉 fine-tune feasibility")
    p.add_argument("--sats-run", type=Path, default=repo / "sats/training/runs/ecomesh_v6_deploy_all4")
    p.add_argument("--holdout", type=int, default=3, help="CASES 인덱스(0-3) 홀드아웃")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = args.device

    sats = load_frozen_sats(args.sats_run, dev)
    for pp in sats.parameters():
        pp.requires_grad_(True)
    base = sats_flat_baseline(args.sats_run)
    tgt = torch.from_numpy(_gauss(0, 0)).to(dev)
    fwin, fgt = _flat_probe(dev)
    hold_rel, hold_sz = CASES[args.holdout]
    train_cases = [c for i, c in enumerate(CASES) if i != args.holdout]

    def ev_bent():
        sats.eval()
        with torch.no_grad():
            pm = _fwd(sats, torch.from_numpy(_bent_windows(hold_rel, base)).to(dev), hold_sz, dev)
        return float(np.median(np.linalg.norm(_peak(pm), axis=1)))

    def ev_flat():
        sats.eval()
        with torch.no_grad():
            pm = _fwd(sats, fwin, 5.0, dev)
        return float(np.median(np.linalg.norm(_peak(pm) - fgt, axis=1)))

    print(f"홀드아웃={hold_rel}")
    print(f"[前] 홀드아웃밴딩 loc={ev_bent():.2f}mm  flat 위치오차={ev_flat():.2f}mm")

    Xs, Ss = [], []
    for rel, sz in train_cases:
        w = _bent_windows(rel, base); Xs.append(w); Ss.append(np.full(len(w), sz, np.float32))
    Xt = torch.from_numpy(np.concatenate(Xs)).to(dev)
    St = torch.from_numpy(np.concatenate(Ss)).to(dev)
    opt = torch.optim.Adam(sats.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        sats.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 1024):
            idx = perm[i:i + 1024]
            L = torch.full((len(idx),), W, dtype=torch.long, device=dev)
            out = sats(Xt[idx], L, St[idx]); out = out[0] if isinstance(out, tuple) else out
            loss = ((out - tgt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 3 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:2d}  홀드아웃밴딩 loc={ev_bent():.2f}mm  flat 위치오차={ev_flat():.2f}mm")
    print("\nfeasibility: 홀드아웃 밴딩조건 재구성 + flat 위치보존이면 → jig-bend 접촉 학습 가능."
          " (접촉 중심만이라 off-center 완전검증엔 위치 다양화 취득 필요)")


if __name__ == "__main__":
    main()
