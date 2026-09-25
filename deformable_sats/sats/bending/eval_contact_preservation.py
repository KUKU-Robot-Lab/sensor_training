#!/usr/bin/env python3
"""준-합성 접촉 보존 검증 (기존 데이터만) — restorer가 밴딩만 제거하고 접촉은 보존하나.

가법가정(§5.4): p_bent+contact ≈ p_baseline(κ) + r(contact)  [상대변화% 공간].
실측 접촉 pct(SATS 학습 trial) + 실측 밴딩 오프셋 pct(v0, 곡률별)를 더해 bent+contact 합성.
파이프라인이 원래 접촉 재구성 SATS(pct_contact)을 되살리는지 측정:
  reference = SATS(contact)              [정답]
  uncorrected = SATS(contact+bend)       [밴딩으로 오염]
  corrected = SATS(restorer(합성, κ))     [보정]
지표: 맵 회복률 = 1 − |corr−ref|/|uncorr−ref|, 접촉 위치오차(argmax vs 실제 xy).

★ restorer_mode 비교: seq_deg(레거시, 접촉 파괴) vs deg_only(§5.3, 접촉 보존).
⚠ 가법성을 가정해 만든 데이터 + 접촉/밴딩이 다른 유닛 → 절대 회복률은 지시적(상대 비교가 핵심).
독립 증명은 실측 bending+contact(G2) 필요.
"""
from __future__ import annotations

import argparse
import glob
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .baseline_restorer import BaselineRestorer
from .config import BendingConfig
from .dataset import BendingTrial, make_windows
from .eval_pipeline import _sats_map, _windows
from .pipeline import load_frozen_sats

SKIN = [f"s{i}" for i in range(1, 17)]
GRID_MIN_MM, GRID_MAX_MM = -10.0, 10.0   # 물리 범위는 모든 해상도에서 동일


def _grid_step_mm(width: int) -> float:
    """맵 한 변 크기 → 격자 간격. 41→0.5 · 81→0.25 · 201→0.1.

    ★간격을 상수로 박으면 안 된다. 배포 run 이 g05/g025/g01 로 갈리는데 0.5 로 고정하면
    81² 모델에서 좌표가 2배로 계산되어 위치오차가 통째로 부풀려진다(실제로 발생했다).
    """
    return (GRID_MAX_MM - GRID_MIN_MM) / max(width - 1, 1)


def contact_windows(trial_dir: str | Path, window: int, n: int, fz_min: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """접촉 trial → (pct 윈도우[K,W,16], 접촉 위치 xy[K,2]). 끝 프레임 Fz>fz_min, stride 샘플."""
    from sats.preprocessing.merged_bin import merged_bin_to_frame
    from sats.training.dataset import _load_baseline
    trial_dir = Path(trial_dir)
    mb = glob.glob(str(trial_dir / "*_merged.bin"))[0]
    df = merged_bin_to_frame(mb)
    base = np.asarray(_load_baseline(Path(glob.glob(str(trial_dir / "*_baseline.json"))[0]),
                                     merged_bin=Path(mb)), dtype=np.float64)
    s = df[SKIN].to_numpy(np.float64)
    pct = ((s - base) / np.where(np.abs(base) < 1e-9, 1e-9, base) * 100.0).astype(np.float32)
    fz = df["Fz"].to_numpy(); xy = df[["x_mm", "y_mm"]].to_numpy()
    ends = np.where(fz > fz_min)[0]
    ends = ends[ends >= window - 1]
    ends = ends[:: max(1, len(ends) // n)][:n]
    win = np.stack([pct[e - window + 1:e + 1] for e in ends])
    return win, xy[ends]


def train_restorer(cfg: BendingConfig, data_dir: Path, trials: list[str], device: str,
                   epochs: int = 120, lr: float = 1e-3,
                   contact_win: np.ndarray | None = None, lam: float = 1.0) -> BaselineRestorer:
    """밴딩%(+deg) → flat(0%) restorer 학습. cfg.restorer_mode 에 따라 구조 결정.

    손실:
      L_suppress = ||restore(bend)||²                    (무접촉 변형 → 0)
      L_contact  = ||restore(bend+contact) − contact||²  (접촉 보존; contact_win 제공 시)
    L_contact 는 seq 의존 모드(latent/seq_deg)의 붕괴(접촉 삭제)를 직접 막는다.
    deg 비의존 모드(deg_only/deg_cnn)는 구조적으로 이미 보존되므로 영향이 작다.
    """
    Xs, ds = [], []
    for t in trials:
        z = np.load(data_dir / f"{t}.npz"); m = z["valid"].astype(bool)
        raw = z["sensor"][m].astype(np.float64); b = z["baseline"].astype(np.float64)
        p = ((raw - b) / np.where(np.abs(b) < 1e-9, 1e-9, b) * 100).astype(np.float32)
        w, d = make_windows(BendingTrial(sensor=p, bend_deg=z["bend_deg"][m].astype(np.float32)),
                            cfg.window_size)
        Xs.append(w); ds.append(d)
    X = torch.from_numpy(np.concatenate(Xs)).to(device)
    dg = torch.from_numpy(np.concatenate(ds)).to(device)
    model = BaselineRestorer(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    C = (torch.from_numpy(np.asarray(contact_win, np.float32)).to(device)
         if contact_win is not None else None)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device)
        for i in range(0, len(X), 512):
            bi = perm[i:i + 512]
            xb, db = X[bi], dg[bi]
            loss = nn.functional.mse_loss(model(xb, db), torch.zeros_like(xb))
            if C is not None:                      # 준-합성 접촉 보존(가법 §5.4)
                cb = C[torch.randint(0, len(C), (len(bi),), device=device)]
                loss = loss + lam * nn.functional.mse_loss(model(xb + cb, db), cb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def _peak_xy(m: torch.Tensor) -> np.ndarray:
    """맵 최대점의 물리 좌표[B,2] mm. 격자 간격은 맵 크기에서 유도한다."""
    B, _, Wd = m.shape
    step = _grid_step_mm(Wd)
    fi = m.view(B, -1).argmax(1)
    return np.stack([GRID_MIN_MM + (fi % Wd).cpu().numpy() * step,
                     GRID_MIN_MM + (fi // Wd).cpu().numpy() * step], 1)


def evaluate(
    sats_run_dir: str | Path, contact_trial: str | Path, bending_dir: str | Path,
    *, bending_val: list[str], bending_train: list[str], modes: tuple[str, ...] = ("seq_deg", "deg_only"),
    train_contact_trial: str | Path | None = None,
    device: str = "cpu", n_contact: int = 300, fz_min: float = 0.5, seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    sats = load_frozen_sats(sats_run_dir, device)
    base_cfg = BendingConfig()
    W = base_cfg.window_size
    cwin, cxy = contact_windows(contact_trial, W, n_contact, fz_min)
    bX, bdeg, bdelta, _ = _windows(Path(bending_dir), list(bending_val), W)
    idx = np.linspace(0, len(bX) - 1, len(cwin)).astype(int)
    bwin, bdeg_k, bd = bX[idx], bdeg[idx], bdelta[idx]

    # 학습용 접촉(붕괴 방지 손실)은 ★평가와 다른 trial★ 에서 — 누수 방지
    train_contact = (contact_windows(train_contact_trial, W, n_contact, fz_min)[0]
                     if train_contact_trial is not None else None)
    ct = torch.from_numpy(cwin).to(device); bt = torch.from_numpy(bwin).to(device)
    dg = torch.from_numpy(bdeg_k).to(device); synth = ct + bt
    with torch.no_grad():
        ref = _sats_map(sats, ct)
        unc = _sats_map(sats, synth)
    eu = (unc - ref).abs().mean(dim=(1, 2)).cpu().numpy()
    loc_ref = float(np.linalg.norm(_peak_xy(ref) - cxy, axis=1).mean())
    loc_unc = np.linalg.norm(_peak_xy(unc) - cxy, axis=1)

    def bin_stats(err, loc):
        out = []
        for lo in range(0, 11, 2):
            m = (bd >= lo) & (bd < lo + 2)
            if m.sum() > 3:
                out.append({"delta_lo": lo, "n": int(m.sum()),
                            "map_err": float(err[m].mean()), "loc_mm": float(loc[m].mean())})
        return out

    result = {"n": int(len(cwin)), "delta_range": [float(bd.min()), float(bd.max())],
              "loc_ref_mm": loc_ref, "contact_trial": str(contact_trial),
              "uncorrected": {"map_err": float(eu.mean()), "loc_mm": float(loc_unc.mean()),
                              "by_delta": bin_stats(eu, loc_unc)},
              "modes": {}}
    for mode in modes:
        # "latent:k" 형식이면 잠재차원 k 지정(정보 병목 스윕용)
        if mode.startswith("latent:"):
            cfg = replace(base_cfg, restorer_mode="latent", latent_dim=int(mode.split(":")[1]))
        else:
            cfg = replace(base_cfg, restorer_mode=mode)
        use_contact = train_contact is not None and cfg.restorer_mode in ("latent", "seq_deg")
        restorer = train_restorer(cfg, Path(bending_dir), list(bending_train), device,
                                  contact_win=train_contact if use_contact else None)
        with torch.no_grad():
            cor = _sats_map(sats, restorer(synth, dg))
        ec = (cor - ref).abs().mean(dim=(1, 2)).cpu().numpy()
        loc_c = np.linalg.norm(_peak_xy(cor) - cxy, axis=1)
        result["modes"][mode] = {
            "map_err": float(ec.mean()), "loc_mm": float(loc_c.mean()),
            "recovery": 1.0 - float(ec.mean()) / float(eu.mean()),
            "loc_improve_mm": float(loc_unc.mean() - loc_c.mean()),
            "by_delta": bin_stats(ec, loc_c)}
    return result


def write_figure(result: dict, out_png: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    d = [b["delta_lo"] + 1 for b in result["uncorrected"]["by_delta"]]
    ax1.plot(d, [b["loc_mm"] for b in result["uncorrected"]["by_delta"]], "o-", label="uncorrected")
    ax2.plot(d, [b["map_err"] for b in result["uncorrected"]["by_delta"]], "o-", label="uncorrected")
    for mode, r in result["modes"].items():
        ax1.plot(d, [b["loc_mm"] for b in r["by_delta"]], "s-", label=mode)
        ax2.plot(d, [b["map_err"] for b in r["by_delta"]], "s-", label=mode)
    ax1.axhline(result["loc_ref_mm"], color="k", ls="--", lw=1, label=f"reference {result['loc_ref_mm']:.2f}mm")
    ax1.set_xlabel("|delta| (mm)"); ax1.set_ylabel("contact localization error (mm)")
    ax1.set_title("contact preservation vs curvature"); ax1.grid(alpha=.3); ax1.legend()
    ax2.set_xlabel("|delta| (mm)"); ax2.set_ylabel("map error vs contact-only reference")
    ax2.set_title("reconstruction error vs curvature"); ax2.grid(alpha=.3); ax2.legend()
    fig.suptitle("semi-synthetic contact preservation (existing data; additivity assumed)")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="준-합성 접촉 보존 검증 (restorer 모드 비교).")
    p.add_argument("--sats-run", type=Path,
                   default=repo / "sats/training/runs/size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3")
    p.add_argument("--contact-trial", type=Path,
                   default=repo / "learning_data/sensor_raw_bin/ecomesh_xy0p5/d5/z_2.5mm/test1")
    p.add_argument("--bending-dir", type=Path, default=repo / "learning_data/bending/v0")
    p.add_argument("--bending-val", nargs="+", default=["20260727_test4"])
    p.add_argument("--bending-train", nargs="+", default=["20260727_test2", "20260727_test3"])
    p.add_argument("--train-contact-trial", type=Path, default=None,
                   help="접촉보존 손실용 접촉 trial(★평가용과 다른 trial 지정). 미지정=손실 미사용")
    p.add_argument("--modes", nargs="+", default=["seq_deg", "deg_only"],
                   help="비교할 restorer 모드. latent 는 'latent:k'(k=잠재차원)로도 지정 가능")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--figure", type=Path, default=None)
    args = p.parse_args()

    r = evaluate(args.sats_run, args.contact_trial, args.bending_dir,
                 bending_val=args.bending_val, bending_train=args.bending_train,
                 modes=tuple(args.modes), train_contact_trial=args.train_contact_trial,
                 device=args.device)
    print(f"[접촉보존] n={r['n']} 접촉윈도우 × v0 밴딩(δ {r['delta_range'][0]:.1f}~{r['delta_range'][1]:.1f}mm)")
    print(f"  정답(접촉만) 위치오차 상한 = {r['loc_ref_mm']:.2f}mm")
    u = r["uncorrected"]
    print(f"  {'uncorrected':28s}: map_err={u['map_err']:.4f}  loc={u['loc_mm']:.2f}mm")
    for mode, m in r["modes"].items():
        print(f"  {mode:28s}: map_err={m['map_err']:.4f}  loc={m['loc_mm']:.2f}mm  "
              f"recovery={m['recovery']*100:+.0f}%  loc개선={m['loc_improve_mm']:+.2f}mm")
    if args.figure:
        write_figure(r, args.figure); print(f"  figure: {args.figure}")


if __name__ == "__main__":
    main()
