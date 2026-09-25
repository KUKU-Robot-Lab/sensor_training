#!/usr/bin/env python3
"""v6 ③ G2 실측 — 밴딩+접촉에서 곡률 보정이 접촉 재구성을 회복하는가 (4-조건).

준-합성(eval_contact_preservation)을 **실측 밴딩+접촉**으로 대체. 취득 요건(핵심):
같은 세션에 **같은 위치의 flat(δ=0) 기준 접촉**을 함께 취득해야 "flat 등가 회복" 비교 가능.

조건(프레임별, SATS 상대변화% 입력):
  reference   = SATS(flat 접촉 pct)              [정답 상한, 같은 위치 무밴딩]
  uncorrected = SATS(밴딩+접촉 pct)              [곡률 오염]
  corrected   = SATS(restorer(밴딩+접촉 pct, deg))[제안 보정; deg=held 곡률 또는 estimator]
  flat_sub    = SATS(밴딩+접촉 pct − 밴딩baseline%)[단순 차감, 선택]
지표: 접촉 위치오차(argmax vs GT xy), reference 대비 맵오차, 곡률(δ)별 분해.

데이터 계약(V6_ACQUISITION_EVAL_SPEC): merged bin(s1..16,Fz,x_mm,y_mm) + 세션 flat baseline +
프레임별 held 곡률 δ(사이드카/컬럼). restorer 는 무접촉 밴딩(learning_data/bending/v5)로 학습.

evaluate_g2() 는 순수(로드된 맵/GT). CLI 는 SATS·estimator·restorer 배선 + 포맷 로더.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .baseline_restorer import BaselineRestorer
from .config import BendingConfig
from .eval_contact_preservation import _peak_xy, train_restorer
from .eval_pipeline import _sats_map
from .pipeline import load_frozen_sats

SKIN = [f"s{i}" for i in range(1, 17)]


def evaluate_g2(ref_maps: torch.Tensor, uncorr_maps: torch.Tensor, corr_maps: torch.Tensor,
                gt_xy: np.ndarray, delta_mm: np.ndarray) -> dict:
    """로드된 맵 3종 + GT → 조건별 위치오차·맵오차, δ별 분해. 순수(테스트 가능)."""
    gt = np.asarray(gt_xy, float).reshape(-1, 2)
    d = np.asarray(delta_mm, float)

    def loc_err(maps):
        return np.linalg.norm(_peak_xy(maps) - gt, axis=1)

    def map_err(maps):
        return (maps - ref_maps).abs().mean(dim=(1, 2)).cpu().numpy()

    conds = {"uncorrected": uncorr_maps, "corrected": corr_maps}
    loc = {"reference": loc_err(ref_maps), **{k: loc_err(v) for k, v in conds.items()}}
    merr = {k: map_err(v) for k, v in conds.items()}

    bins = []
    for lo in range(0, int(np.ceil(np.abs(d).max())) + 2, 2):
        m = (np.abs(d) >= lo) & (np.abs(d) < lo + 2)
        if m.sum() < 3:
            continue
        row = {"delta_lo": lo, "n": int(m.sum()),
               "loc_ref": float(loc["reference"][m].mean())}
        for k in conds:
            row[f"loc_{k}"] = float(loc[k][m].mean())
            row[f"maperr_{k}"] = float(merr[k][m].mean())
        bins.append(row)

    return {
        "n": int(len(gt)),
        "loc_reference_mm": float(loc["reference"].mean()),
        "loc_uncorrected_mm": float(loc["uncorrected"].mean()),
        "loc_corrected_mm": float(loc["corrected"].mean()),
        "maperr_uncorrected": float(merr["uncorrected"].mean()),
        "maperr_corrected": float(merr["corrected"].mean()),
        "map_recovery": 1.0 - float(merr["corrected"].mean()) / max(float(merr["uncorrected"].mean()), 1e-9),
        "by_delta": bins,
    }


def _pct_windows(merged_bin: Path, window: int, fz_min: float):
    """밴딩/flat 접촉 merged → (pct 윈도우[M,W,16], 끝프레임 xy[M,2], time_s[M])."""
    import glob
    from sats.preprocessing.merged_bin import merged_bin_to_frame
    from sats.training.dataset import _load_baseline
    df = merged_bin_to_frame(merged_bin)
    base = np.asarray(_load_baseline(Path(glob.glob(str(merged_bin.parent / "*_baseline.json"))[0]),
                                     merged_bin=merged_bin), float)
    s = df[SKIN].to_numpy(float)
    pct = ((s - base) / np.where(np.abs(base) < 1e-9, 1e-9, base) * 100.0).astype(np.float32)
    fz = df["Fz"].to_numpy(); xy = df[["x_mm", "y_mm"]].to_numpy(float); t = df["timestep_sec"].to_numpy()
    ends = np.where(fz > fz_min)[0]; ends = ends[ends >= window - 1]
    win = np.stack([pct[e - window + 1:e + 1] for e in ends]) if len(ends) else np.zeros((0, window, 16), np.float32)
    return win, xy[ends], t[ends]


def _match_ref(bent_xy: np.ndarray, ref_xy: np.ndarray, max_mm: float = 1.5) -> np.ndarray:
    """밴딩 접촉 프레임 ↔ flat 기준 프레임을 GT (x,y) 최근접 매칭. 반환: bent별 ref 인덱스(-1=없음)."""
    idx = np.full(len(bent_xy), -1, dtype=int)
    for i, b in enumerate(bent_xy):
        d = np.linalg.norm(ref_xy - b, axis=1)
        j = int(np.argmin(d))
        if d[j] <= max_mm:
            idx[i] = j
    return idx


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="v6 G2 실측: 밴딩+접촉 4-조건 회복 평가.")
    p.add_argument("--sats-run", type=Path, required=True)
    p.add_argument("--bent-contact-bin", type=Path, required=True, help="밴딩+접촉 merged bin")
    p.add_argument("--flat-ref-bin", type=Path, required=True, help="같은 위치 flat 기준 접촉 merged bin")
    p.add_argument("--delta-csv", type=Path, required=True,
                   help="밴딩 프레임 held 곡률 δ: 컬럼 time_s, delta_mm (bent-contact와 시간정합)")
    p.add_argument("--bending-dir", type=Path, default=repo / "learning_data/bending/v5",
                   help="restorer 학습용 무접촉 밴딩 npz 디렉토리")
    p.add_argument("--bending-train", nargs="+", required=True, help="restorer 학습 세션 이름")
    p.add_argument("--restorer-mode", default="deg_only", choices=["deg_only", "seq_deg"])
    p.add_argument("--fz-min", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--figure", type=Path, default=None)
    args = p.parse_args()

    import pandas as pd
    cfg = replace(BendingConfig(), restorer_mode=args.restorer_mode)
    W = cfg.window_size
    sats = load_frozen_sats(args.sats_run, args.device)
    restorer = train_restorer(cfg, args.bending_dir, list(args.bending_train), args.device)

    bwin, bxy, bt = _pct_windows(args.bent_contact_bin, W, args.fz_min)
    rwin, rxy, rt = _pct_windows(args.flat_ref_bin, W, args.fz_min)
    if len(bwin) == 0 or len(rwin) == 0:
        print("접촉 프레임 부족(fz_min 확인)"); return
    # held 곡률 δ 시간정합
    dcsv = pd.read_csv(args.delta_csv)
    delta = np.array([dcsv.loc[(dcsv["time_s"] - t).abs().idxmin(), "delta_mm"] for t in bt], float)
    # 밴딩 접촉 ↔ flat 기준 매칭(같은 위치)
    mi = _match_ref(bxy, rxy)
    keep = mi >= 0
    bwin, bxy, delta, mi = bwin[keep], bxy[keep], delta[keep], mi[keep]
    if not len(bwin):
        print("flat 기준과 매칭되는 밴딩 접촉 없음 — 같은 위치 flat 취득 확인"); return

    bt_t = torch.from_numpy(bwin).to(args.device)
    deg = torch.from_numpy(delta.astype(np.float32)).to(args.device)  # held 곡률(deg 스케일은 restorer가 정규화)
    with torch.no_grad():
        ref_maps = _sats_map(sats, torch.from_numpy(rwin[mi]).to(args.device))
        uncorr = _sats_map(sats, bt_t)
        corr = _sats_map(sats, restorer(bt_t, deg))
    r = evaluate_g2(ref_maps, uncorr, corr, bxy, delta)
    print(f"[G2 실측] n={r['n']}  loc: ref={r['loc_reference_mm']:.2f} "
          f"uncorr={r['loc_uncorrected_mm']:.2f} corr={r['loc_corrected_mm']:.2f}mm  "
          f"맵회복={r['map_recovery']*100:+.0f}%")
    for b in r["by_delta"]:
        print(f"  δ {b['delta_lo']:2d}-{b['delta_lo']+2}mm n={b['n']:4d}  "
              f"loc ref={b['loc_ref']:.2f} uncorr={b['loc_uncorrected']:.2f} corr={b['loc_corrected']:.2f}")
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        c = [b["delta_lo"] + 1 for b in r["by_delta"]]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(c, [b["loc_uncorrected"] for b in r["by_delta"]], "o-", label="uncorrected")
        ax.plot(c, [b["loc_corrected"] for b in r["by_delta"]], "s-", label="corrected")
        ax.plot(c, [b["loc_ref"] for b in r["by_delta"]], "k--", label="flat reference")
        ax.set_xlabel("|delta| (mm)"); ax.set_ylabel("contact localization error (mm)")
        ax.set_title("G2: bent+contact restoration vs curvature"); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(args.figure, dpi=120)
        print(f"  그림: {args.figure}")


if __name__ == "__main__":
    main()
