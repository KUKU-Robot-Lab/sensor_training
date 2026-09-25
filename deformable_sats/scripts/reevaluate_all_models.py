"""전 모델 재평가 — rel RMSE 저force 왜곡을 걷어낸 공정 지표로 재판정 (2026-07-20).

배경: rel = rmse / target_RMS 는 저force(target 압력≈0)에서 분모가 0 → nan/폭발.
계단식 xy0.5 는 저force 정지 프레임이 압도적이라 rel 이 부당하게 나빠 보였다.
저장된 per-sample 진단(rmse·rel·fz)만으로 저force 제외 재집계(추론 불필요).

per-sample target_rms_i = rmse_i / rel_i (rel 유한한 것에서 역산) → 유의미 접촉 필터.
공정 지표:
  - rel(fair): target_rms 상위(유의미 접촉)만 median rel — 저force 분모폭발 제거
  - abs_rmse: per-sample rmse median (스케일 의존이나 분모 왜곡 없음)
  - 대조로 rel(all): 기존 방식(저force 포함) — 얼마나 왜곡됐는지 비교

실행: .venv/bin/python scripts/reevaluate_all_models.py
산출: history/fig_data/experiments_archive/reeval/{reeval_all.csv, reeval_report.md}
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "history/fig_data/experiments_archive/reeval"

# 유의미 접촉 정의: target_rms(압력 스케일) 하위 30% 제외 → 저force 분모왜곡 컷.
TARGET_RMS_PCTL = 30.0


def target_rms(rmse: np.ndarray, rel: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = rmse / rel
    return t  # rel=0/nan → inf/nan


def fair_stats(npz_path: Path) -> list[dict]:
    d = np.load(npz_path)
    rmse, rel, fz, is_d5 = d["rmse"], d["rel"], d["fz"], d["is_d5"]
    trms = target_rms(rmse, rel)
    rows = []
    for lab, mask in [("d5", is_d5), ("d10", ~is_d5)]:
        if mask.sum() < 50:
            continue
        r, e, t = rel[mask], rmse[mask], trms[mask]
        finite = np.isfinite(t) & (t > 0)
        # 유의미 접촉 = target_rms 하위 백분위 초과 (저force 분모왜곡 컷)
        if finite.sum() > 50:
            thr = np.percentile(t[finite], TARGET_RMS_PCTL)
            keep = finite & (t > thr)
        else:
            keep = finite
        rel_all = float(np.nanmedian(r[np.isfinite(r)])) if np.isfinite(r).any() else float("nan")
        rel_fair = float(np.nanmedian(r[keep])) if keep.sum() else float("nan")
        rows.append({
            "run": npz_path.stem.replace("samples_", ""),
            "diam": lab,
            "n": int(mask.sum()),
            "rel_all(구지표)": round(rel_all, 3),
            "rel_fair(저force제외)": round(rel_fair, 3),
            "abs_rmse_med": round(float(np.median(e)), 4),
            "fz_med": round(float(np.median(fz[mask])), 2),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(REPO / "history/fig_data/experiments_archive/**/samples_*.npz"),
                             recursive=True))
    all_rows = []
    for f in files:
        all_rows.extend(fair_stats(Path(f)))

    cols = ["run", "diam", "n", "rel_all(구지표)", "rel_fair(저force제외)", "abs_rmse_med", "fz_med"]
    with open(OUT / "reeval_all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(all_rows)

    # 리포트: 핵심 모델만 발췌 표
    key = {
        "sizeA_ecomesh_xy1_fold3_e2e_g05": "ecomesh xy1 (최종 소재비교)",
        "sizeA_eco20_xy1_fold2_e2e_g05": "eco20 xy1",
        "sizeA_eco50_xy1_fold1_e2e_g05": "eco50 xy1",
        "ecomesh_xy0p5_sizeinput_val_d5t10_d10t3": "ecomesh xy0.5 (최종 flat, A)",
        "ecomesh_xy0p5_datarich_val_d5test10_d10test3": "ecomesh xy0.5 (datarich, 구)",
        "d5only_beta_g0p5": "d5-only 0.5mm",
    }
    lines = ["| 모델 | 지름 | rel(구지표) | **rel(공정)** | 절대rmse | fz中 |",
             "|---|---|---|---|---|---|"]
    by = {(r["run"], r["diam"]): r for r in all_rows}
    for run, name in key.items():
        for dm in ("d5", "d10"):
            r = by.get((run, dm))
            if r:
                lines.append(f"| {name} | {dm} | {r['rel_all(구지표)']} | **{r['rel_fair(저force제외)']}** "
                             f"| {r['abs_rmse_med']} | {r['fz_med']} |")
    (OUT / "reeval_report.md").write_text(
        "# 전 모델 재평가 — rel 저force 왜곡 제거 (2026-07-20)\n\n"
        "> `scripts/reevaluate_all_models.py`. rel(공정)=target_rms 하위30%(저force 분모왜곡) 제외 후 median.\n"
        "> rel(구지표)=기존(저force 포함). 절대rmse=per-sample median(분모 왜곡 없음). 전체 CSV=reeval_all.csv.\n\n"
        + "\n".join(lines)
        + "\n\n## 핵심\n"
        "- rel(구지표)와 rel(공정)의 괴리가 큰 곳 = 저force 분모왜곡이 심했던 곳(특히 d10·xy0.5).\n"
        "- 절대rmse 로 보면 저force 샘플이 오히려 정확(작은 오차) → '나쁘다' 판정은 지표 아티팩트.\n"
        "- 위치(loc)·peak 상관 은 npz 에 없어 별도 추론 재평가 필요(다음 단계).\n",
        encoding="utf-8")
    for r in all_rows:
        print(f"{r['run'][:45]:45s} {r['diam']:3s} rel_all={r['rel_all(구지표)']!s:>6} "
              f"rel_fair={r['rel_fair(저force제외)']!s:>6} abs={r['abs_rmse_med']}")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
