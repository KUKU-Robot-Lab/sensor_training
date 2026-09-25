"""데이터 효율/전이 리허설 분석 — 새 센서 취득량 결정 근거.

수집:
  - 학습 4종(scratch_1pair / warm_1pair / warm_2pair / crosswarm_2pair) best val RMSE
  - 참조 scratch_2pair = 기존 sizeA_ecomesh_xy1_fold3
  - zero-shot 2종: xy0.5 최종 A 모델·eco20 A 모델을 ecomesh_xy1 fold3 홀드아웃에 그대로 평가
    (xy0.5 모델 = 같은 센서·다른 프로토콜의 전이 / eco20 = 다른 소재 = 유닛 편차의 비관적 프록시)

실행: .venv/bin/python scripts/analyze_transfer_efficiency.py
산출: history/fig_data/experiments_archive/transfer_efficiency/{transfer_result.csv, transfer_efficiency.png, transfer_report.md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sats.tools.eval_diagnostics import load_cfg, _load_model  # noqa: E402
from sats.training.dataset import build_dataloaders  # noqa: E402
from sats.training.gt_gpu import BatchGPUTargetGenerator  # noqa: E402

RUNS = REPO / "sats/training/runs"
OUT = REPO / "history/fig_data/experiments_archive/transfer_efficiency"
FOLD3 = RUNS / "size_input_material/sizeA_ecomesh_xy1_fold3_e2e_g05"   # 홀드아웃 정의 재사용
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def best_val(run_dir: Path) -> float:
    h = json.loads((run_dir / "history.json").read_text())
    vals = [e.get("val_rmse") for e in h if isinstance(e, dict) and e.get("val_rmse") is not None]
    return float(min(vals))


@torch.no_grad()
def zero_shot(model_run: Path) -> float:
    """model_run 의 모델을 fold3 홀드아웃(ecomesh_xy1 t3 pair)에 평가한 rel RMSE."""
    cfg = load_cfg(FOLD3)                       # val loader 정의 = fold3 홀드아웃
    model = _load_model(model_run, load_cfg(model_run), DEVICE)
    tgen = BatchGPUTargetGenerator(cfg, DEVICE)
    _, val_loader = build_dataloaders(cfg)
    se, tms = [], []
    for sensor_b, meta_b, lengths in val_loader:
        sensor_b, meta_b, lengths = sensor_b.to(DEVICE), meta_b.to(DEVICE), lengths.to(DEVICE)
        target = tgen(meta_b)
        pred, _ = model(sensor_b, lengths, meta_b[:, 0])
        se.append(((pred - target) ** 2).mean(dim=(1, 2)).cpu().numpy())
        tms.append((target ** 2).mean(dim=(1, 2)).cpu().numpy())
    se, tms = np.concatenate(se), np.concatenate(tms)
    return float(np.sqrt(se.mean()) / np.sqrt(tms.mean()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, float]] = []   # (라벨, 종류, 값)

    rows.append(("zero-shot: xy0.5 model (protocol shift)", "zero-shot rel",
                 zero_shot(RUNS / "size_input/ecomesh_xy0p5_sizeinput_val_d5t10_d10t3")))
    rows.append(("zero-shot: eco20 model (unit-variation proxy)", "zero-shot rel",
                 zero_shot(RUNS / "size_input_material/sizeA_eco20_xy1_fold2_e2e_g05")))

    for name, run in [
        ("scratch 1-pair (2 trials)", "transfer_efficiency/scratch_1pair"),
        ("warm(xy0.5) 1-pair", "transfer_efficiency/warm_1pair"),
        ("scratch 2-pair (4 trials, ref fold3)", "size_input_material/sizeA_ecomesh_xy1_fold3_e2e_g05"),
        ("warm(xy0.5) 2-pair", "transfer_efficiency/warm_2pair"),
        ("cross-warm(eco20) 2-pair", "transfer_efficiency/crosswarm_2pair"),
        ("xy1 2-pair + 0.25mm output (81x81)", "transfer_efficiency/xy1_2pair_g025"),
    ]:
        rows.append((name, "best val rmse", best_val(RUNS / run)))

    import csv
    with open(OUT / "transfer_result.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "metric", "value"])
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [r[0] for r in rows]
    vals = [r[2] for r in rows]
    colors = ["#999" if r[1].startswith("zero") else
              ("#2ca25f" if "warm" in r[0] else "#5b8def") for r in rows]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("rel RMSE (zero-shot) / best val RMSE (trained)")
    ax.set_title("Transfer / data-efficiency rehearsal — holdout = ecomesh_xy1 d5t3+d10t3")
    fig.tight_layout()
    fig.savefig(OUT / "transfer_efficiency.png", dpi=160)

    lines = [f"| {r[0]} | {r[1]} | {r[2]:.4f} |" for r in rows]
    (OUT / "transfer_report.md").write_text(
        "# 데이터 효율/전이 리허설 결과 (새 센서 취득량 결정 근거)\n\n"
        "> 생성: `scripts/analyze_transfer_efficiency.py`. 러너: `scripts/rehearse_transfer_efficiency.sh`.\n"
        "> 홀드아웃 = ecomesh_xy1 fold3(d5 test3 + d10 test3). zero-shot 은 rel RMSE, 학습은 best val RMSE — 단위 다름 주의.\n\n"
        "| setting | metric | value |\n|---|---|---|\n" + "\n".join(lines) + "\n",
        encoding="utf-8")
    for r in rows:
        print(f"{r[0]:45s} {r[1]:14s} {r[2]:.4f}")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
