#!/usr/bin/env python3
"""S19 ablation 학습 — ecomesh_xy1 fold3 설정 재사용, 모듈별 제거 변형 재학습.

full(SATS) = 기존 run 재사용, 여기서는 noLSTM/noAttention/noCNN 3개만 학습.
출력: sats/training/runs/ablation_ecomesh/{variant}/
"""
from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

from sats.training.config import SATSConfig
from sats.training.train_e2e import train

BASE_CFG = Path("sats/training/runs/xy1_material_d5d10/xy1_d5d10_ecomesh_xy1_fold3_e2e_g05/config.json")
OUT_DIR = "sats/training/runs/ablation_ecomesh"

VARIANTS = {
    "noLSTM": {"ablate_lstm": True},
    "noAttention": {"ablate_attention": True},
    "noCNN": {"ablate_cnn": True},
}


def load_base() -> SATSConfig:
    data = json.loads(BASE_CFG.read_text())
    valid = {f.name for f in fields(SATSConfig)}
    return SATSConfig(**{k: v for k, v in data.items() if k in valid})


def main() -> None:
    base = load_base()
    for name, flags in VARIANTS.items():
        cfg = replace(base, out_dir=OUT_DIR, run_name=name, **flags)
        print(f"\n===== training ablation: {name}  {flags} =====", flush=True)
        train(cfg)
    print("\nALL ABLATIONS DONE", flush=True)


if __name__ == "__main__":
    main()
