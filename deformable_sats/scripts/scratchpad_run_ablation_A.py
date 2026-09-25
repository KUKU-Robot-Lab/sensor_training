#!/usr/bin/env python3
"""S19 ablation을 A(indenter-size input) 베이스로 재학습.

full 모델 = sizeA_ecomesh_xy1_fold3(이미 있음). 여기서는 noLSTM/noAttention/noCNN
3변형을 동일 A 설정(use_indenter_size_input=True)으로 재학습해 S19를 최종 모델과 일치시킨다.
"""
import json
from dataclasses import fields, replace
from pathlib import Path

from sats.training.config import SATSConfig
from sats.training.train_e2e import train

BASE_CFG = Path("sats/training/runs/size_input_material/sizeA_ecomesh_xy1_fold3_e2e_g05/config.json")
OUT_DIR = "sats/training/runs/ablation_ecomesh_A"

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
    assert bool(getattr(base, "use_indenter_size_input", False)), "base must be A model"
    for name, flags in VARIANTS.items():
        cfg = replace(base, out_dir=OUT_DIR, run_name=name, **flags)
        print(f"\n===== training ablation(A): {name}  {flags} =====", flush=True)
        train(cfg)
    print("\nALL A-ABLATIONS DONE", flush=True)


if __name__ == "__main__":
    main()
