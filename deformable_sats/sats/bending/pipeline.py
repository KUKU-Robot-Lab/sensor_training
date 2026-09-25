"""BendingPipeline — [BendingEstimator → BaselineRestorer → ❄️Frozen SATS].

밴딩 상태 신호를 받아 곡률(signed deg)을 추정하고 flat 등가로 복원한 뒤,
**동결된** 사전학습 SATS에 넣어 압력맵을 추론한다. SATS는 재학습되지 않는다
(밴딩 모듈만 학습). 이로써 flat 학습 SATS로 밴딩 상황에서도 추론 가능.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from sats.training.cnn_module import SATSCNNStage
from sats.training.config import SATSConfig

from .baseline_restorer import BaselineRestorer
from .bending_estimator import BendingEstimator
from .config import BendingConfig


def load_frozen_sats(run_dir: str | Path, device: str = "cpu") -> SATSCNNStage:
    """run_dir/config.json + best_model.pt 로 SATS를 로드하고 동결(eval)한다."""
    import json
    from dataclasses import fields

    run_dir = Path(run_dir)
    data = json.loads((run_dir / "config.json").read_text())
    valid = {f.name for f in fields(SATSConfig)}
    scfg = SATSConfig(**{k: v for k, v in data.items() if k in valid})
    model = SATSCNNStage(scfg).to(device).eval()
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt.get("state_dict", ckpt)))
    model.load_state_dict(state)
    return model


class BendingPipeline(nn.Module):
    def __init__(self, cfg: BendingConfig, sats: SATSCNNStage) -> None:
        super().__init__()
        self.cfg = cfg
        self.estimator = BendingEstimator(cfg)
        self.restorer = BaselineRestorer(cfg)
        self.sats = sats
        # SATS 동결: 밴딩 모듈만 학습되고 SATS 가중치는 불변.
        self.sats.eval()
        for p in self.sats.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        bent_seq: torch.Tensor,
        lengths: torch.Tensor,
        size: torch.Tensor | None = None,
    ):
        """밴딩 신호[B,T,16] (+인덴터 지름 size[B]) → (signed deg[B], pressure_map[B,grid,grid]).

        size 는 동결 SATS 가 크기입력(A, use_indenter_size_input) 모델이면 필수 —
        누락 시 FiLM 이 통째로 건너뛰어져 학습 분포와 다른 잘못된 추론이 된다.
        """
        deg = self.estimator(bent_seq, lengths)
        restored = self.restorer(bent_seq, deg)
        if getattr(self.sats, "use_size_input", False) and size is None:
            raise ValueError(
                "frozen SATS uses indenter-size input(A); pass size=[B] diameters (mm)"
            )
        # SATS 파라미터는 동결(requires_grad=False)이라 갱신 안 되지만, 입력(restored)
        # 경로로 gradient가 통과해 restorer의 end-to-end 학습이 가능하다.
        pmap, _ = self.sats(restored, lengths, size)
        return deg, pmap
