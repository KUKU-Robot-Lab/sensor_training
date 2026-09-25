"""BaselineRestorer — 밴딩 유발 오프셋을 예측해 flat 등가 신호를 복원.

가정: 밴딩 상태 신호 ≈ flat 접촉 신호 + 밴딩 오프셋(곡률 의존). 곡률(signed deg)을
조건으로 오프셋을 예측하고 빼서 flat 등가 신호를 만든다. 동결 SATS 입력 분포와 정합.

★ 모드(`cfg.restorer_mode`):
- **deg_only(기본)**: 오프셋 = f(deg) 만의 함수(입력 seq 비의존). §5.3 Δp_bend≈κ·k_i z_i =
  곡률×고정 공간패턴. 입력에 안 의존하므로 **접촉을 건드릴 수 없음(접촉 보존)**.
- deg_cnn: deg → 4×4 공간 seed → conv. 역시 seq 비의존(접촉 보존) + 공간 구조.
- seq_deg(레거시): (seq+deg) MLP. 표현력↑이나 무접촉 학습 시 offset≈seq 로 오학습해
  **접촉을 파괴**함이 준-합성 검증에서 드러남(eval_contact_preservation).
- **latent(각도-프리)**: deg 대신 **seq에서 저차원 잠재 변형 코드 z(k=cfg.latent_dim)**
  를 추출해 오프셋 생성. 각도 라벨 없이 임의 변형(비원호·비틀림)까지 표현하려는 모드.
  seq 의존이라 원리상 붕괴(접촉 삭제) 위험이 있으나, **정보 병목(k 작게)** 이 이를 막는다:
  변형은 전역·저주파(16채널 공통)라 k차원에 담기지만 접촉은 국소·고주파라 압축 불가.
  입력은 윈도우 시간평균([B,16])이라 시간 국소 접촉 이벤트도 희석된다.

zero-init 마지막 층 → 오프셋 0 → 복원=원신호(항등 웜스타트, 안전).
부호 있는 deg → 오프셋 방향이 밴딩 방향(양/음)에 따라 달라짐.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import BendingConfig


class BaselineRestorer(nn.Module):
    def __init__(self, cfg: BendingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = getattr(cfg, "restorer_mode", "deg_only")
        if self.mode not in ("deg_only", "seq_deg", "deg_cnn", "latent"):
            raise ValueError(
                f"restorer_mode must be deg_only/seq_deg/deg_cnn/latent, got {self.mode!r}")
        if self.mode == "latent":
            # seq(시간평균) → z[k] 정보 병목 → 4×4 공간 오프셋. deg 불필요(각도-프리).
            k = int(getattr(cfg, "latent_dim", 2))
            self.encoder = nn.Sequential(
                nn.Linear(cfg.n_sensors, cfg.mlp_hidden), nn.ReLU(),
                nn.Linear(cfg.mlp_hidden, k))
            self.seed = nn.Linear(k, 8 * 4 * 4)
            self.deconv = nn.Sequential(
                nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                nn.Conv2d(8, 1, 3, padding=1))
            nn.init.zeros_(self.deconv[-1].weight)
            nn.init.zeros_(self.deconv[-1].bias)
        elif self.mode == "deg_cnn":
            # deg → 4×4 공간 seed → conv → 4×4 오프셋. 입력(seq) 비의존 = 접촉 보존 + 공간 구조.
            self.seed = nn.Linear(1, 8 * 4 * 4)
            self.deconv = nn.Sequential(
                nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                nn.Conv2d(8, 1, 3, padding=1))
            nn.init.zeros_(self.deconv[-1].weight)
            nn.init.zeros_(self.deconv[-1].bias)
        else:
            in_dim = 1 if self.mode == "deg_only" else cfg.n_sensors + 1
            self.net = nn.Sequential(
                nn.Linear(in_dim, cfg.mlp_hidden),
                nn.ReLU(),
                nn.Linear(cfg.mlp_hidden, cfg.n_sensors),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, seq: torch.Tensor, deg: torch.Tensor | None = None) -> torch.Tensor:
        """[B, T, 16] (+ signed deg[B]) → flat 등가 [B, T, 16] = seq − 오프셋.

        latent 모드는 deg 를 쓰지 않는다(호출 호환을 위해 인자만 받음).
        """
        b, t, _ = seq.shape
        if self.mode == "latent":
            z = self.encoder(seq.mean(dim=1))                  # [B,16] 시간평균 → z[B,k]
            s = self.seed(z).view(b, 8, 4, 4)
            offset = self.deconv(s).reshape(b, 1, self.cfg.n_sensors)   # [B,1,16] broadcast
            return seq - offset
        if deg is None:
            raise ValueError(f"restorer_mode={self.mode!r} 는 deg 가 필요합니다")
        deg_n = deg / float(self.cfg.deg_scale)
        if self.mode == "deg_cnn":
            s = self.seed(deg_n.view(b, 1)).view(b, 8, 4, 4)
            offset = self.deconv(s).reshape(b, 1, self.cfg.n_sensors)  # [B,1,16] broadcast (seq 비의존)
        elif self.mode == "deg_only":
            offset = self.net(deg_n.view(b, 1)).unsqueeze(1)   # [B,1,16] broadcast (seq 비의존)
        else:
            dg = deg_n.view(b, 1, 1).expand(b, t, 1)
            offset = self.net(torch.cat([seq, dg], dim=-1))    # [B, T, 16]
        return seq - offset
