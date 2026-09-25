"""BendingEstimator — 센서 시계열에서 밴딩 곡률(signed deg)을 추정.

구조: 공유 LSTM(16채널 시계열 인코딩, 이력/드리프트 포착) → 마지막 유효 hidden →
MLP head → **부호 있는** 스칼라 deg. 양/음 방향을 모두 표현하도록 출력에 활성화 없음.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import BendingConfig


class BendingEstimator(nn.Module):
    def __init__(self, cfg: BendingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.n_sensors,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, 1),   # signed deg (활성화 없음)
        )

    def _last_hidden(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(seq)                      # [B, T, H]
        idx = (lengths - 1).clamp(min=0)
        b = torch.arange(seq.shape[0], device=seq.device)
        return out[b, idx]                           # [B, H]

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """[B, T, 16] → signed deg [B]. deg_scale로 정규화된 회귀."""
        h = self._last_hidden(seq, lengths)
        deg = self.head(h).squeeze(-1) * float(self.cfg.deg_scale)
        return deg


class CNN2DEstimator(nn.Module):
    """프레임별 4×4 공간 CNN → 윈도우 평균 → signed deg.

    16 taxel을 물리 배치(4×4, row=y·col=x)로 보고 2D conv로 **밴딩 공간 gradient**를 포착.
    시계열은 평균 집약(정적 hold에 강건 — LSTM처럼 램프 전이에 과의존 안 함).
    """

    def __init__(self, cfg: BendingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                       # [.,32,1,1]
        )
        self.head = nn.Sequential(
            nn.Linear(32, cfg.mlp_hidden), nn.ReLU(), nn.Linear(cfg.mlp_hidden, 1))

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, t, _ = seq.shape
        x = seq
        if getattr(self, "shape_norm", False):
            # 프레임별 크기 정규화(shape만) → 응력완화(크기 감소)에 불변 기대.
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        g = x.reshape(b * t, 1, 4, 4)                      # S1..16 → row=y,col=x
        f = self.conv(g).reshape(b, t, 32).mean(dim=1)     # 윈도우 평균 집약
        return self.head(f).squeeze(-1) * float(self.cfg.deg_scale)


class CNNLSTMEstimator(nn.Module):
    """프레임별 4×4 CNN(공간 gradient) → LSTM(시계열 이력) → MLP head.

    cnn2d(공간)+lstm(시계열) 결합. 밴딩 공간패턴을 프레임마다 인코딩한 뒤 시퀀스로 LSTM.
    """

    def __init__(self, cfg: BendingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                       # [.,32,1,1]
        )
        self.lstm = nn.LSTM(32, cfg.lstm_hidden, cfg.lstm_layers, batch_first=True,
                            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden, cfg.mlp_hidden), nn.ReLU(), nn.Linear(cfg.mlp_hidden, 1))

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, t, _ = seq.shape
        f = self.conv(seq.reshape(b * t, 1, 4, 4)).reshape(b, t, 32)   # 프레임별 공간특징 시계열
        out, _ = self.lstm(f)
        idx = (lengths - 1).clamp(min=0)
        bi = torch.arange(b, device=seq.device)
        return self.head(out[bi, idx]).squeeze(-1) * float(self.cfg.deg_scale)


class MLPFrameEstimator(nn.Module):
    """프레임별 MLP(16→h→h→1) → 윈도우 평균. 시계열·공간구조 없는 최소 기준선."""

    def __init__(self, cfg: BendingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.n_sensors, cfg.mlp_hidden), nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden), nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, 1))

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.net(seq).squeeze(-1).mean(dim=1) * float(self.cfg.deg_scale)


def build_estimator(cfg: BendingConfig) -> nn.Module:
    """cfg.estimator_arch 로 구조 선택(체크포인트 cfg에 저장되어 로드 시 자동 복원)."""
    arch = getattr(cfg, "estimator_arch", "lstm")
    if arch == "cnn2d_shape":
        m = CNN2DEstimator(cfg)
        m.shape_norm = True                    # 프레임 크기 정규화(완화 불변 실험)
        return m
    table = {"lstm": BendingEstimator, "cnn2d": CNN2DEstimator,
             "cnn_lstm": CNNLSTMEstimator, "mlp_frame": MLPFrameEstimator}
    if arch not in table:
        raise ValueError(f"estimator_arch must be one of {list(table)}+cnn2d_shape, got {arch!r}")
    return table[arch](cfg)
