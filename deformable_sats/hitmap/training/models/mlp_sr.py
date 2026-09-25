"""
mlp_sr.py

MLP 기준 SR 모델.
  Input : s16(16) + diam(1) = 17-dim
  Output: x_mm, y_mm, z_depth_mm, fz  (4-dim)
"""

import torch
import torch.nn as nn


class MLPSR(nn.Module):
    """
    Multi-Layer Perceptron for tactile super-resolution.

    Args:
        in_dim  : 입력 차원 (16 channels + 1 diameter = 17)
        hidden  : 은닉층 크기 리스트
        out_dim : 출력 차원 (x, y, z, fz = 4)
        dropout : Dropout 비율 (마지막 hidden 제외)
    """

    def __init__(
        self,
        in_dim: int = 17,
        hidden: list = None,
        out_dim: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden is None:
            hidden = [256, 256, 128, 64]

        layers = []
        prev = in_dim
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            # 마지막 hidden layer 전까지만 Dropout 적용
            if dropout > 0 and i < len(hidden) - 1:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s16: torch.Tensor, diam: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s16  : (B, 16) s_norm
            diam : (B, 1)  diameter_norm
        Returns:
            (B, 4) [x_mm, y_mm, z_depth_mm, fz] (normalized scale)
        """
        x = torch.cat([s16, diam], dim=1)  # (B, 17)
        return self.net(x)
