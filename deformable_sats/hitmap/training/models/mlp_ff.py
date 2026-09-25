"""
mlp_ff.py

Force Field 예측을 위한 MLP 모델.
Input: s1..16 (16) + x, y, z_depth (3) + radius (1) = 20-dim
Output: fz_bc (1-dim)
"""

import torch
import torch.nn as nn


class MLPFF(nn.Module):
    """
    Multi-Layer Perceptron for Force Field (Fz) prediction.
    """

    def __init__(
        self,
        in_dim: int = 20,
        hidden: list = None,
        out_dim: int = 1,
        dropout: float = 0.1,
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
            if dropout > 0 and i < len(hidden) - 1:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, tactile: torch.Tensor, radius: torch.Tensor, sr_pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tactile : (B, 16) s_norm
            radius  : (B, 1)  radius_mm
            sr_pos  : (B, 3)  [x, y, z_depth]
        Returns:
            (B, 1) fz_bc
        """
        x = torch.cat([tactile, radius, sr_pos], dim=1)  # (B, 20)
        return self.net(x)
