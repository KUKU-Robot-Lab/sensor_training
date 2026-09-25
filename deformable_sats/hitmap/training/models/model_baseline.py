"""
model_baseline.py

Phase 1 — MLP Encoder + CNN Decoder

구조:
  Input: tactile (B, 14), aux (B, 4)
  MLP Encoder:
    Linear(14, 128) → BN → ReLU
    Linear(128, 256) → BN → ReLU
    Linear(256, 128) → BN → ReLU  → latent (128,)
  Conditioning:
    Linear(4, 32) → ReLU → aux_enc (32,)
    cat(latent, aux_enc) → Linear(160, 512) → reshape (B, 8, 8, 8)
  CNN Decoder (8×8 → 64×64, 3 upsampling steps):
    ConvTranspose2d(8, 64, 4,2,1)  → BN → ReLU  → (B, 64, 16, 16)
    ConvTranspose2d(64, 32, 4,2,1) → BN → ReLU  → (B, 32, 32, 32)
    ConvTranspose2d(32, 16, 4,2,1) → BN → ReLU  → (B, 16, 64, 64)
    Conv2d(16, 1, 3,1,1) → Sigmoid               → (B,  1, 64, 64)
Output: hr_map (B, 1, 64, 64)
"""

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, n_tactile: int = 14, latent_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_tactile, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, latent_dim)


class AuxEncoder(nn.Module):
    def __init__(self, n_aux: int = 4, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_aux, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, out_dim)


class CNNDecoder(nn.Module):
    """8×8 spatial feature → 64×64 HR map."""

    def __init__(self, in_channels: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # (B, 8, 8, 8) → (B, 64, 16, 16)
            nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # → (B, 32, 32, 32)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # → (B, 16, 64, 64)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # → (B, 1, 64, 64)
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 1, 64, 64)


class BaselineModel(nn.Module):
    """
    Phase 1 baseline: MLP encoder + conditioning + CNN decoder.

    Args:
        n_tactile:  live tactile 채널 수 (dead ch 제거 후, 기본 14)
        n_aux:      보조 feature 수 (기본 4: fx, fy, depth_mm, radius_mm)
        latent_dim: MLP encoder 출력 차원
        spatial_ch: reshape 시 spatial channel 수 (spatial_size = 8 고정)
    """

    SPATIAL_SIZE: int = 8  # 8×8 spatial feature map

    def __init__(
        self,
        n_tactile: int = 14,
        n_aux: int = 4,
        latent_dim: int = 128,
        spatial_ch: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = MLPEncoder(n_tactile, latent_dim)
        self.aux_enc = AuxEncoder(n_aux, out_dim=32)

        fused_dim = latent_dim + 32
        spatial_total = spatial_ch * self.SPATIAL_SIZE * self.SPATIAL_SIZE  # 8*8*8=512
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, spatial_total),
            nn.ReLU(inplace=True),
        )
        self.spatial_ch = spatial_ch
        self.decoder = CNNDecoder(in_channels=spatial_ch)

    def forward(self, tactile: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tactile: (B, n_tactile)
            aux:     (B, n_aux)
        Returns:
            hr_map:  (B, 1, 64, 64)
        """
        latent = self.encoder(tactile)                     # (B, 128)
        aux_enc = self.aux_enc(aux)                        # (B, 32)
        fused = torch.cat([latent, aux_enc], dim=1)        # (B, 160)
        spatial = self.fuse(fused)                         # (B, 512)
        spatial = spatial.view(
            -1, self.spatial_ch, self.SPATIAL_SIZE, self.SPATIAL_SIZE
        )                                                  # (B, 8, 8, 8)
        hr_map = self.decoder(spatial)                     # (B, 1, 64, 64)
        return hr_map
