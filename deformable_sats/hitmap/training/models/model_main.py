"""
model_main.py

Phase 2 — 1D CNN Encoder + FiLM Conditioning + 2D UNet-lite Decoder

구조:
  Input: tactile (B, 14) 또는 depth 시퀀스 (B, K, 14), aux (B, 4)
  1D CNN Encoder:
    Conv1d(14, 64, 3) → ReLU
    Conv1d(64, 128, 3) → ReLU
    Conv1d(128, 256, 3) → ReLU
    AdaptiveAvgPool1d(1) → squeeze → (B, 256)
  FiLM Conditioning:
    cond = aux → Linear(4, 256) → (γ, β) via split(128 each)
    latent = γ * latent + β
  Reshape: (B, 256) → (B, 16, 4, 4)
  2D UNet-lite Decoder:
    up1: ConvTranspose2d(16, 64, 4,2,1)  → (B, 64,  8,  8)
    up2: ConvTranspose2d(64, 32, 4,2,1)  → (B, 32, 16, 16)
    up3: ConvTranspose2d(32, 16, 4,2,1)  → (B, 16, 32, 32)
    up4: ConvTranspose2d(16,  8, 4,2,1)  → (B,  8, 64, 64)
    out: Conv2d(8, 1, 1) → Sigmoid        → (B,  1, 64, 64)

단일 프레임 입력 시 (B, 14) → unsqueeze → (B, 1, 14)로 처리.
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    cond → γ, β → out = γ * x + β
    """

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, feature_dim * 2)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, feature_dim)
            cond: (B, cond_dim)
        """
        gb = self.to_gamma_beta(cond)              # (B, 2*feature_dim)
        gamma, beta = gb.chunk(2, dim=-1)          # (B, feature_dim) each
        return gamma * x + beta


class CNN1DEncoder(nn.Module):
    """
    1D CNN encoder for depth-axis tactile sequence.

    Input:  (B, K, C)  — K: sequence length (depth steps), C: tactile channels
    Output: (B, out_dim)
    """

    def __init__(self, in_channels: int = 14, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # (B, out_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, C) or (B, C) → (B, out_dim)"""
        if x.dim() == 2:
            x = x.unsqueeze(1)               # (B, 1, C)
        x = x.transpose(1, 2)               # (B, C, K)
        out = self.net(x)                    # (B, out_dim, 1)
        return out.squeeze(-1)               # (B, out_dim)


class UNetLiteDecoder(nn.Module):
    """
    4×4 spatial feature → 64×64 HR map (4 upsampling steps).
    """

    def __init__(self, in_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # (B, 16, 4,  4) → (B, 64, 8,  8)
            nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # → (B, 32, 16, 16)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # → (B, 16, 32, 32)
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # → (B, 8, 64, 64)
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            # → (B, 1, 64, 64)
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MainModel(nn.Module):
    """
    Phase 2: 1D CNN Encoder + FiLM + 2D UNet-lite Decoder.

    Args:
        n_tactile:  live tactile 채널 수 (기본 14)
        n_aux:      보조 feature 수 (기본 4: fx, fy, depth_mm, radius_mm)
        latent_dim: 1D CNN 출력 차원 (기본 256)
        spatial_ch: reshape 시 spatial channel (기본 16, 4×4×16=256)
    """

    SPATIAL_SIZE: int = 4

    def __init__(
        self,
        n_tactile: int = 14,
        n_aux: int = 4,
        latent_dim: int = 256,
        spatial_ch: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = CNN1DEncoder(in_channels=n_tactile, out_dim=latent_dim)
        self.film = FiLMLayer(cond_dim=n_aux, feature_dim=latent_dim)
        self.spatial_ch = spatial_ch
        self.decoder = UNetLiteDecoder(in_channels=spatial_ch)

    def forward(self, tactile: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tactile: (B, 14) 단일 프레임 또는 (B, K, 14) depth 시퀀스
            aux:     (B, 4)
        Returns:
            hr_map:  (B, 1, 64, 64)
        """
        latent = self.encoder(tactile)           # (B, latent_dim)
        latent = self.film(latent, aux)          # (B, latent_dim)

        spatial = latent.view(
            -1, self.spatial_ch, self.SPATIAL_SIZE, self.SPATIAL_SIZE
        )                                        # (B, 16, 4, 4)
        hr_map = self.decoder(spatial)           # (B, 1, 64, 64)
        return hr_map
