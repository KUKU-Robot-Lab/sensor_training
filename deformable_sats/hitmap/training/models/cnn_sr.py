
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNSR(nn.Module):
    """
    4x4 센서 그리드의 공간적 특징을 추출하는 CNN 기반 Super-Resolution 모델.
    """
    def __init__(self, in_channels=1, out_dim=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        # 64 channels * 4 * 4 grid = 1024
        self.fc = nn.Sequential(
            nn.Linear(1024 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, grid, radius):
        # grid: (B, 1, 4, 4), radius: (B, 1)
        feat = self.conv(grid)
        combined = torch.cat([feat, radius], dim=1)
        return self.fc(combined)
