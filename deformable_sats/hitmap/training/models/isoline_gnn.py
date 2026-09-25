
import torch
import torch.nn as nn
import torch.nn.functional as F

class IsolineGNN(nn.Module):
    """
    센서 간의 인접 관계를 임베딩하여 Isoline(등압선)의 물리적 변화를 학습하는 모델.
    단순화된 형태의 Graph-like 공간 처리를 수행함.
    """
    def __init__(self, n_sensors=16, out_dim=4):
        super().__init__()
        # 센서 간 상호작용 레이어 (Node to Edge-like processing)
        self.spatial_interact = nn.Sequential(
            nn.Linear(n_sensors, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, s16, radius):
        # s16: (B, 16), radius: (B, 1)
        spatial_feat = self.spatial_interact(s16)
        combined = torch.cat([spatial_feat, radius], dim=1)
        return self.fc(combined)
