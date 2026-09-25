
import torch
import torch.nn as nn

class SATSModel(nn.Module):
    """
    Self-Attention-Assisted Tactile SR (SATS) 모델.
    각 센서의 시계열 특징을 LSTM으로 추출하고, Self-Attention으로 센서 간 공간 상호작용을 학습함.
    """
    def __init__(self, n_sensors=16, embed_dim=64, n_heads=4):
        super().__init__()
        self.n_sensors = n_sensors
        # 1. Temporal Encoder (Per-sensor LSTM)
        self.lstm = nn.LSTM(input_size=1, hidden_size=embed_dim, batch_first=True)
        # 2. Self-Attention Module
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        # 3. MLP Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * n_sensors, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4) # [x, y, z, Fz]
        )

    def forward(self, x_seq):
        # x_seq: (B, T, 16)
        B, T, N = x_seq.shape
        # (B*N, T, 1) 로 변환하여 각 센서별 시계열 처리
        x_reshaped = x_seq.transpose(1, 2).reshape(B * N, T, 1)
        _, (h, _) = self.lstm(x_reshaped)
        feat = h.squeeze(0).view(B, N, -1) # (B, 16, embed_dim)
        
        # Spatial Attention (X-Y축 비대칭성 및 전역적 상관관계 학습)
        attn_out, _ = self.attn(feat, feat, feat)
        
        # 최종 회귀
        out = self.decoder(attn_out.reshape(B, -1))
        return out
