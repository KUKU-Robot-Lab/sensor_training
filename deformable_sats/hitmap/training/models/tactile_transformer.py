
import torch
import torch.nn as nn
import math

class TactileTransformer(nn.Module):
    """
    Transformer 기반 촉각 추정 모델.
    센서 배열의 전역적 공간 상호작용을 Self-attention으로 모델링함.
    """
    def __init__(self, n_sensors=16, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)
        
        # 고정된 센서 위치 정보를 위한 Positional Encoding 역할을 하는 Parameter
        self.pos_encoder = nn.Parameter(torch.randn(1, n_sensors, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model * n_sensors, 256),
            nn.ReLU(),
            nn.Linear(256, 6) # [x, y, z, Fx, Fy, Fz]
        )

    def forward(self, s16):
        # s16: (B, 16)
        B = s16.shape[0]
        # (B, 16, 1) -> (B, 16, d_model)
        x = self.input_proj(s16.unsqueeze(-1))
        x = x + self.pos_encoder
        
        x = self.transformer(x) # (B, 16, d_model)
        
        return self.regressor(x.reshape(B, -1))
