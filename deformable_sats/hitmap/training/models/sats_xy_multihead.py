import torch
import torch.nn as nn


class SATSXYMultiHead(nn.Module):
    """
    SATS 백본 + 완전 분리된 X/Y/Z 헤드.

    - 백본: 센서별 시계열을 LSTM으로 인코딩 후 self-attention.
    - Head_X: x만 예측 (y 신호를 섞지 않도록 별도 MLP)
    - Head_Y: y만 예측 (x 신호를 섞지 않도록 별도 MLP)
    - Head_Z: z (압입 깊이) 예측
    - Head_Fz: 추가로 Fz를 유지하여 기존 인터페이스 호환 (총 출력 4차원: [x, y, z, Fz])

    분리 헤드를 둬서 x↔y 누수를 최소화하고, 축별 스케일을 독립적으로 학습하도록 설계했다.
    """

    def __init__(self, n_sensors: int = 16, embed_dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.n_sensors = n_sensors

        # 1) Temporal encoder (per-sensor)
        self.lstm = nn.LSTM(input_size=1, hidden_size=embed_dim, batch_first=True)

        # 2) Spatial attention across sensors
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

        feat_dim = embed_dim * n_sensors

        # 3) Independent heads
        self.head_x = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.head_y = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.head_z = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.head_fz = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, 16)
        B, T, N = x_seq.shape
        # (B*N, T, 1)
        x_reshaped = x_seq.transpose(1, 2).reshape(B * N, T, 1)
        _, (h, _) = self.lstm(x_reshaped)
        feat = h.squeeze(0).view(B, N, -1)  # (B, 16, embed_dim)

        attn_out, _ = self.attn(feat, feat, feat)  # (B, 16, embed_dim)
        flat = attn_out.reshape(B, -1)             # (B, 16*embed_dim)

        x_pred = self.head_x(flat)
        y_pred = self.head_y(flat)
        z_pred = self.head_z(flat)
        fz_pred = self.head_fz(flat)

        out = torch.cat([x_pred, y_pred, z_pred, fz_pred], dim=-1)
        return out
