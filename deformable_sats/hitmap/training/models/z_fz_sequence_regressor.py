import torch
import torch.nn as nn


class ZFzSequenceRegressor(nn.Module):
    """Predict normalized [z, Fz] from tactile sequence and xy/radius condition."""

    def __init__(self, seq_len: int = 50, cond_dim: int = 3, out_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.lstm = nn.LSTM(64 * 4 * 4, 128, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.cond = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, out_dim),
        )

    def forward(self, grid_seq: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = grid_seq.shape
        cnn_out = self.cnn(grid_seq.reshape(batch * steps, channels, height, width)).view(batch, steps, -1)
        lstm_out, _ = self.lstm(cnn_out)
        tactile_feat = self.dropout(lstm_out[:, -1, :])
        cond_feat = self.cond(condition)
        return self.head(torch.cat([tactile_feat, cond_feat], dim=1))
