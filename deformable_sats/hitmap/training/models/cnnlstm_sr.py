
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMSR(nn.Module):
    """
    CNN을 통한 공간 정보와 LSTM을 통한 시간 정보(히스테리시스 보정)를 결합한 모델.
    """
    def __init__(self, in_channels=1, hidden_dim=128, out_dim=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # 32 * 4 * 4 = 512
        self.lstm = nn.LSTM(input_size=512 + 1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, grid_seq, radius_seq):
        # grid_seq: (B, T, 1, 4, 4), radius_seq: (B, T, 1)
        B, T, C, H, W = grid_seq.shape
        cnn_in = grid_seq.view(B * T, C, H, W)
        cnn_out = self.cnn(cnn_in).view(B, T, -1)
        
        combined = torch.cat([cnn_out, radius_seq], dim=-1)
        lstm_out, _ = self.lstm(combined)
        
        # 마지막 타임스텝의 정보로 결과 예측
        last_feat = lstm_out[:, -1, :]
        return self.fc(last_feat)
