
import torch
import torch.nn as nn

class CNNBiLSTM(nn.Module):
    """
    CNN + Bidirectional LSTM 하이브리드 모델.
    CNN으로 공간 패턴을, Bi-LSTM으로 정/역방향 시간 의존성(히스테리시스)을 학습함.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), # 4x4 -> 2x2
            nn.Flatten()
        )
        # Bi-LSTM: 32*2*2 = 128 features
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, 
                            batch_first=True, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 6) # [x, y, z, Fx, Fy, Fz]
        )

    def forward(self, grid_seq):
        # grid_seq: (B, T, 1, 4, 4)
        B, T, C, H, W = grid_seq.shape
        cnn_in = grid_seq.view(B * T, C, H, W)
        cnn_out = self.cnn(cnn_in).view(B, T, -1)
        
        lstm_out, _ = self.lstm(cnn_out)
        
        # 마지막 타임스텝의 양방향 Hidden state 결합
        feat = lstm_out[:, -1, :]
        return self.fc(feat)
