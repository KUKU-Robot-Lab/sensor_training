
import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, output_size=6):
        super(CNN_LSTM, self).__init__()
        # CNN Part: 4x4 이미지를 받아 특징 추출
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # CNN 출력 크기 계산: 16 channels * 4 * 4 = 256
        cnn_output_size = 16 * 4 * 4

        # LSTM Part: CNN 특징의 시계열을 받아 시간적 패턴 학습
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=64, num_layers=1, batch_first=True)
        
        # Fully Connected Part: 최종 예측
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, 1, 4, 4)
        batch_size, seq_len, c, h, w = x.size()
        
        # 각 시점(time step)의 이미지를 CNN에 통과
        c_in = x.view(batch_size * seq_len, c, h, w)
        c_out = self.cnn(c_in)
        
        # LSTM 입력을 위해 shape 변경
        r_in = c_out.view(batch_size, seq_len, -1)
        
        # LSTM 통과
        lstm_out, _ = self.lstm(r_in)
        
        # 마지막 시점의 LSTM 출력만 사용
        last_lstm_out = lstm_out[:, -1, :]
        
        # 최종 예측
        output = self.fc(last_lstm_out)
        return output
