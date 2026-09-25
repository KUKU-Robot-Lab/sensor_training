
import torch
import torch.nn as nn

class MLPBaseline(nn.Module):
    """
    가장 기본적인 MLP 베이스라인 모델.
    16채널 정규화 신호와 인덴터 반지름을 입력받아 [x, y, z, Fz]를 회귀함.
    """
    def __init__(self, in_dim=17, out_dim=4, hidden=[512, 512, 256, 128]):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for h in hidden:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, radius):
        # x: (B, 16), radius: (B, 1)
        inp = torch.cat([x, radius], dim=1)
        return self.net(inp)
