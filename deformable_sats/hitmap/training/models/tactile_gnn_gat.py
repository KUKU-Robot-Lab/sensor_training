
import torch
import torch.nn as nn
import torch.nn.functional as F

class TactileGAT(nn.Module):
    """
    Graph Attention Network (GAT) 기반 촉각 모델.
    센서 간의 관계를 그래프로 정의하고, 동적인 가중치를 통해 공간적 특징을 집계함.
    """
    def __init__(self, in_feat=1, out_feat=32, n_heads=2):
        super().__init__()
        self.out_feat = out_feat
        self.n_heads = n_heads
        
        # GAT 파라미터
        self.w = nn.Linear(in_feat, out_feat * n_heads, bias=False)
        # Per-head attention vector for concatenated pair feature (2*out_feat)
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_feat))
        nn.init.xavier_uniform_(self.a, gain=1.414)
        
        self.fc = nn.Sequential(
            nn.Linear(out_feat * n_heads * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 6) # [x, y, z, Fx, Fy, Fz]
        )

    def forward(self, s16):
        B = s16.shape[0]
        # (B, 16, 1) -> (B, 16, heads * out_feat)
        h = self.w(s16.unsqueeze(-1)).view(B, 16, self.n_heads, self.out_feat)
        
        # Attention Mechanism (Dense Graph: All sensors connected)
        # (B, 16, 16, n_heads, 2*out_feat)
        h_i = h.unsqueeze(2).repeat(1, 1, 16, 1, 1)
        h_j = h.unsqueeze(1).repeat(1, 16, 1, 1, 1)
        combined = torch.cat([h_i, h_j], dim=-1).transpose(1, 3) # (B, n_heads, 16, 16, 2*out_feat)
        
        # Energy: (B, n_heads, 16, 16)
        # combined: (B, H, 16, 16, 2F), a: (H, 2F)
        e = F.leaky_relu(torch.einsum("bhijc,hc->bhij", combined, self.a))
        attention = F.softmax(e, dim=-1)
        
        # Aggregate: (B, n_heads, 16, out_feat)
        h_prime = torch.matmul(attention, h.transpose(1, 2))
        
        return self.fc(h_prime.reshape(B, -1))
