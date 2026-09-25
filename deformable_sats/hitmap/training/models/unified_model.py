
import torch
import torch.nn as nn
import torch.nn.functional as F

class IsolineMLP(nn.Module):
    """
    Isoline Theory based MLP branch.
    Handles drift, temperature, and other meta features to provide physical corrections.
    """
    def __init__(self, input_dim=19, hidden_dim=64, output_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (B, T, input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Temporal aggregation: taking mean over sequence
        x = torch.mean(x, dim=1)  # (B, output_dim)
        return x

class CNNEncoder(nn.Module):
    """
    Spatial 4x4 grid CNN Encoder.
    Extracts spatial features from 16-channel sensor array.
    """
    def __init__(self, in_channels=1, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 32 channels * 4 * 4 grid = 512
        self.fc = nn.Linear(32 * 4 * 4, out_dim)

    def forward(self, x):
        # x shape: (B*T, 1, 4, 4)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class UnifiedSensorModel(nn.Module):
    """
    Unified Multi-head Architecture: CNN + LSTM + Isoline MLP.
    
    Branch 1: SR Localization (x, y, z) + Force Distribution (fz_map)
    Branch 2: 3-Axis Force Vector (Fx, Fy, Fz) at center point
    """
    def __init__(self, use_branch1=True, use_branch2=True, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        self.use_branch1 = use_branch1
        self.use_branch2 = use_branch2

        # 1. CNN-LSTM Backbone
        self.encoder = CNNEncoder(in_channels=1, out_dim=64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        
        # 2. Isoline Branch (Physics-informed corrections)
        self.isoline = IsolineMLP(input_dim=19, hidden_dim=64, output_dim=32)

        fused_dim = 128 + 32 # LSTM hidden + Isoline out

        # 3. Branch 1: Position & Distribution
        if use_branch1:
            self.fc1_shared = nn.Linear(fused_dim, 256)
            self.fc1_xyz = nn.Linear(256, 3)          # [x, y, z]
            self.fc1_map = nn.Linear(256, 25 * 25)    # 25x25 force distribution map

        # 4. Branch 2: 3-Axis Force Vector
        if use_branch2:
            self.fc2_shared = nn.Linear(fused_dim, 128)
            self.fc2_force = nn.Linear(128, 6)        # [x, y, z, Fx, Fy, Fz]

    def forward(self, x_grid, x_iso):
        """
        Args:
            x_grid: (B, T, 1, 4, 4) - 4x4 sensor array sequence
            x_iso:  (B, T, 19)      - Isoline features (16-ch drift + 3-meta)
        Returns:
            z1 (dict): {xyz, fz_map} if use_branch1
            z2 (tensor): [x, y, z, Fx, Fy, Fz] if use_branch2
        """
        B, T, C, H, W = x_grid.shape
        
        # 1. Feature Extraction (CNN)
        x_flat = x_grid.view(B * T, C, H, W)
        cnn_feat = self.encoder(x_flat)
        cnn_feat = cnn_feat.view(B, T, -1)
        
        # 2. Temporal Processing (LSTM)
        lstm_out, _ = self.lstm(cnn_feat)
        lstm_feat = lstm_out[:, -1, :] # Take last state
        
        # 3. Isoline processing
        iso_feat = self.isoline(x_iso)
        
        # 4. Feature Fusion
        fused = torch.cat([lstm_feat, iso_feat], dim=-1)

        res1, res2 = None, None
        
        # 5. Output Branches
        if self.use_branch1:
            f1 = F.relu(self.fc1_shared(fused))
            xyz = self.fc1_xyz(f1)
            fz_map = self.fc1_map(f1).view(B, 1, 25, 25)
            res1 = {"xyz": xyz, "fz_map": fz_map}
            
        if self.use_branch2:
            f2 = F.relu(self.fc2_shared(fused))
            res2 = self.fc2_force(f2)
            
        return res1, res2
