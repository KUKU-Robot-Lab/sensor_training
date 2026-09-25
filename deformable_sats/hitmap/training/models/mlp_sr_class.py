import torch
import torch.nn as nn

class MLPSRClass(nn.Module):
    """
    Classification-based MLP for tactile super-resolution.
    Predicts X-class (40), Y-class (40), and Z-depth (regression).
    """
    def __init__(self, in_dim=17, hidden=[512, 512, 256, 128], x_classes=40, y_classes=40):
        super().__init__()
        
        # Shared Backbone
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        self.backbone = nn.Sequential(*layers)
        
        # Classification Heads
        self.x_head = nn.Linear(prev, x_classes)
        self.y_head = nn.Linear(prev, y_classes)
        
        # Regression Head for Z
        self.z_head = nn.Linear(prev, 1)

    def forward(self, s16, diam):
        feat = torch.cat([s16, diam], dim=1)
        shared = self.backbone(feat)
        
        x_logits = self.x_head(shared)
        y_logits = self.y_head(shared)
        z_depth = self.z_head(shared)
        
        return x_logits, y_logits, z_depth
