
import torch
import torch.nn as nn

class SuperTacVAE(nn.Module):
    """
    VAE + Residual Upsampling 기반 Super-resolution 모델.
    저해상도(4x4) 센서 입력을 잠재 공간(Latent Space)을 거쳐 고해상도(32x32) 압력 맵으로 복원함.
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder: 4x4 -> Latent
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*4*4, latent_dim * 2) # mean and logvar
        )
        
        # Decoder Input
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Residual-In-Residual 스타일 Upsampling: 4x4 -> 32x32
        self.upsample = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid() # Normalize to [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, grid):
        # grid: (B, 1, 4, 4)
        h = self.encoder(grid)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar) if self.training else mu
        
        feat = self.decoder_input(z).view(-1, 256, 4, 4)
        hr_map = self.upsample(feat)
        return hr_map, mu, logvar
