# ScoreNet architecture for NCSN

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreNet(nn.Module):
    def __init__(self, channels=128, num_scales=10):
        super(ScoreNet, self).__init__()
        self.channels = channels
        self.num_scales = num_scales
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, channels // 4),
            nn.SiLU(),
            nn.Linear(channels // 4, channels // 4)
        )
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels, 3, 3, padding=1)
        
        # Group norms
        self.gn1 = nn.GroupNorm(4, channels)
        self.gn2 = nn.GroupNorm(4, channels)
        self.gn3 = nn.GroupNorm(4, channels)

    def forward(self, x, t):
        # t is noise level, shape (batch,)
        t_emb = self.time_embed(t.unsqueeze(-1))
        
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)
        
        h = self.conv2(h)
        h = self.gn2(h)
        h = F.silu(h) + t_emb.unsqueeze(-1).unsqueeze(-1)
        
        h = self.conv3(h)
        h = self.gn3(h)
        h = F.silu(h)
        
        h = self.conv4(h)
        return h