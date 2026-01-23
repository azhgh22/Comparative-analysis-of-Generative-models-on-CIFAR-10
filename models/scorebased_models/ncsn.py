import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from base.base_model import BaseModel
from utils.get_device import get_device


class RefineNet(nn.Module):
    """RefineNet block for NCSN"""
    def __init__(self, in_channels, out_channels):
        super(RefineNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(32, out_channels // 4), out_channels)
        self.gn2 = nn.GroupNorm(min(32, out_channels // 4), out_channels)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.gn1(h)
        h = nn.functional.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = nn.functional.silu(h)
        return h + self.skip(x)

class ScoreNet(nn.Module):
    """Score Network for NCSN - predicts score function (gradient of log density)"""
    def __init__(self, channels=128, num_scales=10, image_size=32, in_channels=3):
        super(ScoreNet, self).__init__()
        self.channels = channels
        self.num_scales = num_scales
        self.image_size = image_size

        # Noise level embedding
        self.noise_level_embed = nn.Sequential(
            nn.Linear(1, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )

        # Encoder (downsampling path)
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)

        self.down1 = RefineNet(channels, channels)
        self.down2 = RefineNet(channels, channels * 2)
        self.down3 = RefineNet(channels * 2, channels * 2)

        # Middle
        self.middle = RefineNet(channels * 2, channels * 2)

        # Decoder (upsampling path)
        self.up1 = RefineNet(channels * 2, channels * 2)
        self.up2 = RefineNet(channels * 2, channels)
        self.up3 = RefineNet(channels, channels)

        # Output
        self.conv_out = nn.Conv2d(channels, in_channels, 3, padding=1)

        # Pooling and upsampling
        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, sigma):
        """
        Args:
            x: Input image batch (B, C, H, W)
            sigma: Noise level (B,) or (B, 1)

        Returns:
            score: Predicted score (gradient of log density) (B, C, H, W)
        """
        # Embed noise level
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        sigma_embed = self.noise_level_embed(sigma)  # (B, channels)
        sigma_embed = sigma_embed.view(sigma_embed.shape[0], -1, 1, 1)  # (B, channels, 1, 1)

        # Encoder
        h = self.conv_in(x)
        h = h + sigma_embed

        h1 = self.down1(h)
        h = self.pool(h1)

        h2 = self.down2(h)
        h = self.pool(h2)

        h3 = self.down3(h)
        h = self.pool(h3)

        # Middle
        h = self.middle(h)

        # Decoder
        h = self.upsample(h)
        h = self.up1(h + h3)

        h = self.upsample(h)
        h = self.up2(h + h2)

        h = self.upsample(h)
        h = self.up3(h + h1)

        # Output
        score = self.conv_out(h)

        return score

class NCSN(BaseModel):
    """
    Noise Conditional Score Network
    Compatible with the Train class interface
    """
    def __init__(self,
                 config=None,
                 config_path='configs/ncsn.yaml',
                 channels: int = None,
                 num_scales: int = None,
                 image_size: int = 32,
                 in_channels: int = 3,
                 lr: float = None,
                 sigma_low: float = None,
                 sigma_high: float = None,
                 device: str = None):
        super(NCSN, self).__init__()

        self.device = device if device is not None else get_device()

        # Load config from file if config dict not provided
        if config is None:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except FileNotFoundError:
                print(f"Config file {config_path} not found. Using default parameters.")
                config = {}

        # Extract parameters from config, with fallback to direct parameters or defaults
        self.channels = channels if channels is not None else config.get('channels', 128)
        self.num_scales = num_scales if num_scales is not None else config.get('num_scales', 10)
        self.image_size = image_size
        self.in_channels = in_channels
        self.lr = lr if lr is not None else config.get('lr', 1e-4)
        self.device = device

        # Handle sigmas from config
        sigmas_config = config.get('sigmas', [0.01, 1.0])
        if sigma_low is None:
            sigma_low = sigmas_config[0] if isinstance(sigmas_config, list) else 0.01
        if sigma_high is None:
            sigma_high = sigmas_config[-1] if isinstance(sigmas_config, list) else 1.0

        # Create geometric progression of noise levels
        self.sigmas = torch.exp(torch.linspace(
            np.log(sigma_high),
            np.log(sigma_low),
            self.num_scales
        )).to(device)

        # Initialize score network
        self.score_net = ScoreNet(self.channels, self.num_scales, self.image_size, in_channels)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.score_net.parameters(), lr=self.lr)

        # Move to device
        self.to(device)

    def forward(self, x, sigma):
        """Forward pass through score network"""
        return self.score_net(x, sigma)

    def train_step(self, x, epoch):
        """
        Perform one training step

        Args:
            x: Input batch (B, C, H, W)
            epoch: Current epoch number

        Returns:
            Dictionary of losses
        """
        self.score_net.train()
        self.optimizer.zero_grad()

        # Sample random noise level for each sample in batch
        batch_size = x.shape[0]
        sigma_idx = torch.randint(0, self.num_scales, (batch_size,), device=self.device)
        sigma = self.sigmas[sigma_idx]

        # Compute loss
        loss = self.compute_loss(x, sigma)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return {'total_loss': loss.item()}

    def compute_loss(self, x, sigma):
        """
        Compute denoising score matching loss

        Args:
            x: Clean data (B, C, H, W)
            sigma: Noise levels (B,)

        Returns:
            Loss value
        """
        # Add noise to data
        noise = torch.randn_like(x)
        perturbed_x = x + sigma.view(-1, 1, 1, 1) * noise

        # Predict score
        predicted_score = self.score_net(perturbed_x, sigma)

        # Target is -noise/sigma (gradient of log p(perturbed_x | x))
        target = -noise / sigma.view(-1, 1, 1, 1)

        # Compute L2 loss weighted by sigma^2
        loss = torch.mean(
            sigma.view(-1, 1, 1, 1) ** 2 *
            torch.sum((predicted_score - target) ** 2, dim=[1, 2, 3])
        )

        return loss

    def epoch_step(self):
        """Called at the end of each epoch"""
        pass  # Can add learning rate scheduling here if needed

    def get_init_loss_dict(self):
        """Return initial loss dictionary structure"""
        return {'total_loss': 0.0}

    def get_model_state(self, epoch):
        """
        Get model state for checkpointing

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing model state
        """
        return {
            'epoch': epoch,
            'score_net_state_dict': self.score_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'sigmas': self.sigmas.cpu(),
            'config': {
                'channels': self.channels,
                'num_scales': self.num_scales,
                'image_size': self.image_size,
                'in_channels': self.in_channels,
                'lr': self.lr
            }
        }

    def load_state(self, model_state):
        """
        Load model state from checkpoint

        Args:
            model_state: Dictionary containing model state
        """
        self.score_net.load_state_dict(model_state['score_net_state_dict'])
        self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        self.sigmas = model_state['sigmas'].to(self.device)

    def sample(self, batch_size: int, num_steps: int = 100):
        """
        Generate samples using Langevin dynamics

        Args:
            batch_size: Number of samples to generate
            num_steps: Number of Langevin steps per noise level

        Returns:
            Generated samples (batch_size, C, H, W)
        """
        self.score_net.eval()

        # Start from random noise
        x = torch.randn(batch_size, self.in_channels, self.image_size, self.image_size).to(self.device)

        with torch.no_grad():
            # Iterate through noise levels from high to low
            for sigma in self.sigmas:
                sigma_batch = sigma.repeat(batch_size)
                step_size = sigma ** 2 * 2e-5

                for _ in range(num_steps):
                    # Predict score
                    score = self.score_net(x, sigma_batch)

                    # Langevin dynamics step
                    noise = torch.randn_like(x)
                    x = x + step_size * score + torch.sqrt(2 * step_size) * noise

        # Clip to valid range
        x = torch.clamp(x, 0, 1)

        return x

