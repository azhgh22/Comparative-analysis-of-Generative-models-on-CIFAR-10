import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import yaml
from base.base_model import BaseModel
from utils.get_device import get_device


class ConditionalInstanceNorm2d(nn.Module):
    """
    Conditional Instance Normalization
    Normalizes the input and then applies a scale and shift computed from the noise level sigma.
    """

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.inst_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)

        # Embed the noise level index/value into scale (gamma) and shift (beta)
        # The paper typically maps embedding -> 2 * num_features
        self.embed = nn.Linear(num_classes, num_features * 2)

        # Initialize weights to effectively perform identity at start
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initial scale ~1
        self.embed.weight.data[:, num_features:].zero_()  # Initial shift 0

    def forward(self, x, sigma_emb):
        # x: (B, C, H, W)
        # sigma_emb: (B, embed_dim)

        out = self.inst_norm(x)

        # Get gamma and beta for the current noise level
        style = self.embed(sigma_emb)  # (B, 2*C)
        style = style.view(style.shape[0], 2 * self.num_features, 1, 1)
        gamma, beta = style.chunk(2, dim=1)

        return out * gamma + beta

class CondRefineBlock(nn.Module):
    """
    Residual Block with Conditional Instance Normalization
    """

    def __init__(self, in_channels, out_channels, dilation=1, num_classes=10):
        super().__init__()
        self.norm1 = ConditionalInstanceNorm2d(in_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = ConditionalInstanceNorm2d(out_channels, num_classes)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation)

        self.activation = nn.ELU()  # Paper uses ELU

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, sigma_emb):
        h = self.activation(self.norm1(x, sigma_emb))
        h = self.conv1(h)
        h = self.activation(self.norm2(h, sigma_emb))
        h = self.conv2(h)
        return h + self.shortcut(x)

class ScoreNet(nn.Module):
    def __init__(self, channels=128, num_scales=10, image_size=32, in_channels=3):
        super().__init__()

        # Gaussian Random Feature embedding for sigma (optional but stable)
        # Or simple One-hot/Linear embedding. Paper often used simple Dense layers.
        self.embed_dim = 256
        self.noise_embed = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.ELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # U-Net Encoder
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.block1 = CondRefineBlock(channels, channels, num_classes=self.embed_dim)
        self.block2 = CondRefineBlock(channels, 2 * channels, num_classes=self.embed_dim)
        self.block3 = CondRefineBlock(2 * channels, 2 * channels, dilation=2, num_classes=self.embed_dim)  # Dilated

        # Middle
        self.mid_block = CondRefineBlock(2 * channels, 2 * channels, dilation=4, num_classes=self.embed_dim)

        # U-Net Decoder (RefineNet style often combines features differently,
        # but this U-Net structure closely mimics the official NCSN code for CIFAR-10)
        self.up_block1 = CondRefineBlock(2 * channels, channels, num_classes=self.embed_dim)
        self.up_block2 = CondRefineBlock(channels, channels, num_classes=self.embed_dim)

        self.norm_out = ConditionalInstanceNorm2d(channels, self.embed_dim)
        self.conv_out = nn.Conv2d(channels, in_channels, 3, padding=1)

        # Initialize final conv layer to near zero to start as identity
        self.score_net.conv_out.weight.data.normal_(0, 1e-10)  # Almost zero
        if self.score_net.conv_out.bias is not None:
            self.score_net.conv_out.bias.data.zero_()

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
            sigma = sigma.view(-1, 1)

        # Log-transform sigma often helps stability
        sigma_emb = self.noise_embed(torch.log(sigma))

        # 2. Encoder
        h = self.conv_in(x)
        h1 = self.block1(h, sigma_emb)
        h2 = self.block2(F.avg_pool2d(h1, 2), sigma_emb)
        h3 = self.block3(F.avg_pool2d(h2, 2), sigma_emb)

        # 3. Middle
        h_mid = self.mid_block(h3, sigma_emb)

        # 4. Decoder (with upsampling)
        # Upsample h_mid
        h_up1 = F.interpolate(h_mid, scale_factor=2, mode='nearest')
        # Add skip connection (simple addition or concat, paper typically adds in RefineNet)
        h_up1 = self.up_block1(h_up1 + h2, sigma_emb)

        h_up2 = F.interpolate(h_up1, scale_factor=2, mode='nearest')
        h_up2 = self.up_block2(h_up2 + h1, sigma_emb)

        # 5. Output
        out = self.norm_out(h_up2, sigma_emb)
        out = F.elu(out)
        out = self.conv_out(out)

        return out

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
                 sigma_min: float = None,
                 sigma_max: float = None,
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

        # -- 1. Define Sigmas (Geometric Sequence) --
        # Paper settings for CIFAR-10: sigma_max=50, sigma_min=0.01
        sigmas_config = config.get('sigmas', [0.01, 50.0])
        if sigma_min is None:
            sigma_min = sigmas_config[0] if isinstance(sigmas_config, list) else 0.01
        if sigma_max is None:
            sigma_max = sigmas_config[-1] if isinstance(sigmas_config, list) else 50.0

        # Create geometric progression of noise levels
        self.sigmas = torch.exp(torch.linspace(
            np.log(sigma_max),
            np.log(sigma_min),
            self.num_scales
        )).to(self.device)

        # Initialize score network
        self.score_net = ScoreNet(self.channels, self.num_scales, self.image_size, self.in_channels)

        # initialize weights
        self.score_net.apply(init_weights)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.score_net.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # Move to device
        self.to(self.device)

    def forward(self, x, sigma):
        """Forward pass through score network"""
        return self.score_net(x, sigma)

    def compute_loss(self, x):
        """
        Compute denoising score matching loss

        Args:
            x: Clean data (B, C, H, W)

        Returns:
            Loss value
        """
        # 1. Sample random sigma indices for the batch
        indices = torch.randint(0, self.num_scales, (x.size(0),), device=self.device)
        sigmas = self.sigmas[indices].view(-1, 1, 1, 1)

        # 2. Perturb data
        noise = torch.randn_like(x)
        perturbed_x = x + sigmas * noise

        # 3. Predict Score
        # We pass sigmas.view(-1) to the network embedding
        target = -noise / sigmas  # Score of Gaussian is -(x - mu) / sigma^2
        score = self.score_net(perturbed_x, sigmas.view(-1))

        # 4. Weighted Loss
        # Loss weighting: sigma^2
        loss = 0.5 * ((score - target) ** 2).sum(dim=(1, 2, 3)) * (sigmas.squeeze() ** 2)
        return loss.mean()

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

        # Compute loss
        loss = self.compute_loss(x)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return {'total_loss': loss.item()}

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

    def sample(self, batch_size=64, n_steps=100, eps=2e-5):
        """
        Annealed Langevin Dynamics Sampling

        Args:
            batch_size: Number of samples to generate
            n_steps: Number of Langevin steps per noise level
            eps: Base step size

        Returns:
            Generated samples (batch_size, C, H, W)
        """
        self.score_net.eval()

        # 1. Initialize with random noise
        x = torch.rand(batch_size, self.in_channels, self.image_size, self.image_size).to(self.device)

        with torch.no_grad():
            for i, sigma in enumerate(self.sigmas):
                sigma_val = sigma.item()
                # Step size calculation from paper
                alpha = eps * (sigma_val ** 2) / (self.sigmas[-1].item() ** 2)

                for t in range(n_steps):
                    z = torch.randn_like(x) if t < n_steps - 1 else 0

                    # Get score
                    sigma_input = torch.ones(batch_size, device=self.device) * sigma_val
                    score = self.score_net(x, sigma_input)

                    # Langevin step
                    x = x + alpha / 2 * score + np.sqrt(alpha) * z

        return x.clamp(0.0, 1.0)


def init_weights(m):
    """
    Initialize weights according to the techniques used in the NCSN paper.
    """
    if isinstance(m, nn.Conv2d):
        # 1. Convolutional Layers: Kaiming/He Normal
        # Standard for ReLU/ELU networks to maintain variance
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        # 2. Linear Layers (Embeddings): Xavier/Glorot Uniform
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, ConditionalInstanceNorm2d):
        # 3. Conditional Instance Norm Embeddings (CRITICAL STEP)
        # The embedding outputs parameters [gamma, beta] for normalization.
        # We want the initial state to be: gamma ≈ 1, beta ≈ 0
        # This ensures the network starts as an Identity function regarding normalization.

        num_features = m.num_features

        # The weight shape is (2 * num_features, embed_dim)
        # Initialize Gamma part (first half) to be close to 1
        m.embed.weight.data[:num_features, :].normal_(1, 0.02)

        # Initialize Beta part (second half) to be 0
        m.embed.weight.data[num_features:, :].zero_()

        # Biases for the embedding layer
        if m.embed.bias is not None:
            m.embed.bias.data[:num_features].fill_(1)  # Gamma bias = 1
            m.embed.bias.data[num_features:].fill_(0)  # Beta bias = 0