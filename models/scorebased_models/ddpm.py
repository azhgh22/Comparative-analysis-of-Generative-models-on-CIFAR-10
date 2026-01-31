"""
Denoising Diffusion Probabilistic Models (DDPM)
Based on: "Denoising Diffusion Probabilistic Models" by Ho et al. (2020)
https://arxiv.org/abs/2006.11239
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

from torch import optim

from base.base_model import BaseModel
from utils.get_device import get_device
from utils.helper_for_overfitting import load_test_batch, show_images


def get_beta_schedule(timesteps: int, schedule: str = "linear") -> torch.Tensor:
    """
    Generate beta schedule for the forward diffusion process.

    Args:
        timesteps: Number of diffusion steps (T)
        schedule: Type of schedule ("linear", "cosine")

    Returns:
        Beta values for each timestep
    """
    if schedule == "linear":
        # Linear schedule from the paper: β1 = 10^-4 to βT = 0.02
        beta_start = 1e-4
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    elif schedule == "cosine":
        # Improved cosine schedule (not in original paper)
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class SinusoidalPositionEmbeddings(nn.Module):
    """Transformer-style sinusoidal position embeddings for timestep encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Residual block with group normalization."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        # Residual connection
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # Add time embedding
        time_emb = F.silu(self.time_mlp(t_emb))
        h = h + time_emb[:, :, None, None]

        h = self.conv2(F.silu(self.norm2(h)))
        h = self.dropout(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)

        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, HW, C_per_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C // self.num_heads)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        h = torch.matmul(attn, v)
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return x + self.proj(h)


class UNet(nn.Module):
    """
    U-Net architecture for noise prediction.
    Based on the architecture described in the DDPM paper.
    """

    def __init__(
            self,
            image_channels: int = 3,
            base_channels: int = 128,
            channel_mults: Tuple[int, ...] = (1, 2, 2, 2),
            num_res_blocks: int = 2,
            time_emb_dim: int = 512,
            dropout: float = 0.1,
            attention_resolutions: Tuple[int, ...] = (16,),
    ):
        super().__init__()

        self.image_channels = image_channels
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for level, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    Block(now_channels, out_channels, time_emb_dim, dropout)
                )
                now_channels = out_channels
                channels.append(now_channels)

                # Add attention at specified resolutions
                # Assume input is 32x32, so resolutions are 32, 16, 8, 4
                current_resolution = 32 // (2 ** level)
                if current_resolution in attention_resolutions:
                    self.down_blocks.append(AttentionBlock(now_channels))
                    channels.append(now_channels)

            # Downsample (except for the last level)
            if level != len(channel_mults) - 1:
                self.down_blocks.append(nn.Conv2d(now_channels, now_channels, kernel_size=3, stride=2, padding=1))
                channels.append(now_channels)

        # Middle
        self.mid_block1 = Block(now_channels, now_channels, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(now_channels)
        self.mid_block2 = Block(now_channels, now_channels, time_emb_dim, dropout)

        # Upsampling
        self.up_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    Block(now_channels + channels.pop(), out_channels, time_emb_dim, dropout)
                )
                now_channels = out_channels

                # Add attention at specified resolutions
                current_resolution = 32 // (2 ** level)
                if current_resolution in attention_resolutions:
                    self.up_blocks.append(AttentionBlock(now_channels))

            # Upsample (except for the first level)
            if level != 0:
                self.up_blocks.append(
                    nn.ConvTranspose2d(now_channels, now_channels, kernel_size=4, stride=2, padding=1))

        # Output
        self.out_norm = nn.GroupNorm(8, now_channels)
        self.conv_out = nn.Conv2d(now_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_mlp(t)

        # Initial conv
        h = self.conv_in(x)

        # Downsampling
        hs = [h]
        for module in self.down_blocks:
            if isinstance(module, Block):
                h = module(h, t_emb)
            else:
                h = module(h)
            hs.append(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Upsampling
        for module in self.up_blocks:
            if isinstance(module, Block):
                if h.shape[-2:] != hs[-1].shape[-2:]:
                    h = F.interpolate(h, size=hs[-1].shape[-2:], mode='nearest')
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, t_emb)
            elif isinstance(module, AttentionBlock):
                h = module(h)
            else:  # Upsample
                h = module(h)

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class DDPM(BaseModel):
    """
    Denoising Diffusion Probabilistic Model.

    Implements the training and sampling algorithms from the DDPM paper.
    """

    def __init__(
            self,
            model: UNet,
            timesteps: int = 1000,
            beta_schedule: str = "linear",
            image_size: int = 32,
            image_channels: int = 3,
            lr: float = 1e-4,
            device: str = None
    ):
        super().__init__()

        self.model = model
        self.timesteps = timesteps
        self.image_size = image_size
        self.image_channels = image_channels
        self.lr = lr
        self.device = device if device is not None else get_device()

        # Define beta schedule
        betas = get_beta_schedule(timesteps, beta_schedule)

        # Pre-compute constants for the diffusion process
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (these will be moved to GPU with the model)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance',
                             betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

        # For sampling
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # Move to device
        self.to(self.device)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        Sample from q(x_t | x_0) using the reparameterization trick.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Training loss: L_simple from the paper (Equation 14)

        L_simple = E_{t, x_0, epsilon} [|| epsilon - epsilon_theta(x_t, t) ||^2]
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Get noisy image
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        predicted_noise = self.model(x_t, t)

        # MSE loss
        loss = F.mse_loss(noise, predicted_noise)

        return loss

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Reverse diffusion process: sample x_{t-1} from p_theta(x_{t-1} | x_t)

        Uses the formula from Algorithm 2 in the paper.
        """
        batch_size = x_t.shape[0]

        # Create time tensor
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)

        # Predict noise
        predicted_noise = self.model(x_t, t_tensor)

        # Get coefficients
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        betas_t = self.betas[t]

        # Calculate mean
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * predicted_noise)

        if t == 0:
            return model_mean
        else:
            # Add noise
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, batch_size: int = 16, return_all_timesteps: bool = False) -> torch.Tensor:
        """
        Sample from the model using Algorithm 2 from the paper.

        Args:
            batch_size: Number of images to generate
            return_all_timesteps: If True, return all intermediate steps

        Returns:
            Generated images
        """
        device = next(self.model.parameters()).device

        # Start from random noise
        x = torch.randn(batch_size, self.image_channels, self.image_size, self.image_size, device=device)

        if return_all_timesteps:
            imgs = [x]

        # Reverse diffusion process
        imgs = []
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t)
            if return_all_timesteps:
                imgs.append(x)

        if return_all_timesteps:
            return torch.stack(imgs, dim=1)  # (batch, timesteps, channels, height, width)
        else:
            return x

    def forward(self, x: torch.Tensor, t):
        """
        Forward pass for training.

        Args:
            x: Clean images from the dataset
            t: Current timestep
        Returns:
            Loss value
        """
        pass

    def train_step(self, x_0: torch.Tensor, epoch=None) -> dict:
        """
        Training forward pass: Algorithm 1 from the paper.

        Args:
            x_0: Clean images from the dataset

        Returns:
            Loss value
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        self.model.train()

        self.optimizer.zero_grad()

        # Sample random timesteps
        random_t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        # Calculate loss
        loss = self.p_losses(x_0, random_t)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)

        self.optimizer.step()
        return {'total_loss': loss.item()}

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
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
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
        self.load_state_dict(model_state['model_state_dict'])
        self.optimizer.load_state_dict(model_state['optimizer_state_dict'])


def create_ddpm(
        image_size: int = 32,
        image_channels: int = 3,
        timesteps: int = 1000,
        base_channels: int = 128,
) -> DDPM:
    """
    Create a DDPM model with default settings from the paper.

    Args:
        image_size: Size of input images (assumes square images)
        image_channels: Number of channels (3 for RGB)
        timesteps: Number of diffusion steps (T)
        base_channels: Base number of channels in U-Net

    Returns:
        DDPM model
    """
    # Create U-Net model
    model = UNet(
        image_channels=image_channels,
        base_channels=base_channels,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=base_channels * 4,
        dropout=0.1,
        attention_resolutions=(16,),
    )

    # Create DDPM
    ddpm = DDPM(
        model=model,
        timesteps=timesteps,
        beta_schedule="linear",
        image_size=image_size,
        image_channels=image_channels,
    )

    return ddpm


if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = get_device()
    num_samples = 1

    # Create model
    model = create_ddpm(image_size=32, image_channels=3, timesteps=1000)
    model = model.to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    test_batch = load_test_batch(num_samples)
    print(test_batch.shape)

    # Training
    loss_dict = model.train_step(test_batch)

    sampled_images = model.sample(1)
    sampled_images.detach_()

    show_images(sampled_images, title="Sampled Images", n_row=1)
    show_images(test_batch, title="Test Images", n_row=1)

    print(f"Training loss: {loss_dict["total_loss"]:.6f}")

    # Sampling
    samples = model.sample(batch_size=4)
    print(f"Generated samples shape: {samples.shape}")
