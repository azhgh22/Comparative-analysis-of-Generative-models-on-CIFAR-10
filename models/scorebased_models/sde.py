import os

import yaml
from torch import optim

from base.base_model import BaseModel
from models.scorebased_models import *
from utils.get_device import get_device
from utils.helper_for_overfitting import show_images, load_test_batch

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'sde_diffusion.yaml')

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t_emb):
        out = self.conv1(nn.functional.silu(x))                      # 256 -> 128
        time_emb_proj = self.time_mlp(nn.functional.silu(t_emb))     # project time embedding

        out = out + time_emb_proj[:, :, None, None]                  # add time embedding
        out = self.conv2(nn.functional.silu(out))                    # 128 -> 128
        return self.shortcut(x) + out                                # skip connection

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, time):
        # 1. Create the sinusoidal patterns
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(int(half_dim), device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]

        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # 2. Process with a small MLP (Multi-Layer Perceptron)
        embeddings = self.linear_1(embeddings)
        embeddings = nn.functional.silu(embeddings)
        embeddings = self.linear_2(embeddings)
        return embeddings

class DiffusionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_embedded_dim = 64

        # initial layers
        self.time_embed = TimeEmbedding(dim=time_embedded_dim)
        self.first_conv = nn.Conv2d(3, 16, 3, padding=1)

        # Downsampling blocks
        self.down1 = ResBlock(16, 16, time_embedded_dim)
        self.downsample1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)

        self.down2 = ResBlock(32, 32, time_embedded_dim)
        self.downsample2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        self.down3 = ResBlock(64, 64, time_embedded_dim)
        self.downsample3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.down4 = ResBlock(128, 128, time_embedded_dim)
        self.downsample4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        # Middle block
        self.middle = ResBlock(256, 256, time_embedded_dim)

        # Upsampling blocks
        self.upsample1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.up1 = ResBlock(256, 128, time_embedded_dim)

        self.upsample2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up2 = ResBlock(128, 64, time_embedded_dim)

        self.upsample3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.up3 = ResBlock(64, 32, time_embedded_dim)

        self.upsample4 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.up4 = ResBlock(32, 16, time_embedded_dim)

        # Final output layer
        self.final_conv = nn.Conv2d(16, 3, 1)

    def forward(self, x, t):
        # 1. Create Time Embedding
        t_emb = self.time_embed(t)

        # Initial Convolution
        x = self.first_conv(x)

        # 2. Encoder (Down)
        x1 = self.down1(x, t_emb)
        x2 = self.downsample1(x1)
        x3 = self.down2(x2, t_emb)
        x4 = self.downsample2(x3)
        x5 = self.down3(x4, t_emb)
        x6 = self.downsample3(x5)
        x7 = self.down4(x6, t_emb)
        x8 = self.downsample4(x7)

        # 3. Bottleneck
        x_mid = self.middle(x8, t_emb)

        # 4. Decoder (Up)
        x_up1 = self.upsample1(x_mid)
        x_up1 = self.up1(torch.cat([x_up1, x7], dim=1), t_emb)
        x_up2 = self.upsample2(x_up1)
        x_up2 = self.up2(torch.cat([x_up2, x5], dim=1), t_emb)
        x_up3 = self.upsample3(x_up2)
        x_up3 = self.up3(torch.cat([x_up3, x3], dim=1), t_emb)
        x_up4 = self.upsample4(x_up3)
        x_up4 = self.up4(torch.cat([x_up4, x1], dim=1), t_emb)

        output = self.final_conv(x_up4)
        return output

class NoiseScheduler(nn.Module):
    def __init__(self, num_steps, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alpha_cumprod = self.get_alpha_cumprod().to(get_device())

    def get_alpha_cumprod(self):
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod_t = self.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1)

        sqrt_one_minus_alpha_cumprod_t = (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

class SDEModel(BaseModel):
    def __init__(self,
                 config=None,
                 config_path=_DEFAULT_CONFIG_PATH,
                 num_steps=1000,
                 image_size: int = 32,
                 in_channels: int = 3,
                 lr: float = None,
                 device: str = None):
        super().__init__()

        self.in_channels = in_channels
        self.image_size = image_size
        self.device = device if device is not None else get_device()

        # Load config from file if config dict not provided
        if config is None:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"Reading from config file {config_path}")
                print(f"Configs:\n{config}")
            except FileNotFoundError:
                print(f"Config file {config_path} not found. Using default parameters.")
                config = {}
        self.unet = DiffusionUNet()
        self.device = get_device()
        self.noise_scheduler = NoiseScheduler(num_steps)

        self.lr = lr if lr is not None else config.get('lr', 1e-4)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.unet.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # Move to device
        self.to(self.device)

    def forward(self, x, t):
        return self.unet(x, t)

    def compute_loss(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)

        x_noisy = self.noise_scheduler.q_sample(x, t, noise)

        predicted_noise = self.forward(x_noisy, t)

        return nn.functional.mse_loss(predicted_noise, noise)

    def train_step(self, x, epoch):
        """
        Perform one training step

        Args:
            x: Input batch (B, C, H, W)
            epoch: Current epoch number

        Returns:
            Dictionary of losses
        """
        batch_size = x.shape[0]

        self.unet.train()

        self.optimizer.zero_grad()

        # Compute loss
        random_t = torch.randint(0, int(self.noise_scheduler.num_steps), (batch_size,), device=self.device).long()
        loss = self.compute_loss(x, random_t)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {'total_loss': loss}

    def sample(self, batch_size: int, n_steps=None) -> torch.Tensor:
        if n_steps is None:
            n_steps = self.noise_scheduler.num_steps

        self.unet.eval()

        # 1. Initialize with Gaussian noise
        x = torch.randn(batch_size, int(self.in_channels), int(self.image_size), int(self.image_size)).to(self.device)

        device = next(self.unet.parameters()).device
        with torch.no_grad():
            for time_stamp in reversed(range(n_steps)):
                t_batch = torch.full((len(x),), time_stamp, device=device, dtype=torch.long)
                predicted_noise = self.unet(x, t_batch)
                beta_t = self.noise_scheduler.betas[time_stamp]
                alpha_t = 1 - beta_t
                alpha_cumprod_t = self.noise_scheduler.alpha_cumprod[time_stamp]

                # $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$
                coef1 = 1 / torch.sqrt(alpha_t)
                coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)
                sigma_t = torch.sqrt(beta_t)
                random_noise = torch.randn_like(x) if time_stamp > 0 else 0
                x = coef1 * (x - coef2 * predicted_noise) + sigma_t * random_noise

                x = x.clamp(-1.0, 1.0)
        return x.clamp(-1.0, 1.0)

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


if __name__ == "__main__":
    model = SDEModel(device=get_device())
    num_samples = 1

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_params)

    test_batch = load_test_batch(num_samples)
    print(test_batch.shape)
    loss_dict = model.train_step(test_batch, epoch=0)

    sampled_images = model.sample(1)
    # sampled_images = (sampled_images - sampled_images.min()) / (sampled_images.max() - sampled_images.min())
    sampled_images.detach_()

    show_images(sampled_images, title="Sampled Images", n_row=1)
    show_images(test_batch, title="test Images", n_row=1)

    # assert if the sampled images are different from the random samples
    assert not torch.allclose(sampled_images, test_batch), "Sampled images are identical to random samples!"

    print(f"Loss: {loss_dict["total_loss"].item()}")
    print(f"Sampled images:\n {sampled_images}")
