"""
SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations
Based on: https://arxiv.org/pdf/2108.01073

SDEdit performs image synthesis and editing by:
1. Adding noise to a guide image (stroke painting, user edit, etc.) at noise level t0
2. Running the reverse SDE (denoising) from t0 to t=0 to generate realistic images

The key insight is that adding noise "projects" the guide image onto the data manifold,
and the reverse SDE then refines it to a realistic image while preserving the structure.
"""

import os
import numpy as np
import torch
import torch.optim as optim
import yaml

from base.base_model import BaseModel
from models.scorebased_models.ncsn import DoubleScoreNet, ScoreNet, init_weights
from utils.get_device import get_device

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'sdedit.yaml')


class SDEdit(BaseModel):
    """
    SDEdit: Stochastic Differential Editing

    Uses a pre-trained score-based model (NCSN) for guided image synthesis and editing.
    Compatible with the Train class interface for fine-tuning capabilities.

    Key Features:
    - Image-to-image translation via stochastic differential equations
    - Stroke-based image synthesis
    - Image editing with user-provided guides
    - Controllable realism vs. faithfulness trade-off via t0 parameter
    """

    def __init__(self,
                 config=None,
                 config_path=_DEFAULT_CONFIG_PATH,
                 channels: int = None,
                 num_scales: int = None,
                 image_size: int = 32,
                 in_channels: int = 3,
                 lr: float = None,
                 sigma_min: float = None,
                 sigma_max: float = None,
                 t0: float = None,
                 device: str = None,
                 use_simple_loss: bool = False,
                 pretrained_path: str = None,
                 use_double_scorenet: bool = True):
        """
        Initialize SDEdit model.

        Args:
            config: Configuration dictionary
            config_path: Path to YAML configuration file
            channels: Number of channels in score network
            num_scales: Number of noise scales (L)
            image_size: Size of input images
            in_channels: Number of input image channels
            lr: Learning rate for fine-tuning
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            t0: Default noise level ratio for editing (0 < t0 < 1)
                - t0 closer to 1: More realism, less faithfulness to guide
                - t0 closer to 0: More faithfulness, less realism
            device: Device to run on
            use_simple_loss: Whether to use simple loss formulation
            pretrained_path: Path to pretrained NCSN checkpoint
            use_double_scorenet: Whether to use DoubleScoreNet (deeper) or ScoreNet
        """
        super(SDEdit, self).__init__()

        self.device = device if device is not None else get_device()
        self.use_simple_loss = use_simple_loss

        # Load config from file if config dict not provided
        if config is None:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except FileNotFoundError:
                print(f"Config file {config_path} not found. Using default parameters.")
                config = {}

        # Extract parameters from config
        self.channels = channels if channels is not None else config.get('channels', 128)
        self.num_scales = num_scales if num_scales is not None else config.get('num_scales', 10)
        self.image_size = image_size
        self.in_channels = in_channels
        self.lr = lr if lr is not None else config.get('lr', 1e-4)

        # SDEdit specific parameter: t0 controls the noise level for editing
        # t0 ∈ (0, 1) where higher values mean more noise (more realism, less faithfulness)
        self.t0 = t0 if t0 is not None else config.get('t0', 0.5)

        # Sigma configuration
        sigmas_config = config.get('sigmas', [0.01, 1.0])
        if sigma_min is None:
            sigma_min = sigmas_config[0] if isinstance(sigmas_config, list) else 0.01
        if sigma_max is None:
            sigma_max = sigmas_config[-1] if isinstance(sigmas_config, list) else 1.0

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Create geometric progression of noise levels (from max to min)
        self.sigmas = torch.exp(torch.linspace(
            np.log(sigma_max),
            np.log(sigma_min),
            self.num_scales
        )).to(self.device)

        # Initialize score network
        if use_double_scorenet:
            self.score_net = DoubleScoreNet(self.channels, self.num_scales, self.image_size, self.in_channels)
        else:
            self.score_net = ScoreNet(self.channels, self.num_scales, self.image_size, self.in_channels)

        # Initialize weights
        self.score_net.apply(init_weights)

        # Initialize optimizer for fine-tuning
        self.optimizer = optim.Adam(self.score_net.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # Move to device
        self.to(self.device)

        # Load pretrained weights if provided
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, checkpoint_path: str):
        """
        Load pretrained NCSN weights for the score network.

        Args:
            checkpoint_path: Path to NCSN checkpoint file
        """
        print(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state' in checkpoint:
            model_state = checkpoint['model_state']
            if 'score_net_state_dict' in model_state:
                self.score_net.load_state_dict(model_state['score_net_state_dict'])
            if 'sigmas' in model_state:
                self.sigmas = model_state['sigmas'].to(self.device)
        elif 'score_net_state_dict' in checkpoint:
            self.score_net.load_state_dict(checkpoint['score_net_state_dict'])
            if 'sigmas' in checkpoint:
                self.sigmas = checkpoint['sigmas'].to(self.device)
        else:
            # Try loading directly as state dict
            self.score_net.load_state_dict(checkpoint)

        print("Pretrained weights loaded successfully!")

    def forward(self, x, sigma):
        """Forward pass through score network"""
        return self.score_net(x, sigma)

    def add_noise(self, x, sigma):
        """
        Add Gaussian noise to image at specified noise level.

        Args:
            x: Input image (B, C, H, W)
            sigma: Noise level (scalar or tensor)

        Returns:
            Noisy image
        """
        noise = torch.randn_like(x)
        if isinstance(sigma, (int, float)):
            return x + sigma * noise
        else:
            sigma = sigma.view(-1, 1, 1, 1)
            return x + sigma * noise

    def get_sigma_from_t(self, t):
        """
        Get sigma value for a given t ∈ [0, 1].

        t=0 corresponds to sigma_min (clean)
        t=1 corresponds to sigma_max (most noise)

        Args:
            t: Time value in [0, 1]

        Returns:
            Corresponding sigma value
        """
        # Geometric interpolation in log space
        log_sigma = np.log(self.sigma_max) * t + np.log(self.sigma_min) * (1 - t)
        return np.exp(log_sigma)

    def get_start_index(self, t0=None):
        """
        Get the starting index in sigmas array for SDEdit.

        Args:
            t0: Noise level ratio (0 < t0 < 1), uses self.t0 if None

        Returns:
            Starting index for reverse SDE
        """
        if t0 is None:
            t0 = self.t0

        # t0=1 means start from the beginning (index 0, highest noise)
        # t0=0 means start from the end (index num_scales-1, lowest noise)
        start_idx = int((1 - t0) * (self.num_scales - 1))
        return max(0, min(start_idx, self.num_scales - 1))

    def edit(self, guide_image, t0=None, n_steps=100, eps=2e-5, return_trajectory=False):
        """
        Perform SDEdit: Add noise to guide image and denoise.

        This is the main editing function that implements the SDEdit algorithm:
        1. Add noise to guide image at level corresponding to t0
        2. Run reverse SDE (Annealed Langevin Dynamics) from t0 to generate output

        Args:
            guide_image: Input guide/stroke image (B, C, H, W)
            t0: Noise level ratio (0 < t0 < 1)
                - Higher t0: More noise added, more realistic but less faithful
                - Lower t0: Less noise added, more faithful but potentially less realistic
            n_steps: Number of Langevin dynamics steps per noise level
            eps: Base step size for Langevin dynamics
            return_trajectory: Whether to return intermediate results

        Returns:
            edited_image: Output image (B, C, H, W)
            trajectory (optional): List of intermediate images
        """
        if t0 is None:
            t0 = self.t0

        self.score_net.eval()
        guide_image = guide_image.to(self.device)
        batch_size = guide_image.shape[0]

        # Get starting index based on t0
        start_idx = self.get_start_index(t0)

        # Get the starting sigma
        sigma_start = self.sigmas[start_idx].item()

        # Step 1: Add noise to guide image
        x = self.add_noise(guide_image, sigma_start)

        trajectory = [x.clone()] if return_trajectory else None

        # Step 2: Run reverse SDE from start_idx to end
        with torch.no_grad():
            for i in range(start_idx, self.num_scales):
                sigma = self.sigmas[i]
                sigma_val = sigma.item()

                # Step size: α_i = ε * (σ_i / σ_L)²
                alpha = eps * (sigma_val / self.sigmas[-1].item()) ** 2

                for t in range(n_steps):
                    # Add noise except on the last step of the last sigma
                    if i == self.num_scales - 1 and t == n_steps - 1:
                        z = torch.zeros_like(x)
                    else:
                        z = torch.randn_like(x)

                    # Get score
                    sigma_input = torch.ones(batch_size, device=self.device) * sigma_val
                    score = self(x, sigma_input)

                    # Langevin step: x_{t+1} = x_t + (α/2) * s_θ(x_t, σ) + √α * z
                    x = x + (alpha / 2) * score + np.sqrt(alpha) * z

                    # Clamp to valid range
                    x = x.clamp(0.0, 1.0)

                if return_trajectory:
                    trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x

    def stroke_to_image(self, stroke_image, t0=None, n_steps=100, eps=2e-5):
        """
        Convert a stroke painting to a realistic image.

        Args:
            stroke_image: Stroke/sketch input (B, C, H, W)
            t0: Noise level ratio (default uses self.t0)
            n_steps: Number of Langevin steps per noise level
            eps: Base step size

        Returns:
            Realistic image
        """
        return self.edit(stroke_image, t0=t0, n_steps=n_steps, eps=eps)

    def image_compositing(self, foreground, background, mask, t0=None, n_steps=100, eps=2e-5):
        """
        Composite foreground onto background and harmonize.

        Args:
            foreground: Foreground image (B, C, H, W)
            background: Background image (B, C, H, W)
            mask: Binary mask where 1 indicates foreground (B, 1, H, W)
            t0: Noise level ratio
            n_steps: Number of Langevin steps per noise level
            eps: Base step size

        Returns:
            Harmonized composite image
        """
        # Create composite
        mask = mask.to(self.device)
        foreground = foreground.to(self.device)
        background = background.to(self.device)

        composite = mask * foreground + (1 - mask) * background

        # Run SDEdit to harmonize
        return self.edit(composite, t0=t0, n_steps=n_steps, eps=eps)

    def compute_loss(self, x):
        """
        Compute denoising score matching loss (same as NCSN).
        Used for fine-tuning the model on specific domains.

        Args:
            x: Clean data batch (B, C, H, W)

        Returns:
            loss: Scalar loss value
            indices: Indices of sampled noise levels
        """
        batch_size = x.shape[0]

        # Randomly sample noise levels
        indices = torch.randint(0, self.num_scales, (batch_size,), device=self.device)
        sigmas = self.sigmas[indices]

        # Sample Gaussian noise
        noise = torch.randn_like(x)

        # Perturb the data
        sigmas_expanded = sigmas.view(batch_size, 1, 1, 1)
        perturbed_x = x + sigmas_expanded * noise

        # Predict the score
        predicted_score = self.score_net(perturbed_x, sigmas)

        # Compute loss
        if self.use_simple_loss:
            target_score = -noise / sigmas_expanded
            loss = torch.mean(torch.sum((predicted_score - target_score) ** 2, dim=[1, 2, 3]))
        else:
            weighted_error = sigmas_expanded * predicted_score + noise
            loss = torch.mean(torch.sum(weighted_error ** 2, dim=[1, 2, 3]))

        return loss, indices

    def train_step(self, x, epoch):
        """
        Perform one training step (for fine-tuning).

        Args:
            x: Input batch (B, C, H, W)
            epoch: Current epoch number

        Returns:
            Dictionary of losses
        """
        self.score_net.train()
        self.optimizer.zero_grad()

        loss, indices = self.compute_loss(x)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {'total_loss': loss.item()}

    def epoch_step(self):
        """Called at the end of each epoch"""
        pass

    def get_init_loss_dict(self):
        """Return initial loss dictionary structure"""
        return {'total_loss': 0.0}

    def get_model_state(self, epoch):
        """
        Get model state for checkpointing.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary containing model state
        """
        return {
            'epoch': epoch,
            'score_net_state_dict': self.score_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'sigmas': self.sigmas,
            't0': self.t0,
            'config': {
                'channels': self.channels,
                'num_scales': self.num_scales,
                'image_size': self.image_size,
                'in_channels': self.in_channels,
                'lr': self.lr,
                't0': self.t0,
                'sigma_min': self.sigma_min,
                'sigma_max': self.sigma_max
            }
        }

    def load_state(self, model_state):
        """
        Load model state from checkpoint.

        Args:
            model_state: Dictionary containing model state
        """
        self.score_net.load_state_dict(model_state['score_net_state_dict'])
        self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        self.sigmas = model_state['sigmas'].to(self.device)
        if 't0' in model_state:
            self.t0 = model_state['t0']

    def sample(self, batch_size=64, n_steps=100, eps=2e-5):
        """
        Generate samples from scratch using Annealed Langevin Dynamics.
        (Same as NCSN sampling - starts from pure noise)

        Args:
            batch_size: Number of samples to generate
            n_steps: Number of Langevin steps per noise level
            eps: Base step size

        Returns:
            Generated samples (batch_size, C, H, W)
        """
        self.score_net.eval()

        # Initialize with Gaussian noise
        x = torch.randn(batch_size, self.in_channels, self.image_size, self.image_size).to(self.device)

        with torch.no_grad():
            for i, sigma in enumerate(self.sigmas):
                sigma_val = sigma.item()
                alpha = eps * (sigma_val / self.sigmas[-1].item()) ** 2

                for t in range(n_steps):
                    z = torch.randn_like(x) if t < n_steps - 1 else torch.zeros_like(x)

                    sigma_input = torch.ones(batch_size, device=self.device) * sigma_val
                    score = self(x, sigma_input)

                    x = x + (alpha / 2) * score + np.sqrt(alpha) * z
                    x = x.clamp(0.0, 1.0)

        return x
