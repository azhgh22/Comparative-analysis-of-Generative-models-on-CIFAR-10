"""
SDE-Based Diffusion Model
Based on: "Score-Based Generative Modeling through Stochastic Differential Equations"
Paper: https://arxiv.org/pdf/2011.13456

This implementation provides:
1. VE-SDE (Variance Exploding) - similar to NCSN
2. VP-SDE (Variance Preserving) - similar to DDPM
3. sub-VP SDE - tighter ELBO
4. Unified sampling methods (Predictor-Corrector, Probability Flow ODE)

The key insight is that score-based models can be unified under the SDE framework:
- Forward process: dx = f(x,t)dt + g(t)dw (diffusion SDE)
- Reverse process: dx = [f(x,t) - g(t)²∇_x log p_t(x)]dt + g(t)dw̄
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from base.base_model import BaseModel
from base.predictor import Predictor
from base.corrector import Corrector
from base.sde import SDE

from models.scorebased_models.ncsn import ScoreNet, DoubleScoreNet
from models.scorebased_models.score_networks.score_net_sde import ScoreNetworkSDE
from utils.get_device import get_device

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'sde_diffusion.yaml')


# ============================================================================
# Specific SDE Implementations
# ============================================================================

class VESDE(SDE):
    """
    Variance Exploding SDE (VE-SDE).

    dx = σ(t) * sqrt(2 * log(σ_max/σ_min)) * dw

    This corresponds to the NCSN/SMLD model where noise levels increase geometrically.
    """

    def __init__(self, sigma_min=0.01, sigma_max=50., N=1000):
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @property
    def T(self):
        return 1.

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * math.sqrt(2 * (math.log(self.sigma_max) - math.log(self.sigma_min)))
        return drift, diffusion.expand(x.shape[0])

    def marginal_prob(self, x, t):
        """
        p_t(x | x_0) = N(x_0, σ(t)^2 * I)
        """
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape, device='cpu'):
        return torch.randn(*shape, device=device) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

    def discretize(self, x, t):
        """SMLD discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t),
            self.sigma_min * (self.sigma_max / self.sigma_min) ** ((timestep - 1) / (self.N - 1))
        )
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G


class VPSDE(SDE):
    """
    Variance Preserving SDE (VP-SDE).

    dx = -0.5 * β(t) * x * dt + sqrt(β(t)) * dw

    This corresponds to the DDPM model. The variance is preserved (roughly 1) throughout.
    """

    def __init__(self, beta_min=0.1, beta_max=20., N=1000):
        super().__init__(N)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    @property
    def T(self):
        return 1.

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        """
        p_t(x | x_0) = N(sqrt(α_bar(t)) * x_0, (1 - α_bar(t)) * I)

        where α_bar(t) = exp(-0.5 * ∫_0^t β(s) ds)
        """
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape, device='cpu'):
        return torch.randn(*shape, device=device)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.

    def discretize(self, x, t):
        """DDPM discretization."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.beta_0 + t * (self.beta_1 - self.beta_0)
        alpha = 1. - beta / self.N
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = torch.sqrt(beta / self.N)
        return f, G


class subVPSDE(SDE):
    """
    Sub-VP SDE (tighter ELBO than VP-SDE).

    dx = -0.5 * β(t) * x * dt + sqrt(β(t) * (1 - exp(-2 * ∫_0^t β(s)ds))) * dw
    """

    def __init__(self, beta_min=0.1, beta_max=20., N=1000):
        super().__init__(N)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    @property
    def T(self):
        return 1.

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = 1. - torch.exp(2. * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape, device='cpu'):
        return torch.randn(*shape, device=device)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


# ============================================================================
# NCSN Score Network Wrapper for SDE Framework
# ============================================================================

class NCSNScoreNetworkWrapper(nn.Module):
    """
    Wrapper for NCSN-style score networks (ScoreNet, DoubleScoreNet) to be used
    with the SDE-based diffusion framework.

    This wrapper converts the continuous time t ∈ [0, 1] used in SDE diffusion
    to the sigma-based conditioning used by NCSN networks.

    Args:
        network_type: 'scorenet' or 'double_scorenet'
        channels: Base number of channels
        image_size: Size of input images
        in_channels: Number of input channels
        sde: The SDE object (VESDE, VPSDE, etc.) for computing sigma from t
    """

    def __init__(self, network_type='double_scorenet', channels=128, image_size=32,
                 in_channels=3, sde=None):
        super().__init__()

        self.sde = sde
        self.network_type = network_type

        if network_type == 'scorenet':
            self.net = ScoreNet(
                channels=channels,
                num_scales=1000,  # Not used for continuous time
                image_size=image_size,
                in_channels=in_channels
            )
        elif network_type == 'double_scorenet':
            self.net = DoubleScoreNet(
                channels=channels,
                num_scales=1000,  # Not used for continuous time
                image_size=image_size,
                in_channels=in_channels
            )
        else:
            raise ValueError(f"Unknown network type: {network_type}. "
                           f"Use 'scorenet' or 'double_scorenet'")

    def forward(self, x, t):
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)
            t: Time tensor (B,) in [0, 1]

        Returns:
            Score estimate (B, C, H, W)
        """
        # Convert time t to sigma for NCSN networks
        # The NCSN networks use sigma (noise level) for conditioning via log(sigma)
        # We need to get the std/sigma from the SDE's marginal distribution
        if self.sde is not None:
            _, std = self.sde.marginal_prob(torch.zeros_like(x), t)
            # std is the standard deviation of p(x_t | x_0)
            # For VESDE: std = sigma_min * (sigma_max/sigma_min)^t
            # For VPSDE: std = sqrt(1 - exp(-integral of beta))
            if isinstance(std, torch.Tensor):
                sigma = std
            else:
                sigma = torch.ones(x.shape[0], device=x.device) * std
        else:
            # Fallback: interpret t directly as sigma scale
            sigma = t

        # Ensure sigma has correct shape (B,)
        if sigma.dim() == 0:
            sigma = sigma.expand(x.shape[0])
        elif sigma.dim() > 1:
            # If std came out with extra dims, flatten to (B,)
            sigma = sigma.view(-1)

        # Ensure sigma is positive (avoid log(0) in NCSN networks)
        sigma = sigma.clamp(min=1e-5)

        return self.net(x, sigma)


class EulerMaruyamaPredictor(Predictor):
    """Euler-Maruyama predictor for reverse SDE."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * math.sqrt(-dt) * z
        return x, x_mean


class ReverseDiffusionPredictor(Predictor):
    """Reverse diffusion predictor (ancestral sampling)."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


class AncestralSamplingPredictor(Predictor):
    """Ancestral sampling predictor for VP-SDE (DDPM style)."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.beta_0 + t * (sde.beta_1 - sde.beta_0)
        score = self.score_fn(x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta[:, None, None, None] / sde.N)
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta[:, None, None, None] / sde.N) * noise
        return x, x_mean


class NonePredictor(Predictor):
    """No predictor (just returns input)."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        return x, x


class LangevinCorrector(Corrector):
    """Langevin dynamics corrector."""

    def __init__(self, sde, score_fn, snr=0.16, n_steps=1):
        super().__init__(sde, score_fn, snr, n_steps)

    def update_fn(self, x, t):
        sde = self.sde
        target_snr = self.snr
        x_mean = x  # Initialize for case where n_steps=0

        for _ in range(self.n_steps):
            grad = self.score_fn(x, t)
            noise = torch.randn_like(x)

            # Compute step size
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2

            x_mean = x + step_size * grad
            x = x_mean + torch.sqrt(2 * step_size) * noise

        return x, x_mean


class AnnealedLangevinCorrector(Corrector):
    """Annealed Langevin dynamics corrector (for VE-SDE)."""

    def __init__(self, sde, score_fn, snr=0.16, n_steps=1):
        super().__init__(sde, score_fn, snr, n_steps)

    def update_fn(self, x, t):
        sde = self.sde
        target_snr = self.snr
        x_mean = x  # Initialize for case where n_steps=0

        if isinstance(sde, VESDE):
            alpha = torch.ones_like(t)
        else:
            _, alpha = sde.marginal_prob(x, t)

        for _ in range(self.n_steps):
            grad = self.score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * alpha) ** 2 * 2
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(2 * step_size[:, None, None, None]) * noise

        return x, x_mean


class NoneCorrector(Corrector):
    """No corrector."""

    def __init__(self, sde, score_fn, snr=0.16, n_steps=1):
        super().__init__(sde, score_fn, snr, n_steps)

    def update_fn(self, x, t):
        return x, x


# ============================================================================
# Main SDE Diffusion Model Class
# ============================================================================

class SDEDiffusion(BaseModel):
    """
    SDE-Based Diffusion Model.

    Implements the unified framework from "Score-Based Generative Modeling
    through Stochastic Differential Equations" (https://arxiv.org/pdf/2011.13456).

    Compatible with the Train class interface.
    """

    def __init__(self,
                 config=None,
                 config_path=_DEFAULT_CONFIG_PATH,
                 # Architecture parameters
                 channels: int = None,
                 ch_mult: tuple = None,
                 num_res_blocks: int = None,
                 attn_resolutions: tuple = None,
                 dropout: float = None,
                 image_size: int = 32,
                 in_channels: int = 3,
                 score_network_type: str = None,  # 'sde' (default), 'scorenet', 'double_scorenet'
                 # SDE parameters
                 sde_type: str = None,  # 'vesde', 'vpsde', 'subvpsde'
                 beta_min: float = None,
                 beta_max: float = None,
                 sigma_min: float = None,
                 sigma_max: float = None,
                 N: int = None,  # Number of discretization steps
                 # Training parameters
                 lr: float = None,
                 ema_rate: float = None,
                 # Sampling parameters
                 predictor: str = None,  # 'euler_maruyama', 'reverse_diffusion', 'ancestral', 'none'
                 corrector: str = None,  # 'langevin', 'annealed_langevin', 'none'
                 snr: float = None,
                 n_steps_corrector: int = None,
                 probability_flow: bool = None,
                 # Other
                 device: str = None,
                 continuous: bool = True):
        """
        Initialize SDE Diffusion model.

        Args:
            config: Configuration dictionary
            config_path: Path to YAML configuration file
            channels: Base number of channels in score network
            ch_mult: Channel multipliers for each resolution level
            num_res_blocks: Number of residual blocks per resolution
            attn_resolutions: Resolutions at which to apply attention
            dropout: Dropout rate
            image_size: Size of input images
            in_channels: Number of input image channels
            score_network_type: Type of score network architecture:
                - 'sde': Default NCSN++ style U-Net with Fourier embeddings
                - 'scorenet': NCSN-style ScoreNet with conditional instance norm
                - 'double_scorenet': NCSN-style DoubleScoreNet (deeper architecture)
            sde_type: Type of SDE ('vesde', 'vpsde', 'subvpsde')
            beta_min: Minimum beta for VP-SDE
            beta_max: Maximum beta for VP-SDE
            sigma_min: Minimum sigma for VE-SDE
            sigma_max: Maximum sigma for VE-SDE
            N: Number of discretization steps
            lr: Learning rate
            ema_rate: EMA rate for model weights
            predictor: Type of predictor for sampling
            corrector: Type of corrector for sampling
            snr: Signal-to-noise ratio for Langevin corrector
            n_steps_corrector: Number of corrector steps
            probability_flow: Whether to use probability flow ODE
            device: Device to run on
            continuous: Whether to use continuous time
        """
        super(SDEDiffusion, self).__init__()

        self.device = device if device is not None else get_device()
        self.continuous = continuous

        # Load config
        if config is None:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except FileNotFoundError:
                print(f"Config file {config_path} not found. Using default parameters.")
                config = {}

        # Architecture parameters
        self.channels = channels if channels is not None else config.get('channels', 128)
        self.ch_mult = ch_mult if ch_mult is not None else tuple(config.get('ch_mult', [1, 2, 2, 2]))
        self.num_res_blocks = num_res_blocks if num_res_blocks is not None else config.get('num_res_blocks', 2)
        self.attn_resolutions = attn_resolutions if attn_resolutions is not None else tuple(config.get('attn_resolutions', [16]))
        self.dropout = dropout if dropout is not None else config.get('dropout', 0.1)
        self.image_size = image_size
        self.in_channels = in_channels
        self.score_network_type = score_network_type if score_network_type is not None else config.get('score_network_type', 'sde')

        # SDE parameters
        self.sde_type = sde_type if sde_type is not None else config.get('sde_type', 'vpsde')
        self.beta_min = beta_min if beta_min is not None else config.get('beta_min', 0.1)
        self.beta_max = beta_max if beta_max is not None else config.get('beta_max', 20.)
        self.sigma_min = sigma_min if sigma_min is not None else config.get('sigma_min', 0.01)
        self.sigma_max = sigma_max if sigma_max is not None else config.get('sigma_max', 50.)
        self.N = N if N is not None else config.get('N', 1000)

        # Training parameters
        self.lr = lr if lr is not None else config.get('lr', 2e-4)
        self.ema_rate = ema_rate if ema_rate is not None else config.get('ema_rate', 0.9999)

        # Sampling parameters
        self.predictor_type = predictor if predictor is not None else config.get('predictor', 'euler_maruyama')
        self.corrector_type = corrector if corrector is not None else config.get('corrector', 'langevin')
        self.snr = snr if snr is not None else config.get('snr', 0.16)
        self.n_steps_corrector = n_steps_corrector if n_steps_corrector is not None else config.get('n_steps_corrector', 1)
        self.probability_flow = probability_flow if probability_flow is not None else config.get('probability_flow', False)

        # Initialize SDE
        self.sde = self._create_sde()

        # Initialize score network based on score_network_type
        self.score_net = self._create_score_network()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.score_net.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # EMA for model parameters
        self.ema_params = None

        # Move to device
        self.to(self.device)

    def _create_sde(self):
        """Create the appropriate SDE based on sde_type."""
        if self.sde_type.lower() == 'vesde':
            return VESDE(sigma_min=self.sigma_min, sigma_max=self.sigma_max, N=self.N)
        elif self.sde_type.lower() == 'vpsde':
            return VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.N)
        elif self.sde_type.lower() == 'subvpsde':
            return subVPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.N)
        else:
            raise ValueError(f"Unknown SDE type: {self.sde_type}")

    def _create_score_network(self):
        """Create the score network based on score_network_type."""
        network_type = self.score_network_type.lower()

        if network_type == 'sde':
            # Default NCSN++ style U-Net with Fourier embeddings
            return ScoreNetworkSDE(
                in_channels=self.in_channels,
                channels=self.channels,
                ch_mult=self.ch_mult,
                num_res_blocks=self.num_res_blocks,
                attn_resolutions=self.attn_resolutions,
                dropout=self.dropout,
                image_size=self.image_size,
                embedding_type='fourier'
            )
        elif network_type in ['scorenet', 'double_scorenet']:
            # NCSN-style networks wrapped for SDE framework
            return NCSNScoreNetworkWrapper(
                network_type=network_type,
                channels=self.channels,
                image_size=self.image_size,
                in_channels=self.in_channels,
                sde=self.sde
            )
        else:
            raise ValueError(f"Unknown score network type: {self.score_network_type}. "
                           f"Use 'sde', 'scorenet', or 'double_scorenet'")

    def forward(self, x, t):
        """
        Forward pass through score network.

        Args:
            x: Input tensor (B, C, H, W)
            t: Time tensor (B,)

        Returns:
            Score estimate
        """
        return self.score_net(x, t)

    def score_fn(self, x, t):
        """
        Compute the score function.

        For continuous training, the network directly predicts the score.
        The score is scaled by the standard deviation for numerical stability.
        """
        if self.continuous:
            # Network predicts s(x, t) = ∇_x log p_t(x)
            # We parameterize as: score = -model_output / std
            # This is because s(x, t) ≈ -(x - mean) / std^2 for Gaussian
            _, std = self.sde.marginal_prob(torch.zeros_like(x), t)
            score = self.score_net(x, t)
            # The network is trained to predict epsilon (noise), so score = -epsilon / std
            if isinstance(std, torch.Tensor):
                score = -score / std[:, None, None, None]
            else:
                score = -score / std
        else:
            # Discrete case
            score = self.score_net(x, t)
        return score

    def compute_loss(self, x):
        """
        Compute the training loss.

        Uses denoising score matching:
        L = E_t E_{x_0} E_{x_t|x_0} [λ(t) ||s_θ(x_t, t) - ∇_{x_t} log p(x_t|x_0)||²]

        For continuous training, we sample t uniformly from [eps, T].

        Args:
            x: Clean data batch (B, C, H, W)

        Returns:
            loss: Scalar loss value
        """
        batch_size = x.shape[0]

        # Sample time uniformly in [eps, T]
        eps = 1e-5
        t = torch.rand(batch_size, device=self.device) * (self.sde.T - eps) + eps

        # Sample noise
        z = torch.randn_like(x)

        # Get mean and std for p(x_t | x_0)
        mean, std = self.sde.marginal_prob(x, t)

        # Perturb data: x_t = mean + std * z
        if isinstance(std, torch.Tensor):
            perturbed_x = mean + std[:, None, None, None] * z
        else:
            perturbed_x = mean + std * z

        # Get network prediction (predicts epsilon/noise)
        score = self.score_net(perturbed_x, t)

        # Compute loss: network should match the sampled noise z (epsilon prediction)
        # This keeps score_fn = -epsilon/std in sync with the true score -z/std.
        if isinstance(std, torch.Tensor):
            loss = torch.mean(torch.sum((score - z) ** 2, dim=[1, 2, 3]))
        else:
            loss = torch.mean(torch.sum((score - z) ** 2, dim=[1, 2, 3]))

        return loss, t

    def train_step(self, x, epoch):
        """
        Perform one training step.

        Args:
            x: Input batch (B, C, H, W)
            epoch: Current epoch number

        Returns:
            Dictionary of losses
        """
        self.score_net.train()
        self.optimizer.zero_grad()

        # Compute loss
        loss, t = self.compute_loss(x)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update EMA
        self._update_ema()

        return {'total_loss': loss.item()}

    def _update_ema(self):
        """Update exponential moving average of model parameters."""
        if self.ema_params is None:
            self.ema_params = [p.clone().detach() for p in self.score_net.parameters()]
        else:
            for ema_p, p in zip(self.ema_params, self.score_net.parameters()):
                ema_p.mul_(self.ema_rate).add_(p.data, alpha=1 - self.ema_rate)

    def _swap_to_ema(self):
        """Swap model parameters with EMA parameters."""
        if self.ema_params is not None:
            self._backup_params = [p.clone() for p in self.score_net.parameters()]
            for ema_p, p in zip(self.ema_params, self.score_net.parameters()):
                p.data.copy_(ema_p)

    def _restore_from_ema(self):
        """Restore original model parameters."""
        if hasattr(self, '_backup_params'):
            for backup_p, p in zip(self._backup_params, self.score_net.parameters()):
                p.data.copy_(backup_p)
            del self._backup_params

    def epoch_step(self):
        """Called at the end of each epoch."""
        pass

    def get_init_loss_dict(self):
        """Return initial loss dictionary structure."""
        return {'total_loss': 0.0}

    def get_model_state(self, epoch):
        """Get model state for checkpointing."""
        return {
            'epoch': epoch,
            'score_net_state_dict': self.score_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ema_params': self.ema_params,
            'config': {
                'channels': self.channels,
                'ch_mult': self.ch_mult,
                'num_res_blocks': self.num_res_blocks,
                'attn_resolutions': self.attn_resolutions,
                'dropout': self.dropout,
                'image_size': self.image_size,
                'in_channels': self.in_channels,
                'score_network_type': self.score_network_type,
                'sde_type': self.sde_type,
                'beta_min': self.beta_min,
                'beta_max': self.beta_max,
                'sigma_min': self.sigma_min,
                'sigma_max': self.sigma_max,
                'N': self.N,
                'lr': self.lr,
            }
        }

    def load_state(self, model_state):
        """Load model state from checkpoint."""
        self.score_net.load_state_dict(model_state['score_net_state_dict'])
        self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        if 'ema_params' in model_state:
            self.ema_params = model_state['ema_params']

    def _get_predictor(self, score_fn):
        """Get predictor instance."""
        predictors = {
            'euler_maruyama': EulerMaruyamaPredictor,
            'reverse_diffusion': ReverseDiffusionPredictor,
            'ancestral': AncestralSamplingPredictor,
            'none': NonePredictor,
        }
        predictor_cls = predictors.get(self.predictor_type.lower(), EulerMaruyamaPredictor)
        return predictor_cls(self.sde, score_fn, self.probability_flow)

    def _get_corrector(self, score_fn):
        """Get corrector instance."""
        correctors = {
            'langevin': LangevinCorrector,
            'annealed_langevin': AnnealedLangevinCorrector,
            'none': NoneCorrector,
        }
        corrector_cls = correctors.get(self.corrector_type.lower(), LangevinCorrector)
        return corrector_cls(self.sde, score_fn, self.snr, self.n_steps_corrector)

    def sample(self, batch_size=64, use_ema=True, denoise=True):
        """
        Generate samples using Predictor-Corrector sampling.

        Args:
            batch_size: Number of samples to generate
            use_ema: Whether to use EMA parameters
            denoise: Whether to apply final denoising step

        Returns:
            Generated samples (batch_size, C, H, W)
        """
        self.score_net.eval()

        if use_ema and self.ema_params is not None:
            self._swap_to_ema()

        # Define score function for sampling
        def score_fn(x, t):
            with torch.no_grad():
                return self.score_fn(x, t)

        # Get predictor and corrector
        predictor = self._get_predictor(score_fn)
        corrector = self._get_corrector(score_fn)

        # Sample from prior
        x = self.sde.prior_sampling((batch_size, self.in_channels, self.image_size, self.image_size), device=self.device)

        # Time steps
        timesteps = torch.linspace(self.sde.T, 1e-5, self.sde.N, device=self.device)

        with torch.no_grad():
            for t in timesteps:
                t_batch = torch.ones(batch_size, device=self.device) * t

                # Corrector step
                x, _ = corrector.update_fn(x, t_batch)

                # Predictor step
                x, x_mean = predictor.update_fn(x, t_batch)

        # Final denoising step
        if denoise:
            x = x_mean

        if use_ema and self.ema_params is not None:
            self._restore_from_ema()

        return x.clamp(0., 1.)

    def sample_probability_flow(self, batch_size=64, use_ema=True, denoise=True):
        """
        Generate samples using the probability flow ODE.

        This is deterministic sampling that traces the same marginal distributions.

        Args:
            batch_size: Number of samples to generate
            use_ema: Whether to use EMA parameters
            denoise: Whether to apply final denoising step

        Returns:
            Generated samples (batch_size, C, H, W)
        """
        original_probability_flow = self.probability_flow
        self.probability_flow = True

        samples = self.sample(batch_size, use_ema, denoise)

        self.probability_flow = original_probability_flow
        return samples


# Factory function for creating models
def create_sde_diffusion(sde_type='vpsde', **kwargs):
    """
    Factory function to create SDE diffusion models.

    Args:
        sde_type: Type of SDE ('vesde', 'vpsde', 'subvpsde')
        **kwargs: Additional arguments passed to SDEDiffusion

    Returns:
        SDEDiffusion model instance
    """
    return SDEDiffusion(sde_type=sde_type, **kwargs)
