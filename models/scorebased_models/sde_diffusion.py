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
from abc import ABC, abstractmethod

from base.base_model import BaseModel
from models.scorebased_models.diffusions import (
    get_timestep_embedding, nonlinearity, Normalize,
    Upsample, Downsample, ResnetBlock, AttnBlock
)
from utils.get_device import get_device

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'sde_diffusion.yaml')


# ============================================================================
# SDE Abstract Base Class
# ============================================================================

class SDE(ABC):
    """
    Abstract base class for Stochastic Differential Equations.

    Forward SDE: dx = f(x, t)dt + g(t)dw
    where:
        - f(x, t) is the drift coefficient
        - g(t) is the diffusion coefficient
        - w is a standard Wiener process
    """

    def __init__(self, N=1000):
        """
        Initialize SDE.

        Args:
            N: Number of discretization steps
        """
        super().__init__()
        self.N = N

    @property
    @abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abstractmethod
    def sde(self, x, t):
        """
        Compute drift and diffusion coefficients of the SDE.

        Args:
            x: Input tensor
            t: Time tensor

        Returns:
            drift: f(x, t)
            diffusion: g(t)
        """
        pass

    @abstractmethod
    def marginal_prob(self, x, t):
        """
        Parameters to compute the marginal distribution of the SDE, p_t(x).

        For most SDEs, p_t(x) = N(mean, std^2 * I) given x_0.

        Args:
            x: Input tensor (clean data x_0)
            t: Time tensor

        Returns:
            mean: Mean of p_t(x | x_0)
            std: Standard deviation of p_t(x | x_0)
        """
        pass

    @abstractmethod
    def prior_sampling(self, shape):
        """
        Sample from the prior distribution at t=T.

        Args:
            shape: Shape of samples

        Returns:
            Samples from p_T(x)
        """
        pass

    @abstractmethod
    def prior_logp(self, z):
        """
        Compute log probability of the prior distribution.

        Args:
            z: Latent samples

        Returns:
            Log probability
        """
        pass

    def discretize(self, x, t):
        """
        Discretize the SDE in the form: x_{i+1} = x_i + f_i + G_i z_i

        Useful for ancestral sampling.

        Args:
            x: Input tensor
            t: Time tensor

        Returns:
            f: Discretized drift
            G: Discretized diffusion
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * math.sqrt(dt)
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """
        Create the reverse-time SDE/ODE.

        Args:
            score_fn: A function that computes the score ∇_x log p_t(x)
            probability_flow: If True, returns probability flow ODE (deterministic)

        Returns:
            ReverseSDE object
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        class ReverseSDE(self.__class__):
            """Reverse-time SDE."""

            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """
                Reverse SDE: dx = [f(x,t) - g(t)^2 * score]dt + g(t)dw_bar
                """
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.0)
                # Set diffusion to 0 for ODE
                diffusion = torch.zeros_like(diffusion) if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Discretize the reverse SDE."""
                f, G = discretize_fn(x, t)
                score = score_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.0)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return ReverseSDE()


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
# Score Network Architecture (NCSNv2 / DDPM++ style)
# ============================================================================

class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for noise levels (positional encoding).
    As used in the score SDE paper for time conditioning.
    """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Register as buffer so it moves with model to device
        self.register_buffer('W', torch.randn(embed_dim // 2) * scale)

    def forward(self, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class ScoreNetworkSDE(nn.Module):
    """
    Score Network for SDE-based diffusion models.

    Based on the NCSN++ architecture from the score SDE paper.
    Uses time conditioning via Gaussian Fourier features or sinusoidal embeddings.
    """

    def __init__(self,
                 in_channels=3,
                 channels=128,
                 out_channels=None,
                 ch_mult=(1, 2, 2, 2),
                 num_res_blocks=2,
                 attn_resolutions=(16,),
                 dropout=0.1,
                 resamp_with_conv=True,
                 image_size=32,
                 embedding_type='fourier',
                 fourier_scale=16.):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        out_channels = out_channels if out_channels is not None else in_channels
        self.out_channels = out_channels
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.image_size = image_size

        # Time embedding
        self.temb_ch = channels * 4
        if embedding_type == 'fourier':
            self.time_embed = nn.Sequential(
                GaussianFourierProjection(channels, scale=fourier_scale),
                nn.Linear(channels, self.temb_ch),
                nn.SiLU(),
                nn.Linear(self.temb_ch, self.temb_ch),
            )
        else:  # 'positional'
            self.time_embed = nn.Sequential(
                nn.Linear(channels, self.temb_ch),
                nn.SiLU(),
                nn.Linear(self.temb_ch, self.temb_ch),
            )
        self.embedding_type = embedding_type

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)

        curr_res = image_size
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * ch_mult[i_level]

            for _ in range(num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout
                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout
        )

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * ch_mult[i_level]
            skip_in = channels * ch_mult[i_level]

            for i_block in range(num_res_blocks + 1):
                if i_block == num_res_blocks:
                    skip_in = channels * in_ch_mult[i_level]
                block.append(ResnetBlock(
                    in_channels=block_in + skip_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout
                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # Output
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, 3, padding=1)

        # Initialize final layer to near zero
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, t):
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)
            t: Time tensor (B,) in [0, 1]

        Returns:
            Score estimate (B, C, H, W)
        """
        # Time embedding
        if self.embedding_type == 'fourier':
            temb = self.time_embed(t)
        else:
            temb = get_timestep_embedding(t * 1000, self.channels)  # Scale t for better embeddings
            temb = self.time_embed(temb)

        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


# ============================================================================
# Predictor and Corrector Classes
# ============================================================================

class Predictor(ABC):
    """Abstract class for predictors."""

    def __init__(self, sde, score_fn, probability_flow=False):
        self.sde = sde
        self.score_fn = score_fn
        self.probability_flow = probability_flow
        self.rsde = sde.reverse(score_fn, probability_flow)

    @abstractmethod
    def update_fn(self, x, t):
        """One update step."""
        pass


class Corrector(ABC):
    """Abstract class for correctors."""

    def __init__(self, sde, score_fn, snr, n_steps):
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abstractmethod
    def update_fn(self, x, t):
        """One correction step."""
        pass


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

        # Initialize score network
        self.score_net = ScoreNetworkSDE(
            in_channels=self.in_channels,
            channels=self.channels,
            ch_mult=self.ch_mult,
            num_res_blocks=self.num_res_blocks,
            attn_resolutions=self.attn_resolutions,
            dropout=self.dropout,
            image_size=self.image_size,
            embedding_type='fourier'
        )

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

        # Get network prediction
        score = self.score_net(perturbed_x, t)

        # Target: the network should predict -z (scaled noise)
        # The true score is: ∇_{x_t} log p(x_t|x_0) = -(x_t - mean) / std² = -z / std
        # So we train the network to predict -z (the negative noise)
        # And then scale by 1/std to get the score

        # Compute loss: ||score + z||² (network predicts -z)
        # This is equivalent to the standard denoising score matching loss
        if isinstance(std, torch.Tensor):
            # Weight by std² to balance contributions across noise levels
            loss = torch.mean(torch.sum((score + z) ** 2, dim=[1, 2, 3]))
        else:
            loss = torch.mean(torch.sum((score + z) ** 2, dim=[1, 2, 3]))

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
