from base import *

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

