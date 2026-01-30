from base import *

# ============================================================================
# Corrector Classes
# ============================================================================

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
