from base import *

# ============================================================================
# Predictor Classes
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

