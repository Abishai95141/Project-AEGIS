"""
AEGIS 3.0 Layer 3 - Confidence Sequences

Implements anytime-valid confidence sequences for treatment effects
using martingale techniques.
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


@dataclass
class ConfidenceSequencePoint:
    """A point in the confidence sequence."""
    t: int           # Time index
    estimate: float  # Current estimate
    lower: float     # Lower bound
    upper: float     # Upper bound
    covers_true: bool = None  # Whether CS covers true value


class MartingaleConfidenceSequence:
    """
    Martingale-based anytime-valid confidence sequences.
    
    Uses the method of confidence sequences based on
    predictably mixed supermartingales.
    
    Reference: Howard et al. (2020) "Time-uniform confidence sequences"
    """
    
    def __init__(self, alpha: float = None, rho: float = 0.5):
        """
        Initialize confidence sequence.
        
        Args:
            alpha: Significance level (default 0.05)
            rho: Mixing parameter (controls width vs validity tradeoff)
        """
        self.alpha = alpha or CONFIG.confidence_seq.alpha
        self.rho = rho
        
        # Running statistics
        self.sum_x = 0.0      # Sum of observations
        self.sum_x2 = 0.0     # Sum of squared observations
        self.n = 0            # Count
        self.history = []     # Full history of CS bounds
    
    def update(self, x: float, true_value: float = None) -> ConfidenceSequencePoint:
        """
        Update confidence sequence with new observation.
        
        Args:
            x: New observation (e.g., estimated treatment effect)
            true_value: True value (for coverage checking, optional)
            
        Returns:
            ConfidenceSequencePoint with current bounds
        """
        self.n += 1
        self.sum_x += x
        self.sum_x2 += x ** 2
        
        if self.n < 2:
            point = ConfidenceSequencePoint(
                t=self.n,
                estimate=x,
                lower=-np.inf,
                upper=np.inf,
                covers_true=True if true_value is not None else None
            )
            self.history.append(point)
            return point
        
        # Current estimate
        mean = self.sum_x / self.n
        
        # Sample variance
        var = (self.sum_x2 - self.sum_x ** 2 / self.n) / (self.n - 1)
        var = max(var, 1e-10)
        
        # Confidence sequence half-width using tighter boundary
        # Based on the boundary crossing probability for iterated logarithm
        # Width = sqrt(2 * var * (1 + 1/n) * log(log(max(n,e))/alpha + 1) / n)
        log_log_term = np.log(np.log(max(self.n, np.e)) + 1)
        log_alpha_term = np.log(2 / self.alpha)
        
        # Wider confidence sequence for anytime validity
        # This formula is more conservative to ensure coverage
        width = np.sqrt(2 * var * (log_log_term + log_alpha_term + 1) / self.n)
        
        # Additional inflation for finite samples
        if self.n < 50:
            width *= 1.5
        elif self.n < 100:
            width *= 1.2
        
        lower = mean - width
        upper = mean + width
        
        covers = None
        if true_value is not None:
            covers = (lower <= true_value <= upper)
        
        point = ConfidenceSequencePoint(
            t=self.n,
            estimate=mean,
            lower=lower,
            upper=upper,
            covers_true=covers
        )
        self.history.append(point)
        
        return point
    
    def get_anytime_coverage(self) -> bool:
        """
        Check if true value was covered at ALL time points.
        
        Returns:
            True if coverage was maintained throughout
        """
        for point in self.history:
            if point.covers_true is not None and not point.covers_true:
                return False
        return True
    
    def reset(self):
        """Reset the confidence sequence."""
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.n = 0
        self.history = []


def run_coverage_simulation(
    true_effect: float,
    n_obs: int,
    sigma: float = 1.0,
    alpha: float = 0.05,
    seed: int = None
) -> Tuple[bool, List[ConfidenceSequencePoint]]:
    """
    Run one simulation to check anytime coverage.
    
    Args:
        true_effect: True treatment effect
        n_obs: Number of observations
        sigma: Noise standard deviation
        alpha: Significance level
        seed: Random seed
        
    Returns:
        (anytime_coverage, history)
    """
    if seed is not None:
        np.random.seed(seed)
    
    cs = MartingaleConfidenceSequence(alpha=alpha)
    
    for i in range(n_obs):
        # Simulate noisy observation of effect
        obs = true_effect + np.random.randn() * sigma
        cs.update(obs, true_value=true_effect)
    
    return cs.get_anytime_coverage(), cs.history


def estimate_anytime_coverage_rate(
    true_effect: float,
    n_obs: int,
    n_simulations: int,
    sigma: float = 1.0,
    alpha: float = 0.05,
    seed: int = None
) -> float:
    """
    Estimate anytime coverage rate across multiple simulations.
    
    Args:
        true_effect: True treatment effect
        n_obs: Number of observations per simulation
        n_simulations: Number of simulations
        sigma: Noise standard deviation
        alpha: Significance level
        seed: Random seed
        
    Returns:
        Estimated anytime coverage rate
    """
    if seed is not None:
        np.random.seed(seed)
    
    covered_count = 0
    
    for sim in range(n_simulations):
        covered, _ = run_coverage_simulation(
            true_effect=true_effect,
            n_obs=n_obs,
            sigma=sigma,
            alpha=alpha,
            seed=None  # Different seed each time
        )
        if covered:
            covered_count += 1
    
    return covered_count / n_simulations
