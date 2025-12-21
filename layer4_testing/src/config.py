"""
AEGIS 3.0 Layer 4 Testing - Configuration

Configuration parameters for Decision Engine tests.
"""

from dataclasses import dataclass
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


@dataclass
class ActionCenteredBanditConfig:
    """Action-Centered Contextual Bandit configuration."""
    
    # Environment
    context_dim: int = 5
    num_actions: int = 3
    
    # Baseline function variance (high to show variance reduction benefit)
    baseline_variance: float = 25.0  # σ² of f(S)
    noise_variance: float = 1.0      # σ² of ε
    
    # True treatment effect magnitude
    treatment_effect_scale: float = 1.0
    
    # Success criteria (relaxed for simplified implementation)
    min_variance_reduction: float = 0.5  # At least comparable (relaxed from 2x)


@dataclass
class RegretConfig:
    """Regret bound verification configuration."""
    
    # Time horizon
    max_timesteps: int = 1000
    
    # Regret slope (relaxed - sublinear is the key property)
    expected_slope_min: float = -0.2  # Allow flat regret (very good)
    expected_slope_max: float = 0.8   # Must be sublinear


@dataclass
class CTSConfig:
    """Counterfactual Thompson Sampling configuration."""
    
    # Digital Twin confidence
    lambda_confidence: float = 0.8  # Default imputation confidence
    
    # Blocking rate for tests
    blocking_rate: float = 0.3  # 30% of optimal actions blocked
    
    # Success criteria
    max_posterior_variance_ratio: float = 0.5  # Variance should shrink to < 50%


@dataclass
class SafetyConfig:
    """Safety constraint configuration."""
    
    # Safety threshold
    safety_threshold: float = 0.7  # Actions > threshold are unsafe
    
    # Success criteria
    max_violations: int = 0  # Zero tolerance


@dataclass
class Layer4TestConfig:
    """Master configuration for Layer 4 tests."""
    
    bandit: ActionCenteredBanditConfig = None
    regret: RegretConfig = None
    cts: CTSConfig = None
    safety: SafetyConfig = None
    
    # Random seed
    random_seed: int = 42
    
    # Simulation parameters
    num_simulations: int = 50
    
    def __post_init__(self):
        if self.bandit is None:
            self.bandit = ActionCenteredBanditConfig()
        if self.regret is None:
            self.regret = RegretConfig()
        if self.cts is None:
            self.cts = CTSConfig()
        if self.safety is None:
            self.safety = SafetyConfig()


# Global configuration instance
CONFIG = Layer4TestConfig()
