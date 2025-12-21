"""
AEGIS 3.0 Layer 4 - Source Package Initialization
"""

from .config import CONFIG, DATA_DIR, RESULTS_DIR
from .action_centered_bandit import (
    ActionCenteredBandit,
    StandardQBandit,
    BanditState,
    generate_bandit_environment
)
from .cts import (
    CounterfactualThompsonSampling,
    StandardThompsonSampling,
    DigitalTwin,
    SafetyEvaluator,
    CTSResult
)

__all__ = [
    'CONFIG',
    'DATA_DIR', 
    'RESULTS_DIR',
    'ActionCenteredBandit',
    'StandardQBandit',
    'BanditState',
    'generate_bandit_environment',
    'CounterfactualThompsonSampling',
    'StandardThompsonSampling',
    'DigitalTwin',
    'SafetyEvaluator',
    'CTSResult'
]
