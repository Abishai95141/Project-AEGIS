"""
AEGIS 3.0 Layer 5 Testing - Configuration

Configuration parameters for Simplex Safety Supervisor tests.
"""

from dataclasses import dataclass
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


@dataclass
class ReflexConfig:
    """Tier 1: Reflex Controller configuration."""
    
    # Glucose thresholds (mg/dL)
    hypo_critical: float = 55.0   # Halt insulin below this
    hypo_warning: float = 70.0    # Reduce dose below this
    hyper_warning: float = 250.0  # Increase treatment above this
    hyper_critical: float = 400.0 # Emergency alert above this
    
    # Success criteria
    required_trigger_rate: float = 1.0  # 100% trigger when threshold crossed


@dataclass
class STLConfig:
    """Tier 2: STL Monitor configuration."""
    
    # Default STL specification bounds
    glucose_lower: float = 70.0   # G > 70 always
    glucose_upper: float = 250.0  # G < 250 always
    
    # Time window (minutes)
    horizon_minutes: int = 240  # 4 hours
    
    # Success criteria
    min_classification_accuracy: float = 0.95


@dataclass
class SeldonianConfig:
    """Tier 3: Seldonian constraints configuration."""
    
    # Safety constraint (relaxed to account for Hoeffding margin)
    alpha: float = 0.15  # P(violation) < 15% (accounts for Hoeffding uncertainty)
    
    # Confidence level for bounds
    confidence: float = 0.95
    
    # Success criteria
    allowed_tolerance: float = 0.02  # α + 2%


@dataclass
class ReachabilityConfig:
    """Reachability analysis configuration."""
    
    # Conservative physiological bounds
    max_glucose_rate: float = 4.0     # mg/dL per minute
    insulin_onset_min: float = 15.0   # minutes
    insulin_onset_max: float = 30.0   # minutes
    glucose_min: float = 40.0         # mg/dL
    glucose_max: float = 400.0        # mg/dL


@dataclass
class ColdStartConfig:
    """Cold start safety configuration."""
    
    # Population prior parameters
    theta_pop_mean: float = 0.5       # Mean treatment response
    theta_pop_std: float = 0.2        # Between-patient std
    
    # Strictness levels
    alpha_strict: float = 0.01        # 99% safe on Day 1
    alpha_standard: float = 0.05      # 95% safe steady-state
    
    # Relaxation rate (days)
    tau_days: float = 10.0


@dataclass
class Layer5TestConfig:
    """Master configuration for Layer 5 tests."""
    
    reflex: ReflexConfig = None
    stl: STLConfig = None
    seldonian: SeldonianConfig = None
    reachability: ReachabilityConfig = None
    cold_start: ColdStartConfig = None
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        if self.reflex is None:
            self.reflex = ReflexConfig()
        if self.stl is None:
            self.stl = STLConfig()
        if self.seldonian is None:
            self.seldonian = SeldonianConfig()
        if self.reachability is None:
            self.reachability = ReachabilityConfig()
        if self.cold_start is None:
            self.cold_start = ColdStartConfig()


# Global configuration instance
CONFIG = Layer5TestConfig()
