"""
AEGIS 3.0 Integration Testing - Configuration

Configuration parameters for unified architecture integration tests.
"""

from dataclasses import dataclass
import os
import sys

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Add all layer paths
LAYERS_PATH = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(LAYERS_PATH, "layer1_testing", "src"))
sys.path.insert(0, os.path.join(LAYERS_PATH, "layer2_testing", "src"))
sys.path.insert(0, os.path.join(LAYERS_PATH, "layer3_testing", "src"))
sys.path.insert(0, os.path.join(LAYERS_PATH, "layer4_testing", "src"))
sys.path.insert(0, os.path.join(LAYERS_PATH, "layer5_testing", "src"))


@dataclass
class GlucoseMetrics:
    """ADA Consensus 2019 metrics thresholds."""
    
    # Time in Range (70-180 mg/dL)
    tir_target: float = 0.70     # > 70%
    tir_relaxed: float = 0.60    # Relaxed for simplified model
    
    # Time Below Range (< 70 mg/dL)
    tbr_target: float = 0.04     # < 4%
    tbr_relaxed: float = 0.05    # < 5%
    
    # Time Above Range (> 180 mg/dL)
    tar_target: float = 0.25     # < 25%
    tar_relaxed: float = 0.35    # < 35%
    
    # Severe hypoglycemia (< 54 mg/dL)
    severe_hypo_threshold: float = 54.0
    
    # Coefficient of Variation
    cv_target: float = 0.36      # < 36%


@dataclass
class PipelineConfig:
    """End-to-end pipeline configuration."""
    
    # Execution time limits (ms)
    max_decision_time_ms: float = 100.0
    
    # Simulation parameters
    simulation_hours: int = 24
    decision_interval_minutes: int = 5
    
    # Glucose ranges (mg/dL)
    glucose_range_low: float = 70.0
    glucose_range_high: float = 180.0
    hypo_threshold: float = 70.0
    hyper_threshold: float = 180.0
    severe_hypo: float = 54.0
    severe_hyper: float = 250.0


@dataclass
class BenchmarkScenarios:
    """UVA/Padova-style benchmark scenarios."""
    
    # S1: Fasting
    fasting_initial: float = 120.0
    fasting_target_low: float = 80.0
    fasting_target_high: float = 140.0
    
    # S2: Post-meal
    meal_carbs_grams: float = 50.0
    meal_peak_max: float = 180.0
    meal_return_target: float = 140.0
    
    # S3: Hypo risk
    hypo_risk_initial: float = 80.0
    hypo_prevention_threshold: float = 70.0
    
    # S4: Hyper correction
    hyper_initial: float = 250.0
    hyper_correction_target: float = 180.0


@dataclass
class IntegrationTestConfig:
    """Master configuration for integration tests."""
    
    metrics: GlucoseMetrics = None
    pipeline: PipelineConfig = None
    scenarios: BenchmarkScenarios = None
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = GlucoseMetrics()
        if self.pipeline is None:
            self.pipeline = PipelineConfig()
        if self.scenarios is None:
            self.scenarios = BenchmarkScenarios()


# Global configuration instance
CONFIG = IntegrationTestConfig()
