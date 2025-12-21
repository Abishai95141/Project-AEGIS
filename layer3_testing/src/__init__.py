"""
AEGIS 3.0 Layer 3 - Source Package Initialization
"""

from .config import CONFIG, DATA_DIR, RESULTS_DIR
from .g_estimation import (
    HarmonicGEstimator, 
    GEstimationResult,
    generate_harmonic_outcome_data
)
from .confidence_sequences import (
    MartingaleConfidenceSequence,
    ConfidenceSequencePoint,
    run_coverage_simulation,
    estimate_anytime_coverage_rate
)

__all__ = [
    'CONFIG',
    'DATA_DIR',
    'RESULTS_DIR',
    'HarmonicGEstimator',
    'GEstimationResult',
    'generate_harmonic_outcome_data',
    'MartingaleConfidenceSequence',
    'ConfidenceSequencePoint',
    'run_coverage_simulation',
    'estimate_anytime_coverage_rate',
]
