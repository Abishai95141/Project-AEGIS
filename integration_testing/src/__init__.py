"""
AEGIS 3.0 Integration Testing - Source Package Initialization
"""

from .config import CONFIG, RESULTS_DIR
from .unified_pipeline import (
    UnifiedPipeline,
    PatientState,
    PipelineOutput,
    compute_glucose_metrics,
    Layer1Adapter,
    Layer2Adapter,
    Layer3Adapter,
    Layer4Adapter,
    Layer5Adapter
)

__all__ = [
    'CONFIG',
    'RESULTS_DIR',
    'UnifiedPipeline',
    'PatientState',
    'PipelineOutput',
    'compute_glucose_metrics',
    'Layer1Adapter',
    'Layer2Adapter',
    'Layer3Adapter',
    'Layer4Adapter',
    'Layer5Adapter'
]
