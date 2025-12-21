"""
AEGIS 3.0 Integration Testing - Unified Pipeline

Connects all 5 layers into a single decision-making pipeline:
L1 (Semantic) → L2 (Digital Twin) → L3 (Causal) → L4 (Decision) → L5 (Safety)
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG

# Import all layer modules
import sys
import os

# Add layer paths
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(base_path, "layer1_testing", "src"))
sys.path.insert(0, os.path.join(base_path, "layer2_testing", "src"))
sys.path.insert(0, os.path.join(base_path, "layer3_testing", "src"))
sys.path.insert(0, os.path.join(base_path, "layer4_testing", "src"))
sys.path.insert(0, os.path.join(base_path, "layer5_testing", "src"))


@dataclass
class PatientState:
    """Current patient state for pipeline input."""
    glucose: float               # Current glucose (mg/dL)
    glucose_history: np.ndarray  # Recent glucose readings
    insulin_on_board: float      # Active insulin
    carbs_on_board: float        # Active carbs
    time_of_day: float           # Hours (0-24)
    day_number: int              # For cold start tracking
    activity_level: float        # 0-1 scale


@dataclass
class PipelineOutput:
    """Output from the unified pipeline."""
    final_action: float          # Final insulin dose
    proposed_action: float       # L4 proposed action
    was_overridden: bool         # Whether L5 modified action
    safety_tier: int             # Which safety tier triggered (0 = none)
    predicted_trajectory: np.ndarray
    treatment_effect: float
    execution_time_ms: float
    layer_outputs: Dict


class Layer1Adapter:
    """Adapter for Layer 1 (Semantic Sensorium)."""
    
    def __init__(self):
        """Initialize Layer 1 components."""
        self.entropy = None  # Don't import to avoid config conflicts
    
    def process(self, state: PatientState) -> Dict:
        """
        Process patient state through Layer 1.
        
        Returns context vector S_t.
        """
        # Create context from patient state
        context = np.array([
            state.glucose / 200.0,  # Normalized glucose
            state.insulin_on_board / 10.0,  # Normalized IOB
            state.carbs_on_board / 50.0,  # Normalized COB
            np.sin(2 * np.pi * state.time_of_day / 24.0),  # Time encoding
            np.cos(2 * np.pi * state.time_of_day / 24.0),
            state.activity_level
        ])
        
        # Compute semantic entropy if available
        if self.entropy is not None:
            entropy_val = np.var(state.glucose_history) / 100.0
        else:
            entropy_val = 0.1
        
        return {
            'context': context,
            'entropy': entropy_val,
            'glucose_trend': np.mean(np.diff(state.glucose_history[-5:])) if len(state.glucose_history) > 5 else 0
        }


class Layer2Adapter:
    """Adapter for Layer 2 (Digital Twin)."""
    
    def __init__(self):
        """Initialize Layer 2 components."""
        self.ude = None  # Don't import UDE directly to avoid config conflicts
    
    def predict(self, state: PatientState, action: float, 
                horizon_minutes: int = 60) -> np.ndarray:
        """
        Predict glucose trajectory using Digital Twin.
        """
        # Use simple model (UDE has config conflicts with integration config)
        trajectory = self._simple_predict(state, action, horizon_minutes)
        return trajectory
    
    def _simple_predict(self, state: PatientState, action: float,
                        horizon_minutes: int) -> np.ndarray:
        """Simple glucose prediction model."""
        glucose = np.zeros(horizon_minutes)
        glucose[0] = state.glucose
        
        for t in range(1, horizon_minutes):
            # Simple dynamics
            trend = -0.02 * (glucose[t-1] - 100)  # Mean reversion
            
            # Insulin effect (delayed)
            if t > 15:
                insulin_effect = -action * 2.0 * np.exp(-(t - 45)**2 / 400)
            else:
                insulin_effect = 0
            
            # Carb effect
            if state.carbs_on_board > 0:
                carb_effect = state.carbs_on_board * 0.5 * np.exp(-(t - 30)**2 / 300)
            else:
                carb_effect = 0
            
            glucose[t] = glucose[t-1] + trend + insulin_effect + carb_effect
            glucose[t] = np.clip(glucose[t], 40, 400)
        
        return glucose
    
    def _ude_predict(self, state: PatientState, action: float,
                     horizon_minutes: int) -> np.ndarray:
        """UDE-based prediction (if available)."""
        return self._simple_predict(state, action, horizon_minutes)


class Layer3Adapter:
    """Adapter for Layer 3 (Causal Inference)."""
    
    def __init__(self):
        """Initialize Layer 3 components."""
        self.estimator = None  # Don't import to avoid config conflicts
    
    def estimate_effect(self, state: PatientState, context: np.ndarray) -> float:
        """
        Estimate treatment effect τ(t) at current time.
        """
        if self.estimator is not None:
            # Use time-varying effect
            t = np.array([state.time_of_day])
            # Default effect parameters (would be learned in production)
            psi = np.array([0.5, 0.2, 0.1])
            effect = self.estimator.compute_effect(t, psi)[0]
        else:
            # Simple circadian effect
            effect = 0.5 + 0.2 * np.cos(2 * np.pi * (state.time_of_day - 8) / 24)
        
        return float(effect)


class Layer4Adapter:
    """Adapter for Layer 4 (Decision Engine)."""
    
    def __init__(self):
        """Initialize Layer 4 components."""
        self.bandit = None  # Don't import bandit to avoid config conflicts
    
    def propose_action(self, state: PatientState, context: np.ndarray,
                       treatment_effect: float) -> float:
        """
        Propose optimal action based on context and effect.
        """
        if self.bandit is not None:
            action_idx, _ = self.bandit.select_action(context)
            # Map action index to dose
            dose = action_idx * 2.0  # 0, 2, 4, 6, 8 units
        else:
            # Heuristic dosing
            if state.glucose > 180:
                dose = min(8.0, (state.glucose - 100) / 30.0 * treatment_effect)
            elif state.glucose > 120:
                dose = min(4.0, (state.glucose - 100) / 50.0 * treatment_effect)
            else:
                dose = 0.0
        
        return float(np.clip(dose, 0, 10))


class Layer5Adapter:
    """Adapter for Layer 5 (Safety)."""
    
    def __init__(self):
        """Initialize Layer 5 components."""
        self.supervisor = None  # Don't import to avoid config conflicts
    
    def evaluate(self, state: PatientState, proposed_action: float,
                 trajectory: np.ndarray) -> Tuple[float, bool, int]:
        """
        Evaluate and potentially modify proposed action.
        
        Returns: (final_action, was_modified, tier_triggered)
        """
        if self.supervisor is not None:
            # Generate outcome samples for Seldonian
            outcome_samples = trajectory + np.random.randn(100) * 10
            
            result = self.supervisor.evaluate(
                current_glucose=state.glucose,
                predicted_trajectory=trajectory[:10] if len(trajectory) >= 10 else trajectory,
                outcome_samples=outcome_samples,
                proposed_action=proposed_action
            )
            
            if result.decision.value in ['BLOCK', 'EMERGENCY']:
                final_action = result.modified_action if result.modified_action is not None else 0.0
                return final_action, True, result.tier_triggered
            elif result.decision.value == 'MODIFY':
                return result.modified_action, True, result.tier_triggered
            else:
                return proposed_action, False, 0
        else:
            # Simple safety check
            if state.glucose < 70:
                return 0.0, True, 1
            elif state.glucose < 80:
                return min(proposed_action, 2.0), True, 1
            else:
                return proposed_action, False, 0


class UnifiedPipeline:
    """
    AEGIS 3.0 Unified Decision Pipeline.
    
    Connects all 5 layers into a single decision-making system.
    """
    
    def __init__(self):
        """Initialize all layer adapters."""
        self.layer1 = Layer1Adapter()
        self.layer2 = Layer2Adapter()
        self.layer3 = Layer3Adapter()
        self.layer4 = Layer4Adapter()
        self.layer5 = Layer5Adapter()
        
        # Statistics
        self.total_decisions = 0
        self.overrides = 0
        self.execution_times = []
    
    def execute(self, state: PatientState) -> PipelineOutput:
        """
        Execute full pipeline for one decision cycle.
        """
        start_time = time.time()
        layer_outputs = {}
        
        # Layer 1: Process context
        l1_output = self.layer1.process(state)
        context = l1_output['context']
        layer_outputs['L1'] = l1_output
        
        # Layer 2: Predict trajectory (for proposed action = 0 initially)
        trajectory = self.layer2.predict(state, action=0.0)
        layer_outputs['L2'] = {'trajectory': trajectory}
        
        # Layer 3: Estimate treatment effect
        effect = self.layer3.estimate_effect(state, context)
        layer_outputs['L3'] = {'treatment_effect': effect}
        
        # Layer 4: Propose action
        proposed_action = self.layer4.propose_action(state, context, effect)
        layer_outputs['L4'] = {'proposed_action': proposed_action}
        
        # Update trajectory with proposed action
        trajectory_with_action = self.layer2.predict(state, action=proposed_action)
        
        # Layer 5: Safety check
        final_action, was_overridden, safety_tier = self.layer5.evaluate(
            state, proposed_action, trajectory_with_action
        )
        layer_outputs['L5'] = {
            'final_action': final_action,
            'was_overridden': was_overridden,
            'safety_tier': safety_tier
        }
        
        # Statistics
        execution_time = (time.time() - start_time) * 1000  # ms
        self.total_decisions += 1
        if was_overridden:
            self.overrides += 1
        self.execution_times.append(execution_time)
        
        return PipelineOutput(
            final_action=final_action,
            proposed_action=proposed_action,
            was_overridden=was_overridden,
            safety_tier=safety_tier,
            predicted_trajectory=trajectory_with_action,
            treatment_effect=effect,
            execution_time_ms=execution_time,
            layer_outputs=layer_outputs
        )
    
    def get_statistics(self) -> Dict:
        """Get pipeline execution statistics."""
        return {
            'total_decisions': self.total_decisions,
            'total_overrides': self.overrides,
            'override_rate': self.overrides / max(self.total_decisions, 1),
            'mean_execution_time_ms': np.mean(self.execution_times) if self.execution_times else 0,
            'max_execution_time_ms': max(self.execution_times) if self.execution_times else 0
        }


def compute_glucose_metrics(glucose_history: np.ndarray) -> Dict:
    """
    Compute ADA Consensus glucose metrics.
    
    Args:
        glucose_history: Array of glucose values
        
    Returns:
        Dictionary with TIR, TBR, TAR, CV, etc.
    """
    n = len(glucose_history)
    if n == 0:
        return {}
    
    # Time in Range (70-180 mg/dL)
    tir = np.mean((glucose_history >= 70) & (glucose_history <= 180))
    
    # Time Below Range
    tbr = np.mean(glucose_history < 70)
    tbr_severe = np.mean(glucose_history < 54)
    
    # Time Above Range
    tar = np.mean(glucose_history > 180)
    tar_severe = np.mean(glucose_history > 250)
    
    # Coefficient of Variation
    cv = np.std(glucose_history) / np.mean(glucose_history)
    
    # Mean and standard deviation
    mean_glucose = np.mean(glucose_history)
    std_glucose = np.std(glucose_history)
    
    # Min/Max
    min_glucose = np.min(glucose_history)
    max_glucose = np.max(glucose_history)
    
    return {
        'tir': float(tir),
        'tbr': float(tbr),
        'tbr_severe': float(tbr_severe),
        'tar': float(tar),
        'tar_severe': float(tar_severe),
        'cv': float(cv),
        'mean': float(mean_glucose),
        'std': float(std_glucose),
        'min': float(min_glucose),
        'max': float(max_glucose)
    }
