"""
AEGIS 3.0 Layer 5 - Simplex Safety Supervisor

Implements the three-tier safety hierarchy:
- Tier 1: Reflex Controller (model-free, highest priority)
- Tier 2: STL Monitor (formal verification)
- Tier 3: Seldonian Constraints (probabilistic)

Paper reference: Section 5.5
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


class SafetyDecision(Enum):
    """Safety decision outcomes."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    MODIFY = "MODIFY"
    EMERGENCY = "EMERGENCY"


@dataclass
class SafetyResult:
    """Result from safety evaluation."""
    decision: SafetyDecision
    tier_triggered: int  # 0 = none, 1/2/3 = tier number
    reason: str
    modified_action: Optional[float] = None


class ReflexController:
    """
    Tier 1: Reflex Controller (Highest Priority)
    
    Model-free threshold logic operating directly on sensor measurements.
    Cannot be fooled by Digital Twin errors; operates on raw reality.
    """
    
    def __init__(self):
        """Initialize with threshold configuration."""
        self.hypo_critical = CONFIG.reflex.hypo_critical
        self.hypo_warning = CONFIG.reflex.hypo_warning
        self.hyper_warning = CONFIG.reflex.hyper_warning
        self.hyper_critical = CONFIG.reflex.hyper_critical
    
    def evaluate(self, glucose: float, proposed_action: float) -> SafetyResult:
        """
        Evaluate safety based on current glucose reading.
        
        Args:
            glucose: Current glucose level (mg/dL)
            proposed_action: Proposed insulin dose (units)
            
        Returns:
            SafetyResult with decision
        """
        # Critical hypoglycemia - HALT all insulin
        if glucose < self.hypo_critical:
            return SafetyResult(
                decision=SafetyDecision.EMERGENCY,
                tier_triggered=1,
                reason=f"CRITICAL HYPO: G={glucose:.1f} < {self.hypo_critical}",
                modified_action=0.0
            )
        
        # Warning hypoglycemia - reduce/block insulin
        if glucose < self.hypo_warning:
            return SafetyResult(
                decision=SafetyDecision.BLOCK,
                tier_triggered=1,
                reason=f"HYPO WARNING: G={glucose:.1f} < {self.hypo_warning}",
                modified_action=0.0
            )
        
        # Critical hyperglycemia - allow and alert
        if glucose > self.hyper_critical:
            return SafetyResult(
                decision=SafetyDecision.EMERGENCY,
                tier_triggered=1,
                reason=f"CRITICAL HYPER: G={glucose:.1f} > {self.hyper_critical}",
                modified_action=proposed_action  # Still allow treatment
            )
        
        # Warning hyperglycemia - allow
        if glucose > self.hyper_warning:
            return SafetyResult(
                decision=SafetyDecision.ALLOW,
                tier_triggered=0,
                reason=f"HYPER WARNING: G={glucose:.1f} > {self.hyper_warning}"
            )
        
        # Normal glucose - allow
        return SafetyResult(
            decision=SafetyDecision.ALLOW,
            tier_triggered=0,
            reason="Glucose in normal range"
        )


class STLMonitor:
    """
    Tier 2: STL Monitor (Signal Temporal Logic)
    
    Formal verification of predicted trajectories against temporal specifications.
    Specification: □[0,T](G > 70) ∧ □[0,T](G < 250)
    """
    
    def __init__(self):
        """Initialize with STL specification."""
        self.glucose_lower = CONFIG.stl.glucose_lower
        self.glucose_upper = CONFIG.stl.glucose_upper
        self.horizon = CONFIG.stl.horizon_minutes
    
    def compute_robustness(self, trajectory: np.ndarray) -> float:
        """
        Compute STL robustness value for trajectory.
        
        For □[0,T](G > 70) ∧ □[0,T](G < 250):
        - ρ = min(min(G - 70), min(250 - G))
        - ρ > 0 means safe, ρ < 0 means violation
        
        Args:
            trajectory: Array of glucose values over time
            
        Returns:
            Robustness value (positive = safe, negative = unsafe)
        """
        if len(trajectory) == 0:
            return float('-inf')
        
        # Always lower bound
        lower_margin = trajectory - self.glucose_lower
        
        # Always upper bound
        upper_margin = self.glucose_upper - trajectory
        
        # Robustness is the minimum of all margins
        robustness = min(np.min(lower_margin), np.min(upper_margin))
        
        return float(robustness)
    
    def evaluate(self, predicted_trajectory: np.ndarray, 
                 proposed_action: float) -> SafetyResult:
        """
        Evaluate trajectory against STL specification.
        
        Args:
            predicted_trajectory: Array of predicted glucose values
            proposed_action: Proposed action
            
        Returns:
            SafetyResult with decision
        """
        robustness = self.compute_robustness(predicted_trajectory)
        
        if robustness < 0:
            return SafetyResult(
                decision=SafetyDecision.BLOCK,
                tier_triggered=2,
                reason=f"STL VIOLATION: robustness={robustness:.2f}"
            )
        
        if robustness < 10:  # Close to boundary
            return SafetyResult(
                decision=SafetyDecision.ALLOW,
                tier_triggered=0,
                reason=f"STL MARGINAL: robustness={robustness:.2f}"
            )
        
        return SafetyResult(
            decision=SafetyDecision.ALLOW,
            tier_triggered=0,
            reason=f"STL SAFE: robustness={robustness:.2f}"
        )
    
    def classify_trajectory(self, trajectory: np.ndarray) -> bool:
        """Classify trajectory as safe (True) or unsafe (False)."""
        return self.compute_robustness(trajectory) >= 0


class SeldonianConstraint:
    """
    Tier 3: Seldonian Constraints (Probabilistic)
    
    High-confidence bounds on safety-relevant outcome probabilities.
    Specification: P(g(θ) > 0) ≤ α
    """
    
    def __init__(self):
        """Initialize with constraint configuration."""
        self.alpha = CONFIG.seldonian.alpha
        self.confidence = CONFIG.seldonian.confidence
    
    def compute_violation_probability(self, 
                                       outcome_samples: np.ndarray,
                                       threshold: float = 70.0) -> float:
        """
        Compute probability of safety violation.
        
        Args:
            outcome_samples: Monte Carlo samples of predicted outcomes
            threshold: Safety threshold (glucose < threshold = violation)
            
        Returns:
            Estimated probability of violation
        """
        if len(outcome_samples) == 0:
            return 1.0  # Conservative if no data
        
        violations = np.sum(outcome_samples < threshold)
        return violations / len(outcome_samples)
    
    def compute_upper_bound(self, 
                            outcome_samples: np.ndarray,
                            threshold: float = 70.0) -> float:
        """
        Compute high-confidence upper bound on violation probability.
        
        Uses Hoeffding bound for conservative estimate.
        
        Args:
            outcome_samples: Monte Carlo samples
            threshold: Safety threshold
            
        Returns:
            Upper bound on P(violation)
        """
        n = len(outcome_samples)
        if n == 0:
            return 1.0
        
        # Point estimate
        p_hat = self.compute_violation_probability(outcome_samples, threshold)
        
        # Hoeffding bound: P_upper = p_hat + sqrt(log(1/δ) / 2n)
        delta = 1 - self.confidence
        margin = np.sqrt(np.log(1 / delta) / (2 * n))
        
        return min(1.0, p_hat + margin)
    
    def evaluate(self, outcome_samples: np.ndarray,
                 threshold: float = 70.0) -> SafetyResult:
        """
        Evaluate constraint satisfaction.
        
        Args:
            outcome_samples: Monte Carlo samples of predicted outcomes
            threshold: Safety threshold
            
        Returns:
            SafetyResult with decision
        """
        upper_bound = self.compute_upper_bound(outcome_samples, threshold)
        
        if upper_bound > self.alpha:
            return SafetyResult(
                decision=SafetyDecision.BLOCK,
                tier_triggered=3,
                reason=f"SELDONIAN VIOLATION: P_upper={upper_bound:.3f} > α={self.alpha}"
            )
        
        return SafetyResult(
            decision=SafetyDecision.ALLOW,
            tier_triggered=0,
            reason=f"SELDONIAN SAFE: P_upper={upper_bound:.3f} ≤ α={self.alpha}"
        )


class SimplexSafetySupervisor:
    """
    Complete Simplex Safety Supervisor.
    
    Combines all three tiers with priority resolution:
    Tier 1 > Tier 2 > Tier 3
    """
    
    def __init__(self):
        """Initialize all safety components."""
        self.reflex = ReflexController()
        self.stl = STLMonitor()
        self.seldonian = SeldonianConstraint()
    
    def evaluate(self,
                 current_glucose: float,
                 predicted_trajectory: np.ndarray,
                 outcome_samples: np.ndarray,
                 proposed_action: float) -> SafetyResult:
        """
        Full safety evaluation with tier priority.
        
        Args:
            current_glucose: Current glucose reading
            predicted_trajectory: Predicted glucose trajectory
            outcome_samples: Monte Carlo outcome samples
            proposed_action: Proposed insulin dose
            
        Returns:
            SafetyResult from highest-priority blocking tier
        """
        # Tier 1: Reflex (highest priority)
        tier1_result = self.reflex.evaluate(current_glucose, proposed_action)
        if tier1_result.decision in [SafetyDecision.BLOCK, SafetyDecision.EMERGENCY]:
            return tier1_result
        
        # Tier 2: STL Monitor
        tier2_result = self.stl.evaluate(predicted_trajectory, proposed_action)
        if tier2_result.decision == SafetyDecision.BLOCK:
            return tier2_result
        
        # Tier 3: Seldonian Constraints
        tier3_result = self.seldonian.evaluate(outcome_samples)
        if tier3_result.decision == SafetyDecision.BLOCK:
            return tier3_result
        
        # All tiers allow
        return SafetyResult(
            decision=SafetyDecision.ALLOW,
            tier_triggered=0,
            reason="All safety tiers passed"
        )
