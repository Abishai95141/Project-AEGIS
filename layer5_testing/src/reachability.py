"""
AEGIS 3.0 Layer 5 - Reachability Analysis and Cold Start Safety

Implements:
- Reachability analysis with conservative physiological bounds
- Cold start safety via hierarchical Bayesian priors
- Safety relaxation schedule

Paper reference: Section 5.5.2 and 5.5.3
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


@dataclass
class ReachabilitySet:
    """Result of reachability analysis."""
    min_state: float      # Minimum reachable state
    max_state: float      # Maximum reachable state
    contains_unsafe: bool # Whether unsafe states are reachable
    unsafe_overlap: float # Amount of overlap with unsafe region


class ReachabilityAnalyzer:
    """
    Reachability Analysis for Safety Verification.
    
    Computes worst-case future states using conservative physiological bounds
    independent of the patient-specific Digital Twin.
    
    This breaks the circularity problem: safety doesn't depend on potentially
    inaccurate patient models.
    """
    
    def __init__(self):
        """Initialize with conservative bounds."""
        self.max_rate = CONFIG.reachability.max_glucose_rate
        self.onset_min = CONFIG.reachability.insulin_onset_min
        self.onset_max = CONFIG.reachability.insulin_onset_max
        self.glucose_min = CONFIG.reachability.glucose_min
        self.glucose_max = CONFIG.reachability.glucose_max
    
    def compute_reachable_set(self,
                               current_state: float,
                               action: float,
                               time_horizon: float) -> ReachabilitySet:
        """
        Compute reachable states from current state under action.
        
        Uses overapproximation: all actual trajectories are contained.
        
        Args:
            current_state: Current glucose (mg/dL)
            action: Insulin dose (units)
            time_horizon: Prediction horizon (minutes)
            
        Returns:
            ReachabilitySet with bounds
        """
        # Without treatment effect (natural variation)
        max_natural_change = self.max_rate * time_horizon
        
        # Treatment effect bounds (insulin lowers glucose)
        # After onset, insulin has effect proportional to dose
        # Conservative: assume effect could be anywhere in range
        insulin_effect_min = 0  # Could have no effect
        insulin_effect_max = action * 10  # ~10 mg/dL per unit (conservative)
        
        # Reachable set: current +/- natural variation - insulin effect
        # Most conservative bounds:
        min_reachable = max(
            self.glucose_min,
            current_state - max_natural_change - insulin_effect_max
        )
        max_reachable = min(
            self.glucose_max,
            current_state + max_natural_change - insulin_effect_min
        )
        
        # Check unsafe regions (hypoglycemia)
        unsafe_threshold = CONFIG.reflex.hypo_warning  # 70 mg/dL
        contains_unsafe = min_reachable < unsafe_threshold
        
        if contains_unsafe:
            unsafe_overlap = unsafe_threshold - min_reachable
        else:
            unsafe_overlap = 0.0
        
        return ReachabilitySet(
            min_state=min_reachable,
            max_state=max_reachable,
            contains_unsafe=contains_unsafe,
            unsafe_overlap=unsafe_overlap
        )
    
    def is_action_safe(self, 
                       current_state: float,
                       action: float,
                       time_horizon: float = 60.0) -> Tuple[bool, ReachabilitySet]:
        """
        Determine if action is safe using reachability.
        
        Args:
            current_state: Current glucose
            action: Proposed action
            time_horizon: Prediction horizon (minutes)
            
        Returns:
            (is_safe, reachability_set)
        """
        reachable = self.compute_reachable_set(current_state, action, time_horizon)
        is_safe = not reachable.contains_unsafe
        return is_safe, reachable


class ColdStartSafety:
    """
    Cold Start Safety via Hierarchical Bayesian Priors.
    
    On Day 1, uses conservative population-derived bounds.
    As patient data accumulates, relaxes to individual posterior.
    """
    
    def __init__(self):
        """Initialize with population prior and relaxation schedule."""
        self.theta_pop = CONFIG.cold_start.theta_pop_mean
        self.sigma_between = CONFIG.cold_start.theta_pop_std
        self.alpha_strict = CONFIG.cold_start.alpha_strict
        self.alpha_standard = CONFIG.cold_start.alpha_standard
        self.tau = CONFIG.cold_start.tau_days
    
    def get_day1_bound(self) -> float:
        """
        Compute Day 1 conservative safety bound.
        
        Uses 99th percentile of population distribution.
        θ_safe = θ_pop - z_{0.01} * σ_between
        
        Returns:
            Conservative parameter bound
        """
        z_alpha = 2.326  # z for α = 0.01 (one-sided)
        theta_safe = self.theta_pop - z_alpha * self.sigma_between
        return theta_safe
    
    def get_bound_at_day(self, day: int) -> float:
        """
        Compute safety bound for given day.
        
        Uses relaxation schedule to transition from strict to standard.
        
        Args:
            day: Day number (1 = first day)
            
        Returns:
            Safety bound for that day
        """
        alpha_day = self.get_alpha_at_day(day)
        
        # Convert alpha to z-score
        # As alpha increases (less strict), z decreases (bound relaxes)
        from scipy import stats
        z_alpha = stats.norm.ppf(1 - alpha_day)
        
        # Combine population uncertainty with relaxing strictness
        theta_safe = self.theta_pop - z_alpha * self.sigma_between
        return theta_safe
    
    def get_alpha_at_day(self, day: int) -> float:
        """
        Compute safety level α at given day.
        
        α_t = α_strict * exp(-t/τ) + α_standard * (1 - exp(-t/τ))
        
        Args:
            day: Day number
            
        Returns:
            Current α level
        """
        t = max(0, day - 1)  # Day 1 = t=0
        decay = np.exp(-t / self.tau)
        alpha_t = self.alpha_strict * decay + self.alpha_standard * (1 - decay)
        return alpha_t
    
    def get_relaxation_schedule(self, max_days: int = 30) -> Dict[int, float]:
        """
        Get full relaxation schedule.
        
        Args:
            max_days: Number of days to compute
            
        Returns:
            Dictionary mapping day -> α value
        """
        schedule = {}
        for day in range(1, max_days + 1):
            schedule[day] = self.get_alpha_at_day(day)
        return schedule
    
    def is_bound_monotonic(self, max_days: int = 30) -> bool:
        """
        Verify that safety bound monotonically relaxes.
        
        The bound should never become stricter over time.
        
        Returns:
            True if monotonically relaxing
        """
        schedule = self.get_relaxation_schedule(max_days)
        alphas = list(schedule.values())
        
        # Alpha should monotonically increase (less strict)
        for i in range(1, len(alphas)):
            if alphas[i] < alphas[i-1]:
                return False
        return True


def generate_glucose_trajectory(
    initial: float,
    duration_minutes: int,
    insulin_dose: float = 0,
    meal_carbs: float = 0,
    noise_std: float = 5.0,
    seed: int = None
) -> np.ndarray:
    """
    Generate synthetic glucose trajectory.
    
    Simple model for testing purposes.
    
    Args:
        initial: Starting glucose (mg/dL)
        duration_minutes: Trajectory length
        insulin_dose: Insulin given at t=0
        meal_carbs: Carbs consumed at t=0
        noise_std: CGM noise standard deviation
        seed: Random seed
        
    Returns:
        Array of glucose values (one per minute)
    """
    if seed is not None:
        np.random.seed(seed)
    
    glucose = np.zeros(duration_minutes)
    glucose[0] = initial
    
    for t in range(1, duration_minutes):
        # Baseline trend (slight decrease toward 100)
        trend = -0.02 * (glucose[t-1] - 100)
        
        # Insulin effect (delayed, peaks at 30-60 min)
        if insulin_dose > 0 and t > 15:
            # Insulin action profile
            insulin_effect = -insulin_dose * 0.5 * np.exp(-(t - 45)**2 / 500)
        else:
            insulin_effect = 0
        
        # Meal effect (peaks at 30-45 min)
        if meal_carbs > 0:
            meal_effect = meal_carbs * 0.3 * np.exp(-(t - 30)**2 / 300)
        else:
            meal_effect = 0
        
        # Random variation
        noise = np.random.randn() * noise_std * 0.1
        
        # Update glucose
        glucose[t] = glucose[t-1] + trend + insulin_effect + meal_effect + noise
        
        # Physiological bounds
        glucose[t] = np.clip(glucose[t], 40, 400)
    
    # Add measurement noise
    glucose += np.random.randn(duration_minutes) * noise_std
    glucose = np.clip(glucose, 40, 400)
    
    return glucose
