"""
AEGIS 3.0 Layer 4 - Counterfactual Thompson Sampling (CTS)

Implements CTS that prevents posterior collapse for blocked actions
by using Digital Twin predictions for counterfactual updates.

Paper reference: Section 5.4.2, Algorithm 5.1
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass

try:
    from .config import CONFIG
    from .action_centered_bandit import ActionCenteredBandit
except ImportError:
    from config import CONFIG
    from action_centered_bandit import ActionCenteredBandit


@dataclass 
class CTSResult:
    """Result of one CTS step."""
    proposed_action: int
    executed_action: int
    was_blocked: bool
    counterfactual_update: bool
    imputed_outcome: Optional[float]
    lambda_confidence: float


class DigitalTwin:
    """
    Simplified Digital Twin for counterfactual prediction.
    
    In full AEGIS, this would be the UDE from Layer 2.
    Here we use a simple model for testing.
    """
    
    def __init__(self, 
                 true_tau_params: np.ndarray,
                 prediction_noise: float = 0.5,
                 confidence: float = 0.8):
        """
        Initialize Digital Twin.
        
        Args:
            true_tau_params: True treatment effect parameters (for simulation)
            prediction_noise: Noise in predictions
            confidence: Default confidence λ
        """
        self.true_tau_params = true_tau_params
        self.prediction_noise = prediction_noise
        self.base_confidence = confidence
    
    def predict(self, context: np.ndarray, action: int) -> float:
        """
        Predict outcome for counterfactual action.
        
        Returns: Predicted treatment effect τ(S) * A + noise
        """
        true_tau = context @ self.true_tau_params
        prediction = action * true_tau + np.random.randn() * self.prediction_noise
        return prediction
    
    def get_confidence(self, context: np.ndarray, action: int) -> float:
        """
        Get prediction confidence λ ∈ (0, 1).
        
        Higher confidence = stronger posterior update.
        """
        # In reality, confidence depends on how much data we have for this context
        # For testing, use base confidence with some variation
        variation = np.random.uniform(-0.1, 0.1)
        return np.clip(self.base_confidence + variation, 0.3, 0.95)


class SafetyEvaluator:
    """
    Safety evaluator that determines if actions are safe.
    """
    
    def __init__(self, 
                 safety_threshold: float = 0.7,
                 context_dependent: bool = True):
        """
        Initialize safety evaluator.
        
        Args:
            safety_threshold: Actions above this are unsafe
            context_dependent: If True, threshold varies with context
        """
        self.safety_threshold = safety_threshold
        self.context_dependent = context_dependent
    
    def is_safe(self, action: int, context: np.ndarray, 
                num_actions: int = 3) -> bool:
        """Check if action is safe in given context."""
        # Normalize action to [0, 1]
        action_normalized = action / (num_actions - 1)
        
        if self.context_dependent:
            # Threshold varies with context (e.g., lower for vulnerable patients)
            threshold = self.safety_threshold - 0.1 * np.mean(context)
            threshold = np.clip(threshold, 0.3, 0.9)
        else:
            threshold = self.safety_threshold
        
        return action_normalized <= threshold
    
    def get_safe_actions(self, context: np.ndarray, 
                         num_actions: int = 3) -> List[int]:
        """Get list of safe actions for context."""
        safe = []
        for a in range(num_actions):
            if self.is_safe(a, context, num_actions):
                safe.append(a)
        return safe if safe else [0]  # Always allow no-treatment


class CounterfactualThompsonSampling:
    """
    Counterfactual Thompson Sampling (CTS).
    
    Key innovation: When an action is blocked, update the posterior
    using Digital Twin predictions with discounted likelihood.
    
    This prevents posterior collapse for blocked actions.
    """
    
    def __init__(self,
                 bandit: ActionCenteredBandit,
                 digital_twin: DigitalTwin,
                 safety_evaluator: SafetyEvaluator,
                 lambda_default: float = None):
        """
        Initialize CTS.
        
        Args:
            bandit: Underlying bandit learner
            digital_twin: For counterfactual predictions
            safety_evaluator: For safety checks
            lambda_default: Default imputation confidence
        """
        self.bandit = bandit
        self.digital_twin = digital_twin
        self.safety = safety_evaluator
        self.lambda_default = lambda_default or CONFIG.cts.lambda_confidence
        
        # Track blocking statistics
        self.total_steps = 0
        self.blocked_count = 0
        self.counterfactual_updates = 0
        
        # Track posterior variance for blocked actions
        self.blocked_action_variances = []
    
    def step(self, context: np.ndarray) -> CTSResult:
        """
        Execute one CTS step (Algorithm 5.1).
        
        1. Sample θ from posterior
        2. Compute optimal action a*
        3. Check safety
        4. If blocked, do counterfactual update
        5. Execute best safe action
        
        Returns:
            CTSResult with step details
        """
        self.total_steps += 1
        
        # Step 1 & 2: Sample and compute optimal
        theta_sample = self.bandit.sample_theta()
        tau = context @ theta_sample
        
        # Optimal action (without safety)
        if tau > 0:
            proposed_action = self.bandit.num_actions - 1
        else:
            proposed_action = 0
        
        # Step 3: Safety check
        is_safe = self.safety.is_safe(
            proposed_action, context, self.bandit.num_actions
        )
        
        was_blocked = not is_safe
        counterfactual_update = False
        imputed_outcome = None
        lambda_conf = self.lambda_default
        
        if was_blocked:
            self.blocked_count += 1
            
            # Step 4: Counterfactual update
            imputed_outcome = self.digital_twin.predict(context, proposed_action)
            lambda_conf = self.digital_twin.get_confidence(context, proposed_action)
            
            # Update posterior with discounted likelihood
            self._counterfactual_update(
                context, proposed_action, imputed_outcome, lambda_conf
            )
            counterfactual_update = True
            self.counterfactual_updates += 1
            
            # Track variance for blocked action analysis
            var = np.trace(self.bandit.theta_cov)
            self.blocked_action_variances.append(var)
        
        # Step 5: Execute best safe action
        safe_actions = self.safety.get_safe_actions(
            context, self.bandit.num_actions
        )
        executed_action, _ = self.bandit.select_action(context, safe_actions)
        
        return CTSResult(
            proposed_action=proposed_action,
            executed_action=executed_action,
            was_blocked=was_blocked,
            counterfactual_update=counterfactual_update,
            imputed_outcome=imputed_outcome,
            lambda_confidence=lambda_conf
        )
    
    def _counterfactual_update(self, 
                                context: np.ndarray,
                                action: int,
                                imputed_outcome: float,
                                lambda_conf: float):
        """
        Update posterior using counterfactual outcome.
        
        Uses discounted likelihood: weight update by λ.
        """
        if action == 0:
            return
        
        # Imputed τ observation
        tau_imputed = imputed_outcome / action
        
        # Discounted Bayesian update
        # Effective noise variance = original / λ (higher λ = stronger update)
        effective_noise_var = self.bandit.noise_var / (lambda_conf + 1e-10)
        
        x = context.reshape(-1, 1)
        self.bandit.theta_precision += (x @ x.T) / effective_noise_var
        self.bandit.theta_cov = np.linalg.inv(self.bandit.theta_precision)
        self.bandit.theta_mean = self.bandit.theta_cov @ (
            self.bandit.theta_precision @ self.bandit.theta_mean +
            x.flatten() * tau_imputed / effective_noise_var
        )
    
    def update_with_observation(self, context: np.ndarray, 
                                 action: int, reward: float):
        """Update bandit with actual observation."""
        self.bandit.update(context, action, reward)
    
    def get_blocking_rate(self) -> float:
        """Get fraction of steps that were blocked."""
        if self.total_steps == 0:
            return 0.0
        return self.blocked_count / self.total_steps
    
    def get_variance_reduction(self) -> float:
        """
        Get variance reduction for blocked actions.
        
        Returns ratio of final/initial variance (lower is better).
        """
        if len(self.blocked_action_variances) < 2:
            return 1.0
        
        initial = self.blocked_action_variances[0]
        final = self.blocked_action_variances[-1]
        
        return final / initial if initial > 0 else 1.0


class StandardThompsonSampling:
    """
    Standard Thompson Sampling (no counterfactual updates).
    
    For comparison: blocked actions receive no updates,
    leading to posterior collapse.
    """
    
    def __init__(self,
                 bandit: ActionCenteredBandit,
                 safety_evaluator: SafetyEvaluator):
        """Initialize standard TS."""
        self.bandit = bandit
        self.safety = safety_evaluator
        
        self.total_steps = 0
        self.blocked_count = 0
        self.blocked_action_variances = []
    
    def step(self, context: np.ndarray) -> CTSResult:
        """Execute standard TS step (no counterfactual)."""
        self.total_steps += 1
        
        # Sample and compute optimal
        theta_sample = self.bandit.sample_theta()
        tau = context @ theta_sample
        
        proposed_action = self.bandit.num_actions - 1 if tau > 0 else 0
        
        # Safety check
        is_safe = self.safety.is_safe(
            proposed_action, context, self.bandit.num_actions
        )
        
        was_blocked = not is_safe
        
        if was_blocked:
            self.blocked_count += 1
            # NO UPDATE - this causes posterior collapse
            var = np.trace(self.bandit.theta_cov)
            self.blocked_action_variances.append(var)
        
        # Execute best safe action
        safe_actions = self.safety.get_safe_actions(
            context, self.bandit.num_actions
        )
        executed_action, _ = self.bandit.select_action(context, safe_actions)
        
        return CTSResult(
            proposed_action=proposed_action,
            executed_action=executed_action,
            was_blocked=was_blocked,
            counterfactual_update=False,
            imputed_outcome=None,
            lambda_confidence=0.0
        )
    
    def update_with_observation(self, context: np.ndarray,
                                 action: int, reward: float):
        """Update with actual observation."""
        self.bandit.update(context, action, reward)
    
    def get_variance_reduction(self) -> float:
        """Get variance (non-)reduction for blocked actions."""
        if len(self.blocked_action_variances) < 2:
            return 1.0
        
        initial = self.blocked_action_variances[0]
        final = self.blocked_action_variances[-1]
        
        return final / initial if initial > 0 else 1.0
