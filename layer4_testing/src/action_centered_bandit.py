"""
AEGIS 3.0 Layer 4 - Action-Centered Contextual Bandit

Implements the action-centered bandit that learns treatment effects τ(S) 
instead of full Q(S,A), achieving variance reduction.

Paper reference: Section 5.4.1
R_t = f(S_t) + A_t · τ(S_t) + ε_t
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


@dataclass
class BanditState:
    """Current state of the bandit learner."""
    theta_mean: np.ndarray      # Posterior mean for τ parameters
    theta_cov: np.ndarray       # Posterior covariance
    total_reward: float         # Cumulative reward
    regret: float               # Cumulative regret
    timestep: int               # Current timestep


class ActionCenteredBandit:
    """
    Action-Centered Contextual Bandit.
    
    Key insight: Learn only τ(S) - the treatment effect, treating 
    baseline f(S) as noise to be subtracted.
    
    This reduces variance because f(S) has high variance but is 
    unrelated to treatment choice.
    """
    
    def __init__(self, 
                 context_dim: int = None,
                 num_actions: int = None,
                 prior_mean: np.ndarray = None,
                 prior_variance: float = 1.0):
        """Initialize bandit with Bayesian prior."""
        context_dim = context_dim or CONFIG.bandit.context_dim
        num_actions = num_actions or CONFIG.bandit.num_actions
        
        self.context_dim = context_dim
        self.num_actions = num_actions
        
        # For action-centered: θ maps context to treatment effect
        # τ(S) = S @ θ  (linear model)
        self.param_dim = context_dim
        
        # Prior: θ ~ N(prior_mean, prior_variance * I)
        if prior_mean is None:
            self.theta_mean = np.zeros(self.param_dim)
        else:
            self.theta_mean = prior_mean.copy()
        
        self.theta_cov = prior_variance * np.eye(self.param_dim)
        self.theta_precision = np.eye(self.param_dim) / prior_variance
        
        # Noise variance (assumed known for simplicity)
        self.noise_var = CONFIG.bandit.noise_variance
        
        # History
        self.history = []
        self.regret_history = []
        self.cumulative_regret = 0.0
    
    def get_treatment_effect(self, context: np.ndarray, theta: np.ndarray = None) -> float:
        """Compute treatment effect τ(S) = S @ θ."""
        if theta is None:
            theta = self.theta_mean
        return context @ theta
    
    def sample_theta(self) -> np.ndarray:
        """Thompson Sampling: draw θ from posterior."""
        return np.random.multivariate_normal(self.theta_mean, self.theta_cov)
    
    def select_action(self, context: np.ndarray, 
                      safe_actions: List[int] = None) -> Tuple[int, float]:
        """
        Select action using Thompson Sampling.
        
        Args:
            context: Current context S_t
            safe_actions: Optional list of allowed actions
            
        Returns:
            (selected_action, expected_effect)
        """
        if safe_actions is None:
            safe_actions = list(range(self.num_actions))
        
        # Sample θ from posterior
        theta_sample = self.sample_theta()
        
        # Compute expected treatment effect
        tau = self.get_treatment_effect(context, theta_sample)
        
        # For discrete actions: action 0 = no treatment, action a = treatment level a
        # Expected reward difference: action a gives τ * a
        expected_rewards = [tau * a for a in safe_actions]
        
        best_idx = np.argmax(expected_rewards)
        selected_action = safe_actions[best_idx]
        
        return selected_action, expected_rewards[best_idx]
    
    def update(self, context: np.ndarray, action: int, reward: float,
               baseline_estimate: float = 0.0):
        """
        Update posterior using observed outcome.
        
        Uses action-centered update: learns from (R - f̂(S)) / A = τ̂
        
        Args:
            context: Observed context
            action: Taken action
            reward: Observed reward
            baseline_estimate: Estimate of f(S), default 0
        """
        if action == 0:
            # No treatment - can't learn about τ from this observation
            # But can update baseline estimate (not implemented here)
            return
        
        # Action-centered: Y = f(S) + A*τ(S) + ε
        # τ̂ = (Y - f̂(S)) / A
        tau_obs = (reward - baseline_estimate) / action
        
        # Bayesian linear regression update
        # Posterior: θ | data ~ N(μ_n, Σ_n)
        x = context.reshape(-1, 1)  # Column vector
        
        # Update precision and mean
        self.theta_precision += (x @ x.T) / self.noise_var
        self.theta_cov = np.linalg.inv(self.theta_precision)
        self.theta_mean = self.theta_cov @ (
            self.theta_precision @ self.theta_mean + 
            x.flatten() * tau_obs / self.noise_var
        )
        
        # Store history
        self.history.append({
            'context': context.copy(),
            'action': action,
            'reward': reward,
            'tau_obs': tau_obs
        })
    
    def compute_regret(self, context: np.ndarray, action: int, 
                       true_tau: float) -> float:
        """Compute instantaneous regret."""
        # Optimal action would give τ * a_max
        optimal_reward = true_tau * (self.num_actions - 1)  # Best action
        actual_reward = true_tau * action
        
        regret = optimal_reward - actual_reward
        self.cumulative_regret += max(0, regret)
        self.regret_history.append(self.cumulative_regret)
        
        return regret
    
    def get_state(self) -> BanditState:
        """Get current bandit state."""
        return BanditState(
            theta_mean=self.theta_mean.copy(),
            theta_cov=self.theta_cov.copy(),
            total_reward=sum(h['reward'] for h in self.history),
            regret=self.cumulative_regret,
            timestep=len(self.history)
        )


class StandardQBandit:
    """
    Standard Q-learning bandit for comparison.
    
    Learns full Q(S, A) instead of just τ(S).
    This has higher variance due to baseline fluctuations.
    """
    
    def __init__(self, 
                 context_dim: int = None,
                 num_actions: int = None,
                 prior_variance: float = 1.0):
        """Initialize standard Q-learner."""
        context_dim = context_dim or CONFIG.bandit.context_dim
        num_actions = num_actions or CONFIG.bandit.num_actions
        
        self.context_dim = context_dim
        self.num_actions = num_actions
        
        # Separate parameters for each action: Q(S, a) = S @ θ_a
        self.param_dim = context_dim * num_actions
        
        self.theta_mean = np.zeros(self.param_dim)
        self.theta_cov = prior_variance * np.eye(self.param_dim)
        self.theta_precision = np.eye(self.param_dim) / prior_variance
        
        self.noise_var = CONFIG.bandit.noise_variance + CONFIG.bandit.baseline_variance
        
        self.history = []
        self.cumulative_regret = 0.0
        self.regret_history = []
    
    def _get_features(self, context: np.ndarray, action: int) -> np.ndarray:
        """Create feature vector for (S, A) pair."""
        features = np.zeros(self.param_dim)
        start = action * self.context_dim
        features[start:start + self.context_dim] = context
        return features
    
    def sample_theta(self) -> np.ndarray:
        """Thompson Sampling."""
        return np.random.multivariate_normal(self.theta_mean, self.theta_cov)
    
    def select_action(self, context: np.ndarray, 
                      safe_actions: List[int] = None) -> Tuple[int, float]:
        """Select action using Thompson Sampling."""
        if safe_actions is None:
            safe_actions = list(range(self.num_actions))
        
        theta_sample = self.sample_theta()
        
        q_values = []
        for a in safe_actions:
            features = self._get_features(context, a)
            q_values.append(features @ theta_sample)
        
        best_idx = np.argmax(q_values)
        return safe_actions[best_idx], q_values[best_idx]
    
    def update(self, context: np.ndarray, action: int, reward: float, **kwargs):
        """Update Q-function estimate."""
        features = self._get_features(context, action)
        x = features.reshape(-1, 1)
        
        self.theta_precision += (x @ x.T) / self.noise_var
        self.theta_cov = np.linalg.inv(self.theta_precision)
        self.theta_mean = self.theta_cov @ (
            self.theta_precision @ self.theta_mean +
            features * reward / self.noise_var
        )
        
        self.history.append({
            'context': context.copy(),
            'action': action,
            'reward': reward
        })
    
    def compute_regret(self, context: np.ndarray, action: int, 
                       true_optimal_reward: float, actual_reward: float) -> float:
        """Compute regret."""
        regret = true_optimal_reward - actual_reward
        self.cumulative_regret += max(0, regret)
        self.regret_history.append(self.cumulative_regret)
        return regret


def generate_bandit_environment(
    context_dim: int = 5,
    num_actions: int = 3,
    true_tau_params: np.ndarray = None,
    baseline_variance: float = 25.0,
    noise_variance: float = 1.0,
    seed: int = None
) -> Dict:
    """
    Generate synthetic bandit environment.
    
    Returns environment parameters and functions.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if true_tau_params is None:
        true_tau_params = np.random.randn(context_dim) * 0.5
    
    def generate_context():
        return np.random.randn(context_dim)
    
    def baseline_function(context):
        # High variance baseline (nuisance)
        return np.random.randn() * np.sqrt(baseline_variance)
    
    def treatment_effect(context):
        # τ(S) = S @ θ_true
        return context @ true_tau_params
    
    def get_reward(context, action):
        f_s = baseline_function(context)
        tau_s = treatment_effect(context)
        noise = np.random.randn() * np.sqrt(noise_variance)
        return f_s + action * tau_s + noise
    
    def optimal_action(context):
        tau = treatment_effect(context)
        if tau > 0:
            return num_actions - 1  # Highest treatment
        else:
            return 0  # No treatment
    
    return {
        'context_dim': context_dim,
        'num_actions': num_actions,
        'true_tau_params': true_tau_params,
        'generate_context': generate_context,
        'baseline_function': baseline_function,
        'treatment_effect': treatment_effect,
        'get_reward': get_reward,
        'optimal_action': optimal_action,
        'baseline_variance': baseline_variance,
        'noise_variance': noise_variance
    }
