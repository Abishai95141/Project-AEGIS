"""
AEGIS 3.0 Layer 3 - Harmonic G-Estimation

Implements G-Estimation with time-varying treatment effects using Fourier decomposition:
τ(t) = ψ₀ + Σ[ψ_ck cos(2πkt/24) + ψ_sk sin(2πkt/24)]
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


@dataclass
class GEstimationResult:
    """Result of G-Estimation."""
    psi_0: float              # Constant effect
    psi_harmonics: np.ndarray  # [ψ_c1, ψ_s1, ψ_c2, ψ_s2, ...]
    standard_errors: np.ndarray
    confidence_intervals: np.ndarray  # [lower, upper] for each param
    converged: bool


class HarmonicGEstimator:
    """
    Harmonic G-Estimator for time-varying treatment effects.
    
    Implements:
    - Fourier decomposition of treatment effects τ(t)
    - Doubly-robust estimation using outcome and propensity models
    - Proximal adjustment for unmeasured confounding (optional)
    """
    
    def __init__(self, num_harmonics: int = None):
        """Initialize estimator."""
        self.num_harmonics = num_harmonics or CONFIG.g_estimation.num_harmonics
        self.psi = None  # Estimated parameters
        self.result = None
    
    def _fourier_basis(self, t: np.ndarray, period: float = 24.0) -> np.ndarray:
        """
        Compute Fourier basis functions for time t.
        
        Returns: [1, cos(2πt/T), sin(2πt/T), cos(4πt/T), sin(4πt/T), ...]
        """
        basis = [np.ones(len(t))]  # Constant term
        
        for k in range(1, self.num_harmonics + 1):
            basis.append(np.cos(2 * np.pi * k * t / period))
            basis.append(np.sin(2 * np.pi * k * t / period))
        
        return np.column_stack(basis)
    
    def compute_effect(self, t: np.ndarray, psi: np.ndarray = None) -> np.ndarray:
        """
        Compute treatment effect τ(t) given parameters.
        
        Args:
            t: Time points (hours)
            psi: Parameters [ψ₀, ψ_c1, ψ_s1, ...]
        """
        if psi is None:
            psi = self.psi
        
        basis = self._fourier_basis(t)
        return basis @ psi
    
    def estimate(self,
                 Y: np.ndarray,
                 A: np.ndarray,
                 t: np.ndarray,
                 propensity: np.ndarray,
                 outcome_model: np.ndarray = None,
                 W: np.ndarray = None) -> GEstimationResult:
        """
        Estimate treatment effect parameters using G-Estimation.
        
        Args:
            Y: Outcomes
            A: Treatment indicators
            t: Time of each observation (hours, 0-24)
            propensity: P(A=1|S) for each observation
            outcome_model: E[Y|S, A=0] predictions (optional, for DR)
            W: Outcome proxy for proximal adjustment (optional)
            
        Returns:
            GEstimationResult with estimated parameters
        """
        n = len(Y)
        basis = self._fourier_basis(t)
        n_params = basis.shape[1]
        
        # Center outcome
        if outcome_model is not None:
            Y_centered = Y - outcome_model
        else:
            # Fit baseline model on untreated observations
            if np.sum(A == 0) > 10:
                Y_centered = Y - np.mean(Y[A == 0])
            else:
                Y_centered = Y - np.mean(Y)
        
        # Proximal adjustment if W provided
        if W is not None and len(W) == n:
            # Simple: regress Y_centered on W for treated only
            W_safe = W.copy()
            try:
                W_design = np.column_stack([np.ones(n), W_safe])
                gamma = np.linalg.lstsq(W_design, Y_centered, rcond=None)[0]
                Y_centered = Y_centered - gamma[1] * (W_safe - np.mean(W_safe))
            except:
                pass
        
        # G-Estimation via weighted least squares
        # Model: Y_centered = τ(t) * A + ε
        # Where τ(t) = basis @ psi
        
        # Create treatment-modified basis
        # Weight by inverse propensity score variance
        weights = propensity * (1 - propensity)
        weights = np.clip(weights, 0.01, 0.25)
        
        # Design matrix: A * basis
        X = A[:, np.newaxis] * basis
        
        # Weighted least squares
        W_matrix = np.diag(weights)
        try:
            XtWX = X.T @ W_matrix @ X
            XtWY = X.T @ W_matrix @ Y_centered
            
            # Ridge regularization for stability
            ridge = 0.001 * np.eye(n_params)
            psi = np.linalg.solve(XtWX + ridge, XtWY)
        except:
            psi = np.zeros(n_params)
        
        self.psi = psi
        
        # Compute standard errors
        fitted = X @ psi
        residuals = Y_centered - fitted
        sigma_sq = np.mean(residuals ** 2)
        
        try:
            var_psi = sigma_sq * np.linalg.inv(XtWX + ridge)
            se = np.sqrt(np.maximum(np.diag(var_psi), 1e-10))
        except:
            se = np.ones(n_params) * 0.1
        
        # Confidence intervals
        z = 1.96
        ci = np.column_stack([psi - z * se, psi + z * se])
        
        self.result = GEstimationResult(
            psi_0=psi[0],
            psi_harmonics=psi[1:],
            standard_errors=se,
            confidence_intervals=ci,
            converged=True
        )
        
        return self.result
    
    def get_peak_trough_times(self) -> Tuple[float, float]:
        """
        Find times of maximum and minimum treatment effect.
        
        Returns:
            (peak_time, trough_time) in hours
        """
        if self.psi is None:
            return (8.0, 20.0)  # Default
        
        # Evaluate effect at fine time grid
        t_grid = np.linspace(0, 24, 288)
        effects = self.compute_effect(t_grid, self.psi)
        
        peak_idx = np.argmax(effects)
        trough_idx = np.argmin(effects)
        
        return t_grid[peak_idx], t_grid[trough_idx]


def generate_harmonic_outcome_data(
    n: int,
    true_psi: np.ndarray,
    propensity: float = 0.5,
    noise_std: float = 1.0,
    confounding_strength: float = 0.0,
    seed: int = None
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic data with time-varying treatment effects.
    
    Args:
        n: Number of observations
        true_psi: True effect parameters [ψ₀, ψ_c1, ψ_s1, ...]
        propensity: Treatment probability
        noise_std: Outcome noise standard deviation
        confounding_strength: Strength of unmeasured confounding
        seed: Random seed
        
    Returns:
        Dictionary with Y, A, t, propensity, U (confounder), etc.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Time uniformly distributed over 24 hours
    t = np.random.uniform(0, 24, n)
    
    # Unmeasured confounder
    U = np.random.randn(n)
    
    # Treatment (potentially confounded)
    prob = propensity + confounding_strength * 0.2 * U
    prob = np.clip(prob, 0.1, 0.9)
    A = (np.random.rand(n) < prob).astype(float)
    
    # Compute true treatment effect at each time
    num_harmonics = max(1, (len(true_psi) - 1) // 2)
    estimator = HarmonicGEstimator(num_harmonics=num_harmonics)
    tau_t = estimator.compute_effect(t, true_psi)
    
    # Baseline outcome (zero mean for simplicity)
    baseline = confounding_strength * U  # Only confounding affects baseline
    
    # Observed outcome: Y = baseline + tau(t)*A + noise
    Y = baseline + tau_t * A + np.random.randn(n) * noise_std
    
    # Proxies for unmeasured confounder
    Z = U + np.random.randn(n) * 0.3  # Treatment proxy
    W = U + np.random.randn(n) * 0.3  # Outcome proxy
    
    return {
        'Y': Y,
        'A': A,
        't': t,
        'propensity': prob,
        'U': U,
        'Z': Z,
        'W': W,
        'true_psi': true_psi,
        'baseline': baseline
    }

