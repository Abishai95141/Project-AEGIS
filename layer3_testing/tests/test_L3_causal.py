"""
AEGIS 3.0 Layer 3 - Comprehensive Tests

Tests for:
- L3-1: MRT Positivity
- L3-2: Harmonic G-Estimation
- L3-3: Time-Varying Effects
- L3-4: Double Robustness
- L3-5: Proximal G-Estimation
"""

import sys
import os
import json
import numpy as np
from typing import Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from config import CONFIG, DATA_DIR, RESULTS_DIR
from g_estimation import HarmonicGEstimator, generate_harmonic_outcome_data


class TestL3_CausalInference:
    """
    Layer 3 Causal Inference Tests.
    """
    
    def __init__(self):
        self.results = {}
    
    def test_mrt_positivity(self) -> Dict:
        """L3-1: Test MRT positivity constraint."""
        print("\n--- Test L3-1: MRT Positivity Constraint ---")
        
        np.random.seed(CONFIG.random_seed)
        n = 1000
        
        # Generate diverse contexts
        glucose = np.random.normal(120, 30, n)
        time_of_day = np.random.uniform(0, 24, n)
        activity = np.random.choice([0, 1, 2], n)  # sedentary, light, active
        
        # Context-dependent randomization probabilities
        # Higher prob for high glucose, lower for active
        propensity = 0.5 + 0.15 * (glucose - 120) / 30 - 0.1 * (activity - 1)
        propensity = np.clip(propensity, CONFIG.mrt.epsilon, 1 - CONFIG.mrt.epsilon)
        
        # Check positivity
        min_p = np.min(propensity)
        max_p = np.max(propensity)
        
        violations = np.sum((propensity < CONFIG.mrt.min_positivity_bound) | 
                           (propensity > CONFIG.mrt.max_positivity_bound))
        
        passed = (min_p >= CONFIG.mrt.min_positivity_bound and 
                  max_p <= CONFIG.mrt.max_positivity_bound)
        
        result = {
            'test_name': 'MRT Positivity Constraint',
            'test_id': 'L3-1',
            'min_propensity': float(min_p),
            'max_propensity': float(max_p),
            'positivity_bound': CONFIG.mrt.epsilon,
            'violations': int(violations),
            'passed': passed,
            'interpretation': f"Propensity ∈ [{min_p:.3f}, {max_p:.3f}], violations: {violations}"
        }
        
        print(f"  Min propensity: {min_p:.3f}")
        print(f"  Max propensity: {max_p:.3f}")
        print(f"  {result['interpretation']} - {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def test_g_estimation(self) -> Dict:
        """L3-2: Test Harmonic G-Estimation parameter recovery."""
        print("\n--- Test L3-2: Harmonic G-Estimation ---")
        
        # True parameters
        true_psi = np.array([
            CONFIG.g_estimation.true_psi_0,   # ψ₀ = 0.5
            CONFIG.g_estimation.true_psi_c1,  # ψ_c1 = 0.3
            CONFIG.g_estimation.true_psi_s1   # ψ_s1 = 0.2
        ])
        
        # Generate data
        data = generate_harmonic_outcome_data(
            n=CONFIG.num_observations,
            true_psi=true_psi,
            propensity=0.5,
            noise_std=1.0,
            confounding_strength=0.0,  # No confounding
            seed=CONFIG.random_seed
        )
        
        # Estimate
        estimator = HarmonicGEstimator(num_harmonics=1)
        result = estimator.estimate(
            Y=data['Y'],
            A=data['A'],
            t=data['t'],
            propensity=data['propensity']
        )
        
        estimated_psi = np.array([result.psi_0] + list(result.psi_harmonics))
        errors = estimated_psi - true_psi
        rmse = np.sqrt(np.mean(errors ** 2))
        
        # Check coverage
        ci = result.confidence_intervals
        coverage = np.mean([(ci[i, 0] <= true_psi[i] <= ci[i, 1]) for i in range(len(true_psi))])
        
        passed = rmse < CONFIG.g_estimation.max_parameter_rmse
        
        test_result = {
            'test_name': 'Harmonic G-Estimation',
            'test_id': 'L3-2',
            'true_psi': true_psi.tolist(),
            'estimated_psi': estimated_psi.tolist(),
            'errors': errors.tolist(),
            'rmse': float(rmse),
            'coverage': float(coverage),
            'passed': passed,
            'interpretation': f"RMSE={rmse:.4f} {'<' if passed else '>='} {CONFIG.g_estimation.max_parameter_rmse}"
        }
        
        print(f"  True ψ:      {true_psi}")
        print(f"  Estimated ψ: {estimated_psi}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Coverage: {coverage:.0%}")
        print(f"  {test_result['interpretation']} - {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return test_result
    
    def test_time_varying_effects(self) -> Dict:
        """L3-3: Test detection of circadian effect pattern."""
        print("\n--- Test L3-3: Time-Varying Effect Recovery ---")
        
        # Create circadian pattern: τ(t) = 0.5 + 0.3*cos(2πt/24)
        # cos(0)=1, so peak is at t=0 (midnight), trough at t=12 (noon)
        true_peak = 0.0   # cos peak at t=0
        true_trough = 12.0  # cos trough at t=12
        
        true_psi = np.array([0.5, 0.3, 0.0])  # ψ₀=0.5, ψ_c1=0.3, ψ_s1=0
        
        # Generate data
        data = generate_harmonic_outcome_data(
            n=2000,
            true_psi=true_psi,
            propensity=0.5,
            noise_std=0.5,
            seed=CONFIG.random_seed
        )
        
        estimator = HarmonicGEstimator(num_harmonics=1)
        estimator.estimate(
            Y=data['Y'],
            A=data['A'],
            t=data['t'],
            propensity=data['propensity']
        )
        
        est_peak, est_trough = estimator.get_peak_trough_times()
        
        # Allow wrap-around for peak/trough
        peak_error = min(abs(est_peak - true_peak), 24 - abs(est_peak - true_peak))
        trough_error = min(abs(est_trough - true_trough), 24 - abs(est_trough - true_trough))
        
        passed = (peak_error <= 3.0) and (trough_error <= 3.0)  # Within 3 hours
        
        result = {
            'test_name': 'Time-Varying Effect Recovery',
            'test_id': 'L3-3',
            'true_peak': true_peak,
            'estimated_peak': float(est_peak),
            'peak_error_hours': float(peak_error),
            'true_trough': true_trough,
            'estimated_trough': float(est_trough),
            'trough_error_hours': float(trough_error),
            'passed': passed,
            'interpretation': f"Peak error: {peak_error:.1f}h, Trough error: {trough_error:.1f}h"
        }
        
        print(f"  True peak: {true_peak:.1f}h, Estimated: {est_peak:.1f}h (error: {peak_error:.1f}h)")
        print(f"  True trough: {true_trough:.1f}h, Estimated: {est_trough:.1f}h (error: {trough_error:.1f}h)")
        print(f"  {result['interpretation']} - {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def test_double_robustness(self) -> Dict:
        """L3-4: Test double robustness property."""
        print("\n--- Test L3-4: Double Robustness Property ---")
        
        true_psi = np.array([0.5, 0.0, 0.0])  # Constant effect
        true_effect = 0.5
        
        np.random.seed(CONFIG.random_seed)
        n = 1000
        
        # Generate confounded data
        U = np.random.randn(n)
        t = np.random.uniform(0, 24, n)
        
        # True propensity (depends on U)
        true_prop = 0.5 + 0.2 * U
        true_prop = np.clip(true_prop, 0.1, 0.9)
        A = (np.random.rand(n) < true_prop).astype(float)
        
        # True outcome model (zero mean baseline for proper testing)
        baseline = U  # Confounded baseline (zero mean)
        Y = baseline + true_effect * A + np.random.randn(n) * 0.5
        
        results_by_scenario = {}
        
        # Scenario A: Both correct
        estimator = HarmonicGEstimator(num_harmonics=1)
        correct_outcome = baseline
        res_a = estimator.estimate(Y, A, t, true_prop, outcome_model=correct_outcome)
        bias_a = abs(res_a.psi_0 - true_effect)
        results_by_scenario['both_correct'] = {'bias': float(bias_a), 'estimate': float(res_a.psi_0)}
        
        # Scenario B: Outcome correct, propensity wrong
        wrong_prop = np.ones(n) * 0.5  # Wrong constant propensity
        res_b = estimator.estimate(Y, A, t, wrong_prop, outcome_model=correct_outcome)
        bias_b = abs(res_b.psi_0 - true_effect)
        results_by_scenario['outcome_correct'] = {'bias': float(bias_b), 'estimate': float(res_b.psi_0)}
        
        # Scenario C: Propensity correct, outcome wrong
        wrong_outcome = np.zeros(n)  # Wrong constant outcome
        res_c = estimator.estimate(Y, A, t, true_prop, outcome_model=wrong_outcome)
        bias_c = abs(res_c.psi_0 - true_effect)
        results_by_scenario['propensity_correct'] = {'bias': float(bias_c), 'estimate': float(res_c.psi_0)}
        
        # Scenario D: Both wrong
        res_d = estimator.estimate(Y, A, t, wrong_prop, outcome_model=wrong_outcome)
        bias_d = abs(res_d.psi_0 - true_effect)
        results_by_scenario['both_wrong'] = {'bias': float(bias_d), 'estimate': float(res_d.psi_0)}
        
        # Check: bias should be low when at least one is correct
        max_allowed = CONFIG.double_robustness.max_bias_when_one_correct
        passed = (bias_a < max_allowed and bias_b < max_allowed and bias_c < max_allowed)
        
        result = {
            'test_name': 'Double Robustness Property',
            'test_id': 'L3-4',
            'true_effect': true_effect,
            'scenarios': results_by_scenario,
            'passed': passed,
            'interpretation': f"Bias when ≥1 correct: {max(bias_a, bias_b, bias_c):.3f}"
        }
        
        print(f"  Both correct:        estimate={res_a.psi_0:.3f}, bias={bias_a:.3f}")
        print(f"  Outcome correct:     estimate={res_b.psi_0:.3f}, bias={bias_b:.3f}")
        print(f"  Propensity correct:  estimate={res_c.psi_0:.3f}, bias={bias_c:.3f}")
        print(f"  Both wrong:          estimate={res_d.psi_0:.3f}, bias={bias_d:.3f}")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def test_proximal_g_estimation(self) -> Dict:
        """L3-5: Test proximal G-estimation for unmeasured confounding."""
        print("\n--- Test L3-5: Proximal G-Estimation ---")
        
        true_effect = 0.5
        true_psi = np.array([true_effect, 0.0, 0.0])
        
        # Generate confounded data with proxies
        data = generate_harmonic_outcome_data(
            n=CONFIG.num_observations,
            true_psi=true_psi,
            propensity=0.5,
            noise_std=0.5,
            confounding_strength=CONFIG.proximal.confounding_strength,
            seed=CONFIG.random_seed
        )
        
        estimator = HarmonicGEstimator(num_harmonics=1)
        
        # Naive estimation (ignores confounding)
        res_naive = estimator.estimate(
            Y=data['Y'],
            A=data['A'],
            t=data['t'],
            propensity=np.ones(len(data['A'])) * 0.5  # Wrong propensity
        )
        naive_bias = abs(res_naive.psi_0 - true_effect)
        
        # Proximal estimation (uses W proxy)
        res_proximal = estimator.estimate(
            Y=data['Y'],
            A=data['A'],
            t=data['t'],
            propensity=np.ones(len(data['A'])) * 0.5,
            W=data['W']  # Use outcome proxy
        )
        proximal_bias = abs(res_proximal.psi_0 - true_effect)
        
        # Oracle estimation (observes U directly)
        res_oracle = estimator.estimate(
            Y=data['Y'],
            A=data['A'],
            t=data['t'],
            propensity=data['propensity'],  # True propensity
            W=data['U']  # Perfect proxy
        )
        oracle_bias = abs(res_oracle.psi_0 - true_effect)
        
        # Bias reduction
        if naive_bias > 0:
            bias_reduction = (naive_bias - proximal_bias) / naive_bias
        else:
            bias_reduction = 0.0
        
        passed = (proximal_bias < naive_bias) and (bias_reduction >= CONFIG.proximal.min_bias_reduction)
        
        result = {
            'test_name': 'Proximal G-Estimation',
            'test_id': 'L3-5',
            'true_effect': true_effect,
            'naive_estimate': float(res_naive.psi_0),
            'naive_bias': float(naive_bias),
            'proximal_estimate': float(res_proximal.psi_0),
            'proximal_bias': float(proximal_bias),
            'oracle_estimate': float(res_oracle.psi_0),
            'oracle_bias': float(oracle_bias),
            'bias_reduction': float(bias_reduction),
            'passed': passed,
            'interpretation': f"Bias reduction: {bias_reduction:.1%} (required: {CONFIG.proximal.min_bias_reduction:.0%})"
        }
        
        print(f"  True effect: {true_effect}")
        print(f"  Naive:    estimate={res_naive.psi_0:.3f}, bias={naive_bias:.3f}")
        print(f"  Proximal: estimate={res_proximal.psi_0:.3f}, bias={proximal_bias:.3f}")
        print(f"  Oracle:   estimate={res_oracle.psi_0:.3f}, bias={oracle_bias:.3f}")
        print(f"  Bias reduction: {bias_reduction:.1%}")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def run_all_tests(self) -> Dict:
        """Execute all Layer 3 tests."""
        print("=" * 60)
        print("TEST L3: Causal Inference Engine Tests")
        print("=" * 60)
        
        results = {
            'L3_1': self.test_mrt_positivity(),
            'L3_2': self.test_g_estimation(),
            'L3_3': self.test_time_varying_effects(),
            'L3_4': self.test_double_robustness(),
            'L3_5': self.test_proximal_g_estimation()
        }
        
        passed_count = sum(1 for r in results.values() if r.get('passed', False))
        total = len(results)
        
        self.results = {
            'test_suite': 'L3 Causal Inference',
            'individual_results': results,
            'summary': {
                'total': total,
                'passed': passed_count,
                'failed': total - passed_count,
                'pass_rate': passed_count / total
            },
            'overall_passed': passed_count == total
        }
        
        print("\n" + "=" * 60)
        print(f"L3 TESTS: {passed_count}/{total} passed ({100*passed_count/total:.0f}%)")
        print("=" * 60)
        
        return self.results
    
    def save_results(self, output_dir: str = None):
        if output_dir is None:
            output_dir = RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'test_L3_causal_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


def run_test():
    test = TestL3_CausalInference()
    results = test.run_all_tests()
    test.save_results()
    return results


if __name__ == "__main__":
    run_test()
