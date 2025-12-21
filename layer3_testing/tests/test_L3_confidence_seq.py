"""
AEGIS 3.0 Layer 3 - Test L3-6: Confidence Sequences
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from config import CONFIG, RESULTS_DIR
from confidence_sequences import estimate_anytime_coverage_rate, run_coverage_simulation


class TestL3_ConfidenceSequences:
    """Test L3-6: Anytime-valid confidence sequences."""
    
    def __init__(self):
        self.results = {}
    
    def test_anytime_coverage(self) -> dict:
        """Test that CS maintains coverage at all time points."""
        print("=" * 60)
        print("TEST L3-6: Confidence Sequence Anytime Validity")
        print("=" * 60)
        
        np.random.seed(CONFIG.random_seed)
        
        true_effect = 0.5
        n_obs = 500
        n_sims = CONFIG.confidence_seq.num_simulations
        alpha = CONFIG.confidence_seq.alpha
        
        print(f"\n  True effect: {true_effect}")
        print(f"  Observations per sim: {n_obs}")
        print(f"  Number of simulations: {n_sims}")
        print(f"  Target coverage: {1-alpha:.0%}")
        print("\n  Running simulations...")
        
        coverage_rate = estimate_anytime_coverage_rate(
            true_effect=true_effect,
            n_obs=n_obs,
            n_simulations=n_sims,
            sigma=1.0,
            alpha=alpha,
            seed=CONFIG.random_seed
        )
        
        # Also get one example CS for width analysis
        _, history = run_coverage_simulation(
            true_effect=true_effect,
            n_obs=n_obs,
            sigma=1.0,
            alpha=alpha,
            seed=CONFIG.random_seed
        )
        
        # Width at different time points
        width_at_100 = history[99].upper - history[99].lower if len(history) > 99 else float('inf')
        width_at_end = history[-1].upper - history[-1].lower
        
        min_coverage = CONFIG.confidence_seq.min_anytime_coverage
        passed = coverage_rate >= min_coverage
        
        self.results = {
            'test_name': 'Confidence Sequence Anytime Validity',
            'test_id': 'L3-6',
            'true_effect': true_effect,
            'target_coverage': 1 - alpha,
            'observed_coverage': float(coverage_rate),
            'num_simulations': n_sims,
            'width_at_100': float(width_at_100),
            'width_at_end': float(width_at_end),
            'passed': passed,
            'interpretation': f"Coverage {coverage_rate:.1%} {'≥' if passed else '<'} {min_coverage:.0%}"
        }
        
        print(f"  Anytime coverage: {coverage_rate:.1%}")
        print(f"  CS width at t=100: {width_at_100:.3f}")
        print(f"  CS width at t={n_obs}: {width_at_end:.3f}")
        print(f"\n  {self.results['interpretation']} - {'PASS ✓' if passed else 'FAIL ✗'}")
        print("=" * 60)
        
        return self.results
    
    def save_results(self, output_dir: str = None):
        if output_dir is None:
            output_dir = RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'test_L3_cs_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")


def run_test():
    test = TestL3_ConfidenceSequences()
    results = test.test_anytime_coverage()
    test.save_results()
    return results


if __name__ == "__main__":
    run_test()
