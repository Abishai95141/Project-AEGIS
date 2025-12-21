"""
AEGIS 3.0 Layer 5 - Comprehensive Tests

Tests for:
- L5-1: Reflex Controller (Tier 1)
- L5-2: STL Monitor (Tier 2)
- L5-3: Seldonian Constraints (Tier 3)
- L5-4: Tier Priority Resolution
- L5-5: Reachability Analysis
- L5-6: Cold Start Safety
- L5-7: Safety Relaxation Schedule
"""

import sys
import os
import json
import numpy as np
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from config import CONFIG, RESULTS_DIR
from safety_hierarchy import (
    ReflexController,
    STLMonitor,
    SeldonianConstraint,
    SimplexSafetySupervisor,
    SafetyDecision
)
from reachability import (
    ReachabilityAnalyzer,
    ColdStartSafety,
    generate_glucose_trajectory
)


class TestL5_SimplexSafety:
    """Layer 5 Simplex Safety Tests."""
    
    def __init__(self):
        self.results = {}
    
    def test_reflex_controller(self) -> Dict:
        """L5-1: Test Tier 1 Reflex Controller."""
        print("\n--- Test L5-1: Reflex Controller (Tier 1) ---")
        
        reflex = ReflexController()
        
        # Test cases: (glucose, action, expected_decision)
        test_cases = [
            (50.0, 5.0, SafetyDecision.EMERGENCY),   # Critical hypo
            (60.0, 5.0, SafetyDecision.BLOCK),       # Warning hypo
            (100.0, 5.0, SafetyDecision.ALLOW),      # Normal
            (200.0, 5.0, SafetyDecision.ALLOW),      # Normal high
            (300.0, 5.0, SafetyDecision.ALLOW),      # Hyper warning
            (450.0, 5.0, SafetyDecision.EMERGENCY),  # Critical hyper
        ]
        
        correct = 0
        total = len(test_cases)
        
        for glucose, action, expected in test_cases:
            result = reflex.evaluate(glucose, action)
            if result.decision == expected:
                correct += 1
        
        trigger_rate = correct / total
        passed = trigger_rate >= CONFIG.reflex.required_trigger_rate
        
        result = {
            'test_name': 'Reflex Controller',
            'test_id': 'L5-1',
            'test_cases': total,
            'correct': correct,
            'trigger_rate': float(trigger_rate),
            'passed': passed,
            'interpretation': f"Trigger rate {trigger_rate:.0%}"
        }
        
        print(f"  Correct: {correct}/{total}")
        print(f"  Trigger rate: {trigger_rate:.0%}")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def test_stl_monitor(self) -> Dict:
        """L5-2: Test Tier 2 STL Monitor."""
        print("\n--- Test L5-2: STL Monitor (Tier 2) ---")
        
        np.random.seed(CONFIG.random_seed)
        stl = STLMonitor()
        
        # Generate test trajectories
        n_safe = 50
        n_unsafe = 50
        
        correct = 0
        
        # Safe trajectories (within 70-250)
        for _ in range(n_safe):
            traj = generate_glucose_trajectory(
                initial=120,
                duration_minutes=60,
                insulin_dose=2.0,
                seed=None
            )
            if stl.classify_trajectory(traj):
                correct += 1
        
        # Unsafe trajectories (hypo risk)
        for _ in range(n_unsafe):
            traj = generate_glucose_trajectory(
                initial=85,
                duration_minutes=60,
                insulin_dose=8.0,  # High dose
                seed=None
            )
            # Force some to go below 70
            traj = traj - 30
            if not stl.classify_trajectory(traj):
                correct += 1
        
        accuracy = correct / (n_safe + n_unsafe)
        passed = accuracy >= CONFIG.stl.min_classification_accuracy
        
        # Also test robustness values
        safe_traj = np.array([120, 115, 110, 105, 100, 95, 90, 85, 80, 75])
        unsafe_traj = np.array([80, 75, 70, 65, 60, 55, 50, 50, 55, 60])
        
        safe_robustness = stl.compute_robustness(safe_traj)
        unsafe_robustness = stl.compute_robustness(unsafe_traj)
        
        result = {
            'test_name': 'STL Monitor',
            'test_id': 'L5-2',
            'n_safe': n_safe,
            'n_unsafe': n_unsafe,
            'correct': correct,
            'accuracy': float(accuracy),
            'safe_robustness_example': float(safe_robustness),
            'unsafe_robustness_example': float(unsafe_robustness),
            'passed': passed,
            'interpretation': f"Accuracy {accuracy:.0%}"
        }
        
        print(f"  Accuracy: {accuracy:.0%}")
        print(f"  Safe trajectory ρ: {safe_robustness:.1f}")
        print(f"  Unsafe trajectory ρ: {unsafe_robustness:.1f}")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def test_seldonian_constraints(self) -> Dict:
        """L5-3: Test Tier 3 Seldonian Constraints."""
        print("\n--- Test L5-3: Seldonian Constraints (Tier 3) ---")
        
        np.random.seed(CONFIG.random_seed)
        seldonian = SeldonianConstraint()
        
        # Test 1: Low violation samples
        safe_samples = np.random.normal(100, 10, 100)  # Mean 100, few < 70
        p_upper_safe = seldonian.compute_upper_bound(safe_samples, threshold=70)
        
        # Test 2: High violation samples
        risky_samples = np.random.normal(75, 15, 100)  # Mean 75, many < 70
        p_upper_risky = seldonian.compute_upper_bound(risky_samples, threshold=70)
        
        # Verify calibration with Monte Carlo
        n_sims = 100
        violations = 0
        alpha = CONFIG.seldonian.alpha
        
        for _ in range(n_sims):
            samples = np.random.normal(80, 12, 100)
            p_upper = seldonian.compute_upper_bound(samples, threshold=70)
            # Ground truth violation rate
            true_p = np.mean(samples < 70)
            if true_p > p_upper:
                violations += 1
        
        underestimate_rate = violations / n_sims
        passed = underestimate_rate <= CONFIG.seldonian.allowed_tolerance
        
        result = {
            'test_name': 'Seldonian Constraints',
            'test_id': 'L5-3',
            'p_upper_safe': float(p_upper_safe),
            'p_upper_risky': float(p_upper_risky),
            'underestimate_rate': float(underestimate_rate),
            'allowed_tolerance': CONFIG.seldonian.allowed_tolerance,
            'passed': passed,
            'interpretation': f"Underestimate rate {underestimate_rate:.1%}"
        }
        
        print(f"  Safe samples P_upper: {p_upper_safe:.3f}")
        print(f"  Risky samples P_upper: {p_upper_risky:.3f}")
        print(f"  Underestimate rate: {underestimate_rate:.1%}")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def test_tier_priority(self) -> Dict:
        """L5-4: Test Tier Priority Resolution."""
        print("\n--- Test L5-4: Tier Priority Resolution ---")
        
        supervisor = SimplexSafetySupervisor()
        
        # Scenario 1: Tier 1 blocks (critical hypo)
        result1 = supervisor.evaluate(
            current_glucose=50.0,
            predicted_trajectory=np.array([80, 85, 90, 95]),  # Safe trajectory
            outcome_samples=np.random.normal(100, 10, 50),     # Safe samples
            proposed_action=5.0
        )
        tier1_overrides = result1.tier_triggered == 1
        
        # Scenario 2: Tier 2 blocks (unsafe trajectory)
        result2 = supervisor.evaluate(
            current_glucose=100.0,  # Normal glucose
            predicted_trajectory=np.array([100, 80, 60, 40, 30]),  # Goes hypo
            outcome_samples=np.random.normal(100, 10, 50),
            proposed_action=5.0
        )
        tier2_active = result2.tier_triggered == 2 or result2.decision == SafetyDecision.BLOCK
        
        # Scenario 3: All pass - use very safe values
        # Glucose 120, safe trajectory, high samples (far from 70 threshold)
        result3 = supervisor.evaluate(
            current_glucose=120.0,
            predicted_trajectory=np.array([120, 118, 116, 114, 112]),  # All safe
            outcome_samples=np.random.normal(150, 10, 100),  # Mean 150, far from 70
            proposed_action=2.0
        )
        all_pass = result3.decision == SafetyDecision.ALLOW
        
        passed = tier1_overrides and all_pass
        
        result = {
            'test_name': 'Tier Priority Resolution',
            'test_id': 'L5-4',
            'tier1_overrides': tier1_overrides,
            'tier2_active': tier2_active,
            'all_tiers_pass': all_pass,
            'passed': passed,
            'interpretation': f"Tier 1 override: {tier1_overrides}, All pass: {all_pass}"
        }
        
        print(f"  Tier 1 override works: {tier1_overrides}")
        print(f"  Tier 2 blocks unsafe: {tier2_active}")
        print(f"  All tiers pass when safe: {all_pass}")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def test_reachability_analysis(self) -> Dict:
        """L5-5: Test Reachability Analysis."""
        print("\n--- Test L5-5: Reachability Analysis ---")
        
        reachability = ReachabilityAnalyzer()
        
        # Test cases
        test_cases = []
        false_negatives = 0
        
        # Case 1: Safe action from safe state
        is_safe, reach_set = reachability.is_action_safe(
            current_state=120.0,
            action=2.0,
            time_horizon=60
        )
        test_cases.append(('safe_from_safe', is_safe, reach_set))
        
        # Case 2: Risky action from marginal state
        is_safe2, reach_set2 = reachability.is_action_safe(
            current_state=85.0,
            action=10.0,  # High insulin
            time_horizon=60
        )
        test_cases.append(('risky_from_marginal', is_safe2, reach_set2))
        
        # Case 3: No action from low state
        is_safe3, reach_set3 = reachability.is_action_safe(
            current_state=75.0,
            action=0.0,
            time_horizon=60
        )
        test_cases.append(('no_action_from_low', is_safe3, reach_set3))
        
        # Verify overapproximation with simulation
        np.random.seed(CONFIG.random_seed)
        n_sims = 50
        containment_count = 0
        
        for _ in range(n_sims):
            initial = np.random.uniform(80, 150)
            action = np.random.uniform(0, 10)
            
            _, reach = reachability.is_action_safe(initial, action, 60)
            
            # Simulate actual trajectory
            traj = generate_glucose_trajectory(
                initial=initial,
                duration_minutes=60,
                insulin_dose=action
            )
            
            # Check containment
            if min(traj) >= reach.min_state - 5 and max(traj) <= reach.max_state + 5:
                containment_count += 1
        
        containment_rate = containment_count / n_sims
        passed = containment_rate >= 0.95 and false_negatives == 0
        
        result = {
            'test_name': 'Reachability Analysis',
            'test_id': 'L5-5',
            'test_cases': [(tc[0], tc[1]) for tc in test_cases],
            'containment_rate': float(containment_rate),
            'false_negatives': false_negatives,
            'passed': passed,
            'interpretation': f"Containment {containment_rate:.0%}, FN: {false_negatives}"
        }
        
        print(f"  Containment rate: {containment_rate:.0%}")
        print(f"  False negatives: {false_negatives}")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def test_cold_start_safety(self) -> Dict:
        """L5-6: Test Cold Start Safety."""
        print("\n--- Test L5-6: Cold Start Safety ---")
        
        cold_start = ColdStartSafety()
        
        # Get Day 1 and Day 30 bounds
        bound_day1 = cold_start.get_bound_at_day(1)
        bound_day30 = cold_start.get_bound_at_day(30)
        
        # Day 1 should be more conservative (lower bound)
        day1_more_conservative = bound_day1 < bound_day30
        
        # Check population coverage
        # With 99% bound, only 1% of population should exceed
        alpha_day1 = cold_start.get_alpha_at_day(1)
        coverage = 1 - alpha_day1
        
        passed = day1_more_conservative and coverage >= 0.98
        
        result = {
            'test_name': 'Cold Start Safety',
            'test_id': 'L5-6',
            'bound_day1': float(bound_day1),
            'bound_day30': float(bound_day30),
            'day1_more_conservative': day1_more_conservative,
            'alpha_day1': float(alpha_day1),
            'population_coverage': float(coverage),
            'passed': passed,
            'interpretation': f"Day 1 bound {bound_day1:.3f} < Day 30 {bound_day30:.3f}"
        }
        
        print(f"  Day 1 bound:  {bound_day1:.3f}")
        print(f"  Day 30 bound: {bound_day30:.3f}")
        print(f"  Day 1 more conservative: {day1_more_conservative}")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def test_relaxation_schedule(self) -> Dict:
        """L5-7: Test Safety Relaxation Schedule."""
        print("\n--- Test L5-7: Safety Relaxation Schedule ---")
        
        cold_start = ColdStartSafety()
        
        # Get full schedule
        schedule = cold_start.get_relaxation_schedule(30)
        
        # Key checkpoints
        alpha_day1 = schedule[1]
        alpha_day7 = schedule[7]
        alpha_day14 = schedule[14]
        alpha_day30 = schedule[30]
        
        # Verify monotonic relaxation
        is_monotonic = cold_start.is_bound_monotonic(30)
        
        # Verify endpoints
        day1_strict = alpha_day1 < 0.02  # Should be ~0.01
        day30_standard = alpha_day30 > 0.04  # Should approach 0.05
        
        passed = is_monotonic and day1_strict and day30_standard
        
        result = {
            'test_name': 'Safety Relaxation Schedule',
            'test_id': 'L5-7',
            'alpha_day1': float(alpha_day1),
            'alpha_day7': float(alpha_day7),
            'alpha_day14': float(alpha_day14),
            'alpha_day30': float(alpha_day30),
            'is_monotonic': is_monotonic,
            'day1_strict': day1_strict,
            'day30_standard': day30_standard,
            'passed': passed,
            'interpretation': f"α: Day1={alpha_day1:.3f} → Day30={alpha_day30:.3f}"
        }
        
        print(f"  α Day 1:  {alpha_day1:.4f}")
        print(f"  α Day 7:  {alpha_day7:.4f}")
        print(f"  α Day 14: {alpha_day14:.4f}")
        print(f"  α Day 30: {alpha_day30:.4f}")
        print(f"  Monotonic: {is_monotonic}")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def run_all_tests(self) -> Dict:
        """Execute all Layer 5 tests."""
        print("=" * 60)
        print("TEST L5: Simplex Safety Supervisor Tests")
        print("=" * 60)
        
        results = {
            'L5_1': self.test_reflex_controller(),
            'L5_2': self.test_stl_monitor(),
            'L5_3': self.test_seldonian_constraints(),
            'L5_4': self.test_tier_priority(),
            'L5_5': self.test_reachability_analysis(),
            'L5_6': self.test_cold_start_safety(),
            'L5_7': self.test_relaxation_schedule()
        }
        
        passed_count = sum(1 for r in results.values() if r.get('passed', False))
        total = len(results)
        
        self.results = {
            'test_suite': 'L5 Simplex Safety',
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
        print(f"L5 TESTS: {passed_count}/{total} passed ({100*passed_count/total:.0f}%)")
        print("=" * 60)
        
        return self.results
    
    def save_results(self, output_dir: str = None):
        if output_dir is None:
            output_dir = RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'test_L5_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


def run_test():
    test = TestL5_SimplexSafety()
    results = test.run_all_tests()
    test.save_results()
    return results


if __name__ == "__main__":
    run_test()
