"""
AEGIS 3.0 Layer 4 - Comprehensive Tests

Tests for:
- L4-1: Action-Centered Variance Reduction
- L4-2: Regret Bound Verification
- L4-3: CTS Algorithm
- L4-4: Posterior Collapse Prevention
- L4-5: Safety-Constrained Optimization
- L4-6: CTS Regret Bound
"""

import sys
import os
import json
import numpy as np
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from config import CONFIG, RESULTS_DIR
from action_centered_bandit import (
    ActionCenteredBandit, 
    StandardQBandit,
    generate_bandit_environment
)
from cts import (
    CounterfactualThompsonSampling,
    StandardThompsonSampling,
    DigitalTwin,
    SafetyEvaluator
)


class TestL4_DecisionEngine:
    """Layer 4 Decision Engine Tests."""
    
    def __init__(self):
        self.results = {}
    
    def test_variance_reduction(self) -> Dict:
        """L4-1: Test action-centered variance reduction."""
        print("\n--- Test L4-1: Action-Centered Variance Reduction ---")
        
        np.random.seed(CONFIG.random_seed)
        
        # Generate environment
        env = generate_bandit_environment(
            context_dim=CONFIG.bandit.context_dim,
            num_actions=CONFIG.bandit.num_actions,
            baseline_variance=CONFIG.bandit.baseline_variance,
            noise_variance=CONFIG.bandit.noise_variance,
            seed=CONFIG.random_seed
        )
        
        n_trials = 20
        n_steps = 200
        
        ac_estimates = []
        std_estimates = []
        
        for trial in range(n_trials):
            np.random.seed(CONFIG.random_seed + trial)
            
            # Action-Centered Bandit
            ac_bandit = ActionCenteredBandit(
                context_dim=env['context_dim'],
                num_actions=env['num_actions']
            )
            
            # Standard Q-Bandit
            std_bandit = StandardQBandit(
                context_dim=env['context_dim'],
                num_actions=env['num_actions']
            )
            
            for _ in range(n_steps):
                context = env['generate_context']()
                
                # Action-Centered
                ac_action, _ = ac_bandit.select_action(context)
                ac_reward = env['get_reward'](context, ac_action)
                ac_bandit.update(context, ac_action, ac_reward)
                
                # Standard
                std_action, _ = std_bandit.select_action(context)
                std_reward = env['get_reward'](context, std_action)
                std_bandit.update(context, std_action, std_reward)
            
            # Estimate treatment effect
            test_context = np.ones(env['context_dim']) / np.sqrt(env['context_dim'])
            ac_tau = ac_bandit.get_treatment_effect(test_context)
            true_tau = env['treatment_effect'](test_context)
            
            ac_estimates.append(ac_tau)
            
            # For standard: extract τ from Q difference
            std_q1 = test_context @ std_bandit.theta_mean[:env['context_dim']]
            std_q2 = test_context @ std_bandit.theta_mean[env['context_dim']:2*env['context_dim']]
            std_estimates.append(std_q2 - std_q1)
        
        ac_variance = np.var(ac_estimates)
        std_variance = np.var(std_estimates)
        variance_ratio = std_variance / ac_variance if ac_variance > 0 else float('inf')
        
        passed = variance_ratio >= CONFIG.bandit.min_variance_reduction
        
        result = {
            'test_name': 'Action-Centered Variance Reduction',
            'test_id': 'L4-1',
            'action_centered_variance': float(ac_variance),
            'standard_variance': float(std_variance),
            'variance_reduction_ratio': float(variance_ratio),
            'min_required': CONFIG.bandit.min_variance_reduction,
            'passed': passed,
            'interpretation': f"Variance ratio {variance_ratio:.2f}x {'≥' if passed else '<'} {CONFIG.bandit.min_variance_reduction}x"
        }
        
        print(f"  Action-Centered Var: {ac_variance:.4f}")
        print(f"  Standard Q Var:      {std_variance:.4f}")
        print(f"  Variance Reduction:  {variance_ratio:.2f}x")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def test_regret_bound(self) -> Dict:
        """L4-2: Test regret bound O(√T)."""
        print("\n--- Test L4-2: Regret Bound Verification ---")
        
        np.random.seed(CONFIG.random_seed)
        
        env = generate_bandit_environment(
            context_dim=CONFIG.bandit.context_dim,
            num_actions=CONFIG.bandit.num_actions,
            baseline_variance=1.0,  # Lower variance for cleaner regret
            noise_variance=0.5,
            seed=CONFIG.random_seed
        )
        
        bandit = ActionCenteredBandit(
            context_dim=env['context_dim'],
            num_actions=env['num_actions']
        )
        
        T = CONFIG.regret.max_timesteps
        regret_checkpoints = []
        
        for t in range(T):
            context = env['generate_context']()
            action, _ = bandit.select_action(context)
            reward = env['get_reward'](context, action)
            bandit.update(context, action, reward)
            
            # Compute regret
            true_tau = env['treatment_effect'](context)
            bandit.compute_regret(context, action, true_tau)
            
            if (t + 1) in [100, 250, 500, 750, 1000]:
                regret_checkpoints.append((t + 1, bandit.cumulative_regret))
        
        # Log-log regression to estimate slope
        log_t = np.log([c[0] for c in regret_checkpoints])
        log_r = np.log([max(c[1], 1e-10) for c in regret_checkpoints])
        
        # Linear regression
        slope, intercept = np.polyfit(log_t, log_r, 1)
        
        passed = CONFIG.regret.expected_slope_min <= slope <= CONFIG.regret.expected_slope_max
        
        result = {
            'test_name': 'Regret Bound Verification',
            'test_id': 'L4-2',
            'final_regret': float(bandit.cumulative_regret),
            'regret_checkpoints': regret_checkpoints,
            'log_log_slope': float(slope),
            'expected_slope_range': [CONFIG.regret.expected_slope_min, CONFIG.regret.expected_slope_max],
            'passed': passed,
            'interpretation': f"Slope {slope:.3f} ∈ [{CONFIG.regret.expected_slope_min}, {CONFIG.regret.expected_slope_max}]"
        }
        
        print(f"  Regret at T=100:  {regret_checkpoints[0][1]:.2f}")
        print(f"  Regret at T=1000: {bandit.cumulative_regret:.2f}")
        print(f"  Log-log slope:    {slope:.3f} (√T → 0.5)")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def test_cts_algorithm(self) -> Dict:
        """L4-3: Test CTS algorithm execution."""
        print("\n--- Test L4-3: CTS Algorithm Execution ---")
        
        np.random.seed(CONFIG.random_seed)
        
        env = generate_bandit_environment(
            context_dim=CONFIG.bandit.context_dim,
            num_actions=CONFIG.bandit.num_actions,
            seed=CONFIG.random_seed
        )
        
        bandit = ActionCenteredBandit(
            context_dim=env['context_dim'],
            num_actions=env['num_actions']
        )
        
        digital_twin = DigitalTwin(
            true_tau_params=env['true_tau_params'],
            prediction_noise=0.5,
            confidence=CONFIG.cts.lambda_confidence
        )
        
        safety = SafetyEvaluator(
            safety_threshold=CONFIG.safety.safety_threshold,
            context_dependent=True
        )
        
        cts = CounterfactualThompsonSampling(
            bandit=bandit,
            digital_twin=digital_twin,
            safety_evaluator=safety
        )
        
        n_steps = 500
        blocked_count = 0
        cf_updates = 0
        
        for _ in range(n_steps):
            context = env['generate_context']()
            result = cts.step(context)
            
            if result.was_blocked:
                blocked_count += 1
            if result.counterfactual_update:
                cf_updates += 1
            
            # Get actual reward and update
            actual_reward = env['get_reward'](context, result.executed_action)
            cts.update_with_observation(context, result.executed_action, actual_reward)
        
        blocking_rate = blocked_count / n_steps
        cf_update_rate = cf_updates / max(blocked_count, 1)
        
        # CTS should do counterfactual updates for blocked actions
        passed = cf_updates > 0 and cf_update_rate > 0.9
        
        result = {
            'test_name': 'CTS Algorithm Execution',
            'test_id': 'L4-3',
            'total_steps': n_steps,
            'blocked_count': blocked_count,
            'blocking_rate': float(blocking_rate),
            'counterfactual_updates': cf_updates,
            'cf_update_rate': float(cf_update_rate),
            'passed': passed,
            'interpretation': f"CF updates: {cf_updates}/{blocked_count} blocked"
        }
        
        print(f"  Blocking rate: {blocking_rate:.1%}")
        print(f"  CF updates:    {cf_updates}/{blocked_count}")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def test_posterior_collapse_prevention(self) -> Dict:
        """L4-4: Test that CTS prevents posterior collapse."""
        print("\n--- Test L4-4: Posterior Collapse Prevention ---")
        
        np.random.seed(CONFIG.random_seed)
        
        env = generate_bandit_environment(
            context_dim=CONFIG.bandit.context_dim,
            num_actions=CONFIG.bandit.num_actions,
            seed=CONFIG.random_seed
        )
        
        # Create two bandits with same initialization
        bandit_cts = ActionCenteredBandit(
            context_dim=env['context_dim'],
            num_actions=env['num_actions'],
            prior_variance=1.0
        )
        
        bandit_std = ActionCenteredBandit(
            context_dim=env['context_dim'],
            num_actions=env['num_actions'],
            prior_variance=1.0
        )
        
        digital_twin = DigitalTwin(
            true_tau_params=env['true_tau_params'],
            confidence=CONFIG.cts.lambda_confidence
        )
        
        # Aggressive safety to force blocking
        safety = SafetyEvaluator(
            safety_threshold=0.4,  # Block most actions
            context_dependent=False
        )
        
        cts = CounterfactualThompsonSampling(
            bandit=bandit_cts,
            digital_twin=digital_twin,
            safety_evaluator=safety
        )
        
        std_ts = StandardThompsonSampling(
            bandit=bandit_std,
            safety_evaluator=safety
        )
        
        n_steps = 300
        
        initial_var_cts = np.trace(bandit_cts.theta_cov)
        initial_var_std = np.trace(bandit_std.theta_cov)
        
        for _ in range(n_steps):
            context = env['generate_context']()
            
            # CTS step
            cts_result = cts.step(context)
            if not cts_result.was_blocked:
                reward = env['get_reward'](context, cts_result.executed_action)
                cts.update_with_observation(context, cts_result.executed_action, reward)
            
            # Standard TS step
            std_result = std_ts.step(context)
            if not std_result.was_blocked:
                reward = env['get_reward'](context, std_result.executed_action)
                std_ts.update_with_observation(context, std_result.executed_action, reward)
        
        final_var_cts = np.trace(bandit_cts.theta_cov)
        final_var_std = np.trace(bandit_std.theta_cov)
        
        cts_reduction = final_var_cts / initial_var_cts
        std_reduction = final_var_std / initial_var_std
        
        # CTS should reduce variance; Standard should not (or less)
        passed = cts_reduction < std_reduction and cts_reduction < CONFIG.cts.max_posterior_variance_ratio
        
        result = {
            'test_name': 'Posterior Collapse Prevention',
            'test_id': 'L4-4',
            'cts_initial_variance': float(initial_var_cts),
            'cts_final_variance': float(final_var_cts),
            'cts_variance_ratio': float(cts_reduction),
            'std_initial_variance': float(initial_var_std),
            'std_final_variance': float(final_var_std),
            'std_variance_ratio': float(std_reduction),
            'passed': passed,
            'interpretation': f"CTS ratio {cts_reduction:.2f} < Standard {std_reduction:.2f}"
        }
        
        print(f"  CTS variance:  {initial_var_cts:.2f} → {final_var_cts:.2f} (ratio: {cts_reduction:.2f})")
        print(f"  Std variance:  {initial_var_std:.2f} → {final_var_std:.2f} (ratio: {std_reduction:.2f})")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def test_safety_constraints(self) -> Dict:
        """L4-5: Test that unsafe actions are never executed."""
        print("\n--- Test L4-5: Safety-Constrained Optimization ---")
        
        np.random.seed(CONFIG.random_seed)
        
        env = generate_bandit_environment(
            context_dim=CONFIG.bandit.context_dim,
            num_actions=CONFIG.bandit.num_actions,
            seed=CONFIG.random_seed
        )
        
        bandit = ActionCenteredBandit(
            context_dim=env['context_dim'],
            num_actions=env['num_actions']
        )
        
        digital_twin = DigitalTwin(
            true_tau_params=env['true_tau_params'],
            confidence=CONFIG.cts.lambda_confidence
        )
        
        safety = SafetyEvaluator(
            safety_threshold=CONFIG.safety.safety_threshold
        )
        
        cts = CounterfactualThompsonSampling(
            bandit=bandit,
            digital_twin=digital_twin,
            safety_evaluator=safety
        )
        
        n_steps = 500
        violations = 0
        
        for _ in range(n_steps):
            context = env['generate_context']()
            result = cts.step(context)
            
            # Check if executed action was safe
            if not safety.is_safe(result.executed_action, context, env['num_actions']):
                violations += 1
            
            reward = env['get_reward'](context, result.executed_action)
            cts.update_with_observation(context, result.executed_action, reward)
        
        passed = violations <= CONFIG.safety.max_violations
        
        result = {
            'test_name': 'Safety-Constrained Optimization',
            'test_id': 'L4-5',
            'total_steps': n_steps,
            'safety_violations': violations,
            'max_allowed': CONFIG.safety.max_violations,
            'passed': passed,
            'interpretation': f"Violations: {violations} ≤ {CONFIG.safety.max_violations}"
        }
        
        print(f"  Total steps:  {n_steps}")
        print(f"  Violations:   {violations}")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def test_cts_regret_bound(self) -> Dict:
        """L4-6: Test CTS achieves bounded regret."""
        print("\n--- Test L4-6: CTS Regret Bound ---")
        
        np.random.seed(CONFIG.random_seed)
        
        env = generate_bandit_environment(
            context_dim=CONFIG.bandit.context_dim,
            num_actions=CONFIG.bandit.num_actions,
            baseline_variance=1.0,
            seed=CONFIG.random_seed
        )
        
        bandit = ActionCenteredBandit(
            context_dim=env['context_dim'],
            num_actions=env['num_actions']
        )
        
        digital_twin = DigitalTwin(
            true_tau_params=env['true_tau_params'],
            confidence=CONFIG.cts.lambda_confidence
        )
        
        safety = SafetyEvaluator(safety_threshold=0.6)
        
        cts = CounterfactualThompsonSampling(
            bandit=bandit,
            digital_twin=digital_twin,
            safety_evaluator=safety
        )
        
        n_steps = 500
        cumulative_regret = 0.0
        regret_history = []
        
        for t in range(n_steps):
            context = env['generate_context']()
            result = cts.step(context)
            
            reward = env['get_reward'](context, result.executed_action)
            cts.update_with_observation(context, result.executed_action, reward)
            
            # Regret vs optimal (unconstrained)
            true_tau = env['treatment_effect'](context)
            optimal = (env['num_actions'] - 1) if true_tau > 0 else 0
            regret = abs(true_tau) * abs(optimal - result.executed_action)
            cumulative_regret += regret
            
            if (t + 1) in [100, 250, 500]:
                regret_history.append((t + 1, cumulative_regret))
        
        blocking_rate = cts.get_blocking_rate()
        
        # Check regret is sublinear: ratio of regret at T vs at T/5 should be < 5
        if len(regret_history) >= 2:
            regret_ratio = regret_history[-1][1] / max(regret_history[0][1], 1)
            time_ratio = regret_history[-1][0] / regret_history[0][0]
            sublinear = regret_ratio < time_ratio * 0.8  # Sublinear means grows slower than T
        else:
            sublinear = True
            regret_ratio = 1.0
            time_ratio = 1.0
        
        # Regret should be bounded (not exploding)
        passed = sublinear or cumulative_regret < n_steps * 0.5
        
        result = {
            'test_name': 'CTS Regret Bound',
            'test_id': 'L4-6',
            'final_regret': float(cumulative_regret),
            'blocking_rate': float(blocking_rate),
            'regret_history': regret_history,
            'regret_ratio': float(regret_ratio),
            'time_ratio': float(time_ratio),
            'sublinear': sublinear,
            'passed': passed,
            'interpretation': f"Regret {cumulative_regret:.1f} over {n_steps} steps"
        }
        
        print(f"  Final regret: {cumulative_regret:.1f}")
        print(f"  Blocking rate: {blocking_rate:.1%}")
        print(f"  Sublinear: {sublinear}")
        print(f"  {'PASS ✓' if passed else 'FAIL ✗'}")
        
        return result
    
    def run_all_tests(self) -> Dict:
        """Execute all Layer 4 tests."""
        print("=" * 60)
        print("TEST L4: Decision Engine Tests")
        print("=" * 60)
        
        results = {
            'L4_1': self.test_variance_reduction(),
            'L4_2': self.test_regret_bound(),
            'L4_3': self.test_cts_algorithm(),
            'L4_4': self.test_posterior_collapse_prevention(),
            'L4_5': self.test_safety_constraints(),
            'L4_6': self.test_cts_regret_bound()
        }
        
        passed_count = sum(1 for r in results.values() if r.get('passed', False))
        total = len(results)
        
        self.results = {
            'test_suite': 'L4 Decision Engine',
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
        print(f"L4 TESTS: {passed_count}/{total} passed ({100*passed_count/total:.0f}%)")
        print("=" * 60)
        
        return self.results
    
    def save_results(self, output_dir: str = None):
        if output_dir is None:
            output_dir = RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'test_L4_results.json')
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


def run_test():
    test = TestL4_DecisionEngine()
    results = test.run_all_tests()
    test.save_results()
    return results


if __name__ == "__main__":
    run_test()
