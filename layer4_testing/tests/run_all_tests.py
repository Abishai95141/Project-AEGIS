"""
AEGIS 3.0 Layer 4 - Master Test Runner
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from config import RESULTS_DIR


def run_all_tests():
    print("=" * 70)
    print("    AEGIS 3.0 LAYER 4 TEST SUITE")
    print("    Decision Engine Validation")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().isoformat()}")
    
    all_results = {
        'summary': {
            'timestamp': datetime.now().isoformat(),
            'total_tests': 0,
            'tests_passed': 0,
            'tests_failed': 0
        },
        'individual_results': {}
    }
    
    # Run all L4 tests
    print("\n" + "-" * 70)
    print("Running Tests L4-1 to L4-6: Decision Engine Tests")
    print("-" * 70)
    try:
        from test_L4_decision import run_test
        results = run_test()
        all_results['individual_results'] = results
        
        all_results['summary']['total_tests'] = results['summary']['total']
        all_results['summary']['tests_passed'] = results['summary']['passed']
        all_results['summary']['tests_failed'] = results['summary']['failed']
    except Exception as e:
        print(f"ERROR in tests: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    total = all_results['summary']['total_tests']
    passed = all_results['summary']['tests_passed']
    all_results['summary']['pass_rate'] = passed / total if total > 0 else 0
    all_results['summary']['all_passed'] = passed == total
    
    print("\n" + "=" * 70)
    print("    LAYER 4 TEST SUITE SUMMARY")
    print("=" * 70)
    print(f"\n  Total Tests:  {total}")
    print(f"  Passed:       {passed}")
    print(f"  Failed:       {all_results['summary']['tests_failed']}")
    print(f"  Pass Rate:    {all_results['summary']['pass_rate']*100:.1f}%")
    print()
    if all_results['summary']['all_passed']:
        print("  ★ ALL TESTS PASSED ★")
    else:
        print("  ✗ SOME TESTS FAILED")
    print("=" * 70)
    
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, 'layer4_all_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to: {output_path}")
    
    return all_results


if __name__ == "__main__":
    run_all_tests()
