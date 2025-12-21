# Layer 3 (Causal Inference Engine) Test Suite

## Overview

Test suite for Layer 3 (Causal Inference Engine) of AEGIS 3.0.

## Structure

```
layer3_testing/
├── src/
│   ├── config.py              # Test configuration
│   ├── g_estimation.py        # Harmonic G-Estimator
│   └── confidence_sequences.py # Martingale CS
├── tests/
│   ├── test_L3_causal.py      # Tests L3-1 to L3-5
│   ├── test_L3_confidence_seq.py # Test L3-6
│   └── run_all_tests.py       # Master runner
├── results/                   # Test results
└── README.md
```

## Running Tests

```bash
cd layer3_testing/tests
python run_all_tests.py
```

## Test Summary

| Test | Description | Status |
|------|-------------|--------|
| L3-1 | MRT Positivity | ✓ PASS |
| L3-2 | Harmonic G-Estimation | ✓ PASS |
| L3-3 | Time-Varying Effects | ✓ PASS |
| L3-4 | Double Robustness | ✓ PASS |
| L3-5 | Proximal G-Estimation | ✓ PASS |
| L3-6 | Confidence Sequences | ✓ PASS |

## Claims Validated

1. **MRT positivity maintained** (C3-1) ✓
2. **G-Estimation captures time-varying effects** (C3-2) ✓
3. **Double robustness property** (C3-3) ✓
4. **Proximal inference reduces bias** (C3-4) ✓
5. **Anytime-valid confidence sequences** (C3-5) ✓
