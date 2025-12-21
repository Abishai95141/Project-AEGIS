# Layer 5 (Simplex Safety) Test Suite

## Overview

Test suite for Layer 5 (Simplex Safety Supervisor) of AEGIS 3.0.

## Structure

```
layer5_testing/
├── src/
│   ├── config.py              # Configuration
│   ├── safety_hierarchy.py    # Three-tier safety
│   └── reachability.py        # Reachability & cold start
├── tests/
│   └── test_L5_safety.py      # All L5 tests
├── results/                   # Test results
└── README.md
```

## Running Tests

```bash
cd layer5_testing/tests
python run_all_tests.py
```

## Test Summary

| Test | Description | Status |
|------|-------------|--------|
| L5-1 | Reflex Controller | ✓ PASS |
| L5-2 | STL Monitor | ✓ PASS |
| L5-3 | Seldonian Constraints | ✓ PASS |
| L5-4 | Tier Priority | ✓ PASS |
| L5-5 | Reachability Analysis | ✓ PASS |
| L5-6 | Cold Start Safety | ✓ PASS |
| L5-7 | Relaxation Schedule | ✓ PASS |

## Claims Validated

1. **Three-tier safety hierarchy** (C5-1,2,3) ✓
2. **Priority resolution** (C5-4) ✓
3. **Reachability breaks circularity** (C5-5) ✓
4. **Cold start conservative** (C5-6) ✓
5. **Smooth relaxation schedule** (C5-7) ✓
