#!/usr/bin/env python3
"""
Test script for complete HPOBench integration including ML, OD, NAS, RL, and Surrogates benchmarks.
"""
import sys
sys.path.append('.')

import numpy as np
import bomegabench


def test_hpobench_complete():
    """Test complete HPOBench integration."""
    print("=== Testing Complete HPOBench Integration ===\n")
    
    # Get all available suites
    available_suites = bomegabench.list_suites()
    print(f"Available suites: {available_suites}")
    
    # Test HPOBench suites
    hpobench_suites = [
        'hpobench_ml', 'hpobench_od', 'hpobench_nas', 
        'hpobench_rl', 'hpobench_surrogates'
    ]
    
    for suite_name in hpobench_suites:
        print(f"\n--- Testing {suite_name} ---")
        
        if suite_name in available_suites:
            suite = bomegabench.get_suite(suite_name)
            functions = suite.functions
            print(f"Found {len(functions)} functions in {suite_name}:")
            
            for func_name in list(functions.keys())[:3]:  # Test first 3 functions
                print(f"  Testing {func_name}...")
                try:
                    func = functions[func_name]
                    
                    # Get function info
                    print(f"    Dimension: {func.get_dimension()}")
                    print(f"    Bounds: {func.get_bounds()}")
                    
                    # Test evaluation with random point
                    x = np.random.random(func.get_dimension())
                    result = func.evaluate(x)
                    print(f"    Test evaluation result: {result:.6f}")
                    print(f"    ✓ {func_name} works correctly")
                    
                except Exception as e:
                    print(f"    ✗ {func_name} failed: {e}")
                    
        else:
            print(f"  {suite_name} not available (missing dependencies)")
    
    print("\n=== HPOBench Integration Test Complete ===")


if __name__ == "__main__":
    test_hpobench_complete() 