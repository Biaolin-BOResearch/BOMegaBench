#!/usr/bin/env python3
"""
Test all 24 BBOB functions.
"""

import torch
import numpy as np
import bomegabench as bmb

def test_all_bbob_functions():
    """Test all 24 BBOB functions."""
    print("=== Testing All 24 BBOB Functions ===")
    
    # Get BBOB Raw functions
    bbob_suite = bmb.get_functions_by_property('suite', 'BBOB Raw')
    print(f"Found {len(bbob_suite)} BBOB Raw functions")
    
    # Expected function names based on BBOB specification
    expected_functions = [
        'sphere', 'ellipsoid_separable', 'rastrigin_separable', 'skew_rastrigin_bueche',
        'linear_slope', 'attractive_sector', 'step_ellipsoid', 'rosenbrock', 'rosenbrock_rotated',
        'ellipsoid', 'discus', 'bent_cigar', 'sharp_ridge', 'different_powers', 'rastrigin',
        'weierstrass', 'schaffer_f7_cond10', 'schaffer_f7_cond1000', 'griewank_rosenbrock',
        'schwefel', 'gallagher_101', 'gallagher_21', 'katsuura', 'lunacek_bi_rastrigin'
    ]
    
    print(f"\nExpected 24 functions, got {len(bbob_suite)}")
    
    # Test each function
    success_count = 0
    for i, func_name in enumerate(sorted(bbob_suite.keys()), 1):
        try:
            func = bbob_suite[func_name]
            
            # Test basic properties
            print(f"f{func.metadata['function_idx']:2d}: {func.metadata['name']:<30} ", end="")
            
            # Test evaluation at origin
            x_zero = torch.zeros(1, func.dim)
            y_zero = func(x_zero)
            
            # Test evaluation at random point
            x_rand = torch.randn(1, func.dim) * 2
            y_rand = func(x_rand)
            
            # Test batch evaluation
            X_batch = torch.randn(3, func.dim)
            Y_batch = func(X_batch)
            
            print(f"OK (f(0)={y_zero.item():.3f})")
            success_count += 1
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    print(f"\nSummary: {success_count}/{len(bbob_suite)} functions working correctly")
    
    if success_count == 24:
        print("SUCCESS: All 24 BBOB functions implemented and working!")
    else:
        print(f"WARNING: Only {success_count}/24 functions working")
    
    return success_count == 24

def test_function_properties():
    """Test function properties and metadata."""
    print("\n=== Testing Function Properties ===")
    
    bbob_suite = bmb.get_functions_by_property('suite', 'BBOB Raw')
    
    # Count by properties
    unimodal = 0
    multimodal = 0
    separable = 0
    non_separable = 0
    
    for func_name, func in bbob_suite.items():
        props = func.metadata.get('properties', [])
        if 'unimodal' in props:
            unimodal += 1
        if 'multimodal' in props:
            multimodal += 1
        if 'separable' in props:
            separable += 1
        if 'non-separable' in props:
            non_separable += 1
    
    print(f"Function property distribution:")
    print(f"  Unimodal: {unimodal}")
    print(f"  Multimodal: {multimodal}")
    print(f"  Separable: {separable}")
    print(f"  Non-separable: {non_separable}")

def main():
    """Run all tests."""
    print("Complete BBOB Raw Implementation Test")
    print("=" * 50)
    
    success = test_all_bbob_functions()
    test_function_properties()
    
    print("\n" + "=" * 50)
    if success:
        print("All tests completed successfully!")
        print("BBOB Raw implementation is complete with all 24 functions.")
    else:
        print("Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main() 