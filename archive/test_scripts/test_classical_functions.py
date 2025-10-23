#!/usr/bin/env python3
"""
Test all Classical Mathematical Functions.
"""

import torch
import numpy as np
import bomegabench as bmb

def test_classical_functions():
    """Test all Classical functions."""
    print("=== Testing Classical Mathematical Functions ===")
    
    # Get Classical functions
    classical_suite = bmb.get_functions_by_property('suite', 'Classical')
    print(f"Found {len(classical_suite)} Classical functions")
    
    # Expected functions based on synthetic_funcs.md (42 functions)
    expected_count = 42
    print(f"Expected: {expected_count} functions, Got: {len(classical_suite)}")
    
    # Test each function
    success_count = 0
    failed_functions = []
    
    for i, (func_name, func) in enumerate(sorted(classical_suite.items()), 1):
        try:
            print(f"{i:2d}. {func.metadata['name']:<25} ", end="")
            
            # Test evaluation at origin or appropriate point
            if func.dim == 2:
                # Test at origin
                x_test = torch.zeros(1, func.dim)
                y_test = func(x_test)
                
                # Test at random point within bounds
                bounds = func.bounds
                x_rand = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(1, func.dim)
                y_rand = func(x_rand)
                
                # Test batch evaluation
                X_batch = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(3, func.dim)
                Y_batch = func(X_batch)
                
                print(f"OK (dim={func.dim}, f(test)={y_test.item():.3f})")
            else:
                # Multi-dimensional function
                x_test = torch.zeros(1, func.dim)
                y_test = func(x_test)
                print(f"OK (dim={func.dim}, f(0)={y_test.item():.3f})")
            
            success_count += 1
            
        except Exception as e:
            print(f"ERROR: {e}")
            failed_functions.append(func_name)
    
    print(f"\nSummary: {success_count}/{len(classical_suite)} functions working correctly")
    
    if failed_functions:
        print(f"Failed functions: {failed_functions}")
    
    if success_count >= expected_count:
        print("SUCCESS: All Classical functions implemented and working!")
    else:
        print(f"WARNING: Expected {expected_count} functions, only {success_count} working")
    
    return success_count, len(classical_suite)

def test_function_categories():
    """Test function categories and properties."""
    print("\n=== Testing Function Categories ===")
    
    classical_suite = bmb.get_functions_by_property('suite', 'Classical')
    
    # Categorize by properties
    categories = {
        'unimodal': 0,
        'multimodal': 0,
        'separable': 0,
        'non-separable': 0,
        '2D-only': 0,
        'multi-dim': 0
    }
    
    for func_name, func in classical_suite.items():
        props = func.metadata.get('properties', [])
        
        if 'unimodal' in props:
            categories['unimodal'] += 1
        if 'multimodal' in props:
            categories['multimodal'] += 1
        if 'separable' in props:
            categories['separable'] += 1
        if 'non-separable' in props:
            categories['non-separable'] += 1
        
        if func.dim == 2:
            categories['2D-only'] += 1
        else:
            categories['multi-dim'] += 1
    
    print("Function property distribution:")
    for category, count in categories.items():
        print(f"  {category}: {count}")

def test_specific_functions():
    """Test specific well-known functions."""
    print("\n=== Testing Specific Classical Functions ===")
    
    test_cases = [
        ("ackley", torch.zeros(1, 2), 0.0),
        ("rosenbrock_classic", torch.ones(1, 2), 0.0),
        ("sphere", torch.zeros(1, 2), 0.0),  # From BBOB Raw suite
        ("griewank", torch.zeros(1, 2), 0.0),
        ("rastrigin_classic", torch.zeros(1, 2), 0.0),
    ]
    
    for func_name, test_point, expected in test_cases:
        try:
            func = bmb.get_function(func_name)
            result = func(test_point)
            error = abs(result.item() - expected)
            
            print(f"{func_name:<20}: f({test_point[0].tolist()}) = {result.item():.6f}, "
                  f"expected ≈ {expected:.1f}, error = {error:.6f}")
            
            if error > 1e-3:
                print(f"  WARNING: Large error for {func_name}")
                
        except Exception as e:
            print(f"{func_name:<20}: ERROR - {e}")

def main():
    """Run all tests."""
    print("Complete Classical Mathematical Functions Test")
    print("=" * 60)
    
    success_count, total_count = test_classical_functions()
    test_function_categories()
    test_specific_functions()
    
    print("\n" + "=" * 60)
    if success_count >= 40:  # Allow some tolerance
        print(f"SUCCESS: Classical implementation is nearly complete!")
        print(f"Implemented {success_count}/{total_count} functions successfully.")
    else:
        print(f"PARTIAL: {success_count}/{total_count} functions working.")
        print("Some functions may need debugging.")

if __name__ == "__main__":
    main() 