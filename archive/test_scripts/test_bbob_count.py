#!/usr/bin/env python3
"""
Quick test to count BBOB functions.
"""

import bomegabench as bmb

def main():
    print("=== BBOB Function Count Test ===")
    
    # Get all functions
    all_functions = bmb.list_functions()
    print(f"Total functions: {len(all_functions)}")
    
    # Get BBOB Raw functions
    try:
        bbob_suite = bmb.get_functions_by_property('suite', 'BBOB Raw')
        print(f"BBOB Raw functions: {len(bbob_suite)}")
        print(f"BBOB Raw function names: {sorted(bbob_suite.keys())}")
    except Exception as e:
        print(f"Error getting BBOB Raw functions: {e}")
    
    # Test a few specific functions
    test_funcs = ['sphere', 'rosenbrock', 'rastrigin', 'schwefel']
    print(f"\nTesting specific functions:")
    for func_name in test_funcs:
        try:
            func = bmb.get_function(func_name)
            print(f"  {func_name}: OK - {func.metadata['name']} (f{func.metadata['function_idx']})")
        except Exception as e:
            print(f"  {func_name}: ERROR - {e}")

if __name__ == "__main__":
    main() 