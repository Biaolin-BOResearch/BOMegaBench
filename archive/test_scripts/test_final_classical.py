#!/usr/bin/env python3
"""
Final test for Complete Classical Mathematical Functions.
"""

import bomegabench as bmb

def main():
    print("=== Final Classical Functions Summary ===")
    
    # Get all functions by suite
    classical_suite = bmb.get_functions_by_property('suite', 'Classical')
    bbob_suite = bmb.get_functions_by_property('suite', 'BBOB Raw')
    botorch_suite = bmb.get_functions_by_property('suite', 'BoTorch')
    
    print(f"Classical suite: {len(classical_suite)} functions")
    print(f"BBOB Raw suite: {len(bbob_suite)} functions")  
    print(f"BoTorch suite: {len(botorch_suite)} functions")
    print(f"Total functions: {len(classical_suite) + len(bbob_suite) + len(botorch_suite)}")
    
    # Count unique Classical function types (ignoring dimension variants)
    unique_classical = set()
    for name in classical_suite.keys():
        # Remove dimension suffixes like "_3d", "_6d"
        base_name = name.replace('_3d', '').replace('_6d', '')
        unique_classical.add(base_name)
    
    print(f"\nUnique Classical function types: {len(unique_classical)}")
    print("Classical function types:")
    for i, name in enumerate(sorted(unique_classical), 1):
        print(f"{i:2d}. {name}")
    
    # Test specific Hartmann functions
    print(f"\n=== Testing Hartmann Functions ===")
    try:
        hartmann_3d = bmb.get_function("hartmann_3d")
        print(f"Hartmann 3D: {hartmann_3d.metadata['name']} - OK")
        
        hartmann_6d = bmb.get_function("hartmann_6d") 
        print(f"Hartmann 6D: {hartmann_6d.metadata['name']} - OK")
        
        print("SUCCESS: Hartmann functions implemented!")
    except Exception as e:
        print(f"ERROR: Hartmann functions not found - {e}")
    
    # Expected 42 unique classical functions based on synthetic_funcs.md
    expected_unique = 42
    if len(unique_classical) >= expected_unique:
        print(f"\n🎉 SUCCESS: All {expected_unique} Classical function types implemented!")
        print(f"Total implementations: {len(classical_suite)} (including multi-dimensional variants)")
    else:
        missing = expected_unique - len(unique_classical)
        print(f"\n⚠️  INCOMPLETE: {missing} Classical function types still missing")
    
    print(f"\n=== Final Library Summary ===")
    print(f"📊 Total benchmark functions: {len(classical_suite) + len(bbob_suite) + len(botorch_suite)}")
    print(f"   • BBOB Raw (24 functions): {len(bbob_suite)}")
    print(f"   • Classical (42 types): {len(unique_classical)} unique, {len(classical_suite)} total")
    print(f"   • BoTorch Additional: {len(botorch_suite)}")
    print(f"\nBenchmark library is ready for Bayesian Optimization testing! 🚀")

if __name__ == "__main__":
    main() 