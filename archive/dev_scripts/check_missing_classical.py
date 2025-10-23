#!/usr/bin/env python3
"""
Check for missing Classical functions.
"""

import bomegabench as bmb

def main():
    # Expected 42 classical functions based on synthetic_funcs.md
    expected_classical = [
        "Ackley", "Griewank", "Schwefel 1.2", "Schwefel 2.20", "Schwefel 2.21",
        "Schwefel 2.22", "Schwefel 2.23", "Schwefel 2.26", "Styblinski-Tang",
        "Dixon-Price", "Levy", "Levy N.13", "Michalewicz", "Zakharov", "Salomon",
        "Alpine 1", "Alpine 2", "Schaffer F1", "Schaffer F2", "Schaffer F3",
        "Schaffer F4", "Schaffer F5", "Schaffer F6", "Schaffer F7", "Easom",
        "Cross-in-tray", "Eggholder", "Holder Table", "Drop-wave", "Shubert",
        "Powell", "Trid", "Booth", "Matyas", "McCormick", "Six-Hump Camel",
        "Goldstein-Price", "Beale", "Branin", "Hartmann 3D", "Hartmann 6D",
        "Rosenbrock Classic"  # This might be in BBOB or Classical
    ]
    
    print("=== Classical Function Analysis ===")
    
    # Get all functions
    all_funcs = bmb.list_functions()
    print(f"Total functions in library: {len(all_funcs)}")
    
    # Get Classical suite
    classical_suite = bmb.get_functions_by_property('suite', 'Classical')
    print(f"Classical suite functions: {len(classical_suite)}")
    
    # Get BoTorch suite (might contain Hartmann)
    botorch_suite = bmb.get_functions_by_property('suite', 'BoTorch')
    print(f"BoTorch suite functions: {len(botorch_suite)}")
    
    # Get BBOB Raw suite (might contain Rosenbrock)
    bbob_suite = bmb.get_functions_by_property('suite', 'BBOB Raw')
    print(f"BBOB Raw suite functions: {len(bbob_suite)}")
    
    print(f"\nClassical function names:")
    for i, name in enumerate(sorted(classical_suite.keys()), 1):
        print(f"{i:2d}. {name}")
    
    print(f"\nBoTorch function names:")
    for name in sorted(botorch_suite.keys()):
        print(f"    {name}")
    
    # Check if Hartmann functions are in BoTorch
    hartmann_in_botorch = [name for name in botorch_suite.keys() if 'hartmann' in name.lower()]
    print(f"\nHartmann functions in BoTorch: {hartmann_in_botorch}")
    
    # Check if Rosenbrock is in BBOB
    rosenbrock_in_bbob = [name for name in bbob_suite.keys() if 'rosenbrock' in name.lower()]
    print(f"Rosenbrock functions in BBOB: {rosenbrock_in_bbob}")
    
    # Total classical-related functions
    total_classical_related = len(classical_suite) + len(hartmann_in_botorch)
    if 'rosenbrock_classic' in classical_suite:
        total_classical_related += 0  # Already counted
    else:
        # Check if rosenbrock_classic exists anywhere
        rosenbrock_classic = [name for name in all_funcs if 'rosenbrock' in name.lower() and 'classic' in name.lower()]
        total_classical_related += len(rosenbrock_classic)
        print(f"Rosenbrock Classic functions: {rosenbrock_classic}")
    
    print(f"\nTotal Classical-related functions: {total_classical_related}")
    print(f"Expected: 42, Got: {total_classical_related}")
    
    if total_classical_related >= 42:
        print("SUCCESS: All Classical functions accounted for!")
    else:
        missing = 42 - total_classical_related
        print(f"MISSING: {missing} functions still need to be implemented")

if __name__ == "__main__":
    main() 