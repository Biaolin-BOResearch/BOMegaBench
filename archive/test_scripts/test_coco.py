#!/usr/bin/env python3
"""Test COCO platform availability and BBOB functions."""

try:
    import cocoex
    print("COCO imported successfully")
    
    # Create BBOB suite
    suite = cocoex.Suite("bbob", "", "")
    print(f"BBOB suite created with {len(suite)} functions")
    
    # Test a few functions
    for i, problem in enumerate(suite):
        if i >= 5:  # Just test first 5
            break
        print(f"Function {problem.id}: {problem.name}, dim={problem.dimension}")
        
        # Test evaluation
        x = [0.0] * problem.dimension
        y = problem(x)
        print(f"  f({x}) = {y}")
        
except ImportError as e:
    print(f"COCO not available: {e}")
except Exception as e:
    print(f"Error testing COCO: {e}")
    import traceback
    traceback.print_exc() 