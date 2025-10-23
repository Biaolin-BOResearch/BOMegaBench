#!/usr/bin/env python3
"""Minimal COCO test."""

import cocoex
import numpy as np

print("Testing minimal COCO usage...")

try:
    # Create suite without observer
    suite = cocoex.Suite("bbob", "", "")
    print(f"Suite created with {len(suite)} problems")
    
    # Get first few problems
    problems = list(suite)[:5]
    
    for i, problem in enumerate(problems):
        print(f"\nProblem {i+1}: {problem.name}")
        print(f"  ID: {problem.id}")
        print(f"  Dimension: {problem.dimension}")
        
        # Try to evaluate at origin
        x = np.zeros(problem.dimension)
        try:
            y = problem(x)
            print(f"  f(0) = {y}")
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            
        # Try with observer
        try:
            observer = cocoex.Observer("bbob", "result_folder: temp")
            problem.observe_with(observer)
            y = problem(x)
            print(f"  f(0) with observer = {y}")
        except Exception as e:
            print(f"  Observer failed: {e}")
            
except Exception as e:
    print(f"Suite creation failed: {e}")
    import traceback
    traceback.print_exc() 