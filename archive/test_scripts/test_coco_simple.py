#!/usr/bin/env python3
"""Simple COCO test to understand correct usage."""

import cocoex

# Test different ways of using COCO
print("Testing COCO usage patterns...")

# Method 1: Single problem, single evaluation
print("\nMethod 1: Single problem, single evaluation")
try:
    suite = cocoex.Suite("bbob", "", "function_indices:1 dimensions:2 instance_indices:1")
    problem = list(suite)[0]
    print(f"Problem: {problem.name}")
    
    x = [0.0, 0.0]
    y = problem(x)
    print(f"f({x}) = {y}")
    
    # Try another evaluation on same problem
    x2 = [1.0, 1.0]
    y2 = problem(x2)
    print(f"f({x2}) = {y2}")
    
except Exception as e:
    print(f"Method 1 failed: {e}")
    import traceback
    traceback.print_exc()

# Method 2: Multiple evaluations
print("\nMethod 2: Multiple evaluations on same problem")
try:
    suite = cocoex.Suite("bbob", "", "function_indices:1 dimensions:2 instance_indices:1")
    problem = list(suite)[0]
    
    points = [[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]]
    for x in points:
        y = problem(x)
        print(f"f({x}) = {y}")
        
except Exception as e:
    print(f"Method 2 failed: {e}")
    import traceback
    traceback.print_exc()

# Method 3: Fresh suite for each evaluation
print("\nMethod 3: Fresh suite for each evaluation")
try:
    points = [[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]]
    for x in points:
        suite = cocoex.Suite("bbob", "", "function_indices:1 dimensions:2 instance_indices:1")
        problem = list(suite)[0]
        y = problem(x)
        print(f"f({x}) = {y}")
        
except Exception as e:
    print(f"Method 3 failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting different functions...")
for func_id in [1, 2, 8]:  # Sphere, Ellipsoid, Rosenbrock
    try:
        suite = cocoex.Suite("bbob", "", f"function_indices:{func_id} dimensions:2 instance_indices:1")
        problem = list(suite)[0]
        print(f"Function {func_id} ({problem.name}): f([0,0]) = {problem([0.0, 0.0])}")
    except Exception as e:
        print(f"Function {func_id} failed: {e}") 