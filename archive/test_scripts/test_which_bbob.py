#!/usr/bin/env python3
"""Check which BBOB implementation is being used."""

import bomegabench as bmb

# Get sphere function and check its type and metadata
sphere = bmb.get_function("sphere")

print(f"Function type: {type(sphere)}")
print(f"Function module: {type(sphere).__module__}")
print(f"Suite: {sphere.metadata['suite']}")
print()

# Check if it has COCO-specific attributes
if hasattr(sphere, 'coco_problem'):
    print("✓ Using COCO implementation")
    print(f"COCO problem name: {sphere.coco_problem.name}")
    print(f"Function ID: {sphere.function_id}")
    print(f"Instance ID: {sphere.instance_id}")
else:
    print("✗ Using manual implementation")
print()

# Test a few more functions
test_functions = ["ellipsoid_separable", "rastrigin_separable"]
for func_name in test_functions:
    func = bmb.get_function(func_name)
    suite = func.metadata['suite']
    has_coco = hasattr(func, 'coco_problem')
    print(f"{func_name}: {suite} {'(COCO)' if has_coco else '(Manual)'}") 