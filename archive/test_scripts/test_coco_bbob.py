#!/usr/bin/env python3
"""Test COCO BBOB functions integration."""

import torch
import bomegabench as bmb

def test_coco_bbob():
    """Test COCO BBOB functions."""
    print("Testing COCO BBOB functions...")
    
    try:
        # Test sphere function
        sphere = bmb.get_function("sphere")
        print(f"Sphere function: {sphere.metadata['name']}")
        print(f"Suite: {sphere.metadata['suite']}")
        print(f"Properties: {sphere.metadata['properties']}")
        
        # Test evaluation
        x = torch.tensor([[0.0, 0.0]])
        y = sphere(x)
        print(f"f([0, 0]) = {y.item():.6f}")
        
        # Test batch evaluation
        X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])
        Y = sphere(X)
        print("Batch evaluation:")
        for i, (xi, yi) in enumerate(zip(X, Y)):
            print(f"  f({xi.tolist()}) = {yi.item():.6f}")
        print()
        
        # Test other BBOB functions
        test_functions = ["ellipsoid_separable", "rastrigin_separable", "rosenbrock", "schwefel"]
        
        for func_name in test_functions:
            try:
                func = bmb.get_function(func_name)
                print(f"{func.metadata['name']}: {func.metadata['properties']}")
                
                x = torch.zeros(1, func.dim)
                y = func(x)
                print(f"  f(0) = {y.item():.6f}")
                
            except Exception as e:
                print(f"  Error testing {func_name}: {e}")
        print()
        
        # Test function discovery
        all_functions = bmb.list_functions()
        bbob_functions = [f for f in all_functions if "sphere" in f or "rosenbrock" in f or "rastrigin" in f]
        print(f"BBOB-like functions found: {bbob_functions}")
        
    except Exception as e:
        print(f"Error testing COCO BBOB: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_coco_bbob() 