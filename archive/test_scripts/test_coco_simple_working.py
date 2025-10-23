#!/usr/bin/env python3
"""
Simple COCO test without observer to verify basic functionality.
"""

import cocoex
import torch
import numpy as np

def test_coco_without_observer():
    """Test COCO functions without observer."""
    print("Testing COCO without observer...")
    
    try:
        # Test BBOB suite
        suite = cocoex.Suite("bbob", "", "function_indices:1 dimensions:2 instance_indices:1")
        print(f"BBOB suite created with {len(suite)} problems")
        
        if len(suite) > 0:
            problem = list(suite)[0]
            print(f"Problem: {problem.name}")
            print(f"ID: {problem.id}")
            print(f"Dimension: {problem.dimension}")
            
            # Test evaluation
            x = [0.0, 0.0]
            y = problem(x)
            print(f"f([0,0]) = {y}")
            
            # Test multiple evaluations
            test_points = [[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]]
            for point in test_points:
                y = problem(point)
                print(f"f({point}) = {y}")
                
    except Exception as e:
        print(f"BBOB test failed: {e}")
        import traceback
        traceback.print_exc()
        
    print()
    
    try:
        # Test BBOB Large-scale
        suite_ls = cocoex.Suite("bbob-largescale", "", "function_indices:1 dimensions:20 instance_indices:1")
        print(f"BBOB Large-scale suite created with {len(suite_ls)} problems")
        
        if len(suite_ls) > 0:
            problem_ls = list(suite_ls)[0]
            print(f"Large-scale problem: {problem_ls.name}")
            print(f"Dimension: {problem_ls.dimension}")
            
            # Test evaluation at origin
            x_ls = [0.0] * problem_ls.dimension
            y_ls = problem_ls(x_ls)
            print(f"f(0) = {y_ls}")
            
    except Exception as e:
        print(f"BBOB Large-scale test failed: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_functions():
    """Test multiple BBOB functions."""
    print("\nTesting multiple BBOB functions...")
    
    # Test first 5 BBOB functions
    for func_id in range(1, 6):
        try:
            suite = cocoex.Suite("bbob", "", f"function_indices:{func_id} dimensions:2 instance_indices:1")
            if len(suite) > 0:
                problem = list(suite)[0]
                print(f"f{func_id} ({problem.name}): f([0,0]) = {problem([0.0, 0.0])}")
        except Exception as e:
            print(f"f{func_id} failed: {e}")


def create_simple_wrapper():
    """Create a simple wrapper function."""
    print("\nTesting simple wrapper...")
    
    class SimpleBBOBFunction:
        def __init__(self, func_id, dim=2):
            self.suite = cocoex.Suite("bbob", "", f"function_indices:{func_id} dimensions:{dim} instance_indices:1")
            self.problem = list(self.suite)[0]
            self.dim = dim
            
        def __call__(self, x):
            if isinstance(x, torch.Tensor):
                x = x.numpy()
            if x.ndim == 2:
                return torch.tensor([self.problem(xi.tolist()) for xi in x])
            else:
                return torch.tensor(self.problem(x.tolist()))
    
    try:
        # Test wrapper
        sphere = SimpleBBOBFunction(1, dim=2)
        
        # Test single evaluation
        x = torch.tensor([0.0, 0.0])
        y = sphere(x)
        print(f"Wrapper sphere f([0,0]) = {y}")
        
        # Test batch evaluation
        X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])
        Y = sphere(X)
        print(f"Wrapper batch evaluation: {Y}")
        
    except Exception as e:
        print(f"Wrapper test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("COCO Simple Working Test")
    print("=" * 40)
    
    test_coco_without_observer()
    test_multiple_functions()
    create_simple_wrapper()
    
    print("\n" + "=" * 40)
    print("✅ COCO tests completed!")


if __name__ == "__main__":
    main() 