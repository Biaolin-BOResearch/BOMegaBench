#!/usr/bin/env python3
"""
Test for BBOB functions based on COCO raw implementations.
"""

import torch
import numpy as np
import bomegabench as bmb

def test_bbob_raw_basic():
    """Test basic BBOB raw functionality."""
    print("=== Testing BBOB Raw Basic Functions ===")
    
    # Test core BBOB functions
    test_functions = ["sphere", "ellipsoid_separable", "rastrigin_separable", 
                     "rosenbrock", "ellipsoid", "discus", "bent_cigar", 
                     "rastrigin", "schwefel"]
    
    for func_name in test_functions:
        try:
            func = bmb.get_function(func_name)
            print(f"\n{func.metadata['name']} (f{func.metadata.get('function_idx', '?')})")
            print(f"  Suite: {func.metadata['suite']}")
            print(f"  Properties: {func.metadata['properties']}")
            print(f"  Dimension: {func.dim}")
            
            # Test evaluation at origin
            x_origin = torch.zeros(1, func.dim)
            y_origin = func(x_origin)
            print(f"  f(0) = {y_origin.item():.6f}")
            
            # Test evaluation at random points
            x_random = func.sample_random(3)
            y_random = func(x_random)
            print(f"  Random evaluations: {[f'{y:.6f}' for y in y_random.tolist()]}")
            
            # Test batch evaluation
            X_batch = torch.randn(5, func.dim) * 2
            Y_batch = func(X_batch)
            print(f"  Batch evaluation (5 points): mean={Y_batch.mean().item():.3f}, std={Y_batch.std().item():.3f}")
            
            # Test function-specific properties
            if hasattr(func, 'function_idx'):
                print(f"  Function index: {func.function_idx}")
            if hasattr(func, 'instance'):
                print(f"  Instance: {func.instance}")
            
        except Exception as e:
            print(f"  Error testing {func_name}: {e}")
            import traceback
            traceback.print_exc()


def test_raw_functions():
    """Test raw function properties."""
    print("\n=== Testing Raw Function Properties ===")
    
    try:
        # Test sphere function with different instances
        for instance in [1, 2, 3]:
            from bomegabench.functions.bbob_coco_raw import F01_SphereRaw
            
            func = F01_SphereRaw(dim=2, instance=instance)
            print(f"\nSphere instance {instance}:")
            print(f"  Function index: {func.function_idx}")
            print(f"  Instance: {func.instance}")
            
            # Test that optimum is at origin (raw function)
            x_zero = torch.zeros(1, 2)
            y_zero = func(x_zero)
            print(f"  f(0) = {y_zero.item():.6f} (should be 0.0 for raw sphere)")
            
            # Test known point
            x_one = torch.ones(1, 2)
            y_one = func(x_one)
            print(f"  f(1,1) = {y_one.item():.6f} (should be 2.0 for raw sphere)")
            
    except Exception as e:
        print(f"Raw functions test failed: {e}")
        import traceback
        traceback.print_exc()


def test_function_properties():
    """Test function properties and metadata."""
    print("\n=== Testing Function Properties ===")
    
    try:
        # Get all functions
        all_functions = bmb.list_functions()
        print(f"Total functions available: {len(all_functions)}")
        
        # Test property queries
        multimodal = bmb.get_multimodal_functions()
        unimodal = bmb.get_unimodal_functions()
        separable = bmb.get_functions_by_property("properties", "separable")
        
        print(f"Multimodal functions: {len(multimodal)}")
        print(f"  Examples: {list(multimodal.keys())[:5]}")
        
        print(f"Unimodal functions: {len(unimodal)}")
        print(f"  Examples: {list(unimodal.keys())[:5]}")
        
        print(f"Separable functions: {len(separable)}")
        print(f"  Examples: {list(separable.keys())[:5]}")
        
        # Test suite breakdown
        suites = bmb.list_suites()
        print(f"Available suites: {suites}")
        
        for suite_name in suites:
            suite_functions = bmb.list_functions(suite=suite_name)
            print(f"  {suite_name}: {len(suite_functions)} functions")
            
    except Exception as e:
        print(f"Properties test failed: {e}")
        import traceback
        traceback.print_exc()


def test_mathematical_correctness():
    """Test mathematical correctness of raw implementations."""
    print("\n=== Testing Mathematical Correctness ===")
    
    try:
        # Test sphere function
        sphere = bmb.get_function("sphere")
        x_test = torch.tensor([[1.0, 1.0]])
        y_test = sphere(x_test)
        
        # Manual calculation for raw sphere: sum(x_i^2)
        expected = np.sum(x_test.numpy()[0]**2)
        
        print(f"Sphere test:")
        print(f"  Input: {x_test[0].tolist()}")
        print(f"  Output: {y_test.item():.6f}")
        print(f"  Expected: {expected:.6f}")
        print(f"  Difference: {abs(y_test.item() - expected):.8f}")
        
        # Test Rosenbrock function
        rosenbrock = bmb.get_function("rosenbrock")
        x_test = torch.tensor([[1.0, 1.0]])  # Optimum for raw Rosenbrock
        y_test = rosenbrock(x_test)
        
        # Manual calculation for raw Rosenbrock at optimum (1,1)
        x = x_test.numpy()[0]
        expected = 100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2
        
        print(f"\nRosenbrock test:")
        print(f"  Input: {x_test[0].tolist()}")
        print(f"  Output: {y_test.item():.6f}")
        print(f"  Expected: {expected:.6f}")
        print(f"  Difference: {abs(y_test.item() - expected):.8f}")
        
    except Exception as e:
        print(f"Mathematical correctness test failed: {e}")
        import traceback
        traceback.print_exc()


def test_performance():
    """Test evaluation performance."""
    print("\n=== Testing Performance ===")
    
    try:
        import time
        
        # Test sphere function
        func = bmb.get_function("sphere")
        print(f"Testing {func.metadata['name']} performance...")
        
        # Single evaluations
        start_time = time.time()
        for _ in range(100):
            x = torch.randn(1, func.dim)
            y = func(x)
        single_time = time.time() - start_time
        
        # Batch evaluations
        start_time = time.time()
        X = torch.randn(100, func.dim)
        Y = func(X)
        batch_time = time.time() - start_time
        
        print(f"  100 single evaluations: {single_time:.4f}s ({100/single_time:.1f} evals/s)")
        print(f"  1 batch of 100 evaluations: {batch_time:.4f}s ({100/batch_time:.1f} evals/s)")
        if batch_time > 0:
            print(f"  Batch speedup: {single_time/batch_time:.1f}x")
        
    except Exception as e:
        print(f"Performance test failed: {e}")


def main():
    """Run all tests."""
    print("BBOB COCO Raw Implementation Test")
    print("=" * 50)
    
    try:
        test_bbob_raw_basic()
        test_raw_functions()
        test_function_properties()
        test_mathematical_correctness()
        test_performance()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        
    except Exception as e:
        print(f"\nTest suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 