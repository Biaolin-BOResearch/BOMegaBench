#!/usr/bin/env python3
"""
Final test for BBOB COCO implementation.
"""

import torch
import numpy as np
import bomegabench as bmb

def test_bbob_coco_basic():
    """Test basic BBOB COCO functionality."""
    print("=== Testing BBOB COCO Basic Functions ===")
    
    # Test a few core BBOB functions
    test_functions = ["sphere", "rosenbrock", "rastrigin"]
    
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
            x_random = func.sample_random(2)
            y_random = func(x_random)
            print(f"  Random evaluations: {y_random.tolist()}")
            
            # Test if this is actually using COCO
            if hasattr(func, 'problem'):
                print(f"  ✅ Using COCO implementation")
                if hasattr(func.problem, 'name'):
                    print(f"  COCO name: {func.problem.name}")
            else:
                print(f"  ⚠️  Using fallback implementation")
            
        except Exception as e:
            print(f"  ❌ Error testing {func_name}: {e}")
            import traceback
            traceback.print_exc()


def test_function_comparison():
    """Compare COCO vs manual implementation if both available."""
    print("\n=== Testing Function Comparison ===")
    
    try:
        # Try to get both implementations
        from bomegabench.functions.bbob_coco_simple import create_bbob_function
        from bomegabench.functions.synthetic_functions import BBOBSphereFunction
        
        # Test sphere function
        coco_sphere = create_bbob_function(1, dim=2)  # f1 = Sphere
        manual_sphere = BBOBSphereFunction(dim=2)
        
        # Test at same points
        test_points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])
        
        coco_results = coco_sphere(test_points)
        manual_results = manual_sphere(test_points)
        
        print("Sphere function comparison:")
        print(f"  Test points: {test_points.tolist()}")
        print(f"  COCO results: {coco_results.tolist()}")
        print(f"  Manual results: {manual_results.tolist()}")
        print(f"  Difference: {(coco_results - manual_results).abs().max().item():.6f}")
        
    except Exception as e:
        print(f"Comparison failed: {e}")


def test_suite_overview():
    """Test suite overview."""
    print("\n=== Testing Suite Overview ===")
    
    try:
        # List all functions
        all_functions = bmb.list_functions()
        print(f"Total functions available: {len(all_functions)}")
        
        # List by suite
        suites = bmb.list_suites()
        print(f"Available suites: {suites}")
        
        for suite_name in suites:
            suite_functions = bmb.list_functions(suite=suite_name)
            print(f"  {suite_name}: {len(suite_functions)} functions")
            if suite_name.startswith("BBOB") and len(suite_functions) > 0:
                print(f"    Examples: {list(suite_functions.keys())[:3]}")
        
        # Test properties
        multimodal = bmb.get_multimodal_functions()
        unimodal = bmb.get_unimodal_functions()
        separable = bmb.get_functions_by_property("properties", "separable")
        
        print(f"\nFunction properties:")
        print(f"  Multimodal: {len(multimodal)}")
        print(f"  Unimodal: {len(unimodal)}")
        print(f"  Separable: {len(separable)}")
        
    except Exception as e:
        print(f"Suite overview failed: {e}")
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
        for _ in range(50):  # Reduced for COCO
            x = torch.randn(1, func.dim)
            y = func(x)
        single_time = time.time() - start_time
        
        # Batch evaluation
        start_time = time.time()
        X = torch.randn(50, func.dim)  # Reduced batch size
        Y = func(X)
        batch_time = time.time() - start_time
        
        print(f"  50 single evaluations: {single_time:.4f}s ({50/single_time:.1f} evals/s)")
        print(f"  1 batch of 50 evaluations: {batch_time:.4f}s ({50/batch_time:.1f} evals/s)")
        if batch_time > 0:
            print(f"  Batch speedup: {single_time/batch_time:.1f}x")
        
    except Exception as e:
        print(f"Performance test failed: {e}")


def main():
    """Run all tests."""
    print("BBOB COCO Final Test")
    print("=" * 50)
    
    try:
        test_bbob_coco_basic()
        test_function_comparison()
        test_suite_overview()
        test_performance()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 