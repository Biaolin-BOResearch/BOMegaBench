#!/usr/bin/env python3
"""
Complete BBOB test including BBOB-Core and BBOB-LargeScale using COCO platform.
"""

import torch
import numpy as np
import bomegabench as bmb

def test_bbob_core():
    """Test BBOB Core functions."""
    print("=== Testing BBOB Core Functions ===")
    
    # Test basic BBOB functions
    core_functions = ["sphere", "ellipsoid_separable", "rastrigin_separable", 
                     "rosenbrock", "ellipsoid", "discus", "bent_cigar", 
                     "rastrigin", "schwefel"]
    
    for func_name in core_functions:
        try:
            func = bmb.get_function(func_name)
            print(f"\n{func.metadata['name']} (f{func.metadata.get('function_idx', '?')})")
            print(f"  Suite: {func.metadata['suite']}")
            print(f"  Properties: {func.metadata['properties']}")
            print(f"  Domain: {func.metadata['domain']}")
            print(f"  Dimension: {func.dim}")
            
            # Test evaluation at origin
            x_origin = torch.zeros(1, func.dim)
            y_origin = func(x_origin)
            print(f"  f(0) = {y_origin.item():.6f}")
            
            # Test evaluation at random points
            x_random = func.sample_random(3)
            y_random = func(x_random)
            print(f"  Random evaluations: {y_random.tolist()}")
            
            # Test batch evaluation
            X_batch = torch.randn(5, func.dim) * 2  # Random points in [-2, 2]^d
            Y_batch = func(X_batch)
            print(f"  Batch evaluation (5 points): mean={Y_batch.mean().item():.3f}, std={Y_batch.std().item():.3f}")
            
        except Exception as e:
            print(f"  ❌ Error testing {func_name}: {e}")


def test_bbob_largescale():
    """Test BBOB Large-scale functions."""
    print("\n=== Testing BBOB Large-scale Functions ===")
    
    # Test large-scale functions
    try:
        # List all available functions
        all_functions = bmb.list_functions()
        largescale_functions = [f for f in all_functions if "20d" in f or "40d" in f]
        
        print(f"Found {len(largescale_functions)} large-scale functions:")
        print(f"  {largescale_functions}")
        
        # Test a few large-scale functions
        test_functions = largescale_functions[:5] if largescale_functions else []
        
        for func_name in test_functions:
            try:
                func = bmb.get_function(func_name)
                print(f"\n{func.metadata['name']} ({func.dim}D)")
                print(f"  Suite: {func.metadata['suite']}")
                print(f"  Properties: {func.metadata['properties']}")
                
                # Test evaluation at origin
                x_origin = torch.zeros(1, func.dim)
                y_origin = func(x_origin)
                print(f"  f(0) = {y_origin.item():.6f}")
                
                # Test single random evaluation (large-scale can be expensive)
                x_random = func.sample_random(1)
                y_random = func(x_random)
                print(f"  f(random) = {y_random.item():.6f}")
                
            except Exception as e:
                print(f"  ❌ Error testing {func_name}: {e}")
                
    except Exception as e:
        print(f"❌ Error accessing large-scale functions: {e}")


def test_function_properties():
    """Test function property queries."""
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
        print(f"❌ Error testing properties: {e}")


def test_dimensions():
    """Test different dimensions."""
    print("\n=== Testing Different Dimensions ===")
    
    # Test sphere function at different dimensions
    for dim in [2, 5, 10, 20]:
        try:
            # Try to get a function with specific dimension
            func_name = f"sphere_{dim}d" if dim != 2 else "sphere"
            
            if func_name in bmb.list_functions():
                func = bmb.get_function(func_name)
                print(f"Sphere {dim}D: {func.metadata['suite']}")
                
                # Test evaluation
                x = torch.zeros(1, dim)
                y = func(x)
                print(f"  f(0) = {y.item():.6f}")
            else:
                print(f"Sphere {dim}D: Not available")
                
        except Exception as e:
            print(f"❌ Error testing {dim}D: {e}")


def test_coco_specific_features():
    """Test COCO-specific features."""
    print("\n=== Testing COCO-Specific Features ===")
    
    try:
        # Get a COCO function and check its metadata
        sphere = bmb.get_function("sphere")
        
        print(f"Function type: {type(sphere)}")
        print(f"Has COCO problem: {hasattr(sphere, 'coco_problem')}")
        
        if hasattr(sphere, 'coco_problem'):
            print(f"COCO problem name: {sphere.coco_problem.name}")
            print(f"COCO problem ID: {sphere.coco_problem.id}")
            print(f"Function index: {sphere.function_idx}")
            print(f"Instance index: {sphere.instance_idx}")
            print(f"Suite name: {sphere.suite_name}")
        
        # Test metadata
        metadata = sphere.metadata
        coco_keys = [k for k in metadata.keys() if 'coco' in k.lower()]
        print(f"COCO metadata keys: {coco_keys}")
        
        for key in coco_keys:
            print(f"  {key}: {metadata[key]}")
            
    except Exception as e:
        print(f"❌ Error testing COCO features: {e}")


def run_performance_test():
    """Run a simple performance test."""
    print("\n=== Performance Test ===")
    
    try:
        import time
        
        # Test evaluation speed
        func = bmb.get_function("sphere")
        
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
        
        print(f"100 single evaluations: {single_time:.4f}s ({100/single_time:.1f} evals/s)")
        print(f"1 batch of 100 evaluations: {batch_time:.4f}s ({100/batch_time:.1f} evals/s)")
        print(f"Batch speedup: {single_time/batch_time:.1f}x")
        
    except Exception as e:
        print(f"❌ Error in performance test: {e}")


def main():
    """Run all BBOB tests."""
    print("BBOB Complete Test Suite")
    print("=" * 50)
    
    try:
        # Basic tests
        test_bbob_core()
        test_bbob_largescale()
        test_function_properties()
        test_dimensions()
        test_coco_specific_features()
        run_performance_test()
        
        print("\n" + "=" * 50)
        print("✅ All BBOB tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 