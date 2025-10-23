#!/usr/bin/env python3
"""
Test for pure COCO BBOB implementation.
"""

import torch
import numpy as np

def test_pure_coco_import():
    """Test pure COCO import."""
    print("=== Testing Pure COCO Import ===")
    
    try:
        import bomegabench as bmb
        print("✅ BOMegaBench imported successfully")
        
        # Check available functions
        all_functions = bmb.list_functions()
        print(f"Total functions available: {len(all_functions)}")
        
        # Check BBOB functions specifically
        bbob_functions = [f for f in all_functions if any(name in f for name in ['sphere', 'rosenbrock', 'rastrigin'])]
        print(f"BBOB-like functions found: {bbob_functions}")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()


def test_direct_coco_function():
    """Test direct COCO function creation."""
    print("\n=== Testing Direct COCO Function Creation ===")
    
    try:
        from bomegabench.functions.bbob_coco_pure import create_bbob_function
        
        # Try to create a single function
        sphere = create_bbob_function(1, dim=2)  # f1 = Sphere
        print(f"✅ Created sphere function: {sphere.metadata['name']}")
        print(f"  Suite: {sphere.metadata['suite']}")
        print(f"  Implementation: {sphere.metadata.get('implementation', 'COCO')}")
        
        # Test evaluation
        x = torch.tensor([0.0, 0.0])
        y = sphere(x)
        print(f"  f([0,0]) = {y.item():.6f}")
        
        # Test batch evaluation
        X = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        Y = sphere(X)
        print(f"  Batch results: {Y.tolist()}")
        
    except Exception as e:
        print(f"❌ Direct function creation failed: {e}")
        import traceback
        traceback.print_exc()


def test_coco_suite_creation():
    """Test COCO suite creation with limited functions."""
    print("\n=== Testing COCO Suite Creation ===")
    
    try:
        from bomegabench.functions.bbob_coco_pure import create_bbob_pure_coco_suite
        
        # Create a small suite for testing
        suite = create_bbob_pure_coco_suite(dimensions=[2], max_functions=2)
        print(f"✅ Created COCO suite with {len(suite.functions)} functions")
        
        for name, func in suite.functions.items():
            print(f"  {name}: {func.metadata['name']} (f{func.function_idx})")
            
            # Test evaluation
            try:
                x = torch.zeros(1, func.dim)
                y = func(x)
                print(f"    f(0) = {y.item():.6f}")
            except Exception as eval_error:
                print(f"    ❌ Evaluation failed: {eval_error}")
        
    except Exception as e:
        print(f"❌ Suite creation failed: {e}")
        import traceback
        traceback.print_exc()


def test_bmb_integration():
    """Test integration with BOMegaBench."""
    print("\n=== Testing BOMegaBench Integration ===")
    
    try:
        import bomegabench as bmb
        
        # Try to get a function through bmb interface
        try:
            sphere = bmb.get_function("sphere")
            print(f"✅ Got sphere function through bmb: {sphere.metadata['name']}")
            print(f"  Suite: {sphere.metadata['suite']}")
            
            # Test evaluation
            x = torch.tensor([0.0, 0.0])
            y = sphere(x)
            print(f"  f([0,0]) = {y.item():.6f}")
            
        except KeyError:
            print("❌ Sphere function not available through bmb")
            available = bmb.list_functions()
            print(f"Available functions: {list(available.keys())[:10]}...")
            
        # Test suites
        suites = bmb.list_suites()
        print(f"Available suites: {suites}")
        
        for suite_name in suites:
            if 'bbob' in suite_name.lower() or 'coco' in suite_name.lower():
                suite_functions = bmb.list_functions(suite=suite_name)
                print(f"  {suite_name}: {len(suite_functions)} functions")
        
    except Exception as e:
        print(f"❌ BOMegaBench integration failed: {e}")
        import traceback
        traceback.print_exc()


def test_coco_problem_properties():
    """Test COCO problem properties."""
    print("\n=== Testing COCO Problem Properties ===")
    
    try:
        from bomegabench.functions.bbob_coco_pure import create_bbob_function
        
        # Create a function and inspect its COCO properties
        func = create_bbob_function(1, dim=2)
        
        print(f"Function: {func.metadata['name']}")
        print(f"COCO problem available: {hasattr(func, 'problem')}")
        
        if hasattr(func, 'problem'):
            problem = func.problem
            print(f"COCO problem name: {getattr(problem, 'name', 'N/A')}")
            print(f"COCO problem id: {getattr(problem, 'id', 'N/A')}")
            print(f"COCO problem dimension: {getattr(problem, 'dimension', 'N/A')}")
            
            # Test if problem is callable
            if hasattr(problem, '__call__'):
                try:
                    result = problem([0.0, 0.0])
                    print(f"✅ Direct COCO evaluation: {result}")
                except Exception as e:
                    print(f"❌ Direct COCO evaluation failed: {e}")
        
    except Exception as e:
        print(f"❌ COCO properties test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("Pure COCO BBOB Test")
    print("=" * 50)
    
    test_pure_coco_import()
    test_direct_coco_function()
    test_coco_suite_creation()
    test_bmb_integration()
    test_coco_problem_properties()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")


if __name__ == "__main__":
    main() 