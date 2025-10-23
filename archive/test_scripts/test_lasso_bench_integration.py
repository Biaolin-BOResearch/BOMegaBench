#!/usr/bin/env python3
"""
Test script for LassoBench integration with BOMegaBench.

This script tests the integration without requiring LassoBench to be installed,
and provides clear feedback about what's available.
"""

import sys
import warnings

def test_basic_bomegabench():
    """Test basic BOMegaBench functionality."""
    print("Testing basic BOMegaBench functionality...")
    
    try:
        import bomegabench as bmb
        print("✓ BOMegaBench imported successfully")
        
        # List suites
        suites = bmb.list_suites()
        print(f"✓ Found {len(suites)} suites: {suites}")
        
        # Test a basic function
        func = bmb.get_function("styblinski_tang", "classical")
        print(f"✓ Classical function test: {func.metadata['name']}")
        
        return True
    except Exception as e:
        print(f"✗ Error testing basic functionality: {e}")
        return False


def test_lasso_bench_availability():
    """Test if LassoBench is available and integrated."""
    print("\nTesting LassoBench availability...")
    
    try:
        import LassoBench
        print("✓ LassoBench library is installed")
        lasso_available = True
    except ImportError:
        print("✗ LassoBench library not installed")
        print("  Install with: pip install git+https://github.com/ksehic/LassoBench.git")
        lasso_available = False
    
    # Test integration regardless of LassoBench availability
    try:
        import bomegabench as bmb
        suites = bmb.list_suites()
        lasso_suites = [s for s in suites if 'lasso' in s]
        
        if lasso_available:
            if lasso_suites:
                print(f"✓ LassoBench suites integrated: {lasso_suites}")
                return True, lasso_suites
            else:
                print("✗ LassoBench installed but suites not found in registry")
                return False, []
        else:
            if lasso_suites:
                print("✗ LassoBench suites found but library not installed (shouldn't happen)")
                return False, lasso_suites
            else:
                print("✓ LassoBench suites correctly excluded (library not installed)")
                return True, []
                
    except Exception as e:
        print(f"✗ Error testing LassoBench integration: {e}")
        return False, []


def test_lasso_bench_functions():
    """Test LassoBench functions if available."""
    print("\nTesting LassoBench functions...")
    
    try:
        import bomegabench as bmb
        
        # Check if LassoBench suites are available
        suites = bmb.list_suites()
        lasso_suites = [s for s in suites if 'lasso' in s]
        
        if not lasso_suites:
            print("ℹ LassoBench suites not available - skipping function tests")
            return True
        
        # Test synthetic functions
        if 'lasso_synthetic' in lasso_suites:
            print("Testing synthetic functions...")
            functions = bmb.list_functions('lasso_synthetic')
            print(f"  Found {len(functions)} synthetic functions")
            
            # Test one function
            func = bmb.get_function(functions[0], 'lasso_synthetic')
            print(f"  ✓ Created function: {func.metadata['name']}")
            print(f"    Dimension: {func.dim}")
            print(f"    Properties: {func.metadata['properties']}")
        
        # Test real functions
        if 'lasso_real' in lasso_suites:
            print("Testing real-world functions...")
            functions = bmb.list_functions('lasso_real')
            print(f"  Found {len(functions)} real-world functions")
            
            # Test smallest function (diabetes)
            if 'diabetes' in functions:
                func = bmb.get_function('diabetes', 'lasso_real')
                print(f"  ✓ Created function: {func.metadata['name']}")
                print(f"    Dimension: {func.dim}")
        
        # Multi-fidelity functions temporarily removed
        
        print("✓ All LassoBench function tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Error testing LassoBench functions: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lasso_bench_evaluation():
    """Test actual function evaluation if LassoBench is available."""
    print("\nTesting LassoBench function evaluation...")
    
    try:
        import bomegabench as bmb
        import torch
        
        # Check if LassoBench suites are available
        suites = bmb.list_suites()
        if 'lasso_synthetic' not in suites:
            print("ℹ LassoBench synthetic suite not available - skipping evaluation tests")
            return True
        
        # Test function evaluation
        func = bmb.get_function('synt_simple_noiseless', 'lasso_synthetic')
        print(f"Testing evaluation of {func.metadata['name']}")
        
        # Create test input
        X = torch.rand(1, func.dim) * 2 - 1  # Scale to [-1, 1]
        
        # Evaluate function
        result = func(X)
        print(f"  ✓ Function evaluation successful: {result.item():.6f}")
        
        # Test batch evaluation
        X_batch = torch.rand(5, func.dim) * 2 - 1
        results_batch = func(X_batch)
        print(f"  ✓ Batch evaluation successful: {results_batch.shape}")
        
        # Test additional methods if available
        if hasattr(func, 'get_test_metrics'):
            metrics = func.get_test_metrics(X)
            print(f"  ✓ Test metrics: MSE={metrics['mspe']:.6f}, F-score={metrics['fscore']:.6f}")
        
        if hasattr(func, 'get_active_dimensions'):
            active_dims = func.get_active_dimensions()
            print(f"  ✓ Active dimensions: {len(active_dims)} out of {func.dim}")
        
        print("✓ All evaluation tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Error testing function evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("BOMegaBench LassoBench Integration Test")
    print("=" * 50)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    all_passed = True
    
    # Test 1: Basic functionality
    if not test_basic_bomegabench():
        all_passed = False
    
    # Test 2: LassoBench availability
    lasso_ok, lasso_suites = test_lasso_bench_availability()
    if not lasso_ok:
        all_passed = False
    
    # Test 3: Function creation (if LassoBench available)
    if not test_lasso_bench_functions():
        all_passed = False
    
    # Test 4: Function evaluation (if LassoBench available)
    if not test_lasso_bench_evaluation():
        all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed!")
        if lasso_suites:
            print(f"✓ LassoBench integration is working with {len(lasso_suites)} suites")
        else:
            print("ℹ LassoBench not installed - integration ready when library is available")
    else:
        print("✗ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 