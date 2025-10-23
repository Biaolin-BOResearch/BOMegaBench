#!/usr/bin/env python3
"""
Example demonstrating LassoBench integration with BOMegaBench.

This example shows how to use LassoBench synthetic and real-world benchmarks
through the unified BOMegaBench interface.

Requirements:
    pip install git+https://github.com/ksehic/LassoBench.git
    
Note: LassoBench has additional dependencies (celer, sparse-ho, libsvmdata, etc.)
that will be installed automatically.
"""

import numpy as np
import torch
import bomegabench as bmb

def main():
    print("LassoBench Integration Example")
    print("=" * 40)
    
    # Check if LassoBench is available
    try:
        import LassoBench
        print("✓ LassoBench is available")
    except ImportError:
        print("✗ LassoBench not found. Install with:")
        print("  pip install git+https://github.com/ksehic/LassoBench.git")
        return
    
    # List all available suites
    print("\nAvailable suites:")
    suites = bmb.list_suites()
    for suite in suites:
        if 'lasso' in suite:
            print(f"  {suite}")
    
    print("\n" + "="*50)
    print("1. LassoBench Synthetic Functions")
    print("="*50)
    
    # Get synthetic LassoBench functions
    try:
        synthetic_functions = bmb.list_functions("lasso_synthetic")
        print(f"Available synthetic functions: {len(synthetic_functions)}")
        for func_name in synthetic_functions[:4]:  # Show first 4
            print(f"  - {func_name}")
        
        # Test a synthetic function
        func = bmb.get_function("synt_simple_noiseless", "lasso_synthetic")
        print(f"\nTesting {func.metadata['name']}:")
        print(f"  Dimension: {func.dim}")
        print(f"  Properties: {func.metadata['properties']}")
        print(f"  Active dimensions: {func.metadata['active_dimensions']}")
        
        # Generate random test point
        X = torch.rand(1, func.dim) * 2 - 1  # Scale to [-1, 1]
        result = func(X)
        print(f"  Random evaluation: {result.item():.6f}")
        
        # Get test metrics
        if hasattr(func, 'get_test_metrics'):
            metrics = func.get_test_metrics(X)
            print(f"  Test MSE: {metrics['mspe']:.6f}")
            print(f"  F-score: {metrics['fscore']:.6f}")
        
        # Show active dimensions
        if hasattr(func, 'get_active_dimensions'):
            active_dims = func.get_active_dimensions()
            print(f"  Active dimension indices: {active_dims}")
            
    except Exception as e:
        print(f"Error with synthetic functions: {e}")
    
    print("\n" + "="*50)
    print("2. LassoBench Real-world Functions")
    print("="*50)
    
    # Test real-world functions
    try:
        real_functions = bmb.list_functions("lasso_real")
        print(f"Available real-world functions: {len(real_functions)}")
        for func_name in real_functions:
            print(f"  - {func_name}")
        
        # Test a smaller real-world function (diabetes)
        func = bmb.get_function("diabetes", "lasso_real")
        print(f"\nTesting {func.metadata['name']}:")
        print(f"  Dimension: {func.dim}")
        print(f"  Properties: {func.metadata['properties']}")
        print(f"  Dataset: {func.metadata['dataset']}")
        
        # Generate random test point
        X = torch.rand(1, func.dim) * 2 - 1
        result = func(X)
        print(f"  Random evaluation: {result.item():.6f}")
        
        # Get test metrics
        if hasattr(func, 'get_test_metrics'):
            metrics = func.get_test_metrics(X)
            print(f"  Test MSE: {metrics['mspe']:.6f}")
            
    except Exception as e:
        print(f"Error with real-world functions: {e}")
    
    print("\n" + "="*50)
    print("3. LassoBench Multi-Fidelity Functions")
    print("="*50)
    
    # Test multi-fidelity functions
    try:
        mf_functions = bmb.list_functions("lasso_multifidelity")
        print(f"Available multi-fidelity functions: {len(mf_functions)}")
        for func_name in mf_functions[:4]:  # Show first 4
            print(f"  - {func_name}")
        
        # Test a multi-fidelity function
        func = bmb.get_function("synt_simple_mf_discrete", "lasso_multifidelity")
        print(f"\nTesting {func.metadata['name']}:")
        print(f"  Dimension: {func.dim}")
        print(f"  Fidelity type: {func.metadata['fidelity_type']}")
        
        # Test different fidelity levels
        X = torch.rand(1, func.dim) * 2 - 1
        
        print("  Evaluations at different fidelities:")
        for fidelity in [0, 2, 4]:  # Low, medium, high fidelity
            if hasattr(func, 'evaluate_fidelity'):
                result = func.evaluate_fidelity(X, fidelity_index=fidelity)
                print(f"    Fidelity {fidelity}: {result.item():.6f}")
        
    except Exception as e:
        print(f"Error with multi-fidelity functions: {e}")
    
    print("\n" + "="*50)
    print("4. Optimization Example")
    print("="*50)
    
    # Simple optimization example
    try:
        func = bmb.get_function("synt_simple_noiseless", "lasso_synthetic")
        print(f"Simple random search on {func.metadata['name']}")
        
        best_x = None
        best_f = float('inf')
        
        # Random search
        for i in range(100):
            X = torch.rand(1, func.dim) * 2 - 1
            f = func(X).item()
            
            if f < best_f:
                best_f = f
                best_x = X.clone()
        
        print(f"Best value found: {best_f:.6f}")
        
        # Check how well we recovered the sparse structure
        if hasattr(func, 'get_test_metrics') and best_x is not None:
            metrics = func.get_test_metrics(best_x)
            print(f"Best solution test MSE: {metrics['mspe']:.6f}")
            print(f"Best solution F-score: {metrics['fscore']:.6f}")
            
    except Exception as e:
        print(f"Error in optimization example: {e}")
    
    print("\n" + "="*40)
    print("Example completed!")


if __name__ == "__main__":
    main() 