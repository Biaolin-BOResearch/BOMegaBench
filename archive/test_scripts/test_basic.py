#!/usr/bin/env python3
"""
Basic tests to verify BO-MegaBench functionality.
"""

import torch
import numpy as np
import bomegabench as bmb


def test_basic_functionality():
    """Test basic library functionality."""
    print("Testing basic functionality...")
    
    # Test function access
    func = bmb.get_function("sphere")
    assert func is not None
    assert func.dim == 2
    assert hasattr(func, 'metadata')
    
    # Test function evaluation
    x = torch.tensor([[0.0, 0.0]])
    y = func(x)
    assert torch.allclose(y, torch.tensor([0.0]), atol=1e-6)
    
    # Test batch evaluation
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    Y = func(X)
    assert Y.shape == (2,)
    
    # Test numpy interface
    x_np = np.array([[0.0, 0.0]])
    y_np = func(x_np)
    assert isinstance(y_np, np.ndarray)
    
    print("✓ Basic functionality tests passed")


def test_function_suites():
    """Test function suite access."""
    print("Testing function suites...")
    
    # Test suite listing
    suites = bmb.list_suites()
    assert len(suites) > 0
    assert "bbob" in suites
    assert "classical" in suites
    
    # Test function listing
    all_functions = bmb.list_functions()
    assert len(all_functions) > 0
    
    bbob_functions = bmb.list_functions(suite="bbob")
    assert len(bbob_functions) > 0
    
    # Test function summary
    summary = bmb.get_function_summary()
    assert "total" in summary
    assert summary["total"] > 0
    
    print("✓ Function suite tests passed")


def test_function_properties():
    """Test function property queries."""
    print("Testing function properties...")
    
    # Test property queries
    multimodal = bmb.get_multimodal_functions()
    unimodal = bmb.get_unimodal_functions()
    
    assert len(multimodal) > 0
    assert len(unimodal) > 0
    
    # Test specific property query
    separable = bmb.get_functions_by_property("properties", "separable")
    assert len(separable) > 0
    
    print("✓ Function property tests passed")


def test_dimension_scaling():
    """Test dimension scaling."""
    print("Testing dimension scaling...")
    
    # Test different dimensions
    for dim in [2, 5, 10]:
        sphere = bmb.get_function("sphere")(dim=dim)
        assert sphere.dim == dim
        
        # Test evaluation
        x = torch.zeros(1, dim)
        y = sphere(x)
        assert torch.allclose(y, torch.tensor([0.0]), atol=1e-6)
        
        # Test random sampling
        random_points = sphere.sample_random(3)
        assert random_points.shape == (3, dim)
        
        # Check bounds
        lb, ub = sphere.bounds[0], sphere.bounds[1]
        assert len(lb) == dim
        assert len(ub) == dim
    
    print("✓ Dimension scaling tests passed")


def test_benchmark_runner():
    """Test benchmark runner."""
    print("Testing benchmark runner...")
    
    from bomegabench import BenchmarkRunner, simple_random_search
    
    runner = BenchmarkRunner(seed=42)
    
    # Test single run
    result = runner.run_single(
        function_name="sphere",
        algorithm=simple_random_search,
        algorithm_name="random_search",
        n_evaluations=20,
        dim=2
    )
    
    assert result.success
    assert result.n_evaluations <= 20
    assert result.best_value >= 0  # Sphere function minimum is 0
    assert len(result.convergence_history) > 0
    assert len(result.evaluation_history) > 0
    
    # Test multiple runs
    results = runner.run_multiple(
        function_names=["sphere"],
        algorithms={"random_search": simple_random_search},
        n_evaluations=10,
        dim=2,
        show_progress=False
    )
    
    assert len(results) == 1
    assert results[0].success
    
    # Test results dataframe
    df = runner.get_results_dataframe()
    assert len(df) > 0
    assert "function" in df.columns
    assert "algorithm" in df.columns
    assert "best_value" in df.columns
    
    print("✓ Benchmark runner tests passed")


def test_specific_functions():
    """Test specific functions work correctly."""
    print("Testing specific functions...")
    
    # Test sphere function
    sphere = bmb.get_function("sphere")
    x = torch.tensor([[0.0, 0.0]])
    y = sphere(x)
    assert torch.allclose(y, torch.tensor([0.0]), atol=1e-6)
    
    # Test rosenbrock function
    rosenbrock = bmb.get_function("rosenbrock")
    x = torch.tensor([[1.0, 1.0]])  # Global minimum
    y = rosenbrock(x)
    assert torch.allclose(y, torch.tensor([0.0]), atol=1e-6)
    
    # Test ackley function
    ackley = bmb.get_function("ackley")
    x = torch.tensor([[0.0, 0.0]])  # Global minimum
    y = ackley(x)
    assert torch.allclose(y, torch.tensor([0.0]), atol=1e-6)
    
    print("✓ Specific function tests passed")


def main():
    """Run all tests."""
    print("Running BO-MegaBench Tests")
    print("=" * 40)
    
    try:
        test_basic_functionality()
        test_function_suites()
        test_function_properties()
        test_dimension_scaling()
        test_benchmark_runner()
        test_specific_functions()
        
        print("\n" + "=" * 40)
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 