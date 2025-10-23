#!/usr/bin/env python3
"""
Basic usage examples for BO-MegaBench library.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

import bomegabench as bmb
from bomegabench import BenchmarkRunner, simple_random_search
from bomegabench.visualization import plot_function, plot_convergence


def basic_function_usage():
    """Demonstrate basic function usage."""
    print("=== Basic Function Usage ===")
    
    # Get a function
    func = bmb.get_function("sphere")
    print(f"Function: {func.metadata['name']}")
    print(f"Dimension: {func.dim}")
    print(f"Domain: {func.metadata['domain']}")
    print(f"Properties: {func.metadata['properties']}")
    print(f"Global minimum: {func.metadata['global_min']}")
    print()
    
    # Evaluate at a point
    x = torch.tensor([[0.5, -0.3]])
    y = func(x)
    print(f"f({x.tolist()}) = {y.item():.6f}")
    
    # Evaluate multiple points
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])
    Y = func(X)
    print(f"Batch evaluation:")
    for i, (xi, yi) in enumerate(zip(X, Y)):
        print(f"  f({xi.tolist()}) = {yi.item():.6f}")
    print()
    
    # Sample random points
    random_points = func.sample_random(5)
    random_values = func(random_points)
    print(f"Random evaluations:")
    for i, (xi, yi) in enumerate(zip(random_points, random_values)):
        print(f"  f({xi.tolist()}) = {yi.item():.6f}")
    print()


def explore_function_suites():
    """Explore different function suites."""
    print("=== Function Suites ===")
    
    # List all suites
    suites = bmb.list_suites()
    print(f"Available suites: {suites}")
    print()
    
    # Explore each suite
    for suite_name in suites:
        functions = bmb.list_functions(suite=suite_name)
        print(f"{suite_name}: {len(functions)} functions")
        print(f"  Examples: {functions[:3]}...")
        print()
    
    # Get function summary
    summary = bmb.get_function_summary()
    print("Function summary:")
    for suite, count in summary.items():
        print(f"  {suite}: {count}")
    print()


def function_properties_demo():
    """Demonstrate function property queries."""
    print("=== Function Properties ===")
    
    # Get multimodal functions
    multimodal = bmb.get_multimodal_functions()
    print(f"Multimodal functions ({len(multimodal)}): {list(multimodal.keys())[:5]}...")
    
    # Get unimodal functions  
    unimodal = bmb.get_unimodal_functions()
    print(f"Unimodal functions ({len(unimodal)}): {list(unimodal.keys())[:5]}...")
    
    # Get separable functions
    separable = bmb.get_functions_by_property("properties", "separable")
    print(f"Separable functions ({len(separable)}): {list(separable.keys())[:5]}...")
    print()


def dimension_scaling_demo():
    """Demonstrate dimension scaling."""
    print("=== Dimension Scaling ===")
    
    # Test different dimensions
    for dim in [2, 5, 10, 20]:
        sphere = bmb.get_function("sphere")(dim=dim)
        
        # Evaluate at origin
        x = torch.zeros(1, dim)
        y = sphere(x)
        print(f"Sphere {dim}D: f(0) = {y.item():.6f}")
        
        # Evaluate at random point
        x_rand = sphere.sample_random(1)
        y_rand = sphere(x_rand)
        print(f"  f(random) = {y_rand.item():.6f}")
    print()


def simple_benchmark_demo():
    """Demonstrate simple benchmarking."""
    print("=== Simple Benchmark Demo ===")
    
    # Create runner
    runner = BenchmarkRunner(seed=42)
    
    # Define a simple algorithm
    def gradient_descent(objective, bounds, dim, lr=0.1, n_steps=50, **kwargs):
        """Simple gradient descent with finite differences."""
        # Start at random point
        x = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(dim)
        x.requires_grad_(True)
        
        best_x = x.clone().detach()
        best_y = float('inf')
        
        for step in range(n_steps):
            try:
                # Evaluate function
                y = objective(x.unsqueeze(0))
                y_val = float(y)
                
                if y_val < best_y:
                    best_y = y_val
                    best_x = x.clone().detach()
                
                # Simple finite difference gradient
                if step < n_steps - 1:
                    grad = torch.zeros_like(x)
                    eps = 1e-6
                    
                    for i in range(dim):
                        x_plus = x.clone()
                        x_plus[i] += eps
                        x_minus = x.clone()
                        x_minus[i] -= eps
                        
                        y_plus = objective(x_plus.unsqueeze(0))
                        y_minus = objective(x_minus.unsqueeze(0))
                        
                        grad[i] = (float(y_plus) - float(y_minus)) / (2 * eps)
                    
                    # Update x
                    with torch.no_grad():
                        x -= lr * grad
                        # Project back to bounds
                        x = torch.clamp(x, bounds[0], bounds[1])
                        
            except RuntimeError:
                # Max evaluations reached
                break
        
        return {"x": best_x.tolist(), "fun": best_y}
    
    # Test algorithms
    algorithms = {
        "random_search": simple_random_search,
        "gradient_descent": gradient_descent
    }
    
    # Run benchmark on a few functions
    results = runner.run_multiple(
        function_names=["sphere", "rosenbrock"],
        algorithms=algorithms,
        n_evaluations=50,
        dim=2,
        show_progress=True
    )
    
    # Show results
    print("\nResults:")
    for result in results:
        print(f"{result.algorithm_name} on {result.function_name}:")
        print(f"  Best value: {result.best_value:.6f}")
        print(f"  Evaluations: {result.n_evaluations}")
        print(f"  Runtime: {result.runtime:.3f}s")
        print(f"  Success: {result.success}")
        print()
    
    return results


def visualization_demo(results=None):
    """Demonstrate visualization capabilities."""
    print("=== Visualization Demo ===")
    
    # Plot some 2D functions
    functions_2d = ["sphere", "rosenbrock", "ackley", "beale"]
    
    for func_name in functions_2d[:2]:  # Limit to 2 for demo
        try:
            print(f"Plotting {func_name}...")
            fig = plot_function(func_name, resolution=50)
            plt.show()
            plt.close()
        except Exception as e:
            print(f"  Error plotting {func_name}: {e}")
    
    # Plot convergence if results available
    if results and len(results) > 0:
        print("Plotting convergence...")
        try:
            fig = plot_convergence(results)
            plt.show()
            plt.close()
        except Exception as e:
            print(f"  Error plotting convergence: {e}")


def main():
    """Run all demos."""
    print("BO-MegaBench Library Demo")
    print("=" * 50)
    print()
    
    # Basic usage
    basic_function_usage()
    
    # Explore suites
    explore_function_suites()
    
    # Function properties
    function_properties_demo()
    
    # Dimension scaling
    dimension_scaling_demo()
    
    # Simple benchmark
    results = simple_benchmark_demo()
    
    # Visualization (optional - requires matplotlib)
    try:
        visualization_demo(results)
    except ImportError:
        print("Matplotlib not available, skipping visualization demo")
    
    print("Demo completed!")


if __name__ == "__main__":
    main() 