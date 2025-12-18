"""
Dimension Discovery Experiment Example

This example demonstrates how to:
1. Create a dimension-expanded benchmark function with dummy dimensions
2. Run an optimizer on the expanded function
3. Analyze the optimizer's ability to identify real (effective) dimensions
   using GP-ARD (Gaussian Process with Automatic Relevance Determination)

Key Method: GP-ARD
- Fits a Gaussian Process with ARD kernel to the optimization trajectory
- The learned length scales indicate dimension relevance:
  - Short length scale → high relevance (function sensitive to this dim)
  - Long length scale → low relevance (function insensitive to this dim)
- Relevance Score = 1 / length_scale
"""

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from typing import Dict, Any, Callable

# Import BOMegaBench components
from bomegabench.functions.synthetic.classical_core import (
    LevyFunction,
    StyblinskiTangFunction,
)
from bomegabench.utils.dimension_expansion import (
    DimensionExpandedFunction,
    DimensionDiscoveryMetrics,
    create_dimension_expansion_test,
)


def random_search_optimizer(
    objective: Callable,
    bounds: Tensor,
    dim: int,
    n_evaluations: int = 200,
    **kwargs
) -> Dict[str, Any]:
    """Simple random search baseline optimizer."""
    lb, ub = bounds[0].numpy(), bounds[1].numpy()
    
    best_x = None
    best_y = float('inf')
    
    for _ in range(n_evaluations):
        x = lb + np.random.rand(dim) * (ub - lb)
        y = objective(x)
        y_val = float(y.item() if hasattr(y, 'item') else y)
        
        if y_val < best_y:
            best_y = y_val
            best_x = x.copy()
    
    return {"x": best_x, "y": best_y}


def bayesian_optimization_optimizer(
    objective: Callable,
    bounds: Tensor,
    dim: int,
    n_evaluations: int = 200,
    n_initial: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Simple Bayesian Optimization using scikit-learn GP.
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        from scipy.optimize import minimize
        from scipy.stats import norm
    except ImportError:
        print("sklearn/scipy not available, falling back to random search")
        return random_search_optimizer(objective, bounds, dim, n_evaluations)
    
    lb, ub = bounds[0].numpy(), bounds[1].numpy()
    
    # Initial random samples
    X_observed = []
    Y_observed = []
    
    for _ in range(n_initial):
        x = lb + np.random.rand(dim) * (ub - lb)
        y = objective(x)
        y_val = float(y.item() if hasattr(y, 'item') else y)
        X_observed.append(x)
        Y_observed.append(y_val)
    
    # Bayesian optimization loop
    for i in range(n_evaluations - n_initial):
        X_arr = np.array(X_observed)
        Y_arr = np.array(Y_observed)
        
        # Fit GP
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        gp.fit(X_arr, Y_arr)
        
        # Expected Improvement acquisition function
        y_best = Y_arr.min()
        
        def neg_ei(x):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            sigma = np.maximum(sigma, 1e-10)
            z = (y_best - mu) / sigma
            ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
            return -ei.item()
        
        # Optimize acquisition function
        best_x_next = None
        best_ei = float('inf')
        
        for _ in range(10):  # Random restarts
            x0 = lb + np.random.rand(dim) * (ub - lb)
            res = minimize(neg_ei, x0, bounds=list(zip(lb, ub)), method='L-BFGS-B')
            if res.fun < best_ei:
                best_ei = res.fun
                best_x_next = res.x
        
        # Evaluate
        y_next = objective(best_x_next)
        y_val = float(y_next.item() if hasattr(y_next, 'item') else y_next)
        X_observed.append(best_x_next)
        Y_observed.append(y_val)
    
    best_idx = np.argmin(Y_observed)
    return {"x": X_observed[best_idx], "y": Y_observed[best_idx]}


def visualize_dimension_importance(
    result,
    metrics: DimensionDiscoveryMetrics,
    title: str = "Dimension Importance Analysis"
):
    """Visualize dimension importance scores and length scales."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Bar chart of relevance scores (1/length_scale)
    ax1 = axes[0]
    n_dims = len(result.importance_scores)
    colors = ['green' if i in metrics.mapping.real_dim_indices else 'red' 
              for i in range(n_dims)]
    
    sorted_idx = np.argsort(result.importance_scores)[::-1]
    sorted_scores = result.importance_scores[sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]
    
    bars = ax1.bar(range(n_dims), sorted_scores, color=sorted_colors, alpha=0.7)
    ax1.set_xlabel('Dimension (sorted by relevance)')
    ax1.set_ylabel('Relevance Score (1/length_scale)')
    ax1.set_title(f'{title}\n(Green=Real, Red=Dummy)')
    ax1.axhline(y=result.importance_scores.mean(), color='blue', linestyle='--', 
                label='Mean relevance')
    ax1.legend()
    
    # Plot 2: Length scales (lower = more important)
    ax2 = axes[1]
    if result.learned_length_scales is not None:
        sorted_by_ls = np.argsort(result.learned_length_scales)
        ls_colors = ['green' if i in metrics.mapping.real_dim_indices else 'red' 
                     for i in sorted_by_ls]
        
        ax2.barh(range(n_dims), result.learned_length_scales[sorted_by_ls], 
                 color=ls_colors, alpha=0.7)
        ax2.set_ylabel('Dimension (sorted by length scale)')
        ax2.set_xlabel('Length Scale (shorter = more important)')
        ax2.set_title('GP-ARD Learned Length Scales')
        ax2.invert_yaxis()  # Shortest at top
    
    plt.tight_layout()
    return fig


def main():
    """Run dimension discovery experiment."""
    print("=" * 70)
    print("Dimension Discovery Experiment (GP-ARD Method)")
    print("=" * 70)
    
    # Configuration
    ORIGINAL_DIM = 5       # Number of real (effective) dimensions
    N_DUMMY_DIMS = 15      # Number of dummy (ineffective) dimensions
    N_EVALUATIONS = 100    # Budget for each optimizer
    SEED = 42
    
    print(f"\nConfiguration:")
    print(f"  - Original dimensions: {ORIGINAL_DIM}")
    print(f"  - Dummy dimensions: {N_DUMMY_DIMS}")
    print(f"  - Total dimensions: {ORIGINAL_DIM + N_DUMMY_DIMS}")
    print(f"  - Evaluation budget: {N_EVALUATIONS}")
    
    # Create dimension-expanded function
    print("\n1. Creating dimension-expanded test function...")
    func, metrics = create_dimension_expansion_test(
        base_function_class=LevyFunction,
        original_dim=ORIGINAL_DIM,
        n_dummy_dims=N_DUMMY_DIMS,
        shuffle=True,
        seed=SEED
    )
    
    print(f"   Function: {func.metadata['name']}")
    print(f"   Real dimension indices (hidden): {func.mapping.real_dim_indices}")
    
    # Test different optimizers
    optimizers = {
        "Random Search": random_search_optimizer,
        "Bayesian Optimization": bayesian_optimization_optimizer,
    }
    
    results = {}
    
    for opt_name, optimizer_fn in optimizers.items():
        print(f"\n2. Running {opt_name}...")
        
        # Track trajectory
        trajectory_X = []
        trajectory_Y = []
        
        def tracked_objective(x):
            if isinstance(x, np.ndarray):
                x_tensor = torch.from_numpy(x).float()
            else:
                x_tensor = x.float() if isinstance(x, torch.Tensor) else torch.tensor(x).float()
            
            if x_tensor.ndim == 1:
                x_tensor = x_tensor.unsqueeze(0)
            
            y = func(x_tensor)
            
            for i in range(len(x_tensor)):
                trajectory_X.append(x_tensor[i].numpy())
                trajectory_Y.append(float(y[i].item() if hasattr(y, '__getitem__') else y.item()))
            
            return y
        
        # Run optimizer
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        
        opt_result = optimizer_fn(
            objective=tracked_objective,
            bounds=func.bounds,
            dim=func.dim,
            n_evaluations=N_EVALUATIONS
        )
        
        # Analyze trajectory using GP-ARD
        analysis = metrics.analyze_trajectory(trajectory_X, trajectory_Y)
        results[opt_name] = {
            "opt_result": opt_result,
            "analysis": analysis,
            "trajectory_X": trajectory_X,
            "trajectory_Y": trajectory_Y
        }
        
        print(f"   Best objective value: {opt_result['y']:.6f}")
        print(f"   Evaluations used: {len(trajectory_Y)}")
        
        # Print analysis report
        print("\n" + metrics.print_analysis_report(analysis))
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("Summary Comparison")
    print("=" * 70)
    print(f"{'Optimizer':<25} {'AUC-ROC':>10} {'Top-K Acc':>10} {'Discovery':>12} {'Best Y':>12}")
    print("-" * 70)
    
    for opt_name, res in results.items():
        analysis = res["analysis"]
        best_y = res["opt_result"]["y"]
        print(f"{opt_name:<25} {analysis.auc_roc:>10.4f} {analysis.precision_at_k:>10.4f} "
              f"{analysis.discovery_score:>12.4f} {best_y:>12.4f}")
    
    # Visualize results
    try:
        print("\n3. Generating visualizations...")
        for opt_name, res in results.items():
            fig = visualize_dimension_importance(
                res["analysis"], 
                metrics, 
                title=f"{opt_name} - Dimension Discovery"
            )
            filename = f"dimension_discovery_{opt_name.lower().replace(' ', '_')}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"   Saved: {filename}")
    except Exception as e:
        print(f"   Visualization failed: {e}")
    
    print("\nExperiment completed!")
    return results


def demo_basic_usage():
    """Demonstrate basic usage of dimension expansion utilities."""
    print("Basic Usage Demo")
    print("=" * 50)
    
    # 1. Create an expanded function
    base_func = LevyFunction(dim=3)  # 3D Levy function
    
    expanded_func = DimensionExpandedFunction(
        base_function=base_func,
        n_dummy_dims=7,          # Add 7 dummy dimensions
        shuffle=True,            # Randomly shuffle all dimensions
        seed=42                  # For reproducibility
    )
    
    print(f"Original dimension: {base_func.dim}")
    print(f"Expanded dimension: {expanded_func.dim}")
    print(f"Real dim indices: {expanded_func.mapping.real_dim_indices}")
    print(f"Dummy dim indices: {expanded_func.mapping.dummy_dim_indices}")
    
    # 2. Evaluate function
    x = torch.rand(5, 10)  # 5 samples in 10D space
    y = expanded_func(x)
    print(f"\nEvaluated 5 random samples, shape: {y.shape}")
    
    # 3. The dummy dimensions don't affect the output
    print("\nVerifying dummy dimensions don't matter:")
    x1 = torch.zeros(1, 10)
    x2 = x1.clone()
    
    # Change only a dummy dimension
    dummy_idx = expanded_func.mapping.dummy_dim_indices[0]
    x2[0, dummy_idx] = 1.0
    
    y1 = expanded_func(x1)
    y2 = expanded_func(x2)
    print(f"  Changed dummy dim {dummy_idx}: y1={y1.item():.6f}, y2={y2.item():.6f}")
    print(f"  Difference: {abs(y1.item() - y2.item()):.10f} (should be ~0)")
    
    # 4. Run GP-ARD analysis
    print("\n4. Running GP-ARD analysis...")
    func, metrics = create_dimension_expansion_test(
        LevyFunction, original_dim=3, n_dummy_dims=7, seed=42
    )
    
    # Generate some trajectory data
    np.random.seed(42)
    X = np.random.rand(50, 10)
    Y = func(torch.from_numpy(X).float()).numpy()
    
    result = metrics.analyze_trajectory(X, Y)
    print(f"  AUC-ROC: {result.auc_roc:.4f}")
    print(f"  Top-K Accuracy: {result.precision_at_k:.4f}")
    print(f"  Learned Length Scales (first 5): {result.learned_length_scales[:5]}")


if __name__ == "__main__":
    # Run basic demo first
    demo_basic_usage()
    print("\n" + "=" * 70 + "\n")
    
    # Run full experiment
    results = main()
