"""
LassoBench integration for BOMegaBench framework.

This module wraps LassoBench synthetic and real-world benchmarks
into the unified BOMegaBench interface.
"""

import torch
from torch import Tensor
import numpy as np
from typing import Dict, List, Optional, Any

from ..core import BenchmarkFunction, BenchmarkSuite
from ..utils.dependencies import check_dependency, require_dependency

# Check LassoBench availability
LASSO_BENCH_AVAILABLE = check_dependency("lassobench")

# Import if available
if LASSO_BENCH_AVAILABLE:
    from LassoBench.LassoBench import SyntheticBenchmark, RealBenchmark
else:
    SyntheticBenchmark = None
    RealBenchmark = None


class LassoBenchSyntheticFunction(BenchmarkFunction):
    """Wrapper for LassoBench synthetic benchmarks."""
    
    def __init__(self, bench_name: str, noise: bool = False, **kwargs):
        """
        Initialize LassoBench synthetic function.

        Args:
            bench_name: Name of benchmark ('synt_simple', 'synt_medium', 'synt_high', 'synt_hard')
            noise: Whether to add noise to the benchmark
            **kwargs: Additional arguments passed to parent class
        """
        require_dependency("lassobench", "LassoBench synthetic benchmarks")
        
        self.bench_name = bench_name
        self.noise = noise
        
        # Create LassoBench instance
        self.lasso_bench = SyntheticBenchmark(pick_bench=bench_name, noise=noise)
        
        # Get dimension and bounds
        dim = self.lasso_bench.n_features
        bounds = torch.tensor([[-1.0] * dim, [1.0] * dim])
        
        super().__init__(dim=dim, bounds=bounds, **kwargs)
        
    def _get_metadata(self) -> Dict[str, Any]:
        """Get function metadata."""
        # Dimension mapping from LassoBench README
        dim_info = {
            'synt_simple': {'dim': 60, 'active_dims': 3},
            'synt_medium': {'dim': 100, 'active_dims': 5}, 
            'synt_high': {'dim': 300, 'active_dims': 15},
            'synt_hard': {'dim': 1000, 'active_dims': 50}
        }
        
        info = dim_info.get(self.bench_name, {'dim': self.dim, 'active_dims': 'unknown'})
        
        properties = ["high_dimensional", "sparse", "lasso_regression"]
        if self.noise:
            properties.append("noisy")
        else:
            properties.append("noiseless")
            
        return {
            "name": f"LassoBench {self.bench_name}" + (" (noisy)" if self.noise else " (noiseless)"),
            "suite": "LassoBench Synthetic",
            "properties": properties,
            "domain": "[-1,1]^" + str(self.dim),
            "global_min": "Variable (depends on oracle)",
            "description": f"LassoBench synthetic benchmark with {info['active_dims']} active dimensions out of {info['dim']} total",
            "active_dimensions": info['active_dims'],
            "total_dimensions": info['dim']
        }
        
    def _evaluate_true(self, X: Tensor) -> Tensor:
        """Evaluate the function."""
        # Convert torch tensor to numpy
        X_np = X.detach().cpu().numpy()
        
        # Handle batch evaluation
        if X_np.ndim == 1:
            # Single point
            return torch.tensor(self.lasso_bench.evaluate(X_np), dtype=X.dtype, device=X.device)
        else:
            # Batch evaluation
            results = []
            for i in range(X_np.shape[0]):
                result = self.lasso_bench.evaluate(X_np[i])
                results.append(result)
            return torch.tensor(results, dtype=X.dtype, device=X.device)
    
    def get_test_metrics(self, X: Tensor) -> Dict[str, float]:
        """Get test metrics (MSE and F-score) for synthetic benchmarks."""
        X_np = X.detach().cpu().numpy()
        if X_np.ndim > 1:
            X_np = X_np[0]  # Take first point if batch
        
        return self.lasso_bench.test(X_np)
    
    def get_true_weights(self) -> np.ndarray:
        """Get the true regression coefficients."""
        return self.lasso_bench.w_true
    
    def get_active_dimensions(self) -> np.ndarray:
        """Get indices of active (non-zero) dimensions."""
        return np.argwhere(self.lasso_bench.w_true != 0).flatten()


class LassoBenchRealFunction(BenchmarkFunction):
    """Wrapper for LassoBench real-world benchmarks."""
    
    def __init__(self, dataset_name: str, **kwargs):
        """
        Initialize LassoBench real-world function.
        
        Args:
            dataset_name: Name of dataset ('Diabetes', 'Breast_cancer', 'Leukemia', 'DNA', 'RCV1')
            **kwargs: Additional arguments passed to parent class
        """
        if not LASSO_BENCH_AVAILABLE:
            raise ImportError("LassoBench is required but not installed. Install with: pip install git+https://github.com/ksehic/LassoBench.git")
        
        self.dataset_name = dataset_name
        
        # Create LassoBench instance
        self.lasso_bench = RealBenchmark(pick_data=dataset_name)
        
        # Get dimension and bounds
        dim = self.lasso_bench.n_features
        bounds = torch.tensor([[-1.0] * dim, [1.0] * dim])
        
        super().__init__(dim=dim, bounds=bounds, **kwargs)
        
    def _get_metadata(self) -> Dict[str, Any]:
        """Get function metadata."""
        # Dimension info from LassoBench README
        dataset_info = {
            'breast_cancer': {'dim': 10, 'approx_active': 3},
            'diabetes': {'dim': 8, 'approx_active': 5},
            'leukemia': {'dim': 7129, 'approx_active': 22},
            'dna': {'dim': 180, 'approx_active': 43},
            'rcv1': {'dim': 19959, 'approx_active': 75}
        }
        
        info = dataset_info.get(self.dataset_name.lower(), {'dim': self.dim, 'approx_active': 'unknown'})
        
        properties = ["high_dimensional", "real_world", "lasso_regression"]
        if info['dim'] > 1000:
            properties.append("very_high_dimensional")
            
        return {
            "name": f"LassoBench {self.dataset_name}",
            "suite": "LassoBench Real",
            "properties": properties,
            "domain": "[-1,1]^" + str(self.dim),
            "global_min": "Variable (real-world data)",
            "description": f"LassoBench real-world dataset with ~{info['approx_active']} important features out of {info['dim']} total",
            "approx_active_dimensions": info['approx_active'],
            "total_dimensions": info['dim'],
            "dataset": self.dataset_name
        }
        
    def _evaluate_true(self, X: Tensor) -> Tensor:
        """Evaluate the function."""
        # Convert torch tensor to numpy
        X_np = X.detach().cpu().numpy()
        
        # Handle batch evaluation
        if X_np.ndim == 1:
            # Single point
            return torch.tensor(self.lasso_bench.evaluate(X_np), dtype=X.dtype, device=X.device)
        else:
            # Batch evaluation
            results = []
            for i in range(X_np.shape[0]):
                result = self.lasso_bench.evaluate(X_np[i])
                results.append(result)
            return torch.tensor(results, dtype=X.dtype, device=X.device)
    
    def get_test_metrics(self, X: Tensor) -> Dict[str, float]:
        """Get test metrics (MSE) for real-world benchmarks."""
        X_np = X.detach().cpu().numpy()
        if X_np.ndim > 1:
            X_np = X_np[0]  # Take first point if batch
        
        return self.lasso_bench.test(X_np)


# Multi-fidelity functionality temporarily removed for simplification


def create_lasso_bench_synthetic_suite() -> BenchmarkSuite:
    """Create LassoBench synthetic benchmarks suite."""
    if not LASSO_BENCH_AVAILABLE:
        raise ImportError("LassoBench is required but not installed. Install with: pip install git+https://github.com/ksehic/LassoBench.git")
    
    functions = {}
    
    # Synthetic benchmarks (both noiseless and noisy versions)
    synthetic_benchmarks = ['synt_simple', 'synt_medium', 'synt_high', 'synt_hard']
    
    for bench_name in synthetic_benchmarks:
        # Noiseless version
        functions[f"{bench_name}_noiseless"] = LassoBenchSyntheticFunction(bench_name, noise=False)
        # Noisy version  
        functions[f"{bench_name}_noisy"] = LassoBenchSyntheticFunction(bench_name, noise=True)
    
    return BenchmarkSuite("LassoBench Synthetic", functions)


def create_lasso_bench_real_suite() -> BenchmarkSuite:
    """Create LassoBench real-world benchmarks suite."""
    if not LASSO_BENCH_AVAILABLE:
        raise ImportError("LassoBench is required but not installed. Install with: pip install git+https://github.com/ksehic/LassoBench.git")
    
    functions = {}
    
    # Real-world datasets
    datasets = ['Diabetes', 'Breast_cancer', 'Leukemia', 'DNA', 'RCV1']
    
    for dataset in datasets:
        functions[dataset.lower()] = LassoBenchRealFunction(dataset)
    
    return BenchmarkSuite("LassoBench Real", functions)


# Multi-fidelity suite creation removed for simplification


# Create suite instances
def get_lasso_bench_suites():
    """Get all LassoBench suites."""
    if not LASSO_BENCH_AVAILABLE:
        return {}
    
    return {
        "lasso_synthetic": create_lasso_bench_synthetic_suite(),
        "lasso_real": create_lasso_bench_real_suite()
    }


# For backwards compatibility, create individual suites
if LASSO_BENCH_AVAILABLE:
    try:
        LassoBenchSyntheticSuite = create_lasso_bench_synthetic_suite()
        LassoBenchRealSuite = create_lasso_bench_real_suite()
    except Exception as e:
        # If suite creation fails, set to None
        print(f"Warning: LassoBench suite creation failed: {e}")
        LassoBenchSyntheticSuite = None
        LassoBenchRealSuite = None
else:
    LassoBenchSyntheticSuite = None
    LassoBenchRealSuite = None 