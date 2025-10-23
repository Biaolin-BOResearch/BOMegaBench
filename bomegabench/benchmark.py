"""
Benchmark runner for comparing optimization algorithms.
"""

import time
import torch
from torch import Tensor
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
import pandas as pd
from tqdm import tqdm

from .core import BenchmarkFunction
from .functions import get_function


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    
    function_name: str
    algorithm_name: str
    best_value: float
    best_point: List[float]
    convergence_history: List[float]
    evaluation_history: List[List[float]]
    n_evaluations: int
    runtime: float
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_regret(self, true_optimum: Optional[float] = None) -> List[float]:
        """Calculate regret over time."""
        if true_optimum is None:
            # Use best found value as reference
            true_optimum = min(self.convergence_history)
            
        return [val - true_optimum for val in self.convergence_history]
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            "function_name": self.function_name,
            "algorithm_name": self.algorithm_name,
            "best_value": self.best_value,
            "best_point": self.best_point,
            "n_evaluations": self.n_evaluations,
            "runtime": self.runtime,
            "success": self.success,
            "error_message": self.error_message,
            "convergence_history": self.convergence_history,
            "evaluation_history": self.evaluation_history,
            **self.metadata
        }


class BenchmarkRunner:
    """
    Runner for executing benchmark experiments.
    
    Supports various optimization algorithms and provides
    unified interface for comparison.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize benchmark runner.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.results: List[BenchmarkResult] = []
        
    def run_single(
        self,
        function_name: str,
        algorithm: Callable,
        algorithm_name: str,
        n_evaluations: int = 100,
        dim: Optional[int] = None,
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        function_kwargs: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Run a single benchmark experiment.
        
        Args:
            function_name: Name of benchmark function
            algorithm: Optimization algorithm callable
            algorithm_name: Name of the algorithm
            n_evaluations: Maximum number of function evaluations
            dim: Problem dimension (if function supports it)
            algorithm_kwargs: Additional arguments for algorithm
            function_kwargs: Additional arguments for function
            
        Returns:
            BenchmarkResult instance
        """
        algorithm_kwargs = algorithm_kwargs or {}
        function_kwargs = function_kwargs or {}
        
        # Get benchmark function
        try:
            if dim is not None:
                func = get_function(function_name)(dim=dim, **function_kwargs)
            else:
                func = get_function(function_name)(**function_kwargs)
        except Exception as e:
            return BenchmarkResult(
                function_name=function_name,
                algorithm_name=algorithm_name,
                best_value=float('inf'),
                best_point=[],
                convergence_history=[],
                evaluation_history=[],
                n_evaluations=0,
                runtime=0.0,
                success=False,
                error_message=f"Function creation failed: {str(e)}"
            )
        
        # Track evaluations
        evaluation_count = 0
        convergence_history = []
        evaluation_history = []
        best_value = float('inf')
        best_point = None
        
        def objective_wrapper(x: Union[Tensor, np.ndarray]) -> Union[float, np.ndarray]:
            """Wrapper to track evaluations."""
            nonlocal evaluation_count, best_value, best_point, convergence_history, evaluation_history
            
            if evaluation_count >= n_evaluations:
                raise RuntimeError("Maximum evaluations reached")
                
            # Ensure input is correct format
            if isinstance(x, np.ndarray) and x.ndim == 1:
                x_eval = x.reshape(1, -1)
            elif isinstance(x, Tensor) and x.ndim == 1:
                x_eval = x.unsqueeze(0)
            else:
                x_eval = x
                
            # Evaluate function
            y = func(x_eval)
            
            # Handle batch evaluation
            if hasattr(y, '__iter__') and not isinstance(y, (int, float)):
                y_vals = y.flatten() if hasattr(y, 'flatten') else list(y)
                x_vals = x_eval.tolist() if hasattr(x_eval, 'tolist') else x_eval
                
                for i, y_val in enumerate(y_vals):
                    if evaluation_count < n_evaluations:
                        evaluation_count += 1
                        evaluation_history.append(x_vals[i] if isinstance(x_vals[i], list) else x_vals[i].tolist())
                        
                        if y_val < best_value:
                            best_value = float(y_val)
                            best_point = x_vals[i] if isinstance(x_vals[i], list) else x_vals[i].tolist()
                            
                        convergence_history.append(best_value)
            else:
                evaluation_count += 1
                y_val = float(y)
                x_val = x_eval.tolist() if hasattr(x_eval, 'tolist') else x_eval.tolist()
                
                evaluation_history.append(x_val[0] if len(x_val) == 1 and isinstance(x_val[0], list) else x_val)
                
                if y_val < best_value:
                    best_value = y_val
                    best_point = x_val[0] if len(x_val) == 1 and isinstance(x_val[0], list) else x_val
                    
                convergence_history.append(best_value)
                
            return y
        
        # Run optimization
        start_time = time.time()
        try:
            result = algorithm(
                objective=objective_wrapper,
                bounds=func.bounds,
                dim=func.dim,
                **algorithm_kwargs
            )
            success = True
            error_message = None
        except Exception as e:
            success = False
            error_message = str(e)
            result = None
            
        runtime = time.time() - start_time
        
        # Create result
        benchmark_result = BenchmarkResult(
            function_name=function_name,
            algorithm_name=algorithm_name,
            best_value=best_value,
            best_point=best_point or [],
            convergence_history=convergence_history,
            evaluation_history=evaluation_history,
            n_evaluations=evaluation_count,
            runtime=runtime,
            success=success,
            error_message=error_message,
            metadata={
                "function_metadata": func.metadata,
                "algorithm_result": result
            }
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
        
    def run_multiple(
        self,
        function_names: List[str],
        algorithms: Dict[str, Callable],
        n_evaluations: int = 100,
        n_runs: int = 1,
        dim: Optional[int] = None,
        algorithm_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        function_kwargs: Optional[Dict[str, Any]] = None,
        show_progress: bool = True
    ) -> List[BenchmarkResult]:
        """
        Run multiple benchmark experiments.
        
        Args:
            function_names: List of function names to test
            algorithms: Dictionary mapping algorithm names to callables
            n_evaluations: Maximum evaluations per run
            n_runs: Number of independent runs per (function, algorithm) pair
            dim: Problem dimension
            algorithm_kwargs: Per-algorithm keyword arguments
            function_kwargs: Function keyword arguments
            show_progress: Whether to show progress bar
            
        Returns:
            List of BenchmarkResult instances
        """
        algorithm_kwargs = algorithm_kwargs or {}
        function_kwargs = function_kwargs or {}
        
        results = []
        total_experiments = len(function_names) * len(algorithms) * n_runs
        
        pbar = tqdm(total=total_experiments, desc="Running benchmarks") if show_progress else None
        
        for func_name in function_names:
            for alg_name, algorithm in algorithms.items():
                for run_idx in range(n_runs):
                    # Add run index to algorithm name if multiple runs
                    run_alg_name = f"{alg_name}_run_{run_idx}" if n_runs > 1 else alg_name
                    
                    result = self.run_single(
                        function_name=func_name,
                        algorithm=algorithm,
                        algorithm_name=run_alg_name,
                        n_evaluations=n_evaluations,
                        dim=dim,
                        algorithm_kwargs=algorithm_kwargs.get(alg_name, {}),
                        function_kwargs=function_kwargs
                    )
                    
                    results.append(result)
                    
                    if pbar:
                        pbar.update(1)
                        
        if pbar:
            pbar.close()
            
        return results
        
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis."""
        data = []
        for result in self.results:
            data.append({
                "function": result.function_name,
                "algorithm": result.algorithm_name,
                "best_value": result.best_value,
                "n_evaluations": result.n_evaluations,
                "runtime": result.runtime,
                "success": result.success,
                "final_regret": result.convergence_history[-1] if result.convergence_history else float('inf')
            })
            
        return pd.DataFrame(data)
        
    def clear_results(self) -> None:
        """Clear all stored results."""
        self.results.clear()


def simple_random_search(
    objective: Callable,
    bounds: Tensor,
    dim: int,
    n_evaluations: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    Simple random search algorithm for testing.
    
    Args:
        objective: Objective function to minimize
        bounds: Parameter bounds
        dim: Problem dimension
        n_evaluations: Maximum evaluations (managed by wrapper)
        
    Returns:
        Dictionary with optimization result
    """
    best_x = None
    best_y = float('inf')
    
    try:
        for _ in range(n_evaluations):
            # Sample random point
            lb, ub = bounds[0], bounds[1]
            x = lb + (ub - lb) * torch.rand(dim)
            
            # Evaluate
            y = objective(x)
            y_val = float(y) if hasattr(y, '__float__') else float(y[0])
            
            if y_val < best_y:
                best_y = y_val
                best_x = x.clone()
                
    except RuntimeError:
        # Max evaluations reached
        pass
        
    return {
        "x": best_x.tolist() if best_x is not None else [],
        "fun": best_y,
        "success": best_x is not None
    } 