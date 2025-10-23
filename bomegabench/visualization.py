"""
Visualization utilities for benchmark functions and optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any
import torch

from .core import BenchmarkFunction
from .benchmark import BenchmarkResult
from .functions import get_function


def plot_function(
    function_name: str,
    dim: Optional[int] = None,
    bounds: Optional[Tuple[float, float]] = None,
    resolution: int = 100,
    figsize: Tuple[int, int] = (10, 8),
    **function_kwargs
) -> plt.Figure:
    """
    Plot 2D function landscape.
    
    Args:
        function_name: Name of the function to plot
        dim: Function dimension (for functions that support it)
        bounds: Custom bounds (min, max) for plotting
        resolution: Grid resolution
        figsize: Figure size
        **function_kwargs: Additional function arguments
        
    Returns:
        Matplotlib figure
    """
    # Get function
    if dim is not None:
        func = get_function(function_name)(dim=max(2, dim), **function_kwargs)
    else:
        func = get_function(function_name)(**function_kwargs)
        
    if func.dim != 2:
        raise ValueError("Function plotting only supports 2D functions")
    
    # Determine bounds
    if bounds is not None:
        x_min, x_max = bounds
        y_min, y_max = bounds
    else:
        bounds_tensor = func.bounds
        x_min, y_min = bounds_tensor[0].tolist()
        x_max, y_max = bounds_tensor[1].tolist()
    
    # Create grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate function
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = func(points).reshape(X.shape)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Contour plot
    contour = ax1.contour(X, Y, Z, levels=20, colors='black', alpha=0.6, linewidths=0.5)
    contourf = ax1.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
    ax1.clabel(contour, inline=True, fontsize=8)
    
    # Mark global minimum if known
    metadata = func.metadata
    if 'global_min_location' in metadata and len(metadata['global_min_location']) == 2:
        opt_x, opt_y = metadata['global_min_location']
        if x_min <= opt_x <= x_max and y_min <= opt_y <= y_max:
            ax1.plot(opt_x, opt_y, 'r*', markersize=15, label='Global minimum')
            ax1.legend()
    
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title(f'{metadata.get("name", function_name)} - Contour')
    plt.colorbar(contourf, ax=ax1)
    
    # 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('f(x)')
    ax2.set_title(f'{metadata.get("name", function_name)} - Surface')
    plt.colorbar(surf, ax=ax2, shrink=0.5)
    
    plt.tight_layout()
    return fig


def plot_convergence(
    results: List[BenchmarkResult],
    figsize: Tuple[int, int] = (12, 6),
    log_scale: bool = True,
    show_evaluations: bool = True
) -> plt.Figure:
    """
    Plot convergence curves for optimization results.
    
    Args:
        results: List of benchmark results
        figsize: Figure size
        log_scale: Whether to use log scale for y-axis
        show_evaluations: Whether to show x-axis as evaluations vs iterations
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for result in results:
        if not result.success or not result.convergence_history:
            continue
            
        x_vals = range(len(result.convergence_history)) if not show_evaluations else range(1, len(result.convergence_history) + 1)
        y_vals = result.convergence_history
        
        # Handle infinite or very large values
        y_vals = np.array(y_vals)
        y_vals = np.clip(y_vals, -1e10, 1e10)
        
        label = f"{result.algorithm_name} on {result.function_name}"
        ax.plot(x_vals, y_vals, label=label, linewidth=2)
    
    if log_scale:
        ax.set_yscale('log')
        
    ax.set_xlabel('Function Evaluations' if show_evaluations else 'Iterations')
    ax.set_ylabel('Best Function Value (log scale)' if log_scale else 'Best Function Value')
    ax.set_title('Convergence Comparison')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_comparison(
    results: List[BenchmarkResult],
    metric: str = 'best_value',
    groupby: str = 'algorithm_name',
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot comparison of algorithm performance.
    
    Args:
        results: List of benchmark results
        metric: Metric to compare ('best_value', 'runtime', 'n_evaluations')
        groupby: How to group results ('algorithm_name', 'function_name')
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Prepare data
    data = []
    for result in results:
        if result.success:
            data.append({
                'algorithm': result.algorithm_name,
                'function': result.function_name,
                'best_value': result.best_value,
                'runtime': result.runtime,
                'n_evaluations': result.n_evaluations
            })
    
    if not data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No successful results to plot', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return fig
    
    # Create DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if groupby == 'algorithm_name':
        sns.boxplot(data=df, x='algorithm', y=metric, ax=ax)
        ax.set_xlabel('Algorithm')
        plt.xticks(rotation=45)
    else:
        sns.boxplot(data=df, x='function', y=metric, ax=ax)
        ax.set_xlabel('Function')
        plt.xticks(rotation=45)
    
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison by {groupby.replace("_", " ").title()}')
    
    if metric == 'best_value':
        ax.set_yscale('log')
        ax.set_ylabel('Best Function Value (log scale)')
    
    plt.tight_layout()
    return fig


def plot_optimization_path(
    result: BenchmarkResult,
    function_name: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    max_points: int = 100
) -> plt.Figure:
    """
    Plot optimization path on 2D function landscape.
    
    Args:
        result: Benchmark result with evaluation history
        function_name: Function name (uses result.function_name if None)
        figsize: Figure size
        max_points: Maximum points to show (for readability)
        
    Returns:
        Matplotlib figure
    """
    if not result.evaluation_history:
        raise ValueError("Result must contain evaluation history")
        
    func_name = function_name or result.function_name
    
    # Get function for background
    func = get_function(func_name)()
    if func.dim != 2:
        raise ValueError("Path plotting only supports 2D functions")
    
    # Plot function landscape
    fig = plot_function(func_name, figsize=figsize)
    ax = fig.axes[0]  # Get the contour plot axis
    
    # Extract path points
    path = np.array(result.evaluation_history)
    if len(path) > max_points:
        # Sample points to avoid overcrowding
        indices = np.linspace(0, len(path)-1, max_points, dtype=int)
        path = path[indices]
    
    # Plot path
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], 'r-', alpha=0.7, linewidth=2, label='Optimization path')
        ax.scatter(path[0, 0], path[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(path[-1, 0], path[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
        
        # Add arrows to show direction
        for i in range(0, len(path)-1, max(1, len(path)//20)):
            dx = path[i+1, 0] - path[i, 0]
            dy = path[i+1, 1] - path[i, 1]
            ax.arrow(path[i, 0], path[i, 1], dx, dy, head_width=0.02, head_length=0.02, 
                    fc='red', ec='red', alpha=0.6)
    
    ax.legend()
    ax.set_title(f'Optimization Path - {result.algorithm_name} on {func_name}')
    
    return fig


def plot_function_properties(
    function_names: List[str],
    property_name: str = 'properties',
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot distribution of function properties.
    
    Args:
        function_names: List of function names to analyze
        property_name: Property to analyze
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Collect properties
    property_counts = {}
    
    for func_name in function_names:
        try:
            func = get_function(func_name)()
            properties = func.metadata.get(property_name, [])
            
            if isinstance(properties, list):
                for prop in properties:
                    property_counts[prop] = property_counts.get(prop, 0) + 1
            else:
                property_counts[str(properties)] = property_counts.get(str(properties), 0) + 1
                
        except Exception:
            continue
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if property_counts:
        properties = list(property_counts.keys())
        counts = list(property_counts.values())
        
        bars = ax.bar(properties, counts)
        ax.set_xlabel(property_name.replace('_', ' ').title())
        ax.set_ylabel('Number of Functions')
        ax.set_title(f'Distribution of {property_name.replace("_", " ").title()}')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   str(count), ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, f'No {property_name} data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig 