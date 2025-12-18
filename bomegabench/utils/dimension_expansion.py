"""
Dimension Expansion Utilities for Evaluating Optimizer Dimension Discovery.

This module provides tools to:
1. Expand benchmark functions with dummy (ineffective) dimensions
2. Analyze optimization trajectories to measure dimension discovery capability
"""

import torch
from torch import Tensor
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings

from ..core import BenchmarkFunction


@dataclass
class DimensionMapping:
    """Records the mapping between original and expanded dimensions."""
    
    # Total dimension after expansion
    total_dim: int
    
    # Original function's dimension
    original_dim: int
    
    # Number of dummy dimensions added
    n_dummy_dims: int
    
    # Mapping from expanded index to original index (None for dummy dims)
    # expanded_to_original[i] = j means expanded dim i maps to original dim j
    # expanded_to_original[i] = None means expanded dim i is a dummy dimension
    expanded_to_original: List[Optional[int]]
    
    # Indices of real (effective) dimensions in expanded space
    real_dim_indices: List[int]
    
    # Indices of dummy (ineffective) dimensions in expanded space
    dummy_dim_indices: List[int]
    
    # Permutation used for shuffling (None if not shuffled)
    permutation: Optional[List[int]] = None
    
    def is_real_dim(self, expanded_idx: int) -> bool:
        """Check if a dimension in expanded space is real (effective)."""
        return expanded_idx in self.real_dim_indices
    
    def get_original_idx(self, expanded_idx: int) -> Optional[int]:
        """Get original dimension index for an expanded dimension index."""
        return self.expanded_to_original[expanded_idx]
    
    def get_real_mask(self) -> np.ndarray:
        """Get boolean mask indicating which dimensions are real."""
        mask = np.zeros(self.total_dim, dtype=bool)
        mask[self.real_dim_indices] = True
        return mask
    

class DimensionExpandedFunction(BenchmarkFunction):
    """
    A wrapper that expands a benchmark function with dummy (ineffective) dimensions.
    
    This is useful for testing whether an optimizer can identify which dimensions
    actually affect the objective function.
    
    The wrapper:
    1. Normalizes all parameters to [0, 1] range
    2. Adds n_dummy dimensions that don't affect the objective
    3. Optionally shuffles all dimensions to hide which are real vs dummy
    
    Example:
        >>> base_func = LevyFunction(dim=3)  # 3D function
        >>> expanded = DimensionExpandedFunction(
        ...     base_function=base_func,
        ...     n_dummy_dims=7,
        ...     shuffle=True,
        ...     seed=42
        ... )
        >>> # Now we have a 10D function where only 3 dims actually matter
        >>> expanded.dim  # 10
        >>> expanded.mapping.real_dim_indices  # Shuffled indices of real dims
    """
    
    def __init__(
        self,
        base_function: BenchmarkFunction,
        n_dummy_dims: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
        dummy_bounds: Tuple[float, float] = (0.0, 1.0),
        normalize_to_unit: bool = True,
        **kwargs
    ):
        """
        Initialize dimension-expanded function.
        
        Args:
            base_function: The original benchmark function to wrap
            n_dummy_dims: Number of dummy (ineffective) dimensions to add
            shuffle: Whether to shuffle dimensions (default: True)
            seed: Random seed for reproducibility of shuffling
            dummy_bounds: Bounds for dummy dimensions (default: [0, 1])
            normalize_to_unit: Whether to normalize all bounds to [0, 1]
            **kwargs: Additional arguments passed to base class
        """
        self.base_function = base_function
        self.n_dummy_dims = n_dummy_dims
        self.shuffle = shuffle
        self.seed = seed
        self.normalize_to_unit = normalize_to_unit
        self.dummy_bounds = dummy_bounds
        
        # Store original bounds for denormalization
        self.original_bounds = base_function.bounds.clone()
        original_dim = base_function.problem_dim
        total_dim = original_dim + n_dummy_dims
        
        # Create expanded bounds
        if normalize_to_unit:
            # All dimensions in [0, 1]
            expanded_bounds = torch.zeros(2, total_dim)
            expanded_bounds[1, :] = 1.0
        else:
            # Keep original bounds for real dims, use dummy_bounds for dummy dims
            expanded_bounds = torch.zeros(2, total_dim)
            expanded_bounds[0, :original_dim] = self.original_bounds[0]
            expanded_bounds[1, :original_dim] = self.original_bounds[1]
            expanded_bounds[0, original_dim:] = dummy_bounds[0]
            expanded_bounds[1, original_dim:] = dummy_bounds[1]
        
        # Create dimension mapping (before shuffling)
        expanded_to_original = list(range(original_dim)) + [None] * n_dummy_dims
        real_dim_indices = list(range(original_dim))
        dummy_dim_indices = list(range(original_dim, total_dim))
        permutation = None
        
        # Apply shuffling if requested
        if shuffle:
            rng = np.random.RandomState(seed)
            permutation = rng.permutation(total_dim).tolist()
            
            # Shuffle bounds
            expanded_bounds = expanded_bounds[:, permutation]
            
            # Update mapping for shuffled positions
            inv_permutation = [0] * total_dim
            for i, p in enumerate(permutation):
                inv_permutation[p] = i
            
            shuffled_expanded_to_original = [None] * total_dim
            for new_idx, old_idx in enumerate(permutation):
                shuffled_expanded_to_original[new_idx] = expanded_to_original[old_idx]
            
            expanded_to_original = shuffled_expanded_to_original
            real_dim_indices = [i for i, orig in enumerate(expanded_to_original) if orig is not None]
            dummy_dim_indices = [i for i, orig in enumerate(expanded_to_original) if orig is None]
        
        # Store mapping
        self.mapping = DimensionMapping(
            total_dim=total_dim,
            original_dim=original_dim,
            n_dummy_dims=n_dummy_dims,
            expanded_to_original=expanded_to_original,
            real_dim_indices=real_dim_indices,
            dummy_dim_indices=dummy_dim_indices,
            permutation=permutation
        )
        
        # Initialize parent class
        super().__init__(
            dim=total_dim,
            bounds=expanded_bounds,
            negate=base_function.negate,
            noise_std=base_function.noise_std,
            **kwargs
        )
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get function metadata."""
        base_meta = self.base_function.metadata.copy() if hasattr(self.base_function, 'metadata') else {}
        return {
            "name": f"DimensionExpanded_{base_meta.get('name', 'Unknown')}",
            "suite": "Dimension Expansion Test",
            "properties": base_meta.get("properties", []) + ["dimension_expanded"],
            "base_function": base_meta.get("name", "Unknown"),
            "original_dim": self.mapping.original_dim,
            "n_dummy_dims": self.n_dummy_dims,
            "total_dim": self.mapping.total_dim,
            "shuffled": self.shuffle,
            "real_dim_indices": self.mapping.real_dim_indices,
            "dummy_dim_indices": self.mapping.dummy_dim_indices,
        }
    
    def _extract_real_dims(self, X: Tensor) -> Tensor:
        """
        Extract real dimensions from expanded input and denormalize if needed.
        
        Args:
            X: Input tensor of shape (..., total_dim) in expanded space
            
        Returns:
            Tensor of shape (..., original_dim) in original function's space
        """
        # Get the real dimension values in the correct order
        original_dim = self.mapping.original_dim
        X_original = torch.zeros(*X.shape[:-1], original_dim, dtype=X.dtype, device=X.device)
        
        for expanded_idx, original_idx in enumerate(self.mapping.expanded_to_original):
            if original_idx is not None:
                X_original[..., original_idx] = X[..., expanded_idx]
        
        # Denormalize if we normalized to unit cube
        if self.normalize_to_unit:
            lb = self.original_bounds[0]
            ub = self.original_bounds[1]
            X_original = lb + X_original * (ub - lb)
        
        return X_original
    
    def _evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate the function.
        
        Args:
            X: Input tensor of shape (..., total_dim) in expanded space
            
        Returns:
            Function values of shape (...)
        """
        # Extract real dimensions and convert to original space
        X_original = self._extract_real_dims(X)
        
        # Evaluate base function
        return self.base_function._evaluate_true(X_original)
    
    def get_dimension_importance_ground_truth(self) -> np.ndarray:
        """
        Get ground truth importance values for all dimensions.
        
        Returns:
            Array of shape (total_dim,) with 1.0 for real dims, 0.0 for dummy dims
        """
        importance = np.zeros(self.mapping.total_dim)
        importance[self.mapping.real_dim_indices] = 1.0
        return importance


@dataclass
class DimensionDiscoveryResult:
    """Results from GP-ARD dimension discovery analysis."""
    
    # Relevance scores for each dimension (relevance = 1/length_scale)
    # Higher score means dimension is more important
    importance_scores: np.ndarray
    
    # Ground truth mask (1 for real, 0 for dummy)
    ground_truth: np.ndarray
    
    # Classification Metrics
    auc_roc: float  # Area under ROC curve (1.0 = perfect, 0.5 = random)
    precision_at_k: float  # Precision when selecting top-k dims (k = n_real_dims)
    recall_at_k: float  # Recall when selecting top-k dims (same as top_k_accuracy)
    f1_at_k: float  # F1 score at threshold k
    
    # Normalized overall score (0 to 1, higher is better)
    discovery_score: float  # Weighted combination of metrics
    
    # Detailed statistics
    mean_importance_real: float  # Mean relevance of real dimensions
    mean_importance_dummy: float  # Mean relevance of dummy dimensions
    separation_ratio: float  # Ratio of real vs dummy relevance
    
    # Ranking analysis
    real_dim_ranks: List[int]  # Rank of each real dimension (1 = highest relevance)
    best_real_rank: int  # Best rank achieved by any real dimension
    worst_real_rank: int  # Worst rank achieved by any real dimension
    mean_real_rank: float  # Mean rank of real dimensions
    
    # GP-ARD specific
    learned_length_scales: np.ndarray  # ARD length scales from GP
    method_used: str = "gp_ard"  # Always "gp_ard"


class DimensionDiscoveryMetrics:
    """
    Analyzes optimization trajectories to measure an optimizer's ability
    to discover effective (real) dimensions vs ineffective (dummy) dimensions.
    
    Uses Gaussian Process with ARD (Automatic Relevance Determination) kernel
    to learn dimension importance from the optimization trajectory. 
    
    The learned length scales inversely indicate dimension relevance:
    - Short length scale → high relevance (function changes rapidly in this dim)
    - Long length scale → low relevance (function is insensitive to this dim)
    
    Relevance Score = 1 / length_scale
    
    Example:
        >>> func = DimensionExpandedFunction(LevyFunction(dim=3), n_dummy_dims=7)
        >>> # Run your optimizer and collect trajectory
        >>> trajectory_X = [...]  # List of sampled points (in [0,1] space)
        >>> trajectory_Y = [...]  # List of objective values
        >>> metrics = DimensionDiscoveryMetrics(func)
        >>> result = metrics.analyze_trajectory(trajectory_X, trajectory_Y)
        >>> print(f"AUC-ROC: {result.auc_roc:.3f}")
        >>> print(f"Top-K Accuracy: {result.precision_at_k:.3f}")
    """
    
    def __init__(self, expanded_function: DimensionExpandedFunction):
        """
        Initialize metrics analyzer.
        
        Args:
            expanded_function: The dimension-expanded function being optimized
        
        Raises:
            TypeError: If expanded_function is not a DimensionExpandedFunction
            ImportError: If sklearn is not available
        """
        if not isinstance(expanded_function, DimensionExpandedFunction):
            raise TypeError("Expected DimensionExpandedFunction instance")
        
        # Check sklearn availability
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, ConstantKernel
        except ImportError:
            raise ImportError(
                "sklearn is required for GP-ARD dimension discovery. "
                "Install with: pip install scikit-learn"
            )
        
        self.func = expanded_function
        self.mapping = expanded_function.mapping
        self.ground_truth = expanded_function.get_dimension_importance_ground_truth()
    
    def compute_importance_scores(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_restarts: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dimension importance using GP with ARD kernel.
        
        Fits a Gaussian Process with ARD Matern kernel to the optimization
        trajectory. The learned length scales indicate which dimensions 
        the function is sensitive to.
        
        Args:
            X: Array of shape (n_points, total_dim) with sampled points in [0,1]
            Y: Array of shape (n_points,) with objective values
            n_restarts: Number of optimizer restarts for GP fitting
            
        Returns:
            Tuple of (relevance_scores, learned_length_scales)
            - relevance_scores: 1/length_scale, normalized to [0, 1]
            - learned_length_scales: raw length scales from GP
            
        Raises:
            ValueError: If X has fewer than 5 samples
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
        
        X = np.asarray(X)
        Y = np.asarray(Y).flatten()
        
        if len(X) < 5:
            raise ValueError(f"Need at least 5 samples for GP-ARD, got {len(X)}")
        
        D = X.shape[1]
        
        # Create ARD kernel: each dimension has its own length scale
        # Wide bounds (1e-2 to 1e5) allow distinguishing active vs inactive dimensions
        kernel = C(1.0) * Matern(
            length_scale=np.ones(D), 
            length_scale_bounds=(1e-2, 1e5), 
            nu=2.5
        )
        
        gp = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=n_restarts, 
            normalize_y=True,
            random_state=42
        )
        
        # Fit GP to learn length scales from data
        gp.fit(X, Y)
        
        # Extract learned length scales
        # In sklearn, kernel_.k2 is the Matern kernel after fitting
        learned_length_scales = np.atleast_1d(gp.kernel_.k2.length_scale)
        
        # Compute relevance scores: relevance = 1 / length_scale
        # Short length scale → high relevance (function changes rapidly)
        # Long length scale → low relevance (function is insensitive)
        relevance_scores = 1.0 / learned_length_scales
        
        # Normalize to [0, 1]
        if relevance_scores.max() > 0:
            relevance_scores_normalized = relevance_scores / relevance_scores.max()
        else:
            relevance_scores_normalized = np.ones(D) / D
        
        return relevance_scores_normalized, learned_length_scales
    
    def _compute_auc_roc(self, scores: np.ndarray) -> float:
        """Compute Area Under ROC Curve for dimension classification."""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(self.ground_truth, scores)
    
    def analyze_trajectory(
        self,
        trajectory_X: Union[List, np.ndarray],
        trajectory_Y: Union[List, np.ndarray],
        n_restarts: int = 5
    ) -> DimensionDiscoveryResult:
        """
        Analyze an optimization trajectory to evaluate dimension discovery.
        
        Uses GP-ARD to fit a Gaussian Process with ARD kernel to the trajectory,
        then extracts learned length scales to determine dimension importance.
        
        Args:
            trajectory_X: Array of shape (n_points, total_dim) with sampled points
                         Must be in [0, 1] normalized space
            trajectory_Y: Array of shape (n_points,) with objective values
            n_restarts: Number of optimizer restarts for GP fitting
            
        Returns:
            DimensionDiscoveryResult with comprehensive analysis
            
        Raises:
            ValueError: If inputs have mismatched dimensions or fewer than 5 samples
        """
        X = np.asarray(trajectory_X)
        Y = np.asarray(trajectory_Y).flatten()
        
        if len(X) != len(Y):
            raise ValueError(f"X ({len(X)}) and Y ({len(Y)}) must have same length")
        
        if X.shape[1] != self.mapping.total_dim:
            raise ValueError(
                f"X dimension ({X.shape[1]}) doesn't match function "
                f"total_dim ({self.mapping.total_dim})"
            )
        
        # Compute importance scores using GP-ARD
        importance_scores, learned_length_scales = self.compute_importance_scores(
            X, Y, n_restarts=n_restarts
        )
        
        # Get indices sorted by importance (highest first)
        sorted_indices = np.argsort(importance_scores)[::-1]
        
        # Precision@k and Recall@k (k = number of real dimensions)
        k = self.mapping.original_dim
        top_k_indices = set(sorted_indices[:k])
        real_indices = set(self.mapping.real_dim_indices)
        
        true_positives = len(top_k_indices & real_indices)
        precision_at_k = true_positives / k if k > 0 else 0
        recall_at_k = true_positives / len(real_indices) if len(real_indices) > 0 else 0
        f1_at_k = (
            2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
            if (precision_at_k + recall_at_k) > 0 else 0
        )
        
        # AUC-ROC
        auc_roc = self._compute_auc_roc(importance_scores)
        
        # Importance statistics
        real_mask = self.ground_truth == 1
        mean_importance_real = importance_scores[real_mask].mean() if real_mask.any() else 0
        mean_importance_dummy = importance_scores[~real_mask].mean() if (~real_mask).any() else 0
        separation_ratio = (
            mean_importance_real / (mean_importance_dummy + 1e-10)
            if mean_importance_dummy > 0 else float('inf')
        )
        
        # Ranking analysis
        ranks = np.argsort(np.argsort(importance_scores)[::-1]) + 1  # 1-indexed ranks
        real_dim_ranks = [int(ranks[i]) for i in self.mapping.real_dim_indices]
        best_real_rank = min(real_dim_ranks) if real_dim_ranks else 0
        worst_real_rank = max(real_dim_ranks) if real_dim_ranks else 0
        mean_real_rank = np.mean(real_dim_ranks) if real_dim_ranks else 0
        
        # Overall discovery score (weighted combination)
        discovery_score = 0.4 * auc_roc + 0.3 * f1_at_k + 0.3 * min(1.0, separation_ratio / 5.0)
        
        return DimensionDiscoveryResult(
            importance_scores=importance_scores,
            ground_truth=self.ground_truth,
            auc_roc=auc_roc,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_at_k=f1_at_k,
            discovery_score=discovery_score,
            mean_importance_real=mean_importance_real,
            mean_importance_dummy=mean_importance_dummy,
            separation_ratio=separation_ratio,
            learned_length_scales=learned_length_scales,
            method_used="gp_ard",
            real_dim_ranks=real_dim_ranks,
            best_real_rank=best_real_rank,
            worst_real_rank=worst_real_rank,
            mean_real_rank=mean_real_rank
        )
    
    def print_analysis_report(self, result: DimensionDiscoveryResult) -> str:
        """Generate a human-readable analysis report."""
        lines = [
            "=" * 60,
            "Dimension Discovery Analysis Report",
            "=" * 60,
            "",
            f"Method Used: {result.method_used.upper()}",
            f"Total Dimensions: {self.mapping.total_dim}",
            f"  - Real (effective): {self.mapping.original_dim}",
            f"  - Dummy (ineffective): {self.mapping.n_dummy_dims}",
            "",
            "Classification Metrics:",
            f"  - AUC-ROC: {result.auc_roc:.4f}",
            f"  - Precision@{self.mapping.original_dim}: {result.precision_at_k:.4f}",
            f"  - Recall@{self.mapping.original_dim}: {result.recall_at_k:.4f}",
            f"  - F1@{self.mapping.original_dim}: {result.f1_at_k:.4f}",
            "",
            "Importance Scores (Relevance = 1/length_scale):",
            f"  - Mean (real dims): {result.mean_importance_real:.4f}",
            f"  - Mean (dummy dims): {result.mean_importance_dummy:.4f}",
            f"  - Separation ratio: {result.separation_ratio:.4f}",
            "",
            "Ranking Analysis:",
            f"  - Real dimension ranks: {result.real_dim_ranks}",
            f"  - Best real dim rank: {result.best_real_rank}",
            f"  - Worst real dim rank: {result.worst_real_rank}",
            f"  - Mean real dim rank: {result.mean_real_rank:.2f}",
            "",
            f"Overall Discovery Score: {result.discovery_score:.4f}",
            "  (0 = random, 1 = perfect identification)",
        ]
        
        # Show length scales if available (GP-ARD method)
        if result.learned_length_scales is not None:
            lines.append("")
            lines.append("GP-ARD Learned Length Scales:")
            lines.append("  (shorter = more relevant, longer = less relevant)")
            sorted_by_relevance = np.argsort(result.learned_length_scales)
            for i, idx in enumerate(sorted_by_relevance[:5]):
                dim_type = "REAL" if idx in self.mapping.real_dim_indices else "dummy"
                ls = result.learned_length_scales[idx]
                lines.append(f"  {i+1:2d}. Dim {idx:3d} ({dim_type}): length_scale={ls:.4f}")
            lines.append("  ...")
            for i, idx in enumerate(sorted_by_relevance[-3:]):
                dim_type = "REAL" if idx in self.mapping.real_dim_indices else "dummy"
                ls = result.learned_length_scales[idx]
                rank = len(sorted_by_relevance) - 2 + i
                lines.append(f"  {rank:2d}. Dim {idx:3d} ({dim_type}): length_scale={ls:.4f}")
        
        lines.append("")
        lines.append("Dimension Importance Ranking:")
        
        # Show top dimensions
        sorted_indices = np.argsort(result.importance_scores)[::-1]
        for i, idx in enumerate(sorted_indices[:10]):
            dim_type = "REAL" if idx in self.mapping.real_dim_indices else "dummy"
            lines.append(
                f"  {i+1:2d}. Dim {idx:3d} ({dim_type}): relevance={result.importance_scores[idx]:.4f}"
            )
        
        if len(sorted_indices) > 10:
            lines.append(f"  ... and {len(sorted_indices) - 10} more dimensions")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def create_dimension_expansion_test(
    base_function_class,
    original_dim: int,
    n_dummy_dims: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
    **function_kwargs
) -> Tuple[DimensionExpandedFunction, DimensionDiscoveryMetrics]:
    """
    Convenience function to create a dimension expansion test setup.
    
    Args:
        base_function_class: Class of the base benchmark function
        original_dim: Dimension of the base function
        n_dummy_dims: Number of dummy dimensions to add
        shuffle: Whether to shuffle dimensions
        seed: Random seed for reproducibility
        **function_kwargs: Additional arguments for base function
        
    Returns:
        Tuple of (expanded_function, metrics_analyzer)
        
    Example:
        >>> from bomegabench.functions.synthetic.classical_core import LevyFunction
        >>> func, metrics = create_dimension_expansion_test(
        ...     LevyFunction, 
        ...     original_dim=5, 
        ...     n_dummy_dims=15,
        ...     seed=42
        ... )
        >>> print(f"Testing {func.dim}D function ({func.mapping.original_dim} real dims)")
    """
    base_func = base_function_class(dim=original_dim, **function_kwargs)
    expanded_func = DimensionExpandedFunction(
        base_function=base_func,
        n_dummy_dims=n_dummy_dims,
        shuffle=shuffle,
        seed=seed
    )
    metrics = DimensionDiscoveryMetrics(expanded_func)
    
    return expanded_func, metrics


# Batch testing utilities
def run_dimension_discovery_experiment(
    optimizer_fn,
    base_function_class,
    original_dim: int,
    n_dummy_dims: int,
    n_evaluations: int = 200,
    n_runs: int = 5,
    seed_base: int = 0,
    **optimizer_kwargs
) -> Dict[str, Any]:
    """
    Run a complete dimension discovery experiment.
    
    Args:
        optimizer_fn: Optimizer function with signature:
                     optimizer_fn(objective, bounds, dim, **kwargs) -> dict with 'X' and 'Y'
        base_function_class: Class of base benchmark function
        original_dim: Dimension of base function
        n_dummy_dims: Number of dummy dimensions
        n_evaluations: Budget per run
        n_runs: Number of independent runs
        seed_base: Base seed for reproducibility
        **optimizer_kwargs: Additional arguments for optimizer
        
    Returns:
        Dictionary with aggregated results across runs
    """
    all_results = []
    
    for run_idx in range(n_runs):
        seed = seed_base + run_idx
        
        # Create test setup
        func, metrics = create_dimension_expansion_test(
            base_function_class,
            original_dim=original_dim,
            n_dummy_dims=n_dummy_dims,
            seed=seed
        )
        
        # Track trajectory
        trajectory_X = []
        trajectory_Y = []
        
        def tracked_objective(x):
            if isinstance(x, torch.Tensor):
                x_np = x.numpy()
            else:
                x_np = np.asarray(x)
            
            if x_np.ndim == 1:
                x_np = x_np.reshape(1, -1)
            
            y = func(torch.from_numpy(x_np).float())
            y_np = y.numpy() if isinstance(y, torch.Tensor) else y
            
            for i in range(len(x_np)):
                trajectory_X.append(x_np[i])
                trajectory_Y.append(float(y_np[i] if hasattr(y_np, '__getitem__') else y_np))
            
            return y
        
        # Run optimizer
        try:
            optimizer_fn(
                objective=tracked_objective,
                bounds=func.bounds,
                dim=func.dim,
                n_evaluations=n_evaluations,
                **optimizer_kwargs
            )
        except Exception as e:
            warnings.warn(f"Run {run_idx} failed: {e}")
            continue
        
        # Analyze trajectory
        if len(trajectory_X) > 0:
            result = metrics.analyze_trajectory(trajectory_X, trajectory_Y)
            all_results.append(result)
    
    if not all_results:
        return {"error": "All runs failed"}
    
    # Aggregate results
    return {
        "n_runs": len(all_results),
        "original_dim": original_dim,
        "n_dummy_dims": n_dummy_dims,
        "total_dim": original_dim + n_dummy_dims,
        "auc_roc_mean": np.mean([r.auc_roc for r in all_results]),
        "auc_roc_std": np.std([r.auc_roc for r in all_results]),
        "precision_at_k_mean": np.mean([r.precision_at_k for r in all_results]),
        "precision_at_k_std": np.std([r.precision_at_k for r in all_results]),
        "recall_at_k_mean": np.mean([r.recall_at_k for r in all_results]),
        "recall_at_k_std": np.std([r.recall_at_k for r in all_results]),
        "f1_at_k_mean": np.mean([r.f1_at_k for r in all_results]),
        "f1_at_k_std": np.std([r.f1_at_k for r in all_results]),
        "discovery_score_mean": np.mean([r.discovery_score for r in all_results]),
        "discovery_score_std": np.std([r.discovery_score for r in all_results]),
        "separation_ratio_mean": np.mean([r.separation_ratio for r in all_results]),
        "mean_real_rank_mean": np.mean([r.mean_real_rank for r in all_results]),
        "individual_results": all_results
    }

