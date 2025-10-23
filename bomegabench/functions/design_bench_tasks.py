"""
Design-Bench Tasks Integration (Non-overlapping Tasks Only).

This module integrates Design-Bench tasks that do NOT overlap with existing
benchmarks in BOMegaBench. We exclude robot control tasks (Ant, DKitty, Hopper)
since MuJoCo provides the standard versions used in BO papers.

Integrated tasks (non-overlapping):
- Superconductor: Material science (critical temperature prediction)
- GFP: Protein design (green fluorescent protein)
- TFBind8/10: DNA sequence optimization (transcription factor binding)
- UTR: RNA sequence optimization (5' untranslated region)
- CIFARNAS: Neural architecture search on CIFAR-10
- NASBench: NAS-Bench-101 architecture search
- ChEMBL: Molecule optimization (with learned oracles)

Excluded tasks (overlap with existing benchmarks):
- AntMorphology: Overlaps with MuJoCo Ant
- DKittyMorphology: Overlaps with MuJoCo (robot morphology)
- HopperController: Overlaps with MuJoCo Hopper

Reference: https://github.com/brandontrabucco/design-bench
Paper: "Design-Bench: Benchmarks for Data-Driven Offline Model-Based Optimization"
"""

import sys
import os
from typing import Dict, List, Optional, Any, Union
import torch
from torch import Tensor
import numpy as np
import warnings

# Add design-bench to path
design_bench_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "design-bench")
if design_bench_path not in sys.path:
    sys.path.insert(0, design_bench_path)

from bomegabench.core import BenchmarkFunction, BenchmarkSuite


class DesignBenchWrapper(BenchmarkFunction):
    """
    Wrapper for Design-Bench tasks (non-overlapping only).

    This wrapper integrates Design-Bench tasks that complement existing
    BOMegaBench benchmarks without duplication.
    """

    def __init__(
        self,
        task_name: str,
        dataset_kwargs: Optional[Dict] = None,
        oracle_kwargs: Optional[Dict] = None,
        normalize_x: bool = True,
        normalize_y: bool = False,
        negate: bool = False,
        noise_std: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize Design-Bench task wrapper.

        Args:
            task_name: Name of the Design-Bench task (e.g., 'Superconductor-RandomForest-v0')
            dataset_kwargs: Additional kwargs for dataset initialization
            oracle_kwargs: Additional kwargs for oracle initialization
            normalize_x: Whether to normalize design values
            normalize_y: Whether to normalize prediction values
            negate: Whether to negate the function (for maximization)
            noise_std: Standard deviation of Gaussian noise
            **kwargs: Additional parameters
        """
        from design_bench import make

        self.task_name = task_name
        self.dataset_kwargs = dataset_kwargs or {}
        self.oracle_kwargs = oracle_kwargs or {}
        self.normalize_x_flag = normalize_x
        self.normalize_y_flag = normalize_y

        # Create Design-Bench task
        try:
            self.task = make(
                task_name,
                dataset_kwargs=self.dataset_kwargs,
                oracle_kwargs=self.oracle_kwargs
            )
        except Exception as e:
            raise ImportError(f"Failed to create Design-Bench task {task_name}: {e}")

        # Apply normalization if requested (only for continuous tasks)
        if not self.task.is_discrete:
            if self.normalize_x_flag and not self.task.is_normalized_x:
                self.task.map_normalize_x()
        if self.normalize_y_flag and not self.task.is_normalized_y:
            self.task.map_normalize_y()

        # Store discrete task information
        self.is_discrete_task = self.task.is_discrete
        if self.is_discrete_task:
            # Get number of classes for each position
            if hasattr(self.task, 'num_classes'):
                num_classes = self.task.num_classes
                if isinstance(num_classes, (list, np.ndarray)):
                    self.num_classes_per_position = num_classes
                else:
                    # Uniform number of classes across all dimensions
                    self.num_classes_per_position = [num_classes] * self.task.input_size
            else:
                # Analyze dataset to determine number of unique values per position
                x_data = self.task.x
                self.num_classes_per_position = []
                for i in range(x_data.shape[1]):
                    unique_vals = np.unique(x_data[:, i])
                    self.num_classes_per_position.append(len(unique_vals))

            # For one-hot encoding, dimension = sum of all classes
            onehot_dim = sum(self.num_classes_per_position)
            # Bounds for one-hot: each dimension is [0, 1]
            lower_bounds = [0.0] * onehot_dim
            upper_bounds = [1.0] * onehot_dim
        else:
            # For continuous tasks
            if self.task.is_normalized_x:
                # Normalized to roughly [-2, 2] standard deviations
                lower_bounds = [-2.0] * self.task.input_size
                upper_bounds = [2.0] * self.task.input_size
            else:
                # Use dataset statistics if available
                x_data = self.task.x
                lower_bounds = np.min(x_data, axis=0).tolist()
                upper_bounds = np.max(x_data, axis=0).tolist()
            onehot_dim = self.task.input_size

        bounds = torch.tensor([lower_bounds, upper_bounds], dtype=torch.float32)

        super().__init__(
            dim=onehot_dim,
            bounds=bounds,
            negate=negate,
            noise_std=noise_std,
            **kwargs
        )

    def _get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the Design-Bench task."""
        metadata = {
            "name": f"DesignBench_{self.task_name}",
            "source": "Design-Bench",
            "type": "discrete" if self.is_discrete_task else "continuous",
            "task_name": self.task_name,
            "dataset_name": self.task.dataset_name,
            "oracle_name": self.task.oracle_name,
            "description": f"Design-Bench {self.task.dataset_name} with {self.task.oracle_name} oracle",
            "reference": "https://github.com/brandontrabucco/design-bench",
            "dataset_size": self.task.dataset_size,
            "input_shape": self.task.input_shape,
            "is_normalized_x": self.task.is_normalized_x,
            "is_normalized_y": self.task.is_normalized_y,
        }

        # Add discrete-specific metadata
        if self.is_discrete_task:
            metadata["encoding"] = "one-hot"
            metadata["num_classes_per_position"] = self.num_classes_per_position
            metadata["original_input_size"] = self.task.input_size
            metadata["onehot_dim"] = sum(self.num_classes_per_position)

        return metadata

    def _onehot_to_integers(self, X_onehot: np.ndarray) -> np.ndarray:
        """
        Convert one-hot encoded input to integer representation.

        Args:
            X_onehot: One-hot encoded array of shape (n_samples, sum(num_classes))

        Returns:
            Integer array of shape (n_samples, num_positions)
        """
        n_samples = X_onehot.shape[0]
        num_positions = len(self.num_classes_per_position)
        X_int = np.zeros((n_samples, num_positions), dtype=np.int32)

        offset = 0
        for pos_idx, num_classes in enumerate(self.num_classes_per_position):
            # Extract one-hot segment for this position
            onehot_segment = X_onehot[:, offset:offset+num_classes]
            # Convert to integer by taking argmax
            X_int[:, pos_idx] = np.argmax(onehot_segment, axis=1)
            offset += num_classes

        return X_int

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate using the Design-Bench oracle.

        Args:
            X: Input tensor of shape (..., dim)
               For discrete tasks: one-hot encoded
               For continuous tasks: continuous values

        Returns:
            Function values of shape (...)
        """
        # Convert to numpy for Design-Bench
        X_np = X.detach().cpu().numpy()

        # Remember original shape
        original_shape = X.shape[:-1]

        # Ensure 2D array (n_samples, dim)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
            single_sample = True
        else:
            X_np = X_np.reshape(-1, X.shape[-1])
            single_sample = False

        # For discrete tasks, convert from one-hot to integers
        if self.is_discrete_task:
            X_np = self._onehot_to_integers(X_np)

        # Predict using oracle
        try:
            Y_np = self.task.predict(X_np)
        except Exception as e:
            raise RuntimeError(f"Design-Bench oracle prediction failed: {e}")

        # Ensure Y is 1D array
        if Y_np.ndim > 1:
            Y_np = Y_np.squeeze()

        # Convert back to torch
        Y = torch.tensor(Y_np, dtype=X.dtype, device=X.device)

        # Reshape to match input shape
        if single_sample:
            Y = Y.squeeze()
        else:
            Y = Y.reshape(original_shape)

        return Y


# Define available Design-Bench tasks (NON-OVERLAPPING ONLY)
DESIGN_BENCH_TASKS = {
    # Continuous optimization - Material Science
    "materials": {
        "Superconductor-RandomForest-v0": "Superconductor critical temperature (Random Forest)",
        "Superconductor-GP-v0": "Superconductor critical temperature (Gaussian Process)",
        "Superconductor-FullyConnected-v0": "Superconductor critical temperature (Neural Network)",
    },

    # Discrete optimization - Protein Design
    "protein": {
        "GFP-RandomForest-v0": "GFP protein design (Random Forest)",
        "GFP-GP-v0": "GFP protein design (Gaussian Process)",
        "GFP-FullyConnected-v0": "GFP protein design (Neural Network)",
        "GFP-LSTM-v0": "GFP protein design (LSTM)",
        "GFP-ResNet-v0": "GFP protein design (ResNet)",
        "GFP-Transformer-v0": "GFP protein design (Transformer)",
    },

    # Discrete optimization - DNA/RNA Sequences
    "sequences": {
        "TFBind8-RandomForest-v0": "8bp transcription factor binding (Random Forest)",
        "TFBind8-GP-v0": "8bp transcription factor binding (Gaussian Process)",
        "TFBind8-FullyConnected-v0": "8bp transcription factor binding (Neural Network)",
        "TFBind8-LSTM-v0": "8bp transcription factor binding (LSTM)",
        "TFBind8-Exact-v0": "8bp transcription factor binding (Exact)",

        "TFBind10-Exact-v0": "10bp transcription factor binding (Exact)",

        "UTR-RandomForest-v0": "5' UTR sequence design (Random Forest)",
        "UTR-GP-v0": "5' UTR sequence design (Gaussian Process)",
        "UTR-FullyConnected-v0": "5' UTR sequence design (Neural Network)",
        "UTR-LSTM-v0": "5' UTR sequence design (LSTM)",
        "UTR-ResNet-v0": "5' UTR sequence design (ResNet)",
        "UTR-Transformer-v0": "5' UTR sequence design (Transformer)",
    },

    # Discrete optimization - Neural Architecture Search
    "nas": {
        "CIFARNAS-Exact-v0": "CIFAR-10 neural architecture search (Exact)",
        "NASBench-Exact-v0": "NAS-Bench-101 architecture search (Exact)",
    },
}


def create_design_bench_suite(categories: Optional[List[str]] = None) -> BenchmarkSuite:
    """
    Create a suite of Design-Bench tasks (non-overlapping only).

    Args:
        categories: List of category names to include. If None, includes all.
                   Categories: 'materials', 'protein', 'sequences', 'nas'

    Returns:
        BenchmarkSuite containing non-overlapping Design-Bench tasks
    """
    functions = {}

    # Determine which categories to include
    if categories is None:
        categories = list(DESIGN_BENCH_TASKS.keys())

    for category in categories:
        if category not in DESIGN_BENCH_TASKS:
            warnings.warn(f"Unknown category '{category}'")
            continue

        for task_name, description in DESIGN_BENCH_TASKS[category].items():
            try:
                func = DesignBenchWrapper(task_name=task_name)
                functions[f"designbench_{task_name.lower().replace('-', '_')}"] = func
            except Exception as e:
                # Skip if task cannot be loaded
                warnings.warn(f"Could not load Design-Bench task {task_name}: {e}")
                continue

    suite = BenchmarkSuite(
        name="DesignBench",
        functions=functions
    )
    suite.description = "Non-overlapping Design-Bench tasks (excludes robot tasks covered by MuJoCo)"
    return suite


# Convenience classes for commonly used tasks
class SuperconductorRFFunction(DesignBenchWrapper):
    """Superconductor critical temperature prediction with Random Forest."""
    def __init__(self, **kwargs):
        super().__init__(task_name="Superconductor-RandomForest-v0", **kwargs)


class GFPTransformerFunction(DesignBenchWrapper):
    """GFP protein design with Transformer oracle."""
    def __init__(self, **kwargs):
        super().__init__(task_name="GFP-Transformer-v0", **kwargs)


class TFBind8ExactFunction(DesignBenchWrapper):
    """8bp transcription factor binding with exact oracle."""
    def __init__(self, **kwargs):
        super().__init__(task_name="TFBind8-Exact-v0", **kwargs)


class UTRTransformerFunction(DesignBenchWrapper):
    """5' UTR sequence design with Transformer oracle."""
    def __init__(self, **kwargs):
        super().__init__(task_name="UTR-Transformer-v0", **kwargs)


__all__ = [
    "DesignBenchWrapper",
    "create_design_bench_suite",
    "SuperconductorRFFunction",
    "GFPTransformerFunction",
    "TFBind8ExactFunction",
    "UTRTransformerFunction",
    "DESIGN_BENCH_TASKS",
]
