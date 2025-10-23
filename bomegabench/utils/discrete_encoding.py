"""
Discrete Parameter Encoding Utilities.

This module provides three encoding methods for discrete parameters:
1. "raw": Direct discrete values (no encoding)
2. "interval": Map [0,1] intervals to discrete options
3. "onehot": One-hot encoding to [0,1]^n where n is the number of options

Author: BOMegaBench Team
Date: 2025-10-22
"""

from typing import List, Tuple, Dict, Union, Optional
import torch
from torch import Tensor
import numpy as np


class DiscreteParameterSpec:
    """
    Specification for a discrete parameter.

    Attributes:
        dim_index: Index of this parameter in the original parameter vector
        n_options: Number of discrete options for this parameter
        options: Optional list of actual option values
    """

    def __init__(
        self,
        dim_index: int,
        n_options: int,
        options: Optional[List] = None
    ):
        self.dim_index = dim_index
        self.n_options = n_options
        self.options = options if options is not None else list(range(n_options))

    def __repr__(self):
        return f"DiscreteParam(dim={self.dim_index}, n_options={self.n_options})"


class DiscreteEncoder:
    """
    Encoder/Decoder for discrete parameters with three encoding modes.

    Encoding modes:
    - "raw": No encoding, optimizer works directly with discrete values
    - "interval": Map [0,1] continuous to discrete via interval partitioning
    - "onehot": One-hot encoding, discrete param with n options becomes n dimensions

    Example:
        >>> # Define discrete parameters
        >>> specs = [
        ...     DiscreteParameterSpec(dim_index=2, n_options=3),  # param 2 has 3 options
        ...     DiscreteParameterSpec(dim_index=5, n_options=5),  # param 5 has 5 options
        ... ]
        >>>
        >>> # Create encoder for interval mode
        >>> encoder = DiscreteEncoder(
        ...     continuous_dims=10,  # original has 10 dims
        ...     discrete_specs=specs,
        ...     encoding="interval"
        ... )
        >>>
        >>> # Optimizer space dimension (same as original for interval/raw)
        >>> print(encoder.optimizer_dim)  # 10
        >>>
        >>> # Encode: optimizer space -> problem space
        >>> X_opt = torch.rand(5, 10)  # 5 samples in optimizer space
        >>> X_prob = encoder.encode(X_opt)  # 5 samples in problem space
        >>>
        >>> # For onehot mode, optimizer dim would be different
        >>> encoder_onehot = DiscreteEncoder(
        ...     continuous_dims=10,
        ...     discrete_specs=specs,
        ...     encoding="onehot"
        ... )
        >>> print(encoder_onehot.optimizer_dim)  # 10 - 2 + 3 + 5 = 16
    """

    def __init__(
        self,
        continuous_dims: int,
        discrete_specs: List[DiscreteParameterSpec],
        encoding: str = "onehot"
    ):
        """
        Initialize discrete encoder.

        Args:
            continuous_dims: Total number of dimensions (continuous + discrete)
            discrete_specs: List of discrete parameter specifications
            encoding: Encoding mode, one of ["raw", "interval", "onehot"]
        """
        if encoding not in ["raw", "interval", "onehot"]:
            raise ValueError(
                f"Invalid encoding '{encoding}'. "
                "Must be one of: 'raw', 'interval', 'onehot'"
            )

        self.continuous_dims = continuous_dims
        self.discrete_specs = sorted(discrete_specs, key=lambda x: x.dim_index)
        self.encoding = encoding

        # Validate discrete specs
        discrete_indices = [spec.dim_index for spec in discrete_specs]
        if len(discrete_indices) != len(set(discrete_indices)):
            raise ValueError("Duplicate discrete parameter indices found")

        if discrete_indices and (min(discrete_indices) < 0 or max(discrete_indices) >= continuous_dims):
            raise ValueError(
                f"Discrete indices must be in [0, {continuous_dims-1}], "
                f"got {discrete_indices}"
            )

        # Calculate optimizer dimension
        self._calculate_dimensions()

    def _calculate_dimensions(self):
        """Calculate dimension in optimizer space."""
        if self.encoding == "onehot":
            # For onehot: each discrete param with n options becomes n dims
            # Total = continuous_dims - n_discrete_params + sum(n_options)
            n_discrete = len(self.discrete_specs)
            n_onehot_dims = sum(spec.n_options for spec in self.discrete_specs)
            self.optimizer_dim = self.continuous_dims - n_discrete + n_onehot_dims

            # Create mapping: optimizer dim -> (original dim, option index)
            self._onehot_mapping = self._build_onehot_mapping()
        else:
            # For raw and interval: same dimension as original
            self.optimizer_dim = self.continuous_dims

    def _build_onehot_mapping(self) -> Dict:
        """
        Build mapping for onehot encoding.

        Returns:
            Dictionary with:
            - 'continuous_map': mapping from optimizer dim to original dim
            - 'discrete_map': mapping from optimizer dim range to (original dim, n_options)
        """
        continuous_map = {}  # optimizer_dim -> original_dim
        discrete_map = {}    # optimizer_dim_start -> (original_dim, n_options)

        discrete_indices = {spec.dim_index for spec in self.discrete_specs}

        opt_dim = 0
        for orig_dim in range(self.continuous_dims):
            if orig_dim not in discrete_indices:
                # Continuous dimension
                continuous_map[opt_dim] = orig_dim
                opt_dim += 1
            else:
                # Discrete dimension -> one-hot
                spec = next(s for s in self.discrete_specs if s.dim_index == orig_dim)
                discrete_map[opt_dim] = (orig_dim, spec.n_options, spec)
                opt_dim += spec.n_options

        return {
            'continuous_map': continuous_map,
            'discrete_map': discrete_map
        }

    def encode(self, X: Tensor) -> Tensor:
        """
        Encode from optimizer space to problem space.

        Args:
            X: Tensor of shape (..., optimizer_dim) in optimizer space

        Returns:
            Tensor of shape (..., continuous_dims) in problem space
        """
        if X.shape[-1] != self.optimizer_dim:
            raise ValueError(
                f"Expected input dimension {self.optimizer_dim}, "
                f"got {X.shape[-1]}"
            )

        if self.encoding == "raw":
            return X  # No transformation needed

        elif self.encoding == "interval":
            return self._encode_interval(X)

        else:  # onehot
            return self._encode_onehot(X)

    def _encode_interval(self, X: Tensor) -> Tensor:
        """
        Encode using interval partitioning.

        For a discrete parameter with n options, divide [0,1] into n equal intervals:
        [0, 1/n), [1/n, 2/n), ..., [(n-1)/n, 1]
        Map each interval to corresponding option index.

        Args:
            X: Tensor of shape (..., continuous_dims) with values in [0,1]

        Returns:
            Tensor of shape (..., continuous_dims) with discrete dims converted to indices
        """
        X_encoded = X.clone()

        for spec in self.discrete_specs:
            dim_idx = spec.dim_index
            n_options = spec.n_options

            # Get values for this dimension
            x_dim = X[..., dim_idx]

            # Clip to [0, 1]
            x_dim = torch.clamp(x_dim, 0.0, 1.0)

            # Map to discrete options: floor(x * n_options)
            # Special case: x == 1.0 should map to n_options - 1
            discrete_values = torch.floor(x_dim * n_options).long()
            discrete_values = torch.clamp(discrete_values, 0, n_options - 1)

            # Convert back to float
            X_encoded[..., dim_idx] = discrete_values.float()

        return X_encoded

    def _encode_onehot(self, X: Tensor) -> Tensor:
        """
        Encode from one-hot representation to discrete indices.

        Args:
            X: Tensor of shape (..., optimizer_dim) with one-hot encoded discrete params

        Returns:
            Tensor of shape (..., continuous_dims) with discrete indices
        """
        batch_shape = X.shape[:-1]
        X_encoded = torch.zeros(*batch_shape, self.continuous_dims, dtype=X.dtype, device=X.device)

        # Copy continuous dimensions
        for opt_dim, orig_dim in self._onehot_mapping['continuous_map'].items():
            X_encoded[..., orig_dim] = X[..., opt_dim]

        # Decode one-hot dimensions
        for opt_start, (orig_dim, n_options, spec) in self._onehot_mapping['discrete_map'].items():
            # Extract one-hot segment
            onehot_segment = X[..., opt_start:opt_start+n_options]

            # Convert to discrete index (argmax)
            discrete_idx = torch.argmax(onehot_segment, dim=-1)

            X_encoded[..., orig_dim] = discrete_idx.float()

        return X_encoded

    def decode(self, X_problem: Tensor) -> Tensor:
        """
        Decode from problem space to optimizer space.

        This is useful for initializing optimization with specific values.

        Args:
            X_problem: Tensor of shape (..., continuous_dims) in problem space

        Returns:
            Tensor of shape (..., optimizer_dim) in optimizer space
        """
        if X_problem.shape[-1] != self.continuous_dims:
            raise ValueError(
                f"Expected input dimension {self.continuous_dims}, "
                f"got {X_problem.shape[-1]}"
            )

        if self.encoding == "raw":
            return X_problem  # No transformation needed

        elif self.encoding == "interval":
            return self._decode_interval(X_problem)

        else:  # onehot
            return self._decode_onehot(X_problem)

    def _decode_interval(self, X_problem: Tensor) -> Tensor:
        """
        Decode discrete indices to interval midpoints.

        Args:
            X_problem: Tensor with discrete dimensions as integer indices

        Returns:
            Tensor with discrete dimensions as continuous [0,1] values
        """
        X_opt = X_problem.clone()

        for spec in self.discrete_specs:
            dim_idx = spec.dim_index
            n_options = spec.n_options

            # Get discrete indices
            discrete_idx = X_problem[..., dim_idx].long()
            discrete_idx = torch.clamp(discrete_idx, 0, n_options - 1)

            # Map to interval midpoint
            interval_midpoint = (discrete_idx.float() + 0.5) / n_options
            X_opt[..., dim_idx] = interval_midpoint

        return X_opt

    def _decode_onehot(self, X_problem: Tensor) -> Tensor:
        """
        Decode discrete indices to one-hot representation.

        Args:
            X_problem: Tensor with discrete dimensions as integer indices

        Returns:
            Tensor with discrete dimensions as one-hot vectors
        """
        batch_shape = X_problem.shape[:-1]
        X_opt = torch.zeros(*batch_shape, self.optimizer_dim, dtype=X_problem.dtype, device=X_problem.device)

        # Copy continuous dimensions
        for opt_dim, orig_dim in self._onehot_mapping['continuous_map'].items():
            X_opt[..., opt_dim] = X_problem[..., orig_dim]

        # Encode discrete dimensions as one-hot
        for opt_start, (orig_dim, n_options, spec) in self._onehot_mapping['discrete_map'].items():
            # Get discrete index
            discrete_idx = X_problem[..., orig_dim].long()
            discrete_idx = torch.clamp(discrete_idx, 0, n_options - 1)

            # Create one-hot encoding
            onehot = torch.zeros(*batch_shape, n_options, dtype=X_problem.dtype, device=X_problem.device)
            onehot.scatter_(-1, discrete_idx.unsqueeze(-1), 1.0)

            X_opt[..., opt_start:opt_start+n_options] = onehot

        return X_opt

    def get_optimizer_bounds(self, problem_bounds: Tensor) -> Tensor:
        """
        Get bounds for optimizer space given problem space bounds.

        Args:
            problem_bounds: Tensor of shape (2, continuous_dims) with [lower, upper] bounds

        Returns:
            Tensor of shape (2, optimizer_dim) with bounds for optimizer space
        """
        if self.encoding == "raw":
            return problem_bounds

        elif self.encoding == "interval":
            # Interval encoding: discrete dims should be in [0, 1]
            opt_bounds = problem_bounds.clone()
            for spec in self.discrete_specs:
                opt_bounds[0, spec.dim_index] = 0.0
                opt_bounds[1, spec.dim_index] = 1.0
            return opt_bounds

        else:  # onehot
            # One-hot encoding: create expanded bounds
            opt_bounds = torch.zeros(2, self.optimizer_dim, dtype=problem_bounds.dtype)

            # Copy continuous bounds
            for opt_dim, orig_dim in self._onehot_mapping['continuous_map'].items():
                opt_bounds[:, opt_dim] = problem_bounds[:, orig_dim]

            # One-hot dimensions: [0, 1] for each
            for opt_start, (orig_dim, n_options, spec) in self._onehot_mapping['discrete_map'].items():
                opt_bounds[0, opt_start:opt_start+n_options] = 0.0
                opt_bounds[1, opt_start:opt_start+n_options] = 1.0

            return opt_bounds

    def get_info(self) -> Dict:
        """Get information about the encoding."""
        return {
            'encoding': self.encoding,
            'continuous_dims': self.continuous_dims,
            'optimizer_dim': self.optimizer_dim,
            'n_discrete_params': len(self.discrete_specs),
            'discrete_specs': [
                {
                    'dim_index': spec.dim_index,
                    'n_options': spec.n_options,
                    'options': spec.options
                }
                for spec in self.discrete_specs
            ]
        }


def create_encoder_for_hpo(
    param_config: Dict,
    encoding: str = "onehot"
) -> DiscreteEncoder:
    """
    Helper function to create encoder for HPO benchmarks.

    Args:
        param_config: Dictionary with parameter configuration
            {
                'continuous_dims': int,
                'integer_params': [(dim, min_val, max_val), ...],
                'categorical_params': [(dim, options_list), ...]
            }
        encoding: Encoding mode

    Returns:
        DiscreteEncoder instance
    """
    continuous_dims = param_config['continuous_dims']
    discrete_specs = []

    # Add integer parameters
    for dim, min_val, max_val in param_config.get('integer_params', []):
        n_options = max_val - min_val + 1
        options = list(range(min_val, max_val + 1))
        discrete_specs.append(DiscreteParameterSpec(dim, n_options, options))

    # Add categorical parameters
    for dim, options in param_config.get('categorical_params', []):
        discrete_specs.append(DiscreteParameterSpec(dim, len(options), options))

    return DiscreteEncoder(continuous_dims, discrete_specs, encoding)
