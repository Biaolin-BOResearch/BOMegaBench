"""
Core classes and interfaces for the BO-MegaBench library.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from torch import Tensor
import numpy as np

# Import discrete encoding utilities
try:
    from .utils.discrete_encoding import DiscreteEncoder, DiscreteParameterSpec
    DISCRETE_ENCODING_AVAILABLE = True
except ImportError:
    DiscreteEncoder = None
    DiscreteParameterSpec = None
    DISCRETE_ENCODING_AVAILABLE = False


class BenchmarkFunction(ABC):
    """
    Base class for all benchmark functions in BO-MegaBench.
    
    Inherits from BoTorch's BaseTestProblem to ensure compatibility
    with BoTorch optimizers while providing additional metadata.
    """
    
    def __init__(
        self,
        dim: int,
        bounds: Tensor,
        negate: bool = False,
        noise_std: Optional[float] = None,
        discrete_specs: Optional[List] = None,
        discrete_encoding: str = "onehot",
        **kwargs
    ):
        """
        Initialize benchmark function.

        Args:
            dim: Problem dimension (in problem space)
            bounds: Tensor of shape (2, dim) with lower and upper bounds (in problem space)
            negate: Whether to negate the function (for maximization)
            noise_std: Standard deviation of Gaussian noise to add
            discrete_specs: List of DiscreteParameterSpec for discrete parameters
            discrete_encoding: Encoding mode for discrete parameters: "raw", "interval", or "onehot"
            **kwargs: Additional function-specific parameters
        """
        # Store original problem space dimensions
        self.problem_dim = dim
        self.problem_bounds = bounds

        # Create encoder if discrete parameters exist
        if discrete_specs is not None and len(discrete_specs) > 0:
            if not DISCRETE_ENCODING_AVAILABLE:
                raise ImportError(
                    "Discrete encoding requires bomegabench.utils.discrete_encoding module"
                )
            self.encoder = DiscreteEncoder(dim, discrete_specs, discrete_encoding)
            # Optimizer works in encoded space
            self.dim = self.encoder.optimizer_dim
            self.bounds = self.encoder.get_optimizer_bounds(bounds)
        else:
            self.encoder = None
            self.dim = dim
            self.bounds = bounds

        self.negate = negate
        self.noise_std = noise_std
        self._metadata = self._get_metadata()
        
    @abstractmethod
    def _get_metadata(self) -> Dict[str, Any]:
        """Get function metadata including properties, global minimum, etc."""
        pass
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """Access function metadata."""
        return self._metadata
        
    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        """
        Evaluate the function.

        Args:
            X: Input tensor of shape (..., dim) in optimizer space
            noise: Whether to add noise if noise_std is set

        Returns:
            Function values of shape (...)
        """
        # If encoder exists, convert from optimizer space to problem space
        if self.encoder is not None:
            X_problem = self.encoder.encode(X)
        else:
            X_problem = X

        Y = self._evaluate_true(X_problem)

        if self.negate:
            Y = -Y

        if noise and self.noise_std is not None:
            noise_tensor = torch.randn_like(Y) * self.noise_std
            Y = Y + noise_tensor

        return Y
        
    @abstractmethod  
    def _evaluate_true(self, X: Tensor) -> Tensor:
        """Core function evaluation logic (BoTorch interface)."""
        pass
        
    def _evaluate(self, X: Tensor) -> Tensor:
        """Internal evaluation method."""
        # If encoder exists, convert from optimizer space to problem space
        if self.encoder is not None:
            X_problem = self.encoder.encode(X)
        else:
            X_problem = X
        return self._evaluate_true(X_problem)
        
    def __call__(self, X: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        """
        Convenient call interface supporting both torch and numpy.
        
        Args:
            X: Input of shape (..., dim)
            
        Returns:
            Function values with same type as input
        """
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
            Y_tensor = self.forward(X_tensor)
            return Y_tensor.numpy()
        else:
            return self.forward(X)
            
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds as numpy arrays."""
        bounds_np = self.bounds.numpy()
        return bounds_np[0], bounds_np[1]
        
    def sample_random(self, n_samples: int) -> Tensor:
        """
        Sample random points within bounds (in optimizer space).

        Args:
            n_samples: Number of samples to generate

        Returns:
            Tensor of shape (n_samples, dim) in optimizer space
        """
        lb, ub = self.bounds[0], self.bounds[1]
        return lb + (ub - lb) * torch.rand(n_samples, self.dim)

    def encode_to_problem_space(self, X_optimizer: Tensor) -> Tensor:
        """
        Convert from optimizer space to problem space.

        Args:
            X_optimizer: Tensor in optimizer space

        Returns:
            Tensor in problem space
        """
        if self.encoder is not None:
            return self.encoder.encode(X_optimizer)
        else:
            return X_optimizer

    def decode_from_problem_space(self, X_problem: Tensor) -> Tensor:
        """
        Convert from problem space to optimizer space.

        Useful for initializing optimization with specific problem-space values.

        Args:
            X_problem: Tensor in problem space

        Returns:
            Tensor in optimizer space
        """
        if self.encoder is not None:
            return self.encoder.decode(X_problem)
        else:
            return X_problem

    def get_encoding_info(self) -> Dict[str, Any]:
        """
        Get information about discrete parameter encoding.

        Returns:
            Dictionary with encoding information, or empty dict if no encoding
        """
        if self.encoder is not None:
            return self.encoder.get_info()
        else:
            return {
                'encoding': None,
                'continuous_dims': self.dim,
                'optimizer_dim': self.dim,
                'n_discrete_params': 0
            }


class BenchmarkSuite:
    """
    Container for a collection of benchmark functions.
    """
    
    def __init__(self, name: str, functions: Dict[str, BenchmarkFunction]):
        """
        Initialize benchmark suite.
        
        Args:
            name: Suite name
            functions: Dictionary mapping function names to instances
        """
        self.name = name
        self.functions = functions
        
    def get_function(self, name: str) -> BenchmarkFunction:
        """Get function by name."""
        if name not in self.functions:
            available = list(self.functions.keys())
            raise ValueError(f"Function '{name}' not found. Available: {available}")
        return self.functions[name]
        
    def list_functions(self) -> List[str]:
        """List all available function names."""
        return list(self.functions.keys())
        
    def get_functions_by_property(self, property_name: str, property_value: Any) -> List[str]:
        """Get functions matching a specific property value."""
        matching = []
        for name, func in self.functions.items():
            if func.metadata.get(property_name) == property_value:
                matching.append(name)
        return matching
        
    def __len__(self) -> int:
        """Number of functions in suite."""
        return len(self.functions)
        
    def __iter__(self):
        """Iterate over function names."""
        return iter(self.functions.keys())
        
    def __getitem__(self, name: str) -> BenchmarkFunction:
        """Get function using bracket notation."""
        return self.get_function(name) 