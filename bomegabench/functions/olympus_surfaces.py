"""
Olympus Surfaces Integration for BOMegaBench.

This module integrates Olympus surfaces (test functions) that are not duplicated
in the existing BOMegaBench suite, focusing on:
- Categorical variable versions
- Discrete versions
- Special terrain functions (mountains)
- Funnel functions
- Multi-objective optimization functions

Reference: https://github.com/aspuru-guzik-group/olympus
"""

import sys
import os
from typing import Dict, List, Optional, Any, Union
import torch
from torch import Tensor
import numpy as np

# Add olympus to path
olympus_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "olympus", "src")
if olympus_path not in sys.path:
    sys.path.insert(0, olympus_path)

from bomegabench.core import BenchmarkFunction, BenchmarkSuite


class OlympusSurfaceWrapper(BenchmarkFunction):
    """
    Wrapper for Olympus surfaces to match BOMegaBench interface.
    """

    @staticmethod
    def _class_to_file(class_name):
        """Convert class name to file name (e.g., 'AckleyPath' -> 'ackley_path')"""
        file_name = class_name[0].lower()
        for character in class_name[1:]:
            if character.isupper():
                file_name += f"_{character.lower()}"
            else:
                file_name += character
        return file_name

    def __init__(
        self,
        surface_name: str,
        dim: int = 2,
        num_opts: Optional[int] = None,
        negate: bool = False,
        noise_std: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize Olympus surface wrapper.

        Args:
            surface_name: Name of the Olympus surface (e.g., 'Denali', 'CatAckley')
            dim: Problem dimension (for scalable surfaces)
            num_opts: Number of options for categorical surfaces
            negate: Whether to negate the function
            noise_std: Standard deviation of Gaussian noise
            **kwargs: Additional surface-specific parameters
        """
        # Import directly to avoid circular imports - don't import from olympus package
        import importlib
        import sys

        # Directly load the surface wrapper module
        surface_name_lower = self._class_to_file(surface_name)
        module_path = f"olympus.surfaces.surface_{surface_name_lower}.wrapper_{surface_name_lower}"

        try:
            surface_module = importlib.import_module(module_path)
            surface_class = getattr(surface_module, surface_name)

            # Create instance
            if num_opts is not None:
                # Categorical surface
                self.olympus_surface = surface_class(param_dim=dim, num_opts=num_opts)
            elif surface_name in ["Denali", "Everest", "K2", "Kilimanjaro", "Matterhorn", "MontBlanc"]:
                # Fixed dimension surfaces
                self.olympus_surface = surface_class()
            else:
                # Scalable surfaces
                self.olympus_surface = surface_class(param_dim=dim)

        except Exception as e:
            raise ImportError(f"Failed to load Olympus surface {surface_name}: {e}")

        self.surface_name = surface_name

        # Get bounds from Olympus surface parameter space
        param_space = self.olympus_surface.param_space
        lower_bounds = []
        upper_bounds = []

        for param in param_space:
            if param.type == 'continuous':
                lower_bounds.append(param.low)
                upper_bounds.append(param.high)
            elif param.type == 'discrete':
                lower_bounds.append(0)
                upper_bounds.append(len(param.options) - 1)
            elif param.type == 'categorical':
                lower_bounds.append(0)
                upper_bounds.append(len(param.options) - 1)

        bounds = torch.tensor([lower_bounds, upper_bounds], dtype=torch.float32)

        super().__init__(
            dim=dim,
            bounds=bounds,
            negate=negate,
            noise_std=noise_std,
            **kwargs
        )

    def _get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the Olympus surface."""
        return {
            "name": f"Olympus_{self.surface_name}",
            "source": "Olympus",
            "type": "synthetic",
            "surface_name": self.surface_name,
            "description": f"Olympus {self.surface_name} surface",
            "reference": "https://github.com/aspuru-guzik-group/olympus",
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """Evaluate the Olympus surface."""
        # Convert to numpy for Olympus
        X_np = X.detach().cpu().numpy()

        # Remember original shape
        original_shape = X.shape[:-1]  # All dimensions except last

        # Olympus expects 2D array (n_samples, dim)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
            single_sample = True
        else:
            # Flatten batch dimensions
            X_np = X_np.reshape(-1, X.shape[-1])
            single_sample = False

        # Evaluate using Olympus
        results = []
        for x in X_np:
            # Create parameter list for Olympus
            params = x.tolist()

            # Run Olympus surface
            result = self.olympus_surface.run(params)
            # Olympus returns list of objectives, take first one
            results.append(result[0] if isinstance(result, (list, np.ndarray)) else result)

        # Convert back to torch
        Y = torch.tensor(results, dtype=X.dtype, device=X.device)

        # Reshape to match input shape (remove extra dimensions)
        if single_sample:
            Y = Y.squeeze()
        else:
            Y = Y.reshape(original_shape)

        return Y


# Define unique Olympus surfaces to integrate (avoiding duplicates)
OLYMPUS_UNIQUE_SURFACES = {
    # Categorical versions
    "cat_ackley": {"class_name": "CatAckley", "default_dim": 2, "categorical": True},
    "cat_camel": {"class_name": "CatCamel", "default_dim": 2, "categorical": True},
    "cat_dejong": {"class_name": "CatDejong", "default_dim": 2, "categorical": True},
    "cat_michalewicz": {"class_name": "CatMichalewicz", "default_dim": 2, "categorical": True},
    "cat_slope": {"class_name": "CatSlope", "default_dim": 2, "categorical": True},

    # Discrete versions
    "discrete_ackley": {"class_name": "DiscreteAckley", "default_dim": 2, "discrete": True},
    "discrete_double_well": {"class_name": "DiscreteDoubleWell", "default_dim": 2, "discrete": True},
    "discrete_michalewicz": {"class_name": "DiscreteMichalewicz", "default_dim": 2, "discrete": True},

    # Mountain/Terrain functions
    "denali": {"class_name": "Denali", "default_dim": 2, "fixed_dim": True},
    "everest": {"class_name": "Everest", "default_dim": 2, "fixed_dim": True},
    "k2": {"class_name": "K2", "default_dim": 2, "fixed_dim": True},
    "kilimanjaro": {"class_name": "Kilimanjaro", "default_dim": 2, "fixed_dim": True},
    "matterhorn": {"class_name": "Matterhorn", "default_dim": 2, "fixed_dim": True},
    "mont_blanc": {"class_name": "MontBlanc", "default_dim": 2, "fixed_dim": True},

    # Special functions
    "ackley_path": {"class_name": "AckleyPath", "default_dim": 2, "scalable": True},
    "gaussian_mixture": {"class_name": "GaussianMixture", "default_dim": 2, "scalable": True},
    "hyper_ellipsoid": {"class_name": "HyperEllipsoid", "default_dim": 2, "scalable": True},
    "linear_funnel": {"class_name": "LinearFunnel", "default_dim": 2, "scalable": True},
    "narrow_funnel": {"class_name": "NarrowFunnel", "default_dim": 2, "scalable": True},

    # Multi-objective (note: BOMegaBench may need updates to fully support these)
    "mult_fonseca": {"class_name": "MultFonseca", "default_dim": 2, "fixed_dim": True, "multi_objective": True},
    "mult_viennet": {"class_name": "MultViennet", "default_dim": 2, "fixed_dim": True, "multi_objective": True},
    "mult_zdt1": {"class_name": "MultZdt1", "default_dim": 30, "scalable": True, "multi_objective": True},
    "mult_zdt2": {"class_name": "MultZdt2", "default_dim": 30, "scalable": True, "multi_objective": True},
    "mult_zdt3": {"class_name": "MultZdt3", "default_dim": 30, "scalable": True, "multi_objective": True},
}


def create_olympus_surfaces_suite() -> BenchmarkSuite:
    """
    Create a suite of unique Olympus surfaces.

    Returns:
        BenchmarkSuite containing Olympus surfaces
    """
    functions = {}

    for surface_key, surface_info in OLYMPUS_UNIQUE_SURFACES.items():
        class_name = surface_info["class_name"]
        dim = surface_info["default_dim"]

        # Skip multi-objective for now (may need special handling)
        if surface_info.get("multi_objective", False):
            continue

        # Create wrapper
        kwargs = {}
        if surface_info.get("categorical", False):
            kwargs["num_opts"] = 21  # Default number of categorical options

        try:
            func = OlympusSurfaceWrapper(
                surface_name=class_name,
                dim=dim,
                **kwargs
            )
            functions[f"olympus_{surface_key}"] = func
        except Exception as e:
            # Skip if surface cannot be loaded
            print(f"Warning: Could not load Olympus surface {class_name}: {e}")
            continue

    suite = BenchmarkSuite(
        name="OlympusSurfaces",
        functions=functions
    )
    suite.description = "Unique Olympus test surfaces including categorical, discrete, and terrain functions"
    return suite


# Convenience classes for commonly used surfaces
class OlympusDenaliFunction(OlympusSurfaceWrapper):
    """Olympus Denali mountain terrain function."""
    def __init__(self, **kwargs):
        super().__init__(surface_name="Denali", dim=2, **kwargs)


class OlympusEverestFunction(OlympusSurfaceWrapper):
    """Olympus Everest mountain terrain function."""
    def __init__(self, **kwargs):
        super().__init__(surface_name="Everest", dim=2, **kwargs)


class OlympusCatAckleyFunction(OlympusSurfaceWrapper):
    """Olympus categorical Ackley function."""
    def __init__(self, dim: int = 2, num_opts: int = 21, **kwargs):
        super().__init__(surface_name="CatAckley", dim=dim, num_opts=num_opts, **kwargs)


class OlympusDiscreteAckleyFunction(OlympusSurfaceWrapper):
    """Olympus discrete Ackley function."""
    def __init__(self, dim: int = 2, **kwargs):
        super().__init__(surface_name="DiscreteAckley", dim=dim, **kwargs)


class OlympusGaussianMixtureFunction(OlympusSurfaceWrapper):
    """Olympus Gaussian mixture function."""
    def __init__(self, dim: int = 2, **kwargs):
        super().__init__(surface_name="GaussianMixture", dim=dim, **kwargs)


class OlympusLinearFunnelFunction(OlympusSurfaceWrapper):
    """Olympus linear funnel function."""
    def __init__(self, dim: int = 2, **kwargs):
        super().__init__(surface_name="LinearFunnel", dim=dim, **kwargs)


__all__ = [
    "OlympusSurfaceWrapper",
    "create_olympus_surfaces_suite",
    "OlympusDenaliFunction",
    "OlympusEverestFunction",
    "OlympusCatAckleyFunction",
    "OlympusDiscreteAckleyFunction",
    "OlympusGaussianMixtureFunction",
    "OlympusLinearFunnelFunction",
    "OLYMPUS_UNIQUE_SURFACES",
]
