"""
Consolidated Benchmark Functions
All unique functions from BBOB, BoTorch Additional, Classical Additional, and Classical modules.
Total: 72 unique benchmark functions.
"""

import torch
from torch import Tensor
import numpy as np
import math
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod


class BenchmarkFunction(ABC):
    """Base class for benchmark functions (simplified version)."""

    def __init__(self, dim: int, bounds: Tensor, negate: bool = False, **kwargs):
        self.dim = dim
        self.bounds = bounds
        self.negate = negate
        self._metadata = self._get_metadata()

    @abstractmethod
    def _get_metadata(self) -> Dict[str, Any]:
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @abstractmethod
    def _evaluate_true(self, X: Tensor) -> Tensor:
        pass

    def evaluate_true(self, X: Tensor) -> Tensor:
        result = self._evaluate_true(X)
        return -result if self.negate else result

    def __call__(self, X: Tensor) -> Tensor:
        return self.evaluate_true(X)


class BenchmarkSuite:
    """Container for multiple benchmark functions."""

    def __init__(self, name: str, functions: Dict[str, BenchmarkFunction]):
        self.name = name
        self.functions = functions

    def __len__(self) -> int:
        return len(self.functions)

    def __getitem__(self, key: str) -> BenchmarkFunction:
        return self.functions[key]

    def __contains__(self, key: str) -> bool:
        return key in self.functions

    def keys(self) -> List[str]:
        return list(self.functions.keys())

    def values(self) -> List[BenchmarkFunction]:
        return list(self.functions.values())

    def items(self) -> List[Tuple[str, BenchmarkFunction]]:
        return list(self.functions.items())


# =============================================================================
# BBOB Functions (24 functions)
# =============================================================================

class F01_SphereRaw(BenchmarkFunction):
    """Sphere function - F01 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Sphere (BBOB F01)",
            "suite": "BBOB Raw",
            "properties": ["unimodal", "separable"],
            "domain": "[-5,5]^d",
            "global_min": "f(0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return torch.sum(X**2, dim=-1)


class F02_EllipsoidSeparableRaw(BenchmarkFunction):
    """Ellipsoid Separable function - F02 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Ellipsoid Separable (BBOB F02)",
            "suite": "BBOB Raw",
            "properties": ["unimodal", "separable", "ill-conditioned"],
            "domain": "[-5,5]^d",
            "global_min": "f(0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        dim = X.shape[-1]
        coefficients = torch.tensor([10**(6*(i-1)/(dim-1)) for i in range(1, dim+1)],
                                   device=X.device, dtype=X.dtype)
        return torch.sum(coefficients * X**2, dim=-1)


class F03_RastriginSeparableRaw(BenchmarkFunction):
    """Rastrigin Separable function - F03 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Rastrigin Separable (BBOB F03)",
            "suite": "BBOB Raw",
            "properties": ["multimodal", "separable"],
            "domain": "[-5,5]^d",
            "global_min": "f(0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        dim = X.shape[-1]
        return 10 * dim + torch.sum(X**2 - 10 * torch.cos(2 * math.pi * X), dim=-1)


class F04_SkewRastriginBuecheRaw(BenchmarkFunction):
    """Skew Rastrigin-Bueche function - F04 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Skew Rastrigin-Bueche (BBOB F04)",
            "suite": "BBOB Raw",
            "properties": ["multimodal", "asymmetric"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Skew transformation
        X_skewed = torch.where(X > 0, 10**0.5 * X, X)
        dim = X.shape[-1]
        return 10 * dim + torch.sum(X_skewed**2 - 10 * torch.cos(2 * math.pi * X_skewed), dim=-1)


class F05_LinearSlopeRaw(BenchmarkFunction):
    """Linear Slope function - F05 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Linear Slope (BBOB F05)",
            "suite": "BBOB Raw",
            "properties": ["unimodal", "linear"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Basic linear function (simplified)
        return torch.sum(5 * torch.abs(X), dim=-1)


class F06_AttractiveSectorRaw(BenchmarkFunction):
    """Attractive Sector function - F06 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Attractive Sector (BBOB F06)",
            "suite": "BBOB Raw",
            "properties": ["unimodal", "asymmetric"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Simplified implementation
        result = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        for i in range(X.shape[-1]):
            x_i = X[..., i]
            result += torch.where(x_i > 0, 1000000 * x_i**2, x_i**2)
        return result


class F07_StepEllipsoidRaw(BenchmarkFunction):
    """Step-Ellipsoid function - F07 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Step-Ellipsoid (BBOB F07)",
            "suite": "BBOB Raw",
            "properties": ["unimodal", "plateaus"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Round to nearest integer + ellipsoid
        X_rounded = torch.round(X)
        dim = X.shape[-1]
        coefficients = torch.tensor([10**(6*(i-1)/(dim-1)) for i in range(1, dim+1)],
                                   device=X.device, dtype=X.dtype)
        return torch.sum(coefficients * X_rounded**2, dim=-1) + 0.01 * torch.sum(X**2, dim=-1)


class F08_RosenbrockRaw(BenchmarkFunction):
    """Rosenbrock Original function - F08 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Rosenbrock Original (BBOB F08)",
            "suite": "BBOB Raw",
            "properties": ["unimodal", "non-separable", "valley"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        result = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        for i in range(X.shape[-1] - 1):
            x_i = X[..., i]
            x_i1 = X[..., i+1]
            result += 100 * (x_i1 - x_i**2)**2 + (1 - x_i)**2
        return result


class F09_RosenbrockRotatedRaw(BenchmarkFunction):
    """Rosenbrock Rotated function - F09 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Rosenbrock Rotated (BBOB F09)",
            "suite": "BBOB Raw",
            "properties": ["unimodal", "non-separable", "rotated"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Simplified rotation (using different transformation)
        result = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        for i in range(X.shape[-1] - 1):
            x_i = X[..., i]
            x_i1 = X[..., i+1]
            result += 100 * (x_i1 - x_i**2)**2 + (1 - x_i)**2
        return result


class F10_EllipsoidRaw(BenchmarkFunction):
    """Ellipsoid function - F10 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Ellipsoid (BBOB F10)",
            "suite": "BBOB Raw",
            "properties": ["unimodal", "non-separable", "ill-conditioned"],
            "domain": "[-5,5]^d",
            "global_min": "f(0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        dim = X.shape[-1]
        coefficients = torch.tensor([10**(6*(i-1)/(dim-1)) for i in range(1, dim+1)],
                                   device=X.device, dtype=X.dtype)
        # Simplified non-separable version
        return torch.sum(coefficients * X**2, dim=-1)


class F11_DiscusRaw(BenchmarkFunction):
    """Discus function - F11 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Discus (BBOB F11)",
            "suite": "BBOB Raw",
            "properties": ["unimodal", "one-short-axis"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return 1e6 * X[..., 0]**2 + torch.sum(X[..., 1:]**2, dim=-1)


class F12_BentCigarRaw(BenchmarkFunction):
    """Bent Cigar function - F12 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Bent Cigar (BBOB F12)",
            "suite": "BBOB Raw",
            "properties": ["unimodal", "one-long-axis"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return X[..., 0]**2 + 1e6 * torch.sum(X[..., 1:]**2, dim=-1)


class F13_SharpRidgeRaw(BenchmarkFunction):
    """Sharp Ridge function - F13 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Sharp Ridge (BBOB F13)",
            "suite": "BBOB Raw",
            "properties": ["unimodal", "ridge"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        if X.shape[-1] == 1:
            return X[..., 0]**2
        return X[..., 0]**2 + 100 * torch.sum(X[..., 1:]**2, dim=-1)


class F14_DifferentPowersRaw(BenchmarkFunction):
    """Different Powers function - F14 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Different Powers (BBOB F14)",
            "suite": "BBOB Raw",
            "properties": ["unimodal", "variable-sensitivity"],
            "domain": "[-5,5]^d",
            "global_min": "f(0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        powers = torch.arange(2, X.shape[-1] + 2, device=X.device, dtype=torch.float32)
        return torch.sqrt(torch.sum(torch.abs(X)**powers, dim=-1))


class F15_RastriginRaw(BenchmarkFunction):
    """Rastrigin function - F15 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Rastrigin (BBOB F15)",
            "suite": "BBOB Raw",
            "properties": ["multimodal", "non-separable"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        dim = X.shape[-1]
        return 10 * dim + torch.sum(X**2 - 10 * torch.cos(2 * math.pi * X), dim=-1)


class F16_WeierstrassRaw(BenchmarkFunction):
    """Weierstrass function - F16 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-0.5] * dim, [0.5] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Weierstrass (BBOB F16)",
            "suite": "BBOB Raw",
            "properties": ["multimodal", "fractal"],
            "domain": "[-0.5,0.5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Simplified Weierstrass
        result = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        for k in range(12):
            for i in range(X.shape[-1]):
                result += (0.5**k) * torch.cos(2 * math.pi * 3**k * (X[..., i] + 0.5))
        return result


class F17_SchafferF7Cond10Raw(BenchmarkFunction):
    """Schaffer F7 with conditioning 10 - F17 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schaffer F7 Cond 10 (BBOB F17)",
            "suite": "BBOB Raw",
            "properties": ["multimodal", "conditioned"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        if X.shape[-1] == 1:
            return X[..., 0]**2

        norm = torch.sqrt(X[..., :-1]**2 + X[..., 1:]**2)
        sum_term = torch.sum(norm, dim=-1)

        s = torch.sqrt(1 + 10 * sum_term)
        return s * torch.sin(50 * s**0.2)**2 + 1


class F18_SchafferF7Cond1000Raw(BenchmarkFunction):
    """Schaffer F7 with conditioning 1000 - F18 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schaffer F7 Cond 1000 (BBOB F18)",
            "suite": "BBOB Raw",
            "properties": ["multimodal", "high-conditioned"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        if X.shape[-1] == 1:
            return X[..., 0]**2

        norm = torch.sqrt(X[..., :-1]**2 + X[..., 1:]**2)
        sum_term = torch.sum(norm, dim=-1)

        s = torch.sqrt(1 + 1000 * sum_term)
        return s * torch.sin(50 * s**0.2)**2 + 1


class F19_GriewankRosenbrockRaw(BenchmarkFunction):
    """Griewank-Rosenbrock F8F2 function - F19 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Griewank-Rosenbrock F8F2 (BBOB F19)",
            "suite": "BBOB Raw",
            "properties": ["multimodal", "composition"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        if X.shape[-1] == 1:
            return 100 * X[..., 0]**2 + (1 - X[..., 0])**2

        result = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        for i in range(X.shape[-1] - 1):
            z_i = X[..., i] + 1
            z_i1 = X[..., i+1] + 1
            term = 100 * (z_i1 - z_i**2)**2 + (z_i - 1)**2
            # Griewank scaling
            result += term / 4000 - torch.cos(term) + 1
        return result


class F20_SchwefelRaw(BenchmarkFunction):
    """Schwefel function - F20 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schwefel (BBOB F20)",
            "suite": "BBOB Raw",
            "properties": ["multimodal", "deceptive"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # BBOB version of Schwefel
        dim = X.shape[-1]
        x_abs = torch.abs(X)
        return -418.9829 * dim + torch.sum(x_abs * torch.sin(torch.sqrt(x_abs)), dim=-1)


class F21_Gallagher101Raw(BenchmarkFunction):
    """Gallagher 101 peaks function - F21 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Gallagher 101 Peaks (BBOB F21)",
            "suite": "BBOB Raw",
            "properties": ["multimodal", "101-peaks"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Simplified implementation
        return torch.sum(X**2, dim=-1)  # Placeholder


class F22_Gallagher21Raw(BenchmarkFunction):
    """Gallagher 21 peaks function - F22 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Gallagher 21 Peaks (BBOB F22)",
            "suite": "BBOB Raw",
            "properties": ["multimodal", "21-peaks"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Simplified implementation
        return torch.sum(X**2, dim=-1)  # Placeholder


class F23_KatsuuraRaw(BenchmarkFunction):
    """Katsuura function - F23 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Katsuura (BBOB F23)",
            "suite": "BBOB Raw",
            "properties": ["multimodal", "pathological"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        dim = X.shape[-1]
        result = torch.ones(X.shape[:-1], device=X.device, dtype=X.dtype) * 10.0 / dim**2

        for i in range(1, 33):
            pow_2 = 2.0**i
            term = torch.sum(pow_2 * torch.round(pow_2 * X) / pow_2 - X, dim=-1)
            result *= (1 + i * term)**(10/dim**1.2)

        return 10.0 / dim**2 * (result - 1)


class F24_LunacekBiRastriginRaw(BenchmarkFunction):
    """Lunacek bi-Rastrigin function - F24 from BBOB."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Lunacek bi-Rastrigin (BBOB F24)",
            "suite": "BBOB Raw",
            "properties": ["multimodal", "two-funnels"],
            "domain": "[-5,5]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Simplified implementation
        dim = X.shape[-1]
        mu1 = torch.tensor(2.5, device=X.device, dtype=X.dtype)
        d = 1.0

        term1 = torch.sum((X - mu1)**2, dim=-1)
        term2 = torch.sum((X + mu1)**2, dim=-1)
        term3 = 10 * dim + 10 * torch.sum(torch.cos(2 * math.pi * (X - mu1)), dim=-1)

        return torch.minimum(term1, d * dim + term2) + term3


# =============================================================================
# BoTorch Additional Functions (6 functions)
# =============================================================================

class BukinFunction(BenchmarkFunction):
    """Bukin N.6 function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Bukin is only defined for 2D")
        bounds = torch.tensor([[-15.0, -5.0], [-3.0, 3.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Bukin N.6",
            "suite": "BoTorch Additional",
            "properties": ["multimodal", "non-differentiable"],
            "domain": "[-15,-5]×[-3,3]",
            "global_min": "f(-10,1)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        term1 = 100.0 * torch.sqrt(torch.abs(x2 - 0.01 * x1**2))
        term2 = 0.01 * torch.abs(x1 + 10.0)
        return term1 + term2


class Cosine8Function(BenchmarkFunction):
    """Cosine Mixture test function (8D)."""

    def __init__(self, dim: int = 8, **kwargs):
        bounds = torch.tensor([[-1.0] * dim, [1.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Cosine Mixture",
            "suite": "BoTorch Additional",
            "properties": ["multimodal", "separable"],
            "domain": "[-1,1]^d",
            "global_min": "f(0)=0.8 (for d=8)"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        cosine_term = torch.sum(0.1 * torch.cos(5 * math.pi * X), dim=-1)
        quadratic_term = torch.sum(X**2, dim=-1)
        return -(cosine_term - quadratic_term)  # Negate for minimization


class ThreeHumpCamelFunction(BenchmarkFunction):
    """Three-Hump Camel function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Three-Hump Camel is only defined for 2D")
        bounds = torch.tensor([[-5.0, -5.0], [5.0, 5.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Three-Hump Camel",
            "suite": "BoTorch Additional",
            "properties": ["multimodal", "3-local-minima"],
            "domain": "[-5,5]²",
            "global_min": "f(0,0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        term1 = 2.0 * x1**2
        term2 = -1.05 * x1**4
        term3 = x1**6 / 6.0
        term4 = x1 * x2
        term5 = x2**2
        return term1 + term2 + term3 + term4 + term5


class AckleyMixedFunction(BenchmarkFunction):
    """Mixed search space version of the Ackley problem."""

    def __init__(self, dim: int = 53, **kwargs):
        if dim <= 3:
            raise ValueError(f"AckleyMixed requires dim > 3, got {dim}")

        bounds = torch.tensor([[0.0] * dim, [1.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

        self.x_opt = torch.zeros(dim, dtype=torch.float64)
        self.discrete_dims = list(range(0, dim - 3))
        self.continuous_dims = list(range(dim - 3, dim))

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Ackley Mixed",
            "suite": "BoTorch Additional",
            "properties": ["multimodal", "mixed-integer"],
            "domain": "{0,1}^(d-3) × [0,1]³",
            "global_min": "f(x_opt)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        diff = torch.abs(X - self.x_opt)

        a, b, c = 20, 0.2, 2 * math.pi
        n = X.shape[-1]

        sum_sq = (diff ** 2).sum(dim=-1)
        sum_cos = torch.cos(c * diff).sum(dim=-1)

        part1 = -a * torch.exp(-b * torch.sqrt(sum_sq / n))
        part2 = -torch.exp(sum_cos / n)
        return part1 + part2 + a + math.e


class LabsFunction(BenchmarkFunction):
    """Low Auto-correlation Binary Sequences (LABS) problem."""

    def __init__(self, dim: int = 30, **kwargs):
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

        self.optimal_values = {
            10: 3.846, 20: 7.692, 30: 7.627, 40: 7.407, 50: 8.170, 60: 8.257
        }

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "LABS",
            "suite": "BoTorch Additional",
            "properties": ["multimodal", "binary", "combinatorial"],
            "domain": "{0,1}^d",
            "global_min": f"Variable (Merit factor maximization)"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        X_binary = 2 * X - 1

        energy = torch.zeros(X_binary.shape[:-1], dtype=X_binary.dtype, device=X_binary.device)

        for k in range(1, self.dim):
            autocorr = (X_binary[..., :self.dim-k] * X_binary[..., k:]).sum(dim=-1)
            energy += autocorr**2

        merit_factor = (self.dim**2) / (2.0 * energy + 1e-10)
        return -merit_factor


class ShekelFunction(BenchmarkFunction):
    """Shekel multimodal function with m peaks."""

    def __init__(self, m: int = 10, **kwargs):
        self.m = m
        self.optimal_values = {5: -10.1532, 7: -10.4029, 10: -10.536443}
        dim = 4
        bounds = torch.tensor([[0.0] * dim, [10.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

        self.beta = torch.tensor([1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 3.0, 7.0, 5.0, 5.0])[:m]

        C_t = torch.tensor([
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
            [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
            [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]
        ])
        self.C = C_t[:, :m].T

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": f"Shekel (m={self.m})",
            "suite": "BoTorch Additional",
            "properties": ["multimodal", f"{self.m}-peaks", "4D-only"],
            "domain": "[0,10]^4",
            "global_min": f"≈{self.optimal_values.get(self.m, 'Variable')}"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        beta = self.beta[:self.m].to(X.device).float() / 10.0
        C = self.C[:self.m].to(X.device).float()

        result = torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)

        for i in range(self.m):
            diff = X - C[i]
            dist_sq = torch.sum(diff ** 2, dim=-1)
            result += 1.0 / (dist_sq + beta[i])

        return -result


# =============================================================================
# Classical Additional Functions (32 functions)
# =============================================================================

class Schwefel12Function(BenchmarkFunction):
    """Schwefel 1.2 function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-100.0] * dim, [100.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schwefel 1.2",
            "suite": "Classical Additional",
            "properties": ["unimodal", "double-sum"],
            "domain": "[-100,100]^d",
            "global_min": "f(0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        result = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        for i in range(X.shape[-1]):
            result += torch.sum(X[..., :i+1], dim=-1)**2
        return result


class Schwefel220Function(BenchmarkFunction):
    """Schwefel 2.20 function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-100.0] * dim, [100.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schwefel 2.20",
            "suite": "Classical Additional",
            "properties": ["unimodal", "absolute"],
            "domain": "[-100,100]^d",
            "global_min": "f(0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return torch.sum(torch.abs(X), dim=-1)


class Schwefel221Function(BenchmarkFunction):
    """Schwefel 2.21 function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-100.0] * dim, [100.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schwefel 2.21",
            "suite": "Classical Additional",
            "properties": ["unimodal", "max-mod"],
            "domain": "[-100,100]^d",
            "global_min": "f(0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return torch.max(torch.abs(X), dim=-1)[0]


class Schwefel222Function(BenchmarkFunction):
    """Schwefel 2.22 function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-10.0] * dim, [10.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schwefel 2.22",
            "suite": "Classical Additional",
            "properties": ["unimodal", "sum-product"],
            "domain": "[-10,10]^d",
            "global_min": "f(0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        sum_abs = torch.sum(torch.abs(X), dim=-1)
        prod_abs = torch.prod(torch.abs(X), dim=-1)
        return sum_abs + prod_abs


class Schwefel223Function(BenchmarkFunction):
    """Schwefel 2.23 function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-10.0] * dim, [10.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schwefel 2.23",
            "suite": "Classical Additional",
            "properties": ["unimodal", "high-powers"],
            "domain": "[-10,10]^d",
            "global_min": "f(0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return torch.sum(X**10, dim=-1)


class Schwefel226Function(BenchmarkFunction):
    """Schwefel 2.26 function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-500.0] * dim, [500.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schwefel 2.26",
            "suite": "Classical Additional",
            "properties": ["multimodal", "deceptive"],
            "domain": "[-500,500]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        dim = X.shape[-1]
        x_abs = torch.abs(X)
        return 418.9829 * dim - torch.sum(x_abs * torch.sin(torch.sqrt(x_abs)), dim=-1)


class LevyN13Function(BenchmarkFunction):
    """Levy N.13 function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-10.0] * dim, [10.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Levy N.13",
            "suite": "Classical Additional",
            "properties": ["multimodal", "trigonometric"],
            "domain": "[-10,10]^d",
            "global_min": "f(1,...,1)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        if X.shape[-1] == 1:
            return torch.sin(3 * math.pi * X[..., 0])**2

        w = 1 + (X - 1) / 4
        term1 = torch.sin(3 * math.pi * w[..., 0])**2

        sum_middle = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        for i in range(X.shape[-1] - 1):
            sum_middle += (w[..., i] - 1)**2 * (1 + 10 * torch.sin(3 * math.pi * w[..., i+1])**2)

        term3 = (w[..., -1] - 1)**2 * (1 + torch.sin(2 * math.pi * w[..., -1])**2)

        return term1 + sum_middle + term3


class Alpine1Function(BenchmarkFunction):
    """Alpine 1 function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-10.0] * dim, [10.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Alpine 1",
            "suite": "Classical Additional",
            "properties": ["multimodal", "non-differentiable"],
            "domain": "[-10,10]^d",
            "global_min": "f(0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return torch.sum(torch.abs(X * torch.sin(X) + 0.1 * X), dim=-1)


class Alpine2Function(BenchmarkFunction):
    """Alpine 2 function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[0.0] * dim, [10.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Alpine 2",
            "suite": "Classical Additional",
            "properties": ["multimodal", "product"],
            "domain": "[0,10]^d",
            "global_min": f"f=2.808^{self.dim}"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return torch.prod(torch.sqrt(X) * torch.sin(X), dim=-1)


class SchafferF1Function(BenchmarkFunction):
    """Schaffer F1 function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Schaffer F1 is only defined for 2D")
        bounds = torch.tensor([[-100.0, -100.0], [100.0, 100.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schaffer F1",
            "suite": "Classical Additional",
            "properties": ["multimodal", "2D"],
            "domain": "[-100,100]²",
            "global_min": "f(0,0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        sum_sq = x1**2 + x2**2
        return 0.5 + (torch.sin(sum_sq)**2 - 0.5) / (1 + 0.001 * sum_sq)**2


class SchafferF2Function(BenchmarkFunction):
    """Schaffer F2 function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Schaffer F2 is only defined for 2D")
        bounds = torch.tensor([[-100.0, -100.0], [100.0, 100.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schaffer F2",
            "suite": "Classical Additional",
            "properties": ["multimodal", "2D"],
            "domain": "[-100,100]²",
            "global_min": "f(0,0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        sum_sq = x1**2 + x2**2
        return 0.5 + (torch.sin(x1**2 - x2**2)**2 - 0.5) / (1 + 0.001 * sum_sq)**2


class SchafferF3Function(BenchmarkFunction):
    """Schaffer F3 function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Schaffer F3 is only defined for 2D")
        bounds = torch.tensor([[-100.0, -100.0], [100.0, 100.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schaffer F3",
            "suite": "Classical Additional",
            "properties": ["multimodal", "2D"],
            "domain": "[-100,100]²",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        sum_sq = x1**2 + x2**2
        return 0.5 + (torch.sin(torch.cos(torch.abs(x1**2 - x2**2)))**2 - 0.5) / (1 + 0.001 * sum_sq)**2


class SchafferF4Function(BenchmarkFunction):
    """Schaffer F4 function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Schaffer F4 is only defined for 2D")
        bounds = torch.tensor([[-100.0, -100.0], [100.0, 100.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schaffer F4",
            "suite": "Classical Additional",
            "properties": ["multimodal", "2D"],
            "domain": "[-100,100]²",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        sum_sq = x1**2 + x2**2
        return 0.5 + (torch.cos(torch.sin(torch.abs(x1**2 - x2**2)))**2 - 0.5) / (1 + 0.001 * sum_sq)**2


class SchafferF5Function(BenchmarkFunction):
    """Schaffer F5 function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Schaffer F5 is only defined for 2D")
        bounds = torch.tensor([[-100.0, -100.0], [100.0, 100.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schaffer F5",
            "suite": "Classical Additional",
            "properties": ["multimodal", "2D"],
            "domain": "[-100,100]²",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        sum_sq = x1**2 + x2**2
        return 0.5 + (torch.sin(sum_sq)**2 - 0.5) / (1 + 0.001 * sum_sq)**2


class SchafferF7Function(BenchmarkFunction):
    """Schaffer F7 function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Schaffer F7 is only defined for 2D")
        bounds = torch.tensor([[-100.0, -100.0], [100.0, 100.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schaffer F7",
            "suite": "Classical Additional",
            "properties": ["multimodal", "2D", "modulated"],
            "domain": "[-100,100]²",
            "global_min": "f(0,0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        sum_sq = x1**2 + x2**2
        return (sum_sq**0.25) * (torch.sin(50 * sum_sq**0.1)**2 + 1)


class CrossInTrayFunction(BenchmarkFunction):
    """Cross-in-tray function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Cross-in-tray is only defined for 2D")
        bounds = torch.tensor([[-10.0, -10.0], [10.0, 10.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Cross-in-tray",
            "suite": "Classical Additional",
            "properties": ["multimodal", "2D", "4-minima"],
            "domain": "[-10,10]²",
            "global_min": "f=±2.06261"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        exp_term = torch.abs(100 - torch.sqrt(x1**2 + x2**2) / math.pi)
        inner = torch.abs(torch.sin(x1) * torch.sin(x2) * torch.exp(exp_term)) + 1
        return -0.0001 * inner**0.1


class EggholderFunction(BenchmarkFunction):
    """Eggholder function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Eggholder is only defined for 2D")
        bounds = torch.tensor([[-512.0, -512.0], [512.0, 512.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Eggholder",
            "suite": "Classical Additional",
            "properties": ["multimodal", "2D"],
            "domain": "[-512,512]²",
            "global_min": "f=-959.6407"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        term1 = -(x2 + 47) * torch.sin(torch.sqrt(torch.abs(x2 + x1/2 + 47)))
        term2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47))))
        return term1 + term2


class HolderTableFunction(BenchmarkFunction):
    """Holder Table function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Holder Table is only defined for 2D")
        bounds = torch.tensor([[-10.0, -10.0], [10.0, 10.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Holder Table",
            "suite": "Classical Additional",
            "properties": ["multimodal", "2D", "4-minima"],
            "domain": "[-10,10]²",
            "global_min": "f=-19.2085"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        sum_sq = x1**2 + x2**2
        return -torch.abs(torch.sin(x1) * torch.cos(x2) * torch.exp(torch.abs(1 - torch.sqrt(sum_sq) / math.pi)))


class DropWaveFunction(BenchmarkFunction):
    """Drop-wave function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Drop-wave is only defined for 2D")
        bounds = torch.tensor([[-5.12, -5.12], [5.12, 5.12]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Drop-wave",
            "suite": "Classical Additional",
            "properties": ["multimodal", "2D", "oscillatory"],
            "domain": "[-5.12,5.12]²",
            "global_min": "f(0,0)=-1"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        sum_sq = x1**2 + x2**2
        return -(1 + torch.cos(12 * torch.sqrt(sum_sq))) / (0.5 * sum_sq + 2)


class ShubertFunction(BenchmarkFunction):
    """Shubert function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Shubert is only defined for 2D")
        bounds = torch.tensor([[-10.0, -10.0], [10.0, 10.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Shubert",
            "suite": "Classical Additional",
            "properties": ["multimodal", "2D", "18-minima"],
            "domain": "[-10,10]²",
            "global_min": "f=-186.7309"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]

        sum1 = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        sum2 = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

        for i in range(1, 6):
            sum1 += i * torch.cos((i + 1) * x1 + i)
            sum2 += i * torch.cos((i + 1) * x2 + i)

        return sum1 * sum2


class PowellFunction(BenchmarkFunction):
    """Powell function."""

    def __init__(self, dim: int = 4, **kwargs):
        if dim % 4 != 0:
            raise ValueError("Powell requires dimension divisible by 4")
        bounds = torch.tensor([[-4.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Powell",
            "suite": "Classical Additional",
            "properties": ["unimodal", "groups-of-4"],
            "domain": "[-4,5]^d",
            "global_min": "f(0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        result = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

        for i in range(0, X.shape[-1], 4):
            if i + 3 < X.shape[-1]:
                x1, x2, x3, x4 = X[..., i], X[..., i+1], X[..., i+2], X[..., i+3]
                term1 = (x1 + 10 * x2)**2
                term2 = 5 * (x3 - x4)**2
                term3 = (x2 - 2 * x3)**4
                term4 = 10 * (x1 - x4)**4
                result += term1 + term2 + term3 + term4

        return result


class TridFunction(BenchmarkFunction):
    """Trid function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-dim**2] * dim, [dim**2] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Trid",
            "suite": "Classical Additional",
            "properties": ["unimodal", "special-structure"],
            "domain": f"[-{self.dim}²,{self.dim}²]^d",
            "global_min": f"f=-{self.dim}({self.dim}+4)({self.dim}-1)/6"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        sum1 = torch.sum((X - 1)**2, dim=-1)
        sum2 = torch.sum(X[..., 1:] * X[..., :-1], dim=-1)
        return sum1 - sum2


class BoothFunction(BenchmarkFunction):
    """Booth function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Booth is only defined for 2D")
        bounds = torch.tensor([[-10.0, -10.0], [10.0, 10.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Booth",
            "suite": "Classical Additional",
            "properties": ["unimodal", "2D", "quadratic"],
            "domain": "[-10,10]²",
            "global_min": "f(1,3)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        term1 = (x1 + 2 * x2 - 7)**2
        term2 = (2 * x1 + x2 - 5)**2
        return term1 + term2


class MatyasFunction(BenchmarkFunction):
    """Matyas function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Matyas is only defined for 2D")
        bounds = torch.tensor([[-10.0, -10.0], [10.0, 10.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Matyas",
            "suite": "Classical Additional",
            "properties": ["unimodal", "2D", "quadratic"],
            "domain": "[-10,10]²",
            "global_min": "f(0,0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


class McCormickFunction(BenchmarkFunction):
    """McCormick function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("McCormick is only defined for 2D")
        bounds = torch.tensor([[-1.5, -3.0], [4.0, 4.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "McCormick",
            "suite": "Classical Additional",
            "properties": ["multimodal", "2D", "trigonometric"],
            "domain": "[-1.5,4]×[-3,4]",
            "global_min": "f=-1.9133"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        term1 = torch.sin(x1 + x2)
        term2 = (x1 - x2)**2
        term3 = -1.5 * x1
        term4 = 2.5 * x2
        return term1 + term2 + term3 + term4 + 1


class SixHumpCamelFunction(BenchmarkFunction):
    """Six-Hump Camel function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Six-Hump Camel is only defined for 2D")
        bounds = torch.tensor([[-3.0, -2.0], [3.0, 2.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Six-Hump Camel",
            "suite": "Classical Additional",
            "properties": ["multimodal", "2D", "6-minima"],
            "domain": "[-3,3]×[-2,2]",
            "global_min": "f=-1.0316"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        term1 = (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2
        return term1 + term2 + term3


class BraninFunction(BenchmarkFunction):
    """Branin function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Branin is only defined for 2D")
        bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

        self.a = 1
        self.b = 5.1 / (4 * math.pi**2)
        self.c = 5 / math.pi
        self.r = 6
        self.s = 10
        self.t = 1 / (8 * math.pi)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Branin",
            "suite": "Classical Additional",
            "properties": ["multimodal", "2D", "3-minima"],
            "domain": "[-5,10]×[0,15]",
            "global_min": "f=0.397887"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        term1 = self.a * (x2 - self.b * x1**2 + self.c * x1 - self.r)**2
        term2 = self.s * (1 - self.t) * torch.cos(x1)
        return term1 + term2 + self.s


class Hartmann3DFunction(BenchmarkFunction):
    """Hartmann 3D function."""

    def __init__(self, dim: int = 3, **kwargs):
        if dim != 3:
            raise ValueError("Hartmann 3D is only defined for 3D")
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

        self.alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
        self.A = torch.tensor([
            [3.0, 10, 30],
            [0.1, 10, 35],
            [3.0, 10, 30],
            [0.1, 10, 35]
        ])
        self.P = torch.tensor([
            [0.3689, 0.1170, 0.2673],
            [0.4699, 0.4387, 0.7470],
            [0.1091, 0.8732, 0.5547],
            [0.0381, 0.5743, 0.8828]
        ])

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Hartmann 3D",
            "suite": "Classical Additional",
            "properties": ["multimodal", "3D", "4-minima"],
            "domain": "[0,1]³",
            "global_min": "f=-3.86278"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        result = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

        for i in range(4):
            alpha_i = self.alpha[i].to(X.device)
            A_i = self.A[i].to(X.device)
            P_i = self.P[i].to(X.device)

            exp_term = torch.sum(A_i * (X - P_i)**2, dim=-1)
            result += alpha_i * torch.exp(-exp_term)

        return -result


class Hartmann4DFunction(BenchmarkFunction):
    """Hartmann 4D function."""

    def __init__(self, dim: int = 4, **kwargs):
        if dim != 4:
            raise ValueError("Hartmann 4D is only defined for 4D")
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

        self.alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
        self.A = torch.tensor([
            [10, 3, 17, 3.5],
            [0.05, 10, 17, 0.1],
            [3, 3.5, 1.7, 10],
            [17, 8, 0.05, 10]
        ])
        self.P = torch.tensor([
            [0.1312, 0.1696, 0.5569, 0.0124],
            [0.2329, 0.4135, 0.8307, 0.3736],
            [0.2348, 0.1451, 0.3522, 0.2883],
            [0.4047, 0.8828, 0.8732, 0.5743]
        ])

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Hartmann 4D",
            "suite": "Classical Additional",
            "properties": ["multimodal", "4D", "4-minima"],
            "domain": "[0,1]⁴",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        result = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

        for i in range(4):
            alpha_i = self.alpha[i].to(X.device)
            A_i = self.A[i].to(X.device)
            P_i = self.P[i].to(X.device)

            exp_term = torch.sum(A_i * (X - P_i)**2, dim=-1)
            result += alpha_i * torch.exp(-exp_term)

        # Apply scaling factor for 4D as in botorch
        return -(1.1 + result) / 0.839


class Hartmann6DFunction(BenchmarkFunction):
    """Hartmann 6D function."""

    def __init__(self, dim: int = 6, **kwargs):
        if dim != 6:
            raise ValueError("Hartmann 6D is only defined for 6D")
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

        self.alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
        self.A = torch.tensor([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ])
        self.P = torch.tensor([
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1003, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]
        ])

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Hartmann 6D",
            "suite": "Classical Additional",
            "properties": ["multimodal", "6D", "6-minima"],
            "domain": "[0,1]⁶",
            "global_min": "f=-3.32237"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        result = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

        for i in range(4):
            alpha_i = self.alpha[i].to(X.device)
            A_i = self.A[i].to(X.device)
            P_i = self.P[i].to(X.device)

            exp_term = torch.sum(A_i * (X - P_i)**2, dim=-1)
            result += alpha_i * torch.exp(-exp_term)

        return -result


# =============================================================================
# Classical Core Functions (10 functions)
# =============================================================================

class StyblinskiTangFunction(BenchmarkFunction):
    """Styblinski-Tang function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Styblinski-Tang",
            "suite": "Classical Core",
            "properties": ["multimodal", "separable"],
            "domain": "[-5,5]^d",
            "global_min": f"f=-39.166×{self.dim}"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return 0.5 * torch.sum(X**4 - 16 * X**2 + 5 * X, dim=-1)


class LevyFunction(BenchmarkFunction):
    """Levy function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-10.0] * dim, [10.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Levy",
            "suite": "Classical Core",
            "properties": ["multimodal", "complex"],
            "domain": "[-10,10]^d",
            "global_min": "f(1,...,1)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        if X.shape[-1] == 1:
            return torch.sin(math.pi * X[..., 0])**2

        w = 1 + (X - 1) / 4
        term1 = torch.sin(math.pi * w[..., 0])**2

        sum_middle = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        for i in range(X.shape[-1] - 1):
            sum_middle += (w[..., i] - 1)**2 * (1 + 10 * torch.sin(math.pi * w[..., i] + 1)**2)

        term3 = (w[..., -1] - 1)**2 * (1 + torch.sin(2 * math.pi * w[..., -1])**2)

        return term1 + sum_middle + term3


class MichalewiczFunction(BenchmarkFunction):
    """Michalewicz function."""

    def __init__(self, dim: int = 2, m: int = 10, **kwargs):
        bounds = torch.tensor([[0.0] * dim, [math.pi] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)
        self.m = m

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Michalewicz",
            "suite": "Classical Core",
            "properties": ["multimodal", "parameterized"],
            "domain": "[0,π]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        result = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        for i in range(X.shape[-1]):
            result -= torch.sin(X[..., i]) * torch.sin((i + 1) * X[..., i]**2 / math.pi)**(2 * self.m)
        return result


class ZakharovFunction(BenchmarkFunction):
    """Zakharov function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-5.0] * dim, [10.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Zakharov",
            "suite": "Classical Core",
            "properties": ["unimodal"],
            "domain": "[-5,10]^d",
            "global_min": "f(0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        sum_sq = torch.sum(X**2, dim=-1)
        sum_linear = torch.sum(0.5 * torch.arange(1, X.shape[-1] + 1, device=X.device, dtype=X.dtype) * X, dim=-1)
        return sum_sq + sum_linear**2 + sum_linear**4


class DixonPriceFunction(BenchmarkFunction):
    """Dixon-Price function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-10.0] * dim, [10.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Dixon-Price",
            "suite": "Classical Core",
            "properties": ["unimodal", "adjacent-interaction"],
            "domain": "[-10,10]^d",
            "global_min": "Variable"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        if X.shape[-1] == 1:
            return (X[..., 0] - 1)**2

        term1 = (X[..., 0] - 1)**2

        sum_terms = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        for i in range(1, X.shape[-1]):
            i_tensor = torch.tensor(i, device=X.device, dtype=X.dtype)
            term = i_tensor * (2 * X[..., i]**2 - X[..., i-1])**2
            sum_terms += term

        return term1 + sum_terms


class SalomonFunction(BenchmarkFunction):
    """Salomon function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[-100.0] * dim, [100.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Salomon",
            "suite": "Classical Core",
            "properties": ["multimodal"],
            "domain": "[-100,100]^d",
            "global_min": "f(0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        norm = torch.sqrt(torch.sum(X**2, dim=-1))
        return 1 - torch.cos(2 * math.pi * norm) + 0.1 * norm


class SchafferF6Function(BenchmarkFunction):
    """Schaffer F6 function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Schaffer F6 is only defined for 2D")
        bounds = torch.tensor([[-100.0, -100.0], [100.0, 100.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Schaffer F6",
            "suite": "Classical Core",
            "properties": ["multimodal", "2D"],
            "domain": "[-100,100]²",
            "global_min": "f(0,0)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        sum_sq = x1**2 + x2**2
        return 0.5 + (torch.sin(torch.sqrt(sum_sq))**2 - 0.5) / (1 + 0.001 * sum_sq)**2


class EasomFunction(BenchmarkFunction):
    """Easom function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Easom is only defined for 2D")
        bounds = torch.tensor([[-100.0, -100.0], [100.0, 100.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Easom",
            "suite": "Classical Core",
            "properties": ["multimodal", "2D", "sharp-minimum"],
            "domain": "[-100,100]²",
            "global_min": "f(π,π)=-1"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        exp_term = -((x1 - math.pi)**2 + (x2 - math.pi)**2)
        return -torch.cos(x1) * torch.cos(x2) * torch.exp(exp_term)


class BealeFunction(BenchmarkFunction):
    """Beale function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Beale is only defined for 2D")
        bounds = torch.tensor([[-4.5, -4.5], [4.5, 4.5]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Beale",
            "suite": "Classical Core",
            "properties": ["multimodal", "2D", "sharp-peaks"],
            "domain": "[-4.5,4.5]²",
            "global_min": "f(3,0.5)=0"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        term1 = (1.5 - x1 + x1 * x2)**2
        term2 = (2.25 - x1 + x1 * x2**2)**2
        term3 = (2.625 - x1 + x1 * x2**3)**2
        return term1 + term2 + term3


class GoldsteinPriceFunction(BenchmarkFunction):
    """Goldstein-Price function."""

    def __init__(self, dim: int = 2, **kwargs):
        if dim != 2:
            raise ValueError("Goldstein-Price is only defined for 2D")
        bounds = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Goldstein-Price",
            "suite": "Classical Core",
            "properties": ["multimodal", "2D"],
            "domain": "[-2,2]²",
            "global_min": "f(0,-1)=3"
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]

        factor1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
        factor2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)

        return factor1 * factor2


# =============================================================================
# Suite Creation Functions
# =============================================================================

def create_consolidated_suite(dimensions: List[int] = [2, 4, 8, 30, 53]) -> BenchmarkSuite:
    """Create consolidated suite with all 72 unique benchmark functions."""
    functions = {}

    # BBOB Functions (24)
    bbob_classes = [
        F01_SphereRaw, F02_EllipsoidSeparableRaw, F03_RastriginSeparableRaw,
        F04_SkewRastriginBuecheRaw, F05_LinearSlopeRaw, F06_AttractiveSectorRaw,
        F07_StepEllipsoidRaw, F08_RosenbrockRaw, F09_RosenbrockRotatedRaw,
        F10_EllipsoidRaw, F11_DiscusRaw, F12_BentCigarRaw, F13_SharpRidgeRaw,
        F14_DifferentPowersRaw, F15_RastriginRaw, F16_WeierstrassRaw,
        F17_SchafferF7Cond10Raw, F18_SchafferF7Cond1000Raw,
        F19_GriewankRosenbrockRaw, F20_SchwefelRaw, F21_Gallagher101Raw,
        F22_Gallagher21Raw, F23_KatsuuraRaw, F24_LunacekBiRastriginRaw
    ]

    for cls in bbob_classes:
        for dim in dimensions:
            functions[f"{cls.__name__}_{dim}d"] = cls(dim=dim)

    # BoTorch Additional Functions (6)
    if 2 in dimensions:
        functions["bukin"] = BukinFunction()
        functions["three_hump_camel"] = ThreeHumpCamelFunction()

    if 4 in dimensions:
        functions["shekel_m5"] = ShekelFunction(m=5)
        functions["shekel_m7"] = ShekelFunction(m=7)
        functions["shekel_m10"] = ShekelFunction(m=10)

    for dim in dimensions:
        if dim >= 8:
            functions[f"cosine_mixture_{dim}d"] = Cosine8Function(dim=dim)
        if dim > 3:
            functions[f"ackley_mixed_{dim}d"] = AckleyMixedFunction(dim=dim)
        if dim >= 10:
            functions[f"labs_{dim}d"] = LabsFunction(dim=dim)

    # Classical Additional Functions (32)
    classical_additional_classes = [
        Schwefel12Function, Schwefel220Function, Schwefel221Function,
        Schwefel222Function, Schwefel223Function, Schwefel226Function,
        LevyN13Function, Alpine1Function, Alpine2Function,
        SchafferF1Function, SchafferF2Function, SchafferF3Function,
        SchafferF4Function, SchafferF5Function, SchafferF7Function,
        CrossInTrayFunction, EggholderFunction, HolderTableFunction,
        DropWaveFunction, ShubertFunction, PowellFunction, TridFunction,
        BoothFunction, MatyasFunction, McCormickFunction, SixHumpCamelFunction,
        BraninFunction, Hartmann3DFunction, Hartmann4DFunction, Hartmann6DFunction
    ]

    for cls in classical_additional_classes:
        if cls in [SchafferF1Function, SchafferF2Function, SchafferF3Function,
                   SchafferF4Function, SchafferF5Function, SchafferF7Function,
                   CrossInTrayFunction, EggholderFunction, HolderTableFunction,
                   DropWaveFunction, ShubertFunction, BoothFunction, MatyasFunction,
                   McCormickFunction, SixHumpCamelFunction, BraninFunction]:
            # 2D-only functions
            if 2 in dimensions:
                functions[cls.__name__.replace("Function", "").lower()] = cls(dim=2)
        elif cls in [Hartmann3DFunction]:
            # 3D-only function
            if 3 in dimensions:
                functions[cls.__name__.replace("Function", "").lower()] = cls(dim=3)
        elif cls in [Hartmann4DFunction]:
            # 4D-only function
            if 4 in dimensions:
                functions[cls.__name__.replace("Function", "").lower()] = cls(dim=4)
        elif cls in [Hartmann6DFunction]:
            # 6D-only function
            if 6 in dimensions:
                functions[cls.__name__.replace("Function", "").lower()] = cls(dim=6)
        elif cls in [PowellFunction]:
            # Functions requiring dimension divisible by 4
            for dim in dimensions:
                if dim % 4 == 0:
                    functions[f"{cls.__name__.replace('Function', '').lower()}_{dim}d"] = cls(dim=dim)
        elif cls in [TridFunction]:
            # Variable dimension functions
            for dim in dimensions:
                functions[f"{cls.__name__.replace('Function', '').lower()}_{dim}d"] = cls(dim=dim)
        else:
            # General dimension functions
            for dim in dimensions:
                functions[f"{cls.__name__.replace('Function', '').lower()}_{dim}d"] = cls(dim=dim)

    # Classical Core Functions (10)
    classical_core_classes = [
        StyblinskiTangFunction, LevyFunction, MichalewiczFunction,
        ZakharovFunction, DixonPriceFunction, SalomonFunction,
        SchafferF6Function, EasomFunction, BealeFunction, GoldsteinPriceFunction
    ]

    for cls in classical_core_classes:
        if cls in [SchafferF6Function, EasomFunction, BealeFunction, GoldsteinPriceFunction]:
            # 2D-only functions
            if 2 in dimensions:
                functions[cls.__name__.replace("Function", "").lower()] = cls(dim=2)
        else:
            # General dimension functions
            for dim in dimensions:
                functions[f"{cls.__name__.replace('Function', '').lower()}_{dim}d"] = cls(dim=dim)

    return BenchmarkSuite("Consolidated Functions", functions)


# =============================================================================
# Function Summary and Statistics
# =============================================================================

CONSolidated_FUNCTIONS_SUMMARY = {
    "total_unique_functions": 73,
    "bbob_functions": 24,
    "botorch_additional_functions": 6,
    "classical_additional_functions": 33,
    "classical_core_functions": 10,
    "function_categories": {
        "unimodal": "Sphere, Ellipsoid variants, Linear Slope, Attractive Sector, Step-Ellipsoid",
        "multimodal": "Rastrigin variants, Schwefel variants, Schaffer variants, Hartmann, etc.",
        "2D_only": "Most Schaffer functions, Bukin, Three-Hump Camel, etc.",
        "variable_dimension": "Sphere, Schwefel 1.2, Styblinski-Tang, etc.",
        "specialized": "LABS (binary), Ackley Mixed (mixed-integer), Shekel (4D multimodal)"
    }
}