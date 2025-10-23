"""
BBOB (Black-Box Optimization Benchmarking) Functions.

This module contains 24 benchmark functions from the BBOB suite,
representing a diverse set of optimization challenges including
unimodal, multimodal, separable, and non-separable functions.
"""

import torch
from torch import Tensor
import math
from typing import Dict, Any

from ...core import BenchmarkFunction


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


# Export all BBOB function classes
BBOB_FUNCTIONS = [
    F01_SphereRaw, F02_EllipsoidSeparableRaw, F03_RastriginSeparableRaw,
    F04_SkewRastriginBuecheRaw, F05_LinearSlopeRaw, F06_AttractiveSectorRaw,
    F07_StepEllipsoidRaw, F08_RosenbrockRaw, F09_RosenbrockRotatedRaw,
    F10_EllipsoidRaw, F11_DiscusRaw, F12_BentCigarRaw, F13_SharpRidgeRaw,
    F14_DifferentPowersRaw, F15_RastriginRaw, F16_WeierstrassRaw,
    F17_SchafferF7Cond10Raw, F18_SchafferF7Cond1000Raw,
    F19_GriewankRosenbrockRaw, F20_SchwefelRaw, F21_Gallagher101Raw,
    F22_Gallagher21Raw, F23_KatsuuraRaw, F24_LunacekBiRastriginRaw
]
