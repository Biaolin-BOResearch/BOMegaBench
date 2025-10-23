"""
Classical Core Benchmark Functions.

This module contains 10 core classical benchmark functions
that are frequently used in optimization research.
"""

import torch
from torch import Tensor
import math
from typing import Dict, Any

from ...core import BenchmarkFunction


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




# Export all Classical Core function classes
CLASSICAL_CORE_FUNCTIONS = [
    StyblinskiTangFunction,
    LevyFunction,
    MichalewiczFunction,
    ZakharovFunction,
    DixonPriceFunction,
    SalomonFunction,
    SchafferF6Function,
    EasomFunction,
    BealeFunction,
    GoldsteinPriceFunction
]
