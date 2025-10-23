"""
BoTorch Additional Functions.

This module contains 6 additional benchmark functions from BoTorch
that are not part of the standard BBOB suite, including mixed-integer
and binary optimization problems.
"""

import torch
from torch import Tensor
import math
from typing import Dict, Any

from ...core import BenchmarkFunction


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


# Export all BoTorch Additional function classes
BOTORCH_ADDITIONAL_FUNCTIONS = [
    BukinFunction,
    Cosine8Function,
    ThreeHumpCamelFunction,
    AckleyMixedFunction,
    LabsFunction,
    ShekelFunction
]
