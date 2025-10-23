"""
Classical Additional Benchmark Functions.

This module contains 32 additional classical benchmark functions
including Schwefel variants, Schaffer variants, and other well-known
optimization test problems.
"""

import torch
from torch import Tensor
import math
from typing import Dict, Any

from ...core import BenchmarkFunction


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



# Export all Classical Additional function classes
CLASSICAL_ADDITIONAL_FUNCTIONS = [
    Schwefel12Function, Schwefel220Function, Schwefel221Function,
    Schwefel222Function, Schwefel223Function, Schwefel226Function,
    LevyN13Function, Alpine1Function, Alpine2Function,
    SchafferF1Function, SchafferF2Function, SchafferF3Function,
    SchafferF4Function, SchafferF5Function, SchafferF7Function,
    CrossInTrayFunction, EggholderFunction, HolderTableFunction,
    DropWaveFunction, ShubertFunction, PowellFunction, TridFunction,
    BoothFunction, MatyasFunction, McCormickFunction, SixHumpCamelFunction,
    BraninFunction, Hartmann3DFunction, Hartmann4DFunction, Hartmann6DFunction,
    Hartmann4DFunction, Hartmann6DFunction
]
