"""
Consolidated benchmark functions - modularized structure.

This package contains 72 unique benchmark functions organized into:
- BBOB Functions (24)
- BoTorch Additional (6)
- Classical Additional (32)
- Classical Core (10)
"""

from typing import Dict, List
from ...core import BenchmarkFunction, BenchmarkSuite

# Import all function classes from submodules
from .bbob_functions import BBOB_FUNCTIONS
from .botorch_additional import BOTORCH_ADDITIONAL_FUNCTIONS
from .classical_additional import CLASSICAL_ADDITIONAL_FUNCTIONS
from .classical_core import CLASSICAL_CORE_FUNCTIONS


def create_consolidated_suite(dimensions: List[int] = [2, 4, 8, 30, 53]) -> BenchmarkSuite:
    """Create consolidated suite with all 72 unique benchmark functions."""
    functions = {}

    # BBOB Functions (24)
    for cls in BBOB_FUNCTIONS:
        for dim in dimensions:
            functions[f"{cls.__name__}_{dim}d"] = cls(dim=dim)

    # BoTorch Additional Functions (6)
    if 2 in dimensions:
        from .botorch_additional import BukinFunction, ThreeHumpCamelFunction
        functions["bukin"] = BukinFunction()
        functions["three_hump_camel"] = ThreeHumpCamelFunction()

    if 4 in dimensions:
        from .botorch_additional import ShekelFunction
        functions["shekel_m5"] = ShekelFunction(m=5)
        functions["shekel_m7"] = ShekelFunction(m=7)
        functions["shekel_m10"] = ShekelFunction(m=10)

    for dim in dimensions:
        from .botorch_additional import Cosine8Function, AckleyMixedFunction, LabsFunction
        if dim >= 8:
            functions[f"cosine_mixture_{dim}d"] = Cosine8Function(dim=dim)
        if dim > 3:
            functions[f"ackley_mixed_{dim}d"] = AckleyMixedFunction(dim=dim)
        if dim >= 10:
            functions[f"labs_{dim}d"] = LabsFunction(dim=dim)

    # Classical Additional Functions (32)
    # Import all classes dynamically
    from .classical_additional import (
        Schwefel12Function, Schwefel220Function, Schwefel221Function,
        Schwefel222Function, Schwefel223Function, Schwefel226Function,
        LevyN13Function, Alpine1Function, Alpine2Function,
        SchafferF1Function, SchafferF2Function, SchafferF3Function,
        SchafferF4Function, SchafferF5Function, SchafferF7Function,
        CrossInTrayFunction, EggholderFunction, HolderTableFunction,
        DropWaveFunction, ShubertFunction, PowellFunction, TridFunction,
        BoothFunction, MatyasFunction, McCormickFunction, SixHumpCamelFunction,
        BraninFunction, Hartmann3DFunction, Hartmann4DFunction, Hartmann6DFunction
    )

    classical_additional_2d = [
        SchafferF1Function, SchafferF2Function, SchafferF3Function,
        SchafferF4Function, SchafferF5Function, SchafferF7Function,
        CrossInTrayFunction, EggholderFunction, HolderTableFunction,
        DropWaveFunction, ShubertFunction, BoothFunction, MatyasFunction,
        McCormickFunction, SixHumpCamelFunction, BraninFunction
    ]

    for cls in classical_additional_2d:
        if 2 in dimensions:
            functions[cls.__name__.replace("Function", "").lower()] = cls(dim=2)

    # 3D function
    if 3 in dimensions:
        functions["hartmann3d"] = Hartmann3DFunction(dim=3)

    # 4D function
    if 4 in dimensions:
        functions["hartmann4d"] = Hartmann4DFunction(dim=4)

    # 6D function
    if 6 in dimensions:
        functions["hartmann6d"] = Hartmann6DFunction(dim=6)

    # Powell (divisible by 4)
    for dim in dimensions:
        if dim % 4 == 0:
            functions[f"powell_{dim}d"] = PowellFunction(dim=dim)

    # Variable dimension functions
    variable_dim_classes = [
        Schwefel12Function, Schwefel220Function, Schwefel221Function,
        Schwefel222Function, Schwefel223Function, Schwefel226Function,
        LevyN13Function, Alpine1Function, Alpine2Function, TridFunction
    ]
    for cls in variable_dim_classes:
        for dim in dimensions:
            functions[f"{cls.__name__.replace('Function', '').lower()}_{dim}d"] = cls(dim=dim)

    # Classical Core Functions (10)
    from .classical_core import (
        StyblinskiTangFunction, LevyFunction, MichalewiczFunction,
        ZakharovFunction, DixonPriceFunction, SalomonFunction,
        SchafferF6Function, EasomFunction, BealeFunction, GoldsteinPriceFunction
    )

    classical_core_2d = [SchafferF6Function, EasomFunction, BealeFunction, GoldsteinPriceFunction]
    for cls in classical_core_2d:
        if 2 in dimensions:
            functions[cls.__name__.replace("Function", "").lower()] = cls(dim=2)

    # Variable dimension core functions
    variable_dim_core = [
        StyblinskiTangFunction, LevyFunction, MichalewiczFunction,
        ZakharovFunction, DixonPriceFunction, SalomonFunction
    ]
    for cls in variable_dim_core:
        for dim in dimensions:
            functions[f"{cls.__name__.replace('Function', '').lower()}_{dim}d"] = cls(dim=dim)

    return BenchmarkSuite("Consolidated Functions", functions)


# Function summary and statistics
CONSOLIDATED_FUNCTIONS_SUMMARY = {
    "total_unique_functions": 72,
    "bbob_functions": 24,
    "botorch_additional_functions": 6,
    "classical_additional_functions": 32,
    "classical_core_functions": 10,
    "function_categories": {
        "unimodal": "Sphere, Ellipsoid variants, Linear Slope, Attractive Sector, Step-Ellipsoid",
        "multimodal": "Rastrigin variants, Schwefel variants, Schaffer variants, Hartmann, etc.",
        "2D_only": "Most Schaffer functions, Bukin, Three-Hump Camel, etc.",
        "variable_dimension": "Sphere, Schwefel 1.2, Styblinski-Tang, etc.",
        "specialized": "LABS (binary), Ackley Mixed (mixed-integer), Shekel (4D multimodal)"
    }
}


__all__ = [
    "create_consolidated_suite",
    "CONSOLIDATED_FUNCTIONS_SUMMARY",
    "BBOB_FUNCTIONS",
    "BOTORCH_ADDITIONAL_FUNCTIONS",
    "CLASSICAL_ADDITIONAL_FUNCTIONS",
    "CLASSICAL_CORE_FUNCTIONS"
]
