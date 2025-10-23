"""
Synthetic Benchmark Functions - Backward Compatibility Layer.

This module provides backward compatibility with the old monolithic
synthetic_functions.py file. All functions are now organized in
the synthetic/ subpackage for better maintainability.

Total: 72 unique benchmark functions across:
- BBOB (24 functions)
- BoTorch Additional (6 functions)
- Classical Additional (32 functions)
- Classical Core (10 functions)
"""

# Re-export everything from the new modular structure
from .synthetic import (
    create_synthetic_suite,
    SYNTHETIC_FUNCTIONS_SUMMARY,
    BBOB_FUNCTIONS,
    BOTORCH_ADDITIONAL_FUNCTIONS,
    CLASSICAL_ADDITIONAL_FUNCTIONS,
    CLASSICAL_CORE_FUNCTIONS
)

# Re-export all individual function classes for backward compatibility
from .synthetic.bbob_functions import *
from .synthetic.botorch_additional import *
from .synthetic.classical_additional import *
from .synthetic.classical_core import *


__all__ = [
    "create_synthetic_suite",
    "SYNTHETIC_FUNCTIONS_SUMMARY",
    "BBOB_FUNCTIONS",
    "BOTORCH_ADDITIONAL_FUNCTIONS",
    "CLASSICAL_ADDITIONAL_FUNCTIONS",
    "CLASSICAL_CORE_FUNCTIONS",
]
