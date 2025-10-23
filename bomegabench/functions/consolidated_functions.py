"""
Consolidated Benchmark Functions - Backward Compatibility Layer.

This module provides backward compatibility with the old monolithic
consolidated_functions.py file. All functions are now organized in
the consolidated/ subpackage for better maintainability.

Total: 72 unique benchmark functions across:
- BBOB (24 functions)
- BoTorch Additional (6 functions)
- Classical Additional (32 functions)
- Classical Core (10 functions)
"""

# Re-export everything from the new modular structure
from .consolidated import (
    create_consolidated_suite,
    CONSOLIDATED_FUNCTIONS_SUMMARY,
    BBOB_FUNCTIONS,
    BOTORCH_ADDITIONAL_FUNCTIONS,
    CLASSICAL_ADDITIONAL_FUNCTIONS,
    CLASSICAL_CORE_FUNCTIONS
)

# Re-export all individual function classes for backward compatibility
from .consolidated.bbob_functions import *
from .consolidated.botorch_additional import *
from .consolidated.classical_additional import *
from .consolidated.classical_core import *


__all__ = [
    "create_consolidated_suite",
    "CONSOLIDATED_FUNCTIONS_SUMMARY",
    "BBOB_FUNCTIONS",
    "BOTORCH_ADDITIONAL_FUNCTIONS",
    "CLASSICAL_ADDITIONAL_FUNCTIONS",
    "CLASSICAL_CORE_FUNCTIONS",
]
