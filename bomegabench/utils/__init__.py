"""
Utility modules for BOMegaBench.
"""

from .dependencies import (
    check_dependency,
    require_dependency,
    get_missing_dependencies,
    print_dependency_status
)

from .discrete_encoding import (
    DiscreteParameterSpec,
    DiscreteEncoder,
    create_encoder_for_hpo
)

__all__ = [
    "check_dependency",
    "require_dependency",
    "get_missing_dependencies",
    "print_dependency_status",
    "DiscreteParameterSpec",
    "DiscreteEncoder",
    "create_encoder_for_hpo",
]
