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

from .dimension_expansion import (
    DimensionMapping,
    DimensionExpandedFunction,
    DimensionDiscoveryResult,
    DimensionDiscoveryMetrics,
    create_dimension_expansion_test,
    run_dimension_discovery_experiment,
)

__all__ = [
    "check_dependency",
    "require_dependency",
    "get_missing_dependencies",
    "print_dependency_status",
    "DiscreteParameterSpec",
    "DiscreteEncoder",
    "create_encoder_for_hpo",
    # Dimension expansion utilities
    "DimensionMapping",
    "DimensionExpandedFunction",
    "DimensionDiscoveryResult",
    "DimensionDiscoveryMetrics",
    "create_dimension_expansion_test",
    "run_dimension_discovery_experiment",
]
