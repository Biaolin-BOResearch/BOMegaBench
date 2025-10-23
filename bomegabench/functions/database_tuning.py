"""
Database Knob Tuning - Backward Compatibility Layer.

This module provides backward compatibility with the old monolithic
database_tuning.py file. All functionality is now organized in the
database/ subpackage for better maintainability.

DEPRECATED: This module is deprecated. Please import from
bomegabench.functions.database instead:

    from bomegabench.functions.database import (
        DatabaseTuningFunction,
        create_database_tuning_suite,
        DatabaseTuningSuite
    )
"""

import warnings

# Show deprecation warning on import
warnings.warn(
    "Importing from bomegabench.functions.database_tuning is deprecated. "
    "Please use: from bomegabench.functions.database import DatabaseTuningFunction",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new modular structure for backward compatibility
from .database import (
    # Main classes
    DatabaseTuningFunction,
    create_database_tuning_suite,

    # Default suites
    DatabaseTuningSuite,
    PostgreSQLTuningSuite,
    MySQLTuningSuite,

    # Utility functions
    get_postgresql_knobs,
    get_mysql_knobs,
    get_default_knob_config,
    validate_knob_config,
    get_knob_documentation,
    SpaceConverter,
    DatabaseEvaluator,
)


__all__ = [
    "DatabaseTuningFunction",
    "create_database_tuning_suite",
    "DatabaseTuningSuite",
    "PostgreSQLTuningSuite",
    "MySQLTuningSuite",
    "get_postgresql_knobs",
    "get_mysql_knobs",
    "get_default_knob_config",
    "validate_knob_config",
    "get_knob_documentation",
    "SpaceConverter",
    "DatabaseEvaluator",
]
