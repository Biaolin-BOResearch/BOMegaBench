"""
Database Knob Tuning integration for BOMegaBench framework.

This module provides a unified interface for database configuration tuning
benchmarks, supporting PostgreSQL, MySQL, and other database systems.

The module is organized into:
- core: Main DatabaseTuningFunction class
- knob_configs: Default knob configurations for different databases
- space_converter: Continuous-discrete space conversion utilities
- evaluator: BenchBase integration for configuration evaluation
- suite: Suite creation utilities

Examples
--------
>>> from bomegabench.functions.database import DatabaseTuningFunction
>>>
>>> # Create a database tuning function
>>> func = DatabaseTuningFunction(
...     workload_name="tpcc",
...     database_system="postgresql"
... )
>>>
>>> # Evaluate a configuration
>>> import torch
>>> X = torch.rand(1, func.dim)
>>> performance = func(X)
>>>
>>> # Get knob documentation
>>> print(func.get_knob_documentation())
"""

import warnings

# Core functionality
from .core import DatabaseTuningFunction
from .suite import create_database_tuning_suite

# Utility modules (for advanced users)
from .knob_configs import (
    get_postgresql_knobs,
    get_mysql_knobs,
    get_default_knob_config,
    validate_knob_config,
    get_knob_documentation
)
from .space_converter import SpaceConverter
from .evaluator import DatabaseEvaluator


# Create default suites
try:
    PostgreSQLTuningSuite = create_database_tuning_suite("postgresql")
    MySQLTuningSuite = create_database_tuning_suite("mysql")

    # Combined suite
    from ...core import BenchmarkSuite
    DatabaseTuningSuite = BenchmarkSuite(
        "database_tuning",
        {
            **PostgreSQLTuningSuite.functions,
            **MySQLTuningSuite.functions
        }
    )
except Exception as e:
    warnings.warn(f"Could not create database tuning suites: {e}")
    DatabaseTuningSuite = None
    PostgreSQLTuningSuite = None
    MySQLTuningSuite = None


__all__ = [
    # Main classes
    "DatabaseTuningFunction",
    "create_database_tuning_suite",

    # Default suites
    "DatabaseTuningSuite",
    "PostgreSQLTuningSuite",
    "MySQLTuningSuite",

    # Utility functions (advanced usage)
    "get_postgresql_knobs",
    "get_mysql_knobs",
    "get_default_knob_config",
    "validate_knob_config",
    "get_knob_documentation",
    "SpaceConverter",
    "DatabaseEvaluator",
]
