"""
Database tuning benchmark suite creation utilities.
"""

import warnings
from typing import List, Optional, Dict, Any

from ...core import BenchmarkSuite
from .core import DatabaseTuningFunction


def create_database_tuning_suite(
    database_system: str = "postgresql",
    workloads: Optional[List[str]] = None,
    knob_configs: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
) -> BenchmarkSuite:
    """
    Create database tuning suite with specified workloads.

    Parameters
    ----------
    database_system : str
        Database system to benchmark ("postgresql", "mysql", etc.)
    workloads : List[str], optional
        List of workload names. If None, uses default workloads.
    knob_configs : Dict[str, Dict[str, Dict[str, Any]]], optional
        Custom knob configurations per workload.
        Format: {workload_name: {knob_name: knob_spec}}

    Returns
    -------
    BenchmarkSuite
        Suite containing database tuning benchmark functions

    Examples
    --------
    >>> # Create suite with default PostgreSQL workloads
    >>> suite = create_database_tuning_suite("postgresql")
    >>>
    >>> # Create suite with custom workloads
    >>> suite = create_database_tuning_suite(
    ...     "postgresql",
    ...     workloads=["tpcc", "tpch"]
    ... )
    """
    if workloads is None:
        # Default workloads for common benchmarks
        workloads = ["tpcc", "tpch", "ycsb"]

    functions = {}

    for workload in workloads:
        try:
            # Get knob config for this workload if provided
            if knob_configs and workload in knob_configs:
                knob_config = knob_configs[workload]
            else:
                knob_config = None  # Use default

            func_name = f"{database_system}_{workload}"
            functions[func_name] = DatabaseTuningFunction(
                workload_name=workload,
                database_system=database_system,
                knob_config=knob_config
            )
        except Exception as e:
            warnings.warn(f"Could not create {func_name}: {e}")

    suite_name = f"{database_system}_tuning"
    return BenchmarkSuite(suite_name, functions)
