"""
Default knob configurations for different database systems.

This module provides default tunable knob configurations for PostgreSQL, MySQL,
and other database systems.
"""

from typing import Dict, Any


def get_postgresql_knobs() -> Dict[str, Dict[str, Any]]:
    """
    Get default tunable knobs for PostgreSQL.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping knob names to their specifications.
        Each knob spec contains: type, min/max or choices, default, description, category.
    """
    return {
        "shared_buffers_mb": {
            "type": "int",
            "min": 128,
            "max": 16384,
            "default": 1024,
            "description": "Size of shared memory buffers (MB)",
            "category": "memory"
        },
        "effective_cache_size_mb": {
            "type": "int",
            "min": 256,
            "max": 65536,
            "default": 4096,
            "description": "Planner's assumption about effective cache size (MB)",
            "category": "memory"
        },
        "work_mem_mb": {
            "type": "int",
            "min": 1,
            "max": 2048,
            "default": 4,
            "description": "Memory for sort and hash operations (MB)",
            "category": "memory"
        },
        "max_connections": {
            "type": "int",
            "min": 10,
            "max": 1000,
            "default": 100,
            "description": "Maximum number of concurrent connections",
            "category": "connection"
        },
        "random_page_cost": {
            "type": "float",
            "min": 0.1,
            "max": 10.0,
            "default": 4.0,
            "description": "Planner's estimate of random page access cost",
            "category": "planner"
        },
        "effective_io_concurrency": {
            "type": "int",
            "min": 0,
            "max": 1000,
            "default": 1,
            "description": "Number of concurrent disk I/O operations",
            "category": "io"
        },
        "checkpoint_completion_target": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
            "description": "Target for checkpoint completion as fraction of checkpoint interval",
            "category": "wal"
        },
        "default_statistics_target": {
            "type": "int",
            "min": 10,
            "max": 10000,
            "default": 100,
            "description": "Default statistics target for table columns",
            "category": "planner"
        }
    }


def get_mysql_knobs() -> Dict[str, Dict[str, Any]]:
    """
    Get default tunable knobs for MySQL.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping knob names to their specifications.
    """
    return {
        "innodb_buffer_pool_size_mb": {
            "type": "int",
            "min": 128,
            "max": 32768,
            "default": 1024,
            "description": "InnoDB buffer pool size (MB)",
            "category": "memory"
        },
        "innodb_log_file_size_mb": {
            "type": "int",
            "min": 4,
            "max": 4096,
            "default": 48,
            "description": "Size of each InnoDB log file (MB)",
            "category": "io"
        },
        "max_connections": {
            "type": "int",
            "min": 10,
            "max": 10000,
            "default": 151,
            "description": "Maximum number of concurrent connections",
            "category": "connection"
        },
        "innodb_io_capacity": {
            "type": "int",
            "min": 100,
            "max": 20000,
            "default": 200,
            "description": "I/O operations per second for InnoDB background tasks",
            "category": "io"
        },
        "query_cache_size_mb": {
            "type": "int",
            "min": 0,
            "max": 1024,
            "default": 0,
            "description": "Query cache size (MB), 0 means disabled",
            "category": "cache"
        }
    }


def get_default_knob_config(database_system: str) -> Dict[str, Dict[str, Any]]:
    """
    Get default knob configuration for a database system.

    Parameters
    ----------
    database_system : str
        Database system name (e.g., "postgresql", "mysql")

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Default knob configuration for the specified database system

    Raises
    ------
    ValueError
        If no default configuration exists for the specified database system
    """
    db_system = database_system.lower()

    if db_system in ["postgresql", "postgres"]:
        return get_postgresql_knobs()
    elif db_system == "mysql":
        return get_mysql_knobs()
    else:
        raise ValueError(
            f"No default knob configuration for {database_system}. "
            f"Please provide knob_config parameter."
        )


def validate_knob_config(knob_info: Dict[str, Dict[str, Any]]) -> None:
    """
    Validate knob configuration.

    Parameters
    ----------
    knob_info : Dict[str, Dict[str, Any]]
        Knob configuration to validate

    Raises
    ------
    ValueError
        If knob configuration is invalid
    """
    for knob_name, knob_spec in knob_info.items():
        if "type" not in knob_spec:
            raise ValueError(f"Knob {knob_name} missing 'type' field")

        knob_type = knob_spec["type"]
        if knob_type not in ["int", "float", "enum", "bool"]:
            raise ValueError(
                f"Knob {knob_name} has invalid type {knob_type}. "
                f"Must be one of: int, float, enum, bool"
            )

        if knob_type in ["int", "float"]:
            if "min" not in knob_spec or "max" not in knob_spec:
                raise ValueError(
                    f"Knob {knob_name} of type {knob_type} must have 'min' and 'max' fields"
                )

        if knob_type == "enum":
            if "choices" not in knob_spec or len(knob_spec["choices"]) == 0:
                raise ValueError(
                    f"Knob {knob_name} of type enum must have non-empty 'choices' field"
                )


def get_knob_documentation(
    knob_info: Dict[str, Dict[str, Any]],
    database_system: str,
    workload_name: str,
    total_dims: int
) -> str:
    """
    Generate formatted documentation for knob configuration.

    Parameters
    ----------
    knob_info : Dict[str, Dict[str, Any]]
        Knob configuration
    database_system : str
        Database system name
    workload_name : str
        Workload name
    total_dims : int
        Total number of continuous dimensions

    Returns
    -------
    str
        Formatted documentation string
    """
    doc_lines = [
        f"Database Knob Configuration for {database_system.upper()} - {workload_name.upper()}",
        "=" * 80,
        f"\nTotal Tunable Knobs: {len(knob_info)}",
        f"Total Dimensions (continuous): {total_dims}\n"
    ]

    # Group by category if available
    categories = {}
    for knob_name, knob_spec in knob_info.items():
        category = knob_spec.get("category", "other")
        if category not in categories:
            categories[category] = []
        categories[category].append((knob_name, knob_spec))

    for category, knobs in sorted(categories.items()):
        doc_lines.append(f"\n{category.upper()} KNOBS:")
        doc_lines.append("-" * 80)

        for knob_name, knob_spec in knobs:
            knob_type = knob_spec["type"]
            default = knob_spec.get("default", "N/A")
            desc = knob_spec.get("description", "No description")

            doc_lines.append(f"\n{knob_name}:")
            doc_lines.append(f"  Type: {knob_type}")

            if knob_type in ["int", "float"]:
                doc_lines.append(f"  Range: [{knob_spec['min']}, {knob_spec['max']}]")
            elif knob_type == "enum":
                doc_lines.append(f"  Choices: {knob_spec['choices']}")
            elif knob_type == "bool":
                doc_lines.append(f"  Values: True or False")

            doc_lines.append(f"  Default: {default}")
            doc_lines.append(f"  Description: {desc}")

    return "\n".join(doc_lines)
