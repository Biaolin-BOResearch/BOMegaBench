
# Database Knob Configuration Guide

## Overview

This document provides comprehensive documentation for database configuration knobs (parameters) supported in BOMegaBench's Database Tuning module.

## Table of Contents

1. [Parameter Types](#parameter-types)
2. [PostgreSQL Knobs](#postgresql-knobs)
3. [MySQL Knobs](#mysql-knobs)
4. [Custom Knob Configuration](#custom-knob-configuration)
5. [Usage Examples](#usage-examples)

---

## Parameter Types

BOMegaBench supports four types of database configuration parameters:

### 1. Integer (`int`)
- **Description**: Whole number parameters
- **Representation**: Single continuous dimension in [0,1], mapped linearly to [min, max]
- **Conversion**: Value is rounded to nearest integer
- **Example**: `max_connections`, `shared_buffers_mb`

### 2. Float (`float`)
- **Description**: Continuous numeric parameters
- **Representation**: Single continuous dimension in [0,1], mapped linearly to [min, max]
- **Conversion**: Direct linear scaling
- **Example**: `random_page_cost`, `checkpoint_completion_target`

### 3. Enum (`enum`)
- **Description**: Categorical parameters with fixed set of choices
- **Representation**: Single continuous dimension in [0,1], divided into equal ranges
- **Conversion**: Value range determines selected choice
- **Example**: `log_level` with choices ["DEBUG", "INFO", "WARNING", "ERROR"]

### 4. Boolean (`bool`)
- **Description**: Binary on/off parameters
- **Representation**: Single continuous dimension in [0,1]
- **Conversion**: Threshold at 0.5 (< 0.5 → False, >= 0.5 → True)
- **Example**: `enable_seqscan`, `enable_indexscan`

---

## PostgreSQL Knobs

### Memory Configuration

#### `shared_buffers_mb`
- **Type**: Integer
- **Range**: [128, 16384] MB
- **Default**: 1024 MB
- **Description**: Amount of memory PostgreSQL uses for shared memory buffers. This is the main memory area for caching data.
- **Impact**: Higher values improve performance for frequently accessed data but require more system RAM.
- **Recommendation**: Set to 25% of total system RAM for dedicated database servers.

#### `effective_cache_size_mb`
- **Type**: Integer
- **Range**: [256, 65536] MB
- **Default**: 4096 MB
- **Description**: Estimate of memory available for disk caching by the OS and PostgreSQL. Used by query planner.
- **Impact**: Affects query planning decisions. Higher values make index scans more likely.
- **Recommendation**: Set to 50-75% of total system RAM.

#### `work_mem_mb`
- **Type**: Integer
- **Range**: [1, 2048] MB
- **Default**: 4 MB
- **Description**: Memory allocated for sort operations and hash tables before writing to temp files.
- **Impact**: Higher values speed up complex queries with sorts/joins but can cause memory issues with many concurrent queries.
- **Recommendation**: Start with 4-64 MB, adjust based on query complexity and concurrency.

### Connection Configuration

#### `max_connections`
- **Type**: Integer
- **Range**: [10, 1000]
- **Default**: 100
- **Description**: Maximum number of concurrent database connections.
- **Impact**: Each connection consumes memory (~10 MB). Too high can exhaust system resources.
- **Recommendation**: Set based on expected concurrent users. Use connection pooling for high-traffic applications.

### Query Planner Configuration

#### `random_page_cost`
- **Type**: Float
- **Range**: [0.1, 10.0]
- **Default**: 4.0
- **Description**: Planner's estimate of cost for random disk page access (relative to sequential access).
- **Impact**: Lower values favor index scans. Higher values favor sequential scans.
- **Recommendation**: Use 1.1-2.0 for SSDs, 2.0-4.0 for HDDs.

#### `default_statistics_target`
- **Type**: Integer
- **Range**: [10, 10000]
- **Default**: 100
- **Description**: Default amount of statistics collected by ANALYZE for each table column.
- **Impact**: Higher values improve query planning accuracy but increase ANALYZE time.
- **Recommendation**: Use 100-500 for most workloads. Increase for complex queries.

### I/O Configuration

#### `effective_io_concurrency`
- **Type**: Integer
- **Range**: [0, 1000]
- **Default**: 1
- **Description**: Number of concurrent disk I/O operations PostgreSQL expects the system can handle.
- **Impact**: Affects bitmap heap scan performance. More relevant for RAID systems.
- **Recommendation**: Set to number of separate drives in RAID array. Use 200+ for modern SSDs.

### WAL (Write-Ahead Logging) Configuration

#### `checkpoint_completion_target`
- **Type**: Float
- **Range**: [0.0, 1.0]
- **Default**: 0.5
- **Description**: Target for checkpoint completion as fraction of checkpoint interval.
- **Impact**: Higher values spread checkpoint I/O over more time, reducing spikes.
- **Recommendation**: Use 0.7-0.9 for most workloads to smooth I/O.

---

## MySQL Knobs

### Memory Configuration

#### `innodb_buffer_pool_size_mb`
- **Type**: Integer
- **Range**: [128, 32768] MB
- **Default**: 1024 MB
- **Description**: Size of InnoDB buffer pool for caching data and indexes.
- **Impact**: Most important MySQL tuning parameter. Higher is better for read-heavy workloads.
- **Recommendation**: Set to 70-80% of total RAM for dedicated database servers.

#### `innodb_log_file_size_mb`
- **Type**: Integer
- **Range**: [4, 4096] MB
- **Default**: 48 MB
- **Description**: Size of each InnoDB redo log file.
- **Impact**: Larger values improve write performance but increase crash recovery time.
- **Recommendation**: Use 256-512 MB for write-heavy workloads.

### Connection Configuration

#### `max_connections`
- **Type**: Integer
- **Range**: [10, 10000]
- **Default**: 151
- **Description**: Maximum number of concurrent client connections.
- **Impact**: Each connection uses memory. Too high can cause swapping.
- **Recommendation**: Set based on concurrent users. Consider connection pooling.

### I/O Configuration

#### `innodb_io_capacity`
- **Type**: Integer
- **Range**: [100, 20000]
- **Default**: 200
- **Description**: I/O operations per second available to InnoDB background tasks.
- **Impact**: Should match storage system's IOPS capability.
- **Recommendation**: Use 2000-5000 for SSDs, 200-400 for HDDs.

### Cache Configuration

#### `query_cache_size_mb`
- **Type**: Integer
- **Range**: [0, 1024] MB
- **Default**: 0 (disabled)
- **Description**: Size of query result cache. Disabled by default in MySQL 8.0+.
- **Impact**: Can improve read performance for identical queries but adds overhead.
- **Recommendation**: Usually better to use application-level caching. Set to 0 to disable.

---

## Custom Knob Configuration

### Defining Custom Knobs

You can define custom knob configurations for your specific database setup:

```python
from bomegabench.functions.database_tuning import DatabaseTuningFunction

# Define custom knob configuration
custom_knobs = {
    "my_int_param": {
        "type": "int",
        "min": 1,
        "max": 100,
        "default": 10,
        "description": "My custom integer parameter",
        "category": "custom"
    },
    "my_float_param": {
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.5,
        "description": "My custom float parameter",
        "category": "custom"
    },
    "my_enum_param": {
        "type": "enum",
        "choices": ["option1", "option2", "option3"],
        "default": "option1",
        "description": "My custom enum parameter",
        "category": "custom"
    },
    "my_bool_param": {
        "type": "bool",
        "default": True,
        "description": "My custom boolean parameter",
        "category": "custom"
    }
}

# Create function with custom knobs
func = DatabaseTuningFunction(
    workload_name="my_workload",
    database_system="postgresql",
    knob_config=custom_knobs
)
```

### Required Fields

Each knob specification must include:
- `type`: One of ["int", "float", "enum", "bool"]
- `description`: Human-readable description

#### Additional fields by type:
- **int/float**: `min`, `max`
- **enum**: `choices` (list of valid options)
- **bool**: No additional required fields

#### Optional fields:
- `default`: Default value for the knob
- `category`: Category for grouping (e.g., "memory", "io", "planner")

---

## Usage Examples

### Example 1: Basic Usage with Default Knobs

```python
import bomegabench as bmb
import torch

# Get database tuning function with default PostgreSQL knobs
func = bmb.get_function("postgresql_tpcc", suite="database_tuning")

# View knob documentation
print(func.get_knob_documentation())

# Sample random configuration in [0,1] space
X = torch.rand(1, func.dim)

# Evaluate performance (requires actual database integration)
performance = func(X)
print(f"Performance metric: {performance.item()}")

# Get metadata
print(f"Database: {func.metadata['database_system']}")
print(f"Workload: {func.metadata['workload']}")
print(f"Total knobs: {func.metadata['total_knobs']}")
```

### Example 2: Working with Knob Configurations

```python
from bomegabench.functions.database_tuning import DatabaseTuningFunction
import numpy as np

# Create function
func = DatabaseTuningFunction("tpcc", "postgresql")

# Convert a known good configuration to continuous space
good_config = {
    "shared_buffers_mb": 4096,
    "effective_cache_size_mb": 16384,
    "work_mem_mb": 64,
    "max_connections": 200,
    "random_page_cost": 1.5,
    "effective_io_concurrency": 200,
    "checkpoint_completion_target": 0.9,
    "default_statistics_target": 500
}

# Convert to [0,1] space
X_continuous = func._convert_discrete_to_continuous(good_config)
print(f"Continuous representation: {X_continuous}")

# Convert back to discrete
config_back = func._convert_continuous_to_discrete(X_continuous)
print(f"Recovered configuration: {config_back}")
```

### Example 3: Bayesian Optimization

```python
import bomegabench as bmb
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# Get function
func = bmb.get_function("postgresql_tpcc", suite="database_tuning")

# Initialize with random samples
n_init = 5
X_init = torch.rand(n_init, func.dim)
Y_init = func(X_init).unsqueeze(-1)

# Bayesian optimization loop
for iteration in range(20):
    # Fit GP model
    gp = SingleTaskGP(X_init, Y_init)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Optimize acquisition function
    EI = ExpectedImprovement(gp, best_f=Y_init.min())
    candidate, acq_value = optimize_acqf(
        EI,
        bounds=torch.stack([torch.zeros(func.dim), torch.ones(func.dim)]),
        q=1,
        num_restarts=10,
        raw_samples=512,
    )

    # Evaluate candidate
    Y_new = func(candidate).unsqueeze(-1)

    # Update data
    X_init = torch.cat([X_init, candidate])
    Y_init = torch.cat([Y_init, Y_new])

    print(f"Iteration {iteration+1}, Best: {Y_init.min().item():.4f}")

# Get best configuration
best_idx = Y_init.argmin()
best_X = X_init[best_idx].numpy()
best_config = func._convert_continuous_to_discrete(best_X)
print(f"\nBest configuration found:")
for knob, value in best_config.items():
    print(f"  {knob}: {value}")
```

### Example 4: Custom Workload Integration

To integrate with actual database benchmarking system, override `_evaluate_configuration`:

```python
from bomegabench.functions.database_tuning import DatabaseTuningFunction
import subprocess
import json

class BenchBaseIntegration(DatabaseTuningFunction):
    """Integration with BenchBase benchmarking tool."""

    def __init__(self, workload_name, database_system, **kwargs):
        super().__init__(workload_name, database_system, **kwargs)
        # Initialize database connection, BenchBase, etc.
        self.db_config_file = f"/path/to/db/config/{database_system}.conf"
        self.benchbase_jar = "/path/to/benchbase.jar"

    def _evaluate_configuration(self, knob_config):
        """Evaluate configuration using BenchBase."""
        # 1. Apply knob configuration to database
        self._apply_config_to_database(knob_config)

        # 2. Run BenchBase benchmark
        result = self._run_benchbase()

        # 3. Return performance metric
        if self.performance_metric == "latency":
            return result["avg_latency"]
        else:
            return 1.0 / result["throughput"]  # Convert to minimization

    def _apply_config_to_database(self, knob_config):
        """Apply configuration to database."""
        # Update postgresql.conf or my.cnf
        config_lines = []
        for knob, value in knob_config.items():
            config_lines.append(f"{knob} = {value}")

        with open(self.db_config_file, 'w') as f:
            f.write("\n".join(config_lines))

        # Restart database to apply changes
        subprocess.run(["sudo", "systemctl", "restart", f"{self.database_system}"])

    def _run_benchbase(self):
        """Run BenchBase and collect results."""
        cmd = [
            "java", "-jar", self.benchbase_jar,
            "-b", self.workload_name,
            "-c", "config.xml",
            "--execute=true"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse results (format depends on BenchBase output)
        output = json.loads(result.stdout)
        return {
            "throughput": output["throughput"],
            "avg_latency": output["avg_latency"]
        }

# Use custom integration
func = BenchBaseIntegration("tpcc", "postgresql")
```

---

## Performance Metrics

### Latency-based (minimize)
- **Metric**: Average query latency in seconds
- **Lower is better**
- **Suitable for**: OLTP workloads, interactive queries

### Throughput-based (maximize → converted to minimize)
- **Metric**: 1 / throughput (transactions per second)
- **Lower is better** (higher throughput → lower metric value)
- **Suitable for**: Batch processing, high-volume workloads

---

## Best Practices

### 1. Start with Default Configurations
Use the built-in default knobs as a baseline before customizing.

### 2. Consider Hardware Constraints
- Set memory-related knobs based on available RAM
- Set I/O knobs based on storage system capabilities (SSD vs HDD)

### 3. Use Categories for Organization
Group related knobs by category for better understanding and maintenance.

### 4. Document Custom Knobs
Always provide clear descriptions for custom knobs to aid in interpretation.

### 5. Test Configurations Safely
- Use staging environment for initial tuning
- Monitor system resources during evaluation
- Implement timeouts for runaway configurations

### 6. Consider Interdependencies
Some knobs interact with each other:
- `shared_buffers` + `work_mem` × `max_connections` should not exceed total RAM
- `checkpoint_completion_target` works with checkpoint interval settings

---

## Troubleshooting

### Issue: Dimension count mismatch
**Cause**: Changed knob configuration but didn't reinitialize function
**Solution**: Create new function instance with updated configuration

### Issue: Invalid configuration error
**Cause**: Converted value outside valid range
**Solution**: Check min/max bounds and ensure proper normalization

### Issue: Slow evaluation
**Cause**: Database benchmarking is inherently slow
**Solution**:
- Reduce benchmark duration for initial exploration
- Use multi-fidelity optimization if supported
- Run evaluations in parallel if multiple database instances available

---

## References

- PostgreSQL Documentation: https://www.postgresql.org/docs/current/runtime-config.html
- MySQL Documentation: https://dev.mysql.com/doc/refman/8.0/en/server-system-variables.html
- BenchBase: https://github.com/cmu-db/benchbase

---

**Last Updated**: October 2025
