# BenchBase Integration Architecture

This document provides a visual overview of how BenchBase is integrated with BOMegaBench.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Code / Optimizer                        │
│                                                                      │
│  Example: Bayesian Optimization with BoTorch                       │
│  - SingleTaskGP                                                     │
│  - ExpectedImprovement                                              │
│  - optimize_acqf                                                    │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         │ torch.Tensor [0,1]^d
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    DatabaseTuningFunction                           │
│                  (BenchmarkFunction subclass)                       │
│                                                                      │
│  Responsibilities:                                                  │
│  - Maintain continuous [0,1] space                                  │
│  - One-hot encoding for discrete knobs                              │
│  - Convert continuous → discrete knob values                        │
│  - Orchestrate evaluation workflow                                 │
│                                                                      │
│  Key Methods:                                                       │
│  - _evaluate_true(X) → performance metric                          │
│  - _convert_continuous_to_discrete(x) → knob_config               │
│  - _convert_discrete_to_continuous(config) → x                     │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         │ Dict[str, Any] (knob values)
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        BenchBaseWrapper                             │
│                       (Python Class)                                │
│                                                                      │
│  Responsibilities:                                                  │
│  - Apply database configuration knobs                               │
│  - Generate XML configuration files                                 │
│  - Execute BenchBase via subprocess                                 │
│  - Parse result files (CSV/JSON)                                    │
│  - Error handling and timeouts                                      │
│                                                                      │
│  Key Methods:                                                       │
│  - apply_database_config(knobs) → success/failure                  │
│  - run_benchmark(...) → metrics dict                               │
│  - _create_config_xml(...) → Path to XML                           │
│  - _parse_results(...) → metrics dict                              │
└────┬────────────────────────────────────┬─────────────────────────────┘
     │                                    │
     │ SQL commands                       │ subprocess.run()
     │ (ALTER SYSTEM, SET GLOBAL)         │ java -jar benchbase.jar ...
     │                                    │
     ↓                                    ↓
┌──────────────────────┐      ┌──────────────────────────────┐
│     Database         │      │      BenchBase (Java)        │
│  (PostgreSQL/MySQL)  │◄─────┤    Multi-threaded            │
│                      │ JDBC │    Benchmark Executor        │
│  Configuration:      │      │                              │
│  - shared_buffers    │      │  Workloads:                  │
│  - work_mem          │      │  - TPC-C                     │
│  - max_connections   │      │  - TPC-H                     │
│  - ...               │      │  - YCSB                      │
│                      │      │  - 15+ more                  │
└──────────────────────┘      └──────────────┬───────────────┘
                                             │
                                             │ CSV/JSON output
                                             │
                              ┌──────────────▼──────────────┐
                              │      results/ directory     │
                              │                             │
                              │  - tpcc_*.csv               │
                              │  - tpcc_*.summary.json      │
                              │                             │
                              │  Metrics:                   │
                              │  - Latency (percentiles)    │
                              │  - Throughput (txn/s)       │
                              │  - Per-transaction stats    │
                              └─────────────────────────────┘
```

## Data Flow

### 1. Optimization Request
```
User → BoTorch Optimizer
     → Generate candidate X ∈ [0,1]^d
```

### 2. Knob Conversion
```
X (continuous) → DatabaseTuningFunction
               → _convert_continuous_to_discrete()
               → knob_config = {
                   "shared_buffers_mb": 2048,
                   "work_mem_mb": 32,
                   "max_connections": 200
                 }
```

### 3. Configuration Application
```
knob_config → BenchBaseWrapper.apply_database_config()
            → SQL: ALTER SYSTEM SET shared_buffers = '2048MB'
            → SQL: ALTER SYSTEM SET work_mem = '32MB'
            → SQL: SELECT pg_reload_conf()
```

### 4. Benchmark Execution
```
BenchBaseWrapper.run_benchmark()
  → _create_config_xml() → /tmp/tpcc_config_12345.xml
  → subprocess.run([
      "java", "-jar", "benchbase.jar",
      "-b", "tpcc",
      "-c", "tpcc_config_12345.xml",
      "--execute=true"
    ])
  → Wait for completion (with timeout)
```

### 5. Result Parsing
```
BenchBase → writes to results/tpcc_12345.csv
BenchBaseWrapper → _parse_results()
                 → Extract metrics:
                   {
                     "throughput_txns_sec": 1234.5,
                     "avg_latency_ms": 8.2,
                     "p95_latency_ms": 15.6,
                     "p99_latency_ms": 23.1
                   }
```

### 6. Return to Optimizer
```
metrics → DatabaseTuningFunction
        → Return performance (latency or 1/throughput)
        → BoTorch receives y value
        → Update GP model
        → Next iteration
```

## File Structure

```
BOMegaBench/
├── bomegabench/
│   └── functions/
│       ├── database_tuning.py         # Main integration module
│       └── benchbase_wrapper.py       # Python wrapper for BenchBase
│
├── examples/
│   └── database_tuning/
│       ├── BENCHBASE_SETUP.md         # Complete setup guide
│       ├── QUICKSTART.md              # 15-minute quickstart
│       ├── README.md                  # Examples directory guide
│       ├── ARCHITECTURE.md            # This file
│       └── example_benchbase_integration.py  # Usage examples
│
├── BENCHBASE_INTEGRATION_REPORT.md    # Detailed technical report
└── BENCHBASE_INTEGRATION_SUMMARY.md   # Executive summary
```

## Component Responsibilities

### DatabaseTuningFunction
**Purpose**: Adapt database tuning to BOMegaBench/BoTorch interface

**Responsibilities**:
- Maintain continuous optimization space [0,1]^d
- Handle one-hot encoding for discrete parameters
- Convert between continuous and discrete representations
- Orchestrate evaluation workflow
- Provide metadata about the optimization problem

**Does NOT**:
- Directly interact with database
- Execute benchmarks
- Parse results

### BenchBaseWrapper
**Purpose**: Python interface to BenchBase Java application

**Responsibilities**:
- Apply database configurations via SQL
- Generate BenchBase XML configuration files
- Execute BenchBase via subprocess
- Parse benchmark results
- Handle errors and timeouts

**Does NOT**:
- Maintain optimization state
- Convert between continuous/discrete
- Make optimization decisions

### BenchBase (Java)
**Purpose**: Execute database benchmarks

**Responsibilities**:
- Connect to database via JDBC
- Load and execute benchmark workloads
- Generate synthetic data
- Measure performance (latency, throughput)
- Output results to CSV/JSON

**Does NOT**:
- Apply database configurations
- Optimize anything
- Interact with Python directly

## Knob Encoding Strategy

### Problem
Database knobs have mixed types:
- **Integer**: `shared_buffers_mb` ∈ {128, 129, ..., 16384}
- **Float**: `checkpoint_completion_target` ∈ [0.0, 1.0]
- **Enum**: `log_level` ∈ {DEBUG, INFO, WARNING, ERROR}
- **Boolean**: `enable_seqscan` ∈ {True, False}

Bayesian Optimization requires continuous [0,1] space.

### Solution: One-Hot Encoding + Continuous

#### Float Parameters
Map directly to [0,1]:
```python
# Input: x ∈ [0,1]
# Output: value ∈ [min, max]
value = min + x * (max - min)

# Example: checkpoint_completion_target
# x = 0.3 → 0.0 + 0.3 * (1.0 - 0.0) = 0.3
```

#### Integer Parameters
One-hot encoding:
```python
# shared_buffers_mb ∈ {128, 256, 512, 1024, 2048}
# → 5 dimensions in [0,1]

# Dimensions: [x1, x2, x3, x4, x5]
# Select: argmax([x1, x2, x3, x4, x5])

# Example: [0.2, 0.8, 0.1, 0.3, 0.5] → argmax=1 → 256MB
```

#### Categorical Parameters
One-hot encoding:
```python
# log_level ∈ {DEBUG, INFO, WARNING, ERROR}
# → 4 dimensions in [0,1]

# Dimensions: [x1, x2, x3, x4]
# Select: argmax([x1, x2, x3, x4])

# Example: [0.1, 0.3, 0.9, 0.2] → argmax=2 → WARNING
```

#### Boolean Parameters
One-hot encoding with 2 dimensions:
```python
# enable_seqscan ∈ {False, True}
# → 2 dimensions in [0,1]

# Dimensions: [x1, x2]
# Select: argmax([x1, x2])

# Example: [0.3, 0.7] → argmax=1 → True
```

### Example: Full Configuration

**Knobs**:
- `shared_buffers_mb`: int, {512, 1024, 2048, 4096}
- `work_mem_mb`: int, {4, 8, 16, 32}
- `checkpoint_completion_target`: float, [0.0, 1.0]

**Continuous Space**:
```
Dimension 0: shared_buffers=512  (int one-hot 1/4)
Dimension 1: shared_buffers=1024 (int one-hot 2/4)
Dimension 2: shared_buffers=2048 (int one-hot 3/4)
Dimension 3: shared_buffers=4096 (int one-hot 4/4)
Dimension 4: work_mem=4          (int one-hot 1/4)
Dimension 5: work_mem=8          (int one-hot 2/4)
Dimension 6: work_mem=16         (int one-hot 3/4)
Dimension 7: work_mem=32         (int one-hot 4/4)
Dimension 8: checkpoint_completion_target (float)

Total: 9 dimensions, all in [0,1]
```

**Example Input**:
```python
X = [0.2, 0.9, 0.3, 0.1, 0.1, 0.2, 0.8, 0.4, 0.75]
```

**Decoding**:
```python
shared_buffers_mb = argmax([0.2, 0.9, 0.3, 0.1]) = 1 → 1024
work_mem_mb = argmax([0.1, 0.2, 0.8, 0.4]) = 2 → 16
checkpoint_completion_target = 0.0 + 0.75 * (1.0 - 0.0) = 0.75

Configuration = {
  "shared_buffers_mb": 1024,
  "work_mem_mb": 16,
  "checkpoint_completion_target": 0.75
}
```

## Performance Considerations

### Evaluation Cost
Each evaluation requires:
1. **Apply Configuration**: ~2 seconds (database reload)
2. **JVM Startup**: ~3 seconds
3. **Benchmark Execution**: 60+ seconds (configurable)
4. **Result Parsing**: <1 second

**Total**: ~65 seconds per evaluation

### Optimization Implications
- **Sample Efficiency Critical**: Use Bayesian Optimization, not random search
- **Batch Evaluation**: Not currently supported (could parallelize)
- **Multi-Fidelity**: Could use short runs for exploration (future)

### Dimensionality
Default PostgreSQL config: 8 knobs
- 2 float parameters → 2 dimensions
- 6 integer parameters (avg 1000 values each) → 6000 dimensions
- **Total**: ~6002 dimensions

**Recommendation**: Start with 3-5 knobs to reduce dimensionality

## Error Handling

### Database Connection Failures
```python
try:
    wrapper.apply_database_config(knobs)
except Exception as e:
    warnings.warn(f"Failed to apply config: {e}")
    # Continue with current config
```

### Benchmark Execution Failures
```python
try:
    result = wrapper.run_benchmark(...)
except subprocess.TimeoutExpired:
    return float('inf')  # Penalty for timeout
except Exception as e:
    return float('inf')  # Penalty for failure
```

### Result Parsing Failures
```python
try:
    metrics = wrapper._parse_results(benchmark)
except FileNotFoundError:
    warnings.warn("Results file not found")
    return default_metrics
```

## Extensibility

### Adding New Databases
```python
# In BenchBaseWrapper:
JDBC_DRIVERS["newdb"] = "com.newdb.Driver"

def _apply_newdb_config(self, knobs):
    import newdb_driver
    conn = newdb_driver.connect(...)
    # Apply configuration
```

### Adding New Workloads
```python
# Just use the benchmark name
func = DatabaseTuningFunction(
    workload_name="custom_benchmark",  # Any BenchBase benchmark
    ...
)
```

### Adding New Knobs
```python
custom_knobs = {
    "new_knob": {
        "type": "int",
        "min": 0,
        "max": 100,
        "default": 50,
        "description": "New knob description"
    }
}

func = DatabaseTuningFunction(
    knob_config=custom_knobs,
    ...
)
```

## Security Considerations

### Database Credentials
**Development**:
```python
# Acceptable for examples
db_password = "password"
```

**Production**:
```python
# Use environment variables
import os
db_password = os.environ.get("DB_PASSWORD")

# Or use secret management
from secret_manager import get_secret
db_password = get_secret("database_password")
```

### SQL Injection
Current implementation uses parameterized queries:
```python
# Safe
cursor.execute("ALTER SYSTEM SET shared_buffers = %s", (value,))

# NOT used (unsafe)
cursor.execute(f"ALTER SYSTEM SET shared_buffers = {value}")
```

### Subprocess Security
```python
# Safe: list form prevents shell injection
subprocess.run([
    "java", "-jar", "benchbase.jar",
    "-b", benchmark,
    "-c", config_path
])

# NOT used (unsafe)
subprocess.run(f"java -jar benchbase.jar -b {benchmark}", shell=True)
```

## Monitoring and Debugging

### Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Applying configuration: {knobs}")
logger.info(f"Running benchmark: {benchmark}")
logger.info(f"Results: {metrics}")
```

### Benchmarking History
```python
# Save configurations and results
with open("optimization_history.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow([timestamp, config, performance])
```

### Database Monitoring
```sql
-- PostgreSQL: Check current settings
SELECT name, setting FROM pg_settings WHERE name LIKE '%buffer%';

-- MySQL: Check current settings
SHOW VARIABLES LIKE '%buffer%';
```

## References

- BenchBase: https://github.com/cmu-db/benchbase
- BoTorch: https://botorch.org
- BOMegaBench: (Main repository)
- PostgreSQL Configuration: https://www.postgresql.org/docs/current/runtime-config.html
- MySQL Configuration: https://dev.mysql.com/doc/refman/8.0/en/server-system-variables.html
