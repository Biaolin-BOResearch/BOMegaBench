# Database Knob Tuning Integration Guide for BOMegaBench

## Quick Reference: How to Integrate Database Knob Tuning Tasks

This guide provides a step-by-step approach to integrate database knob tuning benchmarks into BOMegaBench.

---

## 1. Architecture Overview

BOMegaBench uses a modular architecture with three main layers:

```
User API Layer (list_suites, get_function, etc.)
        ↓
Registry Layer (_SUITES dictionary in registry.py)
        ↓
Function Wrapper Layer (BenchmarkFunction subclasses)
        ↓
External Library/System Layer (BenchBase, databases, etc.)
```

For database tuning, you'll create wrapper classes that sit between the registry and the actual database system.

---

## 2. Three Key Files to Understand and Modify

### File 1: `bomegabench/functions/database_tuning.py` (NEW - Create this)
**Purpose**: Core wrapper classes for database tuning functions

**What you'll implement**:
- `DatabaseTuningFunction` - Base class inheriting from `BenchmarkFunction`
- Wrapper classes for specific DB systems (e.g., `PostgreSQLFunction`, `MySQLFunction`)
- Suite factory functions

### File 2: `bomegabench/functions/registry.py` (MODIFY)
**Purpose**: Register your new suite in the global function registry

**What to add**:
```python
# Around line 30-43 where other imports are:
try:
    from .database_tuning import (
        DatabaseTuningSuite,
        BENCHBASE_AVAILABLE
    )
    DATABASE_TUNING_AVAILABLE = True
except ImportError:
    DATABASE_TUNING_AVAILABLE = False
    DatabaseTuningSuite = None

# Around line 49-87 where _SUITES is populated:
if DATABASE_TUNING_AVAILABLE and DatabaseTuningSuite is not None:
    _SUITES.update({
        "database_tuning": DatabaseTuningSuite,
    })
```

### File 3: `bomegabench/functions/__init__.py` (MODIFY)
**Purpose**: Export new suite classes

**What to add**:
```python
# Around line 30-41 where HPOBench imports are:
try:
    from .database_tuning import DatabaseTuningSuite
    DATABASE_TUNING_AVAILABLE = True
except ImportError:
    DatabaseTuningSuite = None
    DATABASE_TUNING_AVAILABLE = False

# Around line 79-85 where optional exports are added:
if DATABASE_TUNING_AVAILABLE and DatabaseTuningSuite is not None:
    __all__.extend([
        "DatabaseTuningSuite"
    ])
```

---

## 3. Implementation Template

### Step 1: Create the Wrapper Class

```python
# bomegabench/functions/database_tuning.py

import torch
from torch import Tensor
import numpy as np
from typing import Dict, List, Optional, Any
from ..core import BenchmarkFunction, BenchmarkSuite

# Check for optional dependency
try:
    import benchbase  # or your DB benchmarking library
    BENCHBASE_AVAILABLE = True
except ImportError:
    BENCHBASE_AVAILABLE = False
    print("BenchBase not available: pip install benchbase")


class DatabaseTuningFunction(BenchmarkFunction):
    """
    Wrapper for database knob tuning benchmarks.
    
    Converts discrete/categorical database configuration knobs
    to continuous [0, 1] representation for Bayesian Optimization.
    """
    
    def __init__(
        self,
        workload_name: str,
        database_system: str = "postgresql",
        **kwargs
    ):
        """
        Initialize database tuning function.
        
        Args:
            workload_name: Name of the workload (e.g., "tpcc", "tpch")
            database_system: Database system (e.g., "postgresql", "mysql")
            **kwargs: Additional arguments for BenchmarkFunction
        """
        if not BENCHBASE_AVAILABLE:
            raise ImportError("BenchBase required: pip install benchbase")
        
        self.workload_name = workload_name
        self.database_system = database_system
        
        # Initialize the actual benchmark
        self.benchmark = self._create_benchmark()
        
        # Get knob configuration space
        self.knob_info = self._get_knob_space()
        
        # Convert to continuous [0,1] representation
        self.continuous_space = self._create_continuous_space()
        
        # Set bounds
        dim = len(self.continuous_space)
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim])
        
        super().__init__(dim=dim, bounds=bounds, **kwargs)
    
    def _create_benchmark(self):
        """Create the benchmark instance."""
        # Example: return benchbase.get_benchmark(self.workload_name, self.database_system)
        pass
    
    def _get_knob_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the database knob space from the benchmark.
        
        Returns:
            Dictionary mapping knob names to their specifications:
            {
                "knob_name": {
                    "type": "int" | "enum" | "float" | "bool",
                    "min": min_value,
                    "max": max_value,
                    "default": default_value,
                    "choices": [options] (for enum types)
                },
                ...
            }
        """
        # Example implementation
        knobs = {}
        for knob_name, knob_config in self.benchmark.get_knobs().items():
            knobs[knob_name] = {
                "type": knob_config["type"],
                "min": knob_config.get("min"),
                "max": knob_config.get("max"),
                "default": knob_config.get("default"),
                "choices": knob_config.get("choices", [])
            }
        return knobs
    
    def _create_continuous_space(self) -> List[Dict[str, Any]]:
        """
        Convert mixed-type knob space to continuous [0,1] dimensions.
        
        Returns:
            List of dimension specifications for mapping continuous -> discrete
        """
        continuous_dims = []
        
        for knob_name, knob_spec in self.knob_info.items():
            knob_type = knob_spec["type"]
            
            if knob_type == "float":
                # Float parameters: normalize to [0,1]
                continuous_dims.append({
                    "name": knob_name,
                    "type": "float",
                    "original_bounds": (knob_spec["min"], knob_spec["max"]),
                    "knob_name": knob_name
                })
            
            elif knob_type == "int":
                # Integer parameters: one dimension per possible value
                min_val, max_val = knob_spec["min"], knob_spec["max"]
                for i, val in enumerate(range(min_val, max_val + 1)):
                    continuous_dims.append({
                        "name": f"{knob_name}_{val}",
                        "type": "int_onehot",
                        "original_param": knob_name,
                        "choice": val,
                        "choice_index": i,
                        "total_choices": max_val - min_val + 1
                    })
            
            elif knob_type == "enum":
                # Enum/categorical: one dimension per choice
                choices = knob_spec["choices"]
                for i, choice in enumerate(choices):
                    continuous_dims.append({
                        "name": f"{knob_name}_{choice}",
                        "type": "enum_onehot",
                        "original_param": knob_name,
                        "choice": choice,
                        "choice_index": i,
                        "total_choices": len(choices)
                    })
            
            elif knob_type == "bool":
                # Boolean: single dimension [0,1] -> {False, True}
                continuous_dims.append({
                    "name": knob_name,
                    "type": "bool",
                    "knob_name": knob_name
                })
        
        return continuous_dims
    
    def _convert_continuous_to_discrete(self, x_continuous: np.ndarray) -> Dict[str, Any]:
        """
        Convert continuous [0,1] point to discrete knob configuration.
        
        Args:
            x_continuous: Continuous point in [0,1]^d
            
        Returns:
            Dictionary mapping knob names to their values
        """
        config = {}
        dim_idx = 0
        
        for knob_name, knob_spec in self.knob_info.items():
            knob_type = knob_spec["type"]
            
            if knob_type == "float":
                # Denormalize to original bounds
                min_val, max_val = knob_spec["min"], knob_spec["max"]
                normalized = x_continuous[dim_idx]
                config[knob_name] = min_val + normalized * (max_val - min_val)
                dim_idx += 1
            
            elif knob_type == "int":
                # Find argmax in one-hot encoding
                min_val, max_val = knob_spec["min"], knob_spec["max"]
                num_choices = max_val - min_val + 1
                one_hot_values = x_continuous[dim_idx:dim_idx + num_choices]
                selected_idx = np.argmax(one_hot_values)
                config[knob_name] = min_val + selected_idx
                dim_idx += num_choices
            
            elif knob_type == "enum":
                # Find argmax in one-hot encoding
                choices = knob_spec["choices"]
                one_hot_values = x_continuous[dim_idx:dim_idx + len(choices)]
                selected_idx = np.argmax(one_hot_values)
                config[knob_name] = choices[selected_idx]
                dim_idx += len(choices)
            
            elif knob_type == "bool":
                # Threshold at 0.5
                config[knob_name] = x_continuous[dim_idx] > 0.5
                dim_idx += 1
        
        return config
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Return function metadata."""
        return {
            "name": f"BenchBase {self.database_system.capitalize()} {self.workload_name.upper()}",
            "suite": "Database Tuning",
            "properties": ["configuration_tuning", "database_workload", "expensive"],
            "database_system": self.database_system,
            "workload": self.workload_name,
            "total_knobs": len(self.knob_info),
            "domain": "Continuous [0,1] mapping to database knobs",
            "global_min": "Unknown - depends on database state",
            "knob_details": self.knob_info
        }
    
    def _evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate the function at given points.
        
        Args:
            X: Tensor of shape (..., dim) with values in [0,1]
            
        Returns:
            Tensor of performance metrics (lower is better)
        """
        # Handle batch evaluation
        if X.ndim == 1:
            X_batch = X.unsqueeze(0)
        else:
            X_batch = X
        
        results = []
        X_np = X_batch.detach().cpu().numpy()
        
        for x_point in X_np:
            # Convert continuous to discrete knob configuration
            knob_config = self._convert_continuous_to_discrete(x_point)
            
            # Apply configuration to database
            # This is the most time-consuming step
            self.benchmark.configure_knobs(knob_config)
            
            # Run workload and get performance metric
            # Return as loss (lower is better)
            performance = self.benchmark.run_workload()
            throughput = performance["throughput"]
            
            # Convert throughput to loss (minimize execution time)
            loss = 1.0 / throughput if throughput > 0 else float('inf')
            results.append(loss)
        
        return torch.tensor(results, dtype=X.dtype, device=X.device)


def create_database_tuning_suite() -> BenchmarkSuite:
    """
    Create database tuning suite with available workloads.
    
    Returns:
        BenchmarkSuite containing all available database tuning functions
    """
    if not BENCHBASE_AVAILABLE:
        return None
    
    functions = {}
    
    # Example: Add common workloads
    workloads = ["tpcc", "tpch", "ycsb"]
    database_systems = ["postgresql", "mysql"]
    
    for db_system in database_systems:
        for workload in workloads:
            try:
                func_name = f"{db_system}_{workload}"
                functions[func_name] = DatabaseTuningFunction(
                    workload_name=workload,
                    database_system=db_system
                )
            except Exception as e:
                print(f"Warning: Could not create {func_name}: {e}")
    
    return BenchmarkSuite("database_tuning", functions)


# Create suite at module load time
DatabaseTuningSuite = create_database_tuning_suite() if BENCHBASE_AVAILABLE else None
```

---

## 4. Key Design Decisions Made

| Decision | Reasoning |
|----------|-----------|
| Continuous [0,1] representation | Aligns with BoTorch/Bayesian Optimization requirements; hidden conversion layer handles discrete knobs |
| One-hot encoding for discrete knobs | Allows gradient-based optimization algorithms to work with discrete choices |
| Expensive evaluation (slow) | Database benchmarks take minutes; inherent property reflected in metadata |
| Optional dependency handling | Users without BenchBase still get rest of BOMegaBench working |
| Metadata with knob_details | Allows algorithms to understand and potentially customize BO strategies |

---

## 5. Usage Examples

### After Integration, Users Will Be Able To:

```python
import bomegabench as bmb
import torch

# List database tuning benchmarks
suites = bmb.list_suites()
if "database_tuning" in suites:
    functions = bmb.list_functions(suite="database_tuning")
    print(f"Available DB benchmarks: {functions}")

# Get a specific benchmark
func = bmb.get_function("postgresql_tpcc", suite="database_tuning")
print(f"Benchmark: {func.metadata['name']}")
print(f"Knobs: {func.metadata['knob_details']}")

# Evaluate a point
X = torch.rand(1, func.dim)
performance = func(X)
print(f"Performance loss: {performance.item()}")

# Run optimization
from bomegabench import BenchmarkRunner

runner = BenchmarkRunner(seed=42)
result = runner.run_single(
    function_name="postgresql_tpcc",
    algorithm=my_bayesian_optimizer,
    algorithm_name="BOTorch",
    n_evaluations=50,  # Limited because each eval is expensive
    function_kwargs={"suite": "database_tuning"}
)

print(f"Best configuration loss: {result.best_value}")
```

---

## 6. Testing Strategy

### Test File: `test_database_tuning_integration.py`

```python
import pytest
import torch
import bomegabench as bmb

class TestDatabaseTuningIntegration:
    
    def test_suite_registration(self):
        """Test that database tuning suite is registered."""
        suites = bmb.list_suites()
        if bmb.functions.database_tuning.DATABASE_TUNING_AVAILABLE:
            assert "database_tuning" in suites
    
    def test_function_discovery(self):
        """Test that functions can be discovered."""
        if "database_tuning" in bmb.list_suites():
            functions = bmb.list_functions(suite="database_tuning")
            assert len(functions) > 0
    
    def test_function_instantiation(self):
        """Test that functions can be created."""
        if "database_tuning" in bmb.list_suites():
            func = bmb.get_function("postgresql_tpcc", suite="database_tuning")
            assert func is not None
            assert hasattr(func, 'metadata')
    
    def test_function_evaluation(self):
        """Test that functions can be evaluated."""
        if "database_tuning" in bmb.list_suites():
            func = bmb.get_function("postgresql_tpcc", suite="database_tuning")
            X = torch.rand(1, func.dim)
            Y = func(X)
            assert Y.shape == (1,)
    
    def test_metadata_completeness(self):
        """Test that metadata includes all required fields."""
        if "database_tuning" in bmb.list_suites():
            func = bmb.get_function("postgresql_tpcc", suite="database_tuning")
            required_fields = ["name", "suite", "properties", "database_system", "workload"]
            for field in required_fields:
                assert field in func.metadata
```

---

## 7. Integration Checklist

- [ ] Create `bomegabench/functions/database_tuning.py`
- [ ] Implement `DatabaseTuningFunction` base class
- [ ] Implement `_get_metadata()` with all required fields
- [ ] Implement `_evaluate_true(X)` with benchmark execution
- [ ] Handle knob space discovery and conversion
- [ ] Create `create_database_tuning_suite()` factory
- [ ] Update `bomegabench/functions/__init__.py` with imports
- [ ] Update `bomegabench/functions/registry.py` with registration
- [ ] Create `examples/database_tuning_example.py`
- [ ] Create `test_database_tuning_integration.py`
- [ ] Create `DATABASE_TUNING_INTEGRATION_NOTES.md` with specifics

---

## 8. Common Implementation Challenges & Solutions

### Challenge 1: Database State Management
**Problem**: Running benchmarks requires starting/stopping database, initializing data
**Solution**: Implement setup/teardown in benchmark class; cache between runs if possible

### Challenge 2: Slow Evaluation
**Problem**: Each knob evaluation takes minutes
**Solution**: 
- Document in metadata as "expensive" property
- Use multi-fidelity evaluation if BenchBase supports it
- Consider warm-start/checkpoint mechanisms

### Challenge 3: Numerical Stability
**Problem**: Float knobs with very different scales
**Solution**: 
- Normalize input knobs in `_get_knob_space()`
- Consider log-scale for large ranges
- Document scaling in metadata

### Challenge 4: Configuration Validity
**Problem**: Not all knob combinations are valid
**Solution**:
- Implement constraint checking in `_convert_continuous_to_discrete()`
- Return worst loss for invalid configurations
- Log warnings for debugging

---

## 9. Files to Review (Reference)

1. **Concrete Implementation Pattern**: 
   - `/mnt/h/BOResearch-25fall/BOMegaBench/bomegabench/functions/lasso_bench.py`

2. **Complex Space Conversion**:
   - `/mnt/h/BOResearch-25fall/BOMegaBench/bomegabench/functions/hpobench_benchmarks.py`

3. **Usage Examples**:
   - `/mnt/h/BOResearch-25fall/BOMegaBench/examples/lasso_bench_example.py`

4. **Registry Pattern**:
   - `/mnt/h/BOResearch-25fall/BOMegaBench/bomegabench/functions/registry.py`

---

## 10. Next Steps

1. Create `bomegabench/functions/database_tuning.py` using template above
2. Adapt it based on actual BenchBase API
3. Test with optional dependency available and unavailable
4. Create example showing end-to-end workflow
5. Document any database-specific considerations

