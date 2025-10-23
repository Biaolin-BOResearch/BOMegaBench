# BOMegaBench Project Structure and Integration Guide

## Executive Summary

BOMegaBench is a comprehensive Bayesian Optimization benchmark library with 126+ synthetic and real-world benchmark functions. The project is well-structured with a modular architecture that supports easy integration of new benchmark suites through a unified interface.

---

## 1. Overall Directory Structure

```
BOMegaBench/
├── bomegabench/                    # Main package directory
│   ├── __init__.py                 # Package initialization with version info
│   ├── core.py                     # Base classes and interfaces
│   ├── benchmark.py                # BenchmarkRunner and result management
│   ├── visualization.py            # Visualization utilities
│   └── functions/                  # Benchmark functions organization
│       ├── __init__.py             # Registry imports
│       ├── registry.py             # Function discovery and registration
│       ├── consolidated_functions.py    # 72 synthetic functions
│       ├── lasso_bench.py          # LassoBench integration (13 functions)
│       ├── hpo_benchmarks.py       # HPO benchmarks via Bayesmark
│       └── hpobench_benchmarks.py  # HPOBench ML/NAS/RL/OD/Surrogates
├── examples/                       # Usage examples
│   ├── basic_usage.py              # Core library usage
│   ├── lasso_bench_example.py      # LassoBench examples
│   ├── lasso_bench_simple.py       # Simple LassoBench example
│   └── hpo_benchmark_example.py    # HPO benchmarks examples
├── HPOBench/                       # Submodule for HPOBench library
├── LassoBench/                     # Submodule for LassoBench library
├── exdata/                         # Experimental data cache
├── setup.py                        # Package configuration
├── requirements.txt                # Dependencies
└── [Documentation files]           # Integration guides and lists
```

---

## 2. Core Architecture and Design Patterns

### 2.1 Base Classes (core.py)

#### BenchmarkFunction (Abstract Base Class)
- **Location**: `/mnt/h/BOResearch-25fall/BOMegaBench/bomegabench/core.py`
- **Key Methods**:
  - `__init__(dim, bounds, negate, noise_std, **kwargs)` - Initialize with problem parameters
  - `forward(X, noise=True)` - Evaluate function with optional noise
  - `_evaluate_true(X)` - Core evaluation logic (abstract)
  - `__call__(X)` - Support both numpy and torch inputs
  - `get_bounds()` - Return bounds as numpy arrays
  - `sample_random(n_samples)` - Generate random points within bounds

- **Metadata Support**:
  - `_get_metadata()` - Abstract method returning dict with:
    - `name`: Function identifier
    - `suite`: Suite name
    - `properties`: List of properties (e.g., ["unimodal", "separable"])
    - `domain`: Domain specification
    - `global_min`: Known global minimum value
    - Other custom metadata fields

#### BenchmarkSuite (Container Class)
- Manages collection of BenchmarkFunction instances
- Provides lookup by name, listing, property-based filtering
- Supports iteration and bracket notation access

### 2.2 Unified Registry Pattern (registry.py)

The registry implements a **discovery and registration pattern**:

```python
# Global registry mapping suite names to BenchmarkSuite instances
_SUITES: Dict[str, BenchmarkSuite] = {
    "consolidated": ConsolidatedSuite,
    "lasso_synthetic": LassoBenchSyntheticSuite,  # Optional dependency
    "lasso_real": LassoBenchRealSuite,            # Optional dependency
    "hpo": HPOBenchmarksSuite,                    # Optional dependency
    "hpobench_ml": HPOBenchMLSuite,               # Optional dependency
    # ... additional suites
}

# Main API functions:
- get_function(name, suite=None) -> BenchmarkFunction
- list_functions(suite=None) -> List[str]
- list_suites() -> List[str]
- get_functions_by_property(property_name, value, suite=None) -> Dict
- get_multimodal_functions(suite=None) -> Dict
- get_unimodal_functions(suite=None) -> Dict
```

**Key Design**: Graceful degradation - if optional dependencies are unavailable, the library continues to work with available suites.

### 2.3 Benchmark Runner (benchmark.py)

**BenchmarkResult Dataclass**:
- Stores optimization results with metadata
- Provides regret calculation
- Supports serialization to dict/DataFrame

**BenchmarkRunner Class**:
- `run_single()` - Execute single (function, algorithm) pair
- `run_multiple()` - Execute multiple experiments with progress tracking
- `get_results_dataframe()` - Convert results to pandas DataFrame

---

## 3. Existing Benchmark Suites

### 3.1 Consolidated Suite (72 functions)
**Location**: `/mnt/h/BOResearch-25fall/BOMegaBench/bomegabench/functions/consolidated_functions.py`

**Organization**:
- **BBOB Functions** (24): Standard Black-Box Optimization Benchmark set
  - Classes like `F01_SphereRaw`, `F02_EllipsoidSeparableRaw`, etc.
  - All inherit from `BenchmarkFunction`
  - Dimension-scalable (default dim=2, can be customized)

- **BoTorch Additional** (6): Functions from BoTorch library
- **Classical Additional** (32): Classical optimization benchmarks
- **Classical Core** (10): Core classical functions

**Integration Pattern for Consolidated Suite**:
```python
# Create function classes inheriting from BenchmarkFunction
class FunctionName(BenchmarkFunction):
    def __init__(self, dim=2, **kwargs):
        bounds = torch.tensor([[-bounds] * dim, [bounds] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)
    
    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "Function display name",
            "suite": "Suite name",
            "properties": ["property1", "property2"],
            "domain": "[-5,5]^d",
            "global_min": "f(0)=0"
        }
    
    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Vectorized computation
        return torch.sum(X**2, dim=-1)  # Returns shape (...,)

# Create suite at module load time
def create_consolidated_suite() -> BenchmarkSuite:
    functions = {
        "function_key": FunctionName(),
        # ... more functions
    }
    return BenchmarkSuite("consolidated", functions)
```

### 3.2 LassoBench Integration (13 functions)
**Location**: `/mnt/h/BOResearch-25fall/BOMegaBench/bomegabench/functions/lasso_bench.py`

**Integration Strategy**:
- Wraps external library (LassoBench) into BOMegaBench interface
- Handles optional dependency gracefully
- Two wrapper classes:
  - `LassoBenchSyntheticFunction` (8 functions)
  - `LassoBenchRealFunction` (5 functions)

**Key Pattern**:
```python
try:
    from LassoBench.LassoBench import SyntheticBenchmark, RealBenchmark
    LASSO_BENCH_AVAILABLE = True
except ImportError:
    LASSO_BENCH_AVAILABLE = False
    # Provide helpful error message

class LassoBenchSyntheticFunction(BenchmarkFunction):
    def __init__(self, bench_name: str, noise: bool = False, **kwargs):
        if not LASSO_BENCH_AVAILABLE:
            raise ImportError("Install with: pip install git+https://github.com/ksehic/LassoBench.git")
        
        # Wrap external library instance
        self.lasso_bench = SyntheticBenchmark(pick_bench=bench_name, noise=noise)
        
        # Extract dimension and bounds
        dim = self.lasso_bench.n_features
        bounds = torch.tensor([[-1.0] * dim, [1.0] * dim])
        
        super().__init__(dim=dim, bounds=bounds, **kwargs)
    
    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Convert torch -> numpy, evaluate, convert back
        X_np = X.detach().cpu().numpy()
        if X_np.ndim == 1:
            return torch.tensor(self.lasso_bench.evaluate(X_np), ...)
        else:
            results = [self.lasso_bench.evaluate(X_np[i]) for i in range(X_np.shape[0])]
            return torch.tensor(results, ...)
    
    # Optional: Additional methods for library-specific features
    def get_test_metrics(self, X: Tensor) -> Dict[str, float]:
        """Extract library-specific metrics"""
        return self.lasso_bench.test(X_np)
```

**Suite Creation**:
```python
def create_lasso_bench_synthetic_suite() -> BenchmarkSuite:
    functions = {}
    for bench_name in ["synt_simple", "synt_medium", "synt_high", "synt_hard"]:
        for noise in [False, True]:
            func_name = f"{bench_name}_{'noisy' if noise else 'noiseless'}"
            functions[func_name] = LassoBenchSyntheticFunction(bench_name, noise=noise)
    return BenchmarkSuite("lasso_synthetic", functions)

LassoBenchSyntheticSuite = create_lasso_bench_synthetic_suite() if LASSO_BENCH_AVAILABLE else None
```

### 3.3 HPOBench Integration (30+ functions)
**Location**: `/mnt/h/BOResearch-25fall/BOMegaBench/bomegabench/functions/hpobench_benchmarks.py`

**Complexity Handled**:
- HPOBench uses ConfigSpace for hyperparameter representation
- Continuous vs. discrete vs. categorical parameters
- Multi-fidelity benchmarks (uses maximum fidelity)

**Key Class**:
```python
class HPOBenchFunction(BenchmarkFunction):
    def __init__(self, benchmark_class, benchmark_name, benchmark_kwargs=None, **kwargs):
        # Create benchmark instance
        self.benchmark = benchmark_class(**benchmark_kwargs)
        
        # Get and normalize hyperparameter configuration space
        self.config_space = self.benchmark.get_configuration_space()
        
        # Convert to continuous [0,1] representation
        self.continuous_space = self._create_continuous_space()
        
        # All benchmarks normalized to [0,1]^d
        dim = len(self.continuous_space)
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)
    
    def _create_continuous_space(self) -> List[Dict]:
        """Convert mixed-type ConfigSpace to continuous [0,1]"""
        # Handle UniformFloatHyperparameter -> normalize
        # Handle UniformIntegerHyperparameter -> one-hot per value
        # Handle CategoricalHyperparameter -> one-hot per choice
        ...
    
    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Denormalize [0,1] back to original spaces
        # Create ConfigSpace Configuration
        # Evaluate benchmark
        # Return performance as loss (minimization)
        ...
```

### 3.4 HPO Benchmarks via Bayesmark
**Location**: `/mnt/h/BOResearch-25fall/BOMegaBench/bomegabench/functions/hpo_benchmarks.py`

**Pattern**:
- Wraps scikit-learn ML models via Bayesmark SklearnModel
- Uses datasets: iris, wine, digits, breast_cancer
- Handles hyperparameter space conversion to continuous [0,1]

---

## 4. Python Interfaces and API Patterns

### 4.1 Function Evaluation Interface

```python
# All functions support multiple input/output formats
func = bmb.get_function("sphere")

# Single point - torch
X = torch.tensor([[0.5, 0.3]])
y = func(X)  # Returns Tensor

# Single point - numpy
X_np = np.array([0.5, 0.3])
y_np = func(X_np)  # Returns ndarray

# Batch evaluation
X_batch = torch.rand(10, 2)
Y_batch = func(X_batch)  # Shape (10,)

# Direct __call__ interface
result = func.__call__(X)
```

### 4.2 Metadata Interface

Every function provides rich metadata:
```python
metadata = func.metadata
# Keys:
# - name: str
# - suite: str
# - properties: List[str] - e.g., ["unimodal", "separable", "multimodal"]
# - domain: str - e.g., "[-5,5]^d"
# - global_min: float or str
# - Additional custom fields per benchmark type
```

### 4.3 Dimension Scaling Interface

Most functions support dimension scaling:
```python
# Fixed dimension
func_2d = bmb.get_function("sphere")(dim=2)

# Variable dimension
func_10d = bmb.get_function("sphere")(dim=10)
func_100d = bmb.get_function("sphere")(dim=100)
```

### 4.4 Optimization Runner Interface

```python
from bomegabench import BenchmarkRunner

runner = BenchmarkRunner(seed=42)

# Single experiment
result = runner.run_single(
    function_name="sphere",
    algorithm=my_optimizer,
    algorithm_name="My Algo",
    n_evaluations=100,
    dim=5,
    algorithm_kwargs={"param": value},
    function_kwargs={}
)

# Multiple experiments
results = runner.run_multiple(
    function_names=["sphere", "rosenbrock"],
    algorithms={"random": algo1, "bayesopt": algo2},
    n_evaluations=100,
    n_runs=3,
    dim=5,
    show_progress=True
)

# Access results
df = runner.get_results_dataframe()
```

### 4.5 Suite Discovery Interface

```python
# List all suites
suites = bmb.list_suites()

# List functions in a suite
functions = bmb.list_functions(suite="consolidated")

# Get all functions with a property
multimodal = bmb.get_multimodal_functions()
separable = bmb.get_functions_by_property("properties", "separable")
```

---

## 5. Integration Patterns Summary

### Pattern 1: Simple Synthetic Functions (Consolidated Suite)

**When to use**: For pure mathematical functions
**Steps**:
1. Create class inheriting from `BenchmarkFunction`
2. Implement `_get_metadata()` returning function properties
3. Implement `_evaluate_true(X)` with vectorized computation
4. Instantiate in module and add to dictionary
5. Register in `registry.py`

**File structure**:
- All classes in `consolidated_functions.py`
- Instantiate all at module load
- Return BenchmarkSuite from `create_consolidated_suite()`

### Pattern 2: External Library Wrapper (LassoBench, HPOBench)

**When to use**: For existing benchmark libraries
**Steps**:
1. Check for optional dependency import
2. Create wrapper class inheriting from `BenchmarkFunction`
3. Initialize external library instance in `__init__`
4. Implement parameter space discovery
5. Implement `_evaluate_true(X)` with library calls
6. Add library-specific methods if needed
7. Create suite factory function
8. Add conditional import in `__init__.py` and `registry.py`

**File structure**:
- Wrapper code in dedicated module (e.g., `lasso_bench.py`)
- Suite creation at module level
- Import guarded by try/except
- Add to global `_SUITES` dict conditionally in `registry.py`

### Pattern 3: Configuration Space Conversion (HPOBench)

**When to use**: For discrete/categorical hyperparameters
**Challenge**: Convert mixed-type hyperparameter spaces to continuous [0,1]
**Solution**:
- FloatHP: normalize to [0,1]
- IntHP: create separate dimension per integer value (one-hot)
- CategoricalHP: create separate dimension per category (one-hot)
- Conversion back: compute which category is "active" (highest value)

---

## 6. Documentation Patterns

### 6.1 Function Metadata

Every benchmark function provides metadata dictionary with at minimum:
```python
{
    "name": "Display name",
    "suite": "Suite name",
    "properties": ["unimodal" | "multimodal", "separable" | "non-separable"],
    "domain": "[-a,b]^d or similar",
    "global_min": "value or 'Variable'"
}
```

### 6.2 Suite Documentation

Each suite has:
- **File header docstring** explaining purpose and structure
- **Inline docstrings** for wrapper classes
- **Example markdown files** showing usage
- **Test files** demonstrating integration

### 6.3 Configuration Documentation

Configuration handled via:
- Constructor keyword arguments
- Metadata fields
- Example usage patterns

Example from LassoBench:
```python
class LassoBenchSyntheticFunction(BenchmarkFunction):
    """
    Wrapper for LassoBench synthetic benchmarks.
    
    Parameters
    ----------
    bench_name : str
        Name of benchmark ('synt_simple', 'synt_medium', 'synt_high', 'synt_hard')
    noise : bool
        Whether to add noise to the benchmark
    
    Properties
    ----------
    dim : int
        Total number of dimensions (60-1000)
    active_dimensions : int
        Number of relevant dimensions (3-50)
    """
```

---

## 7. How Tasks Are Currently Organized

### 7.1 Function Organization Hierarchy

```
BenchmarkFunction (abstract base)
├── Consolidated Suite Functions (72)
│   ├── BBOB (24)
│   ├── BoTorch Additional (6)
│   ├── Classical Additional (32)
│   └── Classical Core (10)
├── LassoBench Wrapper Functions (13)
│   ├── Synthetic (8)
│   └── Real-world (5)
├── HPO Wrapper Functions (100+)
│   └── Bayesmark SklearnModel wrapper
└── HPOBench Wrapper Functions (30+)
    ├── ML Benchmarks (15+)
    ├── NAS Benchmarks (8+)
    ├── OD Benchmarks (2+)
    ├── RL Benchmarks (2+)
    └── Surrogate Benchmarks (3+)
```

### 7.2 Suite Organization

Suites are collections of functions by domain/type:
- Each suite is a `BenchmarkSuite` instance
- Functions accessed via `get_function(name, suite=suite_name)`
- Can query functions by properties
- Can list all functions in suite

### 7.3 Metadata-Driven Organization

Functions can be found by properties:
```python
# Get by property value
bmb.get_multimodal_functions()  # Where properties contains "multimodal"
bmb.get_unimodal_functions()    # Where properties contains "unimodal"
bmb.get_functions_by_property("separable", True)  # Custom properties

# List by suite
functions_in_suite = bmb.list_functions(suite="suite_name")
```

---

## 8. Key Integration Points for Database Knob Tuning

Based on the project structure, here are the recommended integration points:

### 8.1 Create New Module: `database_tuning.py`

Location: `/mnt/h/BOResearch-25fall/BOMegaBench/bomegabench/functions/database_tuning.py`

**Key components**:
1. **DatabaseTuningFunction** - Base wrapper class
2. **BenchbaseFunction** - Wrapper for BenchBase benchmarks
3. **Suite creation** - Factory function returning BenchmarkSuite
4. **Optional dependency handling** - Try/except import pattern

### 8.2 Update Registry (`registry.py`)

Add to `_SUITES` dictionary:
```python
if DATABASE_TUNING_AVAILABLE:
    _SUITES.update({
        "database_tuning": DatabaseTuningSuite,
    })
```

### 8.3 Configuration Handling

For database knobs (typically integers/enumerations):
- Use HPOBench pattern: convert to continuous [0,1]
- Store mapping: continuous value -> actual knob value
- Document: knob names, types, ranges, impact

### 8.4 Metadata Fields

Suggested metadata for database tuning functions:
```python
{
    "name": "BenchBase PostgreSQL tpcc",
    "suite": "Database Tuning",
    "properties": ["configuration_tuning", "database_workload"],
    "database_system": "PostgreSQL",
    "workload": "TPCC",
    "total_knobs": 26,
    "tunable_knobs": 26,
    "global_min": "Best known configuration",
    "domain": "[knob_min, knob_max] for each knob"
}
```

### 8.5 Example Implementation Structure

```python
# database_tuning.py structure

class DatabaseTuningFunction(BenchmarkFunction):
    """Base wrapper for database tuning benchmarks"""
    
    def __init__(self, workload_name: str, **kwargs):
        # Initialize BenchBase or other DB benchmark
        self.workload = WorkloadClass(workload_name)
        
        # Get knob space and convert to continuous
        self.knob_space = self._get_knob_space()
        self.continuous_space = self._convert_to_continuous()
        
        # Set bounds
        dim = len(self.continuous_space)
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)
    
    def _get_knob_space(self) -> Dict:
        # Get database system knobs
        # Returns {knob_name: (min, max, type), ...}
        pass
    
    def _convert_to_continuous(self) -> List[Dict]:
        # Convert knob ranges to continuous [0,1]
        pass
    
    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Denormalize X to knob values
        # Apply configuration to database
        # Run benchmark workload
        # Return performance metric (as loss for minimization)
        pass

def create_database_tuning_suite() -> BenchmarkSuite:
    """Create database tuning suite with multiple workloads"""
    functions = {}
    for workload in AVAILABLE_WORKLOADS:
        functions[workload] = DatabaseTuningFunction(workload)
    return BenchmarkSuite("database_tuning", functions)
```

---

## 9. Summary Table: Integration Checklist

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| Base class | `core.py` | Ready | Use `BenchmarkFunction` |
| Registry | `registry.py` | Ready | Add to `_SUITES` dict |
| Module | TBD | TODO | Create `database_tuning.py` |
| Wrapper class | TBD | TODO | Implement for DB benchmarks |
| Suite factory | TBD | TODO | Create suite instance |
| __init__.py | `functions/__init__.py` | TODO | Add conditional imports |
| Main __init__.py | `__init__.py` | TODO | Export new suite |
| Examples | `examples/` | TODO | Add usage example |
| Tests | TBD | TODO | Add integration tests |
| Documentation | TBD | TODO | Add integration guide |

---

## 10. Quick Start for Database Knob Tuning Integration

### Step 1: Create wrapper module
```bash
touch /mnt/h/BOResearch-25fall/BOMegaBench/bomegabench/functions/database_tuning.py
```

### Step 2: Implement wrapper classes following LassoBench/HPOBench patterns
- Inherit from `BenchmarkFunction`
- Handle optional BenchBase import
- Implement knob space discovery
- Implement evaluation with benchmark execution

### Step 3: Update registry
- Add conditional import in `functions/__init__.py`
- Register suite in `registry.py` `_SUITES` dict
- Export from main `__init__.py`

### Step 4: Create example
- Add `examples/database_tuning_example.py`
- Show suite listing, function selection, evaluation

### Step 5: Add tests
- Test with and without BenchBase available
- Verify metadata accuracy
- Test knob conversion logic

---

## Files Ready for Reference

### Core Files (Already Analyzed)
- `/mnt/h/BOResearch-25fall/BOMegaBench/bomegabench/core.py` - Base classes
- `/mnt/h/BOResearch-25fall/BOMegaBench/bomegabench/benchmark.py` - Runner
- `/mnt/h/BOResearch-25fall/BOMegaBench/bomegabench/functions/registry.py` - Registry
- `/mnt/h/BOResearch-25fall/BOMegaBench/bomegabench/functions/lasso_bench.py` - Integration example

### Examples (Ready to Study)
- `/mnt/h/BOResearch-25fall/BOMegaBench/examples/basic_usage.py`
- `/mnt/h/BOResearch-25fall/BOMegaBench/examples/lasso_bench_example.py`

### Documentation (Ready to Read)
- `/mnt/h/BOResearch-25fall/BOMegaBench/lasso_bench_integration_summary.md`
- `/mnt/h/BOResearch-25fall/BOMegaBench/lasso_bench_integration.md`

