# Discrete Parameter Encoding Implementation Summary

## Overview

Implemented three encoding modes for discrete parameters in BOMegaBench, allowing flexible integration with different Bayesian Optimization algorithms. The encoding mode is controlled via the `discrete_encoding` parameter during benchmark instantiation.

**Implementation Date**: 2025-10-22

---

## Three Encoding Modes

### 1. Raw Mode (`discrete_encoding="raw"`)
- **Description**: No encoding, optimizer works directly with discrete values (integer indices)
- **Dimension**: `optimizer_dim = problem_dim`
- **Use Case**: Mixed-integer Bayesian Optimization algorithms (SMAC, Optuna)

### 2. Interval Mode (`discrete_encoding="interval"`)
- **Description**: Maps [0,1] continuous values to discrete options via interval partitioning
- **Dimension**: `optimizer_dim = problem_dim`
- **Use Case**: Standard BO with continuous relaxation (BoTorch, GPyOpt)
- **Mapping**: For n options: [0, 1/n), [1/n, 2/n), ..., [(n-1)/n, 1] → options 0, 1, ..., n-1

### 3. One-hot Mode (`discrete_encoding="onehot"`)
- **Description**: One-hot encodes discrete parameters into multiple [0,1] dimensions
- **Dimension**: `optimizer_dim = problem_dim - n_discrete + Σ(n_options_i)`
- **Use Case**: Categorical variables with no natural ordering
- **Decoding**: Uses argmax to convert one-hot back to discrete index

---

## Files Modified/Created

### 1. Core Implementation

#### `bomegabench/utils/discrete_encoding.py` (NEW, 427 lines)
**Purpose**: Core encoding/decoding logic

**Key Classes**:
- `DiscreteParameterSpec`: Specification for a discrete parameter
  - `dim_index`: Index in original parameter vector
  - `n_options`: Number of discrete options
  - `options`: Optional list of actual option values

- `DiscreteEncoder`: Main encoder/decoder class
  - `__init__(continuous_dims, discrete_specs, encoding)`
  - `encode(X)`: Optimizer space → Problem space
  - `decode(X_problem)`: Problem space → Optimizer space
  - `get_optimizer_bounds(problem_bounds)`: Get bounds for optimizer space
  - `get_info()`: Return encoding configuration info

**Helper Functions**:
- `create_encoder_for_hpo(param_config, encoding)`: Create encoder for HPO benchmarks

**Key Features**:
- Automatic dimension calculation for each encoding mode
- Bidirectional encoding/decoding
- Proper bounds transformation
- Validation of discrete parameter specifications

---

#### `bomegabench/core.py` (MODIFIED)
**Changes**:

1. **Import discrete encoding utilities** (lines 11-18):
```python
from .utils.discrete_encoding import DiscreteEncoder, DiscreteParameterSpec
```

2. **Updated `BenchmarkFunction.__init__`** (lines 29-72):
   - Added `discrete_specs` parameter
   - Added `discrete_encoding` parameter (default: "onehot")
   - Create encoder if discrete parameters exist
   - Update `self.dim` and `self.bounds` to reflect optimizer space

3. **Updated `forward()` method** (lines 84-110):
   - Encode from optimizer space to problem space before evaluation
   - Automatic handling of discrete parameter conversion

4. **Updated `_evaluate()` method** (lines 117-124):
   - Also handles encoding for internal evaluations

5. **Added helper methods** (lines 161-208):
   - `encode_to_problem_space(X_optimizer)`: Manual encoding
   - `decode_from_problem_space(X_problem)`: Manual decoding (for initialization)
   - `get_encoding_info()`: Get encoding configuration

**Backward Compatibility**:
- All existing benchmarks without `discrete_specs` work unchanged
- Default `discrete_specs=None` preserves original behavior

---

#### `bomegabench/utils/__init__.py` (MODIFIED)
**Changes**: Added exports for discrete encoding classes (lines 13-17):
```python
from .discrete_encoding import (
    DiscreteParameterSpec,
    DiscreteEncoder,
    create_encoder_for_hpo
)
```

---

### 2. Documentation

#### `DISCRETE_ENCODING_GUIDE.md` (NEW, 680 lines)
**Purpose**: Comprehensive user guide

**Contents**:
1. Overview of three encoding modes
2. Detailed description of each mode with examples
3. Complete usage examples
4. Integration with BoTorch example
5. Guidance on choosing the right encoding mode
6. API reference
7. Testing and validation examples
8. FAQs

**Key Examples**:
- Simple benchmark with discrete parameters
- Using with existing benchmarks
- Integration with BoTorch optimizer
- Encoding consistency tests

---

#### `examples/discrete_encoding_demo.py` (NEW, 430 lines)
**Purpose**: Runnable demonstration script

**Features**:
- `HPOBenchmarkExample`: Example benchmark with 2 continuous + 2 discrete parameters
- `demo_raw_encoding()`: Demonstrates raw mode
- `demo_interval_encoding()`: Demonstrates interval mode
- `demo_onehot_encoding()`: Demonstrates one-hot mode
- `demo_encoding_comparison()`: Compares all three modes
- `demo_random_sampling()`: Shows random sampling in each mode

**How to Run**:
```bash
cd /mnt/h/BOResearch-25fall/BOMegaBench
python examples/discrete_encoding_demo.py
```

---

## Usage Examples

### Basic Usage

```python
from bomegabench.core import BenchmarkFunction
from bomegabench.utils import DiscreteParameterSpec

class MyBenchmark(BenchmarkFunction):
    def __init__(self, discrete_encoding="onehot"):
        discrete_specs = [
            DiscreteParameterSpec(dim_index=2, n_options=3),
            DiscreteParameterSpec(dim_index=5, n_options=5)
        ]

        super().__init__(
            dim=10,
            bounds=torch.tensor([[0.0]*10, [1.0]*10]),
            discrete_specs=discrete_specs,
            discrete_encoding=discrete_encoding
        )

    def _evaluate_true(self, X):
        # X is in problem space (discrete dims are indices)
        # Implement your objective function here
        return objective(X)

    def _get_metadata(self):
        return {'name': 'MyBenchmark'}

# Instantiate with different encodings
func_raw = MyBenchmark(discrete_encoding="raw")        # dim=10
func_interval = MyBenchmark(discrete_encoding="interval")  # dim=10
func_onehot = MyBenchmark(discrete_encoding="onehot")      # dim=14
```

### Integration with BoTorch

```python
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement

# Create benchmark with interval encoding
func = MyBenchmark(discrete_encoding="interval")

# Sample initial points in optimizer space
X_init = func.sample_random(10)
Y_init = func(X_init).unsqueeze(-1)

# Fit GP model
gp = SingleTaskGP(X_init, Y_init)

# Optimize acquisition function
ei = ExpectedImprovement(gp, best_f=Y_init.min())
candidate, _ = optimize_acqf(
    ei,
    bounds=func.bounds,  # Already in optimizer space
    q=1,
    num_restarts=10,
    raw_samples=100
)

# Evaluate candidate
y_new = func(candidate)
```

---

## Architecture

### Encoding Pipeline

```
Optimizer Space (BO algorithm works here)
         ↓
    [decode()]  ← For initialization from problem space
         ↓
    [encode()]  ← Automatic in forward()
         ↓
  Problem Space (Function evaluation happens here)
```

### Dimension Transformations

```
Mode: raw
  problem_dim=10, discrete at [2,5] with [3,5] options
  → optimizer_dim=10

Mode: interval
  problem_dim=10, discrete at [2,5] with [3,5] options
  → optimizer_dim=10

Mode: onehot
  problem_dim=10, discrete at [2,5] with [3,5] options
  → optimizer_dim = 10 - 2 + 3 + 5 = 16

  Dimension mapping:
  - optimizer[0:2]   → problem[0:2]   (continuous)
  - optimizer[2:5]   → problem[2]     (one-hot, 3 options)
  - optimizer[5:7]   → problem[3:5]   (continuous)
  - optimizer[7:12]  → problem[5]     (one-hot, 5 options)
  - optimizer[12:16] → problem[6:10]  (continuous)
```

---

## Key Design Decisions

### 1. Separation of Optimizer Space and Problem Space
- **Rationale**: Allows encoding to be transparent to the objective function
- **Benefit**: Existing benchmark implementations don't need modification
- **Implementation**: Encoding happens in `forward()` method automatically

### 2. Default to One-hot Encoding
- **Rationale**: Most theoretically sound for categorical variables
- **Benefit**: No artificial ordering imposed on discrete choices
- **Trade-off**: Increases dimensionality, but provides better kernel behavior

### 3. Support for Custom Options
- **Rationale**: Discrete parameters often have semantic meaning (e.g., ['sgd', 'adam'])
- **Benefit**: Better interpretability and documentation
- **Implementation**: `DiscreteParameterSpec.options` attribute

### 4. Bidirectional Encoding
- **Rationale**: Need to initialize optimization with specific problem-space values
- **Benefit**: Can warm-start optimization from known good configurations
- **Implementation**: Both `encode()` and `decode()` methods

---

## Testing and Validation

### Consistency Tests
All encoding modes have been validated for:
1. **Round-trip consistency**: `decode(encode(X)) ≈ X`
2. **Dimension correctness**: Output dimensions match expected values
3. **Bounds preservation**: Encoded values respect optimizer space bounds
4. **Discrete value validity**: Discrete indices are in valid range [0, n_options-1]

### Example Test Code
```python
def test_encoding_consistency():
    discrete_specs = [
        DiscreteParameterSpec(dim_index=1, n_options=3),
        DiscreteParameterSpec(dim_index=3, n_options=4)
    ]

    for encoding in ["raw", "interval", "onehot"]:
        encoder = DiscreteEncoder(5, discrete_specs, encoding)

        # Test round-trip
        X_problem = torch.tensor([[0.5, 1.0, 0.3, 2.0, 0.8]])
        X_opt = encoder.decode(X_problem)
        X_reconstructed = encoder.encode(X_opt)

        assert torch.allclose(X_problem, X_reconstructed, atol=0.01)
```

---

## Future Extensions

### Potential Enhancements
1. **Mixed encoding modes**: Allow different discrete parameters to use different encodings
2. **Ordinal encoding**: Special handling for ordinal discrete variables
3. **Adaptive encoding**: Switch encoding modes during optimization
4. **Relaxed one-hot**: Use softmax instead of argmax for differentiable optimization
5. **Hierarchical parameters**: Support for conditional discrete parameters

### Benchmark Integration
Currently, the encoding is available but not yet integrated with specific benchmarks that have discrete parameters (HPOBench, Database Tuning, etc.). Future work:
1. Update HPOBench wrappers to expose `discrete_encoding` parameter
2. Update Database Tuning benchmarks similarly
3. Add encoding mode to benchmark metadata
4. Create benchmark-specific usage examples

---

## Performance Considerations

### Memory
- **Raw mode**: No overhead
- **Interval mode**: No overhead
- **One-hot mode**: Increases memory by factor of ~(Σn_options / n_discrete)

### Computation
- **Raw mode**: No encoding overhead
- **Interval mode**: Minimal (floor + clamp operations)
- **One-hot mode**: Moderate (scatter/gather operations for one-hot conversion)

### Optimization
- **Raw mode**: Requires mixed-integer BO (fewer available algorithms)
- **Interval mode**: Fast, but may have poor GP kernel behavior at interval boundaries
- **One-hot mode**: Best for GP-based BO, but higher dimensional

---

## References

### Related Work
1. **Mixed-Integer BO**: Garrido-Merchán & Hernández-Lobato (2020)
2. **One-hot Encoding**: Snoek et al. (2012), SMAC paper
3. **Continuous Relaxation**: Daulton et al. (2022), BoTorch mixed optimization

### BOMegaBench Context
- Part of broader effort to support diverse parameter types
- Complements existing continuous and mixed-type benchmarks
- Enables fair comparison of BO algorithms on discrete/mixed problems

---

## Conclusion

The discrete encoding implementation provides:
- ✅ Three flexible encoding modes
- ✅ Seamless integration with existing benchmarks
- ✅ Backward compatibility
- ✅ Comprehensive documentation and examples
- ✅ Validated round-trip encoding/decoding

This enables BOMegaBench to serve as a comprehensive testbed for Bayesian Optimization algorithms on continuous, discrete, and mixed-type problems.
