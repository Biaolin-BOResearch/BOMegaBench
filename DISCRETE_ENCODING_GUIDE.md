# Discrete Parameter Encoding Guide

## Overview

BOMegaBench provides three encoding modes for discrete parameters, allowing flexible integration with different Bayesian Optimization algorithms. The encoding mode is controlled via the `discrete_encoding` parameter when instantiating a benchmark function.

## Three Encoding Modes

### 1. Raw Mode (`discrete_encoding="raw"`)

**Description**: No encoding transformation. The optimizer works directly with discrete values.

**Use Case**:
- Optimizers that natively support discrete/categorical parameters
- Mixed-integer Bayesian Optimization algorithms
- When you want full control over discrete value handling

**Characteristics**:
- `optimizer_dim == problem_dim`
- Discrete parameters remain as integer indices in [0, n_options-1]
- No transformation applied during evaluation

**Example**:
```python
from bomegabench.utils import DiscreteParameterSpec
from bomegabench.functions import get_function

# Define discrete parameters
discrete_specs = [
    DiscreteParameterSpec(dim_index=2, n_options=3, options=['small', 'medium', 'large']),
    DiscreteParameterSpec(dim_index=5, n_options=4, options=[10, 20, 50, 100])
]

# Create benchmark with raw encoding
func = get_function('some_hpo_benchmark')
func_raw = func.__class__(
    discrete_specs=discrete_specs,
    discrete_encoding="raw"
)

# Optimizer provides discrete values directly
# X[..., 2] ∈ {0, 1, 2} (maps to 'small', 'medium', 'large')
# X[..., 5] ∈ {0, 1, 2, 3} (maps to 10, 20, 50, 100)
```

---

### 2. Interval Mode (`discrete_encoding="interval"`)

**Description**: Maps continuous values in [0,1] to discrete options via equal-interval partitioning.

**Use Case**:
- Standard Bayesian Optimization with continuous relaxation
- When you want smooth optimization over discrete choices
- Compatible with most BO implementations (BoTorch, GPyOpt, etc.)

**Characteristics**:
- `optimizer_dim == problem_dim`
- Discrete dimensions mapped to [0,1]
- For n options: [0, 1/n), [1/n, 2/n), ..., [(n-1)/n, 1] → options 0, 1, ..., n-1

**Example**:
```python
from bomegabench.utils import DiscreteParameterSpec
import torch

discrete_specs = [
    DiscreteParameterSpec(dim_index=2, n_options=3)  # 3 options
]

func = func.__class__(
    discrete_specs=discrete_specs,
    discrete_encoding="interval"
)

# Optimizer provides continuous values in [0, 1]
X_opt = torch.tensor([[0.5, 0.5, 0.15, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])  # dim=10

# For dim 2 with 3 options:
# [0, 0.333) → option 0
# [0.333, 0.667) → option 1
# [0.667, 1.0] → option 2
# X_opt[..., 2] = 0.15 → maps to option 0

y = func(X_opt)  # Discrete encoding handled automatically
```

**Interval Mapping Details**:
- 2 options: [0, 0.5) → 0, [0.5, 1] → 1
- 3 options: [0, 0.33) → 0, [0.33, 0.67) → 1, [0.67, 1] → 2
- 4 options: [0, 0.25) → 0, [0.25, 0.5) → 1, [0.5, 0.75) → 2, [0.75, 1] → 3

---

### 3. One-hot Mode (`discrete_encoding="onehot"`)

**Description**: One-hot encodes discrete parameters into multiple continuous dimensions in [0,1].

**Use Case**:
- When treating discrete choices as independent continuous dimensions
- Maximum flexibility for kernel-based methods
- Avoids artificial ordering of categorical variables

**Characteristics**:
- `optimizer_dim = problem_dim - n_discrete_params + Σ(n_options_i)`
- Each discrete parameter with n options becomes n dimensions
- Uses argmax to decode back to discrete index

**Example**:
```python
from bomegabench.utils import DiscreteParameterSpec
import torch

# Original problem: 10 dimensions
# Dim 2: 3 discrete options
# Dim 5: 4 discrete options
discrete_specs = [
    DiscreteParameterSpec(dim_index=2, n_options=3),
    DiscreteParameterSpec(dim_index=5, n_options=4)
]

func = func.__class__(
    discrete_specs=discrete_specs,
    discrete_encoding="onehot"
)

# Optimizer space: 10 - 2 + 3 + 4 = 15 dimensions
print(f"Original dim: 10")
print(f"Optimizer dim: {func.dim}")  # 15
print(f"Encoding info: {func.get_encoding_info()}")

# Optimizer provides 15-dimensional input
# Dims 0-1: continuous (original dims 0-1)
# Dims 2-4: one-hot for original dim 2 (3 options)
# Dims 5-6: continuous (original dims 3-4)
# Dims 7-10: one-hot for original dim 5 (4 options)
# Dims 11-14: continuous (original dims 6-9)

X_opt = torch.rand(5, 15)  # 5 samples in optimizer space
X_opt[0, 2:5] = torch.tensor([0.1, 0.8, 0.1])  # option 1 (argmax)
X_opt[0, 7:11] = torch.tensor([0.0, 0.0, 0.9, 0.1])  # option 2 (argmax)

y = func(X_opt)  # Automatically decodes one-hot to discrete values
```

**One-hot Dimension Calculation**:
```
optimizer_dim = n_continuous + Σ(n_options_i for each discrete param)

Example 1:
- Original: 10D with discrete params at dims 2 (3 options) and 5 (4 options)
- Continuous: 10 - 2 = 8
- One-hot: 3 + 4 = 7
- Optimizer: 8 + 7 = 15

Example 2:
- Original: 20D with discrete params at dims 0 (5 options), 10 (2 options), 19 (3 options)
- Continuous: 20 - 3 = 17
- One-hot: 5 + 2 + 3 = 10
- Optimizer: 17 + 10 = 27
```

---

## Complete Usage Example

### Example 1: Simple Benchmark with Discrete Parameters

```python
import torch
from bomegabench.core import BenchmarkFunction
from bomegabench.utils import DiscreteParameterSpec

class SimpleMixedFunction(BenchmarkFunction):
    """Example benchmark with 2 continuous + 2 discrete parameters."""

    def __init__(self, discrete_encoding="onehot"):
        # Problem space: 4 dimensions
        # Dims 0-1: continuous
        # Dim 2: discrete with 3 options (algorithm type)
        # Dim 3: discrete with 5 options (batch size)

        discrete_specs = [
            DiscreteParameterSpec(
                dim_index=2,
                n_options=3,
                options=['sgd', 'adam', 'rmsprop']
            ),
            DiscreteParameterSpec(
                dim_index=3,
                n_options=5,
                options=[16, 32, 64, 128, 256]
            )
        ]

        bounds = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],  # lower bounds
            [1.0, 1.0, 2.0, 4.0]   # upper bounds (discrete bounds ignored in encoding)
        ])

        super().__init__(
            dim=4,
            bounds=bounds,
            discrete_specs=discrete_specs,
            discrete_encoding=discrete_encoding
        )

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """Evaluation in problem space (discrete values are indices)."""
        # X shape: (..., 4)
        # X[..., 0:2]: continuous learning rate and momentum
        # X[..., 2]: discrete algorithm index {0, 1, 2}
        # X[..., 3]: discrete batch size index {0, 1, 2, 3, 4}

        lr = X[..., 0]
        momentum = X[..., 1]
        algo_idx = X[..., 2].long()
        batch_idx = X[..., 3].long()

        # Some made-up objective
        loss = (lr - 0.01)**2 + (momentum - 0.9)**2
        loss = loss + 0.1 * algo_idx  # Prefer earlier algorithms
        loss = loss + 0.05 * batch_idx  # Prefer smaller batches

        return loss

    def _get_metadata(self):
        return {
            'name': 'SimpleMixedFunction',
            'description': 'Example mixed continuous-discrete optimization'
        }


# Test all three encoding modes
print("=" * 60)
print("1. RAW ENCODING")
print("=" * 60)
func_raw = SimpleMixedFunction(discrete_encoding="raw")
print(f"Optimizer dim: {func_raw.dim}")  # 4
print(f"Bounds shape: {func_raw.bounds.shape}")  # (2, 4)
print(f"Encoding info: {func_raw.get_encoding_info()}")

X_raw = torch.tensor([[0.01, 0.9, 1.0, 2.0]])  # Direct discrete values
y_raw = func_raw(X_raw)
print(f"Input: {X_raw}")
print(f"Output: {y_raw}\n")


print("=" * 60)
print("2. INTERVAL ENCODING")
print("=" * 60)
func_interval = SimpleMixedFunction(discrete_encoding="interval")
print(f"Optimizer dim: {func_interval.dim}")  # 4
print(f"Bounds: {func_interval.bounds}")
print(f"Encoding info: {func_interval.get_encoding_info()}")

# Optimizer provides [0,1] values for discrete dims
X_interval = torch.tensor([[0.01, 0.9, 0.4, 0.5]])  # 0.4 → option 1, 0.5 → option 2
X_problem = func_interval.encode_to_problem_space(X_interval)
print(f"Optimizer input: {X_interval}")
print(f"Problem space: {X_problem}")  # Discrete dims converted to indices
y_interval = func_interval(X_interval)
print(f"Output: {y_interval}\n")


print("=" * 60)
print("3. ONE-HOT ENCODING")
print("=" * 60)
func_onehot = SimpleMixedFunction(discrete_encoding="onehot")
print(f"Optimizer dim: {func_onehot.dim}")  # 4 - 2 + 3 + 5 = 10
print(f"Bounds shape: {func_onehot.bounds.shape}")  # (2, 10)
print(f"Encoding info: {func_onehot.get_encoding_info()}")

# Optimizer provides 10-dimensional input
# Dims 0-1: continuous (original dims 0-1)
# Dims 2-4: one-hot for algorithm (original dim 2, 3 options)
# Dims 5-9: one-hot for batch size (original dim 3, 5 options)
X_onehot = torch.zeros(1, 10)
X_onehot[0, 0] = 0.01  # learning rate
X_onehot[0, 1] = 0.9   # momentum
X_onehot[0, 2:5] = torch.tensor([0.0, 1.0, 0.0])  # algorithm: option 1 (adam)
X_onehot[0, 5:10] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])  # batch size: option 2 (64)

X_problem_onehot = func_onehot.encode_to_problem_space(X_onehot)
print(f"Optimizer input shape: {X_onehot.shape}")
print(f"Optimizer input: {X_onehot}")
print(f"Problem space: {X_problem_onehot}")
y_onehot = func_onehot(X_onehot)
print(f"Output: {y_onehot}")
```

---

### Example 2: Using with Existing Benchmarks

```python
from bomegabench.functions import get_function
from bomegabench.utils import DiscreteParameterSpec, create_encoder_for_hpo

# Example: HPO benchmark with mixed parameters
# (This is a conceptual example - actual implementation depends on specific benchmark)

# Get an HPO benchmark
func = get_function('some_hpo_task', suite='hpo')

# Check if it has discrete parameters
if hasattr(func, 'param_config'):
    param_config = func.param_config

    # Create encoder with different modes
    encoder_raw = create_encoder_for_hpo(param_config, encoding="raw")
    encoder_interval = create_encoder_for_hpo(param_config, encoding="interval")
    encoder_onehot = create_encoder_for_hpo(param_config, encoding="onehot")

    print(f"Raw encoding - optimizer dim: {encoder_raw.optimizer_dim}")
    print(f"Interval encoding - optimizer dim: {encoder_interval.optimizer_dim}")
    print(f"One-hot encoding - optimizer dim: {encoder_onehot.optimizer_dim}")
```

---

### Example 3: Integration with BoTorch

```python
import torch
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood

# Create benchmark with interval encoding (smooth continuous optimization)
func = SimpleMixedFunction(discrete_encoding="interval")

# Initial random samples in optimizer space
n_init = 10
X_init = func.sample_random(n_init)
Y_init = func(X_init).unsqueeze(-1)

# Fit GP model
gp = SingleTaskGP(X_init, Y_init)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

# Optimize acquisition function
ei = ExpectedImprovement(gp, best_f=Y_init.min())
bounds_botorch = func.bounds  # Already in optimizer space

candidate, acq_value = optimize_acqf(
    ei,
    bounds=bounds_botorch,
    q=1,
    num_restarts=10,
    raw_samples=100,
)

# Evaluate candidate
y_candidate = func(candidate)
print(f"Candidate: {candidate}")
print(f"Value: {y_candidate}")

# Convert to problem space if needed for interpretation
X_problem = func.encode_to_problem_space(candidate)
print(f"Problem space representation: {X_problem}")
```

---

## Choosing the Right Encoding Mode

| Mode | Optimizer Type | Advantages | Disadvantages |
|------|---------------|------------|---------------|
| **raw** | Mixed-integer BO | - No dimension change<br>- Direct discrete handling<br>- Preserves problem structure | - Requires MI-BO algorithm<br>- Not compatible with standard BO |
| **interval** | Standard BO | - No dimension change<br>- Works with any BO<br>- Smooth relaxation | - May have poor kernel behavior at boundaries<br>- Artificial metric on discrete vars |
| **onehot** | Standard BO | - No artificial ordering<br>- Each option independent<br>- Better for categorical vars | - Increases dimensionality<br>- Redundant representation<br>- May need dimension reduction for many options |

### Recommendations

1. **Use `raw`** if:
   - You have a mixed-integer BO algorithm (e.g., SMAC, Optuna)
   - You want to preserve the exact discrete nature
   - Dimension is not a concern

2. **Use `interval`** if:
   - You're using standard Bayesian Optimization (BoTorch, GPyOpt, etc.)
   - You want minimal dimension change
   - Discrete parameters have natural ordering (e.g., integers)

3. **Use `onehot`** if:
   - Your discrete parameters are truly categorical (no ordering)
   - You want to avoid artificial metric structure
   - The dimension increase is manageable (< 20 total options)

---

## API Reference

### DiscreteParameterSpec

```python
class DiscreteParameterSpec:
    """Specification for a discrete parameter."""

    def __init__(
        self,
        dim_index: int,          # Index in original parameter vector
        n_options: int,          # Number of discrete options
        options: Optional[List] = None  # Actual option values (optional)
    )
```

### DiscreteEncoder

```python
class DiscreteEncoder:
    """Encoder/Decoder for discrete parameters."""

    def __init__(
        self,
        continuous_dims: int,              # Total dimensions (continuous + discrete)
        discrete_specs: List[DiscreteParameterSpec],
        encoding: str = "onehot"           # "raw", "interval", or "onehot"
    )

    @property
    def optimizer_dim(self) -> int:
        """Dimension in optimizer space."""

    def encode(self, X: Tensor) -> Tensor:
        """Optimizer space → Problem space."""

    def decode(self, X_problem: Tensor) -> Tensor:
        """Problem space → Optimizer space."""

    def get_optimizer_bounds(self, problem_bounds: Tensor) -> Tensor:
        """Get bounds for optimizer space."""

    def get_info(self) -> Dict:
        """Get encoding information."""
```

### BenchmarkFunction Methods

```python
class BenchmarkFunction:
    def __init__(
        self,
        dim: int,
        bounds: Tensor,
        discrete_specs: Optional[List[DiscreteParameterSpec]] = None,
        discrete_encoding: str = "onehot",
        **kwargs
    )

    def encode_to_problem_space(self, X_optimizer: Tensor) -> Tensor:
        """Convert optimizer space → problem space."""

    def decode_from_problem_space(self, X_problem: Tensor) -> Tensor:
        """Convert problem space → optimizer space."""

    def get_encoding_info(self) -> Dict:
        """Get encoding information."""
```

---

## Testing and Validation

```python
import torch
from bomegabench.utils import DiscreteParameterSpec, DiscreteEncoder

# Test all encoding modes
def test_encoding_consistency():
    """Test that encode/decode are consistent."""

    discrete_specs = [
        DiscreteParameterSpec(dim_index=1, n_options=3),
        DiscreteParameterSpec(dim_index=3, n_options=4)
    ]

    problem_dim = 5
    bounds = torch.tensor([[0.0]*5, [1.0]*5])

    for encoding in ["raw", "interval", "onehot"]:
        print(f"\nTesting {encoding} encoding...")
        encoder = DiscreteEncoder(problem_dim, discrete_specs, encoding)

        # Sample in problem space
        X_problem = torch.tensor([[0.5, 1.0, 0.3, 2.0, 0.8]])  # Discrete values as indices

        # Encode to optimizer space
        X_opt = encoder.decode(X_problem)
        print(f"  Problem → Optimizer: {X_problem} → {X_opt}")

        # Decode back to problem space
        X_problem_reconstructed = encoder.encode(X_opt)
        print(f"  Optimizer → Problem: {X_opt} → {X_problem_reconstructed}")

        # Check consistency
        assert torch.allclose(X_problem, X_problem_reconstructed, atol=0.01)
        print(f"  ✓ Consistency check passed")

test_encoding_consistency()
```

---

## FAQs

**Q: Can I change the encoding mode after instantiation?**
A: No, the encoding mode is fixed at instantiation. Create a new instance if you need a different encoding.

**Q: What happens if I provide invalid discrete values in raw mode?**
A: Values are clamped to the valid range [0, n_options-1] during encoding.

**Q: How does one-hot encoding handle soft assignments?**
A: It uses argmax, so [0.3, 0.5, 0.2] → option 1 (index with maximum value).

**Q: Can I mix discrete encodings (e.g., interval for some, onehot for others)?**
A: No, all discrete parameters use the same encoding mode. If you need different encodings, you'll need to implement custom logic.

**Q: How do I know which encoding mode a benchmark instance is using?**
A: Call `func.get_encoding_info()` to see the encoding configuration.

**Q: Does the encoding affect the optimal solution?**
A: The optimal value should be the same, but the optimizer's path and convergence behavior may differ significantly between encoding modes.
