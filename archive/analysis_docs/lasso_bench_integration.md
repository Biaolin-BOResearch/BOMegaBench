# LassoBench Integration for BOMegaBench

This document describes the integration of LassoBench high-dimensional hyperparameter optimization benchmarks into the BOMegaBench framework.

## Overview

LassoBench provides high-dimensional benchmarks for hyperparameter optimization based on Weighted Lasso regression. The integration adds 23 new benchmark functions across 3 suites:

- **LassoBench Synthetic Suite** (8 functions): Synthetic benchmarks with known sparse structure
- **LassoBench Real-world Suite** (5 functions): Real-world datasets from medicine and finance  
- **LassoBench Multi-Fidelity Suite** (10 functions): Multi-fidelity optimization benchmarks

## Installation

### Basic Installation (without LassoBench)
BOMegaBench will work normally without LassoBench, but the LassoBench suites will not be available.

### Full Installation (with LassoBench)
To use LassoBench functions, install the LassoBench library:

```bash
pip install git+https://github.com/ksehic/LassoBench.git
```

This will automatically install the required dependencies:
- `celer` (for Lasso regression)
- `sparse-ho` (for sparse hyperparameter optimization)
- `libsvmdata` (for real-world datasets)
- `scikit-learn` (for machine learning utilities)

## Available Functions

### Synthetic Benchmarks

| Function Name | Dimensions | Active Dims | Description |
|---------------|------------|-------------|-------------|
| `synt_simple_noiseless` | 60 | 3 | Simple benchmark, noiseless |
| `synt_simple_noisy` | 60 | 3 | Simple benchmark, noisy |
| `synt_medium_noiseless` | 100 | 5 | Medium benchmark, noiseless |
| `synt_medium_noisy` | 100 | 5 | Medium benchmark, noisy |
| `synt_high_noiseless` | 300 | 15 | High-dimensional, noiseless |
| `synt_high_noisy` | 300 | 15 | High-dimensional, noisy |
| `synt_hard_noiseless` | 1000 | 50 | Very high-dimensional, noiseless |
| `synt_hard_noisy` | 1000 | 50 | Very high-dimensional, noisy |

### Real-world Benchmarks

| Function Name | Dimensions | Approx. Active | Dataset Source |
|---------------|------------|----------------|----------------|
| `breast_cancer` | 10 | ~3 | Medical data |
| `diabetes` | 8 | ~5 | Medical data |
| `dna` | 180 | ~43 | Bioinformatics |
| `leukemia` | 7,129 | ~22 | Medical data |
| `rcv1` | 19,959 | ~75 | Text classification |

### Multi-Fidelity Benchmarks

Multi-fidelity versions of synthetic and real benchmarks with both discrete and continuous fidelity:

- `synt_simple_mf_discrete` / `synt_simple_mf_continuous`
- `synt_medium_mf_discrete` / `synt_medium_mf_continuous` 
- `synt_high_mf_discrete` / `synt_high_mf_continuous`
- `rcv1_mf_discrete` / `rcv1_mf_continuous`
- `dna_mf_discrete` / `dna_mf_continuous`

## Usage Examples

### Basic Usage

```python
import bomegabench as bmb
import torch

# List LassoBench suites (if available)
suites = bmb.list_suites()
lasso_suites = [s for s in suites if 'lasso' in s]
print(f"LassoBench suites: {lasso_suites}")

# Get a synthetic function
func = bmb.get_function("synt_simple_noiseless", "lasso_synthetic")
print(f"Function: {func.metadata['name']}")
print(f"Dimension: {func.dim}")
print(f"Active dimensions: {func.metadata['active_dimensions']}")

# Evaluate function
X = torch.rand(1, func.dim) * 2 - 1  # Scale to [-1, 1]
result = func(X)
print(f"Result: {result.item()}")
```

### Advanced Usage with Test Metrics

```python
import bomegabench as bmb
import torch

# Get function
func = bmb.get_function("synt_simple_noiseless", "lasso_synthetic")

# Generate test point
X = torch.rand(1, func.dim) * 2 - 1

# Evaluate function
objective_value = func(X)

# Get test metrics (MSE and F-score for synthetic functions)
if hasattr(func, 'get_test_metrics'):
    metrics = func.get_test_metrics(X)
    print(f"Test MSE: {metrics['mspe']:.6f}")
    print(f"F-score: {metrics['fscore']:.6f}")

# Get active dimensions
if hasattr(func, 'get_active_dimensions'):
    active_dims = func.get_active_dimensions()
    print(f"Active dimension indices: {active_dims}")

# Get true weights (for synthetic functions)
if hasattr(func, 'get_true_weights'):
    true_weights = func.get_true_weights()
    print(f"True weight sparsity: {(true_weights != 0).sum()} / {len(true_weights)}")
```

### Multi-Fidelity Usage

```python
import bomegabench as bmb
import torch

# Get multi-fidelity function
func = bmb.get_function("synt_simple_mf_discrete", "lasso_multifidelity")

# Generate test point
X = torch.rand(1, func.dim) * 2 - 1

# Evaluate at different fidelities
for fidelity in [0, 2, 4]:  # Low, medium, high fidelity
    if hasattr(func, 'evaluate_fidelity'):
        result = func.evaluate_fidelity(X, fidelity_index=fidelity)
        print(f"Fidelity {fidelity}: {result.item():.6f}")

# Highest fidelity evaluation (default)
result = func(X)
print(f"Highest fidelity: {result.item():.6f}")
```

### Real-world Dataset Usage

```python
import bomegabench as bmb
import torch

# Get real-world function (diabetes dataset)
func = bmb.get_function("diabetes", "lasso_real")
print(f"Dataset: {func.metadata['dataset']}")
print(f"Dimensions: {func.dim}")

# Evaluate function
X = torch.rand(1, func.dim) * 2 - 1
result = func(X)

# Get test MSE
if hasattr(func, 'get_test_metrics'):
    metrics = func.get_test_metrics(X)
    print(f"Test MSE: {metrics['mspe']:.6f}")
```

## Function Properties

All LassoBench functions share these properties:

- **Domain**: `[-1, 1]^d` where `d` is the function dimension
- **Objective**: Minimize cross-validation loss (lower is better)
- **Sparse Structure**: Only a subset of dimensions are truly important
- **Differentiable**: Functions are differentiable (through automatic differentiation)

### Synthetic Function Properties

- **Oracle Available**: True optimal value known for comparison
- **F-score Evaluation**: Measures sparse structure recovery
- **Controllable Noise**: Available in both noisy and noiseless versions

### Real-world Function Properties

- **Realistic**: Based on real datasets from medicine, finance, and text processing
- **Variable Difficulty**: From 8 dimensions (diabetes) to 19,959 dimensions (RCV1)
- **Domain Knowledge**: Benchmarks reflect real-world hyperparameter optimization challenges

### Multi-Fidelity Properties

- **Discrete Fidelity**: 5 fidelity levels (0-4, where 4 is highest)
- **Continuous Fidelity**: Continuous fidelity parameter in [0, 1]
- **Computational Trade-off**: Lower fidelity = faster evaluation, less accuracy

## Integration Details

### Architecture

The integration follows BOMegaBench's unified interface:

1. **Wrapper Classes**: `LassoBenchSyntheticFunction`, `LassoBenchRealFunction`, `LassoBenchMultiFidelityFunction`
2. **Suite Creation**: Factory functions create complete suites
3. **Optional Dependency**: Graceful handling when LassoBench is not installed
4. **Metadata**: Rich metadata including active dimensions, properties, descriptions

### Error Handling

- **Import Errors**: Clear messages when LassoBench is not installed
- **Bound Checking**: Input validation for the `[-1, 1]^d` domain
- **Graceful Degradation**: BOMegaBench works normally without LassoBench

### Performance Considerations

- **Lazy Loading**: LassoBench instances created only when needed
- **Batch Evaluation**: Efficient handling of multiple evaluations
- **Memory Management**: Proper cleanup of temporary objects

## Testing

Run the integration test:

```bash
python test_lasso_bench_integration.py
```

This test:
1. Verifies basic BOMegaBench functionality
2. Checks LassoBench availability and integration
3. Tests function creation and evaluation
4. Provides clear feedback about what's working

## Troubleshooting

### LassoBench Not Found
```
ImportError: LassoBench is required but not installed
```
**Solution**: Install LassoBench with `pip install git+https://github.com/ksehic/LassoBench.git`

### Missing Dependencies
```
ImportError: No module named 'celer'
```
**Solution**: LassoBench installation should handle dependencies automatically. If not, install manually:
```bash
pip install celer sparse-ho libsvmdata scikit-learn
```

### Domain Errors
```
ValueError: The configuration is outside the bounds.
```
**Solution**: Ensure input values are in `[-1, 1]`. Scale your inputs accordingly.

### Memory Issues with Large Functions
For very high-dimensional functions (like RCV1 with ~20k dimensions), consider:
- Using smaller batch sizes
- Testing with smaller functions first (diabetes, breast_cancer)
- Monitoring memory usage

## Citation

If you use LassoBench functions in your research, please cite:

```
Šehić Kenan, Gramfort Alexandre, Salmon Joseph and Nardi Luigi, 
"LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso", 
Proceedings of the 1st International Conference on Automated Machine Learning, 2022.
```

## References

- [LassoBench GitHub Repository](https://github.com/ksehic/LassoBench)
- [LassoBench Paper](https://openreview.net/forum?id=wG11m_aSIrz)
- [BOMegaBench Documentation](./README.md) 