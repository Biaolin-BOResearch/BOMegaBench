# Contributing to BOMegaBench

Thank you for your interest in contributing to BOMegaBench!

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/BOResearch/BOMegaBench.git
cd BOMegaBench
```

### 2. Install development dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

### 3. Install pre-commit hooks

```bash
pre-commit install
```

This will automatically run code formatting and linting checks before each commit.

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=bomegabench --cov-report=html

# Run specific test file
pytest tests/test_consolidated.py

# Run tests matching a pattern
pytest -k "test_database"
```

### Code Formatting

We use `black`, `isort`, and `flake8` for code formatting and linting:

```bash
# Format code
black bomegabench
isort bomegabench

# Check formatting without making changes
black --check bomegabench
isort --check-only bomegabench

# Run linter
flake8 bomegabench
```

### Type Checking

We use `mypy` for type checking:

```bash
mypy bomegabench --ignore-missing-imports
```

### Running All Checks

```bash
# Run pre-commit on all files
pre-commit run --all-files
```

## Pull Request Process

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit with descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

3. **Push to your fork** and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Ensure CI passes**: All GitHub Actions workflows must pass before merging.

5. **Request review**: Tag relevant maintainers for review.

## Code Style Guidelines

### General Principles

- Follow PEP 8 style guide
- Use type hints for all function signatures
- Write docstrings in NumPy style
- Keep functions focused and modular
- Add tests for new features

### Type Hints Example

```python
from typing import Dict, List, Optional, Any
import torch
from torch import Tensor

def example_function(
    x: Tensor,
    config: Dict[str, Any],
    optional_param: Optional[int] = None
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Short description of the function.

    Parameters
    ----------
    x : Tensor
        Description of x
    config : Dict[str, Any]
        Description of config
    optional_param : int, optional
        Description of optional parameter

    Returns
    -------
    Tuple[Tensor, Dict[str, float]]
        Description of return values
    """
    # Implementation
    pass
```

### Docstring Example (NumPy Style)

```python
def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics for predictions.

    Parameters
    ----------
    predictions : np.ndarray
        Array of predicted values, shape (n_samples,)
    targets : np.ndarray
        Array of target values, shape (n_samples,)

    Returns
    -------
    Dict[str, float]
        Dictionary containing metric names and their values

    Raises
    ------
    ValueError
        If predictions and targets have different shapes

    Examples
    --------
    >>> predictions = np.array([1.0, 2.0, 3.0])
    >>> targets = np.array([1.1, 2.1, 2.9])
    >>> metrics = compute_metrics(predictions, targets)
    >>> print(metrics['mse'])
    0.01
    """
    pass
```

## Adding New Benchmarks

### 1. Create a new module

```python
# bomegabench/functions/my_new_benchmark.py

import torch
from torch import Tensor
from ..core import BenchmarkFunction, BenchmarkSuite

class MyNewFunction(BenchmarkFunction):
    """Description of your benchmark function."""

    def __init__(self, dim: int = 2, **kwargs):
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim])
        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        # Implementation
        return torch.sum(X ** 2, dim=-1)

    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "My New Function",
            "suite": "My Benchmarks",
            "properties": ["continuous", "differentiable"],
        }
```

### 2. Add tests

```python
# tests/test_my_new_benchmark.py

import pytest
import torch
from bomegabench.functions.my_new_benchmark import MyNewFunction

def test_my_new_function():
    func = MyNewFunction(dim=2)
    assert func.dim == 2

    X = torch.rand(10, 2)
    y = func(X)
    assert y.shape == (10,)
```

### 3. Register in functions/__init__.py

```python
try:
    from .my_new_benchmark import MyNewBenchmarkSuite
    MY_NEW_AVAILABLE = True
except ImportError:
    MY_NEW_AVAILABLE = False
```

## Dependency Management

### Adding New Dependencies

1. **Core dependencies** go in `pyproject.toml` under `dependencies`
2. **Optional dependencies** go under `[project.optional-dependencies]`
3. **Development dependencies** go under `dev`

Example:
```toml
[project.optional-dependencies]
mynew = [
    "my-package>=1.0.0",
]
```

### Using Dependencies

Always use the unified dependency checker:

```python
from bomegabench.utils.dependencies import check_dependency, require_dependency

# Check if available
available, module = check_dependency("my_package")

# Require (raises ImportError if missing)
my_module = require_dependency("my_package", "My new feature")
```

## Testing Guidelines

### Test Structure

```
tests/
├── test_core.py              # Test core functionality
├── test_consolidated.py      # Test consolidated functions
├── test_database/            # Database tuning tests
│   ├── test_core.py
│   ├── test_evaluator.py
│   └── test_space_converter.py
└── test_integration.py       # Integration tests
```

### Writing Tests

```python
import pytest
import torch
from bomegabench import get_function

class TestMyFeature:
    def test_basic_functionality(self):
        """Test basic functionality."""
        func = get_function("sphere_2d")
        assert func is not None

    def test_evaluation(self):
        """Test function evaluation."""
        func = get_function("sphere_2d")
        X = torch.zeros(1, 2)
        y = func(X)
        assert torch.allclose(y, torch.tensor([0.0]))

    @pytest.mark.parametrize("dim", [2, 5, 10])
    def test_dimensions(self, dim):
        """Test different dimensions."""
        # Test implementation
        pass
```

## Documentation

### Building Documentation

```bash
cd docs
make html
```

### Viewing Documentation

```bash
python -m http.server --directory docs/_build/html
```

## Questions?

If you have questions or need help, please:
- Open an issue on GitHub
- Contact the maintainers
- Check existing documentation

Thank you for contributing to BOMegaBench!
