# BOMegaBench 🎯

**Comprehensive Bayesian Optimization Benchmark Library**

BOMegaBench provides 200+ benchmark functions for evaluating Bayesian optimization and hyperparameter tuning algorithms, featuring unified interfaces, modular architecture, and extensive documentation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/BOResearch/BOMegaBench/actions/workflows/ci.yml/badge.svg)](https://github.com/BOResearch/BOMegaBench/actions/workflows/ci.yml)
[![Code Quality](https://github.com/BOResearch/BOMegaBench/actions/workflows/quality.yml/badge.svg)](https://github.com/BOResearch/BOMegaBench/actions/workflows/quality.yml)
[![Documentation](https://github.com/BOResearch/BOMegaBench/actions/workflows/docs.yml/badge.svg)](https://github.com/BOResearch/BOMegaBench/actions/workflows/docs.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 Features

- **200+ Benchmark Functions** across multiple domains
- **Unified Interface** - consistent API for all benchmarks
- **Modular Architecture** - well-organized, maintainable codebase
- **Optional Dependencies** - install only what you need
- **Type Hints** - full type annotation support
- **Comprehensive Documentation** - detailed guides and examples

## 📦 Available Benchmark Suites

| Suite | Functions | Description |
|-------|-----------|-------------|
| **Consolidated** | 72 | BBOB (24) + BoTorch (6) + Classical (42) functions |
| **LassoBench** | 13 | High-dimensional sparse regression benchmarks |
| **HPO** | 100+ | Machine learning hyperparameter optimization |
| **HPOBench** | 50+ | ML, NAS, OD, and RL benchmarks |
| **Database Tuning** | Custom | Database configuration optimization |

## 🚀 Quick Start

### Installation

```bash
# Basic installation (core functions only)
pip install -e .

# With optional dependencies
pip install -e ".[all]"  # All benchmarks
pip install -e ".[lasso]"  # LassoBench only
pip install -e ".[hpo]"  # HPO benchmarks only
pip install -e ".[dev]"  # Development tools
```

### Basic Usage

```python
import bomegabench as bmb

# List available benchmarks
suites = bmb.list_suites()
functions = bmb.list_functions()

# Get a specific function
func = bmb.get_function("sphere_2d")

# Evaluate
import torch
X = torch.rand(10, func.dim) * (func.bounds[1] - func.bounds[0]) + func.bounds[0]
y = func(X)

# Get function metadata
print(func.metadata)
```

### Advanced Usage

```python
# Create benchmark suite
from bomegabench import ConsolidatedSuite

suite = ConsolidatedSuite
func = suite["sphere_2d"]

# Run benchmark with BenchmarkRunner
from bomegabench import BenchmarkRunner

runner = BenchmarkRunner(func)
result = runner.run(optimizer="random", n_iterations=100)

# Visualize results
from bomegabench import plot_convergence
plot_convergence(result)
```

## 📂 Project Structure

```
BOMegaBench/
├── bomegabench/                    # Main package
│   ├── core.py                     # Base classes (BenchmarkFunction, BenchmarkSuite)
│   ├── benchmark.py                # Benchmark runner and result classes
│   ├── visualization.py            # Plotting utilities
│   ├── utils/                      # Utility modules
│   │   ├── dependencies.py         # ✨ Unified dependency management
│   │   └── __init__.py
│   └── functions/                  # Benchmark function implementations
│       ├── consolidated/           # ✨ Modular consolidated functions (72 functions)
│       │   ├── __init__.py
│       │   ├── bbob_functions.py          (599 lines - 24 BBOB functions)
│       │   ├── botorch_additional.py      (217 lines - 6 BoTorch functions)
│       │   ├── classical_additional.py    (838 lines - 32 classical functions)
│       │   └── classical_core.py          (285 lines - 10 core functions)
│       ├── database/               # ✨ NEW: Modular database tuning (1,069 lines)
│       │   ├── __init__.py                (90 lines)
│       │   ├── core.py                    (245 lines - DatabaseTuningFunction)
│       │   ├── knob_configs.py            (274 lines - PostgreSQL/MySQL configs)
│       │   ├── space_converter.py         (247 lines - Continuous-discrete conversion)
│       │   ├── evaluator.py               (143 lines - BenchBase integration)
│       │   └── suite.py                   (70 lines - Suite creation)
│       ├── lasso_bench.py          # LassoBench integration (260 lines)
│       ├── hpo_benchmarks.py       # Bayesmark HPO benchmarks (313 lines)
│       ├── hpobench_benchmarks.py  # HPOBench benchmarks (585 lines)
│       ├── benchbase_wrapper.py    # BenchBase wrapper (655 lines)
│       ├── consolidated_functions.py  # Backward compatibility layer (39 lines)
│       ├── database_tuning.py      # ⚠️ DEPRECATED: Use database/ instead (63 lines)
│       └── registry.py             # Function registry (261 lines)
├── tests/                          # Comprehensive test suite
│   ├── test_consolidated.py       # Consolidated functions tests
│   ├── test_dependencies.py       # Dependency management tests
│   └── test_database/             # ✨ NEW: Database tuning tests (40 tests)
│       ├── __init__.py
│       ├── test_knob_configs.py   # Knob configuration tests (18 tests)
│       ├── test_space_converter.py # Space conversion tests (15 tests)
│       └── test_core.py            # Core functionality tests (7 tests)
├── examples/                       # Example scripts
│   ├── basic_usage.py
│   ├── lasso_bench_example.py
│   └── hpo_benchmark_example.py
├── .github/                        # ✨ NEW: CI/CD workflows
│   ├── workflows/
│   │   ├── ci.yml                 # Main CI pipeline
│   │   ├── docs.yml               # Documentation generation
│   │   ├── quality.yml            # Code quality checks
│   │   └── release.yml            # Release automation
│   └── CONTRIBUTING.md            # ✨ NEW: Contribution guidelines
├── docs/                           # Documentation
├── .pre-commit-config.yaml         # Pre-commit hooks configuration
├── pyproject.toml                  # Modern Python project configuration
├── setup.py                        # Setuptools configuration
├── README.md                       # This file
├── REFACTORING_PROGRESS.md         # ✨ NEW: Refactoring progress tracker
└── REFACTORING_SUMMARY.md          # Previous refactoring summary
```

**Legend:**
- ✨ = Recently added/refactored
- ⚠️ = Deprecated (will be removed in future versions)

## 🔧 Development

### Setup Development Environment

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run tests with coverage
pytest --cov=bomegabench --cov-report=html
```

### Code Quality

```bash
# Format code
black bomegabench/
isort bomegabench/

# Lint
flake8 bomegabench/

# Type checking
mypy bomegabench/
```

## 📊 Benchmark Suites Details

### Consolidated Suite (72 functions)

Combines classic optimization benchmarks:
- **BBOB (24)**: Black-Box Optimization Benchmarking suite
- **BoTorch Additional (6)**: Special functions including mixed-integer and binary
- **Classical (42)**: Well-known test functions (Schwefel, Schaffer, Hartmann, etc.)

### LassoBench (13 functions)

High-dimensional sparse regression benchmarks:
- **Synthetic (8)**: Simple, Medium, High, Hard variants (noisy/noiseless)
- **Real-world (5)**: Diabetes, DNA, Breast Cancer, Leukemia, RCV1

### HPO Suite (100+ functions)

ML hyperparameter optimization via Bayesmark:
- Classification and regression tasks
- Multiple ML algorithms (XGBoost, RF, SVM, Neural Networks)
- Various datasets

### HPOBench Suite (50+ functions)

Comprehensive ML benchmarks:
- **ML (30+)**: Tabular ML benchmarks
- **NAS (8+)**: Neural Architecture Search
- **OD (8+)**: Outlier Detection
- **RL (2+)**: Reinforcement Learning

## 🎯 Key Design Principles

1. **Modularity**: Organized into logical submodules
2. **Extensibility**: Easy to add new benchmarks
3. **Type Safety**: Full type hints throughout
4. **Error Handling**: Graceful degradation for missing dependencies
5. **Documentation**: Comprehensive docstrings and examples
6. **Testing**: Extensive test coverage

## 📖 Documentation

- [Installation Guide](docs/installation.md)
- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Contributing Guide](docs/contributing.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- BBOB suite from COCO framework
- BoTorch team for additional test functions
- LassoBench authors
- HPOBench team
- Bayesmark framework

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Note**: Some benchmark suites require additional dependencies. Use the dependency management system to check installation status:

```python
from bomegabench.utils.dependencies import print_dependency_status
print_dependency_status()
```
