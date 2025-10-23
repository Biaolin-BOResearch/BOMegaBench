# BOMegaBench 🎯

**Comprehensive Bayesian Optimization Benchmark Library**

BOMegaBench provides **500+ benchmark functions** across **11 major categories** for evaluating Bayesian optimization and hyperparameter tuning algorithms, featuring unified interfaces, modular architecture, and extensive documentation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/BOResearch/BOMegaBench/actions/workflows/ci.yml/badge.svg)](https://github.com/BOResearch/BOMegaBench/actions/workflows/ci.yml)
[![Code Quality](https://github.com/BOResearch/BOMegaBench/actions/workflows/quality.yml/badge.svg)](https://github.com/BOResearch/BOMegaBench/actions/workflows/quality.yml)
[![Documentation](https://github.com/BOResearch/BOMegaBench/actions/workflows/docs.yml/badge.svg)](https://github.com/BOResearch/BOMegaBench/actions/workflows/docs.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 Features

- **500+ Benchmark Functions** across 11 major categories
- **Wide Dimension Range** - from 2D to 19,959D problems
- **Diverse Parameter Types** - continuous, discrete, categorical, and mixed-integer
- **Unified Interface** - consistent API for all benchmarks
- **Real-World Applications** - robotics, chemistry, materials science, ML/RL
- **Modular Architecture** - well-organized, maintainable codebase
- **Optional Dependencies** - install only what you need
- **Type Hints** - full type annotation support
- **Comprehensive Documentation** - detailed guides and examples

## 📦 Available Benchmark Suites

| Suite | Functions | Dimensions | Parameter Types | Description |
|-------|-----------|------------|-----------------|-------------|
| **1. Synthetic Functions** | 72 | 2-53D | Continuous, Discrete, Mixed | BBOB (24) + BoTorch (6) + Classical (42) |
| **2. LassoBench** | 13 | 8-19,959D | Continuous | High-dimensional sparse regression |
| **3. HPOBench** | 50+ | Variable | Mixed-integer | ML, NAS, OD, RL benchmarks |
| **4. HPO Benchmarks** | 100+ | Variable | Mixed-integer | Bayesmark hyperparameter optimization |
| **5. Database Tuning** | 2 DBs | 100-200D | Mixed-integer | PostgreSQL & MySQL configuration |
| **6. Olympus Surfaces** | 20+ | 2-10D | Continuous, Discrete, Categorical | Synthetic chemistry/materials surfaces |
| **7. Olympus Datasets** | 40+ | 3-9D | Mixed | Real chemistry/materials experiments |
| **8. MuJoCo Control** | 10 | 36-12,569D | Continuous | Robot locomotion control |
| **9. Robosuite** | 22 | 300-2,000D | Continuous | Robot manipulation tasks |
| **10. HumanoidBench** | 320+ | 1,444-6,099D | Continuous | Humanoid robot locomotion & manipulation |
| **11. Design-Bench** | 25+ | 32-4,740D | Discrete, One-hot | Materials, proteins, DNA/RNA, NAS |

**Total: 500+ benchmark tasks** covering classical optimization, hyperparameter tuning, robotics, chemistry, materials science, and molecular design.

## 🚀 Quick Start

### Installation

```bash
# Basic installation (core + Synthetic functions)
pip install -e .

# With optional dependencies
pip install -e ".[all]"        # All benchmarks (requires significant dependencies)
pip install -e ".[lasso]"      # LassoBench only
pip install -e ".[hpo]"        # HPO benchmarks (Bayesmark)
pip install -e ".[hpobench]"   # HPOBench suite
pip install -e ".[olympus]"    # Olympus surfaces & datasets
pip install -e ".[mujoco]"     # MuJoCo control tasks
pip install -e ".[robosuite]"  # Robosuite manipulation
pip install -e ".[humanoid]"   # HumanoidBench tasks
pip install -e ".[design]"     # Design-Bench tasks
pip install -e ".[database]"   # Database tuning (requires BenchBase)
pip install -e ".[dev]"        # Development tools
```

**Note**: Some suites require additional system dependencies (e.g., MuJoCo license, BenchBase installation). See [BENCHMARK_CATALOG.md](BENCHMARK_CATALOG.md) for details.

### Basic Usage

```python
import bomegabench as bmb
import torch

# List available benchmarks
print("Available suites:", bmb.list_suites())
# Output: ['synthetic', 'lasso_synthetic', 'lasso_real', 'hpo', ...]

# Get a specific function
func = bmb.get_function("F01_SphereRaw_2d")  # Synthetic function
print(f"Dimension: {func.dim}, Bounds: {func.bounds}")

# Evaluate the function
X = torch.rand(10, func.dim) * (func.bounds[1] - func.bounds[0]) + func.bounds[0]
y = func(X)
print(f"Shape: {y.shape}")  # (10,)

# Access metadata
print(func.metadata)
# {'name': 'Sphere', 'suite': 'BBOB', 'properties': ['unimodal', 'separable'], ...}

# Example: Use a suite directly
from bomegabench import SyntheticSuite
suite = SyntheticSuite
rastrigin = suite["F03_RastriginSeparableRaw_4d"]
```

### Advanced Usage

```python
# Example 1: High-dimensional sparse optimization (LassoBench)
from bomegabench.functions import lasso_bench
func = lasso_bench.LassoBenchSyntheticSuite["synt_high"]  # 300D
print(f"Dimension: {func.dim}, Sparsity: ~15 active dims")

# Example 2: Hyperparameter optimization (HPOBench)
from bomegabench.functions import hpobench_benchmarks
xgb_func = hpobench_benchmarks.HPOBenchMLSuite["xgboost_task_31"]
# Optimize XGBoost hyperparameters on OpenML task 31

# Example 3: Robot control (MuJoCo)
from bomegabench.functions import mujoco_control
hopper_func = mujoco_control.MuJoCoSuite["hopper_linear"]  # 36D controller
# Optimize linear controller for Hopper robot

# Example 4: Chemistry optimization (Olympus)
from bomegabench.functions import olympus_datasets
perov_func = olympus_datasets.OlympusDatasetsSuite["perovskites"]
# Optimize perovskite material composition

# Run benchmark with custom optimizer
from bomegabench import BenchmarkRunner
runner = BenchmarkRunner(func)
result = runner.run(optimizer="random", n_iterations=100)

# Visualize results
from bomegabench import plot_convergence, plot_comparison
plot_convergence(result)

# Compare multiple algorithms
results = {
    "Random": runner.run(optimizer="random", n_iterations=100),
    "BO": runner.run(optimizer="bo", n_iterations=100),
}
plot_comparison(results)
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
│       ├── synthetic/           # ✨ Modular synthetic functions (72 functions)
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
│       ├── synthetic_functions.py  # Backward compatibility layer (39 lines)
│       ├── database_tuning.py      # ⚠️ DEPRECATED: Use database/ instead (63 lines)
│       └── registry.py             # Function registry (261 lines)
├── tests/                          # Comprehensive test suite
│   ├── test_synthetic.py       # Synthetic functions tests
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

### 1. Synthetic Functions (72 functions)

Classic optimization test functions:
- **BBOB (24)**: Black-Box Optimization Benchmarking suite - includes Sphere, Rastrigin, Rosenbrock, etc.
- **BoTorch Additional (6)**: Bukin, Cosine8, Three-Hump Camel, AckleyMixed, LABS, Shekel
- **Classical Additional (32)**: Schwefel variants, Schaffer variants, Hartmann, Alpine, etc.
- **Classical Core (10)**: Styblinski-Tang, Levy, Michalewicz, Zakharov, etc.
- **Dimensions**: Flexible (2D to 53D)
- **Parameter Types**: Continuous, discrete (one-hot encoded), mixed-integer

### 2. LassoBench (13 tasks)

High-dimensional sparse regression benchmarks:
- **Synthetic (8)**: Simple (60D), Medium (100D), High (300D), Hard (1000D) - noisy/noiseless variants
- **Real-world (5)**: Diabetes (8D), Breast Cancer (10D), DNA (180D), Leukemia (7,129D), RCV1 (19,959D)
- **Use Case**: Sparse optimization, feature selection
- **Parameter Type**: Continuous

### 3. HPOBench (50+ tasks)

Comprehensive ML hyperparameter optimization:
- **ML Benchmarks (32)**: XGBoost, SVM, RF, LR, NN, HistGB × 4 datasets
- **Outlier Detection (8)**: KDE, OneClassSVM × 4 datasets
- **Neural Architecture Search (8)**: NASBench101, NASBench201, Tabular NAS
- **Reinforcement Learning (2+)**: CartPole RL hyperparameters
- **Surrogate Benchmarks (3)**: ParamNet-Adult, ParamNet-Higgs, SurrogateSVM
- **Parameter Types**: Mixed-integer (continuous + one-hot encoded discrete/categorical)

### 4. HPO Benchmarks / Bayesmark (100+ tasks)

Standardized hyperparameter optimization:
- **Models (8)**: Decision Tree, MLP-sgd, Random Forest, SVM, AdaBoost, kNN, Lasso, Linear
- **Classification Datasets (4)**: Iris, Wine, Digits, Breast Cancer
- **Regression Datasets (2)**: Boston Housing, Diabetes
- **Total Combinations**: 8 models × 6 datasets × multiple metrics ≈ 100+ tasks
- **Parameter Types**: Mixed-integer

### 5. Database Tuning (2 databases)

Database configuration optimization:
- **PostgreSQL (8 knobs)**: shared_buffers, effective_cache_size, work_mem, max_connections, etc.
- **MySQL (5 knobs)**: innodb_buffer_pool_size, innodb_log_file_size, max_connections, etc.
- **Dimensions**: 100-200D (after one-hot encoding)
- **Objective**: Maximize throughput or minimize latency
- **Parameter Types**: Mixed-integer

### 6. Olympus Surfaces (20+ surfaces)

Synthetic test surfaces for chemistry/materials:
- **Categorical (5)**: CatAckley, CatCamel, CatDejong, CatMichalewicz, CatSlope
- **Discrete (3)**: DiscreteAckley, DiscreteDoubleWell, DiscreteMichalewicz
- **Mountain Terrains (6)**: Denali, Everest, K2, Kilimanjaro, Matterhorn, MontBlanc
- **Special Functions (5)**: AckleyPath, GaussianMixture, HyperEllipsoid, LinearFunnel, NarrowFunnel
- **Dimensions**: 2-10D
- **Parameter Types**: Continuous, discrete, categorical

### 7. Olympus Datasets (40+ real experiments)

Real-world chemistry and materials science:
- **Chemical Reactions (14)**: Buchwald (5), Suzuki (7), Benzylation, Alkox, SNAr
- **Materials Science (8)**: Perovskites, Fullerenes, Dye Lasers, Redoxmers, Colors, Thin Films
- **Photovoltaics (4)**: Photo_PCE10, Photo_WF3, P3HT, MMLI_OPV
- **Nanoparticles (3)**: AgNP (silver nanoparticles), LNP3, AutoAM
- **Electrochemistry (5)**: OER catalyst plates
- **Liquid/Solvent Systems (7)**: Acetone, DCE, Heptane, THF, Toluene, Water
- **Other (2)**: HPLC, Vapor Diffusion Crystallization
- **Dimensions**: 3-9D
- **Parameter Types**: Mixed (continuous + categorical)

### 8. MuJoCo Control (10 tasks)

Robot locomotion controller optimization:
- **Environments (5)**: HalfCheetah, Hopper, Walker2d, Ant, Humanoid
- **Controller Types (2)**: Linear, MLP (32 hidden units)
- **Dimensions**:
  - Linear: 36D (Hopper) to 6,409D (Humanoid)
  - MLP: 419D (Hopper) to 12,569D (Humanoid)
- **Parameter Type**: Continuous controller weights
- **Objective**: Maximize cumulative reward in simulation

### 9. Robosuite Manipulation (22 tasks)

Robot manipulation controller optimization:
- **Tasks (11)**: Lift, Stack, PickPlace, NutAssembly, Door, Wipe, ToolHang, TwoArmHandover, TwoArmLift, TwoArmPegInHole, TwoArmTransport
- **Controller Types (2)**: Linear, MLP
- **Robot**: Panda (Franka Emika) - also supports Sawyer, IIWA, Jaco, Kinova3, UR5e
- **Dimensions**:
  - Linear: ~300-400D
  - MLP: ~1,500-2,000D
- **Parameter Type**: Continuous
- **Objective**: Task completion reward

### 10. HumanoidBench (320+ tasks)

Full-body humanoid robot control:
- **Robots (5)**: h1, h1hand, h1strong, h1touch, g1
- **Locomotion Tasks (13)**: Walk, Run, Stand, Crawl, Hurdle, Stair, Slide, Pole, Maze, Sit, Balance
- **Manipulation Tasks (19)**: Reach, Push, Door, Cabinet, Truck, Cube, Bookshelf, Basketball, Window, Spoon, Kitchen, Package, Powerlift, Room, Insert, Highbar
- **Controller Types (2)**: Linear, MLP (64 hidden units)
- **Dimensions**:
  - Linear: ~1,444D (Walk, H1Hand)
  - MLP: ~6,099D (Walk, H1Hand)
- **Parameter Type**: Continuous
- **Total Combinations**: 5 robots × 32 tasks × 2 controllers = 320+ tasks

### 11. Design-Bench (25+ tasks)

Offline design optimization:
- **Superconductor (3 oracles)**: 81D continuous material features
- **GFP Protein (6 oracles)**: 237 amino acids → 4,740D one-hot encoded
- **TFBind8 (5 oracles)**: 8 base pairs → 32D one-hot encoded
- **TFBind10 (1 oracle)**: 10 base pairs → 40D one-hot encoded
- **UTR (6 oracles)**: 50 nucleotides → 200D one-hot encoded
- **CIFARNAS (1)**: Neural architecture search on CIFAR-10
- **NASBench (1)**: NAS-Bench-101 search space
- **Oracle Types**: RandomForest, GP, FullyConnected, LSTM, ResNet, Transformer, Exact
- **Parameter Types**: Discrete sequences (one-hot encoded)
- **Domains**: Materials science, protein engineering, DNA/RNA design, neural architecture search

---

For complete details on all benchmarks, see [BENCHMARK_CATALOG.md](BENCHMARK_CATALOG.md).

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

This benchmark library integrates and builds upon the following excellent projects:

- **BBOB/COCO**: Hansen et al. - Black-Box Optimization Benchmarking suite
- **BoTorch**: Facebook AI Research - Additional test functions
- **LassoBench**: Zhao et al. - High-dimensional sparse regression benchmarks
- **HPOBench**: Eggensperger et al. - Hyperparameter optimization benchmarks
- **Bayesmark**: Uber AI Labs - Standardized HPO benchmarking framework
- **Olympus**: Häse et al. - Self-driving laboratories for chemistry/materials
- **MuJoCo**: OpenAI/DeepMind - Multi-Joint dynamics with Contact (robotics simulation)
- **Robosuite**: Stanford PAIR - Robot manipulation benchmarks
- **HumanoidBench**: Robotics community - Humanoid robot control benchmarks
- **Design-Bench**: Trabucco et al. - Offline model-based optimization
- **BenchBase**: CMU Database Group - Multi-DBMS SQL benchmarking framework

We thank all the original authors for making their benchmarks openly available.

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Note**: Some benchmark suites require additional dependencies. Use the dependency management system to check installation status:

```python
from bomegabench.utils.dependencies import print_dependency_status
print_dependency_status()
```
