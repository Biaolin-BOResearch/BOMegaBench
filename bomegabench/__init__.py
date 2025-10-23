"""
BO-MegaBench: A comprehensive Bayesian Optimization benchmark library.

This library provides 72+ synthetic benchmark functions plus real-world optimization benchmarks:
- Consolidated Suite (72 functions): Complete collection including BBOB (24), BoTorch Additional (6), 
  Classical Additional (32), and Classical Core (10) functions
- LassoBench Synthetic Suite (8 functions - high-dimensional sparse regression)  
- LassoBench Real-world Suite (5 functions - real datasets)
- HPO Benchmarks Suite (100+ functions - machine learning hyperparameter optimization via Bayesmark)
- HPOBench ML Suite (30+ functions - machine learning benchmarks from HPOBench)
- HPOBench OD Suite (8+ functions - outlier detection benchmarks from HPOBench)
- HPOBench NAS Suite (8+ functions - neural architecture search benchmarks)
- HPOBench RL Suite (2+ functions - reinforcement learning benchmarks)  
- HPOBench Surrogates Suite (3+ functions - surrogate-based benchmarks)

All functions are wrapped with a unified interface for easy benchmarking.
Optional dependencies: LassoBench, bayesmark, hpobench, ConfigSpace.
"""

from .core import BenchmarkFunction, BenchmarkSuite
from .functions import (
    ConsolidatedSuite,
    get_function,
    list_functions,
    list_suites,
    get_function_summary,
    get_multimodal_functions,
    get_unimodal_functions,
    get_functions_by_property,
)

# Import LassoBench suites if available
try:
    from .functions import (
        LassoBenchSyntheticSuite,
        LassoBenchRealSuite
    )
    LASSO_BENCH_AVAILABLE = True
except ImportError:
    LassoBenchSyntheticSuite = None
    LassoBenchRealSuite = None
    LASSO_BENCH_AVAILABLE = False

# Import HPO benchmarks if available
try:
    from .functions import HPOBenchmarksSuite
    HPO_AVAILABLE = True
except ImportError:
    HPOBenchmarksSuite = None
    HPO_AVAILABLE = False

# Import HPOBench benchmarks if available
try:
    from .functions import (HPOBenchMLSuite, HPOBenchODSuite,
                           HPOBenchNASSuite, HPOBenchRLSuite,
                           HPOBenchSurrogatesSuite)
    HPOBENCH_AVAILABLE = True
except ImportError:
    HPOBenchMLSuite = None
    HPOBenchODSuite = None
    HPOBenchNASSuite = None
    HPOBenchRLSuite = None
    HPOBenchSurrogatesSuite = None
    HPOBENCH_AVAILABLE = False
from .benchmark import BenchmarkRunner, BenchmarkResult
from .visualization import plot_function, plot_convergence, plot_comparison

__version__ = "0.1.0"
__author__ = "BOResearch"

__all__ = [
    "BenchmarkFunction",
    "BenchmarkSuite", 
    "ConsolidatedSuite",
    "get_function",
    "list_functions",
    "list_suites",
    "get_function_summary",
    "get_multimodal_functions", 
    "get_unimodal_functions",
    "get_functions_by_property",
    "BenchmarkRunner",
    "BenchmarkResult",
    "plot_function",
    "plot_convergence", 
    "plot_comparison",
]

# Add LassoBench suites to exports if available
if LASSO_BENCH_AVAILABLE:
    __all__.extend([
        "LassoBenchSyntheticSuite",
        "LassoBenchRealSuite"
    ])

# Add HPO benchmarks to exports if available
if HPO_AVAILABLE:
    __all__.extend([
        "HPOBenchmarksSuite"
    ])

# Add HPOBench benchmarks to exports if available
if HPOBENCH_AVAILABLE:
    __all__.extend([
        "HPOBenchMLSuite",
        "HPOBenchODSuite",
        "HPOBenchNASSuite",
        "HPOBenchRLSuite",
        "HPOBenchSurrogatesSuite"
    ]) 