"""
HPOBench Benchmarks integration.
Integrates various benchmarks from HPOBench into BOMegaBench with continuous hyperparameter encoding.
"""

import sys
import os
import torch
from torch import Tensor
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings

# Import from BOMegaBench core
from ..core import BenchmarkFunction, BenchmarkSuite

# NOTE: Do NOT add local HPOBench repo to path - it has dependency issues
# Use pip-installed hpobench instead

# Fix sklearn compatibility: several internal modules were renamed in sklearn >= 1.0
# These shims allow older code (like HPOBench surrogates) to work
def _fix_sklearn_compatibility():
    """Add compatibility shims for renamed sklearn internal modules."""
    try:
        # sklearn.ensemble.forest -> sklearn.ensemble._forest
        from sklearn.ensemble import _forest
        if 'sklearn.ensemble.forest' not in sys.modules:
            sys.modules['sklearn.ensemble.forest'] = _forest
    except ImportError:
        pass
    
    try:
        # sklearn.tree.tree -> sklearn.tree._tree or sklearn.tree._classes
        from sklearn import tree as sklearn_tree
        if 'sklearn.tree.tree' not in sys.modules:
            # Create a module-like object with the necessary classes
            import types
            tree_module = types.ModuleType('sklearn.tree.tree')
            # Copy common classes
            if hasattr(sklearn_tree, 'DecisionTreeClassifier'):
                tree_module.DecisionTreeClassifier = sklearn_tree.DecisionTreeClassifier
            if hasattr(sklearn_tree, 'DecisionTreeRegressor'):
                tree_module.DecisionTreeRegressor = sklearn_tree.DecisionTreeRegressor
            if hasattr(sklearn_tree, 'ExtraTreeClassifier'):
                tree_module.ExtraTreeClassifier = sklearn_tree.ExtraTreeClassifier
            if hasattr(sklearn_tree, 'ExtraTreeRegressor'):
                tree_module.ExtraTreeRegressor = sklearn_tree.ExtraTreeRegressor
            # Also try to get internal _tree module
            try:
                from sklearn.tree import _tree
                tree_module._tree = _tree
            except ImportError:
                pass
            sys.modules['sklearn.tree.tree'] = tree_module
    except ImportError:
        pass
    
    try:
        # sklearn.neighbors.base -> sklearn.neighbors._base
        from sklearn.neighbors import _base as neighbors_base
        if 'sklearn.neighbors.base' not in sys.modules:
            sys.modules['sklearn.neighbors.base'] = neighbors_base
    except ImportError:
        pass

_fix_sklearn_compatibility()

# Try to import HPOBench
try:
    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH
    CONFIGSPACE_AVAILABLE = True
except ImportError:
    CONFIGSPACE_AVAILABLE = False
    print("ConfigSpace not available: pip install ConfigSpace")

# Patch HPOBench data_manager for sklearn >= 1.2 compatibility
# sklearn renamed 'sparse' to 'sparse_output' in OneHotEncoder
def _patch_hpobench_for_sklearn():
    """Patch HPOBench data_manager.py to fix sklearn compatibility."""
    try:
        import hpobench.dependencies.ml.data_manager as dm
        import inspect
        source = inspect.getsourcefile(dm)
        if source:
            # Read the file
            with open(source, 'r') as f:
                content = f.read()
            # Check if patching is needed
            if 'sparse=False' in content and 'sparse_output' not in content:
                content_new = content.replace('sparse=False', 'sparse_output=False')
                with open(source, 'w') as f:
                    f.write(content_new)
                # Reload the module
                import importlib
                importlib.reload(dm)
    except Exception:
        pass  # If patching fails, continue anyway

_patch_hpobench_for_sklearn()

# Try to import HPOBench ML benchmarks
try:
    from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmarkBB, XGBoostBenchmarkMF
    from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmarkBB, SVMBenchmarkMF
    from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmarkBB  # Fixed: was RFBenchmark
    from hpobench.benchmarks.ml.lr_benchmark import LRBenchmark
    from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark
    from hpobench.benchmarks.ml.histgb_benchmark import HistGBBenchmark
    # Alias for backwards compatibility
    RFBenchmark = RandomForestBenchmarkBB
    HPOBENCH_ML_AVAILABLE = True
except ImportError as e:
    HPOBENCH_ML_AVAILABLE = False
    print(f"HPOBench ML benchmarks not available: {e}")

# Try to import HPOBench OD benchmarks
try:
    from hpobench.benchmarks.od.od_kde import ODKernelDensityEstimation
    from hpobench.benchmarks.od.od_ocsvm import ODOneClassSVM
    # Note: od_ae is excluded as requested
    HPOBENCH_OD_AVAILABLE = True
except ImportError:
    HPOBENCH_OD_AVAILABLE = False
    # Silently skip - OD benchmarks are optional

# Try to import HPOBench NAS benchmarks
try:
    # NASBench-101 requires tabular_benchmarks package
    from hpobench.benchmarks.nas.nasbench_101 import NASCifar10ABenchmark, NASCifar10BBenchmark, NASCifar10CBenchmark
    NASBENCH101_AVAILABLE = True
except ImportError as e:
    NASBENCH101_AVAILABLE = False
    NASCifar10ABenchmark = NASCifar10BBenchmark = NASCifar10CBenchmark = None

try:
    from hpobench.benchmarks.nas.nasbench_201 import NASBench201Benchmark
    NASBENCH201_AVAILABLE = True
except ImportError as e:
    NASBENCH201_AVAILABLE = False
    NASBench201Benchmark = None

try:
    from hpobench.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark, ProteinStructureBenchmark, NavalPropulsionBenchmark, ParkinsonsTelemonitoringBenchmark
    # Test if tabular_benchmarks (from nas_benchmarks) is actually available
    # The classes import but require nasbench package for instantiation
    try:
        import tabular_benchmarks
        TABULAR_BENCHMARKS_AVAILABLE = True
    except ImportError:
        TABULAR_BENCHMARKS_AVAILABLE = False
        SliceLocalizationBenchmark = ProteinStructureBenchmark = NavalPropulsionBenchmark = ParkinsonsTelemonitoringBenchmark = None
except ImportError as e:
    TABULAR_BENCHMARKS_AVAILABLE = False
    SliceLocalizationBenchmark = ProteinStructureBenchmark = NavalPropulsionBenchmark = ParkinsonsTelemonitoringBenchmark = None

HPOBENCH_NAS_AVAILABLE = NASBENCH101_AVAILABLE or NASBENCH201_AVAILABLE or TABULAR_BENCHMARKS_AVAILABLE
# Silently skip - NAS benchmarks are optional and require complex setup

# Try to import HPOBench RL benchmarks
# Note: cartpole requires tensorflow
try:
    from hpobench.benchmarks.rl.cartpole import CartpoleReduced, CartpoleFull
    CARTPOLE_AVAILABLE = True
except ImportError as e:
    CARTPOLE_AVAILABLE = False
    CartpoleReduced = CartpoleFull = None

try:
    from hpobench.benchmarks.rl.learna_benchmark import Learna, MetaLearna
    LEARNA_AVAILABLE = True
except ImportError as e:
    LEARNA_AVAILABLE = False
    Learna = MetaLearna = None

HPOBENCH_RL_AVAILABLE = CARTPOLE_AVAILABLE or LEARNA_AVAILABLE
# Silently skip - RL benchmarks are optional

# Try to import HPOBench Surrogate benchmarks
# Note: These benchmarks use pickled sklearn models from version 0.18.1
# which are incompatible with sklearn >= 1.0 due to internal dtype changes
try:
    from hpobench.benchmarks.surrogates.svm_benchmark import SurrogateSVMBenchmark
    from hpobench.benchmarks.surrogates.paramnet_benchmark import ParamNetAdultOnTimeBenchmark, ParamNetHiggsOnTimeBenchmark
    # Test if we can actually load the surrogate (sklearn pickle compatibility)
    import sklearn
    sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
    if sklearn_version >= (1, 0):
        # Surrogate models were pickled with sklearn 0.18.1, incompatible with sklearn >= 1.0
        HPOBENCH_SURROGATES_AVAILABLE = False
        SurrogateSVMBenchmark = ParamNetAdultOnTimeBenchmark = ParamNetHiggsOnTimeBenchmark = None
    else:
        HPOBENCH_SURROGATES_AVAILABLE = True
except ImportError:
    HPOBENCH_SURROGATES_AVAILABLE = False
    SurrogateSVMBenchmark = ParamNetAdultOnTimeBenchmark = ParamNetHiggsOnTimeBenchmark = None

# Check overall availability
HPOBENCH_AVAILABLE = CONFIGSPACE_AVAILABLE and (HPOBENCH_ML_AVAILABLE or HPOBENCH_OD_AVAILABLE or 
                                                HPOBENCH_NAS_AVAILABLE or HPOBENCH_RL_AVAILABLE or 
                                                HPOBENCH_SURROGATES_AVAILABLE)


class HPOBenchFunction(BenchmarkFunction):
    """Wrapper for HPOBench benchmark functions with continuous hyperparameter encoding."""
    
    def __init__(self, 
                 benchmark_class: type,
                 benchmark_name: str,
                 benchmark_kwargs: Dict[str, Any] = None,
                 **kwargs):
        """
        Initialize HPOBench benchmark function.
        
        Args:
            benchmark_class: HPOBench benchmark class (e.g., XGBoostBenchmarkBB)
            benchmark_name: Name for this benchmark instance
            benchmark_kwargs: Arguments to pass to benchmark constructor
            **kwargs: Additional arguments passed to parent class
        """
        if not HPOBENCH_AVAILABLE:
            raise ImportError("HPOBench is required but not installed. Install with: pip install hpobench ConfigSpace")
        
        self.benchmark_class = benchmark_class
        self.benchmark_name = benchmark_name
        self.benchmark_kwargs = benchmark_kwargs or {}
        
        # Create benchmark instance
        self.benchmark = benchmark_class(**self.benchmark_kwargs)
        
        # Get configuration space
        self.config_space = self.benchmark.get_configuration_space()
        
        # Get fidelity space (use max fidelity for single-objective optimization)
        self.fidelity_space = self.benchmark.get_fidelity_space()
        self.max_fidelity = self._get_max_fidelity()
        
        # Convert spaces to continuous [0,1] representation
        self.continuous_space = self._create_continuous_space()
        
        # Get dimension and bounds
        dim = len(self.continuous_space)
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim])
        
        super().__init__(dim=dim, bounds=bounds, **kwargs)
        
    def _get_max_fidelity(self) -> Dict[str, Any]:
        """Get maximum fidelity configuration for single-objective optimization."""
        max_fidelity = {}
        for hp_name in self.fidelity_space:
            hp = self.fidelity_space[hp_name]
            if isinstance(hp, CSH.UniformFloatHyperparameter):
                max_fidelity[hp_name] = hp.upper
            elif isinstance(hp, CSH.UniformIntegerHyperparameter):
                max_fidelity[hp_name] = hp.upper
            elif isinstance(hp, CSH.CategoricalHyperparameter):
                # Use default value for categorical fidelities
                max_fidelity[hp_name] = hp.default_value
            elif isinstance(hp, CS.Constant):
                max_fidelity[hp_name] = hp.value
            else:
                # Fallback to default value
                max_fidelity[hp_name] = hp.default_value
        return max_fidelity
        
    def _create_continuous_space(self) -> List[Dict]:
        """Convert ConfigSpace to continuous [0,1] representation."""
        continuous_dims = []
        
        for hp_name in self.config_space:
            hp = self.config_space[hp_name]
            
            if isinstance(hp, CSH.UniformFloatHyperparameter):
                # Real parameters: normalize to [0,1]
                continuous_dims.append({
                    'name': hp_name,
                    'type': 'real',
                    'original_bounds': (hp.lower, hp.upper),
                    'log_scale': hp.log
                })
            elif isinstance(hp, CSH.UniformIntegerHyperparameter):
                # Integer parameters: treat as categorical (each value gets one dimension)
                int_values = list(range(hp.lower, hp.upper + 1))
                for i, value in enumerate(int_values):
                    continuous_dims.append({
                        'name': f"{hp_name}_{value}",
                        'type': 'int_as_cat',
                        'original_param': hp_name,
                        'choice': value,
                        'choice_index': i,
                        'total_choices': len(int_values)
                    })
            elif isinstance(hp, CSH.CategoricalHyperparameter):
                # Categorical parameters: each choice gets one dimension
                choices = hp.choices
                for i, choice in enumerate(choices):
                    continuous_dims.append({
                        'name': f"{hp_name}_{choice}",
                        'type': 'cat',
                        'original_param': hp_name,
                        'choice': choice,
                        'choice_index': i,
                        'total_choices': len(choices)
                    })
            elif isinstance(hp, CS.Constant):
                # Constants are not optimized, skip
                continue
            else:
                warnings.warn(f"Unknown hyperparameter type: {type(hp)} for {hp_name}")
        
        return continuous_dims
        
    def _decode_continuous_params(self, X: np.ndarray) -> Dict[str, Any]:
        """Convert continuous [0,1] parameters back to original ConfigSpace format."""
        params = {}
        cat_params = {}  # Track categorical parameters
        int_params = {}  # Track integer parameters treated as categorical
        
        for i, dim_config in enumerate(self.continuous_space):
            value = X[i]
            
            if dim_config['type'] == 'real':
                # Scale from [0,1] to original bounds
                low, high = dim_config['original_bounds']
                if dim_config.get('log_scale', False):
                    # Log scale transformation
                    log_low, log_high = np.log(low), np.log(high)
                    log_value = log_low + value * (log_high - log_low)
                    params[dim_config['name']] = np.exp(log_value)
                else:
                    # Linear scale
                    params[dim_config['name']] = low + value * (high - low)
                    
            elif dim_config['type'] == 'int_as_cat':
                # For integer as categorical: collect all choice dimensions
                param_name = dim_config['original_param']
                if param_name not in int_params:
                    int_params[param_name] = []
                int_params[param_name].append((value, dim_config['choice']))
                
            elif dim_config['type'] == 'cat':
                # For categorical: collect all choice dimensions
                param_name = dim_config['original_param']
                if param_name not in cat_params:
                    cat_params[param_name] = []
                cat_params[param_name].append((value, dim_config['choice']))
        
        # For integer parameters treated as categorical, choose the one with highest value
        for param_name, choices in int_params.items():
            best_choice = max(choices, key=lambda x: x[0])[1]
            params[param_name] = best_choice
            
        # For categorical parameters, choose the one with highest value
        for param_name, choices in cat_params.items():
            best_choice = max(choices, key=lambda x: x[0])[1]
            params[param_name] = best_choice
            
        return params
        
    def _get_metadata(self) -> Dict[str, Any]:
        """Get function metadata."""
        properties = ["hpo", "hpobench", self.benchmark_class.__name__.lower()]
        
        # Add specific properties based on benchmark type
        if hasattr(self.benchmark, 'task_id'):
            properties.append("openml")
            properties.append(f"task_{self.benchmark.task_id}")
        
        if 'OD' in self.benchmark_class.__name__:
            properties.append("anomaly_detection")
        elif any(x in self.benchmark_class.__name__.lower() for x in ['xgb', 'svm', 'rf', 'nn', 'lr']):
            properties.append("machine_learning")
            
        return {
            "name": f"HPOBench-{self.benchmark_name}",
            "suite": "HPOBench Benchmarks",
            "properties": properties,
            "domain": "[0,1]^" + str(self.dim),
            "global_min": "Variable (depends on benchmark)",
            "description": f"HPOBench benchmark: {self.benchmark_class.__name__}",
            "benchmark_class": self.benchmark_class.__name__,
            "benchmark_kwargs": self.benchmark_kwargs,
            "original_config_space": dict(self.config_space),
            "fidelity_space": dict(self.fidelity_space)
        }
        
    def _evaluate_true(self, X: Tensor) -> Tensor:
        """Evaluate the HPOBench objective function."""
        # Convert torch tensor to numpy
        X_np = X.detach().cpu().numpy()
        
        # Handle batch evaluation
        if X_np.ndim == 1:
            # Single point
            params = self._decode_continuous_params(X_np)
            result = self._evaluate_single(params)
            return torch.tensor(result, dtype=X.dtype, device=X.device)
        else:
            # Batch evaluation
            results = []
            for i in range(X_np.shape[0]):
                params = self._decode_continuous_params(X_np[i])
                result = self._evaluate_single(params)
                results.append(result)
            return torch.tensor(results, dtype=X.dtype, device=X.device)
    
    def _evaluate_single(self, params: Dict[str, Any]) -> float:
        """Evaluate single hyperparameter configuration using HPOBench interface."""
        try:
            # Create ConfigSpace Configuration
            config = CS.Configuration(self.config_space, params)
            
            # Create fidelity configuration (use max fidelity)
            fidelity = CS.Configuration(self.fidelity_space, self.max_fidelity) if self.max_fidelity else None
            
            # Evaluate using HPOBench benchmark
            result = self.benchmark.objective_function(config, fidelity=fidelity)
            
            # Return function value (should be minimized)
            return float(result['function_value'])
                
        except Exception as e:
            # Return high penalty for invalid configurations
            warnings.warn(f"Evaluation failed with params {params}: {e}")
            return 1e6


def create_hpobench_ml_suite() -> BenchmarkSuite:
    """Create HPOBench ML benchmarks suite."""
    if not HPOBENCH_ML_AVAILABLE:
        raise ImportError("HPOBench ML benchmarks not available. Install with: pip install hpobench")
    
    functions = {}
    
    # Common OpenML task IDs for testing (small datasets)
    task_ids = [
        31,    # credit-g (1000 instances, 20 features)
        3917,  # kc1 (2109 instances, 21 features)  
        9952,  # phoneme (5404 instances, 5 features)
        146818, # Australian (690 instances, 14 features)
    ]
    
    # ML benchmark configurations
    ml_benchmarks = [
        (XGBoostBenchmarkBB, "XGBoost-BB"),
        (XGBoostBenchmarkMF, "XGBoost-MF"), 
        (SVMBenchmarkBB, "SVM-BB"),
        (SVMBenchmarkMF, "SVM-MF"),
        (RFBenchmark, "RandomForest"),
        (LRBenchmark, "LogisticRegression"),
        (NNBenchmark, "NeuralNetwork"),
        (HistGBBenchmark, "HistGradientBoosting"),
    ]
    
    # Create benchmark functions
    for benchmark_class, benchmark_name in ml_benchmarks:
        for task_id in task_ids:
            try:
                func_name = f"{benchmark_name}_task_{task_id}"
                functions[func_name] = HPOBenchFunction(
                    benchmark_class=benchmark_class,
                    benchmark_name=f"{benchmark_name}-Task{task_id}",
                    benchmark_kwargs={'task_id': task_id}
                )
            except Exception as e:
                warnings.warn(f"Failed to create {func_name}: {e}")
    
    return BenchmarkSuite("HPOBench ML Benchmarks", functions)


def create_hpobench_od_suite() -> BenchmarkSuite:
    """Create HPOBench Outlier Detection benchmarks suite."""
    if not HPOBENCH_OD_AVAILABLE:
        raise ImportError("HPOBench OD benchmarks not available. Install with: pip install hpobench")
    
    functions = {}
    
    # Available OD datasets (smaller ones for testing)
    od_datasets = [
        'breastw',     # 683 instances
        'ionosphere',  # 351 instances  
        'pima',        # 768 instances
        'wbc',         # 378 instances
    ]
    
    # OD benchmark configurations (excluding od_ae as requested)
    od_benchmarks = [
        (ODKernelDensityEstimation, "KDE"),
        (ODOneClassSVM, "OneClassSVM"),
    ]
    
    # Create benchmark functions
    for benchmark_class, benchmark_name in od_benchmarks:
        for dataset in od_datasets:
            try:
                func_name = f"{benchmark_name}_{dataset}"
                functions[func_name] = HPOBenchFunction(
                    benchmark_class=benchmark_class,
                    benchmark_name=f"{benchmark_name}-{dataset}",
                    benchmark_kwargs={'dataset_name': dataset}
                )
            except Exception as e:
                warnings.warn(f"Failed to create {func_name}: {e}")
    
    return BenchmarkSuite("HPOBench OD Benchmarks", functions)


def create_hpobench_nas_suite() -> BenchmarkSuite:
    """Create HPOBench Neural Architecture Search benchmarks suite."""
    if not HPOBENCH_NAS_AVAILABLE:
        raise ImportError("HPOBench NAS benchmarks not available. Install with: pip install hpobench[nas]")
    
    functions = {}
    
    # NAS benchmark configurations
    nas_benchmarks = [
        # NASBench-101 variants
        (NASCifar10ABenchmark, "NASBench101-A"),
        (NASCifar10BBenchmark, "NASBench101-B"), 
        (NASCifar10CBenchmark, "NASBench101-C"),
        
        # NASBench-201
        (NASBench201Benchmark, "NASBench201"),
        
        # Tabular benchmarks
        (SliceLocalizationBenchmark, "SliceLocalization"),
        (ProteinStructureBenchmark, "ProteinStructure"),
        (NavalPropulsionBenchmark, "NavalPropulsion"),
        (ParkinsonsTelemonitoringBenchmark, "ParkinsonsTelemonitoring"),
    ]
    
    # Create benchmark functions
    for benchmark_class, benchmark_name in nas_benchmarks:
        # Skip if benchmark class is None (import failed)
        if benchmark_class is None:
            continue
        try:
            func_name = benchmark_name.lower().replace("-", "_")
            functions[func_name] = HPOBenchFunction(
                benchmark_class=benchmark_class,
                benchmark_name=benchmark_name,
                benchmark_kwargs={}
            )
        except Exception as e:
            warnings.warn(f"Failed to create {benchmark_name}: {e}")
    
    return BenchmarkSuite("HPOBench NAS Benchmarks", functions)


def create_hpobench_rl_suite() -> BenchmarkSuite:
    """Create HPOBench Reinforcement Learning benchmarks suite."""
    if not HPOBENCH_RL_AVAILABLE:
        raise ImportError("HPOBench RL benchmarks not available. Install with: pip install hpobench[rl]")
    
    functions = {}
    
    # RL benchmark configurations (Cartpole - no data path needed)
    cartpole_benchmarks = [
        (CartpoleReduced, "Cartpole-Reduced"),
        (CartpoleFull, "Cartpole-Full"),
    ]
    
    # Create Cartpole benchmark functions
    for benchmark_class, benchmark_name in cartpole_benchmarks:
        # Skip if benchmark class is None (import failed)
        if benchmark_class is None:
            continue
        try:
            func_name = benchmark_name.lower().replace("-", "_")
            functions[func_name] = HPOBenchFunction(
                benchmark_class=benchmark_class,
                benchmark_name=benchmark_name,
                benchmark_kwargs={}
            )
        except Exception as e:
            warnings.warn(f"Failed to create {benchmark_name}: {e}")
    
    # Learna benchmarks (require data path - use placeholder path or skip if not available)
    learna_benchmarks = [
        (Learna, "Learna"),
        (MetaLearna, "MetaLearna"),
    ]
    
    # Try to create Learna benchmarks with a default data path
    # Note: These require specific RNA data to be downloaded
    try:
        # Try common data paths or use a placeholder
        import hpobench.config
        default_data_path = hpobench.config.config_file.data_dir / "learna_data"
        
        for benchmark_class, benchmark_name in learna_benchmarks:
            # Skip if benchmark class is None (import failed)
            if benchmark_class is None:
                continue
            try:
                func_name = benchmark_name.lower()
                functions[func_name] = HPOBenchFunction(
                    benchmark_class=benchmark_class,
                    benchmark_name=benchmark_name,
                    benchmark_kwargs={'data_path': default_data_path}
                )
            except Exception as e:
                warnings.warn(f"Failed to create {benchmark_name}: {e}. "
                             f"Learna benchmarks require RNA data to be downloaded. "
                             f"See HPOBench documentation for setup instructions.")
    except Exception as e:
        warnings.warn(f"Learna benchmarks not available: {e}. "
                     f"These require special data setup.")
    
    return BenchmarkSuite("HPOBench RL Benchmarks", functions)


def create_hpobench_surrogates_suite() -> BenchmarkSuite:
    """Create HPOBench Surrogate benchmarks suite."""
    if not HPOBENCH_SURROGATES_AVAILABLE:
        raise ImportError("HPOBench Surrogate benchmarks not available. Install with: pip install hpobench[surrogates]")
    
    functions = {}
    
    # Surrogate benchmark configurations
    surrogate_benchmarks = [
        (SurrogateSVMBenchmark, "SurrogateSVM"),
        (ParamNetAdultOnTimeBenchmark, "ParamNet-Adult"),
        (ParamNetHiggsOnTimeBenchmark, "ParamNet-Higgs"),
    ]
    
    # Create benchmark functions
    for benchmark_class, benchmark_name in surrogate_benchmarks:
        # Skip if benchmark class is None (import failed)
        if benchmark_class is None:
            continue
        try:
            func_name = benchmark_name.lower().replace("-", "_")
            functions[func_name] = HPOBenchFunction(
                benchmark_class=benchmark_class,
                benchmark_name=benchmark_name,
                benchmark_kwargs={}
            )
        except Exception as e:
            warnings.warn(f"Failed to create {benchmark_name}: {e}")
    
    return BenchmarkSuite("HPOBench Surrogate Benchmarks", functions)


def create_hpobench_suites() -> Dict[str, BenchmarkSuite]:
    """Create and return all HPOBench suites."""
    if not HPOBENCH_AVAILABLE:
        raise ImportError("HPOBench is required but not installed. Install with: pip install hpobench ConfigSpace")
    
    suites = {}
    
    if HPOBENCH_ML_AVAILABLE:
        try:
            suites["hpobench_ml"] = create_hpobench_ml_suite()
        except Exception as e:
            warnings.warn(f"Failed to create HPOBench ML suite: {e}")
    
    if HPOBENCH_OD_AVAILABLE:
        try:
            suites["hpobench_od"] = create_hpobench_od_suite()
        except Exception as e:
            warnings.warn(f"Failed to create HPOBench OD suite: {e}")
    
    if HPOBENCH_NAS_AVAILABLE:
        try:
            suites["hpobench_nas"] = create_hpobench_nas_suite()
        except Exception as e:
            warnings.warn(f"Failed to create HPOBench NAS suite: {e}")
    
    if HPOBENCH_RL_AVAILABLE:
        try:
            suites["hpobench_rl"] = create_hpobench_rl_suite()
        except Exception as e:
            warnings.warn(f"Failed to create HPOBench RL suite: {e}")
    
    if HPOBENCH_SURROGATES_AVAILABLE:
        try:
            suites["hpobench_surrogates"] = create_hpobench_surrogates_suite()
        except Exception as e:
            warnings.warn(f"Failed to create HPOBench Surrogates suite: {e}")
    
    return suites


# Create suite instances if dependencies are available
if HPOBENCH_AVAILABLE:
    try:
        _hpobench_suites = create_hpobench_suites()
        HPOBenchMLSuite = _hpobench_suites.get("hpobench_ml")
        HPOBenchODSuite = _hpobench_suites.get("hpobench_od")
        HPOBenchNASSuite = _hpobench_suites.get("hpobench_nas")
        HPOBenchRLSuite = _hpobench_suites.get("hpobench_rl")
        HPOBenchSurrogatesSuite = _hpobench_suites.get("hpobench_surrogates")
    except Exception as e:
        print(f"Warning: HPOBench suites creation failed: {e}")
        HPOBenchMLSuite = None
        HPOBenchODSuite = None
        HPOBenchNASSuite = None
        HPOBenchRLSuite = None
        HPOBenchSurrogatesSuite = None
else:
    HPOBenchMLSuite = None
    HPOBenchODSuite = None
    HPOBenchNASSuite = None
    HPOBenchRLSuite = None
    HPOBenchSurrogatesSuite = None

__all__ = [
    "HPOBenchFunction",
    "create_hpobench_ml_suite",
    "create_hpobench_od_suite",
    "create_hpobench_nas_suite",
    "create_hpobench_rl_suite",
    "create_hpobench_surrogates_suite",
    "create_hpobench_suites",
    "HPOBenchMLSuite",
    "HPOBenchODSuite",
    "HPOBenchNASSuite",
    "HPOBenchRLSuite",
    "HPOBenchSurrogatesSuite"
] 