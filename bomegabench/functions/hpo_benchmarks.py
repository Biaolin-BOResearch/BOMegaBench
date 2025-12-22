"""
HPO Benchmarks using Bayesmark interface.
Integrates machine learning hyperparameter optimization benchmarks into BOMegaBench.
"""

import torch
from torch import Tensor
import numpy as np
from typing import Dict, Any, List, Optional, Union
import warnings

# Import from BOMegaBench core
from ..core import BenchmarkFunction, BenchmarkSuite

# Try to import bayesmark
# Note: bayesmark uses load_boston which is removed in sklearn >= 1.2
# We need to patch this before importing bayesmark
try:
    # Patch sklearn.datasets to provide a mock load_boston BEFORE importing bayesmark
    import sklearn.datasets
    
    def _fetch_california_housing_with_fallback():
        """
        Fetch California Housing dataset with multiple fallback methods.
        1. Try fetch_california_housing() first
        2. If it fails (e.g., HTTP 403), try fetch_openml as fallback
        """
        from sklearn.utils import Bunch
        
        # Method 1: Try default fetch_california_housing
        try:
            from sklearn.datasets import fetch_california_housing
            cal = fetch_california_housing()
            return cal
        except Exception as e1:
            warnings.warn(f"fetch_california_housing failed: {e1}, trying OpenML fallback...")
        
        # Method 2: Try OpenML (California housing data_id=42165) with as_frame=True
        try:
            from sklearn.datasets import fetch_openml
            import numpy as np
            openml_data = fetch_openml(data_id=42165, as_frame=True, parser='auto')
            # Convert DataFrame to numpy array, handling string columns
            data_df = openml_data.data
            # Select only numeric columns
            numeric_cols = data_df.select_dtypes(include=[np.number]).columns
            cal = Bunch(
                data=data_df[numeric_cols].values.astype(np.float64),
                target=openml_data.target.values.astype(np.float64) if hasattr(openml_data.target, 'values') else np.array(openml_data.target, dtype=np.float64),
                feature_names=list(numeric_cols),
                DESCR="California Housing (from OpenML)"
            )
            return cal
        except Exception as e2:
            warnings.warn(f"OpenML fallback also failed: {e2}")
        
        # Method 3: Try fetch_openml with name instead of data_id, using as_frame=True
        try:
            from sklearn.datasets import fetch_openml
            import numpy as np
            openml_data = fetch_openml(name='california_housing', as_frame=True, parser='auto')
            data_df = openml_data.data
            # Select only numeric columns
            numeric_cols = data_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                # If no numeric columns, try converting all columns
                numeric_cols = data_df.columns
            cal = Bunch(
                data=data_df[numeric_cols].values.astype(np.float64),
                target=openml_data.target.values.astype(np.float64) if hasattr(openml_data.target, 'values') else np.array(openml_data.target, dtype=np.float64),
                feature_names=list(numeric_cols) if len(numeric_cols) > 0 else 
                    ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"],
                DESCR="California Housing (from OpenML by name)"
            )
            return cal
        except Exception as e3:
            warnings.warn(f"OpenML by name also failed: {e3}")
            
        # Method 4: Use diabetes dataset as last resort (it's always available locally)
        try:
            from sklearn.datasets import load_diabetes
            diabetes = load_diabetes()
            cal = Bunch(
                data=diabetes.data,
                target=diabetes.target,
                feature_names=list(diabetes.feature_names),
                DESCR="Fallback dataset (diabetes) - California Housing unavailable"
            )
            warnings.warn("Using diabetes dataset as fallback for Boston/California housing")
            return cal
        except Exception as e4:
            raise RuntimeError(f"All fallback methods failed. Last error: {e4}")
    
    def _mock_load_boston(return_X_y=False):
        """Mock boston dataset using California housing data with fallbacks."""
        cal = _fetch_california_housing_with_fallback()
        # Use min of available features and 8 (boston had 13 features, but we use 8)
        n_features = min(8, cal.data.shape[1])
        data = cal.data[:, :n_features]
        target = cal.target
        
        if return_X_y:
            return data, target
        
        from sklearn.utils import Bunch
        feature_names = cal.feature_names[:n_features] if hasattr(cal, 'feature_names') and cal.feature_names else \
            [f"feature_{i}" for i in range(n_features)]
        return Bunch(
            data=data,
            target=target,
            feature_names=list(feature_names),
            DESCR="Mock Boston dataset (using California Housing or fallback)"
        )
    
    # Force set the attribute (bypass __getattr__ check)
    sklearn.datasets.load_boston = _mock_load_boston
    
    # Also patch bayesmark's internal reference if it caches the function
    import bayesmark.sklearn_funcs as _bm_sklearn
    if hasattr(_bm_sklearn, 'load_boston'):
        _bm_sklearn.load_boston = _mock_load_boston
    
    from bayesmark.sklearn_funcs import SklearnModel
    from bayesmark.space import JointSpace
    BAYESMARK_AVAILABLE = True
except ImportError as e:
    BAYESMARK_AVAILABLE = False
    # Silently skip - bayesmark is optional
except Exception as e:
    BAYESMARK_AVAILABLE = False
    # Silently skip - bayesmark may have issues

# Sklearn is required by bayesmark, so we don't need separate check
SKLEARN_AVAILABLE = BAYESMARK_AVAILABLE


class HPOBenchmarkFunction(BenchmarkFunction):
    """Wrapper for HPO benchmark functions using Bayesmark SklearnModel interface."""
    
    def __init__(self, model_name: str, dataset_name: str, metric: str, shuffle_seed: int = 0, **kwargs):
        """
        Initialize HPO benchmark function.
        
        Args:
            model_name: Name of the ML model (e.g., 'DecisionTreeClassifier')
            dataset_name: Name of dataset (e.g., 'iris', 'wine', 'digits', 'breast')
            metric: Optimization metric ('nll', 'acc', 'mse', 'mae')
            shuffle_seed: Random seed for data splitting
            **kwargs: Additional arguments passed to parent class
        """
        if not BAYESMARK_AVAILABLE:
            raise ImportError("Bayesmark is required but not installed. Install with: pip install bayesmark")
        
        self.model_name = model_name
        self.dataset_name = dataset_name  
        self.metric = metric
        self.shuffle_seed = shuffle_seed
        
        # Initialize SklearnModel directly with bayesmark interface
        self.sklearn_model = SklearnModel(model_name, dataset_name, metric, shuffle_seed=shuffle_seed)
        
        # Get model's hyperparameter space from bayesmark
        self.space = self.sklearn_model.api_config
        
        # Convert space to continuous [0,1] representation
        self.continuous_space = self._create_continuous_space()
        
        # Get dimension and bounds
        dim = len(self.continuous_space)
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim])
        
        super().__init__(dim=dim, bounds=bounds, **kwargs)
        
    def _create_continuous_space(self) -> List[Dict]:
        """Convert bayesmark space to continuous [0,1] representation using one-hot encoding.
        
        Bayesmark api_config format:
        - 'type': 'real', 'int', 'bool', or 'cat'
        - 'space': 'linear', 'log', 'logit', or 'bilog' (scale type for real/int)
        - 'range': (low, high) for real/int
        - For 'cat': 'space' contains list of choices
        """
        continuous_dims = []
        
        for param_name, param_config in self.space.items():
            param_type = param_config['type']
            
            if param_type == 'real':
                # Already continuous, just normalize to [0,1]
                # Range is in 'range' key, not 'space'
                param_range = param_config.get('range', (0, 1))
                continuous_dims.append({
                    'name': param_name,
                    'type': 'real',
                    'scale': param_config.get('space', 'linear'),
                    'original_bounds': (param_range[0], param_range[1])
                })
            elif param_type == 'int':
                # Integer parameters: one-hot encoding for each value
                param_range = param_config.get('range', (0, 10))
                low, high = int(param_range[0]), int(param_range[1])
                int_values = list(range(low, high + 1))
                for i, value in enumerate(int_values):
                    continuous_dims.append({
                        'name': f"{param_name}_{value}",
                        'type': 'int_as_cat',
                        'original_param': param_name,
                        'choice': value,
                        'choice_index': i,
                        'total_choices': len(int_values)
                    })
            elif param_type == 'bool':
                # Boolean: one-hot with two dimensions (True/False)
                for i, value in enumerate([False, True]):
                    continuous_dims.append({
                        'name': f"{param_name}_{value}",
                        'type': 'bool_as_cat',
                        'original_param': param_name,
                        'choice': value,
                        'choice_index': i,
                        'total_choices': 2
                    })
            elif param_type == 'cat':
                # Categorical parameters: one-hot encoding
                choices = param_config.get('space', [])
                if not choices:
                    continue
                for i, choice in enumerate(choices):
                    continuous_dims.append({
                        'name': f"{param_name}_{choice}",
                        'type': 'cat',
                        'original_param': param_name,
                        'choice': choice,
                        'choice_index': i,
                        'total_choices': len(choices)
                    })
        
        return continuous_dims
        
    def _decode_continuous_params(self, X: np.ndarray) -> Dict[str, Any]:
        """Convert continuous [0,1] parameters back to original space."""
        params = {}
        cat_params = {}  # Track categorical parameters
        int_params = {}  # Track integer parameters treated as categorical
        bool_params = {}  # Track boolean parameters treated as categorical
        
        for i, dim_config in enumerate(self.continuous_space):
            value = float(X[i])  # Convert numpy scalar to Python float
            
            if dim_config['type'] == 'real':
                # Scale from [0,1] to original bounds
                low, high = dim_config['original_bounds']
                params[dim_config['name']] = float(low + value * (high - low))
                
            elif dim_config['type'] == 'int_as_cat':
                # For integer as categorical: collect all choice dimensions
                param_name = dim_config['original_param']
                if param_name not in int_params:
                    int_params[param_name] = []
                int_params[param_name].append((value, dim_config['choice']))
            
            elif dim_config['type'] == 'bool_as_cat':
                # For boolean as categorical: collect both dimensions
                param_name = dim_config['original_param']
                if param_name not in bool_params:
                    bool_params[param_name] = []
                bool_params[param_name].append((value, dim_config['choice']))
                
            elif dim_config['type'] == 'cat':
                # For categorical: collect all choice dimensions
                param_name = dim_config['original_param']
                if param_name not in cat_params:
                    cat_params[param_name] = []
                cat_params[param_name].append((value, dim_config['choice']))
        
        # For integer parameters treated as categorical, choose the one with highest value
        for param_name, choices in int_params.items():
            best_choice = max(choices, key=lambda x: x[0])[1]
            # Ensure integer type
            params[param_name] = int(best_choice) if isinstance(best_choice, (int, float, np.integer, np.floating)) else best_choice
        
        # For boolean parameters, choose the one with highest value
        for param_name, choices in bool_params.items():
            best_choice = max(choices, key=lambda x: x[0])[1]
            params[param_name] = bool(best_choice) if isinstance(best_choice, (bool, np.bool_)) else best_choice
            
        # For categorical parameters, choose the one with highest value
        for param_name, choices in cat_params.items():
            best_choice = max(choices, key=lambda x: x[0])[1]
            params[param_name] = best_choice
            
        return params
        
    def _get_metadata(self) -> Dict[str, Any]:
        """Get function metadata."""
        properties = ["hpo", "machine_learning", self.model_name.lower()]
        
        if 'Classifier' in self.model_name:
            properties.append("classification")
        elif 'Regressor' in self.model_name:
            properties.append("regression")
            
        return {
            "name": f"HPO-{self.model_name}-{self.dataset_name}-{self.metric}",
            "suite": "HPO Benchmarks",
            "properties": properties,
            "domain": "[0,1]^" + str(self.dim),
            "global_min": "Variable (depends on model and dataset)",
            "description": f"Hyperparameter optimization for {self.model_name} on {self.dataset_name} dataset",
            "model": self.model_name,
            "dataset": self.dataset_name,
            "metric": self.metric,
            "original_space": dict(self.space)
        }
        
    def _evaluate_true(self, X: Tensor) -> Tensor:
        """Evaluate the HPO objective function."""
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
        """Evaluate single hyperparameter configuration using SklearnModel.evaluate."""
        try:
            # Use SklearnModel's evaluate method directly
            # This returns (cv_loss, generalization_loss)
            cv_loss, generalization_loss = self.sklearn_model.evaluate(params)
            
            # Convert to Python float if numpy scalar
            if hasattr(cv_loss, 'item'):
                cv_loss = cv_loss.item()
            else:
                cv_loss = float(cv_loss)
            
            # Return CV loss for optimization (already a loss, not a score)
            return cv_loss
                
        except Exception as e:
            # Return high penalty for invalid configurations
            warnings.warn(f"Evaluation failed with params {params}: {e}")
            return 1e6


def create_hpo_benchmarks_suite() -> BenchmarkSuite:
    """Create HPO benchmarks suite using Bayesmark SklearnModel interface."""
    if not BAYESMARK_AVAILABLE:
        raise ImportError("HPO benchmarks require bayesmark. Install with: pip install bayesmark")
    
    functions = {}
    
    # Use Bayesmark's standard model and dataset names
    # Classification models and datasets from bayesmark
    classification_configs = [
        # (model, dataset, metrics)
        ('DT', 'iris', ['nll', 'acc']),
        ('DT', 'wine', ['nll', 'acc']), 
        ('DT', 'digits', ['nll', 'acc']),
        ('DT', 'breast', ['nll', 'acc']),
        
        ('MLP-sgd', 'iris', ['nll', 'acc']),
        ('MLP-sgd', 'wine', ['nll', 'acc']),
        ('MLP-sgd', 'digits', ['nll', 'acc']),
        ('MLP-sgd', 'breast', ['nll', 'acc']),
        
        ('RF', 'iris', ['nll', 'acc']),
        ('RF', 'wine', ['nll', 'acc']),
        ('RF', 'digits', ['nll', 'acc']),
        ('RF', 'breast', ['nll', 'acc']),
        
        ('SVM', 'iris', ['nll', 'acc']),
        ('SVM', 'wine', ['nll', 'acc']),
        ('SVM', 'digits', ['nll', 'acc']),
        ('SVM', 'breast', ['nll', 'acc']),
        
        ('ada', 'iris', ['nll', 'acc']),
        ('ada', 'wine', ['nll', 'acc']),
        ('ada', 'digits', ['nll', 'acc']),
        ('ada', 'breast', ['nll', 'acc']),
        
        ('kNN', 'iris', ['nll', 'acc']),
        ('kNN', 'wine', ['nll', 'acc']),
        ('kNN', 'digits', ['nll', 'acc']),
        ('kNN', 'breast', ['nll', 'acc']),
        
        ('lasso', 'iris', ['nll', 'acc']),
        ('lasso', 'wine', ['nll', 'acc']),
        ('lasso', 'digits', ['nll', 'acc']),
        ('lasso', 'breast', ['nll', 'acc']),
    ]
    
    # Regression models and datasets from bayesmark  
    # NOTE: Boston dataset has been deprecated and removed from sklearn.
    # We only use 'diabetes' dataset for regression benchmarks.
    regression_configs = [
        # (model, dataset, metrics)
        ('DT', 'diabetes', ['mse', 'mae']),
        ('MLP-sgd', 'diabetes', ['mse', 'mae']),
        ('RF', 'diabetes', ['mse', 'mae']),
        ('SVM', 'diabetes', ['mse', 'mae']),
        ('ada', 'diabetes', ['mse', 'mae']),
        ('kNN', 'diabetes', ['mse', 'mae']),
        ('lasso', 'diabetes', ['mse', 'mae']),
        ('linear', 'diabetes', ['mse', 'mae']),
    ]
    
    # Create all benchmark functions
    all_configs = classification_configs + regression_configs
    
    for model, dataset, metrics in all_configs:
        for metric in metrics:
            try:
                func_name = f"{model}_{dataset}_{metric}"
                functions[func_name] = HPOBenchmarkFunction(
                    model_name=model,
                    dataset_name=dataset,
                    metric=metric,
                    shuffle_seed=0  # Fixed seed for reproducibility
                )
            except Exception as e:
                warnings.warn(f"Failed to create {func_name}: {e}")
    
    return BenchmarkSuite("HPO Benchmarks", functions)


# Create suite instance if dependencies are available
if BAYESMARK_AVAILABLE and SKLEARN_AVAILABLE:
    try:
        HPOBenchmarksSuite = create_hpo_benchmarks_suite()
    except Exception as e:
        print(f"Warning: HPO benchmarks suite creation failed: {e}")
        HPOBenchmarksSuite = None
else:
    HPOBenchmarksSuite = None

__all__ = [
    "HPOBenchmarkFunction",
    "create_hpo_benchmarks_suite",
    "HPOBenchmarksSuite"
] 