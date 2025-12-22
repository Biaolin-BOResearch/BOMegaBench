"""
Olympus Datasets Integration for BOMegaBench.

This module integrates Olympus datasets - real-world optimization problems
based on experimental data from chemistry and materials science. These datasets
are particularly valuable for testing Bayesian optimization algorithm performance.

Available dataset categories (verified available in olympus):
- Chemical reactions: suzuki, benzylation, alkox, snar
- Materials: fullerenes, colors_bob, colors_n9
- Photovoltaics: photo_pce10, photo_wf3
- Other: hplc

Reference: https://github.com/aspuru-guzik-group/olympus
"""

import sys
import os
import warnings

# ============================================================================
# CRITICAL: Patch matplotlib BEFORE importing olympus
# matplotlib >= 3.7 removed register_cmap, but olympus uses it internally
# This patch MUST happen before any olympus import
# ============================================================================
def _patch_matplotlib_for_olympus():
    """Patch matplotlib.pyplot.register_cmap for compatibility with olympus."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        
        # Check if register_cmap is missing (matplotlib >= 3.7)
        if not hasattr(plt, 'register_cmap'):
            def _register_cmap_compat(name=None, cmap=None):
                """Compatibility shim for register_cmap."""
                if cmap is not None:
                    # Handle the case where cmap has a name attribute
                    cmap_name = name if name is not None else getattr(cmap, 'name', 'custom_cmap')
                    try:
                        colormaps.register(cmap, name=cmap_name)
                    except ValueError:
                        # Already registered, ignore
                        pass
            
            plt.register_cmap = _register_cmap_compat
            matplotlib.pyplot.register_cmap = _register_cmap_compat
            
            # Also patch the cm module if needed
            if hasattr(matplotlib, 'cm') and not hasattr(matplotlib.cm, 'register_cmap'):
                matplotlib.cm.register_cmap = _register_cmap_compat
    except Exception as e:
        warnings.warn(f"Failed to patch matplotlib for olympus compatibility: {e}")

# Apply the patch immediately
_patch_matplotlib_for_olympus()

# Suppress gym deprecation warning - olympus internally uses gym
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

# Try to provide gymnasium as gym for compatibility
try:
    import gymnasium
    if "gym" not in sys.modules:
        sys.modules["gym"] = gymnasium
        sys.modules["gym.spaces"] = gymnasium.spaces
        sys.modules["gym.envs"] = gymnasium.envs
except ImportError:
    pass

# Now safe to import other modules
from typing import Dict, List, Optional, Any, Union
import torch
from torch import Tensor
import numpy as np

# Add olympus to path
olympus_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "olympus", "src")
if olympus_path not in sys.path:
    sys.path.insert(0, olympus_path)

from ..core import BenchmarkFunction, BenchmarkSuite


class OlympusDatasetWrapper(BenchmarkFunction):
    """
    Wrapper for Olympus datasets to match BOMegaBench interface.

    Olympus datasets are trained emulators on real experimental data,
    providing realistic test functions for BO algorithms.
    """

    def _load_dataset_directly(self, dataset_name: str):
        """
        Load dataset data directly from olympus package data files.
        This completely bypasses the Dataset class to avoid NumPy 2.0 issues.
        """
        import pandas as pd
        import olympus
        
        # Get the olympus package data directory
        olympus_dir = os.path.dirname(olympus.__file__)
        datasets_dir = os.path.join(olympus_dir, 'datasets', 'dataset_' + dataset_name)
        
        # Load the data CSV file
        data_file = os.path.join(datasets_dir, 'data.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Dataset file not found: {data_file}")
        
        self._raw_data = pd.read_csv(data_file)
        
        # Load dataset description for param_space
        desc_file = os.path.join(datasets_dir, 'description.txt')
        
        # Create a simple param_space from the data columns
        # The last column is typically the target value
        columns = list(self._raw_data.columns)
        self._param_names = columns[:-1]  # All but last column are parameters
        self._target_name = columns[-1]   # Last column is target
        
        # Create simple bounds from data min/max
        self._param_bounds = {}
        for col in self._param_names:
            self._param_bounds[col] = {
                'low': float(self._raw_data[col].min()),
                'high': float(self._raw_data[col].max())
            }
        
        return True

    def __init__(
        self,
        dataset_name: str,
        use_train_set: bool = False,
        negate: bool = False,
        noise_std: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize Olympus dataset wrapper.

        Args:
            dataset_name: Name of the Olympus dataset (e.g., 'suzuki', 'benzylation')
            use_train_set: Whether to use training set (False = use test set)
            negate: Whether to negate the function
            noise_std: Standard deviation of Gaussian noise
            **kwargs: Additional parameters
        """
        import pandas as pd

        self.dataset_name = dataset_name
        self.use_train_set = use_train_set
        self.olympus_dataset = None
        self._data_loaded_successfully = False

        # Try multiple loading methods
        load_success = False
        load_error = None
        
        # Method 1: Try direct data loading (fastest, most reliable)
        try:
            self._load_dataset_directly(dataset_name)
            load_success = True
            self._data_loaded_successfully = True
        except Exception as e1:
            load_error = e1
        
        # Method 2: Try olympus Dataset class with monkey-patching
        if not load_success:
            try:
                import importlib
                dataset_module = importlib.import_module("olympus.datasets.dataset")
                Dataset = dataset_module.Dataset
                
                # Monkey-patch the problematic split method
                original_split = getattr(Dataset, 'create_train_validate_test_splits', None)
                Dataset.create_train_validate_test_splits = lambda self, *args, **kwargs: None
                
                try:
                    self.olympus_dataset = Dataset(kind=dataset_name)
                    load_success = True
                    self._data_loaded_successfully = True
                finally:
                    if original_split:
                        Dataset.create_train_validate_test_splits = original_split
            except Exception as e2:
                load_error = e2
        
        if not load_success:
            raise ImportError(f"Failed to load Olympus dataset {dataset_name}: {load_error}")

        # Get bounds and dimensions
        if self.olympus_dataset is not None:
            # Use olympus param_space
            param_space = self.olympus_dataset.param_space
            lower_bounds = []
            upper_bounds = []
            self.param_types = []
            
            for param in param_space:
                self.param_types.append(param.type)
                if param.type == 'continuous':
                    lower_bounds.append(param.low)
                    upper_bounds.append(param.high)
                elif param.type == 'discrete':
                    lower_bounds.append(0)
                    upper_bounds.append(len(param.options) - 1)
                elif param.type == 'categorical':
                    lower_bounds.append(0)
                    upper_bounds.append(len(param.options) - 1)
            
            dim = len(param_space)
            self._dataset_info = {
                "num_train": len(self.olympus_dataset.data) if hasattr(self.olympus_dataset, 'data') else 0,
                "num_test": 0,
                "param_space": param_space,
                "dataset_type": getattr(self.olympus_dataset, 'dataset_type', 'unknown'),
            }
        else:
            # Use directly loaded data
            lower_bounds = [self._param_bounds[p]['low'] for p in self._param_names]
            upper_bounds = [self._param_bounds[p]['high'] for p in self._param_names]
            self.param_types = ['continuous'] * len(self._param_names)
            dim = len(self._param_names)
            self._dataset_info = {
                "num_train": len(self._raw_data),
                "num_test": 0,
                "param_space": None,
                "dataset_type": 'unknown',
            }

        bounds = torch.tensor([lower_bounds, upper_bounds], dtype=torch.float32)

        super().__init__(
            dim=dim,
            bounds=bounds,
            negate=negate,
            noise_std=noise_std,
            **kwargs
        )

    def _get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the Olympus dataset."""
        return {
            "name": f"Olympus_{self.dataset_name}",
            "source": "Olympus",
            "type": "real_world",
            "dataset_name": self.dataset_name,
            "description": f"Olympus {self.dataset_name} dataset - real experimental data",
            "reference": "https://github.com/aspuru-guzik-group/olympus",
            "dataset_type": self._dataset_info["dataset_type"],
            "num_train": self._dataset_info["num_train"],
            "num_test": self._dataset_info["num_test"],
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate using the dataset.

        For Olympus datasets, we use nearest neighbor lookup from
        the experimental data (emulators are not used to avoid compatibility issues).
        """
        # Convert to numpy for processing
        X_np = X.detach().cpu().numpy()

        # Remember original shape
        original_shape = X.shape[:-1]

        # Ensure 2D array (n_samples, dim)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
            single_sample = True
        else:
            X_np = X_np.reshape(-1, X.shape[-1])
            single_sample = False

        # Use nearest neighbor from dataset
        results = self._nearest_neighbor_predict(X_np)

        # Convert back to torch
        Y = torch.tensor(results, dtype=X.dtype, device=X.device)

        # Reshape to match input shape
        if single_sample:
            Y = Y.squeeze()
        else:
            Y = Y.reshape(original_shape)

        return Y

    def _nearest_neighbor_predict(self, X_np: np.ndarray) -> np.ndarray:
        """
        Fallback prediction using nearest neighbor from dataset.

        Args:
            X_np: Input array of shape (n_samples, dim)

        Returns:
            Predictions array of shape (n_samples,)
        """
        # Handle directly loaded data (no olympus Dataset object)
        if self.olympus_dataset is None and hasattr(self, '_raw_data'):
            X_data = self._raw_data[self._param_names].values
            y_data = self._raw_data[self._target_name].values
        elif self.olympus_dataset is not None:
            # Get dataset - handle both split and non-split cases
            if hasattr(self.olympus_dataset, 'data'):
                if isinstance(self.olympus_dataset.data, dict):
                    # Has train/test splits
                    if self.use_train_set:
                        dataset = self.olympus_dataset.data.get("train", self.olympus_dataset.data)
                    else:
                        dataset = self.olympus_dataset.data.get("test", self.olympus_dataset.data.get("train", self.olympus_dataset.data))
                else:
                    # No splits, use entire dataset
                    dataset = self.olympus_dataset.data
            else:
                raise ValueError("Dataset has no data attribute")

            # Extract X and y from dataset
            param_names = [p.name for p in self.olympus_dataset.param_space]

            # Handle different data structures
            if hasattr(dataset, 'values'):
                X_data = dataset[param_names].values
                y_data = dataset[self.olympus_dataset.value_space[0].name].values
            else:
                # If dataset is a DataFrame
                X_data = dataset[param_names].to_numpy()
                y_data = dataset[self.olympus_dataset.value_space[0].name].to_numpy()
        else:
            raise ValueError("No data available for prediction")

        # Find nearest neighbors
        results = []
        for x in X_np:
            distances = np.linalg.norm(X_data - x, axis=1)
            nearest_idx = np.argmin(distances)
            results.append(y_data[nearest_idx])

        return np.array(results)


# Define all Olympus datasets organized by category
# NOTE: Only include datasets that are actually available in the olympus package
# Available datasets (verified): alkox, benzylation, colors_bob, colors_n9, 
# fullerenes, hplc, photo_pce10, photo_wf3, snar, suzuki
# Reference: https://aspuru-guzik-group.github.io/olympus/classes/datasets/index.html
OLYMPUS_DATASETS = {
    # Chemical Reactions
    "chemical_reactions": {
        "suzuki": "Suzuki-Miyaura cross-coupling reaction",
        "benzylation": "N-benzylation reaction optimization",
        "alkox": "Alkoxylation reaction",
        "snar": "SNAr nucleophilic aromatic substitution",
    },

    # Materials Science
    "materials": {
        "fullerenes": "Buckminsterfullerene adducts synthesis",
        "colors_bob": "Bob's color mixing dataset",
        "colors_n9": "N9 color optimization",
    },

    # Photovoltaics and Optoelectronics
    "photovoltaics": {
        "photo_pce10": "Photobleaching PCE10 optimization",
        "photo_wf3": "Photobleaching WF3 work function",
    },

    # Other
    "other": {
        "hplc": "HPLC optimization",
    },
}


def create_olympus_datasets_suite(categories: Optional[List[str]] = None) -> BenchmarkSuite:
    """
    Create a suite of Olympus datasets.

    Args:
        categories: List of category names to include. If None, includes all.
                   Categories: 'chemical_reactions', 'materials', 'photovoltaics',
                   'nanoparticles', 'electrochemistry', 'liquids', 'other'

    Returns:
        BenchmarkSuite containing Olympus datasets
    """
    functions = {}

    # Determine which categories to include
    if categories is None:
        categories = list(OLYMPUS_DATASETS.keys())

    for category in categories:
        if category not in OLYMPUS_DATASETS:
            print(f"Warning: Unknown category '{category}'")
            continue

        for dataset_name, description in OLYMPUS_DATASETS[category].items():
            try:
                func = OlympusDatasetWrapper(dataset_name=dataset_name)
                functions[f"olympus_{dataset_name}"] = func
            except Exception as e:
                # Skip if dataset cannot be loaded
                print(f"Warning: Could not load Olympus dataset {dataset_name}: {e}")
                continue

    suite = BenchmarkSuite(
        name="OlympusDatasets",
        functions=functions
    )
    suite.description = "Real-world optimization problems from Olympus based on experimental data"
    return suite


# Create category-specific suites
def create_olympus_chemistry_suite() -> BenchmarkSuite:
    """Create suite with chemical reaction datasets."""
    return create_olympus_datasets_suite(categories=["chemical_reactions"])


def create_olympus_materials_suite() -> BenchmarkSuite:
    """Create suite with materials science datasets."""
    return create_olympus_datasets_suite(categories=["materials"])


def create_olympus_photovoltaics_suite() -> BenchmarkSuite:
    """Create suite with photovoltaics datasets."""
    return create_olympus_datasets_suite(categories=["photovoltaics"])


# Convenience classes for commonly used datasets
# Only include datasets that are actually available in olympus
class OlympusSuzukiFunction(OlympusDatasetWrapper):
    """Olympus Suzuki-Miyaura reaction dataset."""
    def __init__(self, **kwargs):
        super().__init__(dataset_name="suzuki", **kwargs)


class OlympusBenzylationFunction(OlympusDatasetWrapper):
    """Olympus N-benzylation reaction dataset."""
    def __init__(self, **kwargs):
        super().__init__(dataset_name="benzylation", **kwargs)


class OlympusAlkoxFunction(OlympusDatasetWrapper):
    """Olympus alkoxylation reaction dataset."""
    def __init__(self, **kwargs):
        super().__init__(dataset_name="alkox", **kwargs)


class OlympusSnarFunction(OlympusDatasetWrapper):
    """Olympus SNAr reaction dataset."""
    def __init__(self, **kwargs):
        super().__init__(dataset_name="snar", **kwargs)


class OlympusFullerenesFunction(OlympusDatasetWrapper):
    """Olympus Buckminsterfullerene adducts dataset."""
    def __init__(self, **kwargs):
        super().__init__(dataset_name="fullerenes", **kwargs)


class OlympusHplcFunction(OlympusDatasetWrapper):
    """Olympus HPLC optimization dataset."""
    def __init__(self, **kwargs):
        super().__init__(dataset_name="hplc", **kwargs)


class OlympusPhotoPce10Function(OlympusDatasetWrapper):
    """Olympus photobleaching PCE10 dataset."""
    def __init__(self, **kwargs):
        super().__init__(dataset_name="photo_pce10", **kwargs)


class OlympusPhotoWf3Function(OlympusDatasetWrapper):
    """Olympus photobleaching WF3 dataset."""
    def __init__(self, **kwargs):
        super().__init__(dataset_name="photo_wf3", **kwargs)


__all__ = [
    "OlympusDatasetWrapper",
    "create_olympus_datasets_suite",
    "create_olympus_chemistry_suite",
    "create_olympus_materials_suite",
    "create_olympus_photovoltaics_suite",
    "OlympusSuzukiFunction",
    "OlympusBenzylationFunction",
    "OlympusAlkoxFunction",
    "OlympusSnarFunction",
    "OlympusFullerenesFunction",
    "OlympusHplcFunction",
    "OlympusPhotoPce10Function",
    "OlympusPhotoWf3Function",
    "OLYMPUS_DATASETS",
]
