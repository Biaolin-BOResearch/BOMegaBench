"""
Olympus Datasets Integration for BOMegaBench.

This module integrates Olympus datasets - real-world optimization problems
based on experimental data from chemistry and materials science. These datasets
are particularly valuable for testing Bayesian optimization algorithm performance.

Available dataset categories:
- Chemical reactions: Buchwald, Suzuki, benzylation, alkox, etc.
- Materials: Perovskites, fullerenes, dye lasers, etc.
- Photovoltaics: Photo PCE10, Photo WF3, P3HT, MMLI OPV
- Nanoparticles: AgNP, LNP3
- Electrochemistry: OER plates, electrochem
- Liquids: Various liquid properties
- Others: HPLC, AutoAM, thin films, etc.

Reference: https://github.com/aspuru-guzik-group/olympus
"""

import sys
import os
from typing import Dict, List, Optional, Any, Union
import torch
from torch import Tensor
import numpy as np

# Add olympus to path
olympus_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "olympus", "src")
if olympus_path not in sys.path:
    sys.path.insert(0, olympus_path)

from bomegabench.core import BenchmarkFunction, BenchmarkSuite


class OlympusDatasetWrapper(BenchmarkFunction):
    """
    Wrapper for Olympus datasets to match BOMegaBench interface.

    Olympus datasets are trained emulators on real experimental data,
    providing realistic test functions for BO algorithms.
    """

    def _load_dataset_manually(self, dataset_name: str):
        """
        Manually load dataset data without cross-validation splitting.
        This bypasses NumPy compatibility issues in olympus Dataset class.
        """
        import importlib
        import pandas as pd

        # Import Dataset class
        dataset_module = importlib.import_module("olympus.datasets.dataset")

        # Create a minimal dataset object just to get param_space and load data
        # We'll monkey-patch to avoid the problematic split function
        Dataset = dataset_module.Dataset

        # Temporarily replace the split method
        original_split = Dataset.create_train_validate_test_splits
        Dataset.create_train_validate_test_splits = lambda self, *args, **kwargs: None

        try:
            self.olympus_dataset = Dataset(kind=dataset_name)
        finally:
            # Restore original method
            Dataset.create_train_validate_test_splits = original_split

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
            dataset_name: Name of the Olympus dataset (e.g., 'suzuki', 'buchwald_a')
            use_train_set: Whether to use training set (False = use test set)
            negate: Whether to negate the function
            noise_std: Standard deviation of Gaussian noise
            **kwargs: Additional parameters
        """
        # Import directly to avoid circular imports
        import importlib
        import pandas as pd

        self.dataset_name = dataset_name
        self.use_train_set = use_train_set

        # Try to load dataset, with fallback for NumPy compatibility issues
        try:
            # Import Dataset class without going through olympus.__init__
            dataset_module = importlib.import_module("olympus.datasets.dataset")
            Dataset = dataset_module.Dataset

            # Load specific dataset
            self.olympus_dataset = Dataset(kind=dataset_name)
            self._data_loaded_successfully = True

        except (ValueError, Exception) as e:
            # If loading fails due to NumPy issues, load data manually
            if "inhomogeneous shape" in str(e) or "setting an array element with a sequence" in str(e):
                # Load data manually without cross-validation splitting
                self._load_dataset_manually(dataset_name)
                self._data_loaded_successfully = False
            else:
                raise ImportError(f"Failed to load Olympus dataset {dataset_name}: {e}")

        # Get parameter space
        param_space = self.olympus_dataset.param_space

        # Extract bounds
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

        bounds = torch.tensor([lower_bounds, upper_bounds], dtype=torch.float32)
        dim = len(param_space)

        # Store dataset information - handle both split and non-split cases
        if isinstance(self.olympus_dataset.data, dict):
            # Has train/test splits
            self._dataset_info = {
                "num_train": len(self.olympus_dataset.data.get("train", [])),
                "num_test": len(self.olympus_dataset.data.get("test", [])),
                "param_space": param_space,
                "dataset_type": self.olympus_dataset.dataset_type,
            }
        else:
            # No splits, entire dataset
            self._dataset_info = {
                "num_train": len(self.olympus_dataset.data),
                "num_test": 0,
                "param_space": param_space,
                "dataset_type": self.olympus_dataset.dataset_type,
            }

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

        # Find nearest neighbors
        results = []
        for x in X_np:
            distances = np.linalg.norm(X_data - x, axis=1)
            nearest_idx = np.argmin(distances)
            results.append(y_data[nearest_idx])

        return np.array(results)


# Define all Olympus datasets organized by category
OLYMPUS_DATASETS = {
    # Chemical Reactions
    "chemical_reactions": {
        "buchwald_a": "Buchwald-Hartwig C-N cross-coupling reaction (variant A)",
        "buchwald_b": "Buchwald-Hartwig C-N cross-coupling reaction (variant B)",
        "buchwald_c": "Buchwald-Hartwig C-N cross-coupling reaction (variant C)",
        "buchwald_d": "Buchwald-Hartwig C-N cross-coupling reaction (variant D)",
        "buchwald_e": "Buchwald-Hartwig C-N cross-coupling reaction (variant E)",
        "suzuki": "Suzuki-Miyaura cross-coupling reaction",
        "suzuki_edbo": "Suzuki reaction from EDBO paper",
        "suzuki_i": "Suzuki reaction variant I",
        "suzuki_ii": "Suzuki reaction variant II",
        "suzuki_iii": "Suzuki reaction variant III",
        "suzuki_iv": "Suzuki reaction variant IV",
        "benzylation": "Benzylation reaction optimization",
        "alkox": "Alkoxylation reaction",
        "snar": "SNAr nucleophilic aromatic substitution",
    },

    # Materials Science
    "materials": {
        "perovskites": "Perovskite materials optimization",
        "fullerenes": "Fullerene synthesis",
        "dye_lasers": "Organic dye laser optimization",
        "redoxmers": "Redox-active molecules",
        "colors_bob": "Bob's color mixing dataset",
        "colors_n9": "N9 color optimization",
        "thin_film": "Thin film deposition",
        "crossed_barrel": "Crossed barrel optimization",
    },

    # Photovoltaics and Optoelectronics
    "photovoltaics": {
        "photo_pce10": "Photovoltaic PCE10 optimization",
        "photo_wf3": "Photovoltaic WF3 work function",
        "p3ht": "P3HT polymer optimization",
        "mmli_opv": "MMLI organic photovoltaic",
    },

    # Nanoparticles
    "nanoparticles": {
        "agnp": "Silver nanoparticle synthesis",
        "lnp3": "Lipid nanoparticle formulation",
        "autoam": "Automated additive manufacturing",
    },

    # Electrochemistry
    "electrochemistry": {
        "electrochem": "Electrochemical optimization",
        "oer_plate_3496": "Oxygen evolution reaction plate 3496",
        "oer_plate_3851": "Oxygen evolution reaction plate 3851",
        "oer_plate_3860": "Oxygen evolution reaction plate 3860",
        "oer_plate_4098": "Oxygen evolution reaction plate 4098",
    },

    # Liquids and Solvents
    "liquids": {
        "liquid_ace_100": "Acetone properties (100)",
        "liquid_dce": "Dichloroethane properties",
        "liquid_hep_100": "Heptane properties (100)",
        "liquid_thf_100": "THF properties (100)",
        "liquid_thf_500": "THF properties (500)",
        "liquid_toluene": "Toluene properties",
        "liquid_water": "Water properties",
    },

    # Other
    "other": {
        "hplc": "HPLC optimization",
        "vapdiff_crystal": "Vapor diffusion crystallization",
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
class OlympusBuchwaldAFunction(OlympusDatasetWrapper):
    """Olympus Buchwald-Hartwig reaction dataset A."""
    def __init__(self, **kwargs):
        super().__init__(dataset_name="buchwald_a", **kwargs)


class OlympusSuzukiFunction(OlympusDatasetWrapper):
    """Olympus Suzuki-Miyaura reaction dataset."""
    def __init__(self, **kwargs):
        super().__init__(dataset_name="suzuki", **kwargs)


class OlympusPerovskitesFunction(OlympusDatasetWrapper):
    """Olympus perovskites materials dataset."""
    def __init__(self, **kwargs):
        super().__init__(dataset_name="perovskites", **kwargs)


class OlympusDyeLasersFunction(OlympusDatasetWrapper):
    """Olympus dye lasers dataset."""
    def __init__(self, **kwargs):
        super().__init__(dataset_name="dye_lasers", **kwargs)


__all__ = [
    "OlympusDatasetWrapper",
    "create_olympus_datasets_suite",
    "create_olympus_chemistry_suite",
    "create_olympus_materials_suite",
    "create_olympus_photovoltaics_suite",
    "OlympusBuchwaldAFunction",
    "OlympusSuzukiFunction",
    "OlympusPerovskitesFunction",
    "OlympusDyeLasersFunction",
    "OLYMPUS_DATASETS",
]
