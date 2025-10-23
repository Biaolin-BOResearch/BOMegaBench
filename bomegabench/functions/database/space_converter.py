"""
Space conversion utilities for database knob tuning.

This module handles the conversion between continuous [0,1] space used by
Bayesian optimization and discrete/mixed-type database configuration space.
"""

import numpy as np
from typing import Dict, List, Any, Tuple


class SpaceConverter:
    """
    Converts between continuous [0,1] space and discrete database configuration space.

    Following HPOBench pattern: discrete/categorical parameters are one-hot encoded,
    where each possible value gets its own dimension in [0,1].
    """

    def __init__(self, knob_info: Dict[str, Dict[str, Any]]):
        """
        Initialize space converter.

        Parameters
        ----------
        knob_info : Dict[str, Dict[str, Any]]
            Dictionary mapping knob names to their specifications
        """
        self.knob_info = knob_info
        self.continuous_space = self._create_continuous_space()

    @property
    def dim(self) -> int:
        """Total number of continuous dimensions."""
        return len(self.continuous_space)

    def _create_continuous_space(self) -> List[Dict[str, Any]]:
        """
        Convert mixed-type knob space to continuous [0,1] dimensions.

        Returns
        -------
        List[Dict[str, Any]]
            List of dimension specifications for mapping continuous -> discrete
        """
        continuous_dims = []

        for knob_name, knob_spec in self.knob_info.items():
            knob_type = knob_spec["type"]

            if knob_type == "float":
                # Float parameters: single dimension, normalized to [0,1]
                continuous_dims.append({
                    "name": knob_name,
                    "type": "float",
                    "original_bounds": (knob_spec["min"], knob_spec["max"]),
                    "knob_name": knob_name
                })

            elif knob_type == "int":
                # Integer parameters: one dimension per possible value (one-hot encoding)
                min_val, max_val = knob_spec["min"], knob_spec["max"]
                int_values = list(range(min_val, max_val + 1))
                for i, value in enumerate(int_values):
                    continuous_dims.append({
                        "name": f"{knob_name}_{value}",
                        "type": "int_onehot",
                        "original_param": knob_name,
                        "choice": value,
                        "choice_index": i,
                        "total_choices": len(int_values)
                    })

            elif knob_type == "enum":
                # Enum/categorical: one dimension per choice (one-hot encoding)
                choices = knob_spec["choices"]
                for i, choice in enumerate(choices):
                    continuous_dims.append({
                        "name": f"{knob_name}_{choice}",
                        "type": "enum_onehot",
                        "original_param": knob_name,
                        "choice": choice,
                        "choice_index": i,
                        "total_choices": len(choices)
                    })

            elif knob_type == "bool":
                # Boolean: two dimensions for True/False (one-hot encoding)
                continuous_dims.append({
                    "name": f"{knob_name}_False",
                    "type": "bool_onehot",
                    "original_param": knob_name,
                    "choice": False,
                    "choice_index": 0,
                    "total_choices": 2
                })
                continuous_dims.append({
                    "name": f"{knob_name}_True",
                    "type": "bool_onehot",
                    "original_param": knob_name,
                    "choice": True,
                    "choice_index": 1,
                    "total_choices": 2
                })

        return continuous_dims

    def continuous_to_discrete(self, x_continuous: np.ndarray) -> Dict[str, Any]:
        """
        Convert continuous [0,1] point to discrete knob configuration.

        Uses argmax for one-hot encoded discrete/categorical parameters.

        Parameters
        ----------
        x_continuous : np.ndarray
            Continuous point in [0,1]^d

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping knob names to their values
        """
        config = {}
        int_params = {}  # Track one-hot encoded integer parameters
        enum_params = {}  # Track one-hot encoded enum parameters
        bool_params = {}  # Track one-hot encoded boolean parameters

        for dim_idx, dim_spec in enumerate(self.continuous_space):
            value = x_continuous[dim_idx]
            dim_type = dim_spec["type"]

            if dim_type == "float":
                # Denormalize to original bounds
                knob_name = dim_spec["knob_name"]
                min_val, max_val = dim_spec["original_bounds"]
                config[knob_name] = float(min_val + value * (max_val - min_val))

            elif dim_type == "int_onehot":
                # Collect all dimensions for this integer parameter
                param_name = dim_spec["original_param"]
                if param_name not in int_params:
                    int_params[param_name] = []
                int_params[param_name].append((value, dim_spec["choice"]))

            elif dim_type == "enum_onehot":
                # Collect all dimensions for this enum parameter
                param_name = dim_spec["original_param"]
                if param_name not in enum_params:
                    enum_params[param_name] = []
                enum_params[param_name].append((value, dim_spec["choice"]))

            elif dim_type == "bool_onehot":
                # Collect all dimensions for this boolean parameter
                param_name = dim_spec["original_param"]
                if param_name not in bool_params:
                    bool_params[param_name] = []
                bool_params[param_name].append((value, dim_spec["choice"]))

        # For integer parameters, choose the one with highest value (argmax)
        for param_name, choices in int_params.items():
            best_choice = max(choices, key=lambda x: x[0])[1]
            config[param_name] = best_choice

        # For enum parameters, choose the one with highest value (argmax)
        for param_name, choices in enum_params.items():
            best_choice = max(choices, key=lambda x: x[0])[1]
            config[param_name] = best_choice

        # For boolean parameters, choose the one with highest value (argmax)
        for param_name, choices in bool_params.items():
            best_choice = max(choices, key=lambda x: x[0])[1]
            config[param_name] = best_choice

        return config

    def discrete_to_continuous(self, config: Dict[str, Any]) -> np.ndarray:
        """
        Convert discrete knob configuration to continuous [0,1] point.

        Useful for initializing the optimizer with a known good configuration.
        For one-hot encoded parameters, sets the corresponding dimension to 1.0 and others to 0.0.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary mapping knob names to their values

        Returns
        -------
        np.ndarray
            Continuous point in [0,1]^d
        """
        x_continuous = np.zeros(self.dim)

        for dim_idx, dim_spec in enumerate(self.continuous_space):
            dim_type = dim_spec["type"]

            if dim_type == "float":
                # Normalize to [0,1]
                knob_name = dim_spec["knob_name"]
                value = config.get(knob_name)
                if value is None:
                    value = self.knob_info[knob_name].get(
                        "default",
                        (dim_spec["original_bounds"][0] + dim_spec["original_bounds"][1]) / 2
                    )
                min_val, max_val = dim_spec["original_bounds"]
                x_continuous[dim_idx] = (value - min_val) / (max_val - min_val)

            elif dim_type == "int_onehot":
                # Set to 1.0 if this is the selected choice, 0.0 otherwise
                param_name = dim_spec["original_param"]
                actual_value = config.get(param_name)
                if actual_value is None:
                    actual_value = self.knob_info[param_name].get("default")

                if actual_value == dim_spec["choice"]:
                    x_continuous[dim_idx] = 1.0
                else:
                    x_continuous[dim_idx] = 0.0

            elif dim_type == "enum_onehot":
                # Set to 1.0 if this is the selected choice, 0.0 otherwise
                param_name = dim_spec["original_param"]
                actual_value = config.get(param_name)
                if actual_value is None:
                    actual_value = self.knob_info[param_name].get("default")

                if actual_value == dim_spec["choice"]:
                    x_continuous[dim_idx] = 1.0
                else:
                    x_continuous[dim_idx] = 0.0

            elif dim_type == "bool_onehot":
                # Set to 1.0 if this is the selected choice, 0.0 otherwise
                param_name = dim_spec["original_param"]
                actual_value = config.get(param_name)
                if actual_value is None:
                    actual_value = self.knob_info[param_name].get("default", False)

                if actual_value == dim_spec["choice"]:
                    x_continuous[dim_idx] = 1.0
                else:
                    x_continuous[dim_idx] = 0.0

        return x_continuous
