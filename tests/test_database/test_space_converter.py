"""Tests for space converter."""

import pytest
import numpy as np
from bomegabench.functions.database.space_converter import SpaceConverter


class TestSpaceConverter:
    """Test space conversion between continuous and discrete representations."""

    @pytest.fixture
    def simple_knob_config(self):
        """Simple knob configuration for testing."""
        return {
            "int_param": {
                "type": "int",
                "min": 1,
                "max": 5,
                "default": 3
            },
            "float_param": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5
            },
            "enum_param": {
                "type": "enum",
                "choices": ["small", "medium", "large"],
                "default": "medium"
            },
            "bool_param": {
                "type": "bool",
                "default": True
            }
        }

    @pytest.fixture
    def converter(self, simple_knob_config):
        """Create space converter instance."""
        return SpaceConverter(simple_knob_config)

    def test_initialization(self, converter):
        """Test space converter initialization."""
        assert converter is not None
        assert hasattr(converter, 'continuous_space')
        assert hasattr(converter, 'knob_info')

    def test_dimension_calculation(self, converter):
        """Test dimension calculation."""
        # Expected: 5 int values (1-5) + 1 float + 3 enum choices + 2 bool = 11
        assert converter.dim == 11

    def test_continuous_space_structure(self, converter):
        """Test continuous space structure."""
        space = converter.continuous_space

        assert isinstance(space, list)
        assert len(space) == converter.dim

        # Check that all dimensions have required fields
        for dim_spec in space:
            assert "name" in dim_spec
            assert "type" in dim_spec

    def test_float_parameter_mapping(self, converter):
        """Test float parameter mapping."""
        float_dims = [d for d in converter.continuous_space if d["type"] == "float"]
        assert len(float_dims) == 1
        assert float_dims[0]["name"] == "float_param"
        assert float_dims[0]["original_bounds"] == (0.0, 1.0)

    def test_int_parameter_mapping(self, converter):
        """Test integer parameter one-hot encoding."""
        int_dims = [d for d in converter.continuous_space if d["type"] == "int_onehot"]
        assert len(int_dims) == 5  # Values 1-5

        # Check all values are represented
        values = [d["choice"] for d in int_dims]
        assert values == [1, 2, 3, 4, 5]

    def test_enum_parameter_mapping(self, converter):
        """Test enum parameter one-hot encoding."""
        enum_dims = [d for d in converter.continuous_space if d["type"] == "enum_onehot"]
        assert len(enum_dims) == 3  # 3 choices

        choices = [d["choice"] for d in enum_dims]
        assert set(choices) == {"small", "medium", "large"}

    def test_bool_parameter_mapping(self, converter):
        """Test boolean parameter one-hot encoding."""
        bool_dims = [d for d in converter.continuous_space if d["type"] == "bool_onehot"]
        assert len(bool_dims) == 2  # True and False

        choices = [d["choice"] for d in bool_dims]
        assert set(choices) == {True, False}

    def test_continuous_to_discrete_conversion(self, converter):
        """Test conversion from continuous to discrete space."""
        # Create a continuous point with clear argmax choices
        x_continuous = np.zeros(converter.dim)

        # Set values to select specific choices
        # Find indices for each parameter
        int_idx = next(i for i, d in enumerate(converter.continuous_space)
                      if d.get("type") == "int_onehot" and d.get("choice") == 3)
        x_continuous[int_idx] = 1.0  # Select value 3

        float_idx = next(i for i, d in enumerate(converter.continuous_space)
                        if d.get("type") == "float")
        x_continuous[float_idx] = 0.5  # Middle value

        enum_idx = next(i for i, d in enumerate(converter.continuous_space)
                       if d.get("type") == "enum_onehot" and d.get("choice") == "medium")
        x_continuous[enum_idx] = 1.0  # Select "medium"

        bool_idx = next(i for i, d in enumerate(converter.continuous_space)
                       if d.get("type") == "bool_onehot" and d.get("choice") == True)
        x_continuous[bool_idx] = 1.0  # Select True

        config = converter.continuous_to_discrete(x_continuous)

        assert "int_param" in config
        assert config["int_param"] == 3

        assert "float_param" in config
        assert 0.0 <= config["float_param"] <= 1.0

        assert "enum_param" in config
        assert config["enum_param"] == "medium"

        assert "bool_param" in config
        assert config["bool_param"] == True

    def test_discrete_to_continuous_conversion(self, converter):
        """Test conversion from discrete to continuous space."""
        discrete_config = {
            "int_param": 3,
            "float_param": 0.5,
            "enum_param": "medium",
            "bool_param": True
        }

        x_continuous = converter.discrete_to_continuous(discrete_config)

        assert x_continuous.shape == (converter.dim,)
        assert np.all((x_continuous >= 0.0) & (x_continuous <= 1.0))

        # Check that one-hot encoding has 1.0 in the right places
        # For int_param=3
        int_3_idx = next(i for i, d in enumerate(converter.continuous_space)
                        if d.get("type") == "int_onehot" and d.get("choice") == 3)
        assert x_continuous[int_3_idx] == 1.0

        # For enum_param="medium"
        medium_idx = next(i for i, d in enumerate(converter.continuous_space)
                         if d.get("type") == "enum_onehot" and d.get("choice") == "medium")
        assert x_continuous[medium_idx] == 1.0

    def test_round_trip_conversion(self, converter):
        """Test that continuous -> discrete -> continuous preserves configuration."""
        original_config = {
            "int_param": 4,
            "float_param": 0.75,
            "enum_param": "large",
            "bool_param": False
        }

        # Convert to continuous
        x_continuous = converter.discrete_to_continuous(original_config)

        # Convert back to discrete
        recovered_config = converter.continuous_to_discrete(x_continuous)

        # Check that we get back the same discrete values
        assert recovered_config["int_param"] == original_config["int_param"]
        assert recovered_config["enum_param"] == original_config["enum_param"]
        assert recovered_config["bool_param"] == original_config["bool_param"]
        assert np.isclose(recovered_config["float_param"], original_config["float_param"])

    def test_default_values(self, converter):
        """Test that default values are used for missing parameters."""
        partial_config = {
            "int_param": 3
            # Other parameters missing
        }

        x_continuous = converter.discrete_to_continuous(partial_config)
        recovered_config = converter.continuous_to_discrete(x_continuous)

        # Check defaults are used
        assert "float_param" in recovered_config
        assert "enum_param" in recovered_config
        assert "bool_param" in recovered_config

    def test_argmax_selection(self, converter):
        """Test that argmax is used for discrete parameter selection."""
        x_continuous = np.random.rand(converter.dim)

        config = converter.continuous_to_discrete(x_continuous)

        # All parameters should have valid values
        assert config["int_param"] in range(1, 6)
        assert config["enum_param"] in ["small", "medium", "large"]
        assert config["bool_param"] in [True, False]
        assert 0.0 <= config["float_param"] <= 1.0
