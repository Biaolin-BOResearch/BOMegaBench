"""Tests for database knob configurations."""

import pytest
from bomegabench.functions.database.knob_configs import (
    get_postgresql_knobs,
    get_mysql_knobs,
    get_default_knob_config,
    validate_knob_config,
    get_knob_documentation
)


class TestKnobConfigs:
    """Test knob configuration functions."""

    def test_get_postgresql_knobs(self):
        """Test PostgreSQL knobs retrieval."""
        knobs = get_postgresql_knobs()

        assert isinstance(knobs, dict)
        assert len(knobs) > 0
        assert "shared_buffers_mb" in knobs
        assert "effective_cache_size_mb" in knobs

        # Check knob structure
        for knob_name, knob_spec in knobs.items():
            assert "type" in knob_spec
            assert "description" in knob_spec
            assert "default" in knob_spec

    def test_get_mysql_knobs(self):
        """Test MySQL knobs retrieval."""
        knobs = get_mysql_knobs()

        assert isinstance(knobs, dict)
        assert len(knobs) > 0
        assert "innodb_buffer_pool_size_mb" in knobs

        # Check knob structure
        for knob_name, knob_spec in knobs.items():
            assert "type" in knob_spec
            assert "description" in knob_spec

    def test_get_default_knob_config_postgresql(self):
        """Test getting default config for PostgreSQL."""
        knobs = get_default_knob_config("postgresql")
        assert isinstance(knobs, dict)
        assert "shared_buffers_mb" in knobs

        # Test alias
        knobs_alias = get_default_knob_config("postgres")
        assert knobs_alias == knobs

    def test_get_default_knob_config_mysql(self):
        """Test getting default config for MySQL."""
        knobs = get_default_knob_config("mysql")
        assert isinstance(knobs, dict)
        assert "innodb_buffer_pool_size_mb" in knobs

    def test_get_default_knob_config_unknown(self):
        """Test error handling for unknown database system."""
        with pytest.raises(ValueError, match="No default knob configuration"):
            get_default_knob_config("unknown_db")

    def test_validate_knob_config_valid(self):
        """Test validation of valid knob config."""
        valid_config = {
            "param1": {
                "type": "int",
                "min": 0,
                "max": 100,
                "default": 50
            },
            "param2": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5
            },
            "param3": {
                "type": "enum",
                "choices": ["a", "b", "c"],
                "default": "a"
            },
            "param4": {
                "type": "bool",
                "default": True
            }
        }

        # Should not raise any exception
        validate_knob_config(valid_config)

    def test_validate_knob_config_missing_type(self):
        """Test validation fails when type is missing."""
        invalid_config = {
            "param1": {
                "min": 0,
                "max": 100
            }
        }

        with pytest.raises(ValueError, match="missing 'type' field"):
            validate_knob_config(invalid_config)

    def test_validate_knob_config_invalid_type(self):
        """Test validation fails for invalid type."""
        invalid_config = {
            "param1": {
                "type": "invalid_type",
                "min": 0,
                "max": 100
            }
        }

        with pytest.raises(ValueError, match="invalid type"):
            validate_knob_config(invalid_config)

    def test_validate_knob_config_missing_bounds(self):
        """Test validation fails when int/float missing bounds."""
        invalid_config = {
            "param1": {
                "type": "int",
                "default": 50
            }
        }

        with pytest.raises(ValueError, match="must have 'min' and 'max' fields"):
            validate_knob_config(invalid_config)

    def test_validate_knob_config_missing_choices(self):
        """Test validation fails when enum missing choices."""
        invalid_config = {
            "param1": {
                "type": "enum",
                "default": "a"
            }
        }

        with pytest.raises(ValueError, match="must have non-empty 'choices' field"):
            validate_knob_config(invalid_config)

    def test_get_knob_documentation(self):
        """Test knob documentation generation."""
        knobs = get_postgresql_knobs()
        doc = get_knob_documentation(knobs, "postgresql", "tpcc", 100)

        assert isinstance(doc, str)
        assert "POSTGRESQL" in doc.upper()
        assert "TPCC" in doc.upper()
        assert "Total Tunable Knobs: 8" in doc
        assert "Total Dimensions (continuous): 100" in doc
        assert "shared_buffers_mb" in doc
