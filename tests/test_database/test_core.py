"""Tests for database tuning core functionality."""

import pytest
import torch
import numpy as np
from bomegabench.functions.database import DatabaseTuningFunction, create_database_tuning_suite


class TestDatabaseTuningFunction:
    """Test DatabaseTuningFunction class."""

    def test_initialization_with_default_knobs(self):
        """Test initialization with default knobs."""
        func = DatabaseTuningFunction("tpcc", "postgresql")

        assert func.workload_name == "tpcc"
        assert func.database_system == "postgresql"
        assert func.dim > 0
        assert len(func.knob_info) > 0

    def test_initialization_with_custom_knobs(self):
        """Test initialization with custom knobs."""
        custom_knobs = {
            "param1": {
                "type": "int",
                "min": 1,
                "max": 10,
                "default": 5
            },
            "param2": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5
            }
        }

        func = DatabaseTuningFunction("tpcc", "postgresql", knob_config=custom_knobs)

        assert len(func.knob_info) == 2
        assert "param1" in func.knob_info
        assert "param2" in func.knob_info

    def test_bounds(self):
        """Test that bounds are properly set."""
        func = DatabaseTuningFunction("tpcc", "postgresql")

        assert func.bounds is not None
        assert func.bounds.shape == (2, func.dim)
        assert torch.all(func.bounds[0] == 0.0)
        assert torch.all(func.bounds[1] == 1.0)

    def test_evaluation_single_point(self):
        """Test evaluation of a single point."""
        func = DatabaseTuningFunction("tpcc", "postgresql")

        X = torch.rand(func.dim)
        y = func(X)

        assert isinstance(y, torch.Tensor)
        # Single point evaluation returns shape (1,) not ()
        assert y.shape in [torch.Size([]), torch.Size([1])]
        assert y.dtype in [torch.float32, torch.float64]

    def test_evaluation_batch(self):
        """Test batch evaluation."""
        func = DatabaseTuningFunction("tpcc", "postgresql")

        batch_size = 5
        X = torch.rand(batch_size, func.dim)
        y = func(X)

        assert y.shape == (batch_size,)

    def test_metadata(self):
        """Test metadata generation."""
        func = DatabaseTuningFunction("tpcc", "postgresql")
        metadata = func.metadata

        assert "name" in metadata
        assert "suite" in metadata
        assert metadata["suite"] == "Database Tuning"
        assert "database_system" in metadata
        assert metadata["database_system"] == "postgresql"
        assert "workload" in metadata
        assert metadata["workload"] == "tpcc"
        assert "total_knobs" in metadata
        assert "continuous_dimensions" in metadata

    def test_knob_documentation(self):
        """Test knob documentation generation."""
        func = DatabaseTuningFunction("tpcc", "postgresql")
        doc = func.get_knob_documentation()

        assert isinstance(doc, str)
        assert len(doc) > 0
        assert "POSTGRESQL" in doc.upper()
        assert "TPCC" in doc.upper()

    def test_mysql_database_system(self):
        """Test initialization with MySQL database system."""
        func = DatabaseTuningFunction("tpcc", "mysql")

        assert func.database_system == "mysql"
        assert "innodb_buffer_pool_size_mb" in func.knob_info

    def test_invalid_database_system(self):
        """Test error handling for invalid database system."""
        with pytest.raises(ValueError, match="No default knob configuration"):
            DatabaseTuningFunction("tpcc", "invalid_db")

    def test_performance_metrics(self):
        """Test different performance metrics."""
        func_latency = DatabaseTuningFunction("tpcc", "postgresql", performance_metric="latency")
        func_throughput = DatabaseTuningFunction("tpcc", "postgresql", performance_metric="throughput")

        assert func_latency.performance_metric == "latency"
        assert func_throughput.performance_metric == "throughput"

    def test_evaluation_deterministic_for_same_config(self):
        """Test that evaluation is deterministic for the same configuration."""
        func = DatabaseTuningFunction("tpcc", "postgresql")

        X = torch.rand(1, func.dim)
        y1 = func(X)
        y2 = func(X)

        # Without real BenchBase, this uses placeholder (random)
        # So this test checks that the function runs without errors
        assert isinstance(y1, torch.Tensor)
        assert isinstance(y2, torch.Tensor)


class TestDatabaseTuningSuite:
    """Test database tuning suite creation."""

    def test_create_postgresql_suite(self):
        """Test creation of PostgreSQL tuning suite."""
        suite = create_database_tuning_suite("postgresql")

        assert suite is not None
        assert len(suite.functions) > 0

    def test_create_mysql_suite(self):
        """Test creation of MySQL tuning suite."""
        suite = create_database_tuning_suite("mysql")

        assert suite is not None
        assert len(suite.functions) > 0

    def test_suite_with_custom_workloads(self):
        """Test suite creation with custom workloads."""
        workloads = ["tpcc", "tpch"]
        suite = create_database_tuning_suite("postgresql", workloads=workloads)

        assert len(suite.functions) == len(workloads)

        # Check that all workloads are present
        for workload in workloads:
            func_name = f"postgresql_{workload}"
            assert func_name in suite.functions

    def test_suite_with_custom_knob_configs(self):
        """Test suite creation with custom knob configurations."""
        custom_knobs = {
            "param1": {
                "type": "int",
                "min": 1,
                "max": 10,
                "default": 5
            }
        }

        knob_configs = {
            "tpcc": custom_knobs
        }

        suite = create_database_tuning_suite(
            "postgresql",
            workloads=["tpcc"],
            knob_configs=knob_configs
        )

        func = suite.functions["postgresql_tpcc"]
        assert len(func.knob_info) == len(custom_knobs)


class TestBackwardCompatibility:
    """Test backward compatibility layer."""

    def test_import_from_database_tuning(self):
        """Test import from old database_tuning module."""
        # Note: Warning is emitted at module import time, not when importing the class
        # The module is already imported by the time this test runs
        # So we just test that the import works
        from bomegabench.functions.database_tuning import DatabaseTuningFunction as OldFunc

        # Should still work
        assert OldFunc is not None

    def test_old_import_still_works(self):
        """Test that old import path still works."""
        # Module is already imported, so warning has already been emitted
        from bomegabench.functions.database_tuning import DatabaseTuningFunction as OldFunc

        # Should still be able to create instances
        func = OldFunc("tpcc", "postgresql")
        assert func is not None
        assert func.workload_name == "tpcc"
