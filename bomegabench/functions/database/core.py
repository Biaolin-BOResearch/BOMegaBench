"""
Core database tuning function implementation.

This module provides the main DatabaseTuningFunction class that wraps
database configuration tuning as a Bayesian optimization problem.
"""

import torch
from torch import Tensor
import warnings
from typing import Dict, List, Optional, Any

from ...core import BenchmarkFunction
from .knob_configs import (
    get_default_knob_config,
    validate_knob_config,
    get_knob_documentation
)
from .space_converter import SpaceConverter
from .evaluator import DatabaseEvaluator


class DatabaseTuningFunction(BenchmarkFunction):
    """
    Wrapper for database knob tuning benchmarks.

    Converts mixed-type database configuration knobs (int, float, categorical, boolean)
    to continuous [0, 1] representation for Bayesian Optimization.

    Parameters
    ----------
    workload_name : str
        Name of the workload (e.g., "tpcc", "tpch", "ycsb")
    database_system : str
        Database system (e.g., "postgresql", "mysql")
    knob_config : Dict[str, Dict[str, Any]], optional
        Configuration of tunable knobs. If None, uses default for database_system.
    performance_metric : str
        Performance metric to optimize: "latency" or "throughput"
    benchbase_path : str, optional
        Path to BenchBase installation
    db_host : str
        Database host
    db_port : int, optional
        Database port
    db_name : str
        Database name
    db_user : str
        Database user
    db_password : str
        Database password
    benchmark_runtime : int
        Benchmark runtime in seconds
    benchmark_terminals : int
        Number of concurrent benchmark terminals
    scale_factor : int
        Benchmark scale factor
    **kwargs
        Additional arguments passed to BenchmarkFunction

    Examples
    --------
    >>> knob_config = {
    ...     "shared_buffers_mb": {
    ...         "type": "int", "min": 128, "max": 16384,
    ...         "default": 1024, "description": "Shared memory buffers (MB)"
    ...     }
    ... }
    >>> func = DatabaseTuningFunction("tpcc", "postgresql", knob_config)
    >>> X = torch.rand(1, func.dim)
    >>> performance = func(X)
    """

    def __init__(
        self,
        workload_name: str,
        database_system: str = "postgresql",
        knob_config: Optional[Dict[str, Dict[str, Any]]] = None,
        performance_metric: str = "latency",
        benchbase_path: Optional[str] = None,
        db_host: str = "localhost",
        db_port: Optional[int] = None,
        db_name: str = "benchbase",
        db_user: str = "postgres",
        db_password: str = "password",
        benchmark_runtime: int = 60,
        benchmark_terminals: int = 1,
        scale_factor: int = 1,
        **kwargs
    ):
        """Initialize database tuning function."""
        self.workload_name = workload_name
        self.database_system = database_system
        self.performance_metric = performance_metric

        # Use provided knob config or load default
        if knob_config is None:
            self.knob_info = get_default_knob_config(database_system)
        else:
            self.knob_info = knob_config

        # Validate knob configuration
        validate_knob_config(self.knob_info)

        # Create space converter
        self.space_converter = SpaceConverter(self.knob_info)

        # Initialize BenchBase wrapper if available and path provided
        benchbase_wrapper = None
        if benchbase_path:
            benchbase_wrapper = self._initialize_benchbase(
                benchbase_path, db_host, db_port, db_name, db_user, db_password
            )

        # Create evaluator
        self.evaluator = DatabaseEvaluator(
            workload_name=workload_name,
            database_system=database_system,
            performance_metric=performance_metric,
            benchbase_wrapper=benchbase_wrapper,
            benchmark_runtime=benchmark_runtime,
            benchmark_terminals=benchmark_terminals,
            scale_factor=scale_factor
        )

        # Set bounds
        dim = self.space_converter.dim
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim])

        super().__init__(dim=dim, bounds=bounds, **kwargs)

    def _initialize_benchbase(
        self,
        benchbase_path: str,
        db_host: str,
        db_port: Optional[int],
        db_name: str,
        db_user: str,
        db_password: str
    ) -> Optional[Any]:
        """
        Initialize BenchBase wrapper.

        Returns
        -------
        Optional[Any]
            BenchBase wrapper instance, or None if initialization failed
        """
        try:
            from ..benchbase_wrapper import BenchBaseWrapper

            return BenchBaseWrapper(
                benchbase_path=benchbase_path,
                database_type=self._map_db_system_to_benchbase(),
                db_host=db_host,
                db_port=db_port,
                db_name=db_name,
                db_user=db_user,
                db_password=db_password
            )
        except ImportError:
            warnings.warn(
                "BenchBase wrapper not available. Database tuning benchmarks will use placeholder evaluation."
            )
            return None
        except Exception as e:
            warnings.warn(f"Could not initialize BenchBase: {e}")
            return None

    def _map_db_system_to_benchbase(self) -> str:
        """Map database system name to BenchBase database type."""
        mapping = {
            "postgresql": "postgres",
            "postgres": "postgres",
            "mysql": "mysql",
            "mariadb": "mariadb",
            "sqlite": "sqlite",
            "cockroachdb": "cockroachdb"
        }
        return mapping.get(self.database_system.lower(), "postgres")

    def get_knob_documentation(self) -> str:
        """
        Get formatted documentation of all tunable knobs.

        Returns
        -------
        str
            Formatted documentation string
        """
        return get_knob_documentation(
            self.knob_info,
            self.database_system,
            self.workload_name,
            self.dim
        )

    def _get_metadata(self) -> Dict[str, Any]:
        """Return function metadata."""
        return {
            "name": f"{self.database_system.capitalize()} {self.workload_name.upper()}",
            "suite": "Database Tuning",
            "properties": ["configuration_tuning", "database_workload", "expensive"],
            "database_system": self.database_system,
            "workload": self.workload_name,
            "total_knobs": len(self.knob_info),
            "continuous_dimensions": self.dim,
            "performance_metric": self.performance_metric,
            "domain": f"Continuous [0,1]^{self.dim} mapping to database knobs",
            "global_min": "Unknown - depends on workload and database state",
            "knob_details": self.knob_info
        }

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate the function at given points.

        Parameters
        ----------
        X : Tensor
            Tensor of shape (..., dim) with values in [0,1]

        Returns
        -------
        Tensor
            Performance metrics (lower is better)
        """
        # Handle batch evaluation
        if X.ndim == 1:
            X_batch = X.unsqueeze(0)
        else:
            X_batch = X

        results = []
        X_np = X_batch.detach().cpu().numpy()

        for x_point in X_np:
            # Convert continuous to discrete knob configuration
            knob_config = self.space_converter.continuous_to_discrete(x_point)

            # Evaluate configuration
            performance = self.evaluator.evaluate_configuration(knob_config)
            results.append(performance)

        return torch.tensor(results, dtype=X.dtype, device=X.device)
