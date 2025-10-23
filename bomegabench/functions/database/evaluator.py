"""
Database configuration evaluator using BenchBase.

This module handles the evaluation of database configurations by applying
them to a database and running benchmark workloads.
"""

import warnings
import numpy as np
from typing import Dict, Any, Optional


class DatabaseEvaluator:
    """
    Evaluates database configurations using BenchBase benchmarking tool.
    """

    def __init__(
        self,
        workload_name: str,
        database_system: str,
        performance_metric: str = "latency",
        benchbase_wrapper: Optional[Any] = None,
        benchmark_runtime: int = 60,
        benchmark_terminals: int = 1,
        scale_factor: int = 1,
    ):
        """
        Initialize database evaluator.

        Parameters
        ----------
        workload_name : str
            Name of the workload (e.g., "tpcc", "tpch", "ycsb")
        database_system : str
            Database system (e.g., "postgresql", "mysql")
        performance_metric : str
            Metric to optimize: "latency" or "throughput"
        benchbase_wrapper : Optional[Any]
            BenchBase wrapper instance for running benchmarks
        benchmark_runtime : int
            Benchmark runtime in seconds
        benchmark_terminals : int
            Number of concurrent terminals/threads
        scale_factor : int
            Benchmark scale factor
        """
        self.workload_name = workload_name
        self.database_system = database_system
        self.performance_metric = performance_metric
        self.benchbase = benchbase_wrapper
        self.benchmark_runtime = benchmark_runtime
        self.benchmark_terminals = benchmark_terminals
        self.scale_factor = scale_factor

    def evaluate_configuration(self, knob_config: Dict[str, Any]) -> float:
        """
        Evaluate a specific knob configuration using BenchBase.

        This method:
        1. Applies the knob configuration to the database
        2. Runs the benchmark workload using BenchBase
        3. Returns the performance metric (lower is better)

        Parameters
        ----------
        knob_config : Dict[str, Any]
            Dictionary mapping knob names to their values

        Returns
        -------
        float
            Performance metric (lower is better)
            For latency: return average latency in milliseconds
            For throughput: return 1/throughput (to convert to minimization)

        Examples
        --------
        >>> evaluator = DatabaseEvaluator("tpcc", "postgresql", benchbase_wrapper=wrapper)
        >>> config = {"shared_buffers_mb": 2048, "work_mem_mb": 16}
        >>> performance = evaluator.evaluate_configuration(config)
        """
        if self.benchbase is None:
            # Fallback to placeholder implementation
            warnings.warn(
                "BenchBase not configured. Using placeholder evaluation.\n"
                "To use real benchmarks, provide benchbase_path when creating DatabaseTuningFunction.",
                RuntimeWarning
            )
            # Simple synthetic function for demonstration
            return np.random.random() * 100.0  # Return random latency in ms

        try:
            # Step 1: Apply database configuration
            print(f"Applying configuration: {knob_config}")
            config_applied = self.benchbase.apply_database_config(knob_config)

            if not config_applied:
                warnings.warn("Failed to apply database configuration. Using current settings.")

            # Step 2: Run benchmark
            print(f"Running {self.workload_name} benchmark...")
            result = self.benchbase.run_benchmark(
                benchmark=self.workload_name,
                create=False,  # Assume schema already created
                load=False,    # Assume data already loaded
                execute=True,
                scale_factor=self.scale_factor,
                terminals=self.benchmark_terminals,
                runtime_seconds=self.benchmark_runtime
            )

            # Step 3: Extract and return performance metric
            if not result.get("success", False):
                # Benchmark failed - return penalty value
                warnings.warn(f"Benchmark failed: {result.get('error', 'Unknown error')}")
                return float('inf')

            if self.performance_metric == "latency":
                # Return average latency in milliseconds (lower is better)
                latency = result.get("avg_latency_ms", float('inf'))
                print(f"Average latency: {latency:.2f} ms")
                return float(latency)

            elif self.performance_metric == "throughput":
                # Return inverse throughput to convert to minimization problem
                throughput = result.get("throughput_txns_sec", 0.0)
                if throughput > 0:
                    inverse_throughput = 1.0 / throughput
                    print(f"Throughput: {throughput:.2f} txn/s (inverse: {inverse_throughput:.6f})")
                    return float(inverse_throughput)
                else:
                    print("Zero throughput - returning penalty")
                    return float('inf')

            else:
                raise ValueError(f"Unknown performance metric: {self.performance_metric}")

        except Exception as e:
            warnings.warn(f"Error during benchmark evaluation: {e}")
            import traceback
            traceback.print_exc()
            return float('inf')  # Return penalty for failed evaluation
