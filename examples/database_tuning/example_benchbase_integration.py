"""
Example: BenchBase Integration with BOMegaBench Database Tuning

This example demonstrates how to use BenchBase for database knob tuning
with Bayesian Optimization.

Prerequisites:
1. Install BenchBase (see BENCHBASE_SETUP.md)
2. Set up PostgreSQL database
3. Install Python dependencies: psycopg2-binary, numpy

Usage:
    python example_benchbase_integration.py
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bomegabench.functions.database_tuning import DatabaseTuningFunction
from bomegabench.functions.benchbase_wrapper import BenchBaseWrapper


def example_1_wrapper_usage():
    """Example 1: Direct usage of BenchBase wrapper."""
    print("=" * 80)
    print("Example 1: Direct BenchBase Wrapper Usage")
    print("=" * 80)

    # Initialize BenchBase wrapper
    wrapper = BenchBaseWrapper(
        benchbase_path="/path/to/benchbase-postgres",  # UPDATE THIS PATH
        database_type="postgres",
        db_host="localhost",
        db_port=5432,
        db_name="benchbase",
        db_user="postgres",
        db_password="password"
    )

    # Step 1: Create schema and load data (first time only)
    print("\n--- Creating schema and loading data ---")
    result = wrapper.run_benchmark(
        benchmark="tpcc",
        create=True,
        load=True,
        execute=False,
        scale_factor=1  # 1 warehouse
    )
    print(f"Setup result: {result}")

    # Step 2: Apply database configuration
    print("\n--- Applying database configuration ---")
    knobs = {
        "shared_buffers_mb": 1024,
        "work_mem_mb": 16,
        "max_connections": 200,
        "effective_cache_size_mb": 4096
    }
    wrapper.apply_database_config(knobs)

    # Step 3: Run benchmark
    print("\n--- Running benchmark ---")
    result = wrapper.run_benchmark(
        benchmark="tpcc",
        execute=True,
        terminals=4,
        runtime_seconds=60
    )

    print("\n--- Results ---")
    print(f"Throughput: {result['throughput_txns_sec']:.2f} txn/s")
    print(f"Average Latency: {result['avg_latency_ms']:.2f} ms")
    print(f"P95 Latency: {result['p95_latency_ms']:.2f} ms")
    print(f"P99 Latency: {result['p99_latency_ms']:.2f} ms")


def example_2_database_tuning_function():
    """Example 2: Using DatabaseTuningFunction with BenchBase."""
    print("\n" + "=" * 80)
    print("Example 2: DatabaseTuningFunction with BenchBase")
    print("=" * 80)

    # Create database tuning function
    func = DatabaseTuningFunction(
        workload_name="tpcc",
        database_system="postgresql",
        benchbase_path="/path/to/benchbase-postgres",  # UPDATE THIS PATH
        db_host="localhost",
        db_port=5432,
        db_name="benchbase",
        db_user="postgres",
        db_password="password",
        benchmark_runtime=30,  # Shorter for testing
        benchmark_terminals=2,
        scale_factor=1,
        performance_metric="latency"
    )

    print(f"\nFunction dimension: {func.dim}")
    print(f"Number of knobs: {len(func.knob_info)}")

    # Print knob documentation
    print("\n" + func.get_knob_documentation())

    # Evaluate a random configuration
    print("\n--- Evaluating random configuration ---")
    X = torch.rand(1, func.dim)
    print(f"Random input (first 10 dims): {X[0, :10].numpy()}")

    # This will internally:
    # 1. Convert continuous [0,1] to discrete knob values
    # 2. Apply knobs to database
    # 3. Run benchmark
    # 4. Return performance metric
    performance = func(X)
    print(f"Performance (latency): {performance.item():.2f} ms")


def example_3_bayesian_optimization():
    """Example 3: Bayesian Optimization for database tuning."""
    print("\n" + "=" * 80)
    print("Example 3: Bayesian Optimization for Database Tuning")
    print("=" * 80)

    try:
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_model
        from botorch.acquisition import ExpectedImprovement
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError:
        print("BoTorch not installed. Install with: pip install botorch")
        return

    # Create function
    func = DatabaseTuningFunction(
        workload_name="tpcc",
        database_system="postgresql",
        benchbase_path="/path/to/benchbase-postgres",  # UPDATE THIS PATH
        db_host="localhost",
        db_port=5432,
        db_name="benchbase",
        db_user="postgres",
        db_password="password",
        benchmark_runtime=30,
        benchmark_terminals=2,
        scale_factor=1,
        performance_metric="latency"
    )

    print(f"Optimizing {func.dim}-dimensional configuration space")

    # Initialize with random samples
    n_initial = 3
    print(f"\n--- Initial random sampling ({n_initial} points) ---")
    X_train = torch.rand(n_initial, func.dim)
    Y_train = torch.zeros(n_initial, 1)

    for i in range(n_initial):
        y = func(X_train[i:i+1])
        Y_train[i] = y
        print(f"Sample {i+1}/{n_initial}: Latency = {y.item():.2f} ms")

    # Bayesian Optimization loop
    n_iterations = 5
    print(f"\n--- Bayesian Optimization ({n_iterations} iterations) ---")

    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")

        # Fit GP model
        gp = SingleTaskGP(X_train, Y_train)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        # Optimize acquisition function
        EI = ExpectedImprovement(gp, best_f=Y_train.min())
        candidate, acq_value = optimize_acqf(
            EI,
            bounds=func.bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )

        # Evaluate candidate
        y_new = func(candidate)
        print(f"Candidate latency: {y_new.item():.2f} ms")
        print(f"Best so far: {Y_train.min().item():.2f} ms")

        # Update training data
        X_train = torch.cat([X_train, candidate])
        Y_train = torch.cat([Y_train, y_new.unsqueeze(-1)])

    # Report best configuration
    best_idx = Y_train.argmin()
    best_x = X_train[best_idx]
    best_y = Y_train[best_idx]

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Best latency: {best_y.item():.2f} ms")

    # Convert best configuration to knob values
    from bomegabench.functions.database_tuning import DatabaseTuningFunction
    best_config = func._convert_continuous_to_discrete(best_x.numpy())
    print("\nBest configuration:")
    for knob, value in best_config.items():
        print(f"  {knob}: {value}")


def example_4_custom_knobs():
    """Example 4: Custom knob configuration."""
    print("\n" + "=" * 80)
    print("Example 4: Custom Knob Configuration")
    print("=" * 80)

    # Define custom knobs to tune
    custom_knobs = {
        "shared_buffers_mb": {
            "type": "int",
            "min": 512,
            "max": 8192,
            "default": 1024,
            "description": "Size of shared memory buffers (MB)",
            "category": "memory"
        },
        "work_mem_mb": {
            "type": "int",
            "min": 4,
            "max": 512,
            "default": 16,
            "description": "Memory for sort and hash operations (MB)",
            "category": "memory"
        },
        "checkpoint_completion_target": {
            "type": "float",
            "min": 0.1,
            "max": 0.9,
            "default": 0.5,
            "description": "Checkpoint completion target",
            "category": "wal"
        }
    }

    func = DatabaseTuningFunction(
        workload_name="ycsb",
        database_system="postgresql",
        knob_config=custom_knobs,
        benchbase_path="/path/to/benchbase-postgres",  # UPDATE THIS PATH
        db_host="localhost",
        db_port=5432,
        db_name="benchbase",
        db_user="postgres",
        db_password="password"
    )

    print(f"\nTuning {len(custom_knobs)} knobs")
    print(f"Continuous dimension: {func.dim}")
    print("\nKnob configuration:")
    print(func.get_knob_documentation())


def example_5_setup_database():
    """Example 5: One-time database setup."""
    print("\n" + "=" * 80)
    print("Example 5: Database Setup (Run Once)")
    print("=" * 80)

    wrapper = BenchBaseWrapper(
        benchbase_path="/path/to/benchbase-postgres",  # UPDATE THIS PATH
        database_type="postgres",
        db_host="localhost",
        db_port=5432,
        db_name="benchbase",
        db_user="postgres",
        db_password="password"
    )

    # Setup for different benchmarks
    benchmarks = ["tpcc", "ycsb"]

    for benchmark in benchmarks:
        print(f"\n--- Setting up {benchmark.upper()} ---")
        try:
            result = wrapper.run_benchmark(
                benchmark=benchmark,
                create=True,
                load=True,
                execute=False,
                scale_factor=1
            )
            print(f"Setup complete: {result}")
        except Exception as e:
            print(f"Setup failed: {e}")


if __name__ == "__main__":
    print("BenchBase Integration Examples")
    print("=" * 80)
    print("\nNOTE: Update the benchbase_path in each example before running!")
    print("      Also ensure PostgreSQL is running and accessible.")
    print("=" * 80)

    # Uncomment the example you want to run:

    # example_1_wrapper_usage()
    # example_2_database_tuning_function()
    # example_3_bayesian_optimization()
    # example_4_custom_knobs()
    # example_5_setup_database()

    print("\n" + "=" * 80)
    print("To run an example, uncomment it in the main block and execute:")
    print("  python example_benchbase_integration.py")
    print("=" * 80)
