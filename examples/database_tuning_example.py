"""
Example usage of Database Tuning benchmarks in BOMegaBench.

This example demonstrates how to use the database knob tuning interface,
including configuration exploration, evaluation, and Bayesian optimization.
"""

import torch
import numpy as np
import bomegabench as bmb
from bomegabench.functions.database_tuning import DatabaseTuningFunction

print("=" * 80)
print("Database Tuning Benchmark Example")
print("=" * 80)

# ============================================================================
# Example 1: List available database tuning benchmarks
# ============================================================================
print("\n1. Listing available database tuning benchmarks...")

suites = bmb.list_suites()
print(f"Available benchmark suites: {suites}")

if "database_tuning" in suites:
    db_functions = bmb.list_functions(suite="database_tuning")
    print(f"\nDatabase tuning functions available: {len(db_functions)}")
    print(f"Functions: {db_functions[:5]}...")  # Show first 5
else:
    print("\nDatabase tuning suite not available. This is expected if not registered.")

# ============================================================================
# Example 2: Create a database tuning function
# ============================================================================
print("\n" + "=" * 80)
print("2. Creating PostgreSQL TPC-C tuning function...")
print("=" * 80)

# Create function with default PostgreSQL knobs
func = DatabaseTuningFunction(
    workload_name="tpcc",
    database_system="postgresql",
    performance_metric="latency"  # or "throughput"
)

print(f"\nFunction created:")
print(f"  Database: {func.database_system}")
print(f"  Workload: {func.workload_name}")
print(f"  Total knobs: {len(func.knob_info)}")
print(f"  Continuous dimensions: {func.dim}")
print(f"  Performance metric: {func.performance_metric}")

# ============================================================================
# Example 3: View knob documentation
# ============================================================================
print("\n" + "=" * 80)
print("3. Viewing knob documentation...")
print("=" * 80)

doc = func.get_knob_documentation()
print(doc)

# ============================================================================
# Example 4: Evaluate a random configuration
# ============================================================================
print("\n" + "=" * 80)
print("4. Evaluating a random configuration...")
print("=" * 80)

# Sample a random point in [0,1]^d
X_random = torch.rand(1, func.dim)
print(f"Random point in continuous space: shape {X_random.shape}")

# Convert to discrete configuration
X_np = X_random.numpy()[0]
config = func._convert_continuous_to_discrete(X_np)
print(f"\nDiscrete configuration:")
for knob, value in config.items():
    print(f"  {knob}: {value}")

# Evaluate (note: this is a placeholder, will return random value)
print("\nEvaluating configuration...")
performance = func(X_random)
print(f"Performance (placeholder): {performance.item():.6f}")

# ============================================================================
# Example 5: Convert a known good configuration
# ============================================================================
print("\n" + "=" * 80)
print("5. Converting a known good configuration...")
print("=" * 80)

# Define a known good configuration
good_config = {
    "shared_buffers_mb": 4096,
    "effective_cache_size_mb": 16384,
    "work_mem_mb": 64,
    "max_connections": 200,
    "random_page_cost": 1.5,
    "effective_io_concurrency": 200,
    "checkpoint_completion_target": 0.9,
    "default_statistics_target": 500
}

print("Known good configuration:")
for knob, value in good_config.items():
    print(f"  {knob}: {value}")

# Convert to continuous space
X_continuous = func._convert_discrete_to_continuous(good_config)
print(f"\nContinuous representation: {X_continuous}")

# Verify round-trip conversion
config_back = func._convert_continuous_to_discrete(X_continuous)
print("\nRound-trip conversion:")
for knob in good_config.keys():
    original = good_config[knob]
    recovered = config_back[knob]
    match = "✓" if original == recovered else "✗"
    print(f"  {match} {knob}: {original} -> {recovered}")

# ============================================================================
# Example 6: Batch evaluation
# ============================================================================
print("\n" + "=" * 80)
print("6. Batch evaluation...")
print("=" * 80)

# Generate batch of random configurations
batch_size = 5
X_batch = torch.rand(batch_size, func.dim)
print(f"Batch size: {batch_size}")

# Evaluate all at once
Y_batch = func(X_batch)
print(f"\nBatch evaluation results:")
for i, y in enumerate(Y_batch):
    print(f"  Configuration {i+1}: {y.item():.6f}")

# ============================================================================
# Example 7: Custom knob configuration
# ============================================================================
print("\n" + "=" * 80)
print("7. Creating function with custom knobs...")
print("=" * 80)

# Define custom knob set (fewer knobs for faster optimization)
custom_knobs = {
    "shared_buffers_mb": {
        "type": "int",
        "min": 128,
        "max": 8192,
        "default": 1024,
        "description": "Shared memory buffers (MB)",
        "category": "memory"
    },
    "work_mem_mb": {
        "type": "int",
        "min": 4,
        "max": 256,
        "default": 16,
        "description": "Work memory (MB)",
        "category": "memory"
    },
    "random_page_cost": {
        "type": "float",
        "min": 1.0,
        "max": 4.0,
        "default": 2.0,
        "description": "Random page access cost",
        "category": "planner"
    },
    "enable_parallel_query": {
        "type": "bool",
        "default": True,
        "description": "Enable parallel query execution",
        "category": "performance"
    }
}

func_custom = DatabaseTuningFunction(
    workload_name="tpcc",
    database_system="postgresql",
    knob_config=custom_knobs
)

print(f"Custom function created:")
print(f"  Total knobs: {len(func_custom.knob_info)}")
print(f"  Continuous dimensions: {func_custom.dim}")

# Breakdown of dimensions
int_dims = sum(1 for d in func_custom.continuous_space if d["type"] == "int_onehot")
float_dims = sum(1 for d in func_custom.continuous_space if d["type"] == "float")
bool_dims = sum(1 for d in func_custom.continuous_space if d["type"] == "bool_onehot")

print(f"\nDimension breakdown:")
print(f"  Integer (one-hot): {int_dims} dimensions")
print(f"  Float: {float_dims} dimensions")
print(f"  Boolean (one-hot): {bool_dims} dimensions")
print(f"  Total: {func_custom.dim} dimensions")

# ============================================================================
# Example 8: Simple Bayesian Optimization
# ============================================================================
print("\n" + "=" * 80)
print("8. Running simple Bayesian Optimization...")
print("=" * 80)

try:
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    # Use custom function with fewer dimensions
    print("Optimizing custom function with 4 knobs...")

    # Initialize with random samples
    n_init = 3
    X_init = torch.rand(n_init, func_custom.dim, dtype=torch.float64)
    Y_init = func_custom(X_init).unsqueeze(-1).to(torch.float64)

    print(f"Initial samples: {n_init}")
    print(f"Initial best: {Y_init.min().item():.6f}")

    # Run BO for a few iterations
    n_iterations = 5
    for iteration in range(n_iterations):
        # Fit GP model
        gp = SingleTaskGP(X_init, Y_init)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Optimize acquisition function
        EI = ExpectedImprovement(gp, best_f=Y_init.min())
        candidate, acq_value = optimize_acqf(
            EI,
            bounds=torch.stack([
                torch.zeros(func_custom.dim, dtype=torch.float64),
                torch.ones(func_custom.dim, dtype=torch.float64)
            ]),
            q=1,
            num_restarts=5,
            raw_samples=128,
        )

        # Evaluate candidate
        Y_new = func_custom(candidate).unsqueeze(-1).to(torch.float64)

        # Update data
        X_init = torch.cat([X_init, candidate])
        Y_init = torch.cat([Y_init, Y_new])

        print(f"Iteration {iteration + 1}: Best = {Y_init.min().item():.6f}, "
              f"New = {Y_new.item():.6f}")

    # Get best configuration
    print("\n" + "-" * 80)
    best_idx = Y_init.argmin()
    best_X = X_init[best_idx].numpy()
    best_config = func_custom._convert_continuous_to_discrete(best_X)
    best_value = Y_init[best_idx].item()

    print(f"Best configuration found (value: {best_value:.6f}):")
    for knob, value in best_config.items():
        default = func_custom.knob_info[knob].get("default")
        print(f"  {knob}: {value} (default: {default})")

except ImportError as e:
    print(f"Skipping BO example: {e}")
    print("Install BoTorch for Bayesian Optimization: pip install botorch gpytorch")

# ============================================================================
# Example 9: Using with BOMegaBench registry
# ============================================================================
print("\n" + "=" * 80)
print("9. Using through BOMegaBench registry...")
print("=" * 80)

try:
    # Get function through registry
    func_registry = bmb.get_function("postgresql_tpcc", suite="database_tuning")

    # View metadata
    print("Function metadata:")
    for key, value in func_registry.metadata.items():
        if key != "knob_details":  # Skip detailed knob info for brevity
            print(f"  {key}: {value}")

    # Evaluate
    X_test = torch.rand(1, func_registry.dim)
    Y_test = func_registry(X_test)
    print(f"\nTest evaluation: {Y_test.item():.6f}")

except (ValueError, KeyError) as e:
    print(f"Function not available through registry: {e}")
    print("This is expected if the suite wasn't properly registered.")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
This example demonstrated:
1. Listing available database tuning benchmarks
2. Creating database tuning functions
3. Viewing knob documentation
4. Evaluating configurations
5. Converting between discrete and continuous spaces
6. Batch evaluation
7. Custom knob configurations
8. Simple Bayesian Optimization
9. Registry integration

To use with real database benchmarking:
1. Override _evaluate_configuration() method
2. Integrate with BenchBase or your benchmarking tool
3. Apply configurations to actual database
4. Run workload and collect performance metrics

Note: Current implementation uses placeholder evaluations.
""")

print("\nExample completed successfully!")
