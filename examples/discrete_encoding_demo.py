"""
Discrete Encoding Demonstration Script

This script demonstrates the three encoding modes for discrete parameters:
1. raw: Direct discrete values
2. interval: [0,1] interval mapping
3. onehot: One-hot encoding

Run this script to see how each encoding mode works in practice.
"""

import torch
from bomegabench.core import BenchmarkFunction
from bomegabench.utils import DiscreteParameterSpec


class HPOBenchmarkExample(BenchmarkFunction):
    """
    Example hyperparameter optimization benchmark.

    Problem: Optimize 4 hyperparameters
    - learning_rate (continuous): [0.0001, 0.1]
    - momentum (continuous): [0.0, 0.99]
    - optimizer (discrete): {sgd, adam, rmsprop}
    - batch_size (discrete): {16, 32, 64, 128, 256}

    Total: 4 dimensions in problem space
    - 2 continuous
    - 2 discrete (3 + 5 options)
    """

    def __init__(self, discrete_encoding="onehot"):
        # Define discrete parameters
        discrete_specs = [
            DiscreteParameterSpec(
                dim_index=2,
                n_options=3,
                options=['sgd', 'adam', 'rmsprop']
            ),
            DiscreteParameterSpec(
                dim_index=3,
                n_options=5,
                options=[16, 32, 64, 128, 256]
            )
        ]

        # Problem space bounds
        bounds = torch.tensor([
            [0.0001, 0.0, 0.0, 0.0],    # lower bounds
            [0.1, 0.99, 2.0, 4.0]       # upper bounds
        ])

        super().__init__(
            dim=4,
            bounds=bounds,
            discrete_specs=discrete_specs,
            discrete_encoding=discrete_encoding
        )

        # Store option names for pretty printing
        self.optimizer_names = ['sgd', 'adam', 'rmsprop']
        self.batch_sizes = [16, 32, 64, 128, 256]

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate validation loss (simulated).

        Args:
            X: Tensor in problem space
               X[..., 0]: learning rate
               X[..., 1]: momentum
               X[..., 2]: optimizer index {0, 1, 2}
               X[..., 3]: batch size index {0, 1, 2, 3, 4}

        Returns:
            Simulated validation loss
        """
        lr = X[..., 0]
        momentum = X[..., 1]
        opt_idx = X[..., 2].long()
        batch_idx = X[..., 3].long()

        # Simulated loss function (lower is better)
        # Optimal: lr=0.01, momentum=0.9, optimizer=adam(1), batch_size=64(2)
        loss = torch.zeros_like(lr)

        # Learning rate term
        loss = loss + 100 * (lr - 0.01)**2

        # Momentum term
        loss = loss + 10 * (momentum - 0.9)**2

        # Optimizer term (prefer adam)
        opt_penalties = torch.tensor([0.5, 0.0, 0.3])  # sgd, adam, rmsprop
        for i in range(len(opt_penalties)):
            mask = (opt_idx == i)
            loss = loss + mask.float() * opt_penalties[i]

        # Batch size term (prefer 64)
        batch_penalties = torch.tensor([0.3, 0.1, 0.0, 0.2, 0.4])  # 16, 32, 64, 128, 256
        for i in range(len(batch_penalties)):
            mask = (batch_idx == i)
            loss = loss + mask.float() * batch_penalties[i]

        return loss

    def _get_metadata(self):
        return {
            'name': 'HPOBenchmarkExample',
            'description': 'Example HPO benchmark with mixed continuous-discrete parameters',
            'optimal_config': {
                'learning_rate': 0.01,
                'momentum': 0.9,
                'optimizer': 'adam',
                'batch_size': 64
            }
        }

    def interpret_solution(self, X_optimizer: Tensor) -> dict:
        """Convert optimizer space solution to human-readable format."""
        # Convert to problem space
        X_problem = self.encode_to_problem_space(X_optimizer)

        # Extract values
        lr = X_problem[0, 0].item()
        momentum = X_problem[0, 1].item()
        opt_idx = int(X_problem[0, 2].item())
        batch_idx = int(X_problem[0, 3].item())

        return {
            'learning_rate': lr,
            'momentum': momentum,
            'optimizer': self.optimizer_names[opt_idx],
            'batch_size': self.batch_sizes[batch_idx]
        }


def demo_raw_encoding():
    """Demonstrate raw encoding mode."""
    print("=" * 80)
    print("1. RAW ENCODING MODE")
    print("=" * 80)
    print("\nDescription: Optimizer works directly with discrete values (integer indices)")
    print("Use case: Mixed-integer Bayesian Optimization algorithms\n")

    func = HPOBenchmarkExample(discrete_encoding="raw")

    print(f"Optimizer dimension: {func.dim}")
    print(f"Problem dimension: {func.problem_dim}")
    print(f"Bounds:\n{func.bounds}\n")

    # Create sample input (optimizer must provide discrete indices)
    X_opt = torch.tensor([[
        0.01,   # learning_rate
        0.9,    # momentum
        1.0,    # optimizer index (adam)
        2.0     # batch_size index (64)
    ]])

    print(f"Optimizer input: {X_opt}")
    print(f"  - learning_rate: {X_opt[0, 0].item():.4f}")
    print(f"  - momentum: {X_opt[0, 1].item():.4f}")
    print(f"  - optimizer index: {int(X_opt[0, 2].item())} (adam)")
    print(f"  - batch_size index: {int(X_opt[0, 3].item())} (64)\n")

    loss = func(X_opt)
    print(f"Loss: {loss.item():.6f}")

    config = func.interpret_solution(X_opt)
    print(f"\nInterpreted configuration:")
    for key, value in config.items():
        print(f"  - {key}: {value}")

    print("\n" + "-" * 80 + "\n")


def demo_interval_encoding():
    """Demonstrate interval encoding mode."""
    print("=" * 80)
    print("2. INTERVAL ENCODING MODE")
    print("=" * 80)
    print("\nDescription: Maps [0,1] continuous values to discrete options via intervals")
    print("Use case: Standard Bayesian Optimization with continuous relaxation\n")

    func = HPOBenchmarkExample(discrete_encoding="interval")

    print(f"Optimizer dimension: {func.dim}")
    print(f"Problem dimension: {func.problem_dim}")
    print(f"Bounds:\n{func.bounds}\n")

    print("Interval mapping for discrete parameters:")
    print("  - optimizer (3 options):")
    print("    [0.000, 0.333) → sgd (0)")
    print("    [0.333, 0.667) → adam (1)")
    print("    [0.667, 1.000] → rmsprop (2)")
    print("  - batch_size (5 options):")
    print("    [0.00, 0.20) → 16 (0)")
    print("    [0.20, 0.40) → 32 (1)")
    print("    [0.40, 0.60) → 64 (2)")
    print("    [0.60, 0.80) → 128 (3)")
    print("    [0.80, 1.00] → 256 (4)\n")

    # Create sample input (optimizer provides [0,1] values for discrete dims)
    X_opt = torch.tensor([[
        0.01,   # learning_rate
        0.9,    # momentum
        0.45,   # optimizer → [0.333, 0.667) → adam (1)
        0.50    # batch_size → [0.40, 0.60) → 64 (2)
    ]])

    print(f"Optimizer input: {X_opt}")
    print(f"  - learning_rate: {X_opt[0, 0].item():.4f}")
    print(f"  - momentum: {X_opt[0, 1].item():.4f}")
    print(f"  - optimizer (interval): {X_opt[0, 2].item():.4f} → adam")
    print(f"  - batch_size (interval): {X_opt[0, 3].item():.4f} → 64\n")

    X_problem = func.encode_to_problem_space(X_opt)
    print(f"Problem space (after encoding): {X_problem}")
    print(f"  - optimizer index: {int(X_problem[0, 2].item())}")
    print(f"  - batch_size index: {int(X_problem[0, 3].item())}\n")

    loss = func(X_opt)
    print(f"Loss: {loss.item():.6f}")

    config = func.interpret_solution(X_opt)
    print(f"\nInterpreted configuration:")
    for key, value in config.items():
        print(f"  - {key}: {value}")

    print("\n" + "-" * 80 + "\n")


def demo_onehot_encoding():
    """Demonstrate one-hot encoding mode."""
    print("=" * 80)
    print("3. ONE-HOT ENCODING MODE")
    print("=" * 80)
    print("\nDescription: Each discrete option becomes a separate continuous dimension")
    print("Use case: Categorical variables with no natural ordering\n")

    func = HPOBenchmarkExample(discrete_encoding="onehot")

    print(f"Optimizer dimension: {func.dim}")
    print(f"Problem dimension: {func.problem_dim}")
    print(f"Dimension expansion: {func.problem_dim} → {func.dim}")
    print(f"  (4 - 2 discrete + 3 options + 5 options = 10)\n")

    print(f"Bounds shape: {func.bounds.shape}\n")

    print("Dimension mapping:")
    print("  Optimizer dims [0-1] → Problem dims [0-1] (learning_rate, momentum)")
    print("  Optimizer dims [2-4] → Problem dim [2] one-hot (optimizer: sgd, adam, rmsprop)")
    print("  Optimizer dims [5-9] → Problem dim [3] one-hot (batch_size: 16, 32, 64, 128, 256)\n")

    # Create sample input (10 dimensions)
    X_opt = torch.zeros(1, 10)
    X_opt[0, 0] = 0.01      # learning_rate
    X_opt[0, 1] = 0.9       # momentum
    X_opt[0, 2:5] = torch.tensor([0.1, 0.8, 0.1])    # optimizer: [sgd, adam, rmsprop] → adam
    X_opt[0, 5:10] = torch.tensor([0.0, 0.1, 0.9, 0.0, 0.0])  # batch_size: [16,32,64,128,256] → 64

    print(f"Optimizer input shape: {X_opt.shape}")
    print(f"Optimizer input:\n{X_opt}\n")
    print("Breakdown:")
    print(f"  - Dims [0-1] (continuous): {X_opt[0, 0:2]}")
    print(f"  - Dims [2-4] (optimizer one-hot): {X_opt[0, 2:5]} → argmax = 1 (adam)")
    print(f"  - Dims [5-9] (batch_size one-hot): {X_opt[0, 5:10]} → argmax = 2 (64)\n")

    X_problem = func.encode_to_problem_space(X_opt)
    print(f"Problem space (after encoding): {X_problem}")
    print(f"  - optimizer index: {int(X_problem[0, 2].item())}")
    print(f"  - batch_size index: {int(X_problem[0, 3].item())}\n")

    loss = func(X_opt)
    print(f"Loss: {loss.item():.6f}")

    config = func.interpret_solution(X_opt)
    print(f"\nInterpreted configuration:")
    for key, value in config.items():
        print(f"  - {key}: {value}")

    print("\n" + "-" * 80 + "\n")


def demo_encoding_comparison():
    """Compare all three encoding modes."""
    print("=" * 80)
    print("4. ENCODING COMPARISON")
    print("=" * 80)
    print("\nEvaluating the same configuration with all three encoding modes:\n")

    # Target configuration in problem space
    # lr=0.01, momentum=0.9, optimizer=adam(1), batch_size=64(2)

    # Raw mode
    func_raw = HPOBenchmarkExample(discrete_encoding="raw")
    X_raw = torch.tensor([[0.01, 0.9, 1.0, 2.0]])
    loss_raw = func_raw(X_raw)

    # Interval mode
    func_interval = HPOBenchmarkExample(discrete_encoding="interval")
    X_interval = torch.tensor([[0.01, 0.9, 0.45, 0.50]])  # 0.45→adam, 0.50→64
    loss_interval = func_interval(X_interval)

    # One-hot mode
    func_onehot = HPOBenchmarkExample(discrete_encoding="onehot")
    X_onehot = torch.zeros(1, 10)
    X_onehot[0, 0] = 0.01
    X_onehot[0, 1] = 0.9
    X_onehot[0, 2:5] = torch.tensor([0.0, 1.0, 0.0])  # adam
    X_onehot[0, 5:10] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])  # 64
    loss_onehot = func_onehot(X_onehot)

    print("Configuration: lr=0.01, momentum=0.9, optimizer=adam, batch_size=64\n")

    print(f"{'Mode':<15} {'Optimizer Dim':<15} {'Loss':<15}")
    print("-" * 45)
    print(f"{'raw':<15} {func_raw.dim:<15} {loss_raw.item():<15.6f}")
    print(f"{'interval':<15} {func_interval.dim:<15} {loss_interval.item():<15.6f}")
    print(f"{'onehot':<15} {func_onehot.dim:<15} {loss_onehot.item():<15.6f}")

    print("\nNote: All three modes should produce the same (or very similar) loss values")
    print("      for the same configuration in problem space.\n")

    # Show encoding info
    print("\nEncoding Information:\n")
    for mode in ["raw", "interval", "onehot"]:
        func = HPOBenchmarkExample(discrete_encoding=mode)
        info = func.get_encoding_info()
        print(f"{mode.upper()}:")
        print(f"  - Optimizer dimension: {info['optimizer_dim']}")
        print(f"  - Problem dimension: {info['continuous_dims']}")
        print(f"  - Discrete params: {info['n_discrete_params']}")
        if info['discrete_specs']:
            for spec in info['discrete_specs']:
                print(f"    - Dim {spec['dim_index']}: {spec['n_options']} options")
        print()


def demo_random_sampling():
    """Demonstrate random sampling in different encoding modes."""
    print("=" * 80)
    print("5. RANDOM SAMPLING")
    print("=" * 80)
    print("\nGenerating random samples in optimizer space:\n")

    for mode in ["raw", "interval", "onehot"]:
        print(f"{mode.upper()} mode:")
        func = HPOBenchmarkExample(discrete_encoding=mode)

        # Sample 3 random points
        X_samples = func.sample_random(3)
        print(f"  Shape: {X_samples.shape}")
        print(f"  Bounds: [{func.bounds[0].min().item():.2f}, {func.bounds[1].max().item():.2f}]")
        print(f"  Samples:")
        for i in range(3):
            print(f"    {i+1}. {X_samples[i]}")

        # Evaluate samples
        losses = func(X_samples)
        print(f"  Losses: {losses}\n")


def main():
    """Run all demonstrations."""
    print("\n")
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 20 + "DISCRETE ENCODING DEMONSTRATION" + " " * 27 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print("\n")

    demo_raw_encoding()
    demo_interval_encoding()
    demo_onehot_encoding()
    demo_encoding_comparison()
    demo_random_sampling()

    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nFor more information, see DISCRETE_ENCODING_GUIDE.md")
    print()


if __name__ == "__main__":
    main()
