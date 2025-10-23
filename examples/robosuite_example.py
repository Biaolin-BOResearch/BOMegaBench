"""
Example usage of Robosuite Manipulation tasks in BOMegaBench.

This example demonstrates how to use robosuite manipulation tasks
for controller parameter optimization with Bayesian Optimization.
"""

import torch
import numpy as np
from bomegabench.functions import (
    LiftLinearFunction,
    DoorLinearFunction,
    create_robosuite_manipulation_suite,
    ROBOSUITE_AVAILABLE
)


def basic_example():
    """Basic usage of a single robosuite task."""
    if not ROBOSUITE_AVAILABLE:
        print("Robosuite is not available. Please check installation.")
        return

    print("=" * 60)
    print("Basic Robosuite Example: Lift Task")
    print("=" * 60)

    # Create the Lift task with linear controller
    func = LiftLinearFunction(
        num_episodes=1,  # Number of episodes to average over
        horizon=200,     # Shorter horizon for faster evaluation
    )

    print(f"\nTask: {func.metadata['name']}")
    print(f"Controller type: {func.metadata['controller_type']}")
    print(f"Observation dim: {func.metadata['obs_dim']}")
    print(f"Action dim: {func.metadata['action_dim']}")
    print(f"Controller param dim: {func.dim}")
    print(f"Bounds: [{func.bounds[0, 0].item():.1f}, {func.bounds[1, 0].item():.1f}]")

    # Random controller parameters
    X = torch.randn(1, func.dim) * 0.1  # Small random values
    print(f"\nEvaluating random controller...")

    # Evaluate
    Y = func(X)
    print(f"Reward: {Y.item():.4f}")


def suite_example():
    """Example using a suite of manipulation tasks."""
    if not ROBOSUITE_AVAILABLE:
        print("Robosuite is not available. Please check installation.")
        return

    print("\n" + "=" * 60)
    print("Robosuite Suite Example")
    print("=" * 60)

    # Create a suite of manipulation tasks
    suite = create_robosuite_manipulation_suite(
        controller_type="linear",
        tasks=["Lift", "Door"],  # Subset for quick demo
        horizon=100,
        num_episodes=1,
    )

    print(f"\nSuite: {suite.name}")
    print(f"Number of tasks: {len(suite.functions)}")

    for i, func in enumerate(suite.functions):
        print(f"\n[{i+1}] {func.metadata['name']}")
        print(f"    Dim: {func.dim}")
        print(f"    Env: {func.metadata['env_name']}")


def bayesian_optimization_example():
    """Example of using BO with robosuite tasks."""
    if not ROBOSUITE_AVAILABLE:
        print("Robosuite is not available. Please check installation.")
        return

    print("\n" + "=" * 60)
    print("Bayesian Optimization Example (Mock)")
    print("=" * 60)

    # Create Lift task
    func = LiftLinearFunction(horizon=100, num_episodes=1)

    print(f"\nOptimizing controller for: {func.metadata['env_name']}")
    print(f"Search space dimension: {func.dim}")

    # Mock BO loop (replace with actual BO library)
    print("\nMock BO iterations:")
    best_reward = -float('inf')

    for iteration in range(3):
        # Sample random point (in real BO, this would be from acquisition)
        X = torch.randn(1, func.dim) * 0.1

        # Evaluate
        Y = func(X)
        reward = Y.item()

        if reward > best_reward:
            best_reward = reward
            print(f"[{iteration+1}] New best reward: {best_reward:.4f}")
        else:
            print(f"[{iteration+1}] Reward: {reward:.4f}")

    print(f"\nFinal best reward: {best_reward:.4f}")


def compare_controller_types():
    """Compare linear and MLP controllers."""
    if not ROBOSUITE_AVAILABLE:
        print("Robosuite is not available. Please check installation.")
        return

    print("\n" + "=" * 60)
    print("Controller Type Comparison")
    print("=" * 60)

    from bomegabench.functions import RobosuiteManipulationWrapper

    for controller_type in ["linear", "mlp"]:
        func = RobosuiteManipulationWrapper(
            env_name="Lift",
            controller_type=controller_type,
            horizon=100,
            num_episodes=1,
        )

        print(f"\n{controller_type.upper()} Controller:")
        print(f"  Parameters: {func.dim}")
        print(f"  Obs dim: {func.metadata['obs_dim']}")
        print(f"  Action dim: {func.metadata['action_dim']}")


def main():
    """Run all examples."""
    try:
        basic_example()
        suite_example()
        compare_controller_types()
        bayesian_optimization_example()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
