"""
Example demonstrating MuJoCo control task integration.

This example shows how to use standard MuJoCo locomotion tasks
(HalfCheetah, Hopper, Walker2d, Ant, Humanoid) for benchmarking
Bayesian Optimization algorithms.

These environments dominate BO research for robotics (2020-2025).
"""

import torch
import numpy as np


def example_halfcheetah():
    """Example using HalfCheetah (most common in BO papers)."""
    print("=" * 60)
    print("Example 1: HalfCheetah-v4 with Linear Controller")
    print("=" * 60)

    try:
        from bomegabench.functions import HalfCheetahLinearFunction

        # Create HalfCheetah task
        func = HalfCheetahLinearFunction(
            num_episodes=3,  # Average over 3 episodes
            max_episode_steps=1000,  # Standard episode length
            seed=42  # For reproducibility
        )

        print(f"Environment: {func.metadata['env_name']}")
        print(f"Controller type: {func.metadata['controller_type']}")
        print(f"Controller dimension: {func.dim}")
        print(f"Observation dim: {func.metadata['obs_dim']}")
        print(f"Action dim: {func.metadata['action_dim']}")

        # Try zero-initialized controller
        X_zero = torch.zeros(1, func.dim)
        Y_zero = func(X_zero)
        print(f"\nZero controller reward: {Y_zero.item():.2f}")

        # Try small random initialization (better for BO starting point)
        X_random = torch.randn(1, func.dim) * 0.1
        Y_random = func(X_random)
        print(f"Random controller reward: {Y_random.item():.2f}")

    except ImportError as e:
        print(f"Skipped: {e}")
        print("Install with: pip install 'gymnasium[mujoco]'")


def example_multiple_environments():
    """Example using multiple standard environments."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Standard MuJoCo Environments")
    print("=" * 60)

    try:
        from bomegabench.functions import (
            HalfCheetahLinearFunction,
            HopperLinearFunction,
            Walker2dLinearFunction,
            AntLinearFunction,
        )

        environments = [
            ("HalfCheetah", HalfCheetahLinearFunction),
            ("Hopper", HopperLinearFunction),
            ("Walker2d", Walker2dLinearFunction),
            ("Ant", AntLinearFunction),
        ]

        results = {}

        for env_name, env_class in environments:
            print(f"\n{env_name}:")

            func = env_class(num_episodes=2, max_episode_steps=500, seed=42)

            print(f"  Controller dim: {func.dim}")
            print(f"  Obs dim: {func.metadata['obs_dim']}")
            print(f"  Action dim: {func.metadata['action_dim']}")

            # Evaluate zero controller
            X = torch.zeros(1, func.dim)
            Y = func(X)

            results[env_name] = Y.item()
            print(f"  Zero controller reward: {Y.item():.2f}")

        print("\n" + "-" * 60)
        print("Summary:")
        for env_name, reward in results.items():
            print(f"  {env_name:15s}: {reward:8.2f}")

    except ImportError as e:
        print(f"Skipped: {e}")


def example_linear_vs_mlp():
    """Example comparing linear and MLP controllers."""
    print("\n" + "=" * 60)
    print("Example 3: Linear vs MLP Controllers")
    print("=" * 60)

    try:
        from bomegabench.functions import MuJoCoControlWrapper

        # Linear controller (simpler, fewer parameters)
        func_linear = MuJoCoControlWrapper(
            env_name="Hopper-v4",
            controller_type="linear",
            num_episodes=2,
            max_episode_steps=500,
            seed=42
        )

        # MLP controller (more expressive, more parameters)
        func_mlp = MuJoCoControlWrapper(
            env_name="Hopper-v4",
            controller_type="mlp",
            num_episodes=2,
            max_episode_steps=500,
            seed=42
        )

        print("\nLinear Controller:")
        print(f"  Parameters: {func_linear.dim}")
        print(f"  Formula: action = W @ obs + b")

        print("\nMLP Controller:")
        print(f"  Parameters: {func_mlp.dim}")
        print(f"  Architecture: obs_dim -> 32 -> action_dim")

        # Evaluate both
        X_linear = torch.randn(1, func_linear.dim) * 0.1
        X_mlp = torch.randn(1, func_mlp.dim) * 0.1

        Y_linear = func_linear(X_linear)
        Y_mlp = func_mlp(X_mlp)

        print(f"\nLinear controller reward: {Y_linear.item():.2f}")
        print(f"MLP controller reward: {Y_mlp.item():.2f}")

    except ImportError as e:
        print(f"Skipped: {e}")


def example_suite_usage():
    """Example using MuJoCo control suite."""
    print("\n" + "=" * 60)
    print("Example 4: Using MuJoCo Control Suite")
    print("=" * 60)

    try:
        from bomegabench.functions import create_mujoco_control_suite

        # Create suite with v4 environments
        suite = create_mujoco_control_suite(
            controller_type="linear",
            versions=["v4"],  # Use v4 environments
            num_episodes=1
        )

        print(f"Suite name: {suite.name}")
        print(f"Number of tasks: {len(suite)}")
        print(f"Description: {suite.description}")

        print("\nAvailable tasks:")
        for i, task_name in enumerate(suite.list_functions(), 1):
            func = suite.get_function(task_name)
            print(f"  {i}. {task_name}")
            print(f"     - Dimension: {func.dim}")
            print(f"     - Environment: {func.metadata['env_name']}")

    except ImportError as e:
        print(f"Skipped: {e}")


def example_bo_simulation():
    """Example simulating a simple BO loop."""
    print("\n" + "=" * 60)
    print("Example 5: Simple BO Simulation")
    print("=" * 60)

    try:
        from bomegabench.functions import HopperLinearFunction

        # Create task
        func = HopperLinearFunction(
            num_episodes=3,
            max_episode_steps=500,
            seed=42
        )

        print(f"Running simple BO on {func.metadata['env_name']}")
        print(f"Controller dimension: {func.dim}")

        # Random search baseline
        n_iterations = 5
        best_reward = -np.inf
        best_params = None

        print("\nRandom search iterations:")
        for i in range(n_iterations):
            # Sample random controller (small initialization)
            X = torch.randn(1, func.dim) * 0.1

            # Evaluate
            Y = func(X)
            reward = Y.item()

            print(f"  Iteration {i+1}: reward = {reward:.2f}")

            if reward > best_reward:
                best_reward = reward
                best_params = X
                print(f"    -> New best!")

        print(f"\nBest reward found: {best_reward:.2f}")

    except ImportError as e:
        print(f"Skipped: {e}")


def example_batch_evaluation():
    """Example showing batch evaluation for parallel BO."""
    print("\n" + "=" * 60)
    print("Example 6: Batch Evaluation (Parallel BO)")
    print("=" * 60)

    try:
        from bomegabench.functions import HalfCheetahLinearFunction

        func = HalfCheetahLinearFunction(
            num_episodes=2,
            max_episode_steps=300,
            seed=42
        )

        # Evaluate batch of controllers
        batch_size = 3
        X_batch = torch.randn(batch_size, func.dim) * 0.1

        print(f"Evaluating {batch_size} controllers in batch...")
        Y_batch = func(X_batch)

        print("\nResults:")
        for i, reward in enumerate(Y_batch):
            print(f"  Controller {i+1}: {reward.item():.2f}")

        print(f"\nBest in batch: {Y_batch.max().item():.2f}")
        print(f"Mean reward: {Y_batch.mean().item():.2f}")

    except ImportError as e:
        print(f"Skipped: {e}")


if __name__ == "__main__":
    # Run all examples
    example_halfcheetah()
    example_multiple_environments()
    example_linear_vs_mlp()
    example_suite_usage()
    example_bo_simulation()
    example_batch_evaluation()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nNote: If examples were skipped, install MuJoCo:")
    print("  pip install 'gymnasium[mujoco]'")
