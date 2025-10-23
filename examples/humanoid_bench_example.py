"""
HumanoidBench Integration Example.

This example demonstrates how to use HumanoidBench tasks with BOMegaBench
for Bayesian Optimization of humanoid robot controllers.

HumanoidBench provides whole-body humanoid locomotion and manipulation tasks
that are ideal for BO-based controller tuning.
"""

import torch
import numpy as np


def example_single_task():
    """Example: Single HumanoidBench task."""
    print("=" * 60)
    print("Example 1: Single HumanoidBench Task")
    print("=" * 60)

    from bomegabench.functions import H1HandWalkFunction

    # Create H1 hand walk task with linear controller
    func = H1HandWalkFunction(
        controller_type="linear",
        num_episodes=1,  # Average over 1 episode for faster evaluation
        seed=42,
    )

    print(f"\nTask: {func.metadata['name']}")
    print(f"Robot: {func.metadata['robot']}")
    print(f"Task category: {func.metadata['task_category']}")
    print(f"Controller dim: {func.dim}")
    print(f"Observation dim: {func.metadata['obs_dim']}")
    print(f"Action dim: {func.metadata['action_dim']}")

    # Evaluate a random controller
    X = torch.randn(func.dim)
    Y = func(X)

    print(f"\nRandom controller reward: {Y.item():.4f}")

    # Evaluate multiple controllers
    X_batch = torch.randn(5, func.dim)
    Y_batch = func(X_batch)

    print(f"\nBatch evaluation:")
    for i, reward in enumerate(Y_batch):
        print(f"  Controller {i+1}: {reward.item():.4f}")


def example_task_wrapper():
    """Example: Using HumanoidBenchWrapper directly."""
    print("\n" + "=" * 60)
    print("Example 2: HumanoidBenchWrapper")
    print("=" * 60)

    from bomegabench.functions import HumanoidBenchWrapper

    # Create a manipulation task
    func = HumanoidBenchWrapper(
        task_name="push",
        robot="h1hand",
        controller_type="linear",
        num_episodes=1,
        seed=42,
    )

    print(f"\nTask: {func.metadata['task_name']}")
    print(f"Category: {func.metadata['task_category']}")
    print(f"Controller dim: {func.dim}")

    # Test evaluation
    X = torch.randn(func.dim)
    Y = func(X)

    print(f"\nReward: {Y.item():.4f}")


def example_mlp_controller():
    """Example: Using MLP controller."""
    print("\n" + "=" * 60)
    print("Example 3: MLP Controller")
    print("=" * 60)

    from bomegabench.functions import HumanoidBenchWrapper

    # Create task with MLP controller (larger parameter space)
    func = HumanoidBenchWrapper(
        task_name="cabinet",
        robot="h1hand",
        controller_type="mlp",
        num_episodes=1,
        seed=42,
    )

    print(f"\nTask: {func.metadata['task_name']}")
    print(f"Controller type: {func.metadata['controller_type']}")
    print(f"Controller dim: {func.dim}")
    print(f"  (Linear would be: {func.metadata['obs_dim'] * func.metadata['action_dim'] + func.metadata['action_dim']})")

    # Evaluate
    X = torch.randn(func.dim)
    Y = func(X)

    print(f"\nMLP controller reward: {Y.item():.4f}")


def example_locomotion_suite():
    """Example: Creating a locomotion suite."""
    print("\n" + "=" * 60)
    print("Example 4: Locomotion Suite")
    print("=" * 60)

    from bomegabench.functions import create_humanoid_bench_suite

    # Create suite with only locomotion tasks
    suite = create_humanoid_bench_suite(
        robot="h1hand",
        task_categories=["locomotion"],
        controller_type="linear",
        num_episodes=1,
    )

    print(f"\nLocomotion suite created")
    print(f"Number of tasks: {len(suite.functions)}")
    print(f"\nAvailable tasks:")
    for task_name in list(suite.functions.keys())[:5]:
        print(f"  - {task_name}")
    if len(suite.functions) > 5:
        print(f"  ... and {len(suite.functions) - 5} more")

    # Evaluate first task
    first_task_name = list(suite.functions.keys())[0]
    first_task = suite.functions[first_task_name]

    X = torch.randn(first_task.dim)
    Y = first_task(X)

    print(f"\n{first_task_name} reward: {Y.item():.4f}")


def example_manipulation_suite():
    """Example: Creating a manipulation suite."""
    print("\n" + "=" * 60)
    print("Example 5: Manipulation Suite")
    print("=" * 60)

    from bomegabench.functions import create_humanoid_bench_suite

    # Create suite with only manipulation tasks
    suite = create_humanoid_bench_suite(
        robot="h1hand",
        task_categories=["manipulation"],
        controller_type="linear",
        num_episodes=1,
    )

    print(f"\nManipulation suite created")
    print(f"Number of tasks: {len(suite.functions)}")
    print(f"\nAvailable manipulation tasks:")
    for task_name in list(suite.functions.keys())[:8]:
        print(f"  - {task_name}")
    if len(suite.functions) > 8:
        print(f"  ... and {len(suite.functions) - 8} more")


def example_g1_robot():
    """Example: Using G1 robot."""
    print("\n" + "=" * 60)
    print("Example 6: G1 Robot Tasks")
    print("=" * 60)

    from bomegabench.functions import G1WalkFunction, G1PushFunction

    # G1 walk task
    walk_func = G1WalkFunction(
        controller_type="linear",
        num_episodes=1,
        seed=42,
    )

    print(f"\nG1 Walk Task")
    print(f"Robot: {walk_func.metadata['robot']}")
    print(f"Controller dim: {walk_func.dim}")

    X = torch.randn(walk_func.dim)
    Y = walk_func(X)
    print(f"Reward: {Y.item():.4f}")

    # G1 push task
    push_func = G1PushFunction(
        controller_type="linear",
        num_episodes=1,
        seed=42,
    )

    print(f"\nG1 Push Task")
    print(f"Task category: {push_func.metadata['task_category']}")
    print(f"Controller dim: {push_func.dim}")

    X = torch.randn(push_func.dim)
    Y = push_func(X)
    print(f"Reward: {Y.item():.4f}")


def example_task_categories():
    """Example: Listing available tasks and robots."""
    print("\n" + "=" * 60)
    print("Example 7: Available Tasks and Robots")
    print("=" * 60)

    from bomegabench.functions import (
        LOCOMOTION_TASKS,
        HUMANOID_MANIPULATION_TASKS,
        HUMANOID_ALL_TASKS,
        AVAILABLE_ROBOTS,
    )

    print(f"\nAvailable Robots:")
    for robot in AVAILABLE_ROBOTS:
        print(f"  - {robot}")

    print(f"\nLocomotion Tasks ({len(LOCOMOTION_TASKS)}):")
    for task in LOCOMOTION_TASKS:
        print(f"  - {task}")

    print(f"\nManipulation Tasks ({len(HUMANOID_MANIPULATION_TASKS)}):")
    for task in HUMANOID_MANIPULATION_TASKS[:10]:
        print(f"  - {task}")
    if len(HUMANOID_MANIPULATION_TASKS) > 10:
        print(f"  ... and {len(HUMANOID_MANIPULATION_TASKS) - 10} more")

    print(f"\nTotal Tasks: {len(HUMANOID_ALL_TASKS)}")


def example_bo_integration():
    """Example: Integration with Bayesian Optimization."""
    print("\n" + "=" * 60)
    print("Example 8: BO Integration (Conceptual)")
    print("=" * 60)

    from bomegabench.functions import H1HandCabinetFunction

    # Create a challenging manipulation task
    func = H1HandCabinetFunction(
        controller_type="linear",
        num_episodes=3,  # Average over 3 episodes for more stable evaluation
        seed=42,
    )

    print(f"\nTask: {func.metadata['name']}")
    print(f"Optimization goal: {func.metadata['optimization_goal']}")
    print(f"Search space dimension: {func.dim}")
    print(f"Bounds: [{func.bounds[0][0].item()}, {func.bounds[1][0].item()}]")

    print("\nThis function can be used with any BO library:")
    print("  - BoTorch")
    print("  - GPyOpt")
    print("  - Ax")
    print("  - etc.")

    # Example evaluation
    print(f"\nExample evaluation:")
    X_init = torch.randn(5, func.dim) * 0.1  # Small random initialization
    Y_init = torch.stack([func(x) for x in X_init])

    print(f"Initial controllers evaluated: {len(Y_init)}")
    print(f"Best reward: {Y_init.max().item():.4f}")
    print(f"Worst reward: {Y_init.min().item():.4f}")
    print(f"Mean reward: {Y_init.mean().item():.4f}")


if __name__ == "__main__":
    print("\nHumanoidBench Integration Examples")
    print("=" * 60)

    try:
        from bomegabench.functions import HUMANOID_BENCH_AVAILABLE

        if not HUMANOID_BENCH_AVAILABLE:
            print("\n⚠ HumanoidBench is not available.")
            print("Please ensure humanoid-bench is installed in the repository.")
            exit(1)

        # Run all examples
        example_single_task()
        example_task_wrapper()
        example_mlp_controller()
        example_locomotion_suite()
        example_manipulation_suite()
        example_g1_robot()
        example_task_categories()
        example_bo_integration()

        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)

    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nMake sure humanoid-bench is installed:")
        print("  cd humanoid-bench && pip install -e .")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
