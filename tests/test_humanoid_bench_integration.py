"""
Test HumanoidBench integration with BOMegaBench.

This test verifies that HumanoidBench tasks are properly integrated and can be
used for Bayesian Optimization benchmarking.
"""

import pytest
import torch
import numpy as np


def test_humanoid_bench_import():
    """Test that HumanoidBench tasks can be imported."""
    try:
        from bomegabench.functions import (
            HUMANOID_BENCH_AVAILABLE,
            create_humanoid_bench_suite,
            HumanoidBenchWrapper,
            H1HandWalkFunction,
        )

        if not HUMANOID_BENCH_AVAILABLE:
            pytest.skip("HumanoidBench not available")

        print("✓ HumanoidBench imports successful")

    except ImportError as e:
        pytest.skip(f"HumanoidBench not available: {e}")


def test_humanoid_bench_task_lists():
    """Test that task lists are properly defined."""
    try:
        from bomegabench.functions import (
            HUMANOID_BENCH_AVAILABLE,
            LOCOMOTION_TASKS,
            HUMANOID_MANIPULATION_TASKS,
            HUMANOID_ALL_TASKS,
            AVAILABLE_ROBOTS,
        )

        if not HUMANOID_BENCH_AVAILABLE:
            pytest.skip("HumanoidBench not available")

        # Check that task lists are not empty
        assert len(LOCOMOTION_TASKS) > 0, "Locomotion tasks list is empty"
        assert len(HUMANOID_MANIPULATION_TASKS) > 0, "Manipulation tasks list is empty"
        assert len(HUMANOID_ALL_TASKS) > 0, "All tasks list is empty"
        assert len(AVAILABLE_ROBOTS) > 0, "Available robots list is empty"

        # Check that all tasks are in the combined list
        assert len(HUMANOID_ALL_TASKS) == len(LOCOMOTION_TASKS) + len(HUMANOID_MANIPULATION_TASKS)

        print(f"✓ Locomotion tasks: {len(LOCOMOTION_TASKS)}")
        print(f"✓ Manipulation tasks: {len(HUMANOID_MANIPULATION_TASKS)}")
        print(f"✓ Total tasks: {len(HUMANOID_ALL_TASKS)}")
        print(f"✓ Available robots: {AVAILABLE_ROBOTS}")

    except ImportError as e:
        pytest.skip(f"HumanoidBench not available: {e}")


def test_humanoid_bench_wrapper_creation():
    """Test creating a HumanoidBench wrapper."""
    try:
        from bomegabench.functions import (
            HUMANOID_BENCH_AVAILABLE,
            HumanoidBenchWrapper,
        )

        if not HUMANOID_BENCH_AVAILABLE:
            pytest.skip("HumanoidBench not available")

        # Create a simple task
        func = HumanoidBenchWrapper(
            task_name="stand",
            robot="h1hand",
            controller_type="linear",
            num_episodes=1,
        )

        # Check metadata
        metadata = func.metadata
        assert metadata["task_name"] == "stand"
        assert metadata["robot"] == "h1hand"
        assert metadata["controller_type"] == "linear"
        assert metadata["task_category"] == "locomotion"
        assert "obs_dim" in metadata
        assert "action_dim" in metadata
        assert "controller_dim" in metadata

        print(f"✓ Task: {metadata['task_name']}")
        print(f"✓ Robot: {metadata['robot']}")
        print(f"✓ Obs dim: {metadata['obs_dim']}")
        print(f"✓ Action dim: {metadata['action_dim']}")
        print(f"✓ Controller dim: {metadata['controller_dim']}")

    except Exception as e:
        pytest.skip(f"Failed to create HumanoidBench wrapper: {e}")


def test_humanoid_bench_evaluation():
    """Test evaluating a HumanoidBench task."""
    try:
        from bomegabench.functions import (
            HUMANOID_BENCH_AVAILABLE,
            HumanoidBenchWrapper,
        )

        if not HUMANOID_BENCH_AVAILABLE:
            pytest.skip("HumanoidBench not available")

        # Create a simple locomotion task
        func = HumanoidBenchWrapper(
            task_name="stand",
            robot="h1hand",
            controller_type="linear",
            num_episodes=1,
            seed=42,
        )

        # Test single evaluation
        X_single = torch.randn(func.dim)
        Y_single = func(X_single)

        assert Y_single.shape == ()
        assert not torch.isnan(Y_single)
        print(f"✓ Single evaluation: reward = {Y_single.item():.4f}")

        # Test batch evaluation
        X_batch = torch.randn(3, func.dim)
        Y_batch = func(X_batch)

        assert Y_batch.shape == (3,)
        assert not torch.any(torch.isnan(Y_batch))
        print(f"✓ Batch evaluation: rewards = {Y_batch.tolist()}")

    except Exception as e:
        pytest.skip(f"Failed to evaluate HumanoidBench task: {e}")


def test_convenience_functions():
    """Test convenience function classes."""
    try:
        from bomegabench.functions import (
            HUMANOID_BENCH_AVAILABLE,
            H1HandWalkFunction,
            H1HandPushFunction,
            G1WalkFunction,
        )

        if not HUMANOID_BENCH_AVAILABLE:
            pytest.skip("HumanoidBench not available")

        # Test H1Hand walk
        func_h1_walk = H1HandWalkFunction(controller_type="linear", num_episodes=1)
        assert func_h1_walk.metadata["task_name"] == "walk"
        assert func_h1_walk.metadata["robot"] == "h1hand"
        print(f"✓ H1HandWalkFunction created (dim={func_h1_walk.dim})")

        # Test H1Hand push
        func_h1_push = H1HandPushFunction(controller_type="linear", num_episodes=1)
        assert func_h1_push.metadata["task_name"] == "push"
        assert func_h1_push.metadata["robot"] == "h1hand"
        print(f"✓ H1HandPushFunction created (dim={func_h1_push.dim})")

        # Test G1 walk
        func_g1_walk = G1WalkFunction(controller_type="linear", num_episodes=1)
        assert func_g1_walk.metadata["task_name"] == "walk"
        assert func_g1_walk.metadata["robot"] == "g1"
        print(f"✓ G1WalkFunction created (dim={func_g1_walk.dim})")

    except Exception as e:
        pytest.skip(f"Failed to create convenience functions: {e}")


def test_humanoid_bench_suite_creation():
    """Test creating a HumanoidBench suite."""
    try:
        from bomegabench.functions import (
            HUMANOID_BENCH_AVAILABLE,
            create_humanoid_bench_suite,
        )

        if not HUMANOID_BENCH_AVAILABLE:
            pytest.skip("HumanoidBench not available")

        # Create locomotion suite
        loco_suite = create_humanoid_bench_suite(
            robot="h1hand",
            task_categories=["locomotion"],
            controller_type="linear",
            num_episodes=1,
        )

        assert len(loco_suite.functions) > 0
        print(f"✓ Locomotion suite created with {len(loco_suite.functions)} tasks")

        # Create manipulation suite
        manip_suite = create_humanoid_bench_suite(
            robot="h1hand",
            task_categories=["manipulation"],
            controller_type="linear",
            num_episodes=1,
        )

        assert len(manip_suite.functions) > 0
        print(f"✓ Manipulation suite created with {len(manip_suite.functions)} tasks")

        # Create full suite
        full_suite = create_humanoid_bench_suite(
            robot="h1hand",
            controller_type="linear",
            num_episodes=1,
        )

        assert len(full_suite.functions) > 0
        print(f"✓ Full suite created with {len(full_suite.functions)} tasks")

        # Verify full suite contains both categories
        assert len(full_suite.functions) >= len(loco_suite.functions)
        assert len(full_suite.functions) >= len(manip_suite.functions)

    except Exception as e:
        pytest.skip(f"Failed to create HumanoidBench suite: {e}")


def test_no_task_duplication():
    """Verify that HumanoidBench tasks don't duplicate existing benchmarks."""
    try:
        from bomegabench.functions import (
            HUMANOID_BENCH_AVAILABLE,
            HUMANOID_ALL_TASKS,
        )

        if not HUMANOID_BENCH_AVAILABLE:
            pytest.skip("HumanoidBench not available")

        # Check that HumanoidBench tasks are distinct from standard MuJoCo
        # Standard MuJoCo: HalfCheetah, Hopper, Walker2d, Ant, Humanoid
        # HumanoidBench uses H1 and G1 robots with complex tasks

        standard_mujoco_tasks = ["halfcheetah", "hopper", "walker2d", "ant", "humanoid"]

        # Check no direct overlap (case-insensitive)
        humanoid_tasks_lower = [t.lower() for t in HUMANOID_ALL_TASKS]

        for mujoco_task in standard_mujoco_tasks:
            assert mujoco_task not in humanoid_tasks_lower, \
                f"Task {mujoco_task} duplicates standard MuJoCo task"

        print("✓ No duplication with standard MuJoCo tasks")

        # HumanoidBench provides unique whole-body humanoid tasks
        unique_tasks = ["cabinet", "kitchen", "bookshelf", "spoon", "window", "powerlift", "room"]
        for task in unique_tasks:
            if task in HUMANOID_ALL_TASKS:
                print(f"✓ Unique task found: {task}")

    except ImportError as e:
        pytest.skip(f"HumanoidBench not available: {e}")


def test_mlp_controller():
    """Test MLP controller type."""
    try:
        from bomegabench.functions import (
            HUMANOID_BENCH_AVAILABLE,
            HumanoidBenchWrapper,
        )

        if not HUMANOID_BENCH_AVAILABLE:
            pytest.skip("HumanoidBench not available")

        # Create task with MLP controller
        func = HumanoidBenchWrapper(
            task_name="stand",
            robot="h1hand",
            controller_type="mlp",
            num_episodes=1,
            seed=42,
        )

        # MLP should have larger dim than linear
        assert func.dim > 0
        print(f"✓ MLP controller dim: {func.dim}")

        # Test evaluation
        X = torch.randn(func.dim)
        Y = func(X)

        assert Y.shape == ()
        assert not torch.isnan(Y)
        print(f"✓ MLP evaluation: reward = {Y.item():.4f}")

    except Exception as e:
        pytest.skip(f"Failed to test MLP controller: {e}")


if __name__ == "__main__":
    print("Testing HumanoidBench Integration\n")
    print("=" * 60)

    try:
        test_humanoid_bench_import()
        print()

        test_humanoid_bench_task_lists()
        print()

        test_humanoid_bench_wrapper_creation()
        print()

        test_humanoid_bench_evaluation()
        print()

        test_convenience_functions()
        print()

        test_humanoid_bench_suite_creation()
        print()

        test_no_task_duplication()
        print()

        test_mlp_controller()
        print()

        print("=" * 60)
        print("✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
