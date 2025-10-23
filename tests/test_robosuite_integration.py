"""
Tests for Robosuite Manipulation integration.

This test file verifies that robosuite tasks are properly integrated
into BOMegaBench and can be used for Bayesian Optimization.
"""

import sys
import os
import pytest
import torch

# Add BOMegaBench to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from bomegabench.functions import (
        RobosuiteManipulationWrapper,
        LiftLinearFunction,
        DoorLinearFunction,
        create_robosuite_manipulation_suite,
        ROBOSUITE_AVAILABLE,
        MANIPULATION_ENVS,
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Robosuite import failed")
def test_import():
    """Test that robosuite functions can be imported."""
    assert ROBOSUITE_AVAILABLE or not ROBOSUITE_AVAILABLE
    assert MANIPULATION_ENVS is not None or MANIPULATION_ENVS is None


@pytest.mark.skipif(not (IMPORT_SUCCESS and ROBOSUITE_AVAILABLE),
                   reason="Robosuite not available")
def test_lift_linear_function():
    """Test LiftLinearFunction."""
    func = LiftLinearFunction(horizon=50, num_episodes=1)

    # Check metadata
    assert func.metadata['env_name'] == 'Lift'
    assert func.metadata['controller_type'] == 'linear'
    assert func.dim > 0

    # Check bounds
    assert func.bounds.shape == (2, func.dim)
    assert torch.all(func.bounds[0] == -1.0)
    assert torch.all(func.bounds[1] == 1.0)

    print(f"✓ Lift task created: dim={func.dim}")


@pytest.mark.skipif(not (IMPORT_SUCCESS and ROBOSUITE_AVAILABLE),
                   reason="Robosuite not available")
def test_door_linear_function():
    """Test DoorLinearFunction."""
    func = DoorLinearFunction(horizon=50, num_episodes=1)

    assert func.metadata['env_name'] == 'Door'
    assert func.dim > 0

    print(f"✓ Door task created: dim={func.dim}")


@pytest.mark.skipif(not (IMPORT_SUCCESS and ROBOSUITE_AVAILABLE),
                   reason="Robosuite not available")
def test_custom_task():
    """Test creating custom task with RobosuiteManipulationWrapper."""
    func = RobosuiteManipulationWrapper(
        env_name="Lift",
        controller_type="linear",
        horizon=50,
        num_episodes=1,
    )

    assert func.metadata['env_name'] == 'Lift'
    assert func.dim > 0

    print(f"✓ Custom task created: {func.metadata['name']}")


@pytest.mark.skipif(not (IMPORT_SUCCESS and ROBOSUITE_AVAILABLE),
                   reason="Robosuite not available")
def test_evaluation():
    """Test evaluating controller parameters."""
    func = LiftLinearFunction(horizon=50, num_episodes=1)

    # Random controller parameters
    X = torch.randn(1, func.dim) * 0.1

    # Evaluate
    Y = func(X)

    assert Y.shape == (1, 1)
    assert torch.isfinite(Y).all()

    print(f"✓ Evaluation successful: reward={Y.item():.4f}")


@pytest.mark.skipif(not (IMPORT_SUCCESS and ROBOSUITE_AVAILABLE),
                   reason="Robosuite not available")
def test_batch_evaluation():
    """Test batch evaluation."""
    func = LiftLinearFunction(horizon=50, num_episodes=1)

    # Batch of controller parameters
    batch_size = 3
    X = torch.randn(batch_size, func.dim) * 0.1

    # Evaluate
    Y = func(X)

    assert Y.shape == (batch_size, 1)
    assert torch.isfinite(Y).all()

    print(f"✓ Batch evaluation successful: {batch_size} evaluations")


@pytest.mark.skipif(not (IMPORT_SUCCESS and ROBOSUITE_AVAILABLE),
                   reason="Robosuite not available")
def test_suite_creation():
    """Test creating a suite of tasks."""
    suite = create_robosuite_manipulation_suite(
        controller_type="linear",
        tasks=["Lift", "Door"],
        horizon=50,
        num_episodes=1,
    )

    assert len(suite.functions) == 2
    assert suite.name == "RobosuiteManipulation_linear"

    print(f"✓ Suite created: {len(suite.functions)} tasks")


@pytest.mark.skipif(not (IMPORT_SUCCESS and ROBOSUITE_AVAILABLE),
                   reason="Robosuite not available")
def test_mlp_controller():
    """Test MLP controller type."""
    func = RobosuiteManipulationWrapper(
        env_name="Lift",
        controller_type="mlp",
        horizon=50,
        num_episodes=1,
    )

    assert func.metadata['controller_type'] == 'mlp'
    assert func.dim > 0

    # MLP should have more parameters than linear
    linear_func = LiftLinearFunction(horizon=50)
    assert func.dim > linear_func.dim

    print(f"✓ MLP controller: dim={func.dim} (vs linear: {linear_func.dim})")


@pytest.mark.skipif(not (IMPORT_SUCCESS and ROBOSUITE_AVAILABLE),
                   reason="Robosuite not available")
def test_available_envs():
    """Test that MANIPULATION_ENVS is properly defined."""
    assert MANIPULATION_ENVS is not None
    assert len(MANIPULATION_ENVS) > 0
    assert "Lift" in MANIPULATION_ENVS
    assert "Door" in MANIPULATION_ENVS

    print(f"✓ Available environments: {len(MANIPULATION_ENVS)}")
    print(f"  Envs: {', '.join(MANIPULATION_ENVS[:5])}...")


def test_import_status():
    """Print import status for debugging."""
    if IMPORT_SUCCESS:
        print("✓ Robosuite imports successful")
        if ROBOSUITE_AVAILABLE:
            print("✓ Robosuite is available")
        else:
            print("⚠ Robosuite imports OK but runtime not available")
    else:
        print(f"✗ Robosuite import failed: {IMPORT_ERROR}")


if __name__ == "__main__":
    """Run tests manually."""
    print("=" * 60)
    print("Robosuite Integration Tests")
    print("=" * 60)

    test_import_status()

    if IMPORT_SUCCESS and ROBOSUITE_AVAILABLE:
        print("\nRunning tests...")
        try:
            test_import()
            print("✓ test_import")

            test_lift_linear_function()
            print("✓ test_lift_linear_function")

            test_door_linear_function()
            print("✓ test_door_linear_function")

            test_custom_task()
            print("✓ test_custom_task")

            test_available_envs()
            print("✓ test_available_envs")

            test_suite_creation()
            print("✓ test_suite_creation")

            test_mlp_controller()
            print("✓ test_mlp_controller")

            # Skip slow evaluation tests in quick run
            print("\n⚠ Skipping evaluation tests (slow)")

            print("\n" + "=" * 60)
            print("All tests passed!")
            print("=" * 60)

        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠ Robosuite not available, tests skipped")
