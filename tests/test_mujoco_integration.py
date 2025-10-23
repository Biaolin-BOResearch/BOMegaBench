"""
Test MuJoCo control integration with BOMegaBench.

These tests validate the integration of standard MuJoCo locomotion tasks
(HalfCheetah, Hopper, Walker2d, Ant, Humanoid) which are the de facto
standard benchmarks for BO in robotics (2020-2025).
"""

import pytest
import torch
import numpy as np


def test_mujoco_import():
    """Test that MuJoCo wrapper can be imported."""
    try:
        from bomegabench.functions import (
            MuJoCoControlWrapper,
            create_mujoco_control_suite,
        )
        assert MuJoCoControlWrapper is not None
        assert create_mujoco_control_suite is not None
    except ImportError as e:
        pytest.skip(f"MuJoCo not available: {e}")


def test_halfcheetah_creation():
    """Test creation of HalfCheetah task (most common in BO papers)."""
    try:
        from bomegabench.functions import HalfCheetahLinearFunction

        func = HalfCheetahLinearFunction(num_episodes=1)

        # Check basic properties
        assert func.dim > 0
        assert func.bounds.shape == (2, func.dim)
        assert func.metadata["type"] == "continuous"
        assert "HalfCheetah" in func.metadata["env_name"]
        assert func.metadata["controller_type"] == "linear"

        print(f"HalfCheetah controller dim: {func.dim}")
        print(f"Obs dim: {func.metadata['obs_dim']}, Action dim: {func.metadata['action_dim']}")

    except ImportError as e:
        pytest.skip(f"MuJoCo not available: {e}")


def test_hopper_creation():
    """Test creation of Hopper task (very common in BO papers)."""
    try:
        from bomegabench.functions import HopperLinearFunction

        func = HopperLinearFunction(num_episodes=1)

        # Check basic properties
        assert func.dim > 0
        assert func.bounds.shape == (2, func.dim)
        assert "Hopper" in func.metadata["env_name"]

    except ImportError as e:
        pytest.skip(f"MuJoCo not available: {e}")


def test_controller_evaluation():
    """Test that controller can be evaluated."""
    try:
        from bomegabench.functions import HopperLinearFunction

        func = HopperLinearFunction(num_episodes=1, max_episode_steps=100)

        # Create a simple controller (random initialization)
        X = torch.zeros(1, func.dim)  # Zero initialization

        # Evaluate
        Y = func(X)

        assert isinstance(Y, torch.Tensor)
        assert Y.shape == (1,) or Y.shape == ()
        assert torch.isfinite(Y).all()

        print(f"Zero controller reward: {Y.item():.2f}")

    except ImportError as e:
        pytest.skip(f"MuJoCo not available: {e}")


def test_batch_evaluation():
    """Test batch evaluation of controllers."""
    try:
        from bomegabench.functions import HopperLinearFunction

        func = HopperLinearFunction(num_episodes=1, max_episode_steps=50)

        # Create batch of controllers
        X_batch = torch.randn(3, func.dim) * 0.1  # Small random initialization

        # Evaluate
        Y_batch = func(X_batch)

        assert Y_batch.shape == (3,)
        assert torch.isfinite(Y_batch).all()

        print(f"Batch rewards: {Y_batch}")

    except ImportError as e:
        pytest.skip(f"MuJoCo not available: {e}")


def test_linear_vs_mlp_controller():
    """Test different controller types."""
    try:
        from bomegabench.functions import MuJoCoControlWrapper

        # Linear controller
        func_linear = MuJoCoControlWrapper(
            env_name="Hopper-v4",
            controller_type="linear",
            num_episodes=1,
            max_episode_steps=50
        )

        # MLP controller
        func_mlp = MuJoCoControlWrapper(
            env_name="Hopper-v4",
            controller_type="mlp",
            num_episodes=1,
            max_episode_steps=50
        )

        # MLP should have more parameters
        assert func_mlp.dim > func_linear.dim

        print(f"Linear controller dim: {func_linear.dim}")
        print(f"MLP controller dim: {func_mlp.dim}")

        # Both should be evaluable
        X_linear = torch.zeros(1, func_linear.dim)
        X_mlp = torch.zeros(1, func_mlp.dim)

        Y_linear = func_linear(X_linear)
        Y_mlp = func_mlp(X_mlp)

        assert torch.isfinite(Y_linear).all()
        assert torch.isfinite(Y_mlp).all()

    except ImportError as e:
        pytest.skip(f"MuJoCo not available: {e}")


def test_suite_creation():
    """Test creation of MuJoCo control suite."""
    try:
        from bomegabench.functions import create_mujoco_control_suite

        # Create suite with only v4 environments
        suite = create_mujoco_control_suite(
            controller_type="linear",
            versions=["v4"],
            num_episodes=1
        )

        # Check suite properties
        assert len(suite) > 0
        assert suite.name == "MuJoCoControl"

        # List all tasks
        func_names = suite.list_functions()
        assert len(func_names) > 0

        print(f"\nMuJoCo suite: {len(func_names)} tasks")
        for name in func_names:
            func = suite.get_function(name)
            print(f"  {name}: dim={func.dim}")

    except ImportError as e:
        pytest.skip(f"MuJoCo not available: {e}")


def test_multiple_environments():
    """Test that multiple standard environments can be created."""
    try:
        from bomegabench.functions import (
            HalfCheetahLinearFunction,
            HopperLinearFunction,
            Walker2dLinearFunction,
            AntLinearFunction,
        )

        # Create all standard environments
        envs = {
            "HalfCheetah": HalfCheetahLinearFunction(num_episodes=1, max_episode_steps=50),
            "Hopper": HopperLinearFunction(num_episodes=1, max_episode_steps=50),
            "Walker2d": Walker2dLinearFunction(num_episodes=1, max_episode_steps=50),
            "Ant": AntLinearFunction(num_episodes=1, max_episode_steps=50),
        }

        for name, func in envs.items():
            assert func.dim > 0
            print(f"{name}: dim={func.dim}, obs={func.metadata['obs_dim']}, act={func.metadata['action_dim']}")

    except ImportError as e:
        pytest.skip(f"MuJoCo not available: {e}")


def test_reproducibility_with_seed():
    """Test that seeded evaluations are reproducible."""
    try:
        from bomegabench.functions import HopperLinearFunction

        # Create two identical functions with same seed
        func1 = HopperLinearFunction(num_episodes=1, max_episode_steps=100, seed=42)
        func2 = HopperLinearFunction(num_episodes=1, max_episode_steps=100, seed=42)

        # Same controller
        X = torch.randn(1, func1.dim) * 0.1

        # Evaluate
        Y1 = func1(X)
        Y2 = func2(X)

        # Should be identical with same seed
        assert torch.allclose(Y1, Y2, atol=1e-5)

    except ImportError as e:
        pytest.skip(f"MuJoCo not available: {e}")


def test_metadata():
    """Test that metadata is properly populated."""
    try:
        from bomegabench.functions import HalfCheetahLinearFunction

        func = HalfCheetahLinearFunction()
        metadata = func.metadata

        # Check required metadata fields
        assert "name" in metadata
        assert "source" in metadata
        assert "type" in metadata
        assert "env_name" in metadata
        assert "controller_type" in metadata
        assert "obs_dim" in metadata
        assert "action_dim" in metadata
        assert "controller_dim" in metadata

        # Check values
        assert metadata["source"] == "Gymnasium MuJoCo"
        assert metadata["type"] == "continuous"
        assert "HalfCheetah" in metadata["env_name"]
        assert metadata["optimization_goal"] == "maximize_reward"

    except ImportError as e:
        pytest.skip(f"MuJoCo not available: {e}")


def test_bounds_validity():
    """Test that bounds are valid."""
    try:
        from bomegabench.functions import HalfCheetahLinearFunction

        func = HalfCheetahLinearFunction()

        # Bounds should have correct shape
        assert func.bounds.shape == (2, func.dim)

        # Lower bounds should be less than upper bounds
        lower, upper = func.bounds
        assert (lower < upper).all()

        # Bounds should be reasonable for controller params
        assert lower.min() >= -10.0
        assert upper.max() <= 10.0

    except ImportError as e:
        pytest.skip(f"MuJoCo not available: {e}")


if __name__ == "__main__":
    # Run basic tests
    print("Testing MuJoCo integration...")
    test_mujoco_import()
    test_halfcheetah_creation()
    test_hopper_creation()
    test_suite_creation()
    print("\nAll basic tests passed!")
