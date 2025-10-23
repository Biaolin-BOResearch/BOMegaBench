"""
MuJoCo Control Tasks for BOMegaBench.

This module integrates standard MuJoCo locomotion tasks for controller optimization,
which are the de facto standard benchmarks in Bayesian Optimization for robotics
(2020-2025). These environments dominate BO research papers for robot learning.

Standard MuJoCo control suite:
- HalfCheetah-v2/v3/v4/v5: 6 DoF quadruped running
- Hopper-v2/v3/v4/v5: 3 DoF one-legged hopping
- Walker2d-v2/v3/v4/v5: 6 DoF bipedal walking
- Ant-v2/v3/v4/v5: 8 DoF quadruped
- Humanoid-v2/v3/v4/v5: 17 DoF bipedal humanoid

Reference:
- Gymnasium MuJoCo: https://gymnasium.farama.org/environments/mujoco/
- Original MuJoCo: https://mujoco.org/

Applications:
- Controller parameter optimization
- Policy parameter tuning
- Morphology co-optimization
- Transfer learning benchmarks

Note: These are unconstrained optimization problems - maximize cumulative reward.
"""

import sys
import os
from typing import Dict, List, Optional, Any, Union, Callable
import torch
from torch import Tensor
import numpy as np
import warnings

from bomegabench.core import BenchmarkFunction, BenchmarkSuite


class MuJoCoControlWrapper(BenchmarkFunction):
    """
    Wrapper for MuJoCo control tasks for Bayesian Optimization.

    This wrapper allows optimizing controller parameters (e.g., neural network
    policy weights) using BO. The objective is to maximize the cumulative reward
    (return) from the MuJoCo environment.

    The standard setup:
    - Input X: Controller parameters (e.g., neural network weights)
    - Output Y: Cumulative reward (episode return)
    - Goal: Maximize Y = f(X)
    """

    def __init__(
        self,
        env_name: str,
        controller_dim: Optional[int] = None,
        num_episodes: int = 1,
        max_episode_steps: Optional[int] = 1000,
        seed: Optional[int] = None,
        controller_type: str = "linear",
        negate: bool = False,
        noise_std: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize MuJoCo control task.

        Args:
            env_name: MuJoCo environment name (e.g., 'HalfCheetah-v4')
            controller_dim: Dimension of controller parameters. If None, auto-computed
            num_episodes: Number of episodes to average over for each evaluation
            max_episode_steps: Maximum steps per episode
            seed: Random seed for reproducibility
            controller_type: Type of controller ('linear' or 'mlp')
            negate: Whether to negate (default False, as we maximize reward)
            noise_std: Standard deviation of Gaussian noise
            **kwargs: Additional parameters
        """
        try:
            import gymnasium as gym
        except ImportError:
            raise ImportError(
                "Gymnasium is required for MuJoCo tasks. "
                "Install with: pip install gymnasium[mujoco]"
            )

        self.env_name = env_name
        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        self.seed_value = seed
        self.controller_type = controller_type

        # Create environment to get dimensions
        try:
            self.env = gym.make(env_name)
        except Exception as e:
            raise ImportError(
                f"Failed to create MuJoCo environment {env_name}. "
                f"Make sure MuJoCo is installed: pip install gymnasium[mujoco]\n"
                f"Error: {e}"
            )

        # Get observation and action dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Determine controller parameter dimension
        if controller_dim is None:
            if controller_type == "linear":
                # Linear policy: W (obs_dim × action_dim) + b (action_dim)
                controller_dim = self.obs_dim * self.action_dim + self.action_dim
            elif controller_type == "mlp":
                # Small MLP: obs_dim -> 32 -> action_dim
                hidden_size = 32
                controller_dim = (
                    self.obs_dim * hidden_size + hidden_size +  # layer 1
                    hidden_size * self.action_dim + self.action_dim  # layer 2
                )
            else:
                raise ValueError(f"Unknown controller type: {controller_type}")

        self.controller_dim = controller_dim

        # Set reasonable bounds for controller parameters
        # Initialize near zero for stable policies
        bounds = torch.tensor(
            [[-1.0] * controller_dim, [1.0] * controller_dim],
            dtype=torch.float32
        )

        super().__init__(
            dim=controller_dim,
            bounds=bounds,
            negate=negate,
            noise_std=noise_std,
            **kwargs
        )

    def _get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the MuJoCo task."""
        return {
            "name": f"MuJoCo_{self.env_name}",
            "source": "Gymnasium MuJoCo",
            "type": "continuous",
            "env_name": self.env_name,
            "controller_type": self.controller_type,
            "description": f"MuJoCo {self.env_name} controller optimization",
            "reference": "https://gymnasium.farama.org/environments/mujoco/",
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "controller_dim": self.controller_dim,
            "num_episodes": self.num_episodes,
            "max_episode_steps": self.max_episode_steps,
            "optimization_goal": "maximize_reward",
        }

    def _params_to_controller(self, params: np.ndarray) -> Callable:
        """
        Convert parameter vector to a controller function.

        Args:
            params: Controller parameters

        Returns:
            Controller function: obs -> action
        """
        if self.controller_type == "linear":
            # Linear policy: action = W @ obs + b
            W_size = self.obs_dim * self.action_dim
            W = params[:W_size].reshape(self.action_dim, self.obs_dim)
            b = params[W_size:]

            def controller(obs):
                action = W @ obs + b
                # Clip to action space bounds
                return np.clip(action, self.env.action_space.low, self.env.action_space.high)

        elif self.controller_type == "mlp":
            # Simple 1-hidden-layer MLP
            hidden_size = 32

            # Layer 1: obs_dim -> hidden_size
            W1_size = self.obs_dim * hidden_size
            W1 = params[:W1_size].reshape(hidden_size, self.obs_dim)
            b1 = params[W1_size:W1_size + hidden_size]

            # Layer 2: hidden_size -> action_dim
            offset = W1_size + hidden_size
            W2_size = hidden_size * self.action_dim
            W2 = params[offset:offset + W2_size].reshape(self.action_dim, hidden_size)
            b2 = params[offset + W2_size:]

            def controller(obs):
                # Forward pass with tanh activation
                hidden = np.tanh(W1 @ obs + b1)
                action = W2 @ hidden + b2
                # Clip to action space bounds
                return np.clip(action, self.env.action_space.low, self.env.action_space.high)

        else:
            raise ValueError(f"Unknown controller type: {self.controller_type}")

        return controller

    def _evaluate_controller(self, params: np.ndarray) -> float:
        """
        Evaluate a controller in the MuJoCo environment.

        Args:
            params: Controller parameters

        Returns:
            Average cumulative reward over episodes
        """
        import gymnasium as gym

        controller = self._params_to_controller(params)
        total_rewards = []

        for episode_idx in range(self.num_episodes):
            # Reset environment
            if self.seed_value is not None:
                obs, info = self.env.reset(seed=self.seed_value + episode_idx)
            else:
                obs, info = self.env.reset()

            episode_reward = 0.0
            done = False
            step = 0

            while not done and step < (self.max_episode_steps or 1000):
                # Get action from controller
                action = controller(obs)

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

                done = terminated or truncated
                step += 1

            total_rewards.append(episode_reward)

        # Return average reward
        return np.mean(total_rewards)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate controller parameters in MuJoCo environment.

        Args:
            X: Input tensor of shape (..., dim)

        Returns:
            Cumulative rewards of shape (...)
        """
        # Convert to numpy
        X_np = X.detach().cpu().numpy()

        # Remember original shape
        original_shape = X.shape[:-1]

        # Ensure 2D array (n_samples, dim)
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
            single_sample = True
        else:
            X_np = X_np.reshape(-1, X.shape[-1])
            single_sample = False

        # Evaluate each controller
        rewards = []
        for params in X_np:
            try:
                reward = self._evaluate_controller(params)
                rewards.append(reward)
            except Exception as e:
                warnings.warn(f"Controller evaluation failed: {e}")
                rewards.append(-np.inf)  # Return very bad reward on failure

        rewards = np.array(rewards)

        # Convert back to torch
        Y = torch.tensor(rewards, dtype=X.dtype, device=X.device)

        # Reshape to match input shape
        if single_sample:
            Y = Y.squeeze()
        else:
            Y = Y.reshape(original_shape)

        return Y

    def __del__(self):
        """Clean up environment on deletion."""
        if hasattr(self, 'env'):
            self.env.close()


# Standard MuJoCo environments for BO benchmarking
MUJOCO_ENVIRONMENTS = {
    # Most common in BO papers (2020-2025)
    "halfcheetah": {
        "HalfCheetah-v4": "6-DoF quadruped running (most common in BO papers)",
        "HalfCheetah-v5": "6-DoF quadruped running (latest version)",
    },
    "hopper": {
        "Hopper-v4": "3-DoF one-legged hopping (very common in BO papers)",
        "Hopper-v5": "3-DoF one-legged hopping (latest version)",
    },
    "walker2d": {
        "Walker2d-v4": "6-DoF bipedal walking (common in BO papers)",
        "Walker2d-v5": "6-DoF bipedal walking (latest version)",
    },
    "ant": {
        "Ant-v4": "8-DoF quadruped (common in BO papers)",
        "Ant-v5": "8-DoF quadruped (latest version)",
    },
    "humanoid": {
        "Humanoid-v4": "17-DoF bipedal humanoid (challenging benchmark)",
        "Humanoid-v5": "17-DoF bipedal humanoid (latest version)",
    },
}


def create_mujoco_control_suite(
    controller_type: str = "linear",
    versions: List[str] = ["v4"],
    num_episodes: int = 3,
) -> BenchmarkSuite:
    """
    Create a suite of MuJoCo control tasks.

    Args:
        controller_type: Type of controller ('linear' or 'mlp')
        versions: List of environment versions to include (e.g., ['v4', 'v5'])
        num_episodes: Number of episodes to average for each evaluation

    Returns:
        BenchmarkSuite containing MuJoCo control tasks
    """
    functions = {}

    for category, envs in MUJOCO_ENVIRONMENTS.items():
        for env_name, description in envs.items():
            # Check if this version should be included
            version = env_name.split('-')[-1]
            if version not in versions:
                continue

            try:
                func = MuJoCoControlWrapper(
                    env_name=env_name,
                    controller_type=controller_type,
                    num_episodes=num_episodes,
                )
                # Use naming like: mujoco_halfcheetah_v4_linear
                func_key = f"mujoco_{category}_{version}_{controller_type}"
                functions[func_key] = func
            except Exception as e:
                # Skip if environment cannot be created
                warnings.warn(f"Could not load MuJoCo environment {env_name}: {e}")
                continue

    suite = BenchmarkSuite(
        name="MuJoCoControl",
        functions=functions
    )
    suite.description = "Standard MuJoCo locomotion tasks for controller optimization (BO benchmark standard 2020-2025)"
    return suite


# Convenience classes for commonly used environments
class HalfCheetahLinearFunction(MuJoCoControlWrapper):
    """HalfCheetah-v4 with linear controller (most common in BO papers)."""
    def __init__(self, **kwargs):
        super().__init__(env_name="HalfCheetah-v4", controller_type="linear", **kwargs)


class HopperLinearFunction(MuJoCoControlWrapper):
    """Hopper-v4 with linear controller."""
    def __init__(self, **kwargs):
        super().__init__(env_name="Hopper-v4", controller_type="linear", **kwargs)


class Walker2dLinearFunction(MuJoCoControlWrapper):
    """Walker2d-v4 with linear controller."""
    def __init__(self, **kwargs):
        super().__init__(env_name="Walker2d-v4", controller_type="linear", **kwargs)


class AntLinearFunction(MuJoCoControlWrapper):
    """Ant-v4 with linear controller."""
    def __init__(self, **kwargs):
        super().__init__(env_name="Ant-v4", controller_type="linear", **kwargs)


class HumanoidLinearFunction(MuJoCoControlWrapper):
    """Humanoid-v4 with linear controller (challenging)."""
    def __init__(self, **kwargs):
        super().__init__(env_name="Humanoid-v4", controller_type="linear", **kwargs)


__all__ = [
    "MuJoCoControlWrapper",
    "create_mujoco_control_suite",
    "HalfCheetahLinearFunction",
    "HopperLinearFunction",
    "Walker2dLinearFunction",
    "AntLinearFunction",
    "HumanoidLinearFunction",
    "MUJOCO_ENVIRONMENTS",
]
