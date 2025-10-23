"""
Robosuite Manipulation Tasks for BOMegaBench.

This module integrates robosuite manipulation tasks for controller optimization.
Robosuite provides a diverse set of robot manipulation environments for learning
and testing robot control algorithms.

Robosuite Manipulation Suite:
- Lift: Single-arm object lifting
- Door: Door opening task
- NutAssembly: Nut assembly task
- PickPlace: Pick and place objects
- Stack: Stack objects
- ToolHang: Hang tool on rack
- Wipe: Wiping task
- TwoArmHandover: Two-arm handover
- TwoArmLift: Two-arm lifting
- TwoArmPegInHole: Two-arm peg insertion
- TwoArmTransport: Two-arm object transport

Reference:
- Robosuite: https://robosuite.ai/
- Paper: https://arxiv.org/abs/2009.12293

Applications:
- Manipulation controller optimization
- Policy parameter tuning
- Multi-arm coordination
- Contact-rich task optimization

Note: These are unconstrained optimization problems - maximize task reward.
"""

import sys
import os
from typing import Dict, List, Optional, Any, Union, Callable
import torch
from torch import Tensor
import numpy as np
import warnings

from bomegabench.core import BenchmarkFunction, BenchmarkSuite


# Available manipulation environments
MANIPULATION_ENVS = [
    "Lift",
    "Door",
    "NutAssembly",
    "PickPlace",
    "Stack",
    "ToolHang",
    "Wipe",
    "TwoArmHandover",
    "TwoArmLift",
    "TwoArmPegInHole",
    "TwoArmTransport",
]

# Default robot configurations for each task
DEFAULT_ROBOTS = {
    "Lift": "Panda",
    "Door": "Panda",
    "NutAssembly": "Panda",
    "PickPlace": "Panda",
    "Stack": "Panda",
    "ToolHang": "Panda",
    "Wipe": "Panda",
    "TwoArmHandover": ["Panda", "Panda"],
    "TwoArmLift": ["Panda", "Panda"],
    "TwoArmPegInHole": ["Panda", "Panda"],
    "TwoArmTransport": ["Panda", "Panda"],
}


class RobosuiteManipulationWrapper(BenchmarkFunction):
    """
    Wrapper for Robosuite manipulation tasks for Bayesian Optimization.

    This wrapper allows optimizing controller parameters for manipulation tasks.
    The objective is to maximize the cumulative reward (return) from the
    robosuite environment.

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
        horizon: int = 500,
        seed: Optional[int] = None,
        controller_type: str = "linear",
        robots: Optional[Union[str, List[str]]] = None,
        negate: bool = False,
        noise_std: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize Robosuite manipulation task.

        Args:
            env_name: Robosuite environment name (e.g., 'Lift', 'Door')
            controller_dim: Dimension of controller parameters. If None, auto-computed
            num_episodes: Number of episodes to average over for each evaluation
            horizon: Maximum steps per episode
            seed: Random seed for reproducibility
            controller_type: Type of controller ('linear' or 'mlp')
            robots: Robot(s) to use. If None, uses default for the task
            negate: Whether to negate (default False, as we maximize reward)
            noise_std: Standard deviation of Gaussian noise
            **kwargs: Additional parameters passed to robosuite.make()
        """
        # Add robosuite path if needed
        robosuite_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'robosuite'
        )
        if robosuite_path not in sys.path:
            sys.path.insert(0, robosuite_path)

        try:
            import robosuite
            from robosuite import make
        except ImportError:
            raise ImportError(
                "Robosuite is required for manipulation tasks. "
                "The robosuite package should be in the BOMegaBench directory."
            )

        if env_name not in MANIPULATION_ENVS:
            raise ValueError(
                f"Unknown environment {env_name}. "
                f"Available: {MANIPULATION_ENVS}"
            )

        self.env_name = env_name
        self.num_episodes = num_episodes
        self.horizon = horizon
        self.seed_value = seed
        self.controller_type = controller_type

        # Set robots
        if robots is None:
            robots = DEFAULT_ROBOTS[env_name]
        self.robots = robots

        # Create environment to get dimensions
        # Use headless mode for BO (no rendering)
        env_config = {
            'robots': robots,
            'has_renderer': False,
            'has_offscreen_renderer': False,
            'use_camera_obs': False,
            'horizon': horizon,
            'control_freq': 20,
            'reward_shaping': True,
        }
        env_config.update(kwargs)

        try:
            self.env = make(env_name, **env_config)
        except Exception as e:
            raise ImportError(
                f"Failed to create Robosuite environment {env_name}. "
                f"Error: {e}"
            )

        # Get observation and action dimensions
        self.obs_dim = self.env.observation_spec()['robot0_proprio-state'].shape[0]
        self.action_dim = self.env.action_spec[0].shape[0]

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
        """Get metadata for the Robosuite task."""
        return {
            "name": f"Robosuite_{self.env_name}",
            "source": "Robosuite",
            "type": "continuous",
            "env_name": self.env_name,
            "controller_type": self.controller_type,
            "robots": self.robots,
            "description": f"Robosuite {self.env_name} manipulation task",
            "reference": "https://robosuite.ai/",
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "controller_dim": self.controller_dim,
            "horizon": self.horizon,
        }

    def _create_controller(self, params: np.ndarray) -> Callable:
        """
        Create a controller from parameters.

        Args:
            params: Controller parameters

        Returns:
            Controller function: obs -> action
        """
        if self.controller_type == "linear":
            # Linear controller: action = W @ obs + b
            W_size = self.obs_dim * self.action_dim
            W = params[:W_size].reshape(self.action_dim, self.obs_dim)
            b = params[W_size:]

            def controller(obs):
                return np.tanh(W @ obs + b)  # tanh to keep in [-1, 1]

        elif self.controller_type == "mlp":
            # MLP controller: obs -> 32 (tanh) -> action
            hidden_size = 32
            W1_size = self.obs_dim * hidden_size
            W1 = params[:W1_size].reshape(hidden_size, self.obs_dim)
            b1 = params[W1_size:W1_size + hidden_size]

            W2_start = W1_size + hidden_size
            W2_size = hidden_size * self.action_dim
            W2 = params[W2_start:W2_start + W2_size].reshape(
                self.action_dim, hidden_size
            )
            b2 = params[W2_start + W2_size:]

            def controller(obs):
                hidden = np.tanh(W1 @ obs + b1)
                return np.tanh(W2 @ hidden + b2)

        else:
            raise ValueError(f"Unknown controller type: {self.controller_type}")

        return controller

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate controller parameters on the Robosuite environment.

        Args:
            X: Controller parameters, shape (batch_size, controller_dim)

        Returns:
            Cumulative rewards, shape (batch_size, 1)
        """
        X_np = X.cpu().numpy()
        batch_size = X_np.shape[0]
        rewards = np.zeros(batch_size)

        for i in range(batch_size):
            controller = self._create_controller(X_np[i])
            episode_rewards = []

            for _ in range(self.num_episodes):
                self.env.reset()
                total_reward = 0.0

                for step in range(self.horizon):
                    # Get observation
                    obs_dict = self.env._get_observations()
                    obs = obs_dict['robot0_proprio-state']

                    # Get action from controller
                    action = controller(obs)

                    # Step environment
                    obs_dict, reward, done, info = self.env.step(action)
                    total_reward += reward

                    if done:
                        break

                episode_rewards.append(total_reward)

            # Average over episodes
            rewards[i] = np.mean(episode_rewards)

        return torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1)


# Convenience classes for specific environments
class LiftLinearFunction(RobosuiteManipulationWrapper):
    """Lift task with linear controller."""
    def __init__(self, **kwargs):
        super().__init__(env_name="Lift", controller_type="linear", **kwargs)


class DoorLinearFunction(RobosuiteManipulationWrapper):
    """Door opening task with linear controller."""
    def __init__(self, **kwargs):
        super().__init__(env_name="Door", controller_type="linear", **kwargs)


class StackLinearFunction(RobosuiteManipulationWrapper):
    """Stack task with linear controller."""
    def __init__(self, **kwargs):
        super().__init__(env_name="Stack", controller_type="linear", **kwargs)


class NutAssemblyLinearFunction(RobosuiteManipulationWrapper):
    """Nut assembly task with linear controller."""
    def __init__(self, **kwargs):
        super().__init__(env_name="NutAssembly", controller_type="linear", **kwargs)


class PickPlaceLinearFunction(RobosuiteManipulationWrapper):
    """Pick and place task with linear controller."""
    def __init__(self, **kwargs):
        super().__init__(env_name="PickPlace", controller_type="linear", **kwargs)


def create_robosuite_manipulation_suite(
    controller_type: str = "linear",
    tasks: Optional[List[str]] = None,
    **kwargs
) -> BenchmarkSuite:
    """
    Create a suite of Robosuite manipulation tasks.

    Args:
        controller_type: Type of controller ('linear' or 'mlp')
        tasks: List of task names. If None, uses single-arm tasks
        **kwargs: Additional arguments passed to each task

    Returns:
        BenchmarkSuite containing Robosuite manipulation tasks
    """
    if tasks is None:
        # Default: single-arm tasks (easier for BO)
        tasks = ["Lift", "Door", "Stack", "PickPlace", "NutAssembly"]

    functions = []
    for task_name in tasks:
        func = RobosuiteManipulationWrapper(
            env_name=task_name,
            controller_type=controller_type,
            **kwargs
        )
        functions.append(func)

    return BenchmarkSuite(
        name=f"RobosuiteManipulation_{controller_type}",
        functions=functions
    )


__all__ = [
    "RobosuiteManipulationWrapper",
    "LiftLinearFunction",
    "DoorLinearFunction",
    "StackLinearFunction",
    "NutAssemblyLinearFunction",
    "PickPlaceLinearFunction",
    "create_robosuite_manipulation_suite",
    "MANIPULATION_ENVS",
]
