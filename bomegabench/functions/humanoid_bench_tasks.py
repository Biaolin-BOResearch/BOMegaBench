"""
HumanoidBench Tasks for BOMegaBench.

This module integrates HumanoidBench tasks for whole-body humanoid controller optimization.
HumanoidBench provides a comprehensive benchmark for humanoid robots performing both
locomotion and manipulation tasks, which are ideal for Bayesian Optimization of
control policies.

HumanoidBench Task Suite:
- Locomotion tasks: walk, run, stand, crawl, hurdle, stair, slide, pole, maze, balance
- Manipulation tasks: reach, push, door, cabinet, truck, cube, bookshelf, basketball,
  window, spoon, kitchen, package, powerlift, room, insert, highbar
- Robots: H1 (with hands), G1 (Unitree)

Reference:
- Paper: https://arxiv.org/abs/2403.10506
- Website: https://sferrazza.cc/humanoidbench_site/
- GitHub: https://github.com/carlosferrazza/humanoid-bench

Applications:
- Whole-body humanoid controller optimization
- Locomotion and manipulation policy tuning
- Transfer learning benchmarks
- Complex behavior learning

Note: These are unconstrained optimization problems - maximize cumulative reward.
This integration imports from the humanoid-bench package directly without reimplementation.
"""

import sys
import os
from typing import Dict, List, Optional, Any, Union, Callable
import torch
from torch import Tensor
import numpy as np
import warnings

from bomegabench.core import BenchmarkFunction, BenchmarkSuite


# Available HumanoidBench tasks (non-overlapping with existing benchmarks)
# Locomotion tasks
LOCOMOTION_TASKS = [
    "walk",
    "run",
    "stand",
    "crawl",
    "hurdle",
    "stair",
    "slide",
    "pole",
    "maze",
    "sit_simple",
    "sit_hard",
    "balance_simple",
    "balance_hard",
]

# Manipulation tasks
MANIPULATION_TASKS = [
    "reach",
    "push",
    "door",
    "cabinet",
    "truck",
    "cube",
    "bookshelf_simple",
    "bookshelf_hard",
    "basketball",
    "window",
    "spoon",
    "kitchen",
    "package",
    "powerlift",
    "room",
    "insert_normal",
    "insert_small",
    "highbar_simple",
    "highbar_hard",
]

# All available tasks
ALL_TASKS = LOCOMOTION_TASKS + MANIPULATION_TASKS

# Robots available in HumanoidBench
AVAILABLE_ROBOTS = ["h1", "h1hand", "h1strong", "h1touch", "g1"]

# Default robot for each task category
DEFAULT_ROBOT = "h1hand"  # H1 with hands for full capability


class HumanoidBenchWrapper(BenchmarkFunction):
    """
    Wrapper for HumanoidBench tasks for Bayesian Optimization.

    This wrapper allows optimizing controller parameters for complex humanoid
    locomotion and manipulation tasks. The objective is to maximize the cumulative
    reward (return) from the HumanoidBench environment.

    The standard setup:
    - Input X: Controller parameters (e.g., neural network weights)
    - Output Y: Cumulative reward (episode return)
    - Goal: Maximize Y = f(X)
    """

    def __init__(
        self,
        task_name: str,
        robot: str = DEFAULT_ROBOT,
        controller_dim: Optional[int] = None,
        num_episodes: int = 1,
        max_episode_steps: Optional[int] = None,
        seed: Optional[int] = None,
        controller_type: str = "linear",
        negate: bool = False,
        noise_std: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize HumanoidBench task.

        Args:
            task_name: HumanoidBench task name (e.g., 'walk', 'push', 'cabinet')
            robot: Robot to use (e.g., 'h1hand', 'g1')
            controller_dim: Dimension of controller parameters. If None, auto-computed
            num_episodes: Number of episodes to average over for each evaluation
            max_episode_steps: Maximum steps per episode (uses task default if None)
            seed: Random seed for reproducibility
            controller_type: Type of controller ('linear' or 'mlp')
            negate: Whether to negate (default False, as we maximize reward)
            noise_std: Standard deviation of Gaussian noise
            **kwargs: Additional parameters passed to HumanoidEnv
        """
        # Add humanoid-bench to path if needed
        humanoid_bench_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'humanoid-bench'
        )
        if humanoid_bench_path not in sys.path:
            sys.path.insert(0, humanoid_bench_path)

        try:
            import gymnasium as gym
            import humanoid_bench  # This registers all environments
        except ImportError as e:
            raise ImportError(
                "HumanoidBench is required for humanoid tasks. "
                f"Make sure the humanoid-bench package is available.\n"
                f"Error: {e}"
            )

        if task_name not in ALL_TASKS:
            raise ValueError(
                f"Unknown task {task_name}. "
                f"Available tasks: {ALL_TASKS}"
            )

        if robot not in AVAILABLE_ROBOTS:
            raise ValueError(
                f"Unknown robot {robot}. "
                f"Available robots: {AVAILABLE_ROBOTS}"
            )

        self.task_name = task_name
        self.robot = robot
        self.num_episodes = num_episodes
        self.seed_value = seed
        self.controller_type = controller_type

        # Construct environment ID
        self.env_id = f"{robot}-{task_name}-v0"

        # Create environment to get dimensions
        try:
            self.env = gym.make(self.env_id)
        except Exception as e:
            raise ImportError(
                f"Failed to create HumanoidBench environment {self.env_id}. "
                f"Make sure humanoid-bench is properly installed.\n"
                f"Error: {e}"
            )

        # Get observation and action dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # Set max episode steps
        if max_episode_steps is None:
            # Use environment's default
            self.max_episode_steps = self.env.spec.max_episode_steps
        else:
            self.max_episode_steps = max_episode_steps

        # Determine controller parameter dimension
        if controller_dim is None:
            if controller_type == "linear":
                # Linear policy: W (obs_dim × action_dim) + b (action_dim)
                controller_dim = self.obs_dim * self.action_dim + self.action_dim
            elif controller_type == "mlp":
                # Small MLP: obs_dim -> 64 -> action_dim
                # (larger hidden size for complex humanoid tasks)
                hidden_size = 64
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
        """Get metadata for the HumanoidBench task."""
        task_category = "locomotion" if self.task_name in LOCOMOTION_TASKS else "manipulation"

        return {
            "name": f"HumanoidBench_{self.robot}_{self.task_name}",
            "source": "HumanoidBench",
            "type": "continuous",
            "task_name": self.task_name,
            "robot": self.robot,
            "env_id": self.env_id,
            "task_category": task_category,
            "controller_type": self.controller_type,
            "description": f"HumanoidBench {self.task_name} task with {self.robot} robot",
            "reference": "https://arxiv.org/abs/2403.10506",
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
                # Clip to [-1, 1] (HumanoidBench action space)
                return np.clip(action, -1.0, 1.0)

        elif self.controller_type == "mlp":
            # Simple 1-hidden-layer MLP
            hidden_size = 64

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
                # Clip to [-1, 1] (HumanoidBench action space)
                return np.clip(action, -1.0, 1.0)

        else:
            raise ValueError(f"Unknown controller type: {self.controller_type}")

        return controller

    def _evaluate_controller(self, params: np.ndarray) -> float:
        """
        Evaluate a controller by running episodes in the environment.

        Args:
            params: Controller parameters

        Returns:
            Mean cumulative reward across episodes
        """
        controller = self._params_to_controller(params)

        episode_rewards = []
        for episode_idx in range(self.num_episodes):
            # Set seed for reproducibility
            if self.seed_value is not None:
                seed = self.seed_value + episode_idx
                self.env.seed(seed)
                np.random.seed(seed)

            obs, _ = self.env.reset()
            episode_reward = 0.0
            done = False
            step = 0

            while not done and step < self.max_episode_steps:
                action = controller(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step += 1

            episode_rewards.append(episode_reward)

        return np.mean(episode_rewards)

    def evaluate(self, X: Tensor) -> Tensor:
        """
        Evaluate controller parameters.

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


def create_humanoid_bench_suite(
    robot: str = DEFAULT_ROBOT,
    task_categories: Optional[List[str]] = None,
    controller_type: str = "linear",
    num_episodes: int = 1,
) -> BenchmarkSuite:
    """
    Create a suite of HumanoidBench tasks.

    Args:
        robot: Robot to use (e.g., 'h1hand', 'g1')
        task_categories: List of task categories to include.
                        Options: ['locomotion', 'manipulation']
                        If None, includes all tasks
        controller_type: Type of controller ('linear' or 'mlp')
        num_episodes: Number of episodes to average for each evaluation

    Returns:
        BenchmarkSuite containing HumanoidBench tasks
    """
    functions = {}

    # Determine which tasks to include
    tasks_to_include = []
    if task_categories is None:
        tasks_to_include = ALL_TASKS
    else:
        if "locomotion" in task_categories:
            tasks_to_include.extend(LOCOMOTION_TASKS)
        if "manipulation" in task_categories:
            tasks_to_include.extend(MANIPULATION_TASKS)

    for task_name in tasks_to_include:
        try:
            task_category = "locomotion" if task_name in LOCOMOTION_TASKS else "manipulation"
            func_name = f"HumanoidBench_{robot}_{task_name}_{controller_type}"

            func = HumanoidBenchWrapper(
                task_name=task_name,
                robot=robot,
                controller_type=controller_type,
                num_episodes=num_episodes,
            )

            functions[func_name] = func

        except Exception as e:
            warnings.warn(f"Failed to create task {task_name}: {e}")
            continue

    return BenchmarkSuite(
        name=f"HumanoidBench_{robot}_{controller_type}",
        functions=functions,
        metadata={
            "source": "HumanoidBench",
            "robot": robot,
            "controller_type": controller_type,
            "num_episodes": num_episodes,
            "task_categories": task_categories or ["locomotion", "manipulation"],
            "reference": "https://arxiv.org/abs/2403.10506",
        }
    )


# Convenience classes for specific popular tasks
class H1HandWalkFunction(HumanoidBenchWrapper):
    def __init__(self, **kwargs):
        super().__init__(task_name="walk", robot="h1hand", **kwargs)


class H1HandPushFunction(HumanoidBenchWrapper):
    def __init__(self, **kwargs):
        super().__init__(task_name="push", robot="h1hand", **kwargs)


class H1HandCabinetFunction(HumanoidBenchWrapper):
    def __init__(self, **kwargs):
        super().__init__(task_name="cabinet", robot="h1hand", **kwargs)


class H1HandDoorFunction(HumanoidBenchWrapper):
    def __init__(self, **kwargs):
        super().__init__(task_name="door", robot="h1hand", **kwargs)


class G1WalkFunction(HumanoidBenchWrapper):
    def __init__(self, **kwargs):
        super().__init__(task_name="walk", robot="g1", **kwargs)


class G1PushFunction(HumanoidBenchWrapper):
    def __init__(self, **kwargs):
        super().__init__(task_name="push", robot="g1", **kwargs)


__all__ = [
    "HumanoidBenchWrapper",
    "create_humanoid_bench_suite",
    "H1HandWalkFunction",
    "H1HandPushFunction",
    "H1HandCabinetFunction",
    "H1HandDoorFunction",
    "G1WalkFunction",
    "G1PushFunction",
    "LOCOMOTION_TASKS",
    "MANIPULATION_TASKS",
    "ALL_TASKS",
    "AVAILABLE_ROBOTS",
]
