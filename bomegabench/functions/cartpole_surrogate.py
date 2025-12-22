"""
CartPole RL Benchmark for BOMegaBench using Stable-Baselines3.

This module provides a CartPole reinforcement learning hyperparameter
optimization benchmark using stable-baselines3's PPO implementation.

Reference: https://stable-baselines3.readthedocs.io/
"""

import torch
from torch import Tensor
import numpy as np
from typing import Dict, Any, List, Optional
import warnings

from ..core import BenchmarkFunction, BenchmarkSuite

# Check if stable-baselines3 is available
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    warnings.warn("stable-baselines3 not available. Install with: pip install stable-baselines3 gymnasium")


class CartPolePPOFunction(BenchmarkFunction):
    """
    CartPole PPO hyperparameter optimization benchmark using Stable-Baselines3.
    
    This benchmark trains a PPO agent on CartPole-v1 and returns the
    average evaluation reward. The goal is to find hyperparameters that
    maximize performance.
    
    Hyperparameters (all normalized to [0, 1]):
    - learning_rate: [1e-5, 1e-2] (log scale)
    - gamma (discount factor): [0.9, 0.9999]
    - gae_lambda: [0.9, 1.0]
    - clip_range: [0.1, 0.4]
    - ent_coef (entropy coefficient): [0.0, 0.1]
    - vf_coef (value function coefficient): [0.1, 1.0]
    - n_steps: [16, 2048] (log scale, discretized to powers of 2)
    - batch_size_factor: [0.25, 1.0] (fraction of n_steps)
    
    The objective is to maximize the average episode reward (up to 500 for CartPole).
    """
    
    def __init__(
        self,
        total_timesteps: int = 10000,
        n_eval_episodes: int = 10,
        negate: bool = False,
        noise_std: Optional[float] = None,
        verbose: int = 0,
        **kwargs
    ):
        """
        Initialize CartPole PPO benchmark.
        
        Args:
            total_timesteps: Number of timesteps to train for each evaluation
            n_eval_episodes: Number of episodes for evaluation
            negate: Whether to negate the function (for minimization)
            noise_std: Standard deviation of observation noise
            verbose: Verbosity level (0=silent, 1=info)
            **kwargs: Additional arguments
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 not available. Install with: pip install stable-baselines3 gymnasium")
        
        # 8 hyperparameters
        dim = 8
        
        # All parameters normalized to [0, 1]
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim], dtype=torch.float32)
        
        self.total_timesteps = total_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose
        
        # Parameter specifications for decoding (must be set before super().__init__)
        self.param_specs = [
            {"name": "learning_rate", "low": 1e-5, "high": 1e-2, "scale": "log"},
            {"name": "gamma", "low": 0.9, "high": 0.9999, "scale": "linear"},
            {"name": "gae_lambda", "low": 0.9, "high": 1.0, "scale": "linear"},
            {"name": "clip_range", "low": 0.1, "high": 0.4, "scale": "linear"},
            {"name": "ent_coef", "low": 0.0, "high": 0.1, "scale": "linear"},
            {"name": "vf_coef", "low": 0.1, "high": 1.0, "scale": "linear"},
            {"name": "n_steps", "low": 16, "high": 2048, "scale": "log_int"},
            {"name": "batch_size_factor", "low": 0.25, "high": 1.0, "scale": "linear"},
        ]
        
        super().__init__(
            dim=dim,
            bounds=bounds,
            negate=negate,
            noise_std=noise_std,
            **kwargs
        )
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the benchmark."""
        return {
            "name": "CartPolePPO",
            "source": "BOMegaBench",
            "type": "rl_hyperparameter_optimization",
            "task": "CartPole-v1",
            "algorithm": "PPO",
            "library": "stable-baselines3",
            "description": "PPO hyperparameter optimization on CartPole-v1",
            "parameters": [spec["name"] for spec in self.param_specs],
            "objective": "maximize",
            "optimal_value": 500.0,
            "total_timesteps": self.total_timesteps,
        }
    
    def _decode_params(self, X_normalized: np.ndarray) -> Dict[str, Any]:
        """Decode normalized [0,1] parameters to original scale."""
        params = {}
        for i, spec in enumerate(self.param_specs):
            val = X_normalized[i]
            if spec["scale"] == "log":
                # Log scale: map [0,1] to [log(low), log(high)]
                log_low = np.log(spec["low"])
                log_high = np.log(spec["high"])
                params[spec["name"]] = float(np.exp(log_low + val * (log_high - log_low)))
            elif spec["scale"] == "log_int":
                # Log scale for integers (round to nearest power of 2)
                log_low = np.log2(spec["low"])
                log_high = np.log2(spec["high"])
                log_val = log_low + val * (log_high - log_low)
                params[spec["name"]] = int(2 ** round(log_val))
            else:
                # Linear scale
                params[spec["name"]] = float(spec["low"] + val * (spec["high"] - spec["low"]))
        return params
    
    def _train_and_evaluate(self, params: Dict[str, Any], seed: int = 0) -> float:
        """
        Train PPO with given hyperparameters and return mean evaluation reward.
        
        Args:
            params: Decoded hyperparameters
            seed: Random seed for reproducibility
            
        Returns:
            Mean evaluation reward
        """
        # Create environment
        env = gym.make("CartPole-v1")
        
        # Calculate batch_size from n_steps
        n_steps = params["n_steps"]
        batch_size_factor = params.pop("batch_size_factor")
        batch_size = max(1, int(n_steps * batch_size_factor))
        # batch_size must divide n_steps evenly
        while n_steps % batch_size != 0 and batch_size > 1:
            batch_size -= 1
        
        try:
            # Create PPO model - use auto device selection
            # For small MLP networks, CPU may be faster, but GPU can parallelize batches
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=params["learning_rate"],
                gamma=params["gamma"],
                gae_lambda=params["gae_lambda"],
                clip_range=params["clip_range"],
                ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"],
                n_steps=n_steps,
                batch_size=batch_size,
                verbose=0,
                seed=seed,
                device=device,
            )
            
            # Train
            model.learn(total_timesteps=self.total_timesteps)
            
            # Evaluate
            mean_reward, std_reward = evaluate_policy(
                model, 
                env, 
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            
            if self.verbose > 0:
                print(f"Params: lr={params['learning_rate']:.2e}, gamma={params['gamma']:.4f}, "
                      f"n_steps={n_steps}, batch={batch_size} -> Reward: {mean_reward:.1f} ± {std_reward:.1f}")
            
            return float(mean_reward)
            
        except Exception as e:
            warnings.warn(f"Training failed with params {params}: {e}")
            return 0.0
        finally:
            env.close()
    
    def _evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluate the benchmark function by training PPO.
        """
        X_np = X.detach().cpu().numpy()
        
        # Handle batched input
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
            single_input = True
        else:
            original_shape = X_np.shape[:-1]
            X_np = X_np.reshape(-1, X_np.shape[-1])
            single_input = False
        
        results = []
        for i, x in enumerate(X_np):
            params = self._decode_params(x)
            reward = self._train_and_evaluate(params, seed=i)
            results.append(reward)
        
        Y = torch.tensor(np.array(results), dtype=X.dtype, device=X.device)
        
        if single_input:
            Y = Y.squeeze()
        else:
            Y = Y.reshape(original_shape)
        
        return Y


class CartPolePPOReducedFunction(BenchmarkFunction):
    """
    Reduced-dimensionality CartPole PPO benchmark (4D).
    
    Optimizes only the most important hyperparameters:
    - learning_rate
    - gamma  
    - n_steps
    - clip_range
    
    Other hyperparameters are set to reasonable defaults.
    """
    
    def __init__(
        self,
        total_timesteps: int = 10000,
        n_eval_episodes: int = 10,
        negate: bool = False,
        noise_std: Optional[float] = None,
        verbose: int = 0,
        **kwargs
    ):
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 not available. Install with: pip install stable-baselines3 gymnasium")
        
        dim = 4
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim], dtype=torch.float32)
        
        self.total_timesteps = total_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose
        
        self.param_specs = [
            {"name": "learning_rate", "low": 1e-5, "high": 1e-2, "scale": "log"},
            {"name": "gamma", "low": 0.9, "high": 0.9999, "scale": "linear"},
            {"name": "n_steps", "low": 16, "high": 2048, "scale": "log_int"},
            {"name": "clip_range", "low": 0.1, "high": 0.4, "scale": "linear"},
        ]
        
        # Fixed defaults for other parameters
        self.defaults = {
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
        }
        
        super().__init__(
            dim=dim,
            bounds=bounds,
            negate=negate,
            noise_std=noise_std,
            **kwargs
        )
    
    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "CartPolePPOReduced",
            "source": "BOMegaBench",
            "type": "rl_hyperparameter_optimization",
            "task": "CartPole-v1",
            "algorithm": "PPO",
            "library": "stable-baselines3",
            "description": "Reduced 4D PPO hyperparameter optimization on CartPole-v1",
            "parameters": [spec["name"] for spec in self.param_specs],
            "objective": "maximize",
            "optimal_value": 500.0,
        }
    
    def _decode_params(self, X_normalized: np.ndarray) -> Dict[str, Any]:
        """Decode normalized [0,1] parameters to original scale."""
        params = dict(self.defaults)  # Start with defaults
        for i, spec in enumerate(self.param_specs):
            val = X_normalized[i]
            if spec["scale"] == "log":
                log_low = np.log(spec["low"])
                log_high = np.log(spec["high"])
                params[spec["name"]] = float(np.exp(log_low + val * (log_high - log_low)))
            elif spec["scale"] == "log_int":
                log_low = np.log2(spec["low"])
                log_high = np.log2(spec["high"])
                log_val = log_low + val * (log_high - log_low)
                params[spec["name"]] = int(2 ** round(log_val))
            else:
                params[spec["name"]] = float(spec["low"] + val * (spec["high"] - spec["low"]))
        return params
    
    def _train_and_evaluate(self, params: Dict[str, Any], seed: int = 0) -> float:
        """Train PPO and return mean evaluation reward."""
        env = gym.make("CartPole-v1")
        
        n_steps = params["n_steps"]
        # Use n_steps as batch_size (minibatch = full batch)
        batch_size = n_steps
        
        try:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=params["learning_rate"],
                gamma=params["gamma"],
                gae_lambda=params["gae_lambda"],
                clip_range=params["clip_range"],
                ent_coef=params["ent_coef"],
                vf_coef=params["vf_coef"],
                n_steps=n_steps,
                batch_size=batch_size,
                verbose=0,
                seed=seed,
                device="cpu",
            )
            
            model.learn(total_timesteps=self.total_timesteps)
            
            mean_reward, _ = evaluate_policy(
                model, env, n_eval_episodes=self.n_eval_episodes, deterministic=True
            )
            
            return float(mean_reward)
            
        except Exception as e:
            warnings.warn(f"Training failed: {e}")
            return 0.0
        finally:
            env.close()
    
    def _evaluate_true(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy()
        
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
            single_input = True
        else:
            original_shape = X_np.shape[:-1]
            X_np = X_np.reshape(-1, X_np.shape[-1])
            single_input = False
        
        results = []
        for i, x in enumerate(X_np):
            params = self._decode_params(x)
            reward = self._train_and_evaluate(params, seed=i)
            results.append(reward)
        
        Y = torch.tensor(np.array(results), dtype=X.dtype, device=X.device)
        
        if single_input:
            Y = Y.squeeze()
        else:
            Y = Y.reshape(original_shape)
        
        return Y


class CartPoleSurrogateFunction(BenchmarkFunction):
    """
    Fast surrogate benchmark for CartPole (no actual training).
    
    This is a lightweight alternative that uses a polynomial surrogate
    to approximate the PPO hyperparameter landscape. Useful for:
    - Quick testing and debugging
    - When training time is a concern
    - Comparing BO algorithms without RL overhead
    """
    
    def __init__(
        self,
        negate: bool = False,
        noise_std: Optional[float] = 0.01,
        **kwargs
    ):
        dim = 8
        bounds = torch.tensor([[0.0] * dim, [1.0] * dim], dtype=torch.float32)
        
        super().__init__(
            dim=dim,
            bounds=bounds,
            negate=negate,
            noise_std=noise_std,
            **kwargs
        )
        
        # Optimal region based on typical PPO hyperparameters
        self._optimal_normalized = torch.tensor([
            0.5,   # lr around 3e-4
            0.9,   # gamma around 0.99
            0.5,   # gae_lambda around 0.95
            0.33,  # clip around 0.2
            0.1,   # ent_coef around 0.01
            0.5,   # vf_coef around 0.5
            0.6,   # n_steps around 128-256
            0.5,   # batch_size_ratio around 0.5
        ])
    
    def _get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "CartPoleSurrogate",
            "source": "BOMegaBench",
            "type": "surrogate",
            "description": "Fast surrogate for PPO hyperparameter optimization",
            "objective": "maximize",
            "optimal_value": 500.0,
        }
    
    def _evaluate_true(self, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy()
        
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)
            single_input = True
        else:
            original_shape = X_np.shape[:-1]
            X_np = X_np.reshape(-1, X_np.shape[-1])
            single_input = False
        
        results = []
        for x in X_np:
            weights = np.array([2.0, 1.5, 0.5, 0.5, 0.3, 0.3, 1.0, 0.3])
            diff = x - self._optimal_normalized.numpy()
            weighted_dist_sq = np.sum(weights * diff ** 2)
            
            base_reward = 495 * np.exp(-3.0 * weighted_dist_sq)
            
            if x[0] < 0.1 or x[0] > 0.9:
                base_reward *= 0.5
            if x[1] < 0.3:
                base_reward *= 0.6
            
            local_perturbation = 20 * np.sin(5 * np.pi * x[0]) * np.sin(3 * np.pi * x[1])
            reward = np.clip(base_reward + local_perturbation, 0, 500)
            results.append(reward)
        
        Y = torch.tensor(np.array(results), dtype=X.dtype, device=X.device)
        
        if single_input:
            Y = Y.squeeze()
        else:
            Y = Y.reshape(original_shape)
        
        return Y


def create_cartpole_surrogate_suite() -> BenchmarkSuite:
    """
    Create a suite of CartPole benchmarks.
    
    Returns:
        BenchmarkSuite containing CartPole benchmarks
    """
    functions = {}
    
    # Always add surrogate (fast, no dependencies)
    functions["CartPoleSurrogate"] = CartPoleSurrogateFunction()
    
    # Add real PPO benchmarks if stable-baselines3 is available
    if SB3_AVAILABLE:
        functions["CartPolePPO"] = CartPolePPOFunction(total_timesteps=10000)
        functions["CartPolePPOReduced"] = CartPolePPOReducedFunction(total_timesteps=10000)
        # Faster version for quick experiments
        functions["CartPolePPOFast"] = CartPolePPOFunction(total_timesteps=5000, n_eval_episodes=5)
    
    suite = BenchmarkSuite(
        name="CartPoleSurrogate",
        functions=functions
    )
    suite.description = "CartPole RL hyperparameter optimization benchmarks"
    return suite


__all__ = [
    "CartPolePPOFunction",
    "CartPolePPOReducedFunction",
    "CartPoleSurrogateFunction",
    "create_cartpole_surrogate_suite",
    "SB3_AVAILABLE",
]
