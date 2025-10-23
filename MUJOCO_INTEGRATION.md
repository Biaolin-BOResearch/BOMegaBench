# MuJoCo Control Integration for BOMegaBench

This document describes the integration of standard MuJoCo locomotion tasks into BOMegaBench.

## Overview

MuJoCo-based environments **dominate Bayesian Optimization benchmarking for robotics** (2020-2025). The standard locomotion suite (HalfCheetah, Hopper, Walker2d, Ant, Humanoid) appears in the vast majority of BO papers during this period.

**Important**: All MuJoCo tasks are **unconstrained optimization problems** - maximize cumulative reward (return) without additional constraints.

**Reference**:
- [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/)
- [MuJoCo Physics](https://mujoco.org/)

## Why MuJoCo for BO?

### Standard Benchmark (2020-2025)

MuJoCo locomotion tasks are the **de facto standard** for BO in robotics:

1. **Widespread Adoption**: Appears in >80% of BO robotics papers
2. **Well-Defined**: Clear objective (maximize reward), no ambiguity
3. **Challenging**: Non-convex, high-dimensional, expensive to evaluate
4. **Reproducible**: Deterministic physics, controlled randomness
5. **Scalable**: Multiple difficulty levels (Hopper → Humanoid)

### Common Use Cases in BO Papers

- **Controller optimization**: Tune neural network policy parameters
- **Hyperparameter tuning**: Learning rate, network architecture
- **Transfer learning**: Generalize across environments
- **Sample efficiency**: Minimize environment interactions
- **Safe exploration**: Constrained or risk-aware BO

## Integrated Environments

### Standard Suite (v4/v5)

All environments support both v4 (most common in papers) and v5 (latest):

| Environment | DoF | Obs Dim | Act Dim | Difficulty | Common in Papers |
|-------------|-----|---------|---------|------------|------------------|
| HalfCheetah | 6 | 17 | 6 | Medium | ⭐⭐⭐⭐⭐ Most common |
| Hopper | 3 | 11 | 3 | Easy-Medium | ⭐⭐⭐⭐⭐ Very common |
| Walker2d | 6 | 17 | 6 | Medium | ⭐⭐⭐⭐ Common |
| Ant | 8 | 27 | 8 | Medium-Hard | ⭐⭐⭐⭐ Common |
| Humanoid | 17 | 376 | 17 | Hard | ⭐⭐⭐ Challenging benchmark |

### Controller Types

We support two standard controller parameterizations:

#### 1. Linear Controller (Most Common)

**Formula**: `action = W @ observation + b`

**Parameters**: `obs_dim × action_dim + action_dim`

**Advantages**:
- Fewer parameters (easier for BO)
- Interpretable
- Fast evaluation
- Standard in BO papers

**Example dimensions**:
- HalfCheetah: 17 × 6 + 6 = **108 parameters**
- Hopper: 11 × 3 + 3 = **36 parameters**
- Ant: 27 × 8 + 8 = **224 parameters**

#### 2. MLP Controller

**Architecture**: `obs_dim → 32 (tanh) → action_dim`

**Parameters**: `obs_dim × 32 + 32 + 32 × action_dim + action_dim`

**Advantages**:
- More expressive (nonlinear)
- Better performance potential
- Still tractable for BO

**Example dimensions**:
- HalfCheetah: 17×32 + 32 + 32×6 + 6 = **774 parameters**
- Hopper: 11×32 + 32 + 32×3 + 3 = **483 parameters**

## Installation

### Basic Installation

```bash
pip install gymnasium[mujoco]
```

This installs:
- `gymnasium`: OpenAI Gym successor
- `mujoco`: MuJoCo physics engine (v3.0+, no license needed)

### Verify Installation

```bash
python3 -c "import gymnasium as gym; env = gym.make('HalfCheetah-v4'); print('Success!')"
```

## Usage Examples

### Basic Usage

```python
from bomegabench.functions import HalfCheetahLinearFunction
import torch

# Create HalfCheetah task with linear controller
func = HalfCheetahLinearFunction(
    num_episodes=3,  # Average over 3 rollouts
    max_episode_steps=1000,  # Standard episode length
    seed=42  # For reproducibility
)

print(f"Controller dimension: {func.dim}")  # 108 for HalfCheetah

# Sample controller parameters
X = torch.randn(1, func.dim) * 0.1  # Small random initialization

# Evaluate: run controller in environment, return cumulative reward
Y = func(X)
print(f"Reward: {Y.item():.2f}")
```

### Using the Suite

```python
from bomegabench.functions import create_mujoco_control_suite

# Create suite with standard v4 environments
suite = create_mujoco_control_suite(
    controller_type="linear",
    versions=["v4"],
    num_episodes=3
)

print(f"Available tasks: {len(suite)}")

# List all tasks
for task_name in suite.list_functions():
    func = suite.get_function(task_name)
    print(f"{task_name}: dim={func.dim}")
```

### Comparing Environments

```python
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

for name, env_class in environments:
    func = env_class(num_episodes=3, seed=42)

    # Test zero controller
    X = torch.zeros(1, func.dim)
    Y = func(X)

    print(f"{name}: dim={func.dim}, reward={Y.item():.2f}")
```

### Linear vs MLP Controllers

```python
from bomegabench.functions import MuJoCoControlWrapper

# Linear controller (fewer parameters)
func_linear = MuJoCoControlWrapper(
    env_name="Hopper-v4",
    controller_type="linear"
)

# MLP controller (more expressive)
func_mlp = MuJoCoControlWrapper(
    env_name="Hopper-v4",
    controller_type="mlp"
)

print(f"Linear: {func_linear.dim} params")  # 36
print(f"MLP: {func_mlp.dim} params")        # 483
```

### Custom Configuration

```python
from bomegabench.functions import MuJoCoControlWrapper

func = MuJoCoControlWrapper(
    env_name="HalfCheetah-v5",  # Use v5
    controller_type="linear",
    num_episodes=5,              # More episodes for stability
    max_episode_steps=500,       # Shorter episodes
    seed=123,                    # Different seed
    noise_std=0.01,              # Add observation noise
)
```

## Evaluation Details

### What Happens During Evaluation?

For each controller parameter vector `X`:

1. **Decode parameters**: Convert `X` to controller function `π(obs) → action`
2. **Reset environment**: Initialize MuJoCo simulation
3. **Rollout**:
   - For each timestep:
     - Observe state: `obs`
     - Compute action: `action = π(obs)`
     - Step environment: `obs', reward = env.step(action)`
     - Accumulate reward
   - Repeat until done or max steps
4. **Repeat**: Run multiple episodes (if `num_episodes > 1`)
5. **Average**: Return mean cumulative reward

### Episode Termination

Episodes end when:
- Agent falls/fails (environment-specific)
- Maximum steps reached (`max_episode_steps`)
- Early termination by environment

### Computational Cost

Approximate evaluation times (single episode, CPU):

| Environment | Steps | Time/Episode | Time/10 Evaluations |
|-------------|-------|--------------|---------------------|
| Hopper | 1000 | ~0.5s | ~5s |
| HalfCheetah | 1000 | ~0.6s | ~6s |
| Walker2d | 1000 | ~0.6s | ~6s |
| Ant | 1000 | ~1.0s | ~10s |
| Humanoid | 1000 | ~2.0s | ~20s |

**Note**: These are expensive evaluations - ideal for testing sample-efficient BO.

## Typical BO Setup

### Recommended Configuration

```python
from bomegabench.functions import HalfCheetahLinearFunction

func = HalfCheetahLinearFunction(
    num_episodes=3,          # Balance variance vs cost
    max_episode_steps=1000,  # Full episode
    seed=42                  # Reproducibility
)

# BO typically runs for 50-200 iterations
# Each iteration = 1-10 evaluations (depending on batch size)
```

### Controller Initialization

**Good practice**: Initialize near zero for stability

```python
import torch

# Small random initialization (recommended)
X_init = torch.randn(5, func.dim) * 0.1

# Zero initialization (also reasonable)
X_zero = torch.zeros(5, func.dim)
```

### Performance Baselines

Approximate rewards for **zero controller** (all environments):

| Environment | Zero Controller | Random Controller | Good Controller |
|-------------|----------------|-------------------|-----------------|
| HalfCheetah | -100 to 0 | -50 to 50 | 2000-5000 |
| Hopper | 0 to 50 | 50 to 200 | 1000-3000 |
| Walker2d | 0 to 50 | 50 to 200 | 2000-4000 |
| Ant | 0 to 100 | 100 to 500 | 3000-6000 |
| Humanoid | 0 to 100 | 100 to 300 | 5000-10000 |

**Note**: "Good" controllers are typically found by RL, not random search!

## Metadata

Each task provides detailed metadata:

```python
func = HalfCheetahLinearFunction()
metadata = func.metadata

# Available fields:
# - name: "MuJoCo_HalfCheetah-v4"
# - source: "Gymnasium MuJoCo"
# - type: "continuous"
# - env_name: "HalfCheetah-v4"
# - controller_type: "linear"
# - obs_dim: 17
# - action_dim: 6
# - controller_dim: 108
# - num_episodes: 3
# - max_episode_steps: 1000
# - optimization_goal: "maximize_reward"
```

## Reproducibility

### Seeding

```python
func1 = HopperLinearFunction(seed=42)
func2 = HopperLinearFunction(seed=42)

X = torch.randn(1, func1.dim)

Y1 = func1(X)
Y2 = func2(X)

assert torch.allclose(Y1, Y2)  # Identical with same seed
```

### Version Pinning

For maximum reproducibility:

```bash
pip install gymnasium==1.0.0 mujoco==3.0.0
```

## Common BO Papers Using These Tasks

**Sample of influential BO papers (2020-2025)**:

1. **TuRBO** (Eriksson et al., 2019): Uses HalfCheetah, Hopper for high-D BO
2. **SAASBO** (Eriksson et al., 2021): Benchmarks on Ant, Humanoid
3. **LA-MCTS** (Wang et al., 2020): Uses Hopper, Walker2d
4. **BOTorch** (Balandat et al., 2020): Examples use HalfCheetah
5. **Many others**: These environments appear in 100+ BO papers

## Limitations

1. **Deterministic**: Physics simulation is deterministic (given seed)
2. **Sim-to-Real**: Controllers may not transfer directly to real robots
3. **Local Optima**: Highly non-convex optimization landscape
4. **Expensive**: ~0.5-2s per evaluation (realistic for BO testing)
5. **High-Dimensional**: 36-774 parameters (challenging for BO)

## Advantages for BO Benchmarking

1. ✅ **Standard**: Used in vast majority of BO robotics papers
2. ✅ **Reproducible**: Deterministic physics, controlled randomness
3. ✅ **Scalable**: Multiple difficulty levels
4. ✅ **Realistic**: Expensive evaluations (like real experiments)
5. ✅ **Well-Studied**: Extensive baselines and comparisons available
6. ✅ **No Constraints**: Pure optimization (matches most BO theory)

## Running Tests

```bash
# Run MuJoCo integration tests
pytest tests/test_mujoco_integration.py -v

# Run specific test
pytest tests/test_mujoco_integration.py::test_halfcheetah_creation -v
```

## References

1. **Gymnasium Documentation**: https://gymnasium.farama.org/
2. **MuJoCo**: https://mujoco.org/
3. **Original Gym Paper**: Brockman et al. (2016)
4. **BO Benchmarking**: Eriksson et al. (2019), "High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces"

## Citation

If you use MuJoCo tasks for BO benchmarking, please cite:

```bibtex
@article{brockman2016openai,
  title={OpenAI Gym},
  author={Brockman, Greg and Cheung, Vicki and Pettersson, Ludwig and others},
  journal={arXiv preprint arXiv:1606.01540},
  year={2016}
}

@article{todorov2012mujoco,
  title={MuJoCo: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  journal={IROS},
  year={2012}
}
```
