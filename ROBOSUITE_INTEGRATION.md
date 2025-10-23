# Robosuite Manipulation Integration for BOMegaBench

This document describes the integration of robosuite manipulation tasks into BOMegaBench for Bayesian Optimization benchmarking.

## Overview

**Robosuite** provides a diverse set of robot manipulation environments built on MuJoCo physics. These tasks complement the existing MuJoCo locomotion tasks in BOMegaBench by adding **contact-rich manipulation** scenarios.

**Important**: All robosuite tasks are **unconstrained optimization problems** - maximize cumulative reward (return) without additional constraints.

**Reference**:
- [Robosuite Homepage](https://robosuite.ai/)
- [Paper: robosuite: A Modular Simulation Framework](https://arxiv.org/abs/2009.12293)

## Why Robosuite for BO?

### Complementary to MuJoCo Locomotion

| Aspect | MuJoCo Locomotion | Robosuite Manipulation |
|--------|-------------------|----------------------|
| **Task Type** | Locomotion (walking, running) | Manipulation (grasping, assembly) |
| **Contact** | Ground contact | Rich object contact |
| **Objective** | Distance/speed | Task completion |
| **Examples** | HalfCheetah, Hopper, Ant | Lift, Door, Stack |
| **Status in BOMegaBench** | ✅ Already integrated | ✅ **Now integrated** |

### Key Features for BO

1. **Diverse Tasks**: 11 manipulation environments with varying complexity
2. **Contact-Rich**: Realistic grasping, pushing, assembly
3. **Standardized**: Well-maintained benchmark suite
4. **Flexible**: Support for multiple robots and controllers
5. **Reproducible**: Deterministic physics simulation

## Integrated Environments

### Single-Arm Manipulation (Easier)

| Environment | Description | Difficulty | Recommended for BO |
|-------------|-------------|------------|-------------------|
| **Lift** | Lift a cube to target height | Easy | ⭐⭐⭐⭐⭐ Best starter |
| **Door** | Open a door with handle | Medium | ⭐⭐⭐⭐ Good benchmark |
| **PickPlace** | Pick and place objects | Medium | ⭐⭐⭐⭐ Common task |
| **Stack** | Stack one cube on another | Medium-Hard | ⭐⭐⭐ Challenging |
| **NutAssembly** | Assemble nut on peg | Hard | ⭐⭐⭐ Complex contact |
| **ToolHang** | Hang tool on rack | Hard | ⭐⭐ Advanced |
| **Wipe** | Wipe surface | Medium | ⭐⭐⭐ Tracking task |

### Two-Arm Manipulation (Harder)

| Environment | Description | Difficulty | Recommended for BO |
|-------------|-------------|------------|-------------------|
| **TwoArmLift** | Lift object with two arms | Medium | ⭐⭐⭐ Coordination |
| **TwoArmHandover** | Hand over object between arms | Hard | ⭐⭐ Complex |
| **TwoArmPegInHole** | Insert peg with two arms | Hard | ⭐⭐ Precision |
| **TwoArmTransport** | Transport object together | Hard | ⭐⭐ Multi-agent |

## Controller Types

We support two standard controller parameterizations for BO:

### 1. Linear Controller (Recommended for BO)

**Formula**: `action = tanh(W @ observation + b)`

**Parameters**: `obs_dim × action_dim + action_dim`

**Advantages**:
- Fewer parameters (easier for BO)
- Fast evaluation
- Interpretable
- Good baseline

**Example dimensions**:
- Lift: ~120-180 parameters (depends on obs dim)
- Door: ~120-180 parameters
- Stack: ~120-180 parameters

### 2. MLP Controller

**Architecture**: `obs → 32 (tanh) → action`

**Parameters**: `obs_dim × 32 + 32 + 32 × action_dim + action_dim`

**Advantages**:
- More expressive (nonlinear)
- Better performance potential
- Still tractable for BO

**Example dimensions**:
- Lift: ~800-1000 parameters
- Door: ~800-1000 parameters

## Installation

### Prerequisites

Robosuite requires MuJoCo. The robosuite package is already included in BOMegaBench.

```bash
# Install MuJoCo (if not already installed)
pip install mujoco

# Verify robosuite is available
python -c "import sys; sys.path.insert(0, 'robosuite'); import robosuite; print('Robosuite OK')"
```

### Dependencies

- `mujoco` (>= 2.3.0): Physics engine
- `numpy`: Array operations
- `torch`: PyTorch for BO interface

## Usage

### Basic Usage

```python
from bomegabench.functions import LiftLinearFunction
import torch

# Create the Lift task with linear controller
func = LiftLinearFunction(
    num_episodes=3,  # Average over 3 episodes
    horizon=500,     # 500 steps per episode
)

print(f"Controller dimension: {func.dim}")  # e.g., 180
print(f"Bounds: {func.bounds}")

# Evaluate controller parameters
X = torch.randn(1, func.dim) * 0.1  # Small random initialization
Y = func(X)  # Returns reward
print(f"Reward: {Y.item():.4f}")
```

### Using Different Tasks

```python
from bomegabench.functions import (
    LiftLinearFunction,
    DoorLinearFunction,
    StackLinearFunction,
)

# Easy task: Lift
lift_func = LiftLinearFunction(horizon=500)

# Medium task: Door
door_func = DoorLinearFunction(horizon=500)

# Hard task: Stack
stack_func = StackLinearFunction(horizon=500)
```

### Custom Configuration

```python
from bomegabench.functions import RobosuiteManipulationWrapper

# Create custom task
func = RobosuiteManipulationWrapper(
    env_name="Lift",
    controller_type="mlp",  # Use MLP controller
    robots="Sawyer",        # Use Sawyer robot instead of Panda
    num_episodes=5,         # Average over 5 episodes
    horizon=1000,           # Longer episodes
    seed=42,                # For reproducibility
)
```

### Creating a Suite

```python
from bomegabench.functions import create_robosuite_manipulation_suite

# Create suite of single-arm tasks
suite = create_robosuite_manipulation_suite(
    controller_type="linear",
    tasks=["Lift", "Door", "Stack"],
    horizon=500,
)

print(f"Suite: {suite.name}")
print(f"Number of tasks: {len(suite.functions)}")

# Evaluate all tasks
for func in suite.functions:
    print(f"\n{func.metadata['env_name']}:")
    X = torch.randn(1, func.dim) * 0.1
    Y = func(X)
    print(f"  Reward: {Y.item():.4f}")
```

## Integration with BO Algorithms

### Example: BO Loop (Pseudo-code)

```python
from bomegabench.functions import LiftLinearFunction
from your_bo_library import BayesianOptimizer

# Create task
func = LiftLinearFunction(horizon=500, num_episodes=3)

# Create BO optimizer
optimizer = BayesianOptimizer(
    bounds=func.bounds,
    objective=func,
    maximize=True,  # Maximize reward
)

# Run BO
for i in range(50):
    X_next = optimizer.suggest()
    Y = func(X_next)
    optimizer.observe(X_next, Y)

    print(f"[{i+1}] Best reward: {optimizer.best_value:.4f}")
```

## Comparison with MuJoCo Locomotion

### Already in BOMegaBench

| Task | Type | Difficulty | Parameter Dim (linear) |
|------|------|------------|----------------------|
| HalfCheetah | Locomotion | Medium | 108 |
| Hopper | Locomotion | Easy-Medium | 36 |
| Walker2d | Locomotion | Medium | 108 |
| Ant | Locomotion | Medium-Hard | 224 |
| Humanoid | Locomotion | Hard | 6,416 |

### New in This Integration

| Task | Type | Difficulty | Parameter Dim (linear) |
|------|------|------------|----------------------|
| Lift | Manipulation | Easy | ~180 |
| Door | Manipulation | Medium | ~180 |
| Stack | Manipulation | Medium-Hard | ~180 |
| NutAssembly | Manipulation | Hard | ~180 |

## No Duplication with Existing Benchmarks

✅ **Confirmed**: Robosuite manipulation tasks do **NOT** overlap with:

1. **MuJoCo Locomotion** (`mujoco_control.py`)
   - Different task type: locomotion vs. manipulation
   - Different objective: distance/speed vs. task completion

2. **Design-Bench** (`design_bench_tasks.py`)
   - Design-Bench's robot tasks (Ant, Hopper) were already excluded
   - Only non-robot tasks were integrated (proteins, materials, etc.)

3. **Other benchmarks**
   - Olympus: Chemistry experiments
   - MolOpt: Molecular optimization
   - HPOBench: Hyperparameter optimization
   - LassoBench: Lasso regression
   - Database: Database tuning

## Technical Details

### Observation Space

Robosuite provides rich observations:
- Robot proprioception (joint positions, velocities)
- Object states (position, orientation)
- Task-specific info (e.g., distance to target)

For BO, we use **proprioceptive state** only to keep the problem tractable.

### Action Space

All robosuite manipulation tasks use continuous action spaces:
- Joint torques or velocities
- End-effector control
- Gripper commands

Our controllers map observations to actions in `[-1, 1]` range.

### Reward Structure

Each task has a **dense reward** designed for learning:
- Distance to goal
- Task completion bonuses
- Penalty for failures (e.g., dropping object)

The objective is to **maximize cumulative reward** over an episode.

## Best Practices for BO

### 1. Start Simple

```python
# Use Lift task with short horizon for initial testing
func = LiftLinearFunction(horizon=200, num_episodes=1)
```

### 2. Increase Difficulty Gradually

```python
# Easy → Medium → Hard
tasks = ["Lift", "Door", "Stack"]
```

### 3. Average Over Multiple Episodes

```python
# Reduce noise by averaging
func = LiftLinearFunction(num_episodes=5)
```

### 4. Use Linear Controllers First

```python
# Linear is faster and easier for BO
func = LiftLinearFunction()  # Default is linear
```

### 5. Consider Computation Cost

- Linear controller: ~1-2 sec per evaluation
- MLP controller: ~1-2 sec per evaluation
- Episode length affects total time

## Troubleshooting

### Common Issues

1. **"No module named 'mujoco'"**
   ```bash
   pip install mujoco
   ```

2. **"Robosuite not available"**
   - Check that `robosuite/` directory exists in BOMegaBench
   - Verify path is correct

3. **Slow evaluation**
   - Reduce `horizon` (e.g., 200 instead of 500)
   - Reduce `num_episodes` (e.g., 1 instead of 3)
   - Use `controller_type="linear"` instead of `"mlp"`

4. **Low rewards**
   - Random controllers often get negative rewards
   - This is expected - BO should improve over iterations

## Examples

See `examples/robosuite_example.py` for complete working examples:
- Basic usage
- Suite creation
- Controller comparison
- Mock BO loop

## Summary

✅ **Integrated**: 11 robosuite manipulation tasks
✅ **No Duplication**: Complementary to MuJoCo locomotion
✅ **Unconstrained**: All tasks are maximization problems
✅ **Flexible**: Support for linear and MLP controllers
✅ **Ready for BO**: Standard interface compatible with BO libraries

**Recommended starter tasks**: Lift, Door, PickPlace

**Advanced tasks**: Stack, NutAssembly, TwoArm*

**Next steps**: Try the examples and start optimizing!
