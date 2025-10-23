# HumanoidBench 集成

## 概述

HumanoidBench 是一个用于全身运动和操作的仿真人形机器人基准测试套件，已成功集成到 BOMegaBench 中。该基准包含了 **27+ 个任务**，涵盖运动控制和物体操作两大类别。

## 任务分类

### ✅ 运动任务（Locomotion Tasks）13个

这些任务专注于人形机器人的全身运动能力：

```python
LOCOMOTION_TASKS = [
    "walk",            # 行走
    "run",             # 奔跑
    "stand",           # 站立
    "crawl",           # 爬行
    "hurdle",          # 跨栏
    "stair",           # 爬楼梯
    "slide",           # 滑动
    "pole",            # 爬杆
    "maze",            # 迷宫导航
    "sit_simple",      # 简单坐下
    "sit_hard",        # 困难坐下
    "balance_simple",  # 简单平衡
    "balance_hard",    # 困难平衡
]
```

### ✅ 操作任务（Manipulation Tasks）19个

这些任务涉及人形机器人使用手部进行复杂操作：

```python
MANIPULATION_TASKS = [
    "reach",            # 伸手够物
    "push",             # 推物体
    "door",             # 开门
    "cabinet",          # 打开柜子
    "truck",            # 操作货车
    "cube",             # 操作立方体
    "bookshelf_simple", # 简单书架操作
    "bookshelf_hard",   # 困难书架操作
    "basketball",       # 投篮
    "window",           # 开窗
    "spoon",            # 使用勺子
    "kitchen",          # 厨房任务
    "package",          # 包裹操作
    "powerlift",        # 举重
    "room",             # 房间任务
    "insert_normal",    # 正常插入
    "insert_small",     # 小物件插入
    "highbar_simple",   # 简单单杠
    "highbar_hard",     # 困难单杠
]
```

### 可用机器人（Robots）

```python
AVAILABLE_ROBOTS = [
    "h1",         # Unitree H1（无手）
    "h1hand",     # Unitree H1（带灵巧手，默认）
    "h1strong",   # Unitree H1（增强版，可悬挂）
    "h1touch",    # Unitree H1（带触觉传感器）
    "g1",         # Unitree G1（三指手）
]
```

## 与现有 Benchmark 的关系

### BOMegaBench 中的人形机器人任务对比

| Benchmark | 机器人类型 | 任务数量 | 任务复杂度 | 是否重复 |
|-----------|----------|---------|-----------|---------|
| **MuJoCo Control** | 标准 Humanoid-v2/v3/v4 | 1个基础任务 | 简单全身运动 | ❌ **不重复** |
| **HumanoidBench** | H1/G1 (现代人形) | 32个任务 | 复杂运动+操作 | ✅ **独特** |

### 为什么不重复？

1. **机器人模型不同**：
   - MuJoCo Humanoid：经典的简化人形机器人（17 DoF）
   - HumanoidBench H1/G1：现代真实人形机器人（H1Hand: 76 DoF, G1: 44 DoF）

2. **任务类型不同**：
   - MuJoCo Humanoid：只有基础的站立/行走
   - HumanoidBench：包含复杂的全身协调任务（运动 + 操作）

3. **应用场景不同**：
   - MuJoCo Humanoid：简单的控制器优化基准
   - HumanoidBench：真实机器人任务的仿真，包括物体操作、导航、双手协调等

4. **物理真实性**：
   - MuJoCo Humanoid：简化的物理模型
   - HumanoidBench：基于真实商业人形机器人（Unitree H1/G1）的高保真模型

## 集成状态

### ✅ 已完成的集成

1. **代码文件**：
   - `bomegabench/functions/humanoid_bench_tasks.py` ✅ 已创建
   - 完整的 `HumanoidBenchWrapper` 实现
   - 支持线性控制器和 MLP 控制器

2. **导出接口**：
   - `bomegabench/functions/__init__.py` ✅ 已添加导出
   - 尝试导入所有 HumanoidBench 功能
   - 优雅处理依赖缺失

3. **功能实现**：
   - ✅ 自动路径管理（添加 `humanoid-bench` 到 Python 路径）
   - ✅ 环境自动注册（通过 `import humanoid_bench`）
   - ✅ 支持多种控制器类型（linear, mlp）
   - ✅ 可配置评估参数（num_episodes, max_episode_steps）
   - ✅ 多机器人支持（h1, h1hand, g1 等）

## 依赖关系

### 必需依赖

```bash
# HumanoidBench 核心依赖
pip install gymnasium          # 环境接口
pip install mujoco            # 物理引擎
pip install dm-control        # DeepMind Control Suite（提供 rewards 工具）

# 安装 HumanoidBench
cd humanoid-bench
pip install -e .
```

### 可选依赖

```bash
# JAX 支持（用于 MJX 训练）
pip install "jax[cuda12]==0.4.28"  # GPU 版本
# 或
pip install "jax[cpu]==0.4.28"     # CPU 版本

# 强化学习训练框架
pip install -r requirements_jaxrl.txt   # SAC
pip install -r requirements_dreamer.txt # DreamerV3
pip install -r requirements_tdmpc.txt   # TD-MPC2
pip install stable-baselines3==2.3.2    # PPO
```

## 使用示例

### 1. 基本使用

```python
from bomegabench.functions import HumanoidBenchWrapper
import torch

# 创建 HumanoidBench 任务
func = HumanoidBenchWrapper(
    task_name="walk",      # 任务名称
    robot="h1hand",        # 机器人类型
    controller_type="linear",  # 控制器类型
    num_episodes=1,        # 评估回合数
)

print(f"Controller dimension: {func.dim}")  # 控制器参数维度

# 评估控制器
X = torch.randn(1, func.dim)  # 随机控制器参数
Y = func(X)
print(f"Episode return: {Y.item():.2f}")
```

### 2. 不同机器人

```python
# H1 机器人（无手）- 适用于运动任务
func_h1 = HumanoidBenchWrapper(task_name="walk", robot="h1")

# H1 带手 - 适用于操作任务
func_h1hand = HumanoidBenchWrapper(task_name="push", robot="h1hand")

# G1 机器人 - Unitree 新一代
func_g1 = HumanoidBenchWrapper(task_name="cabinet", robot="g1")
```

### 3. MLP 控制器

```python
# 使用 MLP 控制器（更强大但维度更高）
func_mlp = HumanoidBenchWrapper(
    task_name="door",
    robot="h1hand",
    controller_type="mlp",  # 使用 MLP 而非线性控制器
    num_episodes=3,         # 平均 3 个回合
)

print(f"MLP controller dimension: {func_mlp.dim}")
```

### 4. 使用 Suite 创建多个任务

```python
from bomegabench.functions import create_humanoid_bench_suite

# 创建运动任务套件
locomotion_suite = create_humanoid_bench_suite(
    robot="h1hand",
    task_categories=["locomotion"],  # 只包含运动任务
    controller_type="linear",
)

print(f"Locomotion tasks: {len(locomotion_suite.functions)}")

# 创建操作任务套件
manipulation_suite = create_humanoid_bench_suite(
    robot="h1hand",
    task_categories=["manipulation"],  # 只包含操作任务
    controller_type="mlp",
)

print(f"Manipulation tasks: {len(manipulation_suite.functions)}")

# 创建所有任务
all_tasks = create_humanoid_bench_suite(
    robot="h1hand",
    task_categories=None,  # 包含所有任务
)

print(f"All tasks: {len(all_tasks.functions)}")
```

### 5. 便捷类

```python
from bomegabench.functions import (
    H1HandWalkFunction,
    H1HandPushFunction,
    H1HandCabinetFunction,
    H1HandDoorFunction,
    G1WalkFunction,
    G1PushFunction,
)

# 使用预定义的任务类
walk_func = H1HandWalkFunction()
push_func = H1HandPushFunction()
cabinet_func = H1HandCabinetFunction()

# G1 机器人任务
g1_walk = G1WalkFunction()
g1_push = G1PushFunction()
```

## 控制器类型

### Linear Controller（线性控制器）

- **参数维度**: `obs_dim × action_dim + action_dim`
- **公式**: `action = W @ obs + b`
- **优点**: 参数少，优化快
- **适用**: 简单运动任务（walk, stand, run）

**维度示例（H1Hand）**：
- 观察维度 (obs_dim): ~75
- 动作维度 (action_dim): ~19
- 控制器参数: 75 × 19 + 19 = **1444 维**

### MLP Controller（多层感知机控制器）

- **参数维度**: `obs_dim × 64 + 64 + 64 × action_dim + action_dim`
- **结构**: `obs_dim → 64 → action_dim`（带 tanh 激活）
- **优点**: 表达能力强
- **适用**: 复杂操作任务（cabinet, door, kitchen）

**维度示例（H1Hand）**：
- 隐藏层: 64
- 控制器参数: 75 × 64 + 64 + 64 × 19 + 19 = **6083 维**

## 任务元数据

每个 HumanoidBench 任务都包含丰富的元数据：

```python
func = HumanoidBenchWrapper(task_name="walk", robot="h1hand")

metadata = func.metadata
# {
#   "name": "HumanoidBench_h1hand_walk",
#   "source": "HumanoidBench",
#   "type": "continuous",
#   "task_name": "walk",
#   "robot": "h1hand",
#   "env_id": "h1hand-walk-v0",
#   "task_category": "locomotion",
#   "controller_type": "linear",
#   "description": "HumanoidBench walk task with h1hand robot",
#   "reference": "https://arxiv.org/abs/2403.10506",
#   "obs_dim": 75,
#   "action_dim": 19,
#   "controller_dim": 1444,
#   "num_episodes": 1,
#   "max_episode_steps": 1000,
#   "optimization_goal": "maximize_reward",
# }
```

## 无约束确认

所有 HumanoidBench 任务都是**无约束优化问题**：

- **目标**: `max { cumulative_reward(controller_params) }`
- **约束**: 无显式约束
- **输入**: 控制器参数（连续向量）
- **输出**: 累积奖励（标量）

✅ 完全符合 BOMegaBench 的"无约束优化"要求

## 与其他 Benchmark 的互补性

| Benchmark | 领域 | HumanoidBench 关系 |
|-----------|------|----------------|
| **MuJoCo Control** | 简单机器人控制 | ✅ **互补**：HumanoidBench 提供更复杂的人形任务 |
| **Robosuite** | 机械臂操作 | ✅ **互补**：HumanoidBench 是全身人形机器人 |
| **Design-Bench** | 离散序列设计 | ✅ **互补**：HumanoidBench 是连续控制 |
| **Olympus** | 化学/材料 | ✅ **互补**：完全不同的领域 |
| **LassoBench** | 超参数优化 | ✅ **互补**：完全不同的领域 |

## HumanoidBench 的独特价值

1. **真实机器人仿真**：基于商业人形机器人 Unitree H1/G1
2. **全身协调**：需要同时控制运动和操作（双手+双腿）
3. **任务多样性**：32 个不同复杂度的任务
4. **高维控制**：高自由度机器人（H1Hand: 76 DoF）
5. **BO 应用**：
   - 控制器参数优化（策略搜索）
   - 层次化控制器设计
   - 迁移学习基准

## 导入状态检查

由于 HumanoidBench 有较多依赖，`__init__.py` 中使用了优雅的可选导入：

```python
# Import HumanoidBench Tasks (with optional dependency)
try:
    from .humanoid_bench_tasks import (
        create_humanoid_bench_suite,
        HumanoidBenchWrapper,
        # ...
    )
    HUMANOID_BENCH_AVAILABLE = True
except ImportError as e:
    print(f"HumanoidBench not available: {e}")
    HUMANOID_BENCH_AVAILABLE = False
```

**检查是否可用**：

```python
from bomegabench.functions import HUMANOID_BENCH_AVAILABLE

if HUMANOID_BENCH_AVAILABLE:
    from bomegabench.functions import create_humanoid_bench_suite
    suite = create_humanoid_bench_suite()
    print(f"Loaded {len(suite.functions)} HumanoidBench tasks")
else:
    print("HumanoidBench dependencies not installed")
```

## 故障排除

### 问题 1: `ModuleNotFoundError: No module named 'mujoco'`

**解决方案**：
```bash
pip install mujoco
```

### 问题 2: `ModuleNotFoundError: No module named 'dm_control'`

**解决方案**：
```bash
pip install dm-control
```

### 问题 3: `ModuleNotFoundError: No module named 'humanoid_bench'`

**解决方案**：
```bash
cd /path/to/BOMegaBench/humanoid-bench
pip install -e .
```

### 问题 4: 环境注册失败

**解决方案**：确保 `humanoid_bench` 被成功导入，这会自动注册所有环境：
```python
import sys
sys.path.insert(0, 'path/to/humanoid-bench')
import humanoid_bench  # 自动注册环境
import gymnasium as gym

env = gym.make("h1hand-walk-v0")  # 应该可以成功创建
```

## 性能考虑

### 评估时间

- **Linear Controller + 1 episode**: ~1-5 秒/评估
- **MLP Controller + 1 episode**: ~1-5 秒/评估
- **多回合平均 (3 episodes)**: ~3-15 秒/评估

### 并行化建议

由于每次评估需要运行完整的仿真回合，建议：
1. 使用批量并行评估
2. 减少 `num_episodes` 以加快评估
3. 减少 `max_episode_steps`（但可能影响性能评估准确性）

## 参考文献

```bibtex
@article{sferrazza2024humanoidbench,
    title={HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation},
    author={Carmelo Sferrazza and Dun-Ming Huang and Xingyu Lin and Youngwoon Lee and Pieter Abbeel},
    journal={arXiv Preprint arxiv:2403.10506},
    year={2024}
}
```

- **论文**: https://arxiv.org/abs/2403.10506
- **网站**: https://sferrazza.cc/humanoidbench_site/
- **GitHub**: https://github.com/carlosferrazza/humanoid-bench

## 总结

✅ **集成完成**: HumanoidBench 已完全集成到 BOMegaBench
✅ **无重复**: 与现有 benchmark（MuJoCo Control）不重复，提供互补的复杂人形任务
✅ **无约束**: 所有任务都是无约束优化（最大化奖励）
✅ **易用性**: 提供便捷的包装器和创建函数
✅ **真实性**: 基于真实商业人形机器人的高保真仿真
✅ **多样性**: 32+ 个涵盖运动和操作的任务

**推荐使用场景**：
- 全身人形机器人控制器优化
- 复杂操作任务的策略学习
- 运动和操作的协调优化
- 贝叶斯优化在高维控制问题上的基准测试
