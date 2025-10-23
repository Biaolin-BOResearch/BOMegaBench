# MuJoCo Control 快速入门

## 为什么使用 MuJoCo?

**MuJoCo 控制任务是 2020-2025 年间 BO 机器人学论文的标准基准**,出现在绝大多数相关论文中。

## 安装

```bash
pip install 'gymnasium[mujoco]'
```

验证安装:
```bash
python3 -c "import gymnasium as gym; env = gym.make('HalfCheetah-v4'); print('Success!')"
```

## 快速开始

### 1. 最简单的用法

```python
from bomegabench.functions import HalfCheetahLinearFunction
import torch

# 创建 HalfCheetah 任务(BO 论文最常用)
func = HalfCheetahLinearFunction(
    num_episodes=3,  # 平均 3 次运行
    seed=42          # 可重现性
)

print(f"控制器维度: {func.dim}")  # 108 参数

# 采样控制器参数(小随机初始化)
X = torch.randn(1, func.dim) * 0.1

# 评估:运行控制器,返回累积奖励
Y = func(X)
print(f"奖励: {Y.item():.2f}")
```

### 2. 标准环境对比

```python
from bomegabench.functions import (
    HalfCheetahLinearFunction,
    HopperLinearFunction,
    Walker2dLinearFunction,
    AntLinearFunction,
)

environments = [
    ("HalfCheetah", HalfCheetahLinearFunction),  # 最常用
    ("Hopper", HopperLinearFunction),            # 非常常用
    ("Walker2d", Walker2dLinearFunction),        # 常用
    ("Ant", AntLinearFunction),                  # 常用
]

for name, env_class in environments:
    func = env_class(num_episodes=2, seed=42)

    # 测试零控制器
    X = torch.zeros(1, func.dim)
    Y = func(X)

    print(f"{name}: {func.dim} 参数, 奖励={Y.item():.2f}")
```

### 3. 使用 Suite

```python
from bomegabench.functions import create_mujoco_control_suite

# 创建标准 v4 环境套件
suite = create_mujoco_control_suite(
    controller_type="linear",
    versions=["v4"],
    num_episodes=3
)

print(f"任务数量: {len(suite)}")

# 列出所有任务
for task_name in suite.list_functions():
    func = suite.get_function(task_name)
    print(f"{task_name}: {func.dim} 参数")
```

### 4. 线性 vs MLP 控制器

```python
from bomegabench.functions import MuJoCoControlWrapper

# 线性控制器(更少参数,BO 更容易)
func_linear = MuJoCoControlWrapper(
    env_name="Hopper-v4",
    controller_type="linear"
)

# MLP 控制器(更强表达能力,更多参数)
func_mlp = MuJoCoControlWrapper(
    env_name="Hopper-v4",
    controller_type="mlp"
)

print(f"线性: {func_linear.dim} 参数")   # 36
print(f"MLP: {func_mlp.dim} 参数")       # 483
```

## 可用环境

| 环境 | DoF | 控制器维度(线性) | BO 论文流行度 |
|------|-----|----------------|-------------|
| HalfCheetah | 6 | 108 | ⭐⭐⭐⭐⭐ 最常用 |
| Hopper | 3 | 36 | ⭐⭐⭐⭐⭐ 非常常用 |
| Walker2d | 6 | 108 | ⭐⭐⭐⭐ 常用 |
| Ant | 8 | 224 | ⭐⭐⭐⭐ 常用 |
| Humanoid | 17 | 6409 | ⭐⭐⭐ 挑战性 |

所有环境支持 v4 和 v5 版本。

## 控制器类型

### 线性控制器(推荐用于 BO)

**公式**: `action = W @ observation + b`

**参数数量**: `obs_dim × action_dim + action_dim`

**优点**:
- 参数少(36-224 for 标准环境)
- BO 优化更容易
- 评估快速

### MLP 控制器

**结构**: `obs_dim → 32 (tanh) → action_dim`

**参数数量**: 更多(例如 Hopper 483 vs 36)

**优点**:
- 更强表达能力
- 可能达到更好性能

## 典型 BO 设置

### 推荐配置

```python
from bomegabench.functions import HalfCheetahLinearFunction

func = HalfCheetahLinearFunction(
    num_episodes=3,          # 平衡方差和成本
    max_episode_steps=1000,  # 完整回合
    seed=42                  # 可重现性
)

# BO 通常运行 50-200 次迭代
# 每次迭代评估 1-10 个控制器(取决于批大小)
```

### 控制器初始化

```python
# 小随机初始化(推荐)
X_init = torch.randn(5, func.dim) * 0.1

# 零初始化(也合理)
X_zero = torch.zeros(5, func.dim)
```

### 性能基线

零控制器(全0参数)的典型奖励:

| 环境 | 零控制器 | 好控制器(RL训练) |
|------|---------|---------------|
| HalfCheetah | -100 ~ 0 | 2000-5000 |
| Hopper | 0 ~ 50 | 1000-3000 |
| Walker2d | 0 ~ 50 | 2000-4000 |
| Ant | 0 ~ 100 | 3000-6000 |

## 评估成本

单次评估时间(1000步,CPU):

- Hopper: ~0.5秒
- HalfCheetah: ~0.6秒
- Walker2d: ~0.6秒
- Ant: ~1.0秒
- Humanoid: ~2.0秒

**理想的 BO 测试**: 评估昂贵,需要样本高效的算法!

## 高级选项

### 自定义配置

```python
func = MuJoCoControlWrapper(
    env_name="HalfCheetah-v5",  # 使用 v5
    controller_type="linear",
    num_episodes=5,              # 更多回合
    max_episode_steps=500,       # 更短回合
    seed=123,                    # 不同种子
    noise_std=0.01,              # 添加噪声
)
```

### 批量评估(并行 BO)

```python
from bomegabench.functions import HopperLinearFunction

func = HopperLinearFunction(num_episodes=2, seed=42)

# 批量评估多个控制器
X_batch = torch.randn(5, func.dim) * 0.1
Y_batch = func(X_batch)

print(f"批次奖励: {Y_batch}")
print(f"最佳: {Y_batch.max().item():.2f}")
```

## 元数据

每个任务提供详细元数据:

```python
func = HalfCheetahLinearFunction()
print(func.metadata)

# 输出:
# {
#   'name': 'MuJoCo_HalfCheetah-v4',
#   'env_name': 'HalfCheetah-v4',
#   'controller_type': 'linear',
#   'obs_dim': 17,
#   'action_dim': 6,
#   'controller_dim': 108,
#   'optimization_goal': 'maximize_reward',
#   ...
# }
```

## 运行测试

```bash
# 运行所有 MuJoCo 测试
pytest tests/test_mujoco_integration.py -v

# 运行示例
python3 examples/example_mujoco_control.py
```

## 常见问题

### Q: 为什么选择 MuJoCo 而不是其他环境?

A: MuJoCo 控制任务是 **2020-2025 年 BO 机器人学论文的标准基准**,出现在绝大多数论文中。使用标准基准可以:
- 与文献对比
- 获得可信的评估
- 被社区认可

### Q: 评估很慢怎么办?

A: 这是设计如此!BO 的核心挑战就是**样本高效**:
- 每次评估 ~1秒 模拟了真实实验的昂贵性
- 迫使算法在少量评估内找到好解
- 这正是 BO 相比随机搜索的优势所在

### Q: 控制器性能很差怎么办?

A: 随机初始化的控制器性能确实差:
- 零控制器: -100 ~ 100 奖励
- BO 目标: 找到更好的(不一定达到 RL 水平)
- 即使提升到 500+ 也是成功的 BO

### Q: 线性还是 MLP 控制器?

A: 对于 BO:
- **线性**(推荐): 参数少,BO 更容易优化
- **MLP**: 更挑战,测试高维 BO

### Q: 使用 v4 还是 v5?

A:
- **v4**: 大多数论文使用,更好对比
- **v5**: 最新版本,可能有小改进
- 两者差异不大,建议 v4 用于对比,v5 用于新实验

## 更多信息

- **详细文档**: [MUJOCO_INTEGRATION.md](./MUJOCO_INTEGRATION.md)
- **集成总结**: [MUJOCO_INTEGRATION_SUMMARY.md](./MUJOCO_INTEGRATION_SUMMARY.md)
- **示例代码**: [examples/example_mujoco_control.py](./examples/example_mujoco_control.py)
- **Gymnasium 文档**: https://gymnasium.farama.org/environments/mujoco/

## 引用

使用 MuJoCo 环境进行 BO 研究时,请引用:

```bibtex
@article{brockman2016openai,
  title={OpenAI Gym},
  author={Brockman, Greg and Cheung, Vicki and Pettersson, Ludwig and others},
  journal={arXiv preprint arXiv:1606.01540},
  year={2016}
}
```
