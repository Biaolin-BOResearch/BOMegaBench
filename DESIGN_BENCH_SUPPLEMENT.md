# Design-Bench 补充集成

## 概述

在集成 MuJoCo 标准控制任务后,我们补充集成了 Design-Bench 中**不与现有 benchmark 重复**的任务。

## 集成策略

### ✅ 保留的任务(不重复)

#### 1. 材料科学
- **Superconductor** (超导体临界温度预测)
  - 维度: 86 个材料特征
  - Oracle: RandomForest, GP, FullyConnected
  - **不重复**: Olympus 有化学任务但没有超导体

#### 2. 蛋白质设计
- **GFP** (绿色荧光蛋白)
  - 序列长度: 237 个氨基酸
  - One-hot 维度: ~4977
  - Oracle: RandomForest, GP, FullyConnected, LSTM, ResNet, Transformer
  - **不重复**: BOMegaBench 没有蛋白质设计任务

#### 3. DNA/RNA 序列优化
- **TFBind8** (8bp 转录因子结合)
  - 序列长度: 8 bp
  - Oracle: RandomForest, GP, FullyConnected, LSTM, Exact
  - **不重复**: BOMegaBench 没有 DNA 序列任务

- **TFBind10** (10bp 转录因子结合)
  - 序列长度: 10 bp
  - Oracle: Exact
  - **不重复**: 同上

- **UTR** (5' 非翻译区)
  - 序列长度: 50 核苷酸
  - Oracle: RandomForest, GP, FullyConnected, LSTM, ResNet, Transformer
  - **不重复**: BOMegaBench 没有 RNA 序列任务

#### 4. 神经架构搜索
- **CIFARNAS** (CIFAR-10 架构搜索)
  - Oracle: Exact
  - **不重复**: HPOBench 有 NAS 但数据集不同

- **NASBench** (NAS-Bench-101)
  - Oracle: Exact
  - **不重复**: 同上

**总计**: ~25 个任务 (不同 oracle 组合)

### ❌ 排除的任务(与现有重复)

#### 机器人控制任务(与 MuJoCo 重复)

1. **AntMorphology** → ❌ 排除
   - **原因**: MuJoCo Ant 是 BO 论文标准
   - **说明**: Design-Bench 的 Ant 形态学优化与 MuJoCo Ant 控制器优化重复

2. **DKittyMorphology** → ❌ 排除
   - **原因**: 形态学优化,MuJoCo 已覆盖机器人任务
   - **说明**: MuJoCo 提供更标准的机器人控制基准

3. **HopperController** → ❌ 排除
   - **原因**: MuJoCo Hopper 是 BO 论文标准
   - **说明**: Design-Bench 的 Hopper 控制器与 MuJoCo Hopper 完全重复

#### 分子优化任务(与 MolOpt 部分重复)

4. **ChEMBL** → ⚠️ 保留但可选
   - **说明**: 与 MolOpt 有重叠,但 Design-Bench 提供不同的 oracle
   - **当前**: 在任务列表中但需要 deepchem 依赖,实际很少被加载

## 集成任务列表

### 材料科学(3个任务)

```python
# Superconductor tasks
"Superconductor-RandomForest-v0"
"Superconductor-GP-v0"
"Superconductor-FullyConnected-v0"
```

### 蛋白质设计(6个任务)

```python
# GFP tasks
"GFP-RandomForest-v0"
"GFP-GP-v0"
"GFP-FullyConnected-v0"
"GFP-LSTM-v0"
"GFP-ResNet-v0"
"GFP-Transformer-v0"
```

### DNA/RNA 序列(12个任务)

```python
# TFBind8 tasks
"TFBind8-RandomForest-v0"
"TFBind8-GP-v0"
"TFBind8-FullyConnected-v0"
"TFBind8-LSTM-v0"
"TFBind8-Exact-v0"

# TFBind10 tasks
"TFBind10-Exact-v0"

# UTR tasks
"UTR-RandomForest-v0"
"UTR-GP-v0"
"UTR-FullyConnected-v0"
"UTR-LSTM-v0"
"UTR-ResNet-v0"
"UTR-Transformer-v0"
```

### 神经架构搜索(2个任务)

```python
# NAS tasks
"CIFARNAS-Exact-v0"
"NASBench-Exact-v0"
```

## 使用示例

### 基本使用

```python
from bomegabench.functions import SuperconductorRFFunction
import torch

# 创建超导体任务
func = SuperconductorRFFunction()

print(f"Dimension: {func.dim}")  # 86

# 评估
X = torch.randn(1, func.dim)
Y = func(X)
print(f"Critical temperature prediction: {Y.item():.2f}")
```

### 蛋白质设计(One-Hot 编码)

```python
from bomegabench.functions import GFPTransformerFunction
import torch

# 创建 GFP 任务
func = GFPTransformerFunction()

print(f"Encoding: {func.metadata['encoding']}")  # 'one-hot'
print(f"One-hot dim: {func.dim}")  # ~4977

# 创建合法的 one-hot 输入
X_onehot = torch.zeros(1, func.dim)
offset = 0
for num_classes in func.metadata['num_classes_per_position']:
    class_idx = torch.randint(0, num_classes, (1,))
    X_onehot[0, offset + class_idx] = 1.0
    offset += num_classes

# 评估
Y = func(X_onehot)
print(f"Fluorescence: {Y.item():.2f}")
```

### DNA 序列优化

```python
from bomegabench.functions import TFBind8ExactFunction

func = TFBind8ExactFunction()

print(f"Sequence length: {func.metadata['original_input_size']}")  # 8
print(f"One-hot dim: {func.dim}")
```

### 使用 Suite

```python
from bomegabench.functions import create_design_bench_suite

# 创建所有非重复任务
suite = create_design_bench_suite()

print(f"Non-overlapping tasks: {len(suite)}")

# 只创建特定类别
protein_suite = create_design_bench_suite(categories=["protein"])
materials_suite = create_design_bench_suite(categories=["materials"])
sequences_suite = create_design_bench_suite(categories=["sequences"])
nas_suite = create_design_bench_suite(categories=["nas"])
```

## One-Hot 编码

所有离散任务使用 one-hot 编码:

### 编码方式

- 每个离散位置 → 多个维度(对应每个可能的值)
- 总维度 = sum(所有位置的类别数)

### 示例:GFP 蛋白质

```python
# GFP: 237 个氨基酸位置
# 每个位置: 21 种可能的氨基酸
# One-hot 维度: 237 × 21 = 4977

metadata = func.metadata
# {
#   "encoding": "one-hot",
#   "num_classes_per_position": [21, 21, ..., 21],  # 237 个位置
#   "original_input_size": 237,
#   "onehot_dim": 4977,
# }
```

## 与现有 Benchmark 的互补性

### BOMegaBench 覆盖的领域

| Benchmark | 领域 | Design-Bench 互补 |
|-----------|------|----------------|
| MuJoCo | 机器人控制 | ❌ 排除重复的机器人任务 |
| Olympus | 化学实验 | ✅ 添加超导体材料 |
| HPOBench | 超参数优化 | ✅ 添加 NAS(不同数据集) |
| MolOpt | 分子优化 | ⚠️ ChEMBL 部分重叠 |
| LassoBench | Lasso 调参 | ✅ 添加蛋白质/DNA/RNA |
| Database | 数据库配置 | ✅ 添加生物序列优化 |

### Design-Bench 的独特贡献

1. **蛋白质工程**: GFP 大规模序列设计
2. **基因组学**: TFBind, UTR 序列优化
3. **材料科学**: 超导体设计
4. **神经架构搜索**: CIFARNAS, NASBench

## 依赖关系

### 基础依赖

- `torch`, `numpy` (BOMegaBench 已有)
- `design-bench` (包含在 repo 中)
- `scikit-learn` (推荐,用于 RF/GP oracle)

### 可选依赖

- `tensorflow` (神经网络 oracle)
- `deepchem` (ChEMBL 分子任务,暂未启用)

### 不需要的依赖

- ❌ `mujoco-py` (已排除机器人任务)

## 无约束确认

所有保留的 Design-Bench 任务都是**无约束优化问题**:

- Superconductor: `max { critical_temperature(material_features) }`
- GFP: `max { fluorescence(protein_sequence) }`
- TFBind: `max { binding_affinity(DNA_sequence) }`
- UTR: `max { translation_efficiency(RNA_sequence) }`
- CIFARNAS/NASBench: `max { accuracy(architecture) }`

✅ 全部符合"不涉及 constraint"的要求

## 文件清单

### 创建的文件

- `bomegabench/functions/design_bench_tasks.py` (非重复任务)
- `DESIGN_BENCH_SUPPLEMENT.md` (本文件)

### 修改的文件

- `bomegabench/functions/__init__.py` (添加 Design-Bench 导出)

## 总结

成功补充 Design-Bench 的非重复任务:

✅ **保留**: Superconductor, GFP, TFBind, UTR, NAS (~25 tasks)
❌ **排除**: Ant, DKitty, Hopper (与 MuJoCo 重复)
✅ **互补**: 与 MuJoCo, Olympus, HPOBench, MolOpt 互补
✅ **无约束**: 所有任务都是无约束优化
✅ **One-Hot**: 离散任务正确使用 one-hot 编码

**最终结果**: MuJoCo(标准 BO 基准) + Design-Bench(补充任务) = 完整的 BO benchmark 覆盖
