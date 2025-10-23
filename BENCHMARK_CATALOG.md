# BOMegaBench 完整任务目录

本文档详细列举 BOMegaBench 中所有 benchmark 的黑盒函数，包括每个函数的名称、维度、参数信息、类型和范围。

**目录统计：**
- **11 个主要类别**
- **475+ 个基准测试任务**
- **维度范围：2D ~ 19,959D**
- **参数类型：连续型、离散型、混合型**

---

## 目录

1. [Consolidated Functions (72个基础函数)](#1-consolidated-functions)
   - 1.1 [BBOB Functions (24个)](#11-bbob-functions)
   - 1.2 [BoTorch Additional (6个)](#12-botorch-additional-functions)
   - 1.3 [Classical Additional (32个)](#13-classical-additional-functions)
   - 1.4 [Classical Core (10个)](#14-classical-core-functions)
2. [LassoBench (13个任务)](#2-lassobench)
3. [HPOBench (50+个任务)](#3-hpobench)
4. [HPO Benchmarks (100+个任务)](#4-hpo-benchmarks-bayesmark)
5. [Database Tuning (2个数据库)](#5-database-tuning)
6. [Olympus Surfaces (20+个表面)](#6-olympus-surfaces)
7. [Olympus Datasets (40+个真实数据集)](#7-olympus-datasets)
8. [MuJoCo Control (5个机器人)](#8-mujoco-control)
9. [Robosuite Manipulation (11个任务)](#9-robosuite-manipulation)
10. [HumanoidBench (32个任务)](#10-humanoidbench)
11. [Design-Bench (25+个任务)](#11-design-bench)

---

## 1. Consolidated Functions

**文件位置：** `bomegabench/functions/consolidated/`

**总数：** 72 个独特的基准函数

**用途：** 经典的黑盒优化测试函数，广泛用于评估优化算法性能

### 1.1 BBOB Functions

**文件：** `bbob_functions.py`

**数量：** 24 个函数（BBOB测试套件）

**维度：** 灵活（默认支持 2, 4, 8, 30, 53 维）

| 函数ID | 函数名 | 维度 | 参数类型 | 取值范围 | 特性 | 描述 |
|--------|--------|------|----------|----------|------|------|
| F01 | SphereRaw | d | 连续 | [-5, 5]^d | 单峰、可分离 | 球函数，最简单的测试函数 |
| F02 | EllipsoidSeparableRaw | d | 连续 | [-5, 5]^d | 单峰、可分离、病态 | 椭球函数（可分离） |
| F03 | RastriginSeparableRaw | d | 连续 | [-5, 5]^d | 多峰、可分离 | Rastrigin函数（可分离） |
| F04 | SkewRastriginBuecheRaw | d | 连续 | [-5, 5]^d | 多峰、不对称 | 倾斜Rastrigin函数 |
| F05 | LinearSlopeRaw | d | 连续 | [-5, 5]^d | 单峰、线性 | 线性斜坡函数 |
| F06 | AttractiveSectorRaw | d | 连续 | [-5, 5]^d | 单峰、不对称 | 吸引区域函数 |
| F07 | StepEllipsoidRaw | d | 连续 | [-5, 5]^d | 单峰、平台 | 阶梯椭球函数 |
| F08 | RosenbrockRaw | d | 连续 | [-5, 5]^d | 单峰、不可分离、峡谷 | Rosenbrock函数 |
| F09 | RosenbrockRotatedRaw | d | 连续 | [-5, 5]^d | 单峰、不可分离、旋转 | Rosenbrock旋转函数 |
| F10 | EllipsoidRaw | d | 连续 | [-5, 5]^d | 单峰、不可分离、病态 | 椭球函数 |
| F11 | DiscusRaw | d | 连续 | [-5, 5]^d | 单峰、一短轴 | 铁饼函数 |
| F12 | BentCigarRaw | d | 连续 | [-5, 5]^d | 单峰、一长轴 | 弯曲雪茄函数 |
| F13 | SharpRidgeRaw | d | 连续 | [-5, 5]^d | 单峰、脊 | 尖锐脊函数 |
| F14 | DifferentPowersRaw | d | 连续 | [-5, 5]^d | 单峰、变敏感度 | 不同幂函数 |
| F15 | RastriginRaw | d | 连续 | [-5, 5]^d | 多峰、不可分离 | Rastrigin函数 |
| F16 | WeierstrassRaw | d | 连续 | [-0.5, 0.5]^d | 多峰、分形 | Weierstrass函数 |
| F17 | SchafferF7Cond10Raw | d | 连续 | [-5, 5]^d | 多峰、条件数10 | Schaffer F7（条件数10） |
| F18 | SchafferF7Cond1000Raw | d | 连续 | [-5, 5]^d | 多峰、高条件数 | Schaffer F7（条件数1000） |
| F19 | GriewankRosenbrockRaw | d | 连续 | [-5, 5]^d | 多峰、组合 | Griewank-Rosenbrock组合 |
| F20 | SchwefelRaw | d | 连续 | [-5, 5]^d | 多峰、欺骗性 | Schwefel函数 |
| F21 | Gallagher101Raw | d | 连续 | [-5, 5]^d | 多峰、101个峰 | Gallagher 101峰函数 |
| F22 | Gallagher21Raw | d | 连续 | [-5, 5]^d | 多峰、21个峰 | Gallagher 21峰函数 |
| F23 | KatsuuraRaw | d | 连续 | [-5, 5]^d | 多峰、病态 | Katsuura函数 |
| F24 | LunacekBiRastriginRaw | d | 连续 | [-5, 5]^d | 多峰、双漏斗 | Lunacek双Rastrigin函数 |

**参数说明：**
- 所有维度的参数都是未命名的连续变量
- `d` 表示维度，可灵活配置

### 1.2 BoTorch Additional Functions

**文件：** `botorch_additional.py`

**数量：** 6 个函数

| 函数名 | 维度 | 参数类型 | 取值范围 | 特性 | 参数详情 |
|--------|------|----------|----------|------|----------|
| BukinFunction | 2 (固定) | 连续 | x₁:[-15,-5]<br>x₂:[-3,3] | 多峰、不可微 | x₁, x₂: 未命名连续参数 |
| Cosine8Function | d≥8 | 连续 | [-1, 1]^d | 多峰、可分离 | 所有维度：未命名连续参数 |
| ThreeHumpCamelFunction | 2 (固定) | 连续 | [-5, 5]² | 多峰、3个局部极小 | x₁, x₂: 未命名连续参数 |
| **AckleyMixedFunction** | d≥4 | **混合** | 前(d-3)维: {0,1} (离散)<br>后3维: [0,1] (连续) | 混合整数 | **前(d-3)维：二元离散（one-hot编码）**<br>**后3维：连续** |
| **LabsFunction** | d≥10 | **离散/二元** | {0, 1}^d | 二元、组合 | **所有维度：二元选择** |
| ShekelFunction | 4 (固定) | 连续 | [0, 10]⁴ | 多峰、m个峰(5/7/10) | 所有4维：未命名连续参数 |

**特殊编码说明：**
- **AckleyMixedFunction**: 离散维度使用 one-hot 编码转换为连续 [0,1]
- **LabsFunction**: 二元优化问题，优化低自相关二进制序列（LABS问题）

### 1.3 Classical Additional Functions

**文件：** `classical_additional.py`

**数量：** 32 个函数

#### 1.3.1 变维度函数（可用于多种维度）

| 函数名 | 维度 | 参数类型 | 取值范围 | 特性 |
|--------|------|----------|----------|------|
| Schwefel12Function | d | 连续 | [-100, 100]^d | 单峰、双重求和 |
| Schwefel220Function | d | 连续 | [-100, 100]^d | 单峰、绝对值 |
| Schwefel221Function | d | 连续 | [-100, 100]^d | 单峰、最大模 |
| Schwefel222Function | d | 连续 | [-10, 10]^d | 单峰、求和-求积 |
| Schwefel223Function | d | 连续 | [-10, 10]^d | 单峰、高次幂 |
| Schwefel226Function | d | 连续 | [-500, 500]^d | 多峰、欺骗性 |
| LevyN13Function | d | 连续 | [-10, 10]^d | 多峰、三角函数 |
| Alpine1Function | d | 连续 | [-10, 10]^d | 多峰、不可微 |
| Alpine2Function | d | 连续 | [0, 10]^d | 多峰、求积 |
| TridFunction | d | 连续 | [-d², d²]^d | 单峰、特殊结构 |
| PowellFunction | d (需被4整除) | 连续 | [-4, 5]^d | 单峰、4元素分组 |

#### 1.3.2 固定2维函数

| 函数名 | 维度 | 参数类型 | 取值范围 | 特性 | 全局最优点 |
|--------|------|----------|----------|------|------------|
| SchafferF1Function | 2 | 连续 | [-100, 100]² | 多峰、2D | (0, 0) |
| SchafferF2Function | 2 | 连续 | [-100, 100]² | 多峰、2D | (0, 0) |
| SchafferF3Function | 2 | 连续 | [-100, 100]² | 多峰、2D | (0, 1.253) |
| SchafferF4Function | 2 | 连续 | [-100, 100]² | 多峰、2D | (0, 0) |
| SchafferF5Function | 2 | 连续 | [-100, 100]² | 多峰、2D | ~(0, 0) |
| SchafferF7Function | 2 | 连续 | [-100, 100]² | 多峰、2D、调制 | (0, 0) |
| CrossInTrayFunction | 2 | 连续 | [-10, 10]² | 多峰、4个极小 | (±1.349, ±1.349) |
| EggholderFunction | 2 | 连续 | [-512, 512]² | 多峰、2D | (512, 404.2319) |
| HolderTableFunction | 2 | 连续 | [-10, 10]² | 多峰、4个极小 | (±8.055, ±9.665) |
| DropWaveFunction | 2 | 连续 | [-5.12, 5.12]² | 多峰、振荡 | (0, 0) |
| ShubertFunction | 2 | 连续 | [-10, 10]² | 多峰、18个极小 | 多个等价最优点 |
| BoothFunction | 2 | 连续 | [-10, 10]² | 单峰、二次 | (1, 3) |
| MatyasFunction | 2 | 连续 | [-10, 10]² | 单峰、二次 | (0, 0) |
| McCormickFunction | 2 | 连续 | x₁:[-1.5,4]<br>x₂:[-3,4] | 多峰、三角函数 | (-0.547, -1.547) |
| SixHumpCamelFunction | 2 | 连续 | x₁:[-3,3]<br>x₂:[-2,2] | 多峰、6个驼峰 | (0.0898, -0.7126) |
| BraninFunction | 2 | 连续 | x₁:[-5,10]<br>x₂:[0,15] | 多峰、3个极小 | 3个等价最优点 |

#### 1.3.3 特定维度函数

| 函数名 | 维度 | 参数类型 | 取值范围 | 特性 |
|--------|------|----------|----------|------|
| Hartmann3DFunction | 3 | 连续 | [0, 1]³ | 多峰、4个极小 |
| Hartmann4DFunction | 4 | 连续 | [0, 1]⁴ | 多峰、4个极小 |
| Hartmann6DFunction | 6 | 连续 | [0, 1]⁶ | 多峰、6个极小 |

### 1.4 Classical Core Functions

**文件：** `classical_core.py`

**数量：** 10 个函数

| 函数名 | 维度 | 参数类型 | 取值范围 | 特性 | 全局最优 |
|--------|------|----------|----------|------|----------|
| StyblinskiTangFunction | d | 连续 | [-5, 5]^d | 多峰、可分离 | x=-2.903534 (所有维度) |
| LevyFunction | d | 连续 | [-10, 10]^d | 多峰、复杂 | (1, 1, ..., 1) |
| MichalewiczFunction | d | 连续 | [0, π]^d | 多峰、参数化 | 依赖维度 |
| ZakharovFunction | d | 连续 | [-5, 10]^d | 单峰 | (0, 0, ..., 0) |
| DixonPriceFunction | d | 连续 | [-10, 10]^d | 单峰、相邻交互 | 特定模式 |
| SalomonFunction | d | 连续 | [-100, 100]^d | 多峰 | (0, 0, ..., 0) |
| SchafferF6Function | 2 | 连续 | [-100, 100]² | 多峰、2D | (0, 0) |
| EasomFunction | 2 | 连续 | [-100, 100]² | 多峰、尖锐极小 | (π, π) |
| BealeFunction | 2 | 连续 | [-4.5, 4.5]² | 多峰、尖锐峰 | (3, 0.5) |
| GoldsteinPriceFunction | 2 | 连续 | [-2, 2]² | 多峰、2D | (0, -1) |

---

## 2. LassoBench

**文件位置：** `bomegabench/functions/lasso_bench.py`

**总数：** 13 个任务（4个合成 × 2变体 + 5个真实数据集）

**用途：** LASSO 回归超参数优化（高维稀疏优化）

### 2.1 合成基准（Synthetic Benchmarks）

**数量：** 4 个基准 × 2 个变体 = 8 个任务

| 基准名称 | 总维度 | 活跃维度 | 参数类型 | 取值范围 | 噪声变体 | 描述 |
|----------|--------|----------|----------|----------|----------|------|
| synt_simple | **60** | 3 | 连续 | [-1, 1]⁶⁰ | 有噪/无噪 | 简单LASSO任务 |
| synt_medium | **100** | 5 | 连续 | [-1, 1]¹⁰⁰ | 有噪/无噪 | 中等LASSO任务 |
| synt_high | **300** | 15 | 连续 | [-1, 1]³⁰⁰ | 有噪/无噪 | 高维LASSO任务 |
| synt_hard | **1000** | 50 | 连续 | [-1, 1]¹⁰⁰⁰ | 有噪/无噪 | 困难LASSO任务 |

**参数说明：**
- 所有参数：LASSO 回归系数（未命名）
- 稀疏性：只有"活跃维度"数量的参数是非零的
- 优化目标：最小化回归误差

**示例参数（synt_simple，60维）：**
- 参数 x₁, x₂, ..., x₆₀：LASSO回归系数
- 类型：连续型
- 范围：[-1, 1]
- 含义：线性回归模型的权重，只有3个是真正起作用的

### 2.2 真实数据集（Real-World Datasets）

**数量：** 5 个数据集

| 数据集 | 总维度 | 近似活跃维度 | 参数类型 | 取值范围 | 应用领域 |
|--------|--------|--------------|----------|----------|----------|
| Diabetes | **8** | ~5 | 连续 | [-1, 1]⁸ | 医学（糖尿病预测） |
| Breast_cancer | **10** | ~3 | 连续 | [-1, 1]¹⁰ | 医学（乳腺癌） |
| DNA | **180** | ~43 | 连续 | [-1, 1]¹⁸⁰ | 生物信息学 |
| Leukemia | **7129** | ~22 | 连续 | [-1, 1]⁷¹²⁹ | 医学（白血病） |
| RCV1 | **19959** | ~75 | 连续 | [-1, 1]¹⁹⁹⁵⁹ | 文本分类 |

**特点：**
- 极高维度（最高19,959维）
- 真实数据驱动
- 稀疏解结构

---

## 3. HPOBench

**文件位置：** `bomegabench/functions/hpobench_benchmarks.py`

**总数：** 50+ 个任务

**用途：** 超参数优化基准（机器学习/RL/NAS）

**编码方式：**
- 连续参数：归一化到 [0, 1]
- 整数参数：one-hot 编码
- 类别参数：one-hot 编码

### 3.1 机器学习基准（ML Benchmarks）

**数量：** 8 个模型 × 4 个数据集 = 32 个任务

#### 模型列表
1. XGBoost (Black-Box)
2. XGBoost (Multi-Fidelity)
3. SVM (Black-Box)
4. SVM (Multi-Fidelity)
5. RandomForest
6. LogisticRegression
7. NeuralNetwork
8. HistGradientBoosting

#### 数据集（OpenML任务ID）
- Task 31
- Task 3917
- Task 9952
- Task 146818

#### XGBoost 超参数示例

| 参数名 | 类型 | 原始范围 | 编码后维度 | 含义 |
|--------|------|----------|------------|------|
| n_estimators | 整数 | [50, 500] | 多维 (one-hot) | 树的数量 |
| max_depth | 整数 | [1, 15] | 多维 (one-hot) | 树的最大深度 |
| learning_rate | 浮点 | [0.001, 1.0] | 1维 [0,1] | 学习率 |
| min_child_weight | 浮点 | [0.1, 10.0] | 1维 [0,1] | 子节点最小权重 |
| gamma | 浮点 | [0.0, 5.0] | 1维 [0,1] | 分裂最小损失减少 |
| subsample | 浮点 | [0.5, 1.0] | 1维 [0,1] | 样本子采样比例 |
| colsample_bytree | 浮点 | [0.5, 1.0] | 1维 [0,1] | 特征子采样比例 |

**总维度**：取决于 one-hot 编码后的整数参数，通常 20-50 维

#### SVM 超参数示例

| 参数名 | 类型 | 原始范围 | 含义 |
|--------|------|----------|------|
| C | 浮点 | [0.001, 1000] | 正则化参数 |
| gamma | 浮点 | [0.0001, 1.0] | RBF核参数 |
| kernel | 类别 | {linear, rbf, poly} | 核函数类型 (one-hot) |

### 3.2 异常检测基准（Outlier Detection）

**数量：** 2 个模型 × 4 个数据集 = 8 个任务

**模型：**
- KDE (Kernel Density Estimation)
- OneClassSVM

**数据集：**
- breastw
- ionosphere
- pima
- wbc

**超参数示例（KDE）：**
- bandwidth: 浮点 [0.01, 1.0] → 归一化到 [0,1]
- kernel: 类别 {gaussian, tophat, epanechnikov} → one-hot编码

### 3.3 神经架构搜索（NAS Benchmarks）

**数量：** 8 个基准

| 基准名称 | 参数类型 | 编码维度 | 描述 |
|----------|----------|----------|------|
| NASBench101-A | 离散（架构选择） | 可变 | CIFAR-10 NAS |
| NASBench101-B | 离散（架构选择） | 可变 | CIFAR-10 NAS |
| NASBench101-C | 离散（架构选择） | 可变 | CIFAR-10 NAS |
| NASBench201 | 离散（架构选择） | 可变 | CIFAR-10/100, ImageNet16 |
| SliceLocalization | 离散 | 可变 | Tabular NAS |
| ProteinStructure | 离散 | 可变 | Tabular NAS |
| NavalPropulsion | 离散 | 可变 | Tabular NAS |
| ParkinsonsTelemonitoring | 离散 | 可变 | Tabular NAS |

**参数说明：**
- 架构参数：操作类型选择（卷积、池化、跳跃连接等）
- 编码：每个选择位置使用 one-hot 编码

### 3.4 强化学习基准（RL Benchmarks）

**数量：** 2+ 个基准

| 基准名称 | 参数数量 | 参数类型 | 描述 |
|----------|----------|----------|------|
| Cartpole-Reduced | ~8 | 混合 | CartPole RL超参数（简化） |
| Cartpole-Full | ~15 | 混合 | CartPole RL超参数（完整） |

**超参数示例（RL算法）：**
- learning_rate: 浮点
- discount_factor (gamma): 浮点 [0.9, 0.999]
- batch_size: 整数 → one-hot
- network_width: 整数 → one-hot
- activation: 类别 {relu, tanh} → one-hot

### 3.5 代理基准（Surrogate Benchmarks）

**数量：** 3 个基准

| 基准名称 | 描述 |
|----------|------|
| SurrogateSVM | SVM代理模型 |
| ParamNet-Adult | 神经网络（Adult数据集） |
| ParamNet-Higgs | 神经网络（Higgs数据集） |

---

## 4. HPO Benchmarks (Bayesmark)

**文件位置：** `bomegabench/functions/hpo_benchmarks.py`

**总数：** 100+ 个任务组合

**用途：** 标准化HPO基准（Bayesmark框架）

### 4.1 模型列表（7个）

1. **DT** (Decision Tree)
2. **MLP-sgd** (Multi-Layer Perceptron with SGD)
3. **RF** (Random Forest)
4. **SVM** (Support Vector Machine)
5. **ada** (AdaBoost)
6. **kNN** (k-Nearest Neighbors)
7. **lasso** (Lasso Regression)
8. **linear** (Linear Regression)

### 4.2 数据集

#### 分类数据集（4个）
- iris (鸢尾花)
- wine (葡萄酒)
- digits (手写数字)
- breast (乳腺癌)

**评估指标：** nll (负对数似然), acc (准确率)

#### 回归数据集（2个）
- boston (波士顿房价)
- diabetes (糖尿病)

**评估指标：** mse (均方误差), mae (平均绝对误差)

### 4.3 超参数示例

#### Random Forest 超参数

| 参数名 | 类型 | 原始范围 | 编码方式 | 含义 |
|--------|------|----------|----------|------|
| n_estimators | 整数 | [10, 100] | one-hot → [0,1]^n | 树的数量 |
| max_depth | 整数 | [1, 50] | one-hot → [0,1]^n | 最大深度 |
| min_samples_split | 整数 | [2, 20] | one-hot → [0,1]^n | 最小分裂样本数 |
| min_samples_leaf | 整数 | [1, 10] | one-hot → [0,1]^n | 叶节点最小样本数 |
| max_features | 类别 | {auto, sqrt, log2} | one-hot → [0,1]³ | 最大特征数选择策略 |

#### MLP-sgd 超参数

| 参数名 | 类型 | 原始范围 | 编码方式 | 含义 |
|--------|------|----------|----------|------|
| hidden_layer_sizes | 整数组 | [10, 100] | one-hot | 隐藏层大小 |
| alpha | 浮点 | [1e-5, 1e-1] | 归一化 [0,1] | L2正则化系数 |
| learning_rate_init | 浮点 | [1e-4, 1e-1] | 归一化 [0,1] | 初始学习率 |
| activation | 类别 | {relu, tanh, logistic} | one-hot → [0,1]³ | 激活函数 |
| solver | 类别 | {adam, sgd} | one-hot → [0,1]² | 优化器 |

**总任务数：** 7模型 × 6数据集 × 多个指标 ≈ 100+ 个任务

---

## 5. Database Tuning

**文件位置：** `bomegabench/functions/database_tuning.py`

**总数：** 2 个数据库系统

**用途：** 数据库配置参数优化

### 5.1 PostgreSQL 配置参数

**参数数量：** 8 个关键配置参数

| 参数名 | 类型 | 原始范围 | 单位 | 类别 | 含义 |
|--------|------|----------|------|------|------|
| **shared_buffers_mb** | 整数 | [128, 16384] | MB | 内存 | 共享内存缓冲区大小 |
| **effective_cache_size_mb** | 整数 | [256, 65536] | MB | 内存 | 规划器的缓存假设大小 |
| **work_mem_mb** | 整数 | [1, 2048] | MB | 内存 | 排序/哈希操作的内存 |
| **max_connections** | 整数 | [10, 1000] | - | 连接 | 最大并发连接数 |
| **random_page_cost** | 浮点 | [0.1, 10.0] | - | 规划器 | 随机页访问代价估计 |
| **effective_io_concurrency** | 整数 | [0, 1000] | - | IO | 并发磁盘IO操作数 |
| **checkpoint_completion_target** | 浮点 | [0.0, 1.0] | - | WAL | 检查点完成目标比例 |
| **default_statistics_target** | 整数 | [10, 10000] | - | 规划器 | 默认统计目标 |

**编码方式：**
- 整数参数：one-hot 编码到连续 [0,1]
- 浮点参数：直接归一化到 [0,1]

**总维度**：约 100-200 维（取决于整数参数的离散化粒度）

### 5.2 MySQL 配置参数

**参数数量：** 5 个关键配置参数

| 参数名 | 类型 | 原始范围 | 单位 | 类别 | 含义 |
|--------|------|----------|------|------|------|
| **innodb_buffer_pool_size_mb** | 整数 | [128, 32768] | MB | 内存 | InnoDB缓冲池大小 |
| **innodb_log_file_size_mb** | 整数 | [4, 4096] | MB | IO | InnoDB日志文件大小 |
| **max_connections** | 整数 | [10, 10000] | - | 连接 | 最大并发连接数 |
| **innodb_io_capacity** | 整数 | [100, 20000] | IOPS | IO | 每秒IO操作数 |
| **query_cache_size_mb** | 整数 | [0, 1024] | MB | 缓存 | 查询缓存大小 |

**优化目标：** 最大化数据库吞吐量或最小化查询延迟

---

## 6. Olympus Surfaces

**文件位置：** `bomegabench/functions/olympus_surfaces.py`

**总数：** 20+ 个合成测试表面

**用途：** 化学/材料科学实验规划的合成基准

### 7.1 分类表面（Categorical Surfaces）（5个）

| 表面名称 | 基础函数 | 维度 | 参数类型 | 选项数 | 描述 |
|----------|----------|------|----------|--------|------|
| CatAckley | Ackley | 灵活 | **类别** | 每维可配置 | Ackley函数的类别版本 |
| CatCamel | Camel | 2 | **类别** | 每维可配置 | Camel函数的类别版本 |
| CatDejong | Dejong (Sphere) | 灵活 | **类别** | 每维可配置 | 球函数的类别版本 |
| CatMichalewicz | Michalewicz | 灵活 | **类别** | 每维可配置 | Michalewicz函数的类别版本 |
| CatSlope | Slope | 灵活 | **类别** | 每维可配置 | 斜坡函数的类别版本 |

**参数说明（以CatAckley为例）：**
- 维度：d（可配置）
- 每个维度：类别选择
- 选项数：num_opts（可配置，如3, 5, 10）
- 示例：如果 d=3, num_opts=5，则每维有5个离散选项：{opt0, opt1, opt2, opt3, opt4}

### 7.2 离散表面（Discrete Surfaces）（3个）

| 表面名称 | 维度 | 参数类型 | 取值范围 | 描述 |
|----------|------|----------|----------|------|
| DiscreteAckley | 灵活 | 离散 | 整数网格 | 离散化的Ackley函数 |
| DiscreteDoubleWell | 灵活 | 离散 | 整数网格 | 离散化的双阱函数 |
| DiscreteMichalewicz | 灵活 | 离散 | 整数网格 | 离散化的Michalewicz函数 |

### 7.3 山峰/地形表面（Mountain/Terrain）（6个）

| 表面名称 | 维度 | 参数类型 | 取值范围 | 描述 |
|----------|------|----------|----------|------|
| **Denali** | 2 (固定) | 连续 | [0, 1]² | 北美最高峰地形 |
| **Everest** | 2 (固定) | 连续 | [0, 1]² | 珠穆朗玛峰地形 |
| **K2** | 2 (固定) | 连续 | [0, 1]² | 乔戈里峰地形 |
| **Kilimanjaro** | 2 (固定) | 连续 | [0, 1]² | 乞力马扎罗山地形 |
| **Matterhorn** | 2 (固定) | 连续 | [0, 1]² | 马特洪峰地形 |
| **MontBlanc** | 2 (固定) | 连续 | [0, 1]² | 勃朗峰地形 |

**参数说明：**
- x₁, x₂: 二维坐标
- 范围：[0, 1]²
- 含义：模拟真实山峰地形的复杂景观

### 7.4 特殊函数（5个）

| 表面名称 | 维度 | 参数类型 | 取值范围 | 描述 |
|----------|------|----------|----------|------|
| AckleyPath | 2 | 连续 | [0, 1]² | Ackley函数带路径 |
| GaussianMixture | 2 | 连续 | [0, 1]² | 高斯混合表面 |
| HyperEllipsoid | 灵活 | 连续 | [0, 1]^d | 超椭球函数 |
| LinearFunnel | 灵活 | 连续 | [0, 1]^d | 线性漏斗函数 |
| NarrowFunnel | 灵活 | 连续 | [0, 1]^d | 狭窄漏斗函数 |

---

## 7. Olympus Datasets

**文件位置：** `bomegabench/functions/olympus_datasets.py`

**总数：** 40+ 个真实实验数据集

**用途：** 真实世界的化学/材料科学实验优化

**特点：** 所有数据来自真实实验，非合成

### 8.1 化学反应（Chemical Reactions）（14个）

#### Buchwald系列（5个）

| 数据集名称 | 参数数量 | 参数类型 | 描述 |
|-----------|----------|----------|------|
| buchwald_a | 4 | 类别 | Buchwald偶联反应A |
| buchwald_b | 4 | 类别 | Buchwald偶联反应B |
| buchwald_c | 4 | 类别 | Buchwald偶联反应C |
| buchwald_d | 4 | 类别 | Buchwald偶联反应D |
| buchwald_e | 4 | 类别 | Buchwald偶联反应E |

**参数示例（buchwald）：**
1. **ligand**（配体）：类别，~10个选项
2. **base**（碱）：类别，~5个选项
3. **solvent**（溶剂）：类别，~8个选项
4. **concentration**（浓度）：连续，[0.1, 2.0] M

#### Suzuki系列（7个）

| 数据集名称 | 参数数量 | 参数类型 | 描述 |
|-----------|----------|----------|------|
| suzuki | 4 | 混合 | Suzuki偶联反应 |
| suzuki_edbo | 4 | 混合 | Suzuki反应（EDBO数据） |
| suzuki_i/ii/iii/iv | 4 | 混合 | Suzuki反应变体I-IV |

#### 其他反应（2个）

| 数据集名称 | 参数数量 | 参数类型 | 描述 |
|-----------|----------|----------|------|
| benzylation | 3 | 类别 | 苄基化反应 |
| alkox | 4 | 混合 | 烷氧基化反应 |
| snar | 3 | 类别 | 亲核芳香取代反应 |

### 8.2 材料科学（Materials Science）（8个）

| 数据集名称 | 参数数量 | 参数类型 | 优化目标 | 描述 |
|-----------|----------|----------|----------|------|
| **perovskites** | 4 | 混合 | 带隙能量 | 钙钛矿材料设计 |
| **fullerenes** | 4 | 混合 | 稳定性 | 富勒烯衍生物 |
| **dye_lasers** | 3 | 连续 | 激光效率 | 染料激光材料 |
| **redoxmers** | 3 | 类别 | 氧化还原电位 | 氧化还原活性分子 |
| colors_bob | 4 | 混合 | 颜色匹配 | 颜色材料设计 |
| colors_n9 | 9 | 连续 | 颜色匹配 | 9参数颜色配方 |
| thin_film | 5 | 连续 | 薄膜性质 | 薄膜材料 |
| crossed_barrel | 3 | 混合 | 材料性能 | 交叉桶形反应器 |

**参数示例（perovskites）：**
1. **A_site**（A位离子）：类别，如 {MA, FA, Cs}
2. **B_site**（B位离子）：类别，如 {Pb, Sn}
3. **X_site**（X位离子）：类别，如 {I, Br, Cl}
4. **composition**（组成比例）：连续，[0, 1]

### 8.3 光伏材料（Photovoltaics）（4个）

| 数据集名称 | 参数数量 | 参数类型 | 优化目标 | 描述 |
|-----------|----------|----------|----------|------|
| photo_pce10 | 3 | 混合 | 光电转换效率 | 有机光伏（PCE>10%） |
| photo_wf3 | 3 | 混合 | 功函数 | 有机光伏（功函数） |
| p3ht | 4 | 连续 | 效率 | P3HT聚合物太阳能电池 |
| mmli_opv | 5 | 混合 | 效率 | 有机光伏多参数 |

### 8.4 纳米粒子（Nanoparticles）（3个）

| 数据集名称 | 参数数量 | 参数类型 | 优化目标 | 描述 |
|-----------|----------|----------|----------|------|
| **agnp** | 4 | 连续 | 粒径/形状 | 银纳米粒子合成 |
| **lnp3** | 3 | 连续 | 粒径 | 脂质纳米粒子 |
| **autoam** | 6 | 混合 | 材料性能 | 自动化增材制造 |

**参数示例（agnp - 银纳米粒子）：**
1. **silver_nitrate_conc**（硝酸银浓度）：连续，[0.1, 10] mM
2. **reducing_agent_conc**（还原剂浓度）：连续，[0.5, 50] mM
3. **temperature**（温度）：连续，[20, 100] °C
4. **pH**：连续，[3, 11]

### 8.5 电化学（Electrochemistry）（5个）

| 数据集名称 | 参数数量 | 参数类型 | 优化目标 | 描述 |
|-----------|----------|----------|----------|------|
| electrochem | 4 | 混合 | 电化学性能 | 一般电化学系统 |
| oer_plate_3496 | 5 | 混合 | 析氧反应 | OER催化剂板3496 |
| oer_plate_3851 | 5 | 混合 | 析氧反应 | OER催化剂板3851 |
| oer_plate_3860 | 5 | 混合 | 析氧反应 | OER催化剂板3860 |
| oer_plate_4098 | 5 | 混合 | 析氧反应 | OER催化剂板4098 |

### 8.6 液体/溶剂系统（7个）

| 数据集名称 | 参数数量 | 参数类型 | 优化目标 | 描述 |
|-----------|----------|----------|----------|------|
| liquid_ace_100 | 3 | 连续 | 溶解度 | 丙酮溶液（100数据点） |
| liquid_dce | 3 | 连续 | 溶解度 | 二氯乙烷溶液 |
| liquid_hep_100 | 3 | 连续 | 溶解度 | 正庚烷溶液（100点） |
| liquid_thf_100 | 3 | 连续 | 溶解度 | 四氢呋喃（100点） |
| liquid_thf_500 | 3 | 连续 | 溶解度 | 四氢呋喃（500点） |
| liquid_toluene | 3 | 连续 | 溶解度 | 甲苯溶液 |
| liquid_water | 3 | 连续 | 溶解度 | 水溶液 |

### 8.7 其他应用（2个）

| 数据集名称 | 参数数量 | 参数类型 | 优化目标 | 描述 |
|-----------|----------|----------|----------|------|
| hplc | 6 | 混合 | 分离效果 | 高效液相色谱优化 |
| vapdiff_crystal | 4 | 混合 | 晶体质量 | 蒸气扩散结晶 |

---

## 8. MuJoCo Control

**文件位置：** `bomegabench/functions/mujoco_control.py`

**总数：** 5 个机器人 × 2 个控制器类型 = 10 个任务

**用途：** 机器人运动控制器参数优化

**说明：** 优化控制器参数使机器人在仿真环境中获得最大累积奖励

### 9.1 环境列表

| 环境名称 | 机器人类型 | 观察维度 | 动作维度 | 描述 |
|----------|-----------|----------|----------|------|
| HalfCheetah-v4/v5 | 四足（半猎豹） | ~17 | 6 | 快速奔跑 |
| Hopper-v4/v5 | 单腿跳跃 | ~11 | 3 | 单腿向前跳跃 |
| Walker2d-v4/v5 | 双足行走 | ~17 | 6 | 双足平衡行走 |
| Ant-v4/v5 | 四足（蚂蚁） | ~111 | 8 | 四足协调运动 |
| Humanoid-v4/v5 | 人形机器人 | ~376 | 17 | 人形直立行走 |

### 9.2 控制器类型与参数

#### 线性控制器（Linear Controller）

**参数数量：** `obs_dim × action_dim + action_dim`

**公式：** `action = W @ observation + b`

| 环境 | 观察维度 | 动作维度 | 控制器参数维度 | 参数说明 |
|------|----------|----------|---------------|----------|
| HalfCheetah | 17 | 6 | **108** | W: 17×6=102维, b: 6维 |
| Hopper | 11 | 3 | **36** | W: 11×3=33维, b: 3维 |
| Walker2d | 17 | 6 | **108** | W: 17×6=102维, b: 6维 |
| Ant | 111 | 8 | **896** | W: 111×8=888维, b: 8维 |
| Humanoid | 376 | 17 | **6409** | W: 376×17=6392维, b: 17维 |

**参数类型：** 连续

**取值范围：** [-1, 1]（所有参数）

**参数含义：**
- **W矩阵元素** (W[i,j])：第j个观察量对第i个动作的权重
- **b向量元素** (b[i])：第i个动作的偏置

#### MLP控制器（Multi-Layer Perceptron）

**结构：** `obs_dim → 32 → action_dim`

**参数数量：** `obs_dim × 32 + 32 + 32 × action_dim + action_dim`

| 环境 | 观察维度 | 动作维度 | 控制器参数维度 | 参数说明 |
|------|----------|----------|---------------|----------|
| HalfCheetah | 17 | 6 | **806** | 层1: 17×32+32=576, 层2: 32×6+6=198 |
| Hopper | 11 | 3 | **419** | 层1: 11×32+32=384, 层2: 32×3+3=99 |
| Walker2d | 17 | 6 | **806** | 层1: 17×32+32=576, 层2: 32×6+6=198 |
| Ant | 111 | 8 | **3856** | 层1: 111×32+32=3584, 层2: 32×8+8=264 |
| Humanoid | 376 | 17 | **12569** | 层1: 376×32+32=12064, 层2: 32×17+17=561 |

**参数类型：** 连续

**取值范围：** [-1, 1]（所有参数）

**参数含义：**
- **W1矩阵** (obs_dim × 32)：输入层到隐藏层的权重
- **b1向量** (32)：隐藏层偏置
- **W2矩阵** (32 × action_dim)：隐藏层到输出层的权重
- **b2向量** (action_dim)：输出层偏置

### 9.3 观察空间详解

#### HalfCheetah 观察（17维）
1. 根节点水平速度
2-9. 关节角度（8个关节）
10-17. 关节角速度（8个关节）

#### Hopper 观察（11维）
1. z坐标（高度）
2-5. 关节角度（4个）
6-11. 速度（6个）

#### Humanoid 观察（376维）
1-3. 质心位置
4-27. 关节位置（24个关节）
28-51. 关节角度
52-376. 身体部位速度、加速度等

---

## 9. Robosuite Manipulation

**文件位置：** `bomegabench/functions/robosuite_manipulation.py`

**总数：** 11 个操作任务 × 2 个控制器类型 = 22 个任务

**用途：** 机械臂操作控制器参数优化

### 10.1 任务列表

| 任务名称 | 难度 | 描述 |
|---------|------|------|
| **Lift** | 简单 | 抓取并提升方块 |
| **Stack** | 中等 | 堆叠两个方块 |
| **PickPlace** | 中等 | 拾取并放置物体到目标位置 |
| **NutAssembly** | 困难 | 螺母装配任务 |
| **Door** | 中等 | 打开柜门 |
| **Wipe** | 中等 | 擦拭表面 |
| **ToolHang** | 困难 | 挂工具 |
| **TwoArmHandover** | 困难 | 双臂传递物体 |
| **TwoArmLift** | 中等 | 双臂协作提升 |
| **TwoArmPegInHole** | 困难 | 双臂插孔任务 |
| **TwoArmTransport** | 困难 | 双臂运输物体 |

### 10.2 机器人配置

**默认机器人：** Panda (Franka Emika)

**其他可用机器人：**
- Sawyer
- IIWA
- Jaco
- Kinova3
- UR5e

### 10.3 控制器参数

#### 线性控制器

**维度：** `obs_dim × action_dim + action_dim`

**取值范围：** [-1, 1]

**典型维度（Lift任务，Panda机器人）：**
- 观察维度：~40-50（包括机械臂关节状态、物体位置、抓手状态）
- 动作维度：7（6个关节 + 1个抓手）
- 总维度：约 300-400 维

#### MLP控制器

**结构：** `obs_dim → 32 → action_dim`

**典型维度（Lift任务）：**
- 层1：obs_dim × 32 + 32
- 层2：32 × action_dim + action_dim
- 总维度：约 1500-2000 维

### 10.4 观察空间示例（Lift任务）

| 观察量 | 维度 | 描述 |
|-------|------|------|
| robot_joint_pos | 7 | 机械臂关节位置 |
| robot_joint_vel | 7 | 机械臂关节速度 |
| gripper_pos | 3 | 抓手位置（x,y,z） |
| gripper_quat | 4 | 抓手姿态（四元数） |
| gripper_aperture | 1 | 抓手开合度 |
| cube_pos | 3 | 方块位置 |
| cube_quat | 4 | 方块姿态 |
| gripper_to_cube | 3 | 抓手到方块的相对位置 |
| **总计** | **~32** | - |

---

## 10. HumanoidBench

**文件位置：** `bomegabench/functions/humanoid_bench_tasks.py`

**总数：** 5 个机器人 × 32 个任务 × 2 个控制器 = 320+ 个任务组合

**用途：** 全身人形机器人运动和操作控制

### 11.1 机器人列表

| 机器人 | 自由度(DoF) | 特点 | 适用任务 |
|--------|------------|------|----------|
| **h1** | 26 | Unitree H1（无手） | 运动任务 |
| **h1hand** | 76 | Unitree H1（带灵巧手） | 所有任务 |
| **h1strong** | 76 | H1增强版（可悬挂） | 单杠任务 |
| **h1touch** | 76 | H1（带触觉传感器） | 精细操作 |
| **g1** | 44 | Unitree G1（三指手） | 所有任务 |

### 11.2 运动任务（Locomotion）（13个）

| 任务名称 | 难度 | 描述 | 优化目标 |
|---------|------|------|----------|
| **walk** | 简单 | 向前行走 | 速度 + 稳定性 |
| **run** | 中等 | 快速奔跑 | 速度 |
| **stand** | 简单 | 保持站立 | 平衡 + 能量 |
| **crawl** | 中等 | 爬行运动 | 前进速度 |
| **hurdle** | 困难 | 跨越障碍 | 通过率 + 速度 |
| **stair** | 困难 | 爬楼梯 | 高度增加 |
| **slide** | 中等 | 滑动 | 速度 + 控制 |
| **pole** | 困难 | 爬杆 | 高度增加 |
| **maze** | 困难 | 迷宫导航 | 到达目标 |
| **sit_simple** | 简单 | 简单坐下 | 完成坐姿 |
| **sit_hard** | 中等 | 困难坐下 | 精确坐姿 |
| **balance_simple** | 中等 | 简单平衡 | 平衡时间 |
| **balance_hard** | 困难 | 困难平衡 | 动态平衡 |

### 11.3 操作任务（Manipulation）（19个）

| 任务名称 | 难度 | 描述 | 优化目标 |
|---------|------|------|----------|
| **reach** | 简单 | 伸手够物 | 到达目标 |
| **push** | 简单 | 推动物体 | 物体位移 |
| **door** | 中等 | 开门 | 门打开角度 |
| **cabinet** | 中等 | 打开柜子 | 柜门打开 |
| **truck** | 中等 | 操作货车 | 任务完成度 |
| **cube** | 中等 | 操作立方体 | 物体操作 |
| **bookshelf_simple** | 中等 | 简单书架操作 | 放置书籍 |
| **bookshelf_hard** | 困难 | 困难书架操作 | 精确放置 |
| **basketball** | 困难 | 投篮 | 投篮命中 |
| **window** | 中等 | 开窗 | 窗户打开 |
| **spoon** | 困难 | 使用勺子 | 勺子使用 |
| **kitchen** | 困难 | 厨房任务 | 多步骤完成 |
| **package** | 中等 | 包裹操作 | 搬运包裹 |
| **powerlift** | 困难 | 举重 | 举起重量 |
| **room** | 困难 | 房间任务 | 复杂交互 |
| **insert_normal** | 中等 | 正常插入 | 插入成功 |
| **insert_small** | 困难 | 小物件插入 | 精确插入 |
| **highbar_simple** | 中等 | 简单单杠 | 悬挂时间 |
| **highbar_hard** | 困难 | 困难单杠 | 单杠动作 |

### 11.4 控制器参数

#### 线性控制器（H1Hand机器人）

**典型维度（walk任务）：**
- 观察维度：~75
- 动作维度：~19
- 控制器参数：75 × 19 + 19 = **1444 维**

**参数类型：** 连续

**取值范围：** [-1, 1]

#### MLP控制器（H1Hand机器人）

**结构：** `obs_dim → 64 → action_dim` （注意：HumanoidBench使用64个隐藏单元，比MuJoCo的32更多）

**典型维度（walk任务）：**
- 层1：75 × 64 + 64 = 4864
- 层2：64 × 19 + 19 = 1235
- 总维度：**6099 维**

### 11.5 观察空间示例（H1Hand）

| 观察量类别 | 维度 | 描述 |
|-----------|------|------|
| 本体感觉 | ~30 | 关节位置、速度 |
| 身体状态 | ~20 | 躯干姿态、速度、加速度 |
| 手部状态 | ~20 | 双手位置、姿态、抓取状态 |
| 任务相关 | ~5-10 | 目标位置、物体状态等 |
| **总计** | **~75** | - |

---

## 11. Design-Bench

**文件位置：** `bomegabench/functions/design_bench_tasks.py`

**总数：** 25+ 个任务（不同oracle组合）

**用途：** 设计优化（材料、蛋白质、DNA/RNA、神经架构）

### 12.1 材料科学（Materials Science）

#### 超导体（Superconductor）

**任务数：** 3个（不同oracle）

| Oracle类型 | 维度 | 参数类型 | 取值范围 | 描述 |
|-----------|------|----------|----------|------|
| RandomForest | **81** | 连续 | 归一化 | 随机森林代理模型 |
| GP | **81** | 连续 | 归一化 | 高斯过程代理模型 |
| FullyConnected | **81** | 连续 | 归一化 | 全连接神经网络 |

**参数说明（81维材料特征）：**
- 参数1-81：材料的物理化学特征（如原子组成、晶体结构等）
- 参数类型：连续型
- 优化目标：最大化超导临界温度（Tc）

### 12.2 蛋白质设计（Protein Design）

#### GFP（绿色荧光蛋白）

**任务数：** 6个（不同oracle）

| Oracle类型 | 序列长度 | 编码后维度 | 参数类型 | 描述 |
|-----------|----------|-----------|----------|------|
| RandomForest | 237 | **4740** | 离散→One-hot | 随机森林 |
| GP | 237 | **4740** | 离散→One-hot | 高斯过程 |
| FullyConnected | 237 | **4740** | 离散→One-hot | 全连接网络 |
| LSTM | 237 | **4740** | 离散→One-hot | 长短期记忆网络 |
| ResNet | 237 | **4740** | 离散→One-hot | 残差网络 |
| Transformer | 237 | **4740** | 离散→One-hot | Transformer |

**参数详解：**
- **原始表示：** 237个位置，每个位置是一个氨基酸
- **氨基酸选项：** 20种标准氨基酸
- **One-hot编码：** 237位置 × 20氨基酸 = **4740维**

**One-hot编码示例：**
```
位置1: [1,0,0,...,0] (第1个氨基酸 - Alanine)
位置2: [0,0,0,...,1] (第20个氨基酸 - Tyrosine)
...
位置237: [0,1,0,...,0] (第2个氨基酸 - Cysteine)
```

**参数含义：**
- 维度1-20：位置1的氨基酸选择（one-hot）
- 维度21-40：位置2的氨基酸选择（one-hot）
- ...
- 维度4721-4740：位置237的氨基酸选择（one-hot）

**优化目标：** 最大化荧光强度

### 12.3 DNA/RNA序列优化

#### TFBind8（8碱基对转录因子结合）

**任务数：** 5个（不同oracle）

| Oracle类型 | 序列长度 | 编码后维度 | 参数类型 | 描述 |
|-----------|----------|-----------|----------|------|
| RandomForest | 8 bp | **32** | 离散→One-hot | 随机森林 |
| GP | 8 bp | **32** | 离散→One-hot | 高斯过程 |
| FullyConnected | 8 bp | **32** | 离散→One-hot | 全连接网络 |
| LSTM | 8 bp | **32** | 离散→One-hot | LSTM |
| Exact | 8 bp | **32** | 离散→One-hot | 精确模拟 |

**参数详解：**
- **原始表示：** 8个核苷酸位置
- **核苷酸选项：** 4种碱基 {A, T, C, G}
- **One-hot编码：** 8位置 × 4碱基 = **32维**

**One-hot编码示例：**
```
位置1: [1,0,0,0] (A - Adenine)
位置2: [0,1,0,0] (T - Thymine)
位置3: [0,0,1,0] (C - Cytosine)
位置4: [0,0,0,1] (G - Guanine)
...
```

**优化目标：** 最大化转录因子结合亲和力

#### TFBind10（10碱基对转录因子结合）

| Oracle类型 | 序列长度 | 编码后维度 | 参数类型 |
|-----------|----------|-----------|----------|
| Exact | 10 bp | **40** | 离散→One-hot |

**编码：** 10位置 × 4碱基 = **40维**

#### UTR（5'非翻译区）

**任务数：** 6个（不同oracle）

| Oracle类型 | 序列长度 | 编码后维度 | 参数类型 | 描述 |
|-----------|----------|-----------|----------|------|
| RandomForest | 50 nt | **200** | 离散→One-hot | 随机森林 |
| GP | 50 nt | **200** | 离散→One-hot | 高斯过程 |
| FullyConnected | 50 nt | **200** | 离散→One-hot | 全连接网络 |
| LSTM | 50 nt | **200** | 离散→One-hot | LSTM |
| ResNet | 50 nt | **200** | 离散→One-hot | ResNet |
| Transformer | 50 nt | **200** | 离散→One-hot | Transformer |

**参数详解：**
- **原始表示：** 50个核苷酸位置
- **核苷酸选项：** 4种碱基 {A, U, C, G} (RNA)
- **One-hot编码：** 50位置 × 4碱基 = **200维**

**优化目标：** 最大化翻译效率

### 12.4 神经架构搜索（NAS）

#### CIFARNAS

| Oracle类型 | 参数类型 | 编码方式 | 描述 |
|-----------|----------|----------|------|
| Exact | 离散（架构选择） | One-hot | CIFAR-10上的NAS |

**参数说明：**
- 每个架构参数：操作类型选择（如conv3x3, conv5x5, maxpool, skip等）
- 编码后维度：取决于搜索空间大小
- 优化目标：最大化CIFAR-10验证准确率

#### NASBench

| Oracle类型 | 参数类型 | 编码方式 | 描述 |
|-----------|----------|----------|------|
| Exact | 离散（架构选择） | One-hot | NAS-Bench-101 |

**参数说明：**
- 基于NAS-Bench-101搜索空间
- 包含：操作类型 + 连接模式
- 编码后维度：~100-200维（取决于具体搜索空间定义）

---

## 总结与统计

### 按维度分类

| 维度范围 | 任务数量 | 代表性任务 |
|---------|---------|-----------|
| **2D** | ~50 | BBOB函数、经典2D函数 |
| **3-10D** | ~100 | 小规模HPO、Database（部分） |
| **10-100D** | ~150 | LassoBench（简单/中等）、MuJoCo线性控制器 |
| **100-500D** | ~100 | LassoBench（高维）、NAS、MuJoCo MLP控制器 |
| **500-5000D** | ~50 | HumanoidBench、Design-Bench（蛋白质） |
| **5000+D** | ~20 | LassoBench（RCV1/Leukemia）、Humanoid MLP控制器 |

### 按参数类型分类

| 参数类型 | 任务数量 | 代表性任务 |
|---------|---------|-----------|
| **纯连续** | ~300 | BBOB、MuJoCo、Robosuite、LassoBench、Olympus表面 |
| **纯离散** | ~50 | Design-Bench（蛋白质/DNA）、NAS、LABS |
| **混合型** | ~150 | HPOBench、HPO Benchmarks、Database、AckleyMixed、Olympus数据集 |

### 按应用领域分类

| 领域 | 任务数量 | Benchmark |
|------|---------|-----------|
| **经典优化** | ~100 | Consolidated Functions (BBOB, BoTorch, Classical) |
| **超参数优化** | ~200 | HPOBench, HPO Benchmarks, LassoBench |
| **机器人控制** | ~100 | MuJoCo, Robosuite, HumanoidBench |
| **分子/材料设计** | ~80 | Design-Bench, Olympus Datasets |
| **数据库/系统** | ~20 | Database Tuning |
| **神经架构搜索** | ~15 | HPOBench NAS, Design-Bench NAS |

### 编码方式统计

| 编码方式 | 使用场景 | 示例 |
|---------|---------|------|
| **直接连续** | 连续参数 | BBOB、MuJoCo控制器权重 |
| **归一化** | 连续参数到[0,1] | HPO浮点超参数 |
| **One-hot** | 离散/类别参数 | HPO整数/类别参数、DNA序列、蛋白质 |
| **混合** | 混合型任务 | Database Tuning、部分Olympus数据集 |

### 特殊属性

| 属性 | 任务数量 | 说明 |
|------|---------|------|
| **高维稀疏** | ~15 | LassoBench（只有少数维度活跃） |
| **多峰** | ~200 | 大多数经典函数、Olympus表面 |
| **序列优化** | ~30 | DNA/RNA、蛋白质设计 |
| **真实数据** | ~50 | Olympus Datasets、LassoBench真实数据集 |
| **多保真度** | ~20 | HPOBench MF variants |

---

## 使用建议

### 1. 低维测试（≤10D）
- 适合：BBOB函数、经典2D函数
- 用途：快速原型验证、可视化

### 2. 中维测试（10-100D）
- 适合：LassoBench简单/中等、MuJoCo线性控制器
- 用途：标准BO算法测试

### 3. 高维测试（100-1000D）
- 适合：LassoBench高维、HPOBench、NAS
- 用途：测试高维优化能力

### 4. 极高维测试（>1000D）
- 适合：LassoBench RCV1、HumanoidBench MLP、Design-Bench GFP
- 用途：极限性能测试

### 5. 混合型测试
- 适合：HPOBench、Database Tuning
- 用途：测试混合整数优化

### 6. 真实应用测试
- 适合：Olympus Datasets、LassoBench真实数据集
- 用途：实际应用性能评估

---

**文档版本：** v1.0
**最后更新：** 2025-10-22
**总任务数：** 500+
**覆盖维度：** 2D - 19,959D
