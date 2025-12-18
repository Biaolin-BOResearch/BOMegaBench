# 维度发现能力评估指标设计文档

## 1. 问题定义

在超参数优化中，一个重要但常被忽视的能力是：**优化器能否识别出真正影响目标函数的维度**。

许多实际问题中，参数空间可能包含大量"无效"参数（即改变这些参数不会影响目标函数值）。一个好的优化器应该能够：
1. 快速识别出哪些维度是"有效的"（真正影响目标函数）
2. 将探索资源集中在有效维度上
3. 避免在无效维度上浪费评估预算

### 1.1 实验设计

为了测试优化器的这种能力，我们设计了一个**维度扩展实验框架**：

```
原始函数 f(x₁, x₂, ..., xₘ)  →  扩展函数 g(z₁, z₂, ..., zₙ)
                                     ↑
                               n = m + k (k个假维度)
```

- 将原始 m 维黑盒函数扩展到 n 维
- 添加 k 个"假维度"（dummy dimensions），这些维度的值不影响函数输出
- 可选：随机打乱所有维度的顺序，让优化器无法通过索引判断
- 所有维度统一归一化到 [0, 1] 区间

### 1.2 评估目标

从优化轨迹中分析优化器的行为，定量回答：
- 优化器是否"发现"了真实有效的维度？
- 它是否把探索集中在这些维度上？
- 它区分有效/无效维度的能力有多强？

---

## 2. 核心方法：GP-ARD（高斯过程 + 自动相关性确定）

### 2.1 原理

我们使用高斯过程的**ARD（Automatic Relevance Determination）**核从优化轨迹中学习维度重要性。

ARD核为每个维度分配一个独立的**长度尺度（length scale）**参数 $\ell_d$。在Matern核中：

$$k(x, x') = \sigma^2 \cdot \text{Matern}_\nu\left(\sqrt{\sum_{d=1}^{D} \frac{(x_d - x'_d)^2}{\ell_d^2}}\right)$$

**核心思想**：长度尺度反映了函数在该维度上的变化剧烈程度：

| 长度尺度 | 含义 | 相关性 |
|---------|------|--------|
| **短** ($\ell_d$ 小) | 函数在该维度上变化剧烈 | **高** |
| **长** ($\ell_d$ 大) | 函数在该维度上几乎不变 | **低** |

### 2.2 实现步骤

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

def evaluate_feature_identification(X_history, y_history, orig_dim, active_indices):
    """
    通过优化轨迹定量评估维度识别能力。
    
    参数:
        X_history: 形状 (N, D) 的优化轨迹输入 (必须是 [0,1] 空间)
        y_history: 形状 (N,) 的目标函数值
        orig_dim: 真实有效维度的数量
        active_indices: 真实有效维度的索引列表
    
    返回:
        metrics: 包含 'auroc', 'top_k_accuracy', 'relevance_scores' 的字典
    """
    D = X_history.shape[1]
    
    # 1. 使用带ARD的Matern核拟合高斯过程
    kernel = C(1.0) * Matern(
        length_scale=np.ones(D), 
        length_scale_bounds=(1e-2, 1e5),  # 宽范围，区分活跃/非活跃
        nu=2.5
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=5, 
        normalize_y=True
    )
    
    # 2. 训练GP (核心步骤：从数据中逆向工程维度的重要性)
    gp.fit(X_history, y_history)
    
    # 3. 提取学习到的长度尺度
    learned_length_scales = gp.kernel_.k2.length_scale
    
    # 4. 计算相关性得分 (Relevance Score)
    #    相关性 = 1 / length_scale
    #    长度尺度越短，相关性越高
    relevance_scores = 1.0 / learned_length_scales
    
    # 5. 获取 Ground Truth
    true_mask = np.zeros(D, dtype=bool)
    true_mask[active_indices] = True
    
    # --- 指标计算 ---
    
    # Metric 1: AUROC
    auroc = roc_auc_score(true_mask, relevance_scores)
    
    # Metric 2: Top-K Accuracy
    k = orig_dim
    top_k_indices = np.argsort(relevance_scores)[::-1][:k]
    correct_identified = np.sum(true_mask[top_k_indices])
    top_k_acc = correct_identified / k
    
    return {
        "auroc": auroc,
        "top_k_accuracy": top_k_acc,
        "relevance_scores": relevance_scores,
        "learned_length_scales": learned_length_scales,
    }
```

### 2.3 参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `nu` | 2.5 | Matern核的光滑度参数，2.5适用于大多数优化问题 |
| `length_scale_bounds` | (1e-2, 1e5) | 宽范围以区分活跃/非活跃维度 |
| `n_restarts_optimizer` | 5 | GP拟合时的重启次数，增加稳定性 |
| `normalize_y` | True | 对目标值归一化，提高数值稳定性 |

### 2.4 方法优点

1. **理论基础扎实**：基于贝叶斯推断，是机器学习中标准的特征选择方法
2. **自动学习**：直接从数据中学习，无需人工设定权重或阈值
3. **全局考虑**：考虑了整个函数的结构，而不仅仅是局部统计
4. **处理非线性**：Matern核能捕捉复杂的非线性关系
5. **可解释性强**：length_scale有明确的物理含义

---

## 3. 评估指标

### 3.1 AUC-ROC（主要指标）

将维度分类问题视为二分类：
- 将所有维度按相关性分数（1/length_scale）排序
- 计算不同阈值下的真阳性率(TPR)和假阳性率(FPR)
- AUC = ROC曲线下面积

**解释**：
| AUC值 | 含义 |
|-------|------|
| 1.0 | 完美区分真假维度 |
| 0.5 | 随机猜测水平 |
| < 0.5 | 反向预测（比随机还差） |

### 3.2 Top-K Accuracy（直观指标）

设 k = 真实有效维度的数量，取相关性分数最高的 k 个维度：

```python
top_k_indices = np.argsort(relevance_scores)[::-1][:k]
top_k_accuracy = np.sum(true_mask[top_k_indices]) / k
```

**解释**：
- 1.0：完美识别所有真实维度
- 0.0：完全没有识别出真实维度

### 3.3 分离度指标

```python
mean_real = mean(relevance[real_dims])
mean_dummy = mean(relevance[dummy_dims])
separation_ratio = mean_real / (mean_dummy + ε)
```

**解释**：
- separation_ratio >> 1：很好地区分了真假维度
- separation_ratio ≈ 1：无法区分
- separation_ratio < 1：反向预测

### 3.4 综合发现分数

```python
discovery_score = 0.4 * auc_roc + 0.3 * f1_at_k + 0.3 * min(1.0, separation_ratio / 5.0)
```

**范围**：[0, 1]
- 0：完全没有发现能力
- 1：完美发现所有有效维度

---

## 4. 使用示例

### 4.1 基本使用

```python
from bomegabench.functions.synthetic.classical_core import LevyFunction
from bomegabench.utils.dimension_expansion import create_dimension_expansion_test

# 创建扩展测试函数：3维真实 + 7维假 = 10维
func, metrics = create_dimension_expansion_test(
    base_function_class=LevyFunction,
    original_dim=3,
    n_dummy_dims=7,
    shuffle=True,
    seed=42
)

# 运行你的优化器，收集轨迹 (必须在[0,1]空间)
trajectory_X = [...]  # shape: (n_points, 10)
trajectory_Y = [...]  # shape: (n_points,)

# 分析
result = metrics.analyze_trajectory(trajectory_X, trajectory_Y)
print(f"AUC-ROC: {result.auc_roc:.4f}")
print(f"Top-K Accuracy: {result.precision_at_k:.4f}")

# 查看学习到的length scales
print(f"Length Scales: {result.learned_length_scales}")

# 打印详细报告
print(metrics.print_analysis_report(result))
```

### 4.2 比较不同优化器

```python
optimizers = {
    "Random Search": random_search,
    "Bayesian Optimization": bo,
    "CMA-ES": cma_es,
}

for name, optimizer in optimizers.items():
    # 运行优化器并收集轨迹
    trajectory_X, trajectory_Y = run_optimizer(optimizer, func)
    
    # 分析
    result = metrics.analyze_trajectory(trajectory_X, trajectory_Y)
    print(f"{name}: AUC={result.auc_roc:.3f}, Top-K={result.precision_at_k:.3f}")
```

---

## 5. 设计考量与局限性

### 5.1 设计考量

1. **基于贝叶斯推断**：GP-ARD是特征选择的标准方法，有坚实的理论基础

2. **无需修改优化器**：只需要优化轨迹即可分析，不需要优化器提供额外信息

3. **可解释性**：length_scale有明确的物理含义——函数在该维度上的"相关距离"

### 5.2 局限性

1. **计算开销**：GP拟合是 O(n³)，对于大量采样点可能较慢

2. **样本量要求**：需要至少5个采样点才能可靠拟合

3. **假设**：假设函数可以被GP很好地近似；对于某些病态函数可能不准确

### 5.3 最佳实践

1. **样本量**：建议使用50-200个采样点进行分析

2. **归一化**：确保输入X在[0,1]空间，这对length_scale的可比性很重要

3. **多次运行**：由于GP拟合有随机性，建议多次运行取平均

---

## 附录：完整指标列表

| 指标名称 | 范围 | 说明 |
|---------|------|------|
| `auc_roc` | [0, 1] | ROC曲线下面积（主要指标） |
| `precision_at_k` | [0, 1] | Top-k预测精确率（= Top-K Accuracy） |
| `recall_at_k` | [0, 1] | Top-k预测召回率 |
| `f1_at_k` | [0, 1] | Top-k的F1分数 |
| `discovery_score` | [0, 1] | 综合发现能力分数 |
| `mean_importance_real` | [0, 1] | 真实维度的平均相关性 |
| `mean_importance_dummy` | [0, 1] | 假维度的平均相关性 |
| `separation_ratio` | [0, ∞) | 真/假维度相关性比值 |
| `learned_length_scales` | (0, ∞) | GP学习到的ARD长度尺度 |
| `best_real_rank` | [1, n] | 真实维度最好排名 |
| `worst_real_rank` | [1, n] | 真实维度最差排名 |
| `mean_real_rank` | [1, n] | 真实维度平均排名 |
