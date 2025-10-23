# Olympus集成 - 快速开始

## 一分钟上手

### Surfaces（测试函数）

```python
from bomegabench.functions import create_olympus_surfaces_suite
import torch

# 创建所有19个函数
suite = create_olympus_surfaces_suite()

# 使用某个函数
denali = suite.get_function('olympus_denali')
X = torch.rand(100, 2)  # 100个2D样本
Y = denali(X)           # 评估
print(f"Results shape: {Y.shape}")  # torch.Size([100])
```

### Datasets（真实数据）

```python
from bomegabench.functions import create_olympus_chemistry_suite
import torch

# 创建化学反应数据集
chem = create_olympus_chemistry_suite()  # 14个数据集

# 使用Suzuki反应数据
suzuki = chem.get_function('olympus_suzuki')
print(f"Dimension: {suzuki.dim}")  # 4D
print(f"Samples: {suzuki.metadata['num_train']}")  # 247

X = torch.rand(10, 4)
Y = suzuki(X)  # 基于真实数据的预测
```

### 直接使用单个函数

```python
from bomegabench.functions import (
    OlympusDenaliFunction,      # 山峰地形
    OlympusCatAckleyFunction,   # 分类变量
    OlympusSuzukiFunction,      # 真实化学数据
)

# 山峰函数
denali = OlympusDenaliFunction()

# 分类变量函数
cat_ackley = OlympusCatAckleyFunction(dim=3, num_opts=21)

# 真实数据集
suzuki = OlympusSuzukiFunction()
```

## 可用函数清单

### Surfaces（19个）

**分类变量（5个）**
- olympus_cat_ackley
- olympus_cat_camel
- olympus_cat_dejong
- olympus_cat_michalewicz
- olympus_cat_slope

**离散变量（3个）**
- olympus_discrete_ackley
- olympus_discrete_double_well
- olympus_discrete_michalewicz

**山峰地形（6个）**
- olympus_denali
- olympus_everest
- olympus_k2
- olympus_kilimanjaro
- olympus_matterhorn
- olympus_mont_blanc

**特殊函数（5个）**
- olympus_ackley_path
- olympus_gaussian_mixture
- olympus_hyper_ellipsoid
- olympus_linear_funnel
- olympus_narrow_funnel

### Datasets（14+个已验证）

**化学反应**
```python
create_olympus_chemistry_suite()
```
- olympus_buchwald_a, b, c, d, e（5个）
- olympus_suzuki, suzuki_edbo, suzuki_i, suzuki_ii（4个）
- olympus_benzylation
- olympus_alkox
- olympus_snar
等...

**其他类别**
```python
create_olympus_materials_suite()       # 材料科学
create_olympus_photovoltaics_suite()   # 光伏
create_olympus_datasets_suite()        # 全部
```

## 测试验证

```bash
python examples/test_olympus_integration.py
```

预期输出：
```
✅ Olympus surfaces integration test PASSED
✅ Olympus datasets integration test PASSED
🎉 All tests PASSED!
```

## 常见问题

**Q: 为什么有些dataset加载失败？**
A: 部分dataset需要额外依赖（如gurobipy），跳过即可，不影响其他dataset。

**Q: Datasets使用什么模型？**
A: 使用nearest neighbor从真实数据中查找，简单但有效。

**Q: 与BOMegaBench其他函数有冲突吗？**
A: 无冲突！仔细避免了重复函数（如Ackley, Branin等已在BOMegaBench中）。

**Q: 性能如何？**
A: Surfaces很快；Datasets稍慢（需查找最近邻），但可接受。

## 更多信息

- 完整文档：`OLYMPUS_INTEGRATION.md`
- 技术细节：`OLYMPUS_FINAL_SUMMARY.md`
- 测试脚本：`examples/test_olympus_integration.py`

---
🎉 现在你有了33+个额外的BO benchmark函数！
