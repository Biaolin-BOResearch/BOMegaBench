# 超参数优化基准测试完整用例列表（去重版）

## 1. LassoBench（9个用例）

### 合成数据集
| 用例名称 | 数据规模 | 优化超参数 | 优化目标 |
|---------|---------|------------|---------|
| synt_simple | 30×60维，3个有效特征 | α (正则化强度), 权重向量 | 稀疏回归误差最小化 |
| synt_medium | 50×100维，5个有效特征 | α (正则化强度), 权重向量 | 稀疏回归误差最小化 |
| synt_high | 150×300维，15个有效特征 | α (正则化强度), 权重向量 | 稀疏回归误差最小化 |
| synt_hard | 500×1000维，50个有效特征 | α (正则化强度), 权重向量 | 稀疏回归误差最小化 |

### 真实数据集
| 用例名称 | 数据规模 | 优化超参数 | 优化目标 |
|---------|---------|------------|---------|
| Breast_cancer | 683×10 | α (正则化强度), 容差级别 | 分类准确率 |
| Diabetes | 768×8 | α (正则化强度), 容差级别 | 回归误差 |
| Leukemia | 72×7129 | α (正则化强度), 容差级别 | 分类准确率 |
| DNA | 2000×180 | α (正则化强度), 容差级别 | 分类准确率 |
| RCV1 | 20242×19959 | α (正则化强度), 容差级别 | 文本分类准确率 |

## 2. Bayesmark（108个组合用例）

### 分类模型超参数
| 模型 | 主要超参数 | 优化目标 | 适用数据集 |
|------|-----------|----------|------------|
| DecisionTreeClassifier | max_depth, min_samples_split, min_samples_leaf | 准确率/负对数似然 | iris, wine, digits, breast |
| MLPClassifier-Adam | hidden_layer_sizes, learning_rate_init, alpha, batch_size | 准确率/负对数似然 | iris, wine, digits, breast |
| MLPClassifier-SGD | hidden_layer_sizes, learning_rate, momentum, alpha | 准确率/负对数似然 | iris, wine, digits, breast |
| RandomForestClassifier | n_estimators, max_depth, min_samples_split, max_features | 准确率/负对数似然 | iris, wine, digits, breast |
| SVC | C, gamma, kernel, degree | 准确率/负对数似然 | iris, wine, digits, breast |
| AdaBoostClassifier | n_estimators, learning_rate, algorithm | 准确率/负对数似然 | iris, wine, digits, breast |
| KNeighborsClassifier | n_neighbors, weights, algorithm, leaf_size | 准确率/负对数似然 | iris, wine, digits, breast |
| LogisticRegression | C, penalty, solver, max_iter | 准确率/负对数似然 | iris, wine, digits, breast |

### 回归模型超参数
| 模型 | 主要超参数 | 优化目标 | 适用数据集 |
|------|-----------|----------|------------|
| DecisionTreeRegressor | max_depth, min_samples_split, min_samples_leaf | MAE/MSE | boston, diabetes |
| MLPRegressor-Adam | hidden_layer_sizes, learning_rate_init, alpha, batch_size | MAE/MSE | boston, diabetes |
| MLPRegressor-SGD | hidden_layer_sizes, learning_rate, momentum, alpha | MAE/MSE | boston, diabetes |
| RandomForestRegressor | n_estimators, max_depth, min_samples_split, max_features | MAE/MSE | boston, diabetes |
| SVR | C, epsilon, gamma, kernel | MAE/MSE | boston, diabetes |
| AdaBoostRegressor | n_estimators, learning_rate, loss | MAE/MSE | boston, diabetes |
| KNeighborsRegressor | n_neighbors, weights, algorithm, leaf_size | MAE/MSE | boston, diabetes |
| Lasso | alpha, max_iter, tol | MAE/MSE | boston, diabetes |
| Ridge | alpha, solver, max_iter | MAE/MSE | boston, diabetes |

## 3. NASBench系列

### NASBench-101
| 用例配置 | 搜索空间 | 优化超参数 | 优化目标 |
|---------|---------|------------|---------|
| CIFAR-10架构搜索 | 423,624个DAG架构 | 操作类型(conv3x3, conv1x1, maxpool3x3), 连接模式, 训练轮数(4/12/36/108) | 验证准确率, 训练时间 |

### NASBench-201
| 用例配置 | 搜索空间 | 优化超参数 | 优化目标 |
|---------|---------|------------|---------|
| CIFAR-10 | 15,625个单元架构 | 5种操作(none, skip, conv1x1, conv3x3, avgpool3x3), 训练轮数(12/200) | 验证准确率 |
| CIFAR-100 | 15,625个单元架构 | 同上 | 验证准确率 |
| ImageNet-16-120 | 15,625个单元架构 | 同上 | 验证准确率 |

### NASBench-301
| 用例配置 | 搜索空间 | 优化超参数 | 优化目标 |
|---------|---------|------------|---------|
| DARTS搜索空间 | 10^18个架构(代理) | 8种操作, 单元连接模式, 深度 | 验证准确率(代理预测) |

### JAHS-Bench-201
| 用例配置 | 搜索空间 | 优化超参数 | 优化目标 |
|---------|---------|------------|---------|
| Fashion-MNIST | 15,625架构×混合HP | 架构参数 + 深度乘数(1/3/5) + 宽度乘数(4/8/16) + 分辨率(0.25/0.5/1.0) + 训练轮数(1-200) | 验证准确率, 延迟 |
| CIFAR-10 | 同上 | 同上 | 验证准确率, 延迟 |
| ColorectalHistology | 同上 | 同上 | 验证准确率, 延迟 |

### HW-NAS-Bench
| 硬件平台 | 搜索空间 | 优化超参数 | 优化目标 |
|---------|---------|------------|---------|
| NVIDIA Jetson TX2 | NASBench架构 | 架构配置 | 延迟(ms), 能耗(mJ), 准确率 |
| Raspberry Pi 4 | NASBench架构 | 架构配置 | 延迟(ms), 能耗(mJ), 准确率 |
| Google Edge TPU | NASBench架构 | 架构配置 | 延迟(ms), 能耗(mJ), 准确率 |
| Google Pixel 3 | NASBench架构 | 架构配置 | 延迟(ms), 能耗(mJ), 准确率 |
| Xilinx ZC706 FPGA | NASBench架构 | 架构配置 | 延迟(ms), 能耗(mJ), 准确率 |
| Eyeriss ASIC | NASBench架构 | 架构配置 | 延迟(ms), 能耗(mJ), 准确率 |

### NAS-HPO-Bench-II
| 用例配置 | 搜索空间 | 优化超参数 | 优化目标 |
|---------|---------|------------|---------|
| CIFAR-10 CNN | 4,000架构 | 架构 + 批大小(8/16/32/64/128/256) + 学习率(8个选项) | 验证准确率 |

### NASBench-NLP
| 用例配置 | 搜索空间 | 优化超参数 | 优化目标 |
|---------|---------|------------|---------|
| Penn Treebank | 14,000 RNN架构 | RNN操作(LSTM/GRU等), 隐藏维度, 层数 | 困惑度 |
| WikiText-2迁移 | 同上 | 同上 | 困惑度 |

### NASBench-ASR
| 用例配置 | 搜索空间 | 优化超参数 | 优化目标 |
|---------|---------|------------|---------|
| TIMIT | 8,242 ASR模型 | 编码器架构, 解码器配置, 训练轮数 | 音素错误率 |
| LibriSpeech迁移 | 同上 | 同上 | 词错误率 |

### NAS-Bench-360
| 任务领域 | 数据集规模 | 优化超参数 | 优化目标 |
|---------|-----------|------------|---------|
| CIFAR-100 | 60,000样本 | 架构配置 | 分类准确率 |
| 球面CIFAR-100 | 60,000样本 | 架构配置 | 分类准确率 |
| NinaPro手势识别 | 3,916样本 | 架构配置 | 识别准确率 |
| FSD50k音频分类 | 51,197样本 | 架构配置 | 分类准确率 |
| Darcy Flow | 科学计算 | 架构配置 | 预测误差 |
| PSICOV | 蛋白质结构 | 架构配置 | 结构预测准确率 |
| DeepCov | 生物信息学 | 架构配置 | 预测准确率 |

### TransNAS-Bench-101
| 任务类型 | 架构数 | 优化超参数 | 优化目标 |
|---------|--------|------------|---------|
| 目标分类 | 7,352主干架构 | 架构参数, 任务特定头 | 分类准确率 |
| 场景分类 | 同上 | 同上 | 分类准确率 |
| 房间布局 | 同上 | 同上 | 布局预测准确率 |
| 拼图任务 | 同上 | 同上 | 拼图解决准确率 |
| 语义分割 | 同上 | 同上 | mIoU |
| 表面法线 | 同上 | 同上 | 法线预测误差 |
| 自编码器 | 同上 | 同上 | 重建误差 |

## 4. HPOBench独有基准

### 机器学习算法基准
| 算法 | 数据集 | 优化超参数 | 优化目标 |
|------|--------|------------|---------|
| XGBoost | OpenML任务 | n_estimators, max_depth, learning_rate, subsample, colsample | 验证准确率/AUC |
| SVM | OpenML任务 | C, gamma, kernel, degree | 验证准确率/AUC |
| 神经网络 | OpenML任务 | 层数(1-3), 宽度(16-1024), 学习率, dropout | 验证准确率/AUC |
| 随机森林 | OpenML任务 | n_estimators, max_depth, min_samples_split, max_features | 验证准确率/AUC |
| 逻辑回归 | OpenML任务 | C, penalty, solver | 验证准确率/AUC |
| HistGB | OpenML任务 | max_iter, max_depth, learning_rate, l2_regularization | 验证准确率/AUC |

### 专门应用基准
| 应用领域 | 任务描述 | 优化超参数 | 优化目标 |
|---------|---------|------------|---------|
| Cartpole-PPO | 强化学习控制 | 学习率, 批大小, γ, clip_range, n_steps | 累积奖励 |
| SliceLocalization | CT切片定位 | 网络架构, 学习率, 正则化 | 定位准确率 |
| ProteinStructure | 蛋白质结构预测 | 模型参数, 训练配置 | 结构预测RMSD |
| NavalPropulsion | 系统建模 | 回归模型参数 | 预测误差 |
| ParkinsonsTelemonitoring | 疾病监测 | 特征选择, 模型参数 | 监测准确率 |
| BNN-ToyFunction | 贝叶斯神经网络 | 先验参数, 架构 | 不确定性校准 |
| BNN-BostonHousing | 贝叶斯神经网络 | 先验参数, 架构 | 预测误差+不确定性 |

### 异常检测基准
| 方法 | 数据集数量 | 优化超参数 | 优化目标 |
|------|-----------|------------|---------|
| ODAutoencoder | 15个数据集 | 编码器维度, 学习率, 正则化 | AUC-ROC |
| ODKernelDensity | 15个数据集 | 带宽, 核函数类型 | AUC-ROC |
| ODOneClassSVM | 15个数据集 | nu, gamma, kernel | AUC-ROC |

## 5. OpenML基准套件

### OpenML-CC18（72个分类任务）
| 特征范围 | 任务特点 | 优化超参数 | 优化目标 |
|---------|---------|------------|---------|
| 小规模任务 | 500-5,000样本，<50特征 | 算法特定超参数 | 分类准确率, AUC |
| 中规模任务 | 5,000-20,000样本，50-500特征 | 算法特定超参数 | 分类准确率, AUC |
| 大规模任务 | 20,000-100,000样本，500-5,000特征 | 算法特定超参数 | 分类准确率, AUC |

### AutoML Benchmark（104个任务）
| 任务类型 | 数量 | 优化内容 | 优化目标 |
|---------|------|----------|---------|
| 分类任务 | 71个 | 完整AutoML流程(特征工程+模型选择+超参数) | 准确率, AUC, 对数损失 |
| 回归任务 | 33个 | 完整AutoML流程(特征工程+模型选择+超参数) | RMSE, MAE, R² |

## 统计汇总

### 按框架统计
- **LassoBench**: 9个基准
- **Bayesmark**: 17模型×6数据集 = ~108个组合
- **NASBench系列**: 
  - NASBench-101: 423,624个架构
  - NASBench-201: 15,625个架构×3数据集
  - NASBench-301: 10^18个架构(代理)
  - JAHS-Bench-201: 15,625架构×多保真度
  - HW-NAS-Bench: 6个硬件平台
  - NAS-HPO-Bench-II: 192,000个配置
  - NASBench-NLP: 14,000个架构
  - NASBench-ASR: 8,242个模型
  - NAS-Bench-360: 10个任务
  - TransNAS-Bench-101: 7,352个架构×7任务
- **HPOBench**: 100+个多保真度基准
- **OpenML**: 72个CC18任务 + 104个AutoML任务

### 按优化目标分类
- **准确率优化**: 分类、NAS架构搜索
- **误差最小化**: 回归、预测任务
- **延迟/能耗优化**: 硬件感知NAS
- **稀疏性优化**: Lasso回归
- **多目标优化**: 准确率-延迟权衡、准确率-能耗权衡
- **不确定性量化**: 贝叶斯神经网络
- **迁移学习**: 跨任务、跨数据集性能

### 按超参数类型分类
- **连续参数**: 学习率、正则化强度、dropout率
- **离散参数**: 层数、神经元数、树深度
- **分类参数**: 激活函数、优化器、核函数
- **架构参数**: 操作类型、连接模式
- **多保真度参数**: 训练轮数、数据子集大小、模型规模