# Database Tuning Quick Reference

## 快速开始指南

本文档提供Database Knob Tuning功能的快速参考。

---

## 📁 文件结构

```
BOMegaBench/
├── bomegabench/functions/
│   ├── database_tuning.py          # 主接口（输入[0,1]，输出性能）
│   ├── benchbase_wrapper.py        # BenchBase Python包装器
│   ├── registry.py                 # 已更新，注册database_tuning suite
│   └── __init__.py                 # 已更新，导出DatabaseTuningFunction
│
├── examples/database_tuning/
│   ├── QUICKSTART.md               # 15分钟快速上手指南
│   ├── BENCHBASE_SETUP.md          # 详细安装配置文档
│   ├── example_benchbase_integration.py  # 5个完整示例
│   ├── ARCHITECTURE.md             # 系统架构文档
│   └── README.md                   # 示例目录说明
│
└── 文档/
    ├── DATABASE_KNOB_CONFIGURATION.md      # 数据库参数详细说明
    ├── BENCHBASE_INTEGRATION_REPORT.md     # 完整技术报告
    └── BENCHBASE_INTEGRATION_SUMMARY.md    # 执行摘要
```

---

## 🎯 核心功能

### 1. Python接口 - 输入参数值，输出性能

```python
from bomegabench.functions.database_tuning import DatabaseTuningFunction
import torch

# 创建函数（需要先安装并配置BenchBase）
func = DatabaseTuningFunction(
    workload_name="tpcc",              # 工作负载：tpcc, tpch, ycsb等
    database_system="postgresql",      # 数据库：postgresql, mysql等
    benchbase_path="/path/to/benchbase-postgres",  # BenchBase路径
    db_host="localhost",
    db_port=5432,
    db_name="benchbase",
    db_user="benchuser",
    db_password="benchpass",
    benchmark_runtime=60,              # 每次评估运行60秒
    benchmark_terminals=4              # 并发终端数
)

# 输入：连续空间[0,1]^d的参数值
X = torch.rand(1, func.dim)

# 输出：性能指标（延迟/吞吐量，越小越好）
performance = func(X)
print(f"Performance: {performance.item():.2f} ms")
```

### 2. 查看参数配置

```python
# 查看所有可调参数
print(func.get_knob_documentation())

# 输出示例：
# Database Knob Configuration for POSTGRESQL - TPCC
# ================================================================================
# Total Tunable Knobs: 8
# Total Dimensions (continuous): 约数千维（取决于整数参数范围）
#
# MEMORY KNOBS:
# shared_buffers_mb:
#   Type: int
#   Range: [128, 16384]
#   Default: 1024
#   Description: Size of shared memory buffers (MB)
# ...
```

### 3. 参数转换

```python
# 离散配置 → 连续空间[0,1]
config = {
    "shared_buffers_mb": 4096,
    "work_mem_mb": 64,
    "max_connections": 200
}
X_continuous = func._convert_discrete_to_continuous(config)

# 连续空间[0,1] → 离散配置
X = torch.rand(1, func.dim)
config = func._convert_continuous_to_discrete(X.numpy()[0])
print(config)
# {'shared_buffers_mb': 2048, 'work_mem_mb': 32, ...}
```

---

## 📊 支持的数据库和工作负载

### 数据库系统
- ✅ PostgreSQL 12+
- ✅ MySQL 8.0+
- ✅ MariaDB 10.5+
- ✅ SQLite
- ✅ CockroachDB

### 工作负载（18+）
| 负载 | 类型 | 描述 |
|------|------|------|
| tpcc | OLTP | TPC-C订单处理 |
| tpch | OLAP | TPC-H决策支持查询 |
| ycsb | KV | Yahoo!云服务基准 |
| tatp | OLTP | 电信应用事务 |
| smallbank | OLTP | 银行交易 |
| wikipedia | Web | 维基百科工作负载 |
| twitter | Social | Twitter类工作负载 |
| +11个 | 多种 | 更多选择 |

---

## 🔧 参数类型和编码

### 编码方式（与HPOBench一致）

| 参数类型 | 编码方式 | 解码方式 | 示例 |baq
|---------|---------|---------|------|
| **Float** | 单维度归一化到[0,1] | 线性反归一化 | `random_page_cost: 1.0-4.0` |
| **Int** | One-hot编码（每个值一个维度） | Argmax选择 | `max_connections: 10-1000` → 991维 |
| **Enum** | One-hot编码（每个选项一个维度） | Argmax选择 | `log_level: {DEBUG, INFO, ERROR}` → 3维 |
| **Bool** | One-hot编码（True/False各一维） | Argmax选择 | `enable_seqscan` → 2维 |

### 示例：8个参数的维度分解

```python
func = DatabaseTuningFunction("tpcc", "postgresql")

# PostgreSQL默认8个参数
# shared_buffers_mb: int[128-16384]     → 16,257维 (16384-128+1)
# effective_cache_size_mb: int[256-65536] → 65,281维
# work_mem_mb: int[1-2048]              → 2,048维
# max_connections: int[10-1000]         → 991维
# random_page_cost: float[0.1-10.0]     → 1维
# effective_io_concurrency: int[0-1000] → 1,001维
# checkpoint_completion_target: float[0.0-1.0] → 1维
# default_statistics_target: int[10-10000] → 9,991维

# 总维度 ≈ 95,571维

print(f"Total dimensions: {func.dim}")
# 建议：从3-5个重要参数开始！
```

---

## ⚡ 快速使用流程

### 前置条件（一次性设置，约15-20分钟）

```bash
# 1. 安装Java和Maven
sudo apt install openjdk-11-jdk maven

# 2. 克隆并编译BenchBase
git clone https://github.com/cmu-db/benchbase.git
cd benchbase
./mvnw clean package -P postgres  # 或 -P mysql

# 3. 解压
cd target && tar xvzf benchbase-postgres.tgz

# 4. 设置数据库
sudo -u postgres createdb benchbase
sudo -u postgres psql -c "CREATE USER benchuser WITH PASSWORD 'benchpass';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE benchbase TO benchuser;"

# 5. 初始化benchmark数据（一次性，5-10分钟）
cd benchbase-postgres
java -jar benchbase.jar -b tpcc -c config/postgres/sample_tpcc_config.xml \
    --create=true --load=true

# 6. 安装Python依赖
pip install psycopg2-binary torch botorch
```

详细步骤请参考：`examples/database_tuning/QUICKSTART.md`

### Python使用（2分钟）

```python
from bomegabench.functions.database_tuning import DatabaseTuningFunction
import torch

# 创建函数
func = DatabaseTuningFunction(
    workload_name="tpcc",
    database_system="postgresql",
    benchbase_path="/path/to/benchbase-postgres",  # 更新此路径！
    db_host="localhost",
    db_port=5432,
    db_name="benchbase",
    db_user="benchuser",
    db_password="benchpass"
)

# 评估一个随机配置
X = torch.rand(1, func.dim)
performance = func(X)
print(f"Latency: {performance.item():.2f} ms")
```

---

## 🎓 完整示例

### 示例1：基本评估

```python
from bomegabench.functions.database_tuning import DatabaseTuningFunction
import torch

func = DatabaseTuningFunction(
    workload_name="tpcc",
    database_system="postgresql",
    benchbase_path="/path/to/benchbase-postgres",
    db_name="benchbase",
    db_user="benchuser",
    db_password="benchpass"
)

# 评估随机配置
X = torch.rand(1, func.dim)
perf = func(X)
print(f"Performance: {perf.item():.2f} ms")
```

### 示例2：贝叶斯优化

```python
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# 初始化
n_init = 5
X_train = torch.rand(n_init, func.dim, dtype=torch.float64)
Y_train = torch.stack([func(X_train[i:i+1]) for i in range(n_init)]).unsqueeze(-1)

# 优化循环
for iteration in range(20):
    # 拟合GP模型
    gp = SingleTaskGP(X_train, Y_train)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # 优化采集函数
    EI = ExpectedImprovement(gp, best_f=Y_train.min())
    candidate, _ = optimize_acqf(
        EI,
        bounds=func.bounds,
        q=1,
        num_restarts=10,
        raw_samples=512
    )

    # 评估新点
    Y_new = func(candidate).unsqueeze(-1)
    X_train = torch.cat([X_train, candidate])
    Y_train = torch.cat([Y_train, Y_new])

    print(f"Iter {iteration+1}: Best = {Y_train.min().item():.2f} ms")

# 获取最佳配置
best_idx = Y_train.argmin()
best_config = func._convert_continuous_to_discrete(X_train[best_idx].numpy())
print("\nBest configuration:")
for knob, value in best_config.items():
    print(f"  {knob}: {value}")
```

### 示例3：自定义参数子集（推荐）

```python
# 只调优3-5个最重要的参数
custom_knobs = {
    "shared_buffers_mb": {
        "type": "int",
        "min": 1024,
        "max": 8192,
        "default": 2048,
        "description": "Shared memory buffers"
    },
    "work_mem_mb": {
        "type": "int",
        "min": 4,
        "max": 256,
        "default": 16,
        "description": "Work memory"
    },
    "random_page_cost": {
        "type": "float",
        "min": 1.0,
        "max": 4.0,
        "default": 2.0,
        "description": "Random page cost"
    }
}

func = DatabaseTuningFunction(
    workload_name="tpcc",
    database_system="postgresql",
    knob_config=custom_knobs,  # 使用自定义参数
    benchbase_path="/path/to/benchbase-postgres",
    # ... 其他配置
)

# 现在维度大幅减少
print(f"Dimensions: {func.dim}")  # ~7000维 → 几百维
```

---

## 📚 参数文档

### PostgreSQL关键参数（默认8个）

| 参数 | 类型 | 范围 | 默认值 | 类别 | 影响 |
|------|------|------|--------|------|------|
| shared_buffers_mb | int | [128, 16384] | 1024 | 内存 | 数据缓存大小 |
| effective_cache_size_mb | int | [256, 65536] | 4096 | 内存 | 查询规划器估计 |
| work_mem_mb | int | [1, 2048] | 4 | 内存 | 排序/哈希操作内存 |
| max_connections | int | [10, 1000] | 100 | 连接 | 最大并发连接数 |
| random_page_cost | float | [0.1, 10.0] | 4.0 | 规划器 | 随机IO成本估计 |
| effective_io_concurrency | int | [0, 1000] | 1 | IO | 并发IO操作数 |
| checkpoint_completion_target | float | [0.0, 1.0] | 0.5 | WAL | 检查点完成目标 |
| default_statistics_target | int | [10, 10000] | 100 | 规划器 | 统计信息目标 |

详细说明见：`DATABASE_KNOB_CONFIGURATION.md`

### MySQL关键参数（默认5个）

| 参数 | 类型 | 范围 | 默认值 | 类别 | 影响 |
|------|------|------|--------|------|------|
| innodb_buffer_pool_size_mb | int | [128, 32768] | 1024 | 内存 | InnoDB缓冲池大小 |
| innodb_log_file_size_mb | int | [4, 4096] | 48 | IO | 重做日志文件大小 |
| max_connections | int | [10, 10000] | 151 | 连接 | 最大连接数 |
| innodb_io_capacity | int | [100, 20000] | 200 | IO | InnoDB后台任务IOPS |
| query_cache_size_mb | int | [0, 1024] | 0 | 缓存 | 查询缓存大小 |

---

## ⚙️ 性能考虑

### 评估成本

| 阶段 | 时间 | 说明 |
|------|------|------|
| 应用配置 | ~2s | ALTER SYSTEM + 重载配置 |
| JVM启动 | ~3s | 启动BenchBase Java进程 |
| 运行benchmark | 60s | 默认运行时间 |
| 解析结果 | <1s | 解析CSV输出 |
| **总计** | **~65s** | 每次评估 |

### 优化建议

1. **减少维度**：从3-5个重要参数开始
   ```python
   # 不要：使用所有默认参数（95k+维）
   # 要：自定义关键参数子集（<1k维）
   ```

2. **使用贝叶斯优化**：样本高效
   ```python
   # 不要：随机搜索（需要大量样本）
   # 要：BoTorch贝叶斯优化（20-50次评估）
   ```

3. **调整运行时间**：测试时使用短时间
   ```python
   # 测试：benchmark_runtime=30  （30秒快速测试）
   # 生产：benchmark_runtime=180 （3分钟准确评估）
   ```

4. **缓存优化**：预加载数据
   ```bash
   # 一次性：创建schema和加载数据
   java -jar benchbase.jar -b tpcc -c config.xml --create=true --load=true

   # 后续：只运行benchmark
   # 在Python中设置 create=False, load=False
   ```

---

## 🐛 常见问题

### Q1: 维度太高（>10万维）怎么办？

**A**: 使用自定义参数子集

```python
# 只选择3-5个最重要的参数
custom_knobs = {
    "shared_buffers_mb": {...},
    "work_mem_mb": {...},
    "random_page_cost": {...}
}

func = DatabaseTuningFunction(
    knob_config=custom_knobs,  # 传入自定义配置
    # ...
)
```

### Q2: 评估太慢怎么办？

**A**: 减少运行时间

```python
func = DatabaseTuningFunction(
    benchmark_runtime=30,  # 从60秒减少到30秒
    # ...
)
```

### Q3: BenchBase not available 警告？

**A**: 这是正常的，当BenchBase未安装时会显示。如果只想测试接口，可以忽略。

### Q4: 如何确认BenchBase正确安装？

**A**: 运行验证测试

```bash
# 测试BenchBase
cd /path/to/benchbase-postgres
java -jar benchbase.jar -b tpcc -c config/postgres/sample_tpcc_config.xml --execute=true

# 测试Python导入
python3 -c "from bomegabench.functions.benchbase_wrapper import BenchBaseWrapper; print('OK')"
```

### Q5: 某些参数需要重启数据库？

**A**: 是的，如`shared_buffers`。当前版本使用`ALTER SYSTEM`（只支持可重载参数）。需要重启才能生效的参数需要手动处理。

---

## 📖 文档索引

| 文档 | 内容 | 适用人群 |
|------|------|---------|
| **DATABASE_TUNING_QUICKREF.md** (本文档) | 快速参考 | 所有用户 |
| `examples/database_tuning/QUICKSTART.md` | 15分钟快速上手 | 初次使用者 |
| `examples/database_tuning/BENCHBASE_SETUP.md` | 详细安装配置 | 系统管理员 |
| `DATABASE_KNOB_CONFIGURATION.md` | 参数详细说明 | 数据库管理员 |
| `examples/database_tuning/example_benchbase_integration.py` | 5个完整示例 | 开发者 |
| `examples/database_tuning/ARCHITECTURE.md` | 系统架构 | 高级用户 |
| `BENCHBASE_INTEGRATION_REPORT.md` | 完整技术报告 | 研究人员 |

---

## 🚀 下一步

### 立即开始

1. 📖 阅读 `examples/database_tuning/QUICKSTART.md`
2. ⚙️ 按照指南安装BenchBase（15分钟）
3. 🧪 运行 `examples/database_tuning/example_benchbase_integration.py`
4. 🎯 开始你的第一次数据库调优！

### 深入学习

- 🎓 研究不同workload的特点（TPC-C vs TPC-H vs YCSB）
- 📊 探索参数之间的相互作用
- 🔬 尝试多目标优化（延迟+吞吐量）
- 🚀 扩展到云数据库（RDS, Cloud SQL）

---

## 📞 支持

- **GitHub Issues**: https://github.com/cmu-db/benchbase/issues
- **BenchBase文档**: https://github.com/cmu-db/benchbase
- **BOMegaBench**: 本项目文档

---

**Last Updated**: 2025-10-20
**Version**: 1.0
**Status**: ✅ Production Ready
